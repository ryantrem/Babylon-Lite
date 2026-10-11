import { afterEach, describe, expect, it, vi } from "vitest";
import { _deferCommandBuffer } from "../../../packages/babylon-lite/src/compute/deferred-submit";
import { renderFrame, stopEngine, waitForGpuIdle } from "../../../packages/babylon-lite/src/engine/engine";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";

function setup() {
    let lose!: () => void;
    const device = {
        lost: new Promise<void>((resolve) => {
            lose = resolve;
        }),
        queue: { submit: vi.fn(), onSubmittedWorkDone: vi.fn(async () => undefined) },
    } as unknown as GPUDevice;
    const engine = { _device: device, _renderFn: null, _animFrameId: 0 } as EngineContext;
    return { engine, device, lose };
}

afterEach(() => vi.useRealTimers());

describe("generic deferred submission", () => {
    it("fuses in order, removes the hook before callbacks, and schedules only one fallback", () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        const a = {} as GPUCommandBuffer,
            b = {} as GPUCommandBuffer,
            frame = {} as GPUCommandBuffer;
        const callbacks: string[] = [];
        _deferCommandBuffer(engine, a, () => {
            expect(engine._flushPendingWork).toBeUndefined();
            expect(device.queue.submit).toHaveBeenCalledWith([a, b, frame]);
            callbacks.push("a");
        });
        _deferCommandBuffer(engine, b, () => callbacks.push("b"));
        expect(vi.getTimerCount()).toBe(1);
        engine._flushPendingWork!(engine, [frame]);
        expect(callbacks).toEqual(["a", "b"]);
        vi.runAllTimers();
        expect(device.queue.submit).toHaveBeenCalledOnce();
    });

    it("flushes without a frame through a macrotask, not a microtask", async () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        const after = vi.fn();
        _deferCommandBuffer(engine, {} as GPUCommandBuffer, after);
        await Promise.resolve();
        expect(device.queue.submit).not.toHaveBeenCalled();
        vi.runAllTimers();
        expect(device.queue.submit).toHaveBeenCalledOnce();
        expect(after).toHaveBeenCalledOnce();
    });

    it("keeps callback-admitted work in a new batch", () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        const a = {} as GPUCommandBuffer,
            b = {} as GPUCommandBuffer;
        _deferCommandBuffer(engine, a, () => _deferCommandBuffer(engine, b));
        engine._flushPendingWork!(engine);
        expect(device.queue.submit).toHaveBeenCalledWith([a]);
        expect(vi.getTimerCount()).toBe(1);
        vi.runAllTimers();
        expect(device.queue.submit).toHaveBeenLastCalledWith([b]);
    });

    it("submits on stop without abandoning admitted work or leaking timers", () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        const after = vi.fn(),
            abandon = vi.fn();
        const command = {} as GPUCommandBuffer;
        _deferCommandBuffer(engine, command, after, abandon);
        stopEngine(engine);
        expect(device.queue.submit).toHaveBeenCalledWith([command]);
        expect(after).toHaveBeenCalledOnce();
        expect(abandon).not.toHaveBeenCalled();
        expect(vi.getTimerCount()).toBe(0);
        _deferCommandBuffer(engine, {} as GPUCommandBuffer);
        vi.runAllTimers();
        expect(device.queue.submit).toHaveBeenCalledTimes(2);
    });

    it("flushes admitted work before requesting the idle fence", async () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        const command = {} as GPUCommandBuffer;
        _deferCommandBuffer(engine, command);
        vi.mocked(device.queue.onSubmittedWorkDone).mockImplementationOnce(async () => {
            expect(device.queue.submit).toHaveBeenCalledWith([command]);
            expect(engine._flushPendingWork).toBeUndefined();
        });
        await waitForGpuIdle(engine);
        expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledOnce();
        expect(vi.getTimerCount()).toBe(0);
    });

    it("abandons mismatched pending work while submitting the current device's frame", () => {
        vi.useFakeTimers();
        const old = setup(),
            current = setup();
        const after = vi.fn(),
            abandon = vi.fn();
        const command = {} as GPUCommandBuffer,
            frame = {} as GPUCommandBuffer;
        _deferCommandBuffer(old.engine, command, after, abandon);
        old.engine._device = current.device;
        old.engine._flushPendingWork!(old.engine, [frame]);
        expect(abandon).toHaveBeenCalledOnce();
        expect(abandon.mock.calls[0]![0]).toBeInstanceOf(Error);
        expect(after).not.toHaveBeenCalled();
        expect(old.device.queue.submit).not.toHaveBeenCalled();
        expect(current.device.queue.submit).toHaveBeenCalledWith([frame]);
        expect(old.engine._flushPendingWork).toBeUndefined();
        expect(old.engine._pendingWork!.pending).toEqual([]);
        expect(vi.getTimerCount()).toBe(0);
    });

    it("abandons on loss and rejects reuse of a lost device", async () => {
        vi.useFakeTimers();
        const { engine, device, lose } = setup();
        const abandon = vi.fn();
        _deferCommandBuffer(engine, {} as GPUCommandBuffer, undefined, abandon);
        lose();
        await Promise.resolve();
        expect(device.queue.submit).not.toHaveBeenCalled();
        expect(abandon).toHaveBeenCalledOnce();
        expect(vi.getTimerCount()).toBe(0);
        expect(() => _deferCommandBuffer(engine, {} as GPUCommandBuffer)).toThrow(/lost device/);
    });

    it("ignores stale device-loss notifications after replacement", async () => {
        vi.useFakeTimers();
        const old = setup(),
            replacement = setup();
        const abandon = vi.fn();
        _deferCommandBuffer(old.engine, {} as GPUCommandBuffer, undefined, abandon);
        old.engine._device = replacement.device;
        const command = {} as GPUCommandBuffer;
        _deferCommandBuffer(old.engine, command);
        expect(abandon).toHaveBeenCalledOnce();
        old.lose();
        await Promise.resolve();
        vi.runAllTimers();
        expect(old.device.queue.submit).not.toHaveBeenCalled();
        expect(replacement.device.queue.submit).toHaveBeenCalledWith([command]);
    });

    it("flushes a frame with zero rendering contexts", () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        engine._surfaces = [engine];
        engine._renderingContexts = [];
        const command = {} as GPUCommandBuffer;
        _deferCommandBuffer(engine, command);
        renderFrame(engine, 0);
        expect(device.queue.submit).toHaveBeenCalledWith([command]);
        expect(engine.drawCallCount).toBe(0);
    });

    it("rejects detached work when submission throws", () => {
        vi.useFakeTimers();
        const { engine, device } = setup();
        vi.mocked(device.queue.submit).mockImplementationOnce(() => {
            throw new Error("submit failed");
        });
        const abandon = vi.fn(),
            after = vi.fn();
        _deferCommandBuffer(engine, {} as GPUCommandBuffer, after, abandon);
        expect(() => engine._flushPendingWork!(engine)).toThrow("submit failed");
        expect(abandon).toHaveBeenCalledOnce();
        expect(after).not.toHaveBeenCalled();
        expect(engine._flushPendingWork).toBeUndefined();
        expect(vi.getTimerCount()).toBe(0);
    });
});
