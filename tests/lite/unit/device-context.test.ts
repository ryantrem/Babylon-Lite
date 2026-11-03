import { afterEach, describe, expect, it, vi } from "vitest";
import { createDevice, _installAdapterOptions } from "../../../packages/babylon-lite/src/engine/device.js";
import { disposeDevice } from "../../../packages/babylon-lite/src/engine/device-dispose.js";
import { createEngine, renderFrame } from "../../../packages/babylon-lite/src/engine/engine.js";
import { disposeEngine } from "../../../packages/babylon-lite/src/engine/engine-dispose.js";
import { _enableDeviceLostRecovery } from "../../../packages/babylon-lite/src/engine/device-lost-recovery.js";
import { recoverDeviceContext } from "../../../packages/babylon-lite/src/engine/device-lost-device-recovery.js";
import { _resetMatrixAllocatorForTests } from "../../../packages/babylon-lite/src/math/_matrix-allocator.js";
import { onEngineGpuError } from "../../../packages/babylon-lite/src/engine/engine-gpu-events.js";
import { loadTexture2D, clearTexture2DCache } from "../../../packages/babylon-lite/src/texture/texture-2d.js";
import { createTexture2DFromPixels } from "../../../packages/babylon-lite/src/texture/pixels-texture.js";
import { acquireTexture, _textureOwners } from "../../../packages/babylon-lite/src/resource/gpu-pool.js";

function nativeDevice(features: GPUFeatureName[] = []) {
    let lose!: (info: GPUDeviceLostInfo) => void;
    const destroy = vi.fn();
    const device = {
        features: new Set(features),
        limits: { maxBufferSize: 1024 },
        lost: new Promise<GPUDeviceLostInfo>((resolve) => {
            lose = resolve;
        }),
        destroy,
        createTexture: vi.fn(() => ({ createView: () => ({}), destroy: vi.fn() })),
        createSampler: vi.fn(() => ({})),
        queue: {
            onSubmittedWorkDone: vi.fn(async () => undefined),
            copyExternalImageToTexture: vi.fn(),
            writeTexture: vi.fn(),
        },
    } as unknown as GPUDevice;
    return { device, destroy, lose };
}

function gpu(features: GPUFeatureName[] = []) {
    const native = nativeDevice(features);
    const adapterFeatures = new Set(features);
    const requestDevice = vi.fn(async (descriptor: GPUDeviceDescriptor) => {
        const missing = [...(descriptor.requiredFeatures ?? [])].find((feature) => !adapterFeatures.has(feature));
        if (missing) {
            throw new TypeError(`Unsupported feature: ${missing}`);
        }
        return native.device;
    });
    const adapter = { features: adapterFeatures, requestDevice };
    const requestAdapter = vi.fn(async (_options: GPURequestAdapterOptions): Promise<typeof adapter | null> => adapter);
    vi.stubGlobal("navigator", { gpu: { requestAdapter, getPreferredCanvasFormat: () => "bgra8unorm" } });
    return { ...native, requestAdapter, requestDevice };
}

function lossInfo(): GPUDeviceLostInfo {
    return { reason: "unknown", message: "" } as GPUDeviceLostInfo;
}

function canvas() {
    const context = {
        configure: vi.fn(),
        unconfigure: vi.fn(),
        getCurrentTexture: vi.fn(() => ({ width: 4, height: 4, createView: () => ({}) })),
    };
    const element = { width: 4, height: 4, getContext: vi.fn(() => context) } as unknown as OffscreenCanvas;
    return { element, context };
}

afterEach(() => {
    _resetMatrixAllocatorForTests();
    _installAdapterOptions(() => ({}));
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

describe("device acquisition", () => {
    it("acquires without touching a canvas and merges supported features without duplicates", async () => {
        const mock = gpu(["timestamp-query", "shader-f16"]);
        const context = await createDevice({ powerPreference: "low-power", requiredFeatures: ["shader-f16", "timestamp-query", "shader-f16"] });
        expect(mock.requestAdapter).toHaveBeenCalledExactlyOnceWith({ powerPreference: "low-power" });
        expect(mock.requestDevice).toHaveBeenCalledExactlyOnceWith({ requiredFeatures: ["timestamp-query", "shader-f16"], requiredLimits: undefined });
        expect(context.features).toBe(mock.device.features);
        expect(context.limits).toBe(mock.device.limits);
        expect(Array.from(context._device.features)).toEqual(["timestamp-query", "shader-f16"]);
        expect(context._engines).toEqual([]);
        disposeDevice(context);
        disposeDevice(context);
        expect(mock.destroy).toHaveBeenCalledOnce();
    });

    it("snapshots features, limits, and adapter policy before awaiting", async () => {
        const mock = gpu(["shader-f16"]);
        const requiredFeatures: GPUFeatureName[] = ["shader-f16"];
        const requiredLimits = { maxBufferSize: 4096 };
        _installAdapterOptions(() => ({ forceFallbackAdapter: true }));
        const pending = createDevice({ requiredFeatures, requiredLimits });
        requiredFeatures.length = 0;
        requiredLimits.maxBufferSize = 8192;
        _installAdapterOptions(() => ({}));
        const context = await pending;
        expect(mock.requestDevice).toHaveBeenCalledWith({ requiredFeatures: ["shader-f16"], requiredLimits: { maxBufferSize: 4096 } });
        expect(context._adapterOptions).toEqual({ powerPreference: "high-performance", forceFallbackAdapter: true });
        expect(Array.from(context._device.features)).toEqual(["shader-f16"]);
        expect(context._requiredLimits).toEqual({ maxBufferSize: 4096 });
        disposeDevice(context);
    });

    it("forwards unsupported required features so requestDevice rejects them", async () => {
        const mock = gpu();
        await expect(createDevice({ requiredFeatures: ["shader-f16"] })).rejects.toThrow(/shader-f16/);
        expect(mock.requestDevice).toHaveBeenCalledExactlyOnceWith({ requiredFeatures: ["shader-f16"], requiredLimits: undefined });
    });

    it("reports missing adapters and propagates adapter/device request failures", async () => {
        const mock = gpu();
        mock.requestAdapter.mockResolvedValueOnce(null);
        await expect(createDevice()).rejects.toThrow("WebGPU adapter not available");
        mock.requestAdapter.mockRejectedValueOnce(new Error("adapter failed"));
        await expect(createDevice()).rejects.toThrow("adapter failed");
        mock.requestDevice.mockRejectedValueOnce(new Error("device failed"));
        await expect(createDevice()).rejects.toThrow("device failed");
    });

    it("rejects a device lost while the application bundle was still loading", async () => {
        const mock = gpu();
        const device = await createDevice();
        mock.lose(lossInfo());
        await Promise.resolve();
        await expect(createEngine(canvas().element, { device })).rejects.toThrow(/lost/);
        disposeDevice(device);
    });
});

describe("engine device ownership", () => {
    it("shares one device across independent engines and leaves explicit disposal to the caller", async () => {
        const mock = gpu();
        const device = await createDevice();
        const a = canvas();
        const b = canvas();
        const first = await createEngine(a.element, { device });
        const second = await createEngine(b.element, { device, msaaSamples: 1 });
        expect(first.device).toBe(device);
        expect(second._device).toBe(first._device);
        expect(mock.requestAdapter).toHaveBeenCalledOnce();
        expect(mock.requestDevice).toHaveBeenCalledOnce();
        expect(() => disposeDevice(device)).toThrow(/engines are disposed/);
        disposeEngine(first);
        disposeEngine(first);
        expect(mock.destroy).not.toHaveBeenCalled();
        expect(a.context.unconfigure).toHaveBeenCalledOnce();
        expect(b.context.unconfigure).not.toHaveBeenCalled();
        renderFrame(second, 16);
        disposeEngine(second);
        expect(mock.destroy).not.toHaveBeenCalled();
        disposeDevice(device);
        expect(mock.destroy).toHaveBeenCalledOnce();
    });

    it("keeps a convenience device private to its engine", async () => {
        const mock = gpu();
        const engine = await createEngine(canvas().element);
        expect(engine.device._engines).toBeUndefined();
        expect(engine.device._attach).toBeUndefined();
        expect(() => disposeDevice(engine.device)).toThrow(/createDevice/);
        disposeEngine(engine);
        disposeEngine(engine);
        expect(mock.destroy).toHaveBeenCalledOnce();
    });

    it("rejects disposed devices, conflicting acquisition options, and invalid precision before acquiring", async () => {
        const mock = gpu();
        const device = await createDevice();
        await expect(
            // @ts-expect-error Limits must be configured on createDevice.
            createEngine(canvas().element, { device, requiredLimits: { maxBufferSize: 1024 } })
        ).rejects.toThrow(/createDevice/);
        await expect(
            // @ts-expect-error Features must be configured on createDevice.
            createEngine(canvas().element, { device, requiredFeatures: ["shader-f16"] })
        ).rejects.toThrow(/createDevice/);
        disposeDevice(device);
        await expect(createEngine(canvas().element, { device })).rejects.toThrow(/disposed/);
        await expect(createEngine(canvas().element, { useFloatingOrigin: true })).rejects.toThrow(/useHighPrecisionMatrix/);
        expect(mock.requestAdapter).toHaveBeenCalledOnce();
    });

    it("unconfigures the canvas when attachment is rejected without claiming a borrowed device", async () => {
        const mock = gpu();
        const device = await createDevice();
        const target = canvas();
        await expect(
            // @ts-expect-error Limits must be configured on createDevice.
            createEngine(target.element, { device, requiredLimits: { maxBufferSize: 1024 } })
        ).rejects.toThrow(/createDevice/);
        expect(target.context.unconfigure).toHaveBeenCalledOnce();
        expect(device._engines).toEqual([]);
        expect(mock.destroy).not.toHaveBeenCalled();
        disposeDevice(device);
        expect(mock.destroy).toHaveBeenCalledOnce();
    });

    it("rechecks disposal after asynchronous precision initialization", async () => {
        gpu();
        const device = await createDevice();
        const pending = createEngine(canvas().element, { device, useHighPrecisionMatrix: true });
        disposeDevice(device);
        await expect(pending).rejects.toThrow(/disposed/);
        expect(device._engines).toEqual([]);
    });

    it("detaches optional listeners and recovery registrations while a shared device remains alive", async () => {
        const mock = gpu();
        const removeEventListener = vi.fn();
        Object.assign(mock.device, { addEventListener: vi.fn(), removeEventListener });
        const device = await createDevice();
        const engine = await createEngine(canvas().element, { device });
        const disable = vi.fn();
        const handle = _enableDeviceLostRecovery(engine, { _kind: "test", _recover: vi.fn(), _disable: disable });
        onEngineGpuError(engine, vi.fn());
        disposeEngine(engine);
        handle.disable();
        expect(disable).toHaveBeenCalledOnce();
        expect(removeEventListener).toHaveBeenCalledOnce();
        expect(engine._deviceLostRecovery?._registrations).toHaveLength(0);
        expect(mock.destroy).not.toHaveBeenCalled();
        disposeDevice(device);
    });

    it("keeps mutable URL texture wrappers and cache invalidation engine-local", async () => {
        gpu();
        const device = await createDevice();
        const first = await createEngine(canvas().element, { device });
        const second = await createEngine(canvas().element, { device });
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({ blob: async () => new Blob() }))
        );
        vi.stubGlobal(
            "createImageBitmap",
            vi.fn(async () => ({ width: 1, height: 1, close: vi.fn() }))
        );
        const a = loadTexture2D(first, "texture.png", { mipMaps: false });
        const b = loadTexture2D(second, "texture.png", { mipMaps: false });
        expect(loadTexture2D(first, "texture.png", { mipMaps: false })).toBe(a);
        expect(await a).not.toBe(await b);
        clearTexture2DCache(first);
        expect(loadTexture2D(second, "texture.png", { mipMaps: false })).toBe(b);
        const reloaded = loadTexture2D(first, "texture.png", { mipMaps: false });
        expect(reloaded).not.toBe(a);
        await reloaded;
        disposeEngine(first);
        disposeEngine(second);
        disposeDevice(device);
    });
});

describe("shared device recovery", () => {
    it("replaces once, rebuilds every opted-in engine, and leaves other engines stopped", async () => {
        const mock = gpu(["timestamp-query"]);
        const device = await createDevice({ powerPreference: "low-power", requiredLimits: { maxBufferSize: 2048 } });
        const first = await createEngine(canvas().element, { device });
        const second = await createEngine(canvas().element, { device });
        const unprotected = await createEngine(canvas().element, { device });
        const replacement = nativeDevice(["timestamp-query"]);
        mock.requestDevice.mockResolvedValue(replacement.device);
        const recoverFirst = vi.fn(async () => {
            expect(second._deviceUnavailable).toBe(true);
        });
        const recoverSecond = vi.fn();
        const recovered = vi.fn();
        _enableDeviceLostRecovery(first, { _kind: "first", _recover: recoverFirst, _onRecovered: recovered });
        _enableDeviceLostRecovery(second, { _kind: "second", _recover: recoverSecond, _onRecovered: recovered });
        const pending = recoverDeviceContext(device, mock.device, lossInfo(), first);
        expect(recoverDeviceContext(device, mock.device, lossInfo(), second)).toBe(pending);
        await expect(createEngine(canvas().element, { device })).rejects.toThrow(/recovering/);
        await pending;
        expect(mock.requestDevice).toHaveBeenCalledTimes(2);
        expect(mock.requestDevice).toHaveBeenLastCalledWith({ requiredFeatures: ["timestamp-query"], requiredLimits: { maxBufferSize: 2048 } });
        expect(mock.requestAdapter).toHaveBeenLastCalledWith({ powerPreference: "low-power" });
        expect(first._device).toBe(replacement.device);
        expect(second._device).toBe(replacement.device);
        expect(device.features).toBe(replacement.device.features);
        expect(recoverFirst).toHaveBeenCalledOnce();
        expect(recoverSecond).toHaveBeenCalledOnce();
        expect(recovered).toHaveBeenCalledTimes(2);
        expect(unprotected._deviceUnavailable).toBe(true);
        expect(unprotected._renderFn).toBeNull();
        disposeEngine(first);
        disposeEngine(second);
        disposeEngine(unprotected);
        disposeDevice(device);
        expect(replacement.destroy).toHaveBeenCalledOnce();
    });

    it("dispatches a real device loss through engine registrations only once", async () => {
        const mock = gpu();
        const device = await createDevice();
        const first = await createEngine(canvas().element, { device });
        const second = await createEngine(canvas().element, { device });
        const replacement = nativeDevice();
        mock.requestDevice.mockResolvedValue(replacement.device);
        let finished!: () => void;
        const done = new Promise<void>((resolve) => {
            finished = resolve;
        });
        const recovered = vi.fn(() => {
            if (recovered.mock.calls.length === 2) {
                finished();
            }
        });
        _enableDeviceLostRecovery(first, { _kind: "first", _recover: vi.fn(), _onRecovered: recovered });
        _enableDeviceLostRecovery(second, { _kind: "second", _recover: vi.fn(), _onRecovered: recovered });
        mock.lose(lossInfo());
        await done;
        await device._recovery;
        expect(mock.requestDevice).toHaveBeenCalledTimes(2);
        disposeEngine(first);
        disposeEngine(second);
        disposeDevice(device);
    });

    it("reports replacement failure to every participating engine without restarting", async () => {
        const mock = gpu();
        const device = await createDevice();
        const engines = await Promise.all([createEngine(canvas().element, { device }), createEngine(canvas().element, { device })]);
        const failed = vi.fn();
        for (const engine of engines) {
            _enableDeviceLostRecovery(engine, { _kind: "test", _recover: vi.fn(), _onRecoveryFailed: failed });
        }
        mock.requestDevice.mockRejectedValue(new Error("replacement unavailable"));
        await recoverDeviceContext(device, mock.device, lossInfo(), engines[0]);
        expect(failed).toHaveBeenCalledTimes(2);
        expect(failed.mock.calls[0]![0]).toBeInstanceOf(Error);
        await expect(createEngine(canvas().element, { device })).rejects.toThrow(/lost/);
        for (const engine of engines) {
            expect(engine._deviceUnavailable).toBe(true);
            expect(engine._renderFn).toBeNull();
            disposeEngine(engine);
        }
        disposeDevice(device);
    });

    it("does not restart either engine until both have finished rebuilding", async () => {
        const mock = gpu();
        const device = await createDevice();
        const first = await createEngine(canvas().element, { device });
        const second = await createEngine(canvas().element, { device });
        const frames: FrameRequestCallback[] = [];
        vi.stubGlobal(
            "requestAnimationFrame",
            vi.fn((callback: FrameRequestCallback) => frames.push(callback))
        );
        vi.stubGlobal("cancelAnimationFrame", vi.fn());
        first._renderFn = vi.fn();
        second._renderFn = vi.fn();
        const replacement = nativeDevice();
        mock.requestDevice.mockResolvedValue(replacement.device);
        let finishRebuild!: () => void;
        const rebuilding = new Promise<void>((resolve) => {
            finishRebuild = resolve;
        });
        const recoverSecond = vi.fn(() => rebuilding);
        _enableDeviceLostRecovery(first, { _kind: "test", _recover: vi.fn() });
        _enableDeviceLostRecovery(second, { _kind: "test", _recover: recoverSecond });
        const pending = recoverDeviceContext(device, mock.device, lossInfo(), second);
        await vi.waitFor(() => expect(recoverSecond).toHaveBeenCalledOnce());
        expect(frames).toHaveLength(0);
        finishRebuild();
        await vi.waitFor(() => expect(frames).toHaveLength(2));
        for (const callback of frames.splice(0)) {
            callback(16);
        }
        await pending;
        disposeEngine(first);
        disposeEngine(second);
        disposeDevice(device);
    });

    it("releases a private replacement if its engine is disposed while acquisition is pending", async () => {
        const mock = gpu();
        const engine = await createEngine(canvas().element);
        const device = engine.device;
        const recover = vi.fn();
        _enableDeviceLostRecovery(engine, { _kind: "test", _recover: recover });
        const replacement = nativeDevice();
        let finishRequest!: (device: GPUDevice) => void;
        mock.requestDevice.mockImplementation(
            () =>
                new Promise<GPUDevice>((resolve) => {
                    finishRequest = resolve;
                })
        );
        const pending = recoverDeviceContext(device, mock.device, lossInfo(), engine);
        await vi.waitFor(() => expect(mock.requestDevice).toHaveBeenCalledTimes(2));
        disposeEngine(engine);
        expect(() => disposeDevice(device)).toThrow(/recovery/);
        finishRequest(replacement.device);
        await pending;
        expect(recover).not.toHaveBeenCalled();
        expect(replacement.destroy).toHaveBeenCalledOnce();
    });

    it("settles shared texture ownership after all engines reestablish their references", async () => {
        const mock = gpu();
        const device = await createDevice();
        const first = await createEngine(canvas().element, { device });
        const second = await createEngine(canvas().element, { device });
        const pixels = new Uint8Array([255, 255, 255, 255]);
        const texture = createTexture2DFromPixels(first, pixels, 1, 1);
        texture._recoverySource = { kind: "pixels", data: pixels, width: 1, height: 1, options: {} };
        acquireTexture(texture);
        expect(_textureOwners(texture)).toBe(2);
        for (const engine of [first, second]) {
            _enableDeviceLostRecovery(engine, { _kind: "test", _recover: () => acquireTexture(texture) });
            engine._deviceLostRecovery!._textures.add(new WeakRef(texture));
        }
        const replacement = nativeDevice();
        mock.requestDevice.mockResolvedValue(replacement.device);
        await recoverDeviceContext(device, mock.device, lossInfo(), first);
        expect(_textureOwners(texture)).toBe(2);
        expect(replacement.device.createTexture).toHaveBeenCalledOnce();
        disposeEngine(first);
        disposeEngine(second);
        disposeDevice(device);
    });

    it("finishes recovery if disposal cancels the first restarted frame", async () => {
        const mock = gpu();
        const engine = await createEngine(canvas().element);
        const device = engine.device;
        const requestFrame = vi.fn(() => 1);
        vi.stubGlobal("requestAnimationFrame", requestFrame);
        vi.stubGlobal("cancelAnimationFrame", vi.fn());
        engine._renderFn = vi.fn();
        const recovered = vi.fn();
        _enableDeviceLostRecovery(engine, { _kind: "test", _recover: vi.fn(), _onRecovered: recovered });
        const replacement = nativeDevice();
        mock.requestDevice.mockResolvedValue(replacement.device);
        const pending = recoverDeviceContext(device, mock.device, lossInfo(), engine);
        await vi.waitFor(() => expect(requestFrame).toHaveBeenCalledOnce());
        disposeEngine(engine);
        await pending;
        expect(recovered).not.toHaveBeenCalled();
        expect(replacement.destroy).toHaveBeenCalledOnce();
    });
});
