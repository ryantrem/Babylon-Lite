import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { renderFrame, stopEngine, waitForGpuIdle } from "../../../packages/babylon-lite/src/engine/engine";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import { createComputeTextLayer } from "../../../packages/babylon-lite/src/text/compute-text-layer";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { createTextRenderer, disposeTextRenderer, registerTextRenderer } from "../../../packages/babylon-lite/src/text/text-renderer";
import type { TextRenderer } from "../../../packages/babylon-lite/src/text/text-renderer";
import { clearTextPipelineCache, getTextPipelineCache } from "../../../packages/babylon-lite/src/text/_gpu/text-pipeline";
import { getComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import * as readback from "../../../packages/babylon-lite/src/compute/compute-task-readback";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { fixture } from "./fixtures";
import type { Fixture } from "./fixtures";
import { syntheticFont } from "./synthetic";

describe("native resident deferred frame submission", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    let target: GPUTexture;
    const data: ComputeTextData[] = [];
    const renderers: TextRenderer[] = [];
    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
        target = native.device.createTexture({ size: [64, 64], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const surfaces = [native.engine];
        Object.assign(native.engine, {
            engine: native.engine,
            _surfaces: surfaces,
            surfaces,
            canvas: { width: 64, height: 64 },
            format: "rgba8unorm",
            scRT: {},
            _context: { getCurrentTexture: () => target },
            _renderingContexts: [],
        });
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        renderers.splice(0).forEach(disposeTextRenderer);
        data.splice(0).forEach(disposeComputeTextData);
        await assertNativeClean(native);
    });
    afterAll(async () => {
        getTextPipelineCache(native.engine)?._quadVertexBuffer.destroy();
        clearTextPipelineCache(native.engine);
        target.destroy();
        disposeGlyphStorage(storage);
        await native.close();
    });
    function block(face: Fixture = fixture("Roboto-Regular.ttf")): ComputeTextData {
        const owned = createComputeTextData(native.engine, face.font, storage, `fused-${face.file}`, { fontSizePx: 24 });
        data.push(owned);
        const renderer = createTextRenderer(native.engine, { layers: [createComputeTextLayer(owned)] });
        renderers.push(renderer);
        registerTextRenderer(renderer);
        return owned;
    }
    async function bytes(owned: ComputeTextData) {
        return {
            instances: new Uint8Array(await readStorageBuffer(owned._instances)),
            indirect: new Uint8Array(await readStorageBuffer(owned._indirect)),
            status: new Uint8Array(await readStorageBuffer(owned._statusBuffer)),
        };
    }
    const layout = { maxWidthPx: 100, lineHeightPx: 32 };
    const initial = ["office HH AV To", "office HH AV To"];
    async function warm(owned: ComputeTextData): Promise<void> {
        const update = await updateComputeTextParagraphs(owned, initial, layout);
        renderFrame(native.engine, 0);
        await update.completed;
    }

    it("fuses one update and the real cached outline render, with exact immediate-path bytes", async () => {
        const owned = block();
        await warm(owned);
        const immediate = vi.spyOn(readback, "submitDeferredComputeTaskResult").mockImplementation(readback.submitComputeTaskResult);
        await (
            await updateComputeTextParagraphs(owned, ["HH AV", "office"], layout)
        ).completed;
        renderFrame(native.engine, 0);
        const expected = await bytes(owned);
        immediate.mockRestore();
        await warm(owned);
        const submits = vi.spyOn(native.device.queue, "submit");
        const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
        const timer = vi.fn();
        native.engine._gpuTaskTimerResolve = timer;
        const update = await updateComputeTextParagraphs(owned, ["HH AV", "office"], layout);
        expect(submits).not.toHaveBeenCalled();
        expect(maps).not.toHaveBeenCalled();
        renderFrame(native.engine, 0);
        expect(submits).toHaveBeenCalledOnce();
        expect(submits.mock.calls[0]![0]).toHaveLength(2);
        expect(maps.mock.calls.map((call) => call.slice(0, 3))).toEqual([[GPUMapMode.READ, 0, 4]]);
        expect(timer).toHaveBeenCalledOnce();
        native.engine._gpuTaskTimerResolve = undefined;
        await update.completed;
        expect(await bytes(owned)).toEqual(expected);
    });

    it("settles completed without rendering through exactly one fallback submit", async () => {
        const owned = block();
        await warm(owned);
        const submits = vi.spyOn(native.device.queue, "submit");
        const update = await updateComputeTextData(owned, ["HH"]);
        expect(submits).not.toHaveBeenCalled();
        await update.completed;
        expect(submits).toHaveBeenCalledOnce();
        expect(native.engine._flushPendingWork).toBeUndefined();
    });

    it.each([2, 3])("preserves all %i updates and costs N submits before one frame", async (count) => {
        const owned = block();
        await warm(owned);
        const snapshots = [
            ["HH AV", "office"],
            ["To", "HH AV"],
            ["office", "To"],
        ];
        const immediate = vi.spyOn(readback, "submitDeferredComputeTaskResult").mockImplementation(readback.submitComputeTaskResult);
        for (const snapshot of snapshots.slice(0, count)) await (await updateComputeTextParagraphs(owned, snapshot, layout)).completed;
        const expected = await bytes(owned);
        immediate.mockRestore();
        await warm(owned);
        const submits = vi.spyOn(native.device.queue, "submit");
        const updates = [];
        for (const snapshot of snapshots.slice(0, count)) updates.push(await updateComputeTextParagraphs(owned, snapshot, layout));
        expect(submits).toHaveBeenCalledTimes(count - 1);
        renderFrame(native.engine, 0);
        expect(submits).toHaveBeenCalledTimes(count);
        await Promise.all(updates.map((update) => update.completed));
        expect(await bytes(owned)).toEqual(expected);
    });

    it.each([1, 2])("preserves fused status failure %i and invalidates dependent paragraph caches", async (status) => {
        const owned = block();
        await warm(owned);
        const stage = getComputeTextParagraphStage(owned)!;
        const original = readback.submitDeferredComputeTaskResult;
        vi.spyOn(readback, "submitDeferredComputeTaskResult").mockImplementationOnce((tasks, buffer, length, consume, staging) => {
            const stride = 4 + stage._words[1]! * 6;
            updateStorageBuffer(native.engine, owned._shaped, Int32Array.of(status === 1 ? 1 : 0, status === 2 ? 0x7fffffff : 0, 0, 0), stage._words[8]! * stride * 4);
            return original(tasks.slice(-1), buffer, length, consume, staging);
        });
        const update = await updateComputeTextParagraphs(owned, ["HH", initial[1]!], layout);
        renderFrame(native.engine, 0);
        await expect(update.completed).rejects.toThrow(status === 1 ? /unsupported lookup/ : /reserved capacity/);
        expect(stage._cache).toBeUndefined();
        expect([...new Uint32Array(await readStorageBuffer(owned._indirect))]).toEqual([6, 0, 0, 0]);
    });

    it("propagates a real unsupported shaping lookup after a fused draw", async () => {
        const owned = block(syntheticFont("pair-reject"));
        const first = await updateComputeTextParagraphs(owned, ["A", "BA", "C"], layout);
        renderFrame(native.engine, 0);
        await first.completed;
        const update = await updateComputeTextParagraphs(owned, ["A", "BC", "C"], layout);
        renderFrame(native.engine, 0);
        await expect(update.completed).rejects.toThrow(/unsupported lookup/);
        expect(getComputeTextParagraphStage(owned)!._cache).toBeUndefined();
    });

    it("abandons unsubmitted work on actual device loss without starting a status map", async () => {
        const owned = await nativeHarness();
        const atlas = createGlyphStorage();
        const text = createComputeTextData(owned.engine, fixture("Roboto-Regular.ttf").font, atlas, "pending-loss");
        try {
            const update = await updateComputeTextData(text, ["HH"]);
            const submits = vi.spyOn(owned.device.queue, "submit");
            const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
            owned.destroyExpected = true;
            owned.device.destroy();
            await owned.device.lost;
            await expect(update.completed).rejects.toThrow(/device lost/);
            expect(submits).not.toHaveBeenCalled();
            expect(maps).not.toHaveBeenCalled();
            expect(owned.engine._flushPendingWork).toBeUndefined();
            expect(owned.errors).toEqual([]);
        } finally {
            disposeComputeTextData(text);
            disposeGlyphStorage(atlas);
            await owned.close();
        }
    });

    it("flushes before storage readback without requiring render or fallback", async () => {
        const owned = block();
        await warm(owned);
        const immediate = vi.spyOn(readback, "submitDeferredComputeTaskResult").mockImplementation(readback.submitComputeTaskResult);
        await (
            await updateComputeTextData(owned, ["HH"])
        ).completed;
        const expected = new Uint8Array(await readStorageBuffer(owned._instances));
        immediate.mockRestore();
        await warm(owned);
        const submits = vi.spyOn(native.device.queue, "submit");
        const update = await updateComputeTextData(owned, ["HH"]);
        expect(submits).not.toHaveBeenCalled();
        const instances = readStorageBuffer(owned._instances);
        expect(submits).toHaveBeenCalledTimes(2);
        expect(new Uint8Array(await instances)).toEqual(expected);
        expect([...new Uint32Array(await readStorageBuffer(owned._indirect))]).toEqual([6, 2, 0, 0]);
        await update.completed;
    });

    it("flushes before disposal destroys buffers and settles its pending completion", async () => {
        const owned = block();
        await warm(owned);
        const submits = vi.spyOn(native.device.queue, "submit");
        const update = await updateComputeTextData(owned, ["HH"]);
        disposeComputeTextData(owned);
        expect(submits).toHaveBeenCalledOnce();
        await expect(update.completed).rejects.toThrow(/disposed/);
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
    });

    it("flushes pending GPU work when the frame has no rendering contexts", async () => {
        const owned = block();
        await warm(owned);
        renderers.splice(0).forEach(disposeTextRenderer);
        const submits = vi.spyOn(native.device.queue, "submit");
        const update = await updateComputeTextData(owned, ["HH"]);
        renderFrame(native.engine, 0);
        expect(submits).toHaveBeenCalledOnce();
        expect(submits.mock.calls[0]![0]).toHaveLength(1);
        await update.completed;
    });

    it.each(["idle", "stop"] as const)("completes admitted work through %s without rendering", async (boundary) => {
        const owned = block();
        await warm(owned);
        const immediate = vi.spyOn(readback, "submitDeferredComputeTaskResult").mockImplementation(readback.submitComputeTaskResult);
        await (
            await updateComputeTextData(owned, ["HH"])
        ).completed;
        const expected = await bytes(owned);
        immediate.mockRestore();
        await warm(owned);
        const submits = vi.spyOn(native.device.queue, "submit");
        const fence = vi.spyOn(native.device.queue, "onSubmittedWorkDone");
        const update = await updateComputeTextData(owned, ["HH"]);
        expect(submits).not.toHaveBeenCalled();
        if (boundary === "idle") {
            const idle = waitForGpuIdle(native.engine);
            expect(submits).toHaveBeenCalledOnce();
            expect(fence).toHaveBeenCalledOnce();
            expect(submits.mock.invocationCallOrder[0]!).toBeLessThan(fence.mock.invocationCallOrder[0]!);
            await idle;
            expect(await bytes(owned)).toEqual(expected);
        } else {
            stopEngine(native.engine);
            expect(submits).toHaveBeenCalledOnce();
            await expect(update.completed).resolves.toBeUndefined();
            expect(await bytes(owned)).toEqual(expected);
        }
        await expect(update.completed).resolves.toBeUndefined();
        expect(native.engine._flushPendingWork).toBeUndefined();
    });
});
