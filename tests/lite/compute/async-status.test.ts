import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createComputeTextLayer } from "../../../packages/babylon-lite/src/text/compute-text-layer";
import { clearTextPipelineCache, getOrCreateTextPipeline } from "../../../packages/babylon-lite/src/text/_gpu/text-pipeline";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import * as shapeBatch from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape-batch";
import { fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

describe("independent asynchronous resident status", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    const live: ComputeTextData[] = [];
    const unhandled: unknown[] = [];
    const onUnhandled = (error: unknown) => unhandled.push(error);
    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
        process.on("unhandledRejection", onUnhandled);
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        live.splice(0).forEach(disposeComputeTextData);
        await assertNativeClean(native);
        await new Promise<void>((resolve) => setImmediate(resolve));
        expect(unhandled).toEqual([]);
    });
    afterAll(async () => {
        process.off("unhandledRejection", onUnhandled);
        disposeGlyphStorage(storage);
        await native.close();
    });
    function block(): ComputeTextData {
        const data = createComputeTextData(native.engine, fixture("Roboto-Regular.ttf").font, storage, "async-status");
        live.push(data);
        return data;
    }
    function holdMaps() {
        const original = GPUBuffer.prototype.mapAsync;
        const releases: (() => void)[] = [];
        const buffers: GPUBuffer[] = [];
        const pendingMaps: Promise<void>[] = [];
        vi.spyOn(GPUBuffer.prototype, "mapAsync").mockImplementation(function (this: GPUBuffer, mode, offset, size) {
            const mapped = original.call(this, mode, offset, size);
            if (size !== 4) {
                return mapped;
            }
            buffers.push(this);
            pendingMaps.push(mapped);
            const held = new Promise<undefined>((resolve) => releases.push(() => resolve(undefined)));
            return mapped.then(() => held);
        });
        return {
            buffers,
            async mapped() {
                native.engine._flushPendingWork?.(native.engine);
                await new Promise<void>((resolve) => setTimeout(resolve, 0));
                await Promise.allSettled(pendingMaps);
            },
            release() {
                native.engine._flushPendingWork?.(native.engine);
                releases.splice(0).forEach((resolve) => resolve());
            },
        };
    }
    function inject(data: ComputeTextData, kind: "overflow" | "unsupported" | "missing"): void {
        // Test-only admission bypass: skip the cross-object shape batch and run the real pack kernel against injected shaped records.
        vi.spyOn(shapeBatch, "addComputeTextShapeSlice").mockImplementationOnce(() => {
            const words = new Int32Array(data._shaped.byteLength / 4);
            words[0] = kind === "unsupported" ? 1 : 0;
            words[1] = kind === "overflow" ? 0x7fffffff : kind === "missing" ? 1 : 0;
            words[4] = data._font.glyphCount;
            updateStorageBuffer(native.engine, data._shaped, words);
        });
    }
    async function drawBytes(data: ComputeTextData) {
        return {
            instances: new Uint8Array(await readStorageBuffer(data._instances)),
            indirect: new Uint32Array(await readStorageBuffer(data._indirect)),
        };
    }
    const messages = {
        overflow: "updateComputeTextData: shaped run exceeds its reserved capacity",
        unsupported: "updateComputeTextData: unsupported lookup",
        missing: "updateComputeTextData: missing atlas slot",
    };

    it("resolves while status mapping remains pending and permits an immediate indirect draw", async () => {
        const data = block();
        const held = holdMaps();
        try {
            const update = await updateComputeTextData(data, ["office", "HH"]);
            expect(data.status).toBe("ready");
            expect(held.buffers).toHaveLength(0);
            let settled = false;
            void update.completed.then(
                () => {
                    settled = true;
                },
                () => undefined
            );
            const uniforms = native.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            const draw = createComputeTextLayer(data)._createDraw(native.engine, uniforms);
            const pipeline = getOrCreateTextPipeline(native.engine, "rgba8unorm", 1, null, false);
            const target = native.device.createTexture({ size: [32, 32], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
            try {
                draw._prepare(pipeline._cache._bindGroupLayout);
                expect(draw._active).toBe(true);
                const encoder = native.device.createCommandEncoder();
                const bundle = native.device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
                bundle.setPipeline(pipeline._pipeline);
                expect(draw._record(bundle, pipeline._cache._quadVertexBuffer, pipeline._pipeline, pipeline._variantPipeline)).toBe(1);
                const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
                pass.executeBundles([bundle.finish()]);
                pass.end();
                native.engine._flushPendingWork?.(native.engine, [encoder.finish()]);
                expect(held.buffers).toHaveLength(1);
                const indirect = await readStorageBuffer(data._indirect);
                expect(new Uint32Array(indirect)[1]).toBeGreaterThan(0);
                expect(settled).toBe(false);
                held.release();
                await update.completed;
            } finally {
                draw._dispose();
                uniforms.destroy();
                target.destroy();
                pipeline._cache._quadVertexBuffer.destroy();
                clearTextPipelineCache(native.engine);
            }
        } finally {
            held.release();
        }
    });

    it.each(["success", "overflow", "unsupported", "missing"] as const)("immediately queued draw bytes equal the status-awaited path: %s", async (kind) => {
        const data = block();
        const held = holdMaps();
        try {
            if (kind !== "success") {
                inject(data, kind);
            }
            const update = await updateComputeTextData(data, ["HH"]);
            const immediate = await drawBytes(data);
            if (kind !== "success") {
                expect([...immediate.indirect]).toEqual([6, 0, 0, 0]);
            }
            held.release();
            if (kind === "success") {
                await update.completed;
            } else {
                await expect(update.completed).rejects.toThrow(messages[kind]);
            }
            expect(await drawBytes(data)).toEqual(immediate);
        } finally {
            held.release();
        }
    });

    it.each([true, false])("reports each overlapping update's own status without clearing the newer draw (first fails: %s)", async (firstFails) => {
        const data = block();
        const held = holdMaps();
        try {
            if (firstFails) {
                inject(data, "unsupported");
            }
            const first = await updateComputeTextData(data, ["HH"]);
            if (!firstFails) {
                inject(data, "missing");
            }
            const second = await updateComputeTextData(data, ["AV"]);
            native.engine._flushPendingWork?.(native.engine);
            expect(held.buffers).toHaveLength(2);
            expect(held.buffers[0]).not.toBe(held.buffers[1]);
            const before = await drawBytes(data);
            const writes = vi.spyOn(native.device.queue, "writeBuffer");
            held.release();
            await expect((firstFails ? first : second).completed).rejects.toThrow(firstFails ? messages.unsupported : messages.missing);
            await (firstFails ? second : first).completed;
            expect(writes).not.toHaveBeenCalled();
            expect(await drawBytes(data)).toEqual(before);
            expect(before.indirect[1]).toBe(firstFails ? 2 : 0);
        } finally {
            held.release();
        }
    });

    it("grows to peak in-flight maps, then reuses the unmapped pool", async () => {
        const data = block();
        const held = holdMaps();
        const create = vi.spyOn(native.device, "createBuffer");
        try {
            const updates = [];
            for (let index = 0; index < 4; index++) {
                updates.push(await updateComputeTextData(data, [index % 2 ? "AV" : "HH"]));
            }
            native.engine._flushPendingWork?.(native.engine);
            expect(new Set(held.buffers).size).toBe(4);
            held.release();
            await Promise.all(updates.map((update) => update.completed));
            const allocations = create.mock.calls.filter(([descriptor]) => descriptor.label === "compute-text-status-readback").length;
            expect(allocations).toBe(4);
            for (let index = 0; index < 4; index++) {
                const update = await updateComputeTextData(data, ["HH"]);
                held.release();
                await update.completed;
            }
            expect(create.mock.calls.filter(([descriptor]) => descriptor.label === "compute-text-status-readback")).toHaveLength(4);
        } finally {
            held.release();
        }
    });

    it.each(["dispose", "loss"] as const)("rejects pending completions and destroys the pool on %s", async (terminal) => {
        const data = block();
        const held = holdMaps();
        try {
            const updates = [await updateComputeTextData(data, ["HH"]), await updateComputeTextData(data, ["AV"])];
            await held.mapped();
            const destroys = held.buffers.map((buffer) => vi.spyOn(buffer, "destroy"));
            if (terminal === "dispose") {
                disposeComputeTextData(data);
            } else {
                data._onDeviceLost();
            }
            for (const update of updates) {
                await expect(update.completed).rejects.toThrow(`updateComputeTextData: ${terminal === "dispose" ? "disposed" : "device lost"}`);
            }
            destroys.forEach((destroy) => expect(destroy).toHaveBeenCalledOnce());
            expect(data._pendingCompletions.size).toBe(0);
        } finally {
            held.release();
            await held.mapped();
        }
    });

    it("suppresses ignored GPU-status rejection and preserves CPU rejection/indirect clearing", async () => {
        const data = block();
        inject(data, "unsupported");
        await updateComputeTextData(data, ["HH"]);
        await native.device.queue.onSubmittedWorkDone();
        await new Promise<void>((resolve) => setImmediate(resolve));
        await (
            await updateComputeTextData(data, ["HH"])
        ).completed;
        await expect(updateComputeTextData(data, ["HH", 17 as unknown as string])).rejects.toThrow("updateComputeTextData: run 1 text must be a string");
        expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        expect(data.status).toBe("ready");
    });

    it("rejects a pending completion on actual device destruction", async () => {
        const owned = await nativeHarness();
        const data = createComputeTextData(owned.engine, fixture("Roboto-Regular.ttf").font, storage, "async-actual-loss");
        const held = holdMaps();
        try {
            const update = await updateComputeTextData(data, ["HH"]);
            await held.mapped();
            const destroys = held.buffers.map((buffer) => vi.spyOn(buffer, "destroy"));
            owned.destroyExpected = true;
            owned.device.destroy();
            await owned.device.lost;
            await expect(update.completed).rejects.toThrow("updateComputeTextData: device lost");
            expect(data.status).toBe("lost");
            destroys.forEach((destroy) => expect(destroy).toHaveBeenCalledOnce());
        } finally {
            held.release();
            await held.mapped();
            disposeComputeTextData(data);
            await owned.close();
        }
    });

    it.each(["dispose", "loss"] as const)("translates a rejected map to the terminal message on %s", async (terminal) => {
        const data = block();
        const original = GPUBuffer.prototype.mapAsync;
        vi.spyOn(GPUBuffer.prototype, "mapAsync").mockImplementation(function (this: GPUBuffer, mode, offset, size) {
            return original.call(this, mode, offset, size).then(() => {
                if (size === 4) {
                    if (terminal === "dispose") {
                        disposeComputeTextData(data);
                    } else {
                        data._onDeviceLost();
                    }
                    throw new Error("low-level map destroyed/lost");
                }
                return undefined;
            });
        });
        const update = await updateComputeTextData(data, ["HH"]);
        await expect(update.completed).rejects.toThrow(`updateComputeTextData: ${terminal === "dispose" ? "disposed" : "device lost"}`);
    });
});
