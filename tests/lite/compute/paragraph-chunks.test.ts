import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage, updateGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { computeTextParagraphAllocation, updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import type { ComputeTextParagraphLayout } from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import { getComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { fixture } from "./fixtures";
import type { Fixture } from "./fixtures";
import { oracleFields } from "./fixtures";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import * as sourcePreparation from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { createComputeTextLayer } from "../../../packages/babylon-lite/src/text/compute-text-layer";
import { getOrCreateTextPipeline } from "../../../packages/babylon-lite/src/text/_gpu/text-pipeline";
import { writeFileSync } from "node:fs";
import { syntheticFont } from "./synthetic";
import * as computePreparation from "../../../packages/babylon-lite/src/compute/compute-task";
import * as submission from "../../../packages/babylon-lite/src/compute/compute-task-readback";

async function outputs(data: ComputeTextData) {
    const state = data._chunkState;
    const instances: number[] = [];
    const paragraphs: number[][] = [];
    if (state) {
        for (const chunks of state.chunks) {
            const records: number[] = [];
            for (const chunk of chunks) {
                const shaped = new Int32Array(await readStorageBuffer(chunk.shaped.buffer, chunk.shaped.offset, chunk.shaped.size));
                expect(shaped[0]).toBe(0);
                for (let i = 0; i < shaped[1]!; i++) {
                    const record = [...shaped.slice(4 + i * 6, 10 + i * 6)];
                    record[1]! += chunk.source.scalarStart;
                    records.push(...record);
                }
                const indirect = new Uint32Array(await readStorageBuffer(chunk.indirect.buffer, chunk.indirect.offset, 16));
                expect([...indirect].filter((_value, i) => i !== 1)).toEqual([6, 0, 0]);
                if (indirect[1]) {
                    instances.push(...new Uint32Array(await readStorageBuffer(chunk.instances.buffer, chunk.instances.offset, indirect[1] * 12)));
                }
            }
            paragraphs.push(records);
        }
    } else {
        const stage = getComputeTextParagraphStage(data)!;
        const capacity = stage._words[1]!;
        const shaped = new Int32Array(await readStorageBuffer(data._shaped));
        for (const slot of stage._cache!._order) {
            const offset = slot * (4 + capacity * 6);
            expect(shaped[offset]).toBe(0);
            paragraphs.push([...shaped.slice(offset + 4, offset + 4 + shaped[offset + 1]! * 6)]);
        }
        const indirect = new Uint32Array(await readStorageBuffer(data._indirect));
        if (indirect[1]) {
            instances.push(...new Uint32Array(await readStorageBuffer(data._instances, 0, indirect[1] * 12)));
        }
    }
    return { paragraphs, instances };
}

describe("native persistent paragraph chunks", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    const live: ComputeTextData[] = [];
    function block(face: Fixture, paged: boolean) {
        const data = createComputeTextData(native.engine, face.font, storage, face.file, { fontSizePx: 31.75 });
        if (paged) {
            data._chunkBudgets = { force: true, records: 32, arenaBytes: 2048, bindingBytes: 8192 };
        }
        live.push(data);
        return data;
    }
    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        live.forEach(disposeComputeTextData);
        disposeGlyphStorage(storage);
        await native.close();
    });
    async function compare(paged: ComputeTextData, reference: ComputeTextData, paragraphs: string[], layout: ComputeTextParagraphLayout) {
        await (
            await updateComputeTextParagraphs(reference, paragraphs, layout)
        ).completed;
        await (
            await updateComputeTextParagraphs(paged, paragraphs, layout)
        ).completed;
        expect(await outputs(paged)).toEqual(await outputs(reference));
    }

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("matches six-field records and f32 instance bits across small arena seams: %s", async (file) => {
        const face = fixture(file);
        const paged = block(face, true);
        const reference = block(face, false);
        const paragraphs = ["office affinity AV To A\u0326 ".repeat(8), "", "   ", "office   ".repeat(20)];
        for (const maxWidthPx of [600, 60, 1, Infinity]) {
            await compare(paged, reference, paragraphs, { maxWidthPx });
        }
        for (const align of ["left", "center", "right"] as const) {
            await compare(paged, reference, paragraphs, { maxWidthPx: 143.5, align, lineHeightPx: 33.25, paragraphSpacingPx: 6.5 });
        }
        const chunks = paged._chunkState!.chunks.flat();
        expect(chunks.length).toBeGreaterThan(10);
        expect(new Set(chunks.map((chunk) => chunk.shaped.buffer)).size).toBeGreaterThan(1);
        for (const chunk of chunks) {
            expect(chunk.shaped.offset % native.device.limits.minStorageBufferOffsetAlignment).toBe(0);
        }
    });

    it("places a fitting indivisible span through a one-workgroup grid without truncation", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        reference._chunkBudgets = { monolithic: true };
        const limits = native.device.limits;
        vi.spyOn(native.device, "limits", "get").mockReturnValue(
            new Proxy(limits, {
                get(target, key) {
                    return key === "maxComputeWorkgroupsPerDimension" ? 1 : Reflect.get(target, key, target);
                },
            })
        );
        await compare(data, reference, ["A".repeat(100)], { maxWidthPx: 160, align: "right" });
        expect(data._chunkState!.chunks[0]).toHaveLength(1);
        expect((await outputs(data)).instances).toHaveLength(300);
    });

    it("bounds uneven segment workspace without rejecting a fitting indivisible span", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        data._chunkBudgets = { force: true };
        const text = "a ".repeat(80) + "A".repeat(15000) + " " + "a ".repeat(200);
        await compare(data, reference, [text], { maxWidthPx: 229.5, align: "right" });
        expect(data._chunkState!.chunks[0]!.length).toBeGreaterThan(1);
        expect(data._chunkState!.plan!.workspaceBytes).toBeLessThanOrEqual(3 * 1024 * 1024);
    });

    it("retires obsolete dedicated arenas and transient high-water storage after a smaller replacement", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        data._chunkBudgets = { force: true, arenaBytes: 4096 };
        await (
            await updateComputeTextParagraphs(data, ["A".repeat(40000)], { maxWidthPx: 160 })
        ).completed;
        const state = data._chunkState!;
        const transient = () => [...state.buffers].filter(([name]) => name !== "program" && name !== "tables").reduce((sum, [, entry]) => sum + entry.byteLength, 0);
        expect(transient()).toBeGreaterThan(3 * 1024 * 1024);
        const dedicated = state.arenas
            .get("shaped")!
            .filter((arena) => arena.buffer.byteLength > state.plan!.arenaBytes)
            .map((arena) => arena.buffer);
        expect(dedicated.length).toBeGreaterThan(0);
        await compare(data, reference, ["office ".repeat(30)], { maxWidthPx: 160 });
        expect(transient()).toBeLessThanOrEqual(3 * 1024 * 1024);
        expect(dedicated.every((entry) => entry._destroyed)).toBe(true);
    });

    it("retains paragraph-level script selection and scalar clusters across Common-only and supplementary chunks", async () => {
        const face = fixture("NotoSansGothic-Regular.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        const original = " ".repeat(96) + "𐌰 𐌱 𐌲𐌰 ".repeat(40);
        await compare(data, reference, [original, "𐌱𐌲"], { maxWidthPx: 170.5, align: "right", lineHeightPx: 33.25 });
        expect(data._chunkState!.chunks[0]!.every((chunk) => chunk.source.plan.script === "goth")).toBe(true);
        expect(data._chunkState!.chunks[0]![0]!.source.text.trim()).toBe("");
        const last = data._chunkState!.chunks[0]!.at(-1)!;
        const id = last.id;
        await compare(data, reference, ["𐌰 " + original, "𐌱𐌲"], { maxWidthPx: 170.5, align: "center", lineHeightPx: 33.25 });
        expect(data._chunkState!.chunks[0]!.at(-1)!.id).toBe(id);
    });

    it("uploads immutable font data once per selected plan rather than once per chunk", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        const writes = vi.spyOn(native.device.queue, "writeBuffer");
        const texts = ["office ".repeat(24), "ΑΒΓ ".repeat(24)];
        await compare(data, reference, texts, { maxWidthPx: 170 });
        expect(writes.mock.calls.filter(([buffer]) => buffer.label === "compute-text-chunk-program")).toHaveLength(2);
        expect(data._chunkState!.chunks.flat().length).toBeGreaterThan(4);
        writes.mockClear();
        await compare(data, reference, texts, { maxWidthPx: 160 });
        expect(writes.mock.calls.filter(([buffer]) => buffer.label === "compute-text-chunk-program")).toHaveLength(0);
    });

    it("skips clean shape dispatches and releases paged state when switching to plain updates", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        const paragraphs = ["office affinity ".repeat(20), "A", "B"];
        await compare(data, reference, paragraphs, { maxWidthPx: 220 });
        const first = data._chunkState!.chunks.flat().map((chunk) => ({ id: chunk.id, shapes: chunk.shapes }));
        await compare(data, reference, paragraphs, { maxWidthPx: 180 });
        expect(data._chunkState!.chunks.flat().map((chunk) => ({ id: chunk.id, shapes: chunk.shapes }))).toEqual(first);
        await (
            await updateComputeTextData(data, ["A"])
        ).completed;
        expect(data._chunkState).toBeUndefined();
        expect(data._drawRanges).toBeUndefined();
    });

    it("reuses proven shifted suffix shapes, converges exactly, and handles paragraph split/merge/reorder edits", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        const original = "office affinity AV To ".repeat(24);
        let paragraphs = [original, "office", original + "A"];
        await compare(data, reference, paragraphs, { maxWidthPx: 220 });
        const suffix = data._chunkState!.chunks[0]!.slice(-4).map((chunk) => ({ id: chunk.id, shapes: chunk.shapes }));
        paragraphs = ["A " + original, ...paragraphs.slice(1)];
        await compare(data, reference, paragraphs, { maxWidthPx: 220 });
        expect(data._chunkState!.chunks[0]!.slice(-4).map((chunk) => ({ id: chunk.id, shapes: chunk.shapes }))).toEqual(suffix);
        for (const next of [
            [paragraphs[0]!, paragraphs[0]!.slice(0, 200), paragraphs[0]!.slice(200), ...paragraphs.slice(1)],
            [paragraphs[0]!, "office " + original],
            ["", "office", original],
            [original, "office", ""],
            [],
        ]) {
            paragraphs = next;
            await compare(data, reference, paragraphs, { maxWidthPx: 210.25, align: "center", paragraphSpacingPx: 3.75 });
        }
    });

    it("skips measure/place only at exact checkpoint and reverse-event convergence", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const text = ["office affinity ".repeat(12)];
        await (
            await updateComputeTextParagraphs(data, text, { maxWidthPx: 230 })
        ).completed;
        await (
            await updateComputeTextParagraphs(data, text, { maxWidthPx: 230 })
        ).completed;
        for (const chunk of data._chunkState!.chunks.flat()) {
            const info = new Uint32Array(await readStorageBuffer(chunk.metadata.buffer, chunk.metadata.offset, chunk.metadata.size));
            expect(info[139]).toBe(0);
            expect(info[140]).toBe(0);
        }
        await (
            await updateComputeTextParagraphs(data, text, { maxWidthPx: 230, align: "right" })
        ).completed;
        for (const chunk of data._chunkState!.chunks.flat()) {
            const info = new Uint32Array(await readStorageBuffer(chunk.metadata.buffer, chunk.metadata.offset, chunk.metadata.size));
            expect(info[139]).toBe(0);
            expect(info[140]).toBe(1);
        }
    });

    it("re-places a clean prefix when a future chunk changes the width of their shared open line", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        const original = "A " + "office affinity ".repeat(20);
        const layout: ComputeTextParagraphLayout = { maxWidthPx: 1000000, align: "right" };
        await compare(data, reference, [original], layout);
        const first = data._chunkState!.chunks[0]![0]!;
        const id = first.id;
        const shapes = first.shapes;
        await compare(data, reference, [original + "A"], layout);
        expect(data._chunkState!.chunks[0]![0]!.id).toBe(id);
        expect(first.shapes).toBe(shapes);
        const info = new Uint32Array(await readStorageBuffer(first.metadata.buffer, first.metadata.offset, first.metadata.size));
        expect(info[139]).toBe(0);
        expect(info[140]).toBe(1);
    });

    it.each([1, 2, 3])("keeps every old/candidate range invisible after synchronous or later-chunk GPU failure %i", async (status) => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const text = ["office affinity ".repeat(15)];
        await (
            await updateComputeTextParagraphs(data, text, { maxWidthPx: 200 })
        ).completed;
        const chunks = data._chunkState!.chunks.flat();
        const last = chunks.at(-1)!;
        updateStorageBuffer(
            native.engine,
            last.shaped.buffer,
            Int32Array.of(status === 1 ? 1 : 0, status === 2 ? last.source.capacity + 1 : status === 3 ? 1 : 0, 0, 0),
            last.shaped.offset
        );
        if (status === 3) {
            updateStorageBuffer(native.engine, last.shaped.buffer, Int32Array.of(data._font.glyphCount), last.shaped.offset + 16);
        }
        await expect((await updateComputeTextParagraphs(data, text, { maxWidthPx: 200 })).completed).rejects.toThrow(
            status === 1 ? /unsupported/ : status === 2 ? /reserved capacity/ : /missing atlas slot/
        );
        for (const range of data._drawRanges!) {
            expect([...new Uint32Array(await readStorageBuffer(range.indirect, range.indirectOffset, 16))]).toEqual([6, 0, 0, 0]);
        }
        await (
            await updateComputeTextParagraphs(data, text, { maxWidthPx: 200 })
        ).completed;
        expect(() => updateComputeTextParagraphs(data, ["A\nB"], { maxWidthPx: 200 })).toThrow(/line breaks/);
        for (const range of data._drawRanges!) {
            expect([...new Uint32Array(await readStorageBuffer(range.indirect, range.indirectOffset, 16))]).toEqual([6, 0, 0, 0]);
        }
    });

    it("matches serial signed wrapping, deleted chunks and cross-boundary pen arithmetic", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        data._chunkBudgets!.records = 8;
        const texts = ["A A A A A A A A A A A A A A A A A A A A A A A A A A"];
        const layout = { maxWidthPx: 240 };
        await compare(data, reference, texts, layout);
        const stage = getComputeTextParagraphStage(reference)!;
        const full = new Int32Array(reference._shaped.byteLength / 4);
        let records = 0;
        let index = 0;
        for (const chunk of data._chunkState!.chunks[0]!) {
            const shaped = new Int32Array(await readStorageBuffer(chunk.shaped.buffer, chunk.shaped.offset, chunk.shaped.size));
            const count = index % 4 === 2 ? 0 : shaped[1]!;
            shaped[1] = count;
            for (let i = 0; i < count; i++) {
                const at = 4 + i * 6;
                shaped[at + 2] = [100, -20, 0x7ffffffe, 1000, -2147483647][(index + i) % 5]!;
                shaped[at + 3] = ((index + i) % 3) - 1;
                const record = shaped.slice(at, at + 6);
                record[1]! += chunk.source.scalarStart;
                full.set(record, 4 + records++ * 6);
            }
            updateStorageBuffer(native.engine, chunk.shaped.buffer, shaped, chunk.shaped.offset);
            index++;
        }
        full[1] = records;
        expect(stage._cache!._order[0]).toBe(0);
        updateStorageBuffer(native.engine, reference._shaped, full);
        await compare(data, reference, texts, { maxWidthPx: 239.75, align: "right", lineHeightPx: 32.125 });
    });

    it("carries both retroactive line closures over many chunks without losing hanging-space events", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        data._chunkBudgets!.records = 8;
        const texts = ["A ".repeat(80)];
        await compare(data, reference, texts, { maxWidthPx: 240 });
        const full = new Int32Array(reference._shaped.byteLength / 4);
        const stage = getComputeTextParagraphStage(reference)!;
        let ordinal = 0;
        for (const chunk of data._chunkState!.chunks[0]!) {
            const shaped = new Int32Array(await readStorageBuffer(chunk.shaped.buffer, chunk.shaped.offset, chunk.shaped.size));
            const flags = new Uint8Array(chunk.flags.size);
            for (let i = 0; i < chunk.source.scalars; i++) {
                const global = chunk.source.scalarStart + i;
                flags[i + 1] = global % 2 ? (global === 1 ? 3 : 1) : 0;
            }
            flags[0] = chunk.source.scalarStart ? (chunk.source.scalarStart === 2 ? 3 : 1) : 0;
            updateStorageBuffer(native.engine, chunk.flags.buffer, flags, chunk.flags.offset);
            for (let i = 0; i < shaped[1]!; i++) {
                const at = 4 + i * 6;
                const cluster = shaped[at + 1]! + chunk.source.scalarStart;
                shaped[at + 2] = cluster === 158 ? 100000 : cluster % 2 ? 0 : 1;
                const record = shaped.slice(at, at + 6);
                record[1] = cluster;
                full.set(record, 4 + ordinal++ * 6);
            }
            updateStorageBuffer(native.engine, chunk.shaped.buffer, shaped, chunk.shaped.offset);
        }
        full[1] = ordinal;
        updateStorageBuffer(native.engine, reference._shaped, full);
        const flagStart = 16 + stage._words[11]!;
        const flags = new Uint8Array(stage._words.buffer, flagStart * 4, stage._words[12]! * 4);
        for (let i = 0; i < texts[0]!.length; i++) {
            flags[i] = i % 2 ? (i === 1 ? 3 : 1) : 0;
        }
        updateStorageBuffer(native.engine, stage._input!, stage._words);
        await compare(data, reference, texts, { maxWidthPx: 239.75, align: "right" });
        const incoming = [];
        for (const chunk of data._chunkState!.chunks[0]!) {
            const info = new Uint32Array(await readStorageBuffer(chunk.metadata.buffer, chunk.metadata.offset, chunk.metadata.size));
            incoming.push(info[160]!);
        }
        expect(incoming).toContain(2);
        expect(Math.max(...incoming)).toBe(2);
    });

    it("preserves busy admission, multiple pending results and terminal resource release", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const text = ["office affinity ".repeat(10)];
        const pending = updateComputeTextParagraphs(data, text, { maxWidthPx: 230 });
        expect(() => updateComputeTextParagraphs(data, text, { maxWidthPx: 230 })).toThrow(/busy/);
        const first = await pending;
        const second = await updateComputeTextParagraphs(data, ["A " + text[0]], { maxWidthPx: 230 });
        await Promise.all([first.completed, second.completed]);
        const state = data._chunkState!;
        expect(state.released).toBe(false);
        const originalMap = GPUBuffer.prototype.mapAsync;
        const pendingMaps: Promise<void>[] = [];
        vi.spyOn(GPUBuffer.prototype, "mapAsync").mockImplementation(function (this: GPUBuffer, ...args) {
            const pending = originalMap.apply(this, args);
            pendingMaps.push(pending);
            return pending;
        });
        const third = await updateComputeTextParagraphs(data, text, { maxWidthPx: 230 });
        disposeComputeTextData(data);
        await expect(third.completed).rejects.toThrow(/disposed/);
        await Promise.allSettled(pendingMaps);
        expect(state.released).toBe(true);
        expect(state.arenas.size).toBe(0);
        expect(data._chunkState).toBeUndefined();
        expect(data._drawRanges).toBeUndefined();
    });

    it("preserves status results while switching between submitted legacy and paged resources", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, false);
        const first = await updateComputeTextParagraphs(data, ["office ".repeat(20)], { maxWidthPx: 200 });
        const oldShape = data._shaped;
        data._chunkBudgets = { force: true, records: 32 };
        const paged = await updateComputeTextParagraphs(data, ["office ".repeat(30)], { maxWidthPx: 200 });
        expect(oldShape._destroyed).toBe(true);
        expect(data._shaped.byteLength).toBe(64);
        const plain = await updateComputeTextData(data, ["office"]);
        await Promise.all([first.completed, paged.completed, plain.completed]);
        expect(data._chunkState).toBeUndefined();
    });

    it.each([false, true])("invalidates a failed paged ancestor without clearing a newer independent generation: %s", async (independent) => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const texts = ["office affinity ".repeat(10)];
        await (
            await updateComputeTextParagraphs(data, texts, { maxWidthPx: 210 })
        ).completed;
        let release!: () => void;
        const gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        const original = submission.submitDeferredComputeTaskResult;
        vi.spyOn(submission, "submitDeferredComputeTaskResult").mockImplementationOnce((...args) => original(...args).then(() => gate.then(() => 1)));
        const earlier = await updateComputeTextParagraphs(data, texts, { maxWidthPx: 211 });
        if (independent) {
            await (
                await updateComputeTextData(data, ["A"])
            ).completed;
        }
        await (
            await updateComputeTextParagraphs(data, ["A " + texts[0]], { maxWidthPx: 211 })
        ).completed;
        const state = data._chunkState!;
        const revision = state.revision;
        release();
        await expect(earlier.completed).rejects.toThrow(/updateComputeTextParagraphs:.*unsupported/);
        expect(state.revision).toBe(independent ? revision : -1);
        const ranges = data._drawRanges!;
        for (const range of ranges) {
            expect(new Uint32Array(await readStorageBuffer(range.indirect, range.indirectOffset, 16))[1]).toBeGreaterThan(0);
        }
    });

    it("retains the original fitting-path render command order", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, false);
        await (
            await updateComputeTextParagraphs(data, ["office"], { maxWidthPx: 200 })
        ).completed;
        const pipeline = getOrCreateTextPipeline(native.engine, "rgba8unorm", 1, null, false);
        const uniform = native.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const draw = createComputeTextLayer(data)._createDraw!(native.engine, uniform);
        try {
            draw._prepare(pipeline._cache._bindGroupLayout);
            const encode = native.device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
            encode.setPipeline(pipeline._pipeline);
            const events: string[] = [];
            const vertex = encode.setVertexBuffer.bind(encode);
            const group = encode.setBindGroup.bind(encode);
            const indirect = encode.drawIndirect.bind(encode);
            vi.spyOn(encode, "setVertexBuffer").mockImplementation((...args) => {
                events.push(`vertex${args[0]}`);
                vertex(...args);
            });
            vi.spyOn(encode, "setBindGroup").mockImplementation((index, binding) => {
                events.push(`group${index}`);
                group(index, binding);
            });
            vi.spyOn(encode, "drawIndirect").mockImplementation((...args) => {
                events.push("indirect");
                indirect(...args);
            });
            expect(draw._record(encode, pipeline._cache._quadVertexBuffer, pipeline._pipeline, pipeline._variantPipeline)).toBe(1);
            expect(events).toEqual(["vertex0", "vertex1", "group0", "indirect"]);
            encode.finish();
        } finally {
            draw._dispose();
            uniform.destroy();
        }
    });

    it("encodes every actual ordered window and invalidates bundles only when the range list changes", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const paragraphs = ["office affinity ".repeat(10)];
        await (
            await updateComputeTextParagraphs(data, paragraphs, { maxWidthPx: 200 })
        ).completed;
        const pipeline = getOrCreateTextPipeline(native.engine, "rgba8unorm", 1, null, false);
        const uniform = native.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const texture = native.device.createTexture({ size: [8, 8], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const draw = createComputeTextLayer(data)._createDraw!(native.engine, uniform);
        try {
            draw._prepare(pipeline._cache._bindGroupLayout);
            const version = draw._version;
            const encode = native.device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
            encode.setPipeline(pipeline._pipeline);
            const vertex = vi.spyOn(encode, "setVertexBuffer");
            const indirect = vi.spyOn(encode, "drawIndirect");
            const count = draw._record(encode, pipeline._cache._quadVertexBuffer, pipeline._pipeline, pipeline._variantPipeline);
            expect(count).toBe(data._drawRanges!.length);
            expect(indirect.mock.calls).toEqual(data._drawRanges!.map((range) => [range.indirect._buffer, range.indirectOffset]));
            expect(vertex.mock.calls.slice(1)).toEqual(data._drawRanges!.map((range) => [1, range.instances._buffer, range.offset, range.size]));
            const bundle = encode.finish();
            const encoder = native.device.createCommandEncoder();
            const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
            pass.executeBundles([bundle]);
            pass.end();
            native.device.queue.submit([encoder.finish()]);
            await (
                await updateComputeTextParagraphs(data, paragraphs, { maxWidthPx: 180 })
            ).completed;
            draw._prepare(pipeline._cache._bindGroupLayout);
            expect(draw._version).toBe(version);
            await (
                await updateComputeTextParagraphs(data, [...paragraphs, "office"], { maxWidthPx: 180 })
            ).completed;
            draw._prepare(pipeline._cache._bindGroupLayout);
            expect(draw._version).toBeGreaterThan(version);
        } finally {
            draw._dispose();
            uniform.destroy();
            texture.destroy();
        }
    });

    it.each(["multiple", "context", "backtrack", "ligature-ignore-marks", "adjust"] as const)("preserves fitting indivisible synthetic shaping: %s", async (kind) => {
        const face = syntheticFont(kind);
        const data = block(face, true);
        const reference = block(face, false);
        data._chunkBudgets!.records = 4;
        await compare(data, reference, ["ABCABC", "AABC", "BC"], { maxWidthPx: 3.5, lineHeightPx: 32 });
        expect(data._chunkState!.chunks[0]).toHaveLength(1);
    });

    it("binds bounded windows inside arenas larger than the effective test binding limit", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        data._chunkBudgets = { force: true, records: 32, bindingBytes: 8192, arenaBytes: 32768 };
        await (
            await updateComputeTextParagraphs(data, ["office ".repeat(200)], { maxWidthPx: 200 })
        ).completed;
        const chunks = data._chunkState!.chunks.flat();
        expect(chunks.some((chunk) => chunk.shaped.offset > 0)).toBe(true);
        expect(chunks.some((chunk) => chunk.shaped.buffer.byteLength > 8192)).toBe(true);
        for (const chunk of chunks) {
            for (const range of [chunk.shaped, chunk.instances, chunk.metadata, chunk.params, chunk.indirect]) {
                expect(range.size).toBeLessThanOrEqual(8192);
                expect(range.offset + range.size).toBeLessThanOrEqual(range.buffer.byteLength);
                expect(range.offset % native.device.limits.minStorageBufferOffsetAlignment).toBe(0);
            }
        }
    });

    it("automatically pages materially wasteful mixed-length storage while preserving clean paragraph shapes", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, false);
        const reference = block(face, false);
        reference._chunkBudgets = { monolithic: true };
        const texts = ["office ".repeat(180), ...Array.from({ length: 99 }, () => "A")];
        await compare(data, reference, texts, { maxWidthPx: 160 });
        expect(data._chunkState).toBeDefined();
        expect(data._chunkState!.chunks[1]![0]!.source.capacity).toBe(1);
        const clean = data._chunkState!.chunks.slice(1).map((chunks) => ({ id: chunks[0]!.id, shapes: chunks[0]!.shapes }));
        texts[0] = "affinity " + texts[0]!;
        await compare(data, reference, texts, { maxWidthPx: 160 });
        expect(data._chunkState!.chunks.slice(1).map((chunks) => ({ id: chunks[0]!.id, shapes: chunks[0]!.shapes }))).toEqual(clean);
    });

    it("rebuilds paged placement/slot dependencies after external atlas mutation", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        data._curveSet = reference._curveSet = "paged-external-atlas";
        const texts = ["A A ".repeat(25), "B B"];
        await compare(data, reference, texts, { maxWidthPx: 180 });
        const oldIds = data._chunkState!.chunks.flat().map((chunk) => chunk.id);
        const oldShapes = data._chunkState!.chunks.flat().map((chunk) => chunk.shapes);
        const cs = storage._curveSets.get(data._curveSet)!;
        const source = cs._curves.get(face.oracle.glyphId(0x41))!;
        const space = face.oracle.glyphId(0x20);
        expect(data._slotWords![space]).toBe(0xfffffffe);
        updateGlyphStorage(storage, data._curveSet, new Map([[space, { ...source, glyphId: space }]]));
        await compare(data, reference, texts, { maxWidthPx: 180 });
        expect(data._chunkState!.chunks.flat().map((chunk) => chunk.id)).toEqual(oldIds);
        expect(data._chunkState!.chunks.flat().map((chunk) => chunk.shapes)).toEqual(oldShapes.map((count) => count + 1));
        expect(data._slotWords![space]).toBe(cs._atlas._glyphSlots.get(space)!._index);
    });

    it("rejects an indivisible required span before installing atlas or paged resources", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        data._curveSet = "paged-indivisible-rejection";
        const resources = native.engine._storageBuffers!.size;
        expect(() => updateComputeTextParagraphs(data, ["A".repeat(200)], { maxWidthPx: 100 })).toThrow(
            /paragraph 0 indivisible scalar span.*replay buffer of \d+ bytes exceeds binding limit 8192/
        );
        expect(storage._curveSets.has(data._curveSet)).toBe(false);
        expect(data._chunkState).toBeUndefined();
        expect(data._drawRanges).toBeUndefined();
        expect(native.engine._storageBuffers!.size).toBe(resources);
        expect(data.status).toBe("ready");
    });

    it("retains admitted immutable source/layout snapshots while chunk preparation is pending", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const reference = block(face, false);
        const text = "office affinity ".repeat(10);
        const expected = { maxWidthPx: 120, align: "right" as const, lineHeightPx: 32.25 };
        await (
            await updateComputeTextParagraphs(reference, [text], expected)
        ).completed;
        const paragraphs = [text];
        const layout = { ...expected };
        const pending = updateComputeTextParagraphs(data, paragraphs, layout);
        paragraphs[0] = "A\nB";
        layout.maxWidthPx = 1;
        await (
            await pending
        ).completed;
        expect(data._chunkState!.plan!.paragraphs).toEqual([text]);
        expect(await outputs(data)).toEqual(await outputs(reference));
    });

    it("releases early paged allocations when disposal interrupts pipeline preparation", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const prepare = computePreparation.prepareComputeTask;
        let release!: () => void;
        const gate = new Promise<void>((resolve) => {
            release = resolve;
        });

        vi.spyOn(computePreparation, "prepareComputeTask").mockImplementationOnce(async (task) => {
            await gate;
            await prepare(task);
        });
        const pending = updateComputeTextParagraphs(data, ["office ".repeat(20)], { maxWidthPx: 200 });
        const state = data._chunkState!;
        expect(state.arenas.size).toBeGreaterThan(0);
        disposeComputeTextData(data);
        release();
        await expect(pending).rejects.toThrow(/disposed/);
        expect(state.released).toBe(true);
        expect(state.arenas.size).toBe(0);
        expect(data._drawRanges).toBeUndefined();
    });

    it("retires a failed admitted candidate before retrying pipeline preparation", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, true);
        const resources = native.engine._storageBuffers!.size;
        vi.spyOn(computePreparation, "prepareComputeTask").mockRejectedValueOnce(new Error("chunk pipeline preparation failed"));
        await expect(updateComputeTextParagraphs(data, ["office ".repeat(10)], { maxWidthPx: 180 })).rejects.toThrow(/chunk pipeline preparation failed/);
        expect(data.status).toBe("ready");
        expect(data._chunkState).toBeUndefined();
        expect(data._drawRanges).toBeUndefined();
        expect(native.engine._storageBuffers!.size).toBe(resources);
        await (
            await updateComputeTextParagraphs(data, ["office ".repeat(10)], { maxWidthPx: 180 })
        ).completed;
        expect(data._chunkState).toBeDefined();
    });

    it("forgets all paged resources on actual device loss and never resurrects the owner", async () => {
        const own = await nativeHarness();
        const ownStorage = createGlyphStorage();
        const face = fixture("Inter.ttf");
        const data = createComputeTextData(own.engine, face.font, ownStorage, "loss");
        data._chunkBudgets = { force: true, records: 32, arenaBytes: 256 };
        try {
            await (
                await updateComputeTextParagraphs(data, ["office ".repeat(10)], { maxWidthPx: 200 })
            ).completed;
            const state = data._chunkState!;
            const pending = await updateComputeTextParagraphs(data, ["office"], { maxWidthPx: 200 });
            const retired = [...state.retired.values()].flatMap((batch) => [...batch.buffers]);
            expect(retired.length).toBeGreaterThan(0);
            const destroys = retired.map((buffer) => vi.spyOn(buffer._buffer!, "destroy"));
            own.destroyExpected = true;
            own.device.destroy();
            await own.device.lost;
            await Promise.resolve();
            await expect(pending.completed).rejects.toThrow(/device lost/);
            expect(data.status).toBe("lost");
            expect(state.released).toBe(true);
            expect(state.buffers.size).toBe(0);
            expect(state.retired.size).toBe(0);
            destroys.forEach((destroy) => expect(destroy).toHaveBeenCalledOnce());
            expect(data._drawRanges).toBeUndefined();
            expect(() => updateComputeTextParagraphs(data, ["A"], { maxWidthPx: 200 })).toThrow(/device lost/);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(ownStorage);
            await own.close();
        }
    });

    it("renders a real paragraph beyond required Stage1 line storage with default device limits", async () => {
        const face = fixture("Inter.ttf");
        const expansion = Math.max(1, compileComputeTextProgram(createComputeTextPlan(face.font, selectComputeTextRun("a a a "))).maxExpansion);
        const pairs = Math.ceil((Math.floor(native.device.limits.maxStorageBufferBindingSize / 48) + 1) / (2 * expansion));
        const required = pairs * 2 * expansion;
        expect(required * 48).toBeGreaterThan(native.device.limits.maxStorageBufferBindingSize);
        // Split-capacity line storage also carries one 48-byte measure summary per 2048-record block.
        const lineBytes = (required + Math.ceil(required / 2048)) * 48;
        expect(() => computeTextParagraphAllocation(native.device.limits, 1, required)).toThrow(`layout lines buffer of ${lineBytes} bytes exceeds device storage limits`);
        const periodic = oracleFields(face, "a a a ");
        expect(periodic).toHaveLength(6);
        const small = block(face, false);
        const data = block(face, false);
        small._chunkBudgets = { monolithic: true };
        await (
            await updateComputeTextParagraphs(small, ["a a a "], { maxWidthPx: 1, lineHeightPx: 32 })
        ).completed;
        const smallOutput = await outputs(small);
        expect(smallOutput.paragraphs[0]!.slice(0, 12)).toEqual(periodic.slice(0, 2).flat());
        await (
            await updateComputeTextParagraphs(data, ["a a a "], { maxWidthPx: 1, lineHeightPx: 32 })
        ).completed;
        const text = "a ".repeat(pairs);
        const slab = vi.spyOn(sourcePreparation, "createComputeTextInputSlab");
        const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
        await (
            await updateComputeTextParagraphs(data, [text], { maxWidthPx: 1, lineHeightPx: 32 })
        ).completed;
        expect(maps.mock.calls.map((call) => call.slice(0, 3))).toEqual([[GPUMapMode.READ, 0, 4]]);
        expect(slab).not.toHaveBeenCalled();
        slab.mockRestore();
        maps.mockRestore();
        const state = data._chunkState!;
        expect(state).toBeDefined();
        expect(state.plan!.workspaceBytes).toBeLessThanOrEqual(8 * 1024 * 1024);
        expect(state.chunks[0]!.length).toBeGreaterThan(1);
        let records = 0;
        let visible = 0;
        const bits = (number: number) => new Uint32Array(new Float32Array([number]).buffer)[0]!;
        const slot = data._slotWords![periodic[0]![0]!]!;
        for (const chunk of state.chunks[0]!) {
            const shaped = new Int32Array(await readStorageBuffer(chunk.shaped.buffer, chunk.shaped.offset, chunk.shaped.size));
            expect(shaped[0]).toBe(0);
            expect(shaped[1]).toBe(chunk.source.scalars);
            for (let i = 0; i < shaped[1]!; i++) {
                const actual = 4 + i * 6;
                const expected = periodic[(chunk.source.scalarStart + i) % 2]!;
                for (let field = 0; field < 6; field++) {
                    const value = field === 1 ? i : expected[field]!;
                    if (shaped[actual + field] !== value) {
                        throw new Error(`large shaped record ${records + i} field ${field}: ${shaped[actual + field]} != ${value}`);
                    }
                }
            }
            records += shaped[1]!;
            const indirect = new Uint32Array(await readStorageBuffer(chunk.indirect.buffer, chunk.indirect.offset, 16));
            expect([...indirect].filter((_value, i) => i !== 1)).toEqual([6, 0, 0]);
            const instances = new Uint32Array(await readStorageBuffer(chunk.instances.buffer, chunk.instances.offset, indirect[1]! * 12));
            for (let i = 0; i < indirect[1]!; i++) {
                const expected = [smallOutput.instances[0]!, visible + i ? bits(-(visible + i) * 32) : smallOutput.instances[1]!, slot];
                for (let field = 0; field < 3; field++) {
                    if (instances[i * 3 + field] !== expected[field]) {
                        throw new Error(`large instance ${visible + i} field ${field}: ${instances[i * 3 + field]} != ${expected[field]}`);
                    }
                }
            }
            visible += indirect[1]!;
        }
        expect(records).toBe(pairs * 2);
        expect(visible).toBe(pairs);
        const pipeline = getOrCreateTextPipeline(native.engine, "rgba8unorm", 1, null, false);
        const uniform = native.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const texture = native.device.createTexture({ size: [8, 8], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const draw = createComputeTextLayer(data)._createDraw!(native.engine, uniform);
        let commands: number;
        try {
            draw._prepare(pipeline._cache._bindGroupLayout);
            const bundleEncoder = native.device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
            bundleEncoder.setPipeline(pipeline._pipeline);
            const indirect = vi.spyOn(bundleEncoder, "drawIndirect");
            commands = draw._record(bundleEncoder, pipeline._cache._quadVertexBuffer, pipeline._pipeline, pipeline._variantPipeline);
            expect(indirect.mock.calls).toEqual(data._drawRanges!.map((range) => [range.indirect._buffer, range.indirectOffset]));
            expect(commands).toBe(state.chunks[0]!.length);
            const bundle = bundleEncoder.finish();
            const encoder = native.device.createCommandEncoder();
            const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
            pass.executeBundles([bundle]);
            pass.end();
            native.device.queue.submit([encoder.finish()]);
            await native.device.queue.onSubmittedWorkDone();
        } finally {
            draw._dispose();
            uniform.destroy();
            texture.destroy();
        }
        const arenas = [...state.arenas].map(([kind, entries]) => ({ kind, allocations: entries.length, bytes: entries.reduce((sum, entry) => sum + entry.buffer.byteLength, 0) }));
        const buffers = [...state.buffers].map(([name, entry]) => ({ name, bytes: entry.byteLength }));
        const base = [data._shaped, data._instances, data._indirect, data._slots, data._statusBuffer];
        const statusReadbackBytes = state.staging.reduce((sum, entry) => sum + entry.size, 0);
        const ownerBufferBytes =
            arenas.reduce((sum, entry) => sum + entry.bytes, 0) +
            buffers.reduce((sum, entry) => sum + entry.bytes, 0) +
            base.reduce((sum, entry) => sum + entry.byteLength, 0) +
            data._style.size +
            statusReadbackBytes;
        const diagnostics = [...base, ...state.buffers.values(), ...[...state.arenas.values()].flat().map((arena) => arena.buffer)];
        const diagnosticReadbackBytes = diagnostics.reduce((sum, entry) => sum + (entry._readback?.size ?? 0), 0);
        const evidence = {
            font: face.file,
            expansion,
            scalars: text.length,
            requiredCapacity: required,
            oldRequiredLineBytes: required * 48,
            deviceLimits: { maxBufferSize: native.device.limits.maxBufferSize, maxStorageBufferBindingSize: native.device.limits.maxStorageBufferBindingSize },
            chunks: state.chunks[0]!.length,
            ranges: data._drawRanges!.length,
            actualDrawCommands: commands,
            records,
            visible,
            workspaceBytes: state.plan!.workspaceBytes,
            arenas,
            buffers,
            statusReadbackBytes,
            ownerBufferBytes,
            diagnosticReadbackBytes,
            memoryScope:
                "Application-owned GPU buffers; caller-owned atlas textures/metadata and opaque pipeline/driver allocations excluded. Untimed diagnostic readback buffers reported separately.",
        };
        console.warn(`Bounded paragraph capacity: ${JSON.stringify(evidence)}`);
        if (process.env.LITE_CHUNK_CAPACITY_EVIDENCE) {
            writeFileSync(process.env.LITE_CHUNK_CAPACITY_EVIDENCE, JSON.stringify(evidence, null, 2));
        }
    });
});
