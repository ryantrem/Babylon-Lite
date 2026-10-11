import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
    createComputeTextData,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    getComputeFontLineMetrics,
    updateComputeTextParagraphs,
} from "../../../packages/babylon-lite/src/index";
import type { ComputeTextData, ComputeTextParagraphLayout, GlyphStorage } from "../../../packages/babylon-lite/src/index";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { createComputeTextParagraphStage, getComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { awaitComputeTextUpdate } from "./await-update";
import { fixture, oracleFields } from "./fixtures";
import type { Fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

interface Line {
    first: number;
    count: number;
    width: number;
    cluster: number;
}

function bits(value: number): number {
    return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

/** WGSL permits non-fused fma. Discover the adapter's mode outside all measured update paths. */
async function fusedMultiplyAdd(native: NativeHarness): Promise<boolean> {
    const input = createStorageBuffer(native.engine, Uint32Array.of(bits(1 + 2 ** -23), bits(1 + 2 ** -23), bits(-1 - 2 ** -22)));
    const output = createStorageBuffer(native.engine, 4, { writable: true });
    try {
        const module = native.device.createShaderModule({
            code: `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main() { output[0] = fma(input[0], input[1], input[2]); }
`,
        });
        const pipeline = await native.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        const group = native.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: input._buffer! } },
                { binding: 1, resource: { buffer: output._buffer! } },
            ],
        });
        const encoder = native.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(1);
        pass.end();
        native.device.queue.submit([encoder.finish()]);
        const actual = new Float32Array(await readStorageBuffer(output))[0]!;
        expect([0, 2 ** -46]).toContain(actual);
        return actual !== 0;
    } finally {
        disposeStorageBuffer(input);
        disposeStorageBuffer(output);
    }
}

function sourceFlags(face: Fixture, text: string): number[] {
    const selected = selectComputeTextRun(text, face.font);
    const glyphs = mapComputeFontScalars(
        face.font,
        selected._scalars,
        (_i, error) => {
            throw error;
        },
        selected._nominal
    );
    const safety = analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face.font, selected)));
    return Array.from(selected._scalars, (cp, i) => {
        if (cp !== 32) {
            return cp === 0x200d || /\p{M}/u.test(String.fromCodePoint(cp)) ? 4 : 0;
        }
        const candidate = i + 1 < glyphs.length && selected._scalars[i + 1] !== 32;
        if (candidate) {
            expect(safety, `safety: ${text}`).not.toBeNull();
        }
        return candidate && safety?.safeAfter(glyphs, i) ? 3 : 1;
    });
}

/** Advance-based reference for rules 1–5. All records come from the independent text-shaper oracle. */
function referenceLines(records: readonly number[][], flags: readonly number[], scale: number, maxWidth: number): Line[] {
    const lines: Line[] = [];
    let first = 0;
    let startX = 0;
    let penX = 0;
    let lastContent = 0;
    let opportunity: { after: number; pen: number; content: number } | null = null;
    function emit(end: number, content: number): void {
        lines.push({ first, count: end - first, width: Math.max(0, Math.fround(Math.fround(content - startX) * scale)), cluster: records[first]?.[1] ?? 0 });
    }
    for (let i = 0; i < records.length; i++) {
        const record = records[i]!;
        const flag = flags[record[1]!]!;
        const endX = penX + record[2]!;
        if (!(flag & 1)) {
            while (i !== first && Math.fround(Math.fround(endX - startX) * scale) > maxWidth) {
                if (opportunity) {
                    emit(opportunity.after, opportunity.content);
                    first = opportunity.after;
                    startX = opportunity.pen;
                    opportunity = null;
                    continue;
                }
                if (record[1] !== records[i - 1]![1] && !(flag & 4)) {
                    emit(i, lastContent);
                    first = i;
                    startX = penX;
                }
                break;
            }
            lastContent = endX;
        }
        penX = endX;
        const nextCluster = records[i + 1]?.[1];
        if (nextCluster !== undefined && nextCluster > record[1]! && flags[nextCluster - 1]! & 2) {
            opportunity = { after: i + 1, pen: penX, content: lastContent };
        }
    }
    emit(records.length, lastContent);
    return lines;
}

async function verifyLayout(data: ComputeTextData, face: Fixture, paragraphs: readonly string[], layout: ComputeTextParagraphLayout, fused: boolean): Promise<void> {
    const fma = (a: number, b: number, c: number) => Math.fround((fused ? a * b : Math.fround(a * b)) + c);
    const stage = getComputeTextParagraphStage(data)!;
    const capacity = stage._words[1]!;
    const stride = 4 + capacity * 6;
    const shape = new Int32Array(await readStorageBuffer(data._shaped));
    const info = new Uint32Array(await readStorageBuffer(stage._info!, 0, paragraphs.length * 16));
    const lines = new Uint32Array(await readStorageBuffer(stage._lines!));
    const instances = new DataView(await readStorageBuffer(data._instances));
    const scale = Math.fround(data.fontSizePx / face.font.unitsPerEm);
    const height = Math.fround(layout.lineHeightPx ?? getComputeFontLineMetrics(face.font, data.fontSizePx).lineHeightPx);
    const spacing = Math.fround(layout.paragraphSpacingPx ?? 0);
    const width = Math.fround(layout.maxWidthPx);
    const atlas = data._storage._curveSets.get(data._curveSet)!._atlas;
    let visible = 0;
    let globalLine = 0;
    paragraphs.forEach((text, p) => {
        const records = oracleFields(face, text);
        const flags = sourceFlags(face, text);
        const expected = referenceLines(records, flags, scale, width);
        expect([...shape.slice(p * stride, p * stride + 2)], `${p}: shaped status/count`).toEqual([0, records.length]);
        records.forEach((record, i) => {
            expect([...shape.slice(p * stride + 4 + i * 6, p * stride + 10 + i * 6)], `${p}: shaped ${i}`).toEqual(record);
        });
        expect([...info.slice(p * 4, p * 4 + 4)], `${p}: paragraph info`).toEqual([visible, 0, expected.length, globalLine]);
        const flagOffset = stage._words[8 + p]!;
        expect([...new Uint8Array(stage._words.buffer, flagOffset * 4, flags.length)], `${p}: input flags`).toEqual(flags);
        let penX = 0;
        let penY = 0;
        expected.forEach((line, l) => {
            expect([...lines.slice((p * capacity + l) * 4, (p * capacity + l + 1) * 4)], `${p}: line ${l}`).toEqual([
                line.first,
                line.count,
                bits(line.width),
                line.count ? line.cluster : 0,
            ]);
            const lineX = penX;
            const align =
                layout.align === "center"
                    ? Math.fround(Math.fround(Math.max(0, Math.fround(width - line.width))) * 0.5)
                    : layout.align === "right"
                      ? Math.max(0, Math.fround(width - line.width))
                      : 0;
            const baseline = fma(Math.fround(globalLine + l), height, Math.fround(Math.fround(p) * spacing));
            for (let i = line.first; i < line.first + line.count; i++) {
                const record = records[i]!;
                const slot = atlas._glyphSlots.get(record[0]!);
                if (slot) {
                    const at = visible * 12;
                    expect(instances.getFloat32(at, true), `${p}: instance ${visible} x`).toBe(fma(Math.fround(penX - lineX + record[4]!), scale, align));
                    expect(instances.getFloat32(at + 4, true), `${p}: instance ${visible} y`).toBe(fma(Math.fround(penY + record[5]!), scale, -baseline));
                    expect(instances.getUint32(at + 8, true), `${p}: instance ${visible} slot`).toBe(slot._index);
                    visible++;
                }
                penX += record[2]!;
                penY += record[3]!;
            }
        });
        globalLine += expected.length;
    });
    expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, visible, 0, 0]);
    expect(stage._info!._data).toBeNull();
    expect(stage._lines!._data).toBeNull();
    expect(data._instances._data).toBeNull();
    expect(data._shaped._data).toBeNull();
}

describe("native GPU paragraph wrapping and placement", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    let fused: boolean;
    const live = new Set<ComputeTextData>();
    function block(face: Fixture): ComputeTextData {
        const data = createComputeTextData(native.engine, face.font, storage, face.file, { fontSizePx: 31.75 });
        live.add(data);
        return data;
    }
    async function update(data: ComputeTextData, face: Fixture, paragraphs: readonly string[], layout: ComputeTextParagraphLayout): Promise<void> {
        const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
        try {
            await (
                await updateComputeTextParagraphs(data, paragraphs, layout)
            ).completed;
            expect(maps.mock.calls.map((call) => call.slice(0, 3))).toEqual([[GPUMapMode.READ, 0, 4]]);
        } finally {
            maps.mockRestore();
        }
        await verifyLayout(data, face, paragraphs, layout, fused);
    }
    beforeAll(async () => {
        native = await nativeHarness();
        fused = await fusedMultiplyAdd(native);
        storage = createGlyphStorage();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        for (const data of live) {
            disposeComputeTextData(data);
        }
        disposeGlyphStorage(storage);
        await native?.close();
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("matches oracle records, lines, instances and status-only readback: %s", async (file) => {
        const face = fixture(file);
        const data = block(face);
        const paragraphs = ["office affinity To A\u0326", "AV To", "  A   B C   ", "", "   ", "ABCDEFGHIJKLMNO", "B\u0301\u0307B\u0301\u0307B", "A\u00a0B"];
        for (const maxWidthPx of [Infinity, 130, 47.5, 1]) {
            await update(data, face, paragraphs, { maxWidthPx });
        }
        await update(data, face, [], { maxWidthPx: Infinity });
        await update(data, face, ["", "", "A", ""], { maxWidthPx: 20 });
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("preserves whole-paragraph clusters after segmented shaping: %s", async (file) => {
        const face = fixture(file);
        const data = block(face);
        const modules = vi.spyOn(native.device, "createShaderModule");
        const paragraphs = ["office  affinity To A\u0326 ".repeat(12), "  ABC ".repeat(20), ""];
        await update(data, face, paragraphs, { maxWidthPx: 99.25, lineHeightPx: 37.75, paragraphSpacingPx: 5.5 });
        expect(modules.mock.calls.some(([descriptor]) => descriptor.label === "compute-text-stitch" || descriptor.code.includes("fn main(@builtin(workgroup_id) wg"))).toBe(true);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("aligns hanging and overflowing lines, with custom height and paragraph spacing: %s", async (file) => {
        const face = fixture(file);
        const data = block(face);
        for (const align of ["left", "center", "right"] as const) {
            await update(data, face, ["A   B ", " ", "", "office affinity", "WWW"], { maxWidthPx: 46.25, lineHeightPx: 33.75, paragraphSpacingPx: 7.5, align });
        }
    });

    it("scans spans of more than 256 paragraphs, preserving line and instance bases across plan groups", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        for (const count of [257, 769, 0, 513, 2]) {
            const paragraphs = Array.from({ length: count }, (_, p) => (p % 7 === 0 ? "" : p % 3 === 0 ? "α βγ" : p % 3 === 1 ? "office   affinity " : "Москва Іі"));
            await update(data, face, paragraphs, { maxWidthPx: 33.25, paragraphSpacingPx: 2.25 });
        }
    });

    it("uses scalar-indexed flags and line source indices for supplementary LTR text", async () => {
        const face = fixture("NotoSansGothic-Regular.ttf");
        await update(block(face), face, ["𐌰𐌱 𐌲𐌳", "𐌰𐌱𐌲"], { maxWidthPx: 20.5 });
    });

    it("reuses warm buffers and bindings, resizes dispatches without rebinding, and alternates both update APIs", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const layout = { maxWidthPx: 30, paragraphSpacingPx: 1.5 };
        await update(
            data,
            face,
            Array.from({ length: 130 }, () => "office A\u0326"),
            layout
        );
        const stage = getComputeTextParagraphStage(data)!;
        const version = stage.version;
        const bindings = vi.spyOn(native.device, "createBindGroup");
        const pipelines = vi.spyOn(native.device, "createComputePipelineAsync");
        await update(
            data,
            face,
            Array.from({ length: 65 }, () => "office A\u0326"),
            layout
        );
        expect(bindings).not.toHaveBeenCalled();
        expect(pipelines).not.toHaveBeenCalled();
        expect(stage.version).toBe(version);
        const words = stage._words;
        const floats = stage._floatWords;
        const upload = stage._uploadWords;
        expect(upload.length).toBeLessThan(words.length);
        await update(
            data,
            face,
            Array.from({ length: 65 }, () => "office A\u0326"),
            layout
        );
        expect(stage._words).toBe(words);
        expect(stage._floatWords).toBe(floats);
        expect(stage._uploadWords).toBe(upload);
        expect(bindings).not.toHaveBeenCalled();
        expect(pipelines).not.toHaveBeenCalled();
        bindings.mockRestore();
        pipelines.mockRestore();
        for (const paragraphs of [["AB"], ["office   ".repeat(20), "", "A"], ["A"]]) {
            await awaitComputeTextUpdate(data, paragraphs);
            const plain = new Uint32Array(await readStorageBuffer(data._indirect));
            expect(plain[1]).toBeGreaterThan(0);
            const plainInstances = await readStorageBuffer(data._instances, 0, plain[1]! * 12);
            await update(data, face, paragraphs, { maxWidthPx: Infinity, lineHeightPx: data.fontSizePx });
            expect(await readStorageBuffer(data._instances, 0, plain[1]! * 12)).toEqual(plainInstances);
            await update(data, face, paragraphs, layout);
        }
        await awaitComputeTextUpdate(data, ["A"]);
        const resources = [stage._input!, stage._info!, stage._lines!];
        disposeComputeTextData(data);
        live.delete(data);
        expect(getComputeTextParagraphStage(data)).toBeUndefined();
        expect(resources.every((buffer) => buffer._destroyed)).toBe(true);
    });

    it("rejects busy and disposed data synchronously with paragraph-specific diagnostics", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face);
        const first = updateComputeTextParagraphs(data, ["office"], { maxWidthPx: 100 });
        expect(() => updateComputeTextParagraphs(data, ["A"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs: data is busy/);
        await (
            await first
        ).completed;
        disposeComputeTextData(data);
        live.delete(data);
        expect(() => updateComputeTextParagraphs(data, ["A"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs: data is disposed/);
    });

    it("reports GPU-detected failures through paragraph-prefixed completion and recovers on the next update", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const original = native.device.queue.writeBuffer.bind(native.device.queue);
        const writes = vi.spyOn(native.device.queue, "writeBuffer").mockImplementation((...args) => {
            original(...args);
            if (args[0] === data._slots._buffer) {
                original(data._slots._buffer!, face.oracle.glyphId(0x41) * 4, Uint32Array.of(0xffffffff));
            }
        });
        try {
            const update = await updateComputeTextParagraphs(data, ["A"], { maxWidthPx: 100 });
            await expect(update.completed).rejects.toThrow(/^updateComputeTextParagraphs: missing atlas slot$/);
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
            expect(data.status).toBe("ready");
        } finally {
            writes.mockRestore();
        }
        await update(data, face, ["A"], { maxWidthPx: 100 });
    });

    it("clears all indirect words on synchronous admission failure without changing the atlas", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face);
        for (const rejection of [() => updateComputeTextParagraphs(data, ["A\nB"], { maxWidthPx: 100 }), () => updateComputeTextParagraphs(data, ["A"], { maxWidthPx: 0 })]) {
            await update(data, face, ["A"], { maxWidthPx: 100 });
            const atlas = storage._curveSets.get(face.file)!._atlas;
            const slots = new Map(atlas._glyphSlots);
            expect(rejection).toThrow(/^updateComputeTextParagraphs:/);
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
            expect(atlas._glyphSlots).toEqual(slots);
            expect(data.status).toBe("ready");
        }
    });

    it("retains original paragraph completion semantics across a later plain update", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const paragraph = await updateComputeTextParagraphs(data, ["A A"], { maxWidthPx: 20 });
        const plain = awaitComputeTextUpdate(data, ["AB"]);
        await paragraph.completed;
        await plain;
        await update(data, face, ["B B"], { maxWidthPx: 20 });
    });

    it("releases the cached stage on the block's terminal device-loss path", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face);
        await update(data, face, ["A A"], { maxWidthPx: 20 });
        const stage = getComputeTextParagraphStage(data)!;
        const resources = [stage._input!, stage._info!, stage._lines!];
        data._onDeviceLost();
        expect(data.status).toBe("lost");
        expect(resources.every((buffer) => buffer._destroyed)).toBe(true);
        expect(getComputeTextParagraphStage(data)).toBeUndefined();
        expect(() => updateComputeTextParagraphs(data, ["A"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs: device lost/);
    });

    it("recreates stage resources and bumps its version on a device change", async () => {
        const other = await nativeHarness();
        const token = {};
        const words = Uint32Array.of(0, 1, bits(1), 1, bits(Infinity), bits(32), 0, 0);
        const stage = createComputeTextParagraphStage(token, native.engine, words);
        try {
            stage.prepare(native.engine, 0, 1);
            const before = stage.version;
            const old = [stage._input!, stage._info!, stage._lines!];
            stage.prepare(other.engine, 0, 1);
            expect(stage.version).toBeGreaterThan(before);
            expect(old.every((buffer) => buffer._destroyed)).toBe(true);
            expect(stage._device).toBe(other.device);
            await assertNativeClean(other);
        } finally {
            stage.release();
            await other.close();
        }
    });

    it.each([
        { failureAt: 299, failure: 1, limit: 300, expected: 1 },
        { failureAt: 299, failure: 3, limit: 1, expected: 2 },
        { failureAt: 0, failure: 3, limit: 1, expected: 3 },
        { failureAt: 3, failure: 1, limit: 2, expected: 2 },
        { failureAt: 0, failure: 2, limit: 300, expected: 2 },
        { failureAt: 0, failure: -3, limit: 300, expected: 3 },
    ])("reports the lowest failing paragraph or capacity overflow and draws nothing: %j", async ({ failureAt, failure, limit, expected }) => {
        const token = {};
        const count = 300;
        const capacity = 1;
        const words = new Uint32Array(8 + count * 2);
        words.set([count, capacity, bits(1), 3, bits(Infinity), bits(32), 0, 0]);
        for (let p = 0; p < count; p++) {
            words[8 + p] = 8 + count + p;
        }
        const records = new Int32Array(count * 10);
        for (let p = 0; p < count; p++) {
            records[p * 10 + 1] = 1;
            records[p * 10 + 6] = 20;
        }
        if (Math.abs(failure) === 3) {
            records[failureAt * 10 + 4] = failure < 0 ? 3 : 2;
        } else if (failure === 2) {
            records[failureAt * 10 + 1] = 2;
        } else {
            records[failureAt * 10] = failure;
        }
        const buffers = {
            shaped: createStorageBuffer(native.engine, records, { writable: true }),
            slots: createStorageBuffer(native.engine, Uint32Array.of(0, 0xfffffffe, 0xffffffff)),
            instances: createStorageBuffer(native.engine, limit * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
        const stage = createComputeTextParagraphStage(token, native.engine, words);
        const task = createComputeTask(native.engine, "paragraph-injected-status");
        try {
            stage.prepare(native.engine, count, capacity);
            stage.dispatches(buffers).forEach((dispatch) => addComputeDispatch(task, dispatch));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(buffers.status))]).toEqual([expected]);
            expect([...new Uint32Array(await readStorageBuffer(buffers.indirect))]).toEqual([6, 0, 0, 0]);
        } finally {
            task.dispose();
            stage.release();
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it("counts visible slots and keeps y advances across lines, including visible hanging spaces and protected clusters", async () => {
        const token = {};
        const count = 1;
        const capacity = 5;
        const words = Uint32Array.of(1, capacity, bits(0.5), 3, bits(6), bits(10), 0, 0, 9, 0x04040400, 1);
        const records = Int32Array.of(0, 5, 0, 0, 0, 0, 20, 2, 1, 2, 1, 1, 0, 3, -1, 1, 1, 2, 0, 4, -2, 2, 0, 3, 20, 5, 3, 3, 2, 4, 10, 6, 0, 4);
        const buffers = {
            shaped: createStorageBuffer(native.engine, records, { writable: true }),
            slots: createStorageBuffer(native.engine, Uint32Array.of(7, 0xfffffffe, 9)),
            instances: createStorageBuffer(native.engine, 36, { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
        const stage = createComputeTextParagraphStage(token, native.engine, words);
        const task = createComputeTask(native.engine, "paragraph-protected-clusters");
        try {
            stage.prepare(native.engine, count, capacity);
            stage.dispatches(buffers).forEach((dispatch) => addComputeDispatch(task, dispatch));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const info = new Uint32Array(await readStorageBuffer(stage._info!));
            expect([...info]).toEqual([0, 0, 1, 0]);
            expect([...new Uint32Array(await readStorageBuffer(stage._lines!, 0, 16))]).toEqual([0, 5, bits(20), 0]);
            const actual = new DataView(await readStorageBuffer(buffers.instances));
            expect([0, 12, 24].map((at) => [actual.getFloat32(at, true), actual.getFloat32(at + 4, true), actual.getUint32(at + 8, true)])).toEqual([
                [0.5, 1, 7],
                [11.5, 6, 7],
                [20, 9, 9],
            ]);
            // Protect the marks but allow an emergency break before record 3.
            words[9] = 0x00040400;
            stage.prepare(native.engine, count, capacity);
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(stage._info!))]).toEqual([0, 0, 2, 0]);
            expect([...new Uint32Array(await readStorageBuffer(stage._lines!, 0, 32))]).toEqual([0, 3, bits(10), 0, 3, 2, bits(10), 3]);
            expect(new DataView(await readStorageBuffer(buffers.instances)).getFloat32(16, true)).toBe(-4);
            expect([...new Uint32Array(await readStorageBuffer(buffers.indirect))]).toEqual([6, 3, 0, 0]);
            words[9] = 0;
            records[11] = records[17] = records[23] = 0;
            updateStorageBuffer(native.engine, buffers.shaped, records);
            stage.prepare(native.engine, count, capacity);
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(stage._info!))]).toEqual([0, 0, 1, 0]);
            expect([...new Uint32Array(await readStorageBuffer(stage._lines!, 0, 16))]).toEqual([0, 5, bits(20), 0]);
        } finally {
            task.dispose();
            stage.release();
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
});
