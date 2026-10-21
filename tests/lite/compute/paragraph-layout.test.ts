import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
    createComputeTextData,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    getComputeFontLineMetrics,
    updateComputeTextData,
    updateComputeTextParagraphs,
    updateGlyphStorage,
} from "../../../packages/babylon-lite/src/index";
import type { ComputeTextData, ComputeTextParagraphLayout, GlyphStorage } from "../../../packages/babylon-lite/src/index";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { createComputeTextParagraphStage, getComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import * as sourceSelection from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import * as shapePreparation from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import * as submission from "../../../packages/babylon-lite/src/compute/compute-task-readback";
import * as nominalMapping from "../../../packages/babylon-lite/src/text/compute-font";
import * as closure from "../../../packages/babylon-lite/src/text/compute-text-closure";
import * as glyphExtraction from "../../../packages/babylon-lite/src/text/compute-glyph-extraction";
import { awaitComputeTextUpdate } from "./await-update";
import { fixture, oracleFields } from "./fixtures";
import { syntheticFont } from "./synthetic";
import type { Fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { serialParagraphLayout } from "./serial-paragraph-layout";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createComputeDispatch, setComputeDispatchSize } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";

interface Line {
    first: number;
    count: number;
    width: number;
    cluster: number;
}

function bits(value: number): number {
    return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

function layoutWords(count: number, capacity: number): Uint32Array {
    const stride = Math.ceil(capacity / 4);
    const words = new Uint32Array(16 + count + count * stride);
    words.set([count, capacity, bits(1), 3, bits(Infinity), bits(32), 0, 0, 0, count, 1, count, stride]);
    for (let p = 0; p < count; p++) {
        words[16 + p] = p;
    }
    return words;
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
    const pool = stage._words[11]!;
    const info = paragraphs.length ? new Uint32Array(await readStorageBuffer(stage._info!, pool * 16, paragraphs.length * 16)) : new Uint32Array();
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
        const slotIndex = stage._words[16 + p]!;
        const records = oracleFields(face, text);
        const flags = sourceFlags(face, text);
        const expected = referenceLines(records, flags, scale, width);
        expect([...shape.slice(slotIndex * stride, slotIndex * stride + 2)], `${p}: shaped status/count`).toEqual([0, records.length]);
        records.forEach((record, i) => {
            expect([...shape.slice(slotIndex * stride + 4 + i * 6, slotIndex * stride + 10 + i * 6)], `${p}: shaped ${i}`).toEqual(record);
        });
        expect([...info.slice(p * 4, p * 4 + 4)], `${p}: paragraph info`).toEqual([visible, 0, expected.length, globalLine]);
        const flagOffset = 16 + pool + slotIndex * stage._words[12]!;
        expect([...new Uint8Array(stage._words.buffer, flagOffset * 4, flags.length)], `${p}: input flags`).toEqual(flags);
        let penX = 0;
        let penY = 0;
        expected.forEach((line, l) => {
            expect([...lines.slice((slotIndex * capacity + l) * 4, (slotIndex * capacity + l + 1) * 4)], `${p}: line ${l}`).toEqual([
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

/** Compare only defined live records; spare slots and record padding intentionally retain old contents. */
async function residentOutputs(data: ComputeTextData) {
    const stage = getComputeTextParagraphStage(data)!;
    const words = stage._words;
    const stride = 4 + words[1]! * 6;
    const shaped = new Int32Array(await readStorageBuffer(data._shaped));
    const info = new Uint32Array(await readStorageBuffer(stage._info!));
    const lines = new Uint32Array(await readStorageBuffer(stage._lines!));
    const logicalShape: number[][] = [];
    const logicalLines: number[][] = [];
    const logicalInfo: number[][] = [];
    for (let p = 0; p < words[0]!; p++) {
        const slot = words[16 + p]!;
        const start = slot * stride;
        const count = shaped[start + 1]!;
        logicalShape.push([...shaped.slice(start, start + 4 + count * 6)]);
        const result = info.subarray(slot * 4, slot * 4 + 4);
        logicalInfo.push([...result]);
        logicalLines.push(result[1] ? [] : [...lines.slice(slot * words[1]! * 4, (slot * words[1]! + result[2]!) * 4)]);
    }
    const indirect = new Uint32Array(await readStorageBuffer(data._indirect));
    return {
        shaped: logicalShape,
        lines: logicalLines,
        info: logicalInfo,
        instances: indirect[1] ? new Uint8Array(await readStorageBuffer(data._instances, 0, indirect[1]! * 12)) : new Uint8Array(),
        indirect,
        status: new Uint32Array(await readStorageBuffer(data._statusBuffer)),
    };
}

describe("native GPU paragraph wrapping and placement", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    let fused: boolean;
    const live = new Set<ComputeTextData>();
    function block(face: Fixture, curveSet: string = face.file): ComputeTextData {
        const data = createComputeTextData(native.engine, face.font, storage, curveSet, { fontSizePx: 31.75 });
        data._chunkBudgets = { monolithic: true };
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
    async function compareFresh(data: ComputeTextData, face: Fixture, paragraphs: readonly string[], layout: ComputeTextParagraphLayout): Promise<void> {
        const fresh = block(face, data._curveSet);
        fresh._disableSegmentation = data._disableSegmentation;
        try {
            await (
                await updateComputeTextParagraphs(fresh, paragraphs, layout)
            ).completed;
            expect(await residentOutputs(data)).toEqual(await residentOutputs(fresh));
        } finally {
            disposeComputeTextData(fresh);
            live.delete(fresh);
        }
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

    it.each(["Inter.ttf", "Roboto-Regular.ttf", "NotoSansGothic-Regular.ttf"] as const)(
        "reuses bounded flag windows while native output and exact backing ownership match fresh layout: %s",
        async (file) => {
            const face = fixture(file);
            const data = block(face);
            const token = file.includes("Gothic") ? "𐌰 𐌱 𐌲 " : "office affinity AV To A\u0308 ";
            let paragraphs = [token.repeat(80), token.repeat(40), "  "];
            const layout = { maxWidthPx: 221.75, align: "right" as const, lineHeightPx: 33.25, paragraphSpacingPx: 3.75 };
            await update(data, face, paragraphs, layout);
            const first = getComputeTextParagraphStage(data)!._cache!;
            expect(first._flags).toHaveLength(3);
            const clean = first._flags![1];
            const entries = first._flags!.filter((entry) => entry !== undefined);
            expect(entries.every((entry) => entry.scalars.buffer !== entry.glyphIds.buffer)).toBe(true);
            const buffers = new Set(entries.flatMap((entry) => [entry.scalars.buffer, entry.glyphIds.buffer, entry.flags.buffer]));
            expect([...buffers].reduce((sum, buffer) => sum + buffer.byteLength, 0)).toBe(entries.reduce((sum, entry) => sum + entry.scalars.length * 9, 0));
            const original = paragraphs[0]!;
            const at = Math.floor(original.length / 2);
            const scalarAt = file.includes("Gothic") && original.charCodeAt(at) >= 0xdc00 && original.charCodeAt(at) <= 0xdfff ? at - 1 : at;
            for (const text of [original.slice(0, scalarAt) + (file.includes("Gothic") ? "𐌰" : "A") + original.slice(scalarAt), original, original + " ", " " + original]) {
                paragraphs = [text, ...paragraphs.slice(1)];
                await update(data, face, paragraphs, layout);
                await compareFresh(data, face, paragraphs, layout);
                const cache = getComputeTextParagraphStage(data)!._cache!;
                expect(cache._flags![1]).toBe(clean);
                expect(cache._flags).toHaveLength(paragraphs.length);
            }
            await update(data, face, [paragraphs[0]!, "", ...paragraphs.slice(1)], layout);
            expect(getComputeTextParagraphStage(data)!._cache!._flags).toHaveLength(4);
        }
    );

    it("invalidates flag entries with ready admission errors and never publishes a busy-rejected candidate", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face);
        const text = "office affinity ".repeat(80);
        const layout = { maxWidthPx: 210 };
        await (
            await updateComputeTextParagraphs(data, [text], layout)
        ).completed;
        const old = getComputeTextParagraphStage(data)!._cache!._flags![0]!;
        const pending = updateComputeTextParagraphs(data, ["A" + text], layout);
        expect(() => updateComputeTextParagraphs(data, ["B" + text], layout)).toThrow(/busy/);
        await (
            await pending
        ).completed;
        expect(getComputeTextParagraphStage(data)!._cache).toBeUndefined();
        await (
            await updateComputeTextParagraphs(data, [text], layout)
        ).completed;
        expect(getComputeTextParagraphStage(data)!._cache!._flags![0]).not.toBe(old);
        expect(() => updateComputeTextParagraphs(data, ["A\nB"], layout)).toThrow(/line breaks/);
        expect(getComputeTextParagraphStage(data)!._cache).toBeUndefined();
        await (
            await updateComputeTextParagraphs(data, [text], layout)
        ).completed;
        const stage = getComputeTextParagraphStage(data)!;
        disposeComputeTextData(data);
        expect(stage._cache).toBeUndefined();
    });

    it("performs bounded flag work on eligible edits and fully regenerates every ineligible dirty paragraph", async () => {
        const face = fixture("Inter.ttf");
        for (const mode of ["eligible", "plan", "font", "size", "atlas", "generation", "capacity", "insert", "delete", "plain", "cacheless", "failed"] as const) {
            const data = block(face, `flags-${mode}`);
            const text = "A ".repeat(500);
            let paragraphs = [text, "B B", "C C"];
            const layout = { maxWidthPx: 200 };
            await (
                await updateComputeTextParagraphs(data, paragraphs, layout)
            ).completed;
            paragraphs[0] = text.slice(0, 500) + "B" + text.slice(501);
            if (mode === "plan") {
                paragraphs[0] = "Α ".repeat(500);
            } else if (mode === "font") {
                data._font = fixture("Roboto-Regular.ttf").font;
            } else if (mode === "size") {
                Object.assign(data, { fontSizePx: 32 });
            } else if (mode === "atlas") {
                data._storage._curveSets.get(data._curveSet)!._atlas._version++;
            } else if (mode === "generation") {
                data._generation++;
            } else if (mode === "capacity") {
                paragraphs[0] = "A ".repeat(800);
            } else if (mode === "insert") {
                paragraphs = [paragraphs[0]!, "A A", ...paragraphs.slice(1)];
            } else if (mode === "delete") {
                paragraphs = [paragraphs[0]!, paragraphs[2]!];
            } else if (mode === "plain") {
                await (
                    await updateComputeTextData(data, ["A"])
                ).completed;
            } else if (mode === "cacheless") {
                getComputeTextParagraphStage(data)!._cache = undefined;
            } else if (mode === "failed") {
                const original = submission.submitDeferredComputeTaskResult;
                vi.spyOn(submission, "submitDeferredComputeTaskResult").mockImplementationOnce((...args) => original(...args).then(() => 1));
                await expect((await updateComputeTextParagraphs(data, [text, "B B", "C C"], layout)).completed).rejects.toThrow(/unsupported/);
            }
            const classify = vi.spyOn(sourceSelection, "computeTextNoBreakBefore");
            await (
                await updateComputeTextParagraphs(data, paragraphs, layout)
            ).completed;
            const actual = classify.mock.calls.length;
            classify.mockRestore();
            if (mode === "eligible") {
                expect(actual).toBe(67);
            } else {
                const allDirty = ["font", "size", "atlas", "generation", "capacity", "plain", "cacheless", "failed"].includes(mode);
                const expected = (allDirty ? paragraphs : paragraphs.slice(0, mode === "insert" ? 2 : 1)).reduce(
                    (sum, value) => sum + Array.from(value).filter((cp) => cp !== " ").length,
                    0
                );
                expect(actual, mode).toBe(expected);
            }
            expect(getComputeTextParagraphStage(data)!._cache!._flags).toHaveLength(paragraphs.length);
            vi.restoreAllMocks();
        }
    });

    it("budgets optional capacity, counts growth rebuilds, and rejects only required overflow without drawing", async () => {
        const face = syntheticFont("multiple");
        const data = block(face);
        const limits = native.device.limits;
        vi.spyOn(native.device, "limits", "get").mockReturnValue(
            new Proxy(limits, {
                get(target, key) {
                    return key === "maxStorageBufferBindingSize" ? 3000 : Reflect.get(target, key, target);
                },
            })
        );
        const layout = { maxWidthPx: 40, lineHeightPx: 32 };
        const capacities: number[] = [];
        const revisions = new Set<number>();
        for (const length of [20, 21, 25, 26, 31]) {
            const paragraphs = ["A".repeat(length)];
            await update(data, face, paragraphs, layout);
            const cache = getComputeTextParagraphStage(data)!._cache!;
            expect(cache._pool).toBe(1);
            capacities.push(cache._capacity);
            revisions.add(cache._baseRevision);
            await compareFresh(data, face, paragraphs, layout);
        }
        expect(capacities).toEqual([50, 50, 50, 62, 62]);
        expect(revisions.size).toBe(2);
        const atlas = storage._curveSets.get(data._curveSet)!._atlas;
        const version = atlas._version;
        expect(() => updateComputeTextParagraphs(data, ["A".repeat(32)], layout)).toThrow(
            `updateComputeTextParagraphs: layout lines buffer of 3072 bytes exceeds device storage limits ` +
                `(maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=3000, pool=1, capacity=64)`
        );
        expect(data.status).toBe("ready");
        expect(getComputeTextParagraphStage(data)!._cache).toBeUndefined();
        expect(storage._curveSets.get(data._curveSet)!._atlas).toBe(atlas);
        expect(atlas._version).toBe(version);
        expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        await update(data, face, ["A".repeat(20)], layout);
        expect(getComputeTextParagraphStage(data)!._cache!._capacity).toBe(50);
    });

    it("retains clean multi-paragraph slots with a budgeted pool and compacts when insertion headroom is exhausted", async () => {
        const face = syntheticFont("multiple");
        const data = block(face);
        const limits = native.device.limits;
        vi.spyOn(native.device, "limits", "get").mockReturnValue(
            new Proxy(limits, {
                get(target, key) {
                    return key === "maxStorageBufferBindingSize" ? 4800 : Reflect.get(target, key, target);
                },
            })
        );
        const layout = { maxWidthPx: 40, lineHeightPx: 32 };
        let paragraphs = ["A".repeat(10), "B".repeat(10), "C".repeat(10)];
        await update(data, face, paragraphs, layout);
        const stage = getComputeTextParagraphStage(data)!;
        expect(stage._cache!._pool).toBe(4);
        expect(stage._cache!._capacity).toBe(25);
        const revision = stage._cache!._baseRevision;
        const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
        paragraphs = [paragraphs[0]!, "B".repeat(9), paragraphs[2]!];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(mapping.mock.calls.map((call) => call[1].length)).toEqual([9]);
        mapping.mockRestore();
        expect(stage._cache!._baseRevision).toBe(revision);
        expect([...stage._cache!._order]).toEqual([0, 1, 2]);
        await compareFresh(data, face, paragraphs, layout);
        paragraphs = [paragraphs[0]!, "B".repeat(10), ...paragraphs.slice(1)];
        await update(data, face, paragraphs, layout);
        expect(stage._cache!._baseRevision).toBe(revision);
        expect([...stage._cache!._order]).toEqual([0, 3, 1, 2]);
        await compareFresh(data, face, paragraphs, layout);
        paragraphs.push("C".repeat(10));
        await update(data, face, paragraphs, layout);
        expect(stage._cache!._baseRevision).toBeGreaterThan(revision);
        expect(stage._cache!._pool).toBe(5);
        expect(stage._cache!._capacity).toBe(20);
        expect([...stage._cache!._order]).toEqual([0, 1, 2, 3, 4]);
        await compareFresh(data, face, paragraphs, layout);
    });

    it("clears a previously visible draw on required budget overflow without installing new glyphs", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const limits = native.device.limits;
        vi.spyOn(native.device, "limits", "get").mockReturnValue(
            new Proxy(limits, {
                get(target, key) {
                    return key === "maxStorageBufferBindingSize" ? 8 << 20 : Reflect.get(target, key, target);
                },
            })
        );
        const layout = { maxWidthPx: 600 };
        await (
            await updateComputeTextParagraphs(data, ["A".repeat(5000)], layout)
        ).completed;
        expect(new Uint32Array(await readStorageBuffer(data._indirect))[1]).toBe(5000);
        const atlas = storage._curveSets.get(data._curveSet)!._atlas;
        const version = atlas._version;
        const extract = vi.spyOn(glyphExtraction, "extractComputeGlyphCurves");
        expect(() => updateComputeTextParagraphs(data, ["B".repeat(200000)], layout)).toThrow(/layout lines buffer.*maxStorageBufferBindingSize=8388608, pool=1/);
        expect(extract).not.toHaveBeenCalled();
        expect(atlas._version).toBe(version);
        expect(storage._curveSets.get(data._curveSet)!._atlas).toBe(atlas);
        expect(data.status).toBe("ready");
        expect(getComputeTextParagraphStage(data)!._cache).toBeUndefined();
        expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("is bit-identical to fresh updates through a four-medium-paragraph editing sequence: %s", async (file) => {
        const face = fixture(file);
        const data = block(face);
        const medium = "office affinity AV To A\u0326 ".repeat(40);
        let paragraphs = [medium, medium + "A", medium + "B", medium + "C"];
        let layout: ComputeTextParagraphLayout = { maxWidthPx: 620 };
        const edit = async (next: string[], nextLayout = layout) => {
            paragraphs = next;
            layout = nextLayout;
            await (
                await updateComputeTextParagraphs(data, paragraphs, layout)
            ).completed;
            await compareFresh(data, face, paragraphs, layout);
        };
        await edit(paragraphs);
        await edit([paragraphs[0]!, paragraphs[1]! + "x", paragraphs[2]!, paragraphs[3]!]); // Character insertion.
        await edit([paragraphs[0]!, paragraphs[1]!.slice(0, -1), paragraphs[2]!, paragraphs[3]!]); // Character deletion.
        await edit([paragraphs[0]!, paragraphs[1]!.slice(0, 480), paragraphs[1]!.slice(480), ...paragraphs.slice(2)]); // Enter.
        expect([...getComputeTextParagraphStage(data)!._cache!._order]).not.toEqual(paragraphs.map((_p, i) => i));
        await edit([paragraphs[0]!, paragraphs[1]! + paragraphs[2]!, ...paragraphs.slice(3)]); // Backspace.
        await edit([paragraphs[0]!, "pasted office", "", "affinity AV To", ...paragraphs.slice(2)]);
        await edit([paragraphs[0]!, ...paragraphs.slice(3)]); // Pure paragraph deletion.
        await edit([...paragraphs, "appended office"]);
        await edit([paragraphs[0]! + "A", paragraphs[2]!, paragraphs[1]!, ...paragraphs.slice(3)]); // Disjoint/reorder-like edits.
        await edit(paragraphs, { maxWidthPx: 390.5 });
        await edit(paragraphs, { maxWidthPx: 390.5, align: "center", lineHeightPx: 38.25, paragraphSpacingPx: 7.5 });
        await edit(paragraphs, { ...layout, align: "right" });
        await edit(paragraphs);
        const oldCapacity = getComputeTextParagraphStage(data)!._cache!._capacity;
        await edit([paragraphs[0]!, "office ".repeat(oldCapacity), ...paragraphs.slice(2)]);
        expect(getComputeTextParagraphStage(data)!._cache!._capacity).toBeGreaterThan(oldCapacity);
        expect([...getComputeTextParagraphStage(data)!._cache!._order]).toEqual(paragraphs.map((_p, i) => i));
        await edit([]);
        await edit(["", "A", ""]);
    });

    it.each([
        [100, false],
        [100, true],
        [500, false],
    ] as const)("prepares only one dirty paragraph in a %s-paragraph document (whole-run=%s)", async (count, wholeRun) => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        data._disableSegmentation = wholeRun;
        const text = "office affinity AV To A\u0326 ".repeat(12);
        const paragraphs = Array.from({ length: count }, (_, p) => text + (p % 2 ? "A" : "B"));
        const layout = { maxWidthPx: 620 };
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const stage = getComputeTextParagraphStage(data)!;
        const buffers = [data._shaped, stage._info, stage._lines];
        const before = await residentOutputs(data);
        const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
        const closing = vi.spyOn(closure, "closeComputeTextGlyphs");
        const outlines = vi.spyOn(glyphExtraction, "extractComputeGlyphCurves");
        const prepare = vi.spyOn(shapePreparation, "prepareComputeTextShapeWords");
        const writes = vi.spyOn(native.device.queue, "writeBuffer");
        const textures = vi.spyOn(native.device.queue, "writeTexture");
        const submit = vi.spyOn(submission, "submitDeferredComputeTaskResult");
        paragraphs[57] = paragraphs[57]! + "A"; // Existing closure glyph, nonzero physical runBase.
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(mapping).toHaveBeenCalledTimes(1);
        expect(closing).toHaveBeenCalledTimes(1);
        expect(outlines).not.toHaveBeenCalled();
        expect(textures).not.toHaveBeenCalled();
        expect(writes.mock.calls.some(([buffer]) => buffer === data._slots._buffer)).toBe(false);
        const inputWrites = writes.mock.calls.filter(([buffer]) => buffer === stage._input!._buffer);
        expect(inputWrites.map((call) => call[4])).toEqual([64, stage._words[12]! * 4]);
        expect(writes.mock.calls.reduce((bytes, call) => bytes + (call[4] as number), 0)).toBeLessThan(8 * paragraphs[57]!.length + stage._words[12]! * 4 + 256);
        expect(prepare).toHaveBeenCalledTimes(1);
        const tasks = submit.mock.calls.at(-1)![0];
        const dispatches = tasks.flatMap((task) => task.dispatches);
        expect(dispatches.find((dispatch) => dispatch.shader.name === "compute-text-shape")?._x).toBe(wholeRun ? 1 : prepare.mock.calls[0]![1].length);
        const stitch = dispatches.find((dispatch) => dispatch.shader.name === "compute-text-stitch");
        expect(stitch?._x).toBe(wholeRun ? undefined : 1);
        expect(prepare.mock.calls[0]![2]).toBe(wholeRun ? 57 : 0);
        expect([data._shaped, stage._info, stage._lines]).toEqual(buffers);
        const after = await residentOutputs(data);
        for (let p = 0; p < paragraphs.length; p++) {
            if (p !== 57) {
                expect(after.shaped[p]).toEqual(before.shaped[p]);
                expect(after.lines[p]).toEqual(before.lines[p]);
                expect(after.info[p]).toEqual(before.info[p]);
            }
        }
        vi.restoreAllMocks();
        await compareFresh(data, face, paragraphs, layout);
    });

    it("shapes/stitches a nonzero-base dirty batch across several script groups", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const layout = { maxWidthPx: 190 };
        const paragraphs = ["office affinity ".repeat(20), "α βγ ".repeat(40), "Москва Іі ".repeat(30), "A"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        paragraphs[1] += "α";
        paragraphs[2] += "І";
        const prepare = vi.spyOn(shapePreparation, "prepareComputeTextShapeWords");
        const submit = vi.spyOn(submission, "submitDeferredComputeTaskResult");
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(prepare).toHaveBeenCalledTimes(2);
        expect(prepare.mock.calls[0]![2]).toBe(0);
        expect(prepare.mock.calls[1]![2]).toBe(prepare.mock.calls[0]![1].length);
        expect(submit.mock.calls[0]![0].flatMap((task) => task.dispatches).find((dispatch) => dispatch.shader.name === "compute-text-stitch")?._x).toBe(2);
        expect(getComputeTextParagraphStage(data)!._words[8]).toBe(1);
        vi.restoreAllMocks();
        await compareFresh(data, face, paragraphs, layout);
        await verifyLayout(data, face, paragraphs, layout, fused);
    });

    it("does no preparation/shape/stitch work for placement, width, deletion, or unchanged updates", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        let paragraphs = ["office affinity ".repeat(25), "AV To ".repeat(30), "A"];
        let layout: ComputeTextParagraphLayout = { maxWidthPx: 500 };
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        for (const mode of ["unchanged", "width", "placement", "deletion", "empty"] as const) {
            if (mode === "width") {
                layout = { maxWidthPx: 240 };
            } else if (mode === "placement") {
                layout = { ...layout, align: "right", lineHeightPx: 40, paragraphSpacingPx: 5 };
            } else if (mode === "deletion") {
                paragraphs = [paragraphs[0]!, paragraphs[2]!];
            } else if (mode === "empty") {
                paragraphs = [];
            }
            const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
            const prepare = vi.spyOn(shapePreparation, "prepareComputeTextShapeWords");
            const pipelines = vi.spyOn(native.device, "createComputePipelineAsync");
            const writes = vi.spyOn(native.device.queue, "writeBuffer");
            const submit = vi.spyOn(submission, "submitDeferredComputeTaskResult");
            await (
                await updateComputeTextParagraphs(data, paragraphs, layout)
            ).completed;
            const stage = getComputeTextParagraphStage(data)!;
            expect(mapping).not.toHaveBeenCalled();
            expect(prepare).not.toHaveBeenCalled();
            expect(pipelines).not.toHaveBeenCalled();
            expect(stage._words[9]).toBe(mode === "width" ? paragraphs.length : 0);
            expect(stage._dispatches![0]!.enabled).toBe(mode === "width");
            expect(submit.mock.calls[0]![0].map((task) => task.name)).toEqual(["compute-text-resident-pack"]);
            expect(submit.mock.calls[0]![0][0]!.dispatches.some((dispatch) => dispatch.shader.name === "compute-text-stitch")).toBe(false);
            expect(writes.mock.calls.filter(([buffer]) => buffer === stage._input!._buffer).reduce((bytes, call) => bytes + (call[4] as number), 0)).toBeLessThanOrEqual(72);
            vi.restoreAllMocks();
            await compareFresh(data, face, paragraphs, layout);
        }
    });

    it("compacts on pool exhaustion and retains scalar flags when a dirty slot shrinks", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        let paragraphs = ["A\u0308 A\u0326 ".repeat(16), "B", "C", "D"];
        const layout = { maxWidthPx: 60 };
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const pool = getComputeTextParagraphStage(data)!._cache!._pool;
        paragraphs[0] = "A";
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const stage = getComputeTextParagraphStage(data)!;
        const flagStart = 16 + pool + stage._words[16]! * stage._words[12]!;
        expect([...stage._words.slice(flagStart, flagStart + stage._words[12]!)].every((word) => word === 0)).toBe(true);
        await compareFresh(data, face, paragraphs, layout);
        paragraphs = Array.from({ length: pool + 1 }, () => "A");
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(stage._cache!._pool).toBeGreaterThan(pool);
        expect([...stage._cache!._order]).toEqual(paragraphs.map((_p, i) => i));
        await compareFresh(data, face, paragraphs, layout);
    });

    it("plain updates and synchronous errors invalidate reuse, then recover through a full rebuild", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const paragraphs = ["office affinity", "AV To", "A"];
        const layout = { maxWidthPx: 100 };
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        for (const invalidate of [
            async () => (await updateComputeTextData(data, ["A"])).completed,
            () => {
                expect(() => updateComputeTextParagraphs(data, ["bad\ntext"], layout)).toThrow(/line breaks/);
            },
        ]) {
            await invalidate();
            const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
            await (
                await updateComputeTextParagraphs(data, paragraphs, layout)
            ).completed;
            expect(mapping).toHaveBeenCalledTimes(paragraphs.length);
            mapping.mockRestore();
            await compareFresh(data, face, paragraphs, layout);
        }
    });

    it("adds new glyph outlines/slot entries only and rebuilds on atlas capacity growth", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, "paragraph-additive-closure");
        const layout = { maxWidthPx: 100 };
        let paragraphs = ["ABCDE", "A"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const stage = getComputeTextParagraphStage(data)!;
        const atlas = storage._curveSets.get(data._curveSet)!._atlas;
        const revision = stage._cache!._baseRevision;
        const slots = new Map(atlas._glyphSlots);
        const installedGlyphs = new Set(data._slotGlyphIds);
        const firstCurveTexel = atlas._curveTexelsUsed;
        const firstBandTexel = atlas._bandTexelsUsed;
        const firstSlot = atlas._slotCount;
        const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
        const outlines = vi.spyOn(glyphExtraction, "extractComputeGlyphCurves");
        const writes = vi.spyOn(native.device.queue, "writeBuffer");
        const textures = vi.spyOn(native.device.queue, "writeTexture");
        paragraphs = ["ABCDF", "A"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(stage._cache!._baseRevision).toBe(revision);
        expect(mapping).toHaveBeenCalledTimes(1);
        expect(outlines).toHaveBeenCalledTimes(1);
        for (const [glyph, slot] of slots) {
            expect(atlas._glyphSlots.get(glyph)).toBe(slot);
        }
        const meta = writes.mock.calls.filter(([buffer]) => buffer === atlas._gpu!._metaBuf);
        expect(meta.map((call) => [call[1], call[4]])).toEqual([[firstSlot * 48, (atlas._slotCount - firstSlot) * 48]]);
        const entries = writes.mock.calls.filter(([buffer]) => buffer === data._slots._buffer);
        expect(entries.map((call) => [call[1], call[4]])).toEqual(data._slotGlyphIds.filter((id) => !installedGlyphs.has(id)).map((id) => [id * 4, 4]));
        expect(textures.mock.calls.map((call) => call[2].offset)).toEqual([firstCurveTexel * 16, firstBandTexel * 16]);
        vi.restoreAllMocks();
        await compareFresh(data, face, paragraphs, layout);
        // Keep paragraph lengths inside the existing stride; the new repertoire exceeds atlas metadata capacity.
        const cap = stage._cache!._capacity;
        const alphabet = "GHIJKLMNOPQRSTUVWXYZ".match(/.{1,5}/gu)!;
        paragraphs = [paragraphs[0]!, ...alphabet, "A"];
        const oldMeta = atlas._gpu!._metaBuf;
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(atlas._gpu!._metaBuf).not.toBe(oldMeta);
        expect(stage._cache!._capacity).toBe(cap);
        expect(stage._cache!._baseRevision).toBeGreaterThan(revision);
        expect([...stage._cache!._order]).toEqual(paragraphs.map((_p, i) => i));
        await compareFresh(data, face, paragraphs, layout);
    });

    it("keeps logical failure ordering in cached slot info and invalidates after a failed incremental submission", async () => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const layout = { maxWidthPx: 100 };
        const paragraphs = ["office A", "office B", "office C", "office D"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const stage = getComputeTextParagraphStage(data)!;
        const original = submission.submitDeferredComputeTaskResult;
        const injected = vi.spyOn(submission, "submitDeferredComputeTaskResult").mockImplementationOnce((tasks, buffer, bytes, consume, staging) => {
            const slot = stage._words[16 + 2]!;
            const stride = 4 + stage._words[1]! * 6;
            updateStorageBuffer(native.engine, data._shaped, Int32Array.of(1, 0, 4, 2), slot * stride * 4);
            // Run the real layout over a rejecting clean slot and the previously shaped dirty slot.
            stage._words[9] = paragraphs.length;
            stage._words[10] = 1;
            setComputeDispatchSize(stage._dispatches![0]!, { x: paragraphs.length });
            updateStorageBuffer(native.engine, stage._input!, stage._words.subarray(0, 16));
            return original(tasks.slice(-1), buffer, bytes, consume, staging);
        });
        paragraphs[1] += "A";
        const failed = await updateComputeTextParagraphs(data, paragraphs, layout);
        await expect(failed.completed).rejects.toThrow(/unsupported lookup/);
        expect(stage._cache).toBeUndefined();
        expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        injected.mockRestore();
        const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(mapping).toHaveBeenCalledTimes(paragraphs.length);
        mapping.mockRestore();
        await compareFresh(data, face, paragraphs, layout);
    });

    it("matches fresh failing paragraphs and their later fixes through real unsupported lookups", async () => {
        const face = syntheticFont("pair-reject");
        const data = block(face);
        const fresh = block(face);
        const layout = { maxWidthPx: 100, lineHeightPx: 32 };
        const paragraphs = ["A", "BA", "C"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        paragraphs[1] = "BC";
        await expect((await updateComputeTextParagraphs(data, paragraphs, layout)).completed).rejects.toThrow(/unsupported lookup/);
        await expect((await updateComputeTextParagraphs(fresh, paragraphs, layout)).completed).rejects.toThrow(/unsupported lookup/);
        expect(await residentOutputs(data)).toEqual(await residentOutputs(fresh));
        expect(getComputeTextParagraphStage(data)!._cache).toBeUndefined();
        paragraphs[1] = "BA";
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        await compareFresh(data, face, paragraphs, layout);
    });

    it("rebuilds after external atlas mutation, including a formerly empty glyph becoming visible", async () => {
        const face = fixture("Inter.ttf");
        const data = block(face, "paragraph-external-mutation");
        const layout = { maxWidthPx: 100 };
        const paragraphs = ["A A", "B B"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const cs = storage._curveSets.get(data._curveSet)!;
        const source = cs._curves.get(face.oracle.glyphId(0x41))!;
        const space = face.oracle.glyphId(0x20);
        expect(data._slotWords![space]).toBe(0xfffffffe);
        updateGlyphStorage(storage, data._curveSet, new Map([[space, { ...source, glyphId: space }]]));
        const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(mapping).toHaveBeenCalledTimes(paragraphs.length);
        expect(data._slotWords![space]).toBe(cs._atlas._glyphSlots.get(space)!._index);
        mapping.mockRestore();
        await compareFresh(data, face, paragraphs, layout);
    });

    it.each([false, true])("late completion failure invalidates descendants but not a newer full rebuild (rebuild=%s)", async (full) => {
        const face = fixture("Roboto-Regular.ttf");
        const data = block(face);
        const layout = { maxWidthPx: 100 };
        let paragraphs = ["office A", "office B", "office C"];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        let release!: () => void;
        const gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        const original = submission.submitDeferredComputeTaskResult;
        const held = vi.spyOn(submission, "submitDeferredComputeTaskResult").mockImplementationOnce((...args) => original(...args).then(() => gate.then(() => 1)));
        paragraphs = ["office AA", ...paragraphs.slice(1)];
        const previous = await updateComputeTextParagraphs(data, paragraphs, layout);
        if (full) {
            await (
                await updateComputeTextData(data, ["A"])
            ).completed;
        }
        paragraphs = [paragraphs[0]!, paragraphs[1]! + "A", paragraphs[2]!];
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        const stage = getComputeTextParagraphStage(data)!;
        const latest = stage._cache;
        release();
        await expect(previous.completed).rejects.toThrow(/unsupported lookup/);
        expect(stage._cache).toBe(full ? latest : undefined);
        held.mockRestore();
        const mapping = vi.spyOn(nominalMapping, "mapComputeFontScalars");
        await (
            await updateComputeTextParagraphs(data, paragraphs, layout)
        ).completed;
        expect(mapping).toHaveBeenCalledTimes(full ? 0 : paragraphs.length);
        mapping.mockRestore();
        await compareFresh(data, face, paragraphs, layout);
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
        expect(stage._uploadRanges).toEqual([
            [0, 16],
            [81, 81],
            [16 + stage._words[11]!, 16 + stage._words[11]!],
        ]);
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
        expect(getComputeTextParagraphStage(data)!._cache).toBeUndefined();
        await update(data, face, ["office"], { maxWidthPx: 100 });
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
        const words = layoutWords(0, 1);
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
        const words = layoutWords(count, capacity);
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

    it("preserves cached slot failures and selects the first failing logical paragraph after reordering or deletion", async () => {
        const token = {};
        const count = 4;
        const words = layoutWords(count, 1);
        words.set([2, 0, 3, 1], 16);
        const records = new Int32Array(count * 10);
        for (let p = 0; p < count; p++) {
            records[p * 10 + 1] = 1;
            records[p * 10 + 6] = 20;
        }
        records[0] = 1;
        records[20] = 3;
        const buffers = {
            shaped: createStorageBuffer(native.engine, records, { writable: true }),
            slots: createStorageBuffer(native.engine, Uint32Array.of(0)),
            instances: createStorageBuffer(native.engine, count * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
        const stage = createComputeTextParagraphStage(token, native.engine, words);
        const task = createComputeTask(native.engine, "paragraph-cached-status");
        try {
            stage.prepare(native.engine, count, 1);
            stage.dispatches(buffers).forEach((dispatch) => addComputeDispatch(task, dispatch));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(buffers.status))]).toEqual([3]);
            const slots = await readStorageBuffer(stage._info!, 0, count * 16);
            words.set([3, 0, 1, 2], 16);
            words[8] = 3;
            words[9] = 1;
            words[10] = 0;
            stage.prepare(native.engine, 0, 1);
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(buffers.status))]).toEqual([1]);
            expect(await readStorageBuffer(stage._info!, 0, count * 16)).toEqual(slots);
            words[0] = 2;
            words[9] = 0;
            words.set([3, 1], 16);
            stage.prepare(native.engine, 0, 1);
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(buffers.status))]).toEqual([0]);
            expect([...new Uint32Array(await readStorageBuffer(buffers.indirect))]).toEqual([6, 2, 0, 0]);
            expect(await readStorageBuffer(stage._info!, 0, count * 16)).toEqual(slots);
        } finally {
            task.dispose();
            stage.release();
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it("matches serial WGSL bit-for-bit for blocked scans, candidate events, fallback, failures and incremental reuse (seed 0x31415926)", async () => {
        let seed = 0x31415926;
        const random = (limit: number) => {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
            return seed % limit;
        };
        const capacity = 4097;
        const counts = [0, 1, 257, 2049, 4097, 513, 900, 300, 30, 70, 256, 258, 60, 500, 900, 100];
        const words = layoutWords(counts.length, capacity);
        words[2] = bits(0.02343751);
        words[4] = bits(41.37);
        words[5] = bits(31.12345);
        words[6] = bits(7.12345);
        const records = new Int32Array(counts.length * (4 + capacity * 6));
        counts.forEach((count, p) => {
            const head = p * (4 + capacity * 6);
            records[head + 1] = count;
            let c = 0;
            for (let i = 0; i < count; i++) {
                if (i && random(4)) {
                    c++;
                }
                const at = head + 4 + i * 6;
                const flags = p === 6 ? 4 : p === 7 ? 0 : random(8);
                const flagAt = 16 + words[11]! + p * words[12]! + Math.floor(c / 4);
                words[flagAt] = (words[flagAt]! & ~(255 << ((c % 4) * 8))) | (flags << ((c % 4) * 8));
                records.set([random(2), c, random(90), random(31) - 15, random(70) - 35, random(70) - 35], at);
            }
        });
        // Entirely unbreakable, entirely space, negative advance, and signed-prefix overflow.
        for (let i = 0; i < counts[8]!; i++) {
            records[8 * (4 + capacity * 6) + 5 + i * 6] = 0;
        }
        words.fill(0x01010101, 16 + words[11]! + 9 * words[12]!, 16 + words[11]! + 10 * words[12]!);
        records[10 * (4 + capacity * 6) + 6 + 130 * 6] = -300;
        records[11 * (4 + capacity * 6) + 6] = 0x7fffffff;
        const shared = {
            shaped: createStorageBuffer(native.engine, records, { writable: true }),
            slots: createStorageBuffer(native.engine, Uint32Array.of(13, 0xfffffffe, 0xffffffff)),
        };
        const stages = [0, 1].map(() => createComputeTextParagraphStage({}, native.engine, words.slice()));
        const tasks = [0, 1].map((i) => createComputeTask(native.engine, `parallel-serial-${i}`));
        const outputs = [0, 1].map(() => ({
            instances: createStorageBuffer(native.engine, counts.reduce((a, b) => a + b, 0) * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        }));
        const serialShaders: ReturnType<typeof createComputeShader>[] = [];
        try {
            stages.forEach((stage, index) => {
                stage.prepare(native.engine, counts.length, capacity);
                const dispatches = stage.dispatches({ ...shared, ...outputs[index]! });
                if (index) {
                    for (const [i, entryPoint] of [
                        [0, "serialMeasureTest"],
                        [2, "serialPlaceTest"],
                    ] as const) {
                        const shader = createComputeShader(native.engine, {
                            computeSource: stage._shaders![i]!._source + serialParagraphLayout,
                            entryPoint,
                            bindings: stage._shaders![i]!._decls,
                        });
                        serialShaders.push(shader);
                        dispatches[i] = createComputeDispatch(
                            shader,
                            createComputeBindingSet(shader, { input: stage._input!, ...shared, ...outputs[index]!, info: stage._info!, lines: stage._lines! }),
                            { size: { x: Math.ceil(counts.length / 64) } }
                        );
                    }
                } else {
                    for (const i of [0, 2]) {
                        setComputeDispatchSize(dispatches[i]!, { x: 4, y: Math.ceil(counts.length / 4) });
                    }
                }
                dispatches.forEach((dispatch) => addComputeDispatch(tasks[index]!, dispatch));
                tasks[index]!.record();
            });
            async function compare(): Promise<void> {
                for (const task of tasks) {
                    await prepareComputeTask(task);
                    submitComputeTasks([task]);
                }
                const results = await Promise.all(
                    stages.map(async (stage, i) => ({
                        lines: new Uint8Array(await readStorageBuffer(stage._lines!, 0, counts.length * capacity * 16)),
                        info: new Uint8Array(await readStorageBuffer(stage._info!)),
                        instances: new Uint8Array(await readStorageBuffer(outputs[i]!.instances)),
                        indirect: new Uint8Array(await readStorageBuffer(outputs[i]!.indirect)),
                        status: new Uint8Array(await readStorageBuffer(outputs[i]!.status)),
                    }))
                );
                expect(results[0]).toEqual(results[1]);
            }
            for (const width of [0.0001, 41.37, 1000000, Infinity]) {
                for (const align of width === Infinity ? [0] : [0, 1, 2]) {
                    stages.forEach((stage) => {
                        stage._words[4] = bits(width);
                        stage._words[7] = align;
                        updateStorageBuffer(native.engine, stage._input!, stage._words);
                    });
                    await compare();
                }
            }
            // Clean slots retain tables while one dirty physical slot changes; display order is permuted.
            records[2 * (4 + capacity * 6) + 6] = records[2 * (4 + capacity * 6) + 6]! + 10;
            updateStorageBuffer(native.engine, shared.shaped, records);
            stages.forEach((stage) => {
                stage._words[8] = 2;
                stage._words[9] = 1;
                stage._words[10] = 0;
                stage._words.set(
                    counts.map((_n, i) => counts.length - i - 1),
                    16
                );
                updateStorageBuffer(native.engine, stage._input!, stage._words);
            });
            await compare();
            for (const [failure, count, glyph] of [
                [1, 1, 2],
                [0, capacity + 1, 2],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 0],
            ]) {
                stages.forEach((stage) => {
                    stage._words[8] = 0;
                    updateStorageBuffer(native.engine, stage._input!, stage._words);
                });
                records[0] = failure!;
                records[1] = count!;
                records[4] = glyph!;
                updateStorageBuffer(native.engine, shared.shaped, records);
                await compare();
            }
        } finally {
            tasks.forEach((task) => task.dispose());
            stages.forEach((stage) => stage.release());
            serialShaders.forEach(disposeComputeShader);
            [...Object.values(shared), ...outputs.flatMap(Object.values)].forEach(disposeStorageBuffer);
        }
    });

    it("matches the serial oracle for hinted gallop and lane line sweeps through 120001 records (seed 0x6a09e667)", async () => {
        const capacity = 120001;
        const words = layoutWords(1, capacity);
        words[2] = bits(0.02343751);
        words[5] = bits(31.12345);
        const records = new Int32Array(4 + capacity * 6);
        const shared = {
            shaped: createStorageBuffer(native.engine, records, { writable: true }),
            slots: createStorageBuffer(native.engine, Uint32Array.of(13, 0xfffffffe, 0xffffffff)),
        };
        const stages = [0, 1].map(() => createComputeTextParagraphStage({}, native.engine, words.slice()));
        const tasks = [0, 1].map((i) => createComputeTask(native.engine, `gallop-sweep-oracle-${i}`));
        const outputs = [0, 1].map(() => ({
            instances: createStorageBuffer(native.engine, capacity * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        }));
        const serialShaders: ReturnType<typeof createComputeShader>[] = [];
        let seed = 0x6a09e667;
        const random = (limit: number) => {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
            return seed % limit;
        };
        try {
            stages.forEach((stage, index) => {
                stage.prepare(native.engine, 1, capacity);
                const dispatches = stage.dispatches({ ...shared, ...outputs[index]! });
                if (index) {
                    for (const [i, entryPoint] of [
                        [0, "serialMeasureTest"],
                        [2, "serialPlaceTest"],
                    ] as const) {
                        const shader = createComputeShader(native.engine, {
                            computeSource: stage._shaders![i]!._source + serialParagraphLayout,
                            entryPoint,
                            bindings: stage._shaders![i]!._decls,
                        });
                        serialShaders.push(shader);
                        dispatches[i] = createComputeDispatch(
                            shader,
                            createComputeBindingSet(shader, { input: stage._input!, ...shared, ...outputs[index]!, info: stage._info!, lines: stage._lines! }),
                            { size: { x: 1 } }
                        );
                    }
                }
                dispatches.forEach((dispatch) => addComputeDispatch(tasks[index]!, dispatch));
                tasks[index]!.record();
            });
            const cases = [
                ...[0, 1, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025].map((count) => ({ count, kind: "boundary", width: 0.02343751 * 13 })),
                { count: 4097, kind: "zero-tail", width: 41.37 },
                { count: 4097, kind: "spaces", width: 0.0001 },
                { count: 4097, kind: "guarded", width: 0.0001 },
                { count: 4097, kind: "clusters", width: 41.37 },
                { count: 1025, kind: "negative", width: 41.37 },
                { count: 1025, kind: "overflow", width: 41.37 },
                { count: capacity, kind: "seeded", width: 600 },
                { count: capacity, kind: "zero-tail", width: Infinity },
            ];
            for (const { count, kind, width } of cases) {
                records.fill(0);
                records[1] = count;
                words.fill(0, 17);
                let cluster = 0;
                for (let i = 0; i < count; i++) {
                    if (i && (kind !== "clusters" || random(3))) {
                        cluster++;
                    }
                    const flag = kind === "spaces" ? 1 : kind === "guarded" ? 4 : kind === "seeded" || kind === "clusters" ? random(8) : 0;
                    const flagAt = 17 + Math.floor(cluster / 4);
                    words[flagAt] = (words[flagAt]! & ~(255 << ((cluster % 4) * 8))) | (flag << ((cluster % 4) * 8));
                    const advance = kind === "boundary" ? 13 : kind === "zero-tail" ? (i === 0 ? 1000000 : 0) : random(90);
                    records.set([kind === "boundary" ? 0 : random(2), cluster, advance, random(9) - 4, random(17) - 8, random(17) - 8], 4 + i * 6);
                }
                if (kind === "negative") {
                    records[6 + 513 * 6] = -300;
                } else if (kind === "overflow") {
                    records[6] = 0x7fffffff;
                    records[12] = 1;
                }
                updateStorageBuffer(native.engine, shared.shaped, records);
                stages.forEach((stage) => {
                    stage._words.set(words);
                    stage._words[4] = bits(width);
                    updateStorageBuffer(native.engine, stage._input!, stage._words);
                });
                for (const task of tasks) {
                    await prepareComputeTask(task);
                    submitComputeTasks([task]);
                }
                const results = await Promise.all(
                    stages.map(async (stage, i) => {
                        const info = new Uint32Array(await readStorageBuffer(stage._info!));
                        const visible = info[0]!;
                        return {
                            info,
                            lines: new Uint32Array(await readStorageBuffer(stage._lines!, 0, info[2]! * 16)),
                            instances: new Uint32Array(await readStorageBuffer(outputs[i]!.instances, 0, Math.max(4, visible * 12))),
                            indirect: new Uint32Array(await readStorageBuffer(outputs[i]!.indirect)),
                            status: new Uint32Array(await readStorageBuffer(outputs[i]!.status)),
                        };
                    })
                );
                expect(results[0], `${kind}/${count}/${width}: serial output`).toEqual(results[1]);
                expect(new Int32Array(await readStorageBuffer(shared.shaped))).toEqual(records);
                if (count) {
                    const table = new Uint32Array(await readStorageBuffer(stages[0]!._lines!, capacity * 16, count * 32));
                    const lines = results[1]!.lines;
                    let line = 0;
                    for (let i = 0; i < count; i++) {
                        while (line + 1 < results[1]!.info[2]! && lines[(line + 1) * 4]! <= i) {
                            line++;
                        }
                        expect(table[i * 8 + 4], `${kind}/${count}: record ${i} line`).toBe(line);
                    }
                }
            }
        } finally {
            tasks.forEach((task) => task.dispose());
            stages.forEach((stage) => stage.release());
            serialShaders.forEach(disposeComputeShader);
            [...Object.values(shared), ...outputs.flatMap(Object.values)].forEach(disposeStorageBuffer);
        }
    }, 120000);

    describe("successor-prefix fits", () => {
        interface ParagraphCase {
            records: number[][];
            flags: number[];
            serial?: boolean;
        }

        const scale = Math.fround(0.37);
        const paragraph = (advances: readonly number[]): ParagraphCase => ({
            records: advances.map((advance, i) => [0, i, advance, 0, 0, 0]),
            flags: advances.map(() => 0),
        });
        const widthOf = (advance: number) => Math.fround(Math.fround(advance) * scale);
        const nextDown = (value: number) => {
            expect(value).toBeGreaterThan(0);
            expect(Number.isFinite(value)).toBe(true);
            return new Float32Array(Uint32Array.of(bits(value) - 1).buffer)[0]!;
        };

        async function compare(paragraphs: readonly ParagraphCase[], widths: readonly number[], checkPath = false): Promise<void> {
            const capacity = Math.max(1, ...paragraphs.map((p) => p.records.length));
            const words = layoutWords(paragraphs.length, capacity);
            words[2] = bits(scale);
            words[5] = bits(31.12345);
            const stride = 4 + capacity * 6;
            const records = new Int32Array(paragraphs.length * stride);
            paragraphs.forEach((p, index) => {
                records[index * stride + 1] = p.records.length;
                p.records.forEach((record, i) => records.set(record, index * stride + 4 + i * 6));
                p.flags.forEach((flag, cluster) => {
                    const at = 16 + paragraphs.length + index * words[12]! + Math.floor(cluster / 4);
                    words[at] = words[at]! | (flag << ((cluster % 4) * 8));
                });
            });
            const shared = {
                shaped: createStorageBuffer(native.engine, records, { writable: true }),
                slots: createStorageBuffer(native.engine, Uint32Array.of(13, 0xfffffffe, 0xffffffff)),
            };
            const stages = [0, 1].map(() => createComputeTextParagraphStage({}, native.engine, words.slice()));
            const tasks = [0, 1].map((i) => createComputeTask(native.engine, `successor-prefix-oracle-${i}`));
            const outputs = [0, 1].map(() => ({
                instances: createStorageBuffer(native.engine, Math.max(4, paragraphs.reduce((n, p) => n + p.records.length, 0) * 12), { writable: true }),
                indirect: createStorageBuffer(native.engine, 16, { writable: true }),
                status: createStorageBuffer(native.engine, 4, { writable: true }),
            }));
            const serialShaders: ReturnType<typeof createComputeShader>[] = [];
            try {
                stages.forEach((stage, index) => {
                    stage.prepare(native.engine, paragraphs.length, capacity);
                    const dispatches = stage.dispatches({ ...shared, ...outputs[index]! });
                    if (index) {
                        for (const [i, entryPoint] of [
                            [0, "serialMeasureTest"],
                            [2, "serialPlaceTest"],
                        ] as const) {
                            const shader = createComputeShader(native.engine, {
                                computeSource: stage._shaders![i]!._source + serialParagraphLayout,
                                entryPoint,
                                bindings: stage._shaders![i]!._decls,
                            });
                            serialShaders.push(shader);
                            dispatches[i] = createComputeDispatch(
                                shader,
                                createComputeBindingSet(shader, { input: stage._input!, ...shared, ...outputs[index]!, info: stage._info!, lines: stage._lines! }),
                                { size: { x: Math.ceil(paragraphs.length / 64) } }
                            );
                        }
                    }
                    dispatches.forEach((dispatch) => addComputeDispatch(tasks[index]!, dispatch));
                    tasks[index]!.record();
                });
                for (const width of widths) {
                    stages.forEach((stage) => {
                        stage._words[4] = bits(width);
                        updateStorageBuffer(native.engine, stage._input!, stage._words);
                    });
                    for (const task of tasks) {
                        await prepareComputeTask(task);
                        submitComputeTasks([task]);
                    }
                    const actual = await Promise.all(
                        stages.map(async (stage, i) => ({
                            info: new Uint32Array(await readStorageBuffer(stage._info!)),
                            lines: new Uint32Array(await readStorageBuffer(stage._lines!, 0, paragraphs.length * capacity * 16)),
                            instances: new Uint32Array(await readStorageBuffer(outputs[i]!.instances)),
                            indirect: new Uint32Array(await readStorageBuffer(outputs[i]!.indirect)),
                            status: new Uint32Array(await readStorageBuffer(outputs[i]!.status)),
                        }))
                    );
                    expect(actual[0], `width ${width}: serial WGSL`).toEqual(actual[1]);
                    const result = actual[0]!;
                    expect([...result.status]).toEqual([0]);
                    const table = new Uint32Array(await readStorageBuffer(stages[0]!._lines!, paragraphs.length * capacity * 16, paragraphs.length * capacity * 32));
                    paragraphs.forEach((p, index) => {
                        const lineCount = result.info[index * 4 + 2]!;
                        const lines = result.lines.subarray(index * capacity * 4, (index * capacity + lineCount) * 4);
                        // The JS reference uses unbounded pen sums; signed-overflow cases use the unchanged serial WGSL oracle.
                        if (!p.serial) {
                            const expected = referenceLines(p.records, p.flags, scale, Math.fround(width));
                            expect([...lines], `${index}/${width}: f32 JS reference`).toEqual(expected.flatMap((line) => [line.first, line.count, bits(line.width), line.cluster]));
                        }
                        let line = 0;
                        for (let i = 0; i < p.records.length; i++) {
                            while (line + 1 < lineCount && lines[(line + 1) * 4]! <= i) {
                                line++;
                            }
                            expect(table[(index * capacity + i) * 8 + 4], `${index}/${i}: P9 line assignment`).toBe(line);
                        }
                        if (checkPath) {
                            const next = Array.from({ length: p.records.length }, (_n, i) => table[(index * capacity + i) * 8 + 7]!);
                            if (p.serial) {
                                expect(
                                    next.every((value) => value === p.records.length),
                                    `${index}: serial leaves P7 next values initialized`
                                ).toBe(true);
                            } else {
                                expect(next[0], `${index}: existing P7-only next-value witness`).toBeLessThan(p.records.length);
                                expect(next[0]).toBe(lines[0]! + lines[1]!);
                            }
                        }
                    });
                    expect(new Int32Array(await readStorageBuffer(shared.shaped))).toEqual(records);
                }
            } finally {
                tasks.forEach((task) => task.dispose());
                stages.forEach((stage) => stage.release());
                serialShaders.forEach(disposeComputeShader);
                [...Object.values(shared), ...outputs.flatMap(Object.values)].forEach(disposeStorageBuffer);
            }
        }

        it("keeps final-record decisions exact at counts 1/2/256/257/512/513 and one f32 step below", async () => {
            for (const count of [1, 2, 256, 257, 512, 513]) {
                const p = paragraph(Array.from({ length: count }, () => 13));
                const width = widthOf(count * 13);
                await compare([p], [width, nextDown(width)]);
            }
        });

        it("preserves rounded boundaries for sampled starts and cross-lane successors", async () => {
            const p = paragraph(Array.from({ length: 1025 }, (_n, i) => 1 + ((i * 37) % 89)));
            const prefix = [0];
            for (const record of p.records) {
                prefix.push(prefix.at(-1)! + record[2]!);
            }
            const widths = [0.0001, Infinity];
            for (const [s, k] of [
                [0, 1],
                [1, 4],
                [3, 5],
                [254, 258],
                [510, 513],
                [1000, 1025],
            ]) {
                const width = widthOf(prefix[k!]! - prefix[s!]!);
                widths.push(width, nextDown(width));
            }
            await compare([p], widths);
        });

        it("handles equal prefixes across lane/span boundaries and paragraph tails with emergency-only breaks", async () => {
            const p = paragraph(Array.from({ length: 4097 }, (_n, i) => (i % 17 >= 14 || (i >= 254 && i <= 258) || i >= 4080 ? 0 : 7)));
            await compare([p], [0.0001, widthOf(7), nextDown(widthOf(7)), widthOf(70), Infinity]);
        });

        it("keeps the exact signed-prefix limit parallel and overflow or negative advances serial", async () => {
            const count = 513;
            const exact = paragraph(Array.from({ length: count }, (_n, i) => (i ? 1 : 0x7fffffff - (count - 1))));
            const overflow = paragraph(exact.records.map((record) => record[2]!));
            overflow.records[count - 1]![2]!++;
            overflow.serial = true;
            const negative = paragraph(Array.from({ length: count }, () => 13));
            negative.records[256]![2] = -1;
            negative.serial = true;
            await compare([exact, overflow, negative], [0.0001], true);
        });

        it("matches fixed-seed mixed batches including adjacent final-paragraph bounds", async () => {
            let seed = 0x75a31f29;
            const random = (limit: number) => {
                seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                return seed % limit;
            };
            const paragraphs = [0, 1, 2, 255, 256, 257, 512, 513, 1024, 1025, random(1026), 2, 1, 257].map((count) => {
                const p = paragraph(Array.from({ length: count }, () => random(90)));
                let cluster = 0;
                p.records.forEach((record, i) => {
                    if (i && random(4)) {
                        cluster++;
                    }
                    record[0] = random(2);
                    record[1] = cluster;
                    record[3] = random(9) - 4;
                    record[4] = random(17) - 8;
                    record[5] = random(17) - 8;
                    p.flags[cluster] = random(8);
                });
                return p;
            });
            await compare(paragraphs, [0.0001, widthOf(13), nextDown(widthOf(13)), 41.37, Infinity]);
        });
    });

    it("counts visible slots and keeps y advances across lines, including visible hanging spaces and protected clusters", async () => {
        const token = {};
        const count = 1;
        const capacity = 5;
        const words = layoutWords(count, capacity);
        words[2] = bits(0.5);
        words[4] = bits(6);
        words[5] = bits(10);
        words[17] = 0x04040400;
        words[18] = 1;
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
            const info = new Uint32Array(await readStorageBuffer(stage._info!, count * 16, count * 16));
            expect([...info]).toEqual([0, 0, 1, 0]);
            expect([...new Uint32Array(await readStorageBuffer(stage._lines!, 0, 16))]).toEqual([0, 5, bits(20), 0]);
            const actual = new DataView(await readStorageBuffer(buffers.instances));
            expect([0, 12, 24].map((at) => [actual.getFloat32(at, true), actual.getFloat32(at + 4, true), actual.getUint32(at + 8, true)])).toEqual([
                [0.5, 1, 7],
                [11.5, 6, 7],
                [20, 9, 9],
            ]);
            // Protect the marks but allow an emergency break before record 3.
            words[17] = 0x00040400;
            stage.prepare(native.engine, count, capacity);
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(stage._info!, count * 16, count * 16))]).toEqual([0, 0, 2, 0]);
            expect([...new Uint32Array(await readStorageBuffer(stage._lines!, 0, 32))]).toEqual([0, 3, bits(10), 0, 3, 2, bits(10), 3]);
            expect(new DataView(await readStorageBuffer(buffers.instances)).getFloat32(16, true)).toBe(-4);
            expect([...new Uint32Array(await readStorageBuffer(buffers.indirect))]).toEqual([6, 3, 0, 0]);
            words[17] = 0;
            records[11] = records[17] = records[23] = 0;
            updateStorageBuffer(native.engine, buffers.shaped, records);
            stage.prepare(native.engine, count, capacity);
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Uint32Array(await readStorageBuffer(stage._info!, count * 16, count * 16))]).toEqual([0, 0, 1, 0]);
            expect([...new Uint32Array(await readStorageBuffer(stage._lines!, 0, 16))]).toEqual([0, 5, bits(20), 0]);
        } finally {
            task.dispose();
            stage.release();
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
});
