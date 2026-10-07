import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
    createComputeTextData,
    createComputeTextLayer,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    updateComputeTextData,
} from "../../../packages/babylon-lite/src/index";
import type { ComputeTextData, GlyphStorage } from "../../../packages/babylon-lite/src/index";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { closeComputeTextGlyphs } from "../../../packages/babylon-lite/src/text/compute-text-closure";
import { prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { clearTextPipelineCache, getOrCreateTextPipeline } from "../../../packages/babylon-lite/src/text/_gpu/text-pipeline";
import { fixture, GREEK, CYRILLIC, oracleFields } from "./fixtures";
import type { Fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { readGlyfGlyph } from "../../../packages/babylon-lite/src/text/font-parser/glyf";
import { syntheticFont } from "./synthetic";
import * as programCompiler from "../../../packages/babylon-lite/src/text/compute-text-program";
import * as glyfReader from "../../../packages/babylon-lite/src/text/font-parser/glyf";
import * as shapeWords from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import * as glyphClosure from "../../../packages/babylon-lite/src/text/compute-text-closure";
import * as nominalMapping from "../../../packages/babylon-lite/src/text/compute-font";

function capacity(font: Fixture, texts: readonly string[]): number {
    return Math.max(
        1,
        ...texts.map((text) => {
            const selected = selectComputeTextRun(text);
            return prepareComputeTextShapeWords(createComputeTextPlan(font.font, selected), [
                { scalars: selected._scalars, glyphIds: selected._scalars.map((cp) => mapComputeFontScalar(font.font, cp)) },
            ]).capacity;
        })
    );
}

async function verifyResident(data: ComputeTextData, font: Fixture, texts: readonly string[]): Promise<void> {
    const stride = 4 + capacity(font, texts) * 6;
    const shaped = new Int32Array(await readStorageBuffer(data._shaped));
    const instances = await readStorageBuffer(data._instances);
    const view = new DataView(instances);
    const atlas = data._storage._curveSets.get(data._curveSet)?._atlas;
    const scale = Math.fround(data.fontSizePx / font.font.unitsPerEm);
    let visible = 0;
    texts.forEach((text, run) => {
        const expected = oracleFields(font, text);
        expect([...shaped.slice(run * stride, run * stride + 2)], `${text}: status/count`).toEqual([0, expected.length]);
        let x = 0;
        let y = 0;
        expected.forEach((record, index) => {
            expect([...shaped.slice(run * stride + 4 + index * 6, run * stride + 10 + index * 6)], `${font.file} ${JSON.stringify(text)} record ${index}`).toEqual(record);
            const slot = atlas?._glyphSlots.get(record[0]!);
            if (slot) {
                const at = visible * 12;
                expect(view.getFloat32(at, true), `${text}: instance ${visible} x`).toBe(Math.fround(Math.fround(x + record[4]!) * scale));
                const scaledY = Math.fround(Math.fround(y + record[5]!) * scale);
                expect(view.getFloat32(at + 4, true), `${text}: instance ${visible} y`).toBe(Math.fround(scaledY - Math.fround(run * Math.fround(data.fontSizePx))));
                expect(view.getUint32(at + 8, true), `${text}: atlas/style word`).toBe(slot._index);
                visible++;
            }
            x += record[2]!;
            y += record[3]!;
        });
    });
    expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, visible, 0, 0]);
    expect(data._instances._data).toBeNull();
    expect(data._shaped._data).toBeNull();
}

describe("native M1–3 resident production shaping and packing", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    const live = new Set<ComputeTextData>();
    function block(font: Fixture, key: string = font.file, fontSizePx = 32): ComputeTextData {
        const data = createComputeTextData(native.engine, font.font, storage, key, { fontSizePx });
        live.add(data);
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
        for (const data of live) {
            disposeComputeTextData(data);
        }
        disposeGlyphStorage(storage);
        await native?.close();
    });

    it.each(["pair-first", "pair-second"] as const)("packs correct resident advances for disjoint and overlapping pair commits: %s", async (kind) => {
        const font = syntheticFont(kind);
        const data = block(font, kind);
        const texts = ["AB", "ABC", "AABC", "BC", "ABA", "ABC".repeat(8)];
        await updateComputeTextData(data, texts);
        await verifyResident(data, font, texts);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("resident records, packing and indirect: %s", async (file) => {
        const font = fixture(file);
        const data = block(font);
        expect(createComputeTextLayer(data).data).toBe(data);
        const transitions = [
            [],
            [""],
            ["", ""],
            [" ", "  "],
            ["fi", "A\u0308", "B\u0301\u0307", "1 < A"],
            ...[2, 4, 8, 16, 32, 50, 49, 50, 2].map((length) => ["A".repeat(length), "", "fi ".repeat(Math.floor(length / 2))]),
            [GREEK, CYRILLIC, "A", ""],
            ["A".repeat(512)],
            ["A"],
        ];
        for (const texts of transitions) {
            const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
            await updateComputeTextData(data, texts);
            expect(
                maps.mock.calls.map((call) => call.slice(0, 3)),
                `${file} update must read only status`
            ).toEqual([[GPUMapMode.READ, 0, 4]]);
            maps.mockRestore();
            await verifyResident(data, font, texts);
        }
        const before = data._instances._buffer;
        await updateComputeTextData(data, ["A"]);
        expect(data._instances._buffer).toBe(before);
    });

    it.each(["NotoSansHebrew-Regular.ttf", "NotoSansArabic-Regular.ttf", "NotoSansGothic-Regular.ttf", "NotoSansSC-Regular-subset.ttf"] as const)(
        "resident RTL/scalar repertoire: %s",
        async (file) => {
            const font = fixture(file);
            const data = block(font, file, 32);
            const texts = file.includes("Hebrew")
                ? ["שלום", "שָׁלוֹם", "אבגד"]
                : file.includes("Arabic")
                  ? ["كتب", "سلام", "بَّ", "العربية"]
                  : file.includes("Gothic")
                    ? ["𐌰𐌱𐌲", "𐌳"]
                    : ["中文", "中"];
            await updateComputeTextData(data, texts);
            await verifyResident(data, font, texts);
        }
    );

    it("groups decoded input records directly without allocating a second per-run descriptor", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "group-record-ownership");
        const texts = ["office", "", GREEK, CYRILLIC, "A", ""];
        const prepare = vi.spyOn(shapeWords, "prepareComputeTextShapeWords");
        try {
            await updateComputeTextData(data, texts);
            const runs = prepare.mock.calls.flatMap(([, group]) => group);
            expect(runs.length).toBe(texts.length);
            for (const run of runs) {
                expect(run).toHaveProperty("selection._scalars", run.scalars);
                expect(run).toHaveProperty("selection._nominal.glyphIds", run.glyphIds);
            }
        } finally {
            prepare.mockRestore();
        }
        await verifyResident(data, font, texts);
    });

    it("keeps immutable compilation and pipelines warm while updating words, counts, selections and capacities", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "warm-program");
        await updateComputeTextData(data, ["HH", "", "H".repeat(50)]);
        const compile = vi.spyOn(programCompiler, "compileComputeTextProgram");
        const pipelines = vi.spyOn(native.device, "createComputePipelineAsync");
        await updateComputeTextData(data, ["office", "A\u0326", "", "H".repeat(50)]);
        expect(compile).not.toHaveBeenCalled();
        expect(pipelines).not.toHaveBeenCalled();
        await verifyResident(data, font, ["office", "A\u0326", "", "H".repeat(50)]);
        await updateComputeTextData(data, [GREEK]);
        expect(compile).toHaveBeenCalledTimes(1);
        compile.mockClear();
        await updateComputeTextData(data, ["HH", GREEK, "H", GREEK]);
        expect(compile).not.toHaveBeenCalled();
        expect(pipelines).not.toHaveBeenCalled();
        await verifyResident(data, font, ["HH", GREEK, "H", GREEK]);
    });

    it.each([512, 1024, 2048])("packs a complete resident batch of %i with exact counts and ABI", async (count) => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, `resident-large-${count}`);
        const texts = Array.from({ length: count }, (_, run) => (run % 2 ? "office fi To A\u0326 ".repeat(4).slice(0, 50) : "H".repeat(50)));
        await updateComputeTextData(data, texts);
        await verifyResident(data, font, texts);
        await updateComputeTextData(data, ["H"]);
        await verifyResident(data, font, ["H"]);
    });

    it("reuses admitted immutable outlines and component lists across warm closure revisions, including empty glyphs", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "warm-outlines");
        await updateComputeTextData(data, ["H H"]);
        const reads = vi.spyOn(glyfReader, "readGlyfGlyph");
        await updateComputeTextData(data, ["HH", "H ".repeat(25)]);
        expect(reads).not.toHaveBeenCalled();
        await verifyResident(data, font, ["HH", "H ".repeat(25)]);
        await updateComputeTextData(data, ["é"]);
        expect(reads).toHaveBeenCalled();
        reads.mockClear();
        await updateComputeTextData(data, ["é é", "H"]);
        expect(reads).not.toHaveBeenCalled();
        await verifyResident(data, font, ["é é", "H"]);
    });

    it("reuses resident shape and pack bind groups until buffer identities grow", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "warm-bindings");
        await updateComputeTextData(data, ["office affinity", "H".repeat(50)]);
        const groups = vi.spyOn(native.device, "createBindGroup");
        await updateComputeTextData(data, ["affinity", "H".repeat(49)]);
        expect(groups).not.toHaveBeenCalled();
        await verifyResident(data, font, ["affinity", "H".repeat(49)]);
        await updateComputeTextData(data, ["H".repeat(512)]);
        expect(groups).toHaveBeenCalled();
        await verifyResident(data, font, ["H".repeat(512)]);
    });

    it("copies checked warm nominal mappings directly while decoding current inputs and verifying all records", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "warm-nominal-vector");
        await updateComputeTextData(data, ["H".repeat(50)]);
        const mappings = vi.spyOn(nominalMapping, "mapComputeFontScalar");
        const texts = Array.from({ length: 100 }, () => "H".repeat(49));
        await updateComputeTextData(data, texts);
        expect(mappings.mock.calls.length).toBe(0);
        mappings.mockRestore();
        await verifyResident(data, font, texts);
    });

    it("submits resident shaping, useful packing and its four-byte completion copy together", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "single-completion");
        await updateComputeTextData(data, ["office affinity", "HH"]);
        const submissions = vi.spyOn(native.device.queue, "submit");
        await updateComputeTextData(data, ["affinity", "H"]);
        expect(submissions).toHaveBeenCalledTimes(1);
        submissions.mockRestore();
        await verifyResident(data, font, ["affinity", "H"]);
    });

    it("decodes checked warm nominal vectors into the scalar slab and collects closure seeds without retraversing each glyph vector", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "fused-ingestion");
        const texts = Array.from({ length: 100 }, (_, run) => (run % 2 ? "office fi To A\u0326 ".repeat(4).slice(0, 50) : "H".repeat(50)));
        await updateComputeTextData(data, texts);
        const preparation = vi.spyOn(shapeWords, "prepareComputeTextShapeWords");
        const iterations = vi.spyOn(Uint32Array.prototype, Symbol.iterator);
        await updateComputeTextData(
            data,
            texts.map((text) => text.slice(1))
        );
        expect(iterations.mock.calls.length).toBe(0);
        const runs = preparation.mock.calls[0]![1];
        expect(runs).toHaveLength(100);
        for (const run of runs) {
            expect(run.glyphIds.buffer).toBe(run.scalars.buffer);
            expect(run.scalars.byteOffset + run.scalars.byteLength).toBeLessThanOrEqual(run.glyphIds.byteOffset);
        }
        iterations.mockRestore();
        preparation.mockRestore();
        await verifyResident(
            data,
            font,
            texts.map((text) => text.slice(1))
        );
    });

    it("owns one disjoint batch input slab per update instead of one backing allocation per run", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "batch-input-slab");
        const texts = Array.from({ length: 100 }, (_, run) => (run % 2 ? "office A\u0326" : "HHHH"));
        await updateComputeTextData(data, texts);
        const preparation = vi.spyOn(shapeWords, "prepareComputeTextShapeWords");
        await updateComputeTextData(data, texts);
        const firstRuns = preparation.mock.calls[0]![1];
        const buffer = firstRuns[0]!.scalars.buffer;
        expect(firstRuns.at(-1)!.scalars.buffer).toBe(buffer);
        for (let run = 1; run < firstRuns.length; run++) {
            const previous = firstRuns[run - 1]!;
            expect(firstRuns[run]!.scalars.byteOffset).toBeGreaterThanOrEqual(previous.glyphIds.byteOffset + previous.glyphIds.byteLength);
        }
        await updateComputeTextData(
            data,
            texts.map((text) => text.slice(1))
        );
        expect(preparation.mock.calls[1]![1][0]!.scalars.buffer).not.toBe(buffer);
        preparation.mockRestore();
        await verifyResident(
            data,
            font,
            texts.map((text) => text.slice(1))
        );
    });

    it("packs independent runs in parallel after ordered visible-count prefixing, preserving the exact output ABI", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "parallel-packing");
        const modules = vi.spyOn(native.device, "createShaderModule");
        const texts = Array.from({ length: 100 }, (_, index) => (index % 2 ? "office fi To A\u0326 ".repeat(4).slice(0, 50) : "H".repeat(50)));
        await updateComputeTextData(data, texts);
        expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("fn countRuns(") && descriptor.code.includes("fn prefixRuns("))).toBe(true);
        expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("@workgroup_size(64)"))).toBe(true);
        modules.mockRestore();
        await verifyResident(data, font, texts);
    });

    it("fences lane-striped resident counts before leader prefixing without a separate prefix dispatch", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "fused-count-prefix");
        const texts = Array.from({ length: 129 }, (_, run) => (run % 3 === 0 ? "A  " : run % 2 ? "office A\u0326 " : "HH"));
        await updateComputeTextData(data, texts);
        const dispatches = vi.spyOn(GPUComputePassEncoder.prototype, "dispatchWorkgroups");
        await updateComputeTextData(data, texts);
        expect(dispatches.mock.calls.length).toBe(3);
        dispatches.mockRestore();
        await verifyResident(data, font, texts);
    });

    it("retains the guaranteed 256-lane fused count entry while keeping every large-batch record and packed instance", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "wide-count-prefix");
        const modules = vi.spyOn(native.device, "createShaderModule");
        const texts = Array.from({ length: 512 }, (_, run) => (run % 2 ? "office A\u0326 ".repeat(4) : "H".repeat(50)));
        await updateComputeTextData(data, texts);
        expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("@workgroup_size(256)\nfn countRuns("))).toBe(true);
        modules.mockRestore();
        await verifyResident(data, font, texts);
    });

    it("scales counting over multiple workgroups above the fused tier without truncating any resident run", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "parallel-count-tier");
        const texts = Array.from({ length: 512 }, (_, run) => (run % 2 ? "office A\u0326 ".repeat(4) : "H".repeat(50)));
        await updateComputeTextData(data, texts);
        const dispatches = vi.spyOn(GPUComputePassEncoder.prototype, "dispatchWorkgroups");
        await updateComputeTextData(data, texts);
        expect(dispatches.mock.calls.length).toBe(4);
        expect(dispatches.mock.calls[1]?.[0]).toBe(8);
        dispatches.mockRestore();
        await verifyResident(data, font, texts);
    });

    it("packs mutable resident input words once per plan group, not again to calculate capacity", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "single-ingestion");
        await updateComputeTextData(data, ["office affinity", "H".repeat(50)]);
        const preparation = vi.spyOn(shapeWords, "prepareComputeTextShapeWords");
        await updateComputeTextData(data, ["affinity", "H".repeat(49)]);
        expect(preparation).toHaveBeenCalledTimes(1);
        preparation.mockRestore();
        await verifyResident(data, font, ["affinity", "H".repeat(49)]);
    });

    it("passes the current unique repertoire to closure, not one copied seed per input glyph", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "unique-repertoire");
        const closure = vi.spyOn(glyphClosure, "closeComputeTextGlyphs");
        const texts = Array.from({ length: 100 }, () => "H".repeat(50));
        await updateComputeTextData(data, texts);
        expect(closure.mock.calls[0]![2]).toEqual([mapComputeFontScalar(font.font, 0x48)]);
        closure.mockRestore();
        await verifyResident(data, font, texts);
    });

    it("shared atlas and independent blocks: closure, cold growth, cross-font ownership and 100 blocks", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "shared-roboto");
        await updateComputeTextData(data, ["A"]);
        const atlas = storage._curveSets.get("shared-roboto")!._atlas;
        const stable = atlas._glyphSlots.get(font.oracle.glyphId(0x41))!._index;
        const oldMetadata = atlas._metaData;
        const blocks = Array.from({ length: 100 }, (_, index) => block(font, "shared-roboto", index % 2 ? 48 : 32));
        await Promise.all(blocks.map((item, index) => updateComputeTextData(item, ["A".repeat(2 + (index % 49))])));
        for (let index = 0; index < blocks.length; index++) {
            await verifyResident(blocks[index]!, font, ["A".repeat(2 + (index % 49))]);
        }
        const accented = "é";
        const selected = selectComputeTextRun(accented);
        const closure = closeComputeTextGlyphs(font.font, [createComputeTextPlan(font.font, selected)], [font.oracle.glyphId(0xe9), font.oracle.glyphId(0x66)]);
        expect(closure.glyphIds.length).toBeGreaterThan(2);
        await updateComputeTextData(data, [GREEK, CYRILLIC, accented, "ffi"]);
        expect(atlas._glyphSlots.get(font.oracle.glyphId(0x41))!._index).toBe(stable);
        expect(atlas._metaData).not.toBe(oldMetadata);
        for (const record of oracleFields(font, "ffi")) {
            expect(atlas._glyphSlots.has(record[0]!)).toBe(true);
        }
        const components = readGlyfGlyph(font.font._bytes, font.font._directory, font.font._locaFormat, font.font.glyphCount, font.oracle.glyphId(0xe9)).componentGlyphIds;
        expect(components.length).toBeGreaterThan(0);
        for (const component of components) {
            expect(closure.glyphIds).toContain(component);
            expect(atlas._glyphSlots.has(component)).toBe(true);
        }
        await verifyResident(data, font, [GREEK, CYRILLIC, accented, "ffi"]);
        const inter = fixture("Inter.ttf");
        const sibling = block(inter, "shared-inter", 48);
        await Promise.all([updateComputeTextData(sibling, ["AC", "A\u0326"]), updateComputeTextData(blocks[99]!, ["AAA"])]);
        disposeComputeTextData(data);
        live.delete(data);
        await updateComputeTextData(sibling, ["AC"]);
        await verifyResident(sibling, inter, ["AC"]);
        await verifyResident(blocks[99]!, font, ["A".repeat(3)]);
        expect(storage._curveSets.has("shared-roboto")).toBe(true);
    });

    it("rejection clears every indirect word and recovery keeps the caller atlas", async () => {
        const font = fixture("Inter.ttf");
        const data = block(font, "failure");
        for (const text of ["A\u20de", "A\u20dd", "<-", "\n", "Aא", "\ud800", "😀"]) {
            await updateComputeTextData(data, ["A"]);
            await expect(updateComputeTextData(data, ["A", text])).rejects.toThrow();
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
            expect(data.status).toBe("ready");
        }
        await updateComputeTextData(data, ["B"]);
        await verifyResident(data, font, ["B"]);
    });

    it("transitive selected GSUB closure installs empty-outline outputs, never phantom instances", async () => {
        const font = syntheticFont("transitive");
        const plan = createComputeTextPlan(font.font, selectComputeTextRun("A"));
        expect(closeComputeTextGlyphs(font.font, [plan], [1]).glyphIds).toEqual([1, 2, 3]);
        const data = block(font, "transitive");
        await updateComputeTextData(data, ["A"]);
        expect(oracleFields(font, "A")).toEqual([[3, 0, 300, 0, 0, 0]]);
        await verifyResident(data, font, ["A"]);
        const upload = data._slots._data!;
        const slots = new Uint32Array(upload.buffer, upload.byteOffset, upload.byteLength / 4);
        expect([...slots.slice(1, 4)]).toEqual([0xfffffffe, 0xfffffffe, 0xfffffffe]);
    });

    it("kernel failure statuses (internal slot injection, not public admission)", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "injection");
        await updateComputeTextData(data, ["A"]);
        const selected = selectComputeTextRun("A");
        const plan = createComputeTextPlan(font.font, selected);
        const groups = [{ plan, runs: [{ scalars: selected._scalars, glyphIds: Uint32Array.of(font.oracle.glyphId(0x41)) }], outputRun: 0 }];
        updateStorageBuffer(native.engine, data._slots, Uint32Array.of(0xffffffff), font.oracle.glyphId(0x41) * 4);
        const token = {};
        try {
            const status = await executeResidentComputeText(
                token,
                native.engine,
                groups,
                {
                    shaped: data._shaped,
                    instances: data._instances,
                    indirect: data._indirect,
                    slots: data._slots,
                    status: data._statusBuffer,
                },
                1,
                1,
                32,
                font.font.unitsPerEm,
                font.font.glyphCount,
                () => undefined
            );
            expect(status).toBe(3);
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        } finally {
            releaseResidentComputeText(token);
        }
        await updateComputeTextData(data, ["A"]);
        await verifyResident(data, font, ["A"]);
    });

    it("style ABI uses each font's unitsPerEm and requested pixel size", () => {
        const font = fixture("NotoSansGothic-Regular.ttf");
        const writes = vi.spyOn(native.device.queue, "writeBuffer");
        const data = block(font, "style", 48);
        const write = writes.mock.calls.find(([buffer]) => buffer === data._style)!;
        expect([...new Float32Array(write[2] as ArrayBuffer)]).toEqual([1, 1, 1, 1, Math.fround(font.font.unitsPerEm / 48), 0, 0, 0]);
    });

    it("kernel overflow clears draw count (internal undersized instance allocation)", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const data = block(font, "overflow");
        await updateComputeTextData(data, ["AB"]);
        const selected = selectComputeTextRun("AB");
        const groups = [
            {
                plan: createComputeTextPlan(font.font, selected),
                runs: [{ scalars: selected._scalars, glyphIds: Uint32Array.of(font.oracle.glyphId(0x41), font.oracle.glyphId(0x42)) }],
                outputRun: 0,
            },
        ];
        const instances = createStorageBuffer(native.engine, 12, { writable: true, label: "test-undersized-instances" });
        const token = {};
        try {
            const status = await executeResidentComputeText(
                token,
                native.engine,
                groups,
                {
                    shaped: data._shaped,
                    instances,
                    indirect: data._indirect,
                    slots: data._slots,
                    status: data._statusBuffer,
                },
                2,
                1,
                32,
                font.font.unitsPerEm,
                font.font.glyphCount,
                () => undefined
            );
            expect(status).toBe(2);
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        } finally {
            releaseResidentComputeText(token);
            disposeStorageBuffer(instances);
        }
    });

    it("nonvisual actual indirect draws reuse shared native atlas resources and invalidate sibling bindings on growth", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const first = block(font, "native-atlas");
        const second = block(font, "native-atlas");
        await updateComputeTextData(first, ["A"]);
        await updateComputeTextData(second, ["A"]);
        const device = native.device;
        const uniforms = Array.from({ length: 2 }, () => device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
        const draws = [createComputeTextLayer(first), createComputeTextLayer(second)].map((layer, index) => layer._createDraw(native.engine, uniforms[index]!));
        const pipeline = getOrCreateTextPipeline(native.engine, "rgba8unorm", 1, null, false);
        const target = device.createTexture({ size: [32, 32], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const atlas = storage._curveSets.get("native-atlas")!._atlas;
        function record(): void {
            const encoder = device.createCommandEncoder();
            const bundle = device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
            bundle.setPipeline(pipeline._pipeline);
            const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
            for (const draw of draws) {
                draw._prepare(pipeline._cache._bindGroupLayout);
                if (draw._active) {
                    expect(draw._record(bundle, pipeline._cache._quadVertexBuffer, pipeline._pipeline, pipeline._variantPipeline)).toBe(1);
                }
            }
            pass.executeBundles([bundle.finish()]);
            pass.end();
            device.queue.submit([encoder.finish()]);
        }
        try {
            record();
            await device.queue.onSubmittedWorkDone();
            const old = atlas._gpu!;
            const handles = [old._curveTex, old._bandTex, old._metaBuf];
            const versions = draws.map((draw) => draw._version);
            record();
            expect(draws.map((draw) => draw._version)).toEqual(versions);
            await updateComputeTextData(first, [GREEK, CYRILLIC, "office"]);
            record();
            expect([atlas._gpu!._curveTex, atlas._gpu!._bandTex, atlas._gpu!._metaBuf].some((handle, index) => handle !== handles[index])).toBe(true);
            draws.forEach((draw, index) => expect(draw._version).toBeGreaterThan(versions[index]!));
            await device.queue.onSubmittedWorkDone();
            disposeComputeTextData(first);
            live.delete(first);
            record();
            expect(draws[0]!._active).toBe(false);
            expect(draws[1]!._active).toBe(true);
            await verifyResident(second, font, ["A"]);
        } finally {
            await device.queue.onSubmittedWorkDone();
            draws.forEach((draw) => draw._dispose());
            uniforms.forEach((uniform) => uniform.destroy());
            target.destroy();
            pipeline._cache._quadVertexBuffer.destroy();
            clearTextPipelineCache(native.engine);
        }
    });
});
