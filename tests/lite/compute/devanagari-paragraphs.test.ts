/** CTF-03 03_01/03_02: monolithic and paged Devanagari paragraphs wrap at proven spaces and every line matches HarfBuzz line shaping. */
import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { createComputeTextData, disposeComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { getComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

type ShapedRecord = number[];
interface Row {
    readonly id: string;
    readonly text: string;
    readonly whole: ShapedRecord[];
    readonly lines?: Record<string, ShapedRecord[]>;
    readonly proven: number[];
}
const references = JSON.parse(readFileSync(new URL("./references/deva-paragraph-harfbuzz-11.0.1.json", import.meta.url), "utf8")) as { cases: Row[] };
const row = (id: string) => references.cases.find((c) => c.id === id)!;

/** Per-paragraph shaped records and `[first, count, widthBits, cluster]` lines read back from the resident stage. */
async function readParagraphs(data: ComputeTextData) {
    const stage = getComputeTextParagraphStage(data)!;
    const words = stage._words;
    const capacity = words[1]!;
    const stride = 4 + capacity * 6;
    const shaped = new Int32Array(await readStorageBuffer(data._shaped));
    const info = new Uint32Array(await readStorageBuffer(stage._info!));
    const lines = new Uint32Array(await readStorageBuffer(stage._lines!));
    return Array.from({ length: words[0]! }, (_unused, p) => {
        const slot = words[16 + p]!;
        const count = shaped[slot * stride + 1]!;
        const records = Array.from({ length: count }, (_r, i) => [...shaped.slice(slot * stride + 4 + i * 6, slot * stride + 10 + i * 6)]);
        const result = info.subarray(slot * 4, slot * 4 + 4);
        expect(shaped[slot * stride], `paragraph ${p} shape status`).toBe(0);
        expect(result[1], `paragraph ${p} layout status`).toBe(0);
        return {
            records,
            lines: Array.from({ length: result[2]! }, (_l, l) => [...lines.slice((slot * capacity + l) * 4, (slot * capacity + l + 1) * 4)]),
        };
    });
}

/** The monolithic paragraph equals HarfBuzz whole shaping, wraps, and every line equals the proven HarfBuzz line it covers. */
async function expectReferenceLines(data: ComputeTextData, c: Row) {
    const id = c.id;
    const [paragraph] = await readParagraphs(data);
    const { records, lines } = paragraph!;
    expect(records, `${id}: whole paragraph`).toEqual(c.whole);
    expect(lines.length, `${id}: wraps`).toBeGreaterThan(1);
    const scalars = [...c.text];
    let next = 0;
    lines.forEach(([first, count, , cluster], l) => {
        expect(first, `${id}: line ${l} starts where the previous line's glyphs end`).toBe(next);
        const end = l + 1 < lines.length ? records[lines[l + 1]![0]!]![1]! : scalars.length;
        expect(scalars[cluster! - 1] ?? " ", `${id}: line ${l} starts after a space`).toBe(" ");
        const reference = c.lines?.[`${cluster}:${end}`];
        expect(reference, `${id}: line ${cluster}:${end} is a proven HarfBuzz line`).toBeDefined();
        const actual = records.slice(first!, first! + count!).map(([glyph, at, ...rest]) => [glyph, at! - cluster!, ...rest]);
        expect(actual, `${id}: line ${l} glyphs`).toEqual(reference!.slice(0, count));
        // Glyphs after the drawn count are the line's trailing break space(s), which a line never draws.
        for (const trailing of reference!.slice(count)) {
            expect(scalars[cluster! + trailing[1]!], `${id}: line ${l} trailing record`).toBe(" ");
        }
        next = first! + reference!.length;
    });
    expect(next, `${id}: lines cover every glyph`).toBe(records.length);
}

/** Paged records with paragraph-level clusters, and the drawn instance words of every chunk. */
async function readPaged(data: ComputeTextData) {
    const records: number[][] = [];
    const instances: number[] = [];
    for (const chunk of data._chunkState!.chunks.flat()) {
        const shaped = new Int32Array(await readStorageBuffer(chunk.shaped.buffer, chunk.shaped.offset, chunk.shaped.size));
        expect(shaped[0], "chunk shape status").toBe(0);
        for (let i = 0; i < shaped[1]!; i++) {
            const record = [...shaped.slice(4 + i * 6, 10 + i * 6)];
            record[1]! += chunk.source.scalarStart;
            records.push(record);
        }
        const indirect = new Uint32Array(await readStorageBuffer(chunk.indirect.buffer, chunk.indirect.offset, 16));
        if (indirect[1]) {
            instances.push(...new Uint32Array(await readStorageBuffer(chunk.instances.buffer, chunk.instances.offset, indirect[1] * 12)));
        }
    }
    return { records, instances };
}

async function readMonolithicInstances(data: ComputeTextData) {
    const indirect = new Uint32Array(await readStorageBuffer(data._indirect));
    return indirect[1] ? [...new Uint32Array(await readStorageBuffer(data._instances, 0, indirect[1] * 12))] : [];
}

describe("native Devanagari paragraph wrapping", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    const live = new Set<ComputeTextData>();
    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
    });
    afterEach(async () => {
        for (const data of live) {
            disposeComputeTextData(data);
        }
        live.clear();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        disposeGlyphStorage(storage);
        await native?.close();
    });

    it.each([
        { id: "plain-country", maxWidthPx: 160 },
        { id: "reph", maxWidthPx: 190 },
        { id: "conjunct-sta", maxWidthPx: 120 },
    ])("wraps $id at $maxWidthPx px into lines that each match HarfBuzz line shaping", async ({ id, maxWidthPx }) => {
        const c = row(id);
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        const data = createComputeTextData(native.engine, face.font, storage, `deva-paragraphs-${id}`, { fontSizePx: 32 });
        data._chunkBudgets = { monolithic: true };
        live.add(data);
        await (
            await updateComputeTextParagraphs(data, [c.text], { maxWidthPx })
        ).completed;
        await expectReferenceLines(data, c);
    });

    function block(budgets: ComputeTextData["_chunkBudgets"]) {
        const data = createComputeTextData(native.engine, fixture("NotoSansDevanagari-Regular.ttf").font, storage, "deva-paragraphs", { fontSizePx: 32 });
        data._chunkBudgets = budgets;
        live.add(data);
        return data;
    }

    /** Every chunk seam follows a HarfBuzz-proven independent space. */
    function expectProvenSeams(data: ComputeTextData, c: Row) {
        const proven = new Set(c.proven);
        const chunks = data._chunkState!.chunks[0]!;
        for (const chunk of chunks.slice(1)) {
            expect(proven.has(chunk.source.scalarStart - 1), `${c.id}: seam at scalar ${chunk.source.scalarStart}`).toBe(true);
        }
        return chunks.length;
    }

    it.each([300, 160])("pages multiseam-forced at %i px identically to monolithic layout and HarfBuzz lines", async (maxWidthPx) => {
        const c = row("multiseam-forced");
        const monolithic = block({ monolithic: true });
        const paged = block({ force: true, records: 64 });
        await (
            await updateComputeTextParagraphs(monolithic, [c.text], { maxWidthPx })
        ).completed;
        await (
            await updateComputeTextParagraphs(paged, [c.text], { maxWidthPx })
        ).completed;
        await expectReferenceLines(monolithic, c);
        const { records, instances } = await readPaged(paged);
        expect(records).toEqual(c.whole);
        expect(instances).toEqual(await readMonolithicInstances(monolithic));
        expect(expectProvenSeams(paged, c)).toBeGreaterThan(2);
    });

    it.each([
        // S3: words of more than 15 syllables inside one forced chunk.
        ["long-syllable-words", Array.from({ length: 12 }, (_, i) => (i % 3 ? "नमस्ते दुनिया" : "कमलनयनजलधरगगनपवनसरल")).join(" ")],
        // R4: ◌ insertion and nukta decomposition expand records beyond the scalar count.
        ["dotted-circle-nukta", Array.from({ length: 60 }, (_, i) => ["क़ऩ", "ि", "ाॉ", "ऩक़ि", "ऱ्य"][i % 5]).join(" ")],
    ])("pages %s identically to monolithic layout", async (_id, text) => {
        const monolithic = block({ monolithic: true });
        const paged = block({ force: true, records: 64 });
        for (const data of [monolithic, paged]) {
            await (
                await updateComputeTextParagraphs(data, [text], { maxWidthPx: 240 })
            ).completed;
        }
        const [reference] = await readParagraphs(monolithic);
        const { records, instances } = await readPaged(paged);
        expect(paged._chunkState!.chunks[0]!.length).toBeGreaterThan(1);
        expect(records).toEqual(reference!.records);
        expect(instances).toEqual(await readMonolithicInstances(monolithic));
        expect(reference!.lines.length).toBeGreaterThan(1);
    });

    it("pages multiseam-default under default chunk budgets and matches HarfBuzz whole-paragraph shaping", async () => {
        const c = row("multiseam-default");
        // Single paragraphs page unforced only past the binding limit; `force` keeps the default 16384-record chunk target.
        const data = block({ force: true });
        await (
            await updateComputeTextParagraphs(data, [c.text], { maxWidthPx: 600 })
        ).completed;
        const { records, instances } = await readPaged(data);
        expect(records).toEqual(c.whole);
        expect(instances.length).toBeGreaterThan(0);
        expect(expectProvenSeams(data, c)).toBeGreaterThan(2);
    });
});
