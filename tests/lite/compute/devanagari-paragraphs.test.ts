/** CTF-03: native Devanagari paragraph layout against the HarfBuzz paragraph references and the serial WGSL layout oracle. */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import type { ComputeTextParagraphLayout } from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { createComputeTextParagraphStage, getComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { createComputeTextPlan, ensureComputeTextProfile, profileSupport } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { computeTextParagraphFlags } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-flags";
import { oracleParagraphFlags, paragraphReferences, referenceBoundaries, referenceLine, referenceRow } from "./devanagari-paragraph-references";
import type { ParagraphReferenceRow, ShapedRecord } from "./devanagari-paragraph-references";
import { serialParagraphLayout } from "./serial-paragraph-layout";
import { fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

type Row = ParagraphReferenceRow;
const row = referenceRow;
const FONT = "NotoSansDevanagari-Regular.ttf";
/** Every Devanagari row Lite admits; `vedic-crossing` is reference-only, `multiseam-default` is covered by its own paged test. */
const corpusRows = paragraphReferences.cases.filter((c) => !c.tags.includes("reference-only") && !c.tags.includes("mixed") && c.id !== "multiseam-default");
const WIDTHS = [24, 60, 100, 140, 180, 220, 280, 360, 480, 640, 1000, Infinity];
const ALIGNS = ["left", "center", "right"] as const;

interface Paragraph {
    readonly records: ShapedRecord[];
    /** `[firstRecord, drawnCount, widthBits, startCluster]`. */
    readonly lines: number[][];
    /** The paragraph's whole flag stride; bytes past its scalars are zero. */
    readonly flags: Uint8Array;
}

/** Per display paragraph shaped records, `[first, count, widthBits, cluster]` lines and the uploaded flag bytes. */
async function readParagraphs(data: ComputeTextData): Promise<Paragraph[]> {
    const stage = getComputeTextParagraphStage(data)!;
    const words = stage._words;
    const capacity = words[1]!;
    const stride = 4 + capacity * 6;
    const shaped = new Int32Array(await readStorageBuffer(data._shaped));
    const info = new Uint32Array(await readStorageBuffer(stage._info!));
    const lines = new Uint32Array(await readStorageBuffer(stage._lines!));
    const flagBytes = new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
    return Array.from({ length: words[0]! }, (_unused, p) => {
        const slot = words[16 + p]!;
        const count = shaped[slot * stride + 1]!;
        const records = Array.from({ length: count }, (_r, i) => [...shaped.slice(slot * stride + 4 + i * 6, slot * stride + 10 + i * 6)]);
        const result = info.subarray(slot * 4, slot * 4 + 4);
        expect(shaped[slot * stride], `paragraph ${p} shape status`).toBe(0);
        expect(result[1], `paragraph ${p} layout status`).toBe(0);
        const flagStart = (16 + words[11]! + slot * words[12]!) * 4;
        return {
            records,
            lines: Array.from({ length: result[2]! }, (_l, l) => [...lines.slice((slot * capacity + l) * 4, (slot * capacity + l + 1) * 4)]),
            flags: flagBytes.slice(flagStart, flagStart + words[12]! * 4),
        };
    });
}

/**
 * CORRECT-1/2/4 for one laid-out paragraph: whole records equal HarfBuzz; each line covers exactly the records of its scalar range;
 * space-break lines equal HarfBuzz shaping of the line's own text; emergency lines start at a grapheme start and keep whole-paragraph records.
 */
function expectLines(c: Row, paragraph: Paragraph, label: string) {
    const { records, lines } = paragraph;
    expect(records, `${label}: whole paragraph`).toEqual(c.whole);
    const scalars = [...c.text];
    const boundaries = new Set(referenceBoundaries(c));
    const graphemeStarts = oracleParagraphFlags(c.text, c.proven);
    let next = 0;
    lines.forEach(([first, count, , cluster], l) => {
        const at = `${label}: line ${l}`;
        expect(first, `${at} starts where the previous line's records end`).toBe(next);
        const end = l + 1 < lines.length ? lines[l + 1]![3]! : scalars.length;
        const recordEnd = l + 1 < lines.length ? lines[l + 1]![0]! : records.length;
        for (let i = first!; i < recordEnd; i++) {
            expect(records[i]![1]! >= cluster! && records[i]![1]! < end, `${at} record ${i} cluster lies in [${cluster}, ${end})`).toBe(true);
        }
        expect(graphemeStarts[cluster!], `${at} starts at a grapheme start`).not.toBe(4);
        if (boundaries.has(cluster!) && boundaries.has(end)) {
            const reference = referenceLine(c, cluster!, end);
            expect(reference, `${at} ${cluster}:${end} is a HarfBuzz line`).toBeDefined();
            expect(recordEnd - first!, `${at} record count`).toBe(reference!.length);
            const actual = records.slice(first!, first! + count!).map(([glyph, scalar, ...rest]) => [glyph, scalar! - cluster!, ...rest]);
            expect(actual, `${at} glyphs`).toEqual(reference!.slice(0, count));
        }
        // Records after the drawn count are the line's hanging trailing spaces.
        for (const trailing of records.slice(first! + count!, recordEnd)) {
            expect(scalars[trailing[1]!], `${at} trailing record`).toBe(" ");
        }
        next = recordEnd;
    });
    expect(next, `${label}: lines cover every record`).toBe(records.length);
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
        vi.restoreAllMocks();
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

    function block(budgets: ComputeTextData["_chunkBudgets"]) {
        const data = createComputeTextData(native.engine, fixture(FONT).font, storage, "deva-paragraphs", { fontSizePx: 32 });
        data._chunkBudgets = budgets;
        live.add(data);
        return data;
    }

    async function layout(data: ComputeTextData, texts: readonly string[], options: ComputeTextParagraphLayout) {
        await (
            await updateComputeTextParagraphs(data, texts, options)
        ).completed;
    }

    /**
     * CORRECT-3: run the serial WGSL measure/place oracle on the reference records and oracle flags with the monolithic stage's parameters,
     * then compare paragraph info, every line, the indirect words and the placed instances with the production pipeline bit-for-bit.
     */
    async function expectSerialOracle(data: ComputeTextData, rows: readonly Row[], flags: readonly Uint8Array[]) {
        const stage = getComputeTextParagraphStage(data)!;
        const words = stage._words.slice();
        const [count, capacity] = [words[0]!, words[1]!];
        const stride = 4 + capacity * 6;
        const shaped = new Int32Array(await readStorageBuffer(data._shaped));
        const flagBytes = new Uint8Array(words.buffer);
        words[8] = 0;
        words[9] = count;
        words[10] = 1;
        rows.forEach((c, p) => {
            const slot = words[16 + p]!;
            shaped.fill(0, slot * stride, (slot + 1) * stride);
            shaped[slot * stride + 1] = c.whole.length;
            shaped.set(c.whole.flat(), slot * stride + 4);
            const flagStart = (16 + words[11]! + slot * words[12]!) * 4;
            flagBytes.fill(0, flagStart, flagStart + words[12]! * 4);
            flagBytes.set(flags[p]!, flagStart);
        });
        const total = rows.reduce((sum, c) => sum + c.whole.length, 0);
        const buffers = {
            shaped: createStorageBuffer(native.engine, shaped, { writable: true }),
            instances: createStorageBuffer(native.engine, Math.max(1, total) * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
        const oracle = createComputeTextParagraphStage({}, native.engine, words);
        const task = createComputeTask(native.engine, "deva-serial-oracle");
        const shaders: ReturnType<typeof createComputeShader>[] = [];
        try {
            oracle.prepare(native.engine, 0, capacity);
            const dispatches = oracle.dispatches({ ...buffers, slots: data._slots });
            for (const [i, entryPoint] of [
                [0, "serialMeasureTest"],
                [7, "serialPlaceTest"],
            ] as const) {
                const shader = createComputeShader(native.engine, {
                    computeSource: oracle._shaders![i]!._source + serialParagraphLayout,
                    entryPoint,
                    bindings: oracle._shaders![i]!._decls,
                });
                shaders.push(shader);
                dispatches[i] = createComputeDispatch(
                    shader,
                    createComputeBindingSet(shader, { input: oracle._input!, ...buffers, slots: data._slots, info: oracle._info!, lines: oracle._lines! }),
                    { size: { x: Math.ceil(count / 64) } }
                );
            }
            // The serial oracle measures every paragraph itself; block measuring would overwrite its results.
            dispatches.filter((_, i) => i !== 1 && i !== 2).forEach((dispatch) => addComputeDispatch(task, dispatch));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const [actualIndirect, expectedIndirect] = await Promise.all([readStorageBuffer(data._indirect), readStorageBuffer(buffers.indirect)]);
            expect(new Uint32Array(actualIndirect), "indirect words").toEqual(new Uint32Array(expectedIndirect));
            const instanceBytes = new Uint32Array(expectedIndirect)[1]! * 12;
            if (instanceBytes) {
                expect(new Uint8Array(await readStorageBuffer(data._instances, 0, instanceBytes)), "placed instances").toEqual(
                    new Uint8Array(await readStorageBuffer(buffers.instances, 0, instanceBytes))
                );
            }
            const [actualInfo, expectedInfo] = await Promise.all([readStorageBuffer(stage._info!), readStorageBuffer(oracle._info!)]);
            const [actualLines, expectedLines] = await Promise.all([readStorageBuffer(stage._lines!), readStorageBuffer(oracle._lines!)]);
            for (let p = 0; p < count; p++) {
                const slot = words[16 + p]!;
                const info = new Uint32Array(expectedInfo, slot * 16, 4);
                expect(new Uint32Array(actualInfo, slot * 16, 4), `${rows[p]!.id}: paragraph info`).toEqual(info);
                expect(new Uint32Array(actualLines, slot * capacity * 16, info[2]! * 4), `${rows[p]!.id}: lines`).toEqual(
                    new Uint32Array(expectedLines, slot * capacity * 16, info[2]! * 4)
                );
            }
        } finally {
            task.dispose();
            oracle.release();
            shaders.forEach(disposeComputeShader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    }

    it("matches HarfBuzz lines and the serial WGSL oracle for every corpus row at 12 widths under start, center and end alignment", async () => {
        const data = block({ monolithic: true });
        const flags = corpusRows.map((c) => oracleParagraphFlags(c.text, c.proven));
        let wrapped = 0;
        let emergency = 0;
        for (const maxWidthPx of WIDTHS) {
            for (const align of maxWidthPx === Infinity ? (["left"] as const) : ALIGNS) {
                await layout(
                    data,
                    corpusRows.map((c) => c.text),
                    { maxWidthPx, align }
                );
                const paragraphs = await readParagraphs(data);
                paragraphs.forEach((paragraph, p) => {
                    const c = corpusRows[p]!;
                    expect(paragraph.flags.subarray(0, flags[p]!.length), `${c.id}: flags equal the reference-derived oracle flags`).toEqual(flags[p]);
                    expectLines(c, paragraph, `${c.id} @${maxWidthPx}/${align}`);
                    wrapped += paragraph.lines.length > 1 ? 1 : 0;
                    emergency += paragraph.lines.filter(([, , , cluster]) => !referenceBoundaries(c).includes(cluster!)).length;
                });
                await expectSerialOracle(data, corpusRows, flags);
            }
        }
        expect(wrapped).toBeGreaterThan(corpusRows.length * 10);
        expect(emergency).toBeGreaterThan(0);
    });

    it.each([
        { id: "prebase-edge", maxWidthPx: 60 },
        { id: "reph", maxWidthPx: 120 },
        { id: "conjunct-sta", maxWidthPx: 120 },
    ])("keeps reordered pre-base matra and reph syllables whole at $id line edges under every alignment", async ({ id, maxWidthPx }) => {
        const c = row(id);
        const data = block({ monolithic: true });
        const flags = [oracleParagraphFlags(c.text, c.proven)];
        for (const align of ALIGNS) {
            await layout(data, [c.text], { maxWidthPx, align });
            const [paragraph] = await readParagraphs(data);
            expect(paragraph!.lines.length).toBeGreaterThan(2);
            expectLines(c, paragraph!, `${id}/${align}`);
            await expectSerialOracle(data, [c], flags);
        }
    });

    it("produces single-run records for every one-line paragraph (CORRECT-5 resolution parity)", async () => {
        const paragraphs = block({ monolithic: true });
        await layout(
            paragraphs,
            corpusRows.map((c) => c.text),
            { maxWidthPx: Infinity }
        );
        const read = await readParagraphs(paragraphs);
        const single = block(undefined);
        for (const [p, c] of corpusRows.entries()) {
            if (!c.text.trim()) {
                continue;
            }
            await (
                await updateComputeTextData(single, [c.text])
            ).completed;
            const words = new Int32Array(await readStorageBuffer(single._shaped));
            expect(words[0], `${c.id}: single-run status`).toBe(0);
            expect(read[p]!.lines, `${c.id}: one line`).toHaveLength(1);
            expect(read[p]!.records.flat(), `${c.id}: paragraph records equal single-run records`).toEqual([...words.slice(4, 4 + words[1]! * 6)]);
        }
    });

    describe("paged and incremental paths (CORRECT-5, PATH)", () => {
        /** Every chunk seam follows a HarfBuzz-proven independent space. */
        function expectProvenSeams(data: ComputeTextData, c: Row) {
            const proven = new Set(c.proven);
            const chunks = data._chunkState!.chunks[0]!;
            for (const chunk of chunks.slice(1)) {
                expect(proven.has(chunk.source.scalarStart - 1), `${c.id}: seam at scalar ${chunk.source.scalarStart}`).toBe(true);
            }
            return chunks.length;
        }

        it.each([
            ["multiseam-forced", 300, [24, 64, 200]],
            ["multiseam-forced", 160, [40, 100]],
            ["plain-rain", 200, [24, 48]],
            ["reph", 150, [24, 32]],
            ["overlong-boundary", 140, [80, 140]],
        ] as const)("pages %s at %i px at every seam placement identically to monolithic layout and HarfBuzz lines", async (id, maxWidthPx, budgets) => {
            const c = row(id);
            const monolithic = block({ monolithic: true });
            await layout(monolithic, [c.text], { maxWidthPx });
            const [paragraph] = await readParagraphs(monolithic);
            expect(paragraph!.lines.length).toBeGreaterThan(1);
            expectLines(c, paragraph!, id);
            const expected = await readMonolithicInstances(monolithic);
            for (const records of budgets) {
                const paged = block({ force: true, records });
                await layout(paged, [c.text], { maxWidthPx });
                const actual = await readPaged(paged);
                expect(actual.records, `${id}/${records}: records`).toEqual(c.whole);
                expect(actual.instances, `${id}/${records}: instances`).toEqual(expected);
                expect(expectProvenSeams(paged, c), `${id}/${records}: chunks`).toBeGreaterThan(1);
            }
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
                await layout(data, [text], { maxWidthPx: 240 });
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
            await layout(data, [c.text], { maxWidthPx: 600 });
            const { records, instances } = await readPaged(data);
            expect(records).toEqual(c.whole);
            expect(instances.length).toBeGreaterThan(0);
            expect(expectProvenSeams(data, c)).toBeGreaterThan(2);
        });

        it.each([{ monolithic: true }, { force: true, records: 48 }])("edits words incrementally to the fresh layout of the edited text (budgets %o)", async (budgets) => {
            const c = row("plain-rain");
            const scalars = [...c.text];
            const at = c.proven[9]! + 1;
            const edits = [[...scalars.slice(0, at), ..."नया ", ...scalars.slice(at)].join(""), [...scalars.slice(0, at), ...scalars.slice(c.proven[10]! + 1)].join(""), c.text];
            const edited = block(budgets);
            await layout(edited, [c.text], { maxWidthPx: 220, align: "center" });
            for (const text of edits) {
                await layout(edited, [text], { maxWidthPx: 220, align: "center" });
                const fresh = block(budgets);
                await layout(fresh, [text], { maxWidthPx: 220, align: "center" });
                if (budgets.monolithic) {
                    const [actual] = await readParagraphs(edited);
                    const [expected] = await readParagraphs(fresh);
                    const n = [...text].length;
                    expect({ ...actual, flags: actual!.flags.subarray(0, n) }, `edit to ${n} scalars`).toEqual({ ...expected, flags: expected!.flags.subarray(0, n) });
                    expect(await readMonolithicInstances(edited)).toEqual(await readMonolithicInstances(fresh));
                } else {
                    expect(await readPaged(edited)).toEqual(await readPaged(fresh));
                }
                disposeComputeTextData(fresh);
                live.delete(fresh);
            }
            if (budgets.monolithic) {
                expectLines(c, (await readParagraphs(edited))[0]!, "restored");
            }
        });
    });

    it("lays out Devanagari and Common-only paragraphs in one call; the Common paragraph keeps its existing flag rules (MIX-2)", async () => {
        const rows = [row("plain-country"), row("mixed-common"), row("reph")];
        const face = fixture(FONT);
        const data = block({ monolithic: true });
        const common = row("mixed-common");
        const selection = selectComputeTextRun(common.text);
        const program = compileComputeTextProgram(createComputeTextPlan(face.font, selection));
        const commonFlags = computeTextParagraphFlags(
            selection._scalars,
            selection._scalars.map((cp) => face.oracle.glyphId(cp)),
            analyzeComputeTextSegmentation(program)
        );
        const flags = rows.map((c) => (c === common ? commonFlags : oracleParagraphFlags(c.text, c.proven)));
        for (const maxWidthPx of [60, 160]) {
            await layout(
                data,
                rows.map((c) => c.text),
                { maxWidthPx }
            );
            const paragraphs = await readParagraphs(data);
            paragraphs.forEach((paragraph, p) => {
                expect(paragraph.flags.subarray(0, flags[p]!.length), `${rows[p]!.id}: flags`).toEqual(flags[p]);
                expectLines(rows[p]!, paragraph, `${rows[p]!.id} @${maxWidthPx}`);
            });
            expect(paragraphs[1]!.lines.length).toBeGreaterThan(1);
            await expectSerialOracle(data, rows, flags);
        }
    });

    it("lazy-loads the indic profile for a first paragraph update in a fresh module registry (API-2)", async () => {
        const c = row("plain-country");
        const reference = block({ monolithic: true });
        await layout(reference, [c.text], { maxWidthPx: 160 });
        const expected = await readParagraphs(reference);
        const expectedInstances = await readMonolithicInstances(reference);
        vi.resetModules();
        const fresh = {
            plan: await import("../../../packages/babylon-lite/src/text/compute-text-plan"),
            data: await import("../../../packages/babylon-lite/src/text/compute-text-data"),
            paragraphs: await import("../../../packages/babylon-lite/src/text/compute-text-paragraphs"),
            storage: await import("../../../packages/babylon-lite/src/text/glyph-storage"),
            stage: await import("../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout"),
        };
        expect(fresh.plan.profileSupport("indic")).toBeUndefined();
        const freshStorage = fresh.storage.createGlyphStorage();
        const data = fresh.data.createComputeTextData(native.engine, fixture(FONT).font, freshStorage, "deva-paragraphs", { fontSizePx: 32 });
        data._chunkBudgets = { monolithic: true };
        try {
            await (
                await fresh.paragraphs.updateComputeTextParagraphs(data, [c.text], { maxWidthPx: 160 })
            ).completed;
            expect(fresh.plan.profileSupport("indic")?.paragraph).toBeDefined();
            const stage = fresh.stage.getComputeTextParagraphStage(data)!;
            const [capacity, slot] = [stage._words[1]!, stage._words[16]!];
            const stride = 4 + capacity * 6;
            const shaped = new Int32Array(await readStorageBuffer(data._shaped));
            const info = new Uint32Array(await readStorageBuffer(stage._info!));
            const lines = new Uint32Array(await readStorageBuffer(stage._lines!));
            const records = Array.from({ length: shaped[slot * stride + 1]! }, (_r, i) => [...shaped.slice(slot * stride + 4 + i * 6, slot * stride + 10 + i * 6)]);
            expect(records).toEqual(expected[0]!.records);
            const lineWords = (l: number) => [...lines.slice((slot * capacity + l) * 4, (slot * capacity + l + 1) * 4)];
            expect(Array.from({ length: info[slot * 4 + 2]! }, (_l, l) => lineWords(l))).toEqual(expected[0]!.lines);
            const indirect = new Uint32Array(await readStorageBuffer(data._indirect));
            const instances = new Uint32Array(await readStorageBuffer(data._instances, 0, indirect[1]! * 12));
            // Positions match; atlas slot words belong to the fresh glyph storage.
            expect(instances.filter((_w, i) => i % 3 !== 2)).toEqual(Uint32Array.from(expectedInstances.filter((_w, i) => i % 3 !== 2)));
        } finally {
            fresh.data.disposeComputeTextData(data);
            fresh.storage.disposeGlyphStorage(freshStorage);
        }
    });

    describe("paragraph proof independence (BREAK-4)", () => {
        beforeAll(() => ensureComputeTextProfile("indic"));

        it("breaks at proven spaces although text-block segmentation analysis is null", async () => {
            const c = row("plain-country");
            const face = fixture(FONT);
            expect(analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face.font, selectComputeTextRun(c.text))))).toBeNull();
            const data = block({ monolithic: true });
            await layout(data, [c.text], { maxWidthPx: 160 });
            const [paragraph] = await readParagraphs(data);
            expect(paragraph!.lines.length).toBeGreaterThan(2);
            expectLines(c, paragraph!, c.id);
        });

        it("keeps an unprovable space as a non-breaking SPACE whose width equals the serial reference", async () => {
            const c = row("plain-country");
            const maxWidthPx = 160;
            const baseline = block({ monolithic: true });
            await layout(baseline, [c.text], { maxWidthPx });
            const [before] = await readParagraphs(baseline);
            const broken = before!.lines[1]![3]! - 1;
            expect(c.proven).toContain(broken);
            const paragraph = profileSupport("indic")!.paragraph!;
            const spaces = paragraph.spaces;
            vi.spyOn(paragraph, "spaces").mockImplementation((program) => {
                const proof = spaces(program);
                return proof && { safeAfter: (glyphIds, space, scalars) => space !== broken && proof.safeAfter(glyphIds, space, scalars) };
            });
            const data = block({ monolithic: true });
            await layout(data, [c.text], { maxWidthPx });
            const [after] = await readParagraphs(data);
            const proven = c.proven.filter((s) => s !== broken);
            expect(after!.flags[broken]).toBe(1);
            expect(after!.lines.map(([, , , cluster]) => cluster)).not.toContain(broken + 1);
            expectLines({ ...c, proven }, after!, `${c.id} without ${broken}`);
            await expectSerialOracle(data, [{ ...c, proven }], [oracleParagraphFlags(c.text, proven)]);
        });
    });
});
