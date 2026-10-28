import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { shapeComputeText } from "../../../packages/babylon-lite/src/index";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextShapeShader, kernelSource, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { fixture, expectedFields, FONT_HASHES } from "./fixtures";
import type { Fixture } from "./fixtures";
import { syntheticFont } from "./synthetic";
import type { SyntheticKind } from "./synthetic";

interface Witness {
    id: string;
    kind: SyntheticKind;
    text: string;
    expected: string;
    fontSha256: string;
    records: number[][];
}

function reviewed(id: string): Witness {
    const rows = (JSON.parse(readFileSync("tests/lite/compute/references/synthetic-harfbuzz-11.0.1.json", "utf8")) as { cases: Witness[] }).cases;
    const row = rows.find((entry) => entry.id === id);
    expect(row, id).toBeDefined();
    expect(row!.expected).toBe("positive");
    expect(row!.records).toBeDefined();
    return row!;
}

// Qualified independently before inclusion: pinned HarfBuzz 11.0.1 CLI/ctypes
// agreement and actual hmtx + MarkBase/MarkMark anchors, under README R3.
// Font hash and nominal IDs are checked before these records are used.
const N2_REVIEWED_RECORDS = [
    [1453, 0, 344, 0, 0, 0],
    [1691, 1, 0, 0, 180, 229],
    [1690, 2, 0, 0, 271, 458],
] as const;

describe("cached full-kernel state split", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    it("records the real contextual input positions", async () => {
        const row = reviewed("c5-context-format3-0");
        const font = syntheticFont(row.kind);
        expect(createHash("sha256").update(new Uint8Array(font.bytes)).digest("hex")).toBe(row.fontSha256);
        expect(row.text).toBe("AB");
        const p = prepare(font, row.text);
        expect(p.program.plan._profile).toBe("default");
        const nominal = [...row.text].map((cp) => mapComputeFontScalar(font.font, cp.codePointAt(0)!));
        expect(nominal).toEqual([1, 2]);
        const root = p.program.lookups[p.program.rootOrder[0]!]!;
        expect(root).toMatchObject({ table: "GSUB", lookupType: 5 });
        expect(
            root.rules.some(
                (rule) =>
                    rule.mode === 2 &&
                    rule.action.kind === "context" &&
                    rule.input.length === nominal.length &&
                    rule.input.every((predicate, index) => predicate.kind === "glyphs" && predicate.ids.includes(nominal[index]!)) &&
                    rule.backtrack.length === 0 &&
                    rule.lookahead.length === 0
            )
        ).toBe(true);
        expect(presence(p, font, nominal) & 1).not.toBe(0);
        // The nominal contextual predicate matches, so the simple proof declines;
        // the general applySelected records both positions for its nested Single.
        await check(font, row.text, row.records);
    });

    it.each(["c8-mark-ligature-0", "c8-mark-ligature-1"])("tags skipped/trailing marks and reads ligature components: %s", async (id) => {
        const row = reviewed(id);
        const font = syntheticFont(row.kind);
        expect(createHash("sha256").update(new Uint8Array(font.bytes)).digest("hex")).toBe(row.fontSha256);
        // Reuse the existing shared-core internal fixture route. Public PUA
        // admission is deliberately NOT widened for these primitive witnesses.
        expect(() => selectComputeTextRun(row.text)).toThrow(/unknown script/);
        await expect(shapeComputeText(native.engine, font.font, row.text)).rejects.toThrow(/unknown script/);
        const p = prepare(font, row.text, true);
        expect(p.program.plan._profile).toBe("default");
        const nominal = [...row.text].map((cp) => mapComputeFontScalar(font.font, cp.codePointAt(0)!));
        expect(nominal).toEqual(id.endsWith("-0") ? [1, 4, 2, 5] : [1, 2, 4, 5]);
        expect(p.program.lookups[p.program.rootOrder[0]!]!).toMatchObject({ table: "GSUB", lookupType: 4, lookupFlag: 8 });
        expect(
            p.program.rootOrder.some((root) => {
                const lookup = p.program.lookups[root]!;
                return lookup.table === "GPOS" && lookup.lookupType === 5 && lookup.rules.some((rule) => rule.action.kind === "mark-ligature");
            })
        ).toBe(true);
        await check(font, row.text, row.records, true);
    });

    it.each([2, 64])("rebuilds changed root presence through the original %i-glyph refresh path", async (length) => {
        const font = syntheticFont("refresh-roots");
        const text = "A".repeat(length);
        const p = prepare(font, text);
        const words = p.programWords;
        const encoded = words[words[8]!]!;
        const descriptor = words[4]! + (encoded & 0xffff) * 8;
        const rule = words[descriptor + 3]!;
        const input = words[rule]!;
        const action = words[rule + 6]!;
        expect(encoded >>> 16).toBe(0); // GSUB
        expect(words[descriptor + 2]).toBe(1); // cooperative Single
        expect(words[rule + 8]).toBe(1);
        expect([...words.slice(input, input + 2)]).toEqual([1, 1]);
        expect([...words.slice(action, action + 2)]).toEqual([1, 2]); // ID 1 -> 2
        expect(presence(p, font, [1]) & 2).toBe(0);
        expect(presence(p, font, [words[action + 1]!]) & 2).not.toBe(0);
        expect(p.runs[p.runs[0]!]! & 0x7fffffff).toBe(length); // bit 31 is the mark-free flag
        for (const tier of [0, p.capacity]) {
            const source = kernelSource(tier);
            expect(source).toContain("if (mode == 1u) { replaceGlyph(glyphAt(base, index), program[program[ruleAt + 6u] + 1u]); }");
            expect(source).toContain("if (scratch[at] != i32(id)) {");
            expect(source).toContain("atomicStore(&idsChanged, 1u);");
            expect(source).toContain("| select(0u, 2u, !isGpos && atomicLoad(&idsChanged) != 0u)");
            expect(source).toContain("let parallelRefresh = count > 32u;");
        }
        // The actual packed edit changes IDs, setting flags&2 and enabling the
        // previously absent root 1. Single keeps length: 2 is serial, 64 parallel.
        await check(font, text);
    });

    it("initializes a higher presence word before any edit or refresh through the real default main", async () => {
        // This existing font also has Latin coverage. Its LATN plan is DEFAULT,
        // not masked Arabic: 37 roots, including GPOS 21 at root 35 (word 1).
        // f has no acute composition; both nominal marks enter the mark-mark
        // lookup without depending on a GSUB edit or a refresh to create them.
        const font = fixture("NotoSansArabic-Regular.ttf");
        const text = "f\u0301\u0300";
        expect(createHash("sha256").update(new Uint8Array(font.bytes)).digest("hex")).toBe(FONT_HASHES["NotoSansArabic-Regular.ttf"]);
        expect([...text].map((cp) => mapComputeFontScalar(font.font, cp.codePointAt(0)!))).toEqual([1453, 1691, 1690]);
        const p = prepare(font, text);
        expect(p.program.plan._profile).toBe("default");
        expect(p.program.rootOrder).toHaveLength(37);
        expect(p.program.words[13]).toBe(2);
        const root = 35;
        const lookup = p.program.lookups[p.program.rootOrder[root]!]!;
        expect(lookup).toMatchObject({ table: "GPOS", lookupIndex: 21, lookupType: 6 });
        const acute = mapComputeFontScalar(font.font, 0x301);
        const grave = mapComputeFontScalar(font.font, 0x300);
        expect(
            lookup.rules.some((rule) => rule.input[0]?.ids.includes(grave) && rule.action.kind === "mark-mark" && rule.action.bases.some((base) => base.glyphId === acute))
        ).toBe(true);
        const glyphs = [...text].map((cp) => mapComputeFontScalar(font.font, cp.codePointAt(0)!));
        expect(presence(p, font, glyphs, 1) & (1 << (root % 32))).not.toBe(0);
        await check(font, text, N2_REVIEWED_RECORDS);
    });

    function prepare(font: Fixture, text: string, internalFixture = false) {
        const selected = selectComputeTextRun(internalFixture ? "ABC" : text);
        const scalars = internalFixture ? Uint32Array.from(text, (cp) => cp.codePointAt(0)!) : selected._scalars;
        return prepareComputeTextShapeWords(createComputeTextPlan(font.font, selected), [{ scalars, glyphIds: scalars.map((cp) => mapComputeFontScalar(font.font, cp)) }]);
    }

    function presence(p: ReturnType<typeof prepare>, font: Fixture, glyphs: readonly number[], word = 0): number {
        const masks = p.params[5]! + p.program.rootOrder.length * 4;
        const width = p.program.words[13]!;
        const bits = glyphs.reduce((bits, glyph) => bits | p.programWords[masks + glyph * width + word]!, 0);
        const witnesses = glyphs.reduce((bits, glyph) => bits | p.programWords[masks + font.font.glyphCount * width + glyph * width + word]!, 0);
        return bits & witnesses;
    }

    /** Same raw-buffer scheduling as parallel-refresh; no rewritten main, forced branch or shader seam. */
    async function check(font: Fixture, text: string, records: readonly (readonly number[])[] = expectedFields(font, text), internalFixture = false): Promise<void> {
        const expected = records.flat();
        const p = prepare(font, text, internalFixture);
        const cached = await raw(p, false);
        const cacheless = await raw(p, true);
        for (const actual of [cached, cacheless]) {
            expect(actual.slice(0, 4)).toEqual([0, records.length, 0, 0]);
            expect(actual.slice(4, 4 + expected.length)).toEqual(expected);
        }
        expect(cacheless.slice(0, 4 + expected.length)).toEqual(cached.slice(0, 4 + expected.length));
    }

    async function raw(p: ReturnType<typeof prepare>, cacheless: boolean): Promise<number[]> {
        const inputs = { params: p.params, program: p.programWords, tables: p.tables, runs: p.runs, scratch: p.scratchBytes, output: (4 + p.capacity * 6) * 4 };
        const source = kernelSource(0);
        expect(source).not.toMatch(/activeRootsProbe|matchInputProbe/);
        const shader = cacheless
            ? createComputeShader(native.engine, {
                  name: "state-split-cacheless",
                  computeSource: source,
                  bindings: Object.keys(inputs).map((name, binding) => computeStorageBufferBinding(name, { group: 0, binding, access: binding >= 4 ? "read-write" : "read" })),
              })
            : createComputeTextShapeShader(native.engine, p.capacity);
        if (!cacheless) {
            expect(shader._source).toContain("fn activeRootsLanes(");
        }
        const buffers = Object.fromEntries(
            Object.entries(inputs).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const task = createComputeTask(native.engine, "state-split-real-main");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            return [...new Int32Array(await readStorageBuffer(buffers.output!))];
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    }
});
