/** Checked program compilation. text-shaper is imported only by the witness tests. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { createComputeFontFromBuffer, readComputeFontAdvance } from "../../../packages/babylon-lite/src/text/compute-font";
import {
    COMPUTE_LOOKUP_FLAG,
    COMPUTE_LOOKUP_CANDIDATES,
    COMPUTE_LOOKUP_RULE_ORIGIN,
    COMPUTE_LOOKUP_MEMBERSHIP,
    COMPUTE_LOOKUP_ORDERED,
    COMPUTE_LOOKUP_KIND,
    COMPUTE_LOOKUP_RULE_COUNT,
    COMPUTE_PROGRAM_LOOKUP_WORDS,
    COMPUTE_PROGRAM_RULE_WORDS,
    COMPUTE_RULE_MODE,
    COMPUTE_RULE_MODE_REJECT,
    COMPUTE_RULE_MODE_SUBSTITUTE,
    compileComputeTextProgram,
} from "../../../packages/babylon-lite/src/text/compute-text-program";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { parseLookupActions } from "../../../packages/babylon-lite/src/text/font-parser/layout-actions";
import type { GlyphClassDef } from "../../../packages/babylon-lite/src/text/font-parser/opentype-layout";
import { coverageGlyphs, coverageIndex, glyphClass } from "../../../packages/babylon-lite/src/text/font-parser/opentype-layout";
import { Font, GlyphBuffer, UnicodeBuffer, shapeInto } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

const TEXT_SHAPER_VERSION = JSON.parse(readFileSync(join(process.cwd(), "packages/babylon-lite/node_modules/text-shaper/package.json"), "utf8")).version as string;

function fontBytes(file: string): ArrayBuffer {
    const buf = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

describe("compileComputeTextProgram", () => {
    it("packs a tight program buffer and shares mark-class base lists without changing per-rule anchors", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        expect(program.words.byteOffset).toBe(0);
        expect(program.words.buffer.byteLength).toBe(program.words.byteLength);
        const byClass = new Map<string, object>();
        let shared = 0;
        for (const lookup of program.lookups) {
            lookup.rules.forEach((rule) => {
                if (rule.action.kind !== "mark-base" && rule.action.kind !== "mark-mark") {
                    return;
                }
                const key = `${lookup.table}/${lookup.lookupIndex}/${rule.action.markClass}/${rule.action.kind}`;
                const previous = byClass.get(key);
                if (previous) {
                    expect(rule.action.bases).toBe(previous);
                    shared++;
                } else {
                    byClass.set(key, rule.action.bases);
                }
            });
        }
        expect(shared).toBeGreaterThan(100);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf", "NotoSansArabic-Regular.ttf"])("packs stable sorted mark anchors with exact signed positions: %s", (file) => {
        const font = createComputeFontFromBuffer(fontBytes(file));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun(file.startsWith("Noto") ? "ج" : "A")));
        let tested = 0;
        for (const lookup of program.lookups) {
            const descriptor = (lookup.table === "GSUB" ? program.gsubDescriptorOffset : program.gposDescriptorOffset) + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
            const rules = program.words[descriptor + COMPUTE_LOOKUP_RULE_ORIGIN]!;
            lookup.rules.forEach((rule, index) => {
                if (rule.action.kind !== "mark-base" && rule.action.kind !== "mark-mark") {
                    return;
                }
                const action = program.words[rules + index * COMPUTE_PROGRAM_RULE_WORDS + 6]!;
                const expected = rule.action.bases.slice().sort((left, right) => left.glyphId - right.glyphId);
                expect(program.words[action + 3]).toBe(expected.length);
                expect(Array.from({ length: expected.length }, (_, at) => [...program.words.slice(action + 4 + at * 3, action + 7 + at * 3)])).toEqual(
                    expected.map((base) => [base.glyphId, base.x >>> 0, base.y >>> 0])
                );
                tested++;
            });
        }
        expect(tested).toBeGreaterThan(0);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("indexes every first predicate in original order, including rejection and class zero: %s", (file) => {
        const font = createComputeFontFromBuffer(fontBytes(file));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        const dense = new Map<GlyphClassDef, Uint32Array>();
        for (const def of program.classTables) {
            dense.set(
                def,
                Uint32Array.from({ length: font.glyphCount }, (_, glyph) => glyphClass(def, glyph))
            );
        }
        for (const lookup of program.lookups) {
            const base = (lookup.table === "GSUB" ? program.gsubDescriptorOffset : program.gposDescriptorOffset) + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
            const directory = program.words[base + COMPUTE_LOOKUP_CANDIDATES]!;
            expect(directory, `${lookup.table}/${lookup.lookupIndex} directory`).toBeGreaterThan(0);
            const expected: number[][] = Array.from({ length: font.glyphCount }, () => []);
            lookup.rules.forEach((rule, index) => {
                const first = rule.input[0]!;
                if (first.kind === "glyphs") {
                    for (const glyph of first.ids) {
                        expected[glyph]?.push(index);
                    }
                } else {
                    for (let glyph = 0; glyph < font.glyphCount; glyph++) {
                        if (dense.get(first.classDef!)?.[glyph] === first.classId && (!first.coverage || coverageIndex(first.coverage, glyph) !== null)) {
                            expected[glyph]!.push(index);
                        }
                    }
                }
            });
            const actual = expected.map((_, glyph) => {
                const origin = program.words[directory + glyph * 2]!;
                const count = program.words[directory + glyph * 2 + 1]!;
                return [...program.words.slice(origin, origin + count)];
            });
            expect(actual, `${lookup.table}/${lookup.lookupIndex} ordered exhaustive candidates`).toEqual(expected);
        }
    });

    it("reuses immutable parsed actions without aliasing a different font snapshot or lookup", () => {
        const bytes = fontBytes("Roboto-Regular.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const first = parseLookupActions(font, "GPOS", 13);
        expect(parseLookupActions(font, "GPOS", 13)).toBe(first);
        expect(parseLookupActions(font, "GPOS", 1)).not.toBe(first);
        expect(parseLookupActions(createComputeFontFromBuffer(bytes), "GPOS", 13)).not.toBe(first);
    });

    it("shares class coverage and indexes every nonzero or rejected class-pair cell without changing logical rule order", () => {
        const font = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        const lookup = program.lookups.find((entry) => entry.table === "GPOS" && entry.lookupIndex === 1)!;
        const base = program.gposDescriptorOffset + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
        const ruleOrigin = program.words[base + COMPUTE_LOOKUP_RULE_ORIGIN]!;
        let tested = 0;
        const coverageOrigins = new Map<object, number>();
        lookup.rules.forEach((rule, index) => {
            if (rule.input[0]?.kind !== "class" || rule.input[1]?.kind !== "class") {
                return;
            }
            const slot = ruleOrigin + index * COMPUTE_PROGRAM_RULE_WORDS;
            const input = program.words[slot]!;
            const coverage = rule.input[0].coverage!;
            const covered = coverageGlyphs(coverage, font.glyphCount);
            const origin = program.words[input + 2]!;
            expect([...program.words.slice(origin, origin + 1 + covered.length)]).toEqual([covered.length, ...covered]);
            if (coverageOrigins.has(coverage)) {
                expect(origin).toBe(coverageOrigins.get(coverage));
            }
            coverageOrigins.set(coverage, origin);
            const row = program.words[slot + 10]!;
            expect(row).toBeGreaterThan(0);
            expect(program.words[row + rule.input[1].classId]).toBe(index + 1);
            expect(program.classTables[program.words[slot + 11]!]).toBe(rule.input[1].classDef);
            expect(program.words[slot + 13]).toBe(
                lookup.rules.filter((other) => other.input[0]?.classDef === rule.input[0]?.classDef && other.input[0]?.classId === rule.input[0]?.classId).length
            );
            tested++;
        });
        expect(tested).toBeGreaterThan(1000);
    });

    it("classifies only dependency-free single and pair roots for cooperative predicate selection", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        let singles = 0;
        let pairs = 0;
        for (const lookup of program.lookups) {
            const base = (lookup.table === "GSUB" ? program.gsubDescriptorOffset : program.gposDescriptorOffset) + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
            const independent = lookup.rules.every((rule) => rule.backtrack.length === 0 && rule.lookahead.length === 0);
            const single = lookup.table === "GSUB" && independent && lookup.rules.every((rule) => rule.input.length === 1 && (rule.mode === 1 || rule.mode === 5));
            const pair = lookup.table === "GPOS" && independent && lookup.rules.every((rule) => rule.input.length === 2 && [3, 4, 5].includes(rule.mode));
            const firstOnly = lookup.rules.every(
                (rule) =>
                    rule.action.kind === "reject" ||
                    ((rule.action.kind === "pair" || rule.action.kind === "class-pair") && (!rule.action.applySecond || rule.action.secondXAdvance === 0))
            );
            expect(program.words[base + COMPUTE_LOOKUP_MEMBERSHIP], `${lookup.table}/${lookup.lookupIndex}`).toBe(single ? 1 : pair ? (firstOnly ? 3 : 2) : 0);
            singles += Number(single);
            pairs += Number(pair);
        }
        expect(singles).toBeGreaterThan(0);
        expect(pairs).toBeGreaterThan(0);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("encodes ordered eligibility and forward-only proof without changing cooperative capabilities: %s", (file) => {
        const font = createComputeFontFromBuffer(fontBytes(file));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        for (const lookup of program.lookups) {
            const base = (lookup.table === "GSUB" ? program.gsubDescriptorOffset : program.gposDescriptorOffset) + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
            const modes = lookup.table === "GSUB" ? [1, 2, 5, 9, 11] : [3, 4, 5, 7, 8, 10, 12];
            const eligibility = lookup.rules.every((rule) => modes.includes(rule.mode)) ? (lookup.rules.some((rule) => rule.mode === 11) ? 2 : 1) : 0;
            const forward = eligibility && lookup.table === "GSUB" && lookup.rules.every((rule) => rule.backtrack.length === 0) ? 4 : 0;
            expect(program.words[base + COMPUTE_LOOKUP_ORDERED], `${lookup.table}/${lookup.lookupIndex}`).toBe(eligibility | forward);
        }
    });

    it("compiles Roboto's Latin plan as executable text-shaper actions", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("fi")));
        expect(program.rejections.map((rejection) => `${rejection.table}/${rejection.lookupIndex} ${rejection.reason}`).slice(0, 12)).toEqual([]);
        expect(program.fullyExecutable).toBe(true);
        expect(program.lookups.some((lookup) => lookup.table === "GSUB" && lookup.lookupType === 4 && lookup.rules.some((rule) => rule.action.kind === "ligature"))).toBe(true);
        expect(
            program.lookups.filter((lookup) => lookup.table === "GSUB" && lookup.lookupType === 6).every((lookup) => lookup.rules.every((rule) => rule.action.kind === "context"))
        ).toBe(true);
        const pairs = program.lookups.find((lookup) => lookup.table === "GPOS" && lookup.lookupIndex === 13);
        expect(pairs?.rules.every((rule) => (rule.action.kind === "pair" || rule.action.kind === "class-pair") && rule.action.applySecond)).toBe(true);
        expect(
            program.lookups.some(
                (lookup) => lookup.lookupType === 4 && lookup.table === "GPOS" && lookup.rules.some((rule) => rule.action.kind === "mark-base" && rule.action.clearAdvance)
            )
        ).toBe(true);
        expect(modesOf(program)).not.toContain(6);
        expect(program.words[0]).toBe(0x43545031);
    });

    it("admits Inter design values for zero default device deltas and keeps nested ligatures fail-closed", () => {
        expect(TEXT_SHAPER_VERSION).toBe("0.1.28");
        const implementation = [
            "packages/babylon-lite/src/text/compute-text-program.ts",
            "packages/babylon-lite/src/text/font-parser/device-delta.ts",
            "packages/babylon-lite/src/text/font-parser/layout-actions.ts",
        ]
            .map((file) => readFileSync(join(process.cwd(), file), "utf8"))
            .join("\n");
        expect(implementation).not.toMatch(/from ["']text-shaper["']/);
        const bytes = fontBytes("Inter.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        expect(program.fullyExecutable).toBe(false);
        const reason = (table: "GSUB" | "GPOS", lookupIndex: number): string =>
            program.rejections.find((rejection) => rejection.table === table && rejection.lookupIndex === lookupIndex)?.reason ?? "";
        expect(reason("GSUB", 6)).toMatch(/nested GSUB lookup 7 type 4/);
        expect(reason("GSUB", 9)).toMatch(/nested GSUB lookup 10 type 4/);
        expect(reason("GSUB", 48)).toMatch(/nested GSUB lookup 49 type 4/);
        for (const index of [1, 2, 4, 5, 6, 7, 8, 9]) {
            expect(reason("GPOS", index), `GPOS ${index}`).toBe("");
        }
        expect(program.lookups.find((lookup) => lookup.table === "GSUB" && lookup.lookupIndex === 5)?.rules.every((rule) => rule.action.kind === "ligature")).toBe(true);
        expect(program.lookups.find((lookup) => lookup.table === "GSUB" && lookup.lookupIndex === 3)?.rules.every((rule) => rule.action.kind === "context")).toBe(true);
        expect(
            program.lookups
                .find((lookup) => lookup.table === "GPOS" && lookup.lookupIndex === 3)
                ?.rules.every((rule) => rule.action.kind === "pair" && rule.action.applySecond === false)
        ).toBe(true);
        const filtered = program.lookups.find((lookup) => lookup.table === "GPOS" && lookup.lookupIndex === 1);
        expect(filtered?.markFilteringSetIgnored).toBe(true);
        expect(filtered?.rules.every((rule) => rule.action.kind === "pair" || rule.action.kind === "class-pair")).toBe(true);
        expect(modesOf(program)).not.toContain(6);

        const shaped = shapeFields(bytes);
        expect(shaped("AC")).toEqual([
            [2, 0, 1343, 0, 0, 0],
            [73, 1, 1496, 0, 0, 0],
        ]);
        const classPair = parseLookupActions(font, "GPOS", 1).subtables.find((subtable) => subtable.kind === "class-pair");
        if (!classPair || classPair.kind !== "class-pair") {
            throw new Error("GPOS 1 class pair missing");
        }
        const class1 = classOf(classPair.classDef1, 2);
        const class2 = classOf(classPair.classDef2, 73);
        expect(classPair.rows[class1]?.[class2]?.value1.deviceKind).toBe("variation");
        expect(classPair.rows[class1]?.[class2]?.value1.xAdvance).toBe(-70);
        const compiledClass = filtered?.rules.find((rule) => rule.action.kind === "class-pair" && rule.action.class1 === class1 && rule.action.class2 === class2);
        expect(compiledClass?.action).toMatchObject({ kind: "class-pair", firstXAdvance: -70 });
        expect(readComputeFontAdvance(font, 2) + (compiledClass?.action.kind === "class-pair" ? compiledClass.action.firstXAdvance : 0)).toBe(1343);

        expect(shaped("5,")).toEqual([
            [1347, 0, 1160, 0, 0, 0],
            [1501, 1, 590, 0, 0, 0],
        ]);
        const pairLookup = program.lookups.find((lookup) => lookup.table === "GPOS" && lookup.lookupIndex === 2);
        const compiledPair = pairLookup?.rules.find((rule) => rule.action.kind === "pair" && rule.input[0]?.ids[0] === 1347 && rule.action.secondGlyph === 1501);
        expect(compiledPair?.action).toMatchObject({ kind: "pair", firstXAdvance: -55 });
        expect(readComputeFontAdvance(font, 1347) + (compiledPair?.action.kind === "pair" ? compiledPair.action.firstXAdvance : 0)).toBe(1160);

        expect(shaped("A\u0326")).toEqual([
            [2, 0, 1413, 0, 0, 0],
            [1776, 1, 0, 0, 1309, 0],
        ]);
        const markBase = program.lookups.find((lookup) => lookup.table === "GPOS" && lookup.lookupIndex === 4);
        const comma = markBase?.rules.find((rule) => rule.action.kind === "mark-base" && rule.input[0]?.ids[0] === 1776);
        expect(comma?.action.kind).toBe("mark-base");
        if (comma?.action.kind === "mark-base") {
            const base = comma.action.bases.find((anchor) => anchor.glyphId === 2);
            expect(base).toBeDefined();
            expect((base?.x ?? 0) - comma.action.markX).toBe(1309);
            expect((base?.y ?? 0) - comma.action.markY).toBe(0);
            expect(comma.action.clearAdvance).toBe(true);
        }

        expect(shaped("\u0301\u0300")).toEqual([
            [1770, 0, 0, 0, 0, 0],
            [1771, 1, 0, 0, -167, 440],
        ]);
        const markMark = program.lookups.find((lookup) => lookup.table === "GPOS" && lookup.lookupIndex === 9);
        const grave = markMark?.rules.find((rule) => rule.action.kind === "mark-mark" && rule.input[0]?.ids[0] === 1771);
        expect(grave?.action.kind).toBe("mark-mark");
        if (grave?.action.kind === "mark-mark") {
            const base = grave.action.bases.find((anchor) => anchor.glyphId === 1770);
            expect((base?.x ?? 0) - grave.action.markX).toBe(-167);
            expect((base?.y ?? 0) - grave.action.markY).toBe(440);
            expect(grave.action.clearAdvance).toBe(false);
        }

        expect(shaped("i\u0301")).toEqual([[699, 0, 496, 0, 0, 0]]);
        expect(shaped("e\u0301")).toEqual([[618, 0, 1194, 0, 0, 0]]);
        expect(
            program.lookups.some(
                (lookup) =>
                    lookup.table === "GSUB" && lookup.rules.some((rule) => rule.action.kind === "ligature" && rule.action.glyph === 699 && rule.action.components.includes(1770))
            )
        ).toBe(true);
        expect(shaped("A\u20de")).toEqual([
            [2, 0, 1413, 0, 0, 0],
            [1902, 1, 2828, 0, 0, 0],
        ]);
        expect(shaped("A\u20dd")).toEqual([
            [2, 0, 1413, 0, 0, 0],
            [1901, 1, 2846, 0, 0, 0],
        ]);
        expect(shaped("<-")).toEqual([
            [1516, 0, 1355, 0, 0, 0],
            [1459, 1, 942, 0, 0, 0],
        ]);
    });

    it("rejects an unknown selected lookup type instead of skipping it", () => {
        const font = createComputeFontFromBuffer(fontWithGsub(unknownTypeLookup()));
        expect(() => compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")))).toThrow(/unknown GSUB lookup type 9/);
    });

    it("keeps Arabic Context-3 roots oracle-inert without weakening Inter rejection predicates", () => {
        const bytes = fontBytes("NotoSansArabic-Regular.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("سلام")));
        const parsed = parseLookupActions(font, "GSUB", 35);
        expect(parsed.subtables.every((entry) => entry.kind === "context" && !entry.chaining && entry.format === 3)).toBe(true);
        expect(program.lookups.find((lookup) => lookup.table === "GSUB" && lookup.lookupIndex === 35)?.rules).toEqual([]);
        expect(Font.load(bytes).gsub?.lookups[35]?.digest.mayHave(72)).toBe(false);
        const inter = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        const rejected = compileComputeTextProgram(createComputeTextPlan(inter, selectComputeTextRun("A")));
        for (const lookupIndex of [6, 9, 48]) {
            expect(
                rejected.lookups.find((lookup) => lookup.table === "GSUB" && lookup.lookupIndex === lookupIndex)?.rules.some((rule) => rule.mode === COMPUTE_RULE_MODE_REJECT)
            ).toBe(true);
        }
    });

    it("keeps an understood alternate substitution as a fail-if-matched rejection", () => {
        const font = createComputeFontFromBuffer(fontWithGsub(alternateLookup()));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        expect(program.fullyExecutable).toBe(false);
        const lookup = program.lookups.find((entry) => entry.lookupIndex === 0);
        expect(lookup?.rules).toHaveLength(1);
        expect(lookup?.rules[0]?.mode).toBe(COMPUTE_RULE_MODE_REJECT);
        expect(lookup?.rules[0]?.action).toMatchObject({ kind: "reject" });
        expect(lookup?.rules[0]?.input[0]?.ids).toEqual([1]);
        const origin = program.gsubDescriptorOffset + COMPUTE_LOOKUP_KIND;
        expect(program.words[origin]).toBe(3);
        expect(program.words[origin + COMPUTE_LOOKUP_FLAG]).toBe(0);
        expect(program.words[origin + COMPUTE_LOOKUP_RULE_COUNT]).toBe(1);
        const ruleOrigin = program.words[origin + 3] ?? 0;
        expect(program.words[ruleOrigin + COMPUTE_RULE_MODE]).toBe(COMPUTE_RULE_MODE_REJECT);
        expect(COMPUTE_PROGRAM_RULE_WORDS).toBe(16);
        expect(COMPUTE_PROGRAM_LOOKUP_WORDS).toBe(8);
    });

    it("compiles a single substitution and does not emit a no-op mode", () => {
        const font = createComputeFontFromBuffer(fontWithGsub(singleLookup(0)));
        const program = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A")));
        expect(program.fullyExecutable).toBe(true);
        expect(program.lookups[0]?.rules[0]).toMatchObject({
            mode: COMPUTE_RULE_MODE_SUBSTITUTE,
            action: { kind: "substitute", glyphs: [1] },
        });
        expect(modesOf(program)).not.toContain(5);
        expect(modesOf(program)).not.toContain(6);
    });
});

function shapeFields(bytes: ArrayBuffer): (text: string) => number[][] {
    const shaper = Font.load(bytes);
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    return (text: string) => {
        input.clear();
        input.addStr(text);
        input.setScript("latn");
        shapeInto(shaper, input, output, { script: "latn", language: null, direction: "ltr" });
        return output.infos.slice(0, output.length).map((info, index) => {
            const position = output.positions[index];
            return [info.glyphId, info.cluster, position?.xAdvance ?? 0, position?.yAdvance ?? 0, position?.xOffset ?? 0, position?.yOffset ?? 0];
        });
    };
}

function classOf(def: GlyphClassDef, glyph: number): number {
    if (def.format === 1) {
        return def.classes[glyph - def.startGlyphId] ?? 0;
    }
    for (const range of def.ranges) {
        if (glyph >= range.start && glyph <= range.end) {
            return range.classValue;
        }
    }
    return 0;
}

function modesOf(program: ReturnType<typeof compileComputeTextProgram>): number[] {
    return program.lookups.flatMap((lookup) => lookup.rules.map((rule) => rule.mode));
}

function fontWithGsub(gsub: Uint8Array): ArrayBuffer {
    return buildSfnt({
        cmap: cmapFormat4(0x41, 1),
        glyf: new Uint8Array([0]),
        head: headTable(),
        hhea: hheaTable(),
        hmtx: hmtxTable(),
        loca: new Uint8Array(6),
        maxp: maxpTable(),
        GSUB: gsub,
    });
}

function singleLookup(delta: number): Uint8Array {
    const lookup = new Writer();
    lookup.u16(1);
    lookup.u16(4);
    lookup.u16(1);
    lookup.u16(0);
    lookup.u16(1);
    lookup.u16(8);
    lookup.u16(1);
    lookup.u16(6);
    lookup.i16(delta);
    writeCoverage(lookup);
    return assembleGsub(scriptTable("latn", 0xffff, [0]), featureTable("ccmp", [0]), lookup.toUint8());
}

function unknownTypeLookup(): Uint8Array {
    const lookup = new Writer();
    lookup.u16(1);
    lookup.u16(4);
    lookup.u16(9);
    lookup.u16(0);
    lookup.u16(1);
    lookup.u16(8);
    lookup.u16(1);
    lookup.u16(0);
    return assembleGsub(scriptTable("latn", 0xffff, [0]), featureTable("ccmp", [0]), lookup.toUint8());
}

function alternateLookup(): Uint8Array {
    const lookup = new Writer();
    lookup.u16(1);
    lookup.u16(4);
    lookup.u16(3);
    lookup.u16(0);
    lookup.u16(1);
    lookup.u16(8);
    const subtable = lookup.offset();
    lookup.u16(1);
    const coverageAt = lookup.offset();
    lookup.u16(0);
    lookup.u16(1);
    const setAt = lookup.offset();
    lookup.u16(0);
    lookup.patchU16(coverageAt, lookup.offset() - subtable);
    writeCoverage(lookup);
    lookup.patchU16(setAt, lookup.offset() - subtable);
    lookup.u16(1);
    lookup.u16(2);
    return assembleGsub(scriptTable("latn", 0xffff, [0]), featureTable("ccmp", [0]), lookup.toUint8());
}

function writeCoverage(writer: Writer): void {
    writer.u16(1);
    writer.u16(1);
    writer.u16(1);
}

function scriptTable(tag: string, requiredFeatureIndex: number, featureIndices: number[]): Uint8Array {
    const writer = new Writer();
    writer.u16(1);
    writer.tag(tag);
    writer.u16(8);
    writer.u16(4);
    writer.u16(0);
    writer.u16(0);
    writer.u16(requiredFeatureIndex);
    writer.u16(featureIndices.length);
    for (const index of featureIndices) {
        writer.u16(index);
    }
    return writer.toUint8();
}

function featureTable(tag: string, lookups: number[]): Uint8Array {
    const writer = new Writer();
    writer.u16(1);
    writer.tag(tag);
    writer.u16(8);
    writer.u16(0);
    writer.u16(lookups.length);
    for (const lookup of lookups) {
        writer.u16(lookup);
    }
    return writer.toUint8();
}

function assembleGsub(script: Uint8Array, feature: Uint8Array, lookup: Uint8Array): Uint8Array {
    const out = new Uint8Array(10 + script.length + feature.length + lookup.length);
    const view = new DataView(out.buffer);
    view.setUint16(0, 1);
    const scriptAt = 10;
    const featureAt = scriptAt + script.length;
    const lookupAt = featureAt + feature.length;
    view.setUint16(4, scriptAt);
    view.setUint16(6, featureAt);
    view.setUint16(8, lookupAt);
    out.set(script, scriptAt);
    out.set(feature, featureAt);
    out.set(lookup, lookupAt);
    return out;
}

class Writer {
    readonly bytes: number[] = [];

    offset(): number {
        return this.bytes.length;
    }

    u16(value: number): void {
        this.bytes.push((value >> 8) & 0xff, value & 0xff);
    }

    i16(value: number): void {
        this.u16(value < 0 ? value + 0x10000 : value);
    }

    tag(value: string): void {
        for (let index = 0; index < 4; index++) {
            this.bytes.push(value.charCodeAt(index) || 0x20);
        }
    }

    patchU16(at: number, value: number): void {
        this.bytes[at] = (value >> 8) & 0xff;
        this.bytes[at + 1] = value & 0xff;
    }

    toUint8(): Uint8Array {
        return Uint8Array.from(this.bytes);
    }
}

function buildSfnt(tables: Record<string, Uint8Array>): ArrayBuffer {
    const tags = Object.keys(tables);
    const out = new Uint8Array(12 + tags.length * 16 + tags.reduce((sum, tag) => sum + (tables[tag]?.length ?? 0), 0));
    const view = new DataView(out.buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, tags.length);
    let offset = 12 + tags.length * 16;
    tags.forEach((tag, index) => {
        const body = tables[tag] ?? new Uint8Array();
        const at = 12 + index * 16;
        for (let byte = 0; byte < 4; byte++) {
            view.setUint8(at + byte, tag.charCodeAt(byte) || 0x20);
        }
        view.setUint32(at + 8, offset);
        view.setUint32(at + 12, body.length);
        out.set(body, offset);
        offset += body.length;
    });
    return out.buffer;
}

function headTable(): Uint8Array {
    const bytes = new Uint8Array(54);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint32(12, 0x5f0f3cf5);
    view.setUint16(18, 1000);
    return bytes;
}

function hheaTable(): Uint8Array {
    const bytes = new Uint8Array(36);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint16(34, 2);
    return bytes;
}

function hmtxTable(): Uint8Array {
    const bytes = new Uint8Array(8);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 100);
    view.setUint16(4, 200);
    return bytes;
}

function maxpTable(): Uint8Array {
    const bytes = new Uint8Array(32);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, 2);
    return bytes;
}

function cmapFormat4(codePoint: number, glyphId: number): Uint8Array {
    const bytes = new Uint8Array(4 + 8 + 32);
    const view = new DataView(bytes.buffer);
    view.setUint16(2, 1);
    view.setUint16(6, 3);
    view.setUint32(8, 12);
    const at = 12;
    view.setUint16(at, 4);
    view.setUint16(at + 2, 32);
    view.setUint16(at + 6, 4);
    view.setUint16(at + 8, 4);
    view.setUint16(at + 10, 1);
    view.setUint16(at + 12, 2);
    view.setUint16(at + 14, codePoint);
    view.setUint16(at + 16, 0xffff);
    view.setUint16(at + 20, codePoint);
    view.setUint16(at + 22, 0xffff);
    view.setInt16(at + 26, glyphId - codePoint);
    return bytes;
}
