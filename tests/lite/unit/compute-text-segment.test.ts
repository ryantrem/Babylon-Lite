import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createComputeFontFromBuffer, mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import {
    compileComputeTextProgram,
    COMPUTE_RULE_MODE_MULTIPLE,
    COMPUTE_RULE_MODE_MARK_BASE,
    COMPUTE_RULE_MODE_LIGATURE,
    COMPUTE_RULE_MODE_CONTEXT,
    COMPUTE_RULE_MODE_SUBSTITUTE,
    COMPUTE_RULE_MODE_MARK_LIGATURE,
    COMPUTE_RULE_MODE_POSITION,
} from "../../../packages/babylon-lite/src/text/compute-text-program";
import type { ComputeGlyphPredicate, ComputeProgramLookup, ComputeProgramRule, ComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { computeTextParagraphFlags } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-flags";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import {
    analyzeComputeTextSegmentation,
    clearComputeTextSegmentationCache,
    SAFE_SPACE_FLAG,
    segmentComputeTextRuns,
    SEGMENT_TARGET,
} from "../../../packages/babylon-lite/src/text/compute-text-segment";
import type { SegmentGroup } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import * as fontMapping from "../../../packages/babylon-lite/src/text/compute-font";
import * as profiles from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { syntheticFont } from "../compute/synthetic";
import { STITCH_TILE, stitchTileRows } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";

describe("stitch tile row sizing", () => {
    it("uses the maximum local run segment count, not segment offsets or runBase", () => {
        expect(STITCH_TILE).toBe(64);
        expect(stitchTileRows(new Uint32Array(), 0)).toBe(1);
        expect(stitchTileRows(Uint32Array.of(9000, 0, 10000, 1), 2)).toBe(1);
        expect(stitchTileRows(Uint32Array.of(0, 64, 64, 65, 129, 129), 3)).toBe(3);
        expect(stitchTileRows(Uint32Array.of(0, 1961, 1961, 3, 0xffffffff, 0xffffffff), 2)).toBe(31);
        expect(stitchTileRows(Uint32Array.of(0, 65535), 1)).toBe(1024);
    });
});

const limits = { maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024 };
const layout = { headerWords: 4, recordWords: 6 };
const paragraph =
    "The quick brown fox jumps over the lazy dog. Every letter belongs to a flowing English paragraph with ordinary spaces and punctuation. AV To are adjacent pairs; office and affinity remain ordered shaping controls. ";
const phrases = [
    "Office affinity offers fine AVATAR kerning. ",
    "The efficient flight lifts off at five. ",
    "WAVY typography: coffee, waffles and films. ",
    "A quick brown fox visits the office today. ",
    "Friendly staff review lovely typography. ",
    "Affluent traffic flows over the river. ",
    "Vast valleys offer fresh coffee at noon. ",
    "To verify fonts, try AV WA To fi fl ffi. ",
];
const extra = "Hello (world) - it's a test: x = y + 2; [a] {b} @home <tag> ~tilde* \"quoted\" 'single' 12:30 -> => ... end. ";

function program(file = "Inter.ttf", text = "a b"): ComputeTextProgram {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", file));
    const font = createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    return compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun(text, font)));
}

function glyphs(p: ComputeTextProgram, text: string): Uint32Array {
    return mapComputeFontScalars(p.plan.font, Uint32Array.from([...text].map((c) => c.codePointAt(0)!)), (_i, error) => {
        throw error;
    });
}

function group(p: ComputeTextProgram, texts: readonly string[], outputRun = 0): SegmentGroup {
    return {
        plan: p.plan,
        outputRun,
        runs: texts.map((text) => ({ scalars: Uint32Array.from([...text].map((c) => c.codePointAt(0)!)), glyphIds: glyphs(p, text) })),
    };
}

function predicate(...ids: number[]): ComputeGlyphPredicate {
    return { kind: "glyphs", ids, classId: 0, classDef: null, coverage: null };
}

function orderedProgram(roots: readonly (readonly ComputeProgramRule[])[], dependencies: readonly (readonly ComputeProgramRule[])[] = []): ComputeTextProgram {
    const p = program();
    const lookups: ComputeProgramLookup[] = [...roots, ...dependencies].map((rules, lookupIndex) => ({
        ...p.lookups[0]!,
        table: "GSUB",
        lookupIndex,
        rules,
    }));
    return {
        ...p,
        plan: { ...p.plan, _gsubRoots: roots.map((_rules, index) => index), _gsubDependencies: dependencies.map((_rules, index) => roots.length + index), _gposRoots: [] },
        lookups,
        rootOrder: roots.map((_rules, index) => index),
        maxDependencyDepth: dependencies.length ? 1 : 0,
    };
}

function substitution(input: ComputeGlyphPredicate, output: number): ComputeProgramRule {
    return { input: [input], backtrack: [], lookahead: [], mode: COMPUTE_RULE_MODE_SUBSTITUTE, action: { kind: "substitute", glyphs: [output] } };
}

function ligature(input: number[], output: number): ComputeProgramRule {
    return { input: input.map((g) => predicate(g)), backtrack: [], lookahead: [], mode: COMPUTE_RULE_MODE_LIGATURE, action: { kind: "ligature", components: [], glyph: output } };
}

describe("safe logical-run segmentation", () => {
    beforeEach(() => clearComputeTextSegmentationCache());

    it.each([
        ["Roboto-Regular.ttf", "The quick", true],
        ["Roboto-Regular.ttf", "AV To", false],
        ["Roboto-Regular.ttf", "kerning. The efficient", false],
        ["Roboto-Regular.ttf", "noon. To verify", false],
        ["Roboto-Regular.ttf", "WA To fi", false],
        ["Roboto-Regular.ttf", "quick brown", true],
    ])("checks the spec's %s / %s boundary expectation (safe=%s)", (file, text, safe) => {
        const p = program(file);
        expect(analyzeComputeTextSegmentation(p)!.safeAfter(glyphs(p, text), text.indexOf(" "))).toBe(safe);
    });

    it.each([
        { file: "Inter.ttf", corpus: "english", text: paragraph.repeat(3), boundaries: 101, safe: 101, unsafe: [] },
        { file: "Inter.ttf", corpus: "natural", text: phrases.join(""), boundaries: 54, safe: 54, unsafe: [] },
        {
            file: "Inter.ttf",
            corpus: "extra",
            text: extra.repeat(2),
            boundaries: 45,
            safe: 32,
            unsafe: ["+ 2; [a] ", " [a] {b} ", " {b} @hom", "2:30 -> =", "0 -> => .", "> => ... ", " ... end."],
        },
        { file: "Roboto-Regular.ttf", corpus: "english", text: paragraph.repeat(3), boundaries: 101, safe: 96, unsafe: [". AV To a", "ols. The "] },
        { file: "Roboto-Regular.ttf", corpus: "natural", text: phrases.join(""), boundaries: 54, safe: 51, unsafe: ["ing. The ", "oon. To v", "V WA To f"] },
        { file: "Roboto-Regular.ttf", corpus: "extra", text: extra.repeat(2), boundaries: 45, safe: 45, unsafe: [] },
    ])("matches authoritative $file / $corpus safe counts and unsafe-context sets", ({ file, corpus, text, boundaries, safe, unsafe }) => {
        const p = program(file);
        const a = analyzeComputeTextSegmentation(p)!;
        const gl = glyphs(p, text);
        const contexts = new Set<string>();
        let checked = 0;
        let safeCount = 0;
        for (let s = 0; s + 1 < gl.length; s++) {
            if (gl[s] !== a.space) {
                continue;
            }
            checked++;
            const context = text.slice(s - 4, s + 5);
            const actual = a.safeAfter(gl, s);
            if (actual) {
                safeCount++;
            } else {
                contexts.add(context);
            }
        }
        expect(checked).toBe(boundaries);
        expect(safeCount).toBe(safe);
        expect([...contexts].sort()).toEqual([...unsafe].sort());
        if (file === "Roboto-Regular.ttf" && corpus !== "extra") {
            let periodBoundaries = 0;
            for (const match of text.matchAll(/\. (?:To|The)/g)) {
                expect(a.safeAfter(gl, match.index + 1), `period before To/The at ${match.index}`).toBe(false);
                periodBoundaries++;
            }
            expect(periodBoundaries).toBeGreaterThan(0);
        }
        if (corpus === "english") {
            for (const match of text.matchAll(/The quick/g)) {
                expect(a.safeAfter(gl, match.index + 3), `space after The at ${match.index}`).toBe(true);
            }
        }
    });

    it("accepts every English Inter boundary, but not a final space or a non-space", () => {
        const p = program();
        const a = analyzeComputeTextSegmentation(p)!;
        const text = paragraph.repeat(3);
        const gl = glyphs(p, text);
        let checked = 0;
        for (let s = 0; s + 1 < gl.length; s++) {
            if (text[s] === " ") {
                expect(a.safeAfter(gl, s), `space ${s}`).toBe(true);
                checked++;
            }
        }
        expect(checked).toBe(101);
        expect(a.safeAfter(gl, 0)).toBe(false);
        expect(a.safeAfter(gl, gl.length - 1)).toBe(false);
    });

    it("rejects a possibly-mark first right glyph when space is an attachment base", () => {
        const p = program("Roboto-Regular.ttf");
        const space = glyphs(p, " ")[0]!;
        const mark = glyphs(p, "\u0301")[0]!;
        const modified: ComputeTextProgram = {
            ...p,
            lookups: [
                ...p.lookups,
                {
                    ...p.lookups[0]!,
                    table: "GPOS",
                    rules: [
                        {
                            input: [predicate(mark)],
                            backtrack: [],
                            lookahead: [],
                            mode: COMPUTE_RULE_MODE_MARK_BASE,
                            action: { kind: "mark-base", markClass: 0, markX: 0, markY: 0, bases: [{ glyphId: space, x: 0, y: 0 }] },
                        },
                    ],
                },
            ],
        };
        const a = analyzeComputeTextSegmentation(modified)!;
        expect(a.markBaseHit).toBe(true);
        expect(a.safeAfter(glyphs(p, " \u0301a"), 0)).toBe(false);
        expect(a.safeAfter(glyphs(p, " a"), 0)).toBe(true);
    });

    it("also rejects a mark after a B-star MarkLigature anchor, including propagation", () => {
        const p = program("Roboto-Regular.ttf");
        const space = glyphs(p, " ")[0]!;
        const mark = glyphs(p, "\u0301")[0]!;
        const modified: ComputeTextProgram = {
            ...p,
            lookups: [
                ...p.lookups,
                {
                    ...p.lookups[0]!,
                    table: "GPOS",
                    rules: [
                        {
                            input: [predicate(mark)],
                            backtrack: [],
                            lookahead: [],
                            mode: COMPUTE_RULE_MODE_MARK_LIGATURE,
                            action: {
                                kind: "mark-ligature",
                                markX: 0,
                                markY: 0,
                                ligatures: [{ glyphId: space, componentCount: 1, anchors: [{ component: 0, x: 0, y: 0, reject: false }] }],
                            },
                        },
                    ],
                },
            ],
        };
        const proof = analyzeComputeTextSegmentation(modified)!;
        expect(proof.markBaseHit).toBe(true);
        expect(proof.safeAfter(glyphs(p, " \u0301a"), 0)).toBe(false);
        expect(proof.safeAfter(glyphs(p, " a"), 0)).toBe(true);
    });

    it("keeps the Arabic U boundary rule independent of the current RTL rejection", () => {
        // Isolate the necessary guard with an LTR proof model. This does NOT admit
        // an actual RTL plan, force production segmentation, or establish RTL stitch order.
        const p = program();
        const input = group(p, [paragraph.repeat(3)]);
        const arabic = { ...input, plan: { ...input.plan, _profile: "arabic" as const } };
        const support = vi.spyOn(profiles, "profileSupport").mockReturnValue({ glyphWords: 8, masked: true, planes: 1, joiningType: () => 2 });
        try {
            expect(segmentComputeTextRuns([arabic], limits, layout)).toBeNull();
            support.mockReturnValue({ glyphWords: 8, masked: true, planes: 1, joiningType: () => 0 });
            expect(segmentComputeTextRuns([arabic], limits, layout)).not.toBeNull();
        } finally {
            support.mockRestore();
        }
    });

    it.each([0, 0x10])("bounds contextual GPOS and its consuming full-value pair with flag %s", (lookupFlag) => {
        const p = program();
        const [a, space, b, c] = glyphs(p, "A BC");
        const pair: ComputeProgramRule = {
            input: [predicate(space!), predicate(b!)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_POSITION,
            action: {
                kind: "pair",
                secondGlyph: b!,
                format2Nonzero: true,
                first: { xPlacement: 11, yPlacement: 17, xAdvance: 13, yAdvance: 0 },
                second: { xPlacement: 19, yPlacement: 23, xAdvance: 29, yAdvance: 0 },
            },
        };
        const context: ComputeProgramRule = {
            input: [predicate(space!), predicate(b!)],
            backtrack: [predicate(a!)],
            lookahead: [predicate(c!)],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GPOS", lookups: [{ sequenceIndex: 0, lookupIndex: 1 }] },
        };
        const model: ComputeTextProgram = {
            ...p,
            plan: { ...p.plan, _gsubRoots: [], _gsubDependencies: [], _gposRoots: [0], _gposDependencies: [1] },
            lookups: [context, pair].map((rule, index) => ({ ...p.lookups[0]!, table: "GPOS", lookupIndex: index, lookupFlag, rules: [rule] })),
            rootOrder: [0],
            maxDependencyDepth: 1,
        };
        const proof = analyzeComputeTextSegmentation(model)!;
        expect(proof).not.toBeNull();
        expect(proof.safeAfter(glyphs(p, "A BC"), 1)).toBe(false);
        expect(proof.safeAfter(glyphs(p, "A C"), 1)).toBe(true);
    });

    it("keeps a long nested-Multiple space run whole when its pinned font has no nominal space", () => {
        const face = syntheticFont("c4-nested-multiple");
        const input = "ABC ".repeat(32);
        expect([...input].length).toBeGreaterThan(64);
        expect(face.oracle.glyphId(32)).toBe(0);
        const p = compileComputeTextProgram(createComputeTextPlan(face.font, selectComputeTextRun("ABC")));
        expect(p.maxExpansion).toBe(2);
        expect(analyzeComputeTextSegmentation(p)).toBeNull();
        // The unmapped scalar is still rejected before any GPU input is published.
    });

    it.each(["NotoSansHebrew-Regular.ttf", "NotoSansArabic-Regular.ttf"])("keeps RTL plans ineligible: %s", (file) => {
        expect(analyzeComputeTextSegmentation(program(file, file.includes("Hebrew") ? "שלום" : "سلام"))).toBeNull();
    });

    it("keeps missing-space and two-expanding-lookup programs ineligible", () => {
        const font = syntheticFont("multiple").font;
        const p = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("A", font)));
        expect(analyzeComputeTextSegmentation(p)).toBeNull();
        const real = program();
        const expand = {
            ...real.lookups[0]!,
            table: "GSUB" as const,
            rules: [{ input: [predicate(100)], backtrack: [], lookahead: [], mode: COMPUTE_RULE_MODE_MULTIPLE, action: { kind: "multiple" as const, glyphs: [101, 102] } }],
        };
        expect(analyzeComputeTextSegmentation({ ...real, lookups: [expand, { ...expand, lookupIndex: 999 }] })).toBeNull();
    });

    it("bounds ordered multiple-to-ligature decompose/recompose cycles without re-consuming lookup outputs", () => {
        const decompose: ComputeProgramRule = {
            input: [predicate(103)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_MULTIPLE,
            action: { kind: "multiple", glyphs: [100, 101] },
        };
        const recompose = [ligature([100, 100], 102), ligature([100, 101], 103), ligature([103, 101], 104)];
        const p = orderedProgram([[decompose], recompose]);
        const a = analyzeComputeTextSegmentation(p);
        expect(a).not.toBeNull();
        expect(a!.span).toBe(2);
        expect(a!.safeAfter(glyphs(p, "Plain Latin paragraph."), 5)).toBe(true);
        expect(analyzeComputeTextSegmentation(orderedProgram([recompose, [decompose], [ligature([100, 101], 103)]]))!.span).toBe(4);
    });

    it("propagates ordered spans through class substitutions", () => {
        const cls: ComputeGlyphPredicate = {
            kind: "class",
            ids: [],
            classId: 1,
            classDef: { format: 1, startGlyphId: 103, classes: Uint16Array.of(1), ranges: [] },
            coverage: null,
        };
        const p = orderedProgram([[ligature([100, 101], 103)], [substitution(cls, 104)], [ligature([104, 101], 105)]]);
        expect(analyzeComputeTextSegmentation(p)!.span).toBe(3);
    });

    it("composes nested records at the context root, including a dependency that is also a later root", () => {
        const context: ComputeProgramRule = {
            input: [predicate(103)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: {
                kind: "context",
                table: "GSUB",
                lookups: [
                    { sequenceIndex: 0, lookupIndex: 2 },
                    { sequenceIndex: 0, lookupIndex: 4 },
                ],
            },
        };
        const p = orderedProgram([[ligature([100, 101], 103)], [context], [substitution(predicate(103), 104)], [ligature([105, 101], 106)]], [[substitution(predicate(104), 105)]]);
        expect(analyzeComputeTextSegmentation(p)!.span).toBe(3);
    });

    it("closes span-preserving nested substitutions across overlapping context matches", () => {
        const context = (input: number, child: number): ComputeProgramRule => ({
            input: [predicate(100), predicate(input)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: child }] },
        });
        const p = orderedProgram(
            [[ligature([100, 101], 103)], [context(104, 4), context(103, 3)], [ligature([105, 101], 106)]],
            [[substitution(predicate(103), 104)], [substitution(predicate(104), 105)]]
        );
        expect(analyzeComputeTextSegmentation(p)!.span).toBe(3);
    });

    it("rejects span-growing nested lookups whose outputs can be re-consumed by overlapping contexts", () => {
        const context: ComputeProgramRule = {
            input: [predicate(100), predicate(103)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: 1 }] },
        };
        expect(analyzeComputeTextSegmentation(orderedProgram([[context]], [[ligature([103, 101], 103)]]))).toBeNull();
    });

    it("admits non-reconsumable nested ligatures and composes their finite spans", () => {
        const context: ComputeProgramRule = {
            input: [predicate(100), predicate(102)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: 1 }] },
        };
        const result = analyzeComputeTextSegmentation(orderedProgram([[context]], [[ligature([102, 101], 104)]]));
        expect(result).not.toBeNull();
        expect(result!.span).toBe(2);
    });

    it.each(["glyph", "class"] as const)("rejects growing outputs seen by another context's lookahead (%s)", (kind) => {
        const output: ComputeGlyphPredicate =
            kind === "glyph"
                ? predicate(104)
                : {
                      kind: "class",
                      ids: [],
                      classId: 1,
                      classDef: { format: 1, startGlyphId: 104, classes: Uint16Array.of(1), ranges: [] },
                      coverage: null,
                  };
        const context: ComputeProgramRule = {
            input: [predicate(100), predicate(102)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: 1 }] },
        };
        const other: ComputeProgramRule = { ...context, input: [predicate(110)], lookahead: [output], action: { kind: "context", table: "GSUB", lookups: [] } };
        expect(analyzeComputeTextSegmentation(orderedProgram([[context, other]], [[ligature([102, 101], 104)]]))).toBeNull();
    });

    it("rejects a child ligature output that feeds its own component predicate", () => {
        const context: ComputeProgramRule = {
            input: [predicate(100), predicate(102)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: 1 }] },
        };
        expect(analyzeComputeTextSegmentation(orderedProgram([[context]], [[ligature([102, 101], 101)]]))).toBeNull();
    });

    it("keeps the one-expanding-lookup bound for nested Multiple", () => {
        const context: ComputeProgramRule = {
            input: [predicate(100), predicate(102)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: 2 }] },
        };
        const multiple = (input: number, outputs: number[]): ComputeProgramRule => ({
            input: [predicate(input)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_MULTIPLE,
            action: { kind: "multiple", glyphs: outputs },
        });
        expect(analyzeComputeTextSegmentation(orderedProgram([[context], [multiple(110, [111, 112])]], [[multiple(102, [104, 105])]]))).toBeNull();
        const one = { ...context, action: { kind: "context" as const, table: "GSUB" as const, lookups: [{ sequenceIndex: 1, lookupIndex: 1 }] } };
        expect(analyzeComputeTextSegmentation(orderedProgram([[one]], [[multiple(102, [104, 105])]]))).not.toBeNull();
    });

    it("keeps the B-star ligature-tail exclusion for nested children", () => {
        const context: ComputeProgramRule = {
            input: [predicate(100), predicate(102)],
            backtrack: [],
            lookahead: [],
            mode: COMPUTE_RULE_MODE_CONTEXT,
            action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 1, lookupIndex: 1 }] },
        };
        const base = program();
        const space = glyphs(base, " ")[0]!;
        expect(analyzeComputeTextSegmentation(orderedProgram([[context]], [[ligature([102, space], 104)]]))).toBeNull();
    });

    it("Inter latn analysis is non-null with nested ligature contexts", () => {
        const analysis = analyzeComputeTextSegmentation(program());
        expect(analysis).not.toBeNull();
        expect(analysis!.span).toBeLessThanOrEqual(8);
        expect(analysis!.maxD * analysis!.span).toBeLessThanOrEqual(64);
    });

    it("rejects unknown modes, missing nested dependencies and space ligature tails", () => {
        const p = program();
        const base = p.lookups[0]!;
        const input = [predicate(glyphs(p, " ")[0]!)];
        expect(
            analyzeComputeTextSegmentation({
                ...p,
                lookups: [{ ...base, rules: [{ input, backtrack: [], lookahead: [], mode: 999, action: { kind: "reject", reason: "unknown" } }] }],
            })
        ).toBeNull();
        expect(
            analyzeComputeTextSegmentation({
                ...p,
                lookups: [
                    {
                        ...base,
                        rules: [{ input, backtrack: [], lookahead: [], mode: 2, action: { kind: "context", table: "GSUB", lookups: [{ sequenceIndex: 0, lookupIndex: 999 }] } }],
                    },
                ],
            })
        ).toBeNull();
        expect(
            analyzeComputeTextSegmentation({
                ...p,
                lookups: [
                    {
                        ...base,
                        rules: [
                            {
                                input: [predicate(100), ...input],
                                backtrack: [],
                                lookahead: [],
                                mode: COMPUTE_RULE_MODE_LIGATURE,
                                action: { kind: "ligature", components: [], glyph: 101 },
                            },
                        ],
                    },
                ],
            })
        ).toBeNull();
    });

    it("caches eligible and ineligible results and does not analyze short-only updates", () => {
        const p = program();
        const mapping = vi.spyOn(fontMapping, "mapComputeFontScalars");
        const short = group(p, ["a b"]);
        mapping.mockClear();
        expect(segmentComputeTextRuns([short], limits, layout)).toBeNull();
        expect(mapping).not.toHaveBeenCalled();
        const a = analyzeComputeTextSegmentation(p);
        expect(mapping).toHaveBeenCalled();
        mapping.mockClear();
        expect(analyzeComputeTextSegmentation(p)).toBe(a);
        expect(mapping).not.toHaveBeenCalled();
        const bad = { ...p, plan: { ...p.plan, direction: "rtl" as const } };
        expect(analyzeComputeTextSegmentation(bad)).toBeNull();
        expect(analyzeComputeTextSegmentation(bad)).toBeNull();
        mapping.mockRestore();
    });

    it("honors the target and downward-before-upward search using zero-copy views", () => {
        const p = program();
        const downward = "a".repeat(SEGMENT_TARGET - 3) + " b" + "a".repeat(SEGMENT_TARGET);
        const upward = "a".repeat(SEGMENT_TARGET + 3) + " " + "a".repeat(SEGMENT_TARGET);
        const source = group(p, [downward, upward, "a b", "a".repeat(SEGMENT_TARGET * 3)]);
        const result = segmentComputeTextRuns([source], limits, layout)!;
        expect([...result.table.slice(0, 8)]).toEqual([0, 2, 2, 2, 4, 1, 5, 1]);
        expect([...result.table.slice(8)]).toEqual([0, SEGMENT_TARGET - 2, 0, SEGMENT_TARGET + 4, 0, 0]);
        expect(result.groups[0]!.runs[0]!.scalars.buffer).toBe(source.runs[0]!.scalars.buffer);
        expect(result.groups[0]!.runs[0]!.glyphIds.buffer).toBe(source.runs[0]!.glyphIds.buffer);
        expect(result.groups[0]!.runs[4]).toBe(source.runs[2]);
        expect(result.groups[0]!.runs[5]).toBe(source.runs[3]);
        const regular = segmentComputeTextRuns([group(p, ["a ".repeat(SEGMENT_TARGET * 2)])], limits, layout)!;
        expect(regular.groups[0]!.runs.every((r) => r.scalars.length <= SEGMENT_TARGET)).toBe(true);
    });

    it("returns null when nothing splits, including long unsplittable input", () => {
        const p = program();
        expect(segmentComputeTextRuns([group(p, ["a".repeat(SEGMENT_TARGET * 3), "a b"])], limits, layout)).toBeNull();
    });

    it("falls back on workgroup, output-buffer and table limits", () => {
        const p = program();
        const groups = [group(p, ["a ".repeat(SEGMENT_TARGET * 2)])];
        expect(segmentComputeTextRuns(groups, { ...limits, maxComputeWorkgroupsPerDimension: 1 }, layout)).toBeNull();
        expect(segmentComputeTextRuns(groups, { ...limits, maxStorageBufferBindingSize: 100 }, layout)).toBeNull();
        expect(segmentComputeTextRuns(groups, { ...limits, maxBufferSize: 100 }, layout)).toBeNull();
        expect(segmentComputeTextRuns(groups, { ...limits, maxBufferSize: 12 }, { headerWords: 0, recordWords: 0 })).toBeNull();
    });

    it("maps logical slots correctly even when groups are not source ordered", () => {
        const p = program();
        const result = segmentComputeTextRuns([group(p, ["a ".repeat(SEGMENT_TARGET)], 1), group(p, ["a b"], 0)], limits, layout)!;
        expect([...result.table.slice(0, 4)]).toEqual([2, 1, 0, 2]);
        expect(result.groups.map((g) => g.outputRun)).toEqual([0, 2]);
    });

    it("reuses paragraph safe-space flags with identical boundaries and analyzes only uncovered positions", () => {
        const p = program();
        const analysis = analyzeComputeTextSegmentation(p)!;
        const texts = [(paragraph + phrases.join("") + extra).repeat(6), ("a  b" + "c".repeat(70) + " ").repeat(8), "a b"];
        const source = group(p, texts);
        const flags = source.runs.map((run) => computeTextParagraphFlags(run.scalars, run.glyphIds, analysis));
        const expected = segmentComputeTextRuns([source], limits, layout)!;
        const safeAfter = vi.spyOn(analysis, "safeAfter");
        try {
            const hinted = segmentComputeTextRuns([source], limits, layout, [flags])!;
            expect([...hinted.table]).toEqual([...expected.table]);
            expect(hinted.groups[0]!.runs.map((run) => run.scalars.length)).toEqual(expected.groups[0]!.runs.map((run) => run.scalars.length));
            const covered = safeAfter.mock.calls.filter(([glyphIds, s]) => {
                const run = source.runs.find((r) => r.glyphIds === glyphIds)!;
                return run.scalars[s] === 0x20 && s + 1 < run.scalars.length && run.scalars[s + 1] !== 0x20;
            });
            expect(covered).toHaveLength(0);
            expect(safeAfter.mock.calls.length).toBeGreaterThan(0);
            safeAfter.mockClear();
            const forged = flags.map((f) => f.map((v) => (v === SAFE_SPACE_FLAG ? 1 : v)));
            expect(segmentComputeTextRuns([source], limits, layout, [forged])?.table).not.toEqual(expected.table);
            const mismatched = segmentComputeTextRuns([source], limits, layout, [[flags[0]!.subarray(1), ...flags.slice(1)]])!;
            expect([...mismatched.table]).toEqual([...expected.table]);
        } finally {
            safeAfter.mockRestore();
        }
    });

    it("markBaseHit is false for Inter and Roboto English programs", () => {
        for (const file of ["Inter.ttf", "Roboto-Regular.ttf"]) {
            expect(analyzeComputeTextSegmentation(program(file))!.markBaseHit).toBe(false);
        }
    });
});
