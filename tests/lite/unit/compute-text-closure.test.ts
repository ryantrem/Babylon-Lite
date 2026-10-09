import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createComputeFontFromBuffer, mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import type { ComputeFont } from "../../../packages/babylon-lite/src/text/compute-font";
import { closeComputeTextGlyphs } from "../../../packages/babylon-lite/src/text/compute-text-closure";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import type { ComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { readComputeGlyfGlyph } from "../../../packages/babylon-lite/src/text/compute-glyph-extraction";
import * as extraction from "../../../packages/babylon-lite/src/text/compute-glyph-extraction";
import * as actions from "../../../packages/babylon-lite/src/text/font-parser/layout-actions";
import type { ParsedSubtable } from "../../../packages/babylon-lite/src/text/font-parser/layout-actions";
import { coverageIndex } from "../../../packages/babylon-lite/src/text/font-parser/opentype-layout";

function load(file = "Roboto-Regular.ttf"): ComputeFont {
    const bytes = readFileSync(`lab/public/fonts/${file}`);
    return createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}

// The pre-memo algorithm is deliberately local and never calls the production closure.
function reference(font: ComputeFont, plans: readonly ComputeTextPlan[], seeds: readonly number[]): readonly number[] {
    const selected = new Set<number>();
    for (const plan of plans) {
        if (plan.font !== font) {
            throw new Error("updateComputeTextData: closure plan belongs to a different font");
        }
        for (const index of plan._gsubRoots) {
            selected.add(index);
        }
        for (const index of plan._gsubDependencies) {
            selected.add(index);
        }
    }
    const lookups = new Map<number, readonly ParsedSubtable[]>();
    for (const index of selected) {
        try {
            lookups.set(index, actions.parseLookupActions(font, "GSUB", index).subtables);
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new Error(`updateComputeTextData: closure cannot see GSUB lookup ${index} (${message})`, { cause: error });
        }
    }
    const closure = new Set<number>();
    const queue: number[] = [];
    for (const seed of seeds) {
        requireGlyph(font, seed, "cmap seed");
        if (!closure.has(seed)) {
            closure.add(seed);
            queue.push(seed);
        }
    }
    let steps = 0;
    const stepLimit = font.glyphCount * Math.max(1, selected.size) + 1;
    while (queue.length > 0) {
        if (steps++ > stepLimit) {
            throw new Error("updateComputeTextData: closure did not converge");
        }
        const glyphId = queue.pop() ?? 0;
        for (const [lookupIndex, subtables] of lookups) {
            for (let subtableIndex = 0; subtableIndex < subtables.length; subtableIndex++) {
                for (const output of outputsOf(font, lookupIndex, subtableIndex, subtables[subtableIndex]!, glyphId)) {
                    if (!closure.has(output)) {
                        closure.add(output);
                        queue.push(output);
                    }
                }
            }
        }
    }
    const withComponents = new Set(closure);
    const components = [...closure];
    while (components.length > 0) {
        const glyphId = components.pop() ?? 0;
        let ids: readonly number[] = [];
        if (font.outlineFormat === "truetype") {
            try {
                ids = readComputeGlyfGlyph(font, glyphId).componentGlyphIds;
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                throw new Error(`updateComputeTextData: closure cannot see outline components of glyph ${glyphId} (${message})`, { cause: error });
            }
        }
        for (const component of ids) {
            if (!withComponents.has(component)) {
                withComponents.add(component);
                components.push(component);
            }
        }
    }
    return [...withComponents];
}

function requireGlyph(font: ComputeFont, glyphId: number, witness: string): number {
    if (!Number.isInteger(glyphId) || glyphId < 0 || glyphId >= font.glyphCount) {
        throw new Error(`updateComputeTextData: closure cannot see ${witness} output glyph ${glyphId}`);
    }
    return glyphId;
}

function outputsOf(font: ComputeFont, lookupIndex: number, subtableIndex: number, subtable: ParsedSubtable, glyphId: number): number[] {
    const witness = `GSUB lookup ${lookupIndex} subtable ${subtableIndex}`;
    switch (subtable.kind) {
        case "context":
            return [];
        case "single": {
            const index = coverageIndex(subtable.coverage, glyphId);
            if (index === null) {
                return [];
            }
            if (subtable.format === 1) {
                return [requireGlyph(font, glyphId + subtable.delta, witness)];
            }
            const substitute = subtable.substitutes?.[index];
            if (substitute === undefined) {
                throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
            }
            return [requireGlyph(font, substitute, witness)];
        }
        case "multiple":
        case "alternate":
        case "ligature": {
            const index = coverageIndex(subtable.coverage, glyphId);
            if (index === null) {
                return [];
            }
            const sequence =
                subtable.kind === "multiple"
                    ? subtable.sequences[index]
                    : subtable.kind === "alternate"
                      ? subtable.alternatives[index]
                      : subtable.sets[index]?.map((entry) => entry.glyph);
            if (!sequence) {
                throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
            }
            return sequence.map((output) => requireGlyph(font, output, witness));
        }
        case "unimplemented":
            if (subtable.coverage && coverageIndex(subtable.coverage, glyphId) === null) {
                return [];
            }
            throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId} (${subtable.reason})`);
        default:
            throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
    }
}

describe("text-independent ordered closure memo", () => {
    afterEach(() => vi.restoreAllMocks());

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("matches the unmemoized traversal for many seeds and plan orders: %s", (file) => {
        const font = load(file);
        const plans = ["office fi AV é A\u0326", "Αθήναςϊϋΐ", "МоскваЁІЇЄ", "123 "].map((text) => createComputeTextPlan(font, selectComputeTextRun(text)));
        const repertoire = [..."office fi AV é A\u0326ΑθήναςϊϋΐМоскваЁІЇЄ123 "].map((char) => mapComputeFontScalar(font, char.codePointAt(0)!));
        for (let revision = 0; revision < 48; revision++) {
            const seeds = revision === 0 ? [] : repertoire.filter((_, index) => (index + revision) % (2 + (revision % 9)) === 0);
            const ordered = revision % 3 === 0 ? [...plans].reverse() : revision % 3 === 1 ? [plans[1]!, plans[0]!, plans[2]!] : plans;
            for (const selected of [ordered, [ordered[0]!], [], [...ordered, ordered[0]!]]) {
                const expected = reference(font, selected, seeds);
                expect(closeComputeTextGlyphs(font, selected, seeds).glyphIds).toEqual(expected);
                expect(closeComputeTextGlyphs(font, selected, [...seeds, ...seeds]).glyphIds).toEqual(expected);
                expect(closeComputeTextGlyphs(font, selected, [...seeds].reverse()).glyphIds).toEqual(reference(font, selected, [...seeds].reverse()));
            }
        }
    });

    it("does not conflate the same lookup set in different insertion orders", () => {
        const font = { ...load(), outlineFormat: "cff" as const };
        const plan = createComputeTextPlan(font, selectComputeTextRun("A"));
        vi.spyOn(actions, "parseLookupActions").mockImplementation((_font, table, lookupIndex) => ({
            table,
            lookupIndex,
            lookupType: 1,
            lookupFlag: 0,
            markFilteringSet: null,
            subtables: [{ kind: "single", format: 2, delta: 0, coverage: { format: 1, glyphs: Uint16Array.of(1), ranges: [] }, substitutes: Uint16Array.of(lookupIndex + 2) }],
        }));
        const a = { ...plan, _gsubRoots: [0], _gsubDependencies: [] };
        const b = { ...plan, _gsubRoots: [1], _gsubDependencies: [] };
        for (let repeat = 0; repeat < 3; repeat++) {
            expect(closeComputeTextGlyphs(font, [a, b], [1]).glyphIds).toEqual([1, 2, 3]);
            expect(closeComputeTextGlyphs(font, [b, a], [1]).glyphIds).toEqual([1, 3, 2]);
        }
    });

    it("concatenates every supported output kind in subtable order without dropping duplicates", () => {
        const font = { ...load(), outlineFormat: "cff" as const };
        const plan = { ...createComputeTextPlan(font, selectComputeTextRun("A")), _gsubRoots: [0], _gsubDependencies: [] };
        const coverage = { format: 1 as const, glyphs: Uint16Array.of(1), ranges: [] };
        vi.spyOn(actions, "parseLookupActions").mockReturnValue({
            table: "GSUB",
            lookupIndex: 0,
            lookupType: 1,
            lookupFlag: 0,
            markFilteringSet: null,
            subtables: [
                { kind: "single", format: 1, coverage, delta: 1, substitutes: null },
                { kind: "single", format: 2, coverage, delta: 0, substitutes: Uint16Array.of(3) },
                { kind: "multiple", coverage, sequences: [[3, 4]] },
                { kind: "alternate", coverage, alternatives: [[5, 6, 4]] },
                { kind: "ligature", coverage, sets: [[{ glyph: 7, components: [1] }]] },
                { kind: "context", chaining: false, format: 1, rules: [] },
                { kind: "unimplemented", lookupType: 99, format: 1, reason: "uncovered", coverage: { ...coverage, glyphs: Uint16Array.of(99) } },
            ],
        });
        for (let repeat = 0; repeat < 3; repeat++) {
            expect(closeComputeTextGlyphs(font, [plan], [1]).glyphIds).toEqual([1, 2, 3, 4, 5, 6, 7]);
            expect(closeComputeTextGlyphs(font, [plan], [1]).glyphIds).toEqual(reference(font, [plan], [1]));
        }
    });

    it("retains repeated outline-component errors even after GSUB outputs have been memoized", () => {
        const font = load();
        vi.spyOn(extraction, "readComputeGlyfGlyph").mockImplementation(() => {
            throw new Error("component witness");
        });
        const expected = "updateComputeTextData: closure cannot see outline components of glyph 1 (component witness)";
        expect(() => reference(font, [], [1])).toThrow(expected);
        for (let repeat = 0; repeat < 3; repeat++) {
            expect(() => closeComputeTextGlyphs(font, [], [1])).toThrow(expected);
        }
    });

    it("re-evaluates throwing glyph outputs, preserving first and repeated messages", () => {
        const font = load();
        const plan = { ...createComputeTextPlan(font, selectComputeTextRun("A")), _gsubRoots: [0], _gsubDependencies: [] };
        const coverage = { format: 1 as const, glyphs: Uint16Array.of(1), ranges: [] };
        const lookup = vi.spyOn(actions, "parseLookupActions");
        for (const subtable of [
            { kind: "single", format: 1, coverage, delta: font.glyphCount, substitutes: null },
            { kind: "single", format: 2, coverage, delta: 0, substitutes: null },
            { kind: "multiple", coverage, sequences: [] },
            { kind: "alternate", coverage, alternatives: [] },
            { kind: "ligature", coverage, sets: [] },
            { kind: "unimplemented", coverage, lookupType: 99, format: 1, reason: "test witness" },
        ] as const) {
            lookup.mockReturnValue({ table: "GSUB", lookupIndex: 0, lookupType: 1, lookupFlag: 0, markFilteringSet: null, subtables: [subtable] });
            let expected = "";
            try {
                reference(font, [plan], [1]);
            } catch (error) {
                expected = (error as Error).message;
            }
            expect(expected).not.toBe("");
            for (let repeat = 0; repeat < 3; repeat++) {
                expect(() => closeComputeTextGlyphs(font, [plan], [1])).toThrow(expected);
            }
        }
        let reads = 0;
        lookup.mockReturnValue({
            table: "GSUB",
            lookupIndex: 0,
            lookupType: 1,
            lookupFlag: 0,
            markFilteringSet: null,
            subtables: [
                {
                    kind: "single",
                    format: 2,
                    coverage,
                    delta: 0,
                    get substitutes(): Uint16Array {
                        throw new Error(`throwing read ${++reads}`);
                    },
                },
            ],
        });
        for (let repeat = 1; repeat <= 3; repeat++) {
            expect(() => closeComputeTextGlyphs(font, [plan], [1])).toThrow(`throwing read ${repeat}`);
        }
        lookup.mockImplementation(() => {
            throw new Error("parse witness");
        });
        expect(() => closeComputeTextGlyphs(font, [plan], [-1])).toThrow("updateComputeTextData: closure cannot see GSUB lookup 0 (parse witness)");
        lookup.mockReturnValue({ table: "GSUB", lookupIndex: 0, lookupType: 1, lookupFlag: 0, markFilteringSet: null, subtables: [] });
        for (const seeds of [[-1], [font.glyphCount], [0.5]]) {
            expect(() => closeComputeTextGlyphs(font, [plan], seeds)).toThrow(`updateComputeTextData: closure cannot see cmap seed output glyph ${seeds[0]}`);
        }
        expect(() => closeComputeTextGlyphs(load(), [plan], [])).toThrow("updateComputeTextData: closure plan belongs to a different font");
    });
});
