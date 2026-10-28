import { beforeAll, describe, expect, it } from "vitest";
import { createComputeTextPlan, ensureComputeTextProfile } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { fixture } from "../compute/fixtures";
import { syntheticFont, syntheticFontBytes } from "../compute/synthetic";
import { createComputeFontFromBuffer, ensureComputeFontLayout } from "../../../packages/babylon-lite/src/text/compute-font";
import { coverageGlyphs } from "../../../packages/babylon-lite/src/text/font-parser/opentype-layout";

describe("shared compiler v3", () => {
    beforeAll(async () => ensureComputeTextProfile("arabic"));
    it("normalizes unknown GDEF glyph classes 5 and 11 to class zero rather than aliasing marks", () => {
        const bytes = syntheticFontBytes("c1-mark-base").slice(0);
        const view = new DataView(bytes);
        for (let i = 0; i < view.getUint16(4); i++) {
            const entry = 12 + i * 16;
            if (view.getUint32(entry) === 0x47444546) {
                const origin = view.getUint32(entry + 8);
                const definition = origin + view.getUint16(origin + 4);
                expect(view.getUint16(definition)).toBe(1);
                expect(view.getUint16(definition + 2)).toBe(1);
                view.setUint16(definition + 6, 5);
                view.setUint16(definition + 8, 11);
            }
        }
        const font = createComputeFontFromBuffer(bytes);
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font, selectComputeTextRun("AB")), []);
        expect(prepared.tables[prepared.tables[2]! + 1]! & 0xffff).toBe(0);
        expect(prepared.tables[prepared.tables[2]! + 2]! & 0xffff).toBe(0);
    });
    it("exposes a stage-ready header, four-word roots and multi-word root presence", () => {
        for (const file of ["Inter.ttf", "Roboto-Regular.ttf", "NotoSansHebrew-Regular.ttf", "NotoSansArabic-Regular.ttf"] as const) {
            const face = fixture(file);
            const selection = selectComputeTextRun(file.includes("Arabic") ? "كتب" : file.includes("Hebrew") ? "שלום" : "AB");
            const plan = createComputeTextPlan(face.font, selection);
            const program = compileComputeTextProgram(plan);
            expect(program.words[1]).toBe(3);
            expect(program.words[11]).toBe(0);
            expect(program.words[12]).toBe(plan._stages.length);
            expect(program.words[13]).toBe(Math.ceil(Math.min(program.rootOrder.length, 256) / 32));
            expect(program.words[14]).toBe(0);
            const prepared = prepareComputeTextShapeWords(plan, []);
            const roots = prepared.programWords[8]!;
            const encoded = Array.from({ length: program.rootOrder.length }, (_, i) => prepared.programWords[roots + i * 4]);
            expect(encoded).toEqual(plan._stages.flatMap((stage) => stage.roots.map((root) => (stage.table === "GPOS" ? 0x10000 : 0) | root.lookup)));
            expect(program.rootWitnesses.length).toBe(plan.font.glyphCount * program.words[13]!);
        }
    });

    it("executes nested cardinality and positioning actions, rejecting only genuinely unsupported nesting", () => {
        for (const kind of ["c4-nested-ligature", "c4-nested-multiple", "c5-context-format3", "c6-context-pos", "nested-context"] as const) {
            const face = syntheticFont(kind);
            const program = compileComputeTextProgram(createComputeTextPlan(face.font, selectComputeTextRun("ABC")));
            expect(program.fullyExecutable, kind).toBe(kind !== "nested-context");
            expect(
                program.rootOrder.some((index) => program.lookups[index]!.rules.some((rule) => rule.mode === 2)),
                kind
            ).toBe(kind !== "nested-context");
        }
    });

    it("retains full horizontal pair values and original valueFormat2 continuation", () => {
        const face = syntheticFont("c3-pair-value2");
        const program = compileComputeTextProgram(createComputeTextPlan(face.font, selectComputeTextRun("ABC")));
        const pair = program.lookups.flatMap((lookup) => lookup.rules).find((rule) => rule.action.kind === "class-pair")!;
        const origin = program.words[program.gposDescriptorOffset + 3]!;
        const action = program.words[origin + 6]!;
        expect(pair.mode).toBe(4);
        expect(Array.from(program.words.slice(action + 6, action + 11))).toEqual([19, 23, 29, 0, 1]);
        expect(program.words[program.gposDescriptorOffset + 2]).toBe(0);
    });

    it("packs filtering-set indices and exact GDEF membership and rejects missing sets", () => {
        const face = fixture("NotoSansArabic-Regular.ttf");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("كتب"));
        const packed = prepareComputeTextShapeWords(plan, []);
        const sets = ensureComputeFontLayout(face.font).gdef!.markGlyphSets;
        expect(sets.length).toBeGreaterThan(0);
        const width = packed.tables[7]!;
        expect(width).toBe(Math.ceil(sets.length / 32));
        sets.forEach((coverage, index) => {
            const expected = new Set(coverage ? coverageGlyphs(coverage, face.font.glyphCount) : []);
            for (let glyph = 0; glyph < face.font.glyphCount; glyph++) {
                const bit = packed.tables[packed.tables[6]! + glyph * width + (index >>> 5)]! & (1 << (index % 32));
                expect(bit !== 0).toBe(expected.has(glyph));
            }
        });
        for (const lookup of packed.program.lookups.filter((lookup) => lookup.lookupFlag & 0x10)) {
            const descriptor = packed.program.words[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
            expect(packed.program.words[descriptor + 1]! >>> 16).toBe(lookup.markFilteringSet);
        }
        const bytes = syntheticFontBytes("c2-filter-set").slice(0);
        const view = new DataView(bytes);
        for (let i = 0; i < view.getUint16(4); i++) {
            const entry = 12 + i * 16;
            if (view.getUint32(entry) === 0x47444546) {
                const origin = view.getUint32(entry + 8);
                const markSets = origin + view.getUint16(origin + 12);
                view.setUint32(markSets + 4, 0);
            }
        }
        const invalid = createComputeFontFromBuffer(bytes);
        expect(() => compileComputeTextProgram(createComputeTextPlan(invalid, selectComputeTextRun("ABC")))).toThrow(/mark filtering set 0 is missing/);
    });
});
