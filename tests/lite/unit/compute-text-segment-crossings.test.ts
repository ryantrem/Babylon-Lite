import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { createComputeFontFromBuffer, mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextPlan, ensureComputeTextProfile } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import type { ComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { analyzeComputeTextSegmentation, clearComputeTextSegmentationCache } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { collectComputeTextCrossings } from "../../../packages/babylon-lite/src/text/compute-text-indic-paragraph";

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
const texts = [paragraph.repeat(3), phrases.join(""), extra.repeat(2), "The quick", "AV To", "kerning. The efficient", "noon. To verify", "WA To fi", "quick brown"];

function program(file: string, text: string): ComputeTextProgram {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", file));
    const font = createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    return compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun(text, font)));
}

/** Compact digest of the generic boundary analysis: scalar fields plus safeAfter at every space of the segment-test strings. */
function digest(p: ComputeTextProgram): string {
    const analysis = analyzeComputeTextSegmentation(p);
    if (!analysis) {
        return "null";
    }
    const bits = texts.map((text) => {
        const glyphIds = mapComputeFontScalars(p.plan.font, Uint32Array.from([...text].map((c) => c.codePointAt(0)!)), (_i, error) => {
            throw error;
        });
        let out = "";
        for (let s = 0; s < glyphIds.length; s++) {
            if (glyphIds[s] === analysis.space) {
                out += analysis.safeAfter(glyphIds, s) ? "1" : "0";
            }
        }
        return out;
    });
    return `space=${analysis.space} span=${analysis.span} maxD=${analysis.maxD} crossings=${analysis.crossings} markBaseHit=${analysis.markBaseHit} safe=${bits.join("|")}`;
}

const PROGRAMS: readonly (readonly [file: string, text: string])[] = [
    ["Inter.ttf", "a b"],
    ["Roboto-Regular.ttf", "a b"],
    ["NotoSansGothic-Regular.ttf", "\u{10330} \u{10331}"],
    ["NotoSansSC-Regular-subset.ttf", "\u4f60 \u597d"],
    ["NotoSansHebrew-Regular.ttf", "\u05e9\u05dc\u05d5\u05dd"],
    ["NotoSansArabic-Regular.ttf", "\u0633\u0644\u0627\u0645"],
    ["NotoSansDevanagari-Regular.ttf", "\u0915 \u0916"],
];

/** Captured from the pre-extraction analyzer (fe085671); REQ-COMPAT-1 regression characterization, not shaping expectations. */
const EXPECTED: Record<string, string> = {
    "Inter.ttf":
        "space=1777 span=4 maxD=6 crossings=149 markBaseHit=false safe=111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110|1111111111111111111111111111111111111111111111111111110|1111111111000111110000111111111110001111110000|1|1|11|11|11|1",
    "Roboto-Regular.ttf":
        "space=5 span=3 maxD=1 crossings=1 markBaseHit=false safe=111111111111111111111101111111111011111111111111111111110111111111101111111111111111111111011111111110|1111101111111111111111111111111111111111111101111101110|1111111111111111111111111111111111111111111110|1|0|01|01|01|1",
    "NotoSansGothic-Regular.ttf":
        "space=3 span=2 maxD=1 crossings=0 markBaseHit=false safe=111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110|1111111111111111111111111111111111111111111111111111110|1111111111111111111111111111111111111111111110|1|1|11|11|11|1",
    "NotoSansSC-Regular-subset.ttf": "null",
    "NotoSansHebrew-Regular.ttf": "null",
    "NotoSansArabic-Regular.ttf": "null",
    "NotoSansDevanagari-Regular.ttf": "null",
};

describe("generic segmentation analysis characterization", () => {
    beforeAll(async () => {
        await ensureComputeTextProfile("indic");
        await ensureComputeTextProfile("arabic");
    });
    beforeEach(() => clearComputeTextSegmentationCache());

    it.each(PROGRAMS)("is unchanged for %s", (file, text) => {
        expect(digest(program(file, text))).toBe(EXPECTED[file]);
    });
});

describe("shared crossing collection", () => {
    beforeAll(async () => ensureComputeTextProfile("indic"));

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("agrees with the generic analysis on %s", (file) => {
        const p = program(file, "a b");
        const analysis = analyzeComputeTextSegmentation(p)!;
        const shared = collectComputeTextCrossings(p)!;
        expect(shared.space).toBe(analysis.space);
        expect(shared.crossings.length).toBe(analysis.crossings);
        expect(shared.attached).toBe(analysis.markBaseHit);
        for (const crossing of shared.crossings) {
            expect(crossing.slot).toBeLessThan(crossing.window.length - 1);
            expect(shared.bStar.some((g) => shared.matches(crossing.window[crossing.slot]!, g))).toBe(true);
            expect(crossing.spaceSlots).toBeGreaterThanOrEqual(1);
        }
        expect(shared.closure(shared.space)).toContain(shared.space);
    });

    it("finds only the GPOS lookup 3 crossings on the Noto Sans Devanagari indic program", () => {
        const p = program("NotoSansDevanagari-Regular.ttf", "\u0915 \u0916");
        expect(p.plan._profile).toBe("indic");
        expect(analyzeComputeTextSegmentation(p)).toBeNull();
        const shared = collectComputeTextCrossings(p)!;
        expect(shared).not.toBeNull();
        expect([...shared.bStar]).toEqual([3]);
        expect(shared.space).toBe(3);
        expect(shared.glyphClass(3)).toBe(0);
        expect(shared.skipped(3)).toBe(false);
        expect(shared.attached).toBe(false);
        expect(shared.crossings.map((crossing) => [crossing.lookup.table, crossing.lookup.lookupIndex, crossing.spaceSlots])).toEqual([
            ["GPOS", 3, 1],
            ["GPOS", 3, 1],
        ]);
        expect(collectComputeTextCrossings(p)).not.toBe(shared);
    });
});
