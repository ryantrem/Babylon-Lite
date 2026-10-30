import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fixture } from "../compute/fixtures";
import { mapCmapScalar } from "../../../packages/babylon-lite/src/text/font-parser/cmap";
import {
    INDIC_DENSE,
    INDIC_DENSE_BASE,
    INDIC_SPARSE,
    INDIC_DECOMPOSITIONS,
    INDIC_COMPOSITIONS,
    INDIC_VOWEL_CONSTRAINTS,
    INDIC_TABLE_VERSION,
    INDIC_GRAPHEME,
} from "../../../packages/babylon-lite/src/text/compute-text-indic-unicode";

function properties(cp: number): number {
    if (cp >= 0x900 && cp <= 0x97f) {
        return INDIC_DENSE[cp - 0x900]!;
    }
    for (let i = 0; i < INDIC_SPARSE.length; i += 2) {
        if (INDIC_SPARSE[i] === cp) {
            return INDIC_SPARSE[i + 1]!;
        }
    }
    return 0;
}
const rejected = (cp: number): boolean =>
    cp === 13 || cp === 0xad || cp === 0x200b || cp === 0x25cc || (cp >= 0x1cd0 && cp <= 0x1cff) || (cp >= 0xa8e0 && cp <= 0xa8ff) || (cp >= 0x11b00 && cp <= 0x11b5f);

const GCB_EXTEND = 1,
    GCB_ZWJ = 2,
    GCB_SPACING_MARK = 3,
    GCB_PREPEND = 4;
const INCB_CONSONANT = 1,
    INCB_EXTEND = 2,
    INCB_LINKER = 3;
function grapheme(cp: number): { gcb: number; incb: number } {
    let lo = 0,
        hi = INDIC_GRAPHEME.length - 1;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (INDIC_GRAPHEME[mid]! >>> 5 <= cp) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const code = INDIC_GRAPHEME[lo]! & 31;
    return { gcb: code >> 2, incb: code & 3 };
}
const graphemes = (text: string): number => [...new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(text)].length;

describe("generated pinned Indic scalar properties", () => {
    it("has the exact dense region and sorted sparse pairs, with generator identity and no import side effects", async () => {
        expect(INDIC_DENSE_BASE).toBe(0x900);
        expect(INDIC_DENSE).toHaveLength(128);
        expect(INDIC_SPARSE.length % 2).toBe(0);
        let previous = -1;
        for (let i = 0; i < INDIC_SPARSE.length; i += 2) {
            expect(INDIC_SPARSE[i]).toBeGreaterThan(previous);
            previous = INDIC_SPARSE[i]!;
        }
        expect(INDIC_TABLE_VERSION).toMatch(/^[a-f0-9]{64}$/);
        const source = readFileSync("packages/babylon-lite/src/text/compute-text-indic-unicode.ts", "utf8");
        expect(source).not.toMatch(/new (?:Map|Set|WeakMap)|register\w*\(/);
        const again = await import("../../../packages/babylon-lite/src/text/compute-text-indic-unicode");
        expect(again.INDIC_DENSE).toBe(INDIC_DENSE);
        expect(source).toContain("Unicode License v3");
        expect(source).toContain("43a778492219032b9d747112e412dec2201d2667");
    });

    it("classifies every mapped scalar as admissible or specifically rejected, and admits all 122 strong scalars", () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        let mapped = 0;
        for (let cp = 0; cp <= 0x11b5f; cp++) {
            if (mapCmapScalar(face.font._cmap, cp) === 0) {
                continue;
            }
            mapped++;
            expect((properties(cp) >>> 28) & 1, `U+${cp.toString(16)}`).toBe(rejected(cp) ? 0 : 1);
        }
        expect(mapped).toBeGreaterThan(250);
        const data = JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: { id: string; text: string }[] };
        const singles = data.cases.filter((entry) => entry.id.startsWith("scalar-"));
        expect(singles).toHaveLength(122);
        singles.forEach((entry) => expect(properties(entry.text.codePointAt(0)!) & (1 << 28)).not.toBe(0));
        expect(properties(0x200c) & (1 << 27)).not.toBe(0);
        expect(properties(0x200d) & (1 << 26)).not.toBe(0);
        expect(properties(0x25cc) & 255).toBe(11);
        expect(properties(0x25cc) & (1 << 28)).toBe(0);
    });

    it("encodes exactly the canonical Indic exceptions and three allowed compositions", () => {
        expect(Array.from(INDIC_COMPOSITIONS)).toEqual([0x928, 0x93c, 0x929, 0x930, 0x93c, 0x931, 0x933, 0x93c, 0x934]);
        expect(INDIC_DECOMPOSITIONS).toHaveLength(30);
        expect(Array.from(INDIC_DECOMPOSITIONS).filter((_, i) => i % 3 === 0)).toEqual([0x929, 0x934, 0x958, 0x959, 0x95a, 0x95b, 0x95c, 0x95d, 0x95e, 0x95f]);
        expect(properties(0x931) & (1 << 29)).toBe(0);
        expect(properties(0x958) & (1 << 29)).not.toBe(0);
        expect(INDIC_VOWEL_CONSTRAINTS).toHaveLength(88);
        expect(Array.from(INDIC_VOWEL_CONSTRAINTS.slice(-4))).toEqual([0x930, 0x94d, 0x907, 2]);
        expect(INDIC_VOWEL_CONSTRAINTS.includes(0x20)).toBe(false);
        expect(properties(0x20) & 255).toBe(0);
        expect((properties(0x20) >>> 16) & 255).toBe(0);
    });

    it("the 34 independently observed dotted-circle scalars have broken-starter categories", () => {
        // hb-deva-singles-level1.txt: 30 strong scalars plus the four Inherited
        // signs. The latter still reject when alone through public selection.
        const broken = [
            0x900, 0x901, 0x902, 0x903, 0x93a, 0x93b, 0x93c, 0x93e, 0x93f, 0x940, 0x941, 0x942, 0x943, 0x944, 0x945, 0x946, 0x947, 0x948, 0x949, 0x94a, 0x94b, 0x94c, 0x94d, 0x94e,
            0x94f, 0x951, 0x952, 0x953, 0x954, 0x955, 0x956, 0x957, 0x962, 0x963,
        ];
        expect(broken).toHaveLength(34);
        broken.forEach((cp) => expect([3, 4, 7, 8, 9]).toContain(properties(cp) & 255));
        expect(properties(0x915) & 255).toBe(1);
        expect(properties(0x930) & 255).toBe(15);
        expect((properties(0x93f) >>> 8) & 255).toBe(2);
    });

    it("grapheme table classes every admitted scalar consistently with the Unicode 16 Intl.Segmenter oracle", () => {
        expect(process.versions.unicode).toBe("16.0");
        expect(INDIC_GRAPHEME[0]).toBe(0);
        for (let i = 1; i < INDIC_GRAPHEME.length; i++) {
            expect(INDIC_GRAPHEME[i]! >>> 5).toBeGreaterThan(INDIC_GRAPHEME[i - 1]! >>> 5);
            expect(INDIC_GRAPHEME[i]! & 31).not.toBe(INDIC_GRAPHEME[i - 1]! & 31);
        }
        const ka = "\u0915",
            virama = "\u094d";
        let admitted = 0;
        const seen = new Set<string>();
        for (let cp = 0; cp <= 0x11b5f; cp++) {
            if ((properties(cp) & (1 << 28)) === 0) {
                expect(grapheme(cp), `U+${cp.toString(16)}`).toEqual({ gcb: 0, incb: 0 });
                continue;
            }
            admitted++;
            const c = String.fromCodePoint(cp);
            const { gcb, incb } = grapheme(cp);
            seen.add(`${gcb}/${incb}`);
            const joins = gcb === GCB_EXTEND || gcb === GCB_ZWJ || gcb === GCB_SPACING_MARK;
            const chain = incb === INCB_EXTEND || incb === INCB_LINKER;
            const label = `U+${cp.toString(16)} gcb ${gcb} incb ${incb}`;
            expect(c, label).not.toMatch(/\p{Extended_Pictographic}|\p{Regional_Indicator}/u);
            expect(graphemes(ka + c), label).toBe(joins ? 1 : 2);
            expect(graphemes(c + ka), label).toBe(gcb === GCB_PREPEND ? 1 : 2);
            expect(graphemes(ka + virama + c), label).toBe(joins || incb === INCB_CONSONANT ? 1 : 2);
            expect(graphemes(ka + virama + c + ka), label).toBe(chain ? 1 : joins || incb === INCB_CONSONANT || gcb === GCB_PREPEND ? 2 : 3);
            expect(graphemes(ka + c + ka), label).toBe(incb === INCB_LINKER ? 1 : joins || gcb === GCB_PREPEND ? 2 : 3);
            if (chain) {
                expect(joins, label).toBe(true);
            }
        }
        expect(admitted).toBe(194);
        expect([...seen].sort()).toEqual(["0/0", "0/1", "1/0", "1/2", "1/3", "2/2", "3/0"]);
        expect(grapheme(0x200d)).toEqual({ gcb: GCB_ZWJ, incb: INCB_EXTEND });
        expect(grapheme(0x200c)).toEqual({ gcb: GCB_EXTEND, incb: 0 });
        expect(grapheme(0x20f0).gcb).toBe(GCB_EXTEND);
        expect(grapheme(0x94d)).toEqual({ gcb: GCB_EXTEND, incb: INCB_LINKER });
        expect(grapheme(0x915).incb).toBe(INCB_CONSONANT);
        expect(grapheme(0x1cd0)).toEqual({ gcb: 0, incb: 0 });
        expect(grapheme(0x25cc)).toEqual({ gcb: 0, incb: 0 });
    });
});
