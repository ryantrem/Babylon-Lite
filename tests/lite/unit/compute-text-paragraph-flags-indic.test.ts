import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { indicGraphemeChainEnd, indicParagraphFlags } from "../../../packages/babylon-lite/src/text/compute-text-indic-paragraph";
import { computeTextNoBreakBefore } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import type { ComputeTextSpaceProof } from "../../../packages/babylon-lite/src/text/compute-text-segment";

/** CTF-03 indic paragraph flags. Bit 4 is checked against the Unicode 16 `Intl.Segmenter` grapheme oracle. */
const UNAVAILABLE = "updateComputeTextParagraphs: shaping-safe space analysis is unavailable for a paragraph with break candidates";
const always: ComputeTextSpaceProof = { safeAfter: () => true };
const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });

const toScalars = (text: string): Uint32Array => Uint32Array.from([...text].map((c) => c.codePointAt(0)!));
const fromScalars = (scalars: Uint32Array): string => String.fromCodePoint(...scalars);

function flags(scalars: Uint32Array, proof: ComputeTextSpaceProof | null = always, lo = 0, hi = scalars.length): Uint8Array {
    const out = new Uint8Array(hi - lo);
    indicParagraphFlags(scalars, new Uint32Array(scalars.length), proof, lo, hi, out, 0, UNAVAILABLE);
    return out;
}

/** Scalar indexes that start an extended grapheme cluster according to the oracle. */
function oracleStarts(scalars: Uint32Array): Set<number> {
    const text = fromScalars(scalars);
    const starts = new Set<number>();
    let scalar = 0,
        unit = 0;
    for (const segment of segmenter.segment(text)) {
        while (unit < segment.index) {
            unit += text.codePointAt(unit)! > 0xffff ? 2 : 1;
            scalar++;
        }
        starts.add(scalar);
    }
    return starts;
}

function expectOracle(scalars: Uint32Array, label: string): void {
    const out = flags(scalars);
    const starts = oracleStarts(scalars);
    for (let i = 0; i < scalars.length; i++) {
        if (scalars[i] === 0x20) {
            expect(out[i]! & 4, `${label} space ${i}`).toBe(0);
        } else {
            expect(out[i], `${label} @${i} U+${scalars[i]!.toString(16)}`).toBe(starts.has(i) ? 0 : 4);
        }
    }
}

function prng(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

const CONSONANTS = [...Array.from({ length: 0x939 - 0x915 + 1 }, (_, i) => 0x915 + i), 0x958, 0x95f, 0x978, 0x97f];
const ALPHABET = [
    ...CONSONANTS,
    ...CONSONANTS,
    0x94d,
    0x94d,
    0x94d,
    0x93c,
    0x93e,
    0x93f,
    0x940,
    0x941,
    0x947,
    0x948,
    0x94b,
    0x94c,
    0x902,
    0x901,
    0x903,
    0x900,
    0x951,
    0x905,
    0x906,
    0x200d,
    0x200c,
    0x966,
    0x967,
    0x31,
    0x964,
    0x965,
    0xa0,
    0x20,
    0x20,
    0x20,
    0x2d,
    0x20b9,
    0x20f0,
    0x93a,
    0x93b,
    0x94e,
    0x94f,
];

function randomScalars(random: () => number, length: number): Uint32Array {
    return Uint32Array.from({ length }, () => ALPHABET[Math.floor(random() * ALPHABET.length)]!);
}

interface ParagraphCase {
    id: string;
    tags: string[];
    text: string;
}
const corpus = (JSON.parse(readFileSync("tests/lite/compute/references/deva-paragraph-harfbuzz-11.0.1.json", "utf8")) as { cases: ParagraphCase[] }).cases.filter(
    (entry) => !entry.tags.includes("reference-only")
);

describe("indic paragraph grapheme flags", () => {
    it("runs under the Unicode 16 oracle", () => {
        expect(process.versions.unicode).toBe("16.0");
        expect(corpus.length).toBeGreaterThan(20);
    });

    it("bit 4 equals 'not an extended grapheme boundary' for every corpus row and 2000 seeded random strings", () => {
        corpus.forEach((entry) => expectOracle(toScalars(entry.text), entry.id));
        const random = prng(0x5eed03);
        for (let n = 0; n < 2000; n++) {
            const scalars = randomScalars(random, 1 + Math.floor(random() * 40));
            expectOracle(scalars, `random ${n} ${[...scalars].map((cp) => cp.toString(16)).join(" ")}`);
        }
    });

    it("keeps the स्त conjunct whole where the baseline mark rule allows a break before त", () => {
        const scalars = toScalars("नमस्ते");
        const ta = [...scalars].indexOf(0x924);
        expect(scalars[ta - 1]).toBe(0x94d);
        expect(flags(scalars)[ta]).toBe(4);
        expect(computeTextNoBreakBefore(0x924)).toBe(0);
        expect(oracleStarts(scalars).has(ta)).toBe(false);
    });

    it("keeps joiners attached to the preceding virama and never marks them breakable", () => {
        for (const joiner of ["\u200D", "\u200C"]) {
            const scalars = toScalars(`क्${joiner}ष`);
            const out = flags(scalars);
            expect(out[2]).toBe(4);
            expect(out[3]).toBe(oracleStarts(scalars).has(3) ? 0 : 4);
            out.forEach((flag) => expect(flag & 2).toBe(0));
        }
    });

    it("never gives non-space break opportunities to NBSP, danda, rupee or dashes even with a permissive proof", () => {
        const scalars = toScalars("क\u00A0ख।ग॥घ₹5-च–छ—ज");
        flags(scalars).forEach((flag, i) => expect(flag & 2, `@${i}`).toBe(0));
    });

    it("passes scalars to safeAfter for spaces before non-starters and respects its answer", () => {
        for (const text of [" \u093F", " \u094D", " \u200D", " \u20F0"]) {
            const scalars = toScalars(`क${text}`);
            const calls: [number, Uint32Array | undefined][] = [];
            for (const answer of [true, false]) {
                const proof: ComputeTextSpaceProof = {
                    safeAfter: (_glyphs, index, seen) => {
                        calls.push([index, seen]);
                        return answer;
                    },
                };
                expect(flags(scalars, proof)[1]).toBe(answer ? 3 : 1);
            }
            expect(calls.map(([index]) => index)).toEqual([1, 1]);
            calls.forEach(([, seen]) => expect(seen).toBe(scalars));
            expect(flags(scalars)[2]).toBe(4);
        }
    });

    it("throws the caller's message for an unproven candidate space and leaves trailing or doubled spaces unconsulted", () => {
        expect(() => flags(toScalars("क ख"), null)).toThrow(new Error(UNAVAILABLE));
        const out = new Uint8Array(2);
        expect(() => indicParagraphFlags(toScalars("क ख"), new Uint32Array(3), null, 0, 3, new Uint8Array(3), 0, "custom message")).toThrow(new Error("custom message"));
        let consulted = 0;
        const counting: ComputeTextSpaceProof = {
            safeAfter: () => {
                consulted++;
                return true;
            },
        };
        expect(Array.from(flags(toScalars("क  "), counting))).toEqual([0, 1, 1]);
        expect(Array.from(flags(toScalars("क  "), null))).toEqual([0, 1, 1]);
        expect(consulted).toBe(0);
        indicParagraphFlags(toScalars("क  ख"), new Uint32Array(4), null, 0, 2, out, 0, UNAVAILABLE);
        expect(Array.from(out)).toEqual([0, 1]);
        expect(Array.from(flags(toScalars("क  ख"), counting))).toEqual([0, 1, 3, 0]);
        expect(consulted).toBe(1);
    });

    it("writes sub-ranges equal to the full-range output, including inside conjunct chains longer than 66 scalars", () => {
        const random = prng(0xab5);
        const chain = toScalars(`अ ${"क्".repeat(80)}क ${"स्".repeat(40)}त`);
        const samples = [chain, ...Array.from({ length: 200 }, () => randomScalars(random, 1 + Math.floor(random() * 120)))];
        for (const scalars of samples) {
            const full = flags(scalars);
            for (let n = 0; n < 20; n++) {
                const lo = Math.floor(random() * (scalars.length + 1));
                const hi = lo + Math.floor(random() * (scalars.length - lo + 1));
                const out = new Uint8Array(hi - lo + 3).fill(9);
                indicParagraphFlags(scalars, new Uint32Array(scalars.length), always, lo, hi, out, 3, UNAVAILABLE);
                expect(Array.from(out.subarray(0, 3))).toEqual([9, 9, 9]);
                expect(Array.from(out.subarray(3)), `${lo}..${hi}`).toEqual(Array.from(full.subarray(lo, hi)));
            }
        }
    });

    it("chain end bounds left dependence: mutating scalars before h - 1 never changes bit 4 at h or later", () => {
        const random = prng(0xc4a1);
        for (let n = 0; n < 300; n++) {
            const scalars = randomScalars(random, 2 + Math.floor(random() * 60));
            const hi = Math.floor(random() * (scalars.length + 1));
            const end = indicGraphemeChainEnd(scalars, hi);
            expect(end).toBeGreaterThanOrEqual(hi);
            expect(end).toBeLessThanOrEqual(scalars.length);
            const before = flags(scalars);
            for (let m = 0; m < 8 && end > 1; m++) {
                const mutated = scalars.slice();
                mutated[Math.floor(random() * (end - 1))] = ALPHABET[Math.floor(random() * ALPHABET.length)]!;
                const after = flags(mutated);
                for (let i = end; i < scalars.length; i++) {
                    expect(after[i]! & 4, `${n}.${m} @${i}`).toBe(before[i]! & 4);
                }
            }
        }
        const chain = toScalars(`${"क्".repeat(70)}क ख`);
        expect(indicGraphemeChainEnd(chain, 3)).toBe(142);
        expect(indicGraphemeChainEnd(chain, 0)).toBe(0);
        expect(indicGraphemeChainEnd(chain, chain.length)).toBe(chain.length);
    });

    it("conservatively forbids boundaries after an unmodeled scalar until a resetting admitted scalar", () => {
        // U+1CD0 (Vedic tone, InCB Extend) and U+1F600 (Extended_Pictographic) are rejected by validateIndicRun.
        for (const text of ["क्\u1CD0त", "क\u1CD0\u1CD0ख ग", "\u{1F600}\u200D\u{1F600}क", "क \u1CD0ख", "\u1CD0कि"]) {
            const scalars = toScalars(text);
            const out = flags(scalars);
            const starts = oracleStarts(scalars);
            for (let i = 0; i < scalars.length; i++) {
                if (scalars[i] !== 0x20 && !starts.has(i)) {
                    expect(out[i], `${text} @${i}`).toBe(4);
                }
            }
            expect(out[0]).toBe(0);
        }
        expect(Array.from(flags(toScalars("क्\u1CD0तक ख")))).toEqual([0, 4, 4, 4, 4, 3, 0]);
        expect(Array.from(flags(toScalars("क्\u1CD0त, ख")))).toEqual([0, 4, 4, 4, 4, 3, 0]);
        expect(indicGraphemeChainEnd(toScalars("क\u1CD0, ख"), 1)).toBe(3);
    });

    it("has no import side effects", () => {
        const source = readFileSync("packages/babylon-lite/src/text/compute-text-indic-paragraph.ts", "utf8");
        expect(source).not.toMatch(/new (?:Map|Set|WeakMap)|register\w*\(|Intl\./);
        expect(source).not.toMatch(/^(?:let|var) /m);
    });
});
