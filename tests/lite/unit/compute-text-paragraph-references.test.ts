import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { indicProperties } from "../../../packages/babylon-lite/src/text/compute-text-indic-plan";

/** CTF-03 paragraph references (scripts/generate-devanagari-paragraph-references.mjs). Consumes the
 * checked-in JSON only; HarfBuzz is never invoked here. */
type ShapedRecord = [glyph: number, cluster: number, xAdvance: number, yAdvance: number, xOffset: number, yOffset: number];
interface ParagraphSplit {
    s: number;
    independent: boolean;
    left?: ShapedRecord[];
    right?: ShapedRecord[];
}
interface ParagraphCase {
    id: string;
    tags: string[];
    script: "Deva" | "DFLT";
    text: string;
    note?: string;
    length: number;
    candidates: number[];
    proven: number[];
    whole: ShapedRecord[];
    splitsSampled: boolean;
    splits: ParagraphSplit[];
    lineDistance: number;
    lines?: Record<string, ShapedRecord[]>;
    lineSummary?: { pairs: number; consistent: number };
}
interface ParagraphReferences {
    identity: { font: { path: string; sha256: string }; corpusSha256: string; generatorSha256: string; harfbuzz: { version: string } };
    wordCap: number;
    lineInnerSpaces: number;
    longRow: number;
    cases: ParagraphCase[];
}

const references = JSON.parse(readFileSync("tests/lite/compute/references/deva-paragraph-harfbuzz-11.0.1.json", "utf8")) as ParagraphReferences;
const sha256 = (path: string): string => createHash("sha256").update(readFileSync(path)).digest("hex");
const deva = references.cases.filter((c) => c.script === "Deva");
const isSpace = (scalar: string | undefined): boolean => scalar === " ";

function sameRecords(a: readonly ShapedRecord[], b: readonly ShapedRecord[]): boolean {
    return a.length === b.length && a.every((record, i) => record.every((value, k) => value === b[i]![k]));
}

function slice(whole: readonly ShapedRecord[], a: number, b: number): ShapedRecord[] {
    return whole.filter(([, c]) => c >= a && c < b).map(([g, c, ...rest]) => [g, c - a, ...rest] as ShapedRecord);
}

function boundaries(c: ParagraphCase): number[] {
    const breaks = c.script === "DFLT" ? c.candidates : c.proven;
    return [...new Set([0, ...breaks.map((s) => s + 1), c.length])].sort((x, y) => x - y);
}

function requiredLines(c: ParagraphCase): string[] {
    const b = boundaries(c);
    const keys: string[] = [];
    for (let i = 0; i < b.length; i++) {
        for (let j = i + 1; j < b.length && j - i <= c.lineDistance; j++) {
            keys.push(`${b[i]}:${b[j]}`);
        }
    }
    return keys;
}

/** Scalar indices that start a UAX #29 extended grapheme cluster (ICU, Unicode 16.0). */
function graphemeStarts(text: string): Set<number> {
    const starts = new Set<number>();
    const utf16ToScalar = new Map<number, number>();
    let scalar = 0;
    for (let i = 0; i < text.length;) {
        utf16ToScalar.set(i, scalar++);
        i += text.codePointAt(i)! > 0xffff ? 2 : 1;
    }
    utf16ToScalar.set(text.length, scalar);
    for (const { index } of new Intl.Segmenter("und", { granularity: "grapheme" }).segment(text)) {
        starts.add(utf16ToScalar.get(index)!);
    }
    starts.add(scalar);
    return starts;
}

/** Independent re-derivation of the declared proven set: a space s (not followed by a space) is proven when a
 * grapheme boundary exists at s+1 and the maximal non-space words on both sides have <= wordCap scalars. */
function predictedProven(c: ParagraphCase): number[] {
    const scalars = [...c.text];
    const starts = graphemeStarts(c.text);
    const proven: number[] = [];
    for (const s of c.candidates) {
        let left = 0;
        while (s - 1 - left >= 0 && !isSpace(scalars[s - 1 - left])) {
            left++;
        }
        let right = 0;
        while (s + 1 + right < scalars.length && !isSpace(scalars[s + 1 + right])) {
            right++;
        }
        if (starts.has(s + 1) && left <= references.wordCap && right <= references.wordCap) {
            proven.push(s);
        }
    }
    return proven;
}

describe("compute text Devanagari paragraph references", () => {
    it("pins the corpus, generator, font and HarfBuzz identity", () => {
        expect(references.identity.harfbuzz.version).toBe("11.0.1");
        expect(references.identity.corpusSha256).toBe(sha256("tests/lite/compute/references/devanagari-paragraph-corpus.mjs"));
        expect(references.identity.generatorSha256).toBe(sha256("scripts/generate-devanagari-paragraph-references.mjs"));
        expect(references.identity.font.sha256).toBe(sha256(references.identity.font.path));
    });

    it("stores well-formed records, candidates and line tables", () => {
        expect(new Set(references.cases.map((c) => c.id)).size).toBe(references.cases.length);
        for (const c of references.cases) {
            const scalars = [...c.text];
            expect(c.length, c.id).toBe(scalars.length);
            expect(c.candidates, c.id).toEqual(scalars.flatMap((ch, s) => (isSpace(ch) && s + 1 < scalars.length && !isSpace(scalars[s + 1]) ? [s] : [])));
            for (const s of c.proven) {
                expect(c.candidates, `${c.id} proven ${s}`).toContain(s);
            }
            const checkRecords = (records: readonly ShapedRecord[], limit: number, label: string): void => {
                let previous = 0;
                for (const record of records) {
                    expect(record.length, label).toBe(6);
                    expect(record.every(Number.isInteger), label).toBe(true);
                    expect(record[1] >= previous && record[1] < limit, `${label} cluster ${record[1]}`).toBe(true);
                    previous = record[1];
                }
            };
            checkRecords(c.whole, c.length, `${c.id} whole`);
            if (c.script === "Deva" && !c.tags.includes("reference-only")) {
                expect(
                    c.whole.every(([g]) => g !== 0),
                    `${c.id} .notdef`
                ).toBe(true);
            }
            for (const split of c.splits) {
                expect(c.candidates, `${c.id} split ${split.s}`).toContain(split.s);
                if (split.left && split.right) {
                    checkRecords(split.left, split.s + 1, `${c.id} left ${split.s}`);
                    checkRecords(split.right, c.length - split.s - 1, `${c.id} right ${split.s}`);
                }
            }
            if (c.splitsSampled) {
                expect(c.length, c.id).toBeGreaterThan(references.longRow);
                expect(c.splits.length, c.id).toBeLessThanOrEqual(24);
                expect(c.splits[0]!.s, c.id).toBe(c.candidates[0]);
                expect(c.splits.at(-1)!.s, c.id).toBe(c.candidates.at(-1));
            } else {
                expect(
                    c.splits.map((split) => split.s),
                    c.id
                ).toEqual(c.candidates);
            }
            expect(c.lineDistance, c.id).toBe(c.length > references.longRow ? 3 : references.lineInnerSpaces + 1);
            const keys = requiredLines(c);
            if (c.lines) {
                expect(Object.keys(c.lines).sort(), c.id).toEqual([...keys].sort());
                for (const key of keys) {
                    const [a, b] = key.split(":").map(Number) as [number, number];
                    checkRecords(c.lines[key]!, b - a, `${c.id} line ${key}`);
                }
            } else {
                expect(c.lineSummary?.pairs, c.id).toBe(keys.length);
            }
        }
    });

    it("declares exactly the proven spaces the UAX #29 + word-cap rule predicts (drift guard)", () => {
        // The prediction depends on the ICU grapheme rules of the running Node.
        expect(process.versions.unicode).toBe("16.0");
        for (const c of deva) {
            expect(c.proven, c.id).toEqual(predictedProven(c));
        }
        expect(references.cases.find((c) => c.script === "DFLT")?.proven).toEqual([]);
    });

    it("G1: every declared-proven space is HarfBuzz-independent", () => {
        for (const c of deva) {
            for (const split of c.splits) {
                let independent = split.independent;
                if (split.left && split.right) {
                    const joined = [...split.left, ...split.right.map(([g, cl, ...rest]) => [g, cl + split.s + 1, ...rest] as ShapedRecord)];
                    independent = sameRecords(joined, c.whole);
                    expect(split.independent, `${c.id} stored independence ${split.s}`).toBe(independent);
                }
                if (c.proven.includes(split.s)) {
                    expect(independent, `${c.id} proven space ${split.s}`).toBe(true);
                }
            }
        }
    });

    it("REQ-PROOF-3: every candidate space in plain rows is proven", () => {
        for (const c of deva.filter((row) => row.tags.includes("plain"))) {
            expect(c.proven, c.id).toEqual(c.candidates);
        }
    });

    it("line slices between proven boundaries equal the whole-paragraph records", () => {
        for (const c of deva) {
            if (c.lineSummary) {
                expect(c.lineSummary.consistent, c.id).toBe(c.lineSummary.pairs);
                continue;
            }
            for (const [key, records] of Object.entries(c.lines!)) {
                const [a, b] = key.split(":").map(Number) as [number, number];
                expect(sameRecords(records, slice(c.whole, a, b)), `${c.id} line ${key}`).toBe(true);
            }
        }
    });

    it("multiseam-default needs at least two default-budget seams", () => {
        const row = references.cases.find((c) => c.id === "multiseam-default")!;
        expect(row.whole.length).toBeGreaterThan(2 * 16384);
        expect(row.proven).toEqual(row.candidates);
    });

    it("keeps supported Devanagari rows inside the validateIndicRun repertoire", () => {
        for (const c of deva.filter((row) => !row.tags.includes("reference-only"))) {
            for (const scalar of c.text) {
                const cp = scalar.codePointAt(0)!;
                expect(indicProperties(cp) & 0x10000000, `${c.id} U+${cp.toString(16)}`).not.toBe(0);
            }
        }
        const vedic = references.cases.find((c) => c.id === "vedic-crossing")!;
        expect([...vedic.text].some((scalar) => (indicProperties(scalar.codePointAt(0)!) & 0x10000000) === 0)).toBe(true);
    });

    it("reports independent but unproven spaces (conservatism, informational)", () => {
        const conservative = references.cases.flatMap((c) => c.splits.filter((split) => split.independent && !c.proven.includes(split.s)).map((split) => `${c.id}@${split.s}`));
        console.info(`independent but unproven spaces (${conservative.length}): ${conservative.join(", ")}`);
        expect(conservative.length).toBeGreaterThanOrEqual(0);
    });
});
