import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { fixture } from "../compute/fixtures";
import { createComputeTextPlan, ensureComputeTextProfile, profileSupport } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import type { ComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import type { ComputeGlyphPredicate, ComputeProgramLookup, ComputeProgramRule, ComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { analyzeComputeTextSegmentation, computeTextParagraphSpaceProof } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import {
    collectComputeTextCrossings,
    indicParagraphFlags,
    indicParagraphProofReason,
    proveIndicParagraphSpaces,
} from "../../../packages/babylon-lite/src/text/compute-text-indic-paragraph";
import { safeComputeTextSourceBoundary } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-chunks";
import { INDIC_SYLLABLE_MACHINE } from "../../../packages/babylon-lite/src/text/compute-text-indic-unicode";

/** CTF-03 barrier proof (architecture §4): availability obligations O1–O7 and per-space D1–D3 on NotoSansDevanagari. */
interface Split {
    readonly s: number;
    readonly independent: boolean;
}
interface Row {
    readonly id: string;
    readonly script: string;
    readonly tags: readonly string[];
    readonly text: string;
    readonly proven: readonly number[];
    readonly splits: Readonly<Record<string, Split>>;
}
const references = JSON.parse(readFileSync(new URL("../compute/references/deva-paragraph-harfbuzz-11.0.1.json", import.meta.url), "utf8")) as { cases: Row[] };
const font = () => fixture("NotoSansDevanagari-Regular.ttf").font;
const scalarsOf = (text: string): Uint32Array => Uint32Array.from([...text].map((c) => c.codePointAt(0)!));
const glyphsOf = (scalars: Uint32Array): Uint32Array =>
    mapComputeFontScalars(font(), scalars, (_i, error) => {
        throw error;
    });
const nominal = (cp: number): number => glyphsOf(Uint32Array.of(cp))[0]!;

let program: ComputeTextProgram;

function provenSpaces(proof: { safeAfter: (g: Uint32Array, s: number, scalars?: Uint32Array) => boolean }, text: string): number[] {
    const scalars = scalarsOf(text);
    const glyphs = glyphsOf(scalars);
    const proven: number[] = [];
    for (let s = 0; s < scalars.length; s++) {
        if (scalars[s] === 0x20 && proof.safeAfter(glyphs, s, scalars)) {
            proven.push(s);
        }
    }
    return proven;
}

function withPlanData(
    source: ComputeTextProgram,
    patch: (data: NonNullable<ComputeTextPlan["_profileData"]>) => Partial<NonNullable<ComputeTextPlan["_profileData"]>>
): ComputeTextProgram {
    const data = source.plan._profileData!;
    return { ...source, plan: { ...source.plan, _profileData: { ...data, ...patch(data) } } };
}

function withTables(source: ComputeTextProgram, mutate: (words: Uint32Array) => void): ComputeTextProgram {
    return withPlanData(source, (data) => {
        const words = data.tables().slice();
        mutate(words);
        return { tables: () => words };
    });
}

function withRule(source: ComputeTextProgram, table: "GSUB" | "GPOS", make: (template: ComputeProgramRule) => ComputeProgramRule): ComputeTextProgram {
    const index = source.lookups.findIndex((lookup) => lookup.table === table && lookup.rules.some((rule) => rule.action.kind === "context"));
    const lookup = source.lookups[index]!;
    const rule = make(lookup.rules.find((entry) => entry.action.kind === "context")!);
    const lookups: ComputeProgramLookup[] = [...source.lookups];
    lookups[index] = { ...lookup, rules: [...lookup.rules, rule] };
    return { ...source, lookups };
}

const glyphs = (...ids: number[]): ComputeGlyphPredicate => ({ kind: "glyphs", ids, classId: 0, classDef: null, coverage: null });

/** Decodes one category-X transition of the packed Ragel scanner exactly like the test walker. */
function onX(machine: Uint32Array, state: number): { trans: number; target: number; action: number } {
    const data = (slot: number, index: number) => machine[machine[slot]! + index]!;
    const lo = data(1, state * 2),
        hi = data(1, state * 2 + 1),
        span = data(2, state);
    const trans = data(4, data(3, state) + (span > 0 && 0 >= lo && 0 <= hi ? 0 - lo : span));
    return { trans, target: data(5, trans), action: data(6, trans) };
}

describe("CTF-03 indic paragraph space proof", () => {
    beforeAll(async () => {
        await ensureComputeTextProfile("indic");
        program = compileComputeTextProgram(createComputeTextPlan(font(), selectComputeTextRun("क")));
    });

    it("is available on Noto with B* = {space} and only the GPOS lookup 3 Vedic crossings", () => {
        expect(indicParagraphProofReason(program)).toBeNull();
        expect(proveIndicParagraphSpaces(program)).not.toBeNull();
        const crossings = collectComputeTextCrossings(program)!;
        expect(crossings.space).toBe(3);
        expect([...crossings.bStar]).toEqual([3]);
        expect(crossings.attached).toBe(false);
        expect(crossings.crossings.map((rule) => `${rule.lookup.table}${rule.lookup.lookupIndex}`)).toEqual(["GPOS3", "GPOS3"]);
        expect(analyzeComputeTextSegmentation(program)).toBeNull();
    });

    it("installs the paragraph seam and returns the cached indic proof through the dispatcher", () => {
        const paragraph = profileSupport("indic")!.paragraph!;
        expect(paragraph.spaces).toBe(proveIndicParagraphSpaces);
        expect(paragraph.flags).toBe(indicParagraphFlags);
        expect(computeTextParagraphSpaceProof(program)).toBe(proveIndicParagraphSpaces(program));
    });

    it("G3: proves exactly the corpus spaces, each HarfBuzz-independent", () => {
        const proof = proveIndicParagraphSpaces(program)!;
        const rows = references.cases.filter((row) => row.script === "Deva" && !row.tags.includes("reference-only"));
        expect(rows.length).toBeGreaterThan(20);
        for (const row of rows) {
            const proven = provenSpaces(proof, row.text);
            expect(proven, row.id).toEqual([...row.proven]);
            const independent = new Map(Object.values(row.splits).map((split) => [split.s, split.independent]));
            for (const s of proven) {
                expect(independent.get(s) ?? true, `${row.id} @${s}`).toBe(true);
            }
        }
    });

    it("leaves reference-only spaces before Vedic marks unsafe through D1", () => {
        const proof = proveIndicParagraphSpaces(program)!;
        const row = references.cases.find((entry) => entry.id === "vedic-crossing")!;
        const scalars = scalarsOf(row.text);
        const ids = glyphsOf(scalars);
        const out = new Uint8Array(scalars.length);
        indicParagraphFlags(scalars, ids, proof, 0, scalars.length, out, 0, "unavailable");
        expect(scalars[2]).toBe(0x1cd0);
        expect(out[1]).toBe(1);
        expect(provenSpaces(proof, row.text)).toEqual([...row.proven]);
    });

    it("D1 rejects a right line that starts inside a grapheme cluster", () => {
        const proof = proveIndicParagraphSpaces(program)!;
        for (const lead of ["\u093F", "\u094D", "\u200D", "\u20F0", "\u0902", "\u093C"]) {
            expect(provenSpaces(proof, `कख ${lead}ग`), `U+${lead.codePointAt(0)!.toString(16)}`).toEqual([]);
        }
        expect(provenSpaces(proof, "कख ग")).toEqual([2]);
        expect(provenSpaces(proof, "क  ग")).toEqual([2]);
    });

    it("D2 bounds both words at 64 scalars", () => {
        const proof = proveIndicParagraphSpaces(program)!;
        const word = (n: number) => "क".repeat(n);
        expect(provenSpaces(proof, `${word(64)} ${word(64)}`)).toEqual([64]);
        expect(provenSpaces(proof, `${word(65)} ${word(64)}`)).toEqual([]);
        expect(provenSpaces(proof, `${word(64)} ${word(65)}`)).toEqual([]);
        expect(provenSpaces(proof, `ख ${word(65)} ग`)).toEqual([]);
        expect(provenSpaces(proof, `ख ${word(64)} ग`)).toEqual([1, 66]);
    });

    it("requires source scalars", () => {
        const proof = proveIndicParagraphSpaces(program)!;
        expect(() => proof.safeAfter(glyphsOf(scalarsOf("क ख")), 1)).toThrow(/scalars/);
    });

    it("REQ-PATH-2: the 65/66 source window decides every space like the full paragraph", () => {
        const proof = proveIndicParagraphSpaces(program)!;
        for (const left of [63, 64, 65, 66]) {
            for (const right of [63, 64, 65, 66]) {
                const text = `गघ च ${"क".repeat(left)} ${"ख".repeat(right)} छ जझ`;
                const scalars = scalarsOf(text);
                const ids = glyphsOf(scalars);
                for (let s = 0; s < scalars.length; s++) {
                    if (scalars[s] === 0x20) {
                        expect(safeComputeTextSourceBoundary(font(), proof, text, s), `${left}/${right} @${s}`).toBe(proof.safeAfter(ids, s, scalars));
                    }
                }
                expect(proof.safeAfter(ids, 5 + left, scalars), `${left}/${right}`).toBe(left <= 64 && right <= 64);
            }
        }
    });

    it("D3: an added crossing rule makes exactly the spaces whose words can match it unsafe", () => {
        // ट/ठ have no nukta form, so they stay outside the transitive normalization closures of these words.
        const tta = nominal(0x91f),
            ttha = nominal(0x920),
            dotted = nominal(0x25cc);
        const right = withRule(program, "GPOS", (template) => ({ ...template, backtrack: [glyphs(3)], input: [glyphs(tta)], lookahead: [] }));
        expect(indicParagraphProofReason(right)).toBeNull();
        expect(provenSpaces(proveIndicParagraphSpaces(right)!, "ख टग ग घट च छ")).toEqual([4, 9, 11]);
        // The dotted circle is always in a word's closure (A12), so this rule constrains only the left word.
        const left = withRule(program, "GPOS", (template) => ({ ...template, backtrack: [glyphs(tta)], input: [glyphs(3)], lookahead: [glyphs(dotted)] }));
        expect(provenSpaces(proveIndicParagraphSpaces(left)!, "गट ख ग")).toEqual([4]);
        const both = withRule(program, "GPOS", (template) => ({ ...template, backtrack: [glyphs(tta)], input: [glyphs(3)], lookahead: [glyphs(ttha)] }));
        expect(provenSpaces(proveIndicParagraphSpaces(both)!, "ट ठ ठ ट ग")).toEqual([3, 5, 7]);
    });

    it("D3 seeds normalization partners transitively", () => {
        const ka = nominal(0x915);
        const right = withRule(program, "GPOS", (template) => ({ ...template, backtrack: [glyphs(3)], input: [glyphs(ka)], lookahead: [] }));
        // ग → ग़ → nukta → क़ → क: conservative, but never misses a normalization image.
        expect(provenSpaces(proveIndicParagraphSpaces(right)!, "ट ग ट ठ")).toEqual([3, 5]);
    });

    it("is unavailable when a window holds two space slots", () => {
        const mutated = withRule(program, "GPOS", (template) => ({ ...template, backtrack: [glyphs(3)], input: [glyphs(3)], lookahead: [] }));
        expect(proveIndicParagraphSpaces(mutated)).toBeNull();
        expect(indicParagraphProofReason(mutated)).toMatch(/^O5/);
    });

    it("is unavailable when the space can attach as a mark or base", () => {
        const index = program.lookups.findIndex((lookup) => lookup.rules.some((rule) => rule.action.kind === "mark-base"));
        const lookup = program.lookups[index]!;
        const template = lookup.rules.find((rule) => rule.action.kind === "mark-base")!;
        const lookups = [...program.lookups];
        lookups[index] = { ...lookup, rules: [...lookup.rules, { ...template, input: [glyphs(3)] }] };
        const mutated: ComputeTextProgram = { ...program, lookups };
        expect(proveIndicParagraphSpaces(mutated)).toBeNull();
        expect(indicParagraphProofReason(mutated)).toMatch(/^O5/);
    });

    it("is unavailable when U+0020 joins vowel constraints, compositions or decompositions", () => {
        const at = (words: Uint32Array, slot: number) => words[slot]!;
        const vowel = withTables(program, (words) => void (words[at(words, 8) + 1] = 0x20));
        expect(indicParagraphProofReason(vowel)).toMatch(/^O1/);
        const composition = withTables(program, (words) => void (words[at(words, 6)] = 0x20));
        expect(indicParagraphProofReason(composition)).toMatch(/^O2/);
        const decomposition = withTables(program, (words) => void (words[at(words, 4) + 2] = 0x20));
        expect(indicParagraphProofReason(decomposition)).toMatch(/^O2/);
        const local = withPlanData(program, () => ({ spaceLocal: false }));
        expect(indicParagraphProofReason(local)).toMatch(/^O3/);
    });

    it("is unavailable when the syllable DFA can continue a token through category X", () => {
        const machineAt = (words: Uint32Array) => words.subarray(words[21]!);
        const start = INDIC_SYLLABLE_MACHINE[0]!;
        const fromStart = withTables(program, (words) => {
            const machine = machineAt(words);
            machine[machine[5]! + onX(machine, start).trans] = start + 1;
        });
        expect(indicParagraphProofReason(fromStart)).toMatch(/^O3/);
        const absorbing = withTables(program, (words) => {
            const machine = machineAt(words);
            const state = start === 0 ? 1 : 0;
            machine[machine[6]! + onX(machine, state).trans] = 2;
        });
        expect(indicParagraphProofReason(absorbing)).toMatch(/^O3/);
    });

    it("is unavailable for init or any unmodeled applied feature", () => {
        const init = withPlanData(program, (data) => ({ features: [...data.features!, "init"] }));
        expect(indicParagraphProofReason(init)).toMatch(/^O4/);
        const frac = withPlanData(program, (data) => ({ features: [...data.features!, "frac"] }));
        expect(indicParagraphProofReason(frac)).toMatch(/^O7/);
        expect(program.plan._profileData!.features).toEqual([
            "abvs",
            "akhn",
            "blwf",
            "blws",
            "cjct",
            "half",
            "nukt",
            "pres",
            "psts",
            "rkrf",
            "rphf",
            "abvm",
            "blwm",
            "dist",
            "mark",
        ]);
    });

    it("A11: category X is a one-scalar token from the start state and ends every other pending token", () => {
        const machine = INDIC_SYLLABLE_MACHINE;
        const start = machine[0]!;
        const states = machine[3]! - machine[2]!;
        expect(states).toBeGreaterThan(100);
        for (let state = 0; state < states; state++) {
            const { target, action } = onX(machine, state);
            expect(target, `state ${state}`).toBe(start);
            if (state === start) {
                expect(action).toBe(11);
            } else {
                expect([1, 3, 4, 6, 7, 8, 14, 15, 16, 17, 18, 20], `state ${state}`).toContain(action);
            }
        }
    });
});
