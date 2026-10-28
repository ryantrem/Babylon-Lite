import { deepStrictEqual } from "node:assert";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fixture } from "../compute/fixtures";
import { readImpactValidation } from "../compute/references/impact-validation";

interface PreLookup {
    passed: boolean;
    changed: boolean;
    cliCtypesAgreed: boolean;
    keys: { scalar: number; cluster: number; markClass: number; isMcm: boolean }[];
    expectedLogical: number[][];
    observedLogical: number[][];
    lookupsOffRecords: number[][];
    disabledFeatures: string[];
    lookupsOffTrace: unknown[];
}
interface ArabicRow {
    id: string;
    text: string;
    script: string;
    category: string;
    treatment: string;
    flags: string[];
    records: number[][];
    info_only?: number[][];
    harfbuzzRecords?: number[][];
    markOrderKeyed?: unknown;
    preLookup: PreLookup;
}
const base = "tests/lite/compute/references/";
const existing = readImpactValidation() as { cases: ArabicRow[] };
const witnesses = JSON.parse(readFileSync(`${base}arabic-harfbuzz-11.0.1.json`, "utf8")) as { cases: ArabicRow[] };
const rows = [...existing.cases.filter((entry) => entry.script === "arab"), ...witnesses.cases];

describe("C10 independent original-input mark order", () => {
    it("keeps the 220 MCM block before the 230 block on both original-input F2 witnesses", () => {
        const first = witnesses.cases.find((entry) => entry.id === "arabic-c10-both-0")!;
        const second = witnesses.cases.find((entry) => entry.id === "arabic-c10-both-1")!;
        expect([...first.text].map((cp) => cp.codePointAt(0))).toEqual([0x628, 0x654, 0x655]);
        expect([...second.text].map((cp) => cp.codePointAt(0))).toEqual([0x628, 0x650, 0x655, 0x64e, 0x654]);
        expect(first.preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x655, 1],
            [0x654, 1],
        ]);
        expect(second.preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x655, 1],
            [0x654, 1],
            [0x64e, 1],
            [0x650, 1],
        ]);
        for (const entry of [first, second]) {
            expect(entry.preLookup.observedLogical).toEqual(entry.preLookup.expectedLogical);
            expect(entry.preLookup.cliCtypesAgreed).toBe(true);
        }
    });

    it("validates every Arabic row and witness with all font/default features disabled in CLI and ctypes", () => {
        const font = fixture("NotoSansArabic-Regular.ttf").oracle;
        const tags = [font.gsub, font.gpos].flatMap((table) =>
            table!.featureList.features.map(({ featureTag }) => String.fromCharCode(featureTag >>> 24, (featureTag >>> 16) & 255, (featureTag >>> 8) & 255, featureTag & 255))
        );
        const validation = JSON.parse(readFileSync(`${base}arabic-prelookup-validation.json`, "utf8")) as { tested: number; failed: number; cases: PreLookup[] };
        expect(validation.tested).toBe(139);
        expect(validation.cases).toHaveLength(validation.tested);
        expect(validation.failed).toBe(0);
        for (const entry of rows) {
            const proof = entry.preLookup;
            expect(proof.passed, entry.id).toBe(true);
            expect(proof.cliCtypesAgreed, entry.id).toBe(true);
            deepStrictEqual(
                proof.keys.map((key) => key.scalar),
                [...entry.text].map((cp) => cp.codePointAt(0)),
                `${entry.id}: original, not pre-sorted input`
            );
            deepStrictEqual(
                proof.keys.map((key) => key.cluster),
                [...entry.text].map((_, i) => i)
            );
            deepStrictEqual(proof.expectedLogical, proof.observedLogical, entry.id);
            const inverse = new Map(proof.keys.map((key) => [font.glyphId(key.scalar), key.scalar]));
            deepStrictEqual(
                proof.lookupsOffRecords
                    .slice()
                    .reverse()
                    .map(([glyph, cluster]) => [inverse.get(glyph!), cluster]),
                proof.expectedLogical,
                entry.id
            );
            expect(proof.lookupsOffTrace, entry.id).toEqual([]);
            expect([...new Set(tags)].every((tag) => proof.disabledFeatures.includes(tag))).toBe(true);
            expect(["ccmp", "locl", "rlig", "calt", "mark", "mkmk", "isol", "init", "medi", "fina", "rvrn", "rtlm"].every((tag) => proof.disabledFeatures.includes(tag))).toBe(
                true
            );
            expect(entry.category, entry.id).toBe("H");
            expect(entry.markOrderKeyed).toBeUndefined();
            deepStrictEqual(entry.records, entry.harfbuzzRecords ?? entry.info_only, `${entry.id}: all six final fields exactly matched HB`);
        }
    });

    it("covers all seven reviewed additions, controls and the 33-mark cap with exact scalar clusters", () => {
        const find = (i: number) => witnesses.cases.find((entry) => entry.id === `arabic-c10-${i}`)!;
        expect(witnesses.cases.filter((entry) => /^arabic-c10-\d+$/.test(entry.id))).toHaveLength(7);
        expect(find(0).preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x651, 1],
            [0x650, 1],
        ]);
        expect(find(1).preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x651, 1],
            [0x64f, 1],
        ]);
        expect(find(2).preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x651, 1],
            [0x652, 2],
        ]);
        expect(find(3).preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x64e, 1],
            [0x650, 2],
        ]);
        expect(find(4).preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x654, 1],
            [0x64e, 1],
        ]);
        expect(find(5).preLookup.expectedLogical).toEqual([
            [0x628, 0],
            [0x654, 1],
            [0x651, 1],
            [0x64e, 1],
        ]);
        expect(find(6).preLookup.keys).toHaveLength(34);
        expect(find(6).preLookup.changed).toBe(false);
        expect(find(6).preLookup.expectedLogical).toEqual([...find(6).text].map((cp, cluster) => [cp.codePointAt(0), cluster]));
        expect(find(0).preLookup.keys.map((key) => key.markClass)).toEqual([0, 33, 27]);
        expect(find(4).preLookup.keys.map((key) => key.markClass)).toEqual([0, 31, 230]);
        expect(find(4).preLookup.keys.map((key) => key.isMcm)).toEqual([false, false, true]);
        for (const index of [2, 3, 6]) {
            expect(find(index).preLookup.changed).toBe(false);
        }
    });

    it("lists exactly every reordered row with its complete exact expectation", () => {
        const impact = JSON.parse(readFileSync(`${base}arabic-c10-impact.json`, "utf8")) as {
            arabicRows: number;
            reorderedRows: number;
            cases: { id: string; records: number[][]; expectedPreLookup: number[][]; exactMatchedHarfbuzz: boolean }[];
        };
        const changed = rows.filter((entry) => entry.preLookup.changed);
        expect(impact.arabicRows).toBe(rows.length);
        expect(impact.reorderedRows).toBe(9);
        expect(impact.cases.map((entry) => entry.id)).toEqual(changed.map((entry) => entry.id));
        for (const entry of rows) {
            expect(entry.flags.includes("C10")).toBe(entry.preLookup.changed);
            expect(entry.treatment).toBe(entry.preLookup.changed ? "H-C10" : "H-C9");
        }
        impact.cases.forEach((entry, i) => {
            deepStrictEqual(entry.records, changed[i]!.records);
            deepStrictEqual(entry.expectedPreLookup, changed[i]!.preLookup.expectedLogical);
            expect(entry.exactMatchedHarfbuzz).toBe(true);
        });
        const stacked = witnesses.cases.find((entry) => entry.id === "arabic-stacked-0")!;
        expect(stacked.records).toEqual([
            [370, 1, 0, 0, 307, 40],
            [367, 1, 0, 0, 315, -160],
            [316, 0, 0, 0, 410, -23],
            [14, 0, 993, 0, 0, 0],
        ]);
    });

    it("locks UTR53 MCM membership and exercises independent sort/merge/cap code without the UCD cache", () => {
        const validation = JSON.parse(readFileSync(`${base}arabic-prelookup-validation.json`, "utf8")) as { mcmScalars: number[] };
        expect(validation.mcmScalars).toEqual([0x654, 0x655, 0x658, 0x6dc, 0x6e3, 0x6e7, 0x6e8, 0x8ca, 0x8cb, 0x8cd, 0x8ce, 0x8cf, 0x8d3, 0x8f3]);
        const code = [
            'import assert from "node:assert/strict";',
            'import {arabicPreLookup,ARABIC_MCM} from "./scripts/shaping-reference-unicode.mjs";',
            "const classes={1608:0,1576:0,1614:31,1616:33,1617:27,1620:230,1621:220};",
            "const data={markClass:cp=>classes[cp]??0,isMcm:cp=>ARABIC_MCM.includes(cp)};",
            'assert.deepEqual(arabicPreLookup(data,"بَٔ").expectedLogical,[[1576,0],[1620,1],[1614,1]]);',
            'assert.deepEqual(arabicPreLookup(data,"بِّ").expectedLogical,[[1576,0],[1617,1],[1616,1]]);',
            'assert.deepEqual(arabicPreLookup(data,"بِٕ").expectedLogical,[[1576,0],[1621,1],[1616,1]]);',
            'assert.deepEqual(arabicPreLookup(data,"\\u0628\\u0654\\u0655").expectedLogical,[[1576,0],[1621,1],[1620,1]]);',
            'assert.deepEqual(arabicPreLookup(data,"\\u0628\\u0650\\u0655\\u064e\\u0654").expectedLogical,[[1576,0],[1621,1],[1620,1],[1614,1],[1616,1]]);',
            'const text="بَ"+"ّ".repeat(32);assert.equal(arabicPreLookup(data,text).changed,false);',
        ].join("\n");
        const result = spawnSync(process.execPath, ["--input-type=module", "-e", code], { encoding: "utf8" });
        expect(result.status, result.stderr).toBe(0);
    });
});
