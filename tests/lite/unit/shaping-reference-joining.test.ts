import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

interface ArabicCase {
    id: string;
    text: string;
    category: string;
    joiningValidated: boolean;
    joiningForms: { scalar: number; joiningType: string; form: string | null }[];
    count: number;
    harfbuzzRecords: number[][];
    lookupTrace: [string, number, string][];
}
const corpus = JSON.parse(readFileSync("tests/lite/compute/references/arabic-harfbuzz-11.0.1.json", "utf8")) as { cases: ArabicCase[] };

describe("independent C9 joining evidence", () => {
    it("has all planned witnesses and real lookup traces, without needing HarfBuzz installed", () => {
        expect(corpus.cases).toHaveLength(30);
        for (const entry of corpus.cases) {
            expect(entry.joiningValidated, entry.id).toBe(true);
            expect(entry.joiningForms).toHaveLength([...entry.text].length);
            expect(entry.harfbuzzRecords).toHaveLength(entry.count);
            expect(
                entry.harfbuzzRecords.every((row) => row.length === 6 && row.every(Number.isInteger)),
                entry.id
            ).toBe(true);
            expect(entry.lookupTrace.length, entry.id).toBeGreaterThan(0);
        }
        expect([...corpus.cases.find((entry) => entry.id === "arabic-long-600")!.text]).toHaveLength(600);
    });

    it("uses correct Unicode contextual forms, transparent-neighbour skipping and join-causing tatweel", () => {
        const forms = (text: string) => corpus.cases.find((entry) => entry.text === text)!.joiningForms.map((entry) => entry.form);
        expect(forms("كتب")).toEqual(["init", "medi", "fina"]);
        expect(forms("ببب")).toEqual(["init", "medi", "fina"]);
        expect(forms("ابت")).toEqual(["isol", "init", "fina"]);
        expect(forms("دار")).toEqual(["isol", "isol", "isol"]);
        expect(forms("بـب")).toEqual(["init", "medi", "fina"]);
        const transparent = corpus.cases.find((entry) => entry.id === "arabic-transparent-2")!;
        expect(transparent.joiningForms[0]!.form).toBe("init");
        expect(transparent.joiningForms.at(-1)!.form).toBe("fina");
        expect(transparent.joiningForms.slice(1, -1).every((entry) => entry.joiningType === "T" && entry.form === null)).toBe(true);
        expect(corpus.cases.find((entry) => entry.text === "ج")!.harfbuzzRecords.map((row) => row[0])).toEqual([333, 22]);
    });

    it("requires exact G4 inert cross-checks with only the approved internal-space addition", () => {
        const validation = JSON.parse(readFileSync("tests/lite/compute/references/restricted-cmap-validation.json", "utf8")) as {
            tested: number;
            failed: number;
            contract: string;
        };
        expect(validation.tested).toBeGreaterThan(5273);
        expect(validation.failed).toBe(0);
        expect(validation.contract).toContain("U+0020 only");
        const { unicode } = JSON.parse(readFileSync("tests/lite/compute/references/reference-identity.json", "utf8")) as {
            unicode: { version: string; license: string; crossCheckedScalars: number; sources: Record<string, { sha256: string; url: string }> };
        };
        expect(unicode.version).toBe("16.0.0");
        expect(unicode.license).toBe("Unicode License v3");
        expect(unicode.crossCheckedScalars).toBe(0x110000 - 0x800);
        expect(Object.keys(unicode.sources).sort()).toEqual(["ArabicShaping.txt", "DerivedJoiningType.txt", "LICENSE.txt", "UnicodeData.txt"]);
        expect(Object.values(unicode.sources).every((entry) => /^[a-f0-9]{64}$/.test(entry.sha256))).toBe(true);
    });

    it("flags a nested-only SinglePos orthogonal advance before any runtime change", () => {
        const code = [
            'import syn from "./tests/lite/compute/synthetic.ts";',
            'import { Font } from "./packages/babylon-lite/node_modules/text-shaper/dist/index.js";',
            'import { analyzeImpact } from "./scripts/shaping-reference-analysis.mjs";',
            'const font=Font.load(syn.syntheticFontBytes("c6-context-pos"));',
            'const result=analyzeImpact(font,{text:"AB",script:"latn",direction:"ltr"},[[1,0,100,0,0,0],[2,1,213,19,11,17]]);',
            "process.stdout.write(JSON.stringify(result.flags));",
        ].join("\n");
        const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", code], { encoding: "utf8" });
        expect(result.status, result.stderr).toBe(0);
        expect(JSON.parse(result.stdout)).toEqual(["C3", "C6"]);
    });
});
