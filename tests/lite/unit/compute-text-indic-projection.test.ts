import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
const base = "tests/lite/compute/references/";
const raw = JSON.parse(readFileSync(`${base}deva-harfbuzz-11.0.1.json`, "utf8")) as {
    cases: { id: string; text: string; pauseSnapshots?: { message: string; glyphs: number[][] }[] }[];
};
const evidence = JSON.parse(readFileSync(`${base}deva-projected-pauses.json`, "utf8")) as {
    compositions: { scalars: number[]; glyphs: number[] }[];
    preNuktCoverage: { roots: number[]; checks: unknown[]; passed: boolean };
    cases: {
        id: string;
        passed: boolean;
        finalSixFieldsExact: boolean;
        clusterMap: number[];
        pauseSnapshots: { message: string; relation: string; glyphs: number[][]; rawGlyphs: number[][]; decomposedGlyphs: number[][] }[];
    }[];
};
describe("independent narrow B1 projection", () => {
    it("uses exactly three pinned compositions, never excluded U0958", () => {
        expect(evidence.compositions.map((entry) => entry.scalars)).toEqual([
            [0x928, 0x93c, 0x929],
            [0x930, 0x93c, 0x931],
            [0x933, 0x93c, 0x934],
        ]);
        expect(evidence.compositions.some((entry) => entry.scalars[2] === 0x958)).toBe(false);
        expect(raw.cases.find((entry) => entry.text === "क़")).toBeDefined();
        expect(raw.cases.find((entry) => entry.text === "क़")).toBeDefined();
    });
    it("independently validates every composed-input corpus case, preserving raw snapshots", () => {
        const affected = raw.cases.filter((entry) => /[\u0929\u0931\u0934]/u.test(entry.text));
        expect(evidence.cases.map((entry) => entry.id)).toEqual(affected.map((entry) => entry.id));
        for (const row of evidence.cases) {
            expect(row.passed).toBe(true);
            expect(row.finalSixFieldsExact).toBe(true);
            const original = raw.cases.find((entry) => entry.id === row.id)!;
            for (const snapshot of row.pauseSnapshots) {
                expect(snapshot.rawGlyphs).toEqual(original.pauseSnapshots!.find((entry) => entry.message === snapshot.message)!.glyphs);
                expect(snapshot.glyphs).toEqual(snapshot.decomposedGlyphs);
                if (!["start reordering indic initial", "end reordering indic initial"].includes(snapshot.message)) {
                    expect(snapshot.relation).toBe("exact");
                    expect(snapshot.glyphs).toEqual(snapshot.rawGlyphs);
                }
            }
        }
    });
    it("checks pre-nukt default-dev2 roots explicitly and rejects unsafe coverage in the independent helper", () => {
        expect(evidence.preNuktCoverage.passed).toBe(true);
        expect(evidence.preNuktCoverage.roots).toEqual([]);
        const result = spawnSync(
            process.execPath,
            [
                "--input-type=module",
                "-e",
                `
import assert from "node:assert/strict";
import {projectIndicSnapshot,indicCompositionEvidence} from "./scripts/shaping-reference-indic-projection.mjs";
const entries=[{scalars:[0x928,0x93c,0x929],glyphs:[75,104,227]}];
assert.deepEqual(projectIndicSnapshot([[75,0],[104,0]],entries),[[227,0]]);
assert.deepEqual(projectIndicSnapshot([[75,0],[104,1]],entries),[[75,0],[104,1]]);
assert.deepEqual(projectIndicSnapshot([[56,0],[104,0]],entries),[[56,0],[104,0]]);
const tag=s=>Buffer.from(s).readUInt32BE();
const unsafe={glyphId:cp=>cp===0x928?75:cp, gsub:{
 scriptList:{scripts:[{scriptTag:tag("dev2"),script:{defaultLangSys:{featureIndices:[0]}}}]},
 featureList:{features:[{featureTag:tag("locl"),feature:{lookupListIndices:[0]}}]},
 lookups:[{subtables:[{coverage:{covers:g=>g===75}}]}]
}};
assert.throws(()=>indicCompositionEvidence(unsafe),/distinguishes/);
`,
            ],
            { encoding: "utf8" }
        );
        expect(result.status, result.stderr).toBe(0);
    });
});
