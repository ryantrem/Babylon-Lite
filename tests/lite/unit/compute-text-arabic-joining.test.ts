import { createHash } from "node:crypto";
import { readFileSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { arabicJoiningType, arabicMarkClass, arabicIsMcm } from "../../../packages/babylon-lite/src/text/compute-text-arabic";
import { ARABIC_JOINING_RANGES, ARABIC_MARK_RANGES, ARABIC_MCM_SCALARS, ARABIC_TABLE_SHA256 } from "../../../packages/babylon-lite/src/text/compute-text-arabic-joining";

describe("lazy Arabic Unicode scalar properties", () => {
    it("has pinned digest, sorted non-overlapping ranges and valid type codes", () => {
        expect(
            createHash("sha256")
                .update(JSON.stringify([Array.from(ARABIC_JOINING_RANGES), Array.from(ARABIC_MARK_RANGES), Array.from(ARABIC_MCM_SCALARS)]))
                .digest("hex")
        ).toBe(ARABIC_TABLE_SHA256);
        for (const ranges of [ARABIC_JOINING_RANGES, ARABIC_MARK_RANGES]) {
            expect(ranges.length % 3).toBe(0);
            let end = -1;
            for (let i = 0; i < ranges.length; i += 3) {
                expect(ranges[i]).toBeGreaterThan(end);
                expect(ranges[i + 1]).toBeGreaterThanOrEqual(ranges[i]!);
                end = ranges[i + 1]!;
            }
        }
        expect(
            Array.from(ARABIC_JOINING_RANGES)
                .filter((_, i) => i % 3 === 2)
                .every((value) => value >= 1 && value <= 4)
        ).toBe(true);
        const cache = resolve(homedir(), ".cache/babylon-lite/unicode/16.0.0");
        if (existsSync(resolve(cache, "DerivedJoiningType.txt"))) {
            const checked = spawnSync(process.execPath, ["scripts/generate-arabic-joining-table.mjs", "--source-dir", cache, "--check"], { encoding: "utf8" });
            expect(checked.status, checked.stderr).toBe(0);
        }
    });

    it("matches cited UCD joining classes and the reviewed Arabic mark keys", () => {
        expect([0x627, 0x628, 0x62f, 0x640, 0x64e, 0x621, 0x20].map(arabicJoiningType)).toEqual([1, 2, 1, 2, 4, 0, 0]);
        expect([0x64b, 0x64c, 0x64d, 0x64e, 0x64f, 0x650, 0x651, 0x652, 0x654].map(arabicMarkClass)).toEqual([28, 29, 30, 31, 32, 33, 27, 34, 230]);
        expect(Array.from(ARABIC_MCM_SCALARS)).toEqual([0x654, 0x655, 0x658, 0x6dc, 0x6e3, 0x6e7, 0x6e8, 0x8ca, 0x8cb, 0x8cd, 0x8ce, 0x8cf, 0x8d3, 0x8f3]);
        expect(Array.from(ARABIC_MCM_SCALARS).every(arabicIsMcm)).toBe(true);
        expect(arabicIsMcm(0x64e)).toBe(false);
    });

    it("test-only neighbour transcription equals every committed contextual form", () => {
        const corpus = JSON.parse(readFileSync("tests/lite/compute/references/arabic-harfbuzz-11.0.1.json", "utf8")) as {
            cases: { text: string; joiningForms: { form: string | null }[]; preLookup: { keys: { scalar: number; markClass: number; isMcm: boolean }[] } }[];
        };
        for (const entry of corpus.cases) {
            const types = [...entry.text].map((character) => arabicJoiningType(character.codePointAt(0)!));
            const actual = types.map((type, i) => {
                if (type === 0 || type === 4) {
                    return null;
                }
                let p = i - 1;
                while (p >= 0 && types[p] === 4) {
                    p--;
                }
                let n = i + 1;
                while (n < types.length && types[n] === 4) {
                    n++;
                }
                const back = p >= 0 && [2, 3].includes(types[p]!) && [1, 2].includes(type);
                const forward = n < types.length && [2, 3].includes(type) && [1, 2].includes(types[n]!);
                return back ? (forward ? "medi" : "fina") : forward ? "init" : "isol";
            });
            expect(actual).toEqual(entry.joiningForms.map((entry) => entry.form));
            for (const key of entry.preLookup.keys) {
                expect(arabicMarkClass(key.scalar)).toBe(key.markClass);
                expect(arabicIsMcm(key.scalar)).toBe(key.isMcm);
            }
        }
    });
});
