import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fixture } from "../compute/fixtures";

interface ReferenceCase {
    id: string;
    text: string;
    expected: string;
    profile: string;
    scalars: string[];
    records?: number[][];
    pauseSnapshots?: { message: string; glyphs: number[][]; serialized: string }[];
}
const data = JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: ReferenceCase[] };

describe("independent Devanagari references", () => {
    it("has exact repertoire, seed, constraint and public-limit coverage", () => {
        expect(data.cases.filter((entry) => entry.id.startsWith("scalar-"))).toHaveLength(122);
        expect(data.cases.filter((entry) => entry.id.startsWith("seeded-"))).toHaveLength(200);
        expect(data.cases.filter((entry) => entry.id.startsWith("vowel-constraint-"))).toHaveLength(22);
        expect(new Set(data.cases.map((entry) => entry.id)).size).toBe(data.cases.length);
        for (const n of [512, 1024, 2048]) {
            expect(data.cases.find((entry) => entry.id === `long-${n}`)?.scalars).toHaveLength(n);
        }
        expect(data.cases.filter((entry) => entry.profile === "DFLT")).toHaveLength(3);
    });

    it("stores exact six-integer records with scalar-index monotone clusters, and no negative outputs", () => {
        for (const entry of data.cases) {
            expect(entry.scalars).toEqual([...entry.text].map((cp) => cp.codePointAt(0)!.toString(16).toUpperCase()));
            if (entry.expected.startsWith("negative:")) {
                expect(entry.records, entry.id).toBeUndefined();
                continue;
            }
            expect(entry.records!.length, entry.id).toBeGreaterThan(0);
            let previous = 0;
            for (const record of entry.records!) {
                expect(record, entry.id).toHaveLength(6);
                expect(record.every(Number.isInteger), entry.id).toBe(true);
                expect(record[1], entry.id).toBeGreaterThanOrEqual(previous);
                expect(record[1], entry.id).toBeLessThan(entry.scalars.length);
                previous = record[1]!;
            }
        }
    });

    it("reproduces the independent conjunct probe and real Indic pause messages", () => {
        const conjunct = data.cases.find((entry) => entry.text === "क्ष")!;
        expect(conjunct.records).toEqual([[90, 0, 717, 0, 0, 0]]);
        expect(conjunct.pauseSnapshots?.map((snapshot) => snapshot.message)).toEqual([
            "start reordering indic initial",
            "end reordering indic initial",
            "start reordering indic final",
            "end reordering indic final",
            "start table GPOS script tag 'dev2'",
            "end table GPOS script tag 'dev2'",
        ]);
    });

    it("hash-verifies the authorized Noto 2.007 fixture without HarfBuzz", () => {
        expect(fixture("NotoSansDevanagari-Regular.ttf").bytes.byteLength).toBe(184228);
    });
});
