import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { existingCorpus, seededStrings } from "../compute/corpus";
import { fixture, FONT_HASHES, oracleFields } from "../compute/fixtures";
import type { ExistingCase } from "../compute/corpus";
import { DIFFERENTIAL_CASES, FIXED_SOURCE_ACCOUNTING, nonReferenceCorpus, PENDING_INPUTS } from "../compute/corpus-inventory";
import { syntheticFont } from "../compute/synthetic";
import type { SyntheticKind } from "../compute/synthetic";
import { Direction, UnicodeBuffer, GlyphBuffer, shapeInto } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import type { Fixture } from "../compute/fixtures";
import { readImpactValidation } from "../compute/references/impact-validation";

interface ImpactCase extends ExistingCase {
    category: "U" | "A" | "A-attach" | "N-proof" | "G0" | "H" | "X";
    flags: string[];
    oldRecords: number[][];
    records?: number[][];
    c1ArithmeticInfo?: number[][];
    syntheticWitnesses?: string[];
    treatment?: string;
    failedDerivation?: string;
    proposedTreatment?: { status: string; reason: string };
    configuration: { requested: string[]; disabled: string[] };
    recordsSha256?: string;
    oldRecordsSha256?: string;
}
const report = readImpactValidation() as {
    sourceReportSha256: string;
    referenceIdentitySha256: string;
    coverageComplete: boolean;
    totals: Record<string, number>;
    cases: ImpactCase[];
    nonReference: { source: string; reason: string; inputSource: string; font?: string }[];
    planIndex: { source: string; kind: string; gsubRoots: number[]; gposRoots: number[] }[];
    pendingInputs: typeof PENDING_INPUTS;
    sourceAccountingComplete: boolean;
};
const liveFonts = new Map<string, Fixture>();
function liveUnchanged(entry: ImpactCase): number[][] {
    let face = liveFonts.get(entry.font);
    if (!face) {
        face = entry.font.startsWith("synthetic:") ? syntheticFont(entry.font.slice(10) as SyntheticKind) : fixture(entry.font as keyof typeof FONT_HASHES);
        liveFonts.set(entry.font, face);
    }
    const input = new UnicodeBuffer(),
        output = new GlyphBuffer();
    input.addStr(entry.text);
    input.setScript(entry.script);
    input.setDirection(entry.direction === "rtl" ? Direction.RTL : Direction.LTR);
    shapeInto(face.oracle, input, output, { script: entry.script, language: null, direction: entry.direction });
    return output.infos.slice(0, output.length).map((info, i) => {
        const p = output.positions[i]!;
        return [info.glyphId, info.cluster, p.xAdvance, p.yAdvance, p.xOffset, p.yOffset];
    });
}

describe("CPU-only existing shaping impact evidence", () => {
    it("binds the compact audit to the generator identity and its diagnostic source digest", () => {
        const bytes = readFileSync("tests/lite/compute/references/reference-identity.json");
        expect(createHash("sha256").update(bytes).digest("hex")).toBe(report.referenceIdentitySha256);
        expect(JSON.parse(bytes.toString()).existingImpactReportSha256).toBe(report.sourceReportSha256);
    });
    it("covers exactly the registered inputs", () => {
        const fonts = new Map(
            Object.keys(FONT_HASHES)
                .filter((file) => !file.includes("Devanagari"))
                .map((file) => {
                    const face = fixture(file as keyof typeof FONT_HASHES);
                    return [file, face.oracle] as const;
                })
        );
        const corpus = existingCorpus(fonts);
        expect(
            report.cases.map(({ id, font, text, source, script, direction, currentlyRejected }) => ({
                id,
                font,
                text,
                source,
                script,
                direction,
                ...(currentlyRejected ? { currentlyRejected } : {}),
            }))
        ).toEqual(corpus);
        expect(report.cases.length).toBe(Object.values(report.totals).reduce((sum, n) => sum + n, 0));
        expect(new Set(report.cases.map((entry) => entry.id)).size).toBe(report.cases.length);
    });

    it("preserves the exact existing LCG seed and output", () => {
        expect(
            createHash("sha256")
                .update(JSON.stringify(seededStrings("ABCDEFXYTabcfivxy 0123456789.,:;!?")))
                .digest("hex")
        ).toBe("93b3980bc811bb09cba49a143f7aa23b517b8f843579d7cf3aeb90fd87992ec6");
    });

    it("retains exact independent six-field records wherever the classification supplies expectations", () => {
        for (const entry of report.cases) {
            expect(["U", "A", "A-attach", "N-proof", "G0", "H", "X"]).toContain(entry.category);
            if (entry.category === "U") {
                expect(entry.flags).toEqual([]);
                expect(entry.recordsSha256, entry.id).toBe(entry.oldRecordsSha256);
                const live = liveUnchanged(entry);
                expect(createHash("sha256").update(JSON.stringify(live)).digest("hex"), entry.id).toBe(entry.recordsSha256);
                expect(
                    live.every((record) => record.length === 6 && record.every(Number.isInteger)),
                    entry.id
                ).toBe(true);
            }
            if (entry.category === "A") {
                expect(entry.flags).toEqual(["C1"]);
            }
            if (entry.category !== "X" && entry.category !== "U") {
                const invalid = entry.records!.findIndex((row) => row.length !== 6 || !row.every(Number.isInteger));
                expect(invalid, `${entry.id}: first non-six-integer record`).toBe(-1);
            }
        }
    });

    it("cross-checks the reviewed Inter comma-below H classification and independent arithmetic", () => {
        const entry = report.cases.find((entry) => entry.font === "Inter.ttf" && entry.text === "A\u0326")!;
        expect(entry.oldRecords).toEqual([
            [2, 0, 1413, 0, 0, 0],
            [1776, 1, 0, 0, 1309, 0],
        ]);
        expect(entry.category).toBe("H");
        expect(entry.flags).toEqual(["C1", "C2", "C5"]);
        expect(entry.records![1]![4]).toBe(-104);
        expect(entry.c1ArithmeticInfo).toEqual(entry.records);
        expect(oracleFields(fixture("Inter.ttf"), "A\u0326")).toEqual(entry.oldRecords);
    });

    it("does not silently assign expected values or field relaxations to any confounded case", () => {
        expect(report.cases.filter((entry) => entry.category === "X")).toEqual([]);
        for (const entry of report.cases.filter((entry) => entry.category === "X")) {
            expect(entry.records, entry.id).toBeUndefined();
            expect(entry.syntheticWitnesses!.length, entry.id).toBeGreaterThan(0);
            expect(entry.proposedTreatment!.status, entry.id).toBe("requires individual Lead approval; no fields relaxed");
            expect(entry.failedDerivation!.length, entry.id).toBeGreaterThan(0);
        }
    });

    it("preserves requested features instead of disabling rlig/rclt/locl on existing fonts", () => {
        for (const entry of report.cases) {
            for (const feature of ["rlig", "rclt", "locl"]) {
                expect(entry.configuration.requested).toContain(feature);
                expect(entry.configuration.disabled).not.toContain(feature);
            }
        }
    });

    it("keeps exact six-field assertions for mapped precomposed scalars and Hebrew no-mark positioning", () => {
        for (const [font, text] of [
            ["Inter.ttf", "Café"],
            ["Inter.ttf", "Łódź"],
            ["NotoSansHebrew-Regular.ttf", "שלום"],
        ]) {
            const entry = report.cases.find((entry) => entry.font === font && entry.text === text)!;
            expect(entry.category).toBe("H");
            expect(entry.records!.length).toBe(entry.oldRecords.length);
            expect(entry.records!.every((record) => record.length === 6 && record.every(Number.isInteger))).toBe(true);
        }
        const hebrew = report.cases.find((entry) => entry.font === "NotoSansHebrew-Regular.ttf" && entry.text === "שלום")!;
        expect(hebrew.records).not.toEqual(hebrew.oldRecords);
    });

    it("validates correct Unicode joining against HarfBuzz instead of retaining the retired legacy pins", () => {
        const validation = JSON.parse(readFileSync("tests/lite/compute/references/arabic-joining-validation.json", "utf8")) as {
            failed: number;
            tested: number;
            cases: { passed: boolean; text: string; harfbuzzRecords: number[][]; explicitCorrectFormRecords: number[][] }[];
        };
        expect(validation.tested).toBe(validation.cases.length);
        expect(validation.failed).toBe(0);
        for (const entry of validation.cases) {
            expect(entry.passed).toBe(true);
            expect(entry.harfbuzzRecords).toEqual(entry.explicitCorrectFormRecords);
        }
        expect(report.cases.some((entry) => entry.treatment === "H-pinned")).toBe(false);
        expect(report.cases.filter((entry) => entry.script === "arab").every((entry) => entry.flags.includes("C9"))).toBe(true);
        expect(report.cases.some((entry) => entry.treatment === "H-C9")).toBe(true);
    });

    it("locks exact equality with the fixed manually audited source-label enumeration, without native test discovery", () => {
        const files = new Map<string, Set<string>>();
        for (const entry of [...report.cases, ...report.nonReference, ...report.planIndex]) {
            const file = entry.source.split("#")[0]!;
            let labels = files.get(file);
            if (!labels) {
                labels = new Set();
                files.set(file, labels);
            }
            labels.add(entry.source);
        }
        const actual = Object.fromEntries(
            [...files].map(([file, labels]) => [
                file,
                {
                    labels: labels.size,
                    sha256: createHash("sha256")
                        .update(JSON.stringify([...labels].sort()))
                        .digest("hex"),
                },
            ])
        );
        expect(actual).toEqual(FIXED_SOURCE_ACCOUNTING);
        expect(report.nonReference).toEqual(nonReferenceCorpus());
        for (const row of report.nonReference) {
            expect(["negative", "mock", "internal-model", "structural", "status-only", "differential"]).toContain(row.reason);
            expect(row.inputSource.length).toBeGreaterThan(0);
        }
        for (const row of report.planIndex) {
            expect(row.kind).toBe("plan-index");
            expect(row.gsubRoots.length).toBeGreaterThan(0);
            expect(row.gposRoots.length).toBeGreaterThan(0);
            expect(row).not.toHaveProperty("records");
        }
        expect(report.sourceAccountingComplete).toBe(true);
        expect(report.pendingInputs).toEqual(PENDING_INPUTS);
        expect(report.coverageComplete).toBe(report.sourceAccountingComplete && report.pendingInputs.length === 0);
    });

    it("accounts for every edit of confirmed differential-only titles without reference shaping rows or guessed capacities", () => {
        expect(report.pendingInputs).toEqual([]);
        expect(report.coverageComplete).toBe(true);
        for (const entry of DIFFERENTIAL_CASES) {
            expect(report.nonReference).toContainEqual(entry);
            expect(entry.reason).toBe("differential");
            expect(
                report.cases.some((row) => row.source === entry.source),
                entry.source
            ).toBe(false);
        }
        expect(report.cases.some((entry) => entry.source.includes("matches whole-run output and oracle at 480/1000"))).toBe(true);
        expect(report.cases.some((entry) => entry.source.includes("matches oracle records, lines, instances and status-only"))).toBe(true);
        expect(DIFFERENTIAL_CASES.filter((entry) => entry.source.includes("four-medium-paragraph")).length).toBe(2);
    });

    it("scans every fixture SinglePos/PairPos value, including dependency lookups, for nonzero yAdvance", () => {
        const { yAdvanceScan } = readImpactValidation() as {
            yAdvanceScan: Record<string, { nonzeroSingleYAdvance: number; nonzeroPairYAdvance: number; nonzeroPairXAdvance: number }>;
        };
        expect(Object.keys(yAdvanceScan)).toHaveLength(6);
        for (const counts of Object.values(yAdvanceScan)) {
            expect(counts.nonzeroSingleYAdvance).toBe(0);
            expect(counts.nonzeroPairYAdvance).toBe(0);
        }
        expect(yAdvanceScan["Inter.ttf"]!.nonzeroPairXAdvance).toBe(12799);
    });

    it("keeps independent trigger analysis and registry serialization free of Lite imports", () => {
        for (const path of ["scripts/shaping-reference-analysis.mjs", "scripts/shaping-references-corpus.ts", "tests/lite/compute/corpus.ts"]) {
            const source = readFileSync(path, "utf8");
            expect(source).not.toMatch(/(?:import|from)\s*["'][^"']*(?:src\/text|\/fixtures|compute-text-program|compute-text-selection)/);
        }
    });
});
