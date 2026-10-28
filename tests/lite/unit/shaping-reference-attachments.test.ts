import { deepStrictEqual } from "node:assert";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Font } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { syntheticFontBytes } from "../compute/synthetic";
import { readImpactValidation } from "../compute/references/impact-validation";

interface AttachRow {
    id: string;
    category: string;
    records: number[][];
    oldRecords: number[][];
    info_only: number[][];
    text: string;
    fontPositionCrossCheck?: {
        passed: boolean;
        allSixFieldsChecked: boolean;
        noGsubEffect?: boolean;
        noGposTable?: boolean;
        spans?: { start: number; end: number; minimum: number; skipped: number[] }[];
    };
    attachmentParentsLogical: [number, number][];
    attachmentCrossCheck: {
        passed: boolean;
        allSixFieldsDerived: boolean;
        attachedGlyphs: number;
        geometryChecks: { index: number; root: number; rootRelative: number[]; harfbuzzRootRelative: number[]; passed: boolean }[];
        unattachedAdvanceDifferences: { index: number; glyph: number; hmtx: number; expectedAdvance: number[]; harfbuzzAdvance: number[] }[];
        downstreamTranslations: unknown[];
    };
}
const report = readImpactValidation() as {
    cases: AttachRow[];
    guards: {
        acceptedHUnattachedNonzeroHmtx: { passed: boolean; violatingRows: number; violations: { id: string }[] };
        devanagariNonzeroClass3: { passed: boolean; nonzeroClass3Glyphs: number[]; violations: unknown[] };
    };
};
const rows = report.cases.filter((entry) => entry.category === "A-attach");

describe("R3 exact attachment-advance evidence", () => {
    it("derives all six fields for exactly the three isolated legacy MarkMark rows", () => {
        expect(rows).toHaveLength(3);
        const font = Font.load(syntheticFontBytes("mark-mark"));
        for (const entry of rows) {
            expect(entry.records).toHaveLength(entry.oldRecords.length);
            const parents = new Map(entry.attachmentParentsLogical);
            const advances = entry.oldRecords.map((record, i) => {
                deepStrictEqual(record.slice(2, 4), [font.advanceWidth(record[0]!), 0], entry.id);
                return parents.has(i) ? 0 : record[2]!;
            });
            entry.oldRecords.forEach((record, i) => {
                let xOffset = record[4]!;
                if (parents.has(i)) {
                    let root = parents.get(i)!;
                    while (parents.has(root)) {
                        root = parents.get(root)!;
                    }
                    xOffset -= advances.slice(root, i).reduce((sum, advance) => sum + advance, 0);
                }
                deepStrictEqual(entry.records[i], [record[0], record[1], advances[i], 0, xOffset, record[5]], `${entry.id}: record ${i}`);
            });
            expect(entry.attachmentCrossCheck.passed).toBe(true);
            expect(entry.attachmentCrossCheck.allSixFieldsDerived).toBe(true);
            expect(entry.attachmentCrossCheck.attachedGlyphs).toBe(parents.size);
            expect(entry.attachmentCrossCheck.downstreamTranslations).toHaveLength(entry.records.length);
            for (const geometry of entry.attachmentCrossCheck.geometryChecks) {
                expect(geometry.passed).toBe(true);
                expect(geometry.rootRelative).toEqual(geometry.harfbuzzRootRelative);
            }
            for (const difference of entry.attachmentCrossCheck.unattachedAdvanceDifferences) {
                expect(parents.has(difference.index)).toBe(false);
                expect(difference.expectedAdvance).toEqual([difference.hmtx, 0]);
                expect(difference.harfbuzzAdvance).toEqual([0, 0]);
            }
        }
    });

    it("reproduces the independently reviewed first stack and its exact pen geometry", () => {
        const first = rows.find((entry) => entry.id.endsWith(":0"))!;
        expect(first.records.slice(0, 6)).toEqual([
            [1, 0, 100, 0, 0, 0],
            [2, 1, 200, 0, 0, 0],
            [2, 2, 0, 0, -180, 40],
            [2, 3, 0, 0, -160, 80],
            [3, 4, 0, 0, -150, 110],
            [1, 5, 100, 0, 0, 0],
        ]);
        let pen = 0;
        const positions = first.records.slice(0, 6).map((record) => {
            const position = pen + record[4]!;
            pen += record[2]!;
            return position;
        });
        expect(positions.slice(2, 5)).toEqual([120, 140, 150]);
        expect(positions[5]).toBe(300);
    });

    it("does not accept GDEF-wide zero widths through an H expectation", () => {
        const guard = report.guards.acceptedHUnattachedNonzeroHmtx;
        expect(guard.passed, JSON.stringify(guard.violations.map((entry) => entry.id))).toBe(true);
        expect(guard.violatingRows).toBe(0);
    });

    it("excludes all nonzero-advance class-3 glyphs from the pinned Devanagari references", () => {
        expect(report.guards.devanagariNonzeroClass3).toMatchObject({
            passed: true,
            nonzeroClass3Glyphs: [149, 150, 151, 152, 153, 154],
            violations: [],
        });
        const deva = JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: { records?: number[][] }[] };
        expect(deva.cases.every((entry) => (entry.records ?? []).every((record) => record[0]! < 149 || record[0]! > 154))).toBe(true);
    });

    it("reports exact existing/new residual namespaces and never self-authorizes production", () => {
        const gate = JSON.parse(readFileSync("tests/lite/compute/references/phase1-gate-results.json", "utf8")) as {
            conditions: Record<string, boolean>;
            eligibleForCoordinatorClearance: boolean;
            productionAuthorized: boolean;
            residuals: { kind: string; id: string; pendingIntegration: string | null }[];
        };
        expect(gate.eligibleForCoordinatorClearance).toBe(Object.values(gate.conditions).every(Boolean));
        expect(gate.productionAuthorized).toBe(false);
        expect(gate.residuals).toEqual([]);
        expect(gate.eligibleForCoordinatorClearance).toBe(true);
    });

    it("accepts exactly the three reviewed nominal proofs, without adopting HB zero widths", () => {
        const nominal = report.cases.filter((entry) => entry.category === "N-proof");
        expect(nominal.map((entry) => entry.id)).toEqual([
            "tests/lite/compute/kernel.test.ts#actual WGSL mode mark-ligature:synthetic:mark-ligature:2",
            "tests/lite/compute/kernel.test.ts#actual WGSL mode mark-ligature:synthetic:mark-ligature:3",
            "tests/lite/compute/kernel.test.ts#GPOS point-anchor rejection identifies its table and original run after empty-run filtering:synthetic:point-anchor:0",
        ]);
        for (const entry of nominal) {
            const font = Font.load(syntheticFontBytes(entry.id.includes("point-anchor") ? "point-anchor" : "mark-ligature"));
            const expected = [...entry.text].map((cp, i) => {
                const glyph = font.glyphId(cp.codePointAt(0)!);
                return [glyph, i, font.advanceWidth(glyph), 0, 0, 0];
            });
            deepStrictEqual(entry.records, expected);
            deepStrictEqual(entry.oldRecords, expected);
            deepStrictEqual(
                entry.info_only,
                expected.map((r) => [r[0], r[1], font.gdef!.glyphClassDef.get(r[0]!) === 3 ? 0 : r[2], 0, 0, 0])
            );
            expect(entry.fontPositionCrossCheck).toMatchObject({ passed: true, allSixFieldsChecked: true, noGsubEffect: true });
        }
    });

    it("proves the three no-GPOS streams and checks every cluster, advance and offset", () => {
        const noPosition = report.cases.filter((entry) => entry.category === "G0");
        expect(noPosition.map((entry) => entry.id.split(":").at(-1))).toEqual(["0", "1", "2"]);
        const font = Font.load(syntheticFontBytes("ligature-ignore-marks"));
        expect(font.gpos).toBeNull();
        for (const entry of noPosition) {
            expect(entry.id).toContain(
                "keeps failed selections, skipped components, insertion cursors and inherited attachment fields exact: ligature-ignore-marks:synthetic:ligature-ignore-marks:"
            );
            const spans = entry.fontPositionCrossCheck!.spans!;
            expect(entry.fontPositionCrossCheck).toMatchObject({ passed: true, allSixFieldsChecked: true, noGposTable: true });
            for (const span of spans) {
                expect(entry.text[span.start]).toBe("A");
                expect(entry.text[span.end]).toBe("C");
                expect(entry.text.slice(span.start + 1, span.end)).toBe("B".repeat(span.end - span.start - 1));
                expect(span.minimum).toBe(span.start);
            }
            entry.records.forEach((r, i) => {
                const old = entry.oldRecords[i]!;
                const glyph = r[0]!;
                const hmtx = font.advanceWidth(glyph);
                deepStrictEqual(r.slice(2), [hmtx, 0, 0, 0]);
                deepStrictEqual(old.slice(2), [hmtx, 0, 0, 0]);
                expect(glyph).toBe(old[0]);
                const span = spans.find((span) => span.skipped.includes(old[1]!));
                expect(r[1]).toBe(span?.minimum ?? old[1]);
                deepStrictEqual(entry.info_only[i], font.gdef!.glyphClassDef.get(glyph) === 3 ? [glyph, r[1], 0, 0, -hmtx, 0] : r);
            });
        }
    });

    it("rejects unreviewed IDs and every changed HB field rather than generalizing the six proofs", () => {
        const code = [
            'import assert from "node:assert/strict";',
            'import {readFileSync} from "node:fs";',
            'import syn from "./tests/lite/compute/synthetic.ts";',
            'import {Font} from "./packages/babylon-lite/node_modules/text-shaper/dist/index.js";',
            'import {reviewedFontPositions} from "./scripts/shaping-reference-attachments.mjs";',
            'import audit from "./tests/lite/compute/references/impact-validation.ts"; const report=audit.readImpactValidation();',
            'for(const e of report.cases.filter(e=>["N-proof","G0"].includes(e.category))){',
            'const f=Font.load(syn.syntheticFontBytes(e.font.slice("synthetic:".length)));',
            'assert.throws(()=>reviewedFontPositions(f,{...e,id:e.id+":unreviewed"},e.oldRecords,e.info_only));',
            "for(let field=0;field<6;field++){const hb=structuredClone(e.info_only);hb[0][field]++;assert.throws(()=>reviewedFontPositions(f,e,e.oldRecords,hb));}",
            "const mark=e.info_only.findIndex(r=>f.gdef.glyphClassDef.get(r[0])===3);",
            "const hb=structuredClone(e.info_only);hb[mark][4]++;assert.throws(()=>reviewedFontPositions(f,e,e.oldRecords,hb));",
            "}",
        ].join("\n");
        const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", code], { encoding: "utf8" });
        expect(result.status, result.stderr).toBe(0);
    });
});
