import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fixture, expectedFields, oracleFields } from "../compute/fixtures";
import { reviewedRecordByInput, reviewedRecords } from "../compute/references/reviewed-records";
import { readImpactValidation } from "../compute/references/impact-validation";

const read = (name: string): unknown => JSON.parse(readFileSync(`tests/lite/compute/references/${name}`, "utf8"));
interface AuditCase {
    id: string;
    font: string;
    text: string;
    script: string;
    direction: string;
    category: string;
}
const auditCases = (): AuditCase[] => (readImpactValidation() as { cases: AuditCase[] }).cases;

describe("reviewed normative expectations and divergence register", () => {
    it("has complete exact records, no X residuals, and exactly the nine special proof rows", () => {
        const rows = reviewedRecords();
        const allRows = auditCases();
        expect(allRows).toHaveLength(7309);
        expect(rows).toHaveLength(1538);
        expect(rows.map((row) => row.id)).toEqual(allRows.filter((row) => row.category !== "U").map((row) => row.id));
        expect(allRows.filter((row) => row.category === "X")).toHaveLength(0);
        expect(rows.filter((row) => row.category === "X")).toHaveLength(0);
        for (const category of ["N-proof", "G0", "A-attach"]) {
            expect(allRows.filter((row) => row.category === category)).toHaveLength(3);
            expect(rows.filter((row) => row.category === category)).toHaveLength(3);
        }
        const malformed: string[] = [];
        rows.forEach((row) =>
            row.records!.forEach((record, index) => {
                if (record.length !== 6 || !record.every(Number.isInteger)) {
                    malformed.push(`${row.id}:${index}`);
                }
            })
        );
        expect(malformed).toEqual([]);
    });

    it("keeps U comparisons live, keys script/font identity, returns copies, and refuses an unreviewed Arabic expectation", () => {
        const roboto = fixture("Roboto-Regular.ttf");
        const unchanged = auditCases().filter((row) => row.font === roboto.file && row.text === "fi" && row.script === "latn" && row.direction === "ltr");
        expect(unchanged.length).toBeGreaterThan(0);
        expect(unchanged.every((row) => row.category === "U")).toBe(true);
        expect(reviewedRecordByInput(roboto.file, "fi", "latn", "ltr")).toBeUndefined();
        expect(expectedFields(roboto, "fi")).toEqual(oracleFields(roboto, "fi"));
        const inter = fixture("Inter.ttf");
        const corrected = expectedFields(inter, "A\u0326");
        expect(corrected[1]![4]).toBe(-104);
        expect(oracleFields(inter, "A\u0326")[1]![4]).toBe(1309);
        corrected[1]![4] = 12345;
        expect(expectedFields(inter, "A\u0326")[1]![4]).toBe(-104);
        expect(expectedFields(roboto, "A\u0326")[1]![4]).toBe(-91);
        expect(() => expectedFields(fixture("NotoSansArabic-Regular.ttf"), "بببببببببب")).toThrow("No reviewed Arabic expectation");
    });

    it("every register link names existing evidence, none admits Devanagari or weakens exact fields", () => {
        const register = read("divergence-register.json") as {
            entries: { id: string; cases: string[]; replacementWitnesses: string[]; evidence: string[] }[];
            publicCoverage: { correction: string }[];
        };
        const synthetic = read("synthetic-harfbuzz-11.0.1.json") as { cases: { id: string }[] };
        const arabic = read("arabic-harfbuzz-11.0.1.json") as { cases: { id: string }[] };
        const rows = reviewedRecords();
        const ids = new Set([...rows.map((row) => row.id), ...synthetic.cases.map((row) => row.id), ...arabic.cases.map((row) => row.id)]);
        const witnesses = new Set([...synthetic.cases.map((row) => row.id), "arabic-no-joining", "segment-reject", "contour-anchor"]);
        for (const entry of register.entries) {
            expect(entry.replacementWitnesses.length, entry.id).toBeGreaterThan(0);
            for (const citation of entry.evidence.filter((value) => value.includes("existing-impact-report.json"))) {
                expect(citation, entry.id).toContain("(diagnostic, not committed; identity in reference-identity.json)");
            }
            entry.replacementWitnesses.forEach((id) => expect(witnesses.has(id), `${entry.id}: ${id}`).toBe(true));
            entry.cases.forEach((id) => {
                expect(ids.has(id), `${entry.id}: ${id}`).toBe(true);
                expect(id).not.toMatch(/devanagari|deva-/i);
            });
        }
        expect(register.entries.map((entry) => entry.id)).toEqual(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "attached-only-advance", "arabic-font-admission"]);
        expect([...new Set(register.publicCoverage.map((entry) => entry.correction))]).toEqual(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]);
    });
});
