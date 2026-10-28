/** Test-only, identity-backed Phase1 records. No Lite selector/compiler/kernel imports.
 * Shared by Vitest and the Node-side browser adapter; never loaded by a browser page.
 */
import { deepStrictEqual, ok } from "node:assert";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export interface ReviewedRecord {
    readonly id: string;
    readonly font: string;
    readonly text: string;
    readonly script: string;
    readonly direction: "ltr" | "rtl";
    readonly category: string;
    readonly flags: readonly string[];
    readonly records: number[][];
}
interface ReviewedData {
    byId: Map<string, ReviewedRecord>;
    byInput: Map<string, ReviewedRecord>;
    fontKeys: Map<string, string>;
    rows: readonly ReviewedRecord[];
}
let data: ReviewedData | null = null;
const inputKey = (font: string, text: string, script: string, direction: string): string => JSON.stringify([font, text, script, direction]);

function load(): ReviewedData {
    if (data) {
        return data;
    }
    const identityBytes = readFileSync(resolve(process.cwd(), "tests/lite/compute/references/reference-identity.json"));
    const identity = JSON.parse(identityBytes.toString()) as { existingImpactReportSha256: string };
    const report = JSON.parse(readFileSync(resolve(process.cwd(), "tests/lite/compute/references/reviewed-expectations.json"), "utf8")) as {
        referenceIdentitySha256: string;
        sourceReportSha256: string;
        fontHashes: Record<string, string>;
        cases: ReviewedRecord[];
    };
    deepStrictEqual(report.referenceIdentitySha256, createHash("sha256").update(identityBytes).digest("hex"), "Stale compact reference identity");
    deepStrictEqual(report.sourceReportSha256, identity.existingImpactReportSha256, "Compact source report identity mismatch");
    const byId = new Map<string, ReviewedRecord>();
    const byInput = new Map<string, ReviewedRecord>();
    for (const row of report.cases) {
        ok(row.category !== "X", `${row.id}: unaccepted residual cannot supply an expectation`);
        ok(row.category !== "U", `${row.id}: unchanged expectations must use the live independent adapter`);
        ok(
            row.records?.every((record) => record.length === 6 && record.every(Number.isInteger)),
            `${row.id}: incomplete exact expectation`
        );
        const key = inputKey(row.font, row.text, row.script, row.direction);
        const previous = byInput.get(key);
        if (previous) {
            deepStrictEqual(previous.records, row.records, `${row.id}: conflicting reviewed configuration`);
        }
        byId.set(row.id, row);
        byInput.set(key, row);
    }
    data = { byId, byInput, rows: report.cases, fontKeys: new Map(Object.entries(report.fontHashes).map(([font, sha]) => [sha, font])) };
    return data;
}

export function reviewedRecordByInput(font: string, text: string, script: string, direction: "ltr" | "rtl"): ReviewedRecord | undefined {
    return load().byInput.get(inputKey(font, text, script, direction));
}

export function expectedFromReport(id: string): number[][] {
    const row = load().byId.get(id);
    ok(row?.records, `No reviewed shaping record for ${id}`);
    return row.records.map((record) => record.slice());
}

export function reviewedFontKey(bytes: ArrayBuffer | Uint8Array): string | undefined {
    return load().fontKeys.get(
        createHash("sha256")
            .update(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes))
            .digest("hex")
    );
}

export function reviewedRecords(): readonly ReviewedRecord[] {
    return load().rows;
}
