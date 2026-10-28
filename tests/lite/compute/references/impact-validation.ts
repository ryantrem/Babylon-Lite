/** Identity-bound CPU audit evidence, never a production/browser import. */
import { deepStrictEqual, ok } from "node:assert";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
export function readImpactValidation(): unknown {
    const bytes = readFileSync("tests/lite/compute/references/reference-identity.json");
    const identity = JSON.parse(bytes.toString()) as { existingImpactReportSha256: string };
    const audit = JSON.parse(readFileSync("tests/lite/compute/references/impact-validation.json", "utf8")) as {
        referenceIdentitySha256: string;
        sourceReportSha256: string;
        configurations: unknown[];
        cases: { configurationIndex: number; configuration?: unknown }[];
    };
    deepStrictEqual(audit.referenceIdentitySha256, createHash("sha256").update(bytes).digest("hex"), "Stale audit reference identity");
    deepStrictEqual(audit.sourceReportSha256, identity.existingImpactReportSha256, "Audit diagnostic-source identity mismatch");
    for (const row of audit.cases) {
        ok(Number.isInteger(row.configurationIndex) && row.configurationIndex >= 0 && row.configurationIndex < audit.configurations.length, "Invalid audit configuration");
        row.configuration = audit.configurations[row.configurationIndex];
    }
    return audit;
}
