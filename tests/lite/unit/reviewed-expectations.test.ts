import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { reviewedRecords } from "../compute/references/reviewed-records";

describe("compact reviewed expectations", () => {
    it("contains only complete reviewed non-U rows, bound to the reference identity and source report", () => {
        const root = "tests/lite/compute/references/";
        const identityBytes = readFileSync(`${root}reference-identity.json`);
        const identity = JSON.parse(identityBytes.toString()) as { existingImpactReportSha256: string };
        const compact = JSON.parse(readFileSync(`${root}reviewed-expectations.json`, "utf8")) as {
            referenceIdentitySha256: string;
            sourceReportSha256: string;
            cases: { category: string }[];
        };
        expect(compact.referenceIdentitySha256).toBe(createHash("sha256").update(identityBytes).digest("hex"));
        expect(compact.sourceReportSha256).toBe(identity.existingImpactReportSha256);
        expect(compact.cases.every((row) => row.category !== "U" && row.category !== "X")).toBe(true);
        expect(reviewedRecords().every((row) => row.records.every((r) => r.length === 6 && r.every(Number.isInteger)))).toBe(true);
    });
    it.each(["referenceIdentitySha256", "sourceReportSha256"])("rejects stale %s without consulting or modifying the diagnostic report", (field) => {
        const code = `
import assert from "node:assert/strict";
import fs from "node:fs";
import {syncBuiltinESMExports} from "node:module";
import reviewed from "./tests/lite/compute/references/reviewed-records.ts";
const {reviewedRecords}=reviewed;
const original = fs.readFileSync;
fs.readFileSync = function(path,...args) {
 const value = original.call(this,path,...args);
 if (String(path).endsWith("reviewed-expectations.json")) {
  const data = JSON.parse(String(value)); data[${JSON.stringify(field)}] = "stale";
  return typeof value === "string" ? JSON.stringify(data) : Buffer.from(JSON.stringify(data));
 }
 return value;
};
syncBuiltinESMExports();
assert.throws(()=>reviewedRecords(),/identity/i);
`;
        const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", code], { encoding: "utf8" });
        expect(result.status, result.stderr).toBe(0);
    });
});
