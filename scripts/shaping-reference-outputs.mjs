import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/** Check committed bytes and bind excluded diagnostics to the persisted identity. */
export function checkReferenceOutputs(destination, pendingOutputs) {
    const identityFile = "reference-identity.json";
    const identityBytes = readFileSync(resolve(destination, identityFile), "utf8");
    assert.equal(identityBytes, pendingOutputs.get(identityFile), `Reference drift: ${identityFile}`);
    const identity = JSON.parse(identityBytes);
    for (const [file, bytes] of pendingOutputs) {
        const diagnosticKey =
            file === "existing-impact-report.json" ? "existingImpactReportSha256" : file === "existing-impact-report.md" ? "existingImpactReportMarkdownSha256" : null;
        if (diagnosticKey) {
            const digest = createHash("sha256").update(bytes).digest("hex");
            assert.equal(digest, identity[diagnosticKey], `Diagnostic reference drift: ${file} (${diagnosticKey})`);
        } else {
            assert.equal(readFileSync(resolve(destination, file), "utf8"), bytes, `Reference drift: ${file}`);
        }
    }
}
