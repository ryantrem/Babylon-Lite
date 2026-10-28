import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const diagnostics = ["existing-impact-report.json", "existing-impact-report.md"] as const;
const digestKeys = ["existingImpactReportSha256", "existingImpactReportMarkdownSha256"] as const;

function fixtureCheck(change?: (directory: string, outputs: Map<string, string>) => void) {
    const directory = mkdtempSync(join(tmpdir(), "shaping-output-check-"));
    try {
        const outputs = new Map([
            ["existing-impact-report.json", '{"cases":[{"records":[[1,0,2,0,0,0]]}]}\n'],
            ["existing-impact-report.md", "# Diagnostic evidence\n"],
            ["reviewed-expectations.json", '{"cases":[]}\n'],
        ]);
        const identity = Object.fromEntries(diagnostics.map((name, i) => [digestKeys[i], createHash("sha256").update(outputs.get(name)!).digest("hex")]));
        outputs.set("reference-identity.json", JSON.stringify(identity));
        for (const name of ["reference-identity.json", "reviewed-expectations.json"]) {
            writeFileSync(join(directory, name), outputs.get(name)!);
        }
        change?.(directory, outputs);
        return spawnSync(
            process.execPath,
            [
                "--input-type=module",
                "-e",
                'import {checkReferenceOutputs} from "./scripts/shaping-reference-outputs.mjs"; checkReferenceOutputs(process.argv[1],new Map(JSON.parse(process.argv[2])));',
                directory,
                JSON.stringify([...outputs]),
            ],
            { encoding: "utf8" }
        );
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
}

describe("identity-bound diagnostic output packaging", () => {
    it("checks all committed outputs with both diagnostics absent", () => {
        const checked = fixtureCheck();
        expect(checked.status, checked.stderr).toBe(0);
    });

    it.each(diagnostics)("rejects regenerated diagnostic drift for %s against the stored identity", (name) => {
        const checked = fixtureCheck((_directory, outputs) => outputs.set(name, outputs.get(name)! + "drift"));
        expect(checked.status).toBe(1);
        expect(checked.stderr).toContain(`Diagnostic reference drift: ${name}`);
    });

    it.each(digestKeys)("rejects a missing persisted %s even if the fresh identity agrees", (key) => {
        const checked = fixtureCheck((directory, outputs) => {
            const identity = JSON.parse(outputs.get("reference-identity.json")!) as Record<string, string>;
            delete identity[key];
            const bytes = JSON.stringify(identity);
            outputs.set("reference-identity.json", bytes);
            writeFileSync(join(directory, "reference-identity.json"), bytes);
        });
        expect(checked.status).toBe(1);
        expect(checked.stderr).toContain("Diagnostic reference drift:");
    });

    it("byte-checks the persisted identity rather than trusting freshly regenerated hashes", () => {
        const checked = fixtureCheck((_directory, outputs) => {
            const name = diagnostics[0];
            outputs.set(name, "changed");
            const identity = JSON.parse(outputs.get("reference-identity.json")!) as Record<string, string>;
            identity.existingImpactReportSha256 = createHash("sha256").update("changed").digest("hex");
            outputs.set("reference-identity.json", JSON.stringify(identity));
        });
        expect(checked.status).toBe(1);
        expect(checked.stderr).toContain("Reference drift: reference-identity.json");
    });

    it("still rejects compact output drift", () => {
        const checked = fixtureCheck((directory) => writeFileSync(join(directory, "reviewed-expectations.json"), "drift"));
        expect(checked.status).toBe(1);
        expect(checked.stderr).toContain("Reference drift: reviewed-expectations.json");
    });

    it("still requires every non-diagnostic output on disk", () => {
        const checked = fixtureCheck((directory) => rmSync(join(directory, "reviewed-expectations.json")));
        expect(checked.status).toBe(1);
        expect(checked.stderr).toContain("ENOENT");
    });

    it("does not treat present local diagnostic bytes as the identity authority", () => {
        const checked = fixtureCheck((directory) => {
            for (const name of diagnostics) {
                writeFileSync(join(directory, name), "arbitrary local diagnostics");
            }
            expect(readFileSync(join(directory, diagnostics[0]), "utf8")).toBe("arbitrary local diagnostics");
        });
        expect(checked.status, checked.stderr).toBe(0);
    });
});
