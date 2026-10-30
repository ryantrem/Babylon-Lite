import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "../parity/parity-fixtures";
import { attachComputeBrowser, captureComputeBrowser } from "./compute-text-m4-evidence";
import type { runComputeTextM4Probe } from "../../../lab/lite/src/lite/compute-text-m4-probe";

test.afterEach(async ({ page }, info) => attachComputeBrowser(page, info));

const pageSource = readFileSync(resolve(process.cwd(), "lab/lite/src/lite/compute-text-resident-test.ts"), "utf8");

test("resident compute text shapes Roboto fi and Inter AC without mapping instances", async ({ page }) => {
    const captured = captureComputeBrowser(page);
    expect(pageSource).not.toMatch(/from ["']text-shaper["']|import\(["']text-shaper["']\)/);
    expect(pageSource).not.toMatch(/mapAsync|getMappedRange|readStorageBuffer/);
    await page.goto("/lite/compute-text-resident-test.html");
    await expect(page.locator("#renderCanvas")).toHaveAttribute("data-ready", "true", { timeout: 60_000 });
    const results = await page.evaluate(
        () =>
            (
                window as unknown as {
                    __computeTextResidentTest: { error: string | null; robotoFiStatus: number | null; interAcStatus: number | null; sharedSurvived: boolean };
                }
            ).__computeTextResidentTest
    );
    expect(results.error).toBeNull();
    expect(results.robotoFiStatus).toBe(0);
    expect(results.interAcStatus).toBe(0);
    expect(results.sharedSurvived).toBe(true);
    const m4 = await page.evaluate(async () => {
        const path = "/lite/src/lite/compute-text-m4-probe.ts";
        const probe = (await import(path)) as { runComputeTextM4Probe: typeof runComputeTextM4Probe };
        return probe.runComputeTextM4Probe();
    });
    expect(m4.residentStatus).toBe(0);
    expect(m4.indirect[0]).toBe(6);
    expect(m4.indirect[1]).toBeGreaterThan(0);
    expect(m4.paragraphError).toBeNull();
    expect(m4.paragraphLines).toBe(2);
    expect(captured.errors).toEqual([]);
});
