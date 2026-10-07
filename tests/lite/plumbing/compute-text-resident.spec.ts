import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "../parity/parity-fixtures";

const pageSource = readFileSync(resolve(process.cwd(), "lab/lite/src/lite/compute-text-resident-test.ts"), "utf8");

test("resident compute text shapes Roboto fi and Inter AC without mapping instances", async ({ page }) => {
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
});
