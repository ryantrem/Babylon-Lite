import { expect, test } from "../parity/parity-fixtures";

import { shapeSixFields, TEXT_SHAPER_VERSION } from "./compute-text-shaper-adapter";
import { readFileSync } from "node:fs";
import { attachComputeBrowser, captureComputeBrowser } from "./compute-text-m4-evidence";
import type { runComputeTextM4Probe } from "../../../lab/lite/src/lite/compute-text-m4-probe";

test.afterEach(async ({ page }, info) => attachComputeBrowser(page, info));

interface ComputeTextShaperResults {
    error: string | null;
    robotoFi: number[][] | null;
    interAc: number[][] | null;
    interComma: number[][] | null;
    interGreek: number[][] | null;
    interCyrillic: number[][] | null;
    hebrew: number[][] | null;
    arabic: number[][] | null;
    han: number[][] | null;
    gothic: number[][] | null;
    placedGlyphCount: number;
    textDataGlyphCount: number;
    rejected: boolean;
    rejectedPublished: boolean;
    rejectError: string | null;
    unmappedRejected: boolean;
    unmappedPublished: boolean;
    unmappedError: string | null;
}

test("compute text shaper matches reviewed independent records on a real WebGPU device", async ({ page }) => {
    const captured = captureComputeBrowser(page);
    expect(TEXT_SHAPER_VERSION).toBe("0.1.28");
    await page.goto("/lite/compute-text-shaper-test.html");
    await expect(page.locator("#renderCanvas")).toHaveAttribute("data-ready", "true", { timeout: 60_000 });
    const results = await page.evaluate(() => (window as unknown as { __computeTextShaperTest: ComputeTextShaperResults }).__computeTextShaperTest);

    expect(results.error).toBeNull();
    expect(results.robotoFi).toEqual(shapeSixFields("Roboto-Regular.ttf", "fi"));
    expect(results.interAc).toEqual(shapeSixFields("Inter.ttf", "AC"));
    expect(results.interComma).toEqual(shapeSixFields("Inter.ttf", "A\u0326"));
    expect(results.interGreek).toEqual(shapeSixFields("Inter.ttf", "Αθήνα", "grek", "ltr"));
    expect(results.interCyrillic).toEqual(shapeSixFields("Inter.ttf", "Москва", "cyrl", "ltr"));
    const hebrew = shapeSixFields("NotoSansHebrew-Regular.ttf", "שלום", "hebr", "rtl");
    expect(results.hebrew).toEqual(hebrew);
    expect(hebrew.map((row) => row[1])).toEqual([3, 2, 1, 0]);
    expect(hebrew[1]?.[2]).toBe(291);
    // Cleared C3 coverage-ordered positioning, not the retired inert-context pin.
    expect(hebrew[1]?.[4]).toBe(-10);
    const arabic = shapeSixFields("NotoSansArabic-Regular.ttf", "كتب", "arab", "rtl");
    expect(results.arabic).toEqual(arabic);
    // Cleared C1/C2/C8/C9 output includes the two attached component marks.
    expect(arabic).toHaveLength(5);
    expect(results.han).toEqual(shapeSixFields("NotoSansSC-Regular-subset.ttf", "汉字", "hani", "ltr"));
    const gothic = shapeSixFields("NotoSansGothic-Regular.ttf", String.fromCodePoint(0x10330, 0x10331, 0x10332), "goth", "ltr");
    expect(results.gothic).toEqual(gothic);
    expect(gothic.map((row) => row[1])).toEqual([0, 1, 2]);
    expect(results.placedGlyphCount).toBe(results.robotoFi?.length);
    expect(results.textDataGlyphCount).toBe(results.placedGlyphCount);
    const m4 = await page.evaluate(async () => {
        const path = "/lite/src/lite/compute-text-m4-probe.ts";
        const probe = (await import(path)) as { runComputeTextM4Probe: typeof runComputeTextM4Probe };
        return probe.runComputeTextM4Probe();
    });
    // The owned sidecar supplies current negative witnesses without publishing
    // or modifying the protected fixture's pre-existing user diff.
    expect(m4.unmappedPublished).toBe(false);
    expect(m4.unmappedError).toMatch(/unmapped/);
    expect(m4.rejectedPublished).toBe(false);
    expect(m4.rejectError).toMatch(/unsupported GSUB 0/);
    const reference = JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: { text: string; records?: number[][] }[] };
    for (const [text, records] of Object.entries(m4.devanagari)) {
        expect(records, text).toEqual(reference.cases.find((row) => row.text === text)!.records);
    }
    expect(m4.homogeneousBatch).toEqual([shapeSixFields("NotoSansDevanagari-Regular.ttf", "12", "DFLT"), m4.devanagari["क्ष"], []]);
    expect(m4.independentFontRequests).toEqual([shapeSixFields("Inter.ttf", "AC"), m4.devanagari["क्ष"]]);
    expect(captured.errors).toEqual([]);
});
