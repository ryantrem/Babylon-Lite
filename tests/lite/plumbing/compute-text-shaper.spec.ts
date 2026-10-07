import { expect, test } from "../parity/parity-fixtures";

import { shapeSixFields, TEXT_SHAPER_VERSION } from "./compute-text-shaper-adapter";

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

test("compute text shaper matches text-shaper on a real WebGPU device", async ({ page }) => {
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
    expect(hebrew[1]?.[4]).toBe(0);
    const arabic = shapeSixFields("NotoSansArabic-Regular.ttf", "كتب", "arab", "rtl");
    expect(results.arabic).toEqual(arabic);
    expect(arabic).toHaveLength(5);
    expect(results.han).toEqual(shapeSixFields("NotoSansSC-Regular-subset.ttf", "汉字", "hani", "ltr"));
    const gothic = shapeSixFields("NotoSansGothic-Regular.ttf", String.fromCodePoint(0x10330, 0x10331, 0x10332), "goth", "ltr");
    expect(results.gothic).toEqual(gothic);
    expect(gothic.map((row) => row[1])).toEqual([0, 1, 2]);
    expect(results.unmappedRejected).toBe(true);
    expect(results.unmappedPublished).toBe(false);
    expect(results.unmappedError).toMatch(/unmapped/);
    expect(results.placedGlyphCount).toBe(results.robotoFi?.length);
    expect(results.textDataGlyphCount).toBe(results.placedGlyphCount);
    expect(results.rejected).toBe(true);
    expect(results.rejectedPublished).toBe(false);
    expect(results.rejectError).toMatch(/unsupported/);
});
