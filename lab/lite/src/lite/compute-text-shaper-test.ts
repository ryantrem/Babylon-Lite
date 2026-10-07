/** Real-WebGPU compute shaper fixture. This entry does not import text-shaper. */

import { createEngine, createGlyphStorage, createTextData, extractComputeGlyphCurves, loadComputeFont, shapeComputeText, updateGlyphStorage } from "babylon-lite";
import type { ComputeFont, ComputeTextGlyph } from "babylon-lite";

interface ComputeTextShaperResults {
    ready: boolean;
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

declare global {
    interface Window {
        __computeTextShaperTest: ComputeTextShaperResults;
    }
}

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const results: ComputeTextShaperResults = {
    ready: false,
    error: null,
    robotoFi: null,
    interAc: null,
    interComma: null,
    interGreek: null,
    interCyrillic: null,
    hebrew: null,
    arabic: null,
    han: null,
    gothic: null,
    placedGlyphCount: 0,
    textDataGlyphCount: 0,
    rejected: false,
    rejectedPublished: false,
    rejectError: null,
    unmappedRejected: false,
    unmappedPublished: false,
    unmappedError: null,
};
window.__computeTextShaperTest = results;

function fields(glyphs: readonly ComputeTextGlyph[]): number[][] {
    return glyphs.map((glyph) => [glyph.glyphId, glyph.cluster, glyph.xAdvance, glyph.yAdvance, glyph.xOffset, glyph.yOffset]);
}

function place(font: ComputeFont, glyphs: readonly ComputeTextGlyph[], curveSet: string, fontSizePx: number) {
    const pixelsPerFontUnit = fontSizePx / font.unitsPerEm;
    let x = 0;
    const placed = glyphs.map((glyph) => {
        const point = { glyphId: glyph.glyphId, x: (x + glyph.xOffset) * pixelsPerFontUnit, y: glyph.yOffset * pixelsPerFontUnit };
        x += glyph.xAdvance;
        return point;
    });
    return { curveSet, glyphs: placed, pixelsPerFontUnit };
}

async function run(): Promise<void> {
    const engine = await createEngine(canvas);
    try {
        const roboto = await loadComputeFont("/fonts/Roboto-Regular.ttf");
        const inter = await loadComputeFont("/fonts/Inter.ttf");
        const hebrewFont = await loadComputeFont("/fonts/NotoSansHebrew-Regular.ttf");
        const arabicFont = await loadComputeFont("/fonts/NotoSansArabic-Regular.ttf");
        const hanFont = await loadComputeFont("/fonts/NotoSansSC-Regular-subset.ttf");
        const gothicFont = await loadComputeFont("/fonts/NotoSansGothic-Regular.ttf");
        const gothicText = String.fromCodePoint(0x10330, 0x10331, 0x10332);
        const fi = await shapeComputeText(engine, roboto, "fi");
        const [ac, comma] = await shapeComputeText(engine, inter, ["AC", "A\u0326"]);
        const greek = await shapeComputeText(engine, inter, "Αθήνα");
        const cyrillic = await shapeComputeText(engine, inter, "Москва");
        const shalom = await shapeComputeText(engine, hebrewFont, "שלום");
        const kutub = await shapeComputeText(engine, arabicFont, "كتب");
        const han = await shapeComputeText(engine, hanFont, "汉字");
        const gothic = await shapeComputeText(engine, gothicFont, gothicText);
        if (!fi || !ac || !comma || !greek || !cyrillic || !shalom || !kutub || !han || !gothic) {
            throw new Error("shapeComputeText did not return every run");
        }
        results.robotoFi = fields(fi);
        results.interAc = fields(ac);
        results.interComma = fields(comma);
        results.interGreek = fields(greek);
        results.interCyrillic = fields(cyrillic);
        results.hebrew = fields(shalom);
        results.arabic = fields(kutub);
        results.han = fields(han);
        results.gothic = fields(gothic);

        const placed = place(roboto, fi, "roboto", 32);
        results.placedGlyphCount = placed.glyphs.length;
        const curves = new Map();
        extractComputeGlyphCurves(roboto, new Set(placed.glyphs.map((glyph) => glyph.glyphId)), curves);
        const storage = createGlyphStorage();
        updateGlyphStorage(storage, placed.curveSet, curves);
        const data = createTextData(storage, [placed]);
        results.textDataGlyphCount = data.runs[0]?.glyphs.length ?? 0;

        let published = false;
        try {
            const rejected = await shapeComputeText(engine, inter, ["A\u20de"]);
            published = rejected.length > 0;
        } catch (error) {
            results.rejected = true;
            results.rejectError = error instanceof Error ? error.message : String(error);
        }
        results.rejectedPublished = published;

        let unmappedPublished = false;
        try {
            const rejected = await shapeComputeText(engine, hanFont, ["A"]);
            unmappedPublished = rejected.length > 0;
        } catch (error) {
            results.unmappedRejected = true;
            results.unmappedError = error instanceof Error ? error.message : String(error);
        }
        results.unmappedPublished = unmappedPublished;
    } catch (error) {
        results.error = error instanceof Error ? error.message : String(error);
    } finally {
        results.ready = true;
        canvas.dataset.ready = "true";
    }
}

void run();
