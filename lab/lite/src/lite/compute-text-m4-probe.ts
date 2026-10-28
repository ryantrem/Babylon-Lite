/** Browser-test sidecar: leaves the protected historical fixture entry unchanged.
 * References/oracles are never imported here. The indirect read is test-only.
 */
import {
    createEngine,
    disposeEngine,
    loadComputeFont,
    shapeComputeText,
    createGlyphStorage,
    disposeGlyphStorage,
    createComputeTextData,
    updateComputeTextData,
    disposeComputeTextData,
    readStorageBuffer,
    updateComputeTextParagraphs,
} from "babylon-lite";
import type { ComputeTextGlyph } from "babylon-lite";

const fields = (glyphs: readonly ComputeTextGlyph[]) => glyphs.map((g) => [g.glyphId, g.cluster, g.xAdvance, g.yAdvance, g.xOffset, g.yOffset]);

export async function runComputeTextM4Probe() {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    canvas.style.cssText = "position:fixed;left:-10000px;width:64px;height:64px";
    document.body.append(canvas);
    const engine = await createEngine(canvas);
    const storage = createGlyphStorage();
    const font = await loadComputeFont("/fonts/NotoSansDevanagari-Regular.ttf");
    const data = createComputeTextData(engine, font, storage, "m4-devanagari", { fontSizePx: 48 });
    try {
        const devanagari: Record<string, number[][]> = {};
        for (const text of ["क्ष", "र्कि", "हिन्दी", "ि", "नमस्ते दुनिया", "द्ध", "नद्ध", "द्ध द्ध"]) {
            devanagari[text] = fields(await shapeComputeText(engine, font, text));
        }
        // Public batches use one font. This valid same-font DFLT/Indic/empty
        // batch does not pretend that multi-font fallback was introduced.
        const homogeneousBatch = (await shapeComputeText(engine, font, ["12", "क्ष", ""])).map(fields);
        const inter = await loadComputeFont("/fonts/Inter.ttf");
        const independentFontRequests = [fields(await shapeComputeText(engine, inter, "AC")), devanagari["क्ष"]!];
        await (
            await updateComputeTextData(data, ["नमस्ते दुनिया"])
        ).completed;
        const indirect = [...new Uint32Array(await readStorageBuffer(data._indirect))];
        let paragraphError: string | null = null;
        try {
            await updateComputeTextParagraphs(data, ["नमस्ते दुनिया"], { maxWidthPx: 400 });
        } catch (error) {
            paragraphError = error instanceof Error ? error.message : String(error);
        }
        const unsupported = await loadComputeFont("/fonts/test-unsupported-reverse-chain.ttf");
        let rejectError: string | null = null,
            rejectedPublished = false,
            unmappedError: string | null = null,
            unmappedPublished = false;
        try {
            rejectedPublished = (await shapeComputeText(engine, unsupported, ["AB"])).length > 0;
        } catch (error) {
            rejectError = error instanceof Error ? error.message : String(error);
        }
        try {
            unmappedPublished = (await shapeComputeText(engine, font, ["a"])).length > 0;
        } catch (error) {
            unmappedError = error instanceof Error ? error.message : String(error);
        }
        return {
            devanagari,
            homogeneousBatch,
            independentFontRequests,
            residentStatus: 0,
            indirect,
            paragraphError,
            rejectError,
            rejectedPublished,
            unmappedError,
            unmappedPublished,
        };
    } finally {
        disposeComputeTextData(data);
        disposeGlyphStorage(storage);
        disposeEngine(engine);
        canvas.remove();
    }
}
