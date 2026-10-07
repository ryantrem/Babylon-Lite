/** Single-line placement of a compute-shaped run. No wrapping and no second RTL reverse. */

import type { CurveSetId } from "./glyph-storage.js";
import type { GlyphRun } from "./text-data.js";
import type { ComputeShapedTextRun } from "./compute-text-shaper.js";

/** Place shaped records on a baseline that starts at (0, 0). Offsets stay Y-up. */
export function placeComputeTextRun(shaped: ComputeShapedTextRun, curveSet: CurveSetId, fontSizePx: number): GlyphRun {
    if (!shaped?.glyphs || !Number.isFinite(shaped.unitsPerEm) || shaped.unitsPerEm <= 0) {
        throw new Error("placeComputeTextRun: shaped run is invalid");
    }
    if (!Number.isFinite(fontSizePx) || fontSizePx <= 0) {
        throw new Error("placeComputeTextRun: fontSizePx must be a positive finite number");
    }
    const pixelsPerFontUnit = fontSizePx / shaped.unitsPerEm;
    let x = 0;
    let y = 0;
    const glyphs = shaped.glyphs.map((glyph) => {
        const placed = {
            glyphId: glyph.glyphId,
            x: (x + glyph.xOffset) * pixelsPerFontUnit,
            y: (y + glyph.yOffset) * pixelsPerFontUnit,
        };
        x += glyph.xAdvance;
        y += glyph.yAdvance;
        return placed;
    });
    return { curveSet, glyphs, pixelsPerFontUnit };
}
