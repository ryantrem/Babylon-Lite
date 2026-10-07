/** Static glyf/CFF outlines for a compute font. No text-shaper import and no GPU handles.
 *
 *  The request is validated and every missing outline is decoded before `target` is mutated.
 *  A valid empty glyph adds nothing. A malformed outline throws and leaves `target` unchanged. */

import type { ComputeFont } from "./compute-font.js";
import { readCffGlyph } from "./font-parser/cff.js";
import { readGlyfGlyph } from "./font-parser/glyf.js";
import type { GlyfGlyph } from "./font-parser/glyf.js";
import { appendQuadraticContour, createOutlineRecorder, finishOutline } from "./outline-path.js";
import { requireTable } from "./font-parser/sfnt.js";
import type { GlyphCurves } from "./glyph-storage.js";

let glyfByFont: WeakMap<ComputeFont, Map<number, GlyfGlyph>> | null = null;

/** @internal Immutable contours and components shared by extraction and closure. */
export function readComputeGlyfGlyph(font: ComputeFont, glyphId: number): GlyfGlyph {
    let glyphs = glyfByFont?.get(font);
    const existing = glyphs?.get(glyphId);
    if (existing) {
        return existing;
    }
    const glyph = readGlyfGlyph(font._bytes, font._directory, font._locaFormat, font.glyphCount, glyphId);
    if (!glyphs) {
        glyphs = new Map();
        (glyfByFont ??= new WeakMap()).set(font, glyphs);
    }
    glyphs.set(glyphId, glyph);
    return glyph;
}

export function extractComputeGlyphCurves(font: ComputeFont, glyphIds: ReadonlySet<number>, target: Map<number, GlyphCurves>): void {
    if (!(target instanceof Map)) {
        throw new Error("extractComputeGlyphCurves: target must be a Map");
    }
    const pending: { glyphId: number; curves: GlyphCurves | null }[] = [];
    for (const glyphId of glyphIds) {
        if (!Number.isInteger(glyphId) || glyphId < 0 || glyphId >= font.glyphCount) {
            throw new Error(`extractComputeGlyphCurves: glyph ${glyphId} is outside 0..${font.glyphCount - 1}`);
        }
        if (target.has(glyphId)) {
            continue;
        }
        pending.push({ glyphId, curves: decodeGlyph(font, glyphId) });
    }
    for (const entry of pending) {
        if (entry.curves) {
            target.set(entry.glyphId, entry.curves);
        }
    }
}

function decodeGlyph(font: ComputeFont, glyphId: number): GlyphCurves | null {
    if (font.outlineFormat === "cff") {
        const table = requireTable(font._directory, "CFF ");
        return readCffGlyph(font._bytes, table, glyphId, font.glyphCount, font.unitsPerEm);
    }
    const glyph = readComputeGlyfGlyph(font, glyphId);
    if (glyph.empty || !glyph.bounds) {
        return null;
    }
    const recorder = createOutlineRecorder();
    for (const contour of glyph.contours) {
        appendQuadraticContour(recorder, contour);
    }
    return finishOutline(recorder, glyphId, glyph.bounds);
}
