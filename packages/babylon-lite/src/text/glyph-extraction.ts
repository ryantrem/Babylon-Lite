/** Glyph outline extraction: the default `text-shaper`-backed pipeline that
 *  produces `GlyphCurves` values from a `Font`'s glyph paths.
 *
 *  The outline value types themselves (`QuadCurve`, `GlyphBounds`, `GlyphCurves`)
 *  live in `glyph-storage.ts` because storage is what owns and persists them.
 *  Callers using their own outline source (DirectWrite, FreeType, hand-rolled)
 *  import those types directly from `glyph-storage.js` and never pull this
 *  module — so `text-shaper` stays out of their bundle. The only helper they
 *  may want is `cubicToQuadratics`, which has no dependency on text-shaper. */

import { getGlyphPath } from "text-shaper";
import { cubicToQuadratics } from "./cubic-to-quadratics.js";
import { appendOutlineLine, appendOutlineQuadratic } from "./outline-curves.js";
import type { Font } from "./font.js";
import type { GlyphBounds, GlyphCurves, QuadCurve } from "./glyph-storage.js";

/** Approximate a cubic Bézier with two quadratics using the "3/4 rule" (matches Slug reference).
 *  Re-exported from the dependency-free helper so existing public imports keep working. */
export { cubicToQuadratics } from "./cubic-to-quadratics.js";

function extractOne(font: Font, glyphId: number): GlyphCurves | null {
    const path = getGlyphPath(font._font, glyphId);
    if (!path || !path.bounds) {
        return null;
    }
    const curves: QuadCurve[] = [];
    let curX = 0;
    let curY = 0;
    let startX = 0;
    let startY = 0;
    for (const cmd of path.commands) {
        switch (cmd.type) {
            case "M":
                // text-shaper commands close only on Z, not implicitly on the next M.
                startX = cmd.x;
                startY = cmd.y;
                break;
            case "L":
                appendOutlineLine(curves, curX, curY, cmd.x, cmd.y);
                break;
            case "Q":
                appendOutlineQuadratic(curves, curX, curY, cmd.x1, cmd.y1, cmd.x, cmd.y);
                break;
            case "C":
                curves.push(...cubicToQuadratics(curX, curY, cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y));
                break;
            case "Z":
                appendOutlineLine(curves, curX, curY, startX, startY, true);
                curX = startX;
                curY = startY;
                continue;
        }
        curX = cmd.x;
        curY = cmd.y;
    }
    const bounds: GlyphBounds = {
        xMin: path.bounds.xMin,
        yMin: path.bounds.yMin,
        xMax: path.bounds.xMax,
        yMax: path.bounds.yMax,
    };
    return curves.length === 0 ? null : { glyphId, curves, bounds };
}

/** Extract outlines for the requested glyph ids and add them to `target`.
 *  Skips ids already present in `target`. Glyphs with no outline are silently skipped.
 *  Mutates `target` directly — no allocation when no new glyphs appear. */
export function extractGlyphCurves(font: Font, glyphIds: ReadonlySet<number>, target: Map<number, GlyphCurves>): void {
    const cache = (font._curvesCache ??= new Map());
    for (const id of glyphIds) {
        if (target.has(id)) {
            continue;
        }
        let entry = cache.get(id);
        if (entry === undefined) {
            const extracted = extractOne(font, id);
            if (extracted == null) {
                // Sentinel: cache the absence by skipping target; future calls re-skip via target.has.
                continue;
            }
            entry = extracted;
            cache.set(id, entry);
        }
        target.set(id, entry);
    }
}
