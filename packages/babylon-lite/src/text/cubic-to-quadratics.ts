/** Approximate a cubic Bézier with two quadratics using the "3/4 rule" (matches Slug reference).
 *  Callers that ingest cubic outlines convert them into the quadratic-only format `GlyphCurves` expects.
 *  This module has no text-shaper dependency. */

import type { QuadCurve } from "./glyph-storage.js";

export function cubicToQuadratics(p0x: number, p0y: number, c1x: number, c1y: number, c2x: number, c2y: number, p1x: number, p1y: number): [QuadCurve, QuadCurve] {
    const q0cx = p0x + (c1x - p0x) * 0.75;
    const q0cy = p0y + (c1y - p0y) * 0.75;
    const q1cx = p1x + (c2x - p1x) * 0.75;
    const q1cy = p1y + (c2y - p1y) * 0.75;
    const midx = (q0cx + q1cx) * 0.5;
    const midy = (q0cy + q1cy) * 0.5;
    return [
        { p0x, p0y, p1x: q0cx, p1y: q0cy, p2x: midx, p2y: midy },
        { p0x: midx, p0y: midy, p1x: q1cx, p1y: q1cy, p2x: p1x, p2y: p1y },
    ];
}
