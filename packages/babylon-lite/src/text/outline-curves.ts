/** Neutral geometry primitives; contour state and source-specific closing policy belong to callers. */

import type { QuadCurve } from "./glyph-storage.js";

export function appendOutlineLine(curves: QuadCurve[], x0: number, y0: number, x1: number, y1: number, closing = false): void {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    if (closing ? dx > 0.1 || dy > 0.1 : dx >= 0.1 || dy >= 0.1) {
        appendOutlineQuadratic(curves, x0, y0, (x0 + x1) * 0.5, (y0 + y1) * 0.5, x1, y1);
    }
}

export function appendOutlineQuadratic(curves: QuadCurve[], x0: number, y0: number, cx: number, cy: number, x1: number, y1: number): void {
    curves.push({ p0x: x0, p0y: y0, p1x: cx, p1y: cy, p2x: x1, p2y: y1 });
}
