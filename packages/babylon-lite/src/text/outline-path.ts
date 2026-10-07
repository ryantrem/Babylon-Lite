/** Dependency-neutral quadratic outline recording for command adapters and font parsers. */

import { cubicToQuadratics } from "./cubic-to-quadratics.js";
import type { GlyphBounds, GlyphCurves, QuadCurve } from "./glyph-storage.js";
import { appendOutlineLine, appendOutlineQuadratic } from "./outline-curves.js";

export interface OutlinePoint {
    readonly x: number;
    readonly y: number;
    readonly onCurve: boolean;
}

export interface OutlineRecorder {
    curves: QuadCurve[];
    x: number;
    y: number;
    startX: number;
    startY: number;
    open: boolean;
}

export function createOutlineRecorder(): OutlineRecorder {
    return { curves: [], x: 0, y: 0, startX: 0, startY: 0, open: false };
}

export function outlineMoveTo(recorder: OutlineRecorder, x: number, y: number): void {
    if (recorder.open) {
        outlineClose(recorder);
    }
    recorder.x = x;
    recorder.y = y;
    recorder.startX = x;
    recorder.startY = y;
    recorder.open = true;
}

export function outlineLineTo(recorder: OutlineRecorder, x: number, y: number): void {
    appendOutlineLine(recorder.curves, recorder.x, recorder.y, x, y);
    recorder.x = x;
    recorder.y = y;
    recorder.open = true;
}

export function outlineQuadraticTo(recorder: OutlineRecorder, cx: number, cy: number, x: number, y: number): void {
    appendOutlineQuadratic(recorder.curves, recorder.x, recorder.y, cx, cy, x, y);
    recorder.x = x;
    recorder.y = y;
    recorder.open = true;
}

export function outlineCubicTo(recorder: OutlineRecorder, c1x: number, c1y: number, c2x: number, c2y: number, x: number, y: number): void {
    const [first, second] = cubicToQuadratics(recorder.x, recorder.y, c1x, c1y, c2x, c2y, x, y);
    recorder.curves.push(first, second);
    recorder.x = x;
    recorder.y = y;
    recorder.open = true;
}

export function outlineClose(recorder: OutlineRecorder): void {
    if (!recorder.open) {
        return;
    }
    appendOutlineLine(recorder.curves, recorder.x, recorder.y, recorder.startX, recorder.startY, true);
    recorder.x = recorder.startX;
    recorder.y = recorder.startY;
    recorder.open = false;
}

export function finishOutline(recorder: OutlineRecorder, glyphId: number, bounds: GlyphBounds | null): GlyphCurves | null {
    if (recorder.open) {
        outlineClose(recorder);
    }
    if (recorder.curves.length === 0 || !bounds) {
        return null;
    }
    return { glyphId, curves: recorder.curves, bounds };
}

/** TrueType contour walk, including implied on-curve points between consecutive off-curve points. */
export function appendQuadraticContour(recorder: OutlineRecorder, contour: readonly OutlinePoint[]): void {
    if (contour.length === 0) {
        return;
    }
    let startIndex = 0;
    for (let index = 0; index < contour.length; index++) {
        if (contour[index]?.onCurve) {
            startIndex = index;
            break;
        }
    }
    const allOffCurve = contour.every((point) => !point.onCurve);
    const first = contour[0];
    const last = contour[contour.length - 1];
    if (!first || !last) {
        return;
    }
    const start = allOffCurve ? { x: (first.x + last.x) / 2, y: (first.y + last.y) / 2 } : { x: contour[startIndex]?.x ?? 0, y: contour[startIndex]?.y ?? 0 };
    outlineMoveTo(recorder, start.x, start.y);
    let index = allOffCurve ? 0 : startIndex + 1;
    if (index >= contour.length) {
        index = 0;
    }
    for (let step = 0; step < contour.length; step++) {
        const point = contour[index];
        if (!point) {
            break;
        }
        if (point.onCurve) {
            outlineLineTo(recorder, point.x, point.y);
        } else {
            const nextIndex = index + 1 < contour.length ? index + 1 : 0;
            const next = contour[nextIndex];
            if (!next) {
                break;
            }
            let endX: number;
            let endY: number;
            if (next.onCurve) {
                endX = next.x;
                endY = next.y;
                index = nextIndex;
                step++;
            } else {
                endX = (point.x + next.x) / 2;
                endY = (point.y + next.y) / 2;
            }
            outlineQuadraticTo(recorder, point.x, point.y, endX, endY);
        }
        index++;
        if (index >= contour.length) {
            index = 0;
        }
        if (recorder.x === start.x && recorder.y === start.y) {
            break;
        }
    }
    outlineClose(recorder);
}
