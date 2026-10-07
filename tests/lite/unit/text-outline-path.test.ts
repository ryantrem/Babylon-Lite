import { describe, expect, it, vi } from "vitest";
import { getGlyphPath } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

import type { Font } from "../../../packages/babylon-lite/src/text/font";
import { extractGlyphCurves } from "../../../packages/babylon-lite/src/text/glyph-extraction";
import type { GlyphCurves } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { cubicToQuadratics } from "../../../packages/babylon-lite/src/text/cubic-to-quadratics";
import {
    appendQuadraticContour,
    createOutlineRecorder,
    finishOutline,
    outlineClose,
    outlineCubicTo,
    outlineLineTo,
    outlineMoveTo,
    outlineQuadraticTo,
} from "../../../packages/babylon-lite/src/text/outline-path";

vi.mock("../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js", () => ({ getGlyphPath: vi.fn() }));

const bounds = { xMin: 0, yMin: 0, xMax: 10, yMax: 10 };

describe("neutral outline recording", () => {
    it("keeps the inclusive line and exclusive close epsilon, including either axis", () => {
        for (const [x, y] of [
            [0.1, 0],
            [0, -0.1],
        ]) {
            const recorder = createOutlineRecorder();
            outlineMoveTo(recorder, 0, 0);
            outlineLineTo(recorder, x!, y!);
            outlineClose(recorder);
            expect(recorder.curves).toHaveLength(1);
            expect(recorder.x).toBe(0);
            expect(recorder.y).toBe(0);
        }
        const tiny = createOutlineRecorder();
        outlineMoveTo(tiny, 0, 0);
        outlineLineTo(tiny, 0.099, -0.099);
        expect(finishOutline(tiny, 1, bounds)).toBeNull();
        const above = createOutlineRecorder();
        outlineMoveTo(above, 0, 0);
        outlineLineTo(above, 0, 0.101);
        outlineClose(above);
        expect(above.curves).toHaveLength(2);
        outlineClose(above);
        expect(above.curves).toHaveLength(2);
    });

    it("retains quadratic controls, cubic subdivision, and the supplied bounds", () => {
        const recorder = createOutlineRecorder();
        outlineMoveTo(recorder, 0, 0);
        outlineQuadraticTo(recorder, 4, 6, 8, 0);
        outlineCubicTo(recorder, 9, 10, 10, 10, 10, 0);
        expect(recorder.curves).toEqual([{ p0x: 0, p0y: 0, p1x: 4, p1y: 6, p2x: 8, p2y: 0 }, ...cubicToQuadratics(8, 0, 9, 10, 10, 10, 10, 0)]);
        expect(finishOutline(recorder, 7, bounds)?.bounds).toBe(bounds);
        expect(finishOutline(recorder, 7, null)).toBeNull();
    });

    it("auto-closes parser contours at a next move and at finish", () => {
        const recorder = createOutlineRecorder();
        outlineMoveTo(recorder, 0, 0);
        outlineLineTo(recorder, 1, 0);
        outlineMoveTo(recorder, 5, 0);
        expect(recorder.curves).toHaveLength(2);
        outlineLineTo(recorder, 6, 0);
        expect(finishOutline(recorder, 1, bounds)?.curves).toHaveLength(4);
    });

    it("walks implied quadratic points without loading either font implementation", () => {
        const recorder = createOutlineRecorder();
        appendQuadraticContour(recorder, [
            { x: 0, y: 0, onCurve: false },
            { x: 4, y: 8, onCurve: false },
            { x: 8, y: 0, onCurve: false },
        ]);
        expect(recorder.curves).toEqual([
            { p0x: 4, p0y: 0, p1x: 0, p1y: 0, p2x: 2, p2y: 4 },
            { p0x: 2, p0y: 4, p1x: 4, p1y: 8, p2x: 6, p2y: 4 },
            { p0x: 6, p0y: 4, p1x: 8, p1y: 0, p2x: 4, p2y: 0 },
        ]);
        expect(recorder.open).toBe(false);
    });
});

describe("text-shaper command adapter", () => {
    it("does not implicitly close on next-M or end-of-path", () => {
        vi.mocked(getGlyphPath).mockReturnValue({
            bounds,
            commands: [
                { type: "M", x: 0, y: 0 },
                { type: "L", x: 1, y: 0 },
                { type: "M", x: 5, y: 0 },
                { type: "L", x: 6, y: 0 },
            ],
        });
        const target = new Map<number, GlyphCurves>();
        extractGlyphCurves({ _font: {} } as Font, new Set([1]), target);
        expect(target.get(1)?.curves).toEqual([
            { p0x: 0, p0y: 0, p1x: 0.5, p1y: 0, p2x: 1, p2y: 0 },
            { p0x: 5, p0y: 0, p1x: 5.5, p1y: 0, p2x: 6, p2y: 0 },
        ]);
    });

    it("skips bounds-less/empty paths and retains explicit close, Q and C", () => {
        const font = { _font: {} } as Font;
        const target = new Map<number, GlyphCurves>();
        vi.mocked(getGlyphPath).mockReturnValue({ bounds: null, commands: [{ type: "L", x: 1, y: 0 }] });
        extractGlyphCurves(font, new Set([1]), target);
        vi.mocked(getGlyphPath).mockReturnValue({ bounds, commands: [{ type: "M", x: 1, y: 0 }] });
        extractGlyphCurves(font, new Set([2]), target);
        expect(target.size).toBe(0);
        vi.mocked(getGlyphPath).mockReturnValue({
            bounds,
            commands: [{ type: "M", x: 0, y: 0 }, { type: "Q", x1: 4, y1: 6, x: 8, y: 0 }, { type: "C", x1: 9, y1: 10, x2: 10, y2: 10, x: 10, y: 0 }, { type: "Z" }],
        });
        extractGlyphCurves(font, new Set([3]), target);
        expect(target.get(3)?.curves).toEqual([
            { p0x: 0, p0y: 0, p1x: 4, p1y: 6, p2x: 8, p2y: 0 },
            ...cubicToQuadratics(8, 0, 9, 10, 10, 10, 10, 0),
            { p0x: 10, p0y: 0, p1x: 5, p1y: 0, p2x: 0, p2y: 0 },
        ]);
        const existing = target.get(3);
        vi.mocked(getGlyphPath).mockClear();
        extractGlyphCurves(font, new Set([3]), target);
        expect(getGlyphPath).not.toHaveBeenCalled();
        expect(target.get(3)).toBe(existing);
    });
});
