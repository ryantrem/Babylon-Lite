import { describe, expect, it } from "vitest";

import { placeComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-placement";
import type { ComputeShapedTextRun } from "../../../packages/babylon-lite/src/text/compute-text-shaper";

describe("placeComputeTextRun", () => {
    it("places visual records from a zero baseline without wrapping or reversing", () => {
        const shaped = {
            glyphs: [
                { glyphId: 2, cluster: 1, xAdvance: 10, yAdvance: 1, xOffset: 2, yOffset: -3 },
                { glyphId: 3, cluster: 0, xAdvance: 4, yAdvance: 0, xOffset: 0, yOffset: 5 },
            ],
            scalarCount: 2,
            unitsPerEm: 10,
            script: "latn",
            direction: "rtl",
            _font: {},
        } as unknown as ComputeShapedTextRun;
        const placed = placeComputeTextRun(shaped, "face", 20);
        expect(placed.curveSet).toBe("face");
        expect(placed.pixelsPerFontUnit).toBe(2);
        expect(placed.glyphs).toEqual([
            { glyphId: 2, x: 4, y: -6 },
            { glyphId: 3, x: 20, y: 12 },
        ]);
        expect(() => placeComputeTextRun(shaped, "face", 0)).toThrow(/fontSizePx/);
    });
});
