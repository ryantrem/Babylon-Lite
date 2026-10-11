import { describe, expect, it, vi } from "vitest";
import { createSharedAtlas, packAppendGlyph } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphCurves, SharedAtlas } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { appendSharedAtlasGpu, ensureSharedAtlasGpu } from "../../../packages/babylon-lite/src/text/_gpu/text-textures";

function glyph(glyphId: number): GlyphCurves {
    return { glyphId, curves: [{ p0x: 0, p0y: 0, p1x: 1, p1y: 2, p2x: 2, p2y: 0 }], bounds: { xMin: 0, yMin: 0, xMax: 2, yMax: 2 } };
}

function setup() {
    const mock = {
        createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
        createTexture: vi.fn(() => ({ destroy: vi.fn() })),
        queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    };
    const device = mock as unknown as GPUDevice;
    const atlas = createSharedAtlas();
    for (const id of [1, 2, 3]) {
        atlas._glyphSlots.set(id, packAppendGlyph(atlas, glyph(id)));
    }
    ensureSharedAtlasGpu(device, atlas);
    vi.clearAllMocks();
    return { device, mock, atlas };
}

function snapshot(atlas: SharedAtlas): readonly [number, number, number, number] {
    return [atlas._curveTexelsUsed, atlas._bandTexelsUsed, atlas._slotCount, atlas._version];
}

describe("append-only resident atlas uploads", () => {
    it("uploads exactly new texels and metadata without reallocating or retransmitting existing glyphs", () => {
        const { device, mock, atlas } = setup();
        const from = snapshot(atlas);
        atlas._glyphSlots.set(4, packAppendGlyph(atlas, glyph(4)));
        expect(appendSharedAtlasGpu(device, atlas, from)).toBe(true);
        expect(mock.createBuffer).not.toHaveBeenCalled();
        expect(mock.createTexture).not.toHaveBeenCalled();
        expect(mock.queue.writeTexture.mock.calls).toEqual([
            [
                { texture: atlas._gpu!._curveTex, origin: { x: from[0], y: 0 } },
                atlas._curveTexData.buffer,
                { offset: from[0] * 16, bytesPerRow: (atlas._curveTexelsUsed - from[0]) * 16 },
                { width: atlas._curveTexelsUsed - from[0], height: 1 },
            ],
            [
                { texture: atlas._gpu!._bandTex, origin: { x: from[1], y: 0 } },
                atlas._bandTexData.buffer,
                { offset: from[1] * 16, bytesPerRow: (atlas._bandTexelsUsed - from[1]) * 16 },
                { width: atlas._bandTexelsUsed - from[1], height: 1 },
            ],
        ]);
        expect(mock.queue.writeBuffer).toHaveBeenCalledExactlyOnceWith(atlas._gpu!._metaBuf, 3 * 48, atlas._metaData.buffer, 3 * 48, 48);
        expect(atlas._gpu!._uploadedVersion).toBe(atlas._version);
        vi.clearAllMocks();
        expect(ensureSharedAtlasGpu(device, atlas)._rebuilt).toBe(false);
        expect(mock.queue.writeBuffer).not.toHaveBeenCalled();
        expect(mock.queue.writeTexture).not.toHaveBeenCalled();
        expect(appendSharedAtlasGpu(device, atlas, snapshot(atlas))).toBe(true);
        expect(mock.queue.writeBuffer).not.toHaveBeenCalled();
        expect(mock.queue.writeTexture).not.toHaveBeenCalled();
    });

    it.each(["metadata", "curves", "bands", "version", "device"] as const)("requests full installation before writes on %s changes", (kind) => {
        const { device, mock, atlas } = setup();
        const from = snapshot(atlas);
        if (kind === "metadata") {
            atlas._slotCount = 5;
        } else if (kind === "curves") {
            atlas._curveTexelsUsed = 4097;
        } else if (kind === "bands") {
            atlas._bandTexelsUsed = 4097;
        } else if (kind === "version") {
            atlas._gpu!._uploadedVersion = -1;
        } else {
            atlas._gpu!._device = {} as GPUDevice;
        }
        expect(appendSharedAtlasGpu(device, atlas, from)).toBe(false);
        expect(mock.queue.writeBuffer).not.toHaveBeenCalled();
        expect(mock.queue.writeTexture).not.toHaveBeenCalled();
        expect(mock.createBuffer).not.toHaveBeenCalled();
        expect(mock.createTexture).not.toHaveBeenCalled();
    });

    it("splits append texels at texture-row boundaries", () => {
        const { device, mock, atlas } = setup();
        atlas._curveTexelsUsed = 4094;
        atlas._version++;
        ensureSharedAtlasGpu(device, atlas);
        atlas._curveTexelsUsed = 4098;
        const grown = new Float32Array(8192 * 4);
        grown.set(atlas._curveTexData);
        atlas._curveTexData = grown;
        atlas._version++;
        ensureSharedAtlasGpu(device, atlas);
        const from = [4094, atlas._bandTexelsUsed, atlas._slotCount, atlas._version] as const;
        vi.clearAllMocks();
        expect(appendSharedAtlasGpu(device, atlas, from)).toBe(true);
        expect(mock.queue.writeTexture.mock.calls.map((call) => [call[0], call[2], call[3]])).toEqual([
            [
                { texture: atlas._gpu!._curveTex, origin: { x: 4094, y: 0 } },
                { offset: 4094 * 16, bytesPerRow: 32 },
                { width: 2, height: 1 },
            ],
            [
                { texture: atlas._gpu!._curveTex, origin: { x: 0, y: 1 } },
                { offset: 4096 * 16, bytesPerRow: 32 },
                { width: 2, height: 1 },
            ],
        ]);
    });
});
