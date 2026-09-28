import { describe, expect, it, vi } from "vitest";

import type { SurfaceContext } from "../../../packages/babylon-lite/src/engine/surface";
import type { GlyphCurves } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { createGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { createTextData } from "../../../packages/babylon-lite/src/text/text-data";
import { addTextRendererLayer, createTextLayer, createTextRenderer, removeTextRendererLayer } from "../../../packages/babylon-lite/src/text/text-renderer";

interface MockBuffer {
    readonly label: string;
    readonly destroy: ReturnType<typeof vi.fn>;
}

interface BufferWrite {
    readonly buffer: MockBuffer;
    readonly offset: number;
    readonly values: number[];
}

function createMockSurface() {
    const writes: BufferWrite[] = [];
    const device = {
        createBuffer: vi.fn((desc: { label?: string; size: number }) => ({
            label: desc.label ?? "",
            getMappedRange: () => new ArrayBuffer(desc.size),
            unmap: vi.fn(),
            destroy: vi.fn(),
        })),
        createTexture: vi.fn(() => ({ createView: () => ({}), destroy: vi.fn() })),
        createBindGroup: vi.fn(() => ({})),
        createBindGroupLayout: vi.fn(() => ({})),
        createPipelineLayout: vi.fn(() => ({})),
        createShaderModule: vi.fn(() => ({})),
        createRenderPipeline: vi.fn(() => ({})),
        queue: {
            writeBuffer: vi.fn((buffer: MockBuffer, offset: number, data: ArrayBuffer, dataOffset = 0, size?: number) => {
                const byteLength = size ?? data.byteLength - dataOffset;
                writes.push({ buffer, offset, values: Array.from(new Float32Array(data, dataOffset, byteLength / 4)) });
            }),
            writeTexture: vi.fn(),
        },
    };
    const surface = {
        engine: { _device: device },
        canvas: { width: 800, height: 600 },
        format: "bgra8unorm",
    } as unknown as SurfaceContext;
    return { surface, writes };
}

function glyph(): GlyphCurves {
    return {
        glyphId: 1,
        curves: [{ p0x: 0, p0y: 0, p1x: 50, p1y: 100, p2x: 100, p2y: 0 }],
        bounds: { xMin: 0, yMin: 0, xMax: 100, yMax: 100 },
    };
}

function createLayer(opacity = 0.5, coverageGamma = 2) {
    const storage = createGlyphStorage(new Map([["font", new Map([[1, glyph()]])]]));
    const data = createTextData(storage, [{ curveSet: "font", glyphs: [{ glyphId: 1, x: 0, y: 0 }], pixelsPerFontUnit: 1 }]);
    return createTextLayer(data, { opacity, coverageGamma });
}

function colorWrites(writes: readonly BufferWrite[]): BufferWrite[] {
    return writes.filter((write) => write.buffer.label === "text-layer-ubo" && write.offset === 80);
}

describe("text renderer gamma/opacity upload", () => {
    it("uploads the float32 gamma reciprocal and opacity on first use", () => {
        const { surface, writes } = createMockSurface();
        const layer = createLayer(0.25, 4);
        const renderer = createTextRenderer(surface, { layers: [layer] });

        renderer._update();

        expect(colorWrites(writes)).toEqual([
            expect.objectContaining({
                values: [Math.fround(0.25), 1, 1, Math.fround(0.25)],
            }),
        ]);
    });

    it("skips the upload when gamma and opacity are unchanged", () => {
        const { surface, writes } = createMockSurface();
        const layer = createLayer();
        const renderer = createTextRenderer(surface, { layers: [layer] });

        renderer._update();
        renderer._update();
        renderer._update();

        expect(colorWrites(writes)).toHaveLength(1);
    });

    it("uploads when opacity changes", () => {
        const { surface, writes } = createMockSurface();
        const layer = createLayer();
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();

        layer.opacity = 0.75;
        renderer._update();

        expect(colorWrites(writes).map((write) => write.values[3])).toEqual([Math.fround(0.5), Math.fround(0.75)]);
    });

    it("uploads when coverage gamma changes", () => {
        const { surface, writes } = createMockSurface();
        const layer = createLayer();
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();

        layer.coverageGamma = 4;
        renderer._update();

        expect(colorWrites(writes).map((write) => write.values[0])).toEqual([Math.fround(0.5), Math.fround(0.25)]);
    });

    it("skips double-precision changes that are float32-equivalent", () => {
        const { surface, writes } = createMockSurface();
        const layer = createLayer();
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();

        const uploadedOpacity = Math.fround(layer.opacity);
        const uploadedGammaInv = Math.fround(1 / layer.coverageGamma);
        layer.opacity = uploadedOpacity + Number.EPSILON;
        layer.coverageGamma = 1 / (uploadedGammaInv + Number.EPSILON);

        expect(Math.fround(layer.opacity)).toBe(uploadedOpacity);
        expect(Math.fround(1 / layer.coverageGamma)).toBe(uploadedGammaInv);
        renderer._update();

        expect(colorWrites(writes)).toHaveLength(1);
    });

    it("uploads for recreated layer GPU state and a new renderer", () => {
        const { surface, writes } = createMockSurface();
        const layer = createLayer();
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();

        expect(removeTextRendererLayer(renderer, layer)).toBe(true);
        addTextRendererLayer(renderer, layer);
        renderer._update();

        const secondRenderer = createTextRenderer(surface, { layers: [layer] });
        secondRenderer._update();

        expect(colorWrites(writes)).toHaveLength(3);
    });
});
