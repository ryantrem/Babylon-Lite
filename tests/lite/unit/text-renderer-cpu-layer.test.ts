/** CPU adapters upload instances and record normal draws, without resident code. */
import { describe, expect, it, vi } from "vitest";

import type { GlyphCurves } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { createGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphRun } from "../../../packages/babylon-lite/src/text/text-data";
import { createTextData } from "../../../packages/babylon-lite/src/text/text-data";
import { createTextLayer } from "../../../packages/babylon-lite/src/text/text-layer-cpu";
import { createTextRenderer } from "../../../packages/babylon-lite/src/text/text-renderer";
import type { SurfaceContext } from "../../../packages/babylon-lite/src/engine/surface";

type Cmd = { op: "pipeline" } | { op: "vertexBuffer"; slot: number } | { op: "bindGroup" } | { op: "draw"; instances: number } | { op: "drawIndirect" };

function createMockSurface() {
    const commands: Cmd[] = [];
    const counters = { bundlesFinished: 0, drawIndirect: 0 };
    const bundleEncoder = {
        setPipeline: () => commands.push({ op: "pipeline" }),
        setVertexBuffer: (slot: number) => commands.push({ op: "vertexBuffer", slot }),
        setBindGroup: () => commands.push({ op: "bindGroup" }),
        draw: (_vertices: number, instances: number) => commands.push({ op: "draw", instances }),
        drawIndirect: () => {
            counters.drawIndirect++;
            commands.push({ op: "drawIndirect" });
        },
        finish: () => {
            counters.bundlesFinished++;
            return { __bundle: counters.bundlesFinished };
        },
    };
    const device = {
        createBuffer: vi.fn((desc: { size: number }) => ({
            getMappedRange: () => new ArrayBuffer(desc.size),
            unmap: vi.fn(),
            destroy: vi.fn(),
        })),
        createTexture: vi.fn(() => ({ createView: () => ({}), destroy: vi.fn() })),
        createBindGroup: vi.fn(() => ({})),
        createBindGroupLayout: vi.fn(() => ({})),
        createPipelineLayout: vi.fn(() => ({})),
        createShaderModule: vi.fn(() => ({ __label: "text-vert" })),
        createRenderPipeline: vi.fn(() => ({ __vertModule: { __label: "text-vert" } })),
        createRenderBundleEncoder: vi.fn(() => bundleEncoder),
        queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    };
    const engine = { _device: device, _currentEncoder: { beginRenderPass: vi.fn(() => ({ executeBundles: vi.fn(), end: vi.fn() })) } };
    const surface = {
        engine,
        canvas: { width: 800, height: 600 },
        format: "bgra8unorm",
        scRT: { _colorView: {} },
    } as unknown as SurfaceContext;
    return { surface, commands, counters, device };
}

function glyph(glyphId: number): GlyphCurves {
    return {
        glyphId,
        curves: [{ p0x: 0, p0y: 0, p1x: 10, p1y: 20, p2x: 20, p2y: 0 }],
        bounds: { xMin: 0, yMin: 0, xMax: 20, yMax: 20 },
    };
}

describe("CPU text layer adapter", () => {
    it("uploads instances and records draw, never drawIndirect", () => {
        const { surface, commands, counters, device } = createMockSurface();
        const storage = createGlyphStorage(
            new Map([
                [
                    "face",
                    new Map([
                        [1, glyph(1)],
                        [2, glyph(2)],
                    ]),
                ],
            ])
        );
        const run: GlyphRun = {
            curveSet: "face",
            glyphs: [
                { glyphId: 1, x: 0, y: 0 },
                { glyphId: 2, x: 12, y: 0 },
            ],
            pixelsPerFontUnit: 0.05,
        };
        const data = createTextData(storage, [run]);
        const layer = createTextLayer(data);
        expect(layer.data).toBe(data);
        const renderer = createTextRenderer(surface, { layers: [layer] });

        renderer._update();
        renderer._record();

        expect(commands.map((command) => command.op)).toEqual(["pipeline", "vertexBuffer", "vertexBuffer", "bindGroup", "draw"]);
        expect(commands.filter((command) => command.op === "draw")).toEqual([{ op: "draw", instances: data._instanceCount }]);
        expect(counters.drawIndirect).toBe(0);
        expect(device.queue.writeBuffer).toHaveBeenCalled();
        const uploadedInstances = device.queue.writeBuffer.mock.calls.some((call) => call[4] === data._instanceCount * 12);
        expect(uploadedInstances).toBe(true);

        commands.length = 0;
        renderer._update();
        renderer._record();
        expect(commands).toEqual([]);
        expect(counters.bundlesFinished).toBe(1);
        expect(counters.drawIndirect).toBe(0);
    });
});
