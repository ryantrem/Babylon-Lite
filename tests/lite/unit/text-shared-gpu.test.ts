import { describe, expect, it, vi } from "vitest";

import type { ComputeFont } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createComputeTextLayer } from "../../../packages/babylon-lite/src/text/compute-text-layer";
import { createGlyphStorage, updateGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { createTextData, updateTextData } from "../../../packages/babylon-lite/src/text/text-data";
import { createEmptyTextData } from "../../../packages/babylon-lite/src/text/text-data-state";
import { TEXT_INSTANCE_BYTES, TEXT_STYLE_BYTES } from "../../../packages/babylon-lite/src/text/text-buffer-layout";
import { createTextBindGroup } from "../../../packages/babylon-lite/src/text/_gpu/text-bind-group";
import { createStyleBuffer } from "../../../packages/babylon-lite/src/text/_gpu/text-style-buffer";
import { ensureStyleGpu } from "../../../packages/babylon-lite/src/text/_gpu/text-style-gpu";
import type { CpuTextLayerDraw } from "../../../packages/babylon-lite/src/text/text-layer-cpu";
import { createTextLayer } from "../../../packages/babylon-lite/src/text/text-layer-cpu";
import { setTextLayerPosition } from "../../../packages/babylon-lite/src/text/text-layer";
import {
    addTextRendererLayer,
    createTextRenderer,
    disposeTextRenderer,
    removeTextRendererLayer,
    registerTextRenderer,
} from "../../../packages/babylon-lite/src/text/text-renderer";
import { rebuildRegisteredTextRenderers } from "../../../packages/babylon-lite/src/text/text-recovery";
import { createStorageBuffer, disposeStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { ensureSharedAtlasGpu } from "../../../packages/babylon-lite/src/text/_gpu/text-textures";
import type { SurfaceContext } from "../../../packages/babylon-lite/src/engine/surface";

function mockSurface() {
    const surfaces: SurfaceContext[] = [];
    let lose!: (info: GPUDeviceLostInfo) => void;
    const pass = { executeBundles: vi.fn(), end: vi.fn() };
    const encoder = {
        setPipeline: vi.fn(),
        setVertexBuffer: vi.fn(),
        setBindGroup: vi.fn((_index: number, _group: GPUBindGroup) => undefined),
        draw: vi.fn(),
        drawIndirect: vi.fn(),
        finish: vi.fn(() => ({ bindGroup: encoder.setBindGroup.mock.calls.at(-1)?.[1] })),
    };
    const device = {
        limits: { maxBufferSize: 1 << 20 },
        lost: new Promise<GPUDeviceLostInfo>((resolve) => {
            lose = resolve;
        }),
        createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
            label: desc.label,
            size: desc.size,
            getMappedRange: () => new ArrayBuffer(desc.size),
            unmap: vi.fn(),
            destroy: vi.fn(),
        })),
        createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
        createBindGroup: vi.fn((desc: GPUBindGroupDescriptor) => ({ ...desc })),
        createBindGroupLayout: vi.fn(() => ({})),
        createPipelineLayout: vi.fn(() => ({})),
        createShaderModule: vi.fn(() => ({})),
        createRenderPipeline: vi.fn(() => ({})),
        createRenderBundleEncoder: vi.fn(() => encoder),
        queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    };
    const surface = {
        engine: { _device: device, _currentEncoder: { beginRenderPass: () => pass }, surfaces, _msaaSamples: 1 },
        _renderingContexts: [],
        canvas: { width: 800, height: 600 },
        format: "bgra8unorm",
        scRT: { _colorView: {} },
    } as unknown as SurfaceContext;
    surfaces.push(surface);
    return { surface, device, encoder, pass, lose };
}

function storage() {
    return createGlyphStorage(
        new Map([["font", new Map([[1, { glyphId: 1, curves: [{ p0x: 0, p0y: 0, p1x: 1, p1y: 2, p2x: 2, p2y: 0 }], bounds: { xMin: 0, yMin: 0, xMax: 2, yMax: 2 } }]])]])
    );
}

describe("neutral text state and GPU helpers", () => {
    it("initializes independent branded state with aliases and the shared ABI", () => {
        const atlas = storage();
        const data = createEmptyTextData(atlas);
        const cpu = createTextData(atlas);
        expect(data).toEqual(cpu);
        expect(data._runs).toBe(data.runs);
        expect(data._instances.buffer).toBe(data._instancesU32.buffer);
        expect(data._instances.byteLength).toBe(TEXT_INSTANCE_BYTES);
        expect(data._styles.byteLength).toBe(TEXT_STYLE_BYTES);
        expect(data._storage).toBe(atlas);
        expect(data._runs).not.toBe(cpu._runs);
        expect(data._styles).not.toBe(cpu._styles);
        updateTextData(data, { update: "addRun", run: { curveSet: "font", glyphs: [{ glyphId: 1, x: 3, y: 4 }], pixelsPerFontUnit: 0.5 } });
        expect(data._instances.subarray(0, 2)).toEqual(new Float32Array([3, 4]));
        expect(data._instancesU32[2]).toBe(0);
        expect(cpu.runs).toHaveLength(0);
    });

    it("constructs exact selected bindings without caching or taking ownership", () => {
        const { device } = mockSurface();
        const gpuDevice = device as unknown as GPUDevice;
        const uniform = gpuDevice.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM });
        const metadata = gpuDevice.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
        const styles = createStyleBuffer(gpuDevice, 2 * TEXT_STYLE_BYTES);
        const curves = { createView: vi.fn(() => ({ name: "curves" })) } as unknown as GPUTexture;
        const bands = { createView: vi.fn(() => ({ name: "bands" })) } as unknown as GPUTexture;
        const layout = {} as GPUBindGroupLayout;
        const atlas = { _curveTex: curves, _bandTex: bands, _metaBuf: metadata };
        createTextBindGroup(gpuDevice, layout, uniform, atlas, styles, "selected");
        expect(device.createBindGroup).toHaveBeenLastCalledWith({
            label: "selected",
            layout,
            entries: [
                { binding: 0, resource: { buffer: uniform } },
                { binding: 1, resource: { name: "curves" } },
                { binding: 2, resource: { name: "bands" } },
                { binding: 3, resource: { buffer: metadata } },
                { binding: 4, resource: { buffer: styles } },
            ],
        });
        expect(device.createBuffer).toHaveBeenLastCalledWith({ label: "text-styles", size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        expect(curves.createView).toHaveBeenCalledOnce();
        expect(bands.createView).toHaveBeenCalledOnce();
        expect(uniform.destroy).not.toHaveBeenCalled();
        expect(metadata.destroy).not.toHaveBeenCalled();
        expect(styles.destroy).not.toHaveBeenCalled();
    });

    it("allocates the exact byte length and preserves caller-selected labels", () => {
        const { device } = mockSurface();
        const gpuDevice = device as unknown as GPUDevice;
        for (const size of [TEXT_STYLE_BYTES, 2 * TEXT_STYLE_BYTES, 4 * TEXT_STYLE_BYTES]) {
            const buffer = createStyleBuffer(gpuDevice, size, "selected-style");
            expect(buffer.size).toBe(size);
            expect(device.createBuffer).toHaveBeenLastCalledWith({
                label: "selected-style",
                size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            expect(device.queue.writeBuffer).not.toHaveBeenCalled();
        }
    });

    it("keeps CPU palette upload/growth with its owner", () => {
        const { device } = mockSurface();
        const gpuDevice = device as unknown as GPUDevice;
        const data = createTextData(storage());
        const gpu = { _styleBuf: createStyleBuffer(gpuDevice, TEXT_STYLE_BYTES), _uploadedStyleVersion: -1 };
        data._styles = new Float32Array(16);
        data._styleCount = 2;
        const previous = gpu._styleBuf;
        expect(ensureStyleGpu(gpuDevice, data, gpu)).toBe(true);
        expect(previous.destroy).toHaveBeenCalledOnce();
        expect(gpu._styleBuf.size).toBe(2 * TEXT_STYLE_BYTES);
        expect(device.queue.writeBuffer).toHaveBeenLastCalledWith(gpu._styleBuf, 0, data._styles.buffer, 0, 64);
        device.queue.writeBuffer.mockClear();
        expect(ensureStyleGpu(gpuDevice, data, gpu)).toBe(false);
        expect(device.queue.writeBuffer).not.toHaveBeenCalled();
    });

    it.each([
        [0, 0, 0, 1],
        [0.25, 0.5, 0.75, 0.5],
        [0.25, 0.5, 0.75, 0],
    ] as const)("uploads the selected resident linear RGBA (%s, %s, %s, %s) without premultiplication", (r, g, b, a) => {
        const { surface, device } = mockSurface();
        const color: [number, number, number, number] = [r, g, b, a];
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, storage(), "font", {
            fontSizePx: 20,
            defaultColor: color,
        });
        color.fill(1);
        const upload = device.queue.writeBuffer.mock.calls.find(([buffer]) => buffer === data._style)!;
        expect(upload[1]).toBe(0);
        expect(upload[4]).toBe(TEXT_STYLE_BYTES);
        expect(new Float32Array(upload[2] as ArrayBuffer)).toEqual(new Float32Array([r, g, b, a, 50, 0, 0, 0]));
        disposeComputeTextData(data);
    });

    it.each(
        [null, "black", [], Array(4), [0, 0, 0], [0, 0, 0, 1, 1], [-0.1, 0, 0, 1], [0, 0, 0, 1.1], [0, NaN, 0, 1], [0, 0, Infinity, 1], [0, 0, 0, "1"]].map((defaultColor) => ({
            defaultColor,
        }))
    )("rejects malformed resident colors before allocating or uploading resources: $defaultColor", ({ defaultColor }) => {
        const { surface, device } = mockSurface();
        expect(() => Reflect.apply(createComputeTextData, undefined, [surface.engine, { glyphCount: 2, unitsPerEm: 1000 }, storage(), "font", { defaultColor }])).toThrow(
            /defaultColor must contain four finite RGBA channels in \[0, 1\]/
        );
        expect(device.createBuffer).not.toHaveBeenCalled();
        expect(device.queue.writeBuffer).not.toHaveBeenCalled();
    });

    it("shares allocation/bindings in CPU and resident draws while preserving cached bundles and atlas ownership", () => {
        const { surface, device, encoder } = mockSurface();
        const atlas = storage();
        const font = { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont;
        const first = createComputeTextData(surface.engine, font, atlas, "font", { fontSizePx: 20 });
        const survivor = createComputeTextData(surface.engine, font, atlas, "font");
        const cpu = createTextData(atlas, [{ curveSet: "font", glyphs: [{ glyphId: 1, x: 0, y: 0 }], pixelsPerFontUnit: 0.02 }]);
        const layers = [createTextLayer(cpu), createComputeTextLayer(first), createComputeTextLayer(survivor)];
        const renderer = createTextRenderer(surface, { layers });
        renderer._update();
        renderer._record();
        expect(device.createBindGroup).toHaveBeenCalledTimes(3);
        const entries = device.createBindGroup.mock.calls.map(([desc]) => [...desc.entries]);
        for (const bindings of entries) {
            expect(bindings.map((entry) => entry.binding)).toEqual([0, 1, 2, 3, 4]);
            expect(bindings[3]?.resource).toEqual({ buffer: atlas._curveSets.get("font")!._atlas._gpu!._metaBuf });
        }
        expect(entries[1]?.[4]?.resource).toEqual({ buffer: first._style });
        expect(entries[2]?.[4]?.resource).toEqual({ buffer: survivor._style });
        expect(device.createBuffer).toHaveBeenCalledWith({ label: "compute-text-style", size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const styleUpload = device.queue.writeBuffer.mock.calls.find(([buffer]) => buffer === first._style)!;
        expect(new Float32Array(styleUpload[2] as ArrayBuffer).slice(0, 8)).toEqual(new Float32Array([1, 1, 1, 1, 50, 0, 0, 0]));
        expect(encoder.draw).toHaveBeenCalledOnce();
        expect(encoder.drawIndirect).toHaveBeenCalledTimes(2);
        renderer._update();
        renderer._record();
        expect(device.createBindGroup).toHaveBeenCalledTimes(3);
        expect(encoder.finish).toHaveBeenCalledTimes(3);
        const atlasGpu = atlas._curveSets.get("font")!._atlas._gpu!;
        disposeComputeTextData(first);
        expect(first._style.destroy).toHaveBeenCalledOnce();
        expect(atlasGpu._curveTex.destroy).not.toHaveBeenCalled();
        expect(atlasGpu._metaBuf.destroy).not.toHaveBeenCalled();
        renderer._update();
        renderer._record();
        expect(encoder.drawIndirect).toHaveBeenCalledTimes(2);
        expect(survivor.status).toBe("ready");
        disposeComputeTextData(survivor);
        expect(atlasGpu._curveTex.destroy).not.toHaveBeenCalled();
    });

    it("isolates two placements of one resident block and preserves bundles on content and placement edits", () => {
        const { surface, device, encoder } = mockSurface();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, storage(), "font");
        const a = createComputeTextLayer(data, { positionPx: { x: 10, y: 20 }, opacity: 0.5, coverageGamma: 2, order: 2 });
        const b = createComputeTextLayer(data, { positionPx: { x: 200, y: 300 }, order: 1 });
        expect(a.data).toBe(data);
        expect(b.data).toBe(data);
        expect(Object.keys(data).some((key) => /placeholder/i.test(key))).toBe(false);
        const renderer = createTextRenderer(surface, { layers: [a, b] });
        renderer._update();
        expect(renderer.layers).toEqual([b, a]);
        expect(renderer._record()).toBe(2);
        const ua = renderer._layerGpu.get(a)!._textU;
        const ub = renderer._layerGpu.get(b)!._textU;
        expect(ua).not.toBe(ub);
        const bindings = device.createBindGroup.mock.calls.map(([desc]) => [...desc.entries]);
        expect(bindings.map((entries) => entries[0]?.resource)).toEqual([{ buffer: ub }, { buffer: ua }]);
        expect(bindings.map((entries) => entries[4]?.resource)).toEqual([{ buffer: data._style }, { buffer: data._style }]);
        const writes = device.queue.writeBuffer.mock.calls;
        const mvpA = writes.find(([buffer, offset]) => buffer === ua && offset === 0)!;
        const mvpB = writes.find(([buffer, offset]) => buffer === ub && offset === 0)!;
        // The renderer's matrix scratch is reused; write-time copies below verify mutation too.
        expect(mvpA[4]).toBe(64);
        expect(mvpB[4]).toBe(64);
        expect(writes.filter(([buffer]) => buffer === data._instances._buffer)).toEqual([]);
        expect(device.createBuffer.mock.calls.filter(([desc]) => desc.label === "text-layer-instances" || desc.label === "text-styles")).toEqual([]);
        device.queue.writeBuffer.mockClear();
        const captured: { buffer: GPUBuffer; offset: number; values: number[] }[] = [];
        device.queue.writeBuffer.mockImplementation((buffer, offset, source, dataOffset, size) => {
            captured.push({ buffer, offset, values: [...new Float32Array(source as ArrayBuffer, dataOffset, size! / 4)] });
        });
        setTextLayerPosition(a, 100, 150);
        a.opacity = 0.25;
        a.coverageGamma = 4;
        a.rotationRad = Math.PI / 2;
        a.scale = 2;
        a.order = 0;
        renderer._update();
        renderer._record();
        expect(renderer.layers).toEqual([a, b]);
        expect(captured.map(({ buffer }) => buffer)).toEqual([ua, ua]);
        expect(captured[0]?.values[12]).toBeCloseTo(-0.75);
        expect(captured[0]?.values[13]).toBeCloseTo(0.5);
        expect(captured[0]?.values[1]).toBeCloseTo(-4 / 600);
        expect(captured[1]?.values).toEqual([0.25, 1, 1, 0.25]);
        expect(encoder.finish).toHaveBeenCalledTimes(2);
        a.visible = false;
        expect(renderer._record()).toBe(1);
        a.visible = true;
        expect(renderer._record()).toBe(2);
        expect(encoder.finish).toHaveBeenCalledTimes(2);
        disposeTextRenderer(renderer);
        expect(ua.destroy).toHaveBeenCalledOnce();
        expect(ub.destroy).toHaveBeenCalledOnce();
        expect(data._instances._buffer!.destroy).not.toHaveBeenCalled();
        disposeComputeTextData(data);
    });

    for (const resource of ["metadata", "curves", "bands"] as const) {
        it(`refreshes cached CPU and resident bundles after sibling ${resource} atlas replacement`, () => {
            const { surface, device, encoder } = mockSurface();
            const shared = storage();
            const resident = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, shared, "font");
            const cpu = createTextData(shared, [{ curveSet: "font", glyphs: [{ glyphId: 1, x: 3, y: 4 }], pixelsPerFontUnit: 1 }]);
            const layers = [createTextLayer(cpu), createComputeTextLayer(resident), createComputeTextLayer(resident)];
            const renderer = createTextRenderer(surface, { layers });
            renderer._update();
            renderer._record();
            expect(encoder.finish).toHaveBeenCalledTimes(3);
            const atlas = shared._curveSets.get("font")!._atlas;
            const previous = { curves: atlas._gpu!._curveTex, bands: atlas._gpu!._bandTex, metadata: atlas._gpu!._metaBuf };
            if (resource === "metadata") {
                updateGlyphStorage(
                    shared,
                    "font",
                    new Map([[2, { glyphId: 2, curves: [{ p0x: 0, p0y: 0, p1x: 1, p1y: 2, p2x: 2, p2y: 0 }], bounds: { xMin: 0, yMin: 0, xMax: 2, yMax: 2 } }]])
                );
            } else if (resource === "curves") {
                atlas._curveTexelsUsed = atlas._gpu!._curveTexRows * 4096 + 1;
                atlas._curveTexData = new Float32Array(2 * atlas._gpu!._curveTexRows * 4096 * 4);
            } else {
                atlas._bandTexelsUsed = atlas._gpu!._bandTexRows * 4096 + 1;
                atlas._bandTexData = new Float32Array(2 * atlas._gpu!._bandTexRows * 4096 * 4);
            }
            // Another block/upload consumed `_rebuilt` before either cached layer runs again.
            expect(ensureSharedAtlasGpu(surface.engine._device, atlas)._rebuilt).toBe(true);
            expect(ensureSharedAtlasGpu(surface.engine._device, atlas)._rebuilt).toBe(false);
            expect(previous[resource].destroy).toHaveBeenCalledOnce();
            // Growth occurs between update and record: preparation must precede replay.
            renderer._record();
            expect(encoder.finish).toHaveBeenCalledTimes(6);
            const newBindings = device.createBindGroup.mock.calls.slice(3).map(([desc]) => [...desc.entries]);
            expect(newBindings).toHaveLength(3);
            for (const entries of newBindings) {
                expect(entries[3]?.resource).toEqual({ buffer: atlas._gpu!._metaBuf });
            }
            const bundles = layers.map((layer) => renderer._layerGpu.get(layer)!._renderBundle);
            expect(bundles).toEqual(encoder.finish.mock.results.slice(3).map(({ value }) => value));
            renderer._update();
            renderer._record();
            expect(encoder.finish).toHaveBeenCalledTimes(6);
            expect(encoder.draw).toHaveBeenCalledTimes(2);
            expect(encoder.drawIndirect).toHaveBeenCalledTimes(4);
            disposeTextRenderer(renderer);
            disposeComputeTextData(resident);
            expect(atlas._gpu!._metaBuf.destroy).not.toHaveBeenCalled();
        });
    }

    it("re-records buffer and layout replacements but not resident content writes", () => {
        const { surface, encoder } = mockSurface();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, storage(), "font");
        const layer = createComputeTextLayer(data);
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();
        renderer._record();
        surface.engine._device.queue.writeBuffer(data._instances._buffer!, 0, new Float32Array([1, 2, 0]));
        surface.engine._device.queue.writeBuffer(data._indirect._buffer!, 0, new Uint32Array([6, 1, 0, 0]));
        renderer._update();
        renderer._record();
        expect(encoder.finish).toHaveBeenCalledOnce();
        const previous = data._instances;
        data._instances = createStorageBuffer(surface.engine, 256, { writable: true, vertex: true });
        disposeStorageBuffer(previous);
        renderer._record();
        expect(encoder.finish).toHaveBeenCalledTimes(2);
        const previousIndirect = data._indirect;
        data._indirect = createStorageBuffer(surface.engine, 16, { writable: true, indirect: true });
        disposeStorageBuffer(previousIndirect);
        renderer._record();
        expect(encoder.finish).toHaveBeenCalledTimes(3);
        const layout = {} as GPUBindGroupLayout;
        const attachment = renderer._layerGpu.get(layer)!;
        attachment._bindGroupLayout = layout;
        renderer._record();
        expect(encoder.finish).toHaveBeenCalledTimes(4);
        const oldStyle = data._style;
        data._style = createStyleBuffer(surface.engine._device, TEXT_STYLE_BYTES);
        oldStyle.destroy();
        renderer._record();
        expect(encoder.finish).toHaveBeenCalledTimes(5);
        disposeComputeTextData(data);
        expect(renderer._record()).toBe(0);
        expect(encoder.finish).toHaveBeenCalledTimes(5);
        expect(renderer._visibleBundles).toEqual([]);
        disposeTextRenderer(renderer);
    });

    it("removes/re-adds attachments without destroying borrowed data or the other layer's binding", () => {
        const { surface, encoder } = mockSurface();
        const shared = storage();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, shared, "font");
        const a = createComputeTextLayer(data);
        const b = createComputeTextLayer(data);
        const renderer = createTextRenderer(surface, { layers: [a, b] });
        renderer._update();
        renderer._record();
        const original = renderer._layerGpu.get(a)!;
        const survivor = renderer._layerGpu.get(b)!._renderBundle;
        expect(removeTextRendererLayer(renderer, a)).toBe(true);
        expect(removeTextRendererLayer(renderer, a)).toBe(false);
        expect(original._textU.destroy).toHaveBeenCalledOnce();
        expect(original._renderBundle).toBeNull();
        expect(data._instances._buffer!.destroy).not.toHaveBeenCalled();
        expect(renderer._record()).toBe(1);
        expect(renderer._layerGpu.get(b)!._renderBundle).toBe(survivor);
        addTextRendererLayer(renderer, a);
        renderer._update();
        renderer._record();
        expect(renderer._layerGpu.get(a)!._textU).not.toBe(original._textU);
        expect(encoder.finish).toHaveBeenCalledTimes(3);
        disposeTextRenderer(renderer);
        expect(data.status).toBe("ready");
        expect(shared._curveSets.get("font")!._atlas._gpu!._curveTex.destroy).not.toHaveBeenCalled();
        disposeComputeTextData(data);
        disposeComputeTextData(data);
        expect(data._style.destroy).toHaveBeenCalledOnce();
    });

    it("loses resident data terminally and recovers mixed renderers through the adapter seam", async () => {
        const { surface, encoder, lose } = mockSurface();
        const shared = storage();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, shared, "font");
        const resident = createComputeTextLayer(data);
        const cpu = createTextLayer(createTextData(shared, [{ curveSet: "font", glyphs: [{ glyphId: 1, x: 0, y: 0 }], pixelsPerFontUnit: 1 }]));
        const renderer = createTextRenderer(surface, { layers: [resident, cpu] });
        registerTextRenderer(renderer);
        renderer._update();
        renderer._record();
        const oldUniform = renderer._layerGpu.get(cpu)!._textU;
        // Recovery can run before the lost promise delivers; the resident adapter detects it.
        const next = mockSurface();
        surface.engine._device = next.surface.engine._device;
        surface.engine._currentEncoder = next.surface.engine._currentEncoder;
        rebuildRegisteredTextRenderers(surface.engine);
        expect(data.status).toBe("lost");
        expect(data._style.destroy).toHaveBeenCalledOnce();
        expect(renderer._layerGpu.get(cpu)!._textU).not.toBe(oldUniform);
        expect(renderer._record()).toBe(1);
        expect(next.encoder.draw).toHaveBeenCalledOnce();
        expect(next.encoder.drawIndirect).not.toHaveBeenCalled();
        expect(encoder.drawIndirect).toHaveBeenCalledOnce();
        lose({ reason: "destroyed", message: "test" } as GPUDeviceLostInfo);
        await Promise.resolve();
        expect(data.status).toBe("lost");
        await expect(updateComputeTextData(data, ["A"])).rejects.toThrow("device lost");
        expect(() => createComputeTextLayer(data)).toThrow("not live");
        expect(data._style.destroy).toHaveBeenCalledOnce();
        disposeTextRenderer(renderer);
        disposeComputeTextData(data);
    });

    it("never replays a cached resident bundle after the device loss promise resolves", async () => {
        const { surface, encoder, pass, lose } = mockSurface();
        const shared = storage();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, shared, "font");
        const layer = createComputeTextLayer(data);
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();
        expect(renderer._record()).toBe(1);
        pass.executeBundles.mockClear();
        lose({ reason: "destroyed", message: "test" } as GPUDeviceLostInfo);
        await Promise.resolve();
        expect(data.status).toBe("lost");
        expect(renderer._record()).toBe(0);
        expect(renderer._visibleBundles).toEqual([]);
        expect(renderer._layerGpu.get(layer)!._renderBundle).toBeNull();
        expect(encoder.finish).toHaveBeenCalledOnce();
        expect(pass.executeBundles).not.toHaveBeenCalled();
        expect(shared._curveSets.get("font")!._atlas._gpu!._metaBuf.destroy).not.toHaveBeenCalled();
        disposeTextRenderer(renderer);
        disposeComputeTextData(data);
    });

    it("isolates one layer's bindings across renderer attachments and disposals", () => {
        const { surface, encoder } = mockSurface();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, storage(), "font");
        const layer = createComputeTextLayer(data);
        const first = createTextRenderer(surface, { layers: [layer] });
        const second = createTextRenderer(surface, { layers: [layer] });
        first._update();
        second._update();
        first._record();
        second._record();
        const firstAttachment = first._layerGpu.get(layer)!;
        const secondAttachment = second._layerGpu.get(layer)!;
        expect(firstAttachment._textU).not.toBe(secondAttachment._textU);
        expect(firstAttachment._draw).not.toBe(secondAttachment._draw);
        const bindings = encoder.setBindGroup.mock.calls.map(([, group]) => (group as unknown as GPUBindGroupDescriptor).entries);
        expect(bindings).toHaveLength(2);
        expect([...bindings[0]!][0]?.resource).toEqual({ buffer: firstAttachment._textU });
        expect([...bindings[1]!][0]?.resource).toEqual({ buffer: secondAttachment._textU });
        const survivor = secondAttachment._renderBundle;
        disposeTextRenderer(first);
        expect(firstAttachment._textU.destroy).toHaveBeenCalledOnce();
        expect(secondAttachment._textU.destroy).not.toHaveBeenCalled();
        expect(second._record()).toBe(1);
        expect(secondAttachment._renderBundle).toBe(survivor);
        expect(encoder.finish).toHaveBeenCalledTimes(2);
        expect(data._style.destroy).not.toHaveBeenCalled();
        disposeTextRenderer(second);
        disposeComputeTextData(data);
    });

    it("diagnoses a removed live curve set instead of replaying its cached bundle", () => {
        const { surface, pass } = mockSurface();
        const shared = storage();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, shared, "font");
        const renderer = createTextRenderer(surface, { layers: [createComputeTextLayer(data)] });
        renderer._update();
        renderer._record();
        pass.executeBundles.mockClear();
        const curveSet = shared._curveSets.get("font")!;
        shared._curveSets.delete("font");
        expect(() => renderer._record()).toThrow("live data's curve set is missing");
        expect(pass.executeBundles).not.toHaveBeenCalled();
        shared._curveSets.set("font", curveSet);
        disposeTextRenderer(renderer);
        disposeComputeTextData(data);
    });

    it("rejects another engine and releases only the failed attachment's uniform", () => {
        const owner = mockSurface();
        const foreign = mockSurface();
        const data = createComputeTextData(owner.surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, storage(), "font");
        const renderer = createTextRenderer(foreign.surface, { layers: [createComputeTextLayer(data)] });
        expect(() => renderer._update()).toThrow("renderer must use the data's engine");
        const index = foreign.device.createBuffer.mock.calls.findIndex(([desc]) => desc.label === "text-layer-ubo");
        expect(foreign.device.createBuffer.mock.results[index]!.value.destroy).toHaveBeenCalledOnce();
        expect(renderer._layerGpu.size).toBe(0);
        expect(data.status).toBe("ready");
        expect(data._style.destroy).not.toHaveBeenCalled();
        expect(data._instances._buffer!.destroy).not.toHaveBeenCalled();
        disposeTextRenderer(renderer);
        disposeComputeTextData(data);
    });

    it("rejects invalid admission with zero indirect instances without invalidating an otherwise live bundle", async () => {
        const { surface, device, encoder } = mockSurface();
        const data = createComputeTextData(surface.engine, { glyphCount: 2, unitsPerEm: 1000 } as ComputeFont, storage(), "font");
        const layer = createComputeTextLayer(data);
        const renderer = createTextRenderer(surface, { layers: [layer] });
        renderer._update();
        renderer._record();
        device.queue.writeBuffer.mockClear();
        await expect(updateComputeTextData(data, ["😀"])).rejects.toThrow();
        const indirectWrites = device.queue.writeBuffer.mock.calls.filter(([buffer]) => buffer === data._indirect._buffer);
        expect(indirectWrites.length).toBeGreaterThan(0);
        for (const [, , source, offset, size] of indirectWrites) {
            expect([...new Uint32Array(source as ArrayBuffer, offset, size! / 4)]).toEqual([6, 0, 0, 0]);
        }
        renderer._update();
        renderer._record();
        expect(encoder.finish).toHaveBeenCalledOnce();
        disposeComputeTextData(data);
        expect(renderer._record()).toBe(0);
        disposeTextRenderer(renderer);
    });

    it("uploads each shared CPU attachment after edits, including previously hidden attachments", () => {
        const { surface, device, encoder } = mockSurface();
        const data = createTextData(storage(), [{ curveSet: "font", glyphs: [{ glyphId: 1, x: 0, y: 0 }], pixelsPerFontUnit: 1 }]);
        const a = createTextLayer(data);
        const b = createTextLayer(data);
        const renderer = createTextRenderer(surface, { layers: [a, b] });
        renderer._update();
        renderer._record();
        b.visible = false;
        updateTextData(data, { update: "replaceRun", previous: 0, run: { curveSet: "font", glyphs: [{ glyphId: 1, x: 10, y: 20 }], pixelsPerFontUnit: 1 } });
        renderer._update();
        updateTextData(data, { update: "replaceRun", previous: 0, run: { curveSet: "font", glyphs: [{ glyphId: 1, x: 30, y: 40 }], pixelsPerFontUnit: 1 } });
        device.queue.writeBuffer.mockClear();
        b.visible = true;
        renderer._update();
        renderer._record();
        for (const layer of [a, b]) {
            const gpu = renderer._layerGpu.get(layer)!._draw as CpuTextLayerDraw;
            const upload = device.queue.writeBuffer.mock.calls.find(([buffer]) => buffer === gpu._instanceBuf)!;
            expect(upload[4]).toBe(12);
            expect([...new Float32Array(upload[2] as ArrayBuffer, upload[3], 2)]).toEqual([30, 40]);
        }
        expect(encoder.finish).toHaveBeenCalledTimes(2);
        disposeTextRenderer(renderer);
    });
});
