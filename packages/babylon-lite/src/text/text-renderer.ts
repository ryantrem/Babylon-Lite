/** Standalone shared text rendering core: placement, ordering, pipelines and cached bundles.
 *  Draw preparation is installed by each layer factory; neither adapter is imported here. */

import { registerRenderingContext, unregisterRenderingContext } from "../engine/engine.js";
import type { RenderingContext } from "../engine/engine.js";
import type { SurfaceContext } from "../engine/surface.js";
import { createEmptyUniformBuffer } from "../resource/empty-uniform-buffer.js";
import type { TextLayer } from "./text-layer-cpu.js";
import type { ComputeTextLayer } from "./compute-text-layer.js";
import type { TextLayerDraw, TextLayerPlacement } from "./text-layer.js";
import { getOrCreateTextPipeline, getTextPipelineCache } from "./_gpu/text-pipeline.js";

/** Either CPU or resident text, with common placement and distinct data ownership. */
export type TextRendererLayer = TextLayer | ComputeTextLayer;

const KIND = "text-renderer" as const;
const TEXT_UBO_BYTES = 96;

/** Options for a standalone text renderer that draws directly to a surface. */
export interface TextRendererOptions {
    layers: readonly TextRendererLayer[];
    /** Default true. Set false for HUD overlays to preserve existing scene color. */
    clear?: boolean;
    /** Default `{ r: 0, g: 0, b: 0, a: 1 }`. */
    clearValue?: GPUColorDict;
}

/** Standalone rendering context that draws sorted CPU and resident text layers. */
export interface TextRenderer extends RenderingContext {
    /** @internal */
    readonly _kind: typeof KIND;
    readonly layers: readonly TextRendererLayer[];
    /** @internal Mutable alias of layers. */
    _layers: TextRendererLayer[];
    /** @internal */
    readonly _surface: SurfaceContext;
    /** @internal Per-attachment GPU resources. */
    _layerGpu: Map<TextRendererLayer, LayerGpu>;
    /** @internal */
    _targetWidth: number;
    /** @internal */
    _targetHeight: number;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _clear: boolean;
    /** @internal */
    _visibleBundles: GPURenderBundle[];
}

interface LayerGpu {
    _layer: TextRendererLayer;
    _textU: GPUBuffer;
    _draw: TextLayerDraw;
    _pipeline: GPURenderPipeline | null;
    _variantPipeline: GPURenderPipeline | null;
    _bindGroupLayout: GPUBindGroupLayout | null;
    _uploadedViewportW: number;
    _uploadedViewportH: number;
    _uploadedGammaInv: number;
    _uploadedOpacity: number;
    _colorUploaded: boolean;
    _lastMvpInputs: Float32Array;
    _mvpUploaded: boolean;
    _renderBundle: GPURenderBundle | null;
    _bundleVersion: number;
    _bundleDrawCalls: number;
}

const _mvpScratch = new Float32Array(16);

function buildLayerMvp(layer: TextLayerPlacement, targetW: number, targetH: number, out: Float32Array): void {
    const s = layer.scale;
    const r = layer.rotationRad;
    const cr = Math.cos(r);
    const sr = Math.sin(r);
    const px = layer.positionPx.x;
    const py = layer.positionPx.y;
    // Font Y-up → canvas Y-down placement → orthographic clip coordinates, column-major.
    const cx = (2 * s) / targetW;
    const cy = (2 * s) / targetH;
    out.fill(0);
    out[0] = cx * cr;
    out[1] = -cy * sr;
    out[4] = cx * sr;
    out[5] = cy * cr;
    out[10] = 1;
    out[12] = (2 * px) / targetW - 1;
    out[13] = 1 - (2 * py) / targetH;
    out[15] = 1;
}

function ensureLayerGpu(rr: TextRenderer, layer: TextRendererLayer): LayerGpu {
    let lg = rr._layerGpu.get(layer);
    if (lg) {
        return lg;
    }
    const engine = rr._surface.engine;
    const uniform = createEmptyUniformBuffer(engine, TEXT_UBO_BYTES, "text-layer-ubo");
    let draw: TextLayerDraw;
    try {
        draw = layer._createDraw(engine, uniform);
    } catch (error) {
        uniform.destroy();
        throw error;
    }
    lg = {
        _layer: layer,
        _textU: uniform,
        _draw: draw,
        _pipeline: null,
        _variantPipeline: null,
        _bindGroupLayout: null,
        _uploadedViewportW: 0,
        _uploadedViewportH: 0,
        _uploadedGammaInv: 0,
        _uploadedOpacity: 0,
        _colorUploaded: false,
        _lastMvpInputs: new Float32Array(6),
        _mvpUploaded: false,
        _renderBundle: null,
        _bundleVersion: -1,
        _bundleDrawCalls: 0,
    };
    rr._layerGpu.set(layer, lg);
    return lg;
}

function uploadLayerUniforms(device: GPUDevice, rr: TextRenderer, lg: LayerGpu): void {
    const layer = lg._layer;
    const W = rr._targetWidth;
    const H = rr._targetHeight;
    const mi = lg._lastMvpInputs;
    if (!lg._mvpUploaded || mi[0] !== layer.positionPx.x || mi[1] !== layer.positionPx.y || mi[2] !== layer.rotationRad || mi[3] !== layer.scale || mi[4] !== W || mi[5] !== H) {
        buildLayerMvp(layer, W, H, _mvpScratch);
        device.queue.writeBuffer(lg._textU, 0, _mvpScratch.buffer as ArrayBuffer, _mvpScratch.byteOffset, 64);
        mi[0] = layer.positionPx.x;
        mi[1] = layer.positionPx.y;
        mi[2] = layer.rotationRad;
        mi[3] = layer.scale;
        mi[4] = W;
        mi[5] = H;
        lg._mvpUploaded = true;
    }
    if (lg._uploadedViewportW !== W || lg._uploadedViewportH !== H) {
        const vp = new Float32Array([W, H, 0, 0]);
        device.queue.writeBuffer(lg._textU, 64, vp.buffer as ArrayBuffer, vp.byteOffset, 16);
        lg._uploadedViewportW = W;
        lg._uploadedViewportH = H;
    }
    const gammaInv = layer.coverageGamma > 0 ? 1 / layer.coverageGamma : 1;
    if (!lg._colorUploaded || !Object.is(lg._uploadedGammaInv, gammaInv) || !Object.is(lg._uploadedOpacity, layer.opacity)) {
        const col = new Float32Array([gammaInv, 1, 1, layer.opacity]);
        device.queue.writeBuffer(lg._textU, 80, col.buffer as ArrayBuffer, col.byteOffset, 16);
        lg._uploadedGammaInv = gammaInv;
        lg._uploadedOpacity = layer.opacity;
        lg._colorUploaded = true;
    }
}

function disposeLayerGpu(lg: LayerGpu): void {
    lg._renderBundle = null;
    lg._draw._dispose();
    lg._textU.destroy();
}

function compareLayers(a: TextRendererLayer, b: TextRendererLayer): number {
    return a.order - b.order;
}

/** Create a standalone renderer for CPU, resident, or mixed text layers. */
export function createTextRenderer(surface: SurfaceContext, opts: TextRendererOptions): TextRenderer {
    const layers = opts.layers.slice();
    const rr: TextRenderer = {
        _kind: KIND,
        _surface: surface,
        _layerGpu: new Map(),
        _targetWidth: surface.canvas.width,
        _targetHeight: surface.canvas.height,
        _disposed: false,
        _clear: opts.clear ?? true,
        _visibleBundles: [],
        layers,
        _layers: layers,
        clearColor: opts.clearValue ?? { r: 0, g: 0, b: 0, a: 1 },
        _drawCallsPre: 0,
        _update(): void {
            textRendererUpdate(rr);
        },
        _record(): number {
            return textRendererRecord(rr);
        },
    };
    return rr;
}

function textRendererUpdate(rr: TextRenderer): void {
    if (rr._disposed) {
        return;
    }
    rr._targetWidth = rr._surface.canvas.width;
    rr._targetHeight = rr._surface.canvas.height;
    if (rr._layers.length > 1) {
        rr._layers.sort(compareLayers);
    }
    const { _pipeline: pipeline, _variantPipeline: variantPipeline, _cache: cache } = getOrCreateTextPipeline(rr._surface.engine, rr._surface.format, 1, null, false);
    for (const layer of rr._layers) {
        if (!layer.visible) {
            continue;
        }
        const lg = ensureLayerGpu(rr, layer);
        if (lg._pipeline !== pipeline || lg._variantPipeline !== variantPipeline) {
            lg._pipeline = pipeline;
            lg._variantPipeline = variantPipeline;
            lg._renderBundle = null;
        }
        lg._bindGroupLayout = cache._bindGroupLayout;
        lg._draw._prepare(cache._bindGroupLayout);
        uploadLayerUniforms(rr._surface.engine._device, rr, lg);
    }
}

function textRendererRecord(rr: TextRenderer): number {
    if (rr._disposed) {
        return 0;
    }
    const eng = rr._surface.engine;
    const pass = eng._currentEncoder.beginRenderPass({
        colorAttachments: [{ view: rr._surface.scRT._colorView!, clearValue: rr.clearColor, loadOp: rr._clear ? "clear" : "load", storeOp: "store" }],
    });
    const quad = getTextPipelineCache(eng)._quadVertexBuffer;
    const bundles = rr._visibleBundles;
    bundles.length = 0;
    let drawCalls = 0;
    for (const layer of rr._layers) {
        if (!layer.visible) {
            continue;
        }
        const lg = rr._layerGpu.get(layer);
        if (!lg || !lg._pipeline || !lg._bindGroupLayout) {
            continue;
        }
        // Re-check resource identities before replay: a sibling can grow a shared atlas
        // after update, and a terminal resident block must never replay a borrowed bundle.
        lg._draw._prepare(lg._bindGroupLayout);
        if (!lg._draw._active) {
            lg._renderBundle = null;
            continue;
        }
        if (!lg._renderBundle || lg._bundleVersion !== lg._draw._version) {
            const encoder = eng._device.createRenderBundleEncoder({ colorFormats: [rr._surface.format], sampleCount: 1 });
            encoder.setPipeline(lg._pipeline);
            lg._bundleDrawCalls = lg._draw._record(encoder, quad, lg._pipeline, lg._variantPipeline!);
            lg._renderBundle = encoder.finish();
            lg._bundleVersion = lg._draw._version;
        }
        bundles.push(lg._renderBundle);
        drawCalls += lg._bundleDrawCalls;
    }
    if (bundles.length > 0) {
        pass.executeBundles(bundles);
    }
    pass.end();
    return drawCalls;
}

/** @internal Recovery uses each attachment's factory rather than assuming CPU data. */
export function rebuildTextRendererGpu(renderer: TextRenderer): void {
    for (const lg of renderer._layerGpu.values()) {
        disposeLayerGpu(lg);
    }
    renderer._layerGpu.clear();
    renderer._visibleBundles.length = 0;
    renderer._update();
}

/** Add either layer kind; order is sorted during update. */
export function addTextRendererLayer(tr: TextRenderer, layer: TextRendererLayer): void {
    if (tr._disposed) {
        throw new Error("TextRenderer has been disposed.");
    }
    if (!tr._layers.includes(layer)) {
        tr._layers.push(layer);
    }
}

/** Remove a layer and release only its attachment-owned resources. */
export function removeTextRendererLayer(tr: TextRenderer, layer: TextRendererLayer): boolean {
    const index = tr._layers.indexOf(layer);
    if (index < 0) {
        return false;
    }
    tr._layers.splice(index, 1);
    const lg = tr._layerGpu.get(layer);
    if (lg) {
        disposeLayerGpu(lg);
        tr._layerGpu.delete(layer);
    }
    tr._visibleBundles.length = 0;
    return true;
}

/** Register a text renderer for per-frame updates and recording. */
export function registerTextRenderer(tr: TextRenderer): void {
    registerRenderingContext(tr._surface, tr);
}

/** Unregister without disposing layer data. */
export function unregisterTextRenderer(tr: TextRenderer): void {
    unregisterRenderingContext(tr._surface, tr);
}

/** Dispose attachment-owned resources. Both kinds of layer data and atlases remain caller-owned. */
export function disposeTextRenderer(tr: TextRenderer): void {
    if (tr._disposed) {
        return;
    }
    unregisterTextRenderer(tr);
    tr._disposed = true;
    for (const lg of tr._layerGpu.values()) {
        disposeLayerGpu(lg);
    }
    tr._layerGpu.clear();
    tr._layers.length = 0;
    tr._visibleBundles.length = 0;
}
