import type { EngineContext } from "../engine/engine.js";

/** Initial placement and compositing options shared by CPU and resident text layers. */
export interface TextLayerOptions {
    /** Top-left origin in canvas pixels. Default (0, 0). */
    readonly positionPx?: { readonly x: number; readonly y: number };
    /** Z-axis rotation about the origin, in radians. Default 0. */
    readonly rotationRad?: number;
    /** Uniform scale. Default 1. */
    readonly scale?: number;
    /** Lower orders draw first. Default 0. */
    readonly order?: number;
    /** Alpha multiplier in [0, 1]. Default 1. */
    readonly opacity?: number;
    /** Coverage is raised to `1/coverageGamma` before compositing. Values greater than
     *  1 thicken anti-aliased edges on linear-blended sRGB surfaces. Default 1. */
    readonly coverageGamma?: number;
    /** Default true. */
    readonly visible?: boolean;
}

/** Mutable pixel-space placement shared by CPU and resident layers. */
export interface TextLayerPlacement {
    positionPx: { x: number; y: number };
    rotationRad: number;
    scale: number;
    order: number;
    opacity: number;
    /** Coverage gamma for anti-aliased edges. Default 1. */
    coverageGamma: number;
    visible: boolean;
    /** @internal */
    _version: number;
    /** @internal Creates resources for one layer/renderer attachment, never for the data block. */
    _createDraw(engine: EngineContext, uniform: GPUBuffer): TextLayerDraw;
}

/** @internal Prepared draws own only attachment resources; borrowed data and atlases remain caller-owned. */
export interface TextLayerDraw {
    /** @internal */
    _active: boolean;
    /** @internal */
    _version: number;
    /** @internal */
    _prepare(layout: GPUBindGroupLayout): void;
    /** @internal */
    _record(encoder: GPURenderBundleEncoder, quad: GPUBuffer, base: GPURenderPipeline, variant: GPURenderPipeline): number;
    /** @internal */
    _dispose(): void;
}

/** @internal Shared placement initialization, with no adapter or GPU allocation. */
export function createTextLayerPlacement(options?: TextLayerOptions): Omit<TextLayerPlacement, "_createDraw"> {
    return {
        positionPx: { x: options?.positionPx?.x ?? 0, y: options?.positionPx?.y ?? 0 },
        rotationRad: options?.rotationRad ?? 0,
        scale: options?.scale ?? 1,
        order: options?.order ?? 0,
        opacity: options?.opacity ?? 1,
        coverageGamma: options?.coverageGamma ?? 1,
        visible: options?.visible ?? true,
        _version: 0,
    };
}

/** Update either kind of text layer's pixel position. */
export function setTextLayerPosition(layer: TextLayerPlacement, x: number, y: number): void {
    layer.positionPx.x = x;
    layer.positionPx.y = y;
    layer._version++;
}
