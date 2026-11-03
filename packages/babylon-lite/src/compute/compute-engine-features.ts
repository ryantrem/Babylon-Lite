import type { EngineContext, EngineDeviceOptions, EngineOptions, RenderCanvas } from "../engine/engine.js";
import { createEngine } from "../engine/engine.js";
import { _acquireDevice, _getAdapterOptions, _withRequiredFeatures } from "../engine/device.js";

/** Engine options that explicitly require additional WebGPU device features. */
export interface EngineFeatureOptions extends EngineOptions {
    readonly requiredFeatures: readonly GPUFeatureName[];
}

/** Create an engine after requesting explicit WebGPU features; WebGPU rejects unsupported ones. */
export async function createEngineWithFeatures(canvas: RenderCanvas, options: EngineFeatureOptions): Promise<EngineContext> {
    const { requiredFeatures, requiredLimits, ...engineOptions } = options;
    // The context lacks createDevice's shared-device seams, so the engine owns and destroys it.
    const device = await _acquireDevice({ powerPreference: "high-performance", ..._getAdapterOptions() }, requiredLimits, _withRequiredFeatures(requiredFeatures));
    return createEngine(canvas, { ...engineOptions, device } as EngineDeviceOptions);
}
