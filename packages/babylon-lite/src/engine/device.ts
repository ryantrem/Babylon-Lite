import type { EngineContext, EngineDeviceOptions } from "./engine.js";

declare const deviceContextBrand: unique symbol;

/** Canvas-independent GPU acquisition settings, fixed for the device's lifetime. */
export interface DeviceOptions {
    powerPreference?: GPUPowerPreference;
    requiredFeatures?: readonly GPUFeatureName[];
    requiredLimits?: Record<string, GPUSize64 | undefined>;
}

/** Shared GPU ownership boundary. Native WebGPU handles remain internal. */
export interface DeviceContext {
    /** Opaque nominal brand; contexts must be created by createDevice. */
    readonly [deviceContextBrand]: true;
    readonly features: GPUSupportedFeatures;
    readonly limits: GPUSupportedLimits;
    /** @internal */
    _device: GPUDevice;
    /** @internal Creation policy retained for device-lost recovery. */
    _adapterOptions: GPURequestAdapterOptions;
    /** @internal */
    _requiredLimits?: Record<string, GPUSize64 | undefined>;
    /** @internal Present only on caller-owned contexts from createDevice. Engine-private contexts have one implicit engine. */
    _engines?: EngineContext[];
    /** @internal */
    _disposed?: boolean;
    /** @internal Reject attachment to a device lost before engine initialization. */
    _lost?: boolean;
    /** @internal Shared-device seams installed only by createDevice. */
    _attach?: (engine: EngineContext) => void;
    /** @internal */
    _detach?: (engine: EngineContext) => void;
    /** @internal Installed only by opt-in recovery. */
    _recovery?: Promise<void>;
}

let _adapterOptionsHook: (() => GPURequestAdapterOptions) | null = null;

/** @internal Install adapter options for subsequent device creation (WebXR opt-in). */
export function _installAdapterOptions(hook: () => GPURequestAdapterOptions): void {
    _adapterOptionsHook = hook;
}

/** @internal Adapter options for creation and pre-creation capability queries. */
export function _getAdapterOptions(): GPURequestAdapterOptions {
    return _adapterOptionsHook ? _adapterOptionsHook() : {};
}

const OPTIONAL_DEVICE_FEATURES = [
    "float32-filterable",
    "texture-compression-astc",
    "texture-compression-bc",
    "texture-compression-etc2",
    "texture-compression-unaligned" as GPUFeatureName,
    "timestamp-query",
    "primitive-index",
] as const satisfies readonly GPUFeatureName[];

/** @internal Select Lite's opportunistic features supported by the adapter. */
export function _getSupportedDeviceFeatures(adapter: GPUAdapter): GPUFeatureName[] {
    return OPTIONAL_DEVICE_FEATURES.filter((feature) => adapter.features.has(feature));
}

/** @internal Acquire an engine-private device. Shared-device seams are added only by createDevice. */
export async function _acquireDevice(
    adapterOptions: GPURequestAdapterOptions,
    requiredLimits?: Record<string, GPUSize64 | undefined>,
    selectFeatures: (adapter: GPUAdapter) => GPUFeatureName[] = _getSupportedDeviceFeatures
): Promise<DeviceContext> {
    const adapter = await navigator.gpu.requestAdapter(adapterOptions);
    if (!adapter) {
        throw new Error("WebGPU adapter not available");
    }
    // requestDevice rejects unsupported required features with a TypeError naming the feature.
    const device = await adapter.requestDevice({ requiredFeatures: selectFeatures(adapter), requiredLimits });
    return {
        features: device.features,
        limits: device.limits,
        _device: device,
        _adapterOptions: adapterOptions,
        _requiredLimits: requiredLimits,
    } as Partial<DeviceContext> as DeviceContext;
}

/** @internal Merge explicitly required features with Lite's opportunistic features. */
export function _withRequiredFeatures(required: readonly GPUFeatureName[]): (adapter: GPUAdapter) => GPUFeatureName[] {
    return (adapter) => [...new Set([..._getSupportedDeviceFeatures(adapter), ...required])];
}

/** Acquire a GPU device without a canvas or engine. Dispose it after all attached engines. */
export async function createDevice(options?: DeviceOptions): Promise<DeviceContext> {
    const adapterOptions = { powerPreference: options?.powerPreference ?? "high-performance", ..._getAdapterOptions() } satisfies GPURequestAdapterOptions;
    const requiredLimits = options?.requiredLimits ? { ...options.requiredLimits } : undefined;
    const context = await _acquireDevice(adapterOptions, requiredLimits, _withRequiredFeatures(options?.requiredFeatures?.slice() ?? []));
    Object.assign(context, { _engines: [], _attach: attachEngine, _detach: detachEngine } satisfies Partial<DeviceContext>);
    _watchDeviceLoss(context);
    return context;
}

function attachEngine(engine: EngineContext): void {
    const context = engine.device!;
    const options = engine._options as EngineDeviceOptions;
    if (options.requiredLimits !== undefined || options.requiredFeatures !== undefined) {
        engine._context.unconfigure();
        throw new Error("Configure requiredFeatures and requiredLimits on createDevice, not an engine using an existing device.");
    }
    if (context._disposed || context._lost || context._recovery) {
        engine._context.unconfigure();
        throw new Error("Cannot create an engine from a disposed, lost, or recovering device.");
    }
    context._engines!.push(engine);
}

function detachEngine(engine: EngineContext): void {
    engine._disposeGpuEvents?.();
    const engines = engine.device!._engines!;
    engines.splice(engines.indexOf(engine), 1);
}

/** @internal Track native device identity across preparation and replacement. */
export function _watchDeviceLoss(context: DeviceContext): void {
    const device = context._device;
    void device.lost.then(() => {
        if (context._device === device) {
            context._lost = true;
        }
    });
}
