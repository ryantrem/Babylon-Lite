import { create } from "webgpu";
import { expect } from "vitest";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";

export interface NativeHarness {
    engine: EngineContext;
    device: GPUDevice;
    errors: string[];
    destroyExpected: boolean;
    close(): Promise<void>;
}

export async function nativeHarness(): Promise<NativeHarness> {
    const backend = process.env.LITE_GPU_BACKEND ?? (process.platform === "darwin" ? "metal" : "vulkan");
    if (!["metal", "vulkan", "d3d12"].includes(backend)) {
        throw new Error(`LITE_GPU_BACKEND=${backend}: use an executing Metal, Vulkan or D3D12 backend; null is forbidden.`);
    }
    let gpu: GPU | null = create([`backend=${backend}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        throw new Error(`Dawn ${backend} adapter unavailable. Install the native driver or configure software Vulkan (VK_ICD_FILENAMES); no mock/CPU fallback is allowed.`);
    }
    const device = await adapter.requestDevice();
    expect(device.createCommandEncoder()).toBeInstanceOf(GPUCommandEncoder);
    // createEngine also seeds a plain EngineContext before adding its canvas surface.
    // Direct compute uses only device/encoder/resource state; no canvas is acquired.
    const state = {
        _device: device,
        _currentEncoder: undefined,
        _currentDelta: 0,
        _cbs: [],
        _animFrameId: 0,
        _renderFn: null,
        drawCallCount: 0,
        gpuFrameTimeMs: 0,
        useHighPrecisionMatrix: false,
        useFloatingOrigin: false,
    } satisfies Partial<EngineContext>;
    const engine = state as unknown as EngineContext;
    const errors: string[] = [];
    const harness: NativeHarness = {
        engine,
        device,
        errors,
        destroyExpected: false,
        async close() {
            harness.destroyExpected = true;
            engine._disposeManagedResources?.();
            engine._disposeStorageBuffers?.();
            device.destroy();
            await device.lost;
            gpu = null;
        },
    };
    device.addEventListener("uncapturederror", (event) => {
        errors.push(event.error.message);
    });
    void device.lost.then((info) => {
        if (!harness.destroyExpected) {
            errors.push(`Unexpected device loss: ${info.reason}: ${info.message}`);
        }
    });
    console.warn(
        `Native Dawn: webgpu 0.6.1 / ${backend} / ${process.platform}-${process.arch} / Node ${process.version} / ` +
            `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.device} ${adapter.info.description}`
    );
    return harness;
}

export async function assertNativeClean(harness: NativeHarness): Promise<void> {
    if (!harness.destroyExpected) {
        await harness.device.queue.onSubmittedWorkDone();
    }
    expect(harness.errors, "Dawn validation/device errors").toEqual([]);
}
