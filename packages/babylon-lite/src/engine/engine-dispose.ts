import { stopEngine, type EngineContext } from "./engine.js";
import { disposeGpuResourceRetirements } from "./gpu-resource-retirement.js";

/** Release engine-owned resources and detach from its device. Explicitly created devices
 *  remain alive; private convenience devices are destroyed after their last engine.
 *  Rendering contexts own their frame graphs/targets and must be disposed separately. */
export function disposeEngine(engine: EngineContext): void {
    if (engine._disposed) {
        return;
    }
    engine._disposed = true;
    engine._disposeDeviceRecovery?.();
    // Claim retirement batches before stopEngine fences them: the device is destroyed below.
    disposeGpuResourceRetirements(engine);
    stopEngine(engine);
    const surfaces = engine._surfaces;
    for (const surface of surfaces) {
        surface._renderingContexts.length = 0;
        surface._ro?.disconnect();
        surface._context.unconfigure();
    }
    surfaces.length = 0;
    try {
        engine._disposeManagedResources?.();
        engine._disposeStorageBuffers?.();
    } finally {
        const detach = engine.device?._detach;
        if (detach) {
            detach(engine);
        } else {
            engine._device.destroy();
        }
    }
}
