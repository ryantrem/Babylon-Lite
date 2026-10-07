import type { EngineContext } from "../engine/engine.js";
import type { TextRenderer } from "./text-renderer.js";
import { rebuildTextRendererGpu } from "./text-renderer.js";

/** Rebuild registered text attachments through their factory seam on the replacement device.
 *  CPU blocks re-upload; resident blocks remain terminal until the caller recreates them. */
export function rebuildRegisteredTextRenderers(engine: EngineContext): void {
    for (const surface of engine.surfaces) {
        for (const context of surface._renderingContexts) {
            if (context._kind === "text-renderer") {
                rebuildTextRendererGpu(context as TextRenderer);
            }
        }
    }
}
