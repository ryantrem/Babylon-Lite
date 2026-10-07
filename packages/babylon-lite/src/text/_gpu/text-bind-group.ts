/** Shared draw bindings. Resource selection, caching, and lifetime stay with the caller. */

import type { SharedAtlasGpu } from "../glyph-storage.js";

export function createTextBindGroup(
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    uniform: GPUBuffer,
    atlas: Pick<SharedAtlasGpu, "_curveTex" | "_bandTex" | "_metaBuf">,
    styles: GPUBuffer,
    label: string
): GPUBindGroup {
    return device.createBindGroup({
        label,
        layout,
        // Resource order is the contiguous group-0 ABI; the selected atlas is only borrowed.
        entries: [{ buffer: uniform }, atlas._curveTex.createView(), atlas._bandTex.createView(), { buffer: atlas._metaBuf }, { buffer: styles }].map((resource, binding) => ({
            binding,
            resource,
        })),
    });
}
