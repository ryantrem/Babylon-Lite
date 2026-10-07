/** Allocation only. Palette synchronization and base-style writes belong to callers. */

export function createStyleBuffer(device: GPUDevice, byteLength: number, label = "text-styles"): GPUBuffer {
    return device.createBuffer({
        label,
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
}
