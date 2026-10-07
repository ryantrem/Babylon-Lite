import { BU } from "../engine/gpu-flags.js";
import { _getStorageBufferHandle } from "../resource/storage-buffer.js";
import type { StorageBuffer } from "../resource/storage-buffer.js";
import { _recordComputeTasks } from "./compute-task.js";
import type { ComputeTask } from "./compute-task.js";

/** @internal Complete prepared tasks and materialize private, exclusively owned output while mapped. */
export async function readComputeTaskResult<T>(tasks: readonly ComputeTask[], buffer: StorageBuffer, byteLength: number, consume: (bytes: ArrayBuffer) => T): Promise<T> {
    const engine = buffer._engine;
    const source = _getStorageBufferHandle(engine, buffer);
    if (!buffer._writable || buffer._readPending || engine._currentEncoder) {
        throw new Error("readComputeTaskResult requires an idle, writable output allocation.");
    }
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || (byteLength & 3) !== 0 || byteLength > buffer.byteLength) {
        throw new Error("readComputeTaskResult requires a positive aligned range within output capacity.");
    }
    if (tasks.length === 0 || tasks.some((task) => task.engine !== engine)) {
        throw new Error("readComputeTaskResult requires tasks from the output's engine.");
    }
    const device = engine._device;
    if (buffer._readback && (buffer._readbackDevice !== device || buffer._readback.size < byteLength)) {
        buffer._readback.destroy();
        buffer._readback = undefined;
    }
    const staging = (buffer._readback ??= device.createBuffer({ label: "compute-result-readback", size: byteLength, usage: BU.COPY_DST | BU.MAP_READ }));
    buffer._readbackDevice = device;
    const encoder = device.createCommandEncoder({ label: "compute-result" });
    _recordComputeTasks(tasks, encoder);
    encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    engine._gpuTaskTimerResolve?.(encoder);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    try {
        return consume(staging.getMappedRange(0, byteLength));
    } finally {
        staging.unmap();
    }
}
