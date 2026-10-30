import { BU } from "../engine/gpu-flags.js";
import { _getStorageBufferHandle } from "../resource/storage-buffer.js";
import type { StorageBuffer } from "../resource/storage-buffer.js";
import { _recordComputeTasks } from "./compute-task.js";
import type { ComputeTask } from "./compute-task.js";
import { _deferCommandBuffer } from "./deferred-submit.js";

/** @internal Resident status copy; mapping starts only after the queued command buffer submits.
 *  Queue order already follows earlier deferred work, so recording does not flush it; callers flush
 *  their own pending work before writing inputs. `owner` scopes later `_flushDeferredWorkFor` calls. */
export function submitDeferredComputeTaskResult<T>(
    tasks: readonly ComputeTask[],
    buffer: StorageBuffer,
    byteLength: number,
    consume: (bytes: ArrayBuffer) => T,
    staging: GPUBuffer,
    owner?: object
): Promise<T> {
    const engine = buffer._engine;
    const encoder = recordResult(tasks, buffer, byteLength, staging, false);
    return new Promise<T>((resolve, reject) => {
        _deferCommandBuffer(
            engine,
            encoder.finish(),
            () => {
                try {
                    staging
                        .mapAsync(GPUMapMode.READ, 0, byteLength)
                        .then(() => {
                            try {
                                return consume(staging.getMappedRange(0, byteLength));
                            } finally {
                                staging.unmap();
                            }
                        })
                        .then(resolve, reject);
                    engine._gpuTaskTimerResolve?.(encoder);
                } catch (error) {
                    reject(error instanceof Error ? error : new Error(String(error)));
                }
            },
            reject,
            owner
        );
    });
}

/** @internal Submit prepared work with a private status copy, without waiting for GPU completion. */
export function submitComputeTaskResult<T>(
    tasks: readonly ComputeTask[],
    buffer: StorageBuffer,
    byteLength: number,
    consume: (bytes: ArrayBuffer) => T,
    staging: GPUBuffer
): Promise<T> {
    const engine = buffer._engine;
    const encoder = recordResult(tasks, buffer, byteLength, staging);
    engine._device.queue.submit([encoder.finish()]);
    engine._gpuTaskTimerResolve?.(encoder);
    return staging.mapAsync(GPUMapMode.READ, 0, byteLength).then(() => {
        try {
            return consume(staging.getMappedRange(0, byteLength));
        } finally {
            staging.unmap();
        }
    });
}

function recordResult(tasks: readonly ComputeTask[], buffer: StorageBuffer, byteLength: number, staging: GPUBuffer, flushPending = true): GPUCommandEncoder {
    const engine = buffer._engine;
    const source = _getStorageBufferHandle(engine, buffer);
    if (!buffer._writable || engine._currentEncoder || staging.mapState !== "unmapped") {
        throw new Error("submitComputeTaskResult requires an idle encoder and unmapped staging allocation.");
    }
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || (byteLength & 3) !== 0 || byteLength > buffer.byteLength || byteLength > staging.size) {
        throw new Error("submitComputeTaskResult requires a positive aligned range within output and staging capacity.");
    }
    if (tasks.length === 0 || tasks.some((task) => task.engine !== engine)) {
        throw new Error("submitComputeTaskResult requires tasks from the output's engine.");
    }
    if (flushPending) {
        engine._flushPendingWork?.(engine);
    }
    const encoder = engine._device.createCommandEncoder({ label: "compute-result" });
    _recordComputeTasks(tasks, encoder);
    encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
    return encoder;
}

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
    engine._flushPendingWork?.(engine);
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
