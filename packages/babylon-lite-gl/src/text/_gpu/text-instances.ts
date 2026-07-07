/** Lazy per-`TextData` GPU instance buffer — the WebGL2 port of the WebGPU renderer's
 *  per-layer instance buffer + dirty-range upload (`text-renderer.ts` → `uploadLayer`).
 *
 *  The buffer hangs off `TextData._gpu` (not the renderer), so every `TextLayer` that wraps
 *  the same `TextData` shares one upload. It is created from — and retains by reference —
 *  the data's `_instances` pool, which the slot allocator mutates in place; that keeps the
 *  buffer's context-restore replay (`GLVertexBuffer._data`) pointing at the live CPU state.
 *  When the slot allocator reallocates the pool (capacity grow) the pool reference changes,
 *  so the buffer is recreated from the new pool. Otherwise only the dirty
 *  `[_dirtyStart, _dirtyEnd)` instance range is re-uploaded via `bufferSubData`. */

import type { GLEngineContext } from "../../context.js";
import { createVertexBuffer, disposeBuffer, updateVertexBuffer } from "../../mesh.js";
import type { TextData, TextDataGpu } from "../text-data.js";
import { TEXT_INSTANCE_BYTES, TEXT_INSTANCE_FLOATS } from "../text-data.js";

/** Ensure `data._gpu` exists, matches `engine`, and is uploaded up to `data._version`.
 *  Returns the companion, or `null` when there is nothing to draw (`_instanceCount === 0`). */
export function ensureTextDataGpu(engine: GLEngineContext, data: TextData): TextDataGpu | null {
    if (data._instanceCount === 0) {
        return null;
    }

    let gpu = data._gpu;
    if (gpu && gpu.engine !== engine) {
        disposeBuffer(gpu.engine, gpu.instanceBuf);
        gpu = null;
        data._gpu = null;
    }

    if (!gpu) {
        // `createVertexBuffer` uploads the whole pool, so the buffer already holds the current
        // instances — no separate first upload needed.
        const buf = createVertexBuffer(engine, data._instances, true);
        gpu = {
            engine,
            instanceBuf: buf,
            instanceBufCapacity: data._instances.length / TEXT_INSTANCE_FLOATS,
            uploadedVersion: data._version,
        };
        data._gpu = gpu;
        data._dirtyStart = 0;
        data._dirtyEnd = 0;
        return gpu;
    }

    // Pool grew → the allocator swapped `_instances` for a larger array. Recreate the buffer
    // from the new pool (which re-uploads everything) and reset the dirty range.
    if (data._instanceCount > gpu.instanceBufCapacity) {
        disposeBuffer(engine, gpu.instanceBuf);
        gpu.instanceBuf = createVertexBuffer(engine, data._instances, true);
        gpu.instanceBufCapacity = data._instances.length / TEXT_INSTANCE_FLOATS;
        gpu.uploadedVersion = data._version;
        data._dirtyStart = 0;
        data._dirtyEnd = 0;
        return gpu;
    }

    if (gpu.uploadedVersion !== data._version) {
        if (data._dirtyEnd > data._dirtyStart) {
            const view = data._instances.subarray(data._dirtyStart * TEXT_INSTANCE_FLOATS, data._dirtyEnd * TEXT_INSTANCE_FLOATS);
            updateVertexBuffer(engine, gpu.instanceBuf, view, data._dirtyStart * TEXT_INSTANCE_BYTES);
        } else {
            // Version moved without a localized dirty range (e.g. a full compaction) → upload all.
            updateVertexBuffer(engine, gpu.instanceBuf, data._instances, 0);
        }
        gpu.uploadedVersion = data._version;
        data._dirtyStart = 0;
        data._dirtyEnd = 0;
    }
    return gpu;
}
