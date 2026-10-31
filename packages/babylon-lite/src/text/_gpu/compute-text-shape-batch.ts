/** Cross-object shape batching: independent text objects that share a plan and kernel tier admit
 *  their runs into one open batch, shaped by a single dispatch at deferred-flush time. Each workgroup
 *  shapes one run; serial per-object dispatches would otherwise leave the GPU idle. The kernel reads
 *  its record stride from params, so one batch shapes at its largest slice capacity and copies every
 *  run's row prefix into its object's exact stride. A batch holding one slice shapes directly into
 *  that object's buffer: the copy adds GPU latency that dominates small single-object updates. */

import type { EngineContext } from "../../engine/engine.js";
import { createComputeBindingSet } from "../../compute/compute-bindings.js";
import { createComputeDispatch, setComputeDispatchSize } from "../../compute/compute-dispatch.js";
import type { ComputeDispatch } from "../../compute/compute-dispatch.js";
import { disposeComputeShader, prepareComputeShader } from "../../compute/compute-shader.js";
import type { ComputeShader } from "../../compute/compute-shader.js";
import { _recordComputeTasks, addComputeDispatch, createComputeTask } from "../../compute/compute-task.js";
import type { ComputeTask } from "../../compute/compute-task.js";
import { _addDeferredOwner, _deferCommandBuffer } from "../../compute/deferred-submit.js";
import { _getStorageBufferHandle, createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import type { ComputeTextPlan } from "../compute-text-plan.js";
import { COMPUTE_TEXT_HEADER_WORDS, COMPUTE_TEXT_RECORD_WORDS, computeTextVariantKey, computeTextVariantTier, createComputeTextShapeShader } from "./compute-text-shape.js";
import type { ComputeTextShapeWords } from "./compute-text-shape.js";

interface ShapeSlice {
    readonly shaped: StorageBuffer;
    /** Packed run words: `runCount` absolute origins, then each run's payload. */
    readonly runs: Uint32Array;
    readonly runCount: number;
    readonly outputRun: number;
    readonly capacity: number;
    readonly scratchPerRun: number;
}

interface BatchKernel {
    /** `computeTextVariantKey` prefix; resident shaping shares kernels on the same key. */
    readonly family: string;
    /** Workgroup scratch tier; zero selects global scratch. */
    readonly tier: number;
    readonly shader: ComputeShader;
    readonly ready: Promise<void>;
}

/** @internal Plan-owned resources and the open slice list for one shared kernel. */
export interface ComputeTextShapeBatch {
    readonly engine: EngineContext;
    readonly kernel: BatchKernel;
    /** Plan-invariant params; run count, capacity and output run are replaced per flush. */
    readonly params: Uint32Array;
    paramsBuffer: StorageBuffer;
    program: StorageBuffer;
    tables: StorageBuffer;
    runs: StorageBuffer;
    scratch: StorageBuffer;
    arena: StorageBuffer;
    task: ComputeTask | null;
    dispatch: ComputeDispatch | null;
    /** Single-slice dispatch bound to that slice's `shaped` buffer; rebuilt when the target changes. */
    direct: { readonly output: StorageBuffer; readonly task: ComputeTask; readonly dispatch: ComputeDispatch } | null;
    open: ShapeSlice[] | null;
}

interface BatchEngineState {
    readonly device: GPUDevice;
    readonly kernels: BatchKernel[];
    readonly plans: WeakMap<ComputeTextPlan, ComputeTextShapeBatch[]>;
}

let statesByEngine: WeakMap<EngineContext, BatchEngineState> | null = null;

/** @internal Resolve the batch for `prepared`'s plan on a kernel whose tier serves its capacity. Kernels
 *  are shared across plans and, like resident shaping, a retained larger tier serves smaller capacities.
 *  Reserves batch buffers for the open slices plus this one, so admission rather than flush grows them. */
export async function prepareComputeTextShapeBatch(engine: EngineContext, plan: ComputeTextPlan, prepared: ComputeTextShapeWords): Promise<ComputeTextShapeBatch> {
    statesByEngine ??= new WeakMap();
    let state = statesByEngine.get(engine);
    if (state && state.device !== engine._device) {
        state.kernels.forEach((kernel) => disposeComputeShader(kernel.shader));
        state = undefined;
    }
    if (!state) {
        state = { device: engine._device, kernels: [], plans: new WeakMap() };
        statesByEngine.set(engine, state);
    }
    const storageLimit = engine._device.limits.maxComputeWorkgroupStorageSize;
    const family = computeTextVariantKey(prepared.variant, 0).slice(0, -1);
    let kernel = state.kernels.find(
        (candidate) => candidate.family === family && computeTextVariantTier(prepared.capacity, prepared.variant, storageLimit, candidate.tier) === candidate.tier
    );
    if (!kernel) {
        const tier = computeTextVariantTier(prepared.capacity, prepared.variant, storageLimit);
        const shader = createComputeTextShapeShader(engine, tier || prepared.capacity, prepared.variant);
        kernel = { family, tier, shader, ready: prepareComputeShader(shader) };
        state.kernels.push(kernel);
    }
    let batches = state.plans.get(plan);
    if (!batches) {
        batches = [];
        state.plans.set(plan, batches);
    }
    let batch = batches.find((candidate) => candidate.kernel === kernel);
    if (!batch) {
        batch = {
            engine,
            kernel,
            params: prepared.params.slice(),
            paramsBuffer: createStorageBuffer(engine, prepared.params.byteLength, { label: "compute-text-batch-params" }),
            program: createStorageBuffer(engine, prepared.programWords, { label: "compute-text-batch-program" }),
            tables: createStorageBuffer(engine, prepared.tables, { label: "compute-text-batch-tables" }),
            runs: createStorageBuffer(engine, 256, { label: "compute-text-batch-runs" }),
            scratch: createStorageBuffer(engine, 256, { label: "compute-text-batch-scratch", writable: true }),
            arena: createStorageBuffer(engine, 256, { label: "compute-text-batch-arena", writable: true }),
            task: null,
            dispatch: null,
            direct: null,
            open: null,
        };
        batches.push(batch);
    }
    await kernel.ready;
    reserve(batch, [...(batch.open ?? []), toSlice(batch.arena, prepared)]);
    return batch;
}

/** @internal Admit one object's prepared group. Each run's header and records land in `shaped` at
 *  the group's output run and `prepared.capacity` stride, before any command the caller defers
 *  afterwards. `prepared.capacity` must bound the group's output (as `computeTextShapeCapacity` does);
 *  a larger count stays in the header, so the pack still reports overflow. `owner` must flush through
 *  `_flushDeferredWorkFor` before writing or replacing `shaped`. */
export function addComputeTextShapeSlice(batch: ComputeTextShapeBatch, owner: object, prepared: ComputeTextShapeWords, shaped: StorageBuffer): void {
    const engine = batch.engine;
    if (prepared.params[0] === 0) {
        throw new Error("addComputeTextShapeSlice: empty slice");
    }
    const slice = toSlice(shaped, prepared);
    if (batch.open && !fits(batch, [...batch.open, slice])) {
        engine._flushPendingWork?.(engine);
    }
    if (batch.open) {
        batch.open.push(slice);
        _addDeferredOwner(engine, owner);
        return;
    }
    const slices = [slice];
    let encoder: GPUCommandEncoder | undefined;
    _deferCommandBuffer(
        engine,
        () => {
            if (batch.open === slices) {
                batch.open = null;
            }
            encoder = engine._device.createCommandEncoder({ label: "compute-text-shape-batch" });
            recordBatch(batch, slices, encoder);
            return encoder.finish();
        },
        () => engine._gpuTaskTimerResolve?.(encoder),
        () => {
            if (batch.open === slices) {
                batch.open = null;
            }
        },
        owner
    );
    batch.open = slices;
}

function toSlice(shaped: StorageBuffer, prepared: ComputeTextShapeWords): ShapeSlice {
    const runCount = prepared.params[0]!;
    return {
        shaped,
        runs: prepared.runs,
        runCount,
        outputRun: prepared.params[6]!,
        capacity: prepared.capacity,
        scratchPerRun: prepared.scratchBytes / Math.max(1, runCount),
    };
}

/** Runs, largest capacity and largest per-run scratch across `slices`. */
function extent(slices: readonly ShapeSlice[]): { runs: number; capacity: number; scratchPerRun: number } {
    let runs = 0;
    let capacity = 1;
    let scratchPerRun = 0;
    for (const slice of slices) {
        runs += slice.runCount;
        capacity = Math.max(capacity, slice.capacity);
        scratchPerRun = Math.max(scratchPerRun, slice.scratchPerRun);
    }
    return { runs, capacity, scratchPerRun };
}

function fits(batch: ComputeTextShapeBatch, slices: readonly ShapeSlice[]): boolean {
    const limits = batch.engine._device.limits;
    const bytes = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
    const { runs, capacity, scratchPerRun } = extent(slices);
    return runs <= limits.maxComputeWorkgroupsPerDimension && runs * rowBytes(capacity) <= bytes && runs * scratchPerRun <= bytes;
}

function rowBytes(capacity: number): number {
    return (COMPUTE_TEXT_HEADER_WORDS + capacity * COMPUTE_TEXT_RECORD_WORDS) * 4;
}

function recordBatch(batch: ComputeTextShapeBatch, slices: readonly ShapeSlice[], encoder: GPUCommandEncoder): void {
    const engine = batch.engine;
    if (slices.length === 1) {
        recordDirect(batch, slices[0]!, encoder);
        return;
    }
    const { runs: total, capacity } = extent(slices);
    let payload = 0;
    for (const slice of slices) {
        payload += slice.runs.length - slice.runCount;
    }
    const words = new Uint32Array(total + payload);
    let run = 0;
    let cursor = total;
    for (const slice of slices) {
        const shift = cursor - slice.runCount;
        for (let i = 0; i < slice.runCount; i++) {
            words[run + i] = slice.runs[i]! + shift;
        }
        words.set(slice.runs.subarray(slice.runCount), cursor);
        cursor += slice.runs.length - slice.runCount;
        run += slice.runCount;
    }
    const sourceRow = rowBytes(capacity);
    reserve(batch, slices, words.byteLength);
    writeParams(batch, total, capacity, 0);
    updateStorageBuffer(engine, batch.runs, words);
    setComputeDispatchSize(batch.dispatch!, { x: total });
    _recordComputeTasks([batch.task!], encoder);
    const source = _getStorageBufferHandle(engine, batch.arena);
    let row = 0;
    for (const slice of slices) {
        // A slice whose owner released its buffers without flushing has no reader left.
        if (!slice.shaped._destroyed) {
            const target = _getStorageBufferHandle(engine, slice.shaped);
            const targetRow = rowBytes(slice.capacity);
            if (targetRow === sourceRow) {
                encoder.copyBufferToBuffer(source, row * sourceRow, target, slice.outputRun * targetRow, slice.runCount * targetRow);
            } else {
                for (let i = 0; i < slice.runCount; i++) {
                    encoder.copyBufferToBuffer(source, (row + i) * sourceRow, target, (slice.outputRun + i) * targetRow, targetRow);
                }
            }
        }
        row += slice.runCount;
    }
}

/** Shape one slice at its own stride and output run, without the arena. Its packed run origins are
 *  already relative to its own payload, so the slice's words upload unchanged. */
function recordDirect(batch: ComputeTextShapeBatch, slice: ShapeSlice, encoder: GPUCommandEncoder): void {
    // An owner that released its buffers without flushing has no reader left.
    if (slice.shaped._destroyed) {
        return;
    }
    const engine = batch.engine;
    reserve(batch, [slice], slice.runs.byteLength);
    writeParams(batch, slice.runCount, slice.capacity, slice.outputRun);
    updateStorageBuffer(engine, batch.runs, slice.runs);
    let direct = batch.direct;
    if (direct?.output !== slice.shaped) {
        direct?.task.dispose();
        const shader = batch.kernel.shader;
        const dispatch = createComputeDispatch(
            shader,
            createComputeBindingSet(shader, {
                params: batch.paramsBuffer,
                program: batch.program,
                tables: batch.tables,
                runs: batch.runs,
                scratch: batch.scratch,
                output: slice.shaped,
            }),
            { size: { x: 1 } }
        );
        const task = createComputeTask(engine, "compute-text-shape-batch-direct");
        addComputeDispatch(task, dispatch);
        task.record();
        direct = batch.direct = { output: slice.shaped, task, dispatch };
    }
    setComputeDispatchSize(direct.dispatch, { x: slice.runCount });
    _recordComputeTasks([direct.task], encoder);
}

function writeParams(batch: ComputeTextShapeBatch, runs: number, capacity: number, outputRun: number): void {
    const params = batch.params.slice();
    params[0] = runs;
    params[1] = capacity;
    params[6] = outputRun;
    updateStorageBuffer(batch.engine, batch.paramsBuffer, params);
}

/** Grow buffers for `slices` and rebind on growth. Only multi-slice batches shape into the arena. `runBytes` defaults to the slices' packed words. */
function reserve(batch: ComputeTextShapeBatch, slices: readonly ShapeSlice[], runBytes?: number): void {
    const engine = batch.engine;
    const { runs: total, capacity, scratchPerRun } = extent(slices);
    if (runBytes === undefined) {
        runBytes = 0;
        for (const slice of slices) {
            runBytes += slice.runs.byteLength;
        }
    }
    const runs = grow(engine, batch.runs, runBytes, { label: "compute-text-batch-runs" });
    const scratch = grow(engine, batch.scratch, total * scratchPerRun, { label: "compute-text-batch-scratch", writable: true });
    const arena = slices.length > 1 ? grow(engine, batch.arena, total * rowBytes(capacity), { label: "compute-text-batch-arena", writable: true }) : batch.arena;
    if (batch.task && runs === batch.runs && scratch === batch.scratch && arena === batch.arena) {
        return;
    }
    batch.runs = runs;
    batch.scratch = scratch;
    batch.arena = arena;
    batch.task?.dispose();
    batch.direct?.task.dispose();
    batch.direct = null;
    const shader = batch.kernel.shader;
    batch.dispatch = createComputeDispatch(
        shader,
        createComputeBindingSet(shader, { params: batch.paramsBuffer, program: batch.program, tables: batch.tables, runs, scratch, output: arena }),
        { size: { x: 1 } }
    );
    batch.task = createComputeTask(engine, "compute-text-shape-batch");
    addComputeDispatch(batch.task, batch.dispatch);
    batch.task.record();
}

/** Disposal bumps the engine resource epoch and rebinds every binding set, so grow geometrically. */
function grow(engine: EngineContext, buffer: StorageBuffer, bytes: number, options: { label: string; writable?: boolean }): StorageBuffer {
    if (buffer.byteLength >= bytes) {
        return buffer;
    }
    let size = buffer.byteLength;
    while (size < bytes) {
        size *= 2;
    }
    disposeStorageBuffer(buffer);
    return createStorageBuffer(engine, size, options);
}
