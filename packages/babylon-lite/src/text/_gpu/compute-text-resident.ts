/** Shape then pack, without mapping shaped records or instances. Only the 4-byte status is read. */

import type { EngineContext } from "../../engine/engine.js";
import { createComputeBindingSet } from "../../compute/compute-bindings.js";
import { createComputeDispatch, setComputeDispatchSize } from "../../compute/compute-dispatch.js";
import type { ComputeDispatch } from "../../compute/compute-dispatch.js";
import { createComputeShader, disposeComputeShader } from "../../compute/compute-shader.js";
import type { ComputeShader } from "../../compute/compute-shader.js";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../compute/compute-task.js";
import type { ComputeTask } from "../../compute/compute-task.js";
import { readComputeTaskResult } from "../../compute/compute-task-readback.js";
import { computeStorageBufferBinding } from "../../compute/compute-storage-buffer-binding.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import type { ComputeTextPlan } from "../compute-text-plan.js";
import {
    COMPUTE_TEXT_HEADER_WORDS,
    COMPUTE_TEXT_RECORD_WORDS,
    COMPUTE_TEXT_STATUS_OK,
    checkComputeTextShapeBounds,
    prepareComputeTextShapeWords,
    createComputeTextShapeShader,
    computeTextScratchCapacity,
} from "./compute-text-shape.js";

const STATUS_MISSING = 3;
const COUNT_LANES = 256;

const PACK_KERNEL = `
struct PackParams { runCount:u32, capacity:u32, scaleBits:u32, glyphCount:u32, lineAdvanceBits:u32 }
struct Instance { x:f32, y:f32, packed:u32 }
@group(0) @binding(0) var<storage, read> params: PackParams;
@group(0) @binding(1) var<storage, read> shaped: array<i32>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> instances: array<Instance>;
@group(0) @binding(4) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;
@group(0) @binding(6) var<storage, read_write> offsets: array<vec2u>;
const HEADER: u32 = ${COMPUTE_TEXT_HEADER_WORDS}u;
const RECORD: u32 = ${COMPUTE_TEXT_RECORD_WORDS}u;
const MISSING: u32 = 0xffffffffu;
const EMPTY: u32 = 0xfffffffeu;
const STATUS_MISSING: u32 = ${STATUS_MISSING}u;
fn countRun(run:u32) {
  if (run < params.runCount) {
    let out = run * (HEADER + params.capacity * RECORD);
    var failed = u32(shaped[out]);
    var visible = 0u;
    if (failed == 0u) {
      let count = u32(shaped[out + 1u]);
      if (count > params.capacity) { failed = 2u; }
      else {
        for (var i = 0u; i < count; i++) {
          let glyph = u32(shaped[out + HEADER + i * RECORD]);
          if (glyph >= params.glyphCount) { failed = STATUS_MISSING; break; }
          let slot = slots[glyph];
          if (slot == MISSING) { failed = STATUS_MISSING; break; }
          if (slot != EMPTY) { visible++; }
        }
      }
    }
    offsets[run] = vec2u(visible, failed);
  }
}
fn prefixOffsets() {
  indirect[0] = 6u;
  indirect[1] = 0u;
  indirect[2] = 0u;
  indirect[3] = 0u;
  var total = 0u;
  for (var run = 0u; run < params.runCount; run++) {
    let result = offsets[run];
    if (result.x > arrayLength(&instances) - total) { status[0] = 2u; return; }
    if (result.y != 0u) { status[0] = result.y; return; }
    offsets[run].x = total;
    total += result.x;
  }
  indirect[1] = total;
  status[0] = 0u;
}
@compute @workgroup_size(${COUNT_LANES})
fn countRuns(@builtin(global_invocation_id) invocation:vec3u) {
  countRun(invocation.x);
  storageBarrier();
  if (invocation.x == 0u) { prefixOffsets(); }
}
@compute @workgroup_size(64)
fn countParallel(@builtin(global_invocation_id) invocation:vec3u) {
  countRun(invocation.x);
}
var<workgroup> summaries: array<vec2u, ${COUNT_LANES}>;
var<workgroup> prefixStatus: u32;
@compute @workgroup_size(${COUNT_LANES})
fn prefixRuns(@builtin(local_invocation_index) lane:u32) {
  let limit = arrayLength(&instances);
  let span = (params.runCount + ${COUNT_LANES - 1}u) / ${COUNT_LANES}u;
  let begin = lane * span;
  let end = min(begin + span, params.runCount);
  var summary = vec2u(0u, MISSING);
  for (var run = begin; run < end; run++) {
    let result = offsets[run];
    summary.x = min(limit + 1u, summary.x + result.x);
    if (result.y != 0u) { summary.y = min(summary.y, run); }
  }
  summaries[lane] = summary;
  workgroupBarrier();
  for (var distance = 1u; distance < ${COUNT_LANES}u; distance *= 2u) {
    var pending = summaries[lane];
    if (lane >= distance) {
      let previous = summaries[lane - distance];
      pending = vec2u(min(limit + 1u, pending.x + previous.x), min(pending.y, previous.y));
    }
    workgroupBarrier();
    summaries[lane] = pending;
    workgroupBarrier();
  }
  if (lane == 0u) {
    indirect[0] = 6u;
    indirect[1] = 0u;
    indirect[2] = 0u;
    indirect[3] = 0u;
    let last = summaries[${COUNT_LANES - 1}u];
    var overflow = MISSING;
    if (last.x > limit) {
      var low = 0u;
      var high = ${COUNT_LANES}u;
      while (low < high) {
        let middle = (low + high) / 2u;
        if (summaries[middle].x > limit) { high = middle; }
        else { low = middle + 1u; }
      }
      var total = 0u;
      if (low > 0u) { total = summaries[low - 1u].x; }
      let stop = min((low + 1u) * span, params.runCount);
      for (var run = low * span; run < stop; run++) {
        let count = offsets[run].x;
        if (count > limit - total) { overflow = run; break; }
        total += count;
      }
    }
    prefixStatus = 0u;
    if (overflow != MISSING && overflow <= last.y) { prefixStatus = 2u; }
    else if (last.y != MISSING) { prefixStatus = offsets[last.y].y; }
    status[0] = prefixStatus;
    if (prefixStatus == 0u) { indirect[1] = last.x; }
  }
  if (workgroupUniformLoad(&prefixStatus) != 0u) { return; }
  var total = 0u;
  if (lane > 0u) { total = summaries[lane - 1u].x; }
  for (var run = begin; run < end; run++) {
    let count = offsets[run].x;
    offsets[run].x = total;
    total += count;
  }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation:vec3u) {
  let run = invocation.x;
  if (run >= params.runCount || status[0] != 0u) { return; }
  let out = run * (HEADER + params.capacity * RECORD);
  let count = u32(shaped[out + 1u]);
  let scale = bitcast<f32>(params.scaleBits);
  let lineAdvance = bitcast<f32>(params.lineAdvanceBits);
  var total = offsets[run].x;
  var penX = 0;
  var penY = 0;
  for (var i = 0u; i < count; i++) {
    let at = out + HEADER + i * RECORD;
    let slot = slots[u32(shaped[at])];
    if (slot != EMPTY) {
      instances[total] = Instance(f32(penX + shaped[at + 4u]) * scale, f32(penY + shaped[at + 5u]) * scale - f32(run) * lineAdvance, slot);
      total++;
    }
    penX += shaped[at + 2u];
    penY += shaped[at + 3u];
  }
}
`;

export interface ResidentShapeGroup {
    readonly plan: ComputeTextPlan;
    readonly runs: readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array }[];
    readonly outputRun: number;
}

export interface ResidentTextBuffers {
    readonly shaped: StorageBuffer;
    readonly instances: StorageBuffer;
    readonly indirect: StorageBuffer;
    readonly slots: StorageBuffer;
    readonly status: StorageBuffer;
}

interface ResidentShaders {
    readonly device: GPUDevice;
    shape: ComputeShader;
    pack: ComputeShader;
    count: ComputeShader;
    countParallel: ComputeShader;
    prefix: ComputeShader;
    params: StorageBuffer;
    program: StorageBuffer;
    tables: StorageBuffer;
    runs: StorageBuffer;
    scratch: StorageBuffer;
    packParams: StorageBuffer;
    packOffsets: StorageBuffer;
    uploadedPlan: ComputeTextPlan | null;
    scratchCapacity: number;
    shapeTask?: ComputeTask;
    shapeDispatch?: ComputeDispatch;
    packTask?: ComputeTask;
    packParallelCount?: boolean;
    packCountDispatch?: ComputeDispatch;
    packDispatch?: ComputeDispatch;
    buffers?: ResidentTextBuffers;
}

let shadersByToken: WeakMap<object, ResidentShaders> | null = null;

/** Dispatch shape, then pack. Returns the status word. Does not map glyph or instance bytes. */
export async function executeResidentComputeText(
    token: object,
    engine: EngineContext,
    groups: readonly ResidentShapeGroup[],
    buffers: ResidentTextBuffers,
    capacity: number,
    runCount: number,
    fontSizePx: number,
    unitsPerEm: number,
    glyphCount: number,
    assertLive: () => void
): Promise<number> {
    assertLive();
    const preparedGroups = groups.map((group) => prepareComputeTextShapeWords(group.plan, group.runs, group.outputRun, capacity, engine._device.limits));
    for (const prepared of preparedGroups) {
        checkComputeTextShapeBounds(engine._device.limits, prepared);
    }
    let programBytes = 256;
    let tableBytes = 256;
    let runBytes = 256;
    let scratchBytes = 256;
    for (const prepared of preparedGroups) {
        programBytes = Math.max(programBytes, prepared.programWords.byteLength);
        tableBytes = Math.max(tableBytes, prepared.tables.byteLength);
        runBytes = Math.max(runBytes, prepared.runs.byteLength);
        scratchBytes = Math.max(scratchBytes, prepared.scratchBytes);
    }
    ensureResidentShaderCapacity(token, engine, scratchBytes, programBytes, tableBytes, runBytes, capacity);
    const shaders = ensureShaders(token, engine);
    if (shaders.packOffsets.byteLength < Math.max(8, runCount * 8)) {
        shaders.packTask?.dispose();
        shaders.packTask = undefined;
        shaders.packOffsets = grow(engine, shaders.packOffsets, Math.max(8, runCount * 8), { label: "compute-text-pack-offsets", writable: true });
    }
    ensureSchedules(shaders, engine, buffers, runCount);
    const packGroups = Math.max(1, Math.ceil(runCount / 64));
    if (shaders.packCountDispatch) {
        setComputeDispatchSize(shaders.packCountDispatch, { x: packGroups });
    }
    setComputeDispatchSize(shaders.packDispatch!, { x: packGroups });
    for (let index = 0; index < preparedGroups.length; index++) {
        const prepared = preparedGroups[index]!;
        const group = groups[index]!;
        if (group.runs.length === 0) {
            continue;
        }
        updateStorageBuffer(engine, shaders.params, prepared.params);
        if (shaders.uploadedPlan !== group.plan) {
            updateStorageBuffer(engine, shaders.program, prepared.programWords);
            updateStorageBuffer(engine, shaders.tables, prepared.tables);
            shaders.uploadedPlan = group.plan;
        }
        updateStorageBuffer(engine, shaders.runs, prepared.runs);
        setComputeDispatchSize(shaders.shapeDispatch!, { x: group.runs.length });
        await prepareComputeTask(shaders.shapeTask!);
        assertLive();
        if (index < preparedGroups.length - 1) {
            submitComputeTasks([shaders.shapeTask!]);
        }
    }
    const scale = fontSizePx / unitsPerEm;
    const packParams = new Uint32Array(5);
    packParams[0] = runCount;
    packParams[1] = capacity;
    packParams[2] = floatBits(scale);
    packParams[3] = glyphCount;
    packParams[4] = floatBits(fontSizePx);
    updateStorageBuffer(engine, shaders.packParams, packParams);
    await prepareComputeTask(shaders.packTask!);
    assertLive();
    const tasks = preparedGroups.length > 0 && groups[groups.length - 1]!.runs.length > 0 ? [shaders.shapeTask!, shaders.packTask!] : [shaders.packTask!];
    const status = await readComputeTaskResult(tasks, buffers.status, 4, (bytes) => new DataView(bytes).getUint32(0, true));
    assertLive();
    return status;
}

/** Drop shaders borrowed by a disposed or lost object. */
export function releaseResidentComputeText(token: object): void {
    const shaders = shadersByToken?.get(token);
    if (!shaders) {
        return;
    }
    disposeComputeShader(shaders.shape);
    disposeComputeShader(shaders.pack);
    disposeComputeShader(shaders.count);
    disposeComputeShader(shaders.countParallel);
    disposeComputeShader(shaders.prefix);
    shaders.shapeTask?.dispose();
    shaders.packTask?.dispose();
    for (const buffer of [shaders.params, shaders.program, shaders.tables, shaders.runs, shaders.scratch, shaders.packParams, shaders.packOffsets]) {
        disposeStorageBuffer(buffer);
    }
    shadersByToken?.delete(token);
}

export function residentStatusMessage(status: number): string | null {
    if (status === COMPUTE_TEXT_STATUS_OK) {
        return null;
    }
    if (status === 1) {
        return "updateComputeTextData: unsupported lookup";
    }
    if (status === 2) {
        return "updateComputeTextData: shaped run exceeds its reserved capacity";
    }
    if (status === STATUS_MISSING) {
        return "updateComputeTextData: missing atlas slot";
    }
    return `updateComputeTextData: kernel status ${status}`;
}

function ensureShaders(token: object, engine: EngineContext): ResidentShaders {
    shadersByToken ??= new WeakMap();
    const existing = shadersByToken.get(token);
    if (existing && existing.device === engine._device) {
        return existing;
    }
    if (existing) {
        releaseResidentComputeText(token);
    }
    const shape = createComputeTextShapeShader(engine, 1);
    const pack = createPackShader(engine, "main");
    const count = createPackShader(engine, "countRuns");
    const countParallel = createPackShader(engine, "countParallel");
    const prefix = createPackShader(engine, "prefixRuns");
    const created: ResidentShaders = {
        device: engine._device,
        shape,
        pack,
        count,
        countParallel,
        prefix,
        params: createStorageBuffer(engine, 28, { label: "compute-text-resident-params" }),
        program: createStorageBuffer(engine, 256, { label: "compute-text-resident-program" }),
        tables: createStorageBuffer(engine, 256, { label: "compute-text-resident-tables" }),
        runs: createStorageBuffer(engine, 256, { label: "compute-text-resident-runs" }),
        scratch: createStorageBuffer(engine, 256, { label: "compute-text-resident-scratch", writable: true }),
        packParams: createStorageBuffer(engine, 20, { label: "compute-text-resident-pack-params" }),
        packOffsets: createStorageBuffer(engine, 8, { label: "compute-text-pack-offsets", writable: true }),
        uploadedPlan: null,
        scratchCapacity: 1,
    };
    shadersByToken.set(token, created);
    return created;
}

function createPackShader(engine: EngineContext, entryPoint: string): ComputeShader {
    return createComputeShader(engine, {
        name: "compute-text-pack",
        computeSource: PACK_KERNEL,
        entryPoint,
        bindings: [
            computeStorageBufferBinding("params", { group: 0, binding: 0 }),
            computeStorageBufferBinding("shaped", { group: 0, binding: 1 }),
            computeStorageBufferBinding("slots", { group: 0, binding: 2 }),
            computeStorageBufferBinding("instances", { group: 0, binding: 3, access: "read-write" }),
            computeStorageBufferBinding("indirect", { group: 0, binding: 4, access: "read-write" }),
            computeStorageBufferBinding("status", { group: 0, binding: 5, access: "read-write" }),
            computeStorageBufferBinding("offsets", { group: 0, binding: 6, access: "read-write" }),
        ],
    });
}

function ensureSchedules(shaders: ResidentShaders, engine: EngineContext, buffers: ResidentTextBuffers, runCount: number): void {
    const previous = shaders.buffers;
    const parallelCount = runCount > COUNT_LANES;
    if (!shaders.shapeTask || previous?.shaped !== buffers.shaped) {
        shaders.shapeTask?.dispose();
        const bindings = createComputeBindingSet(shaders.shape, {
            params: shaders.params,
            program: shaders.program,
            tables: shaders.tables,
            runs: shaders.runs,
            scratch: shaders.scratch,
            output: buffers.shaped,
        });
        shaders.shapeDispatch = createComputeDispatch(shaders.shape, bindings, { size: { x: 1 } });
        shaders.shapeTask = createComputeTask(engine, "compute-text-resident-shape");
        addComputeDispatch(shaders.shapeTask, shaders.shapeDispatch);
        shaders.shapeTask.record();
    }
    if (
        !shaders.packTask ||
        shaders.packParallelCount !== parallelCount ||
        previous?.shaped !== buffers.shaped ||
        previous?.instances !== buffers.instances ||
        previous?.slots !== buffers.slots ||
        previous?.indirect !== buffers.indirect ||
        previous?.status !== buffers.status
    ) {
        shaders.packTask?.dispose();
        const resources = { params: shaders.packParams, ...buffers, offsets: shaders.packOffsets };
        const bindings = createComputeBindingSet(shaders.pack, resources);
        shaders.packTask = createComputeTask(engine, "compute-text-resident-pack");
        shaders.packParallelCount = parallelCount;
        shaders.packCountDispatch = undefined;
        shaders.packDispatch = createComputeDispatch(shaders.pack, bindings, { size: { x: 1 } });
        if (parallelCount) {
            shaders.packCountDispatch = createComputeDispatch(shaders.countParallel, createComputeBindingSet(shaders.countParallel, resources), { size: { x: 1 } });
            addComputeDispatch(shaders.packTask, shaders.packCountDispatch);
            addComputeDispatch(shaders.packTask, createComputeDispatch(shaders.prefix, createComputeBindingSet(shaders.prefix, resources), { size: { x: 1 } }));
        } else {
            addComputeDispatch(shaders.packTask, createComputeDispatch(shaders.count, createComputeBindingSet(shaders.count, resources), { size: { x: 1 } }));
        }
        addComputeDispatch(shaders.packTask, shaders.packDispatch);
        shaders.packTask.record();
    }
    shaders.buffers = buffers;
}

function floatBits(value: number): number {
    const bits = new Uint32Array(1);
    new Float32Array(bits.buffer)[0] = value;
    return bits[0] ?? 0;
}

/** Grow a resident shader buffer in place of the cached one. Not used for draw-borrowed buffers. */
export function ensureResidentShaderCapacity(
    token: object,
    engine: EngineContext,
    scratchBytes: number,
    programBytes: number,
    tableBytes: number,
    runBytes: number,
    capacity = 1
): void {
    const shaders = ensureShaders(token, engine);
    const scratchCapacity = computeTextScratchCapacity(capacity, shaders.scratchCapacity);
    if (shaders.scratchCapacity !== scratchCapacity) {
        disposeComputeShader(shaders.shape);
        shaders.shape = createComputeTextShapeShader(engine, capacity);
        shaders.scratchCapacity = scratchCapacity;
        shaders.shapeTask?.dispose();
        shaders.shapeTask = undefined;
    }
    if (shaders.program.byteLength < programBytes || shaders.tables.byteLength < tableBytes) {
        shaders.uploadedPlan = null;
    }
    if (shaders.program.byteLength < programBytes || shaders.tables.byteLength < tableBytes || shaders.runs.byteLength < runBytes || shaders.scratch.byteLength < scratchBytes) {
        shaders.shapeTask?.dispose();
        shaders.shapeTask = undefined;
    }
    shaders.program = grow(engine, shaders.program, programBytes, { label: "compute-text-resident-program" });
    shaders.tables = grow(engine, shaders.tables, tableBytes, { label: "compute-text-resident-tables" });
    shaders.runs = grow(engine, shaders.runs, runBytes, { label: "compute-text-resident-runs" });
    shaders.scratch = grow(engine, shaders.scratch, scratchBytes, { label: "compute-text-resident-scratch", writable: true });
    shadersByToken?.set(token, shaders);
}

function grow(engine: EngineContext, buffer: StorageBuffer, bytes: number, options: { label: string; writable?: boolean }): StorageBuffer {
    if (buffer.byteLength >= bytes && buffer._engine._device === engine._device) {
        return buffer;
    }
    disposeStorageBuffer(buffer);
    return createStorageBuffer(engine, bytes, options);
}
