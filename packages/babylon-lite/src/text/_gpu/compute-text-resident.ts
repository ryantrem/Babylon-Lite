/** Shape then pack, without mapping shaped records or instances. Only the 4-byte status is read. */

import type { EngineContext } from "../../engine/engine.js";
import { BU } from "../../engine/gpu-flags.js";
import { createComputeBindingSet } from "../../compute/compute-bindings.js";
import { createComputeDispatch, setComputeDispatchSize } from "../../compute/compute-dispatch.js";
import type { ComputeDispatch } from "../../compute/compute-dispatch.js";
import { createComputeShader, disposeComputeShader, prepareComputeShader } from "../../compute/compute-shader.js";
import type { ComputeShader } from "../../compute/compute-shader.js";
import { addComputeDispatch, createComputeTask, prepareComputeTask } from "../../compute/compute-task.js";
import type { ComputeTask } from "../../compute/compute-task.js";
import { submitDeferredComputeTaskResult } from "../../compute/compute-task-readback.js";
import { _flushDeferredWorkFor } from "../../compute/deferred-submit.js";
import { computeStorageBufferBinding } from "../../compute/compute-storage-buffer-binding.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import type { ComputeTextPlan } from "../compute-text-plan.js";
import type { ComputeTextSegments } from "../compute-text-segment.js";
import { COMPUTE_TEXT_HEADER_WORDS, COMPUTE_TEXT_RECORD_WORDS, COMPUTE_TEXT_STATUS_OK, checkComputeTextShapeBounds, prepareComputeTextShapeWords } from "./compute-text-shape.js";
import { addComputeTextShapeSlice, prepareComputeTextShapeBatch } from "./compute-text-shape-batch.js";

const STATUS_MISSING = 3;
const COUNT_LANES = 256;
const WIDE_PACK_LANES = 64;
const WIDE_PACK_MIN_CAPACITY = 33;

/** @internal Number of consecutive segments copied by each stitch row. */
export const STITCH_TILE = 64;

/** @internal Global row count for the current batch-relative segment table. */
export function stitchTileRows(table: Uint32Array, runCount: number): number {
    let rows = 1;
    for (let run = 0; run < runCount; run++) {
        rows = Math.max(rows, Math.ceil(table[run * 2 + 1]! / STITCH_TILE));
    }
    return rows;
}

/** @internal Shared local segment stitching; paged paragraphs never stitch across storage chunks. */
export const COMPUTE_TEXT_STITCH_KERNEL = `
@group(0) @binding(0) var<storage, read> seg: array<i32>;
@group(0) @binding(1) var<storage, read_write> shaped: array<i32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read> params: vec4u;
const HEADER: u32 = ${COMPUTE_TEXT_HEADER_WORDS}u;
const RECORD: u32 = ${COMPUTE_TEXT_RECORD_WORDS}u;
var<workgroup> aTotal: atomic<u32>;
var<workgroup> aPrefix: atomic<u32>;
var<workgroup> aFail: atomic<u32>;
var<workgroup> uTotal: u32;
var<workgroup> uPrefix: u32;
var<workgroup> uFail: u32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (wg.x >= params.x) { return; }
  let run = params.w + wg.x;
  let first = table[wg.x*2u];
  let count = table[wg.x*2u+1u];
  let sStride = HEADER + params.z*RECORD;
  let out = run*(HEADER + params.y*RECORD);
  let segBase = params.x*2u;
  let tileStart = wg.y*${STITCH_TILE}u;
  if (wg.y != 0u && tileStart >= count) { return; }
  let tileEnd = select(min(count, tileStart+${STITCH_TILE}u), count, wg.y == nwg.y-1u);
  if (lane == 0u) {
    atomicStore(&aTotal, 0u);
    atomicStore(&aPrefix, 0u);
    atomicStore(&aFail, 0xffffffffu);
  }
  workgroupBarrier();
  var lt = 0u;
  var lp = 0u;
  var lf = 0xffffffffu;
  for (var i = lane; i < count; i += 64u) {
    let h = (first+i)*sStride;
    let n = u32(seg[h+1u]);
    lt += n;
    if (i < tileStart) { lp += n; }
    if (lf == 0xffffffffu && seg[h] != 0) { lf = i; }
  }
  atomicAdd(&aTotal, lt);
  atomicAdd(&aPrefix, lp);
  atomicMin(&aFail, lf);
  workgroupBarrier();
  if (lane == 0u) {
    uTotal = atomicLoad(&aTotal);
    uPrefix = atomicLoad(&aPrefix);
    uFail = atomicLoad(&aFail);
  }
  let total = workgroupUniformLoad(&uTotal);
  let prefix = workgroupUniformLoad(&uPrefix);
  let failI = workgroupUniformLoad(&uFail);
  var status: i32 = 0;
  var failed = 0u;
  if (failI != 0xffffffffu) { failed = first+failI; status = seg[failed*sStride]; }
  if (status == 0 && total > params.y) { status = 2; failed = 0xffffffffu; }
  if (status != 0) {
    if (wg.y == 0u && lane == 0u) {
      shaped[out] = status;
      shaped[out+1u] = 0;
      if (failed != 0xffffffffu) {
        shaped[out+2u] = seg[failed*sStride+2u];
        shaped[out+3u] = seg[failed*sStride+3u];
      } else { shaped[out+2u] = 0; shaped[out+3u] = 0; }
    }
    return;
  }
  if (wg.y == 0u && lane == 0u) { shaped[out] = 0; shaped[out+1u] = i32(total); shaped[out+2u] = 0; shaped[out+3u] = 0; }
  var base = prefix;
  for (var i = tileStart; i < tileEnd; i++) {
    let h = (first+i)*sStride;
    let n = u32(seg[h+1u]);
    let start = i32(table[segBase + first + i]);
    for (var k = lane; k < n; k += 64u) {
      let src = h + HEADER + k*RECORD;
      let dst = out + HEADER + (base+k)*RECORD;
      shaped[dst] = seg[src];
      shaped[dst+1u] = seg[src+1u] + start;
      shaped[dst+2u] = seg[src+2u];
      shaped[dst+3u] = seg[src+3u];
      shaped[dst+4u] = seg[src+4u];
      shaped[dst+5u] = seg[src+5u];
    }
    base += n;
  }
}
`;

/** @internal Original resident stitch for batches requiring only one row. */
export const COMPUTE_TEXT_STITCH_BASELINE_KERNEL = `
@group(0) @binding(0) var<storage, read> seg: array<i32>;
@group(0) @binding(1) var<storage, read_write> shaped: array<i32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read> params: vec4u;
const HEADER: u32 = ${COMPUTE_TEXT_HEADER_WORDS}u;
const RECORD: u32 = ${COMPUTE_TEXT_RECORD_WORDS}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (wg.x >= params.x) { return; }
  let run = params.w + wg.x;
  let first = table[wg.x*2u];
  let count = table[wg.x*2u+1u];
  let sStride = HEADER + params.z*RECORD;
  let out = run*(HEADER + params.y*RECORD);
  let segBase = params.x*2u;
  var status: i32 = 0;
  var failed = 0u;
  var total = 0u;
  for (var i = 0u; i < count; i++) {
    let h = (first+i)*sStride;
    if (status == 0 && seg[h] != 0) { status = seg[h]; failed = first+i; }
    total += u32(seg[h+1u]);
  }
  if (status == 0 && total > params.y) { status = 2; failed = 0xffffffffu; }
  if (status != 0) {
    if (lane == 0u) {
      shaped[out] = status;
      shaped[out+1u] = 0;
      if (failed != 0xffffffffu) {
        shaped[out+2u] = seg[failed*sStride+2u];
        shaped[out+3u] = seg[failed*sStride+3u];
      } else { shaped[out+2u] = 0; shaped[out+3u] = 0; }
    }
    return;
  }
  if (lane == 0u) { shaped[out] = 0; shaped[out+1u] = i32(total); shaped[out+2u] = 0; shaped[out+3u] = 0; }
  var base = 0u;
  for (var i = 0u; i < count; i++) {
    let h = (first+i)*sStride;
    let n = u32(seg[h+1u]);
    let start = i32(table[segBase + first + i]);
    for (var k = lane; k < n; k += 64u) {
      let src = h + HEADER + k*RECORD;
      let dst = out + HEADER + (base+k)*RECORD;
      shaped[dst] = seg[src];
      shaped[dst+1u] = seg[src+1u] + start;
      shaped[dst+2u] = seg[src+2u];
      shaped[dst+3u] = seg[src+3u];
      shaped[dst+4u] = seg[src+4u];
      shaped[dst+5u] = seg[src+5u];
    }
    base += n;
  }
}
`;

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
fn packRun(run:u32) {
  if (run >= params.runCount) { return; }
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
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation:vec3u) {
  if (status[0] != 0u) { return; }
  packRun(invocation.x);
}
var<workgroup> packStop: u32;
@compute @workgroup_size(${COUNT_LANES})
fn packSmall(@builtin(local_invocation_index) lane:u32) {
  countRun(lane);
  storageBarrier();
  let limit = arrayLength(&instances);
  var summary = vec2u(0u, MISSING);
  if (lane < params.runCount) {
    let result = offsets[lane];
    summary = vec2u(min(limit + 1u, result.x), select(MISSING, lane, result.y != 0u));
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
      overflow = low;
    }
    packStop = min(overflow, last.y);
    prefixStatus = 0u;
    if (overflow != MISSING && overflow <= last.y) { prefixStatus = 2u; }
    else if (last.y != MISSING) { prefixStatus = offsets[last.y].y; }
    status[0] = prefixStatus;
    if (prefixStatus == 0u) { indirect[1] = last.x; }
  }
  let failed = workgroupUniformLoad(&prefixStatus);
  if (lane < params.runCount && lane < packStop) {
    var total = 0u;
    if (lane > 0u) { total = summaries[lane - 1u].x; }
    offsets[lane].x = total;
  }
  if (failed == 0u) { packRun(lane); }
}
const WIDE: u32 = ${WIDE_PACK_LANES}u;
var<workgroup> wideFail: atomic<u32>;
var<workgroup> wideVisible: atomic<u32>;
var<workgroup> wideStop: u32;
@compute @workgroup_size(${WIDE_PACK_LANES})
fn countWide(@builtin(workgroup_id) group:vec3u, @builtin(local_invocation_index) lane:u32) {
  let run = group.x;
  if (lane == 0u) {
    atomicStore(&wideFail, MISSING);
    atomicStore(&wideVisible, 0u);
  }
  workgroupBarrier();
  let valid = run < params.runCount;
  let out = run * (HEADER + params.capacity * RECORD);
  var header = 1u;
  var count = 0u;
  if (valid) {
    header = u32(shaped[out]);
    if (header == 0u) { count = u32(shaped[out + 1u]); }
  }
  let scan = header == 0u && count <= params.capacity;
  if (scan) {
    for (var i = lane; i < count; i += WIDE) {
      let glyph = u32(shaped[out + HEADER + i * RECORD]);
      if (glyph >= params.glyphCount) { atomicMin(&wideFail, i); }
      else if (slots[glyph] == MISSING) { atomicMin(&wideFail, i); }
    }
  }
  workgroupBarrier();
  if (lane == 0u) { wideStop = atomicLoad(&wideFail); }
  workgroupBarrier();
  let stop = min(count, wideStop);
  if (scan) {
    var visible = 0u;
    for (var i = lane; i < stop; i += WIDE) {
      let glyph = u32(shaped[out + HEADER + i * RECORD]);
      if (slots[glyph] != EMPTY) { visible++; }
    }
    atomicAdd(&wideVisible, visible);
  }
  workgroupBarrier();
  if (lane == 0u && valid) {
    var failed = header;
    var visible = 0u;
    if (header == 0u) {
      if (count > params.capacity) { failed = 2u; }
      else {
        failed = select(0u, STATUS_MISSING, wideStop != MISSING);
        visible = atomicLoad(&wideVisible);
      }
    }
    offsets[run] = vec2u(visible, failed);
  }
}
var<workgroup> wideScan: array<vec3i, ${WIDE_PACK_LANES}>;
var<workgroup> wideHead: vec2u;
@compute @workgroup_size(${WIDE_PACK_LANES})
fn placeWide(@builtin(workgroup_id) group:vec3u, @builtin(local_invocation_index) lane:u32) {
  let run = group.x;
  if (lane == 0u) {
    var s = status[0];
    var n = 0u;
    if (run >= params.runCount) { s = 1u; }
    else { n = u32(shaped[run * (HEADER + params.capacity * RECORD) + 1u]); }
    wideHead = vec2u(s, n);
  }
  let head = workgroupUniformLoad(&wideHead);
  if (head.x != 0u) { return; }
  let count = head.y;
  let out = run * (HEADER + params.capacity * RECORD);
  let base = offsets[run].x;
  let scale = bitcast<f32>(params.scaleBits);
  let lineAdvance = bitcast<f32>(params.lineAdvanceBits);
  var carry = vec3i(0);
  for (var chunk = 0u; chunk < count; chunk += WIDE) {
    let i = chunk + lane;
    let isActive = i < count;
    var own = vec3i(0);
    var slot = EMPTY;
    var offX = 0;
    var offY = 0;
    if (isActive) {
      let at = out + HEADER + i * RECORD;
      slot = slots[u32(shaped[at])];
      own = vec3i(shaped[at + 2u], shaped[at + 3u], select(0, 1, slot != EMPTY));
      offX = shaped[at + 4u];
      offY = shaped[at + 5u];
    }
    wideScan[lane] = own;
    workgroupBarrier();
    for (var distance = 1u; distance < WIDE; distance *= 2u) {
      var pending = wideScan[lane];
      if (lane >= distance) { pending += wideScan[lane - distance]; }
      workgroupBarrier();
      wideScan[lane] = pending;
      workgroupBarrier();
    }
    let excl = carry + wideScan[lane] - own;
    if (isActive && slot != EMPTY) {
      instances[base + u32(excl.z)] = Instance(f32(excl.x + offX) * scale, f32(excl.y + offY) * scale - f32(run) * lineAdvance, slot);
    }
    carry += wideScan[WIDE - 1u];
    workgroupBarrier();
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

/** @internal Replaces the default pack dispatches; runs after shaping and stitching in the same task. */
export interface ResidentPackStage {
    /** Bumped whenever the stage's bound resources change, forcing a pack-task rebuild. */
    readonly version: number;
    /** Size and upload stage resources before scheduling. */
    prepare(engine: EngineContext, runCount: number, capacity: number): void;
    /** Create dispatches bound to the current resident buffers. */
    dispatches(buffers: ResidentTextBuffers): ComputeDispatch[];
    release(): void;
}

interface ResidentShaders {
    readonly device: GPUDevice;
    readonly statusBuffers: GPUBuffer[];
    readonly freeStatusBuffers: GPUBuffer[];
    released: boolean;
    pack: ComputeShader;
    small: ComputeShader;
    countParallel: ComputeShader;
    prefix: ComputeShader;
    countWide: ComputeShader;
    placeWide: ComputeShader;
    packParams: StorageBuffer;
    packOffsets: StorageBuffer;
    stitch?: ComputeShader;
    stitchBaseline?: ComputeShader;
    stitchVariantsReady?: boolean;
    stitchRoute?: "baseline" | "tiled";
    segmentOutput?: StorageBuffer;
    segmentTable?: StorageBuffer;
    stitchParams?: StorageBuffer;
    stitchDispatch?: ComputeDispatch;
    segmented?: boolean;
    boundSegmentOutput?: StorageBuffer;
    boundSegmentTable?: StorageBuffer;
    boundStitchParams?: StorageBuffer;
    packTask?: ComputeTask;
    packMode?: "small" | "parallel" | "wide";
    packCountDispatch?: ComputeDispatch;
    packDispatch?: ComputeDispatch;
    buffers?: ResidentTextBuffers;
    stage?: ResidentPackStage;
    boundStage?: ResidentPackStage;
    boundStageVersion?: number;
}

let shadersByToken: WeakMap<object, ResidentShaders> | null = null;

/** Admit shaping into cross-object batches (see `compute-text-shape-batch.ts`), then record pack and status copy.
 *  Returns after admission, with a separate status promise.
 *  `nominalGlyphs`, when given, must contain every glyph in `groups`; see `prepareComputeTextShapeWords`. */
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
    assertLive: () => void,
    segments?: ComputeTextSegments | null,
    stage?: ResidentPackStage,
    runBase = 0,
    nominalGlyphs?: ReadonlySet<number>
): Promise<{ readonly status: Promise<number> }> {
    assertLive();
    _flushDeferredWorkFor(engine, token);
    const shapeCapacity = segments?.capacity ?? capacity;
    const preparedGroups = groups.map((group) =>
        prepareComputeTextShapeWords(group.plan, group.runs, group.outputRun + (segments ? 0 : runBase), shapeCapacity, engine._device.limits, nominalGlyphs)
    );
    for (const prepared of preparedGroups) {
        checkComputeTextShapeBounds(engine._device.limits, prepared);
    }
    const shaders = ensureShaders(token, engine);
    if (segments) {
        ensureSegmentBuffers(shaders, engine, segments, runCount, capacity, runBase);
        if (!shaders.stitchVariantsReady) {
            await Promise.all([prepareComputeShader(shaders.stitch!), prepareComputeShader(shaders.stitchBaseline!)]);
            shaders.stitchVariantsReady = true;
        }
    }
    if (!stage && shaders.packOffsets.byteLength < Math.max(8, runCount * 8)) {
        shaders.packTask?.dispose();
        shaders.packTask = undefined;
        shaders.packOffsets = grow(engine, shaders.packOffsets, Math.max(8, runCount * 8), { label: "compute-text-pack-offsets", writable: true });
    }
    if (stage) {
        if (shaders.stage !== stage) {
            shaders.stage?.release();
        }
        shaders.stage = stage;
        stage.prepare(engine, runCount, capacity);
    }
    const stitchRows = segments ? stitchTileRows(segments.table, runCount) : 1;
    const stitchRoute = segments ? (stitchRows === 1 ? "baseline" : "tiled") : undefined;
    ensureSchedules(shaders, engine, buffers, runCount, capacity, !!segments, stage, stitchRoute);
    if (segments) {
        setComputeDispatchSize(shaders.stitchDispatch!, stitchRoute === "baseline" ? { x: runCount, y: 1 } : { x: runCount, y: stitchRows });
    }
    if (!stage) {
        const packGroups = shaders.packMode === "wide" ? runCount : Math.max(1, Math.ceil(runCount / 64));
        if (shaders.packCountDispatch) {
            setComputeDispatchSize(shaders.packCountDispatch, { x: packGroups });
        }
        setComputeDispatchSize(shaders.packDispatch!, { x: shaders.packMode === "small" ? 1 : packGroups });
    }
    const batches = await Promise.all(
        groups.map((group, index) => (group.runs.length ? prepareComputeTextShapeBatch(engine, group.plan, preparedGroups[index]!) : Promise.resolve(null)))
    );
    assertLive();
    if (!stage) {
        const scale = fontSizePx / unitsPerEm;
        const packParams = new Uint32Array(5);
        packParams[0] = runCount;
        packParams[1] = capacity;
        packParams[2] = floatBits(scale);
        packParams[3] = glyphCount;
        packParams[4] = floatBits(fontSizePx);
        updateStorageBuffer(engine, shaders.packParams, packParams);
    }
    await prepareComputeTask(shaders.packTask!);
    assertLive();
    // No await may follow: a macrotask flush between these and the pack would split the submission.
    // Each slice's batch is queued before this object's pack, so shaping precedes packing.
    const shapeOutput = segments ? shaders.segmentOutput! : buffers.shaped;
    batches.forEach((batch, index) => batch && addComputeTextShapeSlice(batch, token, preparedGroups[index]!, shapeOutput));
    let staging = shaders.freeStatusBuffers.pop();
    if (!staging) {
        staging = engine._device.createBuffer({ label: "compute-text-status-readback", size: 4, usage: BU.COPY_DST | BU.MAP_READ });
        shaders.statusBuffers.push(staging);
    }
    const readback = staging;
    let status: Promise<number>;
    try {
        status = submitDeferredComputeTaskResult([shaders.packTask!], buffers.status, 4, (bytes) => new DataView(bytes).getUint32(0, true), readback, token);
    } catch (error) {
        shaders.freeStatusBuffers.push(readback);
        throw error;
    }
    status = status.finally(() => {
        if (shaders.released) {
            const index = shaders.statusBuffers.indexOf(readback);
            if (index >= 0) {
                readback.destroy();
                shaders.statusBuffers.splice(index, 1);
            }
        } else if (readback.mapState === "unmapped") {
            shaders.freeStatusBuffers.push(readback);
        }
    });
    void status.catch(() => undefined);
    return { status };
}

/** Drop shaders borrowed by a disposed or lost object. */
export function releaseResidentComputeText(token: object, keepPendingStatus = false): void {
    const shaders = shadersByToken?.get(token);
    if (!shaders) {
        return;
    }
    shaders.packParams._engine._flushPendingWork?.(shaders.packParams._engine);
    shaders.released = true;
    shaders.stage?.release();
    let retained = 0;
    for (const buffer of shaders.statusBuffers) {
        if (!keepPendingStatus || buffer.mapState === "unmapped") {
            buffer.destroy();
        } else {
            shaders.statusBuffers[retained++] = buffer;
        }
    }
    shaders.statusBuffers.length = retained;
    shaders.freeStatusBuffers.length = 0;
    disposeComputeShader(shaders.pack);
    disposeComputeShader(shaders.small);
    disposeComputeShader(shaders.countParallel);
    disposeComputeShader(shaders.prefix);
    disposeComputeShader(shaders.countWide);
    disposeComputeShader(shaders.placeWide);
    if (shaders.stitch) {
        disposeComputeShader(shaders.stitch);
    }
    if (shaders.stitchBaseline) {
        disposeComputeShader(shaders.stitchBaseline);
    }
    shaders.packTask?.dispose();
    for (const buffer of [shaders.packParams, shaders.packOffsets, shaders.segmentOutput, shaders.segmentTable, shaders.stitchParams]) {
        if (buffer) {
            disposeStorageBuffer(buffer);
        }
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
    const pack = createPackShader(engine, "main");
    const small = createPackShader(engine, "packSmall");
    const countParallel = createPackShader(engine, "countParallel");
    const prefix = createPackShader(engine, "prefixRuns");
    const countWide = createPackShader(engine, "countWide");
    const placeWide = createPackShader(engine, "placeWide");
    const created: ResidentShaders = {
        device: engine._device,
        statusBuffers: [],
        freeStatusBuffers: [],
        released: false,
        pack,
        small,
        countParallel,
        prefix,
        countWide,
        placeWide,
        packParams: createStorageBuffer(engine, 20, { label: "compute-text-resident-pack-params" }),
        packOffsets: createStorageBuffer(engine, 8, { label: "compute-text-pack-offsets", writable: true }),
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

function ensureSegmentBuffers(shaders: ResidentShaders, engine: EngineContext, segments: ComputeTextSegments, runCount: number, capacity: number, runBase: number): void {
    if (!shaders.stitch) {
        const bindings = [
            computeStorageBufferBinding("seg", { group: 0, binding: 0 }),
            computeStorageBufferBinding("shaped", { group: 0, binding: 1, access: "read-write" }),
            computeStorageBufferBinding("table", { group: 0, binding: 2 }),
            computeStorageBufferBinding("params", { group: 0, binding: 3 }),
        ];
        shaders.stitch = createComputeShader(engine, {
            name: "compute-text-stitch",
            computeSource: COMPUTE_TEXT_STITCH_KERNEL,
            entryPoint: "main",
            bindings,
        });
        shaders.stitchBaseline = createComputeShader(engine, {
            name: "compute-text-stitch-baseline",
            computeSource: COMPUTE_TEXT_STITCH_BASELINE_KERNEL,
            entryPoint: "main",
            bindings,
        });
        shaders.segmentOutput = createStorageBuffer(engine, 64, { label: "compute-text-segment-output", writable: true });
        shaders.segmentTable = createStorageBuffer(engine, 16, { label: "compute-text-segment-table" });
        shaders.stitchParams = createStorageBuffer(engine, 16, { label: "compute-text-stitch-params" });
    }
    shaders.segmentOutput = grow(engine, shaders.segmentOutput!, segments.count * (COMPUTE_TEXT_HEADER_WORDS + segments.capacity * COMPUTE_TEXT_RECORD_WORDS) * 4, {
        label: "compute-text-segment-output",
        writable: true,
    });
    shaders.segmentTable = grow(engine, shaders.segmentTable!, segments.table.byteLength, { label: "compute-text-segment-table" });
    updateStorageBuffer(engine, shaders.segmentTable, segments.table);
    updateStorageBuffer(engine, shaders.stitchParams!, Uint32Array.of(runCount, capacity, segments.capacity, runBase));
}

function ensureSchedules(
    shaders: ResidentShaders,
    engine: EngineContext,
    buffers: ResidentTextBuffers,
    runCount: number,
    capacity: number,
    segmented: boolean,
    stage?: ResidentPackStage,
    stitchRoute?: "baseline" | "tiled"
): void {
    const previous = shaders.buffers;
    const mode = stage
        ? undefined
        : capacity >= WIDE_PACK_MIN_CAPACITY && runCount > 0 && runCount <= engine._device.limits.maxComputeWorkgroupsPerDimension
          ? "wide"
          : runCount > COUNT_LANES
            ? "parallel"
            : "small";
    if (
        !shaders.packTask ||
        shaders.packMode !== mode ||
        shaders.boundStage !== stage ||
        shaders.boundStageVersion !== stage?.version ||
        shaders.segmented !== segmented ||
        shaders.stitchRoute !== stitchRoute ||
        (segmented &&
            (shaders.boundSegmentOutput !== shaders.segmentOutput || shaders.boundSegmentTable !== shaders.segmentTable || shaders.boundStitchParams !== shaders.stitchParams)) ||
        previous?.shaped !== buffers.shaped ||
        previous?.instances !== buffers.instances ||
        previous?.slots !== buffers.slots ||
        previous?.indirect !== buffers.indirect ||
        previous?.status !== buffers.status
    ) {
        shaders.packTask?.dispose();
        shaders.packTask = createComputeTask(engine, "compute-text-resident-pack");
        shaders.stitchDispatch = undefined;
        if (segmented) {
            const stitch = (stitchRoute === "baseline" ? shaders.stitchBaseline : shaders.stitch)!;
            shaders.stitchDispatch = createComputeDispatch(
                stitch,
                createComputeBindingSet(stitch, { seg: shaders.segmentOutput!, shaped: buffers.shaped, table: shaders.segmentTable!, params: shaders.stitchParams! }),
                { size: { x: 1 } }
            );
            addComputeDispatch(shaders.packTask, shaders.stitchDispatch);
        }
        shaders.packMode = mode;
        shaders.packCountDispatch = undefined;
        shaders.boundStage = stage;
        shaders.boundStageVersion = stage?.version;
        if (stage) {
            for (const dispatch of stage.dispatches(buffers)) {
                addComputeDispatch(shaders.packTask, dispatch);
            }
        } else {
            const resources = { params: shaders.packParams, ...buffers, offsets: shaders.packOffsets };
            const pack = mode === "wide" ? shaders.placeWide : mode === "parallel" ? shaders.pack : shaders.small;
            shaders.packDispatch = createComputeDispatch(pack, createComputeBindingSet(pack, resources), { size: { x: 1 } });
            if (mode !== "small") {
                const count = mode === "wide" ? shaders.countWide : shaders.countParallel;
                shaders.packCountDispatch = createComputeDispatch(count, createComputeBindingSet(count, resources), { size: { x: 1 } });
                addComputeDispatch(shaders.packTask, shaders.packCountDispatch);
                addComputeDispatch(shaders.packTask, createComputeDispatch(shaders.prefix, createComputeBindingSet(shaders.prefix, resources), { size: { x: 1 } }));
            }
            addComputeDispatch(shaders.packTask, shaders.packDispatch);
        }
        shaders.packTask.record();
        shaders.segmented = segmented;
        shaders.stitchRoute = stitchRoute;
        shaders.boundSegmentOutput = shaders.segmentOutput;
        shaders.boundSegmentTable = shaders.segmentTable;
        shaders.boundStitchParams = shaders.stitchParams;
    }
    shaders.buffers = buffers;
}

function floatBits(value: number): number {
    const bits = new Uint32Array(1);
    new Float32Array(bits.buffer)[0] = value;
    return bits[0] ?? 0;
}

function grow(engine: EngineContext, buffer: StorageBuffer, bytes: number, options: { label: string; writable?: boolean }): StorageBuffer {
    if (buffer.byteLength >= bytes && buffer._engine._device === engine._device) {
        return buffer;
    }
    disposeStorageBuffer(buffer);
    return createStorageBuffer(engine, bytes, options);
}
