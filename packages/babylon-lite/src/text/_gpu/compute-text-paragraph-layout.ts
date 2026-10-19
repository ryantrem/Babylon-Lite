import type { EngineContext } from "../../engine/engine.js";
import { createComputeBindingSet } from "../../compute/compute-bindings.js";
import { createComputeDispatch, setComputeDispatchSize } from "../../compute/compute-dispatch.js";
import type { ComputeDispatch } from "../../compute/compute-dispatch.js";
import { createComputeShader, disposeComputeShader } from "../../compute/compute-shader.js";
import type { ComputeShader } from "../../compute/compute-shader.js";
import { computeStorageBufferBinding } from "../../compute/compute-storage-buffer-binding.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import type { ResidentPackStage, ResidentTextBuffers } from "./compute-text-resident.js";
import type { ComputeFont } from "../compute-font.js";
import type { SharedAtlas } from "../glyph-storage.js";
import { COMPUTE_TEXT_HEADER_WORDS, COMPUTE_TEXT_RECORD_WORDS } from "./compute-text-shape.js";

const LAYOUT_KERNEL = `
struct Instance { x:f32, y:f32, slot:u32 }
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> shaped: array<i32>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> instances: array<Instance>;
@group(0) @binding(4) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;
@group(0) @binding(6) var<storage, read_write> info: array<vec4u>;
@group(0) @binding(7) var<storage, read_write> lines: array<vec4u>;
const HEADER: u32 = ${COMPUTE_TEXT_HEADER_WORDS}u;
const RECORD: u32 = ${COMPUTE_TEXT_RECORD_WORDS}u;
const MISSING: u32 = 0xffffffffu;
const EMPTY: u32 = 0xfffffffeu;
fn recordAt(p:u32, i:u32) -> u32 {
  return p * (HEADER + input[1] * RECORD) + HEADER + i * RECORD;
}
fn flags(p:u32, c:u32) -> u32 {
  return (input[16u + input[11] + p * input[12] + c / 4u] >> ((c % 4u) * 8u)) & 255u;
}
fn emit(p:u32, line:u32, first:u32, count:u32, width:i32) {
  var cluster = 0u;
  if (count != 0u) { cluster = u32(shaped[recordAt(p, first) + 1u]); }
  lines[p * input[1] + line] = vec4u(first, count, bitcast<u32>(max(0.0, f32(width) * bitcast<f32>(input[2]))), cluster);
}
fn serialMeasure(p:u32, count:u32, visible:u32) {
  let scale = bitcast<f32>(input[2]);
  let maxWidth = bitcast<f32>(input[4]);
  var line = 0u;
  var lineStart = 0u;
  var startPos = 0;
  var pos = 0;
  var contentEnd = 0;
  var lastBreak = MISSING;
  var breakPos = 0;
  var breakContentEnd = 0;
  for (var i = 0u; i < count; i++) {
    let at = recordAt(p, i);
    let c = u32(shaped[at + 1u]);
    let adv = shaped[at + 2u];
    let flag = flags(p, c);
    if ((flag & 1u) == 0u) {
      let end = pos + adv;
      loop {
        if (i == lineStart || f32(end - startPos) * scale <= maxWidth) { break; }
        if (lastBreak != MISSING) {
          emit(p, line, lineStart, lastBreak + 1u - lineStart, breakContentEnd - startPos);
          line++;
          lineStart = lastBreak + 1u;
          startPos = breakPos;
          lastBreak = MISSING;
          continue;
        }
        if (c != u32(shaped[recordAt(p, i - 1u) + 1u]) && (flag & 4u) == 0u) {
          emit(p, line, lineStart, i - lineStart, contentEnd - startPos);
          line++;
          lineStart = i;
          startPos = pos;
        }
        break;
      }
      contentEnd = end;
    }
    pos += adv;
    if (i + 1u < count) {
      let next = u32(shaped[recordAt(p, i + 1u) + 1u]);
      if (next > c && (flags(p, next - 1u) & 2u) != 0u) {
        lastBreak = i;
        breakPos = pos;
        breakContentEnd = contentEnd;
      }
    }
  }
  emit(p, line, lineStart, count - lineStart, contentEnd - startPos);
  info[p] = vec4u(visible, 0u, line + 1u, 0u);
}
var<workgroup> summaries: array<vec3u, 256>;
var<workgroup> prefixStatus: u32;
var<workgroup> measureFailure: atomic<u32>;
var<workgroup> serialRequired: atomic<u32>;
var<workgroup> measureVisible: atomic<u32>;
var<workgroup> measureStatus: u32;
var<workgroup> useSerial: u32;
var<workgroup> measuredLines: u32;
fn tableAt(p:u32, i:u32) -> u32 {
  return input[11] * input[1] + 2u * (p * input[1] + i);
}
fn nonSpace(p:u32, i:u32) -> bool {
  return (flags(p, u32(shaped[recordAt(p, i) + 1u])) & 1u) == 0u;
}
fn breakOpp(p:u32, i:u32, count:u32) -> bool {
  if (i + 1u >= count) { return false; }
  let c = u32(shaped[recordAt(p, i) + 1u]);
  let next = u32(shaped[recordAt(p, i + 1u) + 1u]);
  return next > c && (flags(p, next - 1u) & 2u) != 0u;
}
fn emergency(p:u32, i:u32) -> bool {
  if (i == 0u) { return false; }
  let c = u32(shaped[recordAt(p, i) + 1u]);
  return nonSpace(p, i) && c != u32(shaped[recordAt(p, i - 1u) + 1u]) && (flags(p, c) & 4u) == 0u;
}
fn scanSummary(lane:u32, kind:u32, reverse:bool) {
  workgroupBarrier();
  for (var distance = 1u; distance < 256u; distance *= 2u) {
    var pending = summaries[lane];
    let other = select(lane - distance, lane + distance, reverse);
    if (select(lane >= distance, lane + distance < 256u, reverse)) {
      let previous = summaries[other];
      if (kind == 0u) { pending += previous; }
      else if (kind == 1u) { pending = max(pending, previous); }
      else { pending = min(pending, previous); }
    }
    workgroupBarrier();
    summaries[lane] = pending;
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn measure(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  let index = group.x + group.y * groups.x;
  if (index >= input[9]) { return; }
  var p = input[8] + index;
  if (input[10] == 1u) { p = input[16u + index]; }
  let head = p * (HEADER + input[1] * RECORD);
  let count = u32(shaped[head + 1u]);
  if (lane == 0u) {
    var failed = u32(shaped[head]);
    if (failed == 0u && count > input[1]) { failed = 2u; }
    atomicStore(&measureFailure, failed);
    atomicStore(&serialRequired, 0u);
    atomicStore(&measureVisible, 0u);
    measureStatus = failed;
  }
  if (workgroupUniformLoad(&measureStatus) != 0u) {
    if (lane == 0u) { info[p] = vec4u(0u, measureStatus, 1u, 0u); }
    return;
  }
  let span = (count + 255u) / 256u;
  let begin = min(lane * span, count);
  let end = min(begin + span, count);
  var sum = vec3u(0u);
  for (var i = begin; i < end; i++) {
    let at = recordAt(p, i);
    let glyph = u32(shaped[at]);
    if (glyph >= input[3]) { atomicStore(&measureFailure, 3u); }
    else {
      let slot = slots[glyph];
      if (slot == MISSING) { atomicStore(&measureFailure, 3u); }
      else if (slot != EMPTY) { sum.z++; }
    }
    if (shaped[at + 2u] < 0) { atomicStore(&serialRequired, 1u); }
    sum = vec3u(sum.xy + bitcast<vec2u>(vec2i(shaped[at + 2u], shaped[at + 3u])), sum.z);
  }
  atomicAdd(&measureVisible, sum.z);
  workgroupBarrier();
  if (lane == 0u) { measureStatus = atomicLoad(&measureFailure); }
  if (workgroupUniformLoad(&measureStatus) != 0u) {
    if (lane == 0u) { info[p] = vec4u(0u, measureStatus, 1u, 0u); }
    return;
  }
  summaries[lane] = sum;
  scanSummary(lane, 0u, false);
  sum = vec3u(0u);
  if (lane > 0u) { sum = summaries[lane - 1u]; }
  workgroupBarrier();
  for (var i = begin; i < end; i++) {
    let at = recordAt(p, i);
    lines[tableAt(p, i)] = vec4u(sum, 0u);
    let advance = bitcast<u32>(shaped[at + 2u]);
    if (advance > 0x7fffffffu - min(sum.x, 0x7fffffffu)) { atomicStore(&serialRequired, 1u); }
    sum = vec3u(sum.xy + bitcast<vec2u>(vec2i(shaped[at + 2u], shaped[at + 3u])), sum.z);
    if (slots[u32(shaped[at])] != EMPTY) { sum.z++; }
  }
  storageBarrier();
  var last = vec3u(0u);
  for (var i = begin; i < end; i++) {
    if (nonSpace(p, i)) { last.x = i + 1u; }
    if (breakOpp(p, i, count)) { last.y = i + 1u; }
  }
  summaries[lane] = last;
  scanSummary(lane, 1u, false);
  last = vec3u(0u);
  if (lane > 0u) { last = summaries[lane - 1u]; }
  workgroupBarrier();
  for (var i = begin; i < end; i++) {
    var content = 0u;
    if (nonSpace(p, i)) { last.x = i + 1u; }
    if (breakOpp(p, i, count)) { last.y = i + 1u; }
    if (last.x != 0u) {
      let k = last.x - 1u;
      content = lines[tableAt(p, k)].x + bitcast<u32>(shaped[recordAt(p, k) + 2u]);
    }
    lines[tableAt(p, i)].w = content;
    lines[tableAt(p, i) + 1u] = vec4u(last.y - 1u, MISSING, MISSING, count);
  }
  storageBarrier();
  var nearest = MISSING;
  for (var i = end; i > begin;) {
    i--;
    if (nonSpace(p, i)) { nearest = i; }
  }
  summaries[lane] = vec3u(nearest);
  scanSummary(lane, 2u, true);
  nearest = MISSING;
  if (lane < 255u) { nearest = summaries[lane + 1u].x; }
  workgroupBarrier();
  for (var i = end; i > begin;) {
    i--;
    if (nonSpace(p, i)) { nearest = i; }
    lines[tableAt(p, i) + 1u].y = nearest;
  }
  storageBarrier();
  nearest = MISSING;
  for (var i = end; i > begin;) {
    i--;
    var event = MISSING;
    if (emergency(p, i)) { event = i; }
    else if (breakOpp(p, i, count)) { event = lines[tableAt(p, i + 1u) + 1u].y; }
    nearest = min(nearest, event);
    lines[tableAt(p, i) + 1u].z = nearest;
  }
  summaries[lane] = vec3u(nearest);
  scanSummary(lane, 2u, true);
  nearest = MISSING;
  if (lane < 255u) { nearest = summaries[lane + 1u].x; }
  workgroupBarrier();
  for (var i = begin; i < end; i++) {
    lines[tableAt(p, i) + 1u].z = min(nearest, lines[tableAt(p, i) + 1u].z);
  }
  storageBarrier();
  if (lane == 0u) { useSerial = atomicLoad(&serialRequired); }
  if (workgroupUniformLoad(&useSerial) == 0u) {
    let scale = bitcast<f32>(input[2]);
    let maxWidth = bitcast<f32>(input[4]);
    for (var s = begin; s < end; s++) {
      if (s != 0u && !breakOpp(p, s - 1u, count) && !emergency(p, s)) { continue; }
      let startPos = bitcast<i32>(lines[tableAt(p, s)].x);
      var low = s + 1u;
      var high = count;
      while (low < high) {
        let middle = (low + high) / 2u;
        let endPos = bitcast<i32>(lines[tableAt(p, middle)].x) + shaped[recordAt(p, middle) + 2u];
        if (f32(endPos - startPos) * scale <= maxWidth) { low = middle + 1u; }
        else { high = middle; }
      }
      var next = count;
      if (low < count) {
        let i = lines[tableAt(p, low) + 1u].y;
        if (i != MISSING) {
          var previous = lines[tableAt(p, i - 1u) + 1u].x;
          if (previous != MISSING && previous >= s) { next = previous + 1u; }
          else {
            let k = lines[tableAt(p, i) + 1u].z;
            if (k != MISSING) {
              previous = lines[tableAt(p, k - 1u) + 1u].x;
              if (previous != MISSING && previous >= s) { next = previous + 1u; }
              else { next = k; }
            }
          }
        }
      }
      lines[tableAt(p, s) + 1u].w = next;
    }
  }
  storageBarrier();
  if (lane == 0u) {
    let visible = atomicLoad(&measureVisible);
    if (useSerial != 0u) { serialMeasure(p, count, visible); }
    else if (count == 0u) {
      emit(p, 0u, 0u, 0u, 0);
      info[p] = vec4u(visible, 0u, 1u, 0u);
    } else {
      var s = 0u;
      var line = 0u;
      loop {
        let next = lines[tableAt(p, s) + 1u].w;
        let width = bitcast<i32>(lines[tableAt(p, next - 1u)].w) - bitcast<i32>(lines[tableAt(p, s)].x);
        emit(p, line, s, next - s, width);
        line++;
        if (next == count) { break; }
        s = next;
      }
      info[p] = vec4u(visible, 0u, line, 0u);
    }
    measuredLines = info[p].z;
  }
  storageBarrier();
  let lineCount = workgroupUniformLoad(&measuredLines);
  for (var i = begin; i < end; i++) {
    var low = 0u;
    var high = lineCount;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (lines[p * input[1] + middle].x <= i) { low = middle; }
      else { high = middle; }
    }
    lines[tableAt(p, i) + 1u].x = low;
  }
}
@compute @workgroup_size(256)
fn prefix(@builtin(local_invocation_index) lane:u32) {
  let limit = arrayLength(&instances);
  let span = (input[0] + 255u) / 256u;
  let begin = lane * span;
  let end = min(begin + span, input[0]);
  var summary = vec3u(0u, 0u, MISSING);
  for (var p = begin; p < end; p++) {
    let result = info[input[16u + p]];
    summary.x = min(limit + 1u, summary.x + result.x);
    summary.y += result.z;
    if (result.y != 0u) { summary.z = min(summary.z, p); }
  }
  summaries[lane] = summary;
  workgroupBarrier();
  for (var distance = 1u; distance < 256u; distance *= 2u) {
    var pending = summaries[lane];
    if (lane >= distance) {
      let previous = summaries[lane - distance];
      pending = vec3u(min(limit + 1u, pending.x + previous.x), pending.y + previous.y, min(pending.z, previous.z));
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
    let last = summaries[255u];
    var overflow = MISSING;
    if (last.x > limit) {
      var low = 0u;
      var high = 256u;
      while (low < high) {
        let middle = (low + high) / 2u;
        if (summaries[middle].x > limit) { high = middle; }
        else { low = middle + 1u; }
      }
      var total = 0u;
      if (low > 0u) { total = summaries[low - 1u].x; }
      let stop = min((low + 1u) * span, input[0]);
      for (var p = low * span; p < stop; p++) {
        let count = info[input[16u + p]].x;
        if (count > limit - total) { overflow = p; break; }
        total += count;
      }
    }
    prefixStatus = 0u;
    if (overflow != MISSING && overflow <= last.z) { prefixStatus = 2u; }
    else if (last.z != MISSING) { prefixStatus = info[input[16u + last.z]].y; }
    status[0] = prefixStatus;
    if (prefixStatus == 0u) { indirect[1] = last.x; }
  }
  if (workgroupUniformLoad(&prefixStatus) != 0u) { return; }
  var base = vec2u(0u);
  if (lane > 0u) { base = summaries[lane - 1u].xy; }
  for (var p = begin; p < end; p++) {
    let result = info[input[16u + p]];
    info[input[11] + p] = vec4u(base.x, 0u, result.z, base.y);
    base += result.xz;
  }
}
@compute @workgroup_size(256)
fn place(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  let p = group.x + group.y * groups.x;
  if (p >= input[0] || status[0] != 0u) { return; }
  let slot = input[16u + p];
  let paragraph = info[input[11] + p];
  let scale = bitcast<f32>(input[2]);
  let maxWidth = bitcast<f32>(input[4]);
  let lineHeight = bitcast<f32>(input[5]);
  let spacing = bitcast<f32>(input[6]);
  let count = u32(shaped[slot * (HEADER + input[1] * RECORD) + 1u]);
  for (var i = lane; i < count; i += 256u) {
    let at = recordAt(slot, i);
    let glyphSlot = slots[u32(shaped[at])];
    if (glyphSlot == EMPTY) { continue; }
    let a = lines[tableAt(slot, i)];
    let l = lines[tableAt(slot, i) + 1u].x;
    let line = lines[slot * input[1] + l];
    let lineX = bitcast<i32>(lines[tableAt(slot, line.x)].x);
    var align = 0.0;
    if (input[7] != 0u) {
      align = max(0.0, maxWidth - bitcast<f32>(line.z));
      if (input[7] == 1u) { align *= 0.5; }
    }
    let baseline = fma(f32(paragraph.w + l), lineHeight, f32(p) * spacing);
    instances[paragraph.x + a.z] = Instance(fma(f32(bitcast<i32>(a.x) - lineX + shaped[at + 4u]), scale, align), fma(f32(bitcast<i32>(a.y) + shaped[at + 5u]), scale, -baseline), glyphSlot);
  }
}
`;

/** @internal Last successfully submitted paragraph state. */
export interface ComputeTextParagraphCache {
    /** @internal */
    _paragraphs: readonly string[];
    /** @internal */
    _order: Uint32Array;
    /** @internal */
    _capacity: number;
    /** @internal */
    _pool: number;
    /** @internal */
    _width: number;
    /** @internal */
    _font: ComputeFont;
    /** @internal */
    _fontSize: number;
    /** @internal */
    _device: GPUDevice;
    /** @internal */
    _generation: number;
    /** @internal */
    _revision: number;
    /** @internal Revision of the full rebuild this state descends from. */
    _baseRevision: number;
    /** @internal */
    _shaped: StorageBuffer;
    /** @internal Detect external mutation of the caller-owned atlas. */
    _atlas?: SharedAtlas;
    /** @internal */
    _atlasVersion?: number;
}

/** @internal Per-token layout resources; never part of the public block API. */
export interface ComputeTextParagraphStage extends ResidentPackStage {
    version: number;
    /** @internal */
    _words: Uint32Array;
    /** @internal Cached view of the reusable input words. */
    _floatWords: Float32Array;
    /** @internal Active upload range, excluding unused high-water capacity. */
    _uploadWords: Uint32Array;
    /** @internal Word ranges to upload; params, changed order, dirty flags. */
    _uploadRanges: readonly (readonly [number, number])[];
    /** @internal */
    _cache?: ComputeTextParagraphCache;
    /** @internal Admission invalidation epoch, including busy rejections. */
    _epoch: number;
    /** @internal The block's immutable font/size default, computed only when needed. */
    _naturalLineHeightPx?: number;
    /** @internal */
    _device?: GPUDevice;
    /** @internal */
    _input?: StorageBuffer;
    /** @internal */
    _info?: StorageBuffer;
    /** @internal */
    _lines?: StorageBuffer;
    /** @internal */
    _shaders?: ComputeShader[];
    /** @internal */
    _dispatches?: ComputeDispatch[];
}

let stages: WeakMap<object, ComputeTextParagraphStage> | null = null;

/** @internal Inspect layout records in tests without adding readback to updates. */
export function getComputeTextParagraphStage(token: object): ComputeTextParagraphStage | undefined {
    return stages?.get(token);
}

/** @internal Reserve reusable CPU input storage; no GPU allocation or submission. */
export function reserveComputeTextParagraphStage(
    token: object,
    engine: EngineContext,
    paragraphs: number,
    capacity: number,
    pool: number,
    flagStride: number
): ComputeTextParagraphStage {
    const wordCount = 16 + pool + pool * flagStride;
    const maxWords = Math.floor(Math.min(engine._device.limits.maxBufferSize, engine._device.limits.maxStorageBufferBindingSize) / 4);
    for (const [name, bytes] of computeTextParagraphBufferSizes(pool, capacity, wordCount * 4)) {
        checkComputeTextParagraphBufferSize(engine._device.limits, name, bytes, pool, capacity);
    }
    let words = stages?.get(token)?._words;
    if (!words || words.length < wordCount || words.length > maxWords) {
        words = new Uint32Array(Math.max(wordCount, Math.min(maxWords, (words?.length ?? 0) * 2)));
    }
    words[0] = paragraphs;
    words[1] = capacity;
    words[11] = pool;
    words[12] = flagStride;
    return createComputeTextParagraphStage(token, engine, words, wordCount);
}

/** @internal Validate before atlas mutation; GPU resources are created only by prepare. */
export function createComputeTextParagraphStage(token: object, engine: EngineContext, words: Uint32Array, wordCount = words.length): ComputeTextParagraphStage {
    checkBounds(engine, words);
    let stage = stages?.get(token);
    if (!stage) {
        const created: ComputeTextParagraphStage = {
            version: 0,
            _words: words,
            _floatWords: new Float32Array(words.buffer, words.byteOffset, words.length),
            _uploadWords: wordCount === words.length ? words : words.subarray(0, wordCount),
            _uploadRanges: [[0, wordCount]],
            _epoch: 0,
            prepare(engine, _runCount, capacity) {
                if (capacity !== created._words[1]) {
                    throw new Error("updateComputeTextParagraphs: layout and shaped capacities disagree");
                }
                checkBounds(engine, created._words);
                if (created._device !== engine._device) {
                    if (created._device) {
                        releaseResources(created);
                    }
                    created._device = engine._device;
                }
                created._shaders ??= ["measure", "prefix", "place"].map((entryPoint) => createLayoutShader(engine, entryPoint));
                const pool = created._words[11]!;
                created._input = grow(engine, created, created._input, created._words.byteLength, "input");
                created._info = grow(engine, created, created._info, Math.max(1, pool * 2) * 16, "info");
                created._lines = grow(engine, created, created._lines, Math.max(1, pool) * capacity * 48, "lines");
                for (const [first, end] of created._uploadRanges) {
                    if (end > first) {
                        updateStorageBuffer(engine, created._input, created._words.subarray(first, end), first * 4);
                    }
                }
                created._dispatches?.forEach((dispatch, index) => {
                    const size = dispatchSize(created, index);
                    setComputeDispatchSize(dispatch, size);
                    dispatch.enabled = size.x > 0;
                });
                (stages ??= new WeakMap()).set(token, created);
            },
            dispatches(buffers: ResidentTextBuffers) {
                const resources = { input: created._input!, ...buffers, info: created._info!, lines: created._lines! };
                created._dispatches = created._shaders!.map((shader, index) => {
                    const size = dispatchSize(created, index);
                    return createComputeDispatch(shader, createComputeBindingSet(shader, resources), { size, enabled: size.x > 0 });
                });
                return created._dispatches;
            },
            release() {
                releaseResources(created);
                if (stages?.get(token) === created) {
                    stages.delete(token);
                }
            },
        };
        stage = created;
        (stages ??= new WeakMap()).set(token, created);
    }
    if (stage._words !== words) {
        stage._floatWords = new Float32Array(words.buffer, words.byteOffset, words.length);
    }
    if (stage._words !== words || stage._uploadWords.length !== wordCount) {
        stage._uploadWords = wordCount === words.length ? words : words.subarray(0, wordCount);
    }
    stage._words = words;
    stage._uploadRanges = [[0, wordCount]];
    return stage;
}

/** @internal Layout allocations; inputBytes includes reusable CPU input growth when present. */
export function computeTextParagraphBufferSizes(
    pool: number,
    capacity: number,
    inputBytes = (16 + pool + pool * Math.ceil(capacity / 4)) * 4
): readonly (readonly [string, number])[] {
    return [
        ["layout input", inputBytes],
        ["layout info", Math.max(1, pool * 2) * 16],
        ["layout lines", Math.max(1, pool) * capacity * 48],
    ];
}

/** @internal Report the actual buffer/limits before paragraph resource installation. */
export function checkComputeTextParagraphBufferSize(
    limits: Pick<GPUSupportedLimits, "maxBufferSize" | "maxStorageBufferBindingSize">,
    name: string,
    bytes: number,
    pool: number,
    capacity: number
): void {
    if (!Number.isSafeInteger(bytes) || bytes > limits.maxBufferSize || bytes > limits.maxStorageBufferBindingSize) {
        throw new Error(
            `updateComputeTextParagraphs: ${name} buffer of ${bytes} bytes exceeds device storage limits ` +
                `(maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize}, pool=${pool}, capacity=${capacity})`
        );
    }
}

function checkBounds(engine: EngineContext, words: Uint32Array): void {
    const paragraphs = words[0]!;
    const pool = words[11]!;
    const capacity = words[1]!;
    const limits = engine._device.limits;
    for (const [name, bytes] of computeTextParagraphBufferSizes(pool, capacity, words.byteLength)) {
        checkComputeTextParagraphBufferSize(limits, name, bytes, pool, capacity);
    }
    if (
        Math.ceil(paragraphs / 64) > limits.maxComputeWorkgroupsPerDimension ||
        Math.ceil(Math.max(paragraphs, words[9]!) / limits.maxComputeWorkgroupsPerDimension) > limits.maxComputeWorkgroupsPerDimension
    ) {
        throw new Error("updateComputeTextParagraphs: paragraph dispatch exceeds device workgroup limits");
    }
}

function dispatchSize(stage: ComputeTextParagraphStage, index: number): { x: number; y: number } {
    const count = index === 1 ? 1 : stage._words[index === 0 ? 9 : 0]!;
    const limit = stage._device!.limits.maxComputeWorkgroupsPerDimension;
    return { x: Math.min(count, limit), y: Math.max(1, Math.ceil(count / limit)) };
}

function createLayoutShader(engine: EngineContext, entryPoint: string): ComputeShader {
    return createComputeShader(engine, {
        name: "compute-text-paragraph-layout",
        computeSource: LAYOUT_KERNEL,
        entryPoint,
        bindings: ["input", "shaped", "slots", "instances", "indirect", "status", "info", "lines"].map((name, binding) =>
            computeStorageBufferBinding(name, { group: 0, binding, access: binding < 3 ? "read" : "read-write" })
        ),
    });
}

function grow(engine: EngineContext, stage: ComputeTextParagraphStage, buffer: StorageBuffer | undefined, bytes: number, name: string): StorageBuffer {
    if (buffer && !buffer._destroyed && buffer.byteLength >= bytes) {
        return buffer;
    }
    if (buffer) {
        disposeStorageBuffer(buffer);
    }
    stage.version++;
    stage._dispatches = undefined;
    return createStorageBuffer(engine, bytes, { label: `compute-text-paragraph-${name}`, writable: name !== "input" });
}

function releaseResources(stage: ComputeTextParagraphStage): void {
    stage._cache = undefined;
    stage._epoch++;
    stage._shaders?.forEach(disposeComputeShader);
    for (const buffer of [stage._input, stage._info, stage._lines]) {
        if (buffer) {
            disposeStorageBuffer(buffer);
        }
    }
    stage._shaders = undefined;
    stage._dispatches = undefined;
    stage._input = stage._info = stage._lines = undefined;
    stage._device = undefined;
    stage.version++;
}
