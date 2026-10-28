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
import type { ComputeTextPlan } from "../compute-text-plan.js";
import type { ComputeTextSelectionReuse } from "../compute-text-selection.js";
import type { SharedAtlas } from "../glyph-storage.js";
import { COMPUTE_TEXT_HEADER_WORDS, COMPUTE_TEXT_RECORD_WORDS } from "./compute-text-shape.js";

/** Records placed per workgroup, so long paragraphs spread placement across many workgroups. */
const PLACE_TILE = 1024;

/** Records per lane in the multi-workgroup candidate and line-assignment stages. */
const FITS_SPAN = 4;
const ASSIGN_SPAN = 16;

/** Records per lane in the multi-workgroup measure stages; lane offsets pack into bytes. */
const MEASURE_SPAN = 8;
const MEASURE_BLOCK = MEASURE_SPAN * 256;

/** Above this record capacity, line breaking spreads measuring, candidates and line assignment across workgroups. */
const SPLIT_CAPACITY = 2048;

/** Line rows and the prefix table, then split-mode per-paragraph measure block summaries (3 vec4 per block). */
function layoutLinesBytes(pool: number, capacity: number): number {
    const blocks = capacity > SPLIT_CAPACITY ? Math.ceil(capacity / MEASURE_BLOCK) : 0;
    return Math.max(1, pool) * (capacity + blocks) * 48;
}

/** @internal Input word 13: whether long-paragraph layout uses the split multi-workgroup stages. */
export function computeTextParagraphSplitStages(capacity: number): number {
    return capacity > SPLIT_CAPACITY ? 1 : 0;
}

function placeTiles(capacity: number): number {
    return Math.max(1, Math.ceil(capacity / PLACE_TILE));
}

/** Dispatch order; measure (fused short paragraphs), walk and prefix are paragraph-serial, the others spread blocks across workgroups. */
const LAYOUT_ENTRY_POINTS = ["measure", "summarize", "measureBlocks", "candidates", "walk", "assign", "prefix", "place"] as const;

function layoutGroups(words: Uint32Array, index: number): number {
    const measured = words[9]!;
    const capacity = words[1]!;
    const split = words[13] !== 0;
    switch (LAYOUT_ENTRY_POINTS[index]) {
        case "measure":
            return split ? 0 : measured;
        case "summarize":
        case "measureBlocks":
            return split ? measured * Math.max(1, Math.ceil(capacity / MEASURE_BLOCK)) : 0;
        case "candidates":
            return split ? measured * Math.max(1, Math.ceil(capacity / (FITS_SPAN * 256))) : 0;
        case "walk":
            return split ? measured : 0;
        case "assign":
            return split ? measured * Math.max(1, Math.ceil(capacity / (ASSIGN_SPAN * 256))) : 0;
        case "prefix":
            return 1;
        default:
            return words[0]! * placeTiles(capacity);
    }
}

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
const PLACE_TILE: u32 = ${PLACE_TILE}u;
const FITS_SPAN: u32 = ${FITS_SPAN}u;
const FITS_BLOCK: u32 = ${FITS_SPAN * 256}u;
const ASSIGN_SPAN: u32 = ${ASSIGN_SPAN}u;
const ASSIGN_BLOCK: u32 = ${ASSIGN_SPAN * 256}u;
const MEASURE_SPAN: u32 = ${MEASURE_SPAN}u;
const MEASURE_BLOCK: u32 = ${MEASURE_BLOCK}u;
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
var<workgroup> stageState: u32;
// Block measure reductions: 0-2 sums, 3-4 last non-space/break ends, 5 first non-space, 6 first emergency, 7 first break,
// 8 non-space after that break, 9 failure (1) / negative (2) flags, 10-11 high/low advance halves, 12 content end, 13 visible.
var<workgroup> blockReduce: array<atomic<u32>, 14>;
fn tableAt(p:u32, i:u32) -> u32 {
  return input[11] * input[1] + 2u * (p * input[1] + i);
}
fn cluster(p:u32, i:u32) -> u32 {
  return u32(shaped[recordAt(p, i) + 1u]);
}
// Break opportunity after i / emergency break before i; callers carry neighbouring clusters and flags in registers.
fn breakAfter(p:u32, i:u32, count:u32, c:u32, next:u32, flag:u32) -> bool {
  if (i + 1u >= count || next <= c) { return false; }
  var nextFlag = flag;
  if (next - 1u != c) { nextFlag = flags(p, next - 1u); }
  return (nextFlag & 2u) != 0u;
}
fn emergencyAt(i:u32, previous:u32, c:u32, flag:u32) -> bool {
  return i != 0u && (flag & 5u) == 0u && c != previous;
}
// Non-serial P7 only, i < count: guarded nonnegative prefixes make the successor exact.
fn fits(p:u32, i:u32, count:u32, startPos:i32, scale:f32, maxWidth:f32) -> bool {
  var endPos:i32;
  if (i + 1u < count) { endPos = bitcast<i32>(lines[tableAt(p, i + 1u)].x); }
  else { endPos = bitcast<i32>(lines[tableAt(p, i)].x) + shaped[recordAt(p, i) + 2u]; }
  return f32(endPos - startPos) * scale <= maxWidth;
}
fn scanSummary(lane:u32, kind:u32, reverse:bool) {
  workgroupBarrier();
  for (var distance = 1u; distance < 256u; distance *= 2u) {
    var pending = summaries[lane];
    let other = select(lane - distance, lane + distance, reverse);
    if (select(lane >= distance, lane + distance < 256u, reverse)) {
      let previous = summaries[other];
      if (kind == 0u) { pending += previous; }
      // kind 1: max index pair, carrying z from whichever side holds the larger x.
      else if (kind == 1u) { pending = vec3u(max(pending.xy, previous.xy), select(pending.z, previous.z, previous.x > pending.x)); }
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
  // Line-break classification summaries ride along with the advance sums so each pass loads every record once.
  var last = vec3u(0u);
  var content = 0u;
  var first = MISSING;
  var c = 0u;
  if (begin < end) { c = cluster(p, begin); }
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
    var next = 0u;
    if (i + 1u < count) { next = cluster(p, i + 1u); }
    let flag = flags(p, c);
    if ((flag & 1u) == 0u) {
      last.x = i + 1u;
      content = sum.x;
      first = min(first, i);
    }
    if (breakAfter(p, i, count, c, next, flag)) { last.y = i + 1u; }
    c = next;
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
  summaries[lane] = vec3u(last.xy, sum.x + content);
  scanSummary(lane, 1u, false);
  last = vec3u(0u);
  if (lane > 0u) { last = summaries[lane - 1u]; }
  content = select(0u, last.z, last.x != 0u);
  workgroupBarrier();
  summaries[lane] = vec3u(first);
  scanSummary(lane, 2u, true);
  var nearest = MISSING;
  if (lane < 255u) { nearest = summaries[lane + 1u].x; }
  workgroupBarrier();
  writeTable(p, count, begin, end, sum, last.xy, content);
  workgroupBarrier();
  if (lane == 0u) { useSerial = atomicLoad(&serialRequired); }
  if (workgroupUniformLoad(&useSerial) == 0u) {
    summaries[lane] = vec3u(reverseEvents(p, count, begin, end, nearest, MISSING));
    scanSummary(lane, 2u, true);
    storageBarrier();
    findBreaks(p, count, begin, end, span, false);
  }
  workgroupBarrier();
  let state = select(1u, 2u, workgroupUniformLoad(&useSerial) != 0u);
  storageBarrier();
  if (lane == 0u) {
    chainLines(p, count, state, atomicLoad(&measureVisible));
    measuredLines = info[p].z;
  }
  storageBarrier();
  let lineCount = workgroupUniformLoad(&measuredLines);
  if (state == 1u && count != 0u) {
    emitLines(p, count, lineCount, lane);
    storageBarrier();
  }
  assignLines(p, count, lineCount, begin, end);
}
// Prefix table rows for [begin, end) from the carried-in sums, last non-space/break ends and content end.
fn writeTable(p:u32, count:u32, begin:u32, end:u32, carry:vec3u, lastIn:vec2u, contentIn:u32) {
  var sum = carry;
  var last = lastIn;
  var content = contentIn;
  var c = 0u;
  if (begin < end) { c = cluster(p, begin); }
  for (var i = begin; i < end; i++) {
    let at = recordAt(p, i);
    let start = sum;
    let advance = bitcast<u32>(shaped[at + 2u]);
    if (advance > 0x7fffffffu - min(sum.x, 0x7fffffffu)) { atomicStore(&serialRequired, 1u); }
    sum = vec3u(sum.xy + bitcast<vec2u>(vec2i(shaped[at + 2u], shaped[at + 3u])), sum.z);
    if (slots[u32(shaped[at])] != EMPTY) { sum.z++; }
    var next = 0u;
    if (i + 1u < count) { next = cluster(p, i + 1u); }
    let flag = flags(p, c);
    if ((flag & 1u) == 0u) {
      last.x = i + 1u;
      content = sum.x;
    }
    if (breakAfter(p, i, count, c, next, flag)) { last.y = i + 1u; }
    c = next;
    lines[tableAt(p, i)] = vec4u(start, content);
    lines[tableAt(p, i) + 1u] = vec4u(last.y - 1u, MISSING, MISSING, count);
  }
}
// Reverse pass: nearest non-space at or after i, then the nearest break event at or after i, seeded with later carries.
fn reverseEvents(p:u32, count:u32, begin:u32, end:u32, nearestIn:u32, eventsIn:u32) -> u32 {
  var nearest = nearestIn;
  var events = eventsIn;
  var after = 0u;
  var c = 0u;
  if (begin < end) {
    c = cluster(p, end - 1u);
    if (end < count) { after = cluster(p, end); }
  }
  for (var i = end; i > begin;) {
    i--;
    var previous = 0u;
    if (i != 0u) { previous = cluster(p, i - 1u); }
    let flag = flags(p, c);
    var event = MISSING;
    if (emergencyAt(i, previous, c, flag)) { event = i; }
    else if (breakAfter(p, i, count, c, after, flag)) { event = nearest; }
    events = min(events, event);
    if ((flag & 1u) == 0u) { nearest = i; }
    lines[tableAt(p, i) + 1u].y = nearest;
    lines[tableAt(p, i) + 1u].z = events;
    after = c;
    c = previous;
  }
  return events;
}
// Candidate line starts in [begin, end): next line start into table .w. Unless table events are paragraph-global,
// summaries holds the later-lane event carry.
fn findBreaks(p:u32, count:u32, begin:u32, end:u32, span:u32, globalEvents:bool) {
  var c = 0u;
  let scale = bitcast<f32>(input[2]);
  let maxWidth = bitcast<f32>(input[4]);
  var hint = 0u;
  var previous = 0u;
  var previousFlag = 0u;
  if (begin < end) {
    c = cluster(p, begin);
    if (begin != 0u) {
      previous = cluster(p, begin - 1u);
      previousFlag = flags(p, previous);
    }
  }
  for (var s = begin; s < end; s++) {
    let flag = flags(p, c);
    let candidate = s == 0u || breakAfter(p, s - 1u, count, previous, c, previousFlag) || emergencyAt(s, previous, c, flag);
    previous = c;
    previousFlag = flag;
    if (s + 1u < end) { c = cluster(p, s + 1u); }
    if (!candidate) { continue; }
    let startPos = bitcast<i32>(lines[tableAt(p, s)].x);
    var low = max(s + 1u, hint);
    if (low < count && fits(p, low, count, startPos, scale, maxWidth)) {
      let first = low;
      var step = 1u;
      var high = first + min(step, count - first);
      low++;
      loop {
        if (high == count || !fits(p, high, count, startPos, scale, maxWidth)) { break; }
        low = high + 1u;
        step = min(count - first, step * 2u + 1u);
        high = first + step;
      }
      while (low < high) {
        let middle = (low + high) / 2u;
        if (fits(p, middle, count, startPos, scale, maxWidth)) { low = middle + 1u; }
        else { high = middle; }
      }
    }
    hint = low;
    var next = count;
    if (low < count) {
      let i = lines[tableAt(p, low) + 1u].y;
      if (i != MISSING) {
        var previous = lines[tableAt(p, i - 1u) + 1u].x;
        if (previous != MISSING && previous >= s) { next = previous + 1u; }
        else {
          var k = lines[tableAt(p, i) + 1u].z;
          if (!globalEvents && i / span < 255u) { k = min(k, summaries[i / span + 1u].x); }
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
// Lane 0 only: chase the dependent next-line chain (or the serial fallback) and finalize info[p].
fn chainLines(p:u32, count:u32, state:u32, visible:u32) {
  if (state == 2u) { serialMeasure(p, count, visible); }
  else if (count == 0u) {
    emit(p, 0u, 0u, 0u, 0);
    info[p] = vec4u(visible, 0u, 1u, 0u);
  } else {
    var s = 0u;
    var line = 0u;
    loop {
      lines[p * input[1] + line].x = s;
      line++;
      let next = lines[tableAt(p, s) + 1u].w;
      if (next == count) { break; }
      s = next;
    }
    info[p] = vec4u(visible, 0u, line, 0u);
  }
}
fn emitLines(p:u32, count:u32, lineCount:u32, lane:u32) {
  for (var line = lane; line < lineCount; line += 256u) {
    let row = p * input[1] + line;
    let first = lines[row].x;
    var next = count;
    if (line + 1u < lineCount) { next = lines[row + 1u].x; }
    let width = bitcast<i32>(lines[tableAt(p, next - 1u)].w) - bitcast<i32>(lines[tableAt(p, first)].x);
    lines[row].y = next - first;
    lines[row].z = bitcast<u32>(max(0.0, f32(width) * bitcast<f32>(input[2])));
    lines[row].w = u32(shaped[recordAt(p, first) + 1u]);
  }
}
// Line index of each record in [begin, end); place reads it from table .x.
fn assignLines(p:u32, count:u32, lineCount:u32, begin:u32, end:u32) {
  if (begin >= end) { return; }
  var low = 0u;
  var high = lineCount;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (lines[p * input[1] + middle].x <= begin) { low = middle; }
    else { high = middle; }
  }
  var nextFirst = count;
  if (low + 1u < lineCount) { nextFirst = lines[p * input[1] + low + 1u].x; }
  for (var i = begin; i < end; i++) {
    while (i >= nextFirst && low + 1u < lineCount) {
      low++;
      nextFirst = count;
      if (low + 1u < lineCount) { nextFirst = lines[p * input[1] + low + 1u].x; }
    }
    lines[tableAt(p, i) + 1u].x = low;
  }
}
fn measuredParagraph(index:u32) -> u32 {
  if (input[10] == 1u) { return input[16u + index]; }
  return input[8] + index;
}
fn paragraphBlocks(block:u32) -> u32 {
  return max(1u, (input[1] + block - 1u) / block);
}
// Split stages run only when input[13] is set; pending measure states in info.w: 1 = parallel, 2 = serial; walk finalizes w to 0.
fn scratchAt(p:u32, block:u32) -> u32 {
  return 3u * (input[11] * input[1] + p * paragraphBlocks(MEASURE_BLOCK) + block);
}
fn resetReduce(lane:u32) {
  workgroupBarrier();
  if (lane < 14u) { atomicStore(&blockReduce[lane], select(0u, MISSING, lane >= 5u && lane <= 8u)); }
  workgroupBarrier();
}
fn laneOffset(i:u32, begin:u32) -> u32 {
  return select(i - begin, 255u, i == MISSING);
}
fn laneIndex(packed:u32, shift:u32, begin:u32) -> u32 {
  let offset = (packed >> shift) & 255u;
  return select(begin + offset, MISSING, offset == 255u);
}
// Saturating u32 total of the accumulated high/low 16-bit advance halves.
fn reducedWide() -> u32 {
  let high = atomicLoad(&blockReduce[10]) + (atomicLoad(&blockReduce[11]) >> 16u);
  return select((high << 16u) | (atomicLoad(&blockReduce[11]) & 0xffffu), MISSING, high > 0xffffu);
}
// Block pass 1: per-lane forward summaries into each lane's first table entries, block summary into scratch.
@compute @workgroup_size(256)
fn summarize(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  let blocks = paragraphBlocks(MEASURE_BLOCK);
  for (var item = group.x + group.y * groups.x; item < input[9] * blocks; item += groups.x * groups.y) {
    let p = measuredParagraph(item / blocks);
    let block = item % blocks;
    let head = p * (HEADER + input[1] * RECORD);
    let count = u32(shaped[head + 1u]);
    let blockBegin = block * MEASURE_BLOCK;
    if (shaped[head] != 0 || count > input[1] || blockBegin >= count) { continue; }
    resetReduce(lane);
    let begin = min(blockBegin + lane * MEASURE_SPAN, count);
    let end = min(begin + MEASURE_SPAN, count);
    var sum = vec3u(0u);
    var last = vec2u(0u);
    var content = 0u;
    var first = MISSING;
    var emergency = MISSING;
    var firstBreak = MISSING;
    var after = MISSING;
    var wide = 0u;
    var failed = 0u;
    var c = 0u;
    var previous = 0u;
    if (begin < end) {
      c = cluster(p, begin);
      if (begin != 0u) { previous = cluster(p, begin - 1u); }
    }
    for (var i = begin; i < end; i++) {
      let at = recordAt(p, i);
      let glyph = u32(shaped[at]);
      if (glyph >= input[3]) { failed |= 1u; }
      else {
        let slot = slots[glyph];
        if (slot == MISSING) { failed |= 1u; }
        else if (slot != EMPTY) { sum.z++; }
      }
      let advance = shaped[at + 2u];
      if (advance < 0) { failed |= 2u; }
      else { wide = select(wide + u32(advance), MISSING, u32(advance) > MISSING - wide); }
      sum = vec3u(sum.xy + bitcast<vec2u>(vec2i(advance, shaped[at + 3u])), sum.z);
      var next = 0u;
      if (i + 1u < count) { next = cluster(p, i + 1u); }
      let flag = flags(p, c);
      if ((flag & 1u) == 0u) {
        last.x = i + 1u;
        content = sum.x;
        first = min(first, i);
        if (firstBreak != MISSING) { after = min(after, i); }
      }
      // Events at or after a position reduce to min(first emergency, nearest non-space after the first break).
      if (emergencyAt(i, previous, c, flag)) { emergency = min(emergency, i); }
      if (breakAfter(p, i, count, c, next, flag)) {
        last.y = i + 1u;
        firstBreak = min(firstBreak, i);
      }
      previous = c;
      c = next;
    }
    if (begin < end) {
      lines[tableAt(p, begin)] = vec4u(sum, content);
      lines[tableAt(p, begin) + 1u] = vec4u(last, first, laneOffset(emergency, begin) | (laneOffset(firstBreak, begin) << 8u) | (laneOffset(after, begin) << 16u));
    }
    atomicAdd(&blockReduce[0], sum.x);
    atomicAdd(&blockReduce[1], sum.y);
    atomicAdd(&blockReduce[2], sum.z);
    atomicMax(&blockReduce[3], last.x);
    atomicMax(&blockReduce[4], last.y);
    atomicMin(&blockReduce[5], first);
    atomicMin(&blockReduce[6], emergency);
    atomicMin(&blockReduce[7], firstBreak);
    atomicOr(&blockReduce[9], failed);
    atomicAdd(&blockReduce[10], wide >> 16u);
    atomicAdd(&blockReduce[11], wide & 0xffffu);
    workgroupBarrier();
    let blockLast = atomicLoad(&blockReduce[3]);
    let blockBreak = atomicLoad(&blockReduce[7]);
    // Block-relative content end at the last non-space, and the nearest non-space after the block's first break.
    if (blockLast != 0u) {
      let owner = (blockLast - 1u - blockBegin) / MEASURE_SPAN;
      if (lane < owner) { atomicAdd(&blockReduce[12], sum.x); }
      else if (lane == owner) { atomicAdd(&blockReduce[12], content); }
    }
    if (blockBreak != MISSING) {
      if (lane == (blockBreak - blockBegin) / MEASURE_SPAN) { atomicMin(&blockReduce[8], after); }
      else if (begin > blockBreak) { atomicMin(&blockReduce[8], first); }
    }
    workgroupBarrier();
    if (lane == 0u) {
      let at = scratchAt(p, block);
      lines[at] = vec4u(atomicLoad(&blockReduce[0]), atomicLoad(&blockReduce[1]), atomicLoad(&blockReduce[2]), blockLast);
      lines[at + 1u] = vec4u(atomicLoad(&blockReduce[4]), atomicLoad(&blockReduce[12]), atomicLoad(&blockReduce[5]), atomicLoad(&blockReduce[6]));
      lines[at + 2u] = vec4u(blockBreak, atomicLoad(&blockReduce[8]), reducedWide(), atomicLoad(&blockReduce[9]));
    }
  }
}
// Block pass 2: earlier/later block carries, in-block lane scans, then the table rows and paragraph-global events.
@compute @workgroup_size(256)
fn measureBlocks(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  let blocks = paragraphBlocks(MEASURE_BLOCK);
  for (var item = group.x + group.y * groups.x; item < input[9] * blocks; item += groups.x * groups.y) {
    let p = measuredParagraph(item / blocks);
    let block = item % blocks;
    let head = p * (HEADER + input[1] * RECORD);
    let count = u32(shaped[head + 1u]);
    let blockBegin = block * MEASURE_BLOCK;
    var failed = u32(shaped[head]);
    if (failed == 0u && count > input[1]) { failed = 2u; }
    if (failed != 0u) {
      if (block == 0u && lane == 0u) { info[p] = vec4u(0u, failed, 1u, 0u); }
      continue;
    }
    if (block != 0u && blockBegin >= count) { continue; }
    resetReduce(lane);
    let used = (count + MEASURE_BLOCK - 1u) / MEASURE_BLOCK;
    for (var b = lane; b < used; b += 256u) {
      let at = scratchAt(p, b);
      let first = lines[at];
      let second = lines[at + 1u];
      let third = lines[at + 2u];
      atomicAdd(&blockReduce[13], first.z);
      atomicOr(&blockReduce[9], third.w);
      atomicAdd(&blockReduce[10], third.z >> 16u);
      atomicAdd(&blockReduce[11], third.z & 0xffffu);
      if (b < block) {
        atomicAdd(&blockReduce[0], first.x);
        atomicAdd(&blockReduce[1], first.y);
        atomicAdd(&blockReduce[2], first.z);
        atomicMax(&blockReduce[3], first.w);
        atomicMax(&blockReduce[4], second.x);
      } else if (b > block) {
        atomicMin(&blockReduce[5], second.z);
        atomicMin(&blockReduce[6], second.w);
        atomicMin(&blockReduce[7], third.x);
      }
    }
    workgroupBarrier();
    if (lane == 0u) {
      // Failure, else serial for a negative advance or a paragraph advance total past i32, else parallel.
      let flagged = atomicLoad(&blockReduce[9]);
      measureStatus = select(select(1u, 2u, (flagged & 2u) != 0u || reducedWide() > 0x7fffffffu), 3u, (flagged & 1u) != 0u);
    }
    let state = workgroupUniformLoad(&measureStatus);
    if (state == 3u) {
      if (block == 0u && lane == 0u) { info[p] = vec4u(0u, 3u, 1u, 0u); }
      continue;
    }
    let lastBefore = atomicLoad(&blockReduce[3]);
    let breakLater = atomicLoad(&blockReduce[7]);
    for (var b = lane; b < used; b += 256u) {
      let at = scratchAt(p, b);
      if (lastBefore != 0u) {
        let lastBlock = (lastBefore - 1u) / MEASURE_BLOCK;
        if (b < lastBlock) { atomicAdd(&blockReduce[12], lines[at].x); }
        else if (b == lastBlock) { atomicAdd(&blockReduce[12], lines[at + 1u].y); }
      }
      if (breakLater != MISSING) {
        let breakBlock = breakLater / MEASURE_BLOCK;
        if (b == breakBlock) { atomicMin(&blockReduce[8], lines[at + 2u].y); }
        else if (b > breakBlock) { atomicMin(&blockReduce[8], lines[at + 1u].z); }
      }
    }
    workgroupBarrier();
    let begin = min(blockBegin + lane * MEASURE_SPAN, count);
    let end = min(begin + MEASURE_SPAN, count);
    var own = vec4u(0u);
    var ownLast = vec4u(0u, 0u, MISSING, MISSING);
    if (begin < end) {
      own = lines[tableAt(p, begin)];
      ownLast = lines[tableAt(p, begin) + 1u];
    }
    summaries[lane] = own.xyz;
    scanSummary(lane, 0u, false);
    var sum = vec3u(atomicLoad(&blockReduce[0]), atomicLoad(&blockReduce[1]), atomicLoad(&blockReduce[2]));
    if (lane > 0u) { sum += summaries[lane - 1u]; }
    workgroupBarrier();
    summaries[lane] = vec3u(ownLast.xy, sum.x + own.w);
    scanSummary(lane, 1u, false);
    var last = vec3u(lastBefore, atomicLoad(&blockReduce[4]), atomicLoad(&blockReduce[12]));
    if (lane > 0u) {
      let previous = summaries[lane - 1u];
      last.y = max(last.y, previous.y);
      if (previous.x > last.x) { last = vec3u(previous.x, last.y, previous.z); }
    }
    workgroupBarrier();
    summaries[lane] = vec3u(ownLast.z);
    scanSummary(lane, 2u, true);
    var nearest = atomicLoad(&blockReduce[5]);
    if (lane < 255u) { nearest = min(nearest, summaries[lane + 1u].x); }
    workgroupBarrier();
    let after = laneIndex(ownLast.w, 16u, begin);
    var event = laneIndex(ownLast.w, 0u, begin);
    if (laneIndex(ownLast.w, 8u, begin) != MISSING) { event = min(event, select(after, nearest, after == MISSING)); }
    summaries[lane] = vec3u(event);
    scanSummary(lane, 2u, true);
    var events = min(atomicLoad(&blockReduce[6]), atomicLoad(&blockReduce[8]));
    if (lane < 255u) { events = min(events, summaries[lane + 1u].x); }
    writeTable(p, count, begin, end, sum, last.xy, select(0u, last.z, last.x != 0u));
    if (state == 1u) { _ = reverseEvents(p, count, begin, end, nearest, events); }
    if (block == 0u && lane == 0u) { info[p] = vec4u(atomicLoad(&blockReduce[13]), 0u, 1u, state); }
  }
}
@compute @workgroup_size(256)
fn candidates(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  let blocks = paragraphBlocks(FITS_BLOCK);
  for (var item = group.x + group.y * groups.x; item < input[9] * blocks; item += groups.x * groups.y) {
    let p = measuredParagraph(item / blocks);
    let count = u32(shaped[p * (HEADER + input[1] * RECORD) + 1u]);
    let begin = min((item % blocks) * FITS_BLOCK + lane * FITS_SPAN, count);
    // Block measuring leaves paragraph-global events in the table, so lanes need no carry or barriers here.
    if (begin >= count || info[p].w != 1u) { continue; }
    findBreaks(p, count, begin, min(begin + FITS_SPAN, count), 0u, true);
  }
}
@compute @workgroup_size(256)
fn walk(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  for (var index = group.x + group.y * groups.x; index < input[9]; index += groups.x * groups.y) {
    let p = measuredParagraph(index);
    let count = u32(shaped[p * (HEADER + input[1] * RECORD) + 1u]);
    workgroupBarrier();
    if (lane == 0u) {
      let result = info[p];
      stageState = result.w;
      if (result.w != 0u) { chainLines(p, count, result.w, result.x); }
      measuredLines = info[p].z;
    }
    storageBarrier();
    let lineCount = workgroupUniformLoad(&measuredLines);
    if (workgroupUniformLoad(&stageState) == 1u && count != 0u) { emitLines(p, count, lineCount, lane); }
  }
}
@compute @workgroup_size(256)
fn assign(@builtin(workgroup_id) group:vec3u, @builtin(num_workgroups) groups:vec3u, @builtin(local_invocation_index) lane:u32) {
  let blocks = paragraphBlocks(ASSIGN_BLOCK);
  for (var item = group.x + group.y * groups.x; item < input[9] * blocks; item += groups.x * groups.y) {
    let p = measuredParagraph(item / blocks);
    let count = u32(shaped[p * (HEADER + input[1] * RECORD) + 1u]);
    let begin = min((item % blocks) * ASSIGN_BLOCK + lane * ASSIGN_SPAN, count);
    let end = min(begin + ASSIGN_SPAN, count);
    if (begin >= end) { continue; }
    let result = info[p];
    if (result.y == 0u) { assignLines(p, count, result.z, begin, end); }
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
  if (status[0] != 0u) { return; }
  let tiles = max(1u, (input[1] + PLACE_TILE - 1u) / PLACE_TILE);
  // Workgroup-stride over (paragraph, tile) items keeps placement exact for any dispatch size.
  for (var index = group.x + group.y * groups.x; index < input[0] * tiles; index += groups.x * groups.y) {
    let p = index / tiles;
    let first = (index % tiles) * PLACE_TILE;
    let slot = input[16u + p];
    let paragraph = info[input[11] + p];
    let scale = bitcast<f32>(input[2]);
    let maxWidth = bitcast<f32>(input[4]);
    let lineHeight = bitcast<f32>(input[5]);
    let spacing = bitcast<f32>(input[6]);
    let count = u32(shaped[slot * (HEADER + input[1] * RECORD) + 1u]);
    let last = min(count, first + PLACE_TILE);
    for (var i = first + lane; i < last; i += 256u) {
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
}
`;

/** @internal Exact owned inputs and one flag byte per scalar for the current live paragraph. */
export interface ComputeTextParagraphFlagEntry {
    plan: ComputeTextPlan;
    scalars: Uint32Array;
    glyphIds: Uint32Array;
    flags: Uint8Array;
    /** Plain decode record for incremental selection; shares `scalars` and `glyphIds`. */
    reuse?: ComputeTextSelectionReuse;
}

/** @internal Last successfully submitted paragraph state. */
export interface ComputeTextParagraphCache {
    /** @internal */
    _paragraphs: readonly string[];
    /** @internal Positional live entries, published with this revision/epoch only. */
    _flags?: readonly (ComputeTextParagraphFlagEntry | undefined)[];
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
                created._shaders ??= LAYOUT_ENTRY_POINTS.map((entryPoint) => createLayoutShader(engine, entryPoint));
                const pool = created._words[11]!;
                created._input = grow(engine, created, created._input, created._words.byteLength, "input");
                created._info = grow(engine, created, created._info, Math.max(1, pool * 2) * 16, "info");
                created._lines = grow(engine, created, created._lines, layoutLinesBytes(pool, capacity), "lines");
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
        ["layout lines", layoutLinesBytes(pool, capacity)],
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
        Math.ceil(Math.max(...LAYOUT_ENTRY_POINTS.map((_, index) => layoutGroups(words, index))) / limits.maxComputeWorkgroupsPerDimension) >
            limits.maxComputeWorkgroupsPerDimension
    ) {
        throw new Error("updateComputeTextParagraphs: paragraph dispatch exceeds device workgroup limits");
    }
}

function dispatchSize(stage: ComputeTextParagraphStage, index: number): { x: number; y: number } {
    const count = layoutGroups(stage._words, index);
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
