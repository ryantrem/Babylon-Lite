/** GPU execution of one compute-text batch. Independent runs share no shaping state. */

import type { EngineContext } from "../../engine/engine.js";
import { createComputeBindingSet } from "../../compute/compute-bindings.js";
import { createComputeDispatch, setComputeDispatchSize } from "../../compute/compute-dispatch.js";
import type { ComputeDispatch } from "../../compute/compute-dispatch.js";
import { createComputeShader, disposeComputeShader } from "../../compute/compute-shader.js";
import { addComputeDispatch, createComputeTask, prepareComputeTask } from "../../compute/compute-task.js";
import { readComputeTaskResult } from "../../compute/compute-task-readback.js";
import type { ComputeTask } from "../../compute/compute-task.js";
import { computeStorageBufferBinding } from "../../compute/compute-storage-buffer-binding.js";
import { ensureComputeFontLayout, readComputeFontAdvance } from "../compute-font.js";
import { compileComputeTextProgram } from "../compute-text-program.js";
import type { ComputeTextPlan } from "../compute-text-plan.js";
import type { ComputeTextGlyph, ComputeTextShaper } from "../compute-text-shaper.js";
import type { GlyphClassDef } from "../font-parser/opentype-layout.js";
import { glyphClass } from "../font-parser/opentype-layout.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import { COMPUTE_TEXT_INPUT_PREDICATES, COMPUTE_TEXT_PREDICATES } from "./compute-text-predicates.js";
import { computeTextSimpleSource } from "./compute-text-simple-wgsl.js";
import { prepareSimplePairWords } from "./compute-text-pair-program.js";

const STATUS_OK = 0;
const STATUS_UNSUPPORTED = 1;
const STATUS_OVERFLOW = 2;
const RECORD_WORDS = 6;
const HEADER_WORDS = 4;

/** @internal Shaped-record header words. Resident packing reads the same layout and does not map it. */
export const COMPUTE_TEXT_HEADER_WORDS = HEADER_WORDS;
/** @internal Words per shaped glyph record. */
export const COMPUTE_TEXT_RECORD_WORDS = RECORD_WORDS;
/** @internal */
export const COMPUTE_TEXT_STATUS_OK = STATUS_OK;
/** @internal */
export const COMPUTE_TEXT_STATUS_UNSUPPORTED = STATUS_UNSUPPORTED;
/** @internal */
export const COMPUTE_TEXT_STATUS_OVERFLOW = STATUS_OVERFLOW;

export const COMPUTE_TEXT_KERNEL_SOURCE = /* @__PURE__ */ kernelSource(0);

function cooperativeSource(localScratch: boolean, storageFence: string): string {
    const slot = localScratch ? "index" : "slot";
    const selection = `
for (var ${slot} = lane; ${slot} < ${localScratch ? "count" : "tileCount"}; ${slot} += 32u) {
${localScratch ? "" : "let index = tile + slot;"}
selected[${slot}] = 0u;
if (!live(base, index, count, flag)) { continue; }
if (capability == 1u) {
let entry = directory + u32(scratch[glyphAt(base, index)]) * 2u;
if (program[entry + 1u] == 0u) { continue; }
let ruleAt = ruleOrigin + program[program[entry]] * 16u;
let mode = program[ruleAt + 8u];
selected[${slot}] = select(ruleAt + 1u, 0u, mode == 1u);
if (mode == 1u) { replaceGlyph(glyphAt(base, index), program[program[ruleAt + 6u] + 1u]); }
} else {
let second = pairSecond(base, count, index, flag);
seconds[${slot}] = second;
selected[${slot}] = selectPair(base, index, second, count, ruleOrigin, directory);
if (capability == 3u && selected[${slot}] != 0u) {
let ruleAt = selected[${slot}] - 1u;
let mode = program[ruleAt + 8u];
if (mode != 5u) { commitPairFirst(base, index, program[ruleAt + 6u], mode); selected[${slot}] = 0u; }
}
}
}`;
    const commit = `
for (var ${slot} = 0u; ${slot} < ${localScratch ? "count" : "tileCount"}; ${slot}++) {
if (selected[${slot}] == 0u) { continue; }
${localScratch ? "" : "let index = tile + slot;"}
let ruleAt = selected[${slot}] - 1u;
let mode = program[ruleAt + 8u];
if (mode == 5u) { rejectRun(out, lookupIndex | (tableBit << 16u), index); break; }
if (capability == 2u) { commitPair(base, index, seconds[${slot}], program[ruleAt + 6u], mode); }
}`;
    const ordered = "length = executeRoot(base, count, flag, ruleOrigin, directory, out, lookupIndex | (tableBit << 16u));";
    const refresh = `
if (!isGpos && !cooperative && removed) { length = compactRun(base, length); }
if (!isGpos && atomicLoad(&idsChanged) != 0u) { rootPresence = activeRoots(base, length); }
atomicStore(&idsChanged, 0u);`;
    if (localScratch) {
        return `
if (cooperative) { ${selection} }
else if (lane == 0u) { ${ordered} }
workgroupBarrier();
if (lane == 0u) { if (cooperative) { ${commit} } ${refresh} }
`;
    }
    return `
if (cooperative) {
for (var tile = 0u; tile < count; tile += 256u) {
let tileCount = min(256u, count - tile);
${selection}
${storageFence}
workgroupBarrier();
if (lane == 0u) { ${commit} }
${storageFence}
if (workgroupUniformLoad(&terminal) != 0u) { return; }
}
} else {
if (lane == 0u) { ${ordered} }
${storageFence}
workgroupBarrier();
}
if (lane == 0u) { ${refresh} }
`;
}

function kernelSource(scratchCapacity: number): string {
    const localScratch = scratchCapacity > 0;
    const storageFence = localScratch ? "" : "storageBarrier();";
    return `
struct Params { runCount:u32, capacity:u32, direction:u32, glyphCount:u32, rootCount:u32, programWords:u32, outputRun:u32 }
@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> program: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;
@group(0) @binding(3) var<storage, read> runs: array<u32>;
${localScratch ? `var<workgroup> scratch:array<i32,${scratchCapacity * 8}>;` : "@group(0) @binding(4) var<storage, read_write> scratch: array<i32>;"}
@group(0) @binding(5) var<storage, read_write> output: array<i32>;
const GLYPH_WORDS: u32 = 8u;
var<workgroup> length:u32;
var<workgroup> terminal:u32;
var<workgroup> rootCount:u32;
var<workgroup> gsubCount:u32;
var<workgroup> rootPresence:u32;
var<workgroup> idsChanged:atomic<u32>;
var<workgroup> selected:array<u32,${scratchCapacity || 256}>;
var<workgroup> seconds:array<u32,${scratchCapacity || 256}>;
var<private> removed:bool;
var<private> nominalSource:bool;
${localScratch ? `var<workgroup> nominalMasks:array<u32,${scratchCapacity}>;` : ""}
${
    localScratch
        ? ""
        : `var<workgroup> presenceBits:atomic<u32>;
var<workgroup> presenceWitnesses:atomic<u32>;`
}
${COMPUTE_TEXT_PREDICATES}
fn glyphAt(base:u32, index:u32)->u32 { return base + index * GLYPH_WORDS; }
fn readGlyph(base:u32, index:u32)->u32 {
if (nominalSource) { return runs[base + 1u + index]; }
return u32(scratch[glyphAt(base, index)]);
}
${
    localScratch
        ? ""
        : `fn nominalMask(base:u32, index:u32)->u32 {
return program[params.programWords + params.rootCount * 2u + readGlyph(base, index)];
}`
}
fn live(base:u32, index:u32, count:u32, flag:u32)->bool {
  if (index >= count) { return false; }
  if (nominalSource) { return !skipped(flag, readGlyph(base, index)); }
  let at = glyphAt(base, index);
  return scratch[at + 6] == 0 && !skipped(flag, u32(scratch[at]));
}
fn activeRoots(base:u32, count:u32)->u32 {
  let masks = params.programWords + params.rootCount * 2u;
  var bits = 0u;
  var witnesses = 0u;
  for (var index = 0u; index < count; index++) {
    let at = glyphAt(base, index);
    if (nominalSource || scratch[at + 6] == 0) {
      let glyph = readGlyph(base, index);
      let first = program[masks + glyph];
      bits |= first;
      ${localScratch ? "if (nominalSource) { nominalMasks[index] = first; }" : ""}
      witnesses |= program[masks + params.glyphCount + glyph];
    }
  }
  return bits & witnesses;
}
${
    localScratch
        ? ""
        : `fn activeRootsParallel(base:u32, count:u32, lane:u32) {
let masks = params.programWords + params.rootCount * 2u;
var bits = 0u;
var witnesses = 0u;
for (var index = lane; index < count; index += 32u) {
let glyph = readGlyph(base, index);
let first = program[masks + glyph];
bits |= first;
witnesses |= program[masks + params.glyphCount + glyph];
}
atomicOr(&presenceBits, bits);
atomicOr(&presenceWitnesses, witnesses);
workgroupBarrier();
if (lane == 0u) { rootPresence = atomicLoad(&presenceBits) & atomicLoad(&presenceWitnesses); }
}`
}
fn replaceGlyph(at:u32, id:u32) {
  if (scratch[at] != i32(id)) {
    scratch[at] = i32(id);
    scratch[at + 2] = advanceOf(id);
    atomicStore(&idsChanged, 1u);
  }
}
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group:vec3u, @builtin(local_invocation_index) lane:u32) {
  let local = group.x;
  if (local >= params.runCount) { return; }
  let run = local + params.outputRun;
  let capacity = params.capacity;
  let stride = ${HEADER_WORDS}u + capacity * ${RECORD_WORDS}u;
  let out = run * stride;
  let inputOrigin = runs[local];
  let base = ${localScratch ? "0u" : "local * capacity * GLYPH_WORDS"};
  nominalSource = true;
  if (lane == 0u) {
    output[out] = 0;
    output[out + 1] = 0;
    length = runs[inputOrigin];
    terminal = 0u;
    atomicStore(&idsChanged, 0u);
    rootCount = params.rootCount;
    gsubCount = program[2u];
    if (length > capacity) { output[out] = ${STATUS_OVERFLOW}; terminal = 1u; }
    if (terminal == 0u${localScratch ? "" : " && length <= 256u"}) {
rootPresence = activeRoots(inputOrigin, length);
simpleRoots = proveSimpleRun(inputOrigin, length, rootPresence);
${localScratch ? "atomicStore(&firstFailure, 0xffffffffu);" : ""}
}
${
    localScratch
        ? ""
        : `atomicStore(&presenceBits, 0u);
atomicStore(&presenceWitnesses, 0u);
atomicStore(&firstFailureIndex, 0xffffffffu);`
}
${localScratch ? "" : "atomicStore(&firstFailure, 0xffffffffu);"}
  }
  ${storageFence}
  workgroupBarrier();
  if (workgroupUniformLoad(&terminal) != 0u) { return; }
  let scalarCount = workgroupUniformLoad(&length);
  ${
      localScratch
          ? ""
          : `if (scalarCount > 256u) {
activeRootsParallel(inputOrigin, scalarCount, lane);
let candidates = workgroupUniformLoad(&rootPresence);
let proved = proveSimpleRunParallel(inputOrigin, scalarCount, candidates, lane);
if (lane == 0u) { simpleRoots = proved; }
}`
  }
  let simple = workgroupUniformLoad(&simpleRoots);
if (simple.y != 0u) { finishSimpleRun(inputOrigin, scalarCount, simple.x, out, lane); return; }
nominalSource = false;
  for (var i = lane; i < scalarCount; i += 32u) {
    let at = glyphAt(base, i);
    let id = runs[inputOrigin + 1u + i];
    scratch[at] = i32(id);
    scratch[at + 1] = i32(i);
    scratch[at + 2] = advanceOf(id);
    scratch[at + 3] = 0;
    scratch[at + 4] = 0;
    scratch[at + 5] = 0;
    scratch[at + 6] = 0;
    scratch[at + 7] = i32(gdefOf(id) & 0xffffu);
  }
  ${storageFence}
  workgroupBarrier();
  let rootOrigin = 8u;
  let roots = workgroupUniformLoad(&rootCount);
  let substitutions = workgroupUniformLoad(&gsubCount);
  var candidates = simple.x;
  for (var root = 0u; root < roots; root++) {
    if (root < 32u && (candidates & (1u << root)) == 0u) { continue; }
    let tableBit = program[program[rootOrigin] + root * 2u];
    let lookupIndex = program[program[rootOrigin] + root * 2u + 1u];
    let descBase = select(program[4u], program[5u], tableBit == 1u) + lookupIndex * 8u;
    let flag = program[descBase + 1u];
    let ruleOrigin = program[descBase + 3u];
    let directory = program[descBase + 6u];
    let isGpos = tableBit == 1u;
    let count = ${localScratch ? "length" : "workgroupUniformLoad(&length)"};
    let capability = program[descBase + 2u];
    let cooperative = capability != 0u${localScratch ? " && capacity <= 256u" : ""};
    ${cooperativeSource(localScratch, storageFence)}
    ${storageFence}
    if (workgroupUniformLoad(&terminal) != 0u) { return; }
    if (root < substitutions) { candidates = workgroupUniformLoad(&rootPresence); }
  }
  if (lane == 0u) {
    if (params.direction == 1u) { reverseRun(base, length); }
    output[out + 1] = i32(length);
  }
  ${storageFence}
  workgroupBarrier();
  let count = workgroupUniformLoad(&length);
  for (var i = lane; i < count; i += 32u) {
    let at = glyphAt(base, i);
    let dst = out + ${HEADER_WORDS}u + i * ${RECORD_WORDS}u;
    for (var word = 0u; word < ${RECORD_WORDS}u; word++) { output[dst + word] = scratch[at + word]; }
  }
}
fn rejectRun(out:u32, lookup:u32, index:u32) {
  output[out] = ${STATUS_UNSUPPORTED};
  output[out + 2] = i32(lookup);
  output[out + 3] = i32(index);
  terminal = 1u;
}
fn selectPair(base:u32, index:u32, second:u32, count:u32, ruleOrigin:u32, directory:u32)->u32 {
  if (second >= count) { return 0u; }
  return selectPairIds(readGlyph(base, index), readGlyph(base, second), ruleOrigin, directory);
}
fn executeRoot(base:u32, inputCount:u32, flag:u32, ruleOrigin:u32, directory:u32, out:u32, lookup:u32)->u32 {
    removed = false;
    var count = inputCount;
    var index = 0u;
    loop {
      if (index >= count) { break; }
      var consumed = 1u;
      let entry = directory + u32(scratch[glyphAt(base, index)]) * 2u;
      let candidates = program[entry];
      let candidateCount = program[entry + 1u];
      if (candidateCount == 0u || !live(base, index, count, flag)) { index++; continue; }
      for (var candidate = 0u; candidate < candidateCount; candidate++) {
        var ruleAt = ruleOrigin + program[candidates + candidate] * 16u;
        let row = program[ruleAt + 10u];
        if (row != 0u) {
          candidate += program[ruleAt + 13u] - 1u;
          let second = pairSecond(base, count, index, flag);
          if (second >= count) { continue; }
          let cls = classOf(program[ruleAt + 11u], u32(scratch[glyphAt(base, second)]));
          if (cls >= program[ruleAt + 12u]) { continue; }
          let selected = program[row + cls];
          if (selected == 0u) { continue; }
          ruleAt = ruleOrigin + (selected - 1u) * 16u;
        }
        let mode = program[ruleAt + 8u];
        // mode 5 is an unsupported-lookup failure, never a successful skip
        if (mode == 5u && ruleMatches(base, count, index, flag, ruleAt)) {
          rejectRun(out, lookup, index);
          return count;
        }
        if (mode == 12u && matchInput(base, count, index, flag, ruleAt)) {
          let attached = applyMarkLigature(base, index, program[ruleAt + 6u]);
          if (attached == 2u) {
            rejectRun(out, lookup, index);
            return count;
          }
          if (attached == 1u) { break; }
          continue;
        }
        if (mode == 11u && matchInput(base, count, index, flag, ruleAt)) {
          let produced = applyMultiple(base, count, index, program[ruleAt + 6u]);
          if (produced == 0xffffffffu) { output[out] = ${STATUS_OVERFLOW}; terminal = 1u; return count; }
          if (produced == 0u) { continue; }
          count += produced - 1u;
          consumed = produced;
          break;
        }
        if (mode != 5u && mode != 11u && applyRule(base, count, index, flag, ruleAt, mode)) { break; }
      }
      index += consumed;
    }
    return count;
}
fn compactRun(base:u32, count:u32)->u32 {
  var write = 0u;
  for (var i = 0u; i < count; i++) {
    let at = glyphAt(base, i);
    if (scratch[at + 6] == 0) {
      if (write != i) {
        let dst = glyphAt(base, write);
        for (var word = 0u; word < GLYPH_WORDS; word++) { scratch[dst + word] = scratch[at + word]; }
      }
      write++;
    }
  }
  return write;
}
fn reverseRun(base:u32, count:u32) {
  if (count < 2u) { return; }
  for (var i = 0u; i < count / 2u; i++) {
    let left = glyphAt(base, i);
    let right = glyphAt(base, count - 1u - i);
    for (var word = 0u; word < GLYPH_WORDS; word++) {
      let tmp = scratch[left + word];
      scratch[left + word] = scratch[right + word];
      scratch[right + word] = tmp;
    }
  }
}
fn ruleMatches(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32)->bool {
  let mode = program[ruleAt + 8u];
  if (mode == 3u || mode == 4u) {
    let second = pairSecond(base, count, index, flag);
    return second < count && matchInput(base, count, index, flag, ruleAt);
  }
  return matchInput(base, count, index, flag, ruleAt);
}
fn applyRule(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32, mode:u32)->bool {
  if (!matchInput(base, count, index, flag, ruleAt)) { return false; }
  let action = program[ruleAt + 6u];
  if (mode == 1u) { replaceGlyph(glyphAt(base, index), program[action + 1u]); return true; }
  if (mode == 2u) { return applyContext(base, count, index, action); }
  if (mode == 9u) { return applyLigature(base, count, index, flag, ruleAt, action); }
  if (mode == 3u || mode == 4u) { return applyPair(base, count, index, flag, action, mode); }
  if (mode == 8u || mode == 7u) { return applyMark(base, count, index, action, mode); }
  if (mode == 10u) {
    let at = glyphAt(base, index);
    scratch[at + 4] += bitcast<i32>(program[action]);
    scratch[at + 5] += bitcast<i32>(program[action + 1u]);
    scratch[at + 2] += bitcast<i32>(program[action + 2u]);
    scratch[at + 3] += bitcast<i32>(program[action + 3u]);
    return true;
  }
  return false;
}
fn applyContext(base:u32, count:u32, index:u32, action:u32)->bool {
  let nested = program[action];
  for (var i = 0u; i < nested; i++) {
    let at = index + program[action + 1u + i * 2u];
    let lookup = program[action + 2u + i * 2u];
    if (at >= count) { continue; }
    let desc = program[4u] + lookup * 8u;
    let ruleOrigin = program[desc + 3u];
    let glyph = u32(scratch[glyphAt(base, at)]);
    let entry = program[desc + 6u] + glyph * 2u;
    let candidates = program[entry];
    for (var candidate = 0u; candidate < program[entry + 1u]; candidate++) {
      let ruleAt = ruleOrigin + program[candidates + candidate] * 16u;
      if (program[ruleAt + 8u] == 5u) {
        return true;
      }
      if (program[ruleAt + 8u] == 1u) {
        replaceGlyph(glyphAt(base, at), program[program[ruleAt + 6u] + 1u]);
        break;
      }
    }
  }
  return true;
}
fn applyMarkLigature(base:u32, index:u32, action:u32)->u32 {
  var lig = -1;
  var between = 0;
  for (var i = 0u; i < index; i++) {
    let cls = scratch[glyphAt(base, i) + 7];
    if (cls == 0 || cls == 1 || cls == 2) { lig = i32(i); between = 0; }
    else if (cls == 3 && lig >= 0) { between++; }
  }
  if (lig < 0 || scratch[glyphAt(base, u32(lig)) + 7] != 2) { return 0u; }
  let ligId = u32(scratch[glyphAt(base, u32(lig))]);
  var cursor = action + 3u;
  let ligatures = program[action + 2u];
  for (var n = 0u; n < ligatures; n++) {
    let glyphId = program[cursor];
    let componentCount = program[cursor + 1u];
    let anchorCount = program[cursor + 2u];
    cursor += 3u;
    if (glyphId == ligId) {
      if (componentCount == 0u) { return 0u; }
      let component = min(u32(between), componentCount - 1u);
      for (var a = 0u; a < anchorCount; a++) {
        let at = cursor + a * 4u;
        if (program[at] == component) {
          if (program[at + 1u] != 0u) { return 2u; }
          let mark = glyphAt(base, index);
          let baseAt = glyphAt(base, u32(lig));
          scratch[mark + 4] = bitcast<i32>(program[at + 2u]) - bitcast<i32>(program[action]) + scratch[baseAt + 4];
          scratch[mark + 5] = bitcast<i32>(program[at + 3u]) - bitcast<i32>(program[action + 1u]) + scratch[baseAt + 5];
          scratch[mark + 2] = 0;
          scratch[mark + 3] = 0;
          return 1u;
        }
      }
      return 0u;
    }
    cursor += anchorCount * 4u;
  }
  return 0u;
}
fn applyMultiple(base:u32, count:u32, index:u32, action:u32)->u32 {
  let produced = program[action];
  if (produced == 0u) { return 0u; }
  let extra = produced - 1u;
  if (count + extra > params.capacity) { return 0xffffffffu; }
  var src = count;
  loop {
    if (src <= index + 1u) { break; }
    src--;
    let sourceAt = glyphAt(base, src);
    let destAt = glyphAt(base, src + extra);
    for (var word = 0u; word < GLYPH_WORDS; word++) { scratch[destAt + word] = scratch[sourceAt + word]; }
  }
  let at = glyphAt(base, index);
  let cluster = scratch[at + 1];
  let first = program[action + 1u];
  replaceGlyph(at, first);
  if (produced > 1u) { atomicStore(&idsChanged, 1u); }
  scratch[at + 7] = i32(gdefOf(first) & 0xffffu);
  for (var step = 1u; step < produced; step++) {
    let dst = glyphAt(base, index + step);
    let id = program[action + 1u + step];
    scratch[dst] = i32(id);
    scratch[dst + 1] = cluster;
    scratch[dst + 2] = advanceOf(id);
    scratch[dst + 3] = 0;
    scratch[dst + 4] = 0;
    scratch[dst + 5] = 0;
    scratch[dst + 6] = 0;
    scratch[dst + 7] = i32(gdefOf(id) & 0xffffu);
  }
  return produced;
}
fn applyLigature(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32, action:u32)->bool {
  let inputCount = program[ruleAt + 1u];
  var cursor = index;
  var cluster = scratch[glyphAt(base, index) + 1];
  for (var step = 0u; step < inputCount; step++) {
    if (step > 0u) { cursor = nextLive(base, count, cursor + 1u, flag, 1); }
    if (cursor >= count) { return false; }
    cluster = min(cluster, scratch[glyphAt(base, cursor) + 1]);
    if (step > 0u) { scratch[glyphAt(base, cursor) + 6] = 1; }
  }
  let at = glyphAt(base, index);
  replaceGlyph(at, program[action + 1u]);
  scratch[at + 1] = cluster;
  removed = true;
  atomicStore(&idsChanged, 1u);
  return true;
}
fn applyPair(base:u32, count:u32, index:u32, flag:u32, action:u32, mode:u32)->bool {
  let second = pairSecond(base, count, index, flag);
  if (second >= count) { return false; }
  let secondId = u32(scratch[glyphAt(base, second)]);
  if (mode == 3u && program[action] != secondId) { return false; }
  commitPair(base, index, second, action, mode);
  return true;
}
fn commitPairFirst(base:u32, index:u32, action:u32, mode:u32) {
  scratch[glyphAt(base, index) + 2] += bitcast<i32>(program[action + select(1u, 2u, mode == 4u)]);
}
fn commitPair(base:u32, index:u32, second:u32, action:u32, mode:u32) {
  commitPairFirst(base, index, action, mode);
  let applySecond = program[action + select(3u, 4u, mode == 4u)];
  if (applySecond != 0u) { scratch[glyphAt(base, second) + 2] += bitcast<i32>(program[action + select(2u, 3u, mode == 4u)]); }
}
fn applyMark(base:u32, count:u32, index:u32, action:u32, mode:u32)->bool {
  var baseIndex = -1;
  if (mode == 7u) {
    if (index == 0u || scratch[glyphAt(base, index - 1u) + 7] != 3) { return false; }
    baseIndex = i32(index) - 1;
  } else {
    for (var i = 0; i < i32(index); i++) { if (scratch[glyphAt(base, u32(i)) + 7] != 3) { baseIndex = i; } }
    if (baseIndex < 0) { return false; }
  }
  let baseId = u32(scratch[glyphAt(base, u32(baseIndex))]);
  let baseCount = program[action + 3u];
  var low = 0u;
  var high = baseCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (program[action + 4u + middle * 3u] <= baseId) { low = middle + 1u; }
    else { high = middle; }
  }
  if (low == 0u) { return false; }
  let anchor = action + 4u + (low - 1u) * 3u;
  if (program[anchor] != baseId) { return false; }
  let ax = bitcast<i32>(program[anchor + 1u]);
  let ay = bitcast<i32>(program[anchor + 2u]);
  let at = glyphAt(base, index);
  let baseAt = glyphAt(base, u32(baseIndex));
  scratch[at + 4] = ax - bitcast<i32>(program[action + 1u]) + scratch[baseAt + 4];
  scratch[at + 5] = ay - bitcast<i32>(program[action + 2u]) + scratch[baseAt + 5];
  if (mode == 8u) { scratch[at + 2] = 0; scratch[at + 3] = 0; }
  return true;
}
${COMPUTE_TEXT_INPUT_PREDICATES}
${computeTextSimpleSource(!localScratch)}
`;
}

/** @internal Zero selects global scratch; a sufficient bounded variant survives shrinking. */
export function computeTextScratchCapacity(capacity: number, retained = 0): number {
    if (capacity > 256) {
        return 0;
    }
    let size = Math.max(1, retained);
    while (size < capacity) {
        size *= 2;
    }
    return size;
}

/** @internal One shared kernel constructor for readback and resident execution. */
export function createComputeTextShapeShader(engine: EngineContext, capacity: number): ReturnType<typeof createComputeShader> {
    return createComputeShader(engine, {
        name: "compute-text-shape",
        computeSource: kernelSource(computeTextScratchCapacity(capacity)),
        bindings: [
            computeStorageBufferBinding("params", { group: 0, binding: 0 }),
            computeStorageBufferBinding("program", { group: 0, binding: 1 }),
            computeStorageBufferBinding("tables", { group: 0, binding: 2 }),
            computeStorageBufferBinding("runs", { group: 0, binding: 3 }),
            computeStorageBufferBinding("scratch", { group: 0, binding: 4, access: "read-write" }),
            computeStorageBufferBinding("output", { group: 0, binding: 5, access: "read-write" }),
        ],
    });
}

interface GpuCache {
    readonly device: GPUDevice;
    readonly plan: ComputeTextPlan;
    runCapacity: number;
    glyphCapacity: number;
    scratchCapacity: number;
    shader: ReturnType<typeof createComputeShader>;
    params: StorageBuffer;
    program: StorageBuffer;
    tables: StorageBuffer;
    runs: StorageBuffer;
    scratch: StorageBuffer;
    output: StorageBuffer;
    task: ComputeTask;
    dispatch: ComputeDispatch;
    taskPrepared: boolean;
}

let caches: WeakMap<ComputeTextShaper, GpuCache> | null = null;

interface ComputeTextPreparation {
    readonly program: ReturnType<typeof compileComputeTextProgram>;
    readonly programWords: Uint32Array;
    readonly baseProgramWords: Uint32Array;
    readonly tables: Uint32Array;
    readonly simpleWordCount: number;
}

let preparations: WeakMap<ComputeTextPlan, ComputeTextPreparation> | null = null;

function prepareImmutable(plan: ComputeTextPlan): ComputeTextPreparation {
    const existing = preparations?.get(plan);
    if (existing) {
        return existing;
    }
    const program = compileComputeTextProgram(plan);
    const roots = packRoots(program);
    const simpleOrigin = program.words.length + roots.length + plan.font.glyphCount * 2;
    const simpleWords = prepareSimplePairWords(program, simpleOrigin);
    const programWords = new Uint32Array(simpleOrigin + simpleWords.length);
    programWords.set(program.words);
    programWords.set(roots, program.words.length);
    programWords[8] = program.words.length;
    const masks = program.words.length + roots.length;
    programWords.set(program.rootWitnesses, masks + plan.font.glyphCount);
    programWords.set(simpleWords, simpleOrigin);
    program.rootOrder.slice(0, 32).forEach((index, root) => {
        const lookup = program.lookups[index]!;
        const descriptor = program.words[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
        const directory = program.words[descriptor + 6]!;
        if (directory === 0) {
            return;
        }
        for (let glyph = 0; glyph < plan.font.glyphCount; glyph++) {
            if (program.words[directory + glyph * 2 + 1]! > 0) {
                programWords[masks + glyph] = (programWords[masks + glyph]! | (1 << root)) >>> 0;
            }
        }
    });
    const prepared = {
        program,
        programWords,
        baseProgramWords: programWords.subarray(0, simpleOrigin),
        tables: packTables(plan, program.classTables),
        simpleWordCount: simpleWords.length,
    };
    (preparations ??= new WeakMap()).set(plan, prepared);
    return prepared;
}

/** @internal CPU-side words for one shape dispatch. `outputRun` places this dispatch's records in a shared buffer. */
export interface ComputeTextShapeWords {
    readonly capacity: number;
    readonly params: Uint32Array;
    readonly programWords: Uint32Array;
    readonly tables: Uint32Array;
    readonly runs: Uint32Array;
    readonly scratchBytes: number;
    readonly program: ReturnType<typeof compileComputeTextProgram>;
    readonly simpleWordCount: number;
}

/** @internal Required stride capacity without allocating mutable dispatch input. */
export function computeTextShapeCapacity(plan: ComputeTextPlan, runs: readonly { readonly scalars: Uint32Array }[]): number {
    const expansion = Math.max(1, prepareImmutable(plan).program.maxExpansion);
    let capacity = 1;
    for (const run of runs) {
        capacity = Math.max(capacity, run.scalars.length * expansion);
    }
    return capacity;
}

/** @internal Pack one plan's runs. Pass `capacityOverride` when several dispatches share one record stride. */
export function prepareComputeTextShapeWords(
    plan: ComputeTextPlan,
    runs: readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array }[],
    outputRun = 0,
    capacityOverride?: number,
    limits?: Pick<GPUSupportedLimits, "maxBufferSize" | "maxStorageBufferBindingSize">
): ComputeTextShapeWords {
    const immutable = prepareImmutable(plan);
    const program = immutable.program;
    const needed = computeTextShapeCapacity(plan, runs);
    const capacity = capacityOverride ?? needed;
    if (!Number.isSafeInteger(capacity) || capacity < needed) {
        throw new Error("shapeComputeText: shared capacity is smaller than this plan's expansion");
    }
    const limit = limits ? Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize) : Infinity;
    const programWords = immutable.programWords.byteLength <= limit ? immutable.programWords : immutable.baseProgramWords;
    return {
        capacity,
        params: new Uint32Array([runs.length, capacity, plan.direction === "rtl" ? 1 : 0, plan.font.glyphCount, program.rootOrder.length, program.words.length, outputRun]),
        programWords,
        tables: immutable.tables,
        runs: packRuns(runs),
        scratchBytes: Math.max(runs.length, 1) * capacity * 8 * 4,
        program,
        simpleWordCount: programWords === immutable.programWords ? immutable.simpleWordCount : 0,
    };
}

export async function executeComputeTextShape(
    shaper: ComputeTextShaper,
    plan: ComputeTextPlan,
    runs: readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array; readonly sourceRun?: number }[],
    generation: number
): Promise<ComputeTextGlyph[][]> {
    const engine = shaper._engine;
    assertLive(shaper, generation);
    const prepared = prepareComputeTextShapeWords(plan, runs, 0, undefined, engine._device.limits);
    checkComputeTextShapeBounds(engine._device.limits, prepared);
    const capacity = prepared.capacity;
    const cache = ensureCache(shaper, engine, plan, runs.length, capacity, prepared);
    updateStorageBuffer(engine, cache.params, prepared.params);
    updateStorageBuffer(engine, cache.runs, prepared.runs);
    setComputeDispatchSize(cache.dispatch, { x: runs.length });
    if (!cache.taskPrepared) {
        await prepareComputeTask(cache.task);
        assertLive(shaper, generation);
        cache.taskPrepared = true;
    }
    assertLive(shaper, generation);
    const result = await readComputeTaskResult([cache.task], cache.output, runs.length * (HEADER_WORDS + capacity * RECORD_WORDS) * 4, (bytes) => {
        assertLive(shaper, generation);
        return decode(bytes, runs, capacity);
    });
    assertLive(shaper, generation);
    return result;
}

/** Check packed allocation arithmetic against the executing device before creating buffers. */
export function checkComputeTextShapeBounds(limits: GPUSupportedLimits, prepared: ComputeTextShapeWords): void {
    const runCount = prepared.params[0]!;
    if (runCount > limits.maxComputeWorkgroupsPerDimension) {
        throw new Error("shapeComputeText: run count exceeds device maxComputeWorkgroupsPerDimension");
    }
    const outputBytes = (prepared.params[6]! + runCount) * (HEADER_WORDS + prepared.capacity * RECORD_WORDS) * 4;
    for (const bytes of [outputBytes, prepared.scratchBytes, prepared.params.byteLength, prepared.programWords.byteLength, prepared.tables.byteLength, prepared.runs.byteLength]) {
        if (!Number.isSafeInteger(bytes) || bytes > limits.maxBufferSize) {
            throw new Error(`shapeComputeText: buffer of ${bytes} bytes exceeds device maxBufferSize`);
        }
        if (bytes > limits.maxStorageBufferBindingSize) {
            throw new Error(`shapeComputeText: binding of ${bytes} bytes exceeds device maxStorageBufferBindingSize`);
        }
    }
}

function ensureCache(
    shaper: ComputeTextShaper,
    engine: EngineContext,
    plan: ComputeTextPlan,
    runCount: number,
    glyphCapacity: number,
    prepared: Pick<ComputeTextPreparation, "programWords" | "tables">
): GpuCache {
    const existing = caches?.get(shaper);
    const previous = existing && existing.device === engine._device && existing.plan === plan ? existing : undefined;
    const scratchCapacity = computeTextScratchCapacity(glyphCapacity, previous?.scratchCapacity);
    if (previous && previous.runCapacity >= runCount && previous.glyphCapacity >= glyphCapacity) {
        if (previous.scratchCapacity !== scratchCapacity) {
            disposeComputeShader(previous.shader);
            previous.task.dispose();
            previous.shader = createComputeTextShapeShader(engine, glyphCapacity);
            previous.scratchCapacity = scratchCapacity;
            Object.assign(previous, createShapeSchedule(engine, previous.shader, previous, runCount));
        }
        return previous;
    }
    const limits = engine._device.limits;
    const outputBytes = runCount * (HEADER_WORDS + glyphCapacity * RECORD_WORDS) * 4;
    const scratchBytes = runCount * glyphCapacity * 8 * 4;
    for (const bytes of [outputBytes, scratchBytes, prepared.programWords.byteLength]) {
        if (bytes > limits.maxBufferSize) {
            throw new Error(`shapeComputeText: buffer of ${bytes} bytes exceeds device maxBufferSize`);
        }
    }
    const shader = previous?.scratchCapacity === scratchCapacity ? previous.shader : createComputeTextShapeShader(engine, glyphCapacity);
    if (existing && existing.shader !== shader) {
        disposeComputeShader(existing.shader);
    }
    existing?.task.dispose();
    disposeBuffers(existing, !!previous);
    const buffers = {
        params: previous?.params ?? createStorageBuffer(engine, 28, { label: "compute-text-params" }),
        program: previous?.program ?? createStorageBuffer(engine, prepared.programWords, { label: "compute-text-program" }),
        tables: previous?.tables ?? createStorageBuffer(engine, prepared.tables, { label: "compute-text-tables" }),
        runs: createStorageBuffer(engine, Math.max(4, runCount * (glyphCapacity + 2) * 4), { label: "compute-text-runs", writable: false }),
        scratch: createStorageBuffer(engine, Math.max(4, scratchBytes), { label: "compute-text-scratch", writable: true }),
        output: createStorageBuffer(engine, Math.max(4, outputBytes), { label: "compute-text-output", writable: true }),
    };
    const cache: GpuCache = {
        device: engine._device,
        plan,
        runCapacity: runCount,
        glyphCapacity,
        scratchCapacity,
        shader,
        ...buffers,
        ...createShapeSchedule(engine, shader, buffers, runCount),
        taskPrepared: previous?.shader === shader && previous.taskPrepared,
    };
    (caches ??= new WeakMap()).set(shaper, cache);
    shaper._releaseGpu = () => {
        disposeComputeShader(cache.shader);
        cache.task.dispose();
        disposeBuffers(cache);
        caches?.delete(shaper);
    };
    return cache;
}

function createShapeSchedule(
    engine: EngineContext,
    shader: ReturnType<typeof createComputeShader>,
    buffers: Pick<GpuCache, "params" | "program" | "tables" | "runs" | "scratch" | "output">,
    runCount: number
): { task: ComputeTask; dispatch: ComputeDispatch; taskPrepared: boolean } {
    const bindings = createComputeBindingSet(shader, {
        params: buffers.params,
        program: buffers.program,
        tables: buffers.tables,
        runs: buffers.runs,
        scratch: buffers.scratch,
        output: buffers.output,
    });
    const dispatch = createComputeDispatch(shader, bindings, { size: { x: runCount } });
    const task = createComputeTask(engine, "compute-text-shape");
    addComputeDispatch(task, dispatch);
    task.record();
    return { task, dispatch, taskPrepared: false };
}

function packRoots(program: ReturnType<typeof compileComputeTextProgram>): Uint32Array {
    const words = new Uint32Array(program.rootOrder.length * 2);
    program.rootOrder.forEach((index, at) => {
        const lookup = program.lookups[index];
        words[at * 2] = lookup?.table === "GPOS" ? 1 : 0;
        words[at * 2 + 1] = lookup?.lookupIndex ?? 0;
    });
    return words;
}

function packTables(plan: ComputeTextPlan, classTables: readonly GlyphClassDef[]): Uint32Array {
    const glyphCount = plan.font.glyphCount;
    const advances = new Uint32Array(glyphCount);
    const gdefWords = new Uint32Array(glyphCount);
    const gdef = ensureComputeFontLayout(plan.font).gdef;
    for (let id = 0; id < glyphCount; id++) {
        advances[id] = readComputeFontAdvance(plan.font, id);
        const cls = glyphClass(gdef?.glyphClassDef ?? null, id);
        const attach = glyphClass(gdef?.markAttachClassDef ?? null, id);
        gdefWords[id] = cls | (attach << 16);
    }
    const header = 6 + classTables.length;
    const classOrigin = header + advances.length + gdefWords.length;
    const words = new Uint32Array(classOrigin + classTables.length * glyphCount);
    words[0] = glyphCount;
    words[1] = header;
    words[2] = header + advances.length;
    words[3] = header + advances.length + gdefWords.length;
    words[4] = gdef ? 1 : 0;
    words[5] = classTables.length;
    words.set(advances, header);
    words.set(gdefWords, header + advances.length);
    classTables.forEach((table, index) => {
        const origin = classOrigin + index * glyphCount;
        words[6 + index] = origin;
        for (let glyph = 0; glyph < glyphCount; glyph++) {
            words[origin + glyph] = glyphClass(table, glyph);
        }
    });
    return words;
}

function packRuns(runs: readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array }[]): Uint32Array {
    let length = runs.length;
    for (const run of runs) {
        length += 1 + run.glyphIds.length;
    }
    const words = new Uint32Array(length);
    let cursor = runs.length;
    runs.forEach((run, index) => {
        words[index] = cursor;
        words[cursor] = run.scalars.length;
        cursor++;
        words.set(run.glyphIds, cursor);
        cursor += run.glyphIds.length;
    });
    return words;
}

function decode(buffer: ArrayBuffer, runs: readonly { readonly scalars: Uint32Array; readonly sourceRun?: number }[], capacity: number): ComputeTextGlyph[][] {
    const view = new DataView(buffer);
    const stride = (HEADER_WORDS + capacity * RECORD_WORDS) * 4;
    return runs.map((run, index) => {
        const offset = index * stride;
        const status = view.getInt32(offset, true);
        const count = view.getInt32(offset + 4, true);
        const origin = view.getUint32(offset + 8, true);
        const lookup = origin & 0xffff;
        const table = origin >>> 16 === 1 ? "GPOS" : "GSUB";
        const sourceRun = run.sourceRun ?? index;
        if (status === STATUS_UNSUPPORTED) {
            throw new Error(`shapeComputeText: run ${sourceRun} unsupported ${table} ${lookup} matched at glyph ${view.getInt32(offset + 12, true)}`);
        }
        if (status === STATUS_OVERFLOW) {
            throw new Error(`shapeComputeText: run ${sourceRun} exceeds its reserved capacity`);
        }
        if (status !== STATUS_OK || count < 0 || count > capacity) {
            throw new Error(`shapeComputeText: run ${sourceRun} kernel status ${status}`);
        }
        const glyphs = new Array<ComputeTextGlyph>(count);
        for (let glyph = 0; glyph < count; glyph++) {
            const at = offset + HEADER_WORDS * 4 + glyph * RECORD_WORDS * 4;
            glyphs[glyph] = {
                glyphId: view.getInt32(at, true),
                cluster: view.getInt32(at + 4, true),
                xAdvance: view.getInt32(at + 8, true),
                yAdvance: view.getInt32(at + 12, true),
                xOffset: view.getInt32(at + 16, true),
                yOffset: view.getInt32(at + 20, true),
            };
        }
        return glyphs;
    });
}

function disposeBuffers(cache: GpuCache | undefined, retainImmutable = false): void {
    if (!cache) {
        return;
    }
    if (!retainImmutable) {
        for (const buffer of [cache.params, cache.program, cache.tables]) {
            disposeStorageBuffer(buffer);
        }
    }
    for (const buffer of [cache.runs, cache.scratch, cache.output]) {
        disposeStorageBuffer(buffer);
    }
}

function assertLive(shaper: ComputeTextShaper, generation: number): void {
    if (shaper.status === "disposed" || shaper._generation !== generation) {
        throw new Error("shapeComputeText: shaper is disposed");
    }
    if (shaper.status === "lost" || shaper._engine._device !== shaper._device) {
        throw new Error("shapeComputeText: device lost");
    }
}
