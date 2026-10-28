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
import type { ComputeTextKernelFragments, ComputeTextPlan, ComputeTextProfileId } from "../compute-text-plan.js";
import { profileSupport } from "../compute-text-plan.js";
import type { ComputeTextGlyph, ComputeTextShaper } from "../compute-text-shaper.js";
import type { GlyphClassDef } from "../font-parser/opentype-layout.js";
import { coverageGlyphs, glyphClass } from "../font-parser/opentype-layout.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import { computeTextInputPredicates, COMPUTE_TEXT_PREDICATES } from "./compute-text-predicates.js";
import { computeTextSimpleSource } from "./compute-text-simple-wgsl.js";
import { prepareSimplePairWords } from "./compute-text-pair-program.js";

const STATUS_OK = 0;
const STATUS_UNSUPPORTED = 1;
const STATUS_OVERFLOW = 2;
const RECORD_WORDS = 6;
const HEADER_WORDS = 4;
const GDEF_MARK_CLASS = 3;
/** Run header bit: no nominal glyph is a GDEF mark, so the kernel may ignore mark filtering sets. */
const RUN_MARK_FREE = 0x80000000;
/** @internal Record word 7: GDEF class 0..4, then an unsigned backward distance. */
export const RECORD_CLASS_MASK = 0x7;
/** @internal Keep attachment-class lookup data in tables, not in the glyph record. */
export const ATTACH_SHIFT = 3;

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

/** @internal Compile-time execution footprint and optional script hooks. Semantics stay shared. */
export interface ComputeTextKernelVariant {
    readonly id: ComputeTextProfileId;
    readonly glyphWords: 8 | 16;
    readonly masked: boolean;
    readonly fragments?: ComputeTextKernelFragments;
    readonly planes: number;
    readonly auxBytes?: (capacity: number) => number;
}

/** @internal */
export function computeTextPlanVariant(plan: ComputeTextPlan): ComputeTextKernelVariant {
    const support = profileSupport(plan._profile);
    if (plan._profile === "arabic" && !support) {
        throw new Error("prepareComputeTextShapeWords: Arabic profile is not loaded");
    }
    return {
        id: plan._profile,
        glyphWords: support?.glyphWords ?? 8,
        masked: support?.masked ?? false,
        fragments: support?.fragments,
        planes: support?.planes ?? 1,
        auxBytes: support?.auxBytes,
    };
}

/** @internal Includes the actual memory tier, not the reference implementation. */
export function computeTextVariantKey(variant: ComputeTextKernelVariant, tier: number): string {
    return `${variant.id}:${variant.glyphWords}:${tier}`;
}

/** @internal Conservative workgroup budget including shared interpreter arrays/latches. */
export function computeTextVariantTier(capacity: number, variant: ComputeTextKernelVariant, storageLimit = 16384, retained = 0): number {
    const tier = computeTextScratchCapacity(capacity, retained, variant.glyphWords);
    const bytes = tier * (variant.glyphWords * 4 * variant.planes + 12) + (variant.auxBytes?.(tier) ?? 0) + 1024 + (variant.id === "arabic" ? 256 : 0);
    return bytes > storageLimit ? 0 : tier;
}

export const COMPUTE_TEXT_KERNEL_SOURCE = /* @__PURE__ */ kernelSource(0);

function cooperativeSource(localScratch: boolean, storageFence: string, masked = false): string {
    const slot = localScratch ? "index" : "slot";
    const selection = `
for (var ${slot} = lane; ${slot} < ${localScratch ? "count" : "tileCount"}; ${slot} += 32u) {
${localScratch ? "" : "let index = tile + slot;"}
selected[${slot}] = 0u;
if (!live(base, index, count, flag)) { continue; }
${masked ? "if ((glyphMask(glyphAt(base, index)) & rootMask) == 0u) { continue; }\n" : ""}if (capability == 1u) {
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
    const orderedSelection = `
if (lane < 8u) { atomicStore(&indexBits[lane], 0u); }
workgroupBarrier();
for (var ${slot} = lane; ${slot} < ${localScratch ? "count" : "tileCount"}; ${slot} += 32u) {
${localScratch ? "" : "let index = tile + slot;"}
selected[${slot}] = selectRule(base, count, index, flag, ruleOrigin, directory);
seconds[${slot}] = lowRead;
if (selected[${slot}] != 0u) { atomicOr(&indexBits[${slot} / 32u], 1u << (${slot} % 32u)); }
}`;
    const orderedCommit = `
var frontier = 0u;
var wrote = false;
loop {
if (orderedCursor >= ${localScratch ? "length" : "tile + tileCount"}) { break; }
if (orderedCursor < count && (isGpos || (orderedForward && (!wrote || orderedCursor > frontier)))) {
orderedCursor = ${localScratch ? "" : "tile + "}nextSelected(${localScratch ? "orderedCursor, count" : "orderedCursor - tile, tileCount"});
if (orderedCursor >= ${localScratch ? "count" : "tile + tileCount"}) { break; }
}
let index = orderedCursor;
var chosen = 0u;
if (index < count && (isGpos || !wrote || (index > frontier && seconds[${localScratch ? "index" : "index - tile"}] > frontier))) {
chosen = selected[${localScratch ? "index" : "index - tile"}];
} else { chosen = selectRule(base, length, index, flag, ruleOrigin, directory); }
if (chosen == 0u) { orderedCursor++; continue; }
let applied = applySelected(base, length, index, flag, chosen - 1u, out, lookupIndex | (tableBit << 16u));
if (terminal != 0u) { break; }
frontier = max(frontier, applied.y);
wrote = true;
length = u32(i32(length) + actionDelta);
orderedCursor += applied.x;
}`;
    const refresh = `
refreshFlags = select(0u, 1u, !isGpos && !cooperative && removed)
| select(0u, 2u, !isGpos && atomicLoad(&idsChanged) != 0u)
| select(0u, 4u, terminal != 0u);
atomicStore(&idsChanged, 0u);`;
    if (localScratch) {
        return `
if (cooperative) { ${selection} }
else if (orderedParallel) { ${orderedSelection} }
else if (lane == 0u) { ${ordered} }
workgroupBarrier();
if (lane == 0u) {
if (cooperative) { ${commit} }
else if (orderedParallel) { removed = false; var orderedCursor = 0u; ${orderedCommit} }
${refresh}
}
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
} else if (orderedParallel) {
var orderedCursor = 0u;
if (lane == 0u) { removed = false; }
for (var tile = 0u; tile < count; tile += 256u) {
let tileCount = min(256u, count - tile);
${orderedSelection}
${storageFence}
workgroupBarrier();
if (lane == 0u) { ${orderedCommit} }
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

/** Presence scans share matching/addressing; only the general scan publishes all words. */
function presenceSource(name: "activeRoots" | "activeRootsProbe", mode: "all-words" | "word0-probe", localScratch: boolean): string {
    const allWords = mode === "all-words";
    const offset = allWords ? " + word" : "";
    const value = allWords ? "present" : "first";
    const scan = `  for (var index = 0u; index < count; index++) {
    let at = glyphAt(base, index);
    if (nominalSource || (scratch[at + 6] & 1) == 0) {
      let glyph = readGlyph(base, index);
      let ${value} = program[masks + glyph * width${offset}];
      bits |= ${value};
      ${localScratch ? (allWords ? "if (nominalSource && word == 0u) { nominalMasks[index] = present; }" : "if (nominalSource) { nominalMasks[index] = first; }") : ""}
      witnesses |= program[masks + params.glyphCount * width + glyph * width${offset}];
    }
  }`;
    return `fn ${name}(base:u32, count:u32)->u32 {
  let masks = params.programWords + params.rootCount * 4u;
  let width = program[13u];
${allWords ? "  var first = 0u;\n  for (var word = 0u; word < width; word++) {\n" : ""}  var bits = 0u;
  var witnesses = 0u;
${scan}
${allWords ? "  rootPresence[word] = bits & witnesses;\n  if (word == 0u) { first = bits & witnesses; }\n  }\n  return first;" : "  return bits & witnesses;"}
}`;
}

export function kernelSource(scratchCapacity: number, variant?: ComputeTextKernelVariant): string {
    const glyphWords = variant?.glyphWords ?? 8;
    const fragments = variant?.fragments;
    const masked = variant?.masked ?? false;
    const profileGates = !!fragments?.gates;
    const simpleEnabled = !masked;
    if (masked && !fragments?.glyphMask) {
        throw new Error("compute text masked variant has no glyphMask hook");
    }
    const loadGlyph = (word: string): string => (fragments?.loadGlyph ? `loadGlyph(${word})` : word);
    const localScratch = scratchCapacity > 0;
    const splitFastState = simpleEnabled && localScratch;
    const storageFence = localScratch ? "" : "storageBarrier();";
    return `
struct Params { runCount:u32, capacity:u32, direction:u32, glyphCount:u32, rootCount:u32, programWords:u32, outputRun:u32, profile:u32 }
@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> program: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;
@group(0) @binding(3) var<storage, read> runs: array<u32>;
${localScratch ? `var<workgroup> scratch:array<i32,${scratchCapacity * glyphWords * (variant?.planes ?? 1) + (variant?.auxBytes?.(scratchCapacity) ?? 0) / 4}>;` : "@group(0) @binding(4) var<storage, read_write> scratch: array<i32>;"}
@group(0) @binding(5) var<storage, read_write> output: array<i32>;
const GLYPH_WORDS: u32 = ${glyphWords}u;
const RECORD_CLASS_MASK: u32 = ${RECORD_CLASS_MASK}u;
const ATTACH_SHIFT: u32 = ${ATTACH_SHIFT}u;
var<workgroup> length:u32;
var<workgroup> terminal:u32;
var<workgroup> rootCount:u32;
var<workgroup> gsubCount:u32;
var<workgroup> rootPresence:array<u32,8>;
var<workgroup> idsChanged:atomic<u32>;
var<workgroup> selected:array<u32,${scratchCapacity || 256}>;
var<workgroup> seconds:array<u32,${scratchCapacity || 256}>;
var<workgroup> indexBits:array<atomic<u32>,8>;
var<workgroup> refreshFlags:u32;
var<workgroup> refreshStart:u32;
var<workgroup> presenceBits:array<atomic<u32>,8>;
var<workgroup> presenceWitnesses:array<atomic<u32>,8>;
var<workgroup> matchPositions:array<u32,64>;
var<workgroup> nextLigature:u32;
var<private> removed:bool;
var<private> nominalSource:bool;
var<private> selecting:bool;
var<private> trackReads:bool;
var<private> lowRead:u32;
var<private> recordMatches:bool;
var<private> actionDelta:i32;
var<private> mergeLength:u32;
${masked ? "var<private> rootMask:u32;\n" : ""}${localScratch ? `var<workgroup> nominalMasks:array<u32,${scratchCapacity}>;` : ""}
${fragments?.profilePrepass ? `fn profileBarrier() { ${storageFence} workgroupBarrier(); }\n` : ""}${Object.values(fragments ?? {}).join("\n")}${COMPUTE_TEXT_PREDICATES}
${fragments?.glyphAddress ? "" : "fn glyphAt(base:u32, index:u32)->u32 { return base + index * GLYPH_WORDS; }\n"}fn recordProperties(cls:u32, distance:u32)->i32 { return bitcast<i32>((cls & RECORD_CLASS_MASK) | (distance << ATTACH_SHIFT)); }
fn recordClass(at:u32)->u32 { return u32(scratch[at + 7u]) & RECORD_CLASS_MASK; }
fn attachDistance(at:u32)->u32 { return u32(scratch[at + 7u]) >> ATTACH_SHIFT; }
fn readGlyph(base:u32, index:u32)->u32 {
if (nominalSource) { return ${loadGlyph("runs[base + 1u + index]")}; }
observeRead(index);
return u32(scratch[glyphAt(base, index)]);
}
fn observeRead(index:u32) { if (selecting) { lowRead = min(lowRead, index); } }
fn nextSelected(start:u32, count:u32)->u32 {
var word = start / 32u;
var bits = atomicLoad(&indexBits[word]) & (0xffffffffu << (start % 32u));
loop {
if (bits != 0u) { return word * 32u + countTrailingZeros(bits); }
word++;
if (word * 32u >= count) { return count; }
bits = atomicLoad(&indexBits[word]);
}
}
${
    localScratch
        ? ""
        : `fn nominalMask(base:u32, index:u32)->u32 {
return program[params.programWords + params.rootCount * 4u + readGlyph(base, index) * program[13u]];
}`
}
fn live(base:u32, index:u32, count:u32, flag:u32)->bool {
  if (index >= count) { return false; }
  if (nominalSource) { return !skipped(flag, readGlyph(base, index)); }
  observeRead(index);
  let at = glyphAt(base, index);
  return (scratch[at + 6] & 1) == 0 && !skipped(flag, u32(scratch[at]));
}
${presenceSource("activeRoots", "all-words", localScratch)}${splitFastState ? "\n" + presenceSource("activeRootsProbe", "word0-probe", localScratch) : ""}
${
    localScratch
        ? ""
        : `fn activeRootsParallel(base:u32, count:u32, lane:u32) {
let masks = params.programWords + params.rootCount * 4u;
let width = program[13u];
for (var word = 0u; word < width; word++) {
var bits = 0u;
var witnesses = 0u;
for (var index = lane; index < count; index += 32u) {
let glyph = readGlyph(base, index);
let first = program[masks + glyph * width + word];
bits |= first;
witnesses |= program[masks + params.glyphCount * width + glyph * width + word];
}
atomicOr(&presenceBits[word], bits);
atomicOr(&presenceWitnesses[word], witnesses);
}
workgroupBarrier();
if (lane == 0u) { for (var word = 0u; word < width; word++) { rootPresence[word] = atomicLoad(&presenceBits[word]) & atomicLoad(&presenceWitnesses[word]); } }
}`
}
fn replaceGlyph(at:u32, id:u32) {
  scratch[at + 6u] |= 0x400000;
  scratch[at + 7u] = recordProperties(gdefOf(id), 0u);
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
  let base = ${localScratch ? "0u" : variant?.auxBytes ? `local * (capacity * GLYPH_WORDS * ${variant.planes}u + capacity * ${variant.auxBytes(1) / 4}u)` : `local * capacity * GLYPH_WORDS${(variant?.planes ?? 1) > 1 ? ` * ${variant!.planes}u` : ""}`};
${fragments?.recordInit ? "  profileInit(base, out, lane);\n" : ""}  nominalSource = true;
  if (lane == 0u) {
    output[out + 2] = 0;
    output[out + 3] = 0;
    output[out] = 0;
    output[out + 1] = 0;
    length = runs[inputOrigin] & 0x7fffffffu;
    terminal = 0u;
${splitFastState ? "" : "    nextLigature = 0u;\n"}    atomicStore(&idsChanged, 0u);
    rootCount = params.rootCount;
    gsubCount = program[2u];
    if (length > capacity) { output[out] = ${STATUS_OVERFLOW}; terminal = 1u; }
    if (terminal == 0u${localScratch ? "" : " && length <= 256u"}) {
${simpleEnabled ? `let present = ${splitFastState ? "activeRootsProbe" : "activeRoots"}(inputOrigin, length);\nsimpleRoots = proveSimpleRun(inputOrigin, length, present);` : "simpleRoots = vec2u(0u);"}
${localScratch ? "atomicStore(&firstFailure, 0xffffffffu);" : ""}
}
${
    localScratch || !simpleEnabled
        ? ""
        : `for (var word = 0u; word < 8u; word++) {
atomicStore(&presenceBits[word], 0u);
atomicStore(&presenceWitnesses[word], 0u);
}
atomicStore(&firstFailureIndex, 0xffffffffu);`
}
${localScratch ? "" : "atomicStore(&firstFailure, 0xffffffffu);"}
  }
  ${storageFence}
  workgroupBarrier();
  if (workgroupUniformLoad(&terminal) != 0u) { return; }
  let scalarCount = workgroupUniformLoad(&length);
  ${
      localScratch || !simpleEnabled
          ? ""
          : `if (scalarCount > 256u) {
activeRootsParallel(inputOrigin, scalarCount, lane);
let candidates = workgroupUniformLoad(&rootPresence[0u]);
let proved = proveSimpleRunParallel(inputOrigin, scalarCount, candidates, lane);
if (lane == 0u) { simpleRoots = proved; }
}`
  }
  let simple = workgroupUniformLoad(&simpleRoots);
${simpleEnabled ? "if (simple.y != 0u) {\nif (simple.y == 1u) { finishSimpleRun(inputOrigin, scalarCount, simple.x, out, lane); } else { finishSimpleRunLean(inputOrigin, scalarCount, simple.x, out, lane); }\nreturn;\n}\n" : ""}${
        splitFastState
            ? `if (lane == 0u) {
let generalPresence = activeRoots(inputOrigin, scalarCount);
nextLigature = 0u;
}
workgroupBarrier();
`
            : ""
    }nominalSource = false;
  for (var i = lane; i < scalarCount; i += 32u) {
    let at = glyphAt(base, i);
    let id = ${loadGlyph("runs[inputOrigin + 1u + i]")};
    scratch[at] = i32(id);
    scratch[at + 1] = i32(i);
    scratch[at + 2] = advanceOf(id);
    scratch[at + 3] = 0;
    scratch[at + 4] = 0;
    scratch[at + 5] = 0;
    scratch[at + 6] = 0;
    scratch[at + 7] = recordProperties(gdefOf(id), 0u);
${glyphWords > 8 ? `for (var word = 8u; word < GLYPH_WORDS; word++) { scratch[at + word] = 0; }\n` : ""}  }
  ${storageFence}
  workgroupBarrier();
${fragments?.profilePrepass ? `  mergeLength = scalarCount;\n  profilePrepass(inputOrigin, base, scalarCount, lane);\n${fragments.recordInit ? "  if (workgroupUniformLoad(&terminal) != 0u) { return; }\n" : ""}  if (lane == 0u) { let present = activeRoots(base, length); }\n  ${storageFence}\n  workgroupBarrier();\n` : !simpleEnabled ? `  if (lane == 0u) { let present = activeRoots(base, length); }\n  ${storageFence}\n  workgroupBarrier();\n` : ""}  let rootOrigin = 8u;
  let roots = workgroupUniformLoad(&rootCount);
  let substitutions = workgroupUniformLoad(&gsubCount);
  let stageOrigin = program[10u];
  for (var stage = 0u; stage < program[12u]; stage++) {
  let stageAt = stageOrigin + stage * 4u;
  let endRoot = program[stageAt + 1u] + program[stageAt + 2u];
  for (var root = program[stageAt + 1u]; root < endRoot; root++) {
    if (root < 256u && (workgroupUniformLoad(&rootPresence[root / 32u]) & (1u << (root % 32u))) == 0u) { continue; }
    let encoded = program[program[rootOrigin] + root * 4u];
    let tableBit = encoded >> 16u;
    let lookupIndex = encoded & 0xffffu;
${masked ? "    rootMask = program[program[rootOrigin] + root * 4u + 1u];\n" : ""}${profileGates ? "    profileRoot(program[program[rootOrigin] + root * 4u + 2u], tableBit);\n" : ""}    let descBase = select(program[4u], program[5u], tableBit == 1u) + lookupIndex * 8u;
    let flag = program[descBase + 1u];
    let ruleOrigin = program[descBase + 3u];
    let directory = program[descBase + 6u];
    let isGpos = tableBit == 1u;
    let count = workgroupUniformLoad(&length);
    mergeLength = count;
    let capability = program[descBase + 2u];
    let cooperative = capability != 0u${profileGates ? " && capability == 1u" : ""}${localScratch ? " && capacity <= 256u" : ""};
    let orderedEligibility = program[descBase + 7u];
    let orderedEligible = (orderedEligibility & 3u) ${localScratch ? "!= 0u" : "== 1u"};
    let orderedForward = (orderedEligibility & 4u) != 0u;
    let orderedParallel = count > 32u && orderedEligible && !cooperative;
    trackReads = orderedParallel && !isGpos && !orderedForward;
    ${cooperativeSource(localScratch, storageFence, masked)}
    ${storageFence}
    let flags = workgroupUniformLoad(&refreshFlags);
    refreshRun(base, lane, flags);
    if ((flags & 4u) != 0u) { return; }
  }
${fragments?.pauses ? `  mergeLength = workgroupUniformLoad(&length);\n  profilePause(base, program[stageAt + 3u], lane);\n  if (workgroupUniformLoad(&terminal) != 0u) { return; }\n  if (lane == 0u) { let present = activeRoots(base, length); }\n  ${storageFence}\n  workgroupBarrier();\n` : ""}  }
  let positionedCount = workgroupUniformLoad(&length);
  for (var i = lane; i < positionedCount; i += 32u) {
    let at = glyphAt(base, i);
    if (attachDistance(at) != 0u) { scratch[at + 2u] = 0; scratch[at + 3u] = 0; }
  }
  ${storageFence}
  workgroupBarrier();
  if (lane == 0u) {
    propagateAttachments(base, length);
    if (params.direction == 1u) { reverseRun(base, length); }
    output[out + 1] = i32(length);
  }
  ${storageFence}
  workgroupBarrier();
${fragments?.postprocess ? "  profileFinish(base, positionedCount, lane);\n" : ""}  let count = workgroupUniformLoad(&length);
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
${masked ? "  if ((glyphMask(glyphAt(base, second)) & rootMask) == 0u) { return 0u; }\n" : ""}  return selectPairIds(readGlyph(base, index), readGlyph(base, second), ruleOrigin, directory);
}
fn executeRoot(base:u32, inputCount:u32, flag:u32, ruleOrigin:u32, directory:u32, out:u32, lookup:u32)->u32 {
    removed = false;
    var count = inputCount;
    var index = 0u;
    loop {
      if (index >= count) { break; }
      actionDelta = 0;
      let selectedRule = selectRule(base, count, index, flag, ruleOrigin, directory);
      var applied = vec2u(1u, index);
      if (selectedRule != 0u) { applied = applySelected(base, count, index, flag, selectedRule - 1u, out, lookup); }
      if (terminal != 0u) { return count; }
      count = u32(i32(count) + actionDelta);
      index += applied.x;
    }
    return count;
}
fn selectRule(base:u32, count:u32, index:u32, flag:u32, ruleOrigin:u32, directory:u32)->u32 {
  lowRead = index;
  selecting = trackReads;
  let selectedRule = selectCandidate(base, count, index, flag, ruleOrigin, directory);
  selecting = false;
  return selectedRule;
}
fn selectCandidate(base:u32, count:u32, index:u32, flag:u32, ruleOrigin:u32, directory:u32)->u32 {
      let entry = directory + readGlyph(base, index) * 2u;
      let candidates = program[entry];
      let candidateCount = program[entry + 1u];
      if (candidateCount == 0u || !live(base, index, count, flag)) { return 0u; }
      for (var candidate = 0u; candidate < candidateCount; candidate++) {
        var ruleAt = ruleOrigin + program[candidates + candidate] * 16u;
        let row = program[ruleAt + 10u];
        if (${profileGates ? "false && " : ""}row != 0u) {
          candidate += program[ruleAt + 13u] - 1u;
          let second = ${profileGates ? "profilePairSecond(base, count, index, flag, ruleAt)" : "pairSecond(base, count, index, flag)"};
          if (second >= count) { continue; }
          let cls = classOf(program[ruleAt + 11u], readGlyph(base, second));
          if (cls >= program[ruleAt + 12u]) { continue; }
          let selected = program[row + cls];
          if (selected == 0u) { continue; }
          ruleAt = ruleOrigin + (selected - 1u) * 16u;
        }
        let mode = program[ruleAt + 8u];
        if (!matchInput(base, count, index, flag, ruleAt)) { continue; }
        let action = program[ruleAt + 6u];
        if (mode == 12u) {
          if (probeMarkLigature(base, index, action).x != 0u) { return ruleAt + 1u; }
        } else if (mode == 11u) {
          if (program[action] != 0u) { return ruleAt + 1u; }
        } else if (mode == 3u || mode == 4u) {
          let second = ${profileGates ? "profilePairSecond(base, count, index, flag, ruleAt)" : "pairSecond(base, count, index, flag)"};
          if (second < count && (mode == 4u || program[action] == readGlyph(base, second))) { return ruleAt + 1u; }
        } else if (mode == 7u || mode == 8u) {
          if (probeMark(base, index, flag, action, mode).x != 0u) { return ruleAt + 1u; }
        } else if (mode == 1u || mode == 2u || mode == 5u || mode == 9u || mode == 10u) {
          return ruleAt + 1u;
        }
      }
      return 0u;
}
fn compactRun(base:u32, count:u32)->u32 {
  var write = 0u;
  for (var i = 0u; i < count; i++) {
    let at = glyphAt(base, i);
    if ((scratch[at + 6] & 1) == 0) {
      if (write != i) {
        let dst = glyphAt(base, write);
        for (var word = 0u; word < GLYPH_WORDS; word++) { scratch[dst + word] = scratch[at + word]; }
      }
      write++;
    }
  }
  return write;
}
fn refreshRun(base:u32, lane:u32, flags:u32) {
if ((flags & 3u) == 0u) { return; }
let count = workgroupUniformLoad(&length);
let parallelRefresh = count > 32u;
if (!parallelRefresh) {
if (lane == 0u) {
if ((flags & 1u) != 0u) { length = compactRun(base, count); }
if ((flags & 2u) != 0u) { let present = activeRoots(base, length); }
}
${storageFence}
return;
}
if (lane == 0u) {
for (var word = 0u; word < 8u; word++) {
atomicStore(&presenceBits[word], 0u);
atomicStore(&presenceWitnesses[word], 0u);
}
}
if ((flags & 1u) != 0u) {
var writeBase = 0u;
for (var tile = 0u; tile < count; tile += 256u) {
let tileCount = min(256u, count - tile);
let chunks = (tileCount + 31u) / 32u;
if (lane < 8u) { atomicStore(&indexBits[lane], 0u); }
workgroupBarrier();
for (var index = lane; index < tileCount; index += 32u) {
if ((scratch[glyphAt(base, tile + index) + 6u] & 1) == 0) { atomicOr(&indexBits[index / 32u], 1u << (index % 32u)); }
}
workgroupBarrier();
if (lane == 0u) {
refreshStart = 0u;
if (writeBase == tile) {
for (var word = 0u; word < chunks; word++) {
if (atomicLoad(&indexBits[word]) != 0xffffffffu) { break; }
refreshStart++;
}
}
}
let start = workgroupUniformLoad(&refreshStart);
var prefix = writeBase + start * 32u;
var current:array<i32,${glyphWords}>;
if (start < chunks) {
let bitmap = atomicLoad(&indexBits[start]);
let index = tile + start * 32u + lane;
let dest = prefix + countOneBits(bitmap & ((1u << lane) - 1u));
if ((bitmap & (1u << lane)) != 0u && dest != index) {
let at = glyphAt(base, index);
for (var word = 0u; word < GLYPH_WORDS; word++) { current[word] = scratch[at + word]; }
}
}
for (var chunk = start; chunk < chunks; chunk++) {
let bitmap = atomicLoad(&indexBits[chunk]);
let nextPrefix = prefix + countOneBits(bitmap);
var next:array<i32,${glyphWords}>;
if (chunk + 1u < chunks) {
let nextBits = atomicLoad(&indexBits[chunk + 1u]);
let index = tile + (chunk + 1u) * 32u + lane;
let dest = nextPrefix + countOneBits(nextBits & ((1u << lane) - 1u));
if ((nextBits & (1u << lane)) != 0u && dest != index) {
let at = glyphAt(base, index);
for (var word = 0u; word < GLYPH_WORDS; word++) { next[word] = scratch[at + word]; }
}
}
${storageFence}
workgroupBarrier();
if ((bitmap & (1u << lane)) != 0u) {
let index = tile + chunk * 32u + lane;
let dest = prefix + countOneBits(bitmap & ((1u << lane) - 1u));
if (dest != index) {
let at = glyphAt(base, dest);
for (var word = 0u; word < GLYPH_WORDS; word++) { scratch[at + word] = current[word]; }
}
}
current = next;
prefix = nextPrefix;
}
writeBase = prefix;
}
${storageFence}
workgroupBarrier();
if (lane == 0u) { length = writeBase; }
}
if ((flags & 2u) != 0u) {
let liveCount = workgroupUniformLoad(&length);
let masks = params.programWords + params.rootCount * 4u;
let width = program[13u];
for (var word = 0u; word < width; word++) {
var bits = 0u;
var witnesses = 0u;
for (var index = lane; index < liveCount; index += 32u) {
let at = glyphAt(base, index);
if ((scratch[at + 6u] & 1) != 0) { continue; }
let glyph = u32(scratch[at]);
bits |= program[masks + glyph * width + word];
witnesses |= program[masks + params.glyphCount * width + glyph * width + word];
}
atomicOr(&presenceBits[word], bits);
atomicOr(&presenceWitnesses[word], witnesses);
}
workgroupBarrier();
if (lane == 0u) { for (var word = 0u; word < width; word++) { rootPresence[word] = atomicLoad(&presenceBits[word]) & atomicLoad(&presenceWitnesses[word]); } }
}
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
fn applySelected(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32, out:u32, lookup:u32)->vec2u {
  actionDelta = 0;
  recordMatches = true;
  let matched = matchInput(base, count, index, flag, ruleAt);
  recordMatches = false;
  if (!matched) { return vec2u(1u, index); }
  if (program[ruleAt + 8u] == 2u) { return applyContext(base, count, index, flag, ruleAt, out, lookup); }
  return applyNonContext(base, count, index, flag, ruleAt, out, lookup);
}
fn applyNonContext(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32, out:u32, lookup:u32)->vec2u {
  actionDelta = 0;
  let mode = program[ruleAt + 8u];
  let action = program[ruleAt + 6u];
  if (mode == 5u) { rejectRun(out, lookup, index); }
  if (mode == 12u && applyMarkLigature(base, index, action) == 2u) { rejectRun(out, lookup, index); }
  if (mode == 11u) {
    let produced = applyMultiple(base, count, index, action);
    if (produced == 0xffffffffu) { output[out] = ${STATUS_OVERFLOW}; terminal = 1u; return vec2u(1u, 0xffffffffu); }
    actionDelta = i32(produced) - 1;
    return vec2u(produced, 0xffffffffu);
  }
  if (mode == 1u) { replaceGlyph(glyphAt(base, index), program[action + 1u]); }
  if (mode == 9u) { return vec2u(1u, applyLigature(base, count, index, flag, ruleAt, action)); }
  if (mode == 3u || mode == 4u) {
    let second = ${profileGates ? "profilePairSecond(base, count, index, flag, ruleAt)" : "pairSecond(base, count, index, flag)"};
    commitPair(base, index, second, action, mode);
    let values = action + select(1u, 2u, mode == 4u);
    return vec2u(second - index + select(0u, 1u, program[values + 8u] != 0u), second);
  }
  if (mode == 8u || mode == 7u) { applyMark(base, count, index, flag, action, mode); }
  if (mode == 10u) {
    let at = glyphAt(base, index);
    scratch[at + 4] += bitcast<i32>(program[action]);
    scratch[at + 5] += bitcast<i32>(program[action + 1u]);
    scratch[at + 2] += bitcast<i32>(program[action + 2u]);
  }
  return vec2u(1u, index);
}
// OpenType SequenceLookupRecord bookkeeping, independently implemented.
// Behavioural reference: HarfBuzz 11.0.1 (MIT, HarfBuzz authors).
// A single action has no subsequent SequenceLookupRecord that could observe a
// rewritten sequence. Keep its original boundary and apply the same child/delta
// and capacity rules, without constructing or filtering an unused sequence.
fn applyContextSingle(base:u32, count:u32, index:u32, parent:u32, out:u32, parentLookup:u32)->vec2u {
  actionDelta = 0;
  let action = program[parent + 6u];
  let seqCount = program[parent + 1u];
  var boundary = index + 1u;
  if (seqCount != 0u) { boundary = matchPositions[seqCount - 1u] + 1u; }
  let k = program[action + 1u];
  var high = index;
  if (k < seqCount) {
    let lookup = program[action + 2u];
    let desc = select(program[4u], program[5u], (lookup >> 16u) != 0u) + (lookup & 0xffffu) * 8u;
    let childFlag = program[desc + 1u];
    let at = matchPositions[k];
    let chosen = selectRule(base,count,at,childFlag,program[desc + 3u],program[desc + 6u]);
    if (chosen != 0u) {
      let applied = applyNonContext(base,count,at,childFlag,chosen - 1u,out,lookup);
      high = max(high,applied.y);
      if (terminal != 0u) { return vec2u(1u,high); }
      if (actionDelta > 0) {
        let extra = u32(actionDelta);
        if (seqCount + extra > 64u) { rejectRun(out,parentLookup,index); return vec2u(1u,high); }
        boundary += extra;
      }
    }
  }
  var advance = 1u;
  if (boundary > index) { advance = max(1u,boundary - index); }
  return vec2u(advance,high);
}
fn applyContext(base:u32, count:u32, index:u32, flag:u32, parent:u32, out:u32, parentLookup:u32)->vec2u {
  let action = program[parent + 6u];
  let nested = program[action];
  if (nested == 1u) { return applyContextSingle(base,count,index,parent,out,parentLookup); }
  var seqCount = program[parent + 1u];
  var currentCount = count;
  var boundary = index + 1u;
  if (seqCount != 0u) { boundary = matchPositions[seqCount - 1u] + 1u; }
  var high = index;
  for (var i = 0u; i < nested; i++) {
    let k = program[action + 1u + i * 2u];
    if (k >= seqCount) { continue; }
    let at = matchPositions[k];
    let lookup = program[action + 2u + i * 2u];
    let desc = select(program[4u], program[5u], (lookup >> 16u) != 0u) + (lookup & 0xffffu) * 8u;
    let ruleOrigin = program[desc + 3u];
    let childFlag = program[desc + 1u];
    let chosen = selectRule(base, currentCount, at, childFlag, ruleOrigin, program[desc + 6u]);
    if (chosen == 0u) { continue; }
    let ruleAt = chosen - 1u;
    let mode = program[ruleAt + 8u];
    let applied = applyNonContext(base, currentCount, at, childFlag, ruleAt, out, lookup);
    high = max(high, applied.y);
    if (terminal != 0u) { return vec2u(1u, high); }
    if (actionDelta > 0) {
      let extra = u32(actionDelta);
      if (seqCount + extra > 64u) { rejectRun(out, parentLookup, index); return vec2u(1u, high); }
      var p = seqCount;
      while (p > k + 1u) { p--; matchPositions[p + extra] = matchPositions[p] + extra; }
      for (var j = 1u; j <= extra; j++) { matchPositions[k + j] = at + j; }
      seqCount += extra;
      currentCount += extra;
      boundary += extra;
    } else if (mode == 9u) {
      // A root retains physical tombstone coordinates until refresh. Drop
      // consumed sequence elements, but never move records or renumber indexes.
      var kept = 0u;
      for (var p = 0u; p < seqCount; p++) {
        if ((scratch[glyphAt(base, matchPositions[p]) + 6u] & 1) == 0) {
          matchPositions[kept] = matchPositions[p];
          kept++;
        }
      }
      seqCount = kept;
    }
  }
  actionDelta = i32(currentCount) - i32(count);
  var advance = 1u;
  if (boundary > index) { advance = max(1u, boundary - index); }
  return vec2u(advance, high);
}
fn probeMarkLigature(base:u32, index:u32, action:u32)->vec3u {
  var lig = -1;
  for (var i = 0u; i < index; i++) {
    observeRead(i);
    let cls = recordClass(glyphAt(base, i));
    if ((scratch[glyphAt(base, i) + 6u] & 1) != 0) { continue; }
${profileGates ? "    if (profileHidden(glyphAt(base, i))) { continue; }\n" : ""}    if (cls != 3) { lig = i32(i); }
  }
  if (lig < 0) { return vec3u(0u); }
  let ligId = readGlyph(base, u32(lig));
  var cursor = action + 3u;
  let ligatures = program[action + 2u];
  for (var n = 0u; n < ligatures; n++) {
    let glyphId = program[cursor];
    let componentCount = program[cursor + 1u];
    let anchorCount = program[cursor + 2u];
    cursor += 3u;
    if (glyphId == ligId) {
      if (componentCount == 0u) { return vec3u(0u); }
      let props = u32(scratch[glyphAt(base, index) + 6u]);
      let ligProps = u32(scratch[glyphAt(base, u32(lig)) + 6u]);
      let same = ((props >> 8u) & 255u) != 0u && ((props >> 8u) & 255u) == ((ligProps >> 8u) & 255u);
      let component = select(componentCount - 1u, min(((props >> 16u) & 15u) - 1u, componentCount - 1u), same && ((props >> 16u) & 15u) != 0u);
      for (var a = 0u; a < anchorCount; a++) {
        let at = cursor + a * 4u;
        if (program[at] == component) {
          return vec3u(u32(lig) + 1u, at, program[at + 1u]);
        }
      }
      return vec3u(0u);
    }
    cursor += anchorCount * 4u;
  }
  return vec3u(0u);
}
fn applyMarkLigature(base:u32, index:u32, action:u32)->u32 {
  let probe = probeMarkLigature(base, index, action);
  if (probe.x == 0u) { return 0u; }
  if (probe.z != 0u) { return 2u; }
  let mark = glyphAt(base, index);
  let baseAt = glyphAt(base, probe.x - 1u);
  scratch[mark + 4] = bitcast<i32>(program[probe.y + 2u]) - bitcast<i32>(program[action]);
  scratch[mark + 5] = bitcast<i32>(program[probe.y + 3u]) - bitcast<i32>(program[action + 1u]);
  scratch[mark + 7u] = recordProperties(recordClass(mark), index - (probe.x - 1u));
  return 1u;
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
  let profileFlags = scratch[at + 6u] & 0x0f000000;
  let first = program[action + 1u];
  replaceGlyph(at, first);
  if (produced > 1u) { atomicStore(&idsChanged, 1u); }
  scratch[at + 7] = recordProperties(gdefOf(first), 0u);
  scratch[at + 6u] = profileFlags | 0x600000 | (1 << 16);
  for (var step = 1u; step < produced; step++) {
    let dst = glyphAt(base, index + step);
    let id = program[action + 1u + step];
    scratch[dst] = i32(id);
    scratch[dst + 1] = cluster;
    scratch[dst + 2] = advanceOf(id);
    scratch[dst + 3] = 0;
    scratch[dst + 4] = 0;
    scratch[dst + 5] = 0;
    scratch[dst + 6] = profileFlags | 0x600000 | i32(min(step + 1u, 15u) << 16u);
    scratch[dst + 7] = recordProperties(gdefOf(id), 0u);
${glyphWords > 8 ? "for (var word = 8u; word < GLYPH_WORDS; word++) { scratch[dst + word] = scratch[at + word]; }\n" : ""}  }
  mergeLength = count + extra;
  return produced;
}
fn applyLigature(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32, action:u32)->u32 {
  let inputCount = program[ruleAt + 1u];
  nextLigature = nextLigature % 7u + 1u;
  var cursor = index;
${profileGates ? "  var inputWord = program[ruleAt];\n" : ""}  for (var step = 0u; step < inputCount; step++) {
    if (step > 0u) {
${profileGates ? "      inputWord += predicateSize(inputWord);\n" : ""}      let next = ${profileGates ? "profileInputNext(base, count, cursor + 1u, flag, 1, inputWord, false)" : "nextLive(base, count, cursor + 1u, flag, 1)"};
      for (var mark = cursor + 1u; mark < next; mark++) { tagLigatureMark(base, mark, nextLigature, step); }
      cursor = next;
      scratch[glyphAt(base, cursor) + 6] |= 1;
    }
  }
  mergeClusters(base, index, cursor + 1u);
  var high = cursor;
  for (var mark = cursor + 1u; mark < count; mark++) {
    if (recordClass(glyphAt(base, mark)) != 3u) { break; }
    tagLigatureMark(base, mark, nextLigature, inputCount);
    high = mark;
  }
  let at = glyphAt(base, index);
  replaceGlyph(at, program[action + 1u]);
  scratch[at + 6u] = (scratch[at + 6u] & 0x0f000000) | 0x500000 | i32(nextLigature << 8u);
  removed = true;
  atomicStore(&idsChanged, 1u);
  return high;
}
fn tagLigatureMark(base:u32, index:u32, id:u32, component:u32) {
  let at = glyphAt(base, index);
  if (recordClass(at) == 3u) {
    scratch[at + 6u] = (scratch[at + 6u] & i32(0xfff000ffu)) | i32((id << 8u) | (min(component, 15u) << 16u));
  }
}
fn mergeClusters(base:u32, start:u32, end:u32) {
  var a = start;
  var b = end;
  while (a > 0u && scratch[glyphAt(base, a - 1u) + 1u] == scratch[glyphAt(base, a) + 1u]) { a--; }
  while (b < mergeLength && scratch[glyphAt(base, b - 1u) + 1u] == scratch[glyphAt(base, b) + 1u]) { b++; }
  var minimum = scratch[glyphAt(base, a) + 1u];
  for (var i = a + 1u; i < b; i++) { minimum = min(minimum, scratch[glyphAt(base, i) + 1u]); }
  for (var i = a; i < b; i++) { scratch[glyphAt(base, i) + 1u] = minimum; }
}
fn commitPairFirst(base:u32, index:u32, action:u32, mode:u32) {
  scratch[glyphAt(base, index) + 2] += bitcast<i32>(program[action + select(3u, 4u, mode == 4u)]);
}
fn commitPair(base:u32, index:u32, second:u32, action:u32, mode:u32) {
  let values = action + select(1u, 2u, mode == 4u);
  commitValue(glyphAt(base, index), values);
  commitValue(glyphAt(base, second), values + 4u);
}
fn commitValue(at:u32, value:u32) {
  scratch[at + 4u] += bitcast<i32>(program[value]);
  scratch[at + 5u] += bitcast<i32>(program[value + 1u]);
  scratch[at + 2u] += bitcast<i32>(program[value + 2u]);
}
fn probeMark(base:u32, index:u32, flag:u32, action:u32, mode:u32)->vec2u {
  var baseIndex = -1;
  if (mode == 7u) {
    if (index == 0u) { return vec2u(0u); }
    let previous = nextLive(base, index, index - 1u, flag, -1);
    if (previous >= index || recordClass(glyphAt(base, previous)) != 3u) { return vec2u(0u); }
    let props = u32(scratch[glyphAt(base, index) + 6u]);
    let other = u32(scratch[glyphAt(base, previous) + 6u]);
    if (((props >> 8u) & 255u) != ((other >> 8u) & 255u)) { return vec2u(0u); }
    if (((props >> 8u) & 255u) != 0u && ((props >> 16u) & 15u) != ((other >> 16u) & 15u)) { return vec2u(0u); }
    baseIndex = i32(previous);
  } else {
    var i = index;
    while (i > 0u) {
      i--;
      observeRead(i);
      let props = u32(scratch[glyphAt(base, i) + 6u]);
      if ((props & 1u) != 0u) { continue; }
${profileGates ? "      if (profileHidden(glyphAt(base, i))) { continue; }\n" : ""}      if ((props & 0x200000u) != 0u && ((props >> 16u) & 15u) > 1u) { continue; }
      if (recordClass(glyphAt(base, i)) != 3u) { baseIndex = i32(i); break; }
    }
    if (baseIndex < 0) { return vec2u(0u); }
  }
  let baseId = readGlyph(base, u32(baseIndex));
  let baseCount = program[action + 3u];
  var low = 0u;
  var high = baseCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (program[action + 4u + middle * 3u] <= baseId) { low = middle + 1u; }
    else { high = middle; }
  }
  if (low == 0u) { return vec2u(0u); }
  let anchor = action + 4u + (low - 1u) * 3u;
  if (program[anchor] != baseId) { return vec2u(0u); }
  return vec2u(u32(baseIndex) + 1u, anchor);
}
fn applyMark(base:u32, count:u32, index:u32, flag:u32, action:u32, mode:u32)->bool {
  let probe = probeMark(base, index, flag, action, mode);
  if (probe.x == 0u) { return false; }
  let anchor = probe.y;
  let ax = bitcast<i32>(program[anchor + 1u]);
  let ay = bitcast<i32>(program[anchor + 2u]);
  let at = glyphAt(base, index);
  let baseAt = glyphAt(base, probe.x - 1u);
  scratch[at + 4] = ax - bitcast<i32>(program[action + 1u]);
  scratch[at + 5] = ay - bitcast<i32>(program[action + 2u]);
  scratch[at + 7u] = recordProperties(recordClass(at), index - (probe.x - 1u));
  return true;
}
fn propagateAttachments(base:u32, count:u32) {
  for (var i = 0u; i < count; i++) {
    let at = glyphAt(base, i);
    let distance = attachDistance(at);
    if (distance == 0u) { continue; }
    let parent = i - distance;
    let parentAt = glyphAt(base, parent);
    scratch[at + 4u] += scratch[parentAt + 4u];
    scratch[at + 5u] += scratch[parentAt + 5u];
    let begin = select(parent, parent + 1u, params.direction == 1u);
    let end = select(i, i + 1u, params.direction == 1u);
    let sign = select(-1, 1, params.direction == 1u);
    for (var j = begin; j < end; j++) {
      scratch[at + 4u] += sign * scratch[glyphAt(base, j) + 2u];
      scratch[at + 5u] += sign * scratch[glyphAt(base, j) + 3u];
    }
    scratch[at + 7u] = recordProperties(recordClass(at), 0u);
  }
}
${computeTextInputPredicates(masked, profileGates, splitFastState)}
${simpleEnabled ? computeTextSimpleSource(!localScratch, splitFastState ? "matchInputProbe" : "matchInput") : "var<workgroup> simpleRoots:vec2u;\nvar<workgroup> firstFailure:atomic<u32>;\nvar<workgroup> firstFailureIndex:atomic<u32>;"}
`;
}

/** @internal Zero selects global scratch; a sufficient bounded variant survives shrinking. */
export function computeTextScratchCapacity(capacity: number, retained = 0, glyphWords: 8 | 16 = 8): number {
    const maximum = glyphWords === 16 ? 128 : 256;
    if (capacity > maximum) {
        return 0;
    }
    let size = Math.max(1, Math.min(maximum, retained));
    while (size < capacity) {
        size *= 2;
    }
    return size;
}

/** @internal One shared kernel constructor for readback and resident execution. */
export function createComputeTextShapeShader(engine: EngineContext, capacity: number, variant?: ComputeTextKernelVariant): ReturnType<typeof createComputeShader> {
    return createComputeShader(engine, {
        name: "compute-text-shape",
        computeSource: kernelSource(
            variant ? computeTextVariantTier(capacity, variant, engine._device.limits.maxComputeWorkgroupStorageSize) : computeTextScratchCapacity(capacity),
            variant
        ),
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
    variantKey: string;
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
    const presenceWords = program.words[13]!;
    const simpleOrigin = program.words.length + roots.length + plan.font.glyphCount * presenceWords * 2;
    const simpleWords = prepareSimplePairWords(program, simpleOrigin);
    const programWords = new Uint32Array(simpleOrigin + simpleWords.length);
    programWords.set(program.words);
    programWords.set(roots, program.words.length);
    programWords[8] = program.words.length;
    const masks = program.words.length + roots.length;
    programWords.set(program.rootWitnesses, masks + plan.font.glyphCount * presenceWords);
    programWords.set(simpleWords, simpleOrigin);
    program.rootOrder.slice(0, 256).forEach((index, root) => {
        const lookup = program.lookups[index]!;
        const descriptor = program.words[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
        const directory = program.words[descriptor + 6]!;
        if (directory === 0) {
            return;
        }
        for (let glyph = 0; glyph < plan.font.glyphCount; glyph++) {
            if (program.words[directory + glyph * 2 + 1]! > 0) {
                const at = masks + glyph * presenceWords + (root >>> 5);
                programWords[at] = (programWords[at]! | (1 << (root % 32))) >>> 0;
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
    readonly variant: ComputeTextKernelVariant;
}

/** @internal Required stride capacity without allocating mutable dispatch input. */
export function computeTextShapeCapacity(plan: ComputeTextPlan, runs: readonly { readonly scalars: Uint32Array }[]): number {
    const expansion = Math.max(1, prepareImmutable(plan).program.maxExpansion);
    let capacity = 1;
    for (const run of runs) {
        capacity = Math.max(capacity, plan._profileData?.capacity?.(run.scalars) ?? run.scalars.length * expansion);
    }
    return capacity;
}

/** @internal Pack one plan's runs. Pass `capacityOverride` when several dispatches share one record stride.
 *  `nominalGlyphs`, when given, must contain every glyph in `runs`; it lets packing prove runs mark-free without scanning them. */
export function prepareComputeTextShapeWords(
    plan: ComputeTextPlan,
    runs: readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array }[],
    outputRun = 0,
    capacityOverride?: number,
    limits?: Pick<GPUSupportedLimits, "maxBufferSize" | "maxStorageBufferBindingSize">,
    nominalGlyphs?: ReadonlySet<number>
): ComputeTextShapeWords {
    const immutable = prepareImmutable(plan);
    const program = immutable.program;
    const needed = computeTextShapeCapacity(plan, runs);
    const capacity = capacityOverride ?? needed;
    const variant = computeTextPlanVariant(plan);
    if (!Number.isSafeInteger(capacity) || capacity < needed) {
        throw new Error("shapeComputeText: shared capacity is smaller than this plan's expansion");
    }
    const limit = limits ? Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize) : Infinity;
    const programWords = immutable.programWords.byteLength <= limit ? immutable.programWords : immutable.baseProgramWords;
    return {
        capacity,
        params: new Uint32Array([
            runs.length,
            capacity,
            plan.direction === "rtl" ? 1 : 0,
            plan.font.glyphCount,
            program.rootOrder.length,
            program.words.length,
            outputRun,
            ["default", "arabic", "hebrew", "indic"].indexOf(variant.id),
        ]),
        programWords,
        tables: immutable.tables,
        runs: packRuns(runs, plan, immutable.tables, nominalGlyphs),
        scratchBytes: Math.max(runs.length, 1) * (capacity * variant.glyphWords * 4 * variant.planes + (variant.auxBytes?.(capacity) ?? 0)),
        program,
        simpleWordCount: programWords === immutable.programWords ? immutable.simpleWordCount : 0,
        variant,
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
    const capacity = prepared.capacity,
        runCount = prepared.params[0]!;
    const sizes = [
        (prepared.params[6]! + prepared.params[0]!) * (HEADER_WORDS + prepared.capacity * RECORD_WORDS) * 4,
        prepared.scratchBytes,
        prepared.params.byteLength,
        prepared.programWords.byteLength,
        prepared.tables.byteLength,
        prepared.runs.byteLength,
    ];
    if (capacity > 0x1fffffff) {
        throw new Error("shapeComputeText: capacity exceeds u32 record addressing");
    }
    if (runCount > limits.maxComputeWorkgroupsPerDimension) {
        throw new Error("shapeComputeText: run count exceeds device maxComputeWorkgroupsPerDimension");
    }
    // Keep platform reads at their original first-use/short-circuit positions.
    let maxBufferSize: number | undefined, maxStorageBufferBindingSize: number | undefined;
    for (const bytes of sizes) {
        if (!Number.isSafeInteger(bytes) || bytes > (maxBufferSize ??= limits.maxBufferSize)) {
            throw new Error(`shapeComputeText: buffer of ${bytes} bytes exceeds device maxBufferSize`);
        }
        if (bytes > (maxStorageBufferBindingSize ??= limits.maxStorageBufferBindingSize)) {
            throw new Error(`shapeComputeText: binding of ${bytes} bytes exceeds device maxStorageBufferBindingSize`);
        }
    }
}

/** @internal Resident preflight before atlas mutation. Reuses immutable font/program
 * preparation; mutable run/scalar vectors are packed only by the actual executor.
 * @param scalarCount - Total scalar count of the logical (unsegmented) group; invariant under segmentation.
 */
export function checkComputeTextPlanBounds(
    limits: GPUSupportedLimits,
    plan: ComputeTextPlan,
    runs: readonly { readonly scalars: Uint32Array }[],
    capacity: number,
    outputRun: number,
    scalarCount: number
): void {
    const immutable = prepareImmutable(plan),
        variant = computeTextPlanVariant(plan);
    const maxBufferSize = limits.maxBufferSize,
        maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize;
    const limit = Math.min(maxBufferSize, maxStorageBufferBindingSize);
    const programBytes = (immutable.programWords.byteLength <= limit ? immutable.programWords : immutable.baseProgramWords).byteLength;
    const runWords = 2 * runs.length + (variant.id === "indic" ? 2 : 1) * scalarCount;
    const runCount = runs.length;
    const sizes = [
        (outputRun + runs.length) * (HEADER_WORDS + capacity * RECORD_WORDS) * 4,
        Math.max(runs.length, 1) * (capacity * variant.glyphWords * 4 * variant.planes + (variant.auxBytes?.(capacity) ?? 0)),
        32,
        programBytes,
        immutable.tables.byteLength,
        runWords * 4,
    ];
    if (capacity > 0x1fffffff) {
        throw new Error("shapeComputeText: capacity exceeds u32 record addressing");
    }
    if (runCount > limits.maxComputeWorkgroupsPerDimension) {
        throw new Error("shapeComputeText: run count exceeds device maxComputeWorkgroupsPerDimension");
    }
    checkShapeSizes(maxBufferSize, maxStorageBufferBindingSize, sizes);
}

function checkShapeSizes(maxBufferSize: number, maxStorageBufferBindingSize: number, sizes: readonly number[]): void {
    for (const bytes of sizes) {
        if (!Number.isSafeInteger(bytes) || bytes > maxBufferSize) {
            throw new Error(`shapeComputeText: buffer of ${bytes} bytes exceeds device maxBufferSize`);
        }
        if (bytes > maxStorageBufferBindingSize) {
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
    prepared: Pick<ComputeTextShapeWords, "programWords" | "tables" | "variant" | "scratchBytes">
): GpuCache {
    const existing = caches?.get(shaper);
    const previous = existing && existing.device === engine._device && existing.plan === plan ? existing : undefined;
    const scratchCapacity = computeTextVariantTier(glyphCapacity, prepared.variant, engine._device.limits.maxComputeWorkgroupStorageSize, previous?.scratchCapacity);
    const variantKey = computeTextVariantKey(prepared.variant, scratchCapacity);
    if (previous && previous.runCapacity >= runCount && previous.glyphCapacity >= glyphCapacity) {
        if (previous.variantKey !== variantKey) {
            disposeComputeShader(previous.shader);
            previous.task.dispose();
            previous.shader = createComputeTextShapeShader(engine, glyphCapacity, prepared.variant);
            previous.scratchCapacity = scratchCapacity;
            previous.variantKey = variantKey;
            Object.assign(previous, createShapeSchedule(engine, previous.shader, previous, runCount));
        }
        return previous;
    }
    const limits = engine._device.limits;
    const outputBytes = runCount * (HEADER_WORDS + glyphCapacity * RECORD_WORDS) * 4;
    const scratchBytes = prepared.scratchBytes;
    for (const bytes of [outputBytes, scratchBytes, prepared.programWords.byteLength]) {
        if (bytes > limits.maxBufferSize) {
            throw new Error(`shapeComputeText: buffer of ${bytes} bytes exceeds device maxBufferSize`);
        }
    }
    const shader = previous?.variantKey === variantKey ? previous.shader : createComputeTextShapeShader(engine, glyphCapacity, prepared.variant);
    if (existing && existing.shader !== shader) {
        disposeComputeShader(existing.shader);
    }
    existing?.task.dispose();
    disposeBuffers(existing, !!previous);
    const buffers = {
        params: previous?.params ?? createStorageBuffer(engine, 32, { label: "compute-text-params" }),
        program: previous?.program ?? createStorageBuffer(engine, prepared.programWords, { label: "compute-text-program" }),
        tables: previous?.tables ?? createStorageBuffer(engine, prepared.tables, { label: "compute-text-tables" }),
        runs: createStorageBuffer(engine, Math.max(4, runCount * (glyphCapacity * (prepared.variant.id === "indic" ? 2 : 1) + 2) * 4), {
            label: "compute-text-runs",
            writable: false,
        }),
        scratch: createStorageBuffer(engine, Math.max(4, scratchBytes), { label: "compute-text-scratch", writable: true }),
        output: createStorageBuffer(engine, Math.max(4, outputBytes), { label: "compute-text-output", writable: true }),
    };
    const cache: GpuCache = {
        device: engine._device,
        plan,
        variantKey,
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

/** @internal Shared interpreter schedule for readback and bounded resident shape batches. */
export function createShapeSchedule(
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
    return program.words.slice(program.words[8], program.words[8]! + program.rootOrder.length * 4);
}

function packTables(plan: ComputeTextPlan, classTables: readonly GlyphClassDef[]): Uint32Array {
    const glyphCount = plan.font.glyphCount;
    const advances = new Uint32Array(glyphCount);
    const gdefWords = new Uint32Array(glyphCount);
    const gdef = ensureComputeFontLayout(plan.font).gdef;
    for (let id = 0; id < glyphCount; id++) {
        advances[id] = readComputeFontAdvance(plan.font, id);
        const rawClass = glyphClass(gdef?.glyphClassDef ?? null, id);
        const cls = rawClass > 4 ? 0 : rawClass;
        const attach = glyphClass(gdef?.markAttachClassDef ?? null, id);
        gdefWords[id] = cls | (attach << 16);
    }
    const markSetWords = Math.ceil((gdef?.markGlyphSetCount ?? 0) / 32);
    const header = 8 + classTables.length;
    const markSetOrigin = header + advances.length + gdefWords.length;
    const classOrigin = markSetOrigin + glyphCount * markSetWords;
    const words = new Uint32Array(classOrigin + classTables.length * glyphCount);
    words[0] = glyphCount;
    words[1] = header;
    words[2] = header + advances.length;
    words[3] = header + advances.length + gdefWords.length;
    words[4] = gdef ? 1 : 0;
    words[5] = classTables.length;
    words[6] = markSetOrigin;
    words[7] = markSetWords;
    words.set(advances, header);
    words.set(gdefWords, header + advances.length);
    gdef?.markGlyphSets.forEach((coverage, set) => {
        if (!coverage) {
            return;
        }
        for (const glyph of coverageGlyphs(coverage, glyphCount)) {
            words[markSetOrigin + glyph * markSetWords + (set >>> 5)]! |= 1 << (set % 32);
        }
    });
    classTables.forEach((table, index) => {
        const origin = classOrigin + index * glyphCount;
        words[8 + index] = origin;
        for (let glyph = 0; glyph < glyphCount; glyph++) {
            words[origin + glyph] = glyphClass(table, glyph);
        }
    });
    return words;
}

function isGdefMark(tables: Uint32Array, gdefOrigin: number, glyph: number): boolean {
    return ((tables[gdefOrigin + glyph] ?? 0) & 0xffff) === GDEF_MARK_CLASS;
}

/** Run header: scalar count, plus RUN_MARK_FREE (selects `finishSimpleRunLean`).
 *  `nominalGlyphs`, when given, must contain every glyph in `runs`; if none is a GDEF mark, it proves every run mark-free without per-run scans. */
function packRuns(
    runs: readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array }[],
    plan: ComputeTextPlan,
    tables: Uint32Array,
    nominalGlyphs?: ReadonlySet<number>
): Uint32Array {
    const gdefOrigin = tables[4] ? tables[2]! : 0;
    let scan = gdefOrigin !== 0;
    if (scan && nominalGlyphs) {
        scan = false;
        for (const glyph of nominalGlyphs) {
            if (isGdefMark(tables, gdefOrigin, glyph)) {
                scan = true;
                break;
            }
        }
    }
    let length = runs.length;
    for (const run of runs) {
        length += 1 + run.glyphIds.length + (plan._profile === "indic" ? run.scalars.length : 0);
    }
    const words = new Uint32Array(length);
    let cursor = runs.length;
    runs.forEach((run, index) => {
        words[index] = cursor;
        let markFree = RUN_MARK_FREE;
        for (let i = 0; scan && i < run.glyphIds.length; i++) {
            if (isGdefMark(tables, gdefOrigin, run.glyphIds[i]!)) {
                markFree = 0;
                break;
            }
        }
        words[cursor] = (run.scalars.length | markFree) >>> 0;
        cursor++;
        if (plan._profile === "arabic") {
            const support = profileSupport("arabic");
            if (!support?.joiningType || !support.markClass || !support.isMcm) {
                throw new Error("prepareComputeTextShapeWords: Arabic profile is not loaded");
            }
            for (let i = 0; i < run.glyphIds.length; i++) {
                const glyph = run.glyphIds[i]!;
                if (glyph >= 65536) {
                    throw new Error("prepareComputeTextShapeWords: Arabic glyph exceeds the OpenType glyph-id field");
                }
                const scalar = run.scalars[i]!;
                words[cursor + i] =
                    glyph |
                    (support.inputProperties
                        ? support.inputProperties(scalar)
                        : (support.joiningType(scalar) << 16) | (Number(support.isMcm(scalar)) << 19) | (support.markClass(scalar) << 24));
            }
        } else {
            words.set(run.glyphIds, cursor);
        }
        cursor += run.glyphIds.length;
        if (plan._profile === "indic") {
            words.set(run.scalars, cursor);
            cursor += run.scalars.length;
        }
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
