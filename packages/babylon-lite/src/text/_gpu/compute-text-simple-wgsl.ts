/** Exact nominal-run proof and fused first-only pair positioning. */
export function computeTextSimpleSource(globalScratch: boolean, proofMatch: "matchInput" | "matchInputProbe" = "matchInput"): string {
    return `
var<workgroup> simpleRoots:vec2u;
var<workgroup> firstFailure:atomic<u32>;
${globalScratch ? "var<workgroup> firstFailureIndex:atomic<u32>;\nvar<workgroup> proofResult:u32;\n" : ""}var<workgroup> proofFailure:atomic<u32>;
var<workgroup> simpleRootInfo:array<vec4u,32>;
// Lane r decodes root r once, overlapping the presence scan: x flag, y GSUB or pair directory,
// z GSUB rules, w bit 0 GPOS and bit 1 GPOS fused-pair eligible.
fn loadSimpleRootInfo(lane:u32) {
if (lane >= params.rootCount) { return; }
let encoded = program[program[8u] + lane * 4u];
let table = encoded >> 16u;
let descriptor = select(program[4u], program[5u], table == 1u) + (encoded & 0xffffu) * 8u;
if (table == 1u) {
let footer = params.programWords + params.rootCount * 4u + params.glyphCount * program[13u] * 2u + lane;
var pairs = 0u;
if (footer < arrayLength(&program)) { pairs = program[footer]; }
simpleRootInfo[lane] = vec4u(program[descriptor + 1u], pairs, 0u, select(1u, 3u, program[descriptor + 2u] == 3u && pairs != 0u));
} else {
simpleRootInfo[lane] = vec4u(program[descriptor + 1u], program[descriptor + 6u], program[descriptor + 3u], 0u);
}
}
fn proveSimpleRunLanes(base:u32, count:u32, candidates:u32, lane:u32)->vec2u {
if (params.rootCount > 32u) { return vec2u(candidates, 0u); }
var admittedRoots = candidates;
var failed = false;
var remaining = candidates;
while (remaining != 0u) {
let root = countTrailingZeros(remaining);
remaining &= remaining - 1u;
let info = simpleRootInfo[root];
if ((info.w & 1u) != 0u) {
failed = failed || info.w != 3u;
} else {
admittedRoots &= ~(1u << root);
for (var index = lane; index < count && !failed; index += 32u) {
if ((${globalScratch ? "nominalMask(base, index)" : "nominalMasks[index]"} & (1u << root)) == 0u || !live(base, index, count, info.x)) { continue; }
let entry = info.y + readGlyph(base, index) * 2u;
let entries = program[entry];
for (var candidate = 0u; candidate < program[entry + 1u]; candidate++) {
if (${proofMatch}(base, count, index, info.x, info.z + program[entries + candidate] * 16u)) { failed = true; break; }
}
}
}
}
if (failed) { atomicStore(&proofFailure, 1u); }
workgroupBarrier();
if (atomicLoad(&proofFailure) != 0u) { return vec2u(candidates, 0u); }
return vec2u(admittedRoots, 1u | ((runs[base] >> 31u) << 1u));
}
fn simplePair(first:u32, second:u32, directory:u32)->vec2u {
let entry = directory + first * 2u;
let ops = program[entry];
for (var candidate = 0u; candidate < program[entry + 1u]; candidate++) {
let at = ops + candidate * 5u;
if (program[at] == 1u) {
let cls = classOf(program[at + 1u], second);
if (cls < program[at + 2u]) {
let cell = program[at + 3u] + cls * 2u;
if (program[cell + 1u] != 0u) { return vec2u(program[cell], program[cell + 1u]); }
}
} else if (matches(program[at + 1u], second)) {
return vec2u(program[at + 3u], program[at + 4u]);
}
}
return vec2u(0u);
}
${finishSimpleRun(globalScratch, "", "")}
${finishSimpleRun(globalScratch, "Lean", " & 6u")}
`;
}

/**
 * Fused first-only pair positioning. `Lean` serves runs the CPU proved mark-free
 * (run header bit 31): IgnoreMarks, UseMarkFilteringSet and MarkAttachmentType only
 * skip class-3 glyphs and RightToLeft never skips, so keeping only IgnoreBaseGlyphs and
 * IgnoreLigatures is exact there; typical kerning flags then skip no GDEF lookups.
 */
function finishSimpleRun(globalScratch: boolean, name: string, flagMask: string): string {
    return `fn finishSimpleRun${name}(base:u32, count:u32, candidates:u32, out:u32, lane:u32) {
let roots = program[8u];
for (var index = lane; index < count; index += 32u) {
let id = readGlyph(base, index);
var advance = advanceOf(id);
var remaining = candidates;
while (remaining != 0u) {
let root = countTrailingZeros(remaining);
remaining &= remaining - 1u;
let info = simpleRootInfo[root];
let flag = info.x${flagMask};
let directory = info.y;
if (skipped(flag, id)) { continue; }
let second = pairSecond(base, count, index, flag);
if (second >= count) { continue; }
let selected = simplePair(id, readGlyph(base, second), directory);
if (selected.y == 2u) { atomicMin(&firstFailure, ${globalScratch ? "root" : "(root << 8u) | index"}); }
else if (selected.y == 1u) { advance += bitcast<i32>(selected.x); }
}
let position = select(index, count - 1u - index, params.direction == 1u);
let dst = out + 4u + position * 6u;
output[dst] = i32(id);
output[dst + 1u] = i32(index);
output[dst + 2u] = advance;
output[dst + 3u] = 0;
output[dst + 4u] = 0;
output[dst + 5u] = 0;
}
storageBarrier();
workgroupBarrier();
${
    globalScratch
        ? `if (lane == 0u) { proofResult = atomicLoad(&firstFailure); }
let failure = workgroupUniformLoad(&proofResult);
if (failure != 0xffffffffu) {
let lookup = program[roots + failure * 4u] & 0xffffu;
let info = simpleRootInfo[failure];
let flag = info.x${flagMask};
for (var index = lane; index < count; index += 32u) {
let id = readGlyph(base, index);
if (skipped(flag, id)) { continue; }
let second = pairSecond(base, count, index, flag);
if (second < count && simplePair(id, readGlyph(base, second), info.y).y == 2u) { atomicMin(&firstFailureIndex, index); }
}
workgroupBarrier();
if (lane != 0u) { return; }
output[out] = 1;
output[out + 2u] = i32(lookup | 0x10000u);
output[out + 3u] = i32(atomicLoad(&firstFailureIndex));`
        : `if (lane != 0u) { return; }
let failure = atomicLoad(&firstFailure);
if (failure != 0xffffffffu) {
output[out] = 1;
output[out + 2u] = i32((program[roots + (failure >> 8u) * 4u] & 0xffffu) | 0x10000u);
output[out + 3u] = i32(failure & 255u);`
}
} else {
${globalScratch ? "if (lane != 0u) { return; }" : ""}
output[out] = 0;
output[out + 1u] = i32(count);
}
}
`;
}
