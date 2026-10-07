/** Exact nominal-run proof and fused first-only pair positioning. */
export const COMPUTE_TEXT_SIMPLE_SOURCE = `
var<workgroup> simpleRoots:vec2u;
var<workgroup> firstFailure:atomic<u32>;
fn simplePair(first:u32, second:u32, root:u32)->vec2u {
let footer = params.programWords + params.rootCount * 2u + params.glyphCount * 2u;
let directory = program[footer + root];
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
fn proveSimpleRun(base:u32, count:u32, candidates:u32)->vec2u {
if (params.rootCount > 32u) { return vec2u(candidates, 0u); }
var admittedRoots = candidates;
let roots = program[8u];
var remaining = candidates;
while (remaining != 0u) {
let root = countTrailingZeros(remaining);
let bit = 1u << root;
remaining &= remaining - 1u;
let table = program[roots + root * 2u];
let lookup = program[roots + root * 2u + 1u];
let descriptor = select(program[4u], program[5u], table == 1u) + lookup * 8u;
if (table == 1u) {
let footer = params.programWords + params.rootCount * 2u + params.glyphCount * 2u;
if (program[descriptor + 2u] != 3u || footer + root >= arrayLength(&program) || program[footer + root] == 0u) { return vec2u(candidates, 0u); }
continue;
}
let flag = program[descriptor + 1u];
let rules = program[descriptor + 3u];
let directory = program[descriptor + 6u];
for (var index = 0u; index < count; index++) {
if ((nominalMasks[index] & bit) == 0u) { continue; }
if (!live(base, index, count, flag)) { continue; }
let entry = directory + readGlyph(base, index) * 2u;
let entries = program[entry];
for (var candidate = 0u; candidate < program[entry + 1u]; candidate++) {
let ruleAt = rules + program[entries + candidate] * 16u;
if (matchInput(base, count, index, flag, ruleAt)) { return vec2u(candidates, 0u); }
}
}
admittedRoots &= ~bit;
}
return vec2u(admittedRoots, 1u);
}
fn finishSimpleRun(base:u32, count:u32, candidates:u32, out:u32, lane:u32) {
let roots = program[8u];
for (var index = lane; index < count; index += 32u) {
let id = readGlyph(base, index);
var advance = advanceOf(id);
var remaining = candidates;
while (remaining != 0u) {
let root = countTrailingZeros(remaining);
remaining &= remaining - 1u;
let lookup = program[roots + root * 2u + 1u];
let descriptor = program[5u] + lookup * 8u;
let flag = program[descriptor + 1u] & 0xff0eu;
if (skipped(flag, id)) { continue; }
let second = pairSecond(base, count, index, flag);
if (second >= count) { continue; }
let selected = simplePair(id, readGlyph(base, second), root);
if (selected.y == 2u) { atomicMin(&firstFailure, (root << 8u) | index); }
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
if (lane != 0u) { return; }
let failure = atomicLoad(&firstFailure);
if (failure != 0xffffffffu) {
output[out] = 1;
output[out + 2u] = i32(program[roots + (failure >> 8u) * 2u + 1u] | 0x10000u);
output[out + 3u] = i32(failure & 255u);
} else {
output[out] = 0;
output[out + 1u] = i32(count);
}
}
`;
