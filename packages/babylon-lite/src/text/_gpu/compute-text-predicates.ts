/** Shared packed-program predicates and pair selection; no execution state. */
export const COMPUTE_TEXT_PREDICATES = `
fn advanceOf(id:u32)->i32 { return bitcast<i32>(tables[tables[1u] + id]); }
fn gdefOf(id:u32)->u32 { return tables[tables[2u] + id]; }
fn classOf(tableIndex:u32, glyph:u32)->u32 {
return tables[tables[8u + tableIndex] + glyph];
}
fn skipped(flag:u32, id:u32)->bool {
if ((flag & 0xff1eu) == 0u || tables[4u] == 0u) { return false; }
let cls = gdefOf(id) & 0xffffu;
if ((flag & 2u) != 0u && cls == 1u) { return true; }
if ((flag & 4u) != 0u && cls == 2u) { return true; }
if ((flag & 8u) != 0u && cls == 3u) { return true; }
if (cls == 3u && (flag & 0x10u) != 0u) {
let setIndex = flag >> 16u;
return (tables[tables[6u] + id * tables[7u] + setIndex / 32u] & (1u << (setIndex % 32u))) == 0u;
}
let attach = (flag >> 8u) & 0xffu;
return attach != 0u && cls == 3u && (gdefOf(id) >> 16u) != attach;
}
fn has(origin:u32, glyph:u32)->bool {
let count = program[origin];
var low = 0u;
var high = count;
while (low < high) {
let mid = (low + high) / 2u;
if (program[origin + 1u + mid] < glyph) { low = mid + 1u; } else { high = mid; }
}
return low < count && program[origin + 1u + low] == glyph;
}
fn matches(wordAt:u32, glyph:u32)->bool {
let head = program[wordAt];
if ((head & 0x80000000u) != 0u) {
let classId = head & 0xffffu;
let tableIndex = program[wordAt + 1u];
if (classOf(tableIndex, glyph) != classId) { return false; }
let coverage = program[wordAt + 2u];
if (coverage == 0u) { return true; }
return has(coverage, glyph);
}
return has(wordAt, glyph);
}
fn predicateSize(wordAt:u32)->u32 {
let head = program[wordAt];
if ((head & 0x80000000u) != 0u) { return 3u; }
return 1u + head;
}
fn selectPairIds(firstId:u32, secondId:u32, ruleOrigin:u32, directory:u32)->u32 {
let entry = directory + firstId * 2u;
let candidates = program[entry];
for (var candidate = 0u; candidate < program[entry + 1u]; candidate++) {
let ruleAt = ruleOrigin + program[candidates + candidate] * 16u;
let row = program[ruleAt + 10u];
if (row != 0u) {
candidate += program[ruleAt + 13u] - 1u;
let cls = classOf(program[ruleAt + 11u], secondId);
if (cls < program[ruleAt + 12u]) {
let cell = program[row + cls];
if (cell != 0u) { return ruleOrigin + (cell - 1u) * 16u + 1u; }
}
} else {
let input = program[ruleAt];
if (matches(input + predicateSize(input), secondId)) { return ruleAt + 1u; }
}
}
return 0u;
}
`;

/** Both kernels provide readGlyph/live for their own immutable or mutable input. */
export function computeTextInputPredicates(masked = false, profileGates = false, simpleProbe = false): string {
    return `
fn pairSecond(base:u32, count:u32, index:u32, flag:u32)->u32 {
${profileGates ? "" : "if (flag == 0u || tables[4u] == 0u) { return index + 1u; }\n"}return nextLive(base, count, index + 1u, flag, 1);
}
fn nextLive(base:u32, count:u32, start:u32, flag:u32, direction:i32)->u32 {
var index = i32(start);
loop {
if (index < 0 || u32(index) >= count) { return count; }
if (live(base, u32(index), count, flag)${profileGates ? " && !profileSkip(glyphAt(base, u32(index)), false)" : ""}) { return u32(index); }
index += direction;
}
}
${matchInputSource("matchInput", true, masked, profileGates)}${simpleProbe ? "\n" + matchInputSource("matchInputProbe", false, masked, profileGates) : ""}
`;
}

/** A proof uses the identical matcher without either context-recording store. */
function matchInputSource(name: "matchInput" | "matchInputProbe", record: boolean, masked: boolean, profileGates: boolean): string {
    return `fn ${name}(base:u32, count:u32, index:u32, flag:u32, ruleAt:u32)->bool {
let inputCount = program[ruleAt + 1u];
var cursor = index;
var word = program[ruleAt];
${masked ? "if ((glyphMask(glyphAt(base, index)) & rootMask) == 0u) { return false; }\n" : ""}${record ? "if (recordMatches && inputCount > 0u) { matchPositions[0u] = index; }\n" : ""}// The exhaustive candidate directory proves input predicate zero.
if (inputCount > 0u) { word += predicateSize(word); }
for (var step = 1u; step < inputCount; step++) {
cursor = ${profileGates ? "profileInputNext(base, count, cursor + 1u, flag, 1, word, false)" : "nextLive(base, count, cursor + 1u, flag, 1)"};
if (cursor >= count || !matches(word, readGlyph(base, cursor))) { return false; }
${profileGates ? "if (!profileMayMatch(base, index, cursor, false)) { return false; }\n" : ""}${masked ? "if ((glyphMask(glyphAt(base, cursor)) & rootMask) == 0u) { return false; }\n" : ""}${record ? "if (recordMatches) { matchPositions[step] = cursor; }\n" : ""}word += predicateSize(word);
}
let backCount = program[ruleAt + 3u];
var back = i32(index) - 1;
word = program[ruleAt + 2u];
for (var step = 0u; step < backCount; step++) {
if (back < 0) { return false; }
back = i32(${profileGates ? "profileInputNext(base, count, u32(back), flag, -1, word, true)" : "nextLive(base, count, u32(back), flag, -1)"});
if (back < 0 || u32(back) >= count || !matches(word, readGlyph(base, u32(back)))) { return false; }
${profileGates ? "if (!profileMayMatch(base, index, u32(back), false)) { return false; }\n" : ""}word += predicateSize(word);
back -= 1;
}
let aheadCount = program[ruleAt + 5u];
${profileGates ? "word = program[ruleAt + 4u];\nvar ahead = i32(count);\nif (aheadCount > 0u) { ahead = i32(profileInputNext(base, count, cursor + 1u, flag, 1, word, true)); }\n" : "var ahead = i32(nextLive(base, count, cursor + 1u, flag, 1));\nword = program[ruleAt + 4u];\n"}for (var step = 0u; step < aheadCount; step++) {
if (ahead < 0 || u32(ahead) >= count || !matches(word, readGlyph(base, u32(ahead)))) { return false; }
word += predicateSize(word);
${profileGates ? "if (step + 1u < aheadCount) { ahead = i32(profileInputNext(base, count, u32(ahead) + 1u, flag, 1, word, true)); }" : "ahead = i32(nextLive(base, count, u32(ahead) + 1u, flag, 1));"}
}
return true;
}`;
}
