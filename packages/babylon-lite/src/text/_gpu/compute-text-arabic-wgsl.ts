/** Arabic-only hooks for the shared lookup kernel, not a separate engine.
 * Unicode cursive neighbours; Arabic modified classes and UTR53 MCM ordering.
 * Behavioural reference: HarfBuzz 11.0.1 (MIT, HarfBuzz authors); independently implemented.
 * Capability boundaries: docs/lite/architecture/compute-text-follow-ups.md.
 */
import type { ComputeTextKernelFragments } from "../compute-text-plan.js";

export const ARABIC_KERNEL_FRAGMENTS: ComputeTextKernelFragments = {
    loadGlyph: "fn loadGlyph(word:u32)->u32 { return word & 0xffffu; }\n",
    glyphMask: "fn glyphMask(at:u32)->u32 { return 0x80000000u | ((u32(scratch[at + 6u]) >> 24u) & 15u); }\n",
    profilePrepass: `
var<workgroup> firstNonT:array<u32,32>;
var<workgroup> lastNonT:array<u32,32>;
fn joiningType(origin:u32, index:u32)->u32 { return (runs[origin + 1u + index] >> 16u) & 7u; }
fn arabicMarkWord(origin:u32, index:u32)->u32 { return runs[origin + 1u + index]; }
fn profilePrepass(origin:u32, base:u32, n:u32, lane:u32) {
  let chunk = (n + 31u) / 32u;
  let start = min(n, lane * chunk);
  let end = min(n, start + chunk);
  var first = n;
  var last = n;
  for (var i = start; i < end; i++) {
    if (joiningType(origin, i) != 4u) {
      if (first == n) { first = i; }
      last = i;
    }
  }
  firstNonT[lane] = first;
  lastNonT[lane] = last;
  profileBarrier();
  var previous = n;
  var next = n;
  for (var other = 0u; other < lane; other++) {
    if (lastNonT[other] != n) { previous = lastNonT[other]; }
  }
  for (var other = lane + 1u; other < 32u; other++) {
    if (firstNonT[other] != n) { next = firstNonT[other]; break; }
  }
  var cursor = end;
  while (cursor > start) {
    cursor--;
    scratch[glyphAt(base, cursor) + 5u] = i32(next);
    if (joiningType(origin, cursor) != 4u) { next = cursor; }
  }
  for (var i = start; i < end; i++) {
    let at = glyphAt(base, i);
    let following = u32(scratch[at + 5u]);
    scratch[at + 5u] = 0;
    let currentType = joiningType(origin, i);
    var form = 0u;
    if (currentType >= 1u && currentType <= 3u) {
      var backward = false;
      var forward = false;
      if (previous < n) {
        let p = joiningType(origin, previous);
        backward = (p == 2u || p == 3u) && (currentType == 1u || currentType == 2u);
      }
      if (following < n) {
        let x = joiningType(origin, following);
        forward = (currentType == 2u || currentType == 3u) && (x == 1u || x == 2u);
      }
      form = select(select(1u, 8u, forward), select(2u, 4u, forward), backward);
    }
    scratch[at + 6u] = (scratch[at + 6u] & i32(0xf0ffffffu)) | i32(form << 24u);
    if (currentType != 4u) { previous = i; }
  }
  profileBarrier();
  for (var i = start; i < end; i++) {
    if ((arabicMarkWord(origin, i) >> 24u) == 0u) { continue; }
    if (i > 0u && (arabicMarkWord(origin, i - 1u) >> 24u) != 0u) { continue; }
    var finish = i;
    while (finish < n && finish - i < 33u && (arabicMarkWord(origin, finish) >> 24u) != 0u) { finish++; }
    let count = finish - i;
    if (count > 32u) { continue; }
    var order:array<u32,32>;
    var classes:array<u32,32>;
    var mcm:array<bool,32>;
    var saved:array<array<i32,8>,32>;
    for (var j = 0u; j < count; j++) {
      order[j] = j;
      let word = arabicMarkWord(origin, i + j);
      classes[j] = word >> 24u;
      mcm[j] = (word & (1u << 19u)) != 0u;
      let at = glyphAt(base, i + j);
      for (var field = 0u; field < 8u; field++) { saved[j][field] = scratch[at + field]; }
    }
    for (var j = 1u; j < count; j++) {
      let key = order[j];
      var k = j;
      while (k > 0u && classes[key] < classes[order[k - 1u]]) { k--; }
      if (k != j) {
        mergeClusters(base, i + k, i + j + 1u);
        var p = j;
        while (p > k) { order[p] = order[p - 1u]; p--; }
        order[k] = key;
      }
    }
    var ins = 0u;
    for (var classPass = 0u; classPass < 2u; classPass++) {
      let cc = select(220u, 230u, classPass == 1u);
      var p = ins;
      while (p < count && classes[order[p]] < cc) { p++; }
      var q = p;
      while (q < count && classes[order[q]] == cc && mcm[order[q]]) { q++; }
      if (q > p) {
        mergeClusters(base, i + ins, i + q);
        var moved:array<u32,32>;
        let amount = q - p;
        for (var j = 0u; j < amount; j++) { moved[j] = order[p + j]; }
        var j = p;
        while (j > ins) { j--; order[j + amount] = order[j]; }
        for (var j = 0u; j < amount; j++) { order[ins + j] = moved[j]; }
        ins += amount;
      }
    }
    for (var j = 0u; j < count; j++) {
      let at = glyphAt(base, i + j);
      let cluster = scratch[at + 1u];
      for (var field = 0u; field < 8u; field++) { scratch[at + field] = saved[order[j]][field]; }
      scratch[at + 1u] = cluster;
    }
  }
  profileBarrier();
}
`,
};
