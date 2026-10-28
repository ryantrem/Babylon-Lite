/** GPU Indic profile for the shared OpenType executors, not a second shaper.
 * OpenType dev2 behavioral model, independently tested against HarfBuzz11.0.1
 * (Old MIT; notice/provenance in compute-text-indic-unicode.ts).
 * Every per-text normalization, syllable, mask and reorder decision is GPU work.
 */
import type { ComputeTextKernelFragments } from "../compute-text-plan.js";

export const INDIC_KERNEL_FRAGMENTS: ComputeTextKernelFragments = {
    glyphAddress: `
var<workgroup> indicPlane:u32;
var<workgroup> indicOut:u32;
var<workgroup> indicSyllables:u32;
var<workgroup> indicPrefix:array<u32,32>;
var<private> indicRootFlags:u32;
var<private> indicGpos:bool;
fn glyphAt(base:u32, index:u32)->u32 { return base + indicPlane + index * 16u; }
fn indicAux(base:u32, index:u32)->u32 { return base + params.capacity * 32u + index * 4u; }
fn indicWord(slot:u32)->u32 { return program[program[14u] + slot]; }
fn indicData(slot:u32, index:u32)->u32 { return program[program[14u] + indicWord(slot) + index]; }
fn indicProps(cp:u32)->u32 {
  if (cp >= 0x900u && cp < 0x980u) { return indicData(0u, cp - 0x900u); }
  var low = 0u; var high = indicWord(3u);
  while (low < high) {
    let mid = (low + high) / 2u;
    if (indicData(2u, mid * 2u) < cp) { low = mid + 1u; } else { high = mid; }
  }
  if (low < indicWord(3u) && indicData(2u, low * 2u) == cp) { return indicData(2u, low * 2u + 1u); }
  return 0u;
}
fn indicNominal(cp:u32)->u32 {
  var low = 0u; var high = indicWord(11u);
  while (low < high) {
    let mid = (low + high) / 2u;
    if (indicData(10u, mid * 2u) < cp) { low = mid + 1u; } else { high = mid; }
  }
  if (low < indicWord(11u) && indicData(10u, low * 2u) == cp) { return indicData(10u, low * 2u + 1u); }
  return 0u;
}
fn indicCopy(dst:u32, src:u32) {
  for (var k = 0u; k < 16u; k++) { scratch[dst + k] = scratch[src + k]; }
}
fn indicScalar(at:u32, cp:u32, glyph:u32) {
  let p = indicProps(cp);
  scratch[at] = i32(glyph); scratch[at + 2u] = advanceOf(glyph);
  scratch[at + 3u] = 0; scratch[at + 4u] = 0; scratch[at + 5u] = 0;
  scratch[at + 6u] = 0; scratch[at + 7u] = recordProperties(gdefOf(glyph), 0u);
  scratch[at + 8u] = i32(cp);
  scratch[at + 9u] = bitcast<i32>((p & 0xffffu) | ((p & 0xff0000u) << 8u));
  scratch[at + 11u] = i32(p >> 24u);
}
fn indicCat(base:u32, i:u32)->u32 { return u32(scratch[glyphAt(base,i) + 9u]) & 255u; }
fn indicPos(base:u32, i:u32)->u32 { return (u32(scratch[glyphAt(base,i) + 9u]) >> 8u) & 255u; }
fn indicSyll(base:u32, i:u32)->u32 { return (u32(scratch[glyphAt(base,i) + 9u]) >> 16u) & 255u; }
fn indicPosition(base:u32, i:u32, value:u32) {
  let at = glyphAt(base,i) + 9u;
  scratch[at] = bitcast<i32>((u32(scratch[at]) & 0xffff00ffu) | (value << 8u));
}
fn indicOne(base:u32, i:u32, mask:u32)->bool {
  let cat = indicCat(base,i);
  return (scratch[glyphAt(base,i) + 6u] & 0x100000) == 0 && cat < 32u && (mask & (1u << cat)) != 0u;
}
fn indicConsonant(base:u32, i:u32)->bool { return indicOne(base,i,0x58c06u); }
fn indicHalant(base:u32, i:u32)->bool { return indicOne(base,i,16u); }
fn indicJoiner(base:u32, i:u32)->bool { return indicOne(base,i,96u); }
fn indicMove(base:u32, origin:u32, newIndex:u32) {
  var saved:array<i32,16>;
  for (var k = 0u; k < 16u; k++) { saved[k] = scratch[glyphAt(base,origin) + k]; }
  var i = origin;
  while (i > newIndex) { indicCopy(glyphAt(base,i),glyphAt(base,i - 1u)); i--; }
  while (i < newIndex) { indicCopy(glyphAt(base,i),glyphAt(base,i + 1u)); i++; }
  for (var k = 0u; k < 16u; k++) { scratch[glyphAt(base,newIndex) + k] = saved[k]; }
}
fn indicReverse(base:u32, start:u32, finish:u32) {
  var a = start; var b = finish;
  while (a + 1u < b) {
    b--;
    for (var k = 0u; k < 16u; k++) {
      let x = scratch[glyphAt(base,a) + k];
      scratch[glyphAt(base,a) + k] = scratch[glyphAt(base,b) + k];
      scratch[glyphAt(base,b) + k] = x;
    }
    a++;
  }
}
`,
    recordInit: `
fn profileInit(base:u32, out:u32, lane:u32) {
  if (lane == 0u) { indicPlane = 0u; indicOut = out; }
  profileBarrier();
}
`,
    glyphMask: `fn glyphMask(at:u32)->u32 { return bitcast<u32>(scratch[at + 10u]); }`,
    gates: `
fn profileRoot(flags:u32, tableBit:u32) { indicRootFlags = flags; indicGpos = tableBit != 0u; }
fn profileHidden(at:u32)->bool { return (scratch[at + 11u] & 2) != 0; }
fn profileSkip(at:u32, context:bool)->bool {
  if (!profileHidden(at)) { return false; }
  let flags = u32(scratch[at + 11u]);
  if ((flags & 8u) != 0u) { return indicGpos || (context && (indicRootFlags & 2u) != 0u); }
  return (flags & 4u) != 0u && (indicGpos || context || (indicRootFlags & 1u) != 0u);
}
fn profileMayMatch(base:u32, first:u32, i:u32, ahead:bool)->bool {
  return ahead || indicGpos || (indicRootFlags & 4u) == 0u || indicSyll(base,first) == indicSyll(base,i);
}
fn profileInputNext(base:u32, count:u32, start:u32, flag:u32, direction:i32, word:u32, context:bool)->u32 {
  var i = i32(start);
  loop {
    if (i < 0 || u32(i) >= count) { return count; }
    let at = glyphAt(base,u32(i));
    if (live(base,u32(i),count,flag) && (!profileSkip(at,context) || matches(word,readGlyph(base,u32(i))))) { return u32(i); }
    i += direction;
  }
}
fn profilePairSecond(base:u32, count:u32, first:u32, flag:u32, rule:u32)->u32 {
  let inputWord = program[rule];
  return profileInputNext(base,count,first + 1u,flag,1,inputWord + predicateSize(inputWord),false);
}
`,
    profilePrepass: `
fn indicDecompose(cp:u32)->vec2u {
  for (var i = 0u; i < indicWord(5u); i++) {
    if (indicData(4u,i * 3u) == cp) {
      let a = indicData(4u,i * 3u + 1u); let b = indicData(4u,i * 3u + 2u);
      if (indicNominal(a) != 0u && indicNominal(b) != 0u) { return vec2u(a,b); }
    }
  }
  return vec2u(0u);
}
fn indicVowel(base:u32, i:u32)->bool {
  for (var j = 0u; j < indicWord(9u); j++) {
    let a = indicData(8u,j * 4u); let b = indicData(8u,j * 4u + 1u); let c = indicData(8u,j * 4u + 2u);
    if (c == 0u) {
      if (i > 0u && u32(scratch[glyphAt(base,i - 1u) + 8u]) == a && u32(scratch[glyphAt(base,i) + 8u]) == b) { return true; }
    } else if (i > 1u && u32(scratch[glyphAt(base,i - 2u) + 8u]) == a && u32(scratch[glyphAt(base,i - 1u) + 8u]) == b && u32(scratch[glyphAt(base,i) + 8u]) == c) { return true; }
  }
  return false;
}
fn indicBroken(base:u32, i:u32)->bool {
  if ((indicSyll(base,i) & 15u) != 4u || indicCat(base,i) == 14u) { return false; }
  var j = i;
  while (j > 0u && indicSyll(base,j - 1u) == indicSyll(base,i)) {
    j--;
    if (indicCat(base,j) != 14u) { return false; }
  }
  return true;
}
fn indicScatter(base:u32, lane:u32, kind:u32) {
  let n = workgroupUniformLoad(&length);
  let chunk = (n + 31u) / 32u; let start = min(n,lane * chunk); let finish = min(n,start + chunk);
  var total = 0u;
  for (var i = start; i < finish; i++) {
    var extra = false;
    if (kind == 0u) { extra = indicVowel(base,i); }
    if (kind == 1u) { extra = indicDecompose(u32(scratch[glyphAt(base,i) + 8u])).y != 0u; }
    if (kind == 2u) { extra = indicBroken(base,i); }
    scratch[indicAux(base,i)] = select(1,2,extra);
    total += select(1u,2u,extra);
  }
  indicPrefix[lane] = total;
  profileBarrier();
  if (lane == 0u) {
    var prefix = 0u;
    for (var j = 0u; j < 32u; j++) { let count = indicPrefix[j]; indicPrefix[j] = prefix; prefix += count; }
    if (prefix > params.capacity) { terminal = 1u; output[indicOut] = 2; } else { length = prefix; }
  }
  profileBarrier();
  if (workgroupUniformLoad(&terminal) != 0u) { return; }
  let nextPlane = params.capacity * 16u - indicPlane;
  var outIndex = indicPrefix[lane];
  for (var i = start; i < finish; i++) {
    let src = glyphAt(base,i);
    let extra = scratch[indicAux(base,i)] == 2;
    if (extra && kind == 1u) {
      let parts = indicDecompose(u32(scratch[src + 8u]));
      let first = base + nextPlane + outIndex * 16u;
      indicCopy(first,src); indicScalar(first,parts.x,indicNominal(parts.x));
      let second = first + 16u;
      indicCopy(second,src); indicScalar(second,parts.y,indicNominal(parts.y));
      outIndex += 2u;
    } else {
      if (extra) {
        let dst = base + nextPlane + outIndex * 16u;
        indicCopy(dst,src); indicScalar(dst,0x25ccu,indicWord(16u));
        if (kind == 2u) { scratch[dst + 9u] = bitcast<i32>(11u | (14u << 8u) | (indicSyll(base,i) << 16u)); }
        outIndex++;
      }
      indicCopy(base + nextPlane + outIndex * 16u,src);
      outIndex++;
    }
  }
  profileBarrier();
  if (lane == 0u) { indicPlane = nextPlane; }
  profileBarrier();
}
fn indicComposePair(a:u32,b:u32)->u32 {
  for (var i = 0u; i < indicWord(7u); i++) {
    if (indicData(6u,i * 3u) == a && indicData(6u,i * 3u + 1u) == b) { return indicData(6u,i * 3u + 2u); }
  }
  return 0u;
}
fn indicNormalize(base:u32, lane:u32) {
  let n = workgroupUniformLoad(&length);
  mergeLength = n;
  for (var i = lane; i < n; i += 32u) {
    let cc = u32(scratch[glyphAt(base,i) + 9u]) >> 24u;
    if (cc == 0u || (i > 0u && (u32(scratch[glyphAt(base,i - 1u) + 9u]) >> 24u) != 0u)) { continue; }
    var end = i;
    while (end < n && end - i <= 32u && (u32(scratch[glyphAt(base,end) + 9u]) >> 24u) != 0u) { end++; }
    if (end - i > 32u) { continue; }
    for (var j = i + 1u; j < end; j++) {
      let key = u32(scratch[glyphAt(base,j) + 9u]) >> 24u;
      var k = j;
      while (k > i && key < (u32(scratch[glyphAt(base,k - 1u) + 9u]) >> 24u)) { k--; }
      if (k != j) { mergeClusters(base,k,j + 1u); indicMove(base,j,k); }
    }
  }
  profileBarrier();
  let oldPlane = indicPlane; let nextPlane = params.capacity * 16u - oldPlane;
  if (lane == 0u) {
    var written = 0u; var starter = 0u; var lastCc = 0u;
    for (var i = 0u; i < n; i++) {
      let src = glyphAt(base,i); let cp = u32(scratch[src + 8u]);
      let cc = u32(scratch[src + 9u]) >> 24u;
      var composite = 0u;
      if (written > 0u && (scratch[src + 11u] & 1) != 0 && (lastCc == 0u || lastCc < cc)) {
        composite = indicComposePair(u32(scratch[base + nextPlane + starter * 16u + 8u]),cp);
        if (indicNominal(composite) == 0u) { composite = 0u; }
      }
      if (composite != 0u) {
        let dst = base + nextPlane + starter * 16u;
        mergeClusters(base,u32(scratch[dst + 12u]),i + 1u);
        indicScalar(dst,composite,indicNominal(composite));
      } else {
        let dst = base + nextPlane + written * 16u;
        indicCopy(dst,src); scratch[dst + 12u] = i32(i);
        if (cc == 0u) { starter = written; }
        lastCc = cc; written++;
      }
    }
    length = written;
  }
  profileBarrier();
  let count = workgroupUniformLoad(&length);
  for (var i = lane; i < count; i += 32u) {
    let dst = base + nextPlane + i * 16u;
    scratch[dst + 1u] = scratch[base + oldPlane + u32(scratch[dst + 12u]) * 16u + 1u];
  }
  profileBarrier();
  if (lane == 0u) { indicPlane = nextPlane; }
  profileBarrier();
}
fn profilePrepass(origin:u32, base:u32, n:u32, lane:u32) {
  for (var i = lane; i < n; i += 32u) {
    let at = glyphAt(base,i);
    indicScalar(at,runs[origin + 1u + n + i],runs[origin + 1u + i]);
    scratch[at + 10u] = bitcast<i32>(0x80000000u);
  }
  profileBarrier();
  indicScatter(base,lane,0u);
  if (workgroupUniformLoad(&terminal) != 0u) { return; }
  indicScatter(base,lane,1u);
  if (workgroupUniformLoad(&terminal) != 0u) { return; }
  indicNormalize(base,lane);
}
fn indicDfa(slot:u32,index:u32)->u32 {
  let origin = program[14u] + indicWord(21u);
  return program[origin + program[origin + slot] + index];
}
fn indicSetupSyllables(base:u32,n:u32) {
  var state = 31u; var p:i32 = 0; var ts = 0u; var te = 0u; var action = 0u; var serial = 1u;
  loop {
    var trans = 0u;
    if (u32(p) == n) {
      let eofTransition = indicDfa(9u,state);
      if (eofTransition == 0u) { break; }
      trans = eofTransition - 1u;
    } else {
      if (indicDfa(8u,state) == 10u) { ts = u32(p); }
      let low = indicDfa(1u,state * 2u); let high = indicDfa(1u,state * 2u + 1u);
      let span = indicDfa(2u,state); let cat = indicCat(base,u32(p));
      let key = select(span,cat - low,span > 0u && cat >= low && cat <= high);
      trans = indicDfa(4u,indicDfa(3u,state) + key);
    }
    state = indicDfa(5u,trans);
    let code = indicDfa(6u,trans);
    var kind:i32 = -1;
    switch code {
      case 2u: { te = u32(p) + 1u; }
      case 11u: { te = u32(p) + 1u; kind = 5; }
      case 14u,15u,18u,20u,16u,17u: {
        te = u32(p); p--;
        if (code == 14u) { kind = 0; } else if (code == 15u) { kind = 1; }
        else if (code == 18u) { kind = 2; } else if (code == 20u) { kind = 3; }
        else if (code == 16u) { kind = 4; } else { kind = 5; }
      }
      case 1u,3u,7u,8u,4u: {
        p = i32(te) - 1;
        if (code == 1u) { kind = 0; } else if (code == 3u) { kind = 1; }
        else if (code == 7u) { kind = 2; } else if (code == 8u) { kind = 3; } else { kind = 4; }
      }
      case 6u: { p = i32(te) - 1; kind = select(5,select(4,0,action == 1u),action == 1u || action == 6u); }
      case 19u,13u,5u,12u: {
        te = u32(p) + 1u;
        if (code == 19u) { action = 1u; } else if (code == 13u) { action = 5u; } else if (code == 5u) { action = 6u; } else { action = 7u; }
      }
      default: {}
    }
    if (kind >= 0) {
      for (var i = ts; i < te; i++) {
        let at = glyphAt(base,i) + 9u;
        scratch[at] = bitcast<i32>((u32(scratch[at]) & 0xff00ffffu) | (((serial << 4u) | u32(kind)) << 16u));
      }
      serial++; if (serial == 16u) { serial = 1u; }
    }
    if (indicDfa(7u,state) == 9u) { ts = 0u; }
    p++;
    if (p > i32(n)) { break; }
  }
}
fn indicSyllableList(base:u32) {
  var count = 0u;
  for (var i = 0u; i < length; i++) {
    if (i == 0u || indicSyll(base,i) != indicSyll(base,i - 1u)) { scratch[indicAux(base,count)] = i32(i); count++; }
  }
  indicSyllables = count;
}
`,
    pauses: `
fn indicInitial(base:u32, start:u32, end:u32) {
  var main = end; var reph = false; var limit = start;
  if (indicWord(18u) != 0u && start + 3u <= end && !indicJoiner(base,start + 2u)) {
    let pair = (readGlyph(base,start) << 16u) | readGlyph(base,start + 1u);
    var low = 0u; var high = indicWord(14u);
    while (low < high) { let mid = (low + high) / 2u; if (indicData(13u,mid) < pair) { low = mid + 1u; } else { high = mid; } }
    if (low < indicWord(14u) && indicData(13u,low) == pair) {
      limit += 2u; main = start; reph = true;
      while (limit < end && indicJoiner(base,limit)) { limit++; }
    }
  }
  var i = end; var seenBelow = false;
  loop {
    if (i <= limit) { break; }
    i--;
    if (indicConsonant(base,i)) {
      let p = indicPos(base,i);
      if (p != 8u && (p != 11u || seenBelow)) { main = i; break; }
      if (p == 8u) { seenBelow = true; }
      main = i;
    } else if (i > start && indicCat(base,i) == 6u && indicCat(base,i - 1u) == 4u) { break; }
  }
  if (reph && main == start && limit - main <= 2u) { reph = false; }
  for (i = start; i < main; i++) { indicPosition(base,i,min(3u,indicPos(base,i))); }
  if (main < end) { indicPosition(base,main,4u); }
  if (reph) { indicPosition(base,start,1u); }
  var lastPosition = 0u;
  for (i = start; i < end; i++) {
    let cat = indicCat(base,i);
    if (cat == 5u || cat == 6u || cat == 3u || cat == 12u || cat == 16u || cat == 4u) {
      indicPosition(base,i,lastPosition);
      if (cat == 4u && lastPosition == 2u) {
        var j = i;
        while (j > start) { j--; if (indicPos(base,j) != 2u) { indicPosition(base,i,indicPos(base,j)); break; } }
      }
    } else if (indicPos(base,i) != 13u) {
      if (cat == 13u && i > start && indicCat(base,i - 1u) == 8u) { indicPosition(base,i - 1u,indicPos(base,i)); }
      lastPosition = indicPos(base,i);
    }
  }
  var previous = main;
  for (i = main + 1u; i < end; i++) {
    if (indicConsonant(base,i)) {
      for (var j = previous + 1u; j < i; j++) { if (indicPos(base,j) < 13u) { indicPosition(base,j,indicPos(base,i)); } }
      previous = i;
    } else if (indicCat(base,i) == 7u || indicCat(base,i) == 13u) { previous = i; }
  }
  for (i = start; i < end; i++) { scratch[glyphAt(base,i) + 12u] = i32(i - start); }
  for (i = start + 1u; i < end; i++) {
    let key = indicPos(base,i); var j = i;
    while (j > start && key < indicPos(base,j - 1u)) { j--; }
    if (j != i) { indicMove(base,i,j); }
  }
  var firstMatra = end; var lastMatra = end; main = end;
  for (i = start; i < end; i++) {
    if (indicPos(base,i) == 4u) { main = i; break; }
    if (indicPos(base,i) == 2u) { if (firstMatra == end) { firstMatra = i; } lastMatra = i; }
  }
  if (firstMatra < lastMatra) {
    indicReverse(base,firstMatra,lastMatra + 1u);
    var first = firstMatra;
    for (i = first; i <= lastMatra; i++) {
      if (indicCat(base,i) == 7u || indicCat(base,i) == 13u) { indicReverse(base,first,i + 1u); first = i + 1u; }
    }
  }
  if (end - start > 127u) {
    if (main < end) { mergeClusters(base,main,end); }
  } else {
    for (i = main; i < end; i++) {
      if (scratch[glyphAt(base,i) + 12u] == 255) { continue; }
      var minimum = i; var maximum = i; var j = start + u32(scratch[glyphAt(base,i) + 12u]);
      while (j != i) {
        minimum = min(minimum,j); maximum = max(maximum,j);
        let next = start + u32(scratch[glyphAt(base,j) + 12u]); scratch[glyphAt(base,j) + 12u] = 255; j = next;
      }
      mergeClusters(base,max(main,minimum),maximum + 1u);
    }
  }
  for (i = start; i < end && indicPos(base,i) == 1u; i++) { scratch[glyphAt(base,i) + 10u] |= i32(indicWord(18u)); }
  for (i = start; i < main; i++) { scratch[glyphAt(base,i) + 10u] |= i32(indicWord(19u) | indicWord(20u)); }
  for (i = main + 1u; i < end; i++) { scratch[glyphAt(base,i) + 10u] |= i32(indicWord(20u)); }
  for (i = start + 1u; i < end; i++) {
    if (!indicJoiner(base,i)) { continue; }
    let nonJoiner = indicCat(base,i) == 5u; var j = i;
    loop {
      j--;
      if (nonJoiner) { scratch[glyphAt(base,j) + 10u] &= ~i32(indicWord(19u)); }
      if (j == start || indicConsonant(base,j)) { break; }
    }
  }
}
fn indicFinal(base:u32, start:u32, end:u32) {
  for (var i = start; i < end; i++) {
    let at = glyphAt(base,i);
    if (u32(scratch[at]) == indicWord(15u) && (scratch[at + 6u] & 0x300000) == 0x300000) {
      scratch[at + 9u] = bitcast<i32>((u32(scratch[at + 9u]) & 0xffffff00u) | 4u);
      scratch[at + 6u] &= ~0x300000;
    }
  }
  var main = start;
  while (main < end) {
    if (indicPos(base,main) >= 4u) { if (main > start && indicPos(base,main) > 4u) { main--; } break; }
    main++;
  }
  if (main == end && main > start && indicOne(base,main - 1u,64u)) { main--; }
  if (main < end) { while (main > start && indicOne(base,main,24u)) { main--; } }
  if (start + 1u < end && start < main) {
    var newIndex = select(main - 1u,main - 2u,main == end);
    loop {
      while (newIndex > start && !indicOne(base,newIndex,0x2090u)) { newIndex--; }
      if (indicHalant(base,newIndex) && indicPos(base,newIndex) != 2u) {
        if (newIndex + 1u < end && indicCat(base,newIndex + 1u) == 6u && newIndex > start) { newIndex--; continue; }
      } else { newIndex = start; }
      break;
    }
    if (newIndex > start && indicPos(base,newIndex) != 2u) {
      var i = newIndex;
      while (i > start) {
        i--;
        if (indicPos(base,i) != 2u) { continue; }
        if (i < main && main <= newIndex) { main--; }
        indicMove(base,i,newIndex);
        if (newIndex < min(end,main + 1u)) { mergeClusters(base,newIndex,min(end,main + 1u)); }
        newIndex--;
      }
    } else {
      for (var i = start; i < main; i++) {
        if (indicPos(base,i) == 2u) { mergeClusters(base,i,min(end,main + 1u)); break; }
      }
    }
  }
  let flags = scratch[glyphAt(base,start) + 6u];
  if (start + 1u < end && indicPos(base,start) == 1u && ((indicCat(base,start) == 14u) != ((flags & 0x300000) == 0x100000))) {
    var newIndex = start + 1u;
    while (newIndex < main && !indicHalant(base,newIndex)) { newIndex++; }
    if (newIndex < main && indicHalant(base,newIndex)) {
      if (newIndex + 1u < main && indicJoiner(base,newIndex + 1u)) { newIndex++; }
    } else {
      newIndex = end - 1u;
      while (newIndex > start && indicPos(base,newIndex) == 13u) { newIndex--; }
      if (indicHalant(base,newIndex)) {
        for (var i = main + 1u; i < newIndex; i++) { if (indicCat(base,i) == 7u || indicCat(base,i) == 13u) { newIndex--; } }
      }
    }
    mergeClusters(base,start,newIndex + 1u); indicMove(base,start,newIndex);
  }
}
fn profilePause(base:u32, pause:u32, lane:u32) {
  if (pause == 0u) { return; }
  if (pause == 1u && lane == 0u) { indicSetupSyllables(base,length); }
  profileBarrier();
  if (pause == 2u) {
    let n = workgroupUniformLoad(&length);
    for (var i = lane; i < n; i += 32u) {
      if (indicPos(base,i) == 4u) { let p = indicData(12u,readGlyph(base,i)); indicPosition(base,i,select(select(4u,8u,p == 1u),11u,p == 2u)); }
    }
    profileBarrier();
    indicScatter(base,lane,2u);
    if (workgroupUniformLoad(&terminal) != 0u) { return; }
  }
  if (pause == 2u || pause == 3u) {
    if (lane == 0u) { indicSyllableList(base); }
    profileBarrier();
    mergeLength = workgroupUniformLoad(&length);
    let count = workgroupUniformLoad(&indicSyllables);
    for (var s = lane; s < count; s += 32u) {
      let start = u32(scratch[indicAux(base,s)]);
      var end = length;
      if (s + 1u < count) { end = u32(scratch[indicAux(base,s + 1u)]); }
      if ((indicSyll(base,start) & 15u) < 3u || (indicSyll(base,start) & 15u) == 4u) {
        if (pause == 2u) { indicInitial(base,start,end); } else { indicFinal(base,start,end); }
      }
    }
  }
  if (pause == 4u) {
    let n = workgroupUniformLoad(&length);
    for (var i = lane; i < n; i += 32u) { let at = glyphAt(base,i) + 9u; scratch[at] &= bitcast<i32>(0xff00ffffu); }
  }
  profileBarrier();
}
`,
    postprocess: `
fn profileFinish(base:u32, n:u32, lane:u32) {
  for (var i = lane; i < n; i += 32u) {
    let at = glyphAt(base,i);
    if (profileHidden(at)) { scratch[at] = i32(indicWord(17u)); for (var k = 2u; k < 6u; k++) { scratch[at + k] = 0; } }
  }
  profileBarrier();
}
`,
};
