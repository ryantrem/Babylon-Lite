/** Pre-parallel measure/place entry points; test-only exact-output oracle. */
export const serialParagraphLayout = `
@compute @workgroup_size(64)
fn serialMeasureTest(@builtin(global_invocation_id) invocation:vec3u) {
  let i = invocation.x;
  if (i >= input[9]) { return; }
  var p = input[8] + i;
  if (input[10] == 1u) { p = input[16u + i]; }
  let head = p * (HEADER + input[1] * RECORD);
  var failed = u32(shaped[head]);
  let count = u32(shaped[head + 1u]);
  var visible = 0u;
  if (failed == 0u) {
    if (count > input[1]) { failed = 2u; }
    else {
      for (var i = 0u; i < count; i++) {
        let glyph = u32(shaped[recordAt(p, i)]);
        if (glyph >= input[3]) { failed = 3u; break; }
        let slot = slots[glyph];
        if (slot == MISSING) { failed = 3u; break; }
        if (slot != EMPTY) { visible++; }
      }
    }
  }
  if (failed != 0u) { info[p] = vec4u(0u, failed, 1u, 0u); return; }
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
@compute @workgroup_size(64)
fn serialPlaceTest(@builtin(global_invocation_id) invocation:vec3u) {
  let p = invocation.x;
  if (p >= input[0] || status[0] != 0u) { return; }
  let slot = input[16u + p];
  let paragraph = info[input[11] + p];
  let scale = bitcast<f32>(input[2]);
  let maxWidth = bitcast<f32>(input[4]);
  let lineHeight = bitcast<f32>(input[5]);
  let spacing = bitcast<f32>(input[6]);
  var total = paragraph.x;
  var penX = 0;
  var penY = 0;
  for (var l = 0u; l < paragraph.z; l++) {
    let line = lines[slot * input[1] + l];
    let lineX = penX;
    var align = 0.0;
    if (input[7] != 0u) {
      align = max(0.0, maxWidth - bitcast<f32>(line.z));
      if (input[7] == 1u) { align *= 0.5; }
    }
    let baseline = fma(f32(paragraph.w + l), lineHeight, f32(p) * spacing);
    for (var i = line.x; i < line.x + line.y; i++) {
      let at = recordAt(slot, i);
      let slot = slots[u32(shaped[at])];
      if (slot != EMPTY) {
        instances[total] = Instance(fma(f32(penX - lineX + shaped[at + 4u]), scale, align), fma(f32(penY + shaped[at + 5u]), scale, -baseline), slot);
        total++;
      }
      penX += shaped[at + 2u];
      penY += shaped[at + 3u];
    }
  }
}
`;
