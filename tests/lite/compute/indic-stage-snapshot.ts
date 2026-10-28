/** Test-only pause snapshot. Neither its control nor its stop selector exists in production. */
export function indicStageSnapshot(source: string, pause: number): string {
    const call = "profilePause(base, program[stageAt + 3u], lane);";
    if (!source.includes(call) || source.indexOf(call) !== source.lastIndexOf(call)) {
        throw new Error("Indic snapshot requires exactly one stage-pause call");
    }
    return source.replace(
        call,
        `${call}
if (program[stageAt + 3u] == ${pause}u) {
 let n = workgroupUniformLoad(&length);
 for (var i = lane; i < n; i += 32u) {
  for (var k = 0u; k < 6u; k++) { output[out + 4u + i * 6u + k] = scratch[glyphAt(base,i) + k]; }
 }
 if (lane == 0u) { output[out + 1u] = i32(n); output[out + 3u] = ${pause}; }
 workgroupBarrier(); return;
}`
    );
}
