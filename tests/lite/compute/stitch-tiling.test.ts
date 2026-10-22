import { createHash } from "node:crypto";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { COMPUTE_TEXT_STITCH_KERNEL, stitchTileRows } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

// Verbatim expanded production kernel at 3d598eb01749d25426a553247d99e3704727e78d.
const OLD_STITCH = `
@group(0) @binding(0) var<storage, read> seg: array<i32>;
@group(0) @binding(1) var<storage, read_write> shaped: array<i32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read> params: vec4u;
const HEADER: u32 = 4u;
const RECORD: u32 = 6u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (wg.x >= params.x) { return; }
  let run = params.w + wg.x;
  let first = table[wg.x*2u];
  let count = table[wg.x*2u+1u];
  let sStride = HEADER + params.z*RECORD;
  let out = run*(HEADER + params.y*RECORD);
  let segBase = params.x*2u;
  var status: i32 = 0;
  var failed = 0u;
  var total = 0u;
  for (var i = 0u; i < count; i++) {
    let h = (first+i)*sStride;
    if (status == 0 && seg[h] != 0) { status = seg[h]; failed = first+i; }
    total += u32(seg[h+1u]);
  }
  if (status == 0 && total > params.y) { status = 2; failed = 0xffffffffu; }
  if (status != 0) {
    if (lane == 0u) {
      shaped[out] = status;
      shaped[out+1u] = 0;
      if (failed != 0xffffffffu) {
        shaped[out+2u] = seg[failed*sStride+2u];
        shaped[out+3u] = seg[failed*sStride+3u];
      } else { shaped[out+2u] = 0; shaped[out+3u] = 0; }
    }
    return;
  }
  if (lane == 0u) { shaped[out] = 0; shaped[out+1u] = i32(total); shaped[out+2u] = 0; shaped[out+3u] = 0; }
  var base = 0u;
  for (var i = 0u; i < count; i++) {
    let h = (first+i)*sStride;
    let n = u32(seg[h+1u]);
    let start = i32(table[segBase + first + i]);
    for (var k = lane; k < n; k += 64u) {
      let src = h + HEADER + k*RECORD;
      let dst = out + HEADER + (base+k)*RECORD;
      shaped[dst] = seg[src];
      shaped[dst+1u] = seg[src+1u] + start;
      shaped[dst+2u] = seg[src+2u];
      shaped[dst+3u] = seg[src+3u];
      shaped[dst+4u] = seg[src+4u];
      shaped[dst+5u] = seg[src+5u];
    }
    base += n;
  }
}
`;

interface StitchCase {
    counts: number[];
    runBase?: number;
    first?: number;
    capacityDelta?: number;
    failures?: { run: number; segment: number; code: number }[];
}

describe("native tiled stitch matches frozen production", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        expect(createHash("sha256").update(OLD_STITCH).digest("hex")).toBe("6a4543bb7fda56958d3e1dd99b447ca0c9a8e78cc8d0f66e30d400c08f747412");
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    async function compare(test: StitchCase): Promise<void> {
        const segmentCapacity = 67;
        const firstOffset = test.first ?? 3;
        const runBase = test.runBase ?? 2;
        const segmentCount = firstOffset + test.counts.reduce((a, b) => a + b, 0);
        const stride = 4 + segmentCapacity * 6;
        const table = new Uint32Array(test.counts.length * 2 + segmentCount);
        const seg = new Int32Array(Math.max(4, segmentCount * stride));
        let first = firstOffset;
        let maxTotal = 0;
        const totals: number[] = [];
        test.counts.forEach((count, run) => {
            table.set([first, count], run * 2);
            let total = 0;
            for (let i = 0; i < count; i++) {
                const index = first + i;
                const n = i % 9 === 0 ? 0 : i % 7 === 0 ? 67 : 1 + ((i * 17 + run) % 5);
                seg.set([0, n, 111 + index, -222 - index], index * stride);
                table[test.counts.length * 2 + index] = index * 89;
                for (let k = 0; k < n; k++) {
                    seg.set([index * 97 + k, k * 2, index - k, k - index, -k - 5, index + 7], index * stride + 4 + k * 6);
                }
                total += n;
            }
            totals.push(total);
            maxTotal = Math.max(maxTotal, total);
            first += count;
        });
        for (const failure of test.failures ?? []) {
            const at = (table[failure.run * 2]! + failure.segment) * stride;
            seg[at] = failure.code;
            // Flagged malformed counts must preserve first-failure diagnostics without copying.
            seg[at + 1] = -1;
        }
        const capacity = Math.max(1, maxTotal + (test.capacityDelta ?? 0));
        const initial = new Int32Array((runBase + test.counts.length + 2) * (4 + capacity * 6) + 16).fill(0x13579bdf);
        const shared = {
            seg: createStorageBuffer(native.engine, seg),
            table: createStorageBuffer(native.engine, table),
            params: createStorageBuffer(native.engine, Uint32Array.of(test.counts.length, capacity, segmentCapacity, runBase)),
        };
        const output = [OLD_STITCH, COMPUTE_TEXT_STITCH_KERNEL].map(() => createStorageBuffer(native.engine, initial, { writable: true }));
        const shaders = [OLD_STITCH, COMPUTE_TEXT_STITCH_KERNEL].map((computeSource) =>
            createComputeShader(native.engine, {
                computeSource,
                bindings: [
                    computeStorageBufferBinding("seg", { group: 0, binding: 0 }),
                    computeStorageBufferBinding("shaped", { group: 0, binding: 1, access: "read-write" }),
                    computeStorageBufferBinding("table", { group: 0, binding: 2 }),
                    computeStorageBufferBinding("params", { group: 0, binding: 3 }),
                ],
            })
        );
        try {
            const formula = stitchTileRows(table, test.counts.length);
            for (const rows of [...new Set([1, formula, formula + 2])]) {
                const schedules = shaders.map((_shader, i) => {
                    const task = createComputeTask(native.engine, `stitch-tiling-${i}-${rows}`);
                    addComputeDispatch(
                        task,
                        createComputeDispatch(shaders[i]!, createComputeBindingSet(shaders[i]!, { ...shared, shaped: output[i]! }), {
                            size: { x: test.counts.length + 2, y: i ? rows : 1 },
                        })
                    );
                    task.record();
                    return task;
                });
                try {
                    for (let repeat = 0; repeat < 2; repeat++) {
                        for (let i = 0; i < schedules.length; i++) {
                            updateStorageBuffer(native.engine, output[i]!, initial);
                            await prepareComputeTask(schedules[i]!);
                            submitComputeTasks([schedules[i]!]);
                        }
                        const actual = await readStorageBuffer(output[1]!);
                        const expected = await readStorageBuffer(output[0]!);
                        expect(Buffer.compare(Buffer.from(actual), Buffer.from(expected)), `rows=${rows}, repeat=${repeat}: every output/sentinel byte`).toBe(0);
                        const words = new Int32Array(actual);
                        test.counts.forEach((_count, run) => {
                            const header = (runBase + run) * (4 + capacity * 6);
                            const failure = test.failures?.filter((failure) => failure.run === run).sort((a, b) => a.segment - b.segment)[0];
                            if (failure) {
                                const index = table[run * 2]! + failure.segment;
                                expect([...words.slice(header, header + 4)]).toEqual([failure.code, 0, 111 + index, -222 - index]);
                            } else {
                                expect([...words.slice(header, header + 4)]).toEqual(totals[run]! > capacity ? [2, 0, 0, 0] : [0, totals[run], 0, 0]);
                            }
                        });
                    }
                } finally {
                    schedules.forEach((task) => task.dispose());
                }
            }
        } finally {
            shaders.forEach(disposeComputeShader);
            [...Object.values(shared), ...output].forEach(disposeStorageBuffer);
        }
    }

    it.each([0, 1, 63, 64, 65, 128, 129, 1961])("preserves every output/header/sentinel byte for S=%i at Y=1/formula/overprovided, twice", async (count) => {
        await compare({ counts: [count] });
    });
    it("preserves mixed runs, nonzero first/runBase, exact fit and capacity overflow by one", async () => {
        await compare({ counts: [2, 65, 129, 1], runBase: 7, first: 11 });
        await compare({ counts: [129], capacityDelta: -1 });
    });
    it("preserves the earliest failure's exact signed status and diagnostics across all tiles", async () => {
        for (const segment of [0, 63, 64, 127, 128, 1960]) {
            await compare({ counts: [1961], failures: [{ run: 0, segment, code: segment === 128 ? -7 : 1 }] });
        }
        await compare({
            counts: [129],
            failures: [
                { run: 0, segment: 128, code: 3 },
                { run: 0, segment: 64, code: -11 },
            ],
        });
    });
    it("keeps ragged 300-run global-Y overdispatch and high run indices exact, including failures and overflow", async () => {
        const counts = Array.from({ length: 300 }, (_n, i) => (i === 257 ? 200 : 1 + (i % 3)));
        await compare({ counts, runBase: 13 });
        await compare({
            counts,
            failures: [
                { run: 1, segment: 1, code: 9 },
                { run: 257, segment: 199, code: -8 },
            ],
        });
        await compare({ counts, capacityDelta: -1 });
        await compare({ counts, capacityDelta: -100000 });
    });
});
