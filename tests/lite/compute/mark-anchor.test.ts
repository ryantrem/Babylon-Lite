import { afterAll, afterEach, beforeAll, expect, it } from "vitest";
import { COMPUTE_TEXT_KERNEL_SOURCE } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

let native: NativeHarness;
beforeAll(async () => {
    native = await nativeHarness();
});
afterEach(async () => {
    await assertNativeClean(native);
});
afterAll(async () => {
    await native.close();
});

it.each([7, 8])("preserves missing/first/last/duplicate anchors and signed inherited offsets for mark mode %i", async (mode) => {
    // Execute the production action with controlled immutable anchor words.
    const source =
        COMPUTE_TEXT_KERNEL_SOURCE +
        `
@compute @workgroup_size(1)
fn anchorSearch() {
scratch[0] = i32(params.glyphCount);
scratch[4] = 10;
scratch[5] = -20;
scratch[7] = select(1, 3, params.direction == 7u);
scratch[8] = 3;
scratch[10] = 200;
scratch[11] = 30;
scratch[12] = 5;
scratch[13] = 6;
scratch[15] = 3;
let applied = applyMark(0u, 2u, 1u, 0u, params.direction);
output[0] = i32(applied);
for (var word = 0u; word < 4u; word++) { output[word + 1u] = scratch[10u + word]; }
}`;
    const names = ["params", "program", "tables", "runs", "scratch", "output"] as const;
    const shader = createComputeShader(native.engine, {
        name: "mark-anchor-action",
        computeSource: source,
        entryPoint: "anchorSearch",
        bindings: names.map((name, binding) => computeStorageBufferBinding(name, { group: 0, binding, access: binding >= 4 ? "read-write" : "read" })),
    });
    try {
        for (const anchors of [[], [0, 7, -8], [0, 7, -8, 2, 100, 200, 2, -30, 40, 255, 50, -60]]) {
            for (const base of [0, 1, 2, 3, 254, 255, 256]) {
                const params = Uint32Array.of(0, 2, mode, base, 0, 0, 0);
                const program = Uint32Array.from([0, -2, 3, anchors.length / 3, ...anchors]);
                const buffers = {
                    params: createStorageBuffer(native.engine, params),
                    program: createStorageBuffer(native.engine, program),
                    tables: createStorageBuffer(native.engine, 4),
                    runs: createStorageBuffer(native.engine, 4),
                    scratch: createStorageBuffer(native.engine, 64, { writable: true }),
                    output: createStorageBuffer(native.engine, 20, { writable: true }),
                };
                const task = createComputeTask(native.engine, "mark-anchor-search-boundaries");
                try {
                    addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
                    task.record();
                    await prepareComputeTask(task);
                    submitComputeTasks([task]);
                    let selected = -1;
                    for (let index = 0; index < anchors.length; index += 3) {
                        if (anchors[index] === base) {
                            selected = index;
                        }
                    }
                    const expected = selected < 0 ? [0, 200, 30, 5, 6] : [1, mode === 8 ? 0 : 200, mode === 8 ? 0 : 30, anchors[selected + 1]! + 12, anchors[selected + 2]! - 23];
                    expect([...new Int32Array(await readStorageBuffer(buffers.output))], `mode ${mode}, base ${base}, ${anchors}`).toEqual(expected);
                } finally {
                    task.dispose();
                    Object.values(buffers).forEach(disposeStorageBuffer);
                }
            }
        }
    } finally {
        disposeComputeShader(shader);
    }
});
