import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { createComputeTextShapeShader, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { fixture, expectedFields } from "./fixtures";
import type { Fixture } from "./fixtures";
import { syntheticFont } from "./synthetic";

describe("parallel post-GSUB refresh", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    const forceSerial = (source: string) => source.replace("let parallelRefresh = ", "let parallelRefresh = false && ");

    async function check(font: Fixture, text: string, capacity?: number): Promise<void> {
        const expected = expectedFields(font, text).flat();
        const parallel = await raw(font, text, undefined, 6, capacity);
        const serial = await raw(font, text, forceSerial, 6, capacity);
        expect(parallel.slice(0, 4)).toEqual([0, expected.length / 6, 0, 0]);
        expect(parallel.slice(4, 4 + expected.length)).toEqual(expected);
        expect(serial.slice(0, 4 + expected.length)).toEqual(parallel.slice(0, 4 + expected.length));
    }

    it.each([1, 2, 31, 32, 33, 64, 255, 256, 257, 480, 768])("preserves all eight words across arbitrary tombstones at length %i", async (count) => {
        const font = syntheticFont("refresh-roots");
        const deleted = new Set([0, count - 1, 30, 31, 32, 33, 63, 64, 255, 256, 257, 511, 512, 513]);
        for (let index = 96; index < 128; index++) {
            deleted.add(index);
        }
        const survivors = Array.from({ length: count }, (_, index) => index).filter((index) => !deleted.has(index));
        const inject = (source: string) => {
            const start = source.indexOf("@compute @workgroup_size(32)");
            const end = source.indexOf("fn rejectRun", start);
            const refresh = source.includes("fn refreshRun(")
                ? "refreshRun(base, lane, 3u);"
                : "if (lane == 0u) { length = compactRun(base, length); rootPresence = activeRoots(base, length); }";
            const body = `
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane:u32) {
let base = 0u;
nominalSource = false;
if (lane == 0u) { length = ${count}u; }
for (var index = lane; index < ${count}u; index += 32u) {
let at = glyphAt(base, index);
scratch[at] = i32(1u + index % 3u);
for (var word = 1u; word < 8u; word++) { scratch[at + word] = i32(index * 101u + word); }
scratch[at + 6u] = select(0, 1, ${[...deleted].map((index) => `index == ${index}u`).join(" || ")});
}
${source.includes("var<storage, read_write> scratch") ? "storageBarrier();" : ""}
workgroupBarrier();
${refresh}
${source.includes("var<storage, read_write> scratch") ? "storageBarrier();" : ""}
let count = workgroupUniformLoad(&length);
let presence = workgroupUniformLoad(&rootPresence[0u]);
if (lane == 0u) { output[0] = 0; output[1] = i32(count); output[2] = i32(presence); output[3] = 0; }
for (var index = lane; index < count; index += 32u) {
for (var word = 0u; word < 8u; word++) { output[4u + index * 8u + word] = scratch[glyphAt(base, index) + word]; }
}
}
`;
            return source.slice(0, start) + body + source.slice(end);
        };
        const parallel = await raw(font, "A".repeat(count), inject, 8);
        const serial = await raw(font, "A".repeat(count), (source) => forceSerial(inject(source)), 8);
        const expected = survivors.flatMap((index) => [1 + (index % 3), index * 101 + 1, index * 101 + 2, index * 101 + 3, index * 101 + 4, index * 101 + 5, 0, index * 101 + 7]);
        expect(parallel[1]).toBe(survivors.length);
        expect(parallel.slice(0, 4 + expected.length)).toEqual(serial.slice(0, 4 + expected.length));
        expect(parallel.slice(4, 4 + expected.length)).toEqual(expected);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("compacts real boundary-crossing ligatures exactly: %s", async (file) => {
        const font = fixture(file);
        for (const count of [31, 32, 33, 64, 255, 256, 257, 480, 768]) {
            const text = Array<string>(count).fill("H");
            for (const at of [29, 31, 62, 254, 256, 510, 600]) {
                if (at + 1 < count) {
                    text[at] = "f";
                    text[at + 1] = "i";
                }
            }
            await check(font, text.join(""));
        }
    });

    it("compacts dense real GSUB ligatures at chunk and tile boundaries", async () => {
        const font = syntheticFont("refresh-ligature");
        for (const prefix of [0, 31, 255, 511]) {
            await check(font, "C".repeat(prefix) + "ABC".repeat(40) + "C".repeat(35));
        }
    });

    it("rescans cooperative id changes, gaining and losing later roots in both scratch variants", async () => {
        const font = syntheticFont("refresh-roots");
        for (const count of [1, 2, 31, 32, 33, 64, 255, 256, 257, 480, 768]) {
            await check(font, "A".repeat(count));
        }
    });

    it("retains RTL order and mark offsets in both scratch variants", async () => {
        const font = fixture("NotoSansHebrew-Regular.ttf");
        for (const repeats of [10, 80]) {
            await check(font, "שָׁלוֹם ".repeat(repeats));
        }
    });

    it("fences serial short-run refresh in oversized global scratch", async () => {
        for (const count of [1, 2, 31, 32, 33, 64, 255, 256]) {
            await check(syntheticFont("refresh-roots"), "A".repeat(count), 512);
        }
        const font = syntheticFont("refresh-ligature");
        for (const count of [1, 2, 10, 11]) {
            await check(font, "ABC".repeat(count), 512);
        }
    });

    it("preserves the first unsupported status and source cursor", async () => {
        const font = syntheticFont("backtrack-reject");
        for (const index of [31, 255, 256, 600]) {
            const text = "C".repeat(index) + "ABA";
            const parallel = await raw(font, text);
            const serial = await raw(font, text, forceSerial);
            // Parent 0 is now executable. Its sequence-1 record invokes the
            // genuinely unsupported child lookup 1 at the following A.
            expect(parallel.slice(0, 4)).toEqual([1, 0, 1, index + 2]);
            expect(serial.slice(0, 4)).toEqual(parallel.slice(0, 4));
        }
    });

    it("preserves capacity overflow before any refresh", async () => {
        const font = syntheticFont("multiple");
        for (const count of [64, 600]) {
            const transform = (source: string) => source.replace("let capacity = params.capacity;", "let capacity = 1u;");
            const parallel = await raw(font, "A".repeat(count), transform);
            const serial = await raw(font, "A".repeat(count), (source) => forceSerial(transform(source)));
            expect(parallel.slice(0, 4)).toEqual([2, 0, 0, 0]);
            expect(serial.slice(0, 4)).toEqual(parallel.slice(0, 4));
        }
    });

    async function raw(font: Fixture, text: string, transform = (source: string) => source, recordWords = 6, capacity?: number): Promise<number[]> {
        const selection = selectComputeTextRun(text);
        const prepared = prepareComputeTextShapeWords(
            createComputeTextPlan(font.font, selection),
            [{ scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => mapComputeFontScalar(font.font, cp)) }],
            0,
            capacity
        );
        const inputs = {
            params: prepared.params,
            program: prepared.programWords,
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + prepared.capacity * recordWords) * 4,
        };
        // This diagnostic header holds the first presence word. Every fixture
        // here has <=32 roots; fail rather than silently truncate a wider plan.
        expect(prepared.program.words[13]).toBe(1);
        const buffers = Object.fromEntries(
            Object.entries(inputs).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const production = createComputeTextShapeShader(native.engine, prepared.capacity);
        const shader = createComputeShader(native.engine, {
            name: "parallel-refresh-differential",
            computeSource: transform(production._source).replace("if (simple.y != 0u)", "if (false && simple.y != 0u)"),
            bindings: Object.keys(inputs).map((name, binding) => computeStorageBufferBinding(name, { group: 0, binding, access: binding >= 4 ? "read-write" : "read" })),
        });
        const task = createComputeTask(native.engine, "parallel-refresh-records");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            return [...new Int32Array(await readStorageBuffer(buffers.output!))];
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            disposeComputeShader(production);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    }
});
