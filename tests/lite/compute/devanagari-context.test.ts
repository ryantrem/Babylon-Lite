/** Exact root-window regression for the first Indic nested ligature.
 * Prefix vectors are from pinned HB11.0.1 trace for U0926 U094D U0927.
 * Only a copied test program is truncated; production has no root-limit option.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { fixture } from "./fixtures";
import { nativeHarness, assertNativeClean } from "./native";
import type { NativeHarness } from "./native";
import { ensureComputeTextProfile, createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextShapeShader, prepareComputeTextShapeWords, createShapeSchedule } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { createComputeTask, addComputeDispatch, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { indicStageSnapshot } from "./indic-stage-snapshot";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";

describe("Indic context root windows", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        await ensureComputeTextProfile("indic");
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native.close());
    it("complete single-record context retains the original boundary and exact final fields", async () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf"),
            selection = selectComputeTextRun("द्ध");
        const p = prepareComputeTextShapeWords(createComputeTextPlan(face.font, selection), [
            {
                scalars: selection._scalars,
                glyphIds: selection._scalars.map((cp) => mapComputeFontScalar(face.font, cp)),
            },
        ]);
        const production = createComputeTextShapeShader(native.engine, p.capacity, p.variant);
        const marker = "return vec2u(advance,high);";
        expect(production._source.indexOf(marker)).toBeGreaterThan(0);
        const shader = createComputeShader(native.engine, {
            name: "indic-complete-context-boundary",
            bindings: production._decls,
            computeSource: production._source.replace(
                marker,
                `
if (parentLookup == 120u) {
 let tail = 4u + params.capacity * 6u;
 output[tail] = i32(index); output[tail + 1u] = i32(count); output[tail + 2u] = i32(boundary);
 output[tail + 3u] = i32(advance); output[tail + 4u] = i32(high); output[tail + 5u] = actionDelta;
}
${marker}`
            ),
        });
        const buffers = {
            params: createStorageBuffer(native.engine, p.params),
            program: createStorageBuffer(native.engine, p.programWords),
            tables: createStorageBuffer(native.engine, p.tables),
            runs: createStorageBuffer(native.engine, p.runs),
            scratch: createStorageBuffer(native.engine, p.scratchBytes, { writable: true }),
            output: createStorageBuffer(native.engine, (4 + p.capacity * 6 + 6) * 4, { writable: true }),
        };
        const schedule = createShapeSchedule(native.engine, shader, buffers, 1);
        try {
            await prepareComputeTask(schedule.task);
            submitComputeTasks([schedule.task]);
            const raw = new Int32Array(await readStorageBuffer(buffers.output));
            expect([...raw.slice(0, 2)]).toEqual([0, 1]);
            expect([...raw.slice(4, 10)]).toEqual([448, 0, 799, 0, 0, 0]);
            const state = [...raw.slice(4 + p.capacity * 6)];
            expect(state).toEqual([0, 3, 3, 3, 2, 0]);
            expect(state[2]!).toBeGreaterThanOrEqual(state[0]! + 1);
        } finally {
            schedule.task.dispose();
            disposeComputeShader(shader);
            disposeComputeShader(production);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
    it("CJCT parent/child matching and isolated child mutation are exact", async () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf"),
            selection = selectComputeTextRun("द्ध");
        const plan = createComputeTextPlan(face.font, selection),
            scalars = selection._scalars;
        const prepared = prepareComputeTextShapeWords(plan, [{ scalars, glyphIds: scalars.map((cp) => mapComputeFontScalar(face.font, cp)) }]);
        const production = createComputeTextShapeShader(native.engine, prepared.capacity, prepared.variant);
        const probe = `
if (lookupIndex == 120u && tableBit == 0u) {
 if (lane == 0u) {
   let chosen = selectRule(base,count,0u,flag,ruleOrigin,directory);
   recordMatches = true;
   let parentMatch = matchInput(base,count,0u,flag,chosen - 1u);
   recordMatches = false;
   let action = program[chosen - 1u + 6u];
   let childIndex = program[action + 2u];
   let desc = program[4u] + childIndex * 8u;
   let child = selectRule(base,count,0u,program[desc + 1u],program[desc + 3u],program[desc + 6u]);
   let applied = applyNonContext(base,count,0u,program[desc + 1u],child - 1u,out,childIndex);
   output[out] = i32(terminal); output[out + 1u] = i32(count); output[out + 2u] = i32(chosen); output[out + 3u] = i32(child);
   for (var i = 0u; i < count; i++) { for (var k = 0u; k < 16u; k++) { output[out + 4u + i * 16u + k] = scratch[glyphAt(base,i) + k]; } }
   output[out + 52u] = i32(parentMatch);
   for (var i = 0u; i < 3u; i++) { output[out + 53u + i] = i32(matchPositions[i]); }
 }
 workgroupBarrier();
 return;
}
`;
        const input = {
            params: prepared.params,
            program: prepared.programWords,
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + prepared.capacity * 16) * 4,
        };
        const buffers = Object.fromEntries(
            Object.entries(input).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const source = production._source.replace("let capability = program[descBase + 2u];", "let capability = program[descBase + 2u];" + probe);
        expect(source).toContain(probe);
        const shader = createComputeShader(native.engine, {
            name: "indic-cjct-candidate-state",
            computeSource: source,
            bindings: Object.keys(input).map((name, binding) => computeStorageBufferBinding(name, { group: 0, binding, access: binding >= 4 ? "read-write" : "read" })),
        });
        const task = createComputeTask(native.engine, "indic-cjct-candidate-state");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const data = new Int32Array(await readStorageBuffer(buffers.output!));
            expect(data[0]).toBe(0);
            expect(data[2]).toBeGreaterThan(0);
            expect(data[3]).toBeGreaterThan(0);
            expect(Array.from(data.slice(52, 56))).toEqual([1, 0, 1, 2]);
            expect(data[4]).toBe(448);
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            disposeComputeShader(production);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
    it.each([4, 8, 9])("exact ddha glyph stream after %i stages", async (stages) => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        const selection = selectComputeTextRun("द्ध");
        const plan = createComputeTextPlan(face.font, selection);
        const scalars = selection._scalars;
        const prepared = prepareComputeTextShapeWords(plan, [{ scalars, glyphIds: scalars.map((cp) => mapComputeFontScalar(face.font, cp)) }]);
        const program = prepared.programWords.slice();
        program[12] = stages;
        program[program[10]! + (stages - 1) * 4 + 3] = 4;
        expect(prepared.params[7]).toBe(3);
        const input = { params: prepared.params, program, tables: prepared.tables, runs: prepared.runs, scratch: prepared.scratchBytes, output: (4 + prepared.capacity * 6) * 4 };
        const buffers = Object.fromEntries(
            Object.entries(input).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const production = createComputeTextShapeShader(native.engine, prepared.capacity, prepared.variant);
        const shader = createComputeShader(native.engine, {
            name: `indic-root-window-${stages}`,
            computeSource: indicStageSnapshot(production._source, 4),
            bindings: production._decls,
        });
        disposeComputeShader(production);
        const task = createComputeTask(native.engine, "indic-context-root-window");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const data = new Int32Array(await readStorageBuffer(buffers.output!));
            expect(data[0]).toBe(0);
            expect(data[3], "root-window snapshot reached").toBe(4);
            const actual = Array.from({ length: data[1]! }, (_, i) => [data[4 + i * 6]!, data[5 + i * 6]!]);
            expect(actual).toEqual(
                stages === 9
                    ? [[448, 0]]
                    : [
                          [73, 0],
                          [103, 1],
                          [74, 2],
                      ]
            );
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
});
