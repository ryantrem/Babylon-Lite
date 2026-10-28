import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { fixture } from "./fixtures";
import { nativeHarness, assertNativeClean } from "./native";
import type { NativeHarness } from "./native";
import { createComputeTextPlan, ensureComputeTextProfile } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { prepareComputeTextShapeWords, createComputeTextShapeShader, computeTextVariantTier } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { indicStageSnapshot } from "./indic-stage-snapshot";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { createComputeTask, addComputeDispatch, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";

interface Entry {
    id: string;
    text: string;
    expected: string;
    profile: string;
    records: number[][];
    pauseSnapshots?: { message: string; glyphs: number[][] }[];
}
const corpus = (JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: Entry[] }).cases;
const projected = (
    JSON.parse(readFileSync("tests/lite/compute/references/deva-projected-pauses.json", "utf8")) as {
        cases: { id: string; passed: boolean; pauseSnapshots: { message: string; glyphs: number[][]; relation: string }[] }[];
    }
).cases;

describe("actual shared-core Indic preprocessing and reorder pauses", () => {
    let native: NativeHarness;
    const shaders = new Map<string, ReturnType<typeof createComputeTextShapeShader>>();
    beforeAll(async () => {
        await ensureComputeTextProfile("indic");
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => {
        for (const shader of shaders.values()) {
            disposeComputeShader(shader);
        }
        await native?.close();
    });
    const checks = corpus
        .filter((entry) => entry.expected === "positive" && entry.profile === "deva")
        .flatMap((entry) => [
            { entry, id: entry.id, stop: 1, message: "start reordering indic initial" },
            { entry, id: entry.id, stop: 2, message: "end reordering indic initial" },
            { entry, id: entry.id, stop: 3, message: "end reordering indic final" },
        ]);
    it.each(checks)("$id / pause $stop: exact independent glyphs/clusters", async ({ entry, stop, message }) => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("क"));
        const proof = projected.find((item) => item.id === entry.id);
        if (proof) {
            expect(proof.passed).toBe(true);
        }
        const snapshot = (proof?.pauseSnapshots ?? entry.pauseSnapshots)?.find((item) => item.message === message);
        expect(snapshot, entry.id).toBeDefined();
        const selection = selectComputeTextRun(entry.text);
        const scalars = selection._scalars;
        const prepared = prepareComputeTextShapeWords(plan, [{ scalars, glyphIds: scalars.map((cp) => mapComputeFontScalar(face.font, cp)) }]);
        expect(prepared.params[7]).toBe(3);
        const input = {
            params: prepared.params,
            program: prepared.programWords,
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + prepared.capacity * 6) * 4,
        };
        const buffers = Object.fromEntries(
            Object.entries(input).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const tier = computeTextVariantTier(prepared.capacity, prepared.variant, native.device.limits.maxComputeWorkgroupStorageSize);
        const key = `${tier}:${stop}`;
        let shader = shaders.get(key);
        if (!shader) {
            const production = createComputeTextShapeShader(native.engine, prepared.capacity, prepared.variant);
            shader = createComputeShader(native.engine, {
                name: `indic-snapshot-${key}`,
                computeSource: indicStageSnapshot(production._source, stop),
                bindings: production._decls,
            });
            disposeComputeShader(production);
            shaders.set(key, shader);
        }
        const task = createComputeTask(native.engine, "indic-pause-diagnostic");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const data = new Int32Array(await readStorageBuffer(buffers.output!));
            expect(data[0], entry.id).toBe(0);
            expect(data[3], "requested pause was reached").toBe(stop);
            const actual = Array.from({ length: data[1]! }, (_, i) => [data[4 + i * 6]!, data[5 + i * 6]!]);
            expect(actual, `${entry.id} ${JSON.stringify(entry.text)}`).toEqual(snapshot!.glyphs);
        } finally {
            task.dispose();
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
});
