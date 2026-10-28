import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { shapeComputeText } from "../../../packages/babylon-lite/src/index";
import { compileComputeTextProgram, COMPUTE_LOOKUP_ORDERED } from "../../../packages/babylon-lite/src/text/compute-text-program";
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
import { compare, fields, fixture } from "./fixtures";
import type { Fixture } from "./fixtures";
import { syntheticFont } from "./synthetic";

describe("parallel selection with exact source-ordered commitment", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => native?.close());

    async function checkBoth(font: Fixture, texts: string[]): Promise<void> {
        const parallel = await shapeComputeText(native.engine, font.font, texts);
        parallel.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
        for (let run = 0; run < texts.length; run++) {
            const serial = await rawRecords(font, texts[run]!, (code) => code.replace("let orderedParallel = ", "let orderedParallel = false && "));
            expect(serial.slice(0, 2)).toEqual([0, parallel[run]!.length]);
            expect(serial.slice(4, 4 + parallel[run]!.length * 6)).toEqual(parallel[run]!.flatMap(fields));
        }
    }

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("retains overlapping ffi/ffl/fi precedence, marks and boundary-crossing ligatures: %s", async (file) => {
        const font = fixture(file);
        for (const length of [64, 256, 257, 600]) {
            const texts = ["ffi ffl fi fffi ffli office affinity ", "A\u0326\u0301\u0300 H\u0308\u0301 ", "fi\u0326\u0301 ffl\u0308 "].map((seed) =>
                seed.repeat(Math.ceil(length / seed.length)).slice(0, length)
            );
            if (length > 256) {
                texts.push("H".repeat(255) + "fi" + "H".repeat(length - 257));
            }
            await checkBoth(font, texts);
        }
    });

    it.each(["backtrack", "ligature-ignore-marks", "multiple", "mark-ligature", "mark-mark", "context"] as const)(
        "keeps failed selections, skipped components, insertion cursors and inherited attachment fields exact: %s",
        async (kind) => {
            const font = syntheticFont(kind);
            const plan = createComputeTextPlan(font.font, selectComputeTextRun("ABC"));
            const program = compileComputeTextProgram(plan);
            const lookup = program.lookups[program.rootOrder[0]!]!;
            const descriptor = program.words[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
            expect(program.words[descriptor + COMPUTE_LOOKUP_ORDERED]! & 3).toBe(kind === "multiple" ? 2 : 1);
            if (kind === "backtrack") {
                expect(lookup.rules.some((rule) => rule.backtrack.length > 0)).toBe(true);
            }
            for (const length of [64, 300, 600]) {
                const seed = kind === "backtrack" ? "ABAABC" : kind === "mark-mark" ? "ABBBCA" : "ABC";
                await checkBoth(font, [seed.repeat(Math.ceil(length / seed.length)).slice(0, length)]);
            }
        }
    );

    it("alternates snapshot reuse and forced re-selection after contextual writes", async () => {
        const font = syntheticFont("backtrack");
        const header = await rawRecords(font, "ABAABC".repeat(20), (code) =>
            code
                .replace("var<private> removed:bool;", "var<private> removed:bool;\nvar<private> reused:u32;\nvar<private> reselected:u32;")
                .replace("chosen = selected[", "reused++; chosen = selected[")
                .replace("} else { chosen = selectRule", "} else { reselected++; chosen = selectRule")
                // C4 advances over the complete matched context. That may no
                // longer naturally revisit the written position, so force the
                // conservative frontier in this test-only differential variant.
                .replace("frontier = max(frontier, applied.y);", "frontier = count;")
                .replace("output[out + 1] = i32(length);", "output[out + 1] = i32(length); output[out + 2] = i32(reused); output[out + 3] = i32(reselected);")
        );
        expect(header[0]).toBe(0);
        expect(header[2]).toBeGreaterThan(0);
        expect(header[3]).toBeGreaterThan(0);
        await checkBoth(font, ["ABAABC".repeat(20), "ABAABC".repeat(100)]);
    });

    it("keeps the first of two ordered rejections and the original lookup word in both scratch modes", async () => {
        const font = syntheticFont("backtrack-reject");
        for (const index of [40, 255, 256, 300]) {
            const text = "C".repeat(index) + "ABA" + "C".repeat(10) + "ABA";
            // The matched parent is executable under C4; the failing record is
            // child lookup 1, sequence index 1, not a pre-emptive parent rejection.
            expect((await rawRecords(font, text)).slice(0, 4)).toEqual([1, 0, 1, index + 2]);
            await expect(shapeComputeText(native.engine, font.font, text)).rejects.toThrow(new RegExp(`unsupported GSUB 1 matched at glyph ${index + 2}$`));
        }
    });

    it("preserves RTL ordering and stacked-mark offsets with bounded and global scratch", async () => {
        const font = fixture("NotoSansHebrew-Regular.ttf");
        await checkBoth(font, ["שָׁלוֹם ".repeat(10), "שָׁלוֹם ".repeat(60)]);
    });

    it("rejects the first contour mark-ligature anchor before a later rejection in the same root", async () => {
        const font = syntheticFont("contour-anchor");
        for (const index of [40, 255, 300]) {
            const text = "C".repeat(index) + "ABC" + "C".repeat(10) + "ABC";
            expect((await rawRecords(font, text)).slice(0, 4)).toEqual([1, 0, 0x10000, index + 1]);
        }
    });

    async function rawRecords(font: Fixture, text: string, transform?: (source: string) => string): Promise<number[]> {
        const selection = selectComputeTextRun(text);
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font.font, selection), [
            { scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => mapComputeFontScalar(font.font, cp)) },
        ]);
        const inputs = {
            params: prepared.params,
            program: prepared.programWords,
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + prepared.capacity * 6) * 4,
        };
        const buffers = Object.fromEntries(
            Object.entries(inputs).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const production = createComputeTextShapeShader(native.engine, prepared.capacity);
        const shader = transform
            ? createComputeShader(native.engine, {
                  name: "ordered-selection-variant",
                  computeSource: transform(production._source),
                  bindings: Object.keys(inputs).map((name, binding) => computeStorageBufferBinding(name, { group: 0, binding, access: binding >= 4 ? "read-write" : "read" })),
              })
            : production;
        const task = createComputeTask(native.engine, "ordered-selection-header");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            return [...new Int32Array(await readStorageBuffer(buffers.output!))];
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            if (shader !== production) {
                disposeComputeShader(production);
            }
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    }
});
