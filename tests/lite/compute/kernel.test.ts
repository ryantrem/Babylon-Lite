import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { shapeComputeText } from "../../../packages/babylon-lite/src/index";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import {
    COMPUTE_TEXT_KERNEL_SOURCE,
    checkComputeTextShapeBounds,
    prepareComputeTextShapeWords,
    createComputeTextShapeShader,
} from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { compare, fixture, fields } from "./fixtures";
import { syntheticFont } from "./synthetic";

describe("native production action modes and bounded admission", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    it.each([
        ["pair-first", 3],
        ["pair-second", 2],
    ] as const)("proves disjoint pair commits but preserves nonzero second-advance ordering: %s", async (kind, capability) => {
        const font = syntheticFont(kind);
        const program = compileComputeTextProgram(createComputeTextPlan(font.font, selectComputeTextRun("ABC")));
        expect(program.words[program.gposDescriptorOffset + 2]).toBe(capability);
        const texts = ["AB", "ABC", "AABC", "BC", "ABA", "ABC".repeat(8)];
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
    });

    it("preserves original independent blocks and activates non-leading composition/context witnesses", async () => {
        const font = fixture("Inter.ttf");
        const texts = [
            ...Array.from({ length: 100 }, (_, run) => [`Block ${String(run).padStart(2, "0")} office AVATAR`, "quick fox 0140 ffi fl To"]).flat(),
            "A\u0301",
            "a\u0306",
            "i\u0307",
            "1 < A",
            "H\u0301",
        ];
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("return bits & witnesses;");
    });

    it("finishes proven nominal/pair runs without executing the complete mutation kernel and retains contextual fallback", async () => {
        const font = fixture("Inter.ttf");
        const createModule = native.device.createShaderModule;
        let completeModules = 0;
        const modules = vi.spyOn(native.device, "createShaderModule").mockImplementation((descriptor) => {
            if (descriptor.label === "compute-text-shape-module") {
                completeModules++;
                const code = descriptor.code.replace("output[out] = 0;", "output[out] = 1; output[out + 2] = 777; output[out + 3] = 0;");
                expect(code).not.toBe(descriptor.code);
                return createModule.call(native.device, { ...descriptor, code });
            }
            return createModule.call(native.device, descriptor);
        });
        try {
            const texts = Array.from({ length: 40 }, (_, run) => [`Block ${String(run).padStart(2, "0")} office AVATAR`, "quick fox 0140 ffi fl To"]).flat();
            const shaped = await shapeComputeText(native.engine, font.font, texts);
            shaped.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
            expect(completeModules).toBe(1);
            await expect(shapeComputeText(native.engine, font.font, ["1 x 2", "A\u0301"])).rejects.toThrow(/unsupported GSUB 777/);
        } finally {
            modules.mockRestore();
        }
    });

    it("retains original root-first rejection priority across independently positioned glyphs", async () => {
        const font = syntheticFont("pair-reject");
        const program = compileComputeTextProgram(createComputeTextPlan(font.font, selectComputeTextRun("ABC")));
        expect(program.rootOrder).toHaveLength(2);
        expect(program.words[program.gposDescriptorOffset + 2]).toBe(3);
        expect(program.words[program.gposDescriptorOffset + 10]).toBe(3);
        await expect(shapeComputeText(native.engine, font.font, ["", "A", "ABC"])).rejects.toThrow(/run 2 unsupported GPOS 0 matched at glyph 1/);
        for (const text of ["A", "B", "C", "BA", "CB", "AC"]) {
            compare(font, text, await shapeComputeText(native.engine, font.font, text));
        }
    });

    it("executes complete bounded shaping when optional pair metadata would exceed the device program limit", async () => {
        const font = syntheticFont("pair-first");
        const selection = selectComputeTextRun("ABC", font.font);
        const plan = createComputeTextPlan(font.font, selection);
        const runs = [{ scalars: selection._scalars, glyphIds: Uint32Array.of(1, 2, 3) }];
        const full = prepareComputeTextShapeWords(plan, runs);
        const baseBytes = (full.programWords.length - full.simpleWordCount) * 4;
        const prepared = prepareComputeTextShapeWords(plan, runs, 0, undefined, { maxBufferSize: baseBytes, maxStorageBufferBindingSize: baseBytes });
        expect(prepared.simpleWordCount).toBe(0);
        expect(prepared.programWords.byteLength).toBe(baseBytes);
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
        const shader = createComputeTextShapeShader(native.engine, prepared.capacity);
        const task = createComputeTask(native.engine, "bounded-pair-fallback");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const result = new Int32Array(await readStorageBuffer(buffers.output!));
            expect([...result.slice(0, 4)]).toEqual([0, 3, 0, 0]);
            expect(Array.from({ length: 3 }, (_, index) => [...result.slice(4 + index * 6, 10 + index * 6)])).toEqual([
                [1, 0, 113, 0, 0, 0],
                [2, 1, 219, 0, 0, 0],
                [3, 2, 300, 0, 0, 0],
            ]);
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it.each([
        ["multiple", 11],
        ["mark-ligature", 12],
        ["adjust", 10],
        ["context", 2],
    ] as const)("actual WGSL mode %s", async (kind, mode) => {
        const font = syntheticFont(kind);
        const plan = createComputeTextPlan(font.font, selectComputeTextRun("A"));
        expect(compileComputeTextProgram(plan).lookups.flatMap((lookup) => lookup.rules.map((rule) => rule.mode))).toContain(mode);
        const texts = kind === "mark-ligature" ? ["AB", "ABB", "ACB", "B"] : kind === "context" ? ["AB", "AA", "BA", "BB"] : ["A", "AA", "AB", "BA"];
        for (const text of texts) {
            compare(font, text, await shapeComputeText(native.engine, font.font, text));
        }
        const result = (await shapeComputeText(native.engine, font.font, texts[0]!)).map(fields);
        expect(result).toEqual(
            kind === "multiple"
                ? [
                      [2, 0, 200, 0, 0, 0],
                      [3, 0, 300, 0, 0, 0],
                  ]
                : kind === "adjust"
                  ? [[1, 0, 113, 0, 0, 0]]
                  : kind === "context"
                    ? [
                          [2, 0, 200, 0, 0, 0],
                          [3, 1, 300, 0, 0, 0],
                      ]
                    : [
                          [1, 0, 100, 0, 0, 0],
                          [2, 1, 0, 0, 20, 40],
                      ]
        );
    });

    it("understood alternate is rejected when matched, never a mode-5 noop", async () => {
        const font = syntheticFont("alternate");
        await expect(shapeComputeText(native.engine, font.font, "A")).rejects.toThrow(/unsupported GSUB 0/);
        compare(font, "B", await shapeComputeText(native.engine, font.font, "B"));
    });

    it("refreshes root presence after a nested Arabic substitution activates positioning", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const result = await shapeComputeText(native.engine, font.font, "ج");
        expect(result.map(fields)).toEqual([
            [316, 0, 0, 0, 203, 2],
            [27, 0, 589, 0, 0, 0],
        ]);
        compare(font, "ج", result);
        compare(font, "جج", await shapeComputeText(native.engine, font.font, "جج"));
    });

    it("shares bounded cooperative scratch within a run and executes the complete global fallback for larger runs", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const modules = vi.spyOn(native.device, "createShaderModule");
        try {
            compare(font, "office ".repeat(7), await shapeComputeText(native.engine, font.font, "office ".repeat(7)));
            expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("var<workgroup> scratch:array<i32,512>"))).toBe(true);
            modules.mockClear();
            const text = "office ".repeat(50);
            compare(font, text, await shapeComputeText(native.engine, font.font, text));
            expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("var<storage, read_write> scratch: array<i32>"))).toBe(true);
        } finally {
            modules.mockRestore();
        }
    });

    it("sizes cooperative storage to power-of-two capacity and reuses the warm variant until growth", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const modules = vi.spyOn(native.device, "createShaderModule");
        try {
            compare(font, "H".repeat(50), await shapeComputeText(native.engine, font.font, "H".repeat(50)));
            expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("var<workgroup> scratch:array<i32,512>"))).toBe(true);
            expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("selected:array<u32,64>"))).toBe(true);
            modules.mockClear();
            compare(font, "H".repeat(10), await shapeComputeText(native.engine, font.font, "H".repeat(10)));
            expect(modules).not.toHaveBeenCalled();
            compare(font, "H".repeat(60), await shapeComputeText(native.engine, font.font, "H".repeat(60)));
            expect(modules).not.toHaveBeenCalled();
            compare(font, "H".repeat(100), await shapeComputeText(native.engine, font.font, "H".repeat(100)));
            expect(modules.mock.calls.some(([descriptor]) => descriptor.code.includes("var<workgroup> scratch:array<i32,1024>"))).toBe(true);
        } finally {
            modules.mockRestore();
        }
    });

    it("reuses warm public bind groups while repacking changed glyphs and counts", async () => {
        const font = fixture("Roboto-Regular.ttf");
        await shapeComputeText(native.engine, font.font, ["office", "H".repeat(50)]);
        const groups = vi.spyOn(native.device, "createBindGroup");
        try {
            const texts = ["affinity", "H".repeat(49)];
            const result = await shapeComputeText(native.engine, font.font, texts);
            expect(groups).not.toHaveBeenCalled();
            result.forEach((run, index) => compare(font, texts[index]!, run));
        } finally {
            groups.mockRestore();
        }
    });

    it("submits public shaping and its completion copy together without publishing mapped views", async () => {
        const font = fixture("Roboto-Regular.ttf");
        await shapeComputeText(native.engine, font.font, ["office", "HH"]);
        const submissions = vi.spyOn(native.device.queue, "submit");
        try {
            const texts = ["affinity", "H"];
            const result = await shapeComputeText(native.engine, font.font, texts);
            expect(submissions).toHaveBeenCalledTimes(1);
            result.forEach((run, index) => compare(font, texts[index]!, run));
            const snapshot = result.map((run) => run.map(fields));
            await shapeComputeText(native.engine, font.font, ["office", "HH"]);
            expect(result.map((run) => run.map(fields))).toEqual(snapshot);
        } finally {
            submissions.mockRestore();
        }
    });

    it("materializes the published glyph count without repeated output-array growth", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const text = "office affinity ".repeat(3);
        await shapeComputeText(native.engine, font.font, text);
        const push = Array.prototype.push;
        let glyphPushes = 0;
        Array.prototype.push = function <T>(this: T[], ...items: T[]): number {
            for (const item of items) {
                if (item && typeof item === "object" && "glyphId" in item && "cluster" in item) {
                    glyphPushes++;
                }
            }
            return push.apply(this, items);
        };
        let result;
        try {
            result = await shapeComputeText(native.engine, font.font, text);
        } finally {
            Array.prototype.push = push;
        }
        compare(font, text, result);
        expect(glyphPushes).toBe(0);
    });

    it("reuses the loaded executor before warm input submission without another module-resolution yield", async () => {
        const font = fixture("Roboto-Regular.ttf");
        await shapeComputeText(native.engine, font.font, ["office", "H".repeat(50)]);
        let crossedMicrotask = false;
        queueMicrotask(() => {
            crossedMicrotask = true;
        });
        const queue = native.device.queue;
        const writeBuffer = queue.writeBuffer;
        const yields: boolean[] = [];
        const writes = vi.spyOn(queue, "writeBuffer").mockImplementation((...args) => {
            yields.push(crossedMicrotask);
            writeBuffer.apply(queue, args);
        });
        const texts = ["affinity", "H"];
        try {
            const result = await shapeComputeText(native.engine, font.font, texts);
            expect(yields.length).toBeGreaterThanOrEqual(2);
            expect(yields.every((yielded) => !yielded)).toBe(true);
            result.forEach((run, index) => compare(font, texts[index]!, run));
        } finally {
            writes.mockRestore();
        }
    });

    it("submits an already prepared owned schedule without an extra asynchronous preparation turn", async () => {
        const font = fixture("Roboto-Regular.ttf");
        await shapeComputeText(native.engine, font.font, ["office", "HH"]);
        let crossedMicrotask = false;
        queueMicrotask(() => {
            crossedMicrotask = true;
        });
        const queue = native.device.queue;
        const submit = queue.submit;
        const turns: boolean[] = [];
        const submissions = vi.spyOn(queue, "submit").mockImplementation((...args) => {
            turns.push(crossedMicrotask);
            submit.apply(queue, args);
        });
        try {
            const texts = ["affinity", "H"];
            const result = await shapeComputeText(native.engine, font.font, texts);
            expect(turns).toEqual([false]);
            result.forEach((run, index) => compare(font, texts[index]!, run));
        } finally {
            submissions.mockRestore();
        }
        const grown = ["office ".repeat(40), "H".repeat(300)];
        const result = await shapeComputeText(native.engine, font.font, grown);
        result.forEach((run, index) => compare(font, grown[index]!, run));
        const shorter = ["affinity", "H"];
        const next = await shapeComputeText(native.engine, font.font, shorter);
        next.forEach((run, index) => compare(font, shorter[index]!, run));
    });

    it("derives initial scalar-index clusters from compact bulk input including supplementary and empty runs", async () => {
        const font = fixture("NotoSansGothic-Regular.ttf");
        const texts = ["𐌰𐌱", "𐌲", ""];
        const plan = createComputeTextPlan(font.font, selectComputeTextRun(texts[0]!));
        const input = texts.map((text) => {
            const selection = selectComputeTextRun(text);
            return { scalars: selection._scalars, glyphIds: selection._scalars.map((scalar) => font.oracle.glyphId(scalar)) };
        });
        const packed = prepareComputeTextShapeWords(plan, input);
        expect(packed.runs.length).toBe(9);
        expect([...packed.runs]).toEqual([3, 6, 8, 2, input[0]!.glyphIds[0], input[0]!.glyphIds[1], 1, input[1]!.glyphIds[0], 0]);
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((run, index) => compare(font, texts[index]!, run));
        expect(result[0]?.map((glyph) => glyph.cluster)).toEqual([0, 1]);
        expect(result[2]).toEqual([]);
    });

    it("allocates compact mutable input capacity and reuses it for shorter and changed revisions", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const buffers = vi.spyOn(native.device, "createBuffer");
        const texts = ["office".repeat(10), "H".repeat(50)];
        try {
            const result = await shapeComputeText(native.engine, font.font, texts);
            result.forEach((run, index) => compare(font, texts[index]!, run));
            const inputs = buffers.mock.calls.filter(([descriptor]) => descriptor.label === "compute-text-runs");
            expect(inputs).toHaveLength(1);
            expect(inputs[0]![0].size).toBe(texts.length * (60 + 2) * 4);
            buffers.mockClear();
            const changed = ["H".repeat(59), "affinity"];
            const next = await shapeComputeText(native.engine, font.font, changed);
            expect(buffers.mock.calls.filter(([descriptor]) => descriptor.label === "compute-text-runs")).toHaveLength(0);
            next.forEach((run, index) => compare(font, changed[index]!, run));
        } finally {
            buffers.mockRestore();
        }
    });

    it("GPOS point-anchor rejection identifies its table and original run after empty-run filtering", async () => {
        const font = syntheticFont("point-anchor");
        await expect(shapeComputeText(native.engine, font.font, ["", "B", "AB"])).rejects.toThrow(/run 2 unsupported GPOS 0/);
        compare(font, "B", await shapeComputeText(native.engine, font.font, "B"));
    });

    it("storage/buffer/workgroup exact boundaries use real device limits without near-OOM allocation", async () => {
        const font = fixture("NotoSansGothic-Regular.ttf");
        const plan = createComputeTextPlan(font.font, selectComputeTextRun("𐌰"));
        const prepared = prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(0x10330), glyphIds: Uint32Array.of(font.oracle.glyphId(0x10330)) }]);
        const limits = native.device.limits;
        checkComputeTextShapeBounds(limits, { ...prepared, scratchBytes: limits.maxStorageBufferBindingSize });
        expect(() => checkComputeTextShapeBounds(limits, { ...prepared, scratchBytes: limits.maxStorageBufferBindingSize + 4 })).toThrow(/maxStorageBufferBindingSize/);
        expect(() => checkComputeTextShapeBounds(limits, { ...prepared, scratchBytes: limits.maxBufferSize + 4 })).toThrow(/maxBufferSize/);
        const params = prepared.params.slice();
        params[0] = limits.maxComputeWorkgroupsPerDimension;
        checkComputeTextShapeBounds(limits, { ...prepared, params });
        params[0]++;
        expect(() => checkComputeTextShapeBounds(limits, { ...prepared, params })).toThrow(/maxComputeWorkgroupsPerDimension/);
        await expect(
            shapeComputeText(
                native.engine,
                font.font,
                Array.from({ length: params[0]! }, () => "𐌰")
            )
        ).rejects.toThrow(/maxComputeWorkgroupsPerDimension/);
        expect(() => prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(0x10330), glyphIds: Uint32Array.of(1) }], 0, 0)).toThrow(/capacity/);
        compare(font, "𐌰", await shapeComputeText(native.engine, font.font, "𐌰"));
    });

    it("shape overflow status 2 executes production WGSL (internal capacity injection)", async () => {
        const font = syntheticFont("multiple");
        const plan = createComputeTextPlan(font.font, selectComputeTextRun("A"));
        const prepared = prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(0x41), glyphIds: Uint32Array.of(1) }]);
        prepared.params[1] = 1;
        const inputs = {
            params: prepared.params,
            program: prepared.programWords,
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + 6) * 4,
        };
        const buffers = Object.fromEntries(
            Object.entries(inputs).map(([name, bytes]) => [
                name,
                createStorageBuffer(native.engine, bytes, { writable: name === "scratch" || name === "output", label: `kernel-test-${name}` }),
            ])
        );
        const shader = createComputeShader(native.engine, {
            computeSource: COMPUTE_TEXT_KERNEL_SOURCE,
            bindings: Object.keys(inputs).map((name, binding) => computeStorageBufferBinding(name, { group: 0, binding, access: binding >= 4 ? "read-write" : "read" })),
        });
        try {
            const task = createComputeTask(native.engine, "kernel-failure-fixture");
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const header = new Int32Array(await readStorageBuffer(buffers.output!)).slice(0, 4);
            expect([...header]).toEqual([2, 0, 0, 0]);
        } finally {
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
});
