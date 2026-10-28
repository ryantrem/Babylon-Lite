import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import {
    createComputeTextData,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    shapeComputeText,
    updateComputeTextData,
} from "../../../packages/babylon-lite/src/index";
import { createComputeTextShaper, disposeComputeTextShaper } from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import {
    executeComputeTextShape,
    kernelSource,
    prepareComputeTextShapeWords,
    createComputeTextShapeShader,
    createShapeSchedule,
} from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeShader, disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { computeStorageBufferBinding } from "../../../packages/babylon-lite/src/compute/compute-storage-buffer-binding";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fields, fixture, oracleFields } from "./fixtures";
import type { FontFile } from "./fixtures";
import { syntheticFont } from "./synthetic";
import type { SyntheticKind } from "./synthetic";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { reviewedRecords } from "./references/reviewed-records";

interface Witness {
    id: string;
    kind: SyntheticKind;
    text: string;
    expected: string;
    records?: number[][];
}
const witnesses = (JSON.parse(readFileSync("tests/lite/compute/references/synthetic-harfbuzz-11.0.1.json", "utf8")) as { cases: Witness[] }).cases;
const impactRows = reviewedRecords();
const fontPositionWitnesses = impactRows.filter((entry) => ["N-proof", "G0", "A-attach"].includes(entry.category));

describe("native corrected shared-core semantics", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    it("cleared records explain existing assertion failures without migrating those assertions", async () => {
        const inputs = [
            ["Inter.ttf", "A\u0326 ".repeat(85)],
            ["Roboto-Regular.ttf", "A\u0326 ".repeat(85)],
            ["NotoSansHebrew-Regular.ttf", "שָׁלוֹם ".repeat(80)],
            ["synthetic:pair-second", "ABC".repeat(100)],
            ["synthetic:mark-ligature", "ABC".repeat(100)],
            ["NotoSansHebrew-Regular.ttf", "שלום ".repeat(30)],
            ["Inter.ttf", "a ".repeat(60) + "<-" + " b".repeat(100)],
        ] as const;
        for (const [file, text] of inputs) {
            const entry = impactRows.find((entry) => entry.font === file && entry.text === text)!;
            expect(entry, `${file}: independently cleared input`).toBeDefined();
            expect(entry.category).not.toBe("X");
            const font = file.startsWith("synthetic:") ? syntheticFont(file.slice("synthetic:".length) as SyntheticKind) : fixture(file as FontFile);
            expect((await shapeComputeText(native.engine, font.font, text)).map(fields), entry.id).toEqual(entry.records);
        }
    });

    it("serial cardinality changes do not leak into subsequent unmatched positions", async () => {
        const font = syntheticFont("multiple");
        const text = "ABCABC";
        const expected = oracleFields(font, text);
        expect(expected).toHaveLength(8);
        expect((await shapeComputeText(native.engine, font.font, text)).map(fields)).toEqual(expected);
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "serial-multiple-delta");
        try {
            await (
                await updateComputeTextData(data, [text])
            ).completed;
            const actual = new Int32Array(await readStorageBuffer(data._shaped));
            expect(Array.from(actual.slice(0, 2))).toEqual([0, expected.length]);
            expect(Array.from(actual.slice(4, 4 + expected.length * 6))).toEqual(expected.flat());
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("a generic masked 16-word variant without a script prepass still initializes root presence", async () => {
        const entry = witnesses.find((entry) => entry.id.startsWith("c4-nested-ligature"))!;
        const face = syntheticFont(entry.kind),
            scalars = Uint32Array.from([...entry.text].map((cp) => cp.codePointAt(0)!));
        const p = prepareComputeTextShapeWords(createComputeTextPlan(face.font, selectComputeTextRun("ABC")), [
            {
                scalars,
                glyphIds: scalars.map((cp) => face.oracle.glyphId(cp)),
            },
        ]);
        const shader = createComputeTextShapeShader(native.engine, p.capacity, {
            id: "indic",
            glyphWords: 16,
            planes: 1,
            masked: true,
            fragments: { glyphMask: "fn glyphMask(at:u32)->u32 { return 0xffffffffu; }" },
        });
        const buffers = {
            params: createStorageBuffer(native.engine, p.params),
            program: createStorageBuffer(native.engine, p.programWords),
            tables: createStorageBuffer(native.engine, p.tables),
            runs: createStorageBuffer(native.engine, p.runs),
            scratch: createStorageBuffer(native.engine, p.capacity * 16 * 4, { writable: true }),
            output: createStorageBuffer(native.engine, (4 + p.capacity * 6) * 4, { writable: true }),
        };
        const schedule = createShapeSchedule(native.engine, shader, buffers, 1);
        try {
            await prepareComputeTask(schedule.task);
            submitComputeTasks([schedule.task]);
            const raw = new Int32Array(await readStorageBuffer(buffers.output));
            expect([...raw.slice(0, 2)]).toEqual([0, entry.records!.length]);
            expect([...raw.slice(4, 4 + entry.records!.length * 6)]).toEqual(entry.records!.flat());
        } finally {
            schedule.task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it("word-7 actual GPU accessors preserve distances above 65535 through the 29-bit boundary", async () => {
        const cases = [
            [0, 0],
            [1, 1],
            [3, 65535],
            [4, 65536],
            [2, 70002],
            [3, 1 << 20],
            [4, 0x1fffffff],
        ];
        const shader = createComputeShader(native.engine, {
            name: "compute-text-record-properties-test",
            entryPoint: "packingBoundary",
            computeSource:
                kernelSource(0) +
                `
@group(0) @binding(6) var<storage, read> packingInputs:array<u32>;
@compute @workgroup_size(1)
fn packingBoundary() {
  let count = arrayLength(&packingInputs) / 2u;
  for (var i = 0u; i < count; i++) {
    let at = glyphAt(0u, i);
    scratch[at + 7u] = recordProperties(packingInputs[i * 2u], packingInputs[i * 2u + 1u]);
    let copy = glyphAt(0u, i + count);
    scratch[copy + 7u] = scratch[at + 7u];
    output[i * 4u] = scratch[copy + 7u];
    output[i * 4u + 1u] = i32(recordClass(copy));
    output[i * 4u + 2u] = i32(attachDistance(copy));
    scratch[copy + 7u] = recordProperties(recordClass(copy), 0u);
    output[i * 4u + 3u] = scratch[copy + 7u];
  }
}`,
            bindings: [
                computeStorageBufferBinding("scratch", { group: 0, binding: 4, access: "read-write" }),
                computeStorageBufferBinding("output", { group: 0, binding: 5, access: "read-write" }),
                computeStorageBufferBinding("packingInputs", { group: 0, binding: 6 }),
            ],
        });
        const buffers = {
            scratch: createStorageBuffer(native.engine, cases.length * 2 * 8 * 4, { writable: true }),
            output: createStorageBuffer(native.engine, cases.length * 4 * 4, { writable: true }),
            packingInputs: createStorageBuffer(native.engine, Uint32Array.from(cases.flat())),
        };
        const task = createComputeTask(native.engine, "compute-text-record-properties-test");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const actual = new Uint32Array(await readStorageBuffer(buffers.output));
            expect(Array.from(actual)).toEqual(cases.flatMap(([cls, distance]) => [Number((BigInt(distance!) << 3n) | BigInt(cls!)), cls, distance, cls]));
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it.each(witnesses)("$id: exact GPU readback/resident fields and preserved public admission", async (entry) => {
        const font = syntheticFont(entry.kind);
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, entry.id);
        try {
            if (/\uE000|\uE001/u.test(entry.text)) {
                // The independent fixture uses private-use scalars, rejected by public
                // admission. Preserve that boundary; exercise unchanged font bytes and
                // records on both real GPU execution paths below, not a test executor.
                await expect(shapeComputeText(native.engine, font.font, entry.text)).rejects.toThrow(/unknown script/);
                await expect(updateComputeTextData(data, [entry.text])).rejects.toThrow(/unknown script/);
                const plan = createComputeTextPlan(font.font, selectComputeTextRun("ABC"));
                const scalars = Uint32Array.from([...entry.text].map((cp) => cp.codePointAt(0)!));
                const runs = [{ scalars, glyphIds: scalars.map((cp) => font.oracle.glyphId(cp)) }];
                const shaper = createComputeTextShaper(native.engine);
                const token = {};
                const prepared = prepareComputeTextShapeWords(plan, runs);
                const buffers = {
                    shaped: createStorageBuffer(native.engine, (4 + prepared.capacity * 6) * 4, { writable: true }),
                    instances: createStorageBuffer(native.engine, Math.max(12, prepared.capacity * 12), { writable: true }),
                    indirect: createStorageBuffer(native.engine, Uint32Array.of(6, 0, 0, 0), { writable: true }),
                    slots: createStorageBuffer(native.engine, new Uint32Array(font.font.glyphCount).fill(0xfffffffe)),
                    status: createStorageBuffer(native.engine, 4, { writable: true }),
                };
                try {
                    const shaped = await executeComputeTextShape(shaper, plan, runs, shaper._generation);
                    expect(shaped[0]!.map(fields)).toEqual(entry.records);
                    const submission = await executeResidentComputeText(
                        token,
                        native.engine,
                        [{ plan, runs, outputRun: 0 }],
                        buffers,
                        prepared.capacity,
                        1,
                        32,
                        font.font.unitsPerEm,
                        font.font.glyphCount,
                        () => undefined
                    );
                    expect(await submission.status).toBe(0);
                    const resident = new Int32Array(await readStorageBuffer(buffers.shaped));
                    expect(Array.from(resident.slice(0, 2))).toEqual([0, entry.records!.length]);
                    expect(Array.from(resident.slice(4, 4 + entry.records!.length * 6))).toEqual(entry.records!.flat());
                } finally {
                    disposeComputeTextShaper(shaper);
                    releaseResidentComputeText(token);
                    Object.values(buffers).forEach(disposeStorageBuffer);
                }
                return;
            }
            if (entry.expected !== "positive") {
                await expect(shapeComputeText(native.engine, font.font, entry.text)).rejects.toThrow(/unsupported/);
                await expect(
                    (async () => {
                        await (
                            await updateComputeTextData(data, [entry.text])
                        ).completed;
                    })()
                ).rejects.toThrow(entry.kind === "reverse-chain" ? /closure cannot see GSUB lookup 0.*reverse chaining/ : /unsupported/);
                expect(Array.from(new Uint32Array(await readStorageBuffer(data._indirect)))).toEqual([6, 0, 0, 0]);
                return;
            }
            expect((await shapeComputeText(native.engine, font.font, entry.text)).map(fields)).toEqual(entry.records);
            await (
                await updateComputeTextData(data, [entry.text])
            ).completed;
            const selection = selectComputeTextRun(entry.text);
            const plan = createComputeTextPlan(font.font, selection);
            const prepared = prepareComputeTextShapeWords(plan, [
                { scalars: selection._scalars, glyphIds: Uint32Array.from([...entry.text].map((cp) => font.oracle.glyphId(cp.codePointAt(0)!))) },
            ]);
            const shaped = new Int32Array(await readStorageBuffer(data._shaped));
            expect(Array.from(shaped.slice(0, 2))).toEqual([0, entry.records!.length]);
            expect(Array.from(shaped.slice(4, 4 + entry.records!.length * 6))).toEqual(entry.records!.flat());
            expect(shaped.length).toBeGreaterThanOrEqual(4 + prepared.capacity * 6);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it.each(fontPositionWitnesses)("$category $id: exact public/resident widths and attachment geometry", async (entry) => {
        const font = syntheticFont(entry.font.slice("synthetic:".length) as SyntheticKind);
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, entry.id);
        data._disableSegmentation = true;
        try {
            expect((await shapeComputeText(native.engine, font.font, entry.text)).map(fields)).toEqual(entry.records);
            await (
                await updateComputeTextData(data, [entry.text])
            ).completed;
            const actual = new Int32Array(await readStorageBuffer(data._shaped));
            expect(Array.from(actual.slice(0, 2))).toEqual([0, entry.records.length]);
            expect(Array.from(actual.slice(4, 4 + entry.records.length * 6))).toEqual(entry.records.flat());
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });
});
