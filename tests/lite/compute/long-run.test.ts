import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
    createComputeTextData,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    shapeComputeText,
    updateComputeTextData,
} from "../../../packages/babylon-lite/src/index";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextShapeShader, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { createComputeBindingSet } from "../../../packages/babylon-lite/src/compute/compute-bindings";
import { createComputeDispatch } from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import { disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../../packages/babylon-lite/src/compute/compute-task";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { compare, CYRILLIC, fixture, GREEK, oracleFields } from "./fixtures";
import { syntheticFont } from "./synthetic";

describe("length-independent production shaping", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => native?.close());

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("finishes proven long nominal runs without scratch fallback: %s", async (file) => {
        const font = fixture(file);
        const original = native.device.createShaderModule;
        const modules = vi.spyOn(native.device, "createShaderModule").mockImplementation((descriptor) => {
            if (descriptor.label === "compute-text-shape-module") {
                const code = descriptor.code.replace("output[out] = 0;", "output[out] = 1; output[out + 2] = 777; output[out + 3] = 0;");
                expect(code).not.toBe(descriptor.code);
                return original.call(native.device, { ...descriptor, code });
            }
            return original.call(native.device, descriptor);
        });
        try {
            const texts = ["HH ".repeat(100), "AV To A", "", "H".repeat(1025)];
            const result = await shapeComputeText(native.engine, font.font, texts);
            result.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
        } finally {
            modules.mockRestore();
        }
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("matches all six oracle fields across scratch/tile boundaries: %s", async (file) => {
        const font = fixture(file);
        for (const length of [255, 256, 257, 300, 600, 1025]) {
            const texts = ["HH ", "AV To ", "office fi ", "A\u0326 ", GREEK, CYRILLIC].map((seed) => seed.repeat(Math.ceil(length / seed.length)).slice(0, length));
            texts.push("", ...Array.from({ length: 100 }, (_, run) => (run === 37 ? "HH ".repeat(100) : ["HH HH H", "AV To A", "office ", " A fi H"][run % 4]!)));
            const result = await shapeComputeText(native.engine, font.font, texts);
            result.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
        }
    });

    it("preserves long RTL clusters and ordered mark attachment", async () => {
        const font = fixture("NotoSansHebrew-Regular.ttf");
        const texts = ["אבגד".repeat(150), "שָׁלוֹם ".repeat(80)];
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
    });

    it.each(["pair-first", "pair-second", "multiple", "context", "transitive", "mark-ligature"] as const)(
        "retains cooperative/ordered actions and the complete failed-proof candidate mask: %s",
        async (kind) => {
            const font = syntheticFont(kind);
            const texts = ["ABC".repeat(100), "C".repeat(257) + "AB", "AB".repeat(512)];
            const result = await shapeComputeText(native.engine, font.font, texts);
            result.forEach((glyphs, run) => compare(font, texts[run]!, glyphs));
        }
    );

    it("reports root-first, then full source-index rejection beyond the first tile", async () => {
        const font = syntheticFont("pair-reject");
        for (const index of [256, 257, 300, 600, 1025]) {
            const text = "A".repeat(index) + "BC";
            await expect(shapeComputeText(native.engine, font.font, ["", "A", text])).rejects.toThrow(new RegExp(`run 2 unsupported GPOS 0 matched at glyph ${index}$`));
        }
        const single = syntheticFont("alternate");
        await expect(shapeComputeText(native.engine, single.font, "C".repeat(600) + "AA")).rejects.toThrow(/unsupported GSUB 0 matched at glyph 600$/);
    });

    it("tiled pair fallback preserves rejection priority when compact simple metadata is unavailable", async () => {
        const font = syntheticFont("pair-reject");
        const text = "A".repeat(600) + "BC";
        const selection = selectComputeTextRun(text);
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font.font, selection), [
            { scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => mapComputeFontScalar(font.font, cp)) },
        ]);
        const inputs = {
            params: prepared.params,
            program: prepared.programWords.slice(0, -prepared.simpleWordCount),
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + prepared.capacity * 6) * 4,
        };
        const buffers = Object.fromEntries(
            Object.entries(inputs).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const shader = createComputeTextShapeShader(native.engine, prepared.capacity);
        const task = createComputeTask(native.engine, "long-pair-rejection");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            const result = new Int32Array(await readStorageBuffer(buffers.output!));
            expect([...result.slice(0, 4)]).toEqual([1, 0, 0x10000, 600]);
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it("keeps long multiple-substitution overflow header exact", async () => {
        const font = syntheticFont("multiple");
        const selection = selectComputeTextRun("C".repeat(300) + "A");
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font.font, selection), [
            { scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => mapComputeFontScalar(font.font, cp)) },
        ]);
        prepared.params[1] = selection.scalarCount;
        const inputs = {
            params: prepared.params,
            program: prepared.programWords,
            tables: prepared.tables,
            runs: prepared.runs,
            scratch: prepared.scratchBytes,
            output: (4 + selection.scalarCount * 6) * 4,
        };
        const buffers = Object.fromEntries(
            Object.entries(inputs).map(([name, source]) => [name, createStorageBuffer(native.engine, source, { writable: name === "scratch" || name === "output" })])
        );
        const shader = createComputeTextShapeShader(native.engine, selection.scalarCount);
        const task = createComputeTask(native.engine, "long-overflow");
        try {
            addComputeDispatch(task, createComputeDispatch(shader, createComputeBindingSet(shader, buffers), { size: { x: 1 } }));
            task.record();
            await prepareComputeTask(task);
            submitComputeTasks([task]);
            expect([...new Int32Array(await readStorageBuffer(buffers.output!)).slice(0, 4)]).toEqual([2, 0, 0, 0]);
        } finally {
            task.dispose();
            disposeComputeShader(shader);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("resident shared owner grows, shrinks, reuses and disposes: %s", async (file) => {
        const font = fixture(file);
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, file, { fontSizePx: 32 });
        try {
            for (const length of [256, 257, 7, 600, 255, 1025, 7]) {
                const texts = Array.from({ length: 257 }, (_, run) => (run === 73 ? "H".repeat(length) : run % 2 ? "AV To A" : "A\u0326"));
                texts[3] = GREEK;
                texts[17] = CYRILLIC;
                texts[251] = "";
                const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
                await updateComputeTextData(data, texts);
                expect(maps.mock.calls.map((args) => args.slice(0, 3))).toEqual([[GPUMapMode.READ, 0, 4]]);
                maps.mockRestore();
                const records = new Int32Array(await readStorageBuffer(data._shaped));
                const capacity = Math.max(
                    1,
                    ...texts.map((text) => {
                        const selection = selectComputeTextRun(text);
                        return prepareComputeTextShapeWords(createComputeTextPlan(font.font, selection), [
                            { scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => mapComputeFontScalar(font.font, cp)) },
                        ]).capacity;
                    })
                );
                const stride = 4 + capacity * 6;
                const instances = new DataView(await readStorageBuffer(data._instances));
                const atlas = storage._curveSets.get(file)!._atlas!;
                const scale = Math.fround(32 / font.font.unitsPerEm);
                let visible = 0;
                texts.forEach((text, run) => {
                    const expected = oracleFields(font, text);
                    expect([...records.slice(run * stride, run * stride + 2)]).toEqual([0, expected.length]);
                    let penX = 0;
                    let penY = 0;
                    expected.forEach((record, index) => {
                        expect([...records.slice(run * stride + 4 + index * 6, run * stride + 10 + index * 6)]).toEqual(record);
                        const slot = atlas._glyphSlots.get(record[0]!);
                        if (slot) {
                            const at = visible++ * 12;
                            expect(instances.getFloat32(at, true)).toBe(Math.fround(Math.fround(penX + record[4]!) * scale));
                            expect(instances.getFloat32(at + 4, true)).toBe(Math.fround(Math.fround(Math.fround(penY + record[5]!) * scale) - Math.fround(run * 32)));
                            expect(instances.getUint32(at + 8, true)).toBe(slot._index);
                        }
                        penX += record[2]!;
                        penY += record[3]!;
                    });
                });
                expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, visible, 0, 0]);
            }
        } finally {
            disposeComputeTextData(data);
            expect(data.status).toBe("disposed");
            disposeGlyphStorage(storage);
        }
    });
});
