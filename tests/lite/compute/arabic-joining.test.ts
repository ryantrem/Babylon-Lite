import { readFileSync } from "node:fs";
import { deepStrictEqual } from "node:assert";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import {
    createComputeTextData,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    shapeComputeText,
    updateComputeTextData,
} from "../../../packages/babylon-lite/src/index";
import { createComputeTextPlan, ensureComputeTextProfile, installComputeTextProfileLoader } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { installComputeTextArabicProfile } from "../../../packages/babylon-lite/src/text/compute-text-arabic";
import { createComputeTextShaper, disposeComputeTextShaper, shapeComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { computeTextShapeCapacity } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fields, fixture, oracleFields } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { reviewedRecords } from "./references/reviewed-records";
import { syntheticFont } from "./synthetic";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";

interface ArabicCase {
    id: string;
    text: string;
    records: number[][];
    count: number;
}
const cases = (JSON.parse(readFileSync("tests/lite/compute/references/arabic-harfbuzz-11.0.1.json", "utf8")) as { cases: ArabicCase[] }).cases;

describe("native Arabic joining and mark order on the shared core", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    it("C9/C10 witnesses: public exact count and six fields", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const actual = await shapeComputeText(
            native.engine,
            font.font,
            cases.map((entry) => entry.text)
        );
        expect(actual).toHaveLength(cases.length);
        actual.forEach((glyphs, i) => {
            expect(glyphs, cases[i]!.id).toHaveLength(cases[i]!.count);
            deepStrictEqual(glyphs.map(fields), cases[i]!.records, cases[i]!.id);
        });
    });

    it("C9/C10 witnesses: resident exact count and six fields", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "arabic-reference");
        try {
            await (
                await updateComputeTextData(
                    data,
                    cases.map((entry) => entry.text)
                )
            ).completed;
            await ensureComputeTextProfile("arabic");
            const plan = createComputeTextPlan(font.font, selectComputeTextRun("كتب"));
            const capacity = computeTextShapeCapacity(
                plan,
                cases.map((entry) => ({ scalars: Uint32Array.from([...entry.text].map((cp) => cp.codePointAt(0)!)) }))
            );
            const stride = 4 + capacity * 6;
            const actual = new Int32Array(await readStorageBuffer(data._shaped));
            cases.forEach((entry, i) => {
                const origin = i * stride;
                expect(Array.from(actual.slice(origin, origin + 2)), entry.id).toEqual([0, entry.records.length]);
                deepStrictEqual(Array.from(actual.slice(origin + 4, origin + 4 + entry.records.length * 6)), entry.records.flat(), entry.id);
            });

            expect(data._shaped._data).toBeNull();
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("every cleared existing Arabic row has exact count and all six fields", async () => {
        await ensureComputeTextProfile("arabic");
        const font = fixture("NotoSansArabic-Regular.ttf");
        const rows = reviewedRecords().filter((entry) => entry.script === "arab");
        expect(rows).toHaveLength(129);
        const plan = createComputeTextPlan(font.font, selectComputeTextRun("كتب"));
        const shaper = createComputeTextShaper(native.engine);
        try {
            const actual = await shapeComputeTextPlan(
                shaper,
                plan,
                rows.map((entry) => entry.text)
            );
            actual.forEach((run, i) => deepStrictEqual(run.glyphs.map(fields), rows[i]!.records, rows[i]!.id));
        } finally {
            disposeComputeTextShaper(shaper);
        }
    });

    it("D10 unsupported-font rejection publishes nothing and unmapped errors keep precedence", async () => {
        const font = syntheticFont("arabic-no-joining");
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "arabic-no-joining");
        try {
            await expect(shapeComputeText(native.engine, font.font, "ب")).rejects.toThrow("font has no Arabic joining features");
            await expect(updateComputeTextData(data, ["ب"])).rejects.toThrow("fallback is unsupported");
            expect(Array.from(new Uint32Array(await readStorageBuffer(data._indirect)))).toEqual([6, 0, 0, 0]);
            await expect(shapeComputeText(native.engine, font.font, "ا")).rejects.toThrow(/unmapped|does not map/i);
            await expect(updateComputeTextData(data, ["ا"])).rejects.toThrow(/unmapped|does not map/i);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("Arabic joiners retain explicit admission errors and suppress old resident drawing", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "arabic-joiner");
        try {
            await (
                await updateComputeTextData(data, ["كتب"])
            ).completed;
            for (const joiner of ["\u200c", "\u200d"]) {
                await expect(shapeComputeText(native.engine, font.font, `ب${joiner}`)).rejects.toThrow(/joining control at scalar 1/);
                await expect(updateComputeTextData(data, [`ب${joiner}`])).rejects.toThrow(/joining control at scalar 1/);
                expect(Array.from(new Uint32Array(await readStorageBuffer(data._indirect)))).toEqual([6, 0, 0, 0]);
            }
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("long Arabic whole-run fallback remains identical under segmentation enable/disable", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const entry = cases.find((entry) => entry.id === "arabic-long-600")!;
        const plan = createComputeTextPlan(font.font, selectComputeTextRun(entry.text));
        expect(analyzeComputeTextSegmentation(compileComputeTextProgram(plan))).toBeNull();
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "arabic-long");
        const whole = createComputeTextData(native.engine, font.font, storage, "arabic-long");
        whole._disableSegmentation = true;
        try {
            await (
                await updateComputeTextData(data, [entry.text])
            ).completed;
            await (
                await updateComputeTextData(whole, [entry.text])
            ).completed;
            const a = new Int32Array(await readStorageBuffer(data._shaped));
            const b = new Int32Array(await readStorageBuffer(whole._shaped));
            expect(Array.from(a.slice(0, 4 + a[1]! * 6))).toEqual(Array.from(b.slice(0, 4 + b[1]! * 6)));
            expect(Array.from(a.slice(4, 4 + a[1]! * 6))).toEqual(entry.records.flat());
        } finally {
            disposeComputeTextData(data);
            disposeComputeTextData(whole);
            disposeGlyphStorage(storage);
        }
    });

    it("C10 mark run crossing a lane chunk matches its standalone exact reference", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const word = cases.find((entry) => entry.id === "arabic-stacked-0")!;
        // n=97 => four-scalar lane chunks; the mark run at 3..4 crosses that boundary.
        const text = "  " + word.text + " ".repeat(92);
        expect([...text]).toHaveLength(97);
        const actual = (await shapeComputeText(native.engine, font.font, text)).map(fields);
        const spaceGlyph = font.oracle.glyphId(0x20);
        const firstSpace = [spaceGlyph, 0, font.oracle.advanceWidth(spaceGlyph), 0, 0, 0];
        const expected = [
            ...Array.from({ length: 92 }, (_, i) => [firstSpace[0]!, 96 - i, ...firstSpace.slice(2)]),
            ...word.records.map((record) => [record[0]!, record[1]! + 2, ...record.slice(2)]),
            [firstSpace[0]!, 1, ...firstSpace.slice(2)],
            [firstSpace[0]!, 0, ...firstSpace.slice(2)],
        ];
        deepStrictEqual(actual, expected);
    });

    it("cold loader failure occurs before busy, clears drawing and permits retry", async () => {
        const font = fixture("NotoSansArabic-Regular.ttf");
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "arabic-load-failure");
        try {
            await (
                await updateComputeTextData(data, ["بب"])
            ).completed;
            installComputeTextProfileLoader("arabic", async () => {
                throw new Error("native test Arabic import failure");
            });
            const failed = updateComputeTextData(data, ["كتب"]);
            expect(data.status).toBe("ready");
            await expect(failed).rejects.toThrow("native test Arabic import failure");
            expect(Array.from(new Uint32Array(await readStorageBuffer(data._indirect)))).toEqual([6, 0, 0, 0]);
            installComputeTextProfileLoader("arabic", async () => {
                installComputeTextArabicProfile();
            });
            await (
                await updateComputeTextData(data, ["كتب"])
            ).completed;
            expect(data.status).toBe("ready");
        } finally {
            installComputeTextProfileLoader("arabic", async () => {
                installComputeTextArabicProfile();
            });
            await ensureComputeTextProfile("arabic");
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("internal mixed plan groups retain default, Hebrew and Arabic variants and exact records", async () => {
        // Empty atlas slots isolate shape scheduling; this is not a public multi-font API.
        await ensureComputeTextProfile("arabic");
        const fonts = [fixture("Roboto-Regular.ttf"), fixture("NotoSansHebrew-Regular.ttf"), fixture("NotoSansArabic-Regular.ttf")];
        const texts = ["HH", "שלום", "كتب"];
        const groups = fonts.map((font, i) => {
            const selection = selectComputeTextRun(texts[i]!);
            return {
                plan: createComputeTextPlan(font.font, selection),
                runs: [{ scalars: selection._scalars, glyphIds: selection._scalars.map((scalar) => font.oracle.glyphId(scalar)) }],
                outputRun: i,
            };
        });
        const capacity = Math.max(...groups.map((group) => computeTextShapeCapacity(group.plan, group.runs)));
        const glyphCount = Math.max(...fonts.map((font) => font.font.glyphCount));
        const stride = 4 + capacity * 6;
        const buffers = {
            shaped: createStorageBuffer(native.engine, groups.length * stride * 4, { writable: true }),
            instances: createStorageBuffer(native.engine, groups.length * capacity * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, Uint32Array.of(6, 0, 0, 0), { writable: true }),
            slots: createStorageBuffer(native.engine, new Uint32Array(glyphCount).fill(0xfffffffe)),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
        const report = { cases: reviewedRecords() };
        const expected = [
            oracleFields(fonts[0]!, texts[0]!),
            report.cases.find((entry) => entry.font === "NotoSansHebrew-Regular.ttf" && entry.text === texts[1])!.records,
            cases.find((entry) => entry.text === texts[2])!.records,
        ];
        const token = {};
        try {
            expect(await (await executeResidentComputeText(token, native.engine, groups, buffers, capacity, groups.length, 32, 1000, glyphCount, () => undefined)).status).toBe(0);
            const actual = new Int32Array(await readStorageBuffer(buffers.shaped));
            expected.forEach((records, i) => {
                expect(actual[i * stride]).toBe(0);
                expect(actual[i * stride + 1]).toBe(records.length);
                deepStrictEqual(Array.from(actual.slice(i * stride + 4, i * stride + 4 + records.length * 6)), records.flat());
            });
        } finally {
            releaseResidentComputeText(token);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });
});
