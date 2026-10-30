/** Cross-object shape batching: concurrent independent updates share one shape dispatch per plan and kernel. */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fixture, GREEK } from "./fixtures";
import type { Fixture, FontFile } from "./fixtures";
import * as shapeBatch from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape-batch";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

interface Case {
    font: FontFile;
    texts: string[];
}

const cases: Case[] = [
    { font: "Roboto-Regular.ttf", texts: ["Hi", "office", "To A\u0326"] },
    { font: "Roboto-Regular.ttf", texts: ["H".repeat(50), "affinity", "H".repeat(31)] },
    { font: "Roboto-Regular.ttf", texts: [GREEK, "AB"] },
    { font: "NotoSansDevanagari-Regular.ttf", texts: ["नमस्ते दुनिया", "क्षत्रिय", "हिन्दी"] },
    { font: "NotoSansDevanagari-Regular.ttf", texts: ["संस्कृत भाषा ".repeat(6)] },
    { font: "NotoSansArabic-Regular.ttf", texts: ["مرحبا بالعالم", "سلام"] },
];

interface Snapshot {
    shaped: number[];
    instances: number[];
    indirect: number[];
}

/** Rows hold a four-word header and `count` six-word records; words past `count` are unspecified. */
async function snapshot(data: ComputeTextData, rows: number): Promise<Snapshot> {
    const words = new Int32Array(await readStorageBuffer(data._shaped));
    const stride = words.length / rows;
    const shaped: number[] = [];
    for (let row = 0; row < rows; row++) {
        const base = row * stride;
        shaped.push(...words.subarray(base, base + 4 + words[base + 1]! * 6));
    }
    return {
        shaped,
        instances: Array.from(new Uint32Array(await readStorageBuffer(data._instances))),
        indirect: Array.from(new Uint32Array(await readStorageBuffer(data._indirect))),
    };
}

describe("native cross-object shape batching", () => {
    let native: NativeHarness, storage: GlyphStorage;
    const live = new Set<ComputeTextData>();
    const fonts = new Map<FontFile, Fixture>();

    function block(file: FontFile): ComputeTextData {
        let font = fonts.get(file);
        if (!font) {
            font = fixture(file);
            fonts.set(file, font);
        }
        const data = createComputeTextData(native.engine, font.font, storage, file);
        live.add(data);
        return data;
    }

    function release(data: ComputeTextData): void {
        live.delete(data);
        disposeComputeTextData(data);
    }

    async function settle(data: ComputeTextData, texts: readonly string[]): Promise<void> {
        await (
            await updateComputeTextData(data, texts)
        ).completed;
    }

    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        live.forEach(release);
        await assertNativeClean(native);
    });
    afterAll(async () => {
        disposeGlyphStorage(storage);
        await native?.close();
    });

    it("shapes concurrent mixed-script objects of different capacities exactly as serial updates do", async () => {
        const expected: Snapshot[] = [];
        for (const { font, texts } of cases) {
            const control = block(font);
            await settle(control, texts);
            expected.push(await snapshot(control, texts.length));
        }
        const objects = cases.map(({ font }) => block(font));
        const slices = vi.spyOn(shapeBatch, "addComputeTextShapeSlice");
        const encoders = vi.spyOn(native.device, "createCommandEncoder");
        const updates = await Promise.all(objects.map((data, index) => updateComputeTextData(data, cases[index]!.texts)));
        await Promise.all(updates.map((update) => update.completed));
        // Every object admits its plan groups as slices; slices sharing a plan and kernel record one shape dispatch.
        const batches = encoders.mock.calls.filter(([descriptor]) => descriptor?.label === "compute-text-shape-batch").length;
        expect(slices.mock.calls.length).toBeGreaterThanOrEqual(cases.length);
        expect(batches).toBeGreaterThan(0);
        expect(batches).toBeLessThan(slices.mock.calls.length);
        for (let index = 0; index < objects.length; index++) {
            expect(await snapshot(objects[index]!, cases[index]!.texts.length)).toEqual(expected[index]);
        }
    });

    it("keeps the latest of two updates admitted to the same object in one task", async () => {
        const control = block("NotoSansDevanagari-Regular.ttf");
        await settle(control, ["क्षत्रिय हिन्दी"]);
        const expected = await snapshot(control, 1);
        const data = block("NotoSansDevanagari-Regular.ttf");
        const other = block("Roboto-Regular.ttf");
        const first = await updateComputeTextData(data, ["नमस्ते"]);
        const [second, sibling] = await Promise.all([updateComputeTextData(data, ["क्षत्रिय हिन्दी"]), updateComputeTextData(other, ["office"])]);
        await Promise.allSettled([first.completed]);
        await second.completed;
        await sibling.completed;
        expect(await snapshot(data, 1)).toEqual(expected);
    });

    it("completes the other participants when one object is disposed with a pending slice", async () => {
        const control = block("Roboto-Regular.ttf");
        await settle(control, ["office affinity"]);
        const expected = await snapshot(control, 1);
        const doomed = block("Roboto-Regular.ttf");
        const survivor = block("Roboto-Regular.ttf");
        await settle(doomed, ["warm"]);
        await settle(survivor, ["warm"]);
        const [lost, kept] = await Promise.all([updateComputeTextData(doomed, ["H".repeat(40)]), updateComputeTextData(survivor, ["office affinity"])]);
        release(doomed);
        await expect(lost.completed).rejects.toThrow(/disposed/);
        await kept.completed;
        expect(await snapshot(survivor, 1)).toEqual(expected);
    });
});
