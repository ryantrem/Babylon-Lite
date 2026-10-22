import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { computeTextShapeCapacity } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import * as shapeWords from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import * as dispatchSizes from "../../../packages/babylon-lite/src/compute/compute-dispatch";
import * as resident from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { fixture, oracleFields } from "./fixtures";
import type { Fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

const english =
    "The quick brown fox jumps over the lazy dog. Every letter belongs to a flowing English paragraph with ordinary spaces and punctuation. AV To are adjacent pairs; office and affinity remain ordered shaping controls. ";
const natural =
    "Office affinity offers fine AVATAR kerning. The efficient flight lifts off at five. WAVY typography: coffee, waffles and films. A quick brown fox visits the office today. Friendly staff review lovely typography. Affluent traffic flows over the river. Vast valleys offer fresh coffee at noon. To verify fonts, try AV WA To fi fl ffi. ";
const extra = "Hello (world) - it's a test: x = y + 2; [a] {b} @home <tag> ~tilde* \"quoted\" 'single' 12:30 -> => ... end. ";

describe("native bit-exact segment shape and stitch", () => {
    let native: NativeHarness;
    let storage: GlyphStorage;
    const live: ComputeTextData[] = [];
    function block(font: Fixture, disabled = false): ComputeTextData {
        const data = createComputeTextData(native.engine, font.font, storage, font.file);
        data._disableSegmentation = disabled;
        live.push(data);
        return data;
    }
    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        live.forEach(disposeComputeTextData);
        disposeGlyphStorage(storage);
        await native.close();
    });

    async function update(data: ComputeTextData, texts: readonly string[]): Promise<void> {
        await (
            await updateComputeTextData(data, texts)
        ).completed;
    }
    async function exact(font: Fixture, candidate: ComputeTextData, baseline: ComputeTextData, texts: readonly string[]): Promise<void> {
        const capacity = Math.max(
            1,
            ...texts.map((text) => {
                const selection = selectComputeTextRun(text, font.font);
                return computeTextShapeCapacity(createComputeTextPlan(font.font, selection), [{ scalars: selection._scalars }]);
            })
        );
        const stride = 4 + capacity * 6;
        const a = new Int32Array(await readStorageBuffer(candidate._shaped));
        const b = new Int32Array(await readStorageBuffer(baseline._shaped));
        for (let run = 0; run < texts.length; run++) {
            const expected = oracleFields(font, texts[run]!);
            const start = run * stride;
            expect([...a.slice(start, start + 2)]).toEqual([0, expected.length]);
            expect([...a.slice(start, start + 2)]).toEqual([...b.slice(start, start + 2)]);
            expect([...a.slice(start + 2, start + 4)]).toEqual([0, 0]);
            expect([...a.slice(start + 4, start + 4 + expected.length * 6)]).toEqual([...b.slice(start + 4, start + 4 + expected.length * 6)]);
            expect([...a.slice(start + 4, start + 4 + expected.length * 6)]).toEqual(expected.flat());
            expected.forEach((record, i) => expect(a[start + 5 + i * 6], `global scalar cluster ${i}`).toBe(record[1]));
        }
        const indirect = await readStorageBuffer(candidate._indirect);
        expect(new Uint8Array(indirect)).toEqual(new Uint8Array(await readStorageBuffer(baseline._indirect)));
        const visible = new Uint32Array(indirect)[1]!;
        expect(new Uint8Array((await readStorageBuffer(candidate._instances)).slice(0, visible * 12))).toEqual(
            new Uint8Array((await readStorageBuffer(baseline._instances)).slice(0, visible * 12))
        );
    }

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("matches whole-run output and oracle at 480/1000 glyphs for all corpora: %s", async (file) => {
        const font = fixture(file);
        const candidate = block(font);
        const baseline = block(font, true);
        for (const length of [480, 1000]) {
            for (const corpus of [english, natural, extra]) {
                const texts = [corpus.repeat(12).slice(0, length)];
                const prepare = vi.spyOn(shapeWords, "prepareComputeTextShapeWords");
                await update(candidate, texts);
                expect(prepare.mock.calls[0]![1].length).toBeGreaterThan(1);
                expect(prepare.mock.calls[0]![3]).toBeLessThan(length);
                prepare.mockRestore();
                await update(baseline, texts);
                await exact(font, candidate, baseline, texts);
            }
        }
    });

    it("preserves logical slots across mixed Latin/Greek groups and unsplittable/short runs", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const candidate = block(font);
        const baseline = block(font, true);
        const texts = ["fi", english.repeat(3).slice(0, 480), "ΑΒΓΔ ".repeat(60), "", "A".repeat(300), natural.repeat(3).slice(0, 480), "HH"];
        await update(candidate, texts);
        await update(baseline, texts);
        await exact(font, candidate, baseline, texts);
    });

    it("includes an RTL fallback group alongside long Latin segments", async () => {
        const font = fixture("NotoSansHebrew-Regular.ttf");
        const candidate = block(font);
        const baseline = block(font, true);
        const texts = ["a b ".repeat(120), "שלום ".repeat(30), "a b"];
        await update(candidate, texts);
        await update(baseline, texts);
        await exact(font, candidate, baseline, texts);
    });

    it("tiles actual non-stage segmented callers above 64 segments through overlapping varying-size updates", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const candidate = block(font);
        const baseline = block(font, true);
        const execute = vi.spyOn(resident, "executeResidentComputeText");
        const sizes = vi.spyOn(dispatchSizes, "setComputeDispatchSize");
        const long = english.repeat(32);
        for (const texts of [
            [long, "HH"],
            [long + " A", english.repeat(24)],
            [english.repeat(25), long],
            ["HH", long],
        ]) {
            execute.mockClear();
            sizes.mockClear();
            const first = await updateComputeTextData(candidate, texts);
            const second = await updateComputeTextData(candidate, texts);
            await Promise.all([first.completed, second.completed]);
            const segments = execute.mock.calls.map((call) => call[10]).filter((value) => value != null);
            expect(segments.length).toBeGreaterThan(0);
            const stitch = sizes.mock.calls.filter(([dispatch]) => dispatch.shader.name === "compute-text-stitch");
            expect(stitch.length).toBeGreaterThan(0);
            expect(stitch.some(([, size]) => (size.y ?? 1) > 1)).toBe(true);
            stitch.forEach(([, size], i) => {
                const call = execute.mock.calls.filter((call) => call[10] != null)[i]!;
                expect(size.y).toBe(resident.stitchTileRows(segments[i]!.table, call[5]));
            });
            await update(baseline, texts);
            await exact(font, candidate, baseline, texts);
        }
    });

    it("propagates a rejecting segment to logical status 1 and suppresses all instances", async () => {
        const font = fixture("Inter.ttf");
        const candidate = block(font);
        const baseline = block(font, true);
        const ok = [english.repeat(3).slice(0, 480)];
        await update(candidate, ok);
        await update(baseline, ok);
        const texts = ["a ".repeat(60) + "<-" + " b".repeat(100)];
        const prepared = vi.spyOn(shapeWords, "prepareComputeTextShapeWords");
        const failed = await updateComputeTextData(candidate, texts);
        expect(prepared.mock.calls[0]![1].length).toBeGreaterThan(1);
        prepared.mockRestore();
        await expect(failed.completed).rejects.toThrow("unsupported lookup");
        await expect(update(baseline, texts)).rejects.toThrow("unsupported lookup");
        expect(new Int32Array(await readStorageBuffer(candidate._shaped))[0]).toBe(1);
        expect(new Int32Array(await readStorageBuffer(baseline._shaped))[0]).toBe(1);
        expect([...new Uint32Array(await readStorageBuffer(candidate._indirect))]).toEqual([6, 0, 0, 0]);
        expect(new Uint8Array(await readStorageBuffer(candidate._instances))).toEqual(new Uint8Array(await readStorageBuffer(baseline._instances)));
    });

    it("reuses warm segmented bindings, rebuilds transitions and disposes all new buffers", async () => {
        const font = fixture("Roboto-Regular.ttf");
        const candidate = block(font);
        const baseline = block(font, true);
        const long = [english.repeat(3).slice(0, 480)];
        const createBuffer = native.device.createBuffer.bind(native.device);
        const destroys: ReturnType<typeof vi.spyOn>[] = [];
        vi.spyOn(native.device, "createBuffer").mockImplementation((descriptor) => {
            const buffer = createBuffer(descriptor);
            if (["compute-text-segment-output", "compute-text-segment-table", "compute-text-stitch-params"].includes(descriptor.label ?? "")) {
                destroys.push(vi.spyOn(buffer, "destroy"));
            }
            return buffer;
        });
        await update(candidate, long);
        expect(destroys.length).toBeGreaterThanOrEqual(3);
        const groups = vi.spyOn(native.device, "createBindGroup");
        await update(candidate, long);
        expect(groups).not.toHaveBeenCalled();
        for (const texts of [["HH"], long, ["A".repeat(600)], [english.repeat(5).slice(0, 1000)], long]) {
            groups.mockClear();
            await update(candidate, texts);
            expect(groups).toHaveBeenCalled();
            await update(baseline, texts);
            await exact(font, candidate, baseline, texts);
        }
        disposeComputeTextData(candidate);
        destroys.forEach((destroy) => expect(destroy).toHaveBeenCalled());
    });
});
