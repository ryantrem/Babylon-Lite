import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createComputeFontFromBuffer, getComputeFontLineMetrics, updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/index";
import type { ComputeFont, ComputeTextParagraphLayout } from "../../../packages/babylon-lite/src/index";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { createComputeTextData, disposeComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import {
    allocateComputeTextParagraphSlots,
    computeTextParagraphCacheValid,
    computeTextParagraphDiff,
    computeTextParagraphFlags,
    computeTextParagraphPoolSize,
} from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import { createComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import type { ComputeTextParagraphCache } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import type { SegmentAnalysis } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import * as segmentation from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";

function font(file = "Inter.ttf"): ComputeFont {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", file));
    return createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}

function scalars(text: string): Uint32Array {
    return Uint32Array.from(text, (ch) => ch.codePointAt(0)!);
}

function analysis(safeAfter: SegmentAnalysis["safeAfter"] = () => true): SegmentAnalysis {
    return { space: 0x20, span: 1, crossings: 0, markBaseHit: false, safeAfter };
}

function block(file?: string) {
    const device = {
        limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20, maxComputeWorkgroupsPerDimension: 65535 },
        lost: new Promise<GPUDeviceLostInfo>(() => undefined),
        createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
            getMappedRange: () => new ArrayBuffer(desc.size),
            unmap: vi.fn(),
            destroy: vi.fn(),
        })),
        queue: { writeBuffer: vi.fn() },
    };
    const engine = { _device: device } as unknown as EngineContext;
    const storage = createGlyphStorage();
    const data = createComputeTextData(engine, font(file), storage, "unit");
    return {
        data,
        storage,
        close() {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        },
    };
}

afterEach(() => vi.restoreAllMocks());

describe("paragraph incremental state", () => {
    it.each([
        { previous: [], next: [], expected: [0, 0] },
        { previous: [], next: ["A"], expected: [0, 0] },
        { previous: ["A"], next: [], expected: [0, 0] },
        { previous: ["A", "B"], next: ["A", "B"], expected: [2, 0] },
        { previous: ["A", "B", "C"], next: ["A", "Bx", "C"], expected: [1, 1] },
        { previous: ["A", "BC", "D"], next: ["A", "B", "C", "D"], expected: [1, 1] },
        { previous: ["A", "B", "C", "D"], next: ["A", "BC", "D"], expected: [1, 1] },
        { previous: ["A", "B", "C"], next: ["A", "C"], expected: [1, 1] },
        { previous: ["A", "A", "A"], next: ["A", "A"], expected: [2, 0] },
        { previous: ["A", "B", "A"], next: ["A", "A", "B", "A"], expected: [1, 2] },
        { previous: ["A", "B", "C"], next: ["X", "Y", "Z"], expected: [0, 0] },
        { previous: ["A", "B", "C", "D"], next: ["Ax", "B", "Cx", "D"], expected: [0, 1] },
        { previous: ["", "A", ""], next: ["", "", ""], expected: [1, 1] },
    ])("finds a non-overlapping positional middle: $previous → $next", ({ previous, next, expected }) => {
        expect(computeTextParagraphDiff(previous, next)).toEqual(expected);
    });

    it("reuses a freed slot, preserves suffix slots on split/merge, and first-fits earlier holes", () => {
        const previous = Uint32Array.of(0, 1, 2, 3);
        expect(allocateComputeTextParagraphSlots(previous, 12, 4, 1, 2)).toEqual({ order: Uint32Array.of(0, 1, 2, 3), runBase: 1 });
        const split = allocateComputeTextParagraphSlots(previous, 12, 5, 1, 2)!;
        expect(split).toEqual({ order: Uint32Array.of(0, 4, 5, 2, 3), runBase: 4 });
        expect(allocateComputeTextParagraphSlots(split.order, 12, 4, 1, 2)).toEqual({ order: Uint32Array.of(0, 1, 2, 3), runBase: 1 });
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(4, 5, 2), 12, 3, 1, 1)).toEqual({ order: Uint32Array.of(4, 0, 2), runBase: 0 });
    });

    it("handles deletion and unchanged input without allocating a dirty range", () => {
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(3, 1, 4), 8, 2, 1, 1)).toEqual({ order: Uint32Array.of(3, 4), runBase: 0 });
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(3, 1), 8, 2, 2, 0)).toEqual({ order: Uint32Array.of(3, 1), runBase: 0 });
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(3, 1), 8, 0, 0, 0)).toEqual({ order: new Uint32Array(), runBase: 0 });
    });

    it("requests compaction on fragmented space or pool exhaustion", () => {
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(1, 3, 5), 7, 5, 1, 2)).toBeNull();
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(0, 1), 2, 3, 2, 0)).toBeNull();
        expect(allocateComputeTextParagraphSlots(Uint32Array.of(1, 3, 5), 7, 5, 0, 0)).toEqual({ order: Uint32Array.of(0, 1, 2, 3, 4), runBase: 0 });
        expect([0, 4, 32, 33, 100, 500].map(computeTextParagraphPoolSize)).toEqual([8, 12, 40, 42, 125, 625]);
    });

    it("invalidates stale submission, generation, font, size, device and shaped-buffer state", () => {
        const test = block();
        const data = test.data;
        const cache: ComputeTextParagraphCache = {
            _paragraphs: ["A"],
            _order: Uint32Array.of(0),
            _capacity: 2,
            _pool: 9,
            _width: 100,
            _font: data._font,
            _fontSize: data.fontSizePx,
            _device: data._device,
            _generation: data._generation,
            _revision: data._revision,
            _baseRevision: data._revision,
            _shaped: data._shaped,
        };
        try {
            expect(computeTextParagraphCacheValid(data, cache)).toBe(true);
            for (const override of [{ _revision: -1 }, { _generation: -1 }, { _font: font() }, { _fontSize: 12 }, { _device: {} as GPUDevice }, { _shaped: data._instances }]) {
                expect(computeTextParagraphCacheValid(data, { ...cache, ...override })).toBe(false);
            }
            test.close();
            expect(computeTextParagraphCacheValid(data, cache)).toBe(false);
        } finally {
            test.close();
        }
    });
});

describe("paragraph scalar flags", () => {
    it("marks only ASCII spaces and only the last space in a followed run as break-after", () => {
        const input = scalars("  A   B  \u00a0\u2002");
        const safe = vi.fn((_ids: Uint32Array, _index: number) => true);
        expect([...computeTextParagraphFlags(input, input, analysis(safe))]).toEqual([1, 3, 0, 1, 1, 3, 0, 1, 3, 0, 0]);
        expect(safe.mock.calls.map((call) => call[1])).toEqual([1, 5, 8]);
        expect([...computeTextParagraphFlags(scalars("A   "), scalars("A   "), null)]).toEqual([0, 1, 1, 1]);
        expect(computeTextParagraphFlags(new Uint32Array(0), new Uint32Array(0), null)).toHaveLength(0);
    });

    it("uses nominal glyph ids and the space index, never the next scalar or shaped glyph", () => {
        const input = scalars("A  B C");
        const glyphs = Uint32Array.of(10, 99, 99, 11, 99, 12);
        const safe = vi.fn((ids: Uint32Array, i: number) => ids[i] === 99 && i === 2);
        expect([...computeTextParagraphFlags(input, glyphs, analysis(safe))]).toEqual([0, 1, 3, 0, 1, 0]);
        expect(safe.mock.calls).toEqual([
            [glyphs, 2],
            [glyphs, 4],
        ]);
        expect(() => computeTextParagraphFlags(input, glyphs, null)).toThrow(/updateComputeTextParagraphs:.*analysis is unavailable/);
    });

    it("indexes scalars, not UTF-16 units; marks Mn/Mc/Me and ZWJ as no-break-before", () => {
        const input = scalars("𐌰 \u0308\u0903\u20dd\u200d𐌱");
        expect([
            ...computeTextParagraphFlags(
                input,
                input,
                analysis(() => false)
            ),
        ]).toEqual([0, 1, 4, 4, 4, 4, 0]);
        expect(input).toHaveLength(7);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("common font analysis is available and admits independent space boundaries: %s", (file) => {
        const face = font(file);
        const selected = selectComputeTextRun("office   affinity A\u0308", face);
        const glyphs = mapComputeFontScalars(
            face,
            selected._scalars,
            (_i, error) => {
                throw error;
            },
            selected._nominal
        );
        const safety = segmentation.analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face, selected)));
        expect(safety).not.toBeNull();
        const flags = computeTextParagraphFlags(selected._scalars, glyphs, safety);
        const candidates = [8, 17];
        for (const i of candidates) {
            expect(flags[i], `space ${i}`).toBe(3);
        }
        expect(flags[6]).toBe(1);
        expect(flags[7]).toBe(1);
        expect(flags.at(-1)).toBe(4);
    });

    it("keeps real unsafe Roboto spaces as hanging whitespace without break-after", () => {
        const face = font("Roboto-Regular.ttf");
        const selected = selectComputeTextRun("AV To", face);
        const glyphs = mapComputeFontScalars(
            face,
            selected._scalars,
            (_i, error) => {
                throw error;
            },
            selected._nominal
        );
        const safety = segmentation.analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face, selected)));
        expect(safety).not.toBeNull();
        expect(safety!.safeAfter(glyphs, 2)).toBe(false);
        expect([...computeTextParagraphFlags(selected._scalars, glyphs, safety)]).toEqual([0, 0, 1, 0, 0]);
    });
});

describe("paragraph synchronous admission", () => {
    it.each(["\n", "\r", "\t", "\u2028", "\u2029"])("rejects paragraph controls synchronously: %j", (control) => {
        expect(() => updateComputeTextParagraphs(undefined!, ["A" + control + "B"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs:.*line breaks or tabs/);
    });

    it.each([null, "A", ["A", 1], Array(1)])("requires a dense array of strings: %j", (input) => {
        expect(() => updateComputeTextParagraphs(undefined!, input as readonly string[], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs:.*string/);
    });

    it.each([
        { maxWidthPx: 0 },
        { maxWidthPx: -1 },
        { maxWidthPx: NaN },
        { maxWidthPx: -Infinity },
        { maxWidthPx: 1e100 },
        { maxWidthPx: 1e-100 },
        { maxWidthPx: "12" },
        { maxWidthPx: 100, lineHeightPx: 0 },
        { maxWidthPx: 100, lineHeightPx: Infinity },
        { maxWidthPx: 100, lineHeightPx: 1e100 },
        { maxWidthPx: 100, paragraphSpacingPx: -1 },
        { maxWidthPx: 100, paragraphSpacingPx: NaN },
        { maxWidthPx: 100, paragraphSpacingPx: Infinity },
        { maxWidthPx: 100, align: "justify" },
        { maxWidthPx: Infinity, align: "center" },
        { maxWidthPx: Infinity, align: "right" },
    ])("rejects invalid options synchronously: %j", (layout) => {
        expect(() => updateComputeTextParagraphs(undefined!, ["A"], layout as ComputeTextParagraphLayout)).toThrow(/^updateComputeTextParagraphs:/);
    });

    it.each(["NotoSansHebrew-Regular.ttf", "NotoSansArabic-Regular.ttf"])("rejects RTL before atlas mutation: %s", (file) => {
        const test = block(file);
        try {
            expect(() => updateComputeTextParagraphs(test.data, [file.includes("Hebrew") ? "שלום" : "سلام"], { maxWidthPx: 100 })).toThrow(
                /updateComputeTextParagraphs:.*LTR script/
            );
            expect(test.storage._curveSets.size).toBe(0);
            expect(test.data.status).toBe("ready");
        } finally {
            test.close();
        }
    });

    it("rejects unavailable safety synchronously, but leaves a ready block and its atlas unchanged", () => {
        const test = block();
        vi.spyOn(segmentation, "analyzeComputeTextSegmentation").mockReturnValue(null);
        try {
            expect(() => updateComputeTextParagraphs(test.data, ["A A"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs:.*analysis is unavailable/);
            expect(test.storage._curveSets.size).toBe(0);
            expect(test.data.status).toBe("ready");
        } finally {
            test.close();
        }
    });

    it("checks layout storage limits before atlas mutation", () => {
        const test = block();
        Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 8 });
        try {
            expect(() => updateComputeTextParagraphs(test.data, ["A A"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs:.*storage limits/);
            expect(test.storage._curveSets.size).toBe(0);
            expect(test.data.status).toBe("ready");
        } finally {
            test.close();
        }
    });

    it("checks line-record binding sizes and dispatch limits before atlas mutation", () => {
        for (const kind of ["lines", "dispatch"]) {
            const test = block();
            Object.defineProperty(test.data._device.limits, kind === "lines" ? "maxStorageBufferBindingSize" : "maxComputeWorkgroupsPerDimension", {
                value: kind === "lines" ? 512 : 0,
            });

            try {
                expect(() => updateComputeTextParagraphs(test.data, ["A".repeat(20)], { maxWidthPx: 100 })).toThrow(
                    kind === "lines" ? /updateComputeTextParagraphs: layout buffer.*storage limits/ : /updateComputeTextParagraphs:.*workgroup limits/
                );
                expect(test.storage._curveSets.size).toBe(0);
                expect(test.data.status).toBe("ready");
            } finally {
                test.close();
            }
        }
    });
});

describe("paragraph per-record storage", () => {
    it("linearizes dispatches beyond the per-dimension limit and resets y on smaller edits", () => {
        const test = block();
        Object.assign(test.data._device, { createBindGroupLayout: vi.fn(() => ({})), createPipelineLayout: vi.fn(() => ({})), createBindGroup: vi.fn(() => ({})) });
        Object.defineProperty(test.data._device.limits, "maxComputeWorkgroupsPerDimension", { value: 4 });
        const words = new Uint32Array(23);
        words[0] = words[9] = words[11] = 7;
        words[1] = 1;
        const stage = createComputeTextParagraphStage({}, test.data._engine, words);
        try {
            stage.prepare(test.data._engine, 7, 1);
            const buffer = stage._lines!;
            const dispatches = stage.dispatches({ shaped: buffer, slots: buffer, instances: buffer, indirect: buffer, status: buffer });
            expect(dispatches.map((dispatch) => [dispatch._x, dispatch._y, dispatch.enabled])).toEqual([
                [4, 2, true],
                [1, 1, true],
                [4, 2, true],
            ]);
            words[0] = 1;
            words[9] = 0;
            stage.prepare(test.data._engine, 0, 1);
            expect(dispatches.map((dispatch) => [dispatch._x, dispatch._y, dispatch.enabled])).toEqual([
                [0, 1, false],
                [1, 1, true],
                [1, 1, true],
            ]);
        } finally {
            stage.release();
            test.close();
        }
    });

    it("includes both per-record vec4 tables in line storage bounds and allocation", () => {
        const test = block();
        const words = new Uint32Array(18);
        words[1] = 10;
        words[11] = 2;
        const stage = createComputeTextParagraphStage({}, test.data._engine, words);
        try {
            stage.prepare(test.data._engine, 0, 10);
            expect(stage._lines!.byteLength).toBe(2 * 10 * 48);
            Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 2 * 10 * 48 - 1 });
            expect(() => createComputeTextParagraphStage({}, test.data._engine, words)).toThrow(/layout buffer of 960 bytes exceeds device storage limits/);
        } finally {
            stage.release();
            test.close();
        }
    });
});

describe("compute font vertical metrics", () => {
    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("reads hhea ascender, descender and gap at each size: %s", (file) => {
        const face = font(file);
        const hhea = face._directory.tables.get("hhea")!;
        const view = new DataView(face._bytes.buffer, face._bytes.byteOffset + hhea.offset, hhea.length);
        for (const size of [12, 32, 47.5]) {
            const scale = size / face.unitsPerEm;
            const expected = { ascentPx: view.getInt16(4) * scale, descentPx: -view.getInt16(6) * scale, lineGapPx: Math.max(0, view.getInt16(8)) * scale };
            expect(getComputeFontLineMetrics(face, size)).toEqual({ ...expected, lineHeightPx: expected.ascentPx + expected.descentPx + expected.lineGapPx });
        }
    });

    it("clamps negative line gaps and validates the font size", () => {
        const original = font();
        const bytes = original._bytes.slice();
        const hhea = original._directory.tables.get("hhea")!;
        new DataView(bytes.buffer).setInt16(hhea.offset + 8, -100);
        const face = createComputeFontFromBuffer(bytes.buffer);
        expect(getComputeFontLineMetrics(face, 32).lineGapPx).toBe(0);
        for (const size of [0, -1, NaN, Infinity]) {
            expect(() => getComputeFontLineMetrics(face, size)).toThrow(/fontSizePx/);
        }
    });
});
