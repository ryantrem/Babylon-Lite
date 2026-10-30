import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeFontFromBuffer, getComputeFontLineMetrics, updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/index";
import type { ComputeFont, ComputeTextParagraphLayout } from "../../../packages/babylon-lite/src/index";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { computeTextDrawBufferSizes, createComputeTextData, disposeComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import {
    allocateComputeTextParagraphSlots,
    computeTextParagraphCacheValid,
    computeTextParagraphAllocation,
    computeTextParagraphDiff,
    computeTextParagraphPoolSize,
} from "../../../packages/babylon-lite/src/text/compute-text-paragraphs";
import { computeTextParagraphFlags, computeTextParagraphFlagsIncremental } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-flags";
import { createComputeTextParagraphStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import * as paragraphStages from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import type { ComputeTextParagraphCache } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import type { SegmentAnalysis } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import * as segmentation from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import * as profiles from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { installComputeTextIndicProfile } from "../../../packages/babylon-lite/src/text/compute-text-indic";

function font(file = "Inter.ttf"): ComputeFont {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", file));
    return createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}

function scalars(text: string): Uint32Array {
    return Uint32Array.from(text, (ch) => ch.codePointAt(0)!);
}

function analysis(safeAfter: SegmentAnalysis["safeAfter"] = () => true): SegmentAnalysis {
    return { space: 0x20, span: 1, maxD: 1, crossings: 0, markBaseHit: false, safeAfter };
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
    data._chunkBudgets = { monolithic: true };
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

describe("paragraph allocation budget", () => {
    it.each([
        { count: 0, needed: 1, limit: 1 << 20, pool: 8, capacity: 2 },
        { count: 1, needed: 20, limit: 1 << 20, pool: 9, capacity: 25 },
        { count: 4, needed: 20, limit: 1 << 20, pool: 12, capacity: 25 },
        { count: 1, needed: 20, limit: 4096, pool: 3, capacity: 25 },
        { count: 3, needed: 20, limit: 4096, pool: 3, capacity: 25 },
        { count: 1, needed: 20, limit: 1200, pool: 1, capacity: 25 },
        { count: 1, needed: 20, limit: 1199, pool: 1, capacity: 24 },
        { count: 1, needed: 20, limit: 960, pool: 1, capacity: 20 },
    ])("plans required storage before optional headroom: $count/$needed/$limit", ({ count, needed, limit, pool, capacity }) => {
        const limits = { maxBufferSize: limit, maxStorageBufferBindingSize: limit };
        const planned = computeTextParagraphAllocation(limits, count, needed);
        expect(planned).toEqual({ pool, capacity });
        const sizes = [...paragraphStages.computeTextParagraphBufferSizes(pool, capacity), ...computeTextDrawBufferSizes(pool, capacity)];
        for (const [, bytes] of sizes) {
            expect(bytes % 4).toBe(0);
            expect(bytes).toBeLessThanOrEqual(limit);
        }
        if (pool < computeTextParagraphPoolSize(count)) {
            expect(paragraphStages.computeTextParagraphBufferSizes(pool + 1, capacity).some(([, bytes]) => bytes > limit)).toBe(true);
        }
        if (capacity < Math.ceil(needed * 1.25)) {
            expect(paragraphStages.computeTextParagraphBufferSizes(count, capacity + 1).some(([, bytes]) => bytes > limit)).toBe(true);
        }
    });

    it("accounts for every affected layout/draw buffer with buffer-specific diagnostics", () => {
        const pool = 3;
        const capacity = 25;
        const sizes = [...paragraphStages.computeTextParagraphBufferSizes(pool, capacity), ...computeTextDrawBufferSizes(pool, capacity)];
        expect(sizes).toEqual([
            ["layout input", 160],
            ["layout info", 96],
            ["layout lines", 3600],
            ["shaped", 1848],
            ["instances", 900],
        ]);
        for (const [name, bytes] of sizes) {
            for (const limit of ["maxBufferSize", "maxStorageBufferBindingSize"] as const) {
                const limits = { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20, [limit]: bytes - 1 };
                expect(() => paragraphStages.checkComputeTextParagraphBufferSize(limits, name, bytes, pool, capacity)).toThrow(
                    `updateComputeTextParagraphs: ${name} buffer of ${bytes} bytes exceeds device storage limits ` +
                        `(maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize}, pool=3, capacity=25)`
                );
                limits[limit] = bytes;
                expect(() => paragraphStages.checkComputeTextParagraphBufferSize(limits, name, bytes, pool, capacity)).not.toThrow();
            }
        }
        expect(() =>
            computeTextParagraphAllocation({ maxBufferSize: Number.MAX_SAFE_INTEGER, maxStorageBufferBindingSize: Number.MAX_SAFE_INTEGER }, 1, Number.MAX_SAFE_INTEGER)
        ).toThrow(/layout input buffer.*storage limits/);
    });

    it("preflights each actual layout allocation, including retained/rounded input words", () => {
        const test = block();
        const words = new Uint32Array(100);
        words[1] = 1;
        words[11] = 1;
        try {
            Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 399, configurable: true });
            expect(() => createComputeTextParagraphStage({}, test.data._engine, words, 18)).toThrow(/layout input buffer of 400 bytes.*maxStorageBufferBindingSize=399/);
            words[11] = 30;
            Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 959 });
            expect(() => createComputeTextParagraphStage({}, test.data._engine, words)).toThrow(/layout info buffer of 960 bytes.*pool=30, capacity=1/);
        } finally {
            test.close();
        }
    });

    it("clamps doubled CPU input reservation to the device budget and checks the actual bound size", () => {
        const test = block();
        Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 4096, configurable: true });
        const token = {};
        try {
            const stage = paragraphStages.reserveComputeTextParagraphStage(token, test.data._engine, 1, 1, 1, 500);
            expect(stage._words.byteLength).toBe(2068);
            expect(paragraphStages.reserveComputeTextParagraphStage(token, test.data._engine, 1, 1, 1, 600)).toBe(stage);
            expect(stage._words.byteLength).toBe(4096);
            expect(stage._uploadWords.byteLength).toBe(2468);
            Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 3000 });
            paragraphStages.reserveComputeTextParagraphStage(token, test.data._engine, 1, 1, 1, 600);
            expect(stage._words.byteLength).toBe(3000);
            stage.release();
        } finally {
            test.close();
        }
    });
});

describe("paragraph scalar flags", () => {
    it("matches full bytes and little-endian packed words across 600 fixed-seed scalar edits and proof-exhaustion boundaries", () => {
        let seed = 0x61c3a9d7;
        const next = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0);
        const pack = (flags: Uint8Array) => {
            const words = new Uint32Array(Math.ceil((flags.length + 3) / 4));
            for (let i = 0; i < flags.length; i++) {
                words[i >>> 2]! |= flags[i]! << ((i % 4) * 8);
            }
            return words;
        };
        for (const file of ["Inter.ttf", "Roboto-Regular.ttf", "NotoSansGothic-Regular.ttf"]) {
            const face = font(file);
            const astral = file.includes("Gothic");
            const base = astral ? "𐌰 𐌱 𐌲 𐌰   " : "office affinity AV To A\u0308 A\u0326   ";
            const alphabet = astral ? ["𐌰", "𐌱", " ", "   ", "𐌲 "] : ["A", "B", " ", "   ", "\u0308", "\u0326", "To ", "AV"];
            let text = base.repeat(40) + (astral ? "𐌰   𐌱 " : "A" + "\u0308".repeat(65) + " B ") + base.repeat(20);
            const decode = (value: string) => {
                const selected = selectComputeTextRun(value);
                return {
                    scalars: selected._scalars,
                    glyphIds: mapComputeFontScalars(face, selected._scalars, (_i, error) => {
                        throw error;
                    }),
                };
            };
            const safety = segmentation.analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face, selectComputeTextRun(text))));
            expect(safety).not.toBeNull();
            let input = decode(text);
            let flags = computeTextParagraphFlags(input.scalars, input.glyphIds, safety);
            for (let edit = 0; edit < 200; edit++) {
                const old = { ...input, flags };
                const values = Array.from(text);
                const positions = [0, values.length, Math.floor(values.length / 2), 65, 66, 67, values.length - 66, next() % (values.length + 1)];
                const at = Math.max(0, Math.min(values.length, positions[edit % positions.length]!));
                const insert = Array.from(alphabet[next() % alphabet.length]!);
                if (edit % 3 === 0) {
                    values.splice(at, 0, ...insert);
                } else if (edit % 3 === 1) {
                    values.splice(at, Math.min(1 + (next() % 3), values.length - at));
                } else {
                    values.splice(at, Math.min(1, values.length - at), ...insert);
                }
                text = values.join("");
                input = decode(text);
                flags = computeTextParagraphFlagsIncremental(old, input.scalars, input.glyphIds, () => safety);
                const full = computeTextParagraphFlags(input.scalars, input.glyphIds, safety);
                expect(flags, `${file} edit${edit}`).toEqual(full);
                expect(pack(flags), `${file} packed${edit}`).toEqual(pack(full));
            }
        }
        for (const text of ["", "A", " ", " A ", "A\u0308 "]) {
            const input = scalars(text);
            expect(
                computeTextParagraphFlagsIncremental({ scalars: input, glyphIds: input, flags: computeTextParagraphFlags(input, input, analysis()) }, input, input, () =>
                    analysis()
                )
            ).toEqual(computeTextParagraphFlags(input, input, analysis()));
        }
    });

    it("compares glyph IDs as well as scalars and keeps analysis lazy for a non-candidate window", () => {
        const input = scalars("A".repeat(1000));
        const old = { scalars: input, glyphIds: input, flags: new Uint8Array(1000) };
        const ids = input.slice();
        ids[500] = 99;
        const read = vi.fn(() => null);
        const evaluate = vi.fn((_scalars: Uint32Array, glyphs: Uint32Array, i: number) => (glyphs[i] === 99 ? 4 : 0));
        const flags = computeTextParagraphFlagsIncremental(old, input, ids, read, evaluate);
        expect(flags[500]).toBe(4);
        expect(evaluate).toHaveBeenCalledTimes(133);
        expect(read).not.toHaveBeenCalled();
        expect(() => computeTextParagraphFlagsIncremental(undefined, scalars("A A"), scalars("A A"), () => null)).toThrow(/analysis is unavailable/);
    });

    it("retains exact proof-exhaustion dependencies at R-1/R/R+1 and paragraph edges", () => {
        const face = font("Roboto-Regular.ttf");
        const original = "office ".repeat(70) + "A" + "\u0308".repeat(65) + " B " + "AV To ".repeat(70);
        const selected = selectComputeTextRun(original);
        const ids = mapComputeFontScalars(face, selected._scalars, (_i, error) => {
            throw error;
        });
        const safety = segmentation.analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face, selected)));
        const flags = computeTextParagraphFlags(selected._scalars, ids, safety);
        const space = original.indexOf(" B ");
        for (const distance of [-67, -66, -65, 65, 66, 67]) {
            const at = space + distance;
            const next = original.slice(0, at) + "A" + original.slice(at);
            const run = selectComputeTextRun(next);
            const mapped = mapComputeFontScalars(face, run._scalars, (_i, error) => {
                throw error;
            });
            expect(computeTextParagraphFlagsIncremental({ scalars: selected._scalars, glyphIds: ids, flags }, run._scalars, mapped, () => safety)).toEqual(
                computeTextParagraphFlags(run._scalars, mapped, safety)
            );
        }
    });

    it("avoids actual flag/proof evaluations outside the R66 middle edit window", () => {
        const before = scalars("A ".repeat(5000));
        const safe = vi.fn(() => true);
        const safety = analysis(safe);
        const initial = computeTextParagraphFlags(before, before, safety);
        const after = before.slice();
        after[5000] = 0x42;
        let evaluations = 0;
        const evaluate = (input: Uint32Array, ids: Uint32Array, i: number, read: () => SegmentAnalysis | null) => {
            evaluations++;
            if (input[i] !== 32) {
                return 0;
            }
            return i + 1 < input.length && input[i + 1] !== 32 && read()!.safeAfter(ids, i) ? 3 : 1;
        };
        safe.mockClear();
        const reused = computeTextParagraphFlagsIncremental({ scalars: before, glyphIds: before, flags: initial }, after, after, () => safety, evaluate);
        expect(evaluations).toBe(133);
        expect(safe).toHaveBeenCalledTimes(66);
        expect(reused).toEqual(computeTextParagraphFlags(after, after, safety));
        evaluations = 0;
        computeTextParagraphFlagsIncremental(undefined, after, after, () => safety, evaluate);
        expect(evaluations).toBe(10000);
    });

    it("uses a known unchanged span instead of comparing, and any lower bound of it yields identical flags", () => {
        const before = scalars("A ".repeat(5000));
        const safety = analysis(vi.fn(() => true));
        const initial = computeTextParagraphFlags(before, before, safety);
        const after = new Uint32Array(before.length + 1);
        after.set(before.subarray(0, 5000));
        after[5000] = 0x42;
        after.set(before.subarray(5000), 5001);
        const full = computeTextParagraphFlags(after, after, safety);
        const reads = { count: 0 };
        const counted = (array: Uint32Array) =>
            new Proxy(array, {
                get(target, key) {
                    if (typeof key === "string" && /^\d+$/.test(key)) {
                        reads.count++;
                    }
                    const value = Reflect.get(target, key, target);
                    return typeof value === "function" ? value.bind(target) : value;
                },
            });
        const previous = { scalars: counted(before), glyphIds: counted(before), flags: initial };
        let evaluations = 0;
        const evaluate = (input: Uint32Array, ids: Uint32Array, i: number, read: () => SegmentAnalysis | null) => {
            evaluations++;
            if (input[i] !== 32) {
                return 0;
            }
            return i + 1 < input.length && input[i + 1] !== 32 && read()!.safeAfter(ids, i) ? 3 : 1;
        };
        expect(computeTextParagraphFlagsIncremental(previous, after, after, () => safety, evaluate, [5000, 5000])).toEqual(full);
        expect(reads.count).toBe(0);
        expect(evaluations).toBe(133);
        for (const known of [
            [4000, 5000],
            [5000, 3500],
            [0, 0],
            [10, 4990],
        ] as const) {
            expect(computeTextParagraphFlagsIncremental(previous, after, after, () => safety, undefined, known)).toEqual(full);
        }
    });

    it("marks only ASCII spaces and only the last space in a followed run as break-after", () => {
        const input = scalars("  A   B  \u00a0\u2002");
        const safe = vi.fn((_ids: Uint32Array, _index: number) => true);
        expect([...computeTextParagraphFlags(input, input, analysis(safe))]).toEqual([1, 3, 0, 1, 1, 3, 0, 1, 3, 0, 0]);
        expect(safe.mock.calls.map((call) => call[1])).toEqual([1, 5, 8]);
        expect([...computeTextParagraphFlags(scalars("A   "), scalars("A   "), null)]).toEqual([0, 1, 1, 1]);
        expect(computeTextParagraphFlags(new Uint32Array(0), new Uint32Array(0), null)).toHaveLength(0);
    });

    it("passes nominal glyph ids, the space index and the aligned scalars, never shaped glyphs", () => {
        const input = scalars("A  B C");
        const glyphs = Uint32Array.of(10, 99, 99, 11, 99, 12);
        const safe = vi.fn((ids: Uint32Array, i: number) => ids[i] === 99 && i === 2);
        expect([...computeTextParagraphFlags(input, glyphs, analysis(safe))]).toEqual([0, 1, 3, 0, 1, 0]);
        expect(safe.mock.calls).toEqual([
            [glyphs, 2, input],
            [glyphs, 4, input],
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
    it.each(["maxBufferSize", "maxStorageBufferBindingSize"] as const)("admits required records when preferred slots exceed %s", (limit) => {
        const test = block();
        Object.defineProperty(test.data._device.limits, limit, { value: 4096 });
        const admitted = new Error("test stopped after storage admission");
        const reserve = paragraphStages.reserveComputeTextParagraphStage;
        const spy = vi.spyOn(paragraphStages, "reserveComputeTextParagraphStage").mockImplementation((...args) => {
            const stage = reserve(...args);
            expect(stage._words[1]).toBe(25);
            expect(stage._words[11]).toBe(3);
            throw admitted;
        });
        try {
            expect(() => updateComputeTextParagraphs(test.data, ["A".repeat(20)], { maxWidthPx: 100 })).toThrow(admitted.message);
            expect(spy).toHaveBeenCalledOnce();
            expect(test.storage._curveSets.size).toBe(0);
        } finally {
            test.close();
        }
    });

    it("reduces capacity headroom, but never the conservative required capacity", () => {
        const test = block();
        Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 1000 });
        const admitted = new Error("test stopped after storage admission");
        const reserve = paragraphStages.reserveComputeTextParagraphStage;
        vi.spyOn(paragraphStages, "reserveComputeTextParagraphStage").mockImplementation((...args) => {
            const stage = reserve(...args);
            expect(stage._words[1]).toBe(20);
            expect(stage._words[11]).toBe(1);
            throw admitted;
        });
        try {
            expect(() => updateComputeTextParagraphs(test.data, ["A".repeat(20)], { maxWidthPx: 100 })).toThrow(admitted.message);
        } finally {
            test.close();
        }
    });

    it("reports exact required allocation and limits before atlas mutation and clears drawing", () => {
        const test = block();
        Object.defineProperty(test.data._device.limits, "maxStorageBufferBindingSize", { value: 959 });
        try {
            expect(() => updateComputeTextParagraphs(test.data, ["A".repeat(20)], { maxWidthPx: 100 })).toThrow(
                "updateComputeTextParagraphs: layout lines buffer of 960 bytes exceeds device storage limits " +
                    "(maxBufferSize=1048576, maxStorageBufferBindingSize=959, pool=1, capacity=20)"
            );
            expect(test.storage._curveSets.size).toBe(0);
            expect(test.data.status).toBe("ready");
            const writes = vi.mocked(test.data._device.queue.writeBuffer).mock.calls;
            const indirect = writes.filter((call) => call[0] === test.data._indirect._buffer).at(-1)!;
            expect([...new Uint32Array(indirect[2] as ArrayBuffer, indirect[3], 4)]).toEqual([6, 0, 0, 0]);
        } finally {
            test.close();
        }
    });

    it("loads the Indic profile for Devanagari paragraphs before allocation, then re-enters", async () => {
        const { data, close } = block("NotoSansDevanagari-Regular.ttf");
        const importer = vi.fn(async () => installComputeTextIndicProfile());
        profiles.installComputeTextProfileLoader("indic", importer);
        const allocations = vi.spyOn(data._device, "createBuffer");
        allocations.mockClear();
        const version = data._revision;
        const admitted = new Error("test stopped after storage admission");
        vi.spyOn(paragraphStages, "reserveComputeTextParagraphStage").mockImplementation(() => {
            throw admitted;
        });
        try {
            const pending = updateComputeTextParagraphs(data, ["क"], { maxWidthPx: 200 });
            expect(allocations).not.toHaveBeenCalled();
            expect(data._revision).toBe(version);
            expect(data.status).toBe("ready");
            await expect(pending).rejects.toThrow(`updateComputeTextParagraphs: ${admitted.message}`);
            expect(importer).toHaveBeenCalledOnce();
            expect(profiles.profileSupport("indic")?.paragraph).toBeDefined();
            expect(() => updateComputeTextParagraphs(data, ["क"], { maxWidthPx: 200 })).toThrow(admitted.message);
            expect(importer).toHaveBeenCalledOnce();
        } finally {
            allocations.mockRestore();
            close();
        }
    });

    it.each(["\n", "\r", "\t", "\u2028", "\u2029"])("rejects paragraph controls synchronously: %j", (control) => {
        expect(() => updateComputeTextParagraphs(undefined!, ["A" + control + "B"], { maxWidthPx: 100 })).toThrow(/updateComputeTextParagraphs:.*line breaks or tabs/);
    });

    it.each(["plain ASCII 123 !", "é", "🙂"])("admits text through the paragraph scan before layout validation: %j", (text) => {
        expect(() => updateComputeTextParagraphs(undefined!, [text], { maxWidthPx: 0 })).toThrow("maxWidthPx must be positive");
    });

    it.each([
        { paragraphs: ["A\tक"], message: "paragraph 0 must not contain line breaks or tabs" },
        { paragraphs: ["A", "क", "B\t"], message: "paragraph 2 must not contain line breaks or tabs" },
        { paragraphs: ["A", "B\t", "क"], message: "paragraph 1 must not contain line breaks or tabs" },
    ])("preserves control priority and the first offending paragraph: $paragraphs", ({ paragraphs, message }) => {
        expect(() => updateComputeTextParagraphs(undefined!, paragraphs, { maxWidthPx: 100 })).toThrow(message);
    });

    it("does not recheck Devanagari in the unchanged prefix or suffix", () => {
        const test = block();
        const data = test.data;
        const cache: ComputeTextParagraphCache = {
            _paragraphs: ["क", "old", "क"],
            _order: Uint32Array.of(0, 1, 2),
            _capacity: 4,
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
        const words = new Uint32Array(12);
        words[1] = cache._capacity;
        words[11] = cache._pool;
        const spy = vi
            .spyOn(paragraphStages, "getComputeTextParagraphStage")
            .mockReturnValue({ _cache: cache, _words: words } as ReturnType<typeof paragraphStages.getComputeTextParagraphStage>);
        try {
            expect(computeTextParagraphCacheValid(data, cache)).toBe(true);
            expect(() => updateComputeTextParagraphs(data, ["क", "new", "क"], { maxWidthPx: 0 })).toThrow("maxWidthPx must be positive");
        } finally {
            spy.mockRestore();
            test.close();
        }
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
        vi.spyOn(segmentation, "computeTextParagraphSpaceProof").mockReturnValue(null);
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
                    kind === "lines" ? /updateComputeTextParagraphs: layout lines buffer.*storage limits/ : /updateComputeTextParagraphs:.*workgroup limits/
                );
                expect(test.storage._curveSets.size).toBe(0);
                expect(test.data.status).toBe("ready");
            } finally {
                test.close();
            }
        }
    });
});

describe("RM-1 exact paragraph admission", () => {
    // With the profile loaded, Devanagari admission is synchronous like every other script.
    beforeAll(() => profiles.ensureComputeTextProfile("indic"));
    const sentinel = new Error("RM1_ADMISSION_PASSED");
    const admissionData = {
        get _chunkBudgets(): never {
            throw sentinel;
        },
    } as unknown as ComputeTextData;
    function outcome(action: () => unknown): string {
        try {
            action();
            throw new Error("admission fixture unexpectedly reached submission");
        } catch (error) {
            if (!(error instanceof Error)) {
                throw error;
            }
            return error.message.replace(/^updateComputeTextParagraphs:\s*/u, "");
        }
    }
    function production(paragraphs: readonly string[], layout: ComputeTextParagraphLayout = { maxWidthPx: 100 }): string {
        return outcome(() => updateComputeTextParagraphs(admissionData, paragraphs, layout));
    }
    function c129(paragraphs: readonly string[]): string {
        const prefix = 0,
            suffix = 0;
        return outcome(() => {
            // C129 admission loop, verbatim except for the Devanagari rejection CTF-03 removed; only the
            // surrounding fixture supplies an empty prior cache and a sentinel after successful admission.
            for (let p = 0; p < paragraphs.length; p++) {
                const text = paragraphs[p];
                if (typeof text !== "string") {
                    throw new Error(`paragraph ${p} text must be a string`);
                }
                if (p >= prefix && p < paragraphs.length - suffix && /[\t\n\r\u0080-\u{10ffff}]/u.test(text)) {
                    if (/[\n\r\t\u2028\u2029]/u.test(text)) {
                        throw new Error(`paragraph ${p} must not contain line breaks or tabs`);
                    }
                }
            }
            throw sentinel;
        });
    }
    it.each([
        { paragraphs: [123, "क"], message: "paragraph 0 text must be a string" },
        { paragraphs: ["A", "B\t", "क"], message: "paragraph 1 must not contain line breaks or tabs" },
        { paragraphs: ["A", "क", "B\t"], message: "paragraph 2 must not contain line breaks or tabs" },
        { paragraphs: ["\u2028"], message: "paragraph 0 must not contain line breaks or tabs" },
        { paragraphs: ["\u2029"], message: "paragraph 0 must not contain line breaks or tabs" },
        ...["é", "😀", "\ud800", "\udc00", "", "\u0900", "क"].map((text) => ({ paragraphs: [text], message: sentinel.message })),
    ])("preserves exact first-offender admission: $paragraphs", ({ paragraphs, message }) => {
        expect(production(paragraphs as readonly string[])).toBe(message);
    });
    it("reports invalid layout width for Devanagari paragraphs before any profile load", () => {
        expect(production(["क"], { maxWidthPx: 0 })).toBe("maxWidthPx must be positive or Infinity and representable as f32");
    });
    it("does not scan a tab in a cached clean prefix on the next positional update", () => {
        const test = block();
        const data = test.data;
        const cache: ComputeTextParagraphCache = {
            _paragraphs: ["A\t", "old"],
            _order: Uint32Array.of(0, 1),
            _capacity: 4,
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
        const words = new Uint32Array(12);
        words[1] = cache._capacity;
        words[11] = cache._pool;
        vi.spyOn(paragraphStages, "getComputeTextParagraphStage").mockReturnValue({ _cache: cache, _words: words } as ReturnType<
            typeof paragraphStages.getComputeTextParagraphStage
        >);
        try {
            expect(computeTextParagraphCacheValid(data, cache)).toBe(true);
            expect(outcome(() => updateComputeTextParagraphs(data, ["A\t", "new"], { maxWidthPx: 0 }))).toBe("maxWidthPx must be positive or Infinity and representable as f32");
        } finally {
            test.close();
        }
    });
    it("matches verbatim C129 over all code points, lone surrogates and 100k seeded mixed strings", () => {
        let checked = 0;
        function equal(text: string): void {
            const actual = production([text]);
            const expected = c129([text]);
            if (actual !== expected) {
                throw new Error(`RM1 admission mismatch ${JSON.stringify(text)}: ${actual} !== ${expected}`);
            }
            checked++;
        }
        for (let cp = 0; cp <= 0x10ffff; cp++) {
            equal(String.fromCodePoint(cp));
        }
        for (let unit = 0xd800; unit <= 0xdfff; unit++) {
            equal(String.fromCharCode(unit));
        }
        let state = 0xa11ce;
        function next(): number {
            state ^= state << 13;
            state ^= state >>> 17;
            state ^= state << 5;
            return (state >>> 0) / 4294967296;
        }
        for (let sample = 0; sample < 100000; sample++) {
            const length = Math.floor(next() * 33);
            let text = "";
            while (text.length < length) {
                const category = Math.floor(next() * 6);
                const cp =
                    category === 0
                        ? Math.floor(next() * 128)
                        : category === 1
                          ? [9, 10, 13][Math.floor(next() * 3)]!
                          : category === 2
                            ? 128 + Math.floor(next() * 128)
                            : category === 3
                              ? Math.floor(next() * 65536)
                              : category === 4
                                ? 0x10000 + Math.floor(next() * 0x100000)
                                : 0xd800 + Math.floor(next() * 2048);
                const piece = String.fromCodePoint(cp);
                text += piece.length > length - text.length ? piece[0]! : piece;
            }
            equal(text);
        }
        expect(checked).toBe(1216160);
    }, 120_000);
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
            // measure, the five split-only stages (idle at this capacity), prefix, place.
            expect(dispatches.map((dispatch) => [dispatch._x, dispatch._y, dispatch.enabled])).toEqual([
                [4, 2, true],
                ...Array.from({ length: 5 }, () => [0, 1, false]),
                [1, 1, true],
                [4, 2, true],
            ]);
            words[0] = 1;
            words[9] = 0;
            stage.prepare(test.data._engine, 0, 1);
            expect(dispatches.map((dispatch) => [dispatch._x, dispatch._y, dispatch.enabled])).toEqual([
                ...Array.from({ length: 6 }, () => [0, 1, false]),
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
            expect(() => createComputeTextParagraphStage({}, test.data._engine, words)).toThrow(/layout lines buffer of 960 bytes exceeds device storage limits/);
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
