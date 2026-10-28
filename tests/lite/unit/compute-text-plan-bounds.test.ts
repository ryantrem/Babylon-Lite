import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { fixture } from "../compute/fixtures";
import type { ComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { createComputeTextPlan, ensureComputeTextProfile } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import * as profiles from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { checkComputeTextPlanBounds, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { segmentComputeTextRuns } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import * as segmentation from "../../../packages/babylon-lite/src/text/compute-text-segment";
import * as shapes from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import * as resident from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import * as textures from "../../../packages/babylon-lite/src/text/_gpu/text-textures";
import * as outlines from "../../../packages/babylon-lite/src/text/compute-glyph-extraction";
import { computeTextDrawBufferSizes, createComputeTextData, disposeComputeTextData, submitComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import type { ResidentShapeGroup, ResidentPackStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";

type Runs = readonly { readonly scalars: Uint32Array; readonly glyphIds: Uint32Array }[];
type Bounds = (limits: GPUSupportedLimits, plan: ComputeTextPlan, runs: Runs, capacity: number, outputRun: number, scalarCount: number) => void;
const bounds = checkComputeTextPlanBounds as Bounds;
const high = { maxBufferSize: 2 ** 40, maxStorageBufferBindingSize: 2 ** 40, maxComputeWorkgroupsPerDimension: 65535 } as GPUSupportedLimits;
const layout = { headerWords: 4, recordWords: 6 };
const paragraph = "The quick brown fox jumps over the lazy dog. Every letter belongs to a flowing English paragraph with ordinary spaces and punctuation. ";

function group(file: Parameters<typeof fixture>[0], text: string, outputRun = 0): ResidentShapeGroup {
    const font = fixture(file).font;
    const selection = selectComputeTextRun(text);
    return {
        plan: createComputeTextPlan(font, selection),
        outputRun,
        runs: [
            {
                scalars: selection._scalars,
                glyphIds: mapComputeFontScalars(font, selection._scalars, (_index, error) => {
                    throw error;
                }),
            },
        ],
    };
}
function total(runs: readonly { readonly scalars: Uint32Array }[]): number {
    let count = 0;
    for (const run of runs) {
        count += run.scalars.length;
    }
    return count;
}
function c129RunWords(runs: readonly { readonly scalars: Uint32Array }[], id: string): number {
    return runs.length + runs.reduce((sum, run) => sum + 1 + run.scalars.length * (id === "indic" ? 2 : 1), 0);
}
function block(file: Parameters<typeof fixture>[0] = "Roboto-Regular.ttf") {
    const device = {
        limits: { ...high },
        lost: new Promise<GPUDeviceLostInfo>(() => undefined),
        createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
            getMappedRange: () => new ArrayBuffer(desc.size),
            unmap: vi.fn(),
            destroy: vi.fn(),
        })),
        queue: { writeBuffer: vi.fn() },
    };
    const engine = { _device: device } as unknown as EngineContext;
    const storage = createGlyphStorage(),
        data = createComputeTextData(engine, fixture(file).font, storage, "rm1-unit");
    return {
        data,
        storage,
        device,
        close() {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        },
    };
}
const stage: ResidentPackStage = { version: 0, prepare() {}, dispatches: () => [], release() {} };
function mockExecution() {
    const execute = vi.spyOn(resident, "executeResidentComputeText").mockResolvedValue({ status: Promise.resolve(0) });
    vi.spyOn(textures, "ensureSharedAtlasGpu").mockReturnValue({
        _rebuilt: false,
        _gpu: {} as ReturnType<typeof textures.ensureSharedAtlasGpu>["_gpu"],
    });
    return execute;
}

beforeAll(async () => {
    await ensureComputeTextProfile("arabic");
    await ensureComputeTextProfile("indic");
});
afterEach(() => vi.restoreAllMocks());

describe("RM-1 logical scalar totals and six preflight sizes", () => {
    it("preserves real segmented Latin group partitions, plan identity and logical totals alongside Indic/Arabic groups", () => {
        const groups = [group("Inter.ttf", paragraph.repeat(8), 0), group("NotoSansArabic-Regular.ttf", "كتب", 1), group("NotoSansDevanagari-Regular.ttf", "क्ष", 2)];
        const segments = segmentComputeTextRuns(groups, high, layout);
        expect(segments).not.toBeNull();
        expect(segments!.groups).toHaveLength(groups.length);
        groups.forEach((input, index) => {
            const actual = segments!.groups[index]!;
            expect(actual.plan).toBe(input.plan);
            expect(total(actual.runs)).toBe(total(input.runs));
            const k = input.plan._profile === "indic" ? 2 : 1;
            expect(2 * actual.runs.length + k * total(input.runs)).toBe(c129RunWords(actual.runs, input.plan._profile));
        });
        expect(segments!.groups[0]!.runs.length).toBeGreaterThan(groups[0]!.runs.length);
    });
    it("keeps the unsegmented null path, empty arrays and zero-length run arithmetic", () => {
        const input = group("Inter.ttf", "A");
        expect(segmentComputeTextRuns([input], high, layout)).toBeNull();
        for (const runs of [[], [{ scalars: new Uint32Array(), glyphIds: new Uint32Array() }], input.runs]) {
            for (const id of ["default", "indic"]) {
                expect(2 * runs.length + (id === "indic" ? 2 : 1) * total(runs)).toBe(c129RunWords(runs, id));
            }
        }
    });
    it("checks every exact C129 size and its minus-one binding boundary in unchanged order", () => {
        const latin = group("Inter.ttf", paragraph.repeat(5));
        const segmented = segmentComputeTextRuns([latin], high, layout)!;
        const cases = [
            { ...latin, runs: segmented.groups[0]!.runs },
            group("NotoSansDevanagari-Regular.ttf", "क्ष"),
            { ...latin, runs: [] },
            { ...latin, runs: [{ scalars: new Uint32Array(), glyphIds: new Uint32Array() }] },
            group("Inter.ttf", "A"),
        ];
        for (const input of cases) {
            const capacity = Math.max(1, ...input.runs.map((r) => r.scalars.length)) * 4;
            const outputRun = 3;
            const prepared = prepareComputeTextShapeWords(input.plan, input.runs, outputRun, capacity, high);
            const expected = [
                (outputRun + input.runs.length) * (4 + capacity * 6) * 4,
                prepared.scratchBytes,
                32,
                prepared.programWords.byteLength,
                prepared.tables.byteLength,
                c129RunWords(input.runs, input.plan._profile) * 4,
            ];
            const checked = vi.spyOn(Number, "isSafeInteger");
            let actual: unknown[];
            try {
                bounds(high, input.plan, input.runs, capacity, outputRun, total(input.runs));
                actual = checked.mock.calls.map(([bytes]) => bytes);
            } finally {
                checked.mockRestore();
            }
            expect(actual).toEqual(expected);
            for (const bytes of expected) {
                for (const delta of [-1, 0, 1]) {
                    // Platform limits are stable. A smaller earlier allocation
                    // cannot mask a larger one, and the optional compact program
                    // must be selected using this same binding limit.
                    const limits = { ...high, maxStorageBufferBindingSize: bytes + delta };
                    const selected = prepareComputeTextShapeWords(input.plan, input.runs, outputRun, capacity, limits);
                    const sizes = [...expected];
                    sizes[3] = selected.programWords.byteLength;
                    const firstFailure = sizes.find((size) => size > limits.maxStorageBufferBindingSize);
                    if (firstFailure === undefined) {
                        expect(() => bounds(limits, input.plan, input.runs, capacity, outputRun, total(input.runs))).not.toThrow();
                    } else {
                        expect(() => bounds(limits, input.plan, input.runs, capacity, outputRun, total(input.runs))).toThrowError(
                            new Error(`shapeComputeText: binding of ${firstFailure} bytes exceeds device maxStorageBufferBindingSize`)
                        );
                    }
                }
            }
        }
    });
    it("does not read segmented scalar vectors in the preflight (regression: C129 reads each segment)", () => {
        const input = group("Inter.ttf", "A");
        let reads = 0;
        const runs = Array.from({ length: 7 }, () => ({
            get scalars() {
                reads++;
                return Uint32Array.of(65);
            },
            glyphIds: Uint32Array.of(1),
        }));
        bounds(high, input.plan, runs, 4, 0, 7);
        expect(reads).toBe(0);
    });
    it("rewrites the actual stable-limit bounds error with the same cause before atlas mutation", async () => {
        for (const delta of [0, -1, 1]) {
            const test = block();
            mockExecution();
            const extraction = vi.spyOn(outlines, "extractComputeGlyphCurves");
            let expected = 0;
            let submitted: Promise<unknown> | undefined;
            try {
                function submit() {
                    return submitComputeTextData(test.data, ["A"], (groups, needed) => {
                        const group = groups[0]!;
                        const base = prepareComputeTextShapeWords(group.plan, group.runs, 0, needed, { ...high, maxStorageBufferBindingSize: 1 });
                        expected = Math.max(
                            (4 + needed * 6) * 4,
                            base.scratchBytes,
                            32,
                            base.programWords.byteLength,
                            base.tables.byteLength,
                            c129RunWords(group.runs, group.plan._profile) * 4
                        );
                        expect(expected).toBeGreaterThan(Math.max(...computeTextDrawBufferSizes(1, needed).map(([, bytes]) => bytes)));
                        test.device.limits.maxStorageBufferBindingSize = expected + delta;
                        return stage;
                    });
                }
                if (delta < 0) {
                    let failure: unknown;
                    try {
                        submit();
                    } catch (error) {
                        failure = error;
                    }
                    expect(failure).toBeInstanceOf(Error);
                    expect((failure as Error).message).toBe(`updateComputeTextData: binding of ${expected} bytes exceeds device maxStorageBufferBindingSize`);
                    expect((failure as Error).cause).toBeInstanceOf(Error);
                    expect(((failure as Error).cause as Error).message).toBe(`shapeComputeText: binding of ${expected} bytes exceeds device maxStorageBufferBindingSize`);
                    expect(extraction).not.toHaveBeenCalled();
                    expect(test.storage._curveSets.size).toBe(0);
                } else {
                    submitted = submit();
                    const update = (await submitted) as Awaited<ReturnType<typeof submitComputeTextData>>;
                    await update.completed;
                }
            } finally {
                test.close();
                vi.restoreAllMocks();
            }
        }
    });
});

describe("RM-1 grouped plan reuse and draw preflight", () => {
    it("checks draw allocation sizes only once per submission (regression: C129 repeats the lexical check)", async () => {
        const test = block();
        mockExecution();
        let drawSizeReads = 0;
        Object.defineProperty(test.device.limits, "maxBufferSize", {
            configurable: true,
            get() {
                // Same-module lexical calls do not go through an export spy.
                // Observe actual work at the existing typed limits seam instead:
                // each successful draw check reads this once for each of its
                // two size entries. Other preflight limit reads do not count.
                if (new Error().stack?.includes("checkDrawBufferSizes")) {
                    drawSizeReads++;
                }
                return high.maxBufferSize;
            },
        });
        try {
            const update = await submitComputeTextData(test.data, ["A"]);
            await update.completed;
            expect(drawSizeReads).toBe(2);
        } finally {
            test.close();
        }
    });
    it("keeps actual submitted segmentation aligned with the pre-atlas logical groups", async () => {
        const test = block("Inter.ttf");
        mockExecution();
        const segment = vi.spyOn(segmentation, "segmentComputeTextRuns");
        const checked = vi.spyOn(shapes, "checkComputeTextPlanBounds");
        let logical: readonly ResidentShapeGroup[] = [];
        try {
            const update = await submitComputeTextData(test.data, [paragraph.repeat(8), "", paragraph.repeat(3)], (groups) => {
                logical = groups;
                return stage;
            });
            await update.completed;
            const actual = segment.mock.results[0]!.value as NonNullable<ReturnType<typeof segmentComputeTextRuns>>;
            expect(actual).not.toBeNull();
            expect(actual.groups).toHaveLength(logical.length);
            actual.groups.forEach((g, index) => {
                expect(g.plan).toBe(logical[index]!.plan);
                expect(total(g.runs)).toBe(total(logical[index]!.runs));
                expect(checked.mock.calls[index]![1]).toBe(g.plan);
                expect(checked.mock.calls[index]![2]).toBe(g.runs);
            });
        } finally {
            test.close();
        }
    });
    it("preserves draw growth labels, exact byte sizes, order and instance replacement", async () => {
        const test = block();
        mockExecution();
        const originalInstances = test.data._instances;
        let capacity = 0;
        test.device.createBuffer.mockClear();
        try {
            const update = await submitComputeTextData(test.data, ["A".repeat(12)], (_groups, needed) => {
                capacity = needed;
                return stage;
            });
            await update.completed;
            const sizes = computeTextDrawBufferSizes(1, capacity);
            const allocated = test.device.createBuffer.mock.calls.map(([desc]) => desc).filter((desc) => ["compute-text-shaped", "compute-text-instances"].includes(desc.label!));
            expect(allocated.map((desc) => [desc.label, desc.size])).toEqual([
                ["compute-text-shaped", sizes[0]![1]],
                ["compute-text-instances", sizes[1]![1]],
            ]);
            expect(test.data._shaped.byteLength).toBe(sizes[0]![1]);
            expect(test.data._instances.byteLength).toBe(sizes[1]![1]);
            expect(test.data._instances).not.toBe(originalInstances);
        } finally {
            test.close();
        }
    });
    it.each([
        ["Roboto-Regular.ttf", ["A", "B"]],
        ["NotoSansHebrew-Regular.ttf", ["שלום"]],
        ["NotoSansArabic-Regular.ttf", ["كتب"]],
        ["NotoSansDevanagari-Regular.ttf", ["क्ष"]],
        ["Roboto-Regular.ttf", ["", "A"]],
        ["NotoSansArabic-Regular.ttf", ["", "كتب", "", "ب"]],
    ] as const)("preserves cached plan identities and empty-first fallback: %s / %j", async (file, texts) => {
        const test = block(file);
        mockExecution();
        let captured: readonly ResidentShapeGroup[] = [];
        try {
            const update = await submitComputeTextData(test.data, texts, (groups) => {
                captured = groups;
                return stage;
            });
            await update.completed;
            for (const g of captured) {
                expect(g.plan).toBe(createComputeTextPlan(test.data._font, selectComputeTextRun(texts[g.outputRun]!)));
            }
            expect(captured.flatMap((g) => g.runs)).toHaveLength(texts.length);
        } finally {
            test.close();
        }
    });
    it("creates no redundant plan for a nonempty-first group (regression: C129 calls twice)", async () => {
        const test = block();
        mockExecution();
        const calls = vi.spyOn(profiles, "createComputeTextPlan");
        try {
            const update = await submitComputeTextData(test.data, ["A"]);
            await update.completed;
            expect(calls).toHaveBeenCalledTimes(1);
        } finally {
            test.close();
        }
    });
    it("rejects over-limit draw allocations before closure installation or buffer replacement", () => {
        const test = block();
        mockExecution();
        const extraction = vi.spyOn(outlines, "extractComputeGlyphCurves");
        const shaped = test.data._shaped,
            instances = test.data._instances;
        try {
            Object.defineProperty(test.device.limits, "maxStorageBufferBindingSize", { value: 1, configurable: true });
            expect(() => submitComputeTextData(test.data, ["A"])).toThrow(/updateComputeTextData: draw buffer of \d+ bytes exceeds device storage limits/);
            expect(extraction).not.toHaveBeenCalled();
            expect(test.storage._curveSets.size).toBe(0);
            expect(test.data._shaped).toBe(shaped);
            expect(test.data._instances).toBe(instances);
        } finally {
            test.close();
        }
    });
    it.each([
        { capacity: 100, pool: 1 },
        { capacity: 1, pool: 100 },
    ])("keeps exact pre-atlas draw failure for capacity/pool $capacity/$pool", ({ capacity, pool }) => {
        const test = block();
        mockExecution();
        const extraction = vi.spyOn(outlines, "extractComputeGlyphCurves");
        const shaped = test.data._shaped,
            instances = test.data._instances;
        try {
            Object.defineProperty(test.device.limits, "maxStorageBufferBindingSize", { value: 1000, configurable: true });
            const bytes = Math.max(1, pool) * (4 + capacity * 6) * 4;
            expect(() => submitComputeTextData(test.data, ["A"], undefined, { capacity, pool, runBase: 0 })).toThrow(
                `updateComputeTextData: draw buffer of ${bytes} bytes exceeds device storage limits`
            );
            expect(extraction).not.toHaveBeenCalled();
            expect(test.storage._curveSets.size).toBe(0);
            expect(test.data._shaped).toBe(shaped);
            expect(test.data._instances).toBe(instances);
        } finally {
            test.close();
        }
    });
});
