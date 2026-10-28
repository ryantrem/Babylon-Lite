import { describe, expect, it } from "vitest";
import { fixture } from "../compute/fixtures";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { checkComputeTextPlanBounds, checkComputeTextShapeBounds, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import type { ComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";

const high = { maxBufferSize: 2 ** 40, maxStorageBufferBindingSize: 2 ** 40, maxComputeWorkgroupsPerDimension: 65535 };
function limits(values: Partial<typeof high> = {}) {
    const actual = { ...high, ...values };
    const reads: string[] = [];
    // The guard uses these three platform attributes; values stay fixed for the check.
    const deviceLimits = {
        get maxBufferSize() {
            reads.push("buffer");
            return actual.maxBufferSize;
        },
        get maxStorageBufferBindingSize() {
            reads.push("storage");
            return actual.maxStorageBufferBindingSize;
        },
        get maxComputeWorkgroupsPerDimension() {
            reads.push("workgroups");
            return actual.maxComputeWorkgroupsPerDimension;
        },
    } as GPUSupportedLimits;
    return { deviceLimits, reads };
}
function input() {
    const plan = createComputeTextPlan(fixture("Roboto-Regular.ttf").font, selectComputeTextRun("A"));
    const runs = [{ scalars: Uint32Array.of(65), glyphIds: Uint32Array.of(36) }];
    return { plan, runs };
}
function empty() {
    const { plan } = input();
    return {
        ...prepareComputeTextShapeWords(plan, []),
        capacity: 0,
        params: new Uint32Array(8),
        scratchBytes: 0,
        programWords: new Uint32Array(),
        tables: new Uint32Array(),
        runs: new Uint32Array(),
    } satisfies ComputeTextShapeWords;
}
const capacityError = "shapeComputeText: capacity exceeds u32 record addressing";
const runError = "shapeComputeText: run count exceeds device maxComputeWorkgroupsPerDimension";
const bufferError = (bytes: number) => `shapeComputeText: buffer of ${bytes} bytes exceeds device maxBufferSize`;
const storageError = (bytes: number) => `shapeComputeText: binding of ${bytes} bytes exceeds device maxStorageBufferBindingSize`;

describe("per-check stable GPU limit reads", () => {
    it("reads each limit once per prepared check, without caching across calls", () => {
        const observed = limits();
        const prepared = empty();
        checkComputeTextShapeBounds(observed.deviceLimits, prepared);
        expect(observed.reads).toEqual(["workgroups", "buffer", "storage"]);
        checkComputeTextShapeBounds(observed.deviceLimits, prepared);
        expect(observed.reads).toEqual(["workgroups", "buffer", "storage", "workgroups", "buffer", "storage"]);
    });
    it("reuses the plan's program-selection limits and reads workgroups only after capacity", () => {
        const { plan, runs } = input();
        const observed = limits();
        checkComputeTextPlanBounds(observed.deviceLimits, plan, runs, 4, 0, 1);
        expect(observed.reads).toEqual(["buffer", "storage", "workgroups"]);
        checkComputeTextPlanBounds(observed.deviceLimits, plan, runs, 4, 0, 1);
        expect(observed.reads).toEqual(["buffer", "storage", "workgroups", "buffer", "storage", "workgroups"]);
    });
    it("keeps capacity before workgroups before invalid sizes, including prepared-property evaluation", () => {
        const observed = limits({ maxComputeWorkgroupsPerDimension: 0 });
        const prepared = empty();
        prepared.capacity = 0x20000000;
        prepared.params[0] = 1;
        prepared.scratchBytes = NaN;
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrowError(new Error(capacityError));
        expect(observed.reads).toEqual([]);
        const failure = new Error("opaque prepared size");
        Object.defineProperty(prepared, "scratchBytes", {
            get() {
                throw failure;
            },
        });
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrow(failure);
        expect(observed.reads).toEqual([]);
        const other = empty();
        other.params[0] = 1;
        other.scratchBytes = NaN;
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, other)).toThrowError(new Error(runError));
        expect(observed.reads).toEqual(["workgroups"]);
    });
    it("keeps the plan's first two reads even for capacity errors, then workgroup precedence", () => {
        const { plan, runs } = input();
        const observed = limits({ maxComputeWorkgroupsPerDimension: 0 });
        expect(() => checkComputeTextPlanBounds(observed.deviceLimits, plan, runs, 0x20000000, Infinity, NaN)).toThrowError(new Error(capacityError));
        expect(observed.reads).toEqual(["buffer", "storage"]);
        observed.reads.length = 0;
        expect(() => checkComputeTextPlanBounds(observed.deviceLimits, plan, runs, 4, Infinity, NaN)).toThrowError(new Error(runError));
        expect(observed.reads).toEqual(["buffer", "storage", "workgroups"]);
    });
    it.each([NaN, 0.1])("does not read buffer/storage for a first unsafe size: %s", (capacity) => {
        const prepared = empty();
        prepared.capacity = capacity;
        prepared.params[0] = 1;
        const bytes = 4 * (4 + capacity * 6);
        const observed = limits();
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrowError(new Error(bufferError(bytes)));
        expect(observed.reads).toEqual(["workgroups"]);
    });
    it("rejects an unsafe first allocation without reading either buffer limit", () => {
        const prepared = empty();
        prepared.capacity = 1 << 17;
        prepared.params[6] = 0xffffffff;
        const bytes = prepared.params[6]! * (4 + prepared.capacity * 6) * 4;
        expect(Number.isSafeInteger(bytes)).toBe(false);
        const observed = limits();
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrowError(new Error(bufferError(bytes)));
        expect(observed.reads).toEqual(["workgroups"]);
    });
    it.each([NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, 0.5])("preserves later unsafe-size precedence and one-time reads: %s", (bytes) => {
        const prepared = empty();
        prepared.scratchBytes = bytes;
        const observed = limits();
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrowError(new Error(bufferError(bytes)));
        expect(observed.reads).toEqual(["workgroups", "buffer", "storage"]);
    });
    it("does not eagerly read storage after the first buffer failure", () => {
        const prepared = empty();
        prepared.params[6] = 4;
        const observed = limits({ maxBufferSize: 63, maxStorageBufferBindingSize: 0 });
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrowError(new Error(bufferError(64)));
        expect(observed.reads).toEqual(["workgroups", "buffer"]);
    });
    it.each([0, 1, 2, 3, 4, 5])("preserves exact buffer/storage boundary and error at size entry %s", (index) => {
        const prepared = empty();
        if (index === 0) prepared.params[6] = 4;
        if (index === 1) prepared.scratchBytes = 64;
        if (index === 2) prepared.params = new Uint32Array(16);
        if (index === 3) prepared.programWords = new Uint32Array(16);
        if (index === 4) prepared.tables = new Uint32Array(16);
        if (index === 5) prepared.runs = new Uint32Array(16);
        for (const delta of [-1, 0, 1]) {
            for (const field of ["maxBufferSize", "maxStorageBufferBindingSize"] as const) {
                const observed = limits({ [field]: 64 + delta });
                if (delta < 0) {
                    expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).toThrowError(
                        new Error(field === "maxBufferSize" ? bufferError(64) : storageError(64))
                    );
                } else {
                    expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).not.toThrow();
                }
                expect(observed.reads).toEqual(index === 0 && field === "maxBufferSize" && delta < 0 ? ["workgroups", "buffer"] : ["workgroups", "buffer", "storage"]);
            }
        }
    });
    it("preserves accepted negative integer sizes, zero/exact workgroup bounds, and later NaN scalar rejection", () => {
        const prepared = empty();
        prepared.scratchBytes = -4;
        const observed = limits({ maxComputeWorkgroupsPerDimension: 0 });
        expect(() => checkComputeTextShapeBounds(observed.deviceLimits, prepared)).not.toThrow();
        expect(observed.reads).toEqual(["workgroups", "buffer", "storage"]);
        const { plan, runs } = input();
        const at = limits({ maxComputeWorkgroupsPerDimension: 1 });
        expect(() => checkComputeTextPlanBounds(at.deviceLimits, plan, runs, 4, 0, 1)).not.toThrow();
        expect(at.reads).toEqual(["buffer", "storage", "workgroups"]);
        at.reads.length = 0;
        expect(() => checkComputeTextPlanBounds(at.deviceLimits, plan, runs, 4, 0, NaN)).toThrowError(new Error(bufferError(NaN)));
        expect(at.reads).toEqual(["buffer", "storage", "workgroups"]);
    });
});
