import { createHash } from "node:crypto";
import { beforeAll, describe, expect, it } from "vitest";
import {
    createComputeTextPlan,
    ensureComputeTextProfile,
    installComputeTextProfile,
    installComputeTextProfileLoader,
    profileSupport,
} from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import {
    COMPUTE_TEXT_KERNEL_SOURCE,
    computeTextScratchCapacity,
    computeTextVariantKey,
    computeTextVariantTier,
    kernelSource,
    prepareComputeTextShapeWords,
    checkComputeTextShapeBounds,
    ATTACH_SHIFT,
    RECORD_CLASS_MASK,
} from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { fixture } from "../compute/fixtures";
import type { ComputeTextKernelVariant } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";

function workgroupBytes(source: string): number {
    const size = (type: string): number => {
        if (["u32", "i32", "atomic<u32>"].includes(type)) {
            return 4;
        }
        if (type === "vec2u") {
            return 8;
        }
        const array = /^array<(.+),(\d+)>$/.exec(type);
        if (!array) {
            throw new Error(`Unknown test workgroup declaration: ${type}`);
        }
        return size(array[1]!) * Number(array[2]);
    };
    const declarations = [...source.matchAll(/var<workgroup>\s+\w+\s*:\s*([^;]+);/g)];
    expect(declarations.length).toBeGreaterThan(10);
    return declarations.reduce((sum, [, type]) => sum + size(type!.replace(/\s/g, "")), 0);
}

describe("shared compute text profiles and memory variants", () => {
    beforeAll(async () => ensureComputeTextProfile("arabic"));
    it("checkComputeTextShapeBounds admits capacity 65536 through 2^20 when device limits allow", () => {
        const plan = createComputeTextPlan(fixture("Roboto-Regular.ttf").font, selectComputeTextRun("H"));
        const limits = { maxComputeWorkgroupsPerDimension: 65535, maxBufferSize: 2 ** 40, maxStorageBufferBindingSize: 2 ** 40 } as GPUSupportedLimits;
        for (const capacity of [65535, 65536, 70002, 1 << 18, 1 << 20]) {
            const prepared = prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(72), glyphIds: Uint32Array.of(44) }], 0, capacity);
            expect(() => checkComputeTextShapeBounds(limits, prepared), String(capacity)).not.toThrow();
        }
        const tooLarge = prepareComputeTextShapeWords(plan, [], 0, 0x20000000);
        expect(() => checkComputeTextShapeBounds(limits, tooLarge)).toThrow("capacity exceeds u32 record addressing");
    });

    it("uses three class bits and leaves no sixteen-bit word-7 reads in generated code", () => {
        expect(RECORD_CLASS_MASK).toBe(7);
        expect(ATTACH_SHIFT).toBe(3);
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("fn recordProperties(");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("fn recordClass(");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("fn attachDistance(");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).not.toMatch(/scratch\[[^\n]*\+\s*7u?\][^\n]*(?:0xffff|(?:<<|>>)\s*16u)/);
    });
    it("maps existing scripts to pause-free stages without changing default lookup semantics", () => {
        for (const [file, text, id] of [
            ["Inter.ttf", "AB", "default"],
            ["NotoSansHebrew-Regular.ttf", "שלום", "hebrew"],
            ["NotoSansArabic-Regular.ttf", "كتب", "arabic"],
        ] as const) {
            const plan = createComputeTextPlan(fixture(file).font, selectComputeTextRun(text));
            expect(plan._profile).toBe(id);
            expect(plan._stages.every((stage) => stage.pause === "none")).toBe(true);
            expect([...new Set(plan._stages.filter((stage) => stage.table === "GSUB").flatMap((stage) => stage.roots.map((root) => root.lookup)))].sort((a, b) => a - b)).toEqual(
                plan._gsubRoots
            );
            expect([...new Set(plan._stages.filter((stage) => stage.table === "GPOS").flatMap((stage) => stage.roots.map((root) => root.lookup)))].sort((a, b) => a - b)).toEqual(
                plan._gposRoots
            );
            expect(plan._stages.length).toBe(id === "arabic" ? 7 : 2);
        }
        // B2 retains tombstone coordinates; one-record contexts eliminate only unobserved sequence bookkeeping.
        expect(createHash("sha256").update(COMPUTE_TEXT_KERNEL_SOURCE.replace(", profile:u32", "")).digest("hex")).toBe(
            "1f0501598c4373e12ac5515fe4287c5026594f71d6baba1acf671e75aa07fd1d"
        );
    });

    it("keys real memory tiers and keeps both record widths inside the device workgroup budget", () => {
        const base: ComputeTextKernelVariant = { id: "default", glyphWords: 8, masked: false, planes: 1 };
        const wide: ComputeTextKernelVariant = { id: "indic", glyphWords: 16, masked: false, planes: 1 };
        expect(computeTextScratchCapacity(256)).toBe(256);
        expect(computeTextScratchCapacity(257)).toBe(0);
        expect(computeTextScratchCapacity(128, 0, 16)).toBe(128);
        expect(computeTextScratchCapacity(129, 0, 16)).toBe(0);
        expect(computeTextVariantKey(wide, 128)).toBe("indic:16:128");
        expect(computeTextVariantTier(256, base, 4096)).toBe(0);
        expect(workgroupBytes(kernelSource(256, base))).toBeLessThanOrEqual(16384);
        expect(workgroupBytes(kernelSource(128, wide))).toBeLessThanOrEqual(16384);
    });

    it("memoizes successful loading, retries failures and accounts for planes and auxiliary storage", async () => {
        let attempts = 0;
        installComputeTextProfileLoader("indic", async () => {
            attempts++;
            if (attempts === 1) {
                throw new Error("test import failure");
            }
            installComputeTextProfile("indic", { glyphWords: 16, masked: true, planes: 2, auxBytes: (capacity) => capacity * 4 });
        });
        const first = ensureComputeTextProfile("indic");
        expect(ensureComputeTextProfile("indic")).toBe(first);
        await expect(first).rejects.toThrow("test import failure");
        await ensureComputeTextProfile("indic");
        expect(attempts).toBe(2);
        expect(profileSupport("indic")!.glyphWords).toBe(16);
        await ensureComputeTextProfile("indic");
        expect(attempts).toBe(2);
        const plan = createComputeTextPlan(fixture("Inter.ttf").font, selectComputeTextRun("A"));
        const prepared = prepareComputeTextShapeWords({ ...plan, _profile: "indic" }, [{ scalars: Uint32Array.of(65), glyphIds: Uint32Array.of(2) }]);
        expect(prepared.params).toHaveLength(8);
        expect(prepared.scratchBytes).toBe(prepared.capacity * (16 * 4 * 2 + 4));
    });
});
