import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import {
    createComputeTextPlan,
    ensureComputeTextProfile,
    installComputeTextProfileLoader,
    profileSupport,
    requiredComputeTextProfiles,
    validateComputeTextPlanFont,
} from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { computeTextRequiredProfiles, selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { computeTextPlanVariant, kernelSource, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { installComputeTextArabicProfile } from "../../../packages/babylon-lite/src/text/compute-text-arabic";
import { fixture } from "../compute/fixtures";
import { syntheticFont } from "../compute/synthetic";

describe("Arabic feature stages and profile lifecycle", () => {
    beforeAll(async () => ensureComputeTextProfile("arabic"));

    it("matches independent HarfBuzz feature-stage/root order and one-hot/global masks", () => {
        const face = fixture("NotoSansArabic-Regular.ttf");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("كتب"));
        const corpus = JSON.parse(readFileSync("tests/lite/compute/references/arabic-harfbuzz-11.0.1.json", "utf8")) as {
            cases: { lookupTrace: [string, number, string][] }[];
        };
        const trace = corpus.cases.find((entry) => entry.lookupTrace.some(([table, , feature]) => table === "GSUB" && feature === "liga"))!.lookupTrace;
        const order = trace.map(([table, lookup]) => `${table}/${lookup}`);
        const actual = plan._stages.flatMap((stage) => stage.roots.map((root) => `${stage.table}/${root.lookup}`));
        expect(actual).toEqual(order);
        expect(plan._stages).toHaveLength(7);
        expect(plan._stages.map((stage) => [...new Set(stage.roots.map((root) => root.mask))])).toEqual([[0x80000000], [2], [4], [8], [0x80000000], [0x80000000], [0x80000000]]);
        expect([...new Set(plan._stages.filter((stage) => stage.table === "GSUB").flatMap((stage) => stage.roots.map((root) => root.lookup)))].sort((a, b) => a - b)).toEqual(
            plan._gsubRoots
        );
        const variant = computeTextPlanVariant(plan);
        expect(variant).toMatchObject({ id: "arabic", glyphWords: 8, masked: true });
        const source = kernelSource(256, variant);
        expect(source).toContain("firstNonT");
        expect(source).toContain("glyphMask");
        expect(source).not.toContain("fn proveSimpleRun");
        expect(kernelSource(256)).not.toContain("firstNonT");
        expect(kernelSource(256)).not.toContain("glyphMask");
        const counts = [...source.matchAll(/var<workgroup>\s+\w+\s*:\s*([^;]+);/g)].map(([, type]) => type!.replace(/\s/g, ""));
        const size = (type: string): number => {
            if (type === "u32" || type === "atomic<u32>") {
                return 4;
            }
            if (type === "vec2u") {
                return 8;
            }
            const array = /^array<(.+),(\d+)>$/.exec(type)!;
            return (["i32", "u32", "atomic<u32>"].includes(array[1]!) ? 4 : size(array[1]!)) * Number(array[2]);
        };
        expect(counts.reduce((sum, type) => sum + size(type), 0)).toBeLessThanOrEqual(16384);
    });

    it("packs scalar properties centrally and keeps profile pre-scans non-throwing", () => {
        expect(requiredComputeTextProfiles(["latn", "hebr"])).toEqual([]);
        expect(requiredComputeTextProfiles(["arab"])).toEqual(["arabic"]);
        expect(computeTextRequiredProfiles(["HH", "كتب"])).toEqual(["arabic"]);
        expect(computeTextRequiredProfiles(["Aب", "ب\u200d", "\ud800"])).toEqual([]);
        const face = fixture("NotoSansArabic-Regular.ttf");
        const selection = selectComputeTextRun("بَّ");
        const plan = createComputeTextPlan(face.font, selection);
        const glyphIds = selection._scalars.map((scalar) => face.oracle.glyphId(scalar));
        const words = prepareComputeTextShapeWords(plan, [{ scalars: selection._scalars, glyphIds }]).runs;
        const start = words[0]! + 1;
        expect(words[start]! & 0xffff).toBe(glyphIds[0]);
        expect((words[start]! >>> 16) & 7).toBe(2);
        expect(words[start + 1]! >>> 24).toBe(31);
        expect(words[start + 2]! >>> 24).toBe(27);
    });

    it("applies the explicit D10 boundary without calling a valid font invalid", () => {
        const face = syntheticFont("arabic-no-joining");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("ب"));
        expect(() => validateComputeTextPlanFont(plan)).toThrow("font has no Arabic joining features");
        expect(() => validateComputeTextPlanFont(plan)).toThrow("fallback is unsupported");
    });

    it("clears failed loading state and retries the same shared profile", async () => {
        installComputeTextProfileLoader("arabic", async () => {
            throw new Error("test Arabic import failure");
        });
        await expect(ensureComputeTextProfile("arabic")).rejects.toThrow("test Arabic import failure");
        expect(profileSupport("arabic")).toBeUndefined();
        installComputeTextProfileLoader("arabic", async () => {
            installComputeTextArabicProfile();
        });
        await ensureComputeTextProfile("arabic");
        expect(profileSupport("arabic")!.masked).toBe(true);
    });
});
