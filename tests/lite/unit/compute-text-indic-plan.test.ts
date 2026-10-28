import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { fixture } from "../compute/fixtures";
import { createComputeTextPlan, ensureComputeTextProfile, profileSupport, installComputeTextProfile } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { buildIndicPlan, validateIndicRun, indicProperties, validateIndicCompiledProgram } from "../../../packages/babylon-lite/src/text/compute-text-indic-plan";
import * as compiler from "../../../packages/babylon-lite/src/text/compute-text-program";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import type { ComputeTextRunSelection } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { ensureComputeFontLayout } from "../../../packages/babylon-lite/src/text/compute-font";
import { shapeComputeText } from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { createComputeTextShapeShader, computeTextVariantTier, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { disposeComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";

describe("lazy dev2 plan and explicit font/input boundary", () => {
    beforeAll(async () => ensureComputeTextProfile("indic"));
    it("fails closed on selected compiled multi-record Ligature children, but not unselected or single-record rules", () => {
        const font = fixture("NotoSansDevanagari-Regular.ttf").font;
        const real = compileComputeTextProgram(createComputeTextPlan(font, selectComputeTextRun("क्ष")));
        expect(() => validateIndicCompiledProgram(real)).not.toThrow();
        const parent = real.lookups.find((lookup) => lookup.table === "GSUB" && lookup.rules.some((rule) => rule.action.kind === "context"))!;
        const context = parent.rules.find((rule) => rule.action.kind === "context")!;
        const ligature = real.lookups.find((lookup) => lookup.table === "GSUB" && lookup.rules.some((rule) => rule.action.kind === "ligature"))!;
        const action = {
            kind: "context" as const,
            table: "GSUB" as const,
            lookups: [
                { sequenceIndex: 0, lookupIndex: ligature.lookupIndex },
                { sequenceIndex: 1, lookupIndex: ligature.lookupIndex },
            ],
        };
        const bad = { ...real, lookups: [{ ...parent, rules: [{ ...context, action }] }, ligature] };
        const message = "createComputeTextPlan: font uses multi-record contextual ligatures not yet supported by the GPU shaper";
        expect(() => validateIndicCompiledProgram(bad)).toThrow(message);
        expect(() => validateIndicCompiledProgram({ lookups: [ligature] })).not.toThrow();
        expect(() =>
            validateIndicCompiledProgram({ lookups: [{ ...parent, rules: [{ ...context, action: { ...action, lookups: action.lookups.slice(0, 1) } }] }, ligature] })
        ).not.toThrow();
        const spy = vi.spyOn(compiler, "compileComputeTextProgram").mockReturnValue(bad);
        try {
            expect(() => buildIndicPlan(font)).toThrow(message);
        } finally {
            spy.mockRestore();
        }
    });
    it("rejects an explicitly unavailable profile fixture before dispatch", async () => {
        const device = { lost: new Promise<GPUDeviceLostInfo>(() => undefined) } as GPUDevice;
        const engine = { _device: device } as EngineContext;
        const original = profileSupport("indic")!;
        installComputeTextProfile("indic", { ...original, available: false });
        try {
            await expect(shapeComputeText(engine, fixture("NotoSansDevanagari-Regular.ttf").font, "क")).rejects.toThrow(
                "shapeComputeText: run 0 Indic kernel variant not available"
            );
        } finally {
            installComputeTextProfile("indic", original);
            engine._disposeManagedResources?.();
        }
    });
    it("admits the pinned font with complete shared executable stages, masks and profile tables", () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("क्ष"));
        const program = compileComputeTextProgram(plan);
        expect(program.fullyExecutable).toBe(true);
        expect(plan._profile).toBe("indic");
        expect(plan._stages.map((stage) => stage.pause)).toEqual([
            "setupSyllables",
            "initialReorder",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "finalReorder",
            "clearSyllables",
            "none",
        ]);
        expect(plan._profileData!.tables().length).toBeGreaterThan(1000);
        expect(program.words[14]).toBeGreaterThan(0);
        expect(plan._profileData!.capacity!(Uint32Array.of(0x958, 0x93f))).toBeGreaterThanOrEqual(4);
        expect(profileSupport("indic")?.available).toBe(true);
        expect(createComputeTextPlan(face.font, selectComputeTextRun("क"))).toBe(plan);
    });

    it("production identity has no stage-stop mechanism and actual two-plane declarations respect 16 KiB", () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf"),
            selection = selectComputeTextRun("क");
        const plan = createComputeTextPlan(face.font, selection);
        const p = prepareComputeTextShapeWords(plan, [{ scalars: selection._scalars, glyphIds: Uint32Array.of(mapComputeFontScalar(face.font, 0x915)) }], 0, 128);
        expect(p.params[7]).toBe(3);
        expect(p.variant.planes).toBe(2);
        expect(p.variant.auxBytes!(128)).toBe(128 * 16);
        expect(p.scratchBytes).toBe(18432);
        expect(128 * (16 * 2 + 4) * 4).toBe(18432);
        expect(computeTextVariantTier(128, p.variant, 16384)).toBe(0);
        const engine = { _device: { limits: { maxComputeWorkgroupStorageSize: 16384 } } } as EngineContext;
        for (const capacity of [64, 128]) {
            const shader = createComputeTextShapeShader(engine, capacity, p.variant);
            const source = shader._source;
            expect(source).not.toMatch(/params\.profile\s*>>/);
            expect(source).not.toContain("indicStopped");
            expect(source).not.toContain("profileStopped");
            expect(source).not.toContain("Internal diagnostic");
            const size = (type: string): number => {
                if (["u32", "i32", "atomic<u32>"].includes(type)) {
                    return 4;
                }
                if (type === "vec2u") {
                    return 8;
                }
                const array = /^array<(.+),(\d+)>$/.exec(type);
                if (!array) {
                    throw new Error(`Unaccounted actual declaration ${type}`);
                }
                return size(array[1]!) * Number(array[2]);
            };
            const declarations = [...source.matchAll(/var<workgroup>\s+\w+\s*:\s*([^;]+);/g)];
            expect(declarations.length).toBeGreaterThan(10);
            expect(declarations.reduce((n, item) => n + size(item[1]!.replace(/\s/g, "")), 0)).toBeLessThanOrEqual(16384);
            expect(source.includes("var<workgroup> scratch:")).toBe(capacity === 64);
            expect(source.includes("@binding(4) var<storage, read_write> scratch")).toBe(capacity === 128);
            disposeComputeShader(shader);
        }
    });

    it("the staged roots contain every independently traced lookup in relative application order", () => {
        const plan = createComputeTextPlan(fixture("NotoSansDevanagari-Regular.ttf").font, selectComputeTextRun("क"));
        const order = plan._stages.flatMap((stage) => stage.roots.map((root) => `${stage.table}:${root.lookup}`));
        const corpus = JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as {
            cases: { id: string; lookupTrace?: [string, number, string][] }[];
        };
        for (const row of corpus.cases) {
            let previous = -1;
            // Actual root-start messages define which lookups applied in this
            // reference request. Nested recursion messages are not root starts.
            for (const [table, index] of row.lookupTrace ?? []) {
                const at = order.indexOf(`${table}:${index}`, previous + 1);
                expect(at, `${row.id}: ${table}/${index} after ${previous}`).toBeGreaterThan(previous);
                previous = at;
            }
        }
    });

    it("enumerates the rphf pair, font-only below-base Ra and all internal closure glyphs", () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("र्"));
        const words = plan._profileData!.tables();
        const ra = mapComputeFontScalar(face.font, 0x930),
            virama = mapComputeFontScalar(face.font, 0x94d);
        expect(Array.from(words.slice(words[13]!, words[13]! + words[14]!))).toContain(((ra << 16) | virama) >>> 0);
        expect(words[words[12]! + ra]).toBe(1);
        expect(Array.from(words.slice(18, 21))).toEqual([1, 4, 2]);
        for (const cp of [0x20, 0x25cc, 0x94d, 0x93c, 0x931]) {
            expect(plan._profileData!.closureSeeds).toContain(mapComputeFontScalar(face.font, cp));
        }
        expect(plan._profileData!.spaceLocal).toBe(true);
    });

    it("fails unsupported font boundaries explicitly instead of shaping nominal approximations", () => {
        expect(() => buildIndicPlan(fixture("Inter.ttf").font)).toThrow("Devanagari font is unsupported: variable fonts are unsupported");
        expect(() => buildIndicPlan(fixture("Roboto-Regular.ttf").font)).toThrow("Devanagari font is unsupported: dev2 GSUB and GPOS scripts are required");
        const font = fixture("NotoSansDevanagari-Regular.ttf").font;
        expect(() => buildIndicPlan({ ...font, _hasVariableAxes: true })).toThrow("variable fonts are unsupported");
        const tables = new Map(font._directory.tables);
        tables.set("morx", tables.get("head")!);
        expect(() => buildIndicPlan({ ...font, _directory: { ...font._directory, tables } })).toThrow("AAT tables are unsupported");
        expect(() => buildIndicPlan({ ...font, _layout: { ...ensureComputeFontLayout(font), gdef: null } })).toThrow("GDEF glyph classes are required");
    });

    it("checks the complete scalar boundary independently of public strong-script selection", () => {
        const run = (cp: number) => ({ _scalars: Uint32Array.of(cp), scalarCount: 1 }) as ComputeTextRunSelection;
        for (let cp = 0x900; cp <= 0x97f; cp++) {
            expect(() => validateIndicRun(run(cp))).not.toThrow();
        }
        for (const cp of [0x20, 0xa0, 0x200c, 0x200d, 0x20b9, 0x20f0]) {
            expect(() => validateIndicRun(run(cp))).not.toThrow();
        }
        for (const cp of [0x1cd0, 0xa8e0, 0x11b00, 0xad, 0x200b, 0x2060, 0xfe0f, 0x25cc, 0x41]) {
            expect(() => validateIndicRun(run(cp))).toThrow(`unsupported scalar in Devanagari run at scalar 0 U+${cp.toString(16).toUpperCase().padStart(4, "0")}`);
        }
        expect(indicProperties(0x93f) & 255).toBe(7);
    });
});
