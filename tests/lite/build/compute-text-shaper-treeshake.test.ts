import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { cleanupTempDirs, ensureLibBuilt, LIB_ENTRY, runRollup, runVite } from "./bundler-harness";

const KERNEL_ONLY = "fn rejectRun(";
const PARSER_ONLY = "function compileComputeTextProgram";
const TEXT_SHAPER_RUNTIME = "Font.load";

afterAll(cleanupTempDirs);
beforeAll(ensureLibBuilt);

describe("compute text shaper tree shaking", () => {
    it("keeps text-shaper out of a compute-only import graph", async () => {
        const entrySource = `import { loadComputeFont, shapeComputeText } from ${JSON.stringify(LIB_ENTRY)};\n` + `console.log(loadComputeFont, shapeComputeText);\n`;
        const result = await runRollup({ entrySource, format: "es", minify: false });

        expect(result.errors).toEqual([]);
        expect(result.significantWarnings).toEqual([]);
        const ids = result.modules!.map((module) => module.id);
        expect(result.code).toContain(KERNEL_ONLY);
        expect(result.code).toContain(PARSER_ONLY);
        expect(result.code).not.toContain(TEXT_SHAPER_RUNTIME);
        expect(result.imports!.some((id) => /text-shaper/.test(id))).toBe(false);
        expect(ids.some((id) => /\/vendor\/text-shaper|\/text\/font\.js$|\/text\/layout\.js$|\/text\/glyph-extraction\.js$|\/text\/default-text-data\.js$/.test(id))).toBe(false);
        expect(ids.some((id) => id.endsWith("/text/compute-text-program.js"))).toBe(true);
        expect(ids.some((id) => id.endsWith("/text/_gpu/compute-text-shape.js"))).toBe(true);
        expect([...ids, ...result.imports!].filter((id) => /\/node_modules\/webgpu\/|dawn\.node|setup-native|tests\/lite\/compute/.test(id))).toEqual([]);
        expect(result.code).not.toContain("dawn.node");
    });

    it("drops the compute parser and shader from a text consumer that never imports them", async () => {
        const result = await runRollup({
            entrySource: `import { createTextRenderable } from ${JSON.stringify(LIB_ENTRY)};\nconsole.log(createTextRenderable);\n`,
            format: "es",
            minify: false,
        });

        expect(result.errors).toEqual([]);
        expect(result.significantWarnings).toEqual([]);
        const ids = result.modules!.map((module) => module.id);
        expect(result.code).not.toContain(KERNEL_ONLY);
        expect(result.code).not.toContain(PARSER_ONLY);
        expect(result.code).not.toContain("function parseMarkLigature");
        expect(ids.some((id) => /compute-text-(program|shape|shaper)\.js$/.test(id))).toBe(false);
        expect(ids.some((id) => id.endsWith("/compute/deferred-submit.js"))).toBe(false);
    });

    const common = ["createGlyphStorage", "createTextRenderer"];
    const resident = ["loadComputeFont", "createComputeTextData", "updateComputeTextData", "createComputeTextLayer", "disposeComputeTextData"];
    const cpu = ["loadFont", "createDefaultTextData", "updateDefaultTextData", "createTextLayer"];
    const byo = ["createTextData", "updateTextData", "createTextLayer"];
    const sharedGpu = ["text-bind-group.js", "text-style-buffer.js"];
    const computeOnly = /\/text\/(compute-|font-parser\/|unicode-script-data)|\/text\/_gpu\/compute-/;
    const cpuOnly = /\/vendor\/text-shaper|\/text\/(font|layout|glyph-extraction|default-text-data|text-data|text-data-state|text-layer-cpu)\.js$|\/_gpu\/text-style-gpu\.js$/;

    it("retains paragraph layout only for the opt-in root consumer", async () => {
        const entrySource =
            `import { updateComputeTextParagraphs, getComputeFontLineMetrics } from ${JSON.stringify(LIB_ENTRY)};\n` +
            "console.log(updateComputeTextParagraphs, getComputeFontLineMetrics);\n";
        for (const result of [await runRollup({ entrySource, format: "es", minify: false }), await runVite(entrySource)]) {
            expect(result.errors).toEqual([]);
            const ids = result.modules!.map((module) => module.id);
            expect(ids.some((id) => id.endsWith("/text/compute-text-paragraphs.js"))).toBe(true);
            expect(ids.some((id) => id.endsWith("/compute/deferred-submit.js"))).toBe(true);
            expect(ids.some((id) => id.endsWith("/text/_gpu/compute-text-paragraph-layout.js"))).toBe(true);
            expect(result.code).toContain("fn measure(");
            expect(result.code).toContain("fn prefix(");
            expect(result.code).toContain("fn place(");
            expect(result.code).toContain("fma(");
            expect(ids.filter((id) => cpuOnly.test(id))).toEqual([]);
        }
    });

    it("does not pull CPU palette synchronization into the resident factories", async () => {
        const entrySource =
            `import { createComputeTextData, createComputeTextLayer } from ${JSON.stringify(LIB_ENTRY)};\n` + "console.log(createComputeTextData, createComputeTextLayer);\n";
        for (const result of [await runRollup({ entrySource, format: "es", minify: false }), await runVite(entrySource)]) {
            expect(result.errors).toEqual([]);
            const ids = result.modules!.map((module) => module.id);
            expect(ids.some((id) => id.endsWith("/_gpu/text-style-buffer.js"))).toBe(true);
            expect(ids.some((id) => id.endsWith("/_gpu/text-style-gpu.js"))).toBe(false);
            expect(ids.filter((id) => cpuOnly.test(id))).toEqual([]);
            expect(result.code).not.toContain("ensureStyleGpu");
            expect(result.code).not.toContain("packGlyphAtSlot");
        }
    });

    it("renders caller-provided TextData without retaining any initialization, packing, or shaping implementation", async () => {
        const entrySource =
            `import { createTextRenderer, createTextLayer, createTextRenderable } from ${JSON.stringify(LIB_ENTRY)};\n` +
            "console.log(createTextRenderer, createTextLayer, createTextRenderable);\n";
        for (const result of [await runRollup({ entrySource, format: "es", minify: false }), await runVite(entrySource)]) {
            expect(result.errors).toEqual([]);
            const ids = result.modules!.map((module) => module.id);
            expect(
                ids.filter((id) => computeOnly.test(id) || /\/vendor\/text-shaper|\/text\/(font|layout|glyph-extraction|default-text-data|text-data|text-data-state)\.js$/.test(id))
            ).toEqual([]);
            expect(ids.some((id) => /text-data-state|outline-curves|outline-path/.test(id))).toBe(false);
            expect(result.code).not.toContain("createEmptyTextData");
            expect(result.code).not.toContain("packGlyphAtSlot");
            expect(result.code).not.toContain("fn advanceOf");
        }
    });

    for (const [name, exports] of [
        ["resident", [...common, ...resident]],
        ["cpu", [...common, ...cpu]],
        ["byo", [...common, ...byo]],
        ["combined", [...common, ...cpu, ...resident]],
    ] as const) {
        it(`emits only the required implementation for the root-export ${name} consumer`, async () => {
            const entrySource = `import { ${exports.join(", ")} } from ${JSON.stringify(LIB_ENTRY)};\nconsole.log(${exports.join(", ")});\n`;
            const result = await runRollup({
                entrySource,
                format: "es",
                minify: false,
            });
            expect(result.errors).toEqual([]);
            expect(result.significantWarnings).toEqual([]);
            const modules = result.modules!;
            const ids = modules.map((module) => module.id);
            expect([...ids, ...result.imports!].filter((id) => /\/node_modules\/webgpu\/|dawn\.node|setup-native|tests\/lite\/compute/.test(id))).toEqual([]);
            for (const helper of sharedGpu) {
                expect(ids.filter((id) => id.endsWith(`/text/_gpu/${helper}`))).toHaveLength(1);
            }
            expect(ids.filter((id) => id.endsWith("/text/text-data-state.js"))).toHaveLength(name === "resident" ? 0 : 1);
            if (name === "resident") {
                expect(ids.filter((id) => cpuOnly.test(id))).toEqual([]);
                expect(result.imports!.filter((id) => /text-shaper/.test(id))).toEqual([]);
                expect(result.code).not.toMatch(
                    /function (createTextData|applyReset|packGlyphAtSlot|shapeComputeText|executeComputeTextShape|executeComputeTextReference|decode)\(/
                );
                const kernel = modules.find((module) => module.id.endsWith("/_gpu/compute-text-shape.js"))!;
                expect(kernel.renderedExports).toContain("createComputeTextShapeShader");
                expect(kernel.renderedExports).not.toContain("COMPUTE_TEXT_KERNEL_SOURCE");
                expect(kernel.renderedExports).not.toContain("executeComputeTextShape");
                expect(ids.some((id) => /compute-text-(reference|shaper|placement)\.js$/.test(id))).toBe(false);
                expect(ids.some((id) => /compute-text-paragraph(s|-layout)\.js$/.test(id))).toBe(false);
                expect(result.code).not.toContain("fn measure(");
                expect(result.code).toContain("drawIndirect");
                expect(result.code).not.toContain("createEmptyTextData");
                expect(result.code).not.toContain("ensureStyleGpu");
                expect(ids.some((id) => id.endsWith("/text/compute-text-layer.js"))).toBe(true);
            } else if (name !== "combined") {
                expect(ids.filter((id) => computeOnly.test(id))).toEqual([]);
                expect(ids.some((id) => id.endsWith("/compute/deferred-submit.js"))).toBe(false);
                expect(result.code).not.toContain("fn advanceOf");
                expect(result.code).not.toContain("fn pack");
                expect(result.code).not.toContain(PARSER_ONLY);
            }
            if (name === "byo") {
                expect(result.imports!.filter((id) => /text-shaper/.test(id))).toEqual([]);
                expect(ids.some((id) => /\/(font|layout|glyph-extraction|outline-path|outline-curves)\.js$/.test(id))).toBe(false);
            }
            if (name === "cpu" || name === "combined") {
                expect(ids.filter((id) => id.endsWith("/text/outline-curves.js"))).toHaveLength(1);
                expect(result.imports!.some((id) => /text-shaper/.test(id))).toBe(true);
                expect(result.code).toContain("function extractGlyphCurves");
                if (name === "cpu") {
                    expect(ids.some((id) => id.endsWith("/text/outline-path.js"))).toBe(false);
                    expect(result.code).not.toContain("appendQuadraticContour");
                }
            }

            const vite = await runVite(entrySource);
            expect(vite.errors).toEqual([]);
            const viteIds = vite.modules!.map((module) => module.id);
            if (name === "resident" || name === "byo") {
                expect(viteIds.filter((id) => /\/vendor\/text-shaper/.test(id))).toEqual([]);
                expect(vite.code).not.toContain("Font.load");
            }
            if (name === "resident") {
                expect(viteIds.filter((id) => cpuOnly.test(id))).toEqual([]);
                expect(viteIds.some((id) => /compute-text-paragraph(s|-layout)\.js$/.test(id))).toBe(false);
                expect(vite.code).not.toContain("fn measure(");
                expect(vite.code).not.toMatch(/function (packGlyphAtSlot|executeComputeTextShape|shapeComputeText)\(/);
            } else if (name !== "combined") {
                expect(viteIds.filter((id) => computeOnly.test(id))).toEqual([]);
                expect(viteIds.some((id) => id.endsWith("/compute/deferred-submit.js"))).toBe(false);
                expect(vite.code).not.toContain("fn advanceOf");
            }
            for (const helper of sharedGpu) {
                expect(viteIds.filter((id) => id.endsWith(`/text/_gpu/${helper}`))).toHaveLength(1);
            }
            const minified = await runRollup({ entrySource, format: "es", minify: true });
            expect(minified.errors).toEqual([]);
            const minifiedIds = minified.modules!.map((module) => module.id);
            if (name === "resident") {
                expect(minifiedIds.filter((id) => cpuOnly.test(id))).toEqual([]);
                expect(minifiedIds.some((id) => /compute-text-(reference|shaper|placement)\.js$/.test(id))).toBe(false);
                expect(minifiedIds.some((id) => /compute-text-paragraph(s|-layout)\.js$/.test(id))).toBe(false);
                expect(minified.code).not.toContain("fn measure(");
            } else if (name !== "combined") {
                expect(minifiedIds.filter((id) => computeOnly.test(id) || /compute-text-layer\.js$/.test(id))).toEqual([]);
            }
            for (const helper of sharedGpu) {
                expect(minified.modules!.filter((module) => module.id.endsWith(`/text/_gpu/${helper}`) && module.renderedLength > 0)).toHaveLength(1);
            }
        });
    }

    it("keeps both draw adapters out of a shared-core-only root consumer", async () => {
        const entrySource = `import { createTextRenderer, setTextLayerPosition } from ${JSON.stringify(LIB_ENTRY)};\nconsole.log(createTextRenderer, setTextLayerPosition);\n`;
        for (const result of [
            await runRollup({ entrySource, format: "es", minify: false }),
            await runRollup({ entrySource, format: "es", minify: true }),
            await runVite(entrySource),
        ]) {
            expect(result.errors).toEqual([]);
            const modules = result.modules!;
            expect(modules.some((module) => module.id.endsWith("/text/text-renderer.js") && module.renderedLength > 0)).toBe(true);
            expect(modules.filter(({ id }) => /text-layer-cpu|compute-text-layer|text-data-state|text-style-(gpu|buffer)|text-bind-group/.test(id))).toEqual([]);
            expect(modules.filter(({ id }) => cpuOnly.test(id) || computeOnly.test(id))).toEqual([]);
            expect(result.code).not.toContain("drawIndirect");
            expect(result.code).not.toContain("text-layer-instances");
        }
    });
});
