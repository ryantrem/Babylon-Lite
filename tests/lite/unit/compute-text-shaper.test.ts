/** Compute shaper state machine and reference-kernel semantics. text-shaper is test-only. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import { createComputeFontFromBuffer, mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import {
    createComputeTextShaper,
    disposeComputeTextShaper,
    shapeComputeText,
    shapeComputeTextPlan,
    _installComputeTextShapeExecutorForTests,
} from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import type { ComputeTextGlyph, ComputeTextShaper } from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import type { ComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { executeComputeTextReference } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-reference";
import { COMPUTE_TEXT_KERNEL_SOURCE, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import * as programCompiler from "../../../packages/babylon-lite/src/text/compute-text-program";
import * as selectors from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { glyphClass } from "../../../packages/babylon-lite/src/text/font-parser/opentype-layout";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { Font, GlyphBuffer, UnicodeBuffer, shapeInto } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

const TEXT_SHAPER_VERSION = JSON.parse(readFileSync(join(process.cwd(), "packages/babylon-lite/node_modules/text-shaper/package.json"), "utf8")).version as string;

function fontBytes(file: string): ArrayBuffer {
    const buf = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function mockEngine(): { engine: EngineContext; lose: () => void } {
    let lose = (): void => undefined;
    const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
        lose = () => resolve({ reason: "destroyed", message: "mock" } as GPUDeviceLostInfo);
    });
    const device = {
        limits: { maxBufferSize: 256 * 1024 * 1024, maxComputeWorkgroupsPerDimension: 65535 },
        lost,
        createBuffer: () => ({ destroy() {}, getMappedRange: () => new ArrayBuffer(4), unmap() {} }),
        createShaderModule: () => ({}),
        createBindGroupLayout: () => ({}),
        createPipelineLayout: () => ({}),
        createComputePipeline: () => ({}),
        createComputePipelineAsync: async () => ({}),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({ beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }), finish: () => ({}) }),
        queue: { submit() {}, writeBuffer() {} },
    } as unknown as GPUDevice;
    return { engine: { _device: device } as EngineContext, lose };
}

function shapeFields(bytes: ArrayBuffer, text: string): number[][] {
    const shaper = Font.load(bytes);
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    input.addStr(text);
    input.setScript("latn");
    shapeInto(shaper, input, output, { script: "latn", language: null, direction: "ltr" });
    return output.infos.slice(0, output.length).map((info, index) => {
        const position = output.positions[index];
        return [info.glyphId, info.cluster, position?.xAdvance ?? 0, position?.yAdvance ?? 0, position?.xOffset ?? 0, position?.yOffset ?? 0];
    });
}

function referenceFields(plan: ComputeTextPlan, text: string): number[][] {
    const selection = selectComputeTextRun(text);
    const glyphIds = [...selection._scalars].map((scalar) => mapComputeFontScalar(plan.font, scalar));
    const program = compileComputeTextProgram(plan);
    return executeComputeTextReference(
        program,
        glyphIds,
        selection._scalars.map((_, index) => index)
    ).map(fieldsOf);
}

function fieldsOf(glyph: ComputeTextGlyph): number[] {
    return [glyph.glyphId, glyph.cluster, glyph.xAdvance, glyph.yAdvance, glyph.xOffset, glyph.yOffset];
}

describe("compute text shaper", () => {
    it("omits unused offset vectors in shaping batches without changing standalone offsets or UTF-16 rejection priority", async () => {
        const font = createComputeFontFromBuffer(fontBytes("NotoSansGothic-Regular.ttf"));
        const standalone = selectComputeTextRun("𐌰𐌱");
        expect(standalone._utf16Offsets).toEqual(Uint32Array.of(0, 2));
        const plan = createComputeTextPlan(font, standalone);
        const { engine } = mockEngine();
        const shaper = createComputeTextShaper(engine);
        _installComputeTextShapeExecutorForTests(async (_shaper, _plan, runs) =>
            runs.map((run) => [...run.glyphIds].map((glyphId, cluster) => ({ glyphId, cluster, xAdvance: font._advances[glyphId]!, yAdvance: 0, xOffset: 0, yOffset: 0 })))
        );
        const calls = vi.spyOn(selectors, "selectComputeTextRun");
        try {
            const root = await shapeComputeText(engine, font, ["𐌰𐌱", ""]);
            const direct = await shapeComputeTextPlan(shaper, plan, ["𐌰𐌱", ""]);
            expect(root[0]?.map((glyph) => glyph.cluster)).toEqual([0, 1]);
            expect(direct[0]?.glyphs).toEqual(root[0]);
            expect(root[1]).toEqual([]);
            expect(direct[1]?.glyphs).toEqual([]);
            for (const result of calls.mock.results) {
                if (result.type === "return") {
                    expect(result.value._utf16Offsets).toHaveLength(0);
                }
            }
            await expect(shapeComputeText(engine, font, "𐌰\n\uD800")).rejects.toThrow(/lone surrogate at UTF-16 index 3/);
            await expect(shapeComputeText(engine, font, "𐌰\n")).rejects.toThrow(/control at scalar 1/);
        } finally {
            calls.mockRestore();
            _installComputeTextShapeExecutorForTests(null);
            disposeComputeTextShaper(shaper);
        }
    });

    it("selects each public input once and preserves executor snapshot isolation", async () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const { engine } = mockEngine();
        const produced: ComputeTextGlyph[] = [{ glyphId: 1, cluster: 0, xAdvance: 10, yAdvance: 0, xOffset: 0, yOffset: 0 }];
        _installComputeTextShapeExecutorForTests(async (_shaper, _plan, runs) => runs.map(() => produced));
        const calls = vi.spyOn(selectors, "selectComputeTextRun");
        try {
            const output = await shapeComputeText(engine, font, ["A", "B", ""]);
            expect(calls).toHaveBeenCalledTimes(3);
            produced[0] = { ...produced[0]!, xAdvance: 999 };
            expect(output[0]?.[0]?.xAdvance).toBe(10);
            expect(output[2]).toEqual([]);
        } finally {
            calls.mockRestore();
            _installComputeTextShapeExecutorForTests(null);
        }
    });

    it("rejects an incomplete executor batch instead of publishing missing nonempty runs as empty", async () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const { engine } = mockEngine();
        _installComputeTextShapeExecutorForTests(async () => []);
        try {
            await expect(shapeComputeText(engine, font, ["A", "B"])).rejects.toThrow(/incomplete batch/);
        } finally {
            _installComputeTextShapeExecutorForTests(null);
        }
    });

    it("checks every signed publication field without allocating an intermediate numeric array per glyph", async () => {
        const source = readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/compute-text-shaper.ts"), "utf8");
        expect(source).not.toContain("for (const value of [glyph.xAdvance");
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const { engine } = mockEngine();
        try {
            for (const field of ["xAdvance", "yAdvance", "xOffset", "yOffset"] as const) {
                _installComputeTextShapeExecutorForTests(async () => [[{ glyphId: 1, cluster: 0, xAdvance: 0, yAdvance: 0, xOffset: 0, yOffset: 0, [field]: Infinity }]]);
                await expect(shapeComputeText(engine, font, "A")).rejects.toThrow(/shaped value is not a safe integer/);
            }
        } finally {
            _installComputeTextShapeExecutorForTests(null);
        }
    });

    it("reuses immutable preparation and compiles once while repacking every warm revision and capacity", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const plan = createComputeTextPlan(font, selectComputeTextRun("A"));
        const calls = vi.spyOn(programCompiler, "compileComputeTextProgram");
        try {
            const first = prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(65), glyphIds: Uint32Array.of(mapComputeFontScalar(font, 65)) }]);
            const second = prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(66, 66), glyphIds: Uint32Array.of(38, 38) }], 7, 4);
            expect(calls).toHaveBeenCalledTimes(1);
            expect(second.program).toBe(first.program);
            expect(second.programWords).toBe(first.programWords);
            expect(second.tables).toBe(first.tables);
            expect(second.runs).not.toBe(first.runs);
            expect([...second.runs]).toEqual([1, 2, 38, 38]);
            expect([...second.params]).toEqual([1, 4, 0, font.glyphCount, first.program.rootOrder.length, first.program.words.length, 7]);
            expect(second.scratchBytes).toBe(128);
            const greek = createComputeTextPlan(font, selectComputeTextRun("Α"));
            const other = prepareComputeTextShapeWords(greek, [{ scalars: Uint32Array.of(0x391), glyphIds: Uint32Array.of(mapComputeFontScalar(font, 0x391)) }]);
            expect(calls).toHaveBeenCalledTimes(2);
            expect(other.program).not.toBe(first.program);
            const anotherFont = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
            const anotherPlan = createComputeTextPlan(anotherFont, selectComputeTextRun("A"));
            prepareComputeTextShapeWords(anotherPlan, []);
            expect(calls).toHaveBeenCalledTimes(3);
            expect(createComputeTextPlan(font, selectComputeTextRun("office"))).toBe(plan);
            expect(createComputeTextPlan(font, selectComputeTextRun("Α"))).toBe(greek);
        } finally {
            calls.mockRestore();
        }
    });

    it("bulk-packs current nominal glyph vectors without transmitting redundant initial scalar-index clusters", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const plan = createComputeTextPlan(font, selectComputeTextRun("A"));
        prepareComputeTextShapeWords(plan, []);
        const first = Uint32Array.of(38, 39);
        const empty = new Uint32Array();
        const last = Uint32Array.of(40);
        const copy = vi.spyOn(Uint32Array.prototype, "set");
        try {
            const packed = prepareComputeTextShapeWords(plan, [
                { scalars: Uint32Array.of(65, 66), glyphIds: first },
                { scalars: empty, glyphIds: empty },
                { scalars: Uint32Array.of(67), glyphIds: last },
            ]);
            expect([...packed.runs]).toEqual([3, 6, 7, 2, 38, 39, 0, 1, 40]);
            expect(copy.mock.calls.filter(([source]) => source === first || source === last)).toHaveLength(2);
            first[0] = 41;
            expect(packed.runs[4]).toBe(38);
            const changed = prepareComputeTextShapeWords(plan, [{ scalars: Uint32Array.of(65, 66), glyphIds: first }]);
            expect([...changed.runs]).toEqual([1, 2, 41, 39]);
            expect(changed.runs.buffer).not.toBe(packed.runs.buffer);
        } finally {
            copy.mockRestore();
        }
    });

    it("packs every class definition as a constant-time glyph-indexed table including the class-zero complement", () => {
        const font = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font, selectComputeTextRun("A")), []);
        expect(prepared.program.classTables.length).toBeGreaterThan(0);
        prepared.program.classTables.forEach((def, index) => {
            const origin = prepared.tables[6 + index]!;
            expect([...prepared.tables.slice(origin, origin + font.glyphCount)]).toEqual(Array.from({ length: font.glyphCount }, (_, glyph) => glyphClass(def, glyph)));
        });
    });

    it("compacts first-only pair rows without losing original cell deltas or rejection modes", () => {
        const font = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        const packed = prepareComputeTextShapeWords(createComputeTextPlan(font, selectComputeTextRun("A")), []);
        const words = packed.programWords;
        const footer = packed.programWords[8]! + packed.program.rootOrder.length * 2 + font.glyphCount * 2;
        expect(words.length).toBeGreaterThan(footer + packed.program.rootOrder.length);
        let checkedRows = 0;
        packed.program.rootOrder.forEach((lookupAt, root) => {
            const lookup = packed.program.lookups[lookupAt]!;
            const descriptor = words[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
            if (lookup.table !== "GPOS" || words[descriptor + 2] !== 3) {
                expect(words[footer + root]).toBe(0);
                return;
            }
            const directory = words[footer + root]!;
            expect(directory).toBeGreaterThanOrEqual(footer + packed.program.rootOrder.length);
            const originalDirectory = words[descriptor + 6]!;
            for (let glyph = 0; glyph < font.glyphCount; glyph++) {
                const count = words[directory + glyph * 2 + 1]!;
                if (count === 0) {
                    continue;
                }
                const first = words[directory + glyph * 2]!;
                const originalRule = words[descriptor + 3]! + words[words[originalDirectory + glyph * 2]!]! * 16;
                if (words[first] !== 1) {
                    continue;
                }
                const originalRow = words[originalRule + 10]!;
                expect(words[first + 1]).toBe(words[originalRule + 11]);
                expect(words[first + 2]).toBe(words[originalRule + 12]);
                const row = words[first + 3]!;
                for (let cls = 0; cls < words[first + 2]!; cls++) {
                    const cell = words[originalRow + cls]!;
                    const rule = words[descriptor + 3]! + (cell - 1) * 16;
                    const mode = cell === 0 ? 0 : words[rule + 8]!;
                    expect(words[row + cls * 2 + 1]).toBe(mode === 0 ? 0 : mode === 5 ? 2 : 1);
                    expect(words[row + cls * 2]).toBe(mode === 0 || mode === 5 ? 0 : words[words[rule + 6]! + (mode === 4 ? 2 : 1)]);
                }
                checkedRows++;
                break;
            }
        });
        expect(checkedRows).toBeGreaterThan(0);
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("packs exhaustive first-32-root presence masks from candidate directories: %s", (file) => {
        const font = createComputeFontFromBuffer(fontBytes(file));
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font, selectComputeTextRun("A")), []);
        const masks = prepared.programWords[8]! + prepared.program.rootOrder.length * 2;
        expect(prepared.programWords.length - masks).toBe(font.glyphCount * 2 + prepared.simpleWordCount);
        for (let glyph = 0; glyph < font.glyphCount; glyph++) {
            let expected = 0;
            prepared.program.rootOrder.slice(0, 32).forEach((index, root) => {
                const lookup = prepared.program.lookups[index]!;
                const descriptor = prepared.programWords[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
                const directory = prepared.programWords[descriptor + 6]!;
                if (directory !== 0 && prepared.programWords[directory + glyph * 2 + 1]! > 0) {
                    expected = (expected | (1 << root)) >>> 0;
                }
            });
            expect(prepared.programWords[masks + glyph], `${file} glyph ${glyph}`).toBe(expected);
        }
    });

    it("requires a non-leading predicate witness before visiting dormant Inter composition roots", () => {
        const font = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        const selection = selectComputeTextRun("Block 01 office AVATAR quick fox 0140 ffi fl To");
        const prepared = prepareComputeTextShapeWords(createComputeTextPlan(font, selection), []);
        const masks = prepared.programWords[8]! + prepared.program.rootOrder.length * 2;
        expect(prepared.programWords.length - masks).toBe(font.glyphCount * 2 + prepared.simpleWordCount);
        const glyphs = [...selection._scalars].map((scalar) => mapComputeFontScalar(font, scalar));
        const firstBits = glyphs.reduce((bits, glyph) => (bits | prepared.programWords[masks + glyph]!) >>> 0, 0);
        const witnesses = glyphs.reduce((bits, glyph) => (bits | prepared.programWords[masks + font.glyphCount + glyph]!) >>> 0, 0);
        for (const lookupIndex of [5, 8, 11]) {
            const root = prepared.program.rootOrder.findIndex((index) => {
                const lookup = prepared.program.lookups[index]!;
                return lookup.table === "GSUB" && lookup.lookupIndex === lookupIndex;
            });
            expect(root).toBeGreaterThanOrEqual(0);
            expect(firstBits & (1 << root)).not.toBe(0);
            expect(witnesses & (1 << root)).toBe(0);
            const combining = mapComputeFontScalar(font, 0x301);
            if (lookupIndex === 5) {
                expect(prepared.programWords[masks + font.glyphCount + combining]! & (1 << root)).not.toBe(0);
            }
        }
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("return bits & witnesses;");
    });

    it("bypasses absent roots before fetching their global lookup descriptors", () => {
        const loop = COMPUTE_TEXT_KERNEL_SOURCE.slice(COMPUTE_TEXT_KERNEL_SOURCE.indexOf("for (var root = 0u;"), COMPUTE_TEXT_KERNEL_SOURCE.indexOf("let cooperative"));
        expect(loop.indexOf("candidates & (1u << root)")).toBeLessThan(loop.indexOf("let tableBit"));
    });

    it("does not re-fetch successfully committed disjoint actions in the serial rejection scan", () => {
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("selected[slot] = select(ruleAt + 1u, 0u, mode == 1u);");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("commitPairFirst(base, index, program[ruleAt + 6u], mode); selected[slot] = 0u;");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("let index = tile + slot;");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("tile += 256u");
    });

    it("uses bounded binary membership for remaining glyph-set predicates, not linear coverage scans", () => {
        expect(COMPUTE_TEXT_KERNEL_SOURCE).not.toContain("for (var i = 0u; i < head; i++)");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).not.toContain("for (var i = 0u; i < program[coverage]; i++)");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("while (low < high)");
    });

    it("does not load GDEF classes for flags that have no filtering effect", () => {
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("(flag & 0xff0eu) == 0u");
    });

    it("compacts only ordered substitution roots, never independent roots or an already compact GPOS transition", () => {
        const main = COMPUTE_TEXT_KERNEL_SOURCE.slice(COMPUTE_TEXT_KERNEL_SOURCE.indexOf("fn main("), COMPUTE_TEXT_KERNEL_SOURCE.indexOf("fn rejectRun("));
        expect(main.match(/compactRun\(/g)).toHaveLength(1);
        expect(main).toContain("!isGpos && !cooperative");
    });

    it("compacts ordered substitution scratch only after a successful ligature creates tombstones", () => {
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("!isGpos && !cooperative && removed");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("removed = false;");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("removed = true;");
    });

    it("refreshes root presence only after glyph identities or live membership actually change", () => {
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("!isGpos && atomicLoad(&idsChanged) != 0u");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("atomicStore(&idsChanged, 0u)");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("atomicStore(&idsChanged, 1u)");
    });

    it("maintains nominal advances during initialization and glyph replacement without a redundant whole-run GPOS transition", () => {
        expect(COMPUTE_TEXT_KERNEL_SOURCE).not.toContain("assignAdvances(base, length)");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).not.toContain("let firstGpos =");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("scratch[at + 2] = advanceOf(id)");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("scratch[dst + 2] = advanceOf(id)");
    });

    it("keeps the GPU kernel behind a dynamic import and fails closed on mode 5", () => {
        expect(TEXT_SHAPER_VERSION).toBe("0.1.28");
        expect(readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/_gpu/compute-text-shape.ts"), "utf8")).not.toMatch(/^const caches = new WeakMap/m);
        const source = readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/compute-text-shaper.ts"), "utf8");
        expect(source).toContain('import("./_gpu/compute-text-shape.js")');
        expect(source).not.toMatch(/from ["']text-shaper["']/);
        expect(source).not.toMatch(/from ["'].*\/font\.js["']/);
        expect(source).not.toMatch(/layout\.js/);
        expect(source).not.toMatch(/glyph-extraction\.js/);
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("@workgroup_size(32)");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).toContain("if (mode == 5u) { rejectRun(out, lookup, index); }");
        expect(COMPUTE_TEXT_KERNEL_SOURCE).not.toContain("mode == 5u) { continue;");
    });

    it("rejects a second batch, dispose, and device loss without publishing", async () => {
        const { engine, lose } = mockEngine();
        const shaper = createComputeTextShaper(engine);
        const plan = createComputeTextPlan(createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf")), selectComputeTextRun("A"));
        let release: (glyphs: ComputeTextGlyph[][]) => void = () => undefined;
        _installComputeTextShapeExecutorForTests(
            () =>
                new Promise((resolve) => {
                    release = resolve;
                })
        );
        const first = shapeComputeTextPlan(shaper, plan, ["A"]);
        expect(shaper.status).toBe("busy");
        await expect(shapeComputeTextPlan(shaper, plan, ["B"])).rejects.toThrow(/busy/);
        disposeComputeTextShaper(shaper);
        disposeComputeTextShaper(shaper);
        await expect(first).rejects.toThrow(/disposed/);
        expect(shaper.status).toBe("disposed");
        await expect(shapeComputeTextPlan(shaper, plan, ["A"])).rejects.toThrow(/disposed/);

        const next = createComputeTextShaper(engine);
        let releaseNext: (glyphs: ComputeTextGlyph[][]) => void = () => undefined;
        _installComputeTextShapeExecutorForTests(
            () =>
                new Promise((resolve) => {
                    releaseNext = resolve;
                })
        );
        const pending = shapeComputeTextPlan(next, plan, ["A"]);
        lose();
        await expect(pending).rejects.toThrow(/device lost/);
        expect(next.status).toBe("lost");
        releaseNext([]);
        release([]);
        _installComputeTextShapeExecutorForTests(null);
    });

    it("returns empty results without dispatching and publishes reference records", async () => {
        const { engine } = mockEngine();
        const shaper = createComputeTextShaper(engine);
        const bytes = fontBytes("Roboto-Regular.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const plan = createComputeTextPlan(font, selectComputeTextRun("fi"));
        await expect(shapeComputeTextPlan(shaper, plan, [])).resolves.toEqual([]);
        const empty = await shapeComputeTextPlan(shaper, plan, [""]);
        expect(empty[0]?.glyphs).toEqual([]);
        expect(empty[0]?.script).toBe("latn");
        expect(shaper.status).toBe("ready");
        let calls = 0;
        _installComputeTextShapeExecutorForTests(async (_shaper: ComputeTextShaper, active: ComputeTextPlan, runs) => {
            calls++;
            const program = compileComputeTextProgram(active);
            return runs.map((run) =>
                executeComputeTextReference(
                    program,
                    run.glyphIds,
                    [...run.glyphIds].map((_, index) => index)
                )
            );
        });
        const shaped = await shapeComputeTextPlan(shaper, plan, ["fi"]);
        expect(calls).toBe(1);
        expect(shaped[0]?.glyphs.map(fieldsOf)).toEqual(shapeFields(bytes, "fi"));
        expect(shaped[0]?.glyphs.map(fieldsOf)).toEqual(referenceFields(plan, "fi"));
        await expect(shapeComputeTextPlan(shaper, plan, ["\n"])).rejects.toThrow(/run 0/);
        expect(shaper.status).toBe("ready");
        _installComputeTextShapeExecutorForTests(null);
        disposeComputeTextShaper(shaper);
    });

    it("matches text-shaper for Roboto Latin and executable Inter kerning and marks", () => {
        const robotoBytes = fontBytes("Roboto-Regular.ttf");
        const roboto = createComputeTextPlan(createComputeFontFromBuffer(robotoBytes), selectComputeTextRun("fi"));
        for (const text of ["fi", "A", "To", " "]) {
            expect(referenceFields(roboto, text), text).toEqual(shapeFields(robotoBytes, text));
        }
        const interBytes = fontBytes("Inter.ttf");
        const inter = createComputeTextPlan(createComputeFontFromBuffer(interBytes), selectComputeTextRun("A"));
        for (const text of ["A", "AC", "5,", "A\u0326"]) {
            expect(referenceFields(inter, text), text).toEqual(shapeFields(interBytes, text));
        }
        const marks = [0x0301, 0x0300].map((scalar) => mapComputeFontScalar(inter.font, scalar));
        expect(executeComputeTextReference(compileComputeTextProgram(inter), marks, [0, 1]).map(fieldsOf)).toEqual(shapeFields(interBytes, "\u0301\u0300"));
        for (const text of ["A\u20de", "A\u20dd", "<-"]) {
            expect(() => referenceFields(inter, text), text).toThrow(/unsupported GSUB lookup/);
        }
    });

    it("shapes a character string without a caller-built plan or shaper", async () => {
        const bytes = fontBytes("Roboto-Regular.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const { engine } = mockEngine();
        _installComputeTextShapeExecutorForTests(async (_shaper, active, runs) => {
            const program = compileComputeTextProgram(active);
            return runs.map((run) =>
                executeComputeTextReference(
                    program,
                    run.glyphIds,
                    [...run.glyphIds].map((_, index) => index)
                )
            );
        });
        const glyphs = await shapeComputeText(engine, font, "fi");
        expect(glyphs.map(fieldsOf)).toEqual(shapeFields(bytes, "fi"));
        _installComputeTextShapeExecutorForTests(null);
    });
});
