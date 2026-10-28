import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createComputeFontFromBuffer, mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { scanComputeTextRun, selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { planComputeTextParagraphChunks, safeComputeTextSourceBoundary } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-chunks";
import { checkComputeTextChunkWindow } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-chunks";

function font(name = "Inter.ttf") {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", name));
    return createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}
const limits = {
    maxBufferSize: 268435456,
    maxStorageBufferBindingSize: 134217728,
    maxComputeWorkgroupsPerDimension: 65535,
} as GPUSupportedLimits;

describe("bounded paragraph source planning", () => {
    it("checks actual allocator window offsets, alignment, binding and allocation limits independently", () => {
        const device = { maxBufferSize: 16384, maxStorageBufferBindingSize: 8192, minStorageBufferOffsetAlignment: 256 };
        expect(() => checkComputeTextChunkWindow(device, 4096, "instances", 4096, 256, 16384)).not.toThrow();
        for (const [bytes, offset, allocation] of [
            [4100, 0, 16384],
            [8196, 0, 16384],
            [4096, 1, 16384],
            [4095, 256, 16384],
            [4096, 256, 4096],
            [4096, 256, 16388],
            [Number.MAX_SAFE_INTEGER, 256, Number.MAX_SAFE_INTEGER],
            [NaN, 0, 16384],
            [4096, -256, 16384],
        ]) {
            expect(() => checkComputeTextChunkWindow(device, 4096, "instances", bytes!, offset!, allocation!)).toThrow(
                /instances window.*allocation\/binding bounds.*alignment=256.*bindingLimit=4096.*maxBufferSize=16384.*maxStorageBufferBindingSize=8192/
            );
        }
    });
    it.each(["", "   ", "123A\u0308", "𐌰 𐌱", "\u0308A", "A\u0308", "שלום", "سلام"])("matches whole-run selection without retaining its arrays: %j", (text) => {
        const selected = selectComputeTextRun(text);
        expect(scanComputeTextRun(text)).toEqual({
            script: selected.script,
            direction: selected.direction,
            language: selected.language,
            scalarCount: selected.scalarCount,
        });
        expect("_scalars" in scanComputeTextRun(text)).toBe(false);
    });

    it.each(["Aא", "\u0308", "A\u200dB", "A\nB", "\u2066A", "\ud800A", "A\u200d\ud800", "অ", "A\ufe0f", "A\udc00"])(
        "retains selection rejection precedence and witnesses: %j",
        (text) => {
            let expected: unknown;
            try {
                selectComputeTextRun(text);
            } catch (error) {
                expected = error;
            }
            expect(expected).toBeInstanceOf(Error);
            expect(() => scanComputeTextRun(text)).toThrow((expected as Error).message);
        }
    );

    it.each(["Inter.ttf", "Roboto-Regular.ttf"])("matches the full-vector proof with complete 65-neighbor windows: %s", (name) => {
        const face = font(name);
        const text = "office ".repeat(20) + "A" + "\u0308".repeat(64) + " B " + "\u0308".repeat(65) + " C " + "AV To ".repeat(25);
        const selected = selectComputeTextRun(text);
        const glyphs = mapComputeFontScalars(face, selected._scalars, (_i, error) => {
            throw error;
        });
        const analysis = analyzeComputeTextSegmentation(compileComputeTextProgram(createComputeTextPlan(face, selected)))!;
        for (let scalar = 0; scalar < selected.scalarCount; scalar++) {
            if (selected._scalars[scalar] === 32) {
                expect(safeComputeTextSourceBoundary(face, analysis, text, selected._utf16Offsets[scalar]!)).toBe(analysis.safeAfter(glyphs, scalar));
            }
        }
    });

    it("packs real bounded expanded records without the longest-paragraph pool multiplier", () => {
        const face = font();
        const plan = planComputeTextParagraphChunks(face, ["office affinity ".repeat(300), "A", ""], limits, { records: 64, arenaBytes: 2048 });
        expect(plan.chunks[0]!.length).toBeGreaterThan(10);
        expect(plan.chunks[1]).toHaveLength(1);
        expect(plan.chunks[1]![0]!.capacity).toBeLessThan(4);
        expect(plan.chunks.map((chunks) => chunks.map((chunk) => chunk.text).join(""))).toEqual(plan.paragraphs);
        expect(plan.workspaceBytes).toBeLessThanOrEqual(8 * 1024 * 1024);
        for (const chunks of plan.chunks) {
            for (const chunk of chunks) {
                expect(chunk.capacity).toBeLessThanOrEqual(64);
                expect(chunk.segments.at(-1)).toBe(chunk.scalars);
            }
        }
    });

    it("splits optional uneven shaping groups instead of rejecting independently fitting spans", () => {
        const face = font();
        const text = "a ".repeat(80) + "A".repeat(15000) + " " + "a ".repeat(200);
        const plan = planComputeTextParagraphChunks(face, [text], limits);
        expect(plan.chunks[0]!.map((chunk) => chunk.text).join("")).toBe(text);
        expect(plan.chunks[0]!.length).toBeGreaterThan(1);
        expect(plan.chunks[0]!.some((chunk) => chunk.text.includes("A".repeat(15000)))).toBe(true);
        expect(plan.workspaceBytes).toBeLessThanOrEqual(3 * 1024 * 1024);
    });

    it("preserves shifted suffix seams rather than retiling from scalar zero", () => {
        const face = font();
        const original = "office affinity ".repeat(80);
        const before = planComputeTextParagraphChunks(face, [original], limits, { records: 64 });
        const after = planComputeTextParagraphChunks(face, ["A " + original], limits, { records: 64 }, before);
        const old = before.chunks[0]!;
        const next = after.chunks[0]!;
        expect(next.slice(-5).map((chunk) => chunk.text)).toEqual(old.slice(-5).map((chunk) => chunk.text));
        expect(next.at(-1)!.start).toBe(old.at(-1)!.start + 2);
        expect(next.at(-1)!.scalarStart).toBe(old.at(-1)!.scalarStart + 2);
    });

    it("admits a fitting dedicated unsafe span and reports exact required over-budget bytes", () => {
        const face = font();
        const fitting = planComputeTextParagraphChunks(face, ["A".repeat(100)], limits, { records: 16, bindingBytes: 8192 });
        expect(fitting.chunks[0]).toHaveLength(1);
        expect(fitting.chunks[0]![0]!.capacity).toBe(100);
        expect(() => planComputeTextParagraphChunks(face, ["A".repeat(200)], limits, { records: 16, bindingBytes: 8192 })).toThrow(
            /paragraph 0 indivisible scalar span.*replay buffer of \d+ bytes exceeds binding limit 8192.*maxStorageBufferBindingSize=134217728/
        );
    });
});
