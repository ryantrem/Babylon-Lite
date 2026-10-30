import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { createComputeFontFromBuffer, mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import type { ComputeFont } from "../../../packages/babylon-lite/src/text/compute-font";
import { decodeComputeTextChunk } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { ensureComputeTextProfile, profileSupport } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { computeTextParagraphSpaceProof } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { computeTextParagraphFlagsIncremental } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-flags";
import { planComputeTextParagraphChunks } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-chunks";
import type { ComputeTextChunkBudgets } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-chunks";
import { computeTextChunkFlags } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-chunks";

/** CTF-03 03_02: paged Devanagari planning — profile capacity, run validation and chunk flag bytes. */
function font(name: string): ComputeFont {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", name));
    return createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
}
const limits = {
    maxBufferSize: 268435456,
    maxStorageBufferBindingSize: 134217728,
    maxComputeWorkgroupsPerDimension: 65535,
} as GPUSupportedLimits;
const reference = JSON.parse(readFileSync(join(process.cwd(), "tests/lite/compute/references/deva-paragraph-harfbuzz-11.0.1.json"), "utf8")) as {
    cases: { id: string; text: string }[];
};
const row = (id: string) => reference.cases.find((c) => c.id === id)!.text;
const glyphsOf = (face: ComputeFont, scalars: Uint32Array) =>
    mapComputeFontScalars(face, scalars, (_i, error) => {
        throw error;
    });

let deva: ComputeFont;
beforeAll(async () => {
    await ensureComputeTextProfile("indic");
    deva = font("NotoSansDevanagari-Regular.ttf");
});

/** Paged flag bytes per chunk, compared with the monolithic producer over the whole paragraph. */
function expectPagedFlagsEqualMonolithic(text: string, budgets: ComputeTextChunkBudgets): number {
    const plan = planComputeTextParagraphChunks(deva, [text], limits, budgets);
    const program = compileComputeTextProgram(plan.plans[0]!);
    const proof = computeTextParagraphSpaceProof(program);
    const scalars = decodeComputeTextChunk(text).scalars;
    const monolithic = computeTextParagraphFlagsIncremental(undefined, scalars, glyphsOf(deva, scalars), () => proof, undefined, undefined, profileSupport("indic")!.paragraph);
    const chunks = plan.chunks[0]!;
    for (const chunk of chunks) {
        const context = decodeComputeTextChunk(chunk.before + chunk.text + chunk.after);
        const prefix = context.offsets.indexOf(chunk.before.length);
        const flags = computeTextChunkFlags(chunk.plan, proof, context.scalars, glyphsOf(deva, context.scalars), prefix, chunk.scalars, chunk.scalars + 1);
        const expected = [chunk.scalarStart ? monolithic[chunk.scalarStart - 1]! : 0, ...monolithic.subarray(chunk.scalarStart, chunk.scalarStart + chunk.scalars)];
        expect([...flags], `chunk at scalar ${chunk.scalarStart}`).toEqual(expected);
    }
    expect(chunks.reduce((sum, chunk) => sum + chunk.scalars, 0)).toBe(scalars.length);
    return chunks.length;
}

describe("paged Devanagari paragraph planning", () => {
    it.each(["plain-rain", "reph", "nukta", "nbsp", "danda", "space-before-mark", "overlong-between", "overlong-boundary", "multiseam-unit", "multiseam-forced"])(
        "chunk flag bytes equal the monolithic flags under forced small budgets (SYL-5): %s",
        (id) => {
            let paged = 0;
            for (const records of [24, 64, 200]) {
                paged = Math.max(paged, expectPagedFlagsEqualMonolithic(row(id), { force: true, records }));
            }
            if (row(id).length > 100) {
                expect(paged).toBeGreaterThan(1);
            }
        }
    );

    it("keeps chunk flags equal across a long virama chain and repeated words", () => {
        const chain = "क" + "्क".repeat(40);
        expect(expectPagedFlagsEqualMonolithic(`नमस्ते ${chain} दुनिया ${"राम सीता ".repeat(30)}${chain}`, { force: true, records: 40 })).toBeGreaterThan(1);
    });

    it("sizes chunks with the additive profile capacity for ◌-insertion and decomposition heavy text (R4)", () => {
        const text = Array.from({ length: 60 }, (_, i) => ["क़ऩ", "ि", "ाॉ", "ऩक़ि", "ऱ्य"][i % 5]).join(" ");
        const plan = planComputeTextParagraphChunks(deva, [text], limits, { force: true, records: 64 });
        const capacity = plan.plans[0]!._profileData!.capacity!;
        const expansion = Math.max(1, compileComputeTextProgram(plan.plans[0]!).maxExpansion);
        expect(plan.chunks[0]!.length).toBeGreaterThan(1);
        for (const chunk of plan.chunks[0]!) {
            const scalars = decodeComputeTextChunk(chunk.text).scalars;
            expect(chunk.capacity).toBe(capacity(scalars));
            expect(chunk.capacity).toBeGreaterThan(chunk.scalars * expansion);
        }
        expect(plan.maxCapacity).toBeGreaterThanOrEqual(Math.max(...plan.chunks[0]!.map((chunk) => chunk.capacity)));
    });

    it("bounds default-budget Devanagari batches by the indic kernel's per-record scratch", () => {
        const plan = planComputeTextParagraphChunks(deva, [row("multiseam-default")], limits, { force: true });
        expect(plan.chunks[0]!.length).toBeGreaterThan(2);
        expect(plan.workspaceBytes).toBeLessThanOrEqual(3 * 1024 * 1024);
    });

    it("throws the binding-limit error for an indivisible Devanagari word (PATH-3)", () => {
        const word = "कखगघङचछजझञ".repeat(20);
        expect(() => planComputeTextParagraphChunks(deva, [`नमस्ते ${word} दुनिया`], limits, { records: 16, bindingBytes: 8192 })).toThrow(
            /paragraph 0 indivisible scalar span.*replay buffer of \d+ bytes exceeds binding limit 8192.*maxStorageBufferBindingSize=134217728/
        );
    });

    it("keeps the Latin chunk capacity formula", () => {
        const face = font("Inter.ttf");
        const plan = planComputeTextParagraphChunks(face, [Array.from({ length: 80 }, (_, i) => `word${i} efficient`).join(" ")], limits, { force: true, records: 48 });
        const expansion = Math.max(1, compileComputeTextProgram(plan.plans[0]!).maxExpansion);
        expect(plan.plans[0]!._profileData?.capacity).toBeUndefined();
        expect(plan.chunks[0]!.length).toBeGreaterThan(1);
        for (const chunk of plan.chunks[0]!) {
            expect(chunk.capacity).toBe(Math.max(1, chunk.scalars * expansion));
        }
    });
});
