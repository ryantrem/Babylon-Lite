import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeFontFromBuffer, updateComputeTextParagraphs } from "../../../packages/babylon-lite/src/index";
import type { ComputeFont } from "../../../packages/babylon-lite/src/index";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { createComputeTextData, disposeComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import * as paragraphStages from "../../../packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout";
import { computeTextParagraphFlagsIncremental } from "../../../packages/babylon-lite/src/text/compute-text-paragraph-flags";
import * as profiles from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { computeTextParagraphSpaceProof } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalars } from "../../../packages/babylon-lite/src/text/compute-font";
import { installComputeTextIndicProfile } from "../../../packages/babylon-lite/src/text/compute-text-indic";

/** CTF-03 03_01: monolithic Devanagari paragraph admission, lazy profile loading and the profile flag seam. */
const faces = new Map<string, ComputeFont>();
function font(file: string): ComputeFont {
    let face = faces.get(file);
    if (!face) {
        const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", file));
        face = createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
        faces.set(file, face);
    }
    return face;
}

function block(file = "NotoSansDevanagari-Regular.ttf") {
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
    const storage = createGlyphStorage();
    const data = createComputeTextData({ _device: device } as unknown as EngineContext, font(file), storage, "unit");
    data._chunkBudgets = { monolithic: true };
    device.createBuffer.mockClear();
    return {
        data,
        storage,
        close() {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        },
    };
}

/** Stops a monolithic update after flags are written, before any GPU pipeline work the mock device cannot run. */
function stopAfterFlags() {
    const sentinel = new Error("test stopped after paragraph flags");
    const reserve = paragraphStages.reserveComputeTextParagraphStage;
    let stage: ReturnType<typeof reserve> | undefined;
    vi.spyOn(paragraphStages, "reserveComputeTextParagraphStage").mockImplementation((...args) => {
        stage = reserve(...args);
        Object.defineProperty(stage, "_uploadRanges", {
            configurable: true,
            set() {
                throw sentinel;
            },
        });
        return stage;
    });
    return sentinel;
}

const scalarsOf = (text: string): Uint32Array => Uint32Array.from([...text].map((c) => c.codePointAt(0)!));

afterEach(() => vi.restoreAllMocks());

describe("Devanagari paragraph lazy loading", () => {
    it("validates layout and controls synchronously before loading the profile (API-3)", () => {
        const importer = vi.fn(async () => installComputeTextIndicProfile());
        profiles.installComputeTextProfileLoader("indic", importer);
        const { data, close } = block();
        try {
            expect(() => updateComputeTextParagraphs(data, ["क ख"], { maxWidthPx: 0 })).toThrow("maxWidthPx must be positive");
            expect(() => updateComputeTextParagraphs(data, ["क\tख"], { maxWidthPx: 100 })).toThrow("paragraph 0 must not contain line breaks or tabs");
            expect(() => updateComputeTextParagraphs(data, ["क", "ख\n"], { maxWidthPx: 100 })).toThrow("paragraph 1 must not contain line breaks or tabs");
            expect(importer).not.toHaveBeenCalled();
            expect(data._device.createBuffer).not.toHaveBeenCalled();
        } finally {
            close();
        }
    });

    it("rejects mixed Devanagari and Latin letters synchronously without loading (MIX-3)", () => {
        const importer = vi.fn(async () => installComputeTextIndicProfile());
        profiles.installComputeTextProfileLoader("indic", importer);
        const { data, close } = block();
        try {
            expect(() => updateComputeTextParagraphs(data, ["कa"], { maxWidthPx: 100 })).toThrow(/^updateComputeTextParagraphs: .*mixed scripts "deva" and "latn"/);
            expect(importer).not.toHaveBeenCalled();
            expect(data._device.createBuffer).not.toHaveBeenCalled();
            expect(data.status).toBe("ready");
        } finally {
            close();
        }
    });

    it("propagates an import failure as a paragraph rejection and retries on the next call", async () => {
        const failure = new Error("indic chunk failed to load");
        const importer = vi
            .fn()
            .mockRejectedValueOnce(failure)
            .mockImplementation(async () => installComputeTextIndicProfile());
        profiles.installComputeTextProfileLoader("indic", importer);
        const { data, close } = block();
        const version = data._revision;
        try {
            await expect(updateComputeTextParagraphs(data, ["क"], { maxWidthPx: 100 })).rejects.toThrow("updateComputeTextParagraphs: indic chunk failed to load");
            expect(data._revision).toBe(version);
            expect(data.status).toBe("ready");
            const sentinel = stopAfterFlags();
            await expect(updateComputeTextParagraphs(data, ["क"], { maxWidthPx: 100 })).rejects.toThrow(sentinel.message);
            expect(importer).toHaveBeenCalledTimes(2);
        } finally {
            close();
        }
    });
});

describe("Devanagari monolithic paragraph flags", () => {
    beforeAll(() => profiles.ensureComputeTextProfile("indic"));

    it("routes Devanagari runs through the profile flag seam with the monolithic unavailable message", () => {
        const seam = profiles.profileSupport("indic")!.paragraph!;
        const flags = vi.spyOn(seam, "flags");
        const sentinel = stopAfterFlags();
        const { data, close } = block();
        try {
            expect(() => updateComputeTextParagraphs(data, ["नमस्ते दुनिया"], { maxWidthPx: 100 })).toThrow(sentinel.message);
            expect(flags).toHaveBeenCalledOnce();
            expect(flags.mock.calls[0]![7]).toBe("updateComputeTextParagraphs: shaping-safe space analysis is unavailable for a paragraph with break candidates");
        } finally {
            close();
        }
    });

    it("keeps Latin paragraphs on the per-scalar rule", () => {
        const flags = vi.spyOn(profiles.profileSupport("indic")!.paragraph!, "flags");
        const sentinel = stopAfterFlags();
        const { data, close } = block("Inter.ttf");
        try {
            expect(() => updateComputeTextParagraphs(data, ["Hello world"], { maxWidthPx: 100 })).toThrow(sentinel.message);
            expect(flags).not.toHaveBeenCalled();
        } finally {
            close();
        }
    });

    it.each([
        { text: "क ख", unavailable: true },
        { text: "कख", unavailable: false },
        { text: "क ", unavailable: false },
        { text: "क  ", unavailable: false },
    ])("needs a proof only for candidate spaces (PROOF-4): $text", ({ text, unavailable }) => {
        vi.spyOn(profiles.profileSupport("indic")!.paragraph!, "spaces").mockReturnValue(null);
        const sentinel = stopAfterFlags();
        const { data, storage, close } = block();
        try {
            expect(() => updateComputeTextParagraphs(data, [text], { maxWidthPx: 100 })).toThrow(
                unavailable ? "updateComputeTextParagraphs: shaping-safe space analysis is unavailable for a paragraph with break candidates" : sentinel.message
            );
            expect(storage._curveSets.size).toBe(0);
            expect(data.status).toBe("ready");
        } finally {
            close();
        }
    });

    it("rejects unsupported Devanagari scalars on the paged path with the monolithic message", async () => {
        const { data, close } = block();
        try {
            for (const budgets of [{ monolithic: true }, { force: true }]) {
                data._chunkBudgets = budgets;
                await expect((async () => updateComputeTextParagraphs(data, ["क\u25cc ख"], { maxWidthPx: 100 }))()).rejects.toThrow(
                    "updateComputeTextParagraphs: run 0 unsupported scalar in Devanagari run at scalar 1 U+25CC"
                );
            }
        } finally {
            close();
        }
    });

    it("incremental flags with the grapheme chain extension equal a full recompute across random edits", () => {
        const face = font("NotoSansDevanagari-Regular.ttf");
        const profile = profiles.profileSupport("indic")!.paragraph!;
        const alphabet = [..."कखगघचजटडतथदनपबमयरलवशसह", "्", "ा", "ि", "ी", "ु", "े", "ो", "ं", "ः", "ँ", "़", " ", "\u200d", "\u200c", "।", "१"];
        let seed = 0x2545f491;
        const random = (n: number) => {
            seed ^= seed << 13;
            seed ^= seed >>> 17;
            seed ^= seed << 5;
            return (seed >>> 0) % n;
        };
        const build = (n: number) => Array.from({ length: n }, () => alphabet[random(alphabet.length)]!).join("");
        const flagsOf = (text: string, previous?: { scalars: Uint32Array; glyphIds: Uint32Array; flags: Uint8Array }) => {
            const scalars = scalarsOf(text);
            const glyphIds = mapComputeFontScalars(face, scalars, (_i, error) => {
                throw error;
            });
            const proof = computeTextParagraphSpaceProof(compileComputeTextProgram(createComputeTextPlan(face, selectComputeTextRun("क"))));
            return { scalars, glyphIds, flags: computeTextParagraphFlagsIncremental(previous, scalars, glyphIds, () => proof, undefined, undefined, profile) };
        };
        // A consonant + 100 viramas + consonant: the GB9c flag before the final consonant depends on the first scalar,
        // more than 66 scalars back, so a first-scalar edit is only seen through the grapheme chain extension.
        const padding = " कखग".repeat(60);
        for (const [first, next] of [
            ["क", "अ"],
            ["अ", "क"],
            ["क", " "],
        ] as const) {
            const linked = "्".repeat(100) + "ख" + padding;
            expect(flagsOf(next + linked, flagsOf(first + linked)).flags, `${first} → ${next}`).toEqual(flagsOf(next + linked).flags);
        }
        const chain = "क्" + "ष्\u200d".repeat(40) + "ा";
        for (let trial = 0; trial < 60; trial++) {
            const before = trial % 3 === 0 ? build(80) + chain + build(80) : build(200 + random(200));
            const previous = flagsOf(before);
            const at = random(before.length + 1);
            const removed = random(4);
            const after = before.slice(0, at) + build(random(4)) + before.slice(Math.min(before.length, at + removed));
            expect(flagsOf(after, previous).flags, `trial ${trial}: ${JSON.stringify(after)}`).toEqual(flagsOf(after).flags);
        }
    });
});
