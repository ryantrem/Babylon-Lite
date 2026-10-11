import type { ComputeTextData, ComputeTextUpdate } from "./compute-text-data.js";
import { submitComputeTextData } from "./compute-text-data.js";
import { getComputeFontLineMetrics } from "./compute-font.js";
import { analyzeComputeTextSegmentation } from "./compute-text-segment.js";
import type { SegmentAnalysis } from "./compute-text-segment.js";
import { compileComputeTextProgram } from "./compute-text-program.js";
import { getComputeTextParagraphStage, reserveComputeTextParagraphStage } from "./_gpu/compute-text-paragraph-layout.js";
import { updateStorageBuffer } from "../resource/storage-buffer.js";

/** Line box and paragraph placement for {@link updateComputeTextParagraphs}. */
export interface ComputeTextParagraphLayout {
    /** Line box width in CSS px. Positive; `Infinity` disables wrapping. */
    readonly maxWidthPx: number;
    /** Baseline distance in CSS px. Defaults to the font's natural line height at the block's size. */
    readonly lineHeightPx?: number;
    /** Extra distance between consecutive paragraphs in CSS px. Default 0; finite and non-negative. */
    readonly paragraphSpacingPx?: number;
    /** Default "left". "center"/"right" require finite width. */
    readonly align?: "left" | "center" | "right";
}

/** Shape, wrap and place LTR paragraphs on the GPU. Input and layout admission reject synchronously. */
export function updateComputeTextParagraphs(data: ComputeTextData, paragraphs: readonly string[], layout: ComputeTextParagraphLayout): Promise<ComputeTextUpdate> {
    try {
        if (!Array.isArray(paragraphs)) {
            throw new Error("paragraphs must be an array of strings");
        }
        for (let p = 0; p < paragraphs.length; p++) {
            const text = paragraphs[p];
            if (typeof text !== "string") {
                throw new Error(`paragraph ${p} text must be a string`);
            }
            if (/[\n\r\t\u2028\u2029]/u.test(text)) {
                throw new Error(`paragraph ${p} must not contain line breaks or tabs`);
            }
        }
        if (!layout || !(layout.maxWidthPx > 0) || (layout.maxWidthPx !== Infinity && !positiveFloat(layout.maxWidthPx))) {
            throw new Error("maxWidthPx must be positive or Infinity and representable as f32");
        }
        const align = layout.align ?? "left";
        if (!["left", "center", "right"].includes(align) || (align !== "left" && !Number.isFinite(layout.maxWidthPx))) {
            throw new Error("align must be left, center or right; center/right require finite maxWidthPx");
        }
        const spacing = layout.paragraphSpacingPx ?? 0;
        if (!Number.isFinite(spacing) || spacing < 0 || !Number.isFinite(Math.fround(spacing))) {
            throw new Error("paragraphSpacingPx must be finite, non-negative and representable as f32");
        }
        if (layout.lineHeightPx !== undefined && !positiveFloat(layout.lineHeightPx)) {
            throw new Error("lineHeightPx must be positive, finite and representable as f32");
        }
        // Read defaults in the factory, after the shared lifecycle check has admitted data.
        const maxWidth = layout.maxWidthPx;
        const requestedHeight = layout.lineHeightPx;
        return submitComputeTextData(data, paragraphs, (groups, capacity) => {
            let naturalHeight = getComputeTextParagraphStage(data)?._naturalLineHeightPx;
            const height = requestedHeight ?? (naturalHeight ??= getComputeFontLineMetrics(data._font, data.fontSizePx).lineHeightPx);
            if (!positiveFloat(height) || !positiveFloat(data.fontSizePx / data._font.unitsPerEm)) {
                throw new Error("font line height and scale must be positive, finite and representable as f32");
            }
            let wordCount = 8 + paragraphs.length;
            for (const group of groups) {
                if (group.plan.direction !== "ltr") {
                    throw new Error(`paragraph ${group.outputRun} must select an LTR script (RTL layout is unsupported)`);
                }
                for (const run of group.runs) {
                    wordCount += Math.ceil(run.scalars.length / 4);
                }
            }
            const stage = reserveComputeTextParagraphStage(data, data._engine, paragraphs.length, capacity, wordCount);
            stage._naturalLineHeightPx = naturalHeight;
            const words = stage._words;
            words[3] = data._font.glyphCount;
            words[7] = align === "left" ? 0 : align === "center" ? 1 : 2;
            const floats = stage._floatWords;
            floats[2] = data.fontSizePx / data._font.unitsPerEm;
            floats[4] = maxWidth;
            floats[5] = height;
            floats[6] = spacing;
            let at = 8 + paragraphs.length;
            for (const group of groups) {
                let analysis: SegmentAnalysis | null | undefined;
                const readAnalysis = () => {
                    if (analysis === undefined) {
                        analysis = analyzeComputeTextSegmentation(compileComputeTextProgram(group.plan));
                    }
                    return analysis;
                };
                for (let runIndex = 0; runIndex < group.runs.length; runIndex++) {
                    const run = group.runs[runIndex]!;
                    words[8 + group.outputRun + runIndex] = at;
                    for (let i = 0; i < run.scalars.length;) {
                        let word = 0;
                        for (let shift = 0; shift < 32 && i < run.scalars.length; shift += 8, i++) {
                            word |= paragraphFlag(run.scalars, run.glyphIds, i, readAnalysis) << shift;
                        }
                        words[at++] = word;
                    }
                }
            }
            return stage;
        }).then(
            (update) => {
                const completed = update.completed.catch((error: unknown) => {
                    throw paragraphError(error);
                });
                void completed.catch(() => undefined);
                return { completed };
            },
            (error: unknown) => {
                throw paragraphError(error);
            }
        );
    } catch (error) {
        if (data?.status === "ready" && !data._dead && data._device === data._engine._device) {
            updateStorageBuffer(data._engine, data._indirect, Uint32Array.of(6, 0, 0, 0));
        }
        throw paragraphError(error);
    }
}

/** @internal Scalar-indexed bytes, matching shape clusters and paragraph-level stitched clusters. */
export function computeTextParagraphFlags(scalars: Uint32Array, glyphIds: Uint32Array, analysis: SegmentAnalysis | null): Uint8Array {
    const flags = new Uint8Array(scalars.length);
    const readAnalysis = () => analysis;
    for (let i = 0; i < scalars.length; i++) {
        flags[i] = paragraphFlag(scalars, glyphIds, i, readAnalysis);
    }
    return flags;
}

function paragraphFlag(scalars: Uint32Array, glyphIds: Uint32Array, i: number, readAnalysis: () => SegmentAnalysis | null): number {
    const cp = scalars[i]!;
    if (cp !== 0x20) {
        return noBreakBefore(cp);
    }
    if (i + 1 < scalars.length && scalars[i + 1] !== 0x20) {
        const analysis = readAnalysis();
        if (!analysis) {
            throw new Error("updateComputeTextParagraphs: shaping-safe space analysis is unavailable for a paragraph with break candidates");
        }
        if (analysis.safeAfter(glyphIds, i)) {
            return 3;
        }
    }
    return 1;
}

let markPages: (Uint8Array | undefined)[] | null = null;

function noBreakBefore(cp: number): number {
    if (cp < 0x80) {
        return 0;
    }
    if (cp === 0x200d) {
        return 4;
    }
    const pages = (markPages ??= []);
    const index = cp >>> 8;
    const page = (pages[index] ??= new Uint8Array(256));
    const offset = cp & 255;
    let value = page[offset]!;
    if (value === 0) {
        value = /\p{M}/u.test(String.fromCodePoint(cp)) ? 5 : 1;
        page[offset] = value;
    }
    return value - 1;
}

function positiveFloat(value: number): boolean {
    return Number.isFinite(value) && value > 0 && Math.fround(value) > 0 && Number.isFinite(Math.fround(value));
}

function paragraphError(error: unknown): Error {
    const message = error instanceof Error ? error.message : String(error);
    return new Error(message.startsWith("updateComputeTextParagraphs:") ? message : `updateComputeTextParagraphs: ${message.replace(/^updateComputeTextData:\s*/u, "")}`, {
        cause: error,
    });
}
