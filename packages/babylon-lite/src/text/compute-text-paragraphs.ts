import type { ComputeTextData, ComputeTextUpdate } from "./compute-text-data.js";
import { submitComputeTextData } from "./compute-text-data.js";
import { getComputeFontLineMetrics } from "./compute-font.js";
import { analyzeComputeTextSegmentation } from "./compute-text-segment.js";
import type { SegmentAnalysis } from "./compute-text-segment.js";
import { compileComputeTextProgram } from "./compute-text-program.js";
import { getComputeTextParagraphStage, reserveComputeTextParagraphStage } from "./_gpu/compute-text-paragraph-layout.js";
import type { ComputeTextParagraphCache } from "./_gpu/compute-text-paragraph-layout.js";
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
        const previous = getComputeTextParagraphStage(data)?._cache;
        const cache = previous && computeTextParagraphCacheValid(data, previous) ? previous : undefined;
        const [prefix, suffix] = computeTextParagraphDiff(cache?._paragraphs ?? [], paragraphs);
        for (let p = 0; p < paragraphs.length; p++) {
            const text = paragraphs[p];
            if (typeof text !== "string") {
                throw new Error(`paragraph ${p} text must be a string`);
            }
            if (p >= prefix && p < paragraphs.length - suffix && /[\n\r\t\u2028\u2029]/u.test(text)) {
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
        const snapshot = paragraphs.slice();
        const allocation = cache ? allocateComputeTextParagraphSlots(cache._order, cache._pool, snapshot.length, prefix, suffix) : null;
        const rebuild = new Error("paragraph resident state requires a full rebuild");
        function submit(full: boolean): Promise<ComputeTextUpdate> {
            const dirty = full ? snapshot : snapshot.slice(prefix, snapshot.length - suffix);
            const options = {
                capacity: full ? 0 : cache!._capacity,
                pool: full ? computeTextParagraphPoolSize(snapshot.length) : cache!._pool,
                runBase: full ? 0 : allocation!.runBase,
                rebuild: full ? undefined : rebuild,
            };
            let candidate: ComputeTextParagraphCache;
            let epoch: number;
            const submitted = submitComputeTextData(
                data,
                dirty,
                (groups, needed) => {
                    if (!full && needed > options.capacity) {
                        throw rebuild;
                    }
                    options.capacity = full ? Math.ceil(needed * 1.25) : options.capacity;
                    const capacity = options.capacity;
                    const flagStride = Math.ceil(capacity / 4);
                    let naturalHeight =
                        previous?._font === data._font && previous._fontSize === data.fontSizePx ? getComputeTextParagraphStage(data)?._naturalLineHeightPx : undefined;
                    const height = requestedHeight ?? (naturalHeight ??= getComputeFontLineMetrics(data._font, data.fontSizePx).lineHeightPx);
                    if (!positiveFloat(height) || !positiveFloat(data.fontSizePx / data._font.unitsPerEm)) {
                        throw new Error("font line height and scale must be positive, finite and representable as f32");
                    }
                    for (const group of groups) {
                        if (group.plan.direction !== "ltr") {
                            throw new Error(`paragraph ${(full ? 0 : prefix) + group.outputRun} must select an LTR script (RTL layout is unsupported)`);
                        }
                        for (const run of group.runs) {
                            if (run.scalars.length > flagStride * 4) {
                                throw rebuild;
                            }
                        }
                    }
                    const stage = reserveComputeTextParagraphStage(data, data._engine, snapshot.length, capacity, options.pool, flagStride);
                    stage._naturalLineHeightPx = naturalHeight;
                    const words = stage._words;
                    const order = full ? Uint32Array.from(snapshot, (_text, i) => i) : allocation!.order;
                    let orderStart = 0;
                    let orderEnd = order.length;
                    if (full) {
                        words.fill(0);
                    } else {
                        while (orderStart < orderEnd && cache!._order[orderStart] === order[orderStart]) {
                            orderStart++;
                        }
                        while (orderEnd > orderStart && cache!._order[orderEnd - 1] === order[orderEnd - 1]) {
                            orderEnd--;
                        }
                    }
                    words[0] = snapshot.length;
                    words[1] = capacity;
                    words[3] = data._font.glyphCount;
                    words[7] = align === "left" ? 0 : align === "center" ? 1 : 2;
                    const floats = stage._floatWords;
                    floats[2] = data.fontSizePx / data._font.unitsPerEm;
                    floats[4] = maxWidth;
                    floats[5] = height;
                    floats[6] = spacing;
                    const measureAll = full || cache!._width !== Math.fround(maxWidth);
                    words[8] = options.runBase;
                    words[9] = measureAll ? snapshot.length : dirty.length;
                    words[10] = measureAll ? 1 : 0;
                    words[11] = options.pool;
                    words[12] = flagStride;
                    words.set(order.subarray(orderStart, orderEnd), 16 + orderStart);
                    const flagStart = 16 + options.pool + options.runBase * flagStride;
                    const flagEnd = flagStart + dirty.length * flagStride;
                    words.fill(0, flagStart, flagEnd);
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
                            let at = flagStart + (group.outputRun + runIndex) * flagStride;
                            for (let i = 0; i < run.scalars.length;) {
                                let word = 0;
                                for (let shift = 0; shift < 32 && i < run.scalars.length; shift += 8, i++) {
                                    word |= paragraphFlag(run.scalars, run.glyphIds, i, readAnalysis) << shift;
                                }
                                words[at++] = word;
                            }
                        }
                    }
                    stage._uploadRanges = full
                        ? [[0, stage._uploadWords.length]]
                        : [
                              [0, 16],
                              [16 + orderStart, 16 + orderEnd],
                              [flagStart, flagEnd],
                          ];
                    candidate = {
                        _paragraphs: snapshot,
                        _order: order,
                        _capacity: capacity,
                        _pool: options.pool,
                        _width: Math.fround(maxWidth),
                        _font: data._font,
                        _fontSize: data.fontSizePx,
                        _device: data._device,
                        _generation: data._generation,
                        _revision: data._revision,
                        _baseRevision: full ? data._revision : cache!._baseRevision,
                        _shaped: data._shaped,
                    };
                    epoch = stage._epoch;
                    return stage;
                },
                options
            );
            return submitted.then(
                (update) => {
                    const stage = getComputeTextParagraphStage(data)!;
                    candidate._shaped = data._shaped;
                    candidate._atlas = data._storage._curveSets.get(data._curveSet)?._atlas;
                    candidate._atlasVersion = candidate._atlas?._version;
                    if (stage._epoch === epoch) {
                        stage._cache = candidate;
                    }
                    const completed = update.completed.catch((error: unknown) => {
                        if (stage._cache && stage._cache._baseRevision <= candidate._revision) {
                            invalidateParagraphs(data);
                        }
                        throw paragraphError(error);
                    });
                    void completed.catch(() => undefined);
                    return { completed };
                },
                (error: unknown) => {
                    invalidateParagraphs(data);
                    if (error === rebuild && !full) {
                        return submit(true);
                    }
                    throw paragraphError(error);
                }
            );
        }
        try {
            return submit(!allocation);
        } catch (error) {
            if (error === rebuild && allocation) {
                invalidateParagraphs(data);
                return submit(true);
            }
            throw error;
        }
    } catch (error) {
        invalidateParagraphs(data);
        if (data?.status === "ready" && !data._dead && data._device === data._engine._device) {
            updateStorageBuffer(data._engine, data._indirect, Uint32Array.of(6, 0, 0, 0));
        }
        throw paragraphError(error);
    }
}

/** @internal Positional middle replacement, with a non-overlapping common suffix. */
export function computeTextParagraphDiff(previous: readonly string[], next: readonly string[]): readonly [number, number] {
    let prefix = 0;
    const limit = Math.min(previous.length, next.length);
    while (prefix < limit && previous[prefix] === next[prefix]) {
        prefix++;
    }
    let suffix = 0;
    while (suffix < limit - prefix && previous[previous.length - suffix - 1] === next[next.length - suffix - 1]) {
        suffix++;
    }
    return [prefix, suffix];
}

/** @internal Compacted pool with insertion headroom. */
export function computeTextParagraphPoolSize(count: number): number {
    return count + Math.max(8, Math.ceil(count / 4));
}

/** @internal Free the replaced middle and first-fit one contiguous dirty batch; null requests compaction. */
export function allocateComputeTextParagraphSlots(
    previous: Uint32Array,
    pool: number,
    count: number,
    prefix: number,
    suffix: number
): { order: Uint32Array; runBase: number } | null {
    if (count > pool) {
        return null;
    }
    const used = new Uint8Array(pool);
    const order = new Uint32Array(count);
    order.set(previous.subarray(0, prefix));
    order.set(previous.subarray(previous.length - suffix), count - suffix);
    for (let p = 0; p < count; p++) {
        if (p < prefix || p >= count - suffix) {
            used[order[p]!] = 1;
        }
    }
    const dirty = count - prefix - suffix;
    let free = 0;
    for (let slot = 0; slot < pool && dirty > 0; slot++) {
        free = used[slot] ? 0 : free + 1;
        if (free === dirty) {
            const runBase = slot + 1 - dirty;
            for (let p = 0; p < dirty; p++) {
                order[prefix + p] = runBase + p;
            }
            return { order, runBase };
        }
    }
    return dirty === 0 ? { order, runBase: 0 } : null;
}

/** @internal Reuse only the immediately preceding resident submission and unchanged stride resources. */
export function computeTextParagraphCacheValid(data: ComputeTextData, cache: ComputeTextParagraphCache): boolean {
    const stage = getComputeTextParagraphStage(data);
    return (
        cache._revision === data._revision &&
        cache._generation === data._generation &&
        cache._font === data._font &&
        cache._fontSize === data.fontSizePx &&
        cache._device === data._engine._device &&
        cache._shaped === data._shaped &&
        !data._shaped._destroyed &&
        cache._atlas === data._storage._curveSets.get(data._curveSet)?._atlas &&
        cache._atlasVersion === cache._atlas?._version &&
        (!stage || (stage._words[1] === cache._capacity && stage._words[11] === cache._pool))
    );
}

function invalidateParagraphs(data: ComputeTextData): void {
    const stage = data && getComputeTextParagraphStage(data);
    if (stage) {
        stage._cache = undefined;
        stage._epoch++;
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
