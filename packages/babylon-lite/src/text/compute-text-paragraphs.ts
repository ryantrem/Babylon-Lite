import type { ComputeTextData, ComputeTextUpdate } from "./compute-text-data.js";
import { computeTextDrawBufferSizes, submitComputeTextData } from "./compute-text-data.js";
import { getComputeFontLineMetrics } from "./compute-font.js";
import { SAFE_SPACE_FLAG, computeTextParagraphSpaceProof } from "./compute-text-segment.js";
import type { ComputeTextSpaceProof } from "./compute-text-segment.js";
import { compileComputeTextProgram } from "./compute-text-program.js";
import { ensureComputeTextProfile, profileSupport } from "./compute-text-plan.js";
import type { ComputeTextParagraphProfile } from "./compute-text-plan.js";
import {
    checkComputeTextParagraphBufferSize,
    computeTextParagraphBufferSizes,
    computeTextParagraphSplitStages,
    getComputeTextParagraphStage,
    reserveComputeTextParagraphStage,
} from "./_gpu/compute-text-paragraph-layout.js";
import type { ComputeTextParagraphCache, ComputeTextParagraphFlagEntry } from "./_gpu/compute-text-paragraph-layout.js";
import { updateStorageBuffer } from "../resource/storage-buffer.js";
import { needsComputeTextParagraphChunks, submitChunkedComputeTextParagraphs } from "./compute-text-paragraph-chunks.js";
import { computeTextNoBreakBefore, scanComputeTextRun } from "./compute-text-selection.js";
import type { ComputeTextRunSelection } from "./compute-text-selection.js";

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
    let paged: Error | undefined;
    let admittedSnapshot: readonly string[] = paragraphs;
    let admittedLayout = layout;
    try {
        data?._engine?._flushPendingWork?.(data._engine);
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
            if (p >= prefix && p < paragraphs.length - suffix) {
                if (/[\n\r\t\u2028\u2029]/u.test(text)) {
                    throw new Error(`paragraph ${p} must not contain line breaks or tabs`);
                }
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
        if (!profileSupport("indic") && snapshot.slice(prefix, snapshot.length - suffix).some(selectsDevanagari)) {
            // Load before any mutation, then re-enter: later errors keep their wording and order, rejected only on this first load.
            const admitted = { ...layout };
            return ensureComputeTextProfile("indic").then(
                () => updateComputeTextParagraphs(data, snapshot, admitted),
                (error: unknown) => {
                    throw paragraphError(error);
                }
            );
        }
        admittedSnapshot = snapshot;
        if (needsComputeTextParagraphChunks(data, snapshot, !!cache, prefix, snapshot.length - suffix)) {
            admittedLayout = { maxWidthPx: maxWidth, lineHeightPx: requestedHeight, paragraphSpacingPx: spacing, align };
            return submitChunkedComputeTextParagraphs(data, snapshot, admittedLayout).catch((error: unknown) => {
                throw paragraphError(error);
            });
        }
        const allocation = cache ? allocateComputeTextParagraphSlots(cache._order, cache._pool, snapshot.length, prefix, suffix) : null;
        const rebuild = new Error("paragraph resident state requires a full rebuild");
        function submit(full: boolean): Promise<ComputeTextUpdate> {
            const dirty = full ? snapshot : snapshot.slice(prefix, snapshot.length - suffix);
            const safeSpaces: Uint8Array[][] = [];
            const selections: ComputeTextRunSelection[] = [];
            const priorFlags = cache?._flags;
            const reuseEnd = full ? (priorFlags?.length ?? 0) : (cache?._paragraphs.length ?? 0) - suffix;
            const previousRuns = priorFlags ? dirty.map((_text, i) => (base(i) < reuseEnd ? priorFlags[base(i)]?.reuse : undefined)) : undefined;
            function base(i: number): number {
                return (full ? 0 : prefix) + i;
            }
            const options = {
                capacity: full ? 0 : cache!._capacity,
                pool: full ? snapshot.length : cache!._pool,
                runBase: full ? 0 : allocation!.runBase,
                rebuild: full ? undefined : rebuild,
                safeSpaces,
                previous: previousRuns,
                selections,
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
                    if (full) {
                        const limit = Math.min(data._device.limits.maxBufferSize, data._device.limits.maxStorageBufferBindingSize);
                        let independent = 0;
                        for (const group of groups) {
                            const expansion = Math.max(1, compileComputeTextProgram(group.plan).maxExpansion);
                            for (const run of group.runs) {
                                independent += Math.max(1, run.scalars.length * expansion) * 48;
                            }
                        }
                        if (
                            !data._chunkBudgets?.monolithic &&
                            (snapshot.length * needed * 48 > Math.max(4 * 1024 * 1024, independent * 2) ||
                                [...computeTextParagraphBufferSizes(snapshot.length, needed), ...computeTextDrawBufferSizes(snapshot.length, needed)].some(
                                    ([, bytes]) => !Number.isSafeInteger(bytes) || bytes > limit
                                ))
                        ) {
                            admittedLayout = { maxWidthPx: maxWidth, lineHeightPx: requestedHeight, paragraphSpacingPx: spacing, align };
                            throw (paged ??= new Error("paragraph required storage needs paging"));
                        }
                        Object.assign(options, computeTextParagraphAllocation(data._device.limits, snapshot.length, needed));
                    }
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
                    words[13] = computeTextParagraphSplitStages(capacity);
                    words.set(order.subarray(orderStart, orderEnd), 16 + orderStart);
                    const flagStart = 16 + options.pool + options.runBase * flagStride;
                    const flagEnd = flagStart + dirty.length * flagStride;
                    words.fill(0, flagStart, flagEnd);
                    // Kernels read flags only below each paragraph's scalar count, so the upload stops after the last dirty paragraph's flag words.
                    let flagUploadEnd = flagStart;
                    const flagEntries: (ComputeTextParagraphFlagEntry | undefined)[] = full
                        ? new Array(snapshot.length)
                        : (cache!._flags?.slice() ?? new Array(cache!._paragraphs.length));
                    if (!full) {
                        flagEntries.splice(prefix, cache!._paragraphs.length - prefix - suffix, ...new Array<undefined>(dirty.length));
                    }
                    const reuseFlags = !full && cache!._paragraphs.length === snapshot.length;
                    for (let g = 0; g < groups.length; g++) {
                        const group = groups[g]!;
                        const groupFlags: Uint8Array[] = (safeSpaces[g] = []);
                        let proof: ComputeTextSpaceProof | null | undefined;
                        const readProof = () => {
                            if (proof === undefined) {
                                proof = computeTextParagraphSpaceProof(compileComputeTextProgram(group.plan));
                            }
                            return proof;
                        };
                        for (let runIndex = 0; runIndex < group.runs.length; runIndex++) {
                            const run = group.runs[runIndex]!;
                            const paragraph = (full ? 0 : prefix) + group.outputRun + runIndex;
                            const old = reuseFlags ? cache!._flags?.[paragraph] : undefined;
                            const selection = selections[group.outputRun + runIndex]!;
                            const prior = old?.plan === group.plan ? old : undefined;
                            // The selection's unchanged span is proven against the record it reused; it applies only when that record is this entry's.
                            const known = prior?.reuse && prior.reuse === previousRuns?.[group.outputRun + runIndex] ? selection._unchanged : undefined;
                            const flags = computeTextParagraphFlagsIncremental(
                                prior,
                                run.scalars,
                                run.glyphIds,
                                readProof,
                                undefined,
                                known,
                                profileSupport(group.plan._profile)?.paragraph
                            );
                            groupFlags[runIndex] = flags;
                            // Runs view a per-submit slab nothing writes after selection; retain it unless it is much larger than this run.
                            const runBytes = run.scalars.byteLength + run.glyphIds.byteLength;
                            const scalars = run.scalars.buffer.byteLength > 2 * runBytes ? run.scalars.slice() : run.scalars;
                            const glyphIds = run.glyphIds.buffer.byteLength > 2 * runBytes ? run.glyphIds.slice() : run.glyphIds;
                            const strong = selection._reuseStrong;
                            flagEntries[paragraph] = {
                                plan: group.plan,
                                scalars,
                                glyphIds,
                                flags,
                                reuse:
                                    strong === undefined
                                        ? undefined
                                        : { text: dirty[group.outputRun + runIndex]!, font: data._font, scalars, glyphIds, strong, firsts: selection._firsts },
                            };
                            // Byte i lands at bits 8 * (i % 4) of the little-endian u32 words the kernel reads.
                            const at = flagStart + (group.outputRun + runIndex) * flagStride;
                            new Uint8Array(words.buffer, words.byteOffset + at * 4, flags.length).set(flags);
                            flagUploadEnd = Math.max(flagUploadEnd, at + Math.ceil(flags.length / 4));
                        }
                    }
                    stage._uploadRanges = full
                        ? [[0, stage._uploadWords.length]]
                        : [
                              [0, 16],
                              [16 + orderStart, 16 + orderEnd],
                              [flagStart, flagUploadEnd],
                          ];
                    candidate = {
                        _paragraphs: snapshot,
                        _flags: flagEntries,
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
                        try {
                            return submit(true);
                        } catch (retry) {
                            if (paged && retry === paged) {
                                return submitChunkedComputeTextParagraphs(data, snapshot, admittedLayout);
                            }
                            throw retry;
                        }
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
        if (paged && error === paged) {
            return submitChunkedComputeTextParagraphs(data, admittedSnapshot, admittedLayout).catch((failure: unknown) => {
                throw paragraphError(failure);
            });
        }
        if (data?.status === "ready" && !data._dead && data._device === data._engine._device) {
            updateStorageBuffer(data._engine, data._indirect, Uint32Array.of(6, 0, 0, 0));
            data._paragraphResources?.clear();
            if (data._chunkState) {
                data._chunkState.revision = -1;
            }
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

/** @internal Required live storage first; prefer growth capacity, then insertion slots within the device budget. */
export function computeTextParagraphAllocation(
    limits: Pick<GPUSupportedLimits, "maxBufferSize" | "maxStorageBufferBindingSize">,
    count: number,
    needed: number
): { pool: number; capacity: number } {
    const sizes = (pool: number, capacity: number) => [...computeTextParagraphBufferSizes(pool, capacity), ...computeTextDrawBufferSizes(pool, capacity)];
    for (const [name, bytes] of sizes(count, needed)) {
        checkComputeTextParagraphBufferSize(limits, name, bytes, count, needed);
    }
    const limit = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
    const fits = (pool: number, capacity: number) => sizes(pool, capacity).every(([, bytes]) => Number.isSafeInteger(bytes) && bytes <= limit);
    let capacity = needed;
    let high = Math.ceil(needed * 1.25);
    while (capacity < high) {
        const middle = capacity + Math.ceil((high - capacity) / 2);
        if (fits(count, middle)) {
            capacity = middle;
        } else {
            high = middle - 1;
        }
    }
    let pool = count;
    high = computeTextParagraphPoolSize(count);
    while (pool < high) {
        const middle = pool + Math.ceil((high - pool) / 2);
        if (fits(middle, capacity)) {
            pool = middle;
        } else {
            high = middle - 1;
        }
    }
    return { pool, capacity };
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
export function computeTextParagraphFlags(scalars: Uint32Array, glyphIds: Uint32Array, proof: ComputeTextSpaceProof | null): Uint8Array {
    const flags = new Uint8Array(scalars.length);
    const readProof = () => proof;
    for (let i = 0; i < scalars.length; i++) {
        flags[i] = paragraphFlag(scalars, glyphIds, i, readProof);
    }
    return flags;
}

const UNAVAILABLE = "updateComputeTextParagraphs: shaping-safe space analysis is unavailable for a paragraph with break candidates";

function paragraphFlag(scalars: Uint32Array, glyphIds: Uint32Array, i: number, readProof: () => ComputeTextSpaceProof | null): number {
    const cp = scalars[i]!;
    if (cp !== 0x20) {
        return computeTextNoBreakBefore(cp);
    }
    if (i + 1 < scalars.length && scalars[i + 1] !== 0x20) {
        const proof = readProof();
        if (!proof) {
            throw new Error(UNAVAILABLE);
        }
        if (proof.safeAfter(glyphIds, i, scalars)) {
            return SAFE_SPACE_FLAG;
        }
    }
    return 1;
}

/** @internal Reuse only proven unchanged scalar/glyph neighborhoods; callback injection is for exact untimed tests.
 *  `known` lower-bounds the unchanged prefix and suffix lengths, skipping their comparison.
 *  A `profile` seam replaces the per-scalar rule and extends the recomputed range to the end of its grapheme chain. */
export function computeTextParagraphFlagsIncremental<P extends ComputeTextSpaceProof>(
    previous: Pick<ComputeTextParagraphFlagEntry, "scalars" | "glyphIds" | "flags"> | undefined,
    scalars: Uint32Array,
    glyphIds: Uint32Array,
    readProof: () => P | null,
    evaluate: (scalars: Uint32Array, glyphIds: Uint32Array, i: number, readProof: () => P | null) => number = paragraphFlag,
    known?: readonly [prefix: number, suffix: number],
    profile?: ComputeTextParagraphProfile
): Uint8Array {
    const flags = new Uint8Array(scalars.length);
    let lo = 0;
    let hi = scalars.length;
    if (previous) {
        const oldN = previous.scalars.length;
        const limit = Math.min(oldN, scalars.length);
        let prefix = 0;
        let suffix = 0;
        if (known) {
            [prefix, suffix] = known;
        } else {
            while (prefix < limit && previous.scalars[prefix] === scalars[prefix] && previous.glyphIds[prefix] === glyphIds[prefix]) {
                prefix++;
            }
            while (
                suffix < limit - prefix &&
                previous.scalars[oldN - suffix - 1] === scalars[scalars.length - suffix - 1] &&
                previous.glyphIds[oldN - suffix - 1] === glyphIds[glyphIds.length - suffix - 1]
            ) {
                suffix++;
            }
        }
        lo = Math.max(0, prefix - 66);
        hi = Math.min(scalars.length, scalars.length - suffix + 66);
        if (lo < 66) {
            lo = 0;
        }
        if (scalars.length - hi < 66) {
            hi = scalars.length;
        }
        if (profile) {
            hi = profile.chainEnd(scalars, hi);
        }
        if (hi - lo >= scalars.length / 2) {
            lo = 0;
            hi = scalars.length;
        }
        flags.set(previous.flags.subarray(0, lo));
        flags.set(previous.flags.subarray(hi - scalars.length + oldN), hi);
    }
    if (profile) {
        profile.flags(scalars, glyphIds, readProof(), lo, hi, flags, lo, UNAVAILABLE);
    } else {
        for (let i = lo; i < hi; i++) {
            flags[i] = evaluate(scalars, glyphIds, i, readProof);
        }
    }
    return flags;
}

/** The base selection scan, non-throwing: paragraphs it rejects (e.g. `mixed scripts`) keep their synchronous errors. */
function selectsDevanagari(text: string): boolean {
    try {
        return /[^\0-\x7f]/u.test(text) && scanComputeTextRun(text).script === "deva";
    } catch {
        return false;
    }
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
