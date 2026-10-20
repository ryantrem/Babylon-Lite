import type { ComputeFont } from "./compute-font.js";
import { mapComputeFontScalars } from "./compute-font.js";
import { decodeComputeTextChunk, prepareComputeTextChunk, scanComputeTextRun } from "./compute-text-selection.js";
import { createComputeTextPlanForSelection } from "./compute-text-plan.js";
import type { ComputeTextPlan } from "./compute-text-plan.js";
import { compileComputeTextProgram } from "./compute-text-program.js";
import { analyzeComputeTextSegmentation } from "./compute-text-segment.js";
import type { SegmentAnalysis } from "./compute-text-segment.js";
import { checkComputeTextShapeBounds, prepareComputeTextShapeWords } from "./_gpu/compute-text-shape.js";
import { installComputeTextClosure, resetComputeTextDrawBuffers, submitComputeTextPrepared } from "./compute-text-data.js";
import type { ComputeTextData, ComputeTextUpdate } from "./compute-text-data.js";
import { closeComputeTextGlyphs } from "./compute-text-closure.js";
import { getComputeFontLineMetrics } from "./compute-font.js";
import type { ComputeTextParagraphLayout } from "./compute-text-paragraphs.js";
import { createComputeTextChunkState, executeChunkedResidentComputeText, installComputeTextChunks } from "./_gpu/compute-text-paragraph-chunks.js";

/** @internal Test-only budgets; no device limits or public option changes. */
export interface ComputeTextChunkBudgets {
    records?: number;
    arenaBytes?: number;
    bindingBytes?: number;
    force?: boolean;
    monolithic?: boolean;
}

/** @internal Persistent source identity; typed dispatch inputs are materialized one chunk at a time. */
export interface ComputeTextSourceChunk {
    text: string;
    start: number;
    scalarStart: number;
    scalars: number;
    capacity: number;
    segments: readonly number[];
    before: string;
    after: string;
    plan: ComputeTextPlan;
}

/** @internal */
export interface ComputeTextChunkPlan {
    paragraphs: readonly string[];
    chunks: ComputeTextSourceChunk[][];
    seeds: Set<number>;
    plans: ComputeTextPlan[];
    maxCapacity: number;
    bindingLimit: number;
    arenaBytes: number;
    workspaceBytes: number;
    workspace: ReadonlyMap<string, number>;
}

function move(text: string, at: number, distance: number): number {
    while (distance < 0 && at > 0) {
        at--;
        if (text.charCodeAt(at) >= 0xdc00 && text.charCodeAt(at) <= 0xdfff) {
            at--;
        }
        distance++;
    }
    while (distance > 0 && at < text.length) {
        at += text.codePointAt(at)! > 0xffff ? 2 : 1;
        distance--;
    }
    return at;
}

/** @internal The unchanged proof sees its full 65-neighbor exhaustion window and true ends. */
export function safeComputeTextSourceBoundary(font: ComputeFont, analysis: SegmentAnalysis | null, text: string, space: number): boolean {
    if (!analysis || text.charCodeAt(space) !== 32 || space + 1 >= text.length) {
        return false;
    }
    const first = move(text, space, -65);
    const end = move(text, space + 1, 65);
    const decoded = decodeComputeTextChunk(text.slice(first, end));
    const glyphs = mapComputeFontScalars(font, decoded.scalars, (_i, error) => {
        throw error;
    });
    return analysis.safeAfter(glyphs, decoded.offsets.indexOf(space - first));
}

/** @internal Source planning and every required resource bound precede closure/atlas installation. */
export function planComputeTextParagraphChunks(
    font: ComputeFont,
    paragraphs: readonly string[],
    limits: GPUSupportedLimits,
    budgets?: ComputeTextChunkBudgets,
    previous?: ComputeTextChunkPlan
): ComputeTextChunkPlan {
    const bindingLimit = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize, budgets?.bindingBytes ?? Infinity);
    for (const value of [budgets?.records, budgets?.arenaBytes, budgets?.bindingBytes]) {
        if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) {
            throw new Error("updateComputeTextParagraphs: internal chunk budgets must be positive safe integers");
        }
    }
    if (bindingLimit < 1024) {
        throw new Error(`updateComputeTextParagraphs: chunk metadata buffer of 1024 bytes exceeds binding limit ${bindingLimit}`);
    }
    const target = Math.max(1, Math.min(16384, budgets?.records ?? 16384, Math.floor((bindingLimit - 144) / 48)));
    const maxBatchProduct = Math.floor((3 * 1024 * 1024 - 80 * target - 1024) / 56);
    const seeds = new Set<number>();
    const plans: ComputeTextPlan[] = [];
    const chunks: ComputeTextSourceChunk[][] = [];
    let maxCapacity = 1;
    let totalScalars = 0;
    let totalRecords = 0;
    const workspace = new Map<string, number>([
        ["carry", 512],
        ["replay", 192],
    ]);
    let dedicated = false;
    let paragraphPrefix = 0;
    while (paragraphPrefix < Math.min(previous?.paragraphs.length ?? 0, paragraphs.length) && previous!.paragraphs[paragraphPrefix] === paragraphs[paragraphPrefix]) {
        paragraphPrefix++;
    }
    let paragraphSuffix = 0;
    while (
        paragraphSuffix < Math.min(previous?.paragraphs.length ?? 0, paragraphs.length) - paragraphPrefix &&
        previous!.paragraphs[previous!.paragraphs.length - paragraphSuffix - 1] === paragraphs[paragraphs.length - paragraphSuffix - 1]
    ) {
        paragraphSuffix++;
    }
    for (let p = 0; p < paragraphs.length; p++) {
        const text = paragraphs[p]!;
        const selection = scanComputeTextRun(text);
        if (selection.direction !== "ltr") {
            throw new Error(`updateComputeTextParagraphs: paragraph ${p} must select an LTR script (RTL layout is unsupported)`);
        }

        totalScalars += selection.scalarCount;
        if (selection.scalarCount > 0x7ffffffe || !Number.isSafeInteger(totalScalars) || totalScalars > 0xfffffffe) {
            throw new Error(`updateComputeTextParagraphs: paragraph ${p} scalar/record counters exceed the chunk addressing ABI`);
        }
        const plan = createComputeTextPlanForSelection(font, selection);
        plans.push(plan);
        const program = compileComputeTextProgram(plan);
        const expansion = Math.max(1, program.maxExpansion);
        totalRecords += Math.max(1, selection.scalarCount * expansion);
        if (!Number.isSafeInteger(totalRecords) || totalRecords + paragraphs.length > 0xfffffffe) {
            throw new Error(`updateComputeTextParagraphs: paragraph ${p} conservative record/line counters exceed the chunk addressing ABI`);
        }
        const analysis = analyzeComputeTextSegmentation(program);
        if (!analysis && / [^ ]/u.test(text)) {
            throw new Error(`updateComputeTextParagraphs: paragraph ${p} independent space analysis is unavailable`);
        }
        const oldIndex = p < paragraphPrefix ? p : p >= paragraphs.length - paragraphSuffix ? (previous?.paragraphs.length ?? 0) - (paragraphs.length - p) : p;
        const oldText = previous?.paragraphs[oldIndex] ?? "";
        const oldChunks = previous?.chunks[oldIndex] ?? [];
        let commonPrefix = 0;
        while (commonPrefix < Math.min(oldText.length, text.length) && oldText.charCodeAt(commonPrefix) === text.charCodeAt(commonPrefix)) {
            commonPrefix++;
        }
        let commonSuffix = 0;
        while (
            commonSuffix < Math.min(oldText.length, text.length) - commonPrefix &&
            oldText.charCodeAt(oldText.length - commonSuffix - 1) === text.charCodeAt(text.length - commonSuffix - 1)
        ) {
            commonSuffix++;
        }
        const stable = new Map<number, ComputeTextSourceChunk>();
        for (const old of oldChunks) {
            const nextStart = old.start + (old.start >= oldText.length - commonSuffix ? text.length - oldText.length : 0);
            if (
                old.plan === plan &&
                (old.start + old.text.length <= commonPrefix || old.start >= oldText.length - commonSuffix) &&
                text.slice(nextStart, nextStart + old.text.length) === old.text &&
                text.slice(move(text, nextStart, -66), nextStart) === old.before &&
                text.slice(nextStart + old.text.length, move(text, nextStart + old.text.length, 66)) === old.after
            ) {
                stable.set(nextStart, old);
            }
        }
        const stableStarts = [...stable.keys()];
        let stableIndex = 0;
        const paragraphChunks: ComputeTextSourceChunk[] = [];
        chunks.push(paragraphChunks);
        let start = 0;
        let scalarStart = 0;
        let cursor = 0;
        let count = 0;
        let maxSegmentCapacity = 0;
        let segments: number[] = [];
        do {
            const retained = stable.get(cursor);
            if (retained) {
                if (count) {
                    paragraphChunks.push(makeSource(cursor));
                    scalarStart += count;
                }
                start = cursor;
                count = retained.scalars;
                segments = [...retained.segments];
                cursor += retained.text.length;
                paragraphChunks.push(makeSource(cursor));
                scalarStart += count;
                start = cursor;
                count = 0;
                maxSegmentCapacity = 0;
                segments = [];
                if (cursor === text.length) {
                    break;
                }
                continue;
            }
            let end = move(text, cursor, Math.min(64, Math.max(1, Math.floor(target / expansion))));
            while (stableIndex < stableStarts.length && stableStarts[stableIndex]! <= cursor) {
                stableIndex++;
            }
            if (stableIndex < stableStarts.length && stableStarts[stableIndex]! < end) {
                end = stableStarts[stableIndex]!;
            }
            if (end < text.length) {
                let seam = end;
                while (seam > cursor && !safeComputeTextSourceBoundary(font, analysis, text, seam - 1)) {
                    seam = move(text, seam, -1);
                }
                if (seam === cursor) {
                    seam = end;
                    let spanScalars = decodeComputeTextChunk(text.slice(cursor, end)).scalars.length;
                    while (seam < text.length && !safeComputeTextSourceBoundary(font, analysis, text, seam - 1)) {
                        seam = move(text, seam, 1);
                        spanScalars++;
                        const bytes = spanScalars * expansion * 48 + 144;
                        if (bytes > bindingLimit) {
                            throw new Error(
                                `updateComputeTextParagraphs: paragraph ${p} indivisible scalar span ${scalarStart + count}..${scalarStart + count + spanScalars} replay buffer of ${bytes} bytes exceeds binding limit ${bindingLimit} (maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize})`
                            );
                        }
                    }
                }
                end = seam;
            }
            const length = decodeComputeTextChunk(text.slice(cursor, end)).scalars.length;
            if (
                count &&
                ((count + length) * expansion > target ||
                    segments.length >= limits.maxComputeWorkgroupsPerDimension ||
                    (segments.length + 1) * Math.max(maxSegmentCapacity, length * expansion) > maxBatchProduct)
            ) {
                paragraphChunks.push(makeSource(cursor));
                start = cursor;
                scalarStart += count;
                count = 0;
                maxSegmentCapacity = 0;
                segments = [];
            }
            count += length;
            maxSegmentCapacity = Math.max(maxSegmentCapacity, length * expansion);
            segments.push(count);
            cursor = end;
        } while (cursor < text.length);
        if (count || !paragraphChunks.length) {
            paragraphChunks.push(makeSource(cursor));
        }

        function makeSource(end: number): ComputeTextSourceChunk {
            const capacity = Math.max(1, count * expansion);
            const replay = 16 + capacity * 16 + (capacity + 4) * 32;
            if (!Number.isSafeInteger(replay) || replay > bindingLimit) {
                throw new Error(
                    `updateComputeTextParagraphs: paragraph ${p} scalar span ${scalarStart}..${scalarStart + count} replay buffer of ${replay} bytes exceeds binding limit ${bindingLimit} (maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize})`
                );
            }
            const source: ComputeTextSourceChunk = {
                text: text.slice(start, end),
                start,
                scalarStart,
                scalars: count,
                capacity,
                segments: segments.slice(),
                before: text.slice(move(text, start, -66), start),
                after: text.slice(end, move(text, end, 66)),
                plan,
            };
            const input = prepareComputeTextChunk(source);
            for (const glyph of input.glyphIds) {
                seeds.add(glyph);
            }
            const shapeCapacity = Math.max(1, ...input.runs.map((run) => run.scalars.length * expansion));
            const prepared = prepareComputeTextShapeWords(plan, input.runs, 0, shapeCapacity, limits);
            checkComputeTextShapeBounds(limits, prepared);
            const shapeBytes = input.runs.length * (4 + shapeCapacity * 6) * 4;
            for (const [name, bytes] of [
                ["shape-scratch", prepared.scratchBytes],
                ["shape-output", shapeBytes],
                ["runs", prepared.runs.byteLength],
                ["stitch-table", (input.runs.length + 2) * 4],
                ["replay", replay],
                ["carry", 512],
                ["shape-params", 28],
                ["stitch-params", 16],
            ] as const) {
                workspace.set(name, Math.max(workspace.get(name) ?? 0, bytes));
            }
            dedicated ||= capacity > target;
            const resources: readonly (readonly [string, number])[] = [
                ["shaped", 16 + capacity * 24],
                ["instances", capacity * 12],
                ["flags", Math.ceil((count + 1) / 4) * 4],
                ["replay", 16 + capacity * 16 + (capacity + 4) * 32],
            ];
            for (const [name, bytes] of resources) {
                if (!Number.isSafeInteger(bytes) || bytes > bindingLimit) {
                    throw new Error(
                        `updateComputeTextParagraphs: paragraph ${p} scalar span ${scalarStart}..${scalarStart + count} ${name} buffer of ${bytes} bytes exceeds binding limit ${bindingLimit} (maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize})`
                    );
                }
            }
            maxCapacity = Math.max(maxCapacity, capacity);
            return source;
        }
    }
    if (!paragraphs.length) {
        workspace.set("empty-metadata", 1024);
    }
    const workspaceBytes = [...workspace.values()].reduce((sum, bytes) => sum + bytes, 0);
    if (!dedicated && workspaceBytes > 3 * 1024 * 1024) {
        throw new Error(`updateComputeTextParagraphs: normal GPU chunk workspace of ${workspaceBytes} bytes exceeds 3145728-byte reservation`);
    }
    return {
        paragraphs: paragraphs.slice(),
        chunks,
        seeds,
        plans,
        maxCapacity,
        bindingLimit,
        workspaceBytes,
        workspace,
        arenaBytes: Math.min(Math.floor(limits.maxBufferSize / 4) * 4, Math.ceil((budgets?.arenaBytes ?? 1 << 20) / 4) * 4),
    };
}

/** @internal Cheap source-size gate; warm fitting updates retain their existing preparation path. */
export function needsComputeTextParagraphChunks(data: ComputeTextData, paragraphs: readonly string[], warm: boolean, first = 0, end = paragraphs.length): boolean {
    if (data._chunkBudgets?.monolithic) {
        return false;
    }
    if (data._chunkState || data._chunkBudgets?.force) {
        return true;
    }
    if (!paragraphs.length) {
        return false;
    }
    if (warm) {
        let dirty = 0;
        for (let i = first; i < end; i++) {
            dirty += paragraphs[i]!.length;
        }
        if (dirty * 8 <= 4 * 1024 * 1024) {
            return false;
        }
    }
    let max = 0;
    let sum = 0;
    for (const text of paragraphs) {
        max = Math.max(max, text.length);
        sum += text.length;
    }
    if (paragraphs.length * max * 48 <= 4 * 1024 * 1024) {
        return false;
    }
    max = 1;
    sum = 0;
    for (const text of paragraphs) {
        const selected = scanComputeTextRun(text);
        const expansion = Math.max(1, compileComputeTextProgram(createComputeTextPlanForSelection(data._font, selected)).maxExpansion);
        const capacity = Math.max(1, selected.scalarCount * expansion);
        max = Math.max(max, capacity);
        sum += capacity;
    }
    const uniform = paragraphs.length * max * 48;
    return uniform > Math.min(data._device.limits.maxBufferSize, data._device.limits.maxStorageBufferBindingSize) || (uniform > 4 * 1024 * 1024 && uniform > sum * 96);
}

/** @internal Shared lifecycle with all bounded source/resource admission before closure mutation. */
export function submitChunkedComputeTextParagraphs(data: ComputeTextData, paragraphs: readonly string[], layout: ComputeTextParagraphLayout): Promise<ComputeTextUpdate> {
    let candidateRevision = 0;
    let baseRevision = 0;
    const submitted = submitComputeTextPrepared(data, (assertLive) => {
        const plan = planComputeTextParagraphChunks(data._font, paragraphs, data._device.limits, data._chunkBudgets, data._chunkState?.plan);
        const height = layout.lineHeightPx ?? getComputeFontLineMetrics(data._font, data.fontSizePx).lineHeightPx;
        if (
            !(Math.fround(height) > 0) ||
            !Number.isFinite(Math.fround(height)) ||
            !(Math.fround(data.fontSizePx / data._font.unitsPerEm) > 0) ||
            !Number.isFinite(Math.fround(data.fontSizePx / data._font.unitsPerEm))
        ) {
            throw new Error("updateComputeTextParagraphs: font line height and scale must be positive, finite and representable as f32");
        }
        const previous = data._chunkState;
        const atlas = data._storage._curveSets.get(data._curveSet)?._atlas;
        const reuse = !!previous && previous.revision === data._revision - 1 && previous.atlas === atlas && previous.atlasVersion === atlas?._version;
        const state = previous ?? createComputeTextChunkState(data);
        if (!previous) {
            resetComputeTextDrawBuffers(data);
        }
        baseRevision = reuse ? state.baseRevision : data._revision;
        candidateRevision = data._revision;
        return async () => {
            const closure = closeComputeTextGlyphs(data._font, plan.plans, [...plan.seeds]);
            const rebuild = new Error("paragraph chunk atlas requires full installation");
            let shapeReuse = reuse;
            try {
                installComputeTextClosure(data, closure.glyphIds, reuse ? rebuild : undefined);
            } catch (error) {
                if (error !== rebuild) {
                    throw error;
                }
                shapeReuse = false;
                baseRevision = candidateRevision;
                installComputeTextClosure(data, closure.glyphIds);
            }
            installComputeTextChunks(data, state, plan, shapeReuse);
            const result = await executeChunkedResidentComputeText(data, state, { ...layout, lineHeightPx: height }, assertLive);
            assertLive();
            state.revision = candidateRevision;
            state.baseRevision = baseRevision;
            state.atlas = data._storage._curveSets.get(data._curveSet)?._atlas;
            state.atlasVersion = data._storage._curveSets.get(data._curveSet)?._atlas._version;
            return result;
        };
    });
    const invalidate = () => {
        const state = data._chunkState;
        if (state && state.baseRevision <= candidateRevision) {
            state.revision = -1;
        }
    };
    return submitted.then(
        (update) => {
            const completed = update.completed.catch((error: unknown) => {
                invalidate();
                const message = error instanceof Error ? error.message : String(error);
                throw new Error(`updateComputeTextParagraphs: ${message.replace(/^updateComputeTextData:\s*/, "")}`, { cause: error });
            });
            void completed.catch(() => undefined);
            return { completed };
        },
        (error: unknown) => {
            invalidate();
            if (candidateRevision === data._revision) {
                data._paragraphResources?.release();
            }
            throw error;
        }
    );
}
