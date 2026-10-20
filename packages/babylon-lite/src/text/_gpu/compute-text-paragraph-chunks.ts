import type { ComputeTextData, ComputeTextDrawRange } from "../compute-text-data.js";
import type { ComputeTextChunkPlan, ComputeTextSourceChunk } from "../compute-text-paragraph-chunks.js";
import { computeTextNoBreakBefore, decodeComputeTextChunk } from "../compute-text-selection.js";
import { mapComputeFontScalars } from "../compute-font.js";
import { analyzeComputeTextSegmentation } from "../compute-text-segment.js";
import { compileComputeTextProgram } from "../compute-text-program.js";
import type { ComputeTextParagraphLayout } from "../compute-text-paragraphs.js";
import type { ComputeTextPlan } from "../compute-text-plan.js";
import type { StorageBuffer } from "../../resource/storage-buffer.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../../resource/storage-buffer.js";
import type { ComputeStorageBufferRange } from "../../compute/compute-storage-buffer-binding.js";
import { computeStorageBufferBinding } from "../../compute/compute-storage-buffer-binding.js";
import { createComputeBindingSet, disposeComputeBindingSet } from "../../compute/compute-bindings.js";
import type { ComputeBindingResources } from "../../compute/compute-bindings.js";
import { createComputeDispatch } from "../../compute/compute-dispatch.js";
import { addComputeDispatch, createComputeTask, prepareComputeTask, submitComputeTasks } from "../../compute/compute-task.js";
import type { ComputeTask } from "../../compute/compute-task.js";
import { createComputeShader, disposeComputeShader } from "../../compute/compute-shader.js";
import type { ComputeShader } from "../../compute/compute-shader.js";
import { submitDeferredComputeTaskResult } from "../../compute/compute-task-readback.js";
import { computeTextScratchCapacity, createComputeTextShapeShader, createShapeSchedule, prepareComputeTextShapeWords } from "./compute-text-shape.js";
import { COMPUTE_TEXT_STITCH_KERNEL } from "./compute-text-resident.js";
import { COMPUTE_TEXT_CHUNK_LAYOUT, COMPUTE_TEXT_CHUNK_PUBLISH } from "./compute-text-paragraph-chunk-layout.js";

/** @internal */
export interface ComputeTextChunkWindow extends ComputeStorageBufferRange {
    offset: number;
    size: number;
}

/** @internal Persistent post-shape chunk and exact source dependency. */
export interface ComputeTextResidentChunk {
    id: number;
    source: ComputeTextSourceChunk;
    shaped: ComputeTextChunkWindow;
    flags: ComputeTextChunkWindow;
    instances: ComputeTextChunkWindow;
    metadata: ComputeTextChunkWindow;
    params: ComputeTextChunkWindow;
    indirect: ComputeTextChunkWindow;
    dirty: boolean;
    shapes: number;
}

interface Arena {
    buffer: StorageBuffer;
    cursor: number;
    free: ComputeTextChunkWindow[];
    live: number;
}

interface RetiredChunkResources {
    buffers: Set<StorageBuffer>;
    windows: { kind: string; range: ComputeTextChunkWindow }[];
}

/** @internal Paragraph resource ownership, including failures before resident executor setup. */
export interface ComputeTextChunkState {
    chunks: ComputeTextResidentChunk[][];
    plan?: ComputeTextChunkPlan;
    revision: number;
    baseRevision: number;
    atlasVersion?: number;
    atlas?: object;
    nextId: number;
    released: boolean;
    arenas: Map<string, Arena[]>;
    shaders: Map<string, ComputeShader>;
    buffers: Map<string, StorageBuffer>;
    tasks: ComputeTask[];
    staging: GPUBuffer[];
    freeStaging: GPUBuffer[];
    uploadedPlan?: ComputeTextPlan;
    drawRanges?: readonly ComputeTextDrawRange[];
    retired: Map<number, RetiredChunkResources>;
}

/** @internal */
export function getComputeTextChunkState(data: ComputeTextData): ComputeTextChunkState | undefined {
    return data._chunkState;
}

function shader(data: ComputeTextData, state: ComputeTextChunkState, key: string, code: string, names: readonly string[], read: number, entryPoint = "main") {
    let result = state.shaders.get(key);
    if (!result) {
        result = createComputeShader(data._engine, {
            name: `compute-text-chunk-${key}`,
            computeSource: code,
            entryPoint,
            bindings: names.map((name, binding) =>
                computeStorageBufferBinding(name, {
                    group: 0,
                    binding,
                    access: key === "stitch" ? (binding === 1 ? "read-write" : "read") : binding < read ? "read" : "read-write",
                })
            ),
        });
        state.shaders.set(key, result);
    }
    return result;
}

function buffer(data: ComputeTextData, state: ComputeTextChunkState, key: string, bytes: number): StorageBuffer {
    let result = state.buffers.get(key);
    if (!result || result.byteLength < bytes) {
        if (result) {
            retireBuffer(data, state, result);
        }
        result = createStorageBuffer(data._engine, Math.max(4, bytes), { label: `compute-text-chunk-${key}`, writable: true });
        state.buffers.set(key, result);
    }
    return result;
}

/** @internal Validate the actual aligned allocation/binding, independently of planner admission. */
export function checkComputeTextChunkWindow(
    limits: Pick<GPUSupportedLimits, "maxBufferSize" | "maxStorageBufferBindingSize" | "minStorageBufferOffsetAlignment">,
    bindingLimit: number,
    kind: string,
    bytes: number,
    offset: number,
    allocation: number
): void {
    const alignment = limits.minStorageBufferOffsetAlignment;
    if (
        !Number.isSafeInteger(bindingLimit) ||
        bindingLimit < 4 ||
        !Number.isSafeInteger(bytes) ||
        bytes < 4 ||
        bytes % 4 !== 0 ||
        !Number.isSafeInteger(offset) ||
        offset < 0 ||
        !Number.isSafeInteger(alignment) ||
        alignment < 1 ||
        offset % alignment !== 0 ||
        !Number.isSafeInteger(allocation) ||
        allocation % 4 !== 0 ||
        allocation > limits.maxBufferSize ||
        !Number.isSafeInteger(offset + bytes) ||
        offset + bytes > allocation ||
        bytes > Math.min(bindingLimit, limits.maxStorageBufferBindingSize, limits.maxBufferSize)
    ) {
        throw new Error(
            `updateComputeTextParagraphs: ${kind} window of ${bytes} bytes at offset ${offset} exceeds aligned allocation/binding bounds ` +
                `(allocation=${allocation}, alignment=${alignment}, bindingLimit=${bindingLimit}, maxBufferSize=${limits.maxBufferSize}, maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize})`
        );
    }
}

function window(data: ComputeTextData, state: ComputeTextChunkState, kind: string, bytes: number, plan: ComputeTextChunkPlan): ComputeTextChunkWindow {
    checkComputeTextChunkWindow(data._device.limits, plan.bindingLimit, kind, bytes, 0, Math.max(bytes, plan.arenaBytes));
    bytes = Math.max(4, Math.ceil(bytes / 4) * 4);
    const preferred = ["shaped", "flags", "instances"].includes(kind) ? Math.min(Math.floor((bytes * 1.25) / 4) * 4, Math.floor(plan.bindingLimit / 4) * 4) : bytes;
    const alignment = data._device.limits.minStorageBufferOffsetAlignment;
    let arenas = state.arenas.get(kind);
    if (!arenas) {
        arenas = [];
        state.arenas.set(kind, arenas);
    }
    for (const arena of arenas) {
        const free = arena.free.findIndex((entry) => entry.size >= bytes && entry.size <= preferred);
        if (free >= 0) {
            const entry = arena.free[free]!;
            checkComputeTextChunkWindow(data._device.limits, plan.bindingLimit, kind, entry.size, entry.offset, arena.buffer.byteLength);
            arena.live++;
            return arena.free.splice(free, 1)[0]!;
        }
        const offset = Math.ceil(arena.cursor / alignment) * alignment;
        const size = offset + preferred <= arena.buffer.byteLength ? preferred : bytes;
        if (offset + size <= arena.buffer.byteLength) {
            checkComputeTextChunkWindow(data._device.limits, plan.bindingLimit, kind, size, offset, arena.buffer.byteLength);
            arena.cursor = offset + size;
            arena.live++;
            return { buffer: arena.buffer, offset, size };
        }
    }
    const allocation = Math.max(preferred, plan.arenaBytes);
    checkComputeTextChunkWindow(data._device.limits, plan.bindingLimit, kind, preferred, 0, allocation);
    const created = createStorageBuffer(data._engine, allocation, {
        label: `compute-text-chunk-${kind}-arena`,
        writable: true,
        vertex: kind === "instances",
        indirect: kind === "indirect",
    });
    arenas.push({ buffer: created, cursor: preferred, live: 1, free: [] });
    return { buffer: created, offset: 0, size: preferred };
}

function retirement(data: ComputeTextData, state: ComputeTextChunkState): RetiredChunkResources {
    if (data._chunkState === state && data._drawRanges?.length) {
        data._drawRanges = [];
    }
    let retired = state.retired.get(data._revision);
    if (!retired) {
        retired = { buffers: new Set(), windows: [] };
        state.retired.set(data._revision, retired);
    }
    return retired;
}

function retireBuffer(data: ComputeTextData, state: ComputeTextChunkState, buffer: StorageBuffer): void {
    retirement(data, state).buffers.add(buffer);
}

function settleRetirement(state: ComputeTextChunkState, revision: number): void {
    const retired = state.retired.get(revision);
    if (!retired) {
        return;
    }
    state.retired.delete(revision);
    // Retired allocations are absent from the live maps; no later revision can borrow them.
    retired.buffers.forEach(disposeStorageBuffer);
    for (const { kind, range } of retired.windows) {
        const arena = state.arenas.get(kind)?.find((candidate) => candidate.buffer === range.buffer);
        if (arena && !range.buffer._destroyed) {
            arena.free.push(range);
        }
    }
}

function retire(data: ComputeTextData, state: ComputeTextChunkState, chunk: ComputeTextResidentChunk): void {
    for (const [kind, entry] of Object.entries({
        shaped: chunk.shaped,
        flags: chunk.flags,
        instances: chunk.instances,
        metadata: chunk.metadata,
        params: chunk.params,
        indirect: chunk.indirect,
    })) {
        const arena = state.arenas.get(kind)!.find((candidate) => candidate.buffer === entry.buffer)!;
        arena.live--;
        retirement(data, state).windows.push({ kind, range: entry });
    }
}

/** @internal New allocations are zeroed by WebGPU; all previously published counts clear before reuse. */
export function createComputeTextChunkState(data: ComputeTextData): ComputeTextChunkState {
    const state: ComputeTextChunkState = {
        chunks: [],
        revision: 0,
        baseRevision: data._revision,
        nextId: 1,
        released: false,
        arenas: new Map(),
        shaders: new Map(),
        buffers: new Map(),
        tasks: [],
        staging: [],
        freeStaging: [],
        retired: new Map(),
    };
    data._chunkState = state;
    data._paragraphResources = { clear: () => clearChunks(data, state), release: () => releaseChunks(data, state) };
    return state;
}

function clearChunks(data: ComputeTextData, state: ComputeTextChunkState): void {
    for (const arena of state.arenas.get("indirect") ?? []) {
        const words = new Uint32Array(Math.ceil(arena.cursor / 4));
        const stride = (Math.ceil(16 / data._device.limits.minStorageBufferOffsetAlignment) * data._device.limits.minStorageBufferOffsetAlignment) / 4;
        for (let at = 0; at < words.length; at += stride) {
            words[at] = 6;
        }
        updateStorageBuffer(data._engine, arena.buffer, words);
    }
}

function releaseChunks(data: ComputeTextData, state: ComputeTextChunkState): void {
    if (state.released) {
        return;
    }
    data._engine._flushPendingWork?.(data._engine);
    if (data._chunkState === state) {
        data._drawRanges = [];
    }
    state.released = true;
    releaseTasks(state);
    state.shaders.forEach(disposeComputeShader);
    state.buffers.forEach(disposeStorageBuffer);
    state.arenas.forEach((arenas) => arenas.forEach((arena) => disposeStorageBuffer(arena.buffer)));
    for (const revision of state.retired.keys()) {
        settleRetirement(state, revision);
    }
    state.staging = state.staging.filter((entry) => {
        if (data._dead || entry.mapState === "unmapped") {
            entry.destroy();
            return false;
        }
        return true;
    });
    state.arenas.clear();
    state.buffers.clear();
    state.shaders.clear();
    state.chunks = [];
    state.drawRanges = undefined;
    state.plan = undefined;
    state.uploadedPlan = undefined;
    state.atlas = undefined;
    state.atlasVersion = undefined;
    state.freeStaging.length = 0;
    if (data._chunkState === state) {
        data._chunkState = undefined;
        data._drawRanges = undefined;
    }
}

function releaseTasks(state: ComputeTextChunkState): void {
    for (const task of state.tasks) {
        for (const dispatch of task.dispatches) {
            disposeComputeBindingSet(dispatch.bindings);
        }
        task.dispose();
    }
    state.tasks = [];
}

function task(data: ComputeTextData, state: ComputeTextChunkState, shader: ComputeShader, resources: ComputeBindingResources, x = 1): ComputeTask {
    const result = createComputeTask(data._engine, shader.name);
    const limit = data._device.limits.maxComputeWorkgroupsPerDimension;
    x = Math.min(x, limit * limit);
    addComputeDispatch(
        result,
        createComputeDispatch(shader, createComputeBindingSet(shader, resources), { size: { x: Math.min(x, limit), y: Math.max(1, Math.ceil(x / limit)) } })
    );
    result.record();
    state.tasks.push(result);
    return result;
}

/** @internal Exact identity matching is positional; never search unrelated paragraphs by text/hash. */
export function installComputeTextChunks(data: ComputeTextData, state: ComputeTextChunkState, plan: ComputeTextChunkPlan, reuse: boolean): void {
    const old = state.chunks.flat();
    const retained = new Set<ComputeTextResidentChunk>();
    const previous = state.chunks;
    let prefix = 0;
    while (prefix < Math.min(previous.length, plan.chunks.length) && state.plan?.paragraphs[prefix] === plan.paragraphs[prefix]) {
        prefix++;
    }
    let suffix = 0;
    while (
        suffix < Math.min(previous.length, plan.chunks.length) - prefix &&
        state.plan?.paragraphs[previous.length - suffix - 1] === plan.paragraphs[plan.chunks.length - suffix - 1]
    ) {
        suffix++;
    }
    state.chunks = plan.chunks.map((sources, p) => {
        const oldIndex = p < prefix ? p : p >= plan.chunks.length - suffix ? previous.length - (plan.chunks.length - p) : p;
        const candidates = previous[oldIndex] ?? [];
        let left = 0;
        const right = candidates.length - 1;
        return sources.map((source, i) => {
            const matches = (chunk: ComputeTextResidentChunk | undefined) =>
                chunk &&
                !retained.has(chunk) &&
                chunk.source.text === source.text &&
                chunk.source.plan === source.plan &&
                chunk.source.before === source.before &&
                chunk.source.after === source.after;
            let entry = candidates[left];
            if (matches(entry)) {
                left++;
            } else {
                const candidate = candidates[right - (sources.length - 1 - i)];
                entry = matches(candidate) ? candidate : undefined;
            }
            if (entry && matches(entry)) {
                retained.add(entry);
                entry.source = source;
                entry.dirty = !reuse;
                return entry;
            }
            const capacity = source.capacity;
            return {
                id: state.nextId++,
                source,
                dirty: true,
                shapes: 0,
                shaped: window(data, state, "shaped", 16 + capacity * 24, plan),
                flags: window(data, state, "flags", Math.ceil((source.scalars + 1) / 4) * 4, plan),
                instances: window(data, state, "instances", capacity * 12, plan),
                metadata: window(data, state, "metadata", 1024, plan),
                params: window(data, state, "params", 64, plan),
                indirect: window(data, state, "indirect", 16, plan),
            };
        });
    });
    for (const entry of old) {
        if (!retained.has(entry)) {
            retire(data, state, entry);
        }
    }
    for (const arenas of state.arenas.values()) {
        let spare = false;
        for (let i = arenas.length - 1; i >= 0; i--) {
            const arena = arenas[i]!;
            if (!arena.live) {
                if (spare || arena.buffer.byteLength > plan.arenaBytes) {
                    retireBuffer(data, state, arena.buffer);
                    arenas.splice(i, 1);
                } else {
                    spare = true;
                }
            }
        }
    }
    const ranges: ComputeTextDrawRange[] = state.chunks.flat().map((chunk) => ({
        instances: chunk.instances.buffer,
        offset: chunk.instances.offset,
        size: chunk.instances.size,
        indirect: chunk.indirect.buffer,
        indirectOffset: chunk.indirect.offset,
    }));
    const previousRanges = state.drawRanges;
    if (
        !previousRanges ||
        previousRanges.length !== ranges.length ||
        ranges.some((range, i) => {
            const old = previousRanges[i]!;
            return (
                range.instances !== old.instances ||
                range.offset !== old.offset ||
                range.size !== old.size ||
                range.indirect !== old.indirect ||
                range.indirectOffset !== old.indirectOffset
            );
        })
    ) {
        if (data._drawRanges?.length) {
            data._drawRanges = [];
        }
        state.drawRanges = ranges;
    }
    state.plan = plan;
}

/** @internal Shape only dirty chunks using the existing interpreter/schedule and local stitch ABI. */
async function shapeChunks(data: ComputeTextData, state: ComputeTextChunkState, assertLive: () => void): Promise<void> {
    const engine = data._engine;
    let selectedPlan: ComputeTextPlan | undefined;
    let expansion = 1;
    let analysis: ReturnType<typeof analyzeComputeTextSegmentation> = null;
    for (const paragraph of state.chunks) {
        for (const chunk of paragraph) {
            if (!chunk.dirty) {
                continue;
            }
            const source = chunk.source;
            if (selectedPlan !== source.plan) {
                selectedPlan = source.plan;
                const program = compileComputeTextProgram(selectedPlan);
                expansion = Math.max(1, program.maxExpansion);
                analysis = analyzeComputeTextSegmentation(program);
            }
            const full = source.before + source.text + source.after;
            const context = decodeComputeTextChunk(full);
            const contextGlyphs = mapComputeFontScalars(data._font, context.scalars, (_i, error) => {
                throw error;
            });
            const prefixScalars = context.offsets.indexOf(source.before.length);
            const scalars = context.scalars.subarray(prefixScalars, prefixScalars + source.scalars);
            const glyphIds = contextGlyphs.subarray(prefixScalars, prefixScalars + source.scalars);
            let first = 0;
            const runs = source.segments.map((end) => {
                const run = { scalars: scalars.subarray(first, end), glyphIds: glyphIds.subarray(first, end) };
                first = end;
                return run;
            });
            const capacity = Math.max(1, ...runs.map((run) => run.scalars.length * expansion));
            const prepared = prepareComputeTextShapeWords(source.plan, runs, 0, capacity, data._device.limits);
            const outputBytes = runs.length * (4 + capacity * 6) * 4;
            const shapeBuffers = {
                params: buffer(data, state, "shape-params", 28),
                program: buffer(data, state, "program", prepared.programWords.byteLength),
                tables: buffer(data, state, "tables", prepared.tables.byteLength),
                runs: buffer(data, state, "runs", prepared.runs.byteLength),
                scratch: buffer(data, state, "shape-scratch", prepared.scratchBytes),
                output: buffer(data, state, "shape-output", outputBytes),
            };
            updateStorageBuffer(engine, shapeBuffers.params, prepared.params);
            updateStorageBuffer(engine, shapeBuffers.runs, prepared.runs);
            if (state.uploadedPlan !== source.plan) {
                updateStorageBuffer(engine, shapeBuffers.program, prepared.programWords);
                updateStorageBuffer(engine, shapeBuffers.tables, prepared.tables);
                state.uploadedPlan = source.plan;
            }
            const shapeKey = `shape-${computeTextScratchCapacity(capacity)}`;
            let shape = state.shaders.get(shapeKey);
            if (!shape) {
                shape = createComputeTextShapeShader(engine, capacity);
                state.shaders.set(shapeKey, shape);
            }
            const schedule = createShapeSchedule(engine, shape, shapeBuffers, runs.length);
            state.tasks.push(schedule.task);
            const table = new Uint32Array(2 + runs.length);
            table[1] = runs.length;
            let start = 0;
            runs.forEach((run, i) => {
                table[i + 2] = start;
                start += run.scalars.length;
            });
            const stitchTable = buffer(data, state, "stitch-table", table.byteLength);
            const stitchParams = buffer(data, state, "stitch-params", 16);
            updateStorageBuffer(engine, stitchTable, table);
            updateStorageBuffer(engine, stitchParams, Uint32Array.of(1, source.capacity, capacity, 0));
            const stitch = shader(data, state, "stitch", COMPUTE_TEXT_STITCH_KERNEL, ["seg", "shaped", "table", "params"], 0);
            const stitchTask = task(data, state, stitch, { seg: shapeBuffers.output, shaped: chunk.shaped, table: stitchTable, params: stitchParams });
            await prepareComputeTask(schedule.task);
            await prepareComputeTask(stitchTask);
            assertLive();
            submitComputeTasks([schedule.task, stitchTask]);
            chunk.shapes++;
            const flagBytes = new Uint8Array(chunk.flags.size);
            const classify = (cp: number) => (cp === 32 ? 1 : computeTextNoBreakBefore(cp));
            const before = source.before
                ? source.before.codePointAt(
                      source.before.length - (source.before.charCodeAt(source.before.length - 1) >= 0xdc00 && source.before.charCodeAt(source.before.length - 1) <= 0xdfff ? 2 : 1)
                  )!
                : 0;
            flagBytes[0] = classify(before);
            for (let i = 0; i < scalars.length; i++) {
                const cp = scalars[i]!;
                let f = classify(cp);
                const at = context.offsets[prefixScalars + i]!;
                if (cp === 32 && at + 1 < full.length && full.charCodeAt(at + 1) !== 32) {
                    if (!analysis) {
                        throw new Error("updateComputeTextParagraphs: independent space analysis is unavailable");
                    }
                    if (analysis.safeAfter(contextGlyphs, prefixScalars + i)) {
                        f |= 2;
                    }
                }
                flagBytes[i + 1] = f;
            }
            // The successor metadata may request the scalar preceding its first output cluster.
            if (before === 32 && source.text.length && source.text.charCodeAt(0) !== 32 && analysis?.safeAfter(contextGlyphs, prefixScalars - 1)) {
                flagBytes[0] |= 2;
            }
            updateStorageBuffer(engine, chunk.flags.buffer, flagBytes, chunk.flags.offset);
        }
    }
}

/** @internal Bounded shape submissions, then one deferred ordered layout/publication/status sequence. */
export async function executeChunkedResidentComputeText(
    data: ComputeTextData,
    state: ComputeTextChunkState,
    layout: ComputeTextParagraphLayout,
    assertLive: () => void
): Promise<{ status: Promise<number> }> {
    const engine = data._engine;
    releaseTasks(state);
    clearChunks(data, state);
    let projected = 0;
    const workspace = state.plan!.workspace;
    for (const [name, bytes] of workspace) {
        projected += Math.max(bytes, state.buffers.get(name)?.byteLength ?? 0);
    }
    for (const [name, entry] of state.buffers) {
        if (name !== "program" && name !== "tables" && !workspace.has(name)) {
            projected += entry.byteLength;
        }
    }
    if (projected > Math.max(3 * 1024 * 1024, Math.floor(state.plan!.workspaceBytes * 1.25))) {
        for (const [name, entry] of state.buffers) {
            if (name !== "program" && name !== "tables") {
                retireBuffer(data, state, entry);
                state.buffers.delete(name);
            }
        }
    }
    await shapeChunks(data, state, assertLive);
    assertLive();
    const carry = buffer(data, state, "carry", 512);
    updateStorageBuffer(engine, carry, new Uint32Array(128));
    const scratch = buffer(data, state, "replay", 16 + state.plan!.maxCapacity * 16 + (state.plan!.maxCapacity + 4) * 32);
    const commands: ComputeTask[] = [];
    const names = ["params", "shaped", "flags", "slots", "instances", "info", "carry", "scratch"];
    const make = (chunk: ComputeTextResidentChunk, entry: string, count = 1) =>
        task(
            data,
            state,
            shader(data, state, entry, COMPUTE_TEXT_CHUNK_LAYOUT, names, 4, entry),
            { params: chunk.params, shaped: chunk.shaped, flags: chunk.flags, slots: data._slots, instances: chunk.instances, info: chunk.metadata, carry, scratch },
            count
        );
    let index = 0;
    state.chunks.forEach((paragraph, p) =>
        paragraph.forEach((chunk, i) => {
            const words = new Uint32Array(16);
            const floats = new Float32Array(words.buffer);
            words.set([chunk.source.capacity, chunk.source.scalarStart, chunk.source.scalars, p, Number(i === 0), Number(i === paragraph.length - 1)]);
            floats[6] = layout.maxWidthPx;
            floats[7] = layout.lineHeightPx!;
            floats[8] = layout.paragraphSpacingPx ?? 0;
            floats[9] = data.fontSizePx / data._font.unitsPerEm;
            words[10] = data._font.glyphCount;
            words[11] = index++;
            words[12] = Number(chunk.dirty);
            words[13] = layout.align === "center" ? 1 : layout.align === "right" ? 2 : 0;
            updateStorageBuffer(engine, chunk.params.buffer, words, chunk.params.offset);
        })
    );
    const chunks = state.chunks.flat();
    for (const chunk of chunks.slice().reverse()) {
        commands.push(make(chunk, "summary"));
    }
    for (const chunk of chunks) {
        commands.push(make(chunk, "measure"));
    }
    for (const chunk of chunks.slice().reverse()) {
        commands.push(make(chunk, "replayChunk"), make(chunk, "place", Math.ceil(chunk.source.capacity / 64)));
    }
    const publish = shader(data, state, "publish", COMPUTE_TEXT_CHUNK_PUBLISH, ["info", "carry", "indirect", "status"], 2);
    for (const chunk of chunks) {
        commands.push(task(data, state, publish, { info: chunk.metadata, carry, indirect: chunk.indirect, status: data._statusBuffer }));
    }
    if (!chunks.length) {
        commands.push(task(data, state, publish, { info: buffer(data, state, "empty-metadata", 1024), carry, indirect: data._indirect, status: data._statusBuffer }));
    }
    updateStorageBuffer(engine, data._statusBuffer, Uint32Array.of(0));
    for (const command of commands) {
        await prepareComputeTask(command);
        assertLive();
    }
    let staging = state.freeStaging.pop();
    if (!staging) {
        staging = data._device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: "compute-text-chunk-status" });
        state.staging.push(staging);
    }
    const readback = staging;
    const revision = data._revision;
    const status = submitDeferredComputeTaskResult(commands, data._statusBuffer, 4, (bytes) => new DataView(bytes).getUint32(0, true), readback).finally(() => {
        settleRetirement(state, revision);
        if (state.released) {
            const index = state.staging.indexOf(readback);
            if (index >= 0) {
                readback.destroy();
                state.staging.splice(index, 1);
            }
        } else if (readback.mapState === "unmapped") {
            state.freeStaging.push(readback);
        }
    });
    data._drawRanges = state.drawRanges;
    void status.catch(() => undefined);
    return { status };
}
