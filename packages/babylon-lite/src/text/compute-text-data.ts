/** GPU-resident compute text. Shaping and placement stay on the device; JavaScript reads only a status word. */

import type { EngineContext } from "../engine/engine.js";
import { createStorageBuffer, disposeStorageBuffer, updateStorageBuffer } from "../resource/storage-buffer.js";
import type { StorageBuffer } from "../resource/storage-buffer.js";
import type { ComputeFont } from "./compute-font.js";
import { mapComputeFontScalars } from "./compute-font.js";
import { closeComputeTextGlyphs } from "./compute-text-closure.js";
import { createComputeTextPlan } from "./compute-text-plan.js";
import type { ComputeTextInputSlab, ComputeTextRunSelection } from "./compute-text-selection.js";
import { createComputeTextInputSlab, selectComputeTextRun } from "./compute-text-selection.js";
import { extractComputeGlyphCurves } from "./compute-glyph-extraction.js";
import { updateGlyphStorage } from "./glyph-storage.js";
import type { CurveSetId, GlyphStorage } from "./glyph-storage.js";
import { TEXT_INSTANCE_BYTES, TEXT_STYLE_BYTES, TEXT_STYLE_FLOATS } from "./text-buffer-layout.js";
import { ensureSharedAtlasGpu } from "./_gpu/text-textures.js";
import { createStyleBuffer } from "./_gpu/text-style-buffer.js";
import { COMPUTE_TEXT_HEADER_WORDS, COMPUTE_TEXT_RECORD_WORDS, computeTextShapeCapacity } from "./_gpu/compute-text-shape.js";
import { executeResidentComputeText, releaseResidentComputeText, residentStatusMessage } from "./_gpu/compute-text-resident.js";
import type { ResidentShapeGroup } from "./_gpu/compute-text-resident.js";

declare const computeTextDataBrand: unique symbol;

const MISSING_SLOT = 0xffffffff;
const EMPTY_SLOT = 0xfffffffe;
const MAX_PACKED_INDEX = 0xfffe;

/** Branded GPU-resident text block. Not a CPU text block and never a `PlacedGlyph` list. */
export interface ComputeTextData {
    readonly [computeTextDataBrand]: true;
    readonly status: "ready" | "busy" | "lost" | "disposed";
    readonly fontSizePx: number;
    /** @internal */
    _engine: EngineContext;
    /** @internal */
    _device: GPUDevice;
    /** @internal */
    _font: ComputeFont;
    /** @internal Caller-owned atlas. Disposing this block does not dispose it. */
    _storage: GlyphStorage;
    /** @internal */
    _curveSet: CurveSetId;
    /** @internal */
    _generation: number;
    /** @internal */
    _onDeviceLost(): void;
    /** @internal */
    _dead: boolean;
    /** @internal */
    _rejectPending: ((error: Error) => void) | null;
    /** @internal */
    _shaped: StorageBuffer;
    /** @internal */
    _instances: StorageBuffer;
    /** @internal */
    _indirect: StorageBuffer;
    /** @internal */
    _slots: StorageBuffer;
    /** @internal */
    _statusBuffer: StorageBuffer;
    /** @internal */
    _style: GPUBuffer;
    /** @internal */
    _released: boolean;
}

/** Create resident buffers for one font and one base style. `storage` is caller-owned and may be shared. Does not shape. */
export function createComputeTextData(
    engine: EngineContext,
    font: ComputeFont,
    storage: GlyphStorage,
    curveSet: CurveSetId,
    options?: { readonly fontSizePx?: number }
): ComputeTextData {
    if (!engine?._device) {
        throw new Error("createComputeTextData: engine has no device");
    }
    if (!font || !Number.isSafeInteger(font.glyphCount) || font.glyphCount <= 0 || !Number.isFinite(font.unitsPerEm) || font.unitsPerEm <= 0) {
        throw new Error("createComputeTextData: font is invalid");
    }
    if (!storage) {
        throw new Error("createComputeTextData: storage is required");
    }
    if (typeof curveSet !== "string" || curveSet.length === 0) {
        throw new Error("createComputeTextData: curveSet is required");
    }
    const fontSizePx = options?.fontSizePx ?? 32;
    if (!Number.isFinite(fontSizePx) || fontSizePx <= 0) {
        throw new Error("createComputeTextData: fontSizePx must be a positive finite number");
    }
    const device = engine._device;
    const style = createStyleBuffer(device, TEXT_STYLE_BYTES, "compute-text-style");
    const styleWords = new Float32Array(TEXT_STYLE_FLOATS);
    styleWords[0] = 1;
    styleWords[1] = 1;
    styleWords[2] = 1;
    styleWords[3] = 1;
    styleWords[4] = font.unitsPerEm / fontSizePx;
    device.queue.writeBuffer(style, 0, styleWords.buffer as ArrayBuffer, styleWords.byteOffset, styleWords.byteLength);
    const indirectWords = new Uint32Array([6, 0, 0, 0]);
    const data = {
        status: "ready",
        fontSizePx,
        _engine: engine,
        _device: device,
        _font: font,
        _storage: storage,
        _curveSet: curveSet,
        _generation: 1,
        _onDeviceLost: () => failTerminal(data, "lost", "updateComputeTextData: device lost"),
        _dead: false,
        _rejectPending: null,
        _shaped: createStorageBuffer(engine, 64, { label: "compute-text-shaped", writable: true }),
        _instances: createStorageBuffer(engine, 64, { label: "compute-text-instances", writable: true, vertex: true }),
        _indirect: createStorageBuffer(engine, indirectWords, { label: "compute-text-indirect", writable: true, indirect: true }),
        _slots: createStorageBuffer(engine, font.glyphCount * 4, { label: "compute-text-slots" }),
        _statusBuffer: createStorageBuffer(engine, 4, { label: "compute-text-status", writable: true }),
        _style: style,
        _released: false,
    } as ComputeTextData;
    const missing = new Uint32Array(font.glyphCount);
    missing.fill(MISSING_SLOT);
    updateStorageBuffer(engine, data._slots, missing);
    void device.lost.then(() => {
        if (data._device !== device || data.status === "disposed") {
            return;
        }
        failTerminal(data, "lost", "updateComputeTextData: device lost");
    });
    return data;
}

/** Select, close, install outlines, shape, and pack. Reads status only. Throws unless the draw is complete. */
export async function updateComputeTextData(data: ComputeTextData, texts: readonly string[]): Promise<void> {
    assertReady(data);
    const generation = data._generation;
    const device = data._device;
    clearIndirect(data);
    setStatus(data, "busy");
    let rejectPending: (error: Error) => void = () => undefined;
    const cancelled = new Promise<never>((_, reject) => {
        rejectPending = reject;
    });
    data._rejectPending = rejectPending;
    try {
        const snapshot = texts.map((text, run) => {
            if (typeof text !== "string") {
                throw new Error(`updateComputeTextData: run ${run} text must be a string`);
            }
            return text;
        });
        const seeds = new Set<number>();
        const slab = createComputeTextInputSlab(snapshot, false);
        const prepared = snapshot.map((text, run) => prepareRun(data._font, text, run, seeds, slab));
        const groups = groupRuns(data._font, prepared);
        let closure;
        try {
            closure = closeComputeTextGlyphs(
                data._font,
                groups.map((group) => group.plan),
                [...seeds]
            );
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new Error(`${message} (witness: ${snapshot.join(" | ") || "<empty>"})`, { cause: error });
        }
        installClosure(data, closure.glyphIds);
        const capacity = sharedCapacity(groups);
        const runCount = prepared.length;
        growDrawBuffers(data, runCount, capacity);
        const status = await Promise.race([
            executeResidentComputeText(
                data,
                data._engine,
                groups,
                {
                    shaped: data._shaped,
                    instances: data._instances,
                    indirect: data._indirect,
                    slots: data._slots,
                    status: data._statusBuffer,
                },
                capacity,
                runCount,
                data.fontSizePx,
                data._font.unitsPerEm,
                data._font.glyphCount,
                () => assertCurrent(data, generation, device)
            ),
            cancelled,
        ]);
        assertCurrent(data, generation, device);
        const failure = residentStatusMessage(status);
        if (failure) {
            throw new Error(failure);
        }
    } catch (error) {
        if (!data._dead && data._device === data._engine._device) {
            clearIndirect(data);
        }
        throw error;
    } finally {
        if (data._rejectPending === rejectPending) {
            data._rejectPending = null;
        }
        if (data.status === "busy") {
            setStatus(data, "ready");
        }
    }
}

/** Release resident buffers and invalidate any layer bundle that borrowed them. Idempotent. */
export function disposeComputeTextData(data: ComputeTextData): void {
    if (!data || data.status === "disposed") {
        return;
    }
    failTerminal(data, "disposed", "updateComputeTextData: disposed");
}

function installClosure(data: ComputeTextData, glyphIds: readonly number[]): void {
    const curves = new Map(data._storage._curveSets.get(data._curveSet)?._curves);
    try {
        extractComputeGlyphCurves(data._font, new Set(glyphIds), curves);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`updateComputeTextData: closure cannot install outlines (${message})`, { cause: error });
    }
    updateGlyphStorage(data._storage, data._curveSet, curves);
    const curveSet = data._storage._curveSets.get(data._curveSet);
    if (!curveSet) {
        throw new Error("updateComputeTextData: curve set is missing");
    }
    ensureSharedAtlasGpu(data._device, curveSet._atlas);
    const slots = new Uint32Array(data._font.glyphCount);
    slots.fill(MISSING_SLOT);
    for (const glyphId of glyphIds) {
        if (curves.has(glyphId)) {
            const slot = curveSet._atlas._glyphSlots.get(glyphId);
            if (!slot || slot._index > MAX_PACKED_INDEX) {
                throw new Error(`updateComputeTextData: closure cannot see atlas slot for glyph ${glyphId}`);
            }
            slots[glyphId] = slot._index;
        } else {
            slots[glyphId] = EMPTY_SLOT;
        }
    }
    updateStorageBuffer(data._engine, data._slots, slots);
}

function growDrawBuffers(data: ComputeTextData, runCount: number, capacity: number): void {
    const shapedBytes = Math.max(1, runCount) * (COMPUTE_TEXT_HEADER_WORDS + capacity * COMPUTE_TEXT_RECORD_WORDS) * 4;
    const instanceBytes = Math.max(1, runCount * capacity) * TEXT_INSTANCE_BYTES;
    for (const bytes of [shapedBytes, instanceBytes]) {
        const limits = data._device.limits;
        if (!Number.isSafeInteger(bytes) || bytes > limits.maxBufferSize || bytes > limits.maxStorageBufferBindingSize) {
            throw new Error(`updateComputeTextData: draw buffer of ${bytes} bytes exceeds device storage limits`);
        }
    }
    data._shaped = grow(data, data._shaped, shapedBytes, { label: "compute-text-shaped", writable: true });
    const nextInstances = grow(data, data._instances, instanceBytes, { label: "compute-text-instances", writable: true, vertex: true });
    if (nextInstances !== data._instances) {
        data._instances = nextInstances;
    }
}

function sharedCapacity(groups: readonly ResidentShapeGroup[]): number {
    let capacity = 1;
    for (const group of groups) {
        if (group.runs.length === 0) {
            continue;
        }
        capacity = Math.max(capacity, computeTextShapeCapacity(group.plan, group.runs));
    }
    return capacity;
}

function groupRuns(font: ComputeFont, runs: readonly PreparedRun[]): ResidentShapeGroup[] {
    const groups: ResidentShapeGroup[] = [];
    let index = 0;
    while (index < runs.length) {
        const selection = runs[index]!.selection;
        const start = index;
        const chunk: PreparedRun[] = [];
        while (index < runs.length) {
            const next = runs[index]!;
            if (next.selection.script !== selection.script || next.selection.direction !== selection.direction) {
                break;
            }
            chunk.push(next);
            index++;
        }
        groups.push({
            plan: createComputeTextPlan(font, selection),
            runs: chunk,
            outputRun: start,
        });
    }
    return groups;
}

interface PreparedRun {
    readonly selection: ComputeTextRunSelection;
    readonly scalars: Uint32Array;
    readonly glyphIds: Uint32Array;
}

function prepareRun(font: ComputeFont, text: string, run: number, seeds: Set<number>, slab: ComputeTextInputSlab): PreparedRun {
    let selection: ComputeTextRunSelection;
    try {
        selection = selectComputeTextRun(text, font, seeds, slab);
    } catch (error) {
        const message = error instanceof Error ? error.message.replace(/^selectComputeTextRun:\s*/, "") : "invalid text";
        throw new Error(`updateComputeTextData: run ${run} ${message}`, { cause: error });
    }
    const glyphIds = mapComputeFontScalars(
        font,
        selection._scalars,
        (_index, error) => {
            const message = error instanceof Error ? error.message.replace(/^mapComputeFontScalar:\s*/, "") : "unmapped scalar";
            throw new Error(`updateComputeTextData: run ${run} ${message}`, { cause: error });
        },
        selection._nominal
    );
    return { selection, scalars: selection._scalars, glyphIds };
}

function clearIndirect(data: ComputeTextData): void {
    updateStorageBuffer(data._engine, data._indirect, new Uint32Array([6, 0, 0, 0]));
}

function grow(data: ComputeTextData, buffer: StorageBuffer, bytes: number, options: { label: string; writable?: boolean; vertex?: boolean; indirect?: boolean }): StorageBuffer {
    if (buffer.byteLength >= bytes && buffer._engine._device === data._device && !buffer._destroyed) {
        return buffer;
    }
    disposeStorageBuffer(buffer);
    return createStorageBuffer(data._engine, bytes, options);
}

function failTerminal(data: ComputeTextData, status: "lost" | "disposed", message: string): void {
    if (data._released) {
        setStatus(data, status);
        return;
    }
    data._released = true;
    data._dead = true;
    data._generation++;
    setStatus(data, status);
    const reject = data._rejectPending;
    data._rejectPending = null;
    reject?.(new Error(message));
    releaseResidentComputeText(data);
    for (const buffer of [data._shaped, data._instances, data._indirect, data._slots, data._statusBuffer]) {
        disposeStorageBuffer(buffer);
    }
    data._style.destroy();
}

function assertReady(data: ComputeTextData): void {
    if (data?.status === "lost") {
        throw new Error("updateComputeTextData: device lost");
    }
    if (!data || data.status === "disposed" || data._dead) {
        throw new Error("updateComputeTextData: data is disposed");
    }
    if (data._device !== data._engine._device) {
        failTerminal(data, "lost", "updateComputeTextData: device lost");
        throw new Error("updateComputeTextData: device lost");
    }
    if (data.status === "busy") {
        throw new Error("updateComputeTextData: data is busy");
    }
}

function assertCurrent(data: ComputeTextData, generation: number, device: GPUDevice): void {
    if (data.status === "disposed" || data._dead || data._generation !== generation) {
        throw new Error("updateComputeTextData: disposed");
    }
    if (data.status === "lost" || data._engine._device !== device || data._device !== device) {
        throw new Error("updateComputeTextData: device lost");
    }
}

function setStatus(data: ComputeTextData, status: ComputeTextData["status"]): void {
    (data as { status: ComputeTextData["status"] }).status = status;
}
