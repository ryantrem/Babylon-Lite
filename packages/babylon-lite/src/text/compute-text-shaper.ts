/** Opt-in compute text shaper. CPU validation stays here; the GPU kernel is dynamic-imported. */

import type { EngineContext } from "../engine/engine.js";
import { registerManagedResourceDisposer } from "../resource/managed-resource-hooks.js";
import type { ComputeFont } from "./compute-font.js";
import { mapComputeFontScalars } from "./compute-font.js";
import type { ComputeTextPlan } from "./compute-text-plan.js";
import { createComputeTextPlan, ensureComputeTextProfile, profileSupport, requiredComputeTextProfiles, validateComputeTextPlanFont } from "./compute-text-plan.js";
import type { ComputeTextDirection, ComputeTextInputSlab, ComputeTextRunSelection, ComputeTextScript } from "./compute-text-selection.js";
import { createComputeTextInputSlab, selectComputeTextRun } from "./compute-text-selection.js";

declare const computeTextShaperBrand: unique symbol;
declare const computeShapedTextRunBrand: unique symbol;

/** One shaped glyph. Advances and offsets are signed font units. */
export interface ComputeTextGlyph {
    readonly glyphId: number;
    readonly cluster: number;
    readonly xAdvance: number;
    readonly yAdvance: number;
    readonly xOffset: number;
    readonly yOffset: number;
}

/** Validated visual-order records for one input string. */
export interface ComputeShapedTextRun {
    readonly [computeShapedTextRunBrand]: true;
    readonly glyphs: readonly ComputeTextGlyph[];
    readonly scalarCount: number;
    readonly unitsPerEm: number;
    readonly script: ComputeTextScript;
    readonly direction: ComputeTextDirection;
    /** @internal Font that produced these glyph ids. */
    readonly _font: ComputeFont;
}

/** Engine-bound shaper. No public GPU handles. */
export interface ComputeTextShaper {
    readonly [computeTextShaperBrand]: true;
    readonly status: "ready" | "busy" | "lost" | "disposed";
    /** @internal */
    _engine: EngineContext;
    /** @internal Device observed at creation. Replacement is terminal loss. */
    _device: GPUDevice;
    /** @internal Bumped on dispose and device loss so late continuations cannot publish. */
    _generation: number;
    /** @internal Rejects the in-flight batch. */
    _rejectPending: ((error: Error) => void) | null;
    /** @internal Installed by the GPU module to release its buffers and pipeline. */
    _releaseGpu?: () => void;
}

interface PreparedRun {
    readonly sourceRun: number;
    readonly text: string;
    readonly scalars: Uint32Array;
    readonly glyphIds: Uint32Array;
}

type ShapeExecutor = (shaper: ComputeTextShaper, plan: ComputeTextPlan, runs: readonly PreparedRun[], generation: number) => Promise<readonly ComputeTextGlyph[][]>;

let liveByEngine: WeakMap<EngineContext, Set<ComputeTextShaper>> | null = null;
let testExecutor: ShapeExecutor | null = null;
let gpuExecutor: ShapeExecutor | null = null;

/** @internal Unit-test seam. Production shaping dynamic-imports the GPU kernel. */
export function _installComputeTextShapeExecutorForTests(executor: ShapeExecutor | null): void {
    testExecutor = executor;
}

/** Create engine-bound shaper state. GPU pipelines wait until the first shape call. */
export function createComputeTextShaper(engine: EngineContext): ComputeTextShaper {
    if (!engine?._device) {
        throw new Error("createComputeTextShaper: engine has no device");
    }
    const device = engine._device;
    const shaper = {
        status: "ready",
        _engine: engine,
        _device: device,
        _generation: 1,
        _rejectPending: null,
    } as ComputeTextShaper;
    liveSet(engine).add(shaper);
    void device.lost.then(() => {
        if (shaper._device !== device || shaper.status === "disposed") {
            return;
        }
        markTerminal(shaper, "lost", "shapeComputeText: device lost");
    });
    return shaper;
}

/** Shape independent strings for one already compiled plan. Does not mutate TextData. */
export async function shapeComputeTextPlan(
    shaper: ComputeTextShaper,
    plan: ComputeTextPlan,
    texts: readonly string[],
    sourceIndexes?: readonly number[]
): Promise<readonly ComputeShapedTextRun[]> {
    return shapeRuns(shaper, plan, texts, sourceIndexes);
}

async function shapeRuns(
    shaper: ComputeTextShaper,
    plan: ComputeTextPlan,
    texts: readonly string[],
    sourceIndexes?: readonly number[],
    selections?: readonly ComputeTextRunSelection[]
): Promise<readonly ComputeShapedTextRun[]> {
    assertShaperReady(shaper);
    if (!plan?.font || plan.language !== null) {
        throw new Error("shapeComputeText: plan is invalid");
    }
    const snapshot = texts.map((text, run) => {
        if (typeof text !== "string") {
            throw new Error(`shapeComputeText: run ${run} text must be a string`);
        }
        return text;
    });
    if (snapshot.length === 0) {
        return [];
    }
    const slab = selections ? undefined : createComputeTextInputSlab(snapshot, false);
    const prepared = snapshot.map((text, run) => prepareRun(plan, text, sourceIndexes?.[run] ?? run, selections?.[run], slab));
    if (prepared.every((run) => run.scalars.length === 0)) {
        return prepared.map(() => publish(plan, 0, []));
    }
    const generation = shaper._generation;
    const device = shaper._device;
    for (const id of requiredComputeTextProfiles([plan.script])) {
        if (!profileSupport(id)) {
            await ensureComputeTextProfile(id);
            assertCurrent(shaper, generation, device);
            assertShaperReady(shaper);
        }
    }
    setStatus(shaper, "busy");
    let rejectPending: (error: Error) => void = () => undefined;
    const cancelled = new Promise<never>((_, reject) => {
        rejectPending = reject;
    });
    shaper._rejectPending = rejectPending;
    try {
        const executor = testExecutor ?? gpuExecutor ?? (await loadExecutor());
        const copy = !selections || executor === testExecutor;
        assertCurrent(shaper, generation, device);
        const active = prepared.filter((run) => run.scalars.length > 0);
        const shaped = await Promise.race([executor(shaper, plan, active, generation), cancelled]);
        assertCurrent(shaper, generation, device);
        if (shaped.length !== active.length) {
            throw new Error("shapeComputeText: incomplete batch");
        }
        let shapedIndex = 0;
        return prepared.map((run) => {
            if (run.scalars.length === 0) {
                return publish(plan, 0, []);
            }
            const glyphs = shaped[shapedIndex] ?? [];
            shapedIndex++;
            validateGlyphs(plan, run, glyphs);
            return publish(plan, run.scalars.length, glyphs, copy);
        });
    } finally {
        if (shaper._rejectPending === rejectPending) {
            shaper._rejectPending = null;
        }
        if ((shaper.status as ComputeTextShaper["status"]) === "busy") {
            setStatus(shaper, "ready");
        }
    }
}

/** Release shaper-owned GPU resources. Idempotent. Does not invalidate completed results. */
export function disposeComputeTextShaper(shaper: ComputeTextShaper): void {
    if (!shaper || shaper.status === "disposed") {
        return;
    }
    setStatus(shaper, "disposed");
    shaper._generation++;
    const reject = shaper._rejectPending;
    shaper._rejectPending = null;
    reject?.(new Error("shapeComputeText: shaper is disposed"));
    shaper._releaseGpu?.();
    shaper._releaseGpu = undefined;
    liveByEngine?.get(shaper._engine)?.delete(shaper);
}

let shaperByEngine: WeakMap<EngineContext, ComputeTextShaper> | null = null;

/** Shape characters. Returns glyph ids and font-unit positions. Does not mutate TextData. */
export async function shapeComputeText(engine: EngineContext, font: ComputeFont, text: string): Promise<readonly ComputeTextGlyph[]>;
export async function shapeComputeText(engine: EngineContext, font: ComputeFont, texts: readonly string[]): Promise<readonly (readonly ComputeTextGlyph[])[]>;
export async function shapeComputeText(
    engine: EngineContext,
    font: ComputeFont,
    text: string | readonly string[]
): Promise<readonly ComputeTextGlyph[] | readonly (readonly ComputeTextGlyph[])[]> {
    if (!font || !Number.isSafeInteger(font.glyphCount) || font.glyphCount <= 0) {
        throw new Error("shapeComputeText: font is invalid");
    }
    const single = typeof text === "string";
    const texts = single
        ? [text]
        : text.map((item, run) => {
              if (typeof item !== "string") {
                  throw new Error(`shapeComputeText: run ${run} text must be a string`);
              }
              return item;
          });
    const slab = createComputeTextInputSlab(texts, false);
    const groups = new Map<string, { selections: ComputeTextRunSelection[]; indexes: number[]; items: string[] }>();
    texts.forEach((item, index) => {
        let selection: ComputeTextRunSelection;
        try {
            selection = selectComputeTextRun(item, font, undefined, slab);
        } catch (error) {
            const message = error instanceof Error ? error.message.replace(/^selectComputeTextRun:\s*/, "") : "invalid text";
            throw new Error(`shapeComputeText: run ${index} ${message}`, { cause: error });
        }
        const key = `${selection.script}:${selection.direction}`;
        const group = groups.get(key);
        if (group) {
            group.indexes.push(index);
            group.items.push(item);
            group.selections.push(selection);
        } else {
            groups.set(key, { selections: [selection], indexes: [index], items: [item] });
        }
    });
    const output: (readonly ComputeTextGlyph[])[] = texts.map(() => []);
    const shaper = shaperFor(engine);
    if ([...groups.values()].some((group) => group.selections[0]?.script === "deva") && !profileSupport("indic")) {
        assertShaperReady(shaper);
        const generation = shaper._generation,
            device = shaper._device;
        await ensureComputeTextProfile("indic");
        assertCurrent(shaper, generation, device);
        assertShaperReady(shaper);
    }
    for (const group of groups.values()) {
        if (group.selections[0]?.script !== "deva") {
            continue;
        }
        group.selections.forEach((selection, at) => {
            try {
                profileSupport("indic")!.validateRun!(selection);
            } catch (error) {
                throw new Error(`shapeComputeText: run ${group.indexes[at]} ${error instanceof Error ? error.message : String(error)}`, { cause: error });
            }
        });
    }
    for (const group of groups.values()) {
        let plan: ComputeTextPlan;
        try {
            plan = createComputeTextPlan(font, group.selections[0]!);
        } catch (error) {
            if (group.selections[0]!.script !== "deva") {
                throw error;
            }
            throw new Error(`shapeComputeText: run ${group.indexes[0]} ${error instanceof Error ? error.message : String(error)}`, { cause: error });
        }
        const shaped = await shapeRuns(shaper, plan, group.items, group.indexes, group.selections);
        group.indexes.forEach((index, at) => {
            output[index] = shaped[at]?.glyphs ?? [];
        });
    }
    return single ? (output[0] ?? []) : output;
}

function shaperFor(engine: EngineContext): ComputeTextShaper {
    if (!shaperByEngine) {
        shaperByEngine = new WeakMap();
    }
    const current = shaperByEngine.get(engine);
    if (current && current.status !== "lost" && current.status !== "disposed" && current._device === engine._device) {
        return current;
    }
    if (current && current.status !== "disposed") {
        disposeComputeTextShaper(current);
    }
    const created = createComputeTextShaper(engine);
    shaperByEngine.set(engine, created);
    return created;
}

function prepareRun(plan: ComputeTextPlan, text: string, run: number, checked?: ComputeTextRunSelection, slab?: ComputeTextInputSlab): PreparedRun {
    let selection: ComputeTextRunSelection;
    try {
        selection = checked ?? selectComputeTextRun(text, plan.font, undefined, slab);
        if (plan._profile === "indic") {
            profileSupport("indic")!.validateRun!(selection);
            if (profileSupport("indic")?.available === false) {
                throw new Error("Indic kernel variant not available");
            }
        }
    } catch (error) {
        const message = error instanceof Error ? error.message.replace(/^selectComputeTextRun:\s*/, "") : "invalid text";
        throw new Error(`shapeComputeText: run ${run} ${message}`, { cause: error });
    }
    if (selection.script !== plan.script && selection.script !== "DFLT") {
        throw new Error(`shapeComputeText: run ${run} script "${selection.script}" does not match plan "${plan.script}"`);
    }
    const glyphIds = mapComputeFontScalars(
        plan.font,
        selection._scalars,
        (index, error) => {
            const message = error instanceof Error ? error.message.replace(/^mapComputeFontScalar:\s*/, "") : "unmapped scalar";
            throw new Error(`shapeComputeText: run ${run} scalar ${index} ${message}`, { cause: error });
        },
        selection._nominal
    );
    if (selection.scalarCount !== 0) {
        validateComputeTextPlanFont(plan);
    }
    return { text, sourceRun: run, scalars: selection._scalars, glyphIds };
}

function validateGlyphs(plan: ComputeTextPlan, run: PreparedRun, glyphs: readonly ComputeTextGlyph[]): void {
    for (let index = 0; index < glyphs.length; index++) {
        const glyph = glyphs[index];
        if (!glyph || !Number.isSafeInteger(glyph.glyphId) || glyph.glyphId < 0 || glyph.glyphId >= plan.font.glyphCount) {
            throw new Error(`shapeComputeText: glyph ${glyph?.glyphId} is outside the font`);
        }
        if (!Number.isSafeInteger(glyph.cluster) || glyph.cluster < 0 || glyph.cluster >= run.scalars.length) {
            throw new Error(`shapeComputeText: cluster ${glyph.cluster} is outside run scalar count ${run.scalars.length}`);
        }
        if (!Number.isSafeInteger(glyph.xAdvance) || !Number.isSafeInteger(glyph.yAdvance) || !Number.isSafeInteger(glyph.xOffset) || !Number.isSafeInteger(glyph.yOffset)) {
            throw new Error("shapeComputeText: shaped value is not a safe integer");
        }
    }
}

function publish(plan: ComputeTextPlan, scalarCount: number, glyphs: readonly ComputeTextGlyph[], copy = true): ComputeShapedTextRun {
    return {
        glyphs: copy ? glyphs.map((glyph) => ({ ...glyph })) : glyphs,
        scalarCount,
        unitsPerEm: plan.font.unitsPerEm,
        script: plan.script,
        direction: plan.direction,
        _font: plan.font,
    } as unknown as ComputeShapedTextRun;
}

async function loadExecutor(): Promise<ShapeExecutor> {
    const gpu = await import("./_gpu/compute-text-shape.js");
    return (gpuExecutor = gpu.executeComputeTextShape);
}

function assertCurrent(shaper: ComputeTextShaper, generation: number, device: GPUDevice): void {
    if (shaper.status === "disposed" || shaper._generation !== generation) {
        throw new Error("shapeComputeText: shaper is disposed");
    }
    if (shaper.status === "lost" || shaper._device !== device || shaper._engine._device !== device) {
        failTerminal(shaper, "lost", "shapeComputeText: device lost");
    }
}

function assertShaper(shaper: ComputeTextShaper): void {
    if (!shaper?._device || !shaper._engine) {
        throw new Error("shapeComputeText: shaper is invalid");
    }
}

function assertShaperReady(shaper: ComputeTextShaper): void {
    assertShaper(shaper);
    if (shaper.status === "disposed") {
        throw new Error("shapeComputeText: shaper is disposed");
    }
    if (shaper.status === "lost" || shaper._device !== shaper._engine._device) {
        failTerminal(shaper, "lost", "shapeComputeText: device lost");
    }
    if (shaper.status === "busy") {
        throw new Error("shapeComputeText: shaper is busy");
    }
}
function failTerminal(shaper: ComputeTextShaper, status: "lost" | "disposed", message: string): never {
    markTerminal(shaper, status, message);
    throw new Error(message);
}

function markTerminal(shaper: ComputeTextShaper, status: "lost" | "disposed", message: string): void {
    if (shaper.status === "disposed") {
        return;
    }
    setStatus(shaper, status);
    shaper._generation++;
    const reject = shaper._rejectPending;
    shaper._rejectPending = null;
    reject?.(new Error(message));
}

function setStatus(shaper: ComputeTextShaper, status: ComputeTextShaper["status"]): void {
    (shaper as { status: ComputeTextShaper["status"] }).status = status;
}

function liveSet(engine: EngineContext): Set<ComputeTextShaper> {
    if (!liveByEngine) {
        liveByEngine = new WeakMap();
    }
    const existing = liveByEngine.get(engine);
    if (existing) {
        return existing;
    }
    const created = new Set<ComputeTextShaper>();
    liveByEngine.set(engine, created);
    registerManagedResourceDisposer(engine, () => {
        for (const shaper of [...created]) {
            disposeComputeTextShaper(shaper);
        }
    });
    return created;
}
