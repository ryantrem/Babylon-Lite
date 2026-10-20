/** Default lookup selection transcribed from text-shaper 0.1.28 `createShapePlanInternal`.
 *
 *  Feature policy is the 0.1.28 default set (`ccmp`, `locl`, `rlig`, `rclt`, `calt`,
 *  `liga`, plus Arabic `isol`/`init`/`medi`/`fina`, and GPOS `kern`/`mark`/`mkmk`).
 *  Language is always the default LangSys (`null` / und). Script fallback is the
 *  requested tag, then `DFLT`, then `latn`. Roots are deduplicated and sorted by
 *  lookup index. Nested lookup indexes are closed separately and are not promoted
 *  into the root list. No user features and no variation-axis substitution. */

import type { ComputeFont } from "./compute-font.js";
import { ensureComputeFontLayout } from "./compute-font.js";
import type { ComputeTextDirection, ComputeTextRunSelection, ComputeTextScript } from "./compute-text-selection.js";
import { directionForComputeTextScript } from "./compute-text-selection.js";
import type { LayoutTable } from "./font-parser/opentype-layout.js";
import { findLayoutScript } from "./font-parser/opentype-layout.js";

declare const computeTextPlanBrand: unique symbol;

const REQUIRED_FEATURE_NONE = 0xffff;

/** CPU lookup plan for one font, script, and direction. No GPU resources. */
export interface ComputeTextPlan {
    readonly [computeTextPlanBrand]: true;
    readonly font: ComputeFont;
    readonly script: ComputeTextScript;
    readonly direction: ComputeTextDirection;
    readonly language: null;
    /** @internal Selected GSUB lookup indexes, in application order. */
    readonly _gsubRoots: readonly number[];
    /** @internal Selected GPOS lookup indexes, in application order. */
    readonly _gposRoots: readonly number[];
    /** @internal GSUB lookups reachable from roots but not themselves roots, ascending. */
    readonly _gsubDependencies: readonly number[];
    /** @internal GPOS lookups reachable from roots but not themselves roots, ascending. */
    readonly _gposDependencies: readonly number[];
}

let plansByFont: WeakMap<ComputeFont, Map<string, ComputeTextPlan>> | null = null;

/** Compile the default Script/LangSys/feature lookups for `selection`. Does not shape text. */
export function createComputeTextPlan(font: ComputeFont, selection: ComputeTextRunSelection): ComputeTextPlan {
    if (!selection._scalars || selection.scalarCount !== selection._scalars.length) {
        throw new Error("createComputeTextPlan: selection scalarCount does not match its scalar array");
    }
    return createComputeTextPlanForSelection(font, selection);
}

/** @internal Paragraph scanning validates source without retaining a whole-paragraph scalar vector. */
export function createComputeTextPlanForSelection(font: ComputeFont, selection: Pick<ComputeTextRunSelection, "script" | "direction" | "language">): ComputeTextPlan {
    const script = selection.script;
    const direction = selection.direction;
    if (!isScript(script) || direction !== directionForComputeTextScript(script) || selection.language !== null) {
        throw new Error("createComputeTextPlan: invalid script, direction, or language");
    }
    const key = `${script}:${direction}`;
    let plans = plansByFont?.get(font);
    const existing = plans?.get(key);
    if (existing) {
        return existing;
    }
    const layout = ensureComputeFontLayout(font);
    const gsubRoots = collectRoots(layout.gsub, script, true);
    const gposRoots = collectRoots(layout.gpos, script, false);
    const plan = {
        font,
        script,
        direction,
        language: null,
        _gsubRoots: gsubRoots,
        _gposRoots: gposRoots,
        _gsubDependencies: closeDependencies(layout.gsub, gsubRoots),
        _gposDependencies: closeDependencies(layout.gpos, gposRoots),
    } as unknown as ComputeTextPlan;
    if (!plans) {
        plans = new Map();
        (plansByFont ??= new WeakMap()).set(font, plans);
    }
    plans.set(key, plan);
    return plan;
}

function collectRoots(table: LayoutTable | null, script: ComputeTextScript, gsub: boolean): number[] {
    if (!table) {
        return [];
    }
    const found = findLayoutScript(table, script) ?? findLayoutScript(table, "DFLT") ?? findLayoutScript(table, "latn");
    const langSys = found?.defaultLangSys;
    if (!found || !langSys) {
        return [];
    }
    const indices = new Set<number>();
    if (langSys.requiredFeatureIndex !== REQUIRED_FEATURE_NONE) {
        addFeature(table, langSys.requiredFeatureIndex, indices, null);
    }
    for (const featureIndex of langSys.featureIndices) {
        addFeature(table, featureIndex, indices, (tag) => (gsub ? isDefaultGsubFeature(script, tag) : isDefaultGposFeature(tag)));
    }
    return [...indices].filter((index) => table.lookups[index] !== undefined).sort((left, right) => left - right);
}

function addFeature(table: LayoutTable, featureIndex: number, indices: Set<number>, enabled: ((tag: string) => boolean) | null): void {
    const feature = table.features[featureIndex];
    if (!feature || (enabled && !enabled(feature.tag))) {
        return;
    }
    for (const lookupIndex of feature.lookupListIndices) {
        indices.add(lookupIndex);
    }
}

function closeDependencies(table: LayoutTable | null, roots: readonly number[]): number[] {
    if (!table || roots.length === 0) {
        return [];
    }
    const seen = new Set<number>(roots);
    const extra: number[] = [];
    const queue = [...roots];
    for (let cursor = 0; cursor < queue.length; cursor++) {
        const index = queue[cursor] ?? 0;
        const lookup = table.lookups[index];
        if (!lookup) {
            continue;
        }
        for (const nested of lookup.nestedLookupIndices) {
            if (seen.has(nested)) {
                continue;
            }
            if (!table.lookups[nested]) {
                throw new Error(`createComputeTextPlan: nested ${table.kind} lookup ${nested} is outside the lookup list`);
            }
            seen.add(nested);
            extra.push(nested);
            queue.push(nested);
        }
    }
    extra.sort((left, right) => left - right);
    return extra;
}

/** text-shaper 0.1.28 `DEFAULT_GSUB_FEATURES`, plus Arabic joining features from `getArabicFeatures`. */
function isDefaultGsubFeature(script: ComputeTextScript, tag: string): boolean {
    if (tag === "ccmp" || tag === "locl" || tag === "rlig" || tag === "rclt" || tag === "calt" || tag === "liga") {
        return true;
    }
    return script === "arab" && (tag === "isol" || tag === "init" || tag === "medi" || tag === "fina");
}

/** text-shaper 0.1.28 `DEFAULT_GPOS_FEATURES`. */
function isDefaultGposFeature(tag: string): boolean {
    return tag === "kern" || tag === "mark" || tag === "mkmk";
}

function isScript(value: string): value is ComputeTextScript {
    return value === "latn" || value === "grek" || value === "cyrl" || value === "hebr" || value === "arab" || value === "hani" || value === "goth" || value === "DFLT";
}
