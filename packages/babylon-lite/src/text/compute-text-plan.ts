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
import type { ComputeTextProgram } from "./compute-text-program.js";
import type { ComputeTextSpaceProof } from "./compute-text-segment.js";
import { directionForComputeTextScript } from "./compute-text-selection.js";
import type { LayoutTable } from "./font-parser/opentype-layout.js";
import { findLayoutScript } from "./font-parser/opentype-layout.js";

declare const computeTextPlanBrand: unique symbol;

const REQUIRED_FEATURE_NONE = 0xffff;

export type ComputeTextProfileId = "default" | "arabic" | "hebrew" | "indic";
export type ComputeTextPause = "none" | "setupSyllables" | "initialReorder" | "finalReorder" | "clearSyllables";
export interface ComputeTextStage {
    readonly table: "GSUB" | "GPOS";
    readonly roots: readonly { readonly lookup: number; readonly mask: number; readonly flags: number }[];
    readonly pause: ComputeTextPause;
}
export interface ComputeTextKernelFragments {
    readonly loadGlyph?: string;
    readonly profilePrepass?: string;
    readonly glyphMask?: string;
    readonly pauses?: string;
    readonly glyphAddress?: string;
    readonly recordInit?: string;
    readonly gates?: string;
    readonly postprocess?: string;
}
export interface ComputeTextProfileData {
    readonly tables: () => Uint32Array;
    readonly closureSeeds?: readonly number[];
    readonly capacity?: (scalars: Uint32Array) => number;
    readonly spaceLocal?: boolean;
    readonly normalizationClosure?: readonly (readonly [number, number])[];
    /** Applied feature tags, for profile proofs that only model known stage semantics. */
    readonly features?: readonly string[];
}
export interface ComputeTextProfileSupport {
    readonly glyphWords: 8 | 16;
    readonly masked: boolean;
    readonly fragments?: ComputeTextKernelFragments;
    readonly planes?: number;
    readonly auxBytes?: (capacity: number) => number;
    readonly joiningType?: (scalar: number) => number;
    readonly markClass?: (scalar: number) => number;
    readonly isMcm?: (scalar: number) => boolean;
    /** Packed per-scalar properties only; allows one shared mark-table search. */
    readonly inputProperties?: (scalar: number) => number;
    readonly buildPlan?: (font: ComputeFont) => ComputeTextPlan;
    readonly validateRun?: (selection: ComputeTextRunSelection) => void;
    readonly available?: boolean;
    /** Paragraph wrapping support, installed only by profiles that prove their own break spaces. */
    readonly paragraph?: ComputeTextParagraphProfile;
}

/** @internal Profile paragraph seam: space proof, paragraph flag bytes and grapheme-chain extent. */
export interface ComputeTextParagraphProfile {
    readonly spaces: (program: ComputeTextProgram) => ComputeTextSpaceProof | null;
    /** Writes paragraph flag bytes for scalars `[lo, hi)` to `out[outOffset + i - lo]`, reading context from the full arrays.
     * Throws `unavailable` when a candidate space needs a proof and `proof` is null. */
    readonly flags: (
        scalars: Uint32Array,
        glyphIds: Uint32Array,
        proof: ComputeTextSpaceProof | null,
        lo: number,
        hi: number,
        out: Uint8Array,
        outOffset: number,
        unavailable: string
    ) => void;
    /** Smallest `h >= hi` from which flags no longer depend on scalars before `h - 1`. */
    readonly chainEnd: (scalars: Uint32Array, hi: number) => number;
}

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
    /** @internal Script-specific data and execution footprint, not lookup semantics. */
    readonly _profile: ComputeTextProfileId;
    /** @internal Ordered root records; a lookup may appear in more than one stage. */
    readonly _stages: readonly ComputeTextStage[];
    /** @internal Optional lazy profile tables. */
    readonly _profileData?: ComputeTextProfileData;
}

let plansByFont: WeakMap<ComputeFont, Map<string, ComputeTextPlan>> | null = null;
let supports: Map<ComputeTextProfileId, ComputeTextProfileSupport> | null = null;
let importers: Map<ComputeTextProfileId, () => Promise<void>> | null = null;
let pendingProfiles: Map<ComputeTextProfileId, Promise<void>> | null = null;
let arabicFontSupport: WeakMap<ComputeTextPlan, boolean> | null = null;

/** @internal Called only by a lazy profile installer. */
export function installComputeTextProfile(id: ComputeTextProfileId, support: ComputeTextProfileSupport): void {
    (supports ??= new Map()).set(id, support);
}

/** @internal No module-level registration or eager profile import. */
export function profileSupport(id: ComputeTextProfileId): ComputeTextProfileSupport | undefined {
    return supports?.get(id);
}

/** @internal Importer seam also used by focused lifecycle tests. */
export function installComputeTextProfileLoader(id: ComputeTextProfileId, importer: () => Promise<void>): void {
    (importers ??= new Map()).set(id, importer);
    pendingProfiles?.delete(id);
    supports?.delete(id);
}

/** @internal Lazy, memoized, retryable. Import failures are propagated, never replaced. */
export function ensureComputeTextProfile(id: ComputeTextProfileId): Promise<void> {
    if (profileSupport(id)) {
        return Promise.resolve();
    }
    const pending = pendingProfiles?.get(id);
    if (pending) {
        return pending;
    }
    const importer = importers?.get(id) ?? (id === "arabic" ? loadArabicProfile : id === "indic" ? loadIndicProfile : undefined);
    if (!importer) {
        return Promise.reject(new Error(`compute text profile ${id} is not available`));
    }
    const promise = Promise.resolve()
        .then(importer)
        .then(() => {
            if (!profileSupport(id)) {
                throw new Error(`compute text profile ${id} installer did not install support`);
            }
        })
        .catch((error: unknown) => {
            pendingProfiles?.delete(id);
            supports?.delete(id);
            throw error;
        });
    (pendingProfiles ??= new Map()).set(id, promise);
    return promise;
}

async function loadArabicProfile(): Promise<void> {
    const profile = await import("./compute-text-arabic.js");
    profile.installComputeTextArabicProfile();
}

async function loadIndicProfile(): Promise<void> {
    const profile = await import("./compute-text-indic.js");
    profile.installComputeTextIndicProfile();
}

/** @internal Entry points check this after nominal mapping, preserving unmapped precedence. */
export function validateComputeTextPlanFont(plan: ComputeTextPlan): void {
    if (plan.script !== "arab") {
        return;
    }
    let supported = arabicFontSupport?.get(plan);
    if (supported === undefined) {
        const table = ensureComputeFontLayout(plan.font).gsub;
        const language = selectedLanguage(table, plan.script);
        const indices = language ? [...language.featureIndices, ...(language.requiredFeatureIndex === REQUIRED_FEATURE_NONE ? [] : [language.requiredFeatureIndex])] : [];
        supported = indices.some((index) => ["isol", "fina", "medi", "init"].includes(table!.features[index]?.tag ?? ""));
        (arabicFontSupport ??= new WeakMap()).set(plan, supported);
    }
    if (!supported) {
        throw new Error("createComputeTextPlan: font has no Arabic joining features (isol/fina/medi/init); Arabic joining fallback is unsupported");
    }
}

/** @internal Profiles whose property data/kernel hooks must be loaded before submission. */
export function requiredComputeTextProfiles(scripts: readonly ComputeTextScript[]): ComputeTextProfileId[] {
    return [...(scripts.includes("arab") ? ["arabic" as const] : []), ...(scripts.includes("deva") ? ["indic" as const] : [])];
}

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
    if (script === "deva") {
        const build = profileSupport("indic")?.buildPlan;
        if (!build) {
            throw new Error("createComputeTextPlan: Indic profile is not loaded");
        }
        const plan = build(font);
        if (!plans) {
            plans = new Map();
            (plansByFont ??= new WeakMap()).set(font, plans);
        }
        plans.set(key, plan);
        return plan;
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
        _profile: script === "arab" ? "arabic" : script === "hebr" ? "hebrew" : "default",
        _stages:
            script === "arab"
                ? arabicStages(layout.gsub, layout.gpos)
                : [
                      { table: "GSUB", roots: gsubRoots.map((lookup) => ({ lookup, mask: 0xffffffff, flags: 0 })), pause: "none" },
                      { table: "GPOS", roots: gposRoots.map((lookup) => ({ lookup, mask: 0xffffffff, flags: 0 })), pause: "none" },
                  ],
    } as unknown as ComputeTextPlan;
    if (!plans) {
        plans = new Map();
        (plansByFont ??= new WeakMap()).set(font, plans);
    }
    plans.set(key, plan);
    return plan;
}

/** @internal Generic staged-plan construction for a lazy profile. No text is shaped. */
export function createComputeTextProfilePlan(
    font: ComputeFont,
    script: ComputeTextScript,
    profile: ComputeTextProfileId,
    stages: readonly ComputeTextStage[],
    data: ComputeTextProfileData
): ComputeTextPlan {
    const layout = ensureComputeFontLayout(font);
    const roots = (table: "GSUB" | "GPOS") =>
        [...new Set(stages.filter((stage) => stage.table === table).flatMap((stage) => stage.roots.map((root) => root.lookup)))].sort((a, b) => a - b);
    const gsub = roots("GSUB"),
        gpos = roots("GPOS");
    return {
        font,
        script,
        direction: directionForComputeTextScript(script),
        language: null,
        _profile: profile,
        _stages: stages,
        _profileData: data,
        _gsubRoots: gsub,
        _gposRoots: gpos,
        _gsubDependencies: closeDependencies(layout.gsub, gsub),
        _gposDependencies: closeDependencies(layout.gpos, gpos),
    } as unknown as ComputeTextPlan;
}

function selectedLanguage(table: LayoutTable | null, script: ComputeTextScript) {
    return (table && (findLayoutScript(table, script) ?? findLayoutScript(table, "DFLT") ?? findLayoutScript(table, "latn")))?.defaultLangSys;
}

function arabicStages(gsub: LayoutTable | null, gpos: LayoutTable | null): ComputeTextStage[] {
    const groups: readonly (readonly string[])[] = [["ccmp", "locl"], ["isol"], ["fina"], ["medi"], ["init"], ["rlig"], ["rclt", "calt"], ["liga"]];
    const stages: ComputeTextStage[] = [];
    const build = (table: LayoutTable | null, tags: readonly string[], first: boolean): void => {
        const language = selectedLanguage(table, "arab");
        if (!table || !language) {
            return;
        }
        const roots = new Map<number, number>();
        const indices = [...language.featureIndices, ...(first && language.requiredFeatureIndex !== REQUIRED_FEATURE_NONE ? [language.requiredFeatureIndex] : [])];
        for (const index of indices) {
            const feature = table.features[index];
            if (!feature || (!tags.includes(feature.tag) && !(first && index === language.requiredFeatureIndex))) {
                continue;
            }
            const form = ["isol", "fina", "medi", "init"].indexOf(feature.tag);
            const mask = form < 0 ? 0x80000000 : 1 << form;
            for (const lookup of feature.lookupListIndices) {
                if (table.lookups[lookup]) {
                    roots.set(lookup, ((roots.get(lookup) ?? 0) | mask) >>> 0);
                }
            }
        }
        if (roots.size) {
            stages.push({ table: table.kind, roots: [...roots].sort(([a], [b]) => a - b).map(([lookup, mask]) => ({ lookup, mask, flags: 0 })), pause: "none" });
        }
    };
    groups.forEach((tags, i) => build(gsub, tags, i === 0));
    build(gpos, ["kern", "mark", "mkmk"], true);
    return stages;
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
    return (
        value === "latn" ||
        value === "grek" ||
        value === "cyrl" ||
        value === "hebr" ||
        value === "arab" ||
        value === "hani" ||
        value === "goth" ||
        value === "deva" ||
        value === "DFLT"
    );
}
