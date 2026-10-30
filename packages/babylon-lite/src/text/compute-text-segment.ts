/** Conservative independence proof for boundaries after a nominal space.
 *
 * B* stays class 0, never skipped/deleted or consumed as a ligature tail. Every
 * cross-boundary lookup therefore has a B*-accepting window. Closures and the
 * d*S non-free nominal range overapproximate every slot, including skipped and
 * ligature-interior glyphs. S bounds spans by induction over ordered GSUB roots:
 * each pass reads pre-pass spans, never its own outputs. Nested records compose
 * in order at their context root. Growing children are admitted only when their
 * outputs match none of that root's context predicates or child input predicates,
 * excluding re-consumption during the root pass. Child windows are enumerated as
 * rules too; finite span closure covers overlapping contexts. Excluding all such windows then proves independence by induction
 * over lookups. Filtering sets and masks never certify a boundary: full predicate
 * windows and every possibly-skippable mark are conservatively retained.
 * A non-mark first right glyph prevents cross-space attachment;
 * rejection stays local and one expanding lookup bounds overflow.
 */

import { ensureComputeFontLayout, mapComputeFontScalars } from "./compute-font.js";
import type { ComputeTextPlan } from "./compute-text-plan.js";
import { profileSupport } from "./compute-text-plan.js";
import type { ComputeGlyphPredicate, ComputeProgramLookup, ComputeProgramRule, ComputeTextProgram } from "./compute-text-program.js";
import {
    compileComputeTextProgram,
    COMPUTE_RULE_MODE_ADJUST,
    COMPUTE_RULE_MODE_CLASS_PAIR,
    COMPUTE_RULE_MODE_CONTEXT,
    COMPUTE_RULE_MODE_LIGATURE,
    COMPUTE_RULE_MODE_MARK_BASE,
    COMPUTE_RULE_MODE_MARK_LIGATURE,
    COMPUTE_RULE_MODE_MARK_MARK,
    COMPUTE_RULE_MODE_MULTIPLE,
    COMPUTE_RULE_MODE_POSITION,
    COMPUTE_RULE_MODE_REJECT,
    COMPUTE_RULE_MODE_SUBSTITUTE,
} from "./compute-text-program.js";
import { coverageIndex, glyphClass } from "./font-parser/opentype-layout.js";

/** @internal */
export const SEGMENT_TARGET = 64;

/** @internal Paragraph flag for a U+0020 before a non-space that `safeAfter` proved safe; other such spaces hold 1. */
export const SAFE_SPACE_FLAG = 3;

/** @internal Per-group, per-logical-run paragraph flags, indexed like the `groups` passed to `segmentComputeTextRuns`. */
export type SegmentSafeSpaces = readonly (readonly (Uint8Array | undefined)[] | undefined)[];

/** @internal Paragraph break proof: whether shaping is independent across the U+0020 at `spaceIndex`.
 * `scalars` are the source scalars aligned with `glyphIds`; profile proofs require them, the generic analysis ignores them. */
export interface ComputeTextSpaceProof {
    readonly safeAfter: (glyphIds: Uint32Array, spaceIndex: number, scalars?: Uint32Array) => boolean;
}

/** @internal */
export interface SegmentAnalysis extends ComputeTextSpaceProof {
    readonly space: number;
    readonly span: number;
    readonly maxD: number;
    readonly crossings: number;
    readonly markBaseHit: boolean;
}

/** @internal */
export interface SegmentRun {
    readonly scalars: Uint32Array;
    readonly glyphIds: Uint32Array;
}

/** @internal */
export interface SegmentGroup {
    readonly plan: ComputeTextPlan;
    readonly runs: readonly SegmentRun[];
    readonly outputRun: number;
}

/** @internal */
export interface ComputeTextSegments {
    readonly groups: readonly SegmentGroup[];
    readonly table: Uint32Array;
    readonly capacity: number;
    readonly count: number;
}

/** @internal */
export interface Slot {
    side: number;
    d: number;
    p: ComputeGlyphPredicate;
    bit: number;
}

/** @internal */
export interface Crossing {
    slots: Slot[];
    anchor: Slot;
}

interface GlyphInfo {
    closure: Uint32Array;
    free: boolean;
    mark: boolean;
    anchorHits: number[];
    /** Per-slot memo indexed by Slot.bit: 0 unknown, 1 miss, 2 hit. */
    slots: Uint8Array | null;
}

let analyses: WeakMap<ComputeTextProgram, SegmentAnalysis | null> | null = null;
let predicateIds: WeakMap<ComputeGlyphPredicate, Set<number>> | null = null;

/** @internal Clear only segmentation caches, for independent analysis measurements. */
export function clearComputeTextSegmentationCache(): void {
    analyses = null;
    predicateIds = null;
}

/** @internal Predicate membership, memoizing glyph-id sets. */
export function matches(p: ComputeGlyphPredicate, g: number): boolean {
    if (p.kind === "glyphs") {
        let ids = predicateIds?.get(p);
        if (!ids) {
            ids = new Set(p.ids);
            (predicateIds ??= new WeakMap()).set(p, ids);
        }
        return ids.has(g);
    }
    return (!p.coverage || coverageIndex(p.coverage, g) !== null) && glyphClass(p.classDef, g) === p.classId;
}

/** @internal Lazy, program-identity-cached boundary analysis; null is cached too. */
export function analyzeComputeTextSegmentation(program: ComputeTextProgram): SegmentAnalysis | null {
    if (analyses?.has(program)) {
        return analyses.get(program)!;
    }
    const result = analyze(program);
    (analyses ??= new WeakMap()).set(program, result);
    return result;
}

/** @internal */
export interface SpaceContext {
    readonly space: number;
    readonly cls: (g: number) => number;
    readonly closure: (g: number, any?: boolean) => Uint32Array;
    readonly bStar: Uint32Array;
    readonly skipped: (g: number) => boolean;
}

/** @internal */
export interface CrossingSet {
    readonly crossings: Crossing[];
    readonly allSlots: Slot[];
    readonly maxD: number;
    readonly inB: Uint8Array;
}

/** @internal One B*-crossing rule window: backtrack (reversed) · input · lookahead. */
export interface ComputeTextCrossingRule {
    readonly lookup: ComputeProgramLookup;
    readonly rule: ComputeProgramRule;
    readonly window: readonly ComputeGlyphPredicate[];
    /** Non-final window slot whose predicate meets B*. */
    readonly slot: number;
    /** Window slots, including the final one, whose predicates meet B*. */
    readonly spaceSlots: number;
}

/** @internal Shared space-crossing facts for profile-specific paragraph proofs. */
export interface ComputeTextCrossings {
    readonly space: number;
    /** Closure of the space glyph under GSUB single/multiple substitutions (ligature-tail free, GDEF class 0). */
    readonly bStar: Uint32Array;
    /** Every glyph reachable from `glyph` through any GSUB substitution, multiple or ligature output. */
    readonly closure: (glyph: number) => Uint32Array;
    readonly glyphClass: (glyph: number) => number;
    /** True when a lookup flag of the program can skip `glyph`. */
    readonly skipped: (glyph: number) => boolean;
    readonly matches: (predicate: ComputeGlyphPredicate, glyph: number) => boolean;
    /** True when a B* glyph is covered as a mark, base or ligature by any mark-attachment rule. */
    readonly attached: boolean;
    readonly crossings: readonly ComputeTextCrossingRule[];
}

function outputs(r: ComputeProgramRule): readonly number[] {
    const action = r.action;
    return action.kind === "ligature" ? [action.glyph] : action.kind === "multiple" || action.kind === "substitute" ? action.glyphs : [];
}

/** @internal Rule-mode admission, the GSUB producer index, the space glyph, GDEF classes, memoized `any`-closures and B*.
 * Null for an unsupported rule mode, a B* that is not class 0 or is a ligature tail, or (`limitExpanding`) a second expanding GSUB lookup. */
export function spaceContext(program: ComputeTextProgram, limitExpanding: boolean): SpaceContext | null {
    const modes = [
        COMPUTE_RULE_MODE_SUBSTITUTE,
        COMPUTE_RULE_MODE_MULTIPLE,
        COMPUTE_RULE_MODE_LIGATURE,
        COMPUTE_RULE_MODE_CONTEXT,
        COMPUTE_RULE_MODE_REJECT,
        COMPUTE_RULE_MODE_POSITION,
        COMPUTE_RULE_MODE_CLASS_PAIR,
        COMPUTE_RULE_MODE_ADJUST,
        COMPUTE_RULE_MODE_MARK_BASE,
        COMPUTE_RULE_MODE_MARK_MARK,
        COMPUTE_RULE_MODE_MARK_LIGATURE,
    ];
    const ligatures: ComputeProgramRule[] = [];
    const byGlyph = new Map<number, ComputeProgramRule[]>();
    const classProducers: ComputeProgramRule[] = [];
    let expanding = 0;
    for (const lookup of program.lookups) {
        if (limitExpanding && lookup.table === "GSUB" && lookup.rules.some((r) => r.action.kind === "multiple" && r.action.glyphs.length > 1) && ++expanding > 1) {
            return null;
        }
        for (const rule of lookup.rules) {
            if (!modes.includes(rule.mode)) {
                return null;
            }
            if (lookup.table !== "GSUB") {
                continue;
            }
            if (rule.action.kind === "context") {
                if (rule.action.lookups.some((nested) => !program.lookups.some((l) => l.table === "GSUB" && l.lookupIndex === nested.lookupIndex))) {
                    return null;
                }
            }
            if (rule.action.kind !== "substitute" && rule.action.kind !== "multiple" && rule.action.kind !== "ligature") {
                continue;
            }
            if (rule.action.kind === "ligature") {
                ligatures.push(rule);
            }
            const predicates = rule.action.kind === "ligature" ? rule.input : rule.input.slice(0, 1);
            let hasClass = false;
            const indexed = new Set<number>();
            for (const p of predicates) {
                if (p.kind === "class") {
                    hasClass = true;
                } else {
                    for (const g of p.ids) {
                        if (!indexed.has(g)) {
                            indexed.add(g);
                            let list = byGlyph.get(g);
                            if (!list) {
                                byGlyph.set(g, (list = []));
                            }
                            list.push(rule);
                        }
                    }
                }
            }
            if (hasClass) {
                classProducers.push(rule);
            }
        }
    }
    const font = program.plan.font;
    let space: number;
    try {
        space = mapComputeFontScalars(font, Uint32Array.of(0x20), (_i, error) => {
            throw error;
        })[0]!;
    } catch {
        return null;
    }
    if (!Number.isInteger(space) || space < 0 || space >= font.glyphCount) {
        return null;
    }
    const gdef = ensureComputeFontLayout(font).gdef;
    const cls = (g: number) => (gdef ? glyphClass(gdef.glyphClassDef, g) : 0);
    const accepts = (r: ComputeProgramRule, g: number, any: boolean) =>
        r.action.kind === "ligature" ? any && r.input.some((p) => matches(p, g)) : !!r.input[0] && matches(r.input[0], g);
    const closureMemo = new Map<number, Uint32Array>();
    const closure = (g: number, any = true): Uint32Array => {
        const cached = any ? closureMemo.get(g) : undefined;
        if (cached) {
            return cached;
        }
        const seen = new Set([g]);
        const queue = [g];
        const visit = (r: ComputeProgramRule, x: number) => {
            if (accepts(r, x, any)) {
                for (const out of outputs(r)) {
                    if (!seen.has(out)) {
                        seen.add(out);
                        queue.push(out);
                    }
                }
            }
        };
        while (queue.length) {
            const x = queue.pop()!;
            for (const r of byGlyph.get(x) ?? []) {
                visit(r, x);
            }
            for (const r of classProducers) {
                visit(r, x);
            }
        }
        const result = Uint32Array.from(seen);
        if (any) {
            closureMemo.set(g, result);
        }
        return result;
    };
    const bStar = closure(space, false);
    if ((gdef && bStar.some((g) => cls(g) !== 0)) || ligatures.some((r) => r.input.slice(1).some((p) => bStar.some((g) => matches(p, g))))) {
        return null;
    }
    const skipped = (g: number) => {
        if (!gdef) {
            return false;
        }
        const c = cls(g);
        return program.lookups.some(
            (l) =>
                l.lookupFlag !== 0 &&
                ((l.ignoreBase && c === 1) ||
                    (l.ignoreLigatures && c === 2) ||
                    (l.ignoreMarks && c === 3) ||
                    (l.markAttachmentClass !== 0 && c === 3 && glyphClass(gdef.markAttachClassDef, g) !== l.markAttachmentClass))
        );
    };
    return { space, cls, closure, bStar, skipped };
}

/** @internal Every non-mark-attachment rule window with a non-final slot that can see B*; null above 4096 crossings.
 * `rules`, when given, also receives each window's identity and B*-slot count (profile proofs only). */
export function enumerateCrossings(program: ComputeTextProgram, bStar: Uint32Array, rules?: ComputeTextCrossingRule[]): CrossingSet | null {
    const crossings: Crossing[] = [];
    const allSlots: Slot[] = [];
    let maxD = 1;
    const inB = new Uint8Array(0x10000);
    for (const g of bStar) {
        inB[g] = 1;
    }
    const seesB = (p: ComputeGlyphPredicate) => {
        if (p.kind === "glyphs") {
            const ids = p.ids;
            if (ids.length === 1) {
                return inB[ids[0]!] === 1;
            }
            for (const g of bStar) {
                if (ids.includes(g)) {
                    return true;
                }
            }
            return false;
        }
        return bStar.some((g) => matches(p, g));
    };
    for (const lookup of program.lookups) {
        for (const r of lookup.rules) {
            if (r.mode === COMPUTE_RULE_MODE_MARK_BASE || r.mode === COMPUTE_RULE_MODE_MARK_MARK || r.mode === COMPUTE_RULE_MODE_MARK_LIGATURE) {
                continue;
            }
            const nb = r.backtrack.length;
            const ni = r.input.length;
            const length = nb + ni + r.lookahead.length;
            let window: ComputeGlyphPredicate[] | null = null;
            for (let k = 0; k + 1 < length; k++) {
                // Most predicates cannot see B*: test them before building the reversed window.
                if (!seesB(k < nb ? r.backtrack[nb - 1 - k]! : k < nb + ni ? r.input[k - nb]! : r.lookahead[k - nb - ni]!)) {
                    continue;
                }
                window ??= [...r.backtrack].reverse().concat(r.input, r.lookahead);
                const slots: Slot[] = [];
                for (let j = 0; j < window.length; j++) {
                    if (j !== k) {
                        const slot = { side: j < k ? -1 : 1, d: Math.abs(j - k), p: window[j]!, bit: allSlots.length };
                        slots.push(slot);
                        allSlots.push(slot);
                        maxD = Math.max(maxD, slot.d);
                    }
                }
                crossings.push({ slots, anchor: slots[0]! });
                rules?.push({ lookup, rule: r, window, slot: k, spaceSlots: window.filter(seesB).length });
                if (crossings.length > 4096) {
                    return null;
                }
            }
        }
    }
    return { crossings, allSlots, maxD, inB };
}

/** @internal Paragraph space proof: the profile's lazy proof for Indic plans, otherwise the generic analysis. */
export function computeTextParagraphSpaceProof(program: ComputeTextProgram): ComputeTextSpaceProof | null {
    return program.plan._profile === "indic" ? (profileSupport("indic")?.paragraph?.spaces(program) ?? null) : analyzeComputeTextSegmentation(program);
}

function analyze(program: ComputeTextProgram): SegmentAnalysis | null {
    const font = program.plan.font;
    // Indic normalization expands/contracts nominal slots before the lookup
    // proof's producer-span induction begins. Space locality alone cannot
    // certify that induction: retain exact whole-run execution until those
    // producers are represented. No limit or existing-script proof is relaxed.
    if (program.plan._profile === "indic") {
        return null;
    }
    if (program.plan.direction !== "ltr") {
        return null;
    }
    const context = spaceContext(program, true);
    if (!context) {
        return null;
    }
    const { space, cls, closure, bStar, skipped } = context;
    let spans = new Map<number, number>();
    const sourceSpan = (p: ComputeGlyphPredicate, before: ReadonlyMap<number, number>) => {
        let value = 1;
        if (p.kind === "glyphs") {
            for (const g of p.ids) {
                value = Math.max(value, before.get(g) ?? 1);
            }
        } else {
            for (const [g, span] of before) {
                if (span > value && matches(p, g)) {
                    value = span;
                }
            }
        }
        return value;
    };
    const mergeSpans = (into: Map<number, number>, from: ReadonlyMap<number, number>) => {
        let changed = false;
        for (const [g, value] of from) {
            if (value > (into.get(g) ?? 1)) {
                into.set(g, value);
                changed = true;
            }
        }
        return changed;
    };
    let span = 1;
    const applyRules = (rules: readonly ComputeProgramRule[], before: ReadonlyMap<number, number>, nested = false, admitGrowing = false): Map<number, number> | null => {
        const after = new Map(before);
        for (const r of rules) {
            if (r.action.kind === "context") {
                if (nested) {
                    return null;
                }
                let branch = new Map(before);
                for (const record of r.action.lookups) {
                    const child = program.lookups.find((l) => l.table === "GSUB" && l.lookupIndex === record.lookupIndex)!;
                    const next = applyRules(child.rules, branch, true, admitGrowing);
                    if (!next) {
                        return null;
                    }
                    branch = next;
                }
                mergeSpans(after, branch);
                continue;
            }
            if (r.action.kind !== "ligature" && r.action.kind !== "substitute" && r.action.kind !== "multiple") {
                continue;
            }
            // The root-local predicate-disjointness proof excludes re-consumption.
            // Compose child spans from the current branch exactly as at top level.
            if (nested && !admitGrowing && (r.action.kind === "ligature" || (r.action.kind === "multiple" && r.action.glyphs.length > 1))) {
                return null;
            }
            const value = r.action.kind === "ligature" ? r.input.reduce((v, p) => v + sourceSpan(p, before), 0) : sourceSpan(r.input[0]!, before);
            if (value > 64) {
                return null;
            }
            for (const g of outputs(r)) {
                if (value > (after.get(g) ?? 1)) {
                    after.set(g, value);
                    span = Math.max(span, value);
                }
            }
        }
        return after;
    };
    for (const index of program.rootOrder) {
        const lookup = program.lookups[index];
        if (!lookup) {
            return null;
        }
        if (lookup.table !== "GSUB") {
            continue;
        }
        const contexts = lookup.rules.filter((r) => r.action.kind === "context");
        const childRules = contexts.flatMap((r) =>
            r.action.kind === "context" ? r.action.lookups.flatMap((record) => program.lookups.find((l) => l.table === "GSUB" && l.lookupIndex === record.lookupIndex)!.rules) : []
        );
        const predicates = [...contexts.flatMap((r) => [...r.backtrack, ...r.input, ...r.lookahead]), ...childRules.flatMap((r) => r.input)];
        const growingOutputs = childRules.flatMap((r) => (r.action.kind === "ligature" || (r.action.kind === "multiple" && r.action.glyphs.length > 1) ? outputs(r) : []));
        const admitGrowing = !growingOutputs.some((glyph) => predicates.some((p) => matches(p, glyph)));
        if (!admitGrowing) {
            return null;
        }
        const after = applyRules(lookup.rules, spans, false, admitGrowing);
        if (!after) {
            return null;
        }
        spans = after;
        if (contexts.length) {
            // Non-leading nested targets may be visited by later context matches
            // in this same root pass. Growing outputs cannot be re-consumed:
            // the root-local proof above makes this a finite, capped closure.
            for (;;) {
                const next = applyRules(contexts, spans, false, admitGrowing);
                if (!next) {
                    return null;
                }
                if (!mergeSpans(spans, next)) {
                    break;
                }
            }
        }
    }
    if (span > 8) {
        return null;
    }
    const set = enumerateCrossings(program, bStar);
    if (!set) {
        return null;
    }
    const { crossings, allSlots, maxD, inB } = set;
    let markBaseHit: boolean | undefined;
    // Only boundaries whose first right glyph can become a mark need the mark-base scan.
    const hitsMarkBase = () =>
        (markBaseHit ??= program.lookups.some((l) =>
            l.rules.some((r) =>
                r.action.kind === "mark-ligature"
                    ? r.action.ligatures.some((b) => inB[b.glyphId] === 1)
                    : (r.action.kind === "mark-base" || r.action.kind === "mark-mark") && r.action.bases.some((b) => inB[b.glyphId] === 1)
            )
        ));
    if (maxD * span > 64) {
        return null;
    }
    const common: number[] = [];
    for (const ch of "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") {
        try {
            common.push(
                ...closure(
                    mapComputeFontScalars(font, Uint32Array.of(ch.codePointAt(0)!), (_i, error) => {
                        throw error;
                    })[0]!
                )
            );
        } catch {
            // Unmapped common glyphs do not participate in the anchor heuristic.
        }
    }
    const countMemo = new Map<ComputeGlyphPredicate, number>();
    const hitsMemo = new Map<ComputeGlyphPredicate, number>();
    const commonCounts = new Map<number, number>();
    for (const g of common) {
        commonCounts.set(g, (commonCounts.get(g) ?? 0) + 1);
    }
    const commonWeights = new Uint32Array(font.glyphCount);
    for (const [g, weight] of commonCounts) {
        commonWeights[g] = weight;
    }
    const scored = new Uint32Array(font.glyphCount);
    let scoreRevision = 0;
    const classHistograms = new Map<object, Map<number, number>>();
    const count = (p: ComputeGlyphPredicate) => {
        let count = countMemo.get(p);
        if (count === undefined) {
            count = p.kind === "glyphs" ? new Set(p.ids).size : 0;
            if (p.kind === "class") {
                const cov = p.coverage;
                if (cov) {
                    // Parsed coverage is strictly ascending, so each glyph is counted once.
                    if (cov.format === 1) {
                        for (const g of cov.glyphs) {
                            if (g < font.glyphCount && glyphClass(p.classDef, g) === p.classId) {
                                count++;
                            }
                        }
                    } else {
                        for (const range of cov.ranges) {
                            for (let g = range.start; g <= range.end && g < font.glyphCount; g++) {
                                if (glyphClass(p.classDef, g) === p.classId) {
                                    count++;
                                }
                            }
                        }
                    }
                } else {
                    const key = p.classDef ?? classHistograms;
                    let histogram = classHistograms.get(key);
                    if (!histogram) {
                        histogram = new Map();
                        for (let g = 0; g < font.glyphCount; g++) {
                            const c = glyphClass(p.classDef, g);
                            histogram.set(c, (histogram.get(c) ?? 0) + 1);
                        }
                        classHistograms.set(key, histogram);
                    }
                    count = histogram.get(p.classId) ?? 0;
                }
            }
            countMemo.set(p, count);
        }
        return count;
    };
    const hits = (p: ComputeGlyphPredicate) => {
        let value = hitsMemo.get(p);
        if (value === undefined) {
            value = 0;
            if (p.kind === "glyphs") {
                scoreRevision++;
                for (const g of p.ids) {
                    if (commonWeights[g] && scored[g] !== scoreRevision) {
                        scored[g] = scoreRevision;
                        value += commonWeights[g]!;
                    }
                }
            } else {
                for (const [g, weight] of commonCounts) {
                    if (matches(p, g)) {
                        value += weight;
                    }
                }
            }
            hitsMemo.set(p, value);
        }
        return value;
    };
    for (const c of crossings) {
        for (const e of c.slots) {
            const a = c.anchor;
            if (e === a) {
                continue;
            }
            if (hits(e.p) < hits(a.p) || (hits(e.p) === hits(a.p) && (count(e.p) < count(a.p) || (count(e.p) === count(a.p) && e.d < a.d)))) {
                c.anchor = e;
            }
        }
    }
    const info = new Map<number, GlyphInfo>();
    const get = (g: number): GlyphInfo => {
        let value = info.get(g);
        if (!value) {
            const c = closure(g);
            const anchorHits: number[] = [];
            for (let i = 0; i < crossings.length; i++) {
                if (c.some((y) => matches(crossings[i]!.anchor.p, y))) {
                    anchorHits.push(i);
                }
            }
            value = { closure: c, free: c.some(skipped), mark: c.some((y) => cls(y) === 3), anchorHits, slots: null };
            info.set(g, value);
        }
        return value;
    };
    const slotMatches = (v: GlyphInfo, e: Slot) => {
        const memo = (v.slots ??= new Uint8Array(allSlots.length));
        let value = memo[e.bit]!;
        if (value === 0) {
            value = v.closure.some((y) => matches(e.p, y)) ? 2 : 1;
            memo[e.bit] = value;
        }
        return value === 2;
    };
    const sideG: GlyphInfo[][] = [[], []];
    const sideN: number[][] = [[], []];
    const touched: number[] = [];
    const walk = (gl: Uint32Array, from: number, dir: number) => {
        const k = dir > 0 ? 1 : 0;
        const glyphs = sideG[k]!;
        const counts = sideN[k]!;
        glyphs.length = 0;
        counts.length = 0;
        let n = 0;
        let steps = 0;
        for (let i = from; i >= 0 && i < gl.length && n < maxD * span; i += dir) {
            const v = get(gl[i]!);
            glyphs.push(v);
            counts.push(n);
            for (const ci of v.anchorHits) {
                const e = crossings[ci]!.anchor;
                if (e.side === dir && n < e.d * span) {
                    touched.push(ci);
                }
            }
            if (!v.free) {
                n++;
            }
            if (++steps > 64) {
                return false;
            }
        }
        return true;
    };
    const slotHit = (e: Slot) => {
        const k = e.side > 0 ? 1 : 0;
        const glyphs = sideG[k]!;
        const counts = sideN[k]!;
        for (let i = 0; i < glyphs.length && counts[i]! < e.d * span; i++) {
            if (slotMatches(glyphs[i]!, e)) {
                return true;
            }
        }
        return false;
    };
    const safeAfter = (gl: Uint32Array, s: number) => {
        if (gl[s] !== space || s + 1 >= gl.length) {
            return false;
        }
        touched.length = 0;
        if (!walk(gl, s + 1, 1) || !walk(gl, s - 1, -1)) {
            return false;
        }
        for (const ci of touched) {
            if (crossings[ci]!.slots.every(slotHit)) {
                return false;
            }
        }
        return !get(gl[s + 1]!).mark || !hitsMarkBase();
    };
    return {
        space,
        span,
        maxD,
        crossings: crossings.length,
        get markBaseHit() {
            return hitsMarkBase();
        },
        safeAfter,
    };
}

// A cached paragraph flag answers exactly the spaces it covers; everything else runs the analysis.
function safeAfterAt(analysis: SegmentAnalysis, run: SegmentRun, flags: Uint8Array | undefined, s: number): boolean {
    if (flags !== undefined && run.scalars[s] === 0x20 && s + 1 < run.scalars.length && run.scalars[s + 1] !== 0x20) {
        return flags[s] === SAFE_SPACE_FLAG;
    }
    return analysis.safeAfter(run.glyphIds, s);
}

/** @internal Segment views in shaped-slot order; null preserves the existing resident path. */
export function segmentComputeTextRuns(
    groups: readonly SegmentGroup[],
    limits: { readonly maxComputeWorkgroupsPerDimension: number; readonly maxStorageBufferBindingSize: number; readonly maxBufferSize: number },
    layout: { readonly headerWords: number; readonly recordWords: number },
    safeSpaces?: SegmentSafeSpaces
): ComputeTextSegments | null {
    if (!groups.some((group) => group.runs.some((run) => run.glyphIds.length > SEGMENT_TARGET))) {
        return null;
    }
    const segmented: SegmentGroup[] = [];
    const first: number[] = [];
    const counts: number[] = [];
    const starts: number[] = [];
    let split = false;
    let capacity = 1;
    for (let g = 0; g < groups.length; g++) {
        const group = groups[g]!;
        const runs: SegmentRun[] = [];
        let analysis: SegmentAnalysis | null | undefined;
        const outputRun = starts.length;
        for (let r = 0; r < group.runs.length; r++) {
            const run = group.runs[r]!;
            const logical = group.outputRun + r;
            first[logical] = starts.length;
            let start = 0;
            const n = run.glyphIds.length;
            const known = safeSpaces?.[g]?.[r];
            const flags = known?.length === n ? known : undefined;
            if (n > SEGMENT_TARGET && analysis === undefined) {
                analysis = analyzeComputeTextSegmentation(compileComputeTextProgram(group.plan));
            }
            const joiningType = group.plan._profile === "arabic" ? profileSupport("arabic")?.joiningType : undefined;
            if (group.plan._profile === "arabic" && !joiningType) {
                throw new Error("segmentComputeTextRuns: Arabic profile is not loaded");
            }
            while (analysis && n - start > SEGMENT_TARGET) {
                let s = start + SEGMENT_TARGET - 1;
                while (s > start && (!safeAfterAt(analysis, run, flags, s) || (joiningType !== undefined && joiningType(run.scalars[s]!) !== 0))) {
                    s--;
                }
                if (s === start) {
                    s = start + SEGMENT_TARGET;
                    while (s < n - 1 && (!safeAfterAt(analysis, run, flags, s) || (joiningType !== undefined && joiningType(run.scalars[s]!) !== 0))) {
                        s++;
                    }
                    if (s === n - 1) {
                        break;
                    }
                }
                runs.push({ scalars: run.scalars.subarray(start, s + 1), glyphIds: run.glyphIds.subarray(start, s + 1) });
                starts.push(start);
                start = s + 1;
                split = true;
            }
            runs.push(start === 0 ? run : { scalars: run.scalars.subarray(start), glyphIds: run.glyphIds.subarray(start) });
            starts.push(start);
            counts[logical] = starts.length - first[logical]!;
        }
        if (runs.length > limits.maxComputeWorkgroupsPerDimension) {
            return null;
        }
        const expansion = compileComputeTextProgram(group.plan).maxExpansion;
        for (const run of runs) {
            capacity = Math.max(capacity, run.scalars.length * expansion);
        }
        segmented.push({ plan: group.plan, runs, outputRun });
    }
    if (!split) {
        return null;
    }
    const outputBytes = starts.length * (layout.headerWords + capacity * layout.recordWords) * 4;
    const tableBytes = (first.length * 2 + starts.length) * 4;
    if ([outputBytes, tableBytes].some((bytes) => !Number.isSafeInteger(bytes) || bytes > limits.maxBufferSize || bytes > limits.maxStorageBufferBindingSize)) {
        return null;
    }
    const table = new Uint32Array(first.length * 2 + starts.length);
    for (let r = 0; r < first.length; r++) {
        table[r * 2] = first[r]!;
        table[r * 2 + 1] = counts[r]!;
    }
    table.set(starts, first.length * 2);
    return { groups: segmented, table, capacity, count: starts.length };
}
