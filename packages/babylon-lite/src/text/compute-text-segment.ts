/** Conservative independence proof for boundaries after a nominal space.
 *
 * B* stays class 0, never skipped/deleted or consumed as a ligature tail. Every
 * cross-boundary lookup therefore has a B*-accepting window. Closures and the
 * d*S non-free nominal range overapproximate every slot, including skipped and
 * ligature-interior glyphs. Excluding all such windows proves independence by
 * induction over lookups. A non-mark first right glyph prevents cross-space
 * attachment; rejection stays local and one expanding lookup bounds overflow.
 */

import { ensureComputeFontLayout, mapComputeFontScalars } from "./compute-font.js";
import type { ComputeTextPlan } from "./compute-text-plan.js";
import type { ComputeGlyphPredicate, ComputeProgramRule, ComputeTextProgram } from "./compute-text-program.js";
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

/** @internal */
export interface SegmentAnalysis {
    readonly space: number;
    readonly span: number;
    readonly crossings: number;
    readonly markBaseHit: boolean;
    readonly safeAfter: (glyphIds: Uint32Array, spaceIndex: number) => boolean;
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

interface Slot {
    side: number;
    d: number;
    p: ComputeGlyphPredicate;
    bit: number;
}

interface Crossing {
    slots: Slot[];
    anchor: Slot;
}

interface GlyphInfo {
    closure: Uint32Array;
    free: boolean;
    mark: boolean;
    anchorHits: number[];
    full: Uint32Array | null;
}

let analyses: WeakMap<ComputeTextProgram, SegmentAnalysis | null> | null = null;
let predicateIds: WeakMap<ComputeGlyphPredicate, Set<number>> | null = null;

/** @internal Clear only segmentation caches, for independent analysis measurements. */
export function clearComputeTextSegmentationCache(): void {
    analyses = null;
    predicateIds = null;
}

function matches(p: ComputeGlyphPredicate, g: number): boolean {
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

function analyze(program: ComputeTextProgram): SegmentAnalysis | null {
    const font = program.plan.font;
    if (program.plan.direction !== "ltr") {
        return null;
    }
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
        if (lookup.table === "GSUB" && lookup.rules.some((r) => r.action.kind === "multiple" && r.action.glyphs.length > 1) && ++expanding > 1) {
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
    const outputs = (r: ComputeProgramRule): readonly number[] => {
        const action = r.action;
        return action.kind === "ligature" ? [action.glyph] : action.kind === "multiple" || action.kind === "substitute" ? action.glyphs : [];
    };
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
    const spans = new Map<number, number>();
    const pending = [...ligatures];
    const queued = new Set(pending);
    const sourceSpan = (p: ComputeGlyphPredicate) => {
        let value = 1;
        if (p.kind === "glyphs") {
            for (const g of p.ids) {
                value = Math.max(value, spans.get(g) ?? 1);
            }
        } else {
            for (const [g, span] of spans) {
                if (span > value && matches(p, g)) {
                    value = span;
                }
            }
        }
        return value;
    };
    const enqueue = (r: ComputeProgramRule, g: number) => {
        if (!queued.has(r) && accepts(r, g, true)) {
            queued.add(r);
            pending.push(r);
        }
    };
    let span = 1;
    while (pending.length) {
        const r = pending.pop()!;
        queued.delete(r);
        const value = r.action.kind === "ligature" ? r.input.reduce((v, p) => v + sourceSpan(p), 0) : sourceSpan(r.input[0]!);
        if (value > 64) {
            return null;
        }
        for (const g of outputs(r)) {
            if (value > (spans.get(g) ?? 1)) {
                spans.set(g, value);
                span = Math.max(span, value);
                for (const next of byGlyph.get(g) ?? []) {
                    enqueue(next, g);
                }
                for (const next of classProducers) {
                    enqueue(next, g);
                }
            }
        }
    }
    if (span > 8) {
        return null;
    }
    const crossings: Crossing[] = [];
    const allSlots: Slot[] = [];
    let maxD = 1;
    let markBaseHit = false;
    for (const lookup of program.lookups) {
        for (const r of lookup.rules) {
            if (r.mode === COMPUTE_RULE_MODE_MARK_BASE || r.mode === COMPUTE_RULE_MODE_MARK_MARK) {
                if ((r.action.kind === "mark-base" || r.action.kind === "mark-mark") && r.action.bases.some((b) => bStar.includes(b.glyphId))) {
                    markBaseHit = true;
                }
                continue;
            }
            if (r.mode === COMPUTE_RULE_MODE_MARK_LIGATURE) {
                continue;
            }
            const window = [...r.backtrack].reverse().concat(r.input, r.lookahead);
            for (let k = 0; k + 1 < window.length; k++) {
                const p = window[k]!;
                // Most predicates cannot see B*: avoid constructing their membership sets.
                if (!bStar.some((g) => (p.kind === "glyphs" ? p.ids.includes(g) : matches(p, g)))) {
                    continue;
                }
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
                if (crossings.length > 4096) {
                    return null;
                }
            }
        }
    }
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
    const count = (p: ComputeGlyphPredicate) => {
        let count = countMemo.get(p);
        if (count === undefined) {
            count = p.kind === "glyphs" ? new Set(p.ids).size : 0;
            if (p.kind === "class") {
                for (let g = 0; g < font.glyphCount; g++) {
                    if (matches(p, g)) {
                        count++;
                    }
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
            value = { closure: c, free: c.some(skipped), mark: c.some((y) => cls(y) === 3), anchorHits, full: null };
            info.set(g, value);
        }
        return value;
    };
    const full = (v: GlyphInfo) => {
        if (!v.full) {
            v.full = new Uint32Array(Math.ceil(allSlots.length / 32) || 1);
            for (const e of allSlots) {
                if (v.closure.some((y) => matches(e.p, y))) {
                    v.full[e.bit >> 5]! |= 1 << (e.bit & 31);
                }
            }
        }
        return v.full;
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
            if ((full(glyphs[i]!)[e.bit >> 5]! >>> (e.bit & 31)) & 1) {
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
        return !markBaseHit || !get(gl[s + 1]!).mark;
    };
    return { space, span, crossings: crossings.length, markBaseHit, safeAfter };
}

/** @internal Segment views in shaped-slot order; null preserves the existing resident path. */
export function segmentComputeTextRuns(
    groups: readonly SegmentGroup[],
    limits: { readonly maxComputeWorkgroupsPerDimension: number; readonly maxStorageBufferBindingSize: number; readonly maxBufferSize: number },
    layout: { readonly headerWords: number; readonly recordWords: number }
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
    for (const group of groups) {
        const runs: SegmentRun[] = [];
        let analysis: SegmentAnalysis | null | undefined;
        const outputRun = starts.length;
        for (let r = 0; r < group.runs.length; r++) {
            const run = group.runs[r]!;
            const logical = group.outputRun + r;
            first[logical] = starts.length;
            let start = 0;
            const n = run.glyphIds.length;
            if (n > SEGMENT_TARGET && analysis === undefined) {
                analysis = analyzeComputeTextSegmentation(compileComputeTextProgram(group.plan));
            }
            while (analysis && n - start > SEGMENT_TARGET) {
                let s = start + SEGMENT_TARGET - 1;
                while (s > start && !analysis.safeAfter(run.glyphIds, s)) {
                    s--;
                }
                if (s === start) {
                    s = start + SEGMENT_TARGET;
                    while (s < n - 1 && !analysis.safeAfter(run.glyphIds, s)) {
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
