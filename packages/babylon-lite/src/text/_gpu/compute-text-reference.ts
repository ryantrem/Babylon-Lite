/** Legacy test-only CPU interpreter, retained for pre-migration unit tests.
 *
 *  One run, source order, first matching rule, mode 5 fails instead of skipping.
 *  Nested actions are only the direct single substitutions the compiler marked
 *  executable. text-shaper is not imported. The production path dispatches the
 *  WGSL kernel. Corrected expectations come from independent committed evidence,
 *  never from this historical interpreter; it is not a production fallback. */

import { ensureComputeFontLayout, readComputeFontAdvance } from "../compute-font.js";
import type { ComputeGlyphPredicate, ComputeProgramLookup, ComputeProgramRule, ComputeTextProgram } from "../compute-text-program.js";
import {
    COMPUTE_RULE_MODE_ADJUST,
    COMPUTE_RULE_MODE_CLASS_PAIR,
    COMPUTE_RULE_MODE_CONTEXT,
    COMPUTE_RULE_MODE_LIGATURE,
    COMPUTE_RULE_MODE_MARK_BASE,
    COMPUTE_RULE_MODE_MARK_LIGATURE,
    COMPUTE_RULE_MODE_MARK_MARK,
    COMPUTE_RULE_MODE_POSITION,
    COMPUTE_RULE_MODE_MULTIPLE,
    COMPUTE_RULE_MODE_REJECT,
    COMPUTE_RULE_MODE_SUBSTITUTE,
} from "../compute-text-program.js";
import { coverageIndex, glyphClass } from "../font-parser/opentype-layout.js";

export interface ReferenceGlyph {
    readonly glyphId: number;
    readonly cluster: number;
    readonly xAdvance: number;
    readonly yAdvance: number;
    readonly xOffset: number;
    readonly yOffset: number;
}

interface LiveGlyph {
    id: number;
    cluster: number;
    xAdvance: number;
    yAdvance: number;
    xOffset: number;
    yOffset: number;
    deleted: boolean;
}

export function executeComputeTextReference(program: ComputeTextProgram, glyphIds: readonly number[] | Uint32Array, clusters: readonly number[] | Uint32Array): ReferenceGlyph[] {
    const font = program.plan.font;
    const gdef = ensureComputeFontLayout(font).gdef;
    const glyphs: LiveGlyph[] = Array.from(glyphIds, (id, index) => ({
        id,
        cluster: clusters[index] ?? index,
        xAdvance: 0,
        yAdvance: 0,
        xOffset: 0,
        yOffset: 0,
        deleted: false,
    }));
    let sawGpos = false;
    for (const index of program.rootOrder) {
        const lookup = program.lookups[index];
        if (!lookup) {
            continue;
        }
        if (lookup.table === "GPOS" && !sawGpos) {
            compact(glyphs);
            assignAdvances(glyphs, font);
            sawGpos = true;
        }
        if (lookup.table === "GSUB") {
            applyGsub(program, lookup, glyphs, gdef);
            compact(glyphs);
        } else {
            applyGpos(lookup, glyphs, gdef);
        }
    }
    if (!sawGpos) {
        compact(glyphs);
        assignAdvances(glyphs, font);
    }
    const live = glyphs.filter((glyph) => !glyph.deleted);
    if (program.plan.direction === "rtl") {
        live.reverse();
    }
    return live.map((glyph) => ({
        glyphId: glyph.id,
        cluster: glyph.cluster,
        xAdvance: glyph.xAdvance,
        yAdvance: glyph.yAdvance,
        xOffset: glyph.xOffset,
        yOffset: glyph.yOffset,
    }));
}

function applyGsub(program: ComputeTextProgram, lookup: ComputeProgramLookup, glyphs: LiveGlyph[], gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): void {
    for (let index = 0; index < glyphs.length; index++) {
        if (unavailable(glyphs, index, lookup, gdef)) {
            continue;
        }
        for (const rule of lookup.rules) {
            if (rule.mode === COMPUTE_RULE_MODE_LIGATURE && rule.action.kind === "ligature") {
                const matched = matchForward(glyphs, index, rule.input, lookup, gdef);
                if (!matched) {
                    continue;
                }
                const first = glyphs[matched[0] ?? index];
                if (!first) {
                    break;
                }
                let cluster = first.cluster;
                for (const at of matched) {
                    cluster = Math.min(cluster, glyphs[at]?.cluster ?? cluster);
                }
                first.id = rule.action.glyph;
                first.cluster = cluster;
                for (let consumed = 1; consumed < matched.length; consumed++) {
                    const glyph = glyphs[matched[consumed] ?? -1];
                    if (glyph) {
                        glyph.deleted = true;
                    }
                }
                break;
            }
            if (rule.mode === COMPUTE_RULE_MODE_MULTIPLE && rule.action.kind === "multiple") {
                if (!matches(rule.input[0], glyphs[index]?.id ?? -1) || rule.action.glyphs.length === 0) {
                    continue;
                }
                const glyph = glyphs[index];
                if (!glyph) {
                    break;
                }
                glyph.id = rule.action.glyphs[0] ?? glyph.id;
                const inserted = rule.action.glyphs.slice(1).map((id) => ({
                    id,
                    cluster: glyph.cluster,
                    xAdvance: 0,
                    yAdvance: 0,
                    xOffset: 0,
                    yOffset: 0,
                    deleted: false,
                }));
                glyphs.splice(index + 1, 0, ...inserted);
                index += inserted.length;
                break;
            }
            if (rule.mode === COMPUTE_RULE_MODE_SUBSTITUTE && rule.action.kind === "substitute") {
                if (!matches(rule.input[0], glyphs[index]?.id ?? -1)) {
                    continue;
                }
                const glyph = glyphs[index];
                if (glyph) {
                    glyph.id = rule.action.glyphs[0] ?? glyph.id;
                }
                break;
            }
            if (rule.mode === COMPUTE_RULE_MODE_CONTEXT && rule.action.kind === "context") {
                if (!matchContext(glyphs, index, rule, lookup, gdef)) {
                    continue;
                }
                for (const nested of rule.action.lookups) {
                    const at = index + nested.sequenceIndex;
                    const child = program.lookups.find((entry) => entry.table === lookup.table && entry.lookupIndex === nested.lookupIndex);
                    if (!child || at < 0 || at >= glyphs.length || glyphs[at]?.deleted) {
                        continue;
                    }
                    applyDirectSingle(child, glyphs[at]!, at);
                }
                break;
            }
            if (rule.mode === COMPUTE_RULE_MODE_REJECT && matchContext(glyphs, index, rule, lookup, gdef)) {
                fail(lookup, index);
            }
        }
    }
}

function gposRejectMatches(
    glyphs: readonly LiveGlyph[],
    index: number,
    second: number,
    rule: ComputeProgramRule,
    lookup: ComputeProgramLookup,
    gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]
): boolean {
    if (rule.backtrack.length > 0 || rule.lookahead.length > 0) {
        return matchContext(glyphs, index, rule, lookup, gdef);
    }
    return matches(rule.input[0], glyphs[index]?.id ?? -1) && (rule.input.length < 2 || (second >= 0 && matches(rule.input[1], glyphs[second]?.id ?? -1)));
}

function applyDirectSingle(lookup: ComputeProgramLookup, glyph: LiveGlyph, at: number): void {
    for (const rule of lookup.rules) {
        if (!matches(rule.input[0], glyph.id)) {
            continue;
        }
        if (rule.mode === COMPUTE_RULE_MODE_REJECT || rule.action.kind === "reject") {
            fail(lookup, at);
        }
        if (rule.action.kind === "substitute") {
            glyph.id = rule.action.glyphs[0] ?? glyph.id;
        }
        return;
    }
}

function applyGpos(lookup: ComputeProgramLookup, glyphs: LiveGlyph[], gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): void {
    if (lookup.rules.some((rule) => rule.mode === COMPUTE_RULE_MODE_MARK_BASE || rule.mode === COMPUTE_RULE_MODE_MARK_MARK)) {
        applyMarks(lookup, glyphs, gdef);
        return;
    }
    if (lookup.rules.some((rule) => rule.mode === COMPUTE_RULE_MODE_MARK_LIGATURE)) {
        applyMarkLigatures(lookup, glyphs, gdef);
        return;
    }
    for (let index = 0; index < glyphs.length; index++) {
        if (unavailable(glyphs, index, lookup, gdef)) {
            continue;
        }
        const second = pairSecond(glyphs, index, lookup, gdef);
        for (const rule of lookup.rules) {
            if (rule.mode === COMPUTE_RULE_MODE_REJECT && gposRejectMatches(glyphs, index, second, rule, lookup, gdef)) {
                fail(lookup, index);
            }
            if (
                (rule.mode === COMPUTE_RULE_MODE_POSITION || rule.mode === COMPUTE_RULE_MODE_CLASS_PAIR) &&
                second >= 0 &&
                matchPair(rule, glyphs[index]?.id ?? -1, glyphs[second]?.id ?? -1)
            ) {
                const action = rule.action;
                if (action.kind === "pair" || action.kind === "class-pair") {
                    const first = glyphs[index];
                    const other = glyphs[second];
                    if (first) {
                        first.xAdvance += action.first.xAdvance;
                    }
                    if (other) {
                        other.xAdvance += action.second.xAdvance;
                    }
                }
                break;
            }
            if (rule.mode === COMPUTE_RULE_MODE_ADJUST && rule.action.kind === "adjust" && matches(rule.input[0], glyphs[index]?.id ?? -1)) {
                const glyph = glyphs[index];
                if (glyph) {
                    glyph.xOffset += rule.action.xPlacement;
                    glyph.yOffset += rule.action.yPlacement;
                    glyph.xAdvance += rule.action.xAdvance;
                    glyph.yAdvance += rule.action.yAdvance;
                }
                break;
            }
        }
    }
}

function applyMarkLigatures(lookup: ComputeProgramLookup, glyphs: LiveGlyph[], gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): void {
    for (let index = 0; index < glyphs.length; index++) {
        const glyph = glyphs[index];
        if (!glyph || glyph.deleted || glyphClass(gdef?.glyphClassDef ?? null, glyph.id) !== 3) {
            continue;
        }
        for (const rule of lookup.rules) {
            if (rule.mode !== COMPUTE_RULE_MODE_MARK_LIGATURE || rule.action.kind !== "mark-ligature" || !matches(rule.input[0], glyph.id)) {
                continue;
            }
            const ligature = previousAttachment(glyphs, index, gdef);
            const base = glyphs[ligature];
            if (!base || glyphClass(gdef?.glyphClassDef ?? null, base.id) !== 2) {
                break;
            }
            let between = 0;
            for (let cursor = ligature + 1; cursor < index; cursor++) {
                const betweenGlyph = glyphs[cursor];
                if (betweenGlyph && !betweenGlyph.deleted && glyphClass(gdef?.glyphClassDef ?? null, betweenGlyph.id) === 3) {
                    between++;
                }
            }
            const record = rule.action.ligatures.find((item) => item.glyphId === base.id);
            if (!record || record.componentCount === 0) {
                break;
            }
            const component = Math.min(between, record.componentCount - 1);
            const anchor = record.anchors.find((item) => item.component === component);
            if (!anchor) {
                break;
            }
            if (anchor.reject) {
                fail(lookup, index);
            }
            glyph.xOffset = anchor.x - rule.action.markX + base.xOffset;
            glyph.yOffset = anchor.y - rule.action.markY + base.yOffset;
            glyph.xAdvance = 0;
            glyph.yAdvance = 0;
            break;
        }
    }
}

function previousAttachment(glyphs: readonly LiveGlyph[], index: number, gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): number {
    let found = -1;
    for (let cursor = 0; cursor < index; cursor++) {
        const glyph = glyphs[cursor];
        if (!glyph || glyph.deleted) {
            continue;
        }
        const cls = glyphClass(gdef?.glyphClassDef ?? null, glyph.id);
        if (cls === 0 || cls === 1 || cls === 2) {
            found = cursor;
        }
    }
    return found;
}

function applyMarks(lookup: ComputeProgramLookup, glyphs: LiveGlyph[], gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): void {
    const attach = previousBase(glyphs, gdef);
    for (let index = 0; index < glyphs.length; index++) {
        const glyph = glyphs[index];
        if (!glyph || glyph.deleted || glyphClass(gdef?.glyphClassDef ?? null, glyph.id) !== 3) {
            continue;
        }
        for (const rule of lookup.rules) {
            if (!matches(rule.input[0], glyph.id)) {
                continue;
            }
            if (rule.mode === COMPUTE_RULE_MODE_REJECT || rule.action.kind === "reject") {
                fail(lookup, index);
            }
            if (rule.action.kind !== "mark-base" && rule.action.kind !== "mark-mark") {
                break;
            }
            const baseIndex = rule.mode === COMPUTE_RULE_MODE_MARK_MARK ? previousMark(glyphs, index, gdef) : (attach[index] ?? -1);
            const base = glyphs[baseIndex];
            const anchor = base ? rule.action.bases.find((item) => item.glyphId === base.id) : undefined;
            if (!base || !anchor) {
                continue;
            }
            glyph.xOffset = anchor.x - rule.action.markX + base.xOffset;
            glyph.yOffset = anchor.y - rule.action.markY + base.yOffset;
            if (rule.mode === COMPUTE_RULE_MODE_MARK_BASE) {
                glyph.xAdvance = 0;
                glyph.yAdvance = 0;
            }
            break;
        }
    }
}

function previousBase(glyphs: readonly LiveGlyph[], gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): number[] {
    const attach = glyphs.map(() => -1);
    let remembered = -1;
    glyphs.forEach((glyph, index) => {
        const cls = glyphClass(gdef?.glyphClassDef ?? null, glyph.id);
        if (cls === 3) {
            attach[index] = remembered;
        } else {
            remembered = index;
        }
    });
    return attach;
}

function previousMark(glyphs: readonly LiveGlyph[], index: number, gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): number {
    const previous = glyphs[index - 1];
    return previous && glyphClass(gdef?.glyphClassDef ?? null, previous.id) === 3 ? index - 1 : -1;
}

function pairSecond(glyphs: readonly LiveGlyph[], index: number, lookup: ComputeProgramLookup, gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): number {
    if ((lookup.lookupFlag === 0 || !gdef) && index + 1 < glyphs.length) {
        return index + 1;
    }
    return nextLive(glyphs, index + 1, lookup, gdef, 1);
}

function matchPair(rule: ComputeProgramRule, first: number, second: number): boolean {
    return matches(rule.input[0], first) && matches(rule.input[1], second);
}

function matchContext(
    glyphs: readonly LiveGlyph[],
    index: number,
    rule: ComputeProgramRule,
    lookup: ComputeProgramLookup,
    gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]
): boolean {
    const input = matchForward(glyphs, index, rule.input, lookup, gdef);
    if (!input) {
        return false;
    }
    if (!matchBackward(glyphs, index - 1, rule.backtrack, lookup, gdef)) {
        return false;
    }
    const after = nextLive(glyphs, (input[input.length - 1] ?? index) + 1, lookup, gdef, 1);
    return matchForward(glyphs, after, rule.lookahead, lookup, gdef) !== null || rule.lookahead.length === 0;
}

function matchForward(
    glyphs: readonly LiveGlyph[],
    start: number,
    predicates: readonly ComputeGlyphPredicate[],
    lookup: ComputeProgramLookup,
    gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]
): number[] | null {
    if (predicates.length === 0) {
        return [];
    }
    if (start < 0) {
        return null;
    }
    const matched: number[] = [];
    let cursor = start;
    for (let index = 0; index < predicates.length; index++) {
        if (index > 0) {
            cursor = nextLive(glyphs, cursor + 1, lookup, gdef, 1);
        }
        const glyph = glyphs[cursor];
        if (!glyph || glyph.deleted || !matches(predicates[index], glyph.id)) {
            return null;
        }
        matched.push(cursor);
    }
    return matched;
}

function matchBackward(
    glyphs: readonly LiveGlyph[],
    start: number,
    predicates: readonly ComputeGlyphPredicate[],
    lookup: ComputeProgramLookup,
    gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]
): boolean {
    let cursor = start;
    for (const predicate of predicates) {
        cursor = nextLive(glyphs, cursor, lookup, gdef, -1);
        const glyph = glyphs[cursor];
        if (!glyph || !matches(predicate, glyph.id)) {
            return false;
        }
        cursor -= 1;
    }
    return true;
}

function nextLive(glyphs: readonly LiveGlyph[], start: number, lookup: ComputeProgramLookup, gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"], direction: 1 | -1): number {
    for (let index = start; index >= 0 && index < glyphs.length; index += direction) {
        const glyph = glyphs[index];
        if (glyph && !glyph.deleted && !skipped(lookup, glyph.id, gdef)) {
            return index;
        }
    }
    return -1;
}

function unavailable(glyphs: readonly LiveGlyph[], index: number, lookup: ComputeProgramLookup, gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): boolean {
    const glyph = glyphs[index];
    return !glyph || glyph.deleted || skipped(lookup, glyph.id, gdef);
}

function skipped(lookup: ComputeProgramLookup, glyphId: number, gdef: ReturnType<typeof ensureComputeFontLayout>["gdef"]): boolean {
    if (lookup.lookupFlag === 0 || !gdef) {
        return false;
    }
    const cls = glyphClass(gdef.glyphClassDef, glyphId);
    if (lookup.ignoreBase && cls === 1) {
        return true;
    }
    if (lookup.ignoreLigatures && cls === 2) {
        return true;
    }
    if (lookup.ignoreMarks && cls === 3) {
        return true;
    }
    return lookup.markAttachmentClass !== 0 && cls === 3 && glyphClass(gdef.markAttachClassDef, glyphId) !== lookup.markAttachmentClass;
}

function matches(predicate: ComputeGlyphPredicate | undefined, glyphId: number): boolean {
    if (!predicate) {
        return false;
    }
    if (predicate.kind === "glyphs") {
        return predicate.ids.includes(glyphId);
    }
    if (predicate.coverage && coverageIndex(predicate.coverage, glyphId) === null) {
        return false;
    }
    return glyphClass(predicate.classDef, glyphId) === predicate.classId;
}

function assignAdvances(glyphs: readonly LiveGlyph[], font: ComputeTextProgram["plan"]["font"]): void {
    for (const glyph of glyphs) {
        if (glyph.deleted) {
            continue;
        }
        glyph.xAdvance = readComputeFontAdvance(font, glyph.id);
        glyph.yAdvance = 0;
        glyph.xOffset = 0;
        glyph.yOffset = 0;
    }
}

function compact(glyphs: LiveGlyph[]): void {
    let write = 0;
    for (const glyph of glyphs) {
        if (!glyph.deleted) {
            glyphs[write] = glyph;
            write++;
        }
    }
    glyphs.length = write;
}

function fail(lookup: ComputeProgramLookup, index: number): never {
    throw new Error(`shapeComputeText: unsupported ${lookup.table} lookup ${lookup.lookupIndex} matched at glyph ${index}`);
}
