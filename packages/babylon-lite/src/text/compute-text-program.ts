/** Checked lookup program for a selected compute text plan.
 *
 *  Packed words keep the lab's eight-word lookup descriptors and sixteen-word rule
 *  descriptors. Mode 5 is a fail-if-matched rejection, never a successful skip.
 *  Mode 6 is the lab's HarfBuzz nested-cardinality policy and is never emitted.
 *  A VariationIndex whose default-coordinate delta is zero is ignored and the design
 *  value is used. A nonzero default delta stays a rejection; it is not evaluated into
 *  the action. ppem devices are not evaluated: text-shaper 0.1.28 applies the design
 *  value and this plan has no ppem. Contour anchors stay rejected. UseMarkFilteringSet
 *  is recognized and ignored, matching text-shaper 0.1.28 rather than a HarfBuzz filter.
 *
 *  Lookup descriptor words, at `descriptorOffset + fontLookupIndex * 8`:
 *    0 kind, 1 flag, 2 cooperative capability, 3 rule origin, 4 rule count,
 *    5 expansion, 6 candidate origin, 7 ordered eligibility (low bits: 0 serial, 1 tiled, 2 bounded insertion; bit 2: forward-only GSUB).
 *  Rule words, 16 each:
 *    0 input origin, 1 input count, 2 backtrack origin, 3 backtrack count,
 *    4 lookahead origin, 5 lookahead count, 6 action origin, 7 action length,
 *    8 mode, 9 reason code, 10 class-pair row, 11 second class table,
 *    12 row length, 13 row candidate count, 14-15 zero.
 *  A predicate word with bit 31 set is a class id in the low 16 bits, followed by a
 *  class-definition index and a shared coverage origin (`count, ...glyph ids`). A zero
 *  origin means the class id alone. Otherwise the predicate is a glyph count followed by
 *  glyph ids. Mode 9 is a ligature. Mode 10 is a single positioning adjustment.
 *  Mode 11 is a GSUB multiple substitution. The matched glyph becomes the first
 *  output and the rest are inserted after it with the same cluster. The lookup
 *  cursor then advances past those inserted glyphs, matching text-shaper 0.1.28.
 */

import type { ComputeTextPlan } from "./compute-text-plan.js";
import { ensureComputeFontLayout } from "./compute-font.js";
import { createDefaultVariationDelta } from "./font-parser/device-delta.js";
import type { ParsedGlyphPredicate, ParsedLookupActions, ParsedSubtable, ParsedValue, VariationDeviceIndex } from "./font-parser/layout-actions.js";
import { parseLookupActions } from "./font-parser/layout-actions.js";
import type { GlyphClassDef, GlyphCoverage } from "./font-parser/opentype-layout.js";
import { coverageGlyphs, glyphClass } from "./font-parser/opentype-layout.js";

export const COMPUTE_PROGRAM_LOOKUP_WORDS = 8;
export const COMPUTE_PROGRAM_RULE_WORDS = 16;
export const COMPUTE_LOOKUP_KIND = 0;
export const COMPUTE_LOOKUP_FLAG = 1;
export const COMPUTE_LOOKUP_MEMBERSHIP = 2;
export const COMPUTE_LOOKUP_RULE_ORIGIN = 3;
export const COMPUTE_LOOKUP_RULE_COUNT = 4;
export const COMPUTE_LOOKUP_EXPANSION = 5;
export const COMPUTE_LOOKUP_CANDIDATES = 6;
export const COMPUTE_LOOKUP_ORDERED = 7;
export const COMPUTE_RULE_INPUT = 0;
export const COMPUTE_RULE_INPUT_LENGTH = 1;
export const COMPUTE_RULE_BACKTRACK = 2;
export const COMPUTE_RULE_BACKTRACK_LENGTH = 3;
export const COMPUTE_RULE_LOOKAHEAD = 4;
export const COMPUTE_RULE_LOOKAHEAD_LENGTH = 5;
export const COMPUTE_RULE_ACTION = 6;
export const COMPUTE_RULE_ACTION_LENGTH = 7;
export const COMPUTE_RULE_MODE = 8;
export const COMPUTE_RULE_MODE_SUBSTITUTE = 1;
export const COMPUTE_RULE_MODE_CONTEXT = 2;
export const COMPUTE_RULE_MODE_POSITION = 3;
export const COMPUTE_RULE_MODE_CLASS_PAIR = 4;
export const COMPUTE_RULE_MODE_REJECT = 5;
export const COMPUTE_RULE_MODE_MARK_MARK = 7;
export const COMPUTE_RULE_MODE_MARK_BASE = 8;
export const COMPUTE_RULE_MODE_LIGATURE = 9;
export const COMPUTE_RULE_MODE_ADJUST = 10;
export const COMPUTE_RULE_MODE_MULTIPLE = 11;
export const COMPUTE_RULE_MODE_MARK_LIGATURE = 12;

const RESERVED_FLAG = 0x00e0;
const RIGHT_TO_LEFT = 0x0001;
const IGNORE_BASE = 0x0002;
const IGNORE_LIGATURES = 0x0004;
const IGNORE_MARKS = 0x0008;
const USE_MARK_FILTERING_SET = 0x0010;

export interface ComputeGlyphPredicate {
    readonly kind: "glyphs" | "class";
    readonly ids: readonly number[];
    readonly classId: number;
    readonly classDef: GlyphClassDef | null;
    readonly coverage: GlyphCoverage | null;
}

export type ComputeProgramAction =
    | { readonly kind: "substitute"; readonly glyphs: readonly number[] }
    | { readonly kind: "multiple"; readonly glyphs: readonly number[] }
    | { readonly kind: "ligature"; readonly components: readonly number[]; readonly glyph: number }
    | { readonly kind: "context"; readonly lookups: readonly { readonly sequenceIndex: number; readonly lookupIndex: number }[] }
    | { readonly kind: "pair"; readonly secondGlyph: number; readonly firstXAdvance: number; readonly secondXAdvance: number; readonly applySecond: boolean }
    | {
          readonly kind: "class-pair";
          readonly class1: number;
          readonly class2: number;
          readonly firstXAdvance: number;
          readonly secondXAdvance: number;
          readonly applySecond: boolean;
      }
    | { readonly kind: "adjust"; readonly xPlacement: number; readonly yPlacement: number; readonly xAdvance: number; readonly yAdvance: number }
    | {
          readonly kind: "mark-ligature";
          readonly markX: number;
          readonly markY: number;
          readonly ligatures: readonly {
              readonly glyphId: number;
              readonly componentCount: number;
              readonly anchors: readonly { readonly component: number; readonly x: number; readonly y: number; readonly reject: boolean }[];
          }[];
      }
    | {
          readonly kind: "mark-base" | "mark-mark";
          readonly markClass: number;
          readonly markX: number;
          readonly markY: number;
          readonly bases: readonly { readonly glyphId: number; readonly x: number; readonly y: number }[];
          readonly clearAdvance: boolean;
      }
    | { readonly kind: "reject"; readonly reason: string };

export interface ComputeProgramRule {
    readonly input: readonly ComputeGlyphPredicate[];
    readonly backtrack: readonly ComputeGlyphPredicate[];
    readonly lookahead: readonly ComputeGlyphPredicate[];
    readonly action: ComputeProgramAction;
    readonly mode: number;
}

export interface ComputeProgramLookup {
    readonly table: "GSUB" | "GPOS";
    readonly lookupIndex: number;
    readonly lookupType: number;
    readonly lookupFlag: number;
    readonly ignoreBase: boolean;
    readonly ignoreLigatures: boolean;
    readonly ignoreMarks: boolean;
    readonly markAttachmentClass: number;
    /** text-shaper 0.1.28 does not apply this flag. It is not a successful filter. */
    readonly markFilteringSetIgnored: boolean;
    readonly rules: readonly ComputeProgramRule[];
    readonly expansion: number;
}

export interface ComputeProgramRejection {
    readonly table: "GSUB" | "GPOS";
    readonly lookupIndex: number;
    readonly lookupType: number;
    readonly reason: string;
}

export interface ComputeTextProgram {
    readonly plan: ComputeTextPlan;
    readonly lookups: readonly ComputeProgramLookup[];
    readonly rootOrder: readonly number[];
    readonly maxExpansion: number;
    readonly maxPatternDepth: number;
    readonly maxDependencyDepth: number;
    readonly maxAbsAdjustment: number;
    readonly rejections: readonly ComputeProgramRejection[];
    readonly fullyExecutable: boolean;
    readonly words: Uint32Array;
    /** Necessary non-leading GSUB predicate witnesses, one first-32-root mask per font glyph. */
    readonly rootWitnesses: Uint32Array;
    readonly gsubDescriptorOffset: number;
    readonly gposDescriptorOffset: number;
    /** Class definitions addressed by packed class predicates, in index order. */
    readonly classTables: readonly GlyphClassDef[];
}

let programs: WeakMap<ComputeTextPlan, ComputeTextProgram> | null = null;

export function compileComputeTextProgram(plan: ComputeTextPlan): ComputeTextProgram {
    const existing = programs?.get(plan);
    if (existing) {
        return existing;
    }
    ensureComputeFontLayout(plan.font);
    const parsed = new Map<string, ParsedLookupActions>();
    for (const index of [...plan._gsubRoots, ...plan._gsubDependencies]) {
        parsed.set(`GSUB/${index}`, parseLookupActions(plan.font, "GSUB", index));
    }
    for (const index of [...plan._gposRoots, ...plan._gposDependencies]) {
        parsed.set(`GPOS/${index}`, parseLookupActions(plan.font, "GPOS", index));
    }
    const order = [
        ...plan._gsubRoots.map((index) => `GSUB/${index}`),
        ...plan._gsubDependencies.map((index) => `GSUB/${index}`),
        ...plan._gposRoots.map((index) => `GPOS/${index}`),
        ...plan._gposDependencies.map((index) => `GPOS/${index}`),
    ];
    const indexOf = new Map<string, number>();
    order.forEach((key, index) => indexOf.set(key, index));
    const deltaOf = createDefaultVariationDelta(plan.font);
    const lookups = order.map((key) => compileLookup(parsed.get(key) as ParsedLookupActions, plan, parsed, deltaOf));
    const rootOrder = [...plan._gsubRoots, ...plan._gposRoots].map((index, at) => indexOf.get(`${at < plan._gsubRoots.length ? "GSUB" : "GPOS"}/${index}`) ?? 0);
    const rejections = lookups.flatMap((lookup) =>
        lookup.rules
            .filter((rule) => rule.action.kind === "reject")
            .map((rule) => ({
                table: lookup.table,
                lookupIndex: lookup.lookupIndex,
                lookupType: lookup.lookupType,
                reason: rule.action.kind === "reject" ? rule.action.reason : "",
            }))
    );
    const maxExpansion = Math.max(1, ...lookups.map((lookup) => lookup.expansion));
    const maxPatternDepth = Math.max(1, ...lookups.flatMap((lookup) => lookup.rules.map((rule) => Math.max(rule.input.length, rule.backtrack.length, rule.lookahead.length))));
    const maxDependencyDepth = lookups.some((lookup) => lookup.rules.some((rule) => rule.action.kind === "context")) ? 1 : 0;
    const maxAbsAdjustment = Math.max(0, ...lookups.flatMap((lookup) => lookup.rules.map(adjustmentOf)));
    const packed = packProgram(plan, lookups, rootOrder);
    const program: ComputeTextProgram = {
        plan,
        lookups,
        rootOrder,
        maxExpansion,
        maxPatternDepth,
        maxDependencyDepth,
        maxAbsAdjustment,
        rejections,
        fullyExecutable: rejections.length === 0,
        words: Uint32Array.from(packed.words),
        rootWitnesses: packed.rootWitnesses,
        gsubDescriptorOffset: packed.gsubDescriptorOffset,
        gposDescriptorOffset: packed.gposDescriptorOffset,
        classTables: packed.classTables,
    };
    (programs ??= new WeakMap()).set(plan, program);
    return program;
}

function compileLookup(
    parsed: ParsedLookupActions,
    plan: ComputeTextPlan,
    all: Map<string, ParsedLookupActions>,
    deltaOf: (outer: number, inner: number) => number
): ComputeProgramLookup {
    if (parsed.lookupFlag & RESERVED_FLAG) {
        throw new Error(`compileComputeTextProgram: ${parsed.table} lookup ${parsed.lookupIndex} has reserved flag bits`);
    }
    const applySecond = pairAppliesSecond(parsed, plan);
    const forceReject = (parsed.lookupFlag & RIGHT_TO_LEFT) !== 0 ? "right-to-left lookup flag is not executed" : null;
    // text-shaper 0.1.28's Context-3-only GSUB digest is empty; these roots never match.
    const oracleInert =
        parsed.table === "GSUB" && parsed.lookupType === 5 && parsed.subtables.every((subtable) => subtable.kind === "context" && !subtable.chaining && subtable.format === 3);
    const candidates = parsed.subtables.flatMap((subtable) => compileSubtable(subtable, parsed, plan, all, applySecond, forceReject, deltaOf));
    const rules = oracleInert ? candidates.filter((rule) => rule.mode === COMPUTE_RULE_MODE_REJECT) : candidates;
    return {
        table: parsed.table,
        lookupIndex: parsed.lookupIndex,
        lookupType: parsed.lookupType,
        lookupFlag: parsed.lookupFlag,
        ignoreBase: (parsed.lookupFlag & IGNORE_BASE) !== 0,
        ignoreLigatures: (parsed.lookupFlag & IGNORE_LIGATURES) !== 0,
        ignoreMarks: (parsed.lookupFlag & IGNORE_MARKS) !== 0,
        markAttachmentClass: (parsed.lookupFlag >> 8) & 0xff,
        markFilteringSetIgnored: (parsed.lookupFlag & USE_MARK_FILTERING_SET) !== 0,
        rules,
        expansion: Math.max(1, ...rules.map(expansionOf)),
    };
}

function compileSubtable(
    subtable: ParsedSubtable,
    lookup: ParsedLookupActions,
    plan: ComputeTextPlan,
    all: Map<string, ParsedLookupActions>,
    applySecond: boolean,
    forceReject: string | null,
    deltaOf: (outer: number, inner: number) => number
): ComputeProgramRule[] {
    const reject = (input: readonly ComputeGlyphPredicate[], reason: string): ComputeProgramRule => ({
        input,
        backtrack: [],
        lookahead: [],
        action: { kind: "reject", reason: forceReject ?? reason },
        mode: COMPUTE_RULE_MODE_REJECT,
    });
    if (subtable.kind === "unimplemented") {
        return [reject(subtable.coverage ? [predicateFromCoverage(subtable.coverage, plan.font.glyphCount)] : [], subtable.reason)];
    }
    if (subtable.kind === "multiple") {
        return coverageGlyphs(subtable.coverage, plan.font.glyphCount).flatMap((glyphId, index) => {
            const sequence = subtable.sequences[index] ?? [];
            if (sequence.length === 0) {
                return [];
            }
            if (sequence.some((glyph) => glyph >= plan.font.glyphCount)) {
                throw new Error(`compileComputeTextProgram: ${lookup.table} lookup ${lookup.lookupIndex} multiple substitute is outside the font`);
            }
            return [rule([[glyphId]], [], [], { kind: "multiple", glyphs: sequence }, COMPUTE_RULE_MODE_MULTIPLE)];
        });
    }
    if (subtable.kind === "alternate") {
        return coverageGlyphs(subtable.coverage, plan.font.glyphCount).map((glyphId, index) =>
            reject(
                [{ kind: "glyphs", ids: [glyphId], classId: 0, classDef: null, coverage: null }],
                `GSUB alternate substitution is not executed (${subtable.alternatives[index]?.length ?? 0} outputs)`
            )
        );
    }
    if (forceReject) {
        return rulesOf(subtable, lookup, plan, all, applySecond, deltaOf).map((rule) => ({
            ...rule,
            action: { kind: "reject", reason: forceReject },
            mode: COMPUTE_RULE_MODE_REJECT,
        }));
    }
    return rulesOf(subtable, lookup, plan, all, applySecond, deltaOf);
}

function rulesOf(
    subtable: ParsedSubtable,
    lookup: ParsedLookupActions,
    plan: ComputeTextPlan,
    all: Map<string, ParsedLookupActions>,
    applySecond: boolean,
    deltaOf: (outer: number, inner: number) => number
): ComputeProgramRule[] {
    const glyphCount = plan.font.glyphCount;
    if (subtable.kind === "single") {
        return coverageGlyphs(subtable.coverage, glyphCount).map((glyphId, index) => {
            const substitute = subtable.format === 1 ? (glyphId + subtable.delta) & 0xffff : (subtable.substitutes?.[index] ?? 0);
            if (substitute >= glyphCount) {
                throw new Error(`compileComputeTextProgram: ${lookup.table} lookup ${lookup.lookupIndex} substitute ${substitute} is outside the font`);
            }
            return rule([[glyphId]], [], [], { kind: "substitute", glyphs: [substitute] }, COMPUTE_RULE_MODE_SUBSTITUTE);
        });
    }
    if (subtable.kind === "ligature") {
        const rules: ComputeProgramRule[] = [];
        coverageGlyphs(subtable.coverage, glyphCount).forEach((glyphId, index) => {
            for (const ligature of subtable.sets[index] ?? []) {
                if (ligature.glyph >= glyphCount || ligature.components.some((component) => component >= glyphCount)) {
                    throw new Error(`compileComputeTextProgram: ${lookup.table} lookup ${lookup.lookupIndex} ligature glyph is outside the font`);
                }
                rules.push(
                    rule(
                        [[glyphId], ...ligature.components.map((component) => [component])],
                        [],
                        [],
                        { kind: "ligature", components: ligature.components, glyph: ligature.glyph },
                        COMPUTE_RULE_MODE_LIGATURE
                    )
                );
            }
        });
        return rules;
    }
    if (subtable.kind === "context") {
        return subtable.rules.map((context) => {
            const nested = context.lookups.map((record) => {
                const child = all.get(`${lookup.table}/${record.lookupListIndex}`);
                const executable = child !== undefined && isExecutableSingle(child) && record.sequenceIndex < context.input.length;
                return { record, executable, childType: child?.lookupType ?? -1 };
            });
            const input = context.input.map(toPredicate);
            const backtrack = context.backtrack.map(toPredicate);
            const lookahead = context.lookahead.map(toPredicate);
            if (nested.length === 0 || nested.every((item) => item.executable)) {
                return predicateRule(
                    input,
                    backtrack,
                    lookahead,
                    { kind: "context", lookups: context.lookups.map((record) => ({ sequenceIndex: record.sequenceIndex, lookupIndex: record.lookupListIndex })) },
                    COMPUTE_RULE_MODE_CONTEXT
                );
            }
            const offender = nested.find((item) => !item.executable);
            return predicateRule(
                input,
                backtrack,
                lookahead,
                {
                    kind: "reject",
                    reason: `nested ${lookup.table} lookup ${offender?.record.lookupListIndex} type ${offender?.childType} is not an executable single substitution`,
                },
                COMPUTE_RULE_MODE_REJECT
            );
        });
    }
    if (subtable.kind === "single-adjustment") {
        return coverageGlyphs(subtable.coverage, glyphCount).map((glyphId, index) =>
            valueRule([glyphId], subtable.values[subtable.values.length === 1 ? 0 : index] as ParsedValue, lookup, deltaOf)
        );
    }
    if (subtable.kind === "pair") {
        const rules: ComputeProgramRule[] = [];
        coverageGlyphs(subtable.coverage, glyphCount).forEach((glyphId, index) => {
            for (const pair of subtable.pairSets[index] ?? []) {
                rules.push(pairRule([[glyphId], [pair.secondGlyph]], pair.value1, pair.value2, applySecond, lookup, pair.secondGlyph, deltaOf));
            }
        });
        return rules;
    }
    if (subtable.kind === "class-pair") {
        const rules: ComputeProgramRule[] = [];
        subtable.rows.forEach((row, class1) => {
            row.forEach((cell, class2) => {
                const blocked = variationBlocks(cell.value1.variationIndexes, deltaOf) || variationBlocks(cell.value2.variationIndexes, deltaOf);
                if (!blocked && cell.value1.xAdvance === 0 && cell.value2.xAdvance === 0) {
                    return;
                }
                const input = [classPredicate(subtable.classDef1, class1, subtable.coverage), classPredicate(subtable.classDef2, class2, null)];
                if (blocked) {
                    rules.push(predicateRule(input, [], [], { kind: "reject", reason: deviceReason(lookup, true) }, COMPUTE_RULE_MODE_REJECT));
                    return;
                }
                rules.push(
                    predicateRule(
                        input,
                        [],
                        [],
                        { kind: "class-pair", class1, class2, firstXAdvance: cell.value1.xAdvance, secondXAdvance: cell.value2.xAdvance, applySecond },
                        COMPUTE_RULE_MODE_CLASS_PAIR
                    )
                );
            });
        });
        return rules;
    }
    if (subtable.kind === "mark-ligature") {
        return subtable.marks.flatMap((mark) => {
            const ligatures = subtable.ligatures.map((ligature) => ({
                glyphId: ligature.glyphId,
                componentCount: ligature.components.length,
                anchors: ligature.components.flatMap((component, componentIndex) => {
                    const anchor = component[mark.markClass] ?? null;
                    if (!anchor) {
                        return [];
                    }
                    return [
                        {
                            component: componentIndex,
                            x: anchor.x,
                            y: anchor.y,
                            reject: anchor.contourPoint || variationBlocks(anchor.variationIndexes, deltaOf),
                        },
                    ];
                }),
            }));
            const markBlocked = mark.anchor.contourPoint || variationBlocks(mark.anchor.variationIndexes, deltaOf);
            if (markBlocked) {
                return [
                    rule([[mark.glyphId]], [], [], { kind: "reject", reason: `unevaluated mark anchor in ${lookup.table} lookup ${lookup.lookupIndex}` }, COMPUTE_RULE_MODE_REJECT),
                ];
            }
            return [rule([[mark.glyphId]], [], [], { kind: "mark-ligature", markX: mark.anchor.x, markY: mark.anchor.y, ligatures }, COMPUTE_RULE_MODE_MARK_LIGATURE)];
        });
    }
    if (subtable.kind === "mark-base" || subtable.kind === "mark-mark") {
        return subtable.marks.map((mark) => {
            const bases = subtable.bases.map((base) => ({ glyphId: base.glyphId, anchor: base.anchors[mark.markClass] ?? null }));
            const contour = mark.anchor.contourPoint || bases.some((base) => base.anchor?.contourPoint);
            const blocked =
                variationBlocks(mark.anchor.variationIndexes, deltaOf) || bases.some((base) => base.anchor !== null && variationBlocks(base.anchor.variationIndexes, deltaOf));
            if (contour || blocked) {
                return rule(
                    [[mark.glyphId]],
                    [],
                    [],
                    {
                        kind: "reject",
                        reason: contour
                            ? `unevaluated contour anchor in ${lookup.table} lookup ${lookup.lookupIndex}`
                            : `nonzero default variation delta in ${lookup.table} lookup ${lookup.lookupIndex}`,
                    },
                    COMPUTE_RULE_MODE_REJECT
                );
            }
            return rule(
                [[mark.glyphId]],
                [],
                [],
                {
                    kind: subtable.kind,
                    markClass: mark.markClass,
                    markX: mark.anchor.x,
                    markY: mark.anchor.y,
                    bases: bases.filter((base) => base.anchor).map((base) => ({ glyphId: base.glyphId, x: base.anchor?.x ?? 0, y: base.anchor?.y ?? 0 })),
                    clearAdvance: subtable.kind === "mark-base",
                },
                subtable.kind === "mark-base" ? COMPUTE_RULE_MODE_MARK_BASE : COMPUTE_RULE_MODE_MARK_MARK
            );
        });
    }
    return [];
}

function pairRule(
    input: number[][],
    value1: ParsedValue,
    value2: ParsedValue,
    applySecond: boolean,
    lookup: ParsedLookupActions,
    secondGlyph: number,
    deltaOf: (outer: number, inner: number) => number
): ComputeProgramRule {
    const blocked = variationBlocks(value1.variationIndexes, deltaOf) || variationBlocks(value2.variationIndexes, deltaOf);
    if (blocked) {
        return rule(input, [], [], { kind: "reject", reason: deviceReason(lookup, true) }, COMPUTE_RULE_MODE_REJECT);
    }
    return rule(input, [], [], { kind: "pair", secondGlyph, firstXAdvance: value1.xAdvance, secondXAdvance: value2.xAdvance, applySecond }, COMPUTE_RULE_MODE_POSITION);
}

function valueRule(input: number[], value: ParsedValue | undefined, lookup: ParsedLookupActions, deltaOf: (outer: number, inner: number) => number): ComputeProgramRule {
    if (!value || variationBlocks(value.variationIndexes, deltaOf)) {
        return rule([input], [], [], { kind: "reject", reason: deviceReason(lookup, true) }, COMPUTE_RULE_MODE_REJECT);
    }
    return rule(
        [input],
        [],
        [],
        { kind: "adjust", xPlacement: value.xPlacement, yPlacement: value.yPlacement, xAdvance: value.xAdvance, yAdvance: value.yAdvance },
        COMPUTE_RULE_MODE_ADJUST
    );
}

function pairAppliesSecond(lookup: ParsedLookupActions, plan: ComputeTextPlan): boolean {
    const hasGdef = ensureComputeFontLayout(plan.font).gdef !== null;
    if (lookup.lookupFlag !== 0 && hasGdef) {
        return true;
    }
    if (lookup.subtables.length === 1) {
        return false;
    }
    if (lookup.subtables.length === 2 && lookup.subtables[0]?.kind === "pair" && lookup.subtables[1]?.kind === "class-pair") {
        return false;
    }
    return lookup.lookupType === 2;
}

function isExecutableSingle(lookup: ParsedLookupActions): boolean {
    return lookup.lookupType === 1 && lookup.subtables.every((subtable) => subtable.kind === "single") && (lookup.lookupFlag & (RIGHT_TO_LEFT | RESERVED_FLAG)) === 0;
}

function variationBlocks(indexes: readonly VariationDeviceIndex[], deltaOf: (outer: number, inner: number) => number): boolean {
    return indexes.some((index) => deltaOf(index.outer, index.inner) !== 0);
}

function deviceReason(lookup: ParsedLookupActions, variation: boolean): string {
    return variation
        ? `nonzero default variation delta in ${lookup.table} lookup ${lookup.lookupIndex}`
        : `unsupported positioning value in ${lookup.table} lookup ${lookup.lookupIndex}`;
}

function rule(
    input: readonly (readonly number[])[],
    backtrack: readonly ComputeGlyphPredicate[],
    lookahead: readonly ComputeGlyphPredicate[],
    action: ComputeProgramAction,
    mode: number
): ComputeProgramRule {
    return predicateRule(
        input.map((ids) => ({ kind: "glyphs" as const, ids, classId: 0, classDef: null, coverage: null })),
        backtrack,
        lookahead,
        action,
        mode
    );
}

function predicateRule(
    input: readonly ComputeGlyphPredicate[],
    backtrack: readonly ComputeGlyphPredicate[],
    lookahead: readonly ComputeGlyphPredicate[],
    action: ComputeProgramAction,
    mode: number
): ComputeProgramRule {
    return { input, backtrack, lookahead, action, mode };
}

function toPredicate(predicate: ParsedGlyphPredicate): ComputeGlyphPredicate {
    return { kind: predicate.kind, ids: predicate.ids, classId: predicate.classId, classDef: predicate.classDef, coverage: predicate.coverage };
}

function classPredicate(classDef: GlyphClassDef, classId: number, coverage: GlyphCoverage | null): ComputeGlyphPredicate {
    return { kind: "class", ids: [], classId, classDef, coverage };
}

function predicateFromCoverage(coverage: GlyphCoverage, glyphCount: number): ComputeGlyphPredicate {
    return { kind: "glyphs", ids: coverageGlyphs(coverage, glyphCount), classId: 0, classDef: null, coverage };
}

function expansionOf(ruleRecord: ComputeProgramRule): number {
    if (ruleRecord.action.kind === "substitute" || ruleRecord.action.kind === "multiple") {
        return Math.max(1, ruleRecord.action.glyphs.length);
    }
    return 1;
}

function adjustmentOf(ruleRecord: ComputeProgramRule): number {
    const action = ruleRecord.action;
    if (action.kind === "pair" || action.kind === "class-pair") {
        return Math.max(Math.abs(action.firstXAdvance), action.applySecond ? Math.abs(action.secondXAdvance) : 0);
    }
    if (action.kind === "adjust") {
        return Math.max(Math.abs(action.xPlacement), Math.abs(action.yPlacement), Math.abs(action.xAdvance), Math.abs(action.yAdvance));
    }
    if (action.kind === "mark-base" || action.kind === "mark-mark") {
        return Math.max(Math.abs(action.markX), Math.abs(action.markY), ...action.bases.flatMap((base) => [Math.abs(base.x), Math.abs(base.y)]));
    }
    return 0;
}

function packProgram(
    plan: ComputeTextPlan,
    lookups: readonly ComputeProgramLookup[],
    rootOrder: readonly number[]
): { words: number[]; rootWitnesses: Uint32Array; gsubDescriptorOffset: number; gposDescriptorOffset: number; classTables: readonly GlyphClassDef[] } {
    const words: number[] = [];
    const push = (...values: number[]): number => {
        const origin = words.length;
        words.push(...values);
        return origin;
    };
    const classDefs: GlyphClassDef[] = [];
    const coverages = new Map<GlyphCoverage, number>();
    const coverageOrigin = (coverage: GlyphCoverage | null): number => {
        if (!coverage) {
            return 0;
        }
        const existing = coverages.get(coverage);
        if (existing !== undefined) {
            return existing;
        }
        const glyphs = coverageGlyphs(coverage, plan.font.glyphCount);
        const origin = push(glyphs.length, ...glyphs);
        coverages.set(coverage, origin);
        return origin;
    };
    const denseClasses = new Map<GlyphClassDef, Uint32Array>();
    const firstGlyphs = (predicate: ComputeGlyphPredicate): readonly number[] => {
        if (predicate.kind === "glyphs") {
            return predicate.ids;
        }
        const def = predicate.classDef;
        let dense = def ? denseClasses.get(def) : undefined;
        if (def && !dense) {
            dense = Uint32Array.from({ length: plan.font.glyphCount }, (_, glyph) => glyphClass(def, glyph));
            denseClasses.set(def, dense);
        }
        const domain = predicate.coverage ? coverageGlyphs(predicate.coverage, plan.font.glyphCount) : Array.from({ length: plan.font.glyphCount }, (_, glyph) => glyph);
        return domain.filter((glyph) => (dense?.[glyph] ?? 0) === predicate.classId);
    };
    const classIndex = (def: GlyphClassDef | null): number => {
        if (!def) {
            return 0xffffffff;
        }
        const existing = classDefs.indexOf(def);
        if (existing >= 0) {
            return existing;
        }
        classDefs.push(def);
        return classDefs.length - 1;
    };
    const packPredicates = (predicates: readonly ComputeGlyphPredicate[]): number => {
        if (predicates.length === 0) {
            return 0;
        }
        const coverageOrigins = predicates.map((predicate) => (predicate.kind === "class" ? coverageOrigin(predicate.coverage) : 0));
        const origin = words.length;
        predicates.forEach((predicate, index) => {
            if (predicate.kind === "class") {
                push(0x80000000 | (predicate.classId & 0xffff), classIndex(predicate.classDef), coverageOrigins[index] ?? 0);
            } else {
                push(predicate.ids.length, ...[...predicate.ids].sort((left, right) => left - right));
            }
        });
        return origin;
    };
    const packAction = (action: ComputeProgramAction): { origin: number; length: number } => {
        if (action.kind === "substitute" || action.kind === "multiple") {
            return { origin: push(action.glyphs.length, ...action.glyphs), length: 1 + action.glyphs.length };
        }
        if (action.kind === "ligature") {
            return { origin: push(1, action.glyph, action.components.length, ...action.components), length: 3 + action.components.length };
        }
        if (action.kind === "context") {
            return { origin: push(action.lookups.length, ...action.lookups.flatMap((item) => [item.sequenceIndex, item.lookupIndex])), length: 1 + action.lookups.length * 2 };
        }
        if (action.kind === "pair") {
            return { origin: push(action.secondGlyph, action.firstXAdvance >>> 0, action.secondXAdvance >>> 0, action.applySecond ? 1 : 0), length: 4 };
        }
        if (action.kind === "class-pair") {
            return { origin: push(action.class1, action.class2, action.firstXAdvance >>> 0, action.secondXAdvance >>> 0, action.applySecond ? 1 : 0), length: 5 };
        }
        if (action.kind === "adjust") {
            return { origin: push(action.xPlacement >>> 0, action.yPlacement >>> 0, action.xAdvance >>> 0, action.yAdvance >>> 0), length: 4 };
        }
        if (action.kind === "mark-ligature") {
            return {
                origin: push(
                    action.markX >>> 0,
                    action.markY >>> 0,
                    action.ligatures.length,
                    ...action.ligatures.flatMap((ligature) => [
                        ligature.glyphId,
                        ligature.componentCount,
                        ligature.anchors.length,
                        ...ligature.anchors.flatMap((anchor) => [anchor.component, anchor.reject ? 1 : 0, anchor.x >>> 0, anchor.y >>> 0]),
                    ])
                ),
                length: 3 + action.ligatures.reduce((sum, ligature) => sum + 3 + ligature.anchors.length * 4, 0),
            };
        }
        if (action.kind === "mark-base" || action.kind === "mark-mark") {
            return {
                origin: push(
                    action.markClass,
                    action.markX >>> 0,
                    action.markY >>> 0,
                    action.bases.length,
                    ...action.bases
                        .slice()
                        .sort((left, right) => left.glyphId - right.glyphId)
                        .flatMap((base) => [base.glyphId, base.x >>> 0, base.y >>> 0])
                ),
                length: 4 + action.bases.length * 3,
            };
        }
        return { origin: 0, length: 0 };
    };
    push(0x43545031, 2, plan._gsubRoots.length, plan._gposRoots.length, 0, 0, 0, 0, 0, rootOrder.length);
    const rootOrigin = push(...rootOrder);
    const gsubCapacity = capacity(lookups, "GSUB");
    const gposCapacity = capacity(lookups, "GPOS");
    const gsubDescriptorOffset = words.length;
    words.push(...new Array(gsubCapacity * COMPUTE_PROGRAM_LOOKUP_WORDS).fill(0));
    const gposDescriptorOffset = words.length;
    words.push(...new Array(gposCapacity * COMPUTE_PROGRAM_LOOKUP_WORDS).fill(0));
    words[4] = gsubDescriptorOffset;
    words[5] = gposDescriptorOffset;
    words[6] = gsubCapacity;
    words[7] = gposCapacity;
    words[8] = rootOrigin;
    for (const lookup of lookups) {
        const ruleOrigin = words.length;
        const ruleSlots = lookup.rules.map(() => {
            const slot = words.length;
            words.push(...new Array(COMPUTE_PROGRAM_RULE_WORDS).fill(0));
            return slot;
        });
        lookup.rules.forEach((ruleRecord, index) => {
            const input = packPredicates(ruleRecord.input);
            const backtrack = packPredicates(ruleRecord.backtrack);
            const lookahead = packPredicates(ruleRecord.lookahead);
            const action = packAction(ruleRecord.action);
            const slot = ruleSlots[index] ?? ruleOrigin;
            words[slot + COMPUTE_RULE_INPUT] = input;
            words[slot + COMPUTE_RULE_INPUT_LENGTH] = ruleRecord.input.length;
            words[slot + COMPUTE_RULE_BACKTRACK] = backtrack;
            words[slot + COMPUTE_RULE_BACKTRACK_LENGTH] = ruleRecord.backtrack.length;
            words[slot + COMPUTE_RULE_LOOKAHEAD] = lookahead;
            words[slot + COMPUTE_RULE_LOOKAHEAD_LENGTH] = ruleRecord.lookahead.length;
            words[slot + COMPUTE_RULE_ACTION] = action.origin;
            words[slot + COMPUTE_RULE_ACTION_LENGTH] = action.length;
            words[slot + COMPUTE_RULE_MODE] = ruleRecord.mode;
            words[slot + 9] = ruleRecord.action.kind === "reject" ? 1 : 0;
        });
        let start = 0;
        while (start < lookup.rules.length) {
            const first = lookup.rules[start]!;
            const input = first.input;
            if (
                lookup.table !== "GPOS" ||
                input.length !== 2 ||
                input[0]?.kind !== "class" ||
                input[1]?.kind !== "class" ||
                input[1].coverage !== null ||
                first.backtrack.length !== 0 ||
                first.lookahead.length !== 0 ||
                (first.mode !== COMPUTE_RULE_MODE_CLASS_PAIR && first.mode !== COMPUTE_RULE_MODE_REJECT)
            ) {
                start++;
                continue;
            }
            let end = start + 1;
            while (end < lookup.rules.length) {
                const next = lookup.rules[end]!;
                if (
                    next.input.length !== 2 ||
                    next.input[0]?.classDef !== input[0].classDef ||
                    next.input[0]?.classId !== input[0].classId ||
                    next.input[0]?.coverage !== input[0].coverage ||
                    next.input[1]?.classDef !== input[1].classDef ||
                    next.input[1]?.coverage !== null ||
                    next.backtrack.length !== 0 ||
                    next.lookahead.length !== 0 ||
                    (next.mode !== COMPUTE_RULE_MODE_CLASS_PAIR && next.mode !== COMPUTE_RULE_MODE_REJECT)
                ) {
                    break;
                }
                end++;
            }
            const row = new Array<number>(Math.max(...lookup.rules.slice(start, end).map((rule) => rule.input[1]?.classId ?? 0)) + 1).fill(0);
            for (let index = start; index < end; index++) {
                const cls = lookup.rules[index]?.input[1]?.classId ?? 0;
                if (row[cls] === 0) {
                    row[cls] = index + 1;
                }
            }
            const origin = push(...row);
            for (let index = start; index < end; index++) {
                const slot = ruleSlots[index]!;
                words[slot + 10] = origin;
                words[slot + 11] = classIndex(input[1].classDef);
                words[slot + 12] = row.length;
                words[slot + 13] = end - start;
            }
            start = end;
        }
        const base = (lookup.table === "GSUB" ? gsubDescriptorOffset : gposDescriptorOffset) + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
        words[base + COMPUTE_LOOKUP_KIND] = lookup.lookupType;
        words[base + COMPUTE_LOOKUP_FLAG] = lookup.lookupFlag;
        const independent = lookup.rules.every((rule) => rule.backtrack.length === 0 && rule.lookahead.length === 0);
        words[base + COMPUTE_LOOKUP_MEMBERSHIP] =
            lookup.table === "GSUB" && independent && lookup.rules.every((rule) => rule.input.length === 1 && (rule.mode === 1 || rule.mode === 5))
                ? 1
                : lookup.table === "GPOS" && independent && lookup.rules.every((rule) => rule.input.length === 2 && (rule.mode === 3 || rule.mode === 4 || rule.mode === 5))
                  ? lookup.rules.every(
                        (rule) =>
                            rule.action.kind === "reject" ||
                            ((rule.action.kind === "pair" || rule.action.kind === "class-pair") && (!rule.action.applySecond || rule.action.secondXAdvance === 0))
                    )
                      ? 3
                      : 2
                  : 0;
        words[base + COMPUTE_LOOKUP_RULE_ORIGIN] = ruleOrigin;
        words[base + COMPUTE_LOOKUP_RULE_COUNT] = lookup.rules.length;
        words[base + COMPUTE_LOOKUP_EXPANSION] = lookup.expansion;
        const candidateOrigin = words.length;
        const candidates: number[][] = Array.from({ length: plan.font.glyphCount }, () => []);
        lookup.rules.forEach((ruleRecord, index) => {
            const first = ruleRecord.input[0];
            if (first) {
                for (const glyph of firstGlyphs(first)) {
                    candidates[glyph]?.push(index);
                }
            }
        });
        for (const list of candidates) {
            push(0, list.length);
        }
        candidates.forEach((list, glyph) => {
            words[candidateOrigin + glyph * 2] = push(...list);
        });
        words[base + COMPUTE_LOOKUP_CANDIDATES] = candidateOrigin;
        const orderedModes = lookup.table === "GSUB" ? [1, 2, 5, 9, 11] : [3, 4, 5, 7, 8, 10, 12];
        const ordered = lookup.rules.every((rule) => orderedModes.includes(rule.mode)) ? (lookup.rules.some((rule) => rule.mode === 11) ? 2 : 1) : 0;
        words[base + COMPUTE_LOOKUP_ORDERED] = ordered | (ordered && lookup.table === "GSUB" && lookup.rules.every((rule) => rule.backtrack.length === 0) ? 4 : 0);
    }
    const rootWitnesses = new Uint32Array(plan.font.glyphCount);
    let unconditional = 0;
    rootOrder.slice(0, 32).forEach((index, root) => {
        const lookup = lookups[index]!;
        const bit = 1 << root;
        if (lookup.table !== "GSUB" || lookup.rules.some((rule) => rule.input.length < 2 && rule.backtrack.length === 0 && rule.lookahead.length === 0)) {
            unconditional |= bit;
            return;
        }
        for (const rule of lookup.rules) {
            let witness: readonly number[] | undefined;
            for (const predicate of [...rule.input.slice(1), ...rule.backtrack, ...rule.lookahead]) {
                const glyphs = firstGlyphs(predicate);
                if (!witness || glyphs.length < witness.length) {
                    witness = glyphs;
                }
            }
            if (!witness) {
                throw new Error("compileComputeTextProgram: missing necessary predicate witness");
            }
            for (const glyph of witness) {
                rootWitnesses[glyph] = (rootWitnesses[glyph]! | bit) >>> 0;
            }
        }
    });
    for (let glyph = 0; glyph < rootWitnesses.length; glyph++) {
        rootWitnesses[glyph] = (rootWitnesses[glyph]! | unconditional) >>> 0;
    }
    return { words, rootWitnesses, gsubDescriptorOffset, gposDescriptorOffset, classTables: classDefs };
}

function capacity(lookups: readonly ComputeProgramLookup[], table: "GSUB" | "GPOS"): number {
    const indexes = lookups.filter((lookup) => lookup.table === table).map((lookup) => lookup.lookupIndex);
    return indexes.length === 0 ? 0 : Math.max(...indexes) + 1;
}
