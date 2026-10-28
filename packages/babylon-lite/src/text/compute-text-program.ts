/** Checked lookup program for a selected compute text plan.
 *
 *  Packed words keep the lab's eight-word lookup descriptors and sixteen-word rule
 *  descriptors. Mode 5 is a fail-if-matched rejection, never a successful skip.
 *  A VariationIndex whose default-coordinate delta is zero is ignored and the design
 *  value is used. A nonzero default delta stays a rejection; it is not evaluated into
 *  the action. ppem devices are not evaluated: text-shaper 0.1.28 applies the design
 *  value and this plan has no ppem. Contour anchors stay rejected.
 *  v3 has a 16-word header and four-word roots. Header 10/12 are stage origin/count,
 *  11 is reserved zero, 13 is presence words/glyph, 14 is the optional profile section.
 *  Pair actions encode both complete values; horizontal yAdvance words are reserved zero.
 *
 *  Lookup descriptor words, at `descriptorOffset + fontLookupIndex * 8`:
 *    0 kind, 1 `flag | markFilteringSet << 16`, 2 cooperative capability, 3 rule origin, 4 rule count,
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
const EMPTY_GLYPHS: readonly number[] = [];

type ComputeProgramMarkBases = readonly { readonly glyphId: number; readonly x: number; readonly y: number }[];
type ComputeProgramValue = Pick<ParsedValue, "xPlacement" | "yPlacement" | "xAdvance" | "yAdvance">;

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
    | { readonly kind: "context"; readonly table: "GSUB" | "GPOS"; readonly lookups: readonly { readonly sequenceIndex: number; readonly lookupIndex: number }[] }
    | { readonly kind: "pair"; readonly secondGlyph: number; readonly first: ComputeProgramValue; readonly second: ComputeProgramValue; readonly format2Nonzero: boolean }
    | {
          readonly kind: "class-pair";
          readonly class1: number;
          readonly class2: number;
          readonly first: ComputeProgramValue;
          readonly second: ComputeProgramValue;
          readonly format2Nonzero: boolean;
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
          readonly bases: ComputeProgramMarkBases;
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
    readonly markFilteringSet: number;
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
    readonly maxNestedRecords: number;
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
    const rootOrder = plan._stages.flatMap((stage) =>
        stage.roots.map((root) => {
            const index = indexOf.get(`${stage.table}/${root.lookup}`);
            if (index === undefined) {
                throw new Error(`compileComputeTextProgram: stage root ${stage.table}/${root.lookup} is missing from the plan union`);
            }
            return index;
        })
    );
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
    let maxExpansion = 1;
    let maxPatternDepth = 1;
    let maxAbsAdjustment = 0;
    let maxNestedRecords = 0;
    for (const lookup of lookups) {
        maxExpansion = Math.max(maxExpansion, lookup.expansion);
        for (const rule of lookup.rules) {
            maxPatternDepth = Math.max(maxPatternDepth, rule.input.length, rule.backtrack.length, rule.lookahead.length);
            maxAbsAdjustment = Math.max(maxAbsAdjustment, adjustmentOf(rule));
            if (rule.action.kind === "context") {
                maxNestedRecords = Math.max(maxNestedRecords, rule.action.lookups.length);
            }
        }
    }
    const maxDependencyDepth = lookups.some((lookup) => lookup.rules.some((rule) => rule.action.kind === "context")) ? 1 : 0;
    const packed = packProgram(plan, lookups, rootOrder);
    const program: ComputeTextProgram = {
        plan,
        lookups,
        rootOrder,
        maxExpansion,
        maxPatternDepth,
        maxDependencyDepth,
        maxNestedRecords,
        maxAbsAdjustment,
        rejections,
        fullyExecutable: rejections.length === 0,
        words: packed.words,
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
    const markFilteringSet = parsed.markFilteringSet ?? 0;
    if ((parsed.lookupFlag & USE_MARK_FILTERING_SET) !== 0 && !ensureComputeFontLayout(plan.font).gdef?.markGlyphSets[markFilteringSet]) {
        throw new Error(`compileComputeTextProgram: ${parsed.table} lookup ${parsed.lookupIndex} mark filtering set ${markFilteringSet} is missing`);
    }
    const forceReject = (parsed.lookupFlag & RIGHT_TO_LEFT) !== 0 ? "right-to-left lookup flag is not executed" : null;
    const rules = parsed.subtables.flatMap((subtable) => compileSubtable(subtable, parsed, plan, all, forceReject, deltaOf));
    let expansion = 1;
    for (const rule of rules) {
        expansion = Math.max(expansion, expansionOf(rule));
        if (rule.input.length > 64) {
            throw new Error(`compileComputeTextProgram: ${parsed.table} lookup ${parsed.lookupIndex} input exceeds 64 matched positions`);
        }
        if (rule.action.kind === "context") {
            let nestedExpansion = 1;
            for (const record of rule.action.lookups) {
                const child = all.get(`${parsed.table}/${record.lookupIndex}`);
                nestedExpansion *= Math.max(
                    1,
                    ...child!.subtables.flatMap((subtable) => (subtable.kind === "multiple" ? subtable.sequences.map((sequence) => sequence.length) : [1]))
                );
            }
            expansion = Math.max(expansion, nestedExpansion);
        }
    }
    return {
        table: parsed.table,
        lookupIndex: parsed.lookupIndex,
        lookupType: parsed.lookupType,
        lookupFlag: parsed.lookupFlag,
        ignoreBase: (parsed.lookupFlag & IGNORE_BASE) !== 0,
        ignoreLigatures: (parsed.lookupFlag & IGNORE_LIGATURES) !== 0,
        ignoreMarks: (parsed.lookupFlag & IGNORE_MARKS) !== 0,
        markAttachmentClass: (parsed.lookupFlag >> 8) & 0xff,
        markFilteringSet,
        rules,
        expansion,
    };
}

function compileSubtable(
    subtable: ParsedSubtable,
    lookup: ParsedLookupActions,
    plan: ComputeTextPlan,
    all: Map<string, ParsedLookupActions>,
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
        return rulesOf(subtable, lookup, plan, all, deltaOf).map((rule) => ({
            ...rule,
            action: { kind: "reject", reason: forceReject },
            mode: COMPUTE_RULE_MODE_REJECT,
        }));
    }
    return rulesOf(subtable, lookup, plan, all, deltaOf);
}

function rulesOf(
    subtable: ParsedSubtable,
    lookup: ParsedLookupActions,
    plan: ComputeTextPlan,
    all: Map<string, ParsedLookupActions>,
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
                if (record.sequenceIndex >= context.input.length) {
                    throw new Error(`compileComputeTextProgram: sequence index ${record.sequenceIndex} exceeds context input length`);
                }
                const executable = child !== undefined && !(lookup.table === "GSUB" ? [5, 6] : [7, 8]).includes(child.lookupType);
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
                    {
                        kind: "context",
                        table: lookup.table,
                        lookups: context.lookups.map((record) => ({ sequenceIndex: record.sequenceIndex, lookupIndex: record.lookupListIndex })),
                    },
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
                    reason: `nested contextual lookup ${offender?.record.lookupListIndex} is not executable`,
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
                rules.push(pairRule([[glyphId], [pair.secondGlyph]], pair.value1, pair.value2, subtable.valueFormat2 !== 0, lookup, pair.secondGlyph, deltaOf));
            }
        });
        return rules;
    }
    if (subtable.kind === "class-pair") {
        const rules: ComputeProgramRule[] = [];
        subtable.rows.forEach((row, class1) => {
            row.forEach((cell, class2) => {
                const blocked = variationBlocks(cell.value1.variationIndexes, deltaOf) || variationBlocks(cell.value2.variationIndexes, deltaOf);
                if (!blocked && !applicableValue(cell.value1) && !applicableValue(cell.value2)) {
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
                        {
                            kind: "class-pair",
                            class1,
                            class2,
                            first: horizontalValue(cell.value1),
                            second: horizontalValue(cell.value2),
                            format2Nonzero: subtable.valueFormat2 !== 0,
                        },
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
        const byClass = new Map<number, { contour: boolean; blocked: boolean; bases: ComputeProgramMarkBases }>();
        const classBases = (markClass: number) => {
            let entry = byClass.get(markClass);
            if (!entry) {
                let contour = false;
                let blocked = false;
                const bases: { glyphId: number; x: number; y: number }[] = [];
                for (const base of subtable.bases) {
                    const anchor = base.anchors[markClass] ?? null;
                    if (!anchor) {
                        continue;
                    }
                    if (anchor.contourPoint) {
                        contour = true;
                    }
                    if (!blocked && variationBlocks(anchor.variationIndexes, deltaOf)) {
                        blocked = true;
                    }
                    bases.push({ glyphId: base.glyphId, x: anchor.x, y: anchor.y });
                }
                entry = { contour, blocked, bases };
                byClass.set(markClass, entry);
            }
            return entry;
        };
        return subtable.marks.map((mark) => {
            const entry = classBases(mark.markClass);
            const contour = mark.anchor.contourPoint || entry.contour;
            const blocked = variationBlocks(mark.anchor.variationIndexes, deltaOf) || entry.blocked;
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
                    bases: entry.bases,
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
    format2Nonzero: boolean,
    lookup: ParsedLookupActions,
    secondGlyph: number,
    deltaOf: (outer: number, inner: number) => number
): ComputeProgramRule {
    const blocked = variationBlocks(value1.variationIndexes, deltaOf) || variationBlocks(value2.variationIndexes, deltaOf);
    if (blocked) {
        return rule(input, [], [], { kind: "reject", reason: deviceReason(lookup, true) }, COMPUTE_RULE_MODE_REJECT);
    }
    return rule(input, [], [], { kind: "pair", secondGlyph, first: horizontalValue(value1), second: horizontalValue(value2), format2Nonzero }, COMPUTE_RULE_MODE_POSITION);
}

function valueRule(input: number[], value: ParsedValue | undefined, lookup: ParsedLookupActions, deltaOf: (outer: number, inner: number) => number): ComputeProgramRule {
    if (!value || variationBlocks(value.variationIndexes, deltaOf)) {
        return rule([input], [], [], { kind: "reject", reason: deviceReason(lookup, true) }, COMPUTE_RULE_MODE_REJECT);
    }
    return rule([input], [], [], { kind: "adjust", ...horizontalValue(value) }, COMPUTE_RULE_MODE_ADJUST);
}

function horizontalValue(value: ParsedValue): ComputeProgramValue {
    return { xPlacement: value.xPlacement, yPlacement: value.yPlacement, xAdvance: value.xAdvance, yAdvance: 0 };
}

function applicableValue(value: ComputeProgramValue): boolean {
    return value.xPlacement !== 0 || value.yPlacement !== 0 || value.xAdvance !== 0;
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

let markBasesMax: WeakMap<ComputeProgramMarkBases, number> | null = null;

function adjustmentOfBases(bases: ComputeProgramMarkBases): number {
    const existing = markBasesMax?.get(bases);
    if (existing !== undefined) {
        return existing;
    }
    let maximum = -Infinity;
    for (const base of bases) {
        maximum = Math.max(maximum, Math.abs(base.x), Math.abs(base.y));
    }
    (markBasesMax ??= new WeakMap()).set(bases, maximum);
    return maximum;
}

function adjustmentOf(ruleRecord: ComputeProgramRule): number {
    const action = ruleRecord.action;
    if (action.kind === "pair" || action.kind === "class-pair") {
        return Math.max(...[action.first, action.second].flatMap((value) => [Math.abs(value.xPlacement), Math.abs(value.yPlacement), Math.abs(value.xAdvance)]));
    }
    if (action.kind === "adjust") {
        return Math.max(Math.abs(action.xPlacement), Math.abs(action.yPlacement), Math.abs(action.xAdvance), Math.abs(action.yAdvance));
    }
    if (action.kind === "mark-base" || action.kind === "mark-mark") {
        return Math.max(Math.abs(action.markX), Math.abs(action.markY), adjustmentOfBases(action.bases));
    }
    return 0;
}

function packProgram(
    plan: ComputeTextPlan,
    lookups: readonly ComputeProgramLookup[],
    rootOrder: readonly number[]
): { words: Uint32Array; rootWitnesses: Uint32Array; gsubDescriptorOffset: number; gposDescriptorOffset: number; classTables: readonly GlyphClassDef[] } {
    let words = new Uint32Array(1 << 16);
    let length = 0;
    /** Appends `count` zero words and returns their origin. */
    const reserve = (count: number): number => {
        const origin = length;
        const needed = length + count;
        if (needed > words.length) {
            let size = words.length * 2;
            while (size < needed) {
                size *= 2;
            }
            const grown = new Uint32Array(size);
            grown.set(words.subarray(0, length));
            words = grown;
        }
        length = needed;
        return origin;
    };
    /** Appends `values` and returns their origin. */
    const append = (values: ArrayLike<number>): number => {
        const origin = reserve(values.length);
        words.set(values, origin);
        return origin;
    };
    const classDefs: GlyphClassDef[] = [];
    const coverages = new Map<GlyphCoverage, number>();
    const baseWords = new Map<ComputeProgramMarkBases, Uint32Array>();
    const coverageOrigin = (coverage: GlyphCoverage | null): number => {
        if (!coverage) {
            return 0;
        }
        const existing = coverages.get(coverage);
        if (existing !== undefined) {
            return existing;
        }
        const glyphs = coverageGlyphs(coverage, plan.font.glyphCount);
        const origin = reserve(1 + glyphs.length);
        words[origin] = glyphs.length;
        words.set(glyphs, origin + 1);
        coverages.set(coverage, origin);
        return origin;
    };
    const denseClasses = new Map<GlyphClassDef, Uint32Array>();
    const classBuckets = new Map<GlyphClassDef | null, Map<GlyphCoverage | null, Map<number, number[]>>>();
    const firstGlyphs = (predicate: ComputeGlyphPredicate): readonly number[] => {
        if (predicate.kind === "glyphs") {
            return predicate.ids;
        }
        const def = predicate.classDef;
        let byCoverage = classBuckets.get(def);
        if (!byCoverage) {
            byCoverage = new Map();
            classBuckets.set(def, byCoverage);
        }
        let buckets = byCoverage.get(predicate.coverage);
        if (!buckets) {
            let dense = def ? denseClasses.get(def) : undefined;
            if (def && !dense) {
                dense = Uint32Array.from({ length: plan.font.glyphCount }, (_, glyph) => glyphClass(def, glyph));
                denseClasses.set(def, dense);
            }
            const filled = new Map<number, number[]>();
            const visit = (glyph: number): void => {
                const cls = dense?.[glyph] ?? 0;
                const list = filled.get(cls);
                if (list) {
                    list.push(glyph);
                } else {
                    filled.set(cls, [glyph]);
                }
            };
            if (predicate.coverage) {
                for (const glyph of coverageGlyphs(predicate.coverage, plan.font.glyphCount)) {
                    visit(glyph);
                }
            } else {
                for (let glyph = 0; glyph < plan.font.glyphCount; glyph++) {
                    visit(glyph);
                }
            }
            buckets = filled;
            byCoverage.set(predicate.coverage, buckets);
        }
        return buckets.get(predicate.classId) ?? EMPTY_GLYPHS;
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
        const coverageOrigins = new Array<number>(predicates.length);
        for (let index = 0; index < predicates.length; index++) {
            const predicate = predicates[index]!;
            coverageOrigins[index] = predicate.kind === "class" ? coverageOrigin(predicate.coverage) : 0;
        }
        const origin = length;
        for (let index = 0; index < predicates.length; index++) {
            const predicate = predicates[index]!;
            if (predicate.kind === "class") {
                const at = reserve(3);
                words[at] = 0x80000000 | (predicate.classId & 0xffff);
                words[at + 1] = classIndex(predicate.classDef);
                words[at + 2] = coverageOrigins[index] ?? 0;
            } else {
                const ids = predicate.ids;
                const n = ids.length;
                const at = reserve(1 + n);
                words[at] = n;
                let ordered = true;
                for (let i = 1; i < n; i++) {
                    if (!(ids[i - 1]! <= ids[i]!)) {
                        ordered = false;
                        break;
                    }
                }
                words.set(ordered ? ids : Array.from(ids).sort((left, right) => left - right), at + 1);
            }
        }
        return origin;
    };
    const packAction = (action: ComputeProgramAction): { origin: number; length: number } => {
        if (action.kind === "substitute" || action.kind === "multiple") {
            const origin = reserve(1 + action.glyphs.length);
            words[origin] = action.glyphs.length;
            words.set(action.glyphs, origin + 1);
            return { origin, length: 1 + action.glyphs.length };
        }
        if (action.kind === "ligature") {
            const origin = reserve(3 + action.components.length);
            words[origin] = 1;
            words[origin + 1] = action.glyph;
            words[origin + 2] = action.components.length;
            words.set(action.components, origin + 3);
            return { origin, length: 3 + action.components.length };
        }
        if (action.kind === "context") {
            const origin = reserve(1 + action.lookups.length * 2);
            words[origin] = action.lookups.length;
            for (let index = 0; index < action.lookups.length; index++) {
                const item = action.lookups[index]!;
                words[origin + 1 + index * 2] = item.sequenceIndex;
                words[origin + 2 + index * 2] = item.lookupIndex | (action.table === "GPOS" ? 0x10000 : 0);
            }
            return { origin, length: 1 + action.lookups.length * 2 };
        }
        if (action.kind === "pair") {
            return { origin: append([action.secondGlyph, ...valueWords(action.first), ...valueWords(action.second), Number(action.format2Nonzero)]), length: 10 };
        }
        if (action.kind === "class-pair") {
            return { origin: append([action.class1, action.class2, ...valueWords(action.first), ...valueWords(action.second), Number(action.format2Nonzero)]), length: 11 };
        }
        if (action.kind === "adjust") {
            const origin = reserve(4);
            words[origin] = action.xPlacement >>> 0;
            words[origin + 1] = action.yPlacement >>> 0;
            words[origin + 2] = action.xAdvance >>> 0;
            words[origin + 3] = action.yAdvance >>> 0;
            return { origin, length: 4 };
        }
        if (action.kind === "mark-ligature") {
            let actionLength = 3;
            for (const ligature of action.ligatures) {
                actionLength += 3 + ligature.anchors.length * 4;
            }
            const origin = reserve(actionLength);
            words[origin] = action.markX >>> 0;
            words[origin + 1] = action.markY >>> 0;
            words[origin + 2] = action.ligatures.length;
            let at = origin + 3;
            for (const ligature of action.ligatures) {
                words[at++] = ligature.glyphId;
                words[at++] = ligature.componentCount;
                words[at++] = ligature.anchors.length;
                for (const anchor of ligature.anchors) {
                    words[at++] = anchor.component;
                    words[at++] = anchor.reject ? 1 : 0;
                    words[at++] = anchor.x >>> 0;
                    words[at++] = anchor.y >>> 0;
                }
            }
            return { origin, length: actionLength };
        }
        if (action.kind === "mark-base" || action.kind === "mark-mark") {
            const bases = action.bases;
            let cached = baseWords.get(bases);
            if (!cached) {
                let sorted = true;
                for (let index = 1; index < bases.length; index++) {
                    if (!(bases[index - 1]!.glyphId <= bases[index]!.glyphId)) {
                        sorted = false;
                        break;
                    }
                }
                const ordered = sorted ? bases : bases.slice().sort((left, right) => left.glyphId - right.glyphId);
                cached = new Uint32Array(ordered.length * 3);
                for (let index = 0; index < ordered.length; index++) {
                    const base = ordered[index]!;
                    cached[index * 3] = base.glyphId;
                    cached[index * 3 + 1] = base.x >>> 0;
                    cached[index * 3 + 2] = base.y >>> 0;
                }
                baseWords.set(bases, cached);
            }
            const at = reserve(4 + bases.length * 3);
            words[at] = action.markClass;
            words[at + 1] = action.markX >>> 0;
            words[at + 2] = action.markY >>> 0;
            words[at + 3] = bases.length;
            words.set(cached, at + 4);
            return { origin: at, length: 4 + bases.length * 3 };
        }
        return { origin: 0, length: 0 };
    };
    const headerOrigin = reserve(16);
    words[headerOrigin] = 0x43545031;
    words[headerOrigin + 1] = 3;
    words[headerOrigin + 2] = plan._stages.filter((stage) => stage.table === "GSUB").reduce((sum, stage) => sum + stage.roots.length, 0);
    words[headerOrigin + 3] = rootOrder.length - words[headerOrigin + 2]!;
    words[headerOrigin + 4] = 0;
    words[headerOrigin + 5] = 0;
    words[headerOrigin + 6] = 0;
    words[headerOrigin + 7] = 0;
    words[headerOrigin + 8] = 0;
    words[headerOrigin + 9] = rootOrder.length;
    const rootOrigin = reserve(rootOrder.length * 4);
    const rootRecords = plan._stages.flatMap((stage, stageIndex) => stage.roots.map((root) => ({ ...root, stageIndex })));
    rootOrder.forEach((index, root) => {
        const lookup = lookups[index]!;
        words[rootOrigin + root * 4] = lookup.lookupIndex | (lookup.table === "GPOS" ? 0x10000 : 0);
        words[rootOrigin + root * 4 + 1] = rootRecords[root]!.mask;
        words[rootOrigin + root * 4 + 2] = rootRecords[root]!.flags;
        words[rootOrigin + root * 4 + 3] = rootRecords[root]!.stageIndex;
    });
    let firstRoot = 0;
    words[10] = append(
        plan._stages.flatMap((stage) => {
            const words = [
                stage.table === "GPOS" ? 1 : 0,
                firstRoot,
                stage.roots.length,
                ["none", "setupSyllables", "initialReorder", "finalReorder", "clearSyllables"].indexOf(stage.pause),
            ];
            firstRoot += stage.roots.length;
            return words;
        })
    );
    words[12] = plan._stages.length;
    words[13] = Math.ceil(Math.min(rootOrder.length, 256) / 32);
    const gsubCapacity = capacity(lookups, "GSUB");
    const gposCapacity = capacity(lookups, "GPOS");
    const gsubDescriptorOffset = reserve(gsubCapacity * COMPUTE_PROGRAM_LOOKUP_WORDS);
    const gposDescriptorOffset = reserve(gposCapacity * COMPUTE_PROGRAM_LOOKUP_WORDS);
    words[4] = gsubDescriptorOffset;
    words[5] = gposDescriptorOffset;
    words[6] = gsubCapacity;
    words[7] = gposCapacity;
    words[8] = rootOrigin;
    for (const lookup of lookups) {
        const ruleOrigin = reserve(lookup.rules.length * COMPUTE_PROGRAM_RULE_WORDS);
        lookup.rules.forEach((ruleRecord, index) => {
            const input = packPredicates(ruleRecord.input);
            const backtrack = packPredicates(ruleRecord.backtrack);
            const lookahead = packPredicates(ruleRecord.lookahead);
            const action = packAction(ruleRecord.action);
            const slot = ruleOrigin + index * COMPUTE_PROGRAM_RULE_WORDS;
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
            let rowLength = 0;
            for (let index = start; index < end; index++) {
                rowLength = Math.max(rowLength, (lookup.rules[index]!.input[1]?.classId ?? 0) + 1);
            }
            const row = new Array<number>(rowLength).fill(0);
            for (let index = start; index < end; index++) {
                const cls = lookup.rules[index]?.input[1]?.classId ?? 0;
                if (row[cls] === 0) {
                    row[cls] = index + 1;
                }
            }
            const origin = append(row);
            for (let index = start; index < end; index++) {
                const slot = ruleOrigin + index * COMPUTE_PROGRAM_RULE_WORDS;
                words[slot + 10] = origin;
                words[slot + 11] = classIndex(input[1].classDef);
                words[slot + 12] = row.length;
                words[slot + 13] = end - start;
            }
            start = end;
        }
        const base = (lookup.table === "GSUB" ? gsubDescriptorOffset : gposDescriptorOffset) + lookup.lookupIndex * COMPUTE_PROGRAM_LOOKUP_WORDS;
        words[base + COMPUTE_LOOKUP_KIND] = lookup.lookupType;
        words[base + COMPUTE_LOOKUP_FLAG] = lookup.lookupFlag | (lookup.markFilteringSet << 16);
        const independent = lookup.rules.every((rule) => rule.backtrack.length === 0 && rule.lookahead.length === 0);
        words[base + COMPUTE_LOOKUP_MEMBERSHIP] =
            lookup.table === "GSUB" && independent && lookup.rules.every((rule) => rule.input.length === 1 && (rule.mode === 1 || rule.mode === 5))
                ? 1
                : lookup.table === "GPOS" &&
                    independent &&
                    lookup.rules.every(
                        (rule) =>
                            rule.input.length === 2 &&
                            (rule.mode === 3 || rule.mode === 4 || rule.mode === 5) &&
                            (rule.action.kind === "reject" || ((rule.action.kind === "pair" || rule.action.kind === "class-pair") && !rule.action.format2Nonzero))
                    )
                  ? lookup.rules.every(
                        (rule) =>
                            rule.action.kind === "reject" ||
                            ((rule.action.kind === "pair" || rule.action.kind === "class-pair") &&
                                !applicableValue(rule.action.second) &&
                                rule.action.first.xPlacement === 0 &&
                                rule.action.first.yPlacement === 0)
                    )
                      ? 3
                      : 2
                  : 0;
        words[base + COMPUTE_LOOKUP_RULE_ORIGIN] = ruleOrigin;
        words[base + COMPUTE_LOOKUP_RULE_COUNT] = lookup.rules.length;
        words[base + COMPUTE_LOOKUP_EXPANSION] = lookup.expansion;
        const glyphCount = plan.font.glyphCount;
        const firsts: (readonly number[] | null)[] = new Array(lookup.rules.length);
        const counts = new Uint32Array(glyphCount);
        let total = 0;
        for (let index = 0; index < lookup.rules.length; index++) {
            const first = lookup.rules[index]!.input[0];
            const glyphs = first ? firstGlyphs(first) : null;
            firsts[index] = glyphs;
            if (glyphs) {
                for (const glyph of glyphs) {
                    if (glyph >= 0 && glyph < glyphCount && (glyph | 0) === glyph) {
                        counts[glyph]!++;
                        total++;
                    }
                }
            }
        }
        const candidateOrigin = reserve(glyphCount * 2);
        let cursor = reserve(total);
        const next = new Uint32Array(glyphCount);
        for (let glyph = 0; glyph < glyphCount; glyph++) {
            words[candidateOrigin + glyph * 2] = cursor;
            words[candidateOrigin + glyph * 2 + 1] = counts[glyph]!;
            next[glyph] = cursor;
            cursor += counts[glyph]!;
        }
        for (let index = 0; index < firsts.length; index++) {
            const glyphs = firsts[index];
            if (glyphs) {
                for (const glyph of glyphs) {
                    if (glyph >= 0 && glyph < glyphCount && (glyph | 0) === glyph) {
                        words[next[glyph]!++] = index;
                    }
                }
            }
        }
        words[base + COMPUTE_LOOKUP_CANDIDATES] = candidateOrigin;
        const orderedModes = lookup.table === "GSUB" ? [1, 2, 5, 9, 11] : [2, 3, 4, 5, 7, 8, 10, 12];
        const ordered = lookup.rules.every((rule) => orderedModes.includes(rule.mode)) ? (lookup.expansion > 1 ? 2 : 1) : 0;
        words[base + COMPUTE_LOOKUP_ORDERED] = ordered | (ordered && lookup.table === "GSUB" && lookup.rules.every((rule) => rule.backtrack.length === 0) ? 4 : 0);
    }
    const presenceWords = words[13]!;
    const rootWitnesses = new Uint32Array(plan.font.glyphCount * presenceWords);
    const unconditional = new Uint32Array(presenceWords);
    rootOrder.slice(0, 256).forEach((index, root) => {
        const lookup = lookups[index]!;
        const bit = 1 << (root % 32);
        const word = root >>> 5;
        if (lookup.table !== "GSUB" || lookup.rules.some((rule) => rule.input.length < 2 && rule.backtrack.length === 0 && rule.lookahead.length === 0)) {
            unconditional[word]! |= bit;
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
                const at = glyph * presenceWords + word;
                rootWitnesses[at] = (rootWitnesses[at]! | bit) >>> 0;
            }
        }
    });
    for (let at = 0; at < rootWitnesses.length; at++) {
        rootWitnesses[at] = (rootWitnesses[at]! | unconditional[at % presenceWords]!) >>> 0;
    }

    function valueWords(value: ComputeProgramValue): readonly number[] {
        return [value.xPlacement, value.yPlacement, value.xAdvance, 0];
    }
    const profile = plan._profileData?.tables();
    if (profile?.length) {
        words[14] = append(profile);
    }
    return { words: words.slice(0, length), rootWitnesses, gsubDescriptorOffset, gposDescriptorOffset, classTables: classDefs };
}

function capacity(lookups: readonly ComputeProgramLookup[], table: "GSUB" | "GPOS"): number {
    const indexes = lookups.filter((lookup) => lookup.table === table).map((lookup) => lookup.lookupIndex);
    return indexes.length === 0 ? 0 : Math.max(...indexes) + 1;
}
