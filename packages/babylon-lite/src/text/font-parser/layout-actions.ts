/** Decode selected GSUB/GPOS subtables into checked action records.
 *
 *  Unknown formats throw. A parsed action the compiler does not execute stays in the
 *  record so the caller can fail closed instead of skipping it. */

import type { ComputeFont } from "../compute-font.js";
import { ensureComputeFontLayout } from "../compute-font.js";
import type { GlyphClassDef, GlyphCoverage } from "./opentype-layout.js";
import { parseClassDef, parseCoverage } from "./opentype-layout.js";
import type { ByteCursor } from "./reader.js";
import { checkedOffset, openCursor, readI16, readU16, requireCount, skip } from "./reader.js";

const GSUB_EXTENSION = 7;
const GPOS_EXTENSION = 9;
const VARIATION_DELTA_FORMAT = 0x8000;

export interface VariationDeviceIndex {
    readonly outer: number;
    readonly inner: number;
}

export interface ParsedValue {
    readonly xPlacement: number;
    readonly yPlacement: number;
    readonly xAdvance: number;
    readonly yAdvance: number;
    readonly hasDevice: boolean;
    readonly deviceKind: "none" | "variation" | "ppem";
    /** VariationIndex entries. Empty when the only devices are ppem or absent. */
    readonly variationIndexes: readonly VariationDeviceIndex[];
}

export interface ParsedAnchor {
    readonly x: number;
    readonly y: number;
    readonly hasDevice: boolean;
    readonly contourPoint: boolean;
    readonly variationIndexes: readonly VariationDeviceIndex[];
}

export interface ParsedLookupRecord {
    readonly sequenceIndex: number;
    readonly lookupListIndex: number;
}

export interface ParsedGlyphPredicate {
    readonly kind: "glyphs" | "class";
    readonly ids: readonly number[];
    readonly classDef: GlyphClassDef | null;
    readonly classId: number;
    readonly coverage: GlyphCoverage | null;
}

export interface ParsedContextRule {
    readonly input: readonly ParsedGlyphPredicate[];
    readonly backtrack: readonly ParsedGlyphPredicate[];
    readonly lookahead: readonly ParsedGlyphPredicate[];
    readonly lookups: readonly ParsedLookupRecord[];
}

export type ParsedSubtable =
    | { readonly kind: "single"; readonly format: 1 | 2; readonly coverage: GlyphCoverage; readonly delta: number; readonly substitutes: Uint16Array | null }
    | { readonly kind: "multiple"; readonly coverage: GlyphCoverage; readonly sequences: readonly (readonly number[])[] }
    | { readonly kind: "alternate"; readonly coverage: GlyphCoverage; readonly alternatives: readonly (readonly number[])[] }
    | { readonly kind: "ligature"; readonly coverage: GlyphCoverage; readonly sets: readonly (readonly { readonly glyph: number; readonly components: readonly number[] }[])[] }
    | { readonly kind: "context"; readonly chaining: boolean; readonly format: 1 | 2 | 3; readonly rules: readonly ParsedContextRule[] }
    | { readonly kind: "single-adjustment"; readonly coverage: GlyphCoverage; readonly values: readonly ParsedValue[] }
    | {
          readonly kind: "pair";
          readonly format: 1;
          readonly coverage: GlyphCoverage;
          readonly valueFormat1: number;
          readonly valueFormat2: number;
          readonly pairSets: readonly (readonly { readonly secondGlyph: number; readonly value1: ParsedValue; readonly value2: ParsedValue }[])[];
      }
    | {
          readonly kind: "class-pair";
          readonly coverage: GlyphCoverage;
          readonly valueFormat1: number;
          readonly valueFormat2: number;
          readonly classDef1: GlyphClassDef;
          readonly classDef2: GlyphClassDef;
          readonly class2Count: number;
          readonly rows: readonly (readonly { readonly value1: ParsedValue; readonly value2: ParsedValue }[])[];
      }
    | {
          readonly kind: "mark-ligature";
          readonly markClassCount: number;
          readonly marks: readonly { readonly glyphId: number; readonly markClass: number; readonly anchor: ParsedAnchor }[];
          readonly ligatures: readonly { readonly glyphId: number; readonly components: readonly (readonly (ParsedAnchor | null)[])[] }[];
      }
    | {
          readonly kind: "mark-base" | "mark-mark";
          readonly markCoverage: GlyphCoverage;
          readonly baseCoverage: GlyphCoverage;
          readonly markClassCount: number;
          readonly marks: readonly { readonly glyphId: number; readonly markClass: number; readonly anchor: ParsedAnchor }[];
          readonly bases: readonly { readonly glyphId: number; readonly anchors: readonly (ParsedAnchor | null)[] }[];
      }
    | { readonly kind: "unimplemented"; readonly lookupType: number; readonly format: number; readonly coverage: GlyphCoverage | null; readonly reason: string };

export interface ParsedLookupActions {
    readonly table: "GSUB" | "GPOS";
    readonly lookupIndex: number;
    readonly lookupType: number;
    readonly lookupFlag: number;
    readonly markFilteringSet: number | null;
    readonly subtables: readonly ParsedSubtable[];
}

let actionsByFont: WeakMap<ComputeFont, Map<string, ParsedLookupActions>> | null = null;

export function parseLookupActions(font: ComputeFont, table: "GSUB" | "GPOS", lookupIndex: number): ParsedLookupActions {
    const key = `${table}/${lookupIndex}`;
    let actions = actionsByFont?.get(font);
    const existing = actions?.get(key);
    if (existing) {
        return existing;
    }
    const layout = ensureComputeFontLayout(font);
    const source = table === "GSUB" ? layout.gsub : layout.gpos;
    const lookup = source?.lookups[lookupIndex];
    if (!source || !lookup) {
        throw new Error(`compileComputeTextProgram: ${table} lookup ${lookupIndex} is outside the lookup list`);
    }
    if (lookup.unsupported) {
        throw new Error(`compileComputeTextProgram: ${table} lookup ${lookupIndex} mixes subtable types and cannot be matched`);
    }
    const tableRecord = font._directory.tables.get(table);
    if (!tableRecord) {
        throw new Error(`compileComputeTextProgram: ${table} table is missing`);
    }
    const origin = lookupOrigin(font._bytes, tableRecord.offset, tableRecord.length, lookupIndex, table);
    const limit = tableRecord.offset + tableRecord.length;
    const extensionType = table === "GSUB" ? GSUB_EXTENSION : GPOS_EXTENSION;
    const subtables = lookup.subtableOffsets.map((offset, index) => {
        const resolved = resolveSubtable(font._bytes, origin, limit, offset, lookup.lookupType, extensionType, `${table} lookup ${lookupIndex} subtable ${index}`);
        return parseSubtable(font._bytes, resolved.start, limit, resolved.type, table, `${table} lookup ${lookupIndex} subtable ${index}`);
    });
    const parsed: ParsedLookupActions = {
        table,
        lookupIndex,
        lookupType: lookup.resolvedType,
        lookupFlag: lookup.lookupFlag,
        markFilteringSet: lookup.markFilteringSet,
        subtables,
    };
    if (!actions) {
        actions = new Map();
        (actionsByFont ??= new WeakMap()).set(font, actions);
    }
    actions.set(key, parsed);
    return parsed;
}

function lookupOrigin(bytes: Uint8Array, tableOffset: number, tableLength: number, lookupIndex: number, table: string): number {
    const header = openCursor(bytes, tableOffset, tableLength, table);
    skip(header, 8, `${table} header`);
    const lookupListOffset = readU16(header, `${table} lookupListOffset`);
    const list = openCursor(bytes, tableOffset + lookupListOffset, tableLength - lookupListOffset, `${table} LookupList`);
    const count = readU16(list, `${table} lookupCount`);
    if (lookupIndex >= count) {
        throw new Error(`compileComputeTextProgram: ${table} lookup ${lookupIndex} is outside the lookup list`);
    }
    skip(list, lookupIndex * 2, `${table} lookup offsets`);
    const offset = readU16(list, `${table} lookupOffset`);
    if (offset === 0) {
        throw new Error(`compileComputeTextProgram: ${table} lookup ${lookupIndex} offset is zero`);
    }
    return tableOffset + lookupListOffset + offset;
}

function resolveSubtable(
    bytes: Uint8Array,
    lookupStart: number,
    limit: number,
    subtableOffset: number,
    lookupType: number,
    extensionType: number,
    label: string
): { type: number; start: number } {
    const start = lookupStart + subtableOffset;
    if (lookupType !== extensionType) {
        return { type: lookupType, start };
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    const format = readU16(cursor, `${label} extension format`);
    if (format !== 1) {
        throw new Error(`${label}: unsupported extension format ${format}`);
    }
    const innerType = readU16(cursor, `${label} extensionLookupType`);
    const extensionOffset = readU32(cursor, `${label} extensionOffset`);
    if (innerType === extensionType || extensionOffset < 8 || start + extensionOffset + 2 > limit) {
        throw new Error(`${label}: extension target is outside the lookup`);
    }
    return { type: innerType, start: start + extensionOffset };
}

function readU32(cursor: ByteCursor, label: string): number {
    if (cursor.offset + 4 > cursor.limit) {
        throw new Error(`${label}: truncated`);
    }
    const at = cursor.offset;
    cursor.offset += 4;
    const bytes = cursor.bytes;
    return ((bytes[at] ?? 0) * 0x1000000 + (bytes[at + 1] ?? 0) * 0x10000 + (bytes[at + 2] ?? 0) * 0x100 + (bytes[at + 3] ?? 0)) >>> 0;
}

function parseSubtable(bytes: Uint8Array, start: number, limit: number, lookupType: number, table: "GSUB" | "GPOS", label: string): ParsedSubtable {
    const format = readU16(openCursor(bytes, start, limit - start, label), `${label} format`);
    if (table === "GSUB") {
        return parseGsub(bytes, start, limit, lookupType, format, label);
    }
    return parseGpos(bytes, start, limit, lookupType, format, label);
}

function parseGsub(bytes: Uint8Array, start: number, limit: number, lookupType: number, format: number, label: string): ParsedSubtable {
    if (lookupType === 1) {
        return parseSingle(bytes, start, limit, format, label);
    }
    if (lookupType === 2) {
        return parseMultiple(bytes, start, limit, format, label);
    }
    if (lookupType === 3) {
        return parseAlternate(bytes, start, limit, format, label);
    }
    if (lookupType === 4) {
        return parseLigature(bytes, start, limit, format, label);
    }
    if (lookupType === 5 || lookupType === 6) {
        return parseContext(bytes, start, limit, format, lookupType === 6, label);
    }
    if (lookupType === 8) {
        const coverage = readOffsetCoverage(bytes, start, limit, 2, label);
        return { kind: "unimplemented", lookupType, format, coverage, reason: "GSUB reverse chaining is not executed" };
    }
    throw new Error(`${label}: unknown GSUB lookup type ${lookupType}`);
}

function parseGpos(bytes: Uint8Array, start: number, limit: number, lookupType: number, format: number, label: string): ParsedSubtable {
    if (lookupType === 1) {
        return parseSingleAdjustment(bytes, start, limit, format, label);
    }
    if (lookupType === 2) {
        return parsePair(bytes, start, limit, format, label);
    }
    if (lookupType === 4 || lookupType === 6) {
        return parseMark(bytes, start, limit, format, lookupType === 6 ? "mark-mark" : "mark-base", label);
    }
    if (lookupType === 5) {
        return parseMarkLigature(bytes, start, limit, format, label);
    }
    if (lookupType === 7 || lookupType === 8) {
        return parseContext(bytes, start, limit, format, lookupType === 8, label);
    }
    if (lookupType === 3) {
        const coverage = readableCoverage(bytes, start, limit, label);
        return { kind: "unimplemented", lookupType, format, coverage, reason: `GPOS lookup type ${lookupType} is not executed` };
    }
    throw new Error(`${label}: unknown GPOS lookup type ${lookupType}`);
}

function parseSingle(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    if (format === 1) {
        return { kind: "single", format: 1, coverage, delta: readI16(cursor, `${label} deltaGlyphID`), substitutes: null };
    }
    if (format === 2) {
        const count = readU16(cursor, `${label} glyphCount`);
        requireCount(count, 2, cursor.limit - cursor.offset, label);
        const substitutes = new Uint16Array(count);
        for (let index = 0; index < count; index++) {
            substitutes[index] = readU16(cursor, `${label} substitute`);
        }
        return { kind: "single", format: 2, coverage, delta: 0, substitutes };
    }
    throw new Error(`${label}: unknown single substitution format ${format}`);
}

function parseMultiple(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    if (format !== 1) {
        throw new Error(`${label}: unknown multiple substitution format ${format}`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    const count = readU16(cursor, `${label} sequenceCount`);
    requireCount(count, 2, cursor.limit - cursor.offset, label);
    const offsets: number[] = [];
    for (let index = 0; index < count; index++) {
        offsets.push(readU16(cursor, `${label} sequenceOffset`));
    }
    const sequences = offsets.map((offset) => readGlyphList(bytes, start + offset, limit, label));
    return { kind: "multiple", coverage, sequences };
}

function parseAlternate(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    if (format !== 1) {
        throw new Error(`${label}: unknown alternate substitution format ${format}`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    const count = readU16(cursor, `${label} alternateSetCount`);
    requireCount(count, 2, cursor.limit - cursor.offset, label);
    const offsets: number[] = [];
    for (let index = 0; index < count; index++) {
        offsets.push(readU16(cursor, `${label} alternateSetOffset`));
    }
    const alternatives = offsets.map((offset) => readGlyphList(bytes, start + offset, limit, label));
    return { kind: "alternate", coverage, alternatives };
}

function parseLigature(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    if (format !== 1) {
        throw new Error(`${label}: unknown ligature format ${format}`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    const setCount = readU16(cursor, `${label} ligatureSetCount`);
    requireCount(setCount, 2, cursor.limit - cursor.offset, label);
    const setOffsets: number[] = [];
    for (let index = 0; index < setCount; index++) {
        setOffsets.push(readU16(cursor, `${label} ligatureSetOffset`));
    }
    const sets = setOffsets.map((setOffset) => {
        const set = openCursor(bytes, start + setOffset, limit - (start + setOffset), label);
        const ligatureCount = readU16(set, `${label} ligatureCount`);
        requireCount(ligatureCount, 2, set.limit - set.offset, label);
        const ligatureOffsets: number[] = [];
        for (let index = 0; index < ligatureCount; index++) {
            ligatureOffsets.push(readU16(set, `${label} ligatureOffset`));
        }
        return ligatureOffsets.map((ligatureOffset) => {
            const ligature = openCursor(bytes, start + setOffset + ligatureOffset, limit - (start + setOffset + ligatureOffset), label);
            const glyph = readU16(ligature, `${label} ligatureGlyph`);
            const componentCount = readU16(ligature, `${label} componentCount`);
            if (componentCount < 1) {
                throw new Error(`${label}: ligature componentCount is zero`);
            }
            requireCount(componentCount - 1, 2, ligature.limit - ligature.offset, label);
            const components: number[] = [];
            for (let index = 1; index < componentCount; index++) {
                components.push(readU16(ligature, `${label} component`));
            }
            return { glyph, components };
        });
    });
    return { kind: "ligature", coverage, sets };
}

function parseContext(bytes: Uint8Array, start: number, limit: number, format: number, chaining: boolean, label: string): ParsedSubtable {
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    if (format === 3) {
        const backtrack = chaining ? readCoverageList(bytes, start, limit, cursor, label) : [];
        const input = chaining
            ? { coverages: readCoverageList(bytes, start, limit, cursor, label), lookups: [] as ParsedLookupRecord[] }
            : readContext3Input(bytes, start, limit, cursor, label);
        const lookahead = chaining ? readCoverageList(bytes, start, limit, cursor, label) : [];
        const lookups = chaining ? readLookupRecords(cursor, label) : input.lookups;
        if (input.coverages.length === 0) {
            throw new Error(`${label}: contextual input is empty`);
        }
        return {
            kind: "context",
            chaining,
            format: 3,
            rules: [{ input: input.coverages.map(glyphPredicate), backtrack: backtrack.map(glyphPredicate), lookahead: lookahead.map(glyphPredicate), lookups }],
        };
    }
    if (format !== 1 && format !== 2) {
        throw new Error(`${label}: unknown contextual format ${format}`);
    }
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    const backtrackClass = format === 2 && chaining ? readOptionalClass(bytes, start, limit, readU16(cursor, `${label} backtrackClass`), label) : null;
    const inputClass = format === 2 ? readOptionalClass(bytes, start, limit, readU16(cursor, `${label} inputClass`), label) : null;
    const lookaheadClass = format === 2 && chaining ? readOptionalClass(bytes, start, limit, readU16(cursor, `${label} lookaheadClass`), label) : null;
    const setCount = readU16(cursor, `${label} ruleSetCount`);
    requireCount(setCount, 2, cursor.limit - cursor.offset, label);
    const setOffsets: number[] = [];
    for (let index = 0; index < setCount; index++) {
        setOffsets.push(readU16(cursor, `${label} ruleSetOffset`));
    }
    const rules: ParsedContextRule[] = [];
    setOffsets.forEach((setOffset, setIndex) => {
        if (setOffset === 0) {
            return;
        }
        const ruleSet = openCursor(bytes, start + setOffset, limit - (start + setOffset), label);
        const ruleCount = readU16(ruleSet, `${label} ruleCount`);
        requireCount(ruleCount, 2, ruleSet.limit - ruleSet.offset, label);
        const ruleOffsets: number[] = [];
        for (let index = 0; index < ruleCount; index++) {
            ruleOffsets.push(readU16(ruleSet, `${label} ruleOffset`));
        }
        for (const ruleOffset of ruleOffsets) {
            const rule = openCursor(bytes, start + setOffset + ruleOffset, limit - (start + setOffset + ruleOffset), label);
            if (!chaining) {
                // Context formats 1/2 place recordCount before input[], unlike ChainContext.
                const glyphCount = readU16(rule, `${label} glyphCount`);
                const recordCount = readU16(rule, `${label} lookupCount`);
                if (glyphCount === 0) {
                    throw new Error(`${label}: contextual input is empty`);
                }
                const first = format === 1 ? glyphPredicate(coverageGlyphsAt(coverage, setIndex)) : classPredicate(inputClass, setIndex, coverage);
                const input = [first];
                for (let i = 1; i < glyphCount; i++) {
                    const value = readU16(rule, `${label} input glyph or class`);
                    input.push(inputClass ? classPredicate(inputClass, value, null) : { kind: "glyphs", ids: [value], classDef: null, classId: 0, coverage: null });
                }
                rules.push({ input, backtrack: [], lookahead: [], lookups: readKnownLookupRecords(rule, recordCount, label) });
                continue;
            }
            const backtrack = chaining ? readRuleGlyphs(rule, backtrackClass, false, label) : [];
            const inputRest = readRuleGlyphs(rule, inputClass, true, label);
            const lookahead = chaining ? readRuleGlyphs(rule, lookaheadClass, false, label) : [];
            const first = format === 1 ? glyphPredicate(coverageGlyphsAt(coverage, setIndex)) : classPredicate(inputClass, setIndex, coverage);
            rules.push({ input: [first, ...inputRest], backtrack, lookahead, lookups: readLookupRecords(rule, label) });
        }
    });
    return { kind: "context", chaining, format, rules };
}

function parseSingleAdjustment(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    const valueFormat = readU16(cursor, `${label} valueFormat`);
    if (format === 1) {
        return { kind: "single-adjustment", coverage, values: [readValue(cursor, valueFormat, start, limit, label)] };
    }
    if (format === 2) {
        const count = readU16(cursor, `${label} valueCount`);
        const values: ParsedValue[] = [];
        for (let index = 0; index < count; index++) {
            values.push(readValue(cursor, valueFormat, start, limit, label));
        }
        return { kind: "single-adjustment", coverage, values };
    }
    throw new Error(`${label}: unknown single adjustment format ${format}`);
}

function parsePair(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const coverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} coverage`), label);
    const valueFormat1 = readU16(cursor, `${label} valueFormat1`);
    const valueFormat2 = readU16(cursor, `${label} valueFormat2`);
    if (format === 1) {
        const setCount = readU16(cursor, `${label} pairSetCount`);
        requireCount(setCount, 2, cursor.limit - cursor.offset, label);
        const setOffsets: number[] = [];
        for (let index = 0; index < setCount; index++) {
            setOffsets.push(readU16(cursor, `${label} pairSetOffset`));
        }
        const pairSets = setOffsets.map((setOffset) => {
            const set = openCursor(bytes, start + setOffset, limit - (start + setOffset), label);
            const pairCount = readU16(set, `${label} pairValueCount`);
            const pairs: { secondGlyph: number; value1: ParsedValue; value2: ParsedValue }[] = [];
            for (let index = 0; index < pairCount; index++) {
                const secondGlyph = readU16(set, `${label} secondGlyph`);
                pairs.push({
                    secondGlyph,
                    value1: readValue(set, valueFormat1, start, limit, label),
                    value2: readValue(set, valueFormat2, start, limit, label),
                });
            }
            return pairs;
        });
        return { kind: "pair", format: 1, coverage, valueFormat1, valueFormat2, pairSets };
    }
    if (format === 2) {
        const classDef1 = parseClassDef(bytes, start + readU16(cursor, `${label} classDef1`), limit);
        const classDef2 = parseClassDef(bytes, start + readU16(cursor, `${label} classDef2`), limit);
        const class1Count = readU16(cursor, `${label} class1Count`);
        const class2Count = readU16(cursor, `${label} class2Count`);
        const rows: { value1: ParsedValue; value2: ParsedValue }[][] = [];
        for (let row = 0; row < class1Count; row++) {
            const cells: { value1: ParsedValue; value2: ParsedValue }[] = [];
            for (let column = 0; column < class2Count; column++) {
                cells.push({
                    value1: readValue(cursor, valueFormat1, start, limit, label),
                    value2: readValue(cursor, valueFormat2, start, limit, label),
                });
            }
            rows.push(cells);
        }
        return { kind: "class-pair", coverage, valueFormat1, valueFormat2, classDef1, classDef2, class2Count, rows };
    }
    throw new Error(`${label}: unknown pair format ${format}`);
}

function parseMarkLigature(bytes: Uint8Array, start: number, limit: number, format: number, label: string): ParsedSubtable {
    if (format !== 1) {
        throw new Error(`${label}: unknown mark-to-ligature format ${format}`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const markCoverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} markCoverage`), label);
    const ligatureCoverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} ligatureCoverage`), label);
    const markClassCount = readU16(cursor, `${label} markClassCount`);
    const markArrayOffset = readU16(cursor, `${label} markArray`);
    const ligatureArrayOffset = readU16(cursor, `${label} ligatureArray`);
    return {
        kind: "mark-ligature",
        markClassCount,
        marks: readMarkArray(bytes, start + markArrayOffset, limit, markCoverage, label),
        ligatures: readLigatureArray(bytes, start + ligatureArrayOffset, limit, ligatureCoverage, markClassCount, label),
    };
}

function readLigatureArray(
    bytes: Uint8Array,
    start: number,
    limit: number,
    coverage: GlyphCoverage,
    classCount: number,
    label: string
): { glyphId: number; components: (ParsedAnchor | null)[][] }[] {
    const cursor = openCursor(bytes, start, limit - start, label);
    const count = readU16(cursor, `${label} ligatureCount`);
    const glyphs = expandCoverage(coverage);
    if (glyphs.length !== count) {
        throw new Error(`${label}: ligature coverage count ${glyphs.length} does not match ligature array ${count}`);
    }
    const offsets: number[] = [];
    for (let index = 0; index < count; index++) {
        offsets.push(readU16(cursor, `${label} ligatureAttach`));
    }
    return offsets.map((offset, index) => {
        const attachStart = start + offset;
        const attach = openCursor(bytes, attachStart, limit - attachStart, label);
        const componentCount = readU16(attach, `${label} componentCount`);
        const components: (ParsedAnchor | null)[][] = [];
        for (let component = 0; component < componentCount; component++) {
            const anchors: (ParsedAnchor | null)[] = [];
            for (let cls = 0; cls < classCount; cls++) {
                const anchorOffset = readU16(attach, `${label} ligatureAnchor`);
                anchors.push(anchorOffset === 0 ? null : readAnchor(bytes, attachStart + anchorOffset, limit, label));
            }
            components.push(anchors);
        }
        return { glyphId: glyphs[index] ?? 0, components };
    });
}

function parseMark(bytes: Uint8Array, start: number, limit: number, format: number, kind: "mark-base" | "mark-mark", label: string): ParsedSubtable {
    if (format !== 1) {
        throw new Error(`${label}: unknown mark format ${format}`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    readU16(cursor, `${label} format`);
    const markCoverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} markCoverage`), label);
    const baseCoverage = readCoverage(bytes, start, limit, readU16(cursor, `${label} baseCoverage`), label);
    const markClassCount = readU16(cursor, `${label} markClassCount`);
    const markArrayOffset = readU16(cursor, `${label} markArray`);
    const baseArrayOffset = readU16(cursor, `${label} baseArray`);
    const marks = readMarkArray(bytes, start + markArrayOffset, limit, markCoverage, label);
    const bases = readBaseArray(bytes, start + baseArrayOffset, limit, baseCoverage, markClassCount, label);
    return { kind, markCoverage, baseCoverage, markClassCount, marks, bases };
}

function readMarkArray(bytes: Uint8Array, start: number, limit: number, coverage: GlyphCoverage, label: string): { glyphId: number; markClass: number; anchor: ParsedAnchor }[] {
    const cursor = openCursor(bytes, start, limit - start, label);
    const count = readU16(cursor, `${label} markCount`);
    const glyphs = expandCoverage(coverage);
    if (glyphs.length !== count) {
        throw new Error(`${label}: mark coverage count ${glyphs.length} does not match mark array ${count}`);
    }
    const marks: { glyphId: number; markClass: number; anchor: ParsedAnchor }[] = [];
    for (let index = 0; index < count; index++) {
        const markClass = readU16(cursor, `${label} markClass`);
        const anchorOffset = readU16(cursor, `${label} markAnchor`);
        marks.push({ glyphId: glyphs[index] ?? 0, markClass, anchor: readAnchor(bytes, start + anchorOffset, limit, label) });
    }
    return marks;
}

function readBaseArray(
    bytes: Uint8Array,
    start: number,
    limit: number,
    coverage: GlyphCoverage,
    classCount: number,
    label: string
): { glyphId: number; anchors: (ParsedAnchor | null)[] }[] {
    const cursor = openCursor(bytes, start, limit - start, label);
    const count = readU16(cursor, `${label} baseCount`);
    const glyphs = expandCoverage(coverage);
    if (glyphs.length !== count) {
        throw new Error(`${label}: base coverage count ${glyphs.length} does not match base array ${count}`);
    }
    const bases: { glyphId: number; anchors: (ParsedAnchor | null)[] }[] = [];
    for (let index = 0; index < count; index++) {
        const anchors: (ParsedAnchor | null)[] = [];
        for (let cls = 0; cls < classCount; cls++) {
            const offset = readU16(cursor, `${label} baseAnchor`);
            anchors.push(offset === 0 ? null : readAnchor(bytes, start + offset, limit, label));
        }
        bases.push({ glyphId: glyphs[index] ?? 0, anchors });
    }
    return bases;
}

function readAnchor(bytes: Uint8Array, start: number, limit: number, label: string): ParsedAnchor {
    const cursor = openCursor(bytes, start, limit - start, label);
    const format = readU16(cursor, `${label} anchor format`);
    const x = readI16(cursor, `${label} anchor x`);
    const y = readI16(cursor, `${label} anchor y`);
    if (format === 1) {
        return { x, y, hasDevice: false, contourPoint: false, variationIndexes: [] };
    }
    if (format === 2) {
        return { x, y, hasDevice: false, contourPoint: true, variationIndexes: [] };
    }
    if (format === 3) {
        const xDevice = readU16(cursor, `${label} xDevice`);
        const yDevice = readU16(cursor, `${label} yDevice`);
        const variationIndexes: VariationDeviceIndex[] = [];
        if (xDevice !== 0) {
            collectVariation(classifyDevice(bytes, start + xDevice, limit, label), variationIndexes);
        }
        if (yDevice !== 0) {
            collectVariation(classifyDevice(bytes, start + yDevice, limit, label), variationIndexes);
        }
        return { x, y, hasDevice: xDevice !== 0 || yDevice !== 0, contourPoint: false, variationIndexes };
    }
    throw new Error(`${label}: unknown anchor format ${format}`);
}

function readValue(cursor: ByteCursor, format: number, subtableStart: number, limit: number, label: string): ParsedValue {
    if (format & ~0x00ff) {
        throw new Error(`${label}: unknown value format 0x${format.toString(16)}`);
    }
    const xPlacement = format & 0x0001 ? readI16(cursor, `${label} xPlacement`) : 0;
    const yPlacement = format & 0x0002 ? readI16(cursor, `${label} yPlacement`) : 0;
    const xAdvance = format & 0x0004 ? readI16(cursor, `${label} xAdvance`) : 0;
    const yAdvance = format & 0x0008 ? readI16(cursor, `${label} yAdvance`) : 0;
    let hasDevice = false;
    let deviceKind: "none" | "variation" | "ppem" = "none";
    const variationIndexes: VariationDeviceIndex[] = [];
    for (let bit = 0x0010; bit <= 0x0080; bit *= 2) {
        if (!(format & bit)) {
            continue;
        }
        const offset = readU16(cursor, `${label} device`);
        if (offset === 0) {
            continue;
        }
        hasDevice = true;
        const device = classifyDevice(cursor.bytes, subtableStart + offset, limit, label);
        if (device.kind === "variation") {
            deviceKind = "variation";
            variationIndexes.push({ outer: device.outer, inner: device.inner });
        } else if (deviceKind === "none") {
            deviceKind = "ppem";
        }
    }
    return { xPlacement, yPlacement, xAdvance, yAdvance, hasDevice, deviceKind, variationIndexes };
}

function classifyDevice(bytes: Uint8Array, start: number, limit: number, label: string): { kind: "variation" | "ppem"; outer: number; inner: number } {
    if (start + 6 > limit) {
        throw new Error(`${label}: device table is outside the subtable`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    const outer = readU16(cursor, `${label} device start`);
    const inner = readU16(cursor, `${label} device end`);
    const format = readU16(cursor, `${label} device format`);
    return format === VARIATION_DELTA_FORMAT ? { kind: "variation", outer, inner } : { kind: "ppem", outer: 0, inner: 0 };
}

function collectVariation(device: { kind: "variation" | "ppem"; outer: number; inner: number }, indexes: VariationDeviceIndex[]): void {
    if (device.kind === "variation") {
        indexes.push({ outer: device.outer, inner: device.inner });
    }
}

function readCoverage(bytes: Uint8Array, subtableStart: number, limit: number, offset: number, label: string): GlyphCoverage {
    if (offset === 0) {
        throw new Error(`${label}: coverage offset is zero`);
    }
    return parseCoverage(bytes, subtableStart + offset, limit, label);
}

function readOffsetCoverage(bytes: Uint8Array, start: number, limit: number, headerBytes: number, label: string): GlyphCoverage | null {
    if (start + headerBytes + 2 > limit) {
        return null;
    }
    const offset = readU16(openCursor(bytes, start + headerBytes, limit - start - headerBytes, label), `${label} coverage`);
    return offset === 0 ? null : readCoverage(bytes, start, limit, offset, label);
}

function readableCoverage(bytes: Uint8Array, start: number, limit: number, label: string): GlyphCoverage | null {
    try {
        return readOffsetCoverage(bytes, start, limit, 2, label);
    } catch {
        throw new Error(`${label}: coverage cannot be matched`);
    }
}

function readContext3Input(
    bytes: Uint8Array,
    subtableStart: number,
    limit: number,
    cursor: ByteCursor,
    label: string
): { coverages: GlyphCoverage[]; lookups: ParsedLookupRecord[] } {
    const glyphCount = readU16(cursor, `${label} glyphCount`);
    const recordCount = readU16(cursor, `${label} lookupCount`);
    requireCount(glyphCount, 2, cursor.limit - cursor.offset, label);
    const coverages: GlyphCoverage[] = [];
    for (let index = 0; index < glyphCount; index++) {
        coverages.push(readCoverage(bytes, subtableStart, limit, readU16(cursor, `${label} coverageOffset`), label));
    }
    return { coverages, lookups: readKnownLookupRecords(cursor, recordCount, label) };
}

function readCoverageList(bytes: Uint8Array, subtableStart: number, limit: number, cursor: ByteCursor, label: string): GlyphCoverage[] {
    const count = readU16(cursor, `${label} coverageCount`);
    requireCount(count, 2, cursor.limit - cursor.offset, label);
    const coverages: GlyphCoverage[] = [];
    for (let index = 0; index < count; index++) {
        coverages.push(readCoverage(bytes, subtableStart, limit, readU16(cursor, `${label} coverageOffset`), label));
    }
    return coverages;
}

function readLookupRecords(cursor: ByteCursor, label: string): ParsedLookupRecord[] {
    const count = readU16(cursor, `${label} lookupCount`);
    return readKnownLookupRecords(cursor, count, label);
}

function readKnownLookupRecords(cursor: ByteCursor, count: number, label: string): ParsedLookupRecord[] {
    requireCount(count, 4, cursor.limit - cursor.offset, label);
    const records: ParsedLookupRecord[] = [];
    for (let index = 0; index < count; index++) {
        records.push({ sequenceIndex: readU16(cursor, `${label} sequenceIndex`), lookupListIndex: readU16(cursor, `${label} lookupListIndex`) });
    }
    return records;
}

function readGlyphList(bytes: Uint8Array, start: number, limit: number, label: string): number[] {
    const cursor = openCursor(bytes, start, limit - start, label);
    const count = readU16(cursor, `${label} glyphCount`);
    requireCount(count, 2, cursor.limit - cursor.offset, label);
    const glyphs: number[] = [];
    for (let index = 0; index < count; index++) {
        glyphs.push(readU16(cursor, `${label} glyph`));
    }
    return glyphs;
}

function readOptionalClass(bytes: Uint8Array, subtableStart: number, limit: number, offset: number, label: string): GlyphClassDef | null {
    if (offset === 0) {
        return null;
    }
    return parseClassDef(bytes, checkedOffset(subtableStart, offset, label), limit);
}

function readRuleGlyphs(cursor: ByteCursor, classDef: GlyphClassDef | null, omitFirst: boolean, label: string): ParsedGlyphPredicate[] {
    const count = readU16(cursor, `${label} glyphCount`);
    const stored = omitFirst ? count - 1 : count;
    if (stored < 0) {
        throw new Error(`${label}: glyphCount ${count} is invalid`);
    }
    requireCount(stored, 2, cursor.limit - cursor.offset, label);
    const predicates: ParsedGlyphPredicate[] = [];
    for (let index = 0; index < stored; index++) {
        const value = readU16(cursor, `${label} glyph or class`);
        predicates.push(classDef ? classPredicate(classDef, value, null) : { kind: "glyphs", ids: [value], classDef: null, classId: 0, coverage: null });
    }
    return predicates;
}

function glyphPredicate(coverage: GlyphCoverage): ParsedGlyphPredicate {
    return { kind: "glyphs", ids: expandCoverage(coverage), classDef: null, classId: 0, coverage };
}

function classPredicate(classDef: GlyphClassDef | null, classId: number, coverage: GlyphCoverage | null): ParsedGlyphPredicate {
    return { kind: "class", ids: [], classDef, classId, coverage };
}

function coverageGlyphsAt(coverage: GlyphCoverage, index: number): GlyphCoverage {
    const glyphs = expandCoverage(coverage);
    const glyph = glyphs[index];
    if (glyph === undefined) {
        throw new Error("contextual rule set index is outside coverage");
    }
    return { format: 1, glyphs: Uint16Array.of(glyph), ranges: [] };
}

function expandCoverage(coverage: GlyphCoverage): number[] {
    if (coverage.format === 1) {
        return Array.from(coverage.glyphs);
    }
    const glyphs: number[] = [];
    for (const range of coverage.ranges) {
        for (let glyphId = range.start; glyphId <= range.end; glyphId++) {
            glyphs.push(glyphId);
        }
    }
    return glyphs;
}
