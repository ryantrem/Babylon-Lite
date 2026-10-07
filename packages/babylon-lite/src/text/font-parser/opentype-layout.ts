/** Selected GDEF/GSUB/GPOS structure: script, language, feature, and lookup lists, plus nested lookup references. */

import type { ByteCursor } from "./reader.js";
import { checkedOffset, openCursor, readTag, readU16, readU32, requireCount, skip } from "./reader.js";

export interface GlyphCoverage {
    readonly format: 1 | 2;
    readonly glyphs: Uint16Array;
    readonly ranges: readonly { readonly start: number; readonly end: number; readonly startCoverageIndex: number }[];
}

export interface GlyphClassDef {
    readonly format: 1 | 2;
    readonly startGlyphId: number;
    readonly classes: Uint16Array;
    readonly ranges: readonly { readonly start: number; readonly end: number; readonly classValue: number }[];
}

export interface LayoutLangSys {
    readonly requiredFeatureIndex: number;
    readonly featureIndices: readonly number[];
}

export interface LayoutScript {
    readonly tag: string;
    readonly defaultLangSys: LayoutLangSys | null;
    readonly languages: readonly { readonly tag: string; readonly langSys: LayoutLangSys }[];
}

export interface LayoutFeature {
    readonly tag: string;
    readonly lookupListIndices: readonly number[];
}

export interface LayoutLookup {
    readonly lookupType: number;
    readonly resolvedType: number;
    readonly lookupFlag: number;
    readonly markFilteringSet: number | null;
    readonly subtableOffsets: readonly number[];
    readonly nestedLookupIndices: readonly number[];
    readonly unsupported: boolean;
}

export interface LayoutTable {
    readonly kind: "GSUB" | "GPOS";
    readonly versionMajor: number;
    readonly versionMinor: number;
    readonly scripts: readonly LayoutScript[];
    readonly features: readonly LayoutFeature[];
    readonly lookups: readonly LayoutLookup[];
    readonly featureVariationsOffset: number;
}

export interface GdefTable {
    readonly versionMajor: number;
    readonly versionMinor: number;
    readonly glyphClassDef: GlyphClassDef | null;
    readonly markAttachClassDef: GlyphClassDef | null;
    readonly markGlyphSetCount: number;
    readonly itemVariationStoreOffset: number;
}

const USE_MARK_FILTERING_SET = 0x0010;

export function parseGdef(bytes: Uint8Array, tableOffset: number, tableLength: number): GdefTable {
    const cursor = openCursor(bytes, tableOffset, tableLength, "GDEF");
    const versionMajor = readU16(cursor, "GDEF majorVersion");
    const versionMinor = readU16(cursor, "GDEF minorVersion");
    if (versionMajor !== 1) {
        throw new Error(`GDEF: unsupported version ${versionMajor}.${versionMinor}`);
    }
    const glyphClassOffset = readU16(cursor, "GDEF glyphClassDefOffset");
    const attachOffset = readU16(cursor, "GDEF attachListOffset");
    const ligCaretOffset = readU16(cursor, "GDEF ligCaretListOffset");
    const markAttachOffset = readU16(cursor, "GDEF markAttachClassDefOffset");
    const markSetsOffset = versionMinor >= 2 ? readU16(cursor, "GDEF markGlyphSetsDefOffset") : 0;
    const itemVariationStoreOffset = versionMinor >= 3 ? readU32(cursor, "GDEF itemVarStoreOffset") : 0;
    if (attachOffset !== 0) {
        validateAttachOrCaret(bytes, tableOffset, tableLength, attachOffset, "GDEF attachList");
    }
    if (ligCaretOffset !== 0) {
        validateAttachOrCaret(bytes, tableOffset, tableLength, ligCaretOffset, "GDEF ligCaretList");
    }
    if (itemVariationStoreOffset !== 0 && itemVariationStoreOffset >= tableLength) {
        throw new Error("GDEF: item variation store offset is outside the table");
    }
    return {
        versionMajor,
        versionMinor,
        glyphClassDef: glyphClassOffset === 0 ? null : parseClassDef(bytes, checkedOffset(tableOffset, glyphClassOffset, "GDEF classDef"), tableOffset + tableLength),
        markAttachClassDef: markAttachOffset === 0 ? null : parseClassDef(bytes, checkedOffset(tableOffset, markAttachOffset, "GDEF markAttach"), tableOffset + tableLength),
        markGlyphSetCount: markSetsOffset === 0 ? 0 : parseMarkGlyphSets(bytes, tableOffset, tableLength, markSetsOffset),
        itemVariationStoreOffset,
    };
}

export function parseLayoutTable(bytes: Uint8Array, tableOffset: number, tableLength: number, kind: "GSUB" | "GPOS"): LayoutTable {
    const cursor = openCursor(bytes, tableOffset, tableLength, kind);
    const versionMajor = readU16(cursor, `${kind} majorVersion`);
    const versionMinor = readU16(cursor, `${kind} minorVersion`);
    if (versionMajor !== 1) {
        throw new Error(`${kind}: unsupported version ${versionMajor}.${versionMinor}`);
    }
    const scriptOffset = readU16(cursor, `${kind} scriptListOffset`);
    const featureOffset = readU16(cursor, `${kind} featureListOffset`);
    const lookupOffset = readU16(cursor, `${kind} lookupListOffset`);
    const featureVariationsOffset = versionMinor >= 1 ? readU32(cursor, `${kind} featureVariationsOffset`) : 0;
    if (featureVariationsOffset !== 0 && featureVariationsOffset >= tableLength) {
        throw new Error(`${kind}: feature variations offset is outside the table`);
    }
    const extensionType = kind === "GSUB" ? 7 : 9;
    const contextType = kind === "GSUB" ? 5 : 7;
    const chainingType = kind === "GSUB" ? 6 : 8;
    return {
        kind,
        versionMajor,
        versionMinor,
        scripts: parseScriptList(bytes, tableOffset, tableLength, scriptOffset),
        features: parseFeatureList(bytes, tableOffset, tableLength, featureOffset),
        lookups: parseLookupList(bytes, tableOffset, tableLength, lookupOffset, extensionType, contextType, chainingType),
        featureVariationsOffset,
    };
}

export function findLayoutScript(table: LayoutTable, tag: string): LayoutScript | null {
    for (const script of table.scripts) {
        if (script.tag === tag) {
            return script;
        }
    }
    return null;
}

export function coverageIndex(coverage: GlyphCoverage, glyphId: number): number | null {
    if (glyphId < 0 || glyphId > 0xffff) {
        return null;
    }
    if (coverage.format === 1) {
        let low = 0;
        let high = coverage.glyphs.length - 1;
        while (low <= high) {
            const mid = (low + high) >> 1;
            const glyph = coverage.glyphs[mid] ?? 0;
            if (glyph === glyphId) {
                return mid;
            }
            if (glyph < glyphId) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return null;
    }
    for (const range of coverage.ranges) {
        if (glyphId < range.start) {
            return null;
        }
        if (glyphId <= range.end) {
            return range.startCoverageIndex + (glyphId - range.start);
        }
    }
    return null;
}

/** Glyph ids in coverage order. Rejects a range that would exceed `glyphCount`. */
export function coverageGlyphs(coverage: GlyphCoverage, glyphCount: number): number[] {
    if (coverage.format === 1) {
        return Array.from(coverage.glyphs);
    }
    const glyphs: number[] = [];
    for (const range of coverage.ranges) {
        if (range.end >= glyphCount) {
            throw new Error(`coverage: glyph ${range.end} is outside 0..${glyphCount - 1}`);
        }
        for (let glyphId = range.start; glyphId <= range.end; glyphId++) {
            glyphs.push(glyphId);
        }
    }
    return glyphs;
}

export function glyphClass(def: GlyphClassDef | null, glyphId: number): number {
    if (!def || glyphId < 0 || glyphId > 0xffff) {
        return 0;
    }
    if (def.format === 1) {
        const index = glyphId - def.startGlyphId;
        if (index < 0 || index >= def.classes.length) {
            return 0;
        }
        return def.classes[index] ?? 0;
    }
    for (const range of def.ranges) {
        if (glyphId < range.start) {
            return 0;
        }
        if (glyphId <= range.end) {
            return range.classValue;
        }
    }
    return 0;
}

function parseScriptList(bytes: Uint8Array, tableOffset: number, tableLength: number, offset: number): LayoutScript[] {
    const start = absoluteOffset(tableOffset, tableLength, offset, "ScriptList");
    const cursor = openCursor(bytes, start, tableOffset + tableLength - start, "ScriptList");
    const count = readU16(cursor, "ScriptList scriptCount");
    requireCount(count, 6, cursor.limit - cursor.offset, "ScriptList records");
    const records: { tag: string; offset: number }[] = [];
    for (let index = 0; index < count; index++) {
        records.push({ tag: readTag(cursor, "ScriptList tag"), offset: readU16(cursor, "ScriptList offset") });
    }
    return records.map((record) => parseScript(bytes, start, cursor.limit, record.tag, record.offset));
}

function parseScript(bytes: Uint8Array, listStart: number, listLimit: number, tag: string, offset: number): LayoutScript {
    if (offset === 0 || listStart + offset >= listLimit) {
        throw new Error(`Script ${tag}: offset ${offset} is outside the ScriptList`);
    }
    const cursor = openCursor(bytes, listStart + offset, listLimit - (listStart + offset), `Script ${tag}`);
    const defaultOffset = readU16(cursor, `Script ${tag} defaultLangSys`);
    const langCount = readU16(cursor, `Script ${tag} langSysCount`);
    requireCount(langCount, 6, cursor.limit - cursor.offset, `Script ${tag} languages`);
    const languages: { tag: string; offset: number }[] = [];
    for (let index = 0; index < langCount; index++) {
        languages.push({ tag: readTag(cursor, `Script ${tag} lang tag`), offset: readU16(cursor, `Script ${tag} lang offset`) });
    }
    const scriptStart = listStart + offset;
    return {
        tag,
        defaultLangSys: defaultOffset === 0 ? null : parseLangSys(bytes, scriptStart, cursor.limit, defaultOffset, `${tag} default`),
        languages: languages.map((language) => ({
            tag: language.tag,
            langSys: parseLangSys(bytes, scriptStart, cursor.limit, language.offset, `${tag} ${language.tag}`),
        })),
    };
}

function parseLangSys(bytes: Uint8Array, scriptStart: number, limit: number, offset: number, label: string): LayoutLangSys {
    if (offset === 0 || scriptStart + offset + 6 > limit) {
        throw new Error(`LangSys ${label}: offset ${offset} is outside the script table`);
    }
    const cursor = openCursor(bytes, scriptStart + offset, limit - (scriptStart + offset), `LangSys ${label}`);
    skip(cursor, 2, `LangSys ${label} lookupOrder`);
    const requiredFeatureIndex = readU16(cursor, `LangSys ${label} requiredFeature`);
    const featureCount = readU16(cursor, `LangSys ${label} featureIndexCount`);
    requireCount(featureCount, 2, cursor.limit - cursor.offset, `LangSys ${label} features`);
    const featureIndices: number[] = [];
    for (let index = 0; index < featureCount; index++) {
        featureIndices.push(readU16(cursor, `LangSys ${label} featureIndex`));
    }
    return { requiredFeatureIndex, featureIndices };
}

function parseFeatureList(bytes: Uint8Array, tableOffset: number, tableLength: number, offset: number): LayoutFeature[] {
    const start = absoluteOffset(tableOffset, tableLength, offset, "FeatureList");
    const cursor = openCursor(bytes, start, tableOffset + tableLength - start, "FeatureList");
    const count = readU16(cursor, "FeatureList featureCount");
    requireCount(count, 6, cursor.limit - cursor.offset, "FeatureList records");
    const records: { tag: string; offset: number }[] = [];
    for (let index = 0; index < count; index++) {
        records.push({ tag: readTag(cursor, "FeatureList tag"), offset: readU16(cursor, "FeatureList offset") });
    }
    return records.map((record) => {
        if (record.offset === 0 || start + record.offset + 4 > cursor.limit) {
            throw new Error(`Feature ${record.tag}: offset ${record.offset} is outside the FeatureList`);
        }
        const feature = openCursor(bytes, start + record.offset, cursor.limit - (start + record.offset), `Feature ${record.tag}`);
        skip(feature, 2, `Feature ${record.tag} params`);
        const lookupCount = readU16(feature, `Feature ${record.tag} lookupIndexCount`);
        requireCount(lookupCount, 2, feature.limit - feature.offset, `Feature ${record.tag} lookups`);
        const lookupListIndices: number[] = [];
        for (let index = 0; index < lookupCount; index++) {
            lookupListIndices.push(readU16(feature, `Feature ${record.tag} lookupIndex`));
        }
        return { tag: record.tag, lookupListIndices };
    });
}

function parseLookupList(
    bytes: Uint8Array,
    tableOffset: number,
    tableLength: number,
    offset: number,
    extensionType: number,
    contextType: number,
    chainingType: number
): LayoutLookup[] {
    const start = absoluteOffset(tableOffset, tableLength, offset, "LookupList");
    const cursor = openCursor(bytes, start, tableOffset + tableLength - start, "LookupList");
    const count = readU16(cursor, "LookupList lookupCount");
    requireCount(count, 2, cursor.limit - cursor.offset, "LookupList offsets");
    const offsets: number[] = [];
    for (let index = 0; index < count; index++) {
        offsets.push(readU16(cursor, "LookupList lookupOffset"));
    }
    return offsets.map((lookupOffset, index) => parseLookup(bytes, start, cursor.limit, lookupOffset, index, count, extensionType, contextType, chainingType));
}

function parseLookup(
    bytes: Uint8Array,
    listStart: number,
    listLimit: number,
    offset: number,
    index: number,
    lookupCount: number,
    extensionType: number,
    contextType: number,
    chainingType: number
): LayoutLookup {
    if (offset === 0 || listStart + offset + 6 > listLimit) {
        throw new Error(`Lookup ${index}: offset ${offset} is outside the LookupList`);
    }
    const cursor = openCursor(bytes, listStart + offset, listLimit - (listStart + offset), `Lookup ${index}`);
    const lookupType = readU16(cursor, `Lookup ${index} type`);
    const lookupFlag = readU16(cursor, `Lookup ${index} flag`);
    const subtableCount = readU16(cursor, `Lookup ${index} subtableCount`);
    requireCount(subtableCount, 2, cursor.limit - cursor.offset, `Lookup ${index} subtables`);
    const subtableOffsets: number[] = [];
    for (let subtableIndex = 0; subtableIndex < subtableCount; subtableIndex++) {
        const subtableOffset = readU16(cursor, `Lookup ${index} subtableOffset`);
        if (subtableOffset === 0 || listStart + offset + subtableOffset + 2 > listLimit) {
            throw new Error(`Lookup ${index}: subtable ${subtableIndex} offset ${subtableOffset} is outside the lookup`);
        }
        subtableOffsets.push(subtableOffset);
    }
    const markFilteringSet = lookupFlag & USE_MARK_FILTERING_SET ? readU16(cursor, `Lookup ${index} markFilteringSet`) : null;
    const lookupStart = listStart + offset;
    const nested = new Set<number>();
    let resolvedType = lookupType;
    let unsupported = false;
    for (let subtableIndex = 0; subtableIndex < subtableOffsets.length; subtableIndex++) {
        const subtableOffset = subtableOffsets[subtableIndex] ?? 0;
        const resolved = resolveSubtable(bytes, lookupStart, listLimit, subtableOffset, lookupType, extensionType, `Lookup ${index} subtable ${subtableIndex}`);
        if (subtableIndex === 0) {
            resolvedType = resolved.type;
        } else if (resolved.type !== resolvedType) {
            unsupported = true;
        }
        if (resolved.type === contextType || resolved.type === chainingType) {
            collectContextualLookups(bytes, resolved.start, resolved.limit, resolved.type === chainingType, nested, lookupCount, `Lookup ${index} subtable ${subtableIndex}`);
        } else if (resolved.type === extensionType) {
            throw new Error(`Lookup ${index}: extension subtable ${subtableIndex} resolves to another extension`);
        }
    }
    return {
        lookupType,
        resolvedType,
        lookupFlag,
        markFilteringSet,
        subtableOffsets,
        nestedLookupIndices: [...nested].sort((left, right) => left - right),
        unsupported,
    };
}

function resolveSubtable(
    bytes: Uint8Array,
    lookupStart: number,
    limit: number,
    subtableOffset: number,
    lookupType: number,
    extensionType: number,
    label: string
): { type: number; start: number; limit: number } {
    const start = lookupStart + subtableOffset;
    if (lookupType !== extensionType) {
        return { type: lookupType, start, limit };
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    const format = readU16(cursor, `${label} extension format`);
    if (format !== 1) {
        throw new Error(`${label}: unsupported extension format ${format}`);
    }
    const innerType = readU16(cursor, `${label} extensionLookupType`);
    const extensionOffset = readU32(cursor, `${label} extensionOffset`);
    if (innerType === extensionType) {
        throw new Error(`${label}: extension lookup type cannot wrap another extension`);
    }
    if (extensionOffset < 8 || start + extensionOffset + 2 > limit) {
        throw new Error(`${label}: extension offset ${extensionOffset} is outside the lookup`);
    }
    return { type: innerType, start: start + extensionOffset, limit };
}

function collectContextualLookups(bytes: Uint8Array, start: number, limit: number, chaining: boolean, nested: Set<number>, lookupCount: number, label: string): void {
    const cursor = openCursor(bytes, start, limit - start, label);
    const format = readU16(cursor, `${label} format`);
    if (format === 3) {
        if (chaining) {
            readCoverageOffsets(bytes, start, limit, cursor, `${label} backtrack`);
            readCoverageOffsets(bytes, start, limit, cursor, `${label} input`);
            readCoverageOffsets(bytes, start, limit, cursor, `${label} lookahead`);
            readLookupRecords(cursor, nested, lookupCount, label);
        } else {
            const glyphCount = readU16(cursor, `${label} glyphCount`);
            const recordCount = readU16(cursor, `${label} lookupRecordCount`);
            requireCount(glyphCount, 2, cursor.limit - cursor.offset, `${label} coverages`);
            for (let index = 0; index < glyphCount; index++) {
                const offset = readU16(cursor, `${label} coverageOffset`);
                if (offset === 0) {
                    throw new Error(`${label}: coverage offset is zero`);
                }
                parseCoverage(bytes, start + offset, limit, `${label} coverage ${index}`);
            }
            readKnownLookupRecords(cursor, recordCount, nested, lookupCount, label);
        }
        return;
    }
    if (format !== 1 && format !== 2) {
        throw new Error(`${label}: unsupported contextual format ${format}`);
    }
    const coverageOffset = readU16(cursor, `${label} coverage`);
    parseCoverage(bytes, start + coverageOffset, limit, `${label} coverage`);
    if (format === 2) {
        const classCount = chaining ? 3 : 1;
        for (let index = 0; index < classCount; index++) {
            const classOffset = readU16(cursor, `${label} classDef`);
            if (classOffset !== 0) {
                parseClassDef(bytes, start + classOffset, limit);
            }
        }
    }
    const setCount = readU16(cursor, `${label} ruleSetCount`);
    requireCount(setCount, 2, cursor.limit - cursor.offset, `${label} rule sets`);
    const setOffsets: number[] = [];
    for (let index = 0; index < setCount; index++) {
        setOffsets.push(readU16(cursor, `${label} ruleSetOffset`));
    }
    for (const setOffset of setOffsets) {
        if (setOffset === 0) {
            continue;
        }
        walkRuleSet(bytes, start + setOffset, limit, chaining, nested, lookupCount, label);
    }
}

function walkRuleSet(bytes: Uint8Array, start: number, limit: number, chaining: boolean, nested: Set<number>, lookupCount: number, label: string): void {
    if (start < 0 || start + 2 > limit) {
        throw new Error(`${label}: rule set is outside the subtable`);
    }
    const cursor = openCursor(bytes, start, limit - start, `${label} rule set`);
    const ruleCount = readU16(cursor, `${label} ruleCount`);
    requireCount(ruleCount, 2, cursor.limit - cursor.offset, `${label} rules`);
    const ruleOffsets: number[] = [];
    for (let index = 0; index < ruleCount; index++) {
        ruleOffsets.push(readU16(cursor, `${label} ruleOffset`));
    }
    for (const ruleOffset of ruleOffsets) {
        if (ruleOffset === 0 || start + ruleOffset >= limit) {
            throw new Error(`${label}: rule offset ${ruleOffset} is outside the rule set`);
        }
        const rule = openCursor(bytes, start + ruleOffset, limit - (start + ruleOffset), `${label} rule`);
        if (chaining) {
            skipGlyphIds(rule, `${label} backtrack`, false);
            skipGlyphIds(rule, `${label} input`, true);
            skipGlyphIds(rule, `${label} lookahead`, false);
        } else {
            // The first input glyph or class is implied by coverage or by the rule-set index.
            skipGlyphIds(rule, `${label} input`, true);
        }
        readLookupRecords(rule, nested, lookupCount, label);
    }
}

function skipGlyphIds(cursor: ByteCursor, label: string, omitFirst: boolean): void {
    const count = readU16(cursor, `${label} glyphCount`);
    const stored = omitFirst ? count - 1 : count;
    if (stored < 0) {
        throw new Error(`${label}: glyphCount ${count} is invalid`);
    }
    requireCount(stored, 2, cursor.limit - cursor.offset, label);
    skip(cursor, stored * 2, label);
}

function readCoverageOffsets(bytes: Uint8Array, subtableStart: number, limit: number, cursor: ByteCursor, label: string): void {
    const count = readU16(cursor, `${label} coverageCount`);
    requireCount(count, 2, cursor.limit - cursor.offset, label);
    for (let index = 0; index < count; index++) {
        const offset = readU16(cursor, `${label} coverageOffset`);
        if (offset === 0) {
            throw new Error(`${label}: coverage offset is zero`);
        }
        parseCoverage(bytes, subtableStart + offset, limit, `${label} coverage ${index}`);
    }
}

function readLookupRecords(cursor: ByteCursor, nested: Set<number>, lookupCount: number, label: string): void {
    const count = readU16(cursor, `${label} lookupRecordCount`);
    readKnownLookupRecords(cursor, count, nested, lookupCount, label);
}

function readKnownLookupRecords(cursor: ByteCursor, count: number, nested: Set<number>, lookupCount: number, label: string): void {
    requireCount(count, 4, cursor.limit - cursor.offset, `${label} lookup records`);
    for (let index = 0; index < count; index++) {
        skip(cursor, 2, `${label} sequenceIndex`);
        const lookupIndex = readU16(cursor, `${label} lookupListIndex`);
        if (lookupIndex >= lookupCount) {
            throw new Error(`${label}: nested lookup ${lookupIndex} is outside the lookup list`);
        }
        nested.add(lookupIndex);
    }
}

export function parseCoverage(bytes: Uint8Array, start: number, limit: number, label: string): GlyphCoverage {
    if (start < 0 || start + 4 > limit) {
        throw new Error(`${label}: coverage is outside its parent table`);
    }
    const cursor = openCursor(bytes, start, limit - start, label);
    const format = readU16(cursor, `${label} format`);
    if (format === 1) {
        const glyphCount = readU16(cursor, `${label} glyphCount`);
        requireCount(glyphCount, 2, cursor.limit - cursor.offset, label);
        const glyphs = new Uint16Array(glyphCount);
        let previous = -1;
        for (let index = 0; index < glyphCount; index++) {
            const glyphId = readU16(cursor, `${label} glyph`);
            if (glyphId <= previous) {
                throw new Error(`${label}: coverage glyphs are not strictly ascending`);
            }
            glyphs[index] = glyphId;
            previous = glyphId;
        }
        return { format: 1, glyphs, ranges: [] };
    }
    if (format === 2) {
        const rangeCount = readU16(cursor, `${label} rangeCount`);
        requireCount(rangeCount, 6, cursor.limit - cursor.offset, label);
        const ranges: { start: number; end: number; startCoverageIndex: number }[] = [];
        let previousEnd = -1;
        for (let index = 0; index < rangeCount; index++) {
            const rangeStart = readU16(cursor, `${label} startGlyphID`);
            const rangeEnd = readU16(cursor, `${label} endGlyphID`);
            const startCoverageIndex = readU16(cursor, `${label} startCoverageIndex`);
            if (rangeStart > rangeEnd || rangeStart <= previousEnd) {
                throw new Error(`${label}: coverage ranges are not strictly ascending`);
            }
            ranges.push({ start: rangeStart, end: rangeEnd, startCoverageIndex });
            previousEnd = rangeEnd;
        }
        return { format: 2, glyphs: new Uint16Array(0), ranges };
    }
    throw new Error(`${label}: unsupported coverage format ${format}`);
}

export function parseClassDef(bytes: Uint8Array, start: number, limit: number): GlyphClassDef {
    if (start + 4 > limit) {
        throw new Error("ClassDef is outside its parent table");
    }
    const cursor = openCursor(bytes, start, limit - start, "ClassDef");
    const format = readU16(cursor, "ClassDef format");
    if (format === 1) {
        const startGlyphId = readU16(cursor, "ClassDef startGlyphID");
        const glyphCount = readU16(cursor, "ClassDef glyphCount");
        requireCount(glyphCount, 2, cursor.limit - cursor.offset, "ClassDef values");
        const classes = new Uint16Array(glyphCount);
        for (let index = 0; index < glyphCount; index++) {
            classes[index] = readU16(cursor, "ClassDef classValue");
        }
        return { format: 1, startGlyphId, classes, ranges: [] };
    }
    if (format === 2) {
        const rangeCount = readU16(cursor, "ClassDef classRangeCount");
        requireCount(rangeCount, 6, cursor.limit - cursor.offset, "ClassDef ranges");
        const ranges: { start: number; end: number; classValue: number }[] = [];
        let previousEnd = -1;
        for (let index = 0; index < rangeCount; index++) {
            const rangeStart = readU16(cursor, "ClassDef startGlyphID");
            const rangeEnd = readU16(cursor, "ClassDef endGlyphID");
            const classValue = readU16(cursor, "ClassDef class");
            if (rangeStart > rangeEnd || rangeStart <= previousEnd) {
                throw new Error("ClassDef ranges are not strictly ascending");
            }
            ranges.push({ start: rangeStart, end: rangeEnd, classValue });
            previousEnd = rangeEnd;
        }
        return { format: 2, startGlyphId: 0, classes: new Uint16Array(0), ranges };
    }
    throw new Error(`ClassDef: unsupported format ${format}`);
}

function parseMarkGlyphSets(bytes: Uint8Array, tableOffset: number, tableLength: number, offset: number): number {
    const start = absoluteOffset(tableOffset, tableLength, offset, "GDEF markGlyphSets");
    const cursor = openCursor(bytes, start, tableOffset + tableLength - start, "GDEF markGlyphSets");
    const format = readU16(cursor, "GDEF markGlyphSets format");
    if (format !== 1) {
        throw new Error(`GDEF: unsupported mark glyph set format ${format}`);
    }
    const count = readU16(cursor, "GDEF markGlyphSetCount");
    requireCount(count, 4, cursor.limit - cursor.offset, "GDEF mark glyph sets");
    for (let index = 0; index < count; index++) {
        const coverageOffset = readU32(cursor, "GDEF markGlyphSet coverage");
        if (coverageOffset !== 0) {
            parseCoverage(bytes, start + coverageOffset, cursor.limit, `GDEF mark set ${index}`);
        }
    }
    return count;
}

function validateAttachOrCaret(bytes: Uint8Array, tableOffset: number, tableLength: number, offset: number, label: string): void {
    const start = absoluteOffset(tableOffset, tableLength, offset, label);
    const cursor = openCursor(bytes, start, tableOffset + tableLength - start, label);
    const coverageOffset = readU16(cursor, `${label} coverage`);
    const glyphCount = readU16(cursor, `${label} glyphCount`);
    requireCount(glyphCount, 2, cursor.limit - cursor.offset, label);
    if (coverageOffset !== 0) {
        parseCoverage(bytes, start + coverageOffset, cursor.limit, `${label} coverage`);
    }
}

function absoluteOffset(tableOffset: number, tableLength: number, offset: number, label: string): number {
    if (offset === 0 || offset >= tableLength) {
        throw new Error(`${label}: offset ${offset} is outside the parent table`);
    }
    return checkedOffset(tableOffset, offset, label);
}
