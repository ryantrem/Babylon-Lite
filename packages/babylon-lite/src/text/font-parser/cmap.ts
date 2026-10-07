/** Unicode cmap. Preference matches text-shaper 0.1.28: 3-10, 0-4, 3-1, 0-3, 0-6, 1-0, then the first other non-14 subtable. */

import type { ByteCursor } from "./reader.js";
import { checkedOffset, openCursor, readU16, readU32, requireCount, skip } from "./reader.js";

interface CmapSegment4 {
    readonly endCode: number;
    readonly startCode: number;
    readonly idDelta: number;
    readonly idRangeOffset: number;
    readonly glyphIds: Uint16Array;
}

interface CmapGroup12 {
    readonly start: number;
    readonly end: number;
    readonly startGlyphId: number;
}

export interface CmapMapping {
    readonly format: 4 | 12;
    readonly platformId: number;
    readonly encodingId: number;
    readonly segments: readonly CmapSegment4[];
    readonly groups: readonly CmapGroup12[];
}

const PREFERRED_KEYS = ["3-10", "0-4", "3-1", "0-3", "0-6", "1-0"];

export function parseCmap(bytes: Uint8Array, tableOffset: number, tableLength: number): CmapMapping {
    const cursor = openCursor(bytes, tableOffset, tableLength, "cmap");
    const version = readU16(cursor, "cmap version");
    if (version !== 0) {
        throw new Error(`cmap: unsupported version ${version}`);
    }
    const numTables = readU16(cursor, "cmap numTables");
    requireCount(numTables, 8, cursor.limit - cursor.offset, "cmap encoding records");
    if (numTables === 0) {
        throw new Error("cmap: no encoding records");
    }

    const records: { platformId: number; encodingId: number; offset: number; format: number }[] = [];
    for (let index = 0; index < numTables; index++) {
        const platformId = readU16(cursor, "cmap platformId");
        const encodingId = readU16(cursor, "cmap encodingId");
        const offset = readU32(cursor, "cmap subtable offset");
        if (offset >= tableLength || tableLength - offset < 2) {
            throw new Error(`cmap: encoding ${platformId}-${encodingId} offset ${offset} is outside the table`);
        }
        const format = readU16(openCursor(bytes, tableOffset + offset, tableLength - offset, "cmap subtable"), "cmap format");
        records.push({ platformId, encodingId, offset, format });
    }

    const byKey = new Map<string, (typeof records)[number]>();
    for (const record of records) {
        const key = `${record.platformId}-${record.encodingId}`;
        if (!byKey.has(key)) {
            byKey.set(key, record);
        }
    }

    let chosen: (typeof records)[number] | undefined;
    for (const key of PREFERRED_KEYS) {
        const record = byKey.get(key);
        if (record && record.format !== 14) {
            chosen = record;
            break;
        }
    }
    if (!chosen) {
        chosen = records.find((record) => record.format !== 14);
    }
    if (!chosen) {
        throw new Error("cmap: no Unicode mapping subtable");
    }
    if (chosen.format !== 4 && chosen.format !== 12) {
        throw new Error(`cmap: unsupported subtable format ${chosen.format} at ${chosen.platformId}-${chosen.encodingId}`);
    }

    const subtable = openCursor(bytes, checkedOffset(tableOffset, chosen.offset, "cmap subtable"), tableLength - chosen.offset, "cmap subtable");
    if (chosen.format === 4) {
        return { format: 4, platformId: chosen.platformId, encodingId: chosen.encodingId, segments: parseFormat4(subtable), groups: [] };
    }
    return { format: 12, platformId: chosen.platformId, encodingId: chosen.encodingId, segments: [], groups: parseFormat12(subtable) };
}

function parseFormat4(cursor: ByteCursor): CmapSegment4[] {
    const format = readU16(cursor, "cmap format 4");
    if (format !== 4) {
        throw new Error(`cmap: expected format 4, found ${format}`);
    }
    const length = readU16(cursor, "cmap format 4 length");
    const subtableStart = cursor.offset - 4;
    if (length < 16 || subtableStart + length > cursor.limit) {
        throw new Error(`cmap: format 4 length ${length} exceeds the subtable`);
    }
    const bounded = openCursor(cursor.bytes, subtableStart, length, "cmap format 4");
    skip(bounded, 6, "cmap format 4 header");
    const segCountX2 = readU16(bounded, "cmap segCountX2");
    if (segCountX2 < 2 || segCountX2 % 2 !== 0) {
        throw new Error(`cmap: invalid segCountX2 ${segCountX2}`);
    }
    const segCount = segCountX2 / 2;
    skip(bounded, 6, "cmap format 4 search");
    requireCount(segCount, 2, bounded.limit - bounded.offset, "cmap endCodes");
    const endCodes = readU16List(bounded, segCount, "cmap endCode");
    const reserved = readU16(bounded, "cmap reservedPad");
    if (reserved !== 0) {
        throw new Error(`cmap: format 4 reserved pad is ${reserved}`);
    }
    const startCodes = readU16List(bounded, segCount, "cmap startCode");
    const idDeltas = readU16List(bounded, segCount, "cmap idDelta");
    const idRangeOffsets = readU16List(bounded, segCount, "cmap idRangeOffset");
    const glyphIdArray = readU16List(bounded, Math.floor((bounded.limit - bounded.offset) / 2), "cmap glyphId");

    const segments: CmapSegment4[] = [];
    for (let index = 0; index < segCount; index++) {
        const startCode = startCodes[index] ?? 0;
        const endCode = endCodes[index] ?? 0;
        if (startCode > endCode) {
            throw new Error(`cmap: format 4 segment ${index} has start ${startCode} after end ${endCode}`);
        }
        const idRangeOffset = idRangeOffsets[index] ?? 0;
        let glyphIds = new Uint16Array(0);
        if (idRangeOffset !== 0) {
            const glyphIndex = idRangeOffset / 2 - (segCount - index);
            const count = endCode - startCode + 1;
            if (idRangeOffset % 2 !== 0 || glyphIndex < 0 || glyphIndex + count > glyphIdArray.length) {
                throw new Error(`cmap: format 4 segment ${index} glyph index ${glyphIndex} is outside the glyphId array`);
            }
            glyphIds = glyphIdArray.slice(glyphIndex, glyphIndex + count);
        }
        const rawDelta = idDeltas[index] ?? 0;
        segments.push({
            endCode,
            startCode,
            idDelta: rawDelta >= 0x8000 ? rawDelta - 0x10000 : rawDelta,
            idRangeOffset,
            glyphIds,
        });
    }
    return segments;
}

function parseFormat12(cursor: ByteCursor): CmapGroup12[] {
    const format = readU16(cursor, "cmap format 12");
    if (format !== 12) {
        throw new Error(`cmap: expected format 12, found ${format}`);
    }
    const reserved = readU16(cursor, "cmap format 12 reserved");
    if (reserved !== 0) {
        throw new Error(`cmap: format 12 reserved field is ${reserved}`);
    }
    const length = readU32(cursor, "cmap format 12 length");
    const subtableStart = cursor.offset - 8;
    if (length < 16 || subtableStart + length > cursor.limit) {
        throw new Error(`cmap: format 12 length ${length} exceeds the subtable`);
    }
    readU32(cursor, "cmap format 12 language");
    const numGroups = readU32(cursor, "cmap format 12 numGroups");
    requireCount(numGroups, 12, cursor.limit - cursor.offset, "cmap format 12 groups");
    const groups: CmapGroup12[] = [];
    let previousEnd = -1;
    for (let index = 0; index < numGroups; index++) {
        const start = readU32(cursor, "cmap startCharCode");
        const end = readU32(cursor, "cmap endCharCode");
        const startGlyphId = readU32(cursor, "cmap startGlyphId");
        if (start > end || start > 0x10ffff || end > 0x10ffff) {
            throw new Error(`cmap: format 12 group ${index} range U+${start.toString(16)}..U+${end.toString(16)} is invalid`);
        }
        if (start <= previousEnd) {
            throw new Error(`cmap: format 12 groups are not strictly ascending at ${index}`);
        }
        const span = end - start;
        if (startGlyphId > 0xffff || span > 0xffff - startGlyphId) {
            throw new Error(`cmap: format 12 group ${index} glyph ids exceed 65535`);
        }
        groups.push({ start, end, startGlyphId });
        previousEnd = end;
    }
    return groups;
}

/** Glyph id for a scalar, or 0 when the selected Unicode subtable does not map it. */
export function mapCmapScalar(cmap: CmapMapping, scalar: number): number {
    if (!Number.isInteger(scalar) || scalar < 0 || scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff)) {
        return 0;
    }
    if (cmap.format === 12) {
        return mapFormat12(cmap.groups, scalar);
    }
    if (scalar > 0xffff) {
        return 0;
    }
    return mapFormat4(cmap.segments, scalar);
}

function mapFormat12(groups: readonly CmapGroup12[], scalar: number): number {
    let low = 0;
    let high = groups.length - 1;
    while (low <= high) {
        const mid = (low + high) >>> 1;
        const group = groups[mid];
        if (!group) {
            return 0;
        }
        if (scalar > group.end) {
            low = mid + 1;
        } else if (scalar < group.start) {
            high = mid - 1;
        } else {
            return group.startGlyphId + (scalar - group.start);
        }
    }
    return 0;
}

function mapFormat4(segments: readonly CmapSegment4[], scalar: number): number {
    let low = 0;
    let high = segments.length - 1;
    while (low <= high) {
        const mid = (low + high) >>> 1;
        const segment = segments[mid];
        if (!segment) {
            return 0;
        }
        if (scalar > segment.endCode) {
            low = mid + 1;
            continue;
        }
        if (scalar < segment.startCode) {
            high = mid - 1;
            continue;
        }
        if (segment.idRangeOffset === 0) {
            return (scalar + segment.idDelta) & 0xffff;
        }
        const glyphId = segment.glyphIds[scalar - segment.startCode] ?? 0;
        if (glyphId === 0) {
            return 0;
        }
        return (glyphId + segment.idDelta) & 0xffff;
    }
    return 0;
}

function readU16List(cursor: ByteCursor, count: number, label: string): Uint16Array {
    const values = new Uint16Array(count);
    for (let index = 0; index < count; index++) {
        values[index] = readU16(cursor, label);
    }
    return values;
}
