/** head, hhea, maxp, and hmtx. Advances include the repeated final hmtx metric. */

import type { ByteCursor } from "./reader.js";
import { readI16, readU16, readU32, requireCount, skip } from "./reader.js";

export function parseHead(cursor: ByteCursor): { unitsPerEm: number; locaFormat: 0 | 1 } {
    if (cursor.limit - cursor.offset < 54) {
        throw new Error("head: truncated table");
    }
    const major = readU16(cursor, "head majorVersion");
    const minor = readU16(cursor, "head minorVersion");
    if (major !== 1) {
        throw new Error(`head: unsupported version ${major}.${minor}`);
    }
    skip(cursor, 8, "head revision");
    const magic = readU32(cursor, "head magicNumber");
    if (magic !== 0x5f0f3cf5) {
        throw new Error(`head: invalid magic 0x${magic.toString(16)}`);
    }
    skip(cursor, 2, "head flags");
    const unitsPerEm = readU16(cursor, "head unitsPerEm");
    if (unitsPerEm < 16 || unitsPerEm > 16384) {
        throw new Error(`head: unitsPerEm ${unitsPerEm} is outside 16..16384`);
    }
    skip(cursor, 30, "head bounds");
    const locaFormat = readI16(cursor, "head indexToLocFormat");
    const glyphDataFormat = readI16(cursor, "head glyphDataFormat");
    if (locaFormat !== 0 && locaFormat !== 1) {
        throw new Error(`head: unsupported loca format ${locaFormat}`);
    }
    if (glyphDataFormat !== 0) {
        throw new Error(`head: unsupported glyph data format ${glyphDataFormat}`);
    }
    return { unitsPerEm, locaFormat };
}

export function parseHheaNumberOfMetrics(cursor: ByteCursor): number {
    if (cursor.limit - cursor.offset < 36) {
        throw new Error("hhea: truncated table");
    }
    const major = readU16(cursor, "hhea majorVersion");
    if (major !== 1) {
        throw new Error(`hhea: unsupported version ${major}`);
    }
    skip(cursor, 32, "hhea metrics");
    const numberOfHMetrics = readU16(cursor, "hhea numberOfHMetrics");
    if (numberOfHMetrics === 0) {
        throw new Error("hhea: numberOfHMetrics is zero");
    }
    return numberOfHMetrics;
}

export function parseMaxpGlyphCount(cursor: ByteCursor): number {
    if (cursor.limit - cursor.offset < 6) {
        throw new Error("maxp: truncated table");
    }
    const version = readU32(cursor, "maxp version");
    if (version !== 0x00010000 && version !== 0x00005000) {
        throw new Error(`maxp: unsupported version 0x${version.toString(16)}`);
    }
    if (version === 0x00010000 && cursor.limit - cursor.offset < 28) {
        throw new Error("maxp: truncated TrueType 1.0 table");
    }
    const glyphCount = readU16(cursor, "maxp numGlyphs");
    if (glyphCount === 0) {
        throw new Error("maxp: numGlyphs is zero");
    }
    return glyphCount;
}

export function parseHmtx(cursor: ByteCursor, numberOfHMetrics: number, glyphCount: number): { advances: Uint16Array; leftSideBearings: Int16Array } {
    if (numberOfHMetrics > glyphCount) {
        throw new Error(`hmtx: numberOfHMetrics ${numberOfHMetrics} exceeds glyphCount ${glyphCount}`);
    }
    const sideBearings = glyphCount - numberOfHMetrics;
    requireCount(numberOfHMetrics, 4, cursor.limit - cursor.offset, "hmtx long metrics");
    const afterLong = cursor.limit - cursor.offset - numberOfHMetrics * 4;
    requireCount(sideBearings, 2, afterLong, "hmtx side bearings");

    const advances = new Uint16Array(glyphCount);
    const leftSideBearings = new Int16Array(glyphCount);
    let lastAdvance = 0;
    for (let index = 0; index < numberOfHMetrics; index++) {
        lastAdvance = readU16(cursor, "hmtx advance");
        advances[index] = lastAdvance;
        leftSideBearings[index] = readI16(cursor, "hmtx lsb");
    }
    for (let index = numberOfHMetrics; index < glyphCount; index++) {
        advances[index] = lastAdvance;
        leftSideBearings[index] = readI16(cursor, "hmtx extra lsb");
    }
    return { advances, leftSideBearings };
}

export function validateLoca(length: number, glyphCount: number, locaFormat: 0 | 1): void {
    const required = (glyphCount + 1) * (locaFormat === 0 ? 2 : 4);
    if (length < required) {
        throw new Error(`loca: length ${length} is shorter than ${required} bytes for ${glyphCount} glyphs`);
    }
}

export function parseFvarPresent(cursor: ByteCursor): boolean {
    if (cursor.limit - cursor.offset < 16) {
        throw new Error("fvar: truncated table");
    }
    const major = readU16(cursor, "fvar majorVersion");
    const minor = readU16(cursor, "fvar minorVersion");
    if (major !== 1 || minor !== 0) {
        throw new Error(`fvar: unsupported version ${major}.${minor}`);
    }
    skip(cursor, 4, "fvar axes offset");
    const axisCount = readU16(cursor, "fvar axisCount");
    return axisCount > 0;
}
