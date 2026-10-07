/** Default-instance variation deltas.
 *
 *  text-shaper 0.1.28 applies the design value and does not evaluate VariationIndex
 *  tables. This reader only proves whether the default-coordinate delta is zero.
 *  A nonzero result stays a compiler rejection; it is never applied. */

import type { ComputeFont } from "../compute-font.js";
import { ensureComputeFontLayout } from "../compute-font.js";
import type { ByteCursor } from "./reader.js";
import { openCursor, readF2Dot14, readI16, readI8, readU16, readU32, requireCount } from "./reader.js";

interface RegionAxis {
    readonly start: number;
    readonly peak: number;
    readonly end: number;
}

interface VariationData {
    readonly itemCount: number;
    readonly regionIndexes: readonly number[];
    readonly rows: readonly (readonly number[])[];
}

export function createDefaultVariationDelta(font: ComputeFont): (outer: number, inner: number) => number {
    const layout = ensureComputeFontLayout(font);
    const gdef = font._directory.tables.get("GDEF");
    const storeOffset = layout.gdef?.itemVariationStoreOffset ?? 0;
    if (!gdef || storeOffset === 0) {
        return () => Number.NaN;
    }
    const store = readItemVariationStore(font._bytes, gdef.offset + storeOffset, gdef.offset + gdef.length);
    return (outer, inner) => store.delta(outer, inner);
}

function readItemVariationStore(bytes: Uint8Array, start: number, limit: number): { delta: (outer: number, inner: number) => number } {
    const cursor = openCursor(bytes, start, limit - start, "GDEF item variation store");
    const format = readU16(cursor, "GDEF item variation store format");
    if (format !== 1) {
        throw new Error(`GDEF item variation store: unsupported format ${format}`);
    }
    const regionListOffset = readU32(cursor, "GDEF variationRegionListOffset");
    const dataCount = readU16(cursor, "GDEF itemVariationDataCount");
    requireCount(dataCount, 4, cursor.limit - cursor.offset, "GDEF itemVariationData");
    const dataOffsets: number[] = [];
    for (let index = 0; index < dataCount; index++) {
        dataOffsets.push(readU32(cursor, "GDEF itemVariationDataOffset"));
    }
    const regions = readRegions(bytes, start + regionListOffset, limit);
    const scalars = regions.map(scalarAtDefault);
    const allZero = scalars.every((scalar) => scalar === 0);
    const data = dataOffsets.map((offset) => readVariationData(bytes, start + offset, limit, allZero));
    return {
        delta(outer: number, inner: number): number {
            if (!Number.isInteger(outer) || !Number.isInteger(inner) || outer < 0 || inner < 0) {
                return Number.NaN;
            }
            const item = data[outer];
            if (!item || inner >= item.itemCount) {
                return Number.NaN;
            }
            if (allZero) {
                return 0;
            }
            const row = item.rows[inner];
            if (!row) {
                return Number.NaN;
            }
            let delta = 0;
            item.regionIndexes.forEach((regionIndex, index) => {
                delta += (scalars[regionIndex] ?? 0) * (row[index] ?? 0);
            });
            return Math.round(delta);
        },
    };
}

function readRegions(bytes: Uint8Array, start: number, limit: number): RegionAxis[][] {
    const cursor = openCursor(bytes, start, limit - start, "GDEF variation region list");
    const axisCount = readU16(cursor, "GDEF variation axisCount");
    const regionCount = readU16(cursor, "GDEF variation regionCount");
    requireCount(regionCount, axisCount * 6, cursor.limit - cursor.offset, "GDEF variation regions");
    const regions: RegionAxis[][] = [];
    for (let region = 0; region < regionCount; region++) {
        const axes: RegionAxis[] = [];
        for (let axis = 0; axis < axisCount; axis++) {
            axes.push({
                start: readF2Dot14(cursor, "GDEF region startCoord"),
                peak: readF2Dot14(cursor, "GDEF region peakCoord"),
                end: readF2Dot14(cursor, "GDEF region endCoord"),
            });
        }
        regions.push(axes);
    }
    return regions;
}

function readVariationData(bytes: Uint8Array, start: number, limit: number, headerOnly: boolean): VariationData {
    const cursor = openCursor(bytes, start, limit - start, "GDEF item variation data");
    const itemCount = readU16(cursor, "GDEF itemCount");
    const wordDeltaCount = readU16(cursor, "GDEF wordDeltaCount");
    const regionIndexCount = readU16(cursor, "GDEF regionIndexCount");
    const longWords = (wordDeltaCount & 0x8000) !== 0;
    const wordCount = wordDeltaCount & 0x7fff;
    if (wordCount > regionIndexCount) {
        throw new Error("GDEF item variation data: wordDeltaCount exceeds regionIndexCount");
    }
    requireCount(regionIndexCount, 2, cursor.limit - cursor.offset, "GDEF regionIndexes");
    const regionIndexes: number[] = [];
    for (let index = 0; index < regionIndexCount; index++) {
        regionIndexes.push(readU16(cursor, "GDEF regionIndex"));
    }
    const rowBytes = (longWords ? wordCount * 4 : wordCount * 2) + (regionIndexCount - wordCount);
    requireCount(itemCount, rowBytes, cursor.limit - cursor.offset, "GDEF deltaSets");
    if (headerOnly) {
        return { itemCount, regionIndexes, rows: [] };
    }
    const rows: number[][] = [];
    for (let item = 0; item < itemCount; item++) {
        rows.push(readDeltaRow(cursor, regionIndexCount, wordCount, longWords));
    }
    return { itemCount, regionIndexes, rows };
}

function readDeltaRow(cursor: ByteCursor, regionIndexCount: number, wordCount: number, longWords: boolean): number[] {
    const row: number[] = [];
    for (let index = 0; index < regionIndexCount; index++) {
        if (index < wordCount) {
            row.push(longWords ? readI32(cursor) : readI16(cursor, "GDEF variation delta"));
        } else {
            row.push(readI8(cursor, "GDEF variation delta"));
        }
    }
    return row;
}

function readI32(cursor: ByteCursor): number {
    const value = readU32(cursor, "GDEF variation delta");
    return value >= 0x80000000 ? value - 0x100000000 : value;
}

/** text-shaper 0.1.28 region scalar at normalized coordinate 0. */
function scalarAtDefault(axes: readonly RegionAxis[]): number {
    let scalar = 1;
    for (const axis of axes) {
        if (axis.start > axis.peak || axis.peak > axis.end) {
            continue;
        }
        if (axis.start < 0 && axis.end > 0 && axis.peak !== 0) {
            continue;
        }
        if (axis.peak === 0) {
            continue;
        }
        if (0 < axis.start || 0 > axis.end) {
            return 0;
        }
        if (0 === axis.peak) {
            continue;
        }
        scalar *= 0 < axis.peak ? (0 - axis.start) / (axis.peak - axis.start) : (axis.end - 0) / (axis.end - axis.peak);
    }
    return scalar;
}
