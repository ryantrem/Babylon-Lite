/** sfnt directory. Rejects collection and compressed containers before table parsing. */

import { checkedOffset, openCursor, readTag, readU16, readU32, requireCount } from "./reader.js";
import type { ByteCursor } from "./reader.js";

export interface SfntTable {
    readonly tag: string;
    readonly checksum: number;
    readonly offset: number;
    readonly length: number;
}

export interface SfntDirectory {
    readonly version: number;
    readonly tables: ReadonlyMap<string, SfntTable>;
}

const TRUE_TYPE = 0x00010000;
const TRUE_TYPE_MAC = 0x74727565;
const OTTO = 0x4f54544f;
const TTC = 0x74746366;
const WOFF = 0x774f4646;
const WOFF2 = 0x774f4632;

export function parseSfntDirectory(bytes: Uint8Array): SfntDirectory {
    if (bytes.length < 12) {
        throw new Error("createComputeFontFromBuffer: truncated sfnt header");
    }
    const header = openCursor(bytes, 0, bytes.length, "sfnt");
    const version = readU32(header, "sfnt version");
    if (version === TTC) {
        throw new Error("createComputeFontFromBuffer: TTC collections are unsupported");
    }
    if (version === WOFF) {
        throw new Error("createComputeFontFromBuffer: WOFF containers are unsupported");
    }
    if (version === WOFF2) {
        throw new Error("createComputeFontFromBuffer: WOFF2 containers are unsupported");
    }
    if (version !== TRUE_TYPE && version !== TRUE_TYPE_MAC && version !== OTTO) {
        throw new Error(`createComputeFontFromBuffer: unsupported font container 0x${version.toString(16)}`);
    }

    const numTables = readU16(header, "sfnt numTables");
    readU16(header, "sfnt searchRange");
    readU16(header, "sfnt entrySelector");
    readU16(header, "sfnt rangeShift");
    requireCount(numTables, 16, header.limit - header.offset, "sfnt table records");
    if (numTables === 0) {
        throw new Error("createComputeFontFromBuffer: sfnt directory has no tables");
    }

    const tables = new Map<string, SfntTable>();
    for (let index = 0; index < numTables; index++) {
        const tag = readTag(header, "sfnt table tag");
        const checksum = readU32(header, "sfnt checksum");
        const offset = readU32(header, "sfnt table offset");
        const length = readU32(header, "sfnt table length");
        const end = checkedOffset(offset, length, `table ${tag}`);
        if (end > bytes.length) {
            throw new Error(`createComputeFontFromBuffer: table ${tag} extent ${offset}+${length} exceeds font`);
        }
        if (tables.has(tag)) {
            throw new Error(`createComputeFontFromBuffer: duplicate table ${tag}`);
        }
        tables.set(tag, { tag, checksum, offset, length });
    }
    return { version, tables };
}

export function requireTable(directory: SfntDirectory, tag: string): SfntTable {
    const table = directory.tables.get(tag);
    if (!table) {
        throw new Error(`createComputeFontFromBuffer: missing required table ${tag}`);
    }
    return table;
}

export function tableCursor(bytes: Uint8Array, table: SfntTable): ByteCursor {
    return openCursor(bytes, table.offset, table.length, `table ${table.tag}`);
}

export function outlineFormatFor(directory: SfntDirectory): "truetype" | "cff" {
    if (directory.tables.has("CFF2")) {
        throw new Error("createComputeFontFromBuffer: CFF2 outlines are unsupported");
    }
    const hasGlyf = directory.tables.has("glyf");
    const cff = directory.tables.get("CFF ");
    if (directory.version === OTTO) {
        if (!cff || cff.length === 0) {
            throw new Error("createComputeFontFromBuffer: OTTO font has no CFF outline table");
        }
        return "cff";
    }
    const glyf = directory.tables.get("glyf");
    if (!hasGlyf || !glyf || glyf.length === 0) {
        throw new Error("createComputeFontFromBuffer: TrueType font has no glyf outline table");
    }
    if (cff) {
        throw new Error("createComputeFontFromBuffer: mixed glyf/CFF outline tables are unsupported");
    }
    return "truetype";
}
