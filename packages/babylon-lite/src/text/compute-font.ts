/** Checked, dependency-free font snapshot for compute text shaping.
 *
 *  Construction validates the container and the mandatory metrics/mapping tables.
 *  GDEF/GSUB/GPOS are decoded on first layout access. Outline bytes stay in the
 *  snapshot; glyf/CFF decoding is a separate reader and is not imported here. */

import { mapCmapScalar, parseCmap } from "./font-parser/cmap.js";
import type { CmapMapping } from "./font-parser/cmap.js";
import { parseFvarPresent, parseHead, parseHheaNumberOfMetrics, parseHmtx, parseMaxpGlyphCount, validateLoca } from "./font-parser/metrics.js";
import { parseGdef, parseLayoutTable } from "./font-parser/opentype-layout.js";
import type { GdefTable, LayoutTable } from "./font-parser/opentype-layout.js";
import { outlineFormatFor, parseSfntDirectory, requireTable, tableCursor } from "./font-parser/sfnt.js";
import type { SfntDirectory, SfntTable } from "./font-parser/sfnt.js";

declare const computeFontBrand: unique symbol;

/** Checked TrueType or CFF font snapshot. No engine, scene, or GPU ownership. */
export interface ComputeFont {
    readonly [computeFontBrand]: true;
    readonly unitsPerEm: number;
    readonly glyphCount: number;
    readonly outlineFormat: "truetype" | "cff";
    /** @internal Copied font bytes. */
    readonly _bytes: Uint8Array;
    /** @internal Checked sfnt directory. */
    readonly _directory: SfntDirectory;
    /** @internal `head.indexToLocFormat`. */
    readonly _locaFormat: 0 | 1;
    /** @internal Unicode cmap selected with text-shaper 0.1.28 precedence. */
    readonly _cmap: CmapMapping;
    /** @internal Lazily memoized immutable nominal mappings in 256-scalar pages. */
    _nominalPages?: (Uint32Array | undefined)[];
    /** @internal Horizontal advances, including the repeated final hmtx metric. */
    readonly _advances: Uint16Array;
    /** @internal Left side bearings parallel to `_advances`. */
    readonly _leftSideBearings: Int16Array;
    /** @internal True when `fvar` advertises at least one axis. Default instance only. */
    readonly _hasVariableAxes: boolean;
    /** @internal Lazily decoded layout tables. */
    _layout: ComputeFontLayout | null;
}

/** @internal Layout tables decoded from a {@link ComputeFont}. */
export interface ComputeFontLayout {
    readonly gdef: GdefTable | null;
    readonly gsub: LayoutTable | null;
    readonly gpos: LayoutTable | null;
}

/** Build a compute font from an in-memory sfnt buffer. The buffer is copied. */
export function createComputeFontFromBuffer(data: ArrayBuffer): ComputeFont {
    if (!(data instanceof ArrayBuffer)) {
        throw new Error("createComputeFontFromBuffer: expected an ArrayBuffer");
    }
    const bytes = new Uint8Array(data.slice(0));
    const directory = parseSfntDirectory(bytes);
    const outlineFormat = outlineFormatFor(directory);
    const head = parseHead(tableCursor(bytes, requireTable(directory, "head")));
    const glyphCount = parseMaxpGlyphCount(tableCursor(bytes, requireTable(directory, "maxp")));
    const numberOfHMetrics = parseHheaNumberOfMetrics(tableCursor(bytes, requireTable(directory, "hhea")));
    const horizontal = parseHmtx(tableCursor(bytes, requireTable(directory, "hmtx")), numberOfHMetrics, glyphCount);
    const cmapTable = requireTable(directory, "cmap");
    const cmap = parseCmap(bytes, cmapTable.offset, cmapTable.length);
    if (outlineFormat === "truetype") {
        validateLoca(requireTable(directory, "loca").length, glyphCount, head.locaFormat);
    }
    const fvar = directory.tables.get("fvar");
    return {
        unitsPerEm: head.unitsPerEm,
        glyphCount,
        outlineFormat,
        _bytes: bytes,
        _directory: directory,
        _locaFormat: head.locaFormat,
        _cmap: cmap,
        _advances: horizontal.advances,
        _leftSideBearings: horizontal.leftSideBearings,
        _hasVariableAxes: fvar ? parseFvarPresent(tableCursor(bytes, fvar)) : false,
        _layout: null,
    } as ComputeFont;
}

/** Fetch a font and build a compute font. Nothing is fetched at import time. */
export async function loadComputeFont(url: string): Promise<ComputeFont> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`loadComputeFont: failed to fetch ${url} (${response.status})`);
    }
    return createComputeFontFromBuffer(await response.arrayBuffer());
}

/** Glyph id for a Unicode scalar. Missing mappings throw instead of returning glyph 0. */
export function mapComputeFontScalar(font: ComputeFont, scalar: number): number {
    let glyphId: number;
    if (Number.isInteger(scalar) && scalar >= 0 && scalar <= 0x10ffff && (scalar < 0xd800 || scalar > 0xdfff)) {
        const pages = (font._nominalPages ??= []);
        const index = scalar >>> 8;
        let page = pages[index];
        if (!page) {
            page = new Uint32Array(256).fill(0xffffffff);
            pages[index] = page;
        }
        const at = scalar & 255;
        glyphId = page[at]!;
        if (glyphId === 0xffffffff) {
            glyphId = mapCmapScalar(font._cmap, scalar);
            page[at] = glyphId;
        }
    } else {
        glyphId = mapCmapScalar(font._cmap, scalar);
    }
    if (glyphId === 0 || glyphId >= font.glyphCount) {
        const hex = scalar.toString(16).toUpperCase().padStart(4, "0");
        throw new Error(`mapComputeFontScalar: unmapped scalar U+${hex}`);
    }
    return glyphId;
}

/** @internal Fresh nominal input decoded with its checked scalar selection, not shaped output. */
export interface ComputeNominalVector {
    readonly font: ComputeFont;
    readonly glyphIds: Uint32Array;
    readonly misses: number;
    readonly seeds?: Set<number>;
}

/** @internal Copy checked immutable entries; misses retain full scalar validation and caller attribution. */
export function mapComputeFontScalars(font: ComputeFont, scalars: Uint32Array, onError: (index: number, error: unknown) => never, decoded?: ComputeNominalVector): Uint32Array {
    if (decoded?.font === font) {
        if (decoded.misses) {
            let seedTail = false;
            for (let index = 0; index < scalars.length; index++) {
                if (decoded.glyphIds[index] === 0) {
                    try {
                        decoded.glyphIds[index] = mapComputeFontScalar(font, scalars[index]!);
                        seedTail = true;
                    } catch (error) {
                        onError(index, error);
                    }
                }
                if (seedTail) {
                    decoded.seeds?.add(decoded.glyphIds[index]!);
                }
            }
        }
        return decoded.glyphIds;
    }
    const glyphIds = new Uint32Array(scalars.length);
    for (let index = 0; index < scalars.length; index++) {
        const scalar = scalars[index]!;
        const cached = font._nominalPages?.[scalar >>> 8]?.[scalar & 255];
        if (cached !== undefined && cached !== 0xffffffff && cached > 0 && cached < font.glyphCount) {
            glyphIds[index] = cached;
            continue;
        }
        try {
            glyphIds[index] = mapComputeFontScalar(font, scalar);
        } catch (error) {
            onError(index, error);
        }
    }
    return glyphIds;
}

/** Advance width in font units. Glyph ids outside the font throw. */
export function readComputeFontAdvance(font: ComputeFont, glyphId: number): number {
    if (!Number.isInteger(glyphId) || glyphId < 0 || glyphId >= font.glyphCount) {
        throw new Error(`readComputeFontAdvance: glyph ${glyphId} is outside 0..${font.glyphCount - 1}`);
    }
    return font._advances[glyphId] ?? 0;
}

/** Decode GDEF/GSUB/GPOS once. Optional tables may be absent; present tables must be well formed. */
export function ensureComputeFontLayout(font: ComputeFont): ComputeFontLayout {
    if (font._layout) {
        return font._layout;
    }
    const layout: ComputeFontLayout = {
        gdef: parseOptional(font, "GDEF", (table) => parseGdef(font._bytes, table.offset, table.length)),
        gsub: parseOptional(font, "GSUB", (table) => parseLayoutTable(font._bytes, table.offset, table.length, "GSUB")),
        gpos: parseOptional(font, "GPOS", (table) => parseLayoutTable(font._bytes, table.offset, table.length, "GPOS")),
    };
    font._layout = layout;
    return layout;
}

function parseOptional<T>(font: ComputeFont, tag: string, parse: (table: SfntTable) => T): T | null {
    const table = font._directory.tables.get(tag);
    if (!table) {
        return null;
    }
    if (table.length === 0) {
        throw new Error(`createComputeFontFromBuffer: table ${tag} is empty`);
    }
    return parse(table);
}
