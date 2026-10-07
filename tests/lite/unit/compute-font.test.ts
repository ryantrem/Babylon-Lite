/** Checked compute-font parsing against text-shaper 0.1.28 cmap/hmtx oracles.
 *
 *  text-shaper is imported only from this test. The parser modules do not import it. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import * as cmapReader from "../../../packages/babylon-lite/src/text/font-parser/cmap";

import {
    createComputeFontFromBuffer,
    ensureComputeFontLayout,
    loadComputeFont,
    mapComputeFontScalar,
    mapComputeFontScalars,
    readComputeFontAdvance,
} from "../../../packages/babylon-lite/src/text/compute-font";
import { Font } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";

const TEXT_SHAPER_VERSION = JSON.parse(readFileSync(join(process.cwd(), "packages/babylon-lite/node_modules/text-shaper/package.json"), "utf8")).version as string;

function fontBytes(file: string): ArrayBuffer {
    const buf = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function oracleFont(bytes: ArrayBuffer): Font {
    return Font.load(bytes);
}

const BMP_SAMPLE = [0x20, 0x41, 0x61, 0x00e9, 0x00c6, 0x0141, 0x0391, 0x041c];

describe("compute font parsing", () => {
    it("reuses immutable adjacent-scalar properties while decoding every current glyph and cluster afresh", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const h = mapComputeFontScalar(font, 0x48);
        const a = mapComputeFontScalar(font, 0x41);
        let pagesRead = 0;
        font._nominalPages = new Proxy(font._nominalPages!, {
            get(target, key, receiver) {
                if (key === "0") {
                    pagesRead++;
                }
                return Reflect.get(target, key, receiver);
            },
        });
        const text = "H".repeat(50) + "A".repeat(50) + "H".repeat(50);
        const selected = selectComputeTextRun(text, font);
        expect(pagesRead).toBe(1);
        expect(selected._scalars).toEqual(Uint32Array.from(text, (char) => char.charCodeAt(0)));
        expect(selected._utf16Offsets).toEqual(Uint32Array.from({ length: text.length }, (_, index) => index));
        expect(selected._nominal?.glyphIds).toEqual(Uint32Array.from(text, (char) => (char === "H" ? h : a)));
        expect(selectComputeTextRun(text, font)._scalars.buffer).not.toBe(selected._scalars.buffer);
    });

    it("preserves first-seen closure seed order across cold nominal misses followed by checked warm entries", () => {
        const bytes = fontBytes("Roboto-Regular.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const h = mapComputeFontScalar(font, 0x48);
        const seeds = new Set<number>();
        const selected = selectComputeTextRun("BHH", font, seeds);
        const glyphs = mapComputeFontScalars(
            font,
            selected._scalars,
            (_index, error) => {
                throw error;
            },
            selected._nominal
        );
        const b = oracleFont(bytes).glyphId(0x42);
        expect([...seeds]).toEqual([b, h]);
        expect(glyphs).toEqual(Uint32Array.of(b, h, h));
        expect(glyphs.buffer).toBe(selected._scalars.buffer);
        expect(glyphs.byteOffset).toBeGreaterThanOrEqual(selected._utf16Offsets.byteOffset + selected._utf16Offsets.byteLength);
        expect(() => selectComputeTextRun("H😀\n", font, seeds)).toThrow(/control at scalar 2/);
        expect(() => selectComputeTextRun("H😀\uD800", font, seeds)).toThrow(/lone surrogate at UTF-16 index 3/);
    });

    it("indexes bounded nominal pages directly by scalar rather than hashing each warm glyph", () => {
        const font = createComputeFontFromBuffer(fontBytes("NotoSansGothic-Regular.ttf"));
        for (const scalar of [0x20, 0x10330, 0x10332]) {
            const glyph = mapComputeFontScalar(font, scalar);
            expect(font._nominalPages?.[scalar >>> 8]?.[scalar & 255]).toBe(glyph);
            expect(mapComputeFontScalar(font, scalar)).toBe(glyph);
        }
        expect(font._nominalPages?.length).toBeLessThanOrEqual(4352);
    });

    it("reuses immutable nominal lookups per font and scalar while retaining unmapped failures", () => {
        const inter = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        const roboto = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        const mapping = vi.spyOn(cmapReader, "mapCmapScalar");
        try {
            const first = mapComputeFontScalar(inter, 0x41);
            expect(mapComputeFontScalar(inter, 0x41)).toBe(first);
            const second = mapComputeFontScalar(roboto, 0x41);
            expect(mapComputeFontScalar(roboto, 0x41)).toBe(second);
            expect(mapping).toHaveBeenCalledTimes(2);
            for (let index = 0; index < 2; index++) {
                expect(() => mapComputeFontScalar(inter, 0x10330)).toThrow(/unmapped scalar U\+10330/);
            }
            expect(mapping).toHaveBeenCalledTimes(3);
            expect(() => mapComputeFontScalar(inter, 0xd800)).toThrow(/unmapped scalar U\+D800/);
            expect(() => mapComputeFontScalar(inter, NaN)).toThrow(/unmapped scalar/);
        } finally {
            mapping.mockRestore();
        }
    });

    it("compares against text-shaper 0.1.28", () => {
        expect(TEXT_SHAPER_VERSION).toBe("0.1.28");
    });

    it.each([
        ["Inter.ttf", 2048, 2933, true],
        ["Roboto-Regular.ttf", 2048, 3387, false],
    ] as const)("reads %s metrics, cmap, and layout", (file, unitsPerEm, glyphCount, variable) => {
        const bytes = fontBytes(file);
        const font = createComputeFontFromBuffer(bytes);
        const oracle = oracleFont(bytes);
        expect(font.unitsPerEm).toBe(unitsPerEm);
        expect(font.glyphCount).toBe(glyphCount);
        expect(font.outlineFormat).toBe("truetype");
        expect(font.unitsPerEm).toBe(oracle.unitsPerEm);
        expect(font.glyphCount).toBe(oracle.numGlyphs);
        expect(font._hasVariableAxes).toBe(variable);

        for (const scalar of BMP_SAMPLE) {
            const glyphId = mapComputeFontScalar(font, scalar);
            expect(glyphId, `U+${scalar.toString(16)}`).toBe(oracle.glyphId(scalar));
            expect(readComputeFontAdvance(font, glyphId)).toBe(oracle.advanceWidth(glyphId));
        }

        const supplementary = firstSupplementary(font);
        if (supplementary === null) {
            expect(oracle.glyphId(0x10330)).toBe(0);
            expect(() => mapComputeFontScalar(font, 0x10330)).toThrow(/unmapped scalar U\+10330/);
        } else {
            const glyphId = mapComputeFontScalar(font, supplementary);
            expect(glyphId).toBe(oracle.glyphId(supplementary));
            expect(readComputeFontAdvance(font, glyphId)).toBe(oracle.advanceWidth(glyphId));
        }

        const layout = ensureComputeFontLayout(font);
        expect(layout.gsub?.lookups.length).toBeGreaterThan(0);
        expect(layout.gpos?.lookups.length).toBeGreaterThan(0);
        expect(layout.gdef).not.toBeNull();
        expect(ensureComputeFontLayout(font)).toBe(layout);
    });

    it("rejects an unmapped BMP scalar instead of returning glyph 0", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        expect(() => mapComputeFontScalar(font, 0x1f600)).toThrow(/unmapped scalar U\+1F600/);
    });

    it("rejects a glyph id outside hmtx", () => {
        const font = createComputeFontFromBuffer(fontBytes("Inter.ttf"));
        expect(() => readComputeFontAdvance(font, font.glyphCount)).toThrow(/outside/);
    });

    it("reports a fetch failure without reading a font", async () => {
        const original = globalThis.fetch;
        globalThis.fetch = async () => new Response("missing", { status: 404 });
        try {
            await expect(loadComputeFont("https://example.invalid/font.ttf")).rejects.toThrow(/loadComputeFont: failed to fetch/);
        } finally {
            globalThis.fetch = original;
        }
    });

    it("loads a font through fetch", async () => {
        const bytes = fontBytes("Roboto-Regular.ttf");
        const original = globalThis.fetch;
        globalThis.fetch = async () => new Response(bytes);
        try {
            const font = await loadComputeFont("https://example.invalid/Roboto-Regular.ttf");
            expect(font.glyphCount).toBe(3387);
        } finally {
            globalThis.fetch = original;
        }
    });

    it("propagates transport and body errors and rejects a successful malformed response", async () => {
        const original = globalThis.fetch;
        try {
            globalThis.fetch = async () => {
                throw new Error("transport failed");
            };
            await expect(loadComputeFont("https://example.invalid/font.ttf")).rejects.toThrow("transport failed");
            globalThis.fetch = async () => new Response(Uint8Array.of(0, 1, 2));
            await expect(loadComputeFont("https://example.invalid/font.ttf")).rejects.toThrow(/truncated/);
            const response = new Response();
            response.arrayBuffer = async () => {
                throw new Error("body failed");
            };
            globalThis.fetch = async () => response;
            await expect(loadComputeFont("https://example.invalid/font.ttf")).rejects.toThrow("body failed");
        } finally {
            globalThis.fetch = original;
        }
    });

    it("throws on truncated, collection, and compressed containers", () => {
        expect(() => createComputeFontFromBuffer(new ArrayBuffer(4))).toThrow(/truncated sfnt/);
        expect(() => createComputeFontFromBuffer(container(0x74746366))).toThrow(/TTC/);
        expect(() => createComputeFontFromBuffer(container(0x774f4646))).toThrow(/WOFF containers/);
        expect(() => createComputeFontFromBuffer(container(0x774f4632))).toThrow(/WOFF2/);
        expect(() => createComputeFontFromBuffer(container(0x12345678))).toThrow(/unsupported font container/);
    });

    it("throws when a table extent or required table is truncated", () => {
        expect(() => createComputeFontFromBuffer(buildSfnt(0x00010000, { head: new Uint8Array(4) }))).toThrow(/glyf/);
        expect(() =>
            createComputeFontFromBuffer(
                buildSfnt(0x00010000, {
                    glyf: new Uint8Array([0]),
                    head: new Uint8Array(8),
                })
            )
        ).toThrow(/head: truncated/);
        expect(() =>
            createComputeFontFromBuffer(
                buildSfnt(0x00010000, {
                    glyf: new Uint8Array([0]),
                    head: headTable(2048, 0),
                    maxp: maxpTable(1),
                    hhea: hheaTable(1),
                    hmtx: hmtxTable([100]),
                    loca: new Uint8Array(4),
                    cmap: new Uint8Array([0, 0]),
                })
            )
        ).toThrow(/cmap/);
    });

    it("throws when hmtx or loca is shorter than the glyph count", () => {
        const tables = minimalTables();
        tables.hmtx = hmtxTable([100]);
        expect(() => createComputeFontFromBuffer(buildSfnt(0x00010000, tables))).toThrow(/hmtx/);
        tables.hmtx = hmtxTable([100, 200]);
        tables.loca = new Uint8Array(2);
        expect(() => createComputeFontFromBuffer(buildSfnt(0x00010000, tables))).toThrow(/loca/);
    });

    it("rejects a CFF2 outline and reports a CFF outline without parsing it", () => {
        const cff2 = minimalTables();
        cff2.CFF2 = new Uint8Array([1, 2, 3, 4]);
        expect(() => createComputeFontFromBuffer(buildSfnt(0x00010000, cff2))).toThrow(/CFF2/);

        const cff = minimalTables();
        delete cff.glyf;
        delete cff.loca;
        cff["CFF "] = new Uint8Array([1, 0, 0, 0]);
        const font = createComputeFontFromBuffer(buildSfnt(0x4f54544f, cff));
        expect(font.outlineFormat).toBe("cff");
        expect(font.glyphCount).toBe(2);
        expect(mapComputeFontScalar(font, 0x41)).toBe(1);
        expect(readComputeFontAdvance(font, 1)).toBe(200);
    });

    it("maps a format-12 supplementary scalar", () => {
        const tables = minimalTables();
        tables.cmap = cmapFormat12([
            { start: 0x41, end: 0x41, glyphId: 1 },
            { start: 0x10330, end: 0x10330, glyphId: 1 },
        ]);
        const bytes = buildSfnt(0x00010000, tables);
        const font = createComputeFontFromBuffer(bytes);
        expect(mapComputeFontScalar(font, 0x10330)).toBe(1);
        expect(readComputeFontAdvance(font, 1)).toBe(200);
        const oracle = oracleFont(bytes);
        expect(oracle.glyphId(0x10330)).toBe(1);
        expect(oracle.advanceWidth(1)).toBe(200);
    });
});

function firstSupplementary(font: ReturnType<typeof createComputeFontFromBuffer>): number | null {
    for (const group of font._cmap.groups) {
        if (group.end > 0xffff && group.startGlyphId !== 0) {
            return Math.max(group.start, 0x10000);
        }
    }
    return null;
}

function container(version: number): ArrayBuffer {
    const bytes = new Uint8Array(12);
    new DataView(bytes.buffer).setUint32(0, version);
    return bytes.buffer;
}

function buildSfnt(version: number, tables: Record<string, Uint8Array>): ArrayBuffer {
    const tags = Object.keys(tables);
    const headerSize = 12 + tags.length * 16;
    let offset = headerSize;
    const records = tags.map((tag) => {
        const body = tables[tag] ?? new Uint8Array();
        const record = { tag, offset, length: body.length, body };
        offset += body.length;
        return record;
    });
    const out = new Uint8Array(offset);
    const view = new DataView(out.buffer);
    view.setUint32(0, version);
    view.setUint16(4, tags.length);
    records.forEach((record, index) => {
        const at = 12 + index * 16;
        for (let byte = 0; byte < 4; byte++) {
            view.setUint8(at + byte, record.tag.charCodeAt(byte) || 0x20);
        }
        view.setUint32(at + 8, record.offset);
        view.setUint32(at + 12, record.length);
        out.set(record.body, record.offset);
    });
    return out.buffer;
}

function minimalTables(): Record<string, Uint8Array> {
    return {
        cmap: cmapFormat12([{ start: 0x41, end: 0x41, glyphId: 1 }]),
        glyf: new Uint8Array([0, 0]),
        head: headTable(1000, 0),
        hhea: hheaTable(2),
        hmtx: hmtxTable([100, 200]),
        loca: new Uint8Array(6),
        maxp: maxpTable(2),
    };
}

function headTable(unitsPerEm: number, locaFormat: number): Uint8Array {
    const bytes = new Uint8Array(54);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint32(12, 0x5f0f3cf5);
    view.setUint16(18, unitsPerEm);
    view.setInt16(50, locaFormat);
    return bytes;
}

function maxpTable(glyphCount: number): Uint8Array {
    const bytes = new Uint8Array(32);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, glyphCount);
    return bytes;
}

function hheaTable(numberOfHMetrics: number): Uint8Array {
    const bytes = new Uint8Array(36);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint16(34, numberOfHMetrics);
    return bytes;
}

function hmtxTable(advances: number[]): Uint8Array {
    const bytes = new Uint8Array(advances.length * 4);
    const view = new DataView(bytes.buffer);
    advances.forEach((advance, index) => {
        view.setUint16(index * 4, advance);
    });
    return bytes;
}

function cmapFormat12(groups: { start: number; end: number; glyphId: number }[]): Uint8Array {
    const bytes = new Uint8Array(4 + 8 + 12 + 16 + groups.length * 12);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 0);
    view.setUint16(2, 1);
    view.setUint16(4, 0);
    view.setUint16(6, 4);
    view.setUint32(8, 12);
    const subtable = 12;
    view.setUint16(subtable, 12);
    view.setUint32(subtable + 4, 16 + groups.length * 12);
    view.setUint32(subtable + 12, groups.length);
    groups.forEach((group, index) => {
        const at = subtable + 16 + index * 12;
        view.setUint32(at, group.start);
        view.setUint32(at + 4, group.end);
        view.setUint32(at + 8, group.glyphId);
    });
    return bytes;
}
