/** Compute outline extraction. text-shaper is an oracle for glyf curves only; the reader does not import it. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { createComputeFontFromBuffer } from "../../../packages/babylon-lite/src/text/compute-font";
import { extractComputeGlyphCurves } from "../../../packages/babylon-lite/src/text/compute-glyph-extraction";
import { cubicToQuadratics as movedCubic } from "../../../packages/babylon-lite/src/text/cubic-to-quadratics";
import { MAX_CFF_CALL_DEPTH, MAX_CFF_OPERATIONS, MAX_CFF_STACK } from "../../../packages/babylon-lite/src/text/font-parser/cff";
import { MAX_GLYF_COMPONENT_DEPTH, readGlyfGlyph } from "../../../packages/babylon-lite/src/text/font-parser/glyf";
import { cubicToQuadratics } from "../../../packages/babylon-lite/src/text/glyph-extraction";
import type { GlyphCurves, QuadCurve } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { Font, getGlyphPath } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

function fontBytes(file: string): ArrayBuffer {
    const buf = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

describe("extractComputeGlyphCurves", () => {
    it("keeps the public cubic helper identical and dependency-free", () => {
        expect(cubicToQuadratics).toBe(movedCubic);
        expect(cubicToQuadratics(0, 0, 0, 100, 100, 100, 100, 0)).toEqual(movedCubic(0, 0, 0, 100, 100, 100, 100, 0));
        const source = readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/compute-glyph-extraction.ts"), "utf8");
        expect(source).not.toContain('from "text-shaper"');
        expect(readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/glyph-extraction.ts"), "utf8")).toContain('from "text-shaper"');
    });

    it("matches text-shaper outlines for Inter glyf glyphs, including a composite and an implied point", () => {
        const bytes = fontBytes("Inter.ttf");
        const font = createComputeFontFromBuffer(bytes);
        const oracle = Font.load(bytes);
        expect(font.outlineFormat).toBe("truetype");
        const glyphIds = [oracle.glyphId(0x41), oracle.glyphId(0x6f), oracle.glyphId(0x00e9), oracle.glyphId(0x41)];
        const target = new Map<number, GlyphCurves>();
        extractComputeGlyphCurves(font, new Set(glyphIds), target);
        for (const glyphId of new Set(glyphIds)) {
            expect(curvesOf(target.get(glyphId))).toEqual(curvesFromOracle(oracle, glyphId));
        }
        const accented = readGlyfGlyph(font._bytes, font._directory, font._locaFormat, font.glyphCount, oracle.glyphId(0x00e9));
        expect(accented.composite).toBe(true);
        expect(accented.componentGlyphIds.length).toBeGreaterThan(0);
        expect(hasImpliedPoint(font, oracle.glyphId(0x6f))).toBe(true);
    });

    it("reads Roboto as glyf and leaves an empty glyph out of the target", () => {
        const font = createComputeFontFromBuffer(fontBytes("Roboto-Regular.ttf"));
        expect(font.outlineFormat).toBe("truetype");
        const empty = firstEmpty(font);
        expect(empty).not.toBeNull();
        const target = new Map<number, GlyphCurves>();
        extractComputeGlyphCurves(font, new Set([empty as number]), target);
        expect(target.size).toBe(0);
    });

    it("does not mutate the target when a glyph is malformed or out of range", () => {
        const font = createComputeFontFromBuffer(syntheticGlyf());
        const target = new Map<number, GlyphCurves>();
        extractComputeGlyphCurves(font, new Set([1]), target);
        expect(target.size).toBe(1);
        const before = new Map(target);
        expect(() => extractComputeGlyphCurves(font, new Set([1, 3]), target)).toThrow(/invalid contour count/);
        expect(target).toEqual(before);
        expect(() => extractComputeGlyphCurves(font, new Set([99]), target)).toThrow(/outside/);
        expect(target).toEqual(before);
        expect(() => extractComputeGlyphCurves(font, new Set([4]), target)).toThrow(/component cycle/);
        expect(MAX_GLYF_COMPONENT_DEPTH).toBe(32);
    });

    it("converts a synthetic CFF cubic, including a local subroutine, and rejects a truncated charstring", () => {
        const font = createComputeFontFromBuffer(syntheticCff());
        expect(font.outlineFormat).toBe("cff");
        const target = new Map<number, GlyphCurves>();
        extractComputeGlyphCurves(font, new Set([0, 1]), target);
        expect(target.has(0)).toBe(false);
        const [first, second] = cubicToQuadratics(100, 100, 150, 100, 200, 150, 200, 200);
        expect(target.get(1)?.curves).toEqual([first, second, { p0x: 200, p0y: 200, p1x: 150, p1y: 150, p2x: 100, p2y: 100 }]);
        const before = new Map(target);
        expect(() => extractComputeGlyphCurves(font, new Set([2]), target)).toThrow(/CFF/);
        expect(target).toEqual(before);
        expect(MAX_CFF_STACK).toBe(48);
        expect(MAX_CFF_CALL_DEPTH).toBe(10);
        expect(MAX_CFF_OPERATIONS).toBeGreaterThan(0);
    });

    it("bounds CFF recursive local subroutines and rejects truncated number operands atomically", () => {
        for (const subroutine of [Uint8Array.from([32, 10, 11]), Uint8Array.from([28, 0])]) {
            const font = createComputeFontFromBuffer(syntheticCff(subroutine));
            const target = new Map<number, GlyphCurves>();
            expect(() => extractComputeGlyphCurves(font, new Set([1]), target)).toThrow(/CFF/);
            expect(target.size).toBe(0);
        }
    });

    it("extracts a synthetic composite translation and rejects a truncated component header", () => {
        const font = createComputeFontFromBuffer(syntheticGlyf());
        const target = new Map<number, GlyphCurves>();
        extractComputeGlyphCurves(font, new Set([1, 2]), target);
        expect(target.get(2)?.curves).toEqual(
            target.get(1)?.curves.map((curve) => ({
                p0x: curve.p0x + 20,
                p0y: curve.p0y,
                p1x: curve.p1x + 20,
                p1y: curve.p1y,
                p2x: curve.p2x + 20,
                p2y: curve.p2y,
            }))
        );
        const truncated = createComputeFontFromBuffer(syntheticGlyf(true));
        expect(() => extractComputeGlyphCurves(truncated, new Set([2]), new Map())).toThrow(/truncated/);
    });
});

function curvesOf(curves: GlyphCurves | undefined): QuadCurve[] {
    return curves?.curves.map((curve) => ({ ...curve })) ?? [];
}

function curvesFromOracle(font: Font, glyphId: number): QuadCurve[] {
    const path = getGlyphPath(font, glyphId);
    if (!path?.bounds || path.commands.length === 0) {
        return [];
    }
    const curves: QuadCurve[] = [];
    let x = 0;
    let y = 0;
    let startX = 0;
    let startY = 0;
    for (const command of path.commands) {
        if (command.type === "M") {
            x = command.x;
            y = command.y;
            startX = command.x;
            startY = command.y;
        } else if (command.type === "L") {
            pushLine(curves, x, y, command.x, command.y);
            x = command.x;
            y = command.y;
        } else if (command.type === "Q") {
            curves.push({ p0x: x, p0y: y, p1x: command.x1, p1y: command.y1, p2x: command.x, p2y: command.y });
            x = command.x;
            y = command.y;
        } else if (command.type === "C") {
            curves.push(...cubicToQuadratics(x, y, command.x1, command.y1, command.x2, command.y2, command.x, command.y));
            x = command.x;
            y = command.y;
        } else if (command.type === "Z") {
            const dx = startX - x;
            const dy = startY - y;
            if (Math.abs(dx) > 0.1 || Math.abs(dy) > 0.1) {
                curves.push({ p0x: x, p0y: y, p1x: (x + startX) * 0.5, p1y: (y + startY) * 0.5, p2x: startX, p2y: startY });
            }
            x = startX;
            y = startY;
        }
    }
    return curves;
}

function pushLine(curves: QuadCurve[], x0: number, y0: number, x1: number, y1: number): void {
    if (Math.abs(x1 - x0) >= 0.1 || Math.abs(y1 - y0) >= 0.1) {
        curves.push({ p0x: x0, p0y: y0, p1x: (x0 + x1) * 0.5, p1y: (y0 + y1) * 0.5, p2x: x1, p2y: y1 });
    }
}

function hasImpliedPoint(font: ReturnType<typeof createComputeFontFromBuffer>, glyphId: number): boolean {
    const glyph = readGlyfGlyph(font._bytes, font._directory, font._locaFormat, font.glyphCount, glyphId);
    return glyph.contours.some((contour) => contour.some((point, index) => !point.onCurve && !contour[(index + 1) % contour.length]?.onCurve));
}

function firstEmpty(font: ReturnType<typeof createComputeFontFromBuffer>): number | null {
    for (let glyphId = 0; glyphId < Math.min(font.glyphCount, 64); glyphId++) {
        if (readGlyfGlyph(font._bytes, font._directory, font._locaFormat, font.glyphCount, glyphId).empty) {
            return glyphId;
        }
    }
    return null;
}

function syntheticGlyf(truncatedComposite = false): ArrayBuffer {
    const simple = simpleSquare();
    const composite = truncatedComposite ? compositeOf(1, 20, 0).slice(0, 12) : compositeOf(1, 20, 0);
    const malformed = new Uint8Array(10);
    new DataView(malformed.buffer).setInt16(0, -2);
    const cycle = compositeOf(4, 0, 0);
    const glyphs = [new Uint8Array(0), simple, composite, malformed, cycle];
    const loca = new Uint8Array((glyphs.length + 1) * 4);
    const locaView = new DataView(loca.buffer);
    let cursor = 0;
    const glyfParts: Uint8Array[] = [];
    glyphs.forEach((glyph, index) => {
        locaView.setUint32(index * 4, cursor);
        glyfParts.push(glyph);
        cursor += glyph.length;
    });
    locaView.setUint32(glyphs.length * 4, cursor);
    const glyf = concat(glyfParts);
    return buildSfnt(0x00010000, {
        cmap: cmapFormat4(0x41, 1),
        glyf,
        head: headTable(16, 1),
        hhea: hheaTable(glyphs.length),
        hmtx: hmtxTable(glyphs.length),
        loca,
        maxp: maxpTable(glyphs.length),
    });
}

function simpleSquare(): Uint8Array {
    const writer = new Writer();
    writer.i16(1);
    writer.i16(0);
    writer.i16(0);
    writer.i16(100);
    writer.i16(0);
    writer.u16(1);
    writer.u16(0);
    writer.u8(0x37);
    writer.u8(0x37);
    writer.u8(0);
    writer.u8(100);
    writer.u8(0);
    writer.u8(0);
    return writer.toUint8();
}

function compositeOf(glyphId: number, dx: number, dy: number): Uint8Array {
    const writer = new Writer();
    writer.i16(-1);
    writer.i16(0);
    writer.i16(0);
    writer.i16(10);
    writer.i16(10);
    writer.u16(0x0003);
    writer.u16(glyphId);
    writer.i16(dx);
    writer.i16(dy);
    return writer.toUint8();
}

function syntheticCff(subroutine?: Uint8Array): ArrayBuffer {
    const cff = buildCff(subroutine);
    return buildSfnt(0x4f54544f, {
        "CFF ": cff,
        cmap: cmapFormat4(0x41, 1),
        head: headTable(16, 0),
        hhea: hheaTable(3),
        hmtx: hmtxTable(3),
        maxp: cffMaxp(3),
    });
}

function buildCff(subroutine?: Uint8Array): Uint8Array {
    const name = cffIndex([utf8("Test")]);
    const strings = cffIndex([]);
    const globalSubrs = cffIndex([]);
    const localSubr = subroutine ?? Uint8Array.from([189, 139, 189, 189, 139, 189, 8, 11]);
    const localSubrs = cffIndex([localSubr]);
    const privateBody = dictBytes([[4, 19]]);
    const charStrings = cffIndex([Uint8Array.from([14]), Uint8Array.from([239, 239, 21, 32, 10, 14]), Uint8Array.from([8])]);
    const charset = Uint8Array.from([0, 0, 1, 0, 2]);
    const topPlaceholder = dictBytes([
        [0, 15],
        [0, 17],
        [privateBody.length, 0, 18],
        [0.0625, 0, 0, 0.0625, 0, 0, 0x0c07],
    ]);
    const top = cffIndex([topPlaceholder]);
    const charsetAt = 4 + name.length + top.length + strings.length + globalSubrs.length;
    const charStringsAt = charsetAt + charset.length;
    const privateAt = charStringsAt + charStrings.length;
    const patchedTop = cffIndex([
        dictBytes([
            [charsetAt, 15],
            [charStringsAt, 17],
            [privateBody.length, privateAt, 18],
            [0.0625, 0, 0, 0.0625, 0, 0, 0x0c07],
        ]),
    ]);
    if (patchedTop.length !== top.length) {
        throw new Error("synthetic CFF top dict size changed");
    }
    return concat([Uint8Array.from([1, 0, 4, 1]), name, patchedTop, strings, globalSubrs, charset, charStrings, privateBody, localSubrs]);
}

function dictBytes(items: (number | number[])[]): Uint8Array {
    const bytes: number[] = [];
    for (const item of items) {
        const values = Array.isArray(item) ? item : [item];
        const operator = values[values.length - 1] ?? 0;
        for (const value of values.slice(0, -1)) {
            bytes.push(...cffOperand(value));
        }
        if (operator > 0xff) {
            bytes.push(12, operator & 0xff);
        } else {
            bytes.push(operator);
        }
    }
    return Uint8Array.from(bytes);
}

function cffOperand(value: number): number[] {
    if (value === 0.0625) {
        return [255, 0, 0, 16, 0];
    }
    if (!Number.isInteger(value) || value < -32768 || value > 32767) {
        throw new Error(`synthetic CFF cannot encode ${value}`);
    }
    const signed = value < 0 ? value + 0x10000 : value;
    return [28, (signed >> 8) & 0xff, signed & 0xff];
}

function cffIndex(items: Uint8Array[]): Uint8Array {
    if (items.length === 0) {
        return Uint8Array.from([0, 0]);
    }
    const offsets = [1];
    for (const item of items) {
        offsets.push((offsets[offsets.length - 1] ?? 1) + item.length);
    }
    const offSize = 1;
    const bytes = [items.length >> 8, items.length & 0xff, offSize];
    for (const offset of offsets) {
        bytes.push(offset);
    }
    return concat([Uint8Array.from(bytes), ...items]);
}

function utf8(value: string): Uint8Array {
    return Uint8Array.from(value, (char) => char.charCodeAt(0));
}

function cffMaxp(glyphCount: number): Uint8Array {
    const bytes = new Uint8Array(6);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x00005000);
    view.setUint16(4, glyphCount);
    return bytes;
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

function hheaTable(metrics: number): Uint8Array {
    const bytes = new Uint8Array(36);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint16(34, metrics);
    return bytes;
}

function hmtxTable(glyphCount: number): Uint8Array {
    const bytes = new Uint8Array(glyphCount * 4);
    const view = new DataView(bytes.buffer);
    for (let index = 0; index < glyphCount; index++) {
        view.setUint16(index * 4, 100);
    }
    return bytes;
}

function maxpTable(glyphCount: number): Uint8Array {
    const bytes = new Uint8Array(32);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, glyphCount);
    return bytes;
}

function cmapFormat4(codePoint: number, glyphId: number): Uint8Array {
    const bytes = new Uint8Array(44);
    const view = new DataView(bytes.buffer);
    view.setUint16(2, 1);
    view.setUint16(6, 3);
    view.setUint32(8, 12);
    const at = 12;
    view.setUint16(at, 4);
    view.setUint16(at + 2, 32);
    view.setUint16(at + 6, 4);
    view.setUint16(at + 8, 4);
    view.setUint16(at + 10, 1);
    view.setUint16(at + 12, 2);
    view.setUint16(at + 14, codePoint);
    view.setUint16(at + 16, 0xffff);
    view.setUint16(at + 20, codePoint);
    view.setUint16(at + 22, 0xffff);
    view.setInt16(at + 26, glyphId - codePoint);
    return bytes;
}

function buildSfnt(version: number, tables: Record<string, Uint8Array>): ArrayBuffer {
    const tags = Object.keys(tables);
    const out = new Uint8Array(12 + tags.length * 16 + tags.reduce((sum, tag) => sum + (tables[tag]?.length ?? 0), 0));
    const view = new DataView(out.buffer);
    view.setUint32(0, version);
    view.setUint16(4, tags.length);
    let offset = 12 + tags.length * 16;
    tags.forEach((tag, index) => {
        const body = tables[tag] ?? new Uint8Array();
        const at = 12 + index * 16;
        for (let byte = 0; byte < 4; byte++) {
            view.setUint8(at + byte, tag.charCodeAt(byte) || 0x20);
        }
        view.setUint32(at + 8, offset);
        view.setUint32(at + 12, body.length);
        out.set(body, offset);
        offset += body.length;
    });
    return out.buffer;
}

function concat(parts: Uint8Array[]): Uint8Array {
    const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
    let offset = 0;
    for (const part of parts) {
        out.set(part, offset);
        offset += part.length;
    }
    return out;
}

class Writer {
    readonly bytes: number[] = [];

    u16(value: number): void {
        this.bytes.push((value >> 8) & 0xff, value & 0xff);
    }

    i16(value: number): void {
        this.u16(value < 0 ? value + 0x10000 : value);
    }

    u8(value: number): void {
        this.bytes.push(value & 0xff);
    }

    toUint8(): Uint8Array {
        return Uint8Array.from(this.bytes);
    }
}
