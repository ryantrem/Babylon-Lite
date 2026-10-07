/** TrueType glyf/loca reader. Default-instance outlines only; gvar is not applied.
 *
 *  Component recursion is bounded at 32 and cycles throw. Empty loca entries are valid
 *  glyphs with no geometry. Malformed contours, offsets, or point attachments throw.
 *  This module does not import text-shaper. */

import type { GlyphBounds } from "../glyph-storage.js";
import type { OutlinePoint } from "../outline-path.js";
import type { SfntDirectory } from "./sfnt.js";
import { requireTable } from "./sfnt.js";
import { openCursor, readF2Dot14, readI8, readI16, readU8, readU16, requireCount } from "./reader.js";

/** Hard cap matching the bounded composite walk. Exceeding it throws. */
export const MAX_GLYF_COMPONENT_DEPTH = 32;

const ON_CURVE = 0x01;
const X_SHORT = 0x02;
const Y_SHORT = 0x04;
const REPEAT = 0x08;
const X_SAME = 0x10;
const Y_SAME = 0x20;
const ARG_WORDS = 0x0001;
const ARGS_XY = 0x0002;
const ROUND_XY = 0x0004;
const WE_HAVE_SCALE = 0x0008;
const MORE_COMPONENTS = 0x0020;
const X_AND_Y_SCALE = 0x0040;
const TWO_BY_TWO = 0x0080;
const WE_HAVE_INSTRUCTIONS = 0x0100;
const SCALED_OFFSET = 0x0800;
const UNSCALED_OFFSET = 0x1000;

export interface GlyfComponent {
    readonly glyphId: number;
    readonly flags: number;
}

export interface GlyfGlyph {
    readonly glyphId: number;
    readonly empty: boolean;
    readonly composite: boolean;
    readonly contours: readonly (readonly OutlinePoint[])[];
    readonly bounds: GlyphBounds | null;
    readonly componentGlyphIds: readonly number[];
}

interface SimpleGlyph {
    readonly kind: "simple";
    readonly contours: OutlinePoint[][];
    readonly bounds: GlyphBounds;
}

interface CompositeSource {
    readonly kind: "composite";
    readonly bounds: GlyphBounds;
    readonly components: {
        readonly glyphId: number;
        readonly flags: number;
        readonly arg1: number;
        readonly arg2: number;
        readonly m00: number;
        readonly m10: number;
        readonly m01: number;
        readonly m11: number;
    }[];
}

export function readGlyfGlyph(bytes: Uint8Array, directory: SfntDirectory, locaFormat: 0 | 1, glyphCount: number, glyphId: number): GlyfGlyph {
    return readGlyph(bytes, directory, locaFormat, glyphCount, glyphId, new Set());
}

function readGlyph(bytes: Uint8Array, directory: SfntDirectory, locaFormat: 0 | 1, glyphCount: number, glyphId: number, visiting: Set<number>): GlyfGlyph {
    if (!Number.isInteger(glyphId) || glyphId < 0 || glyphId >= glyphCount) {
        throw new Error(`glyf: glyph ${glyphId} is outside 0..${glyphCount - 1}`);
    }
    if (visiting.has(glyphId)) {
        throw new Error(`glyf: component cycle at glyph ${glyphId}`);
    }
    if (visiting.size >= MAX_GLYF_COMPONENT_DEPTH) {
        throw new Error(`glyf: component depth exceeds ${MAX_GLYF_COMPONENT_DEPTH} at glyph ${glyphId}`);
    }
    const span = glyphSpan(bytes, directory, locaFormat, glyphCount, glyphId);
    if (span.length === 0) {
        return { glyphId, empty: true, composite: false, contours: [], bounds: null, componentGlyphIds: [] };
    }
    const parsed = parseGlyphData(bytes, span.offset, span.length, glyphId);
    if (parsed.kind === "simple") {
        const empty = parsed.contours.every((contour) => contour.length === 0);
        return {
            glyphId,
            empty,
            composite: false,
            contours: parsed.contours,
            bounds: empty ? null : parsed.bounds,
            componentGlyphIds: [],
        };
    }
    visiting.add(glyphId);
    const contours: OutlinePoint[][] = [];
    const componentGlyphIds: number[] = [];
    const placed: OutlinePoint[] = [];
    for (const component of parsed.components) {
        componentGlyphIds.push(component.glyphId);
        const child = readGlyph(bytes, directory, locaFormat, glyphCount, component.glyphId, visiting);
        const flat = child.contours.flat();
        const delta = componentDelta(component, flat, placed, glyphId);
        for (const contour of child.contours) {
            const transformed = contour.map((point) => transformPoint(component, point, delta));
            contours.push(transformed);
            placed.push(...transformed);
        }
    }
    visiting.delete(glyphId);
    if (placed.length === 0) {
        return { glyphId, empty: true, composite: true, contours: [], bounds: null, componentGlyphIds };
    }
    let xMin = placed[0]?.x ?? 0;
    let yMin = placed[0]?.y ?? 0;
    let xMax = xMin;
    let yMax = yMin;
    for (const point of placed) {
        xMin = Math.min(xMin, point.x);
        yMin = Math.min(yMin, point.y);
        xMax = Math.max(xMax, point.x);
        yMax = Math.max(yMax, point.y);
    }
    return { glyphId, empty: false, composite: true, contours, bounds: { xMin, yMin, xMax, yMax }, componentGlyphIds };
}

function glyphSpan(bytes: Uint8Array, directory: SfntDirectory, locaFormat: 0 | 1, glyphCount: number, glyphId: number): { offset: number; length: number } {
    const loca = requireTable(directory, "loca");
    const glyf = requireTable(directory, "glyf");
    const count = glyphCount + 1;
    const offsets = locaFormat === 0 ? readShortLoca(bytes, loca.offset, loca.length, count) : readLongLoca(bytes, loca.offset, loca.length, count);
    const start = offsets[glyphId] ?? 0;
    const end = offsets[glyphId + 1] ?? 0;
    if (end < start || end > glyf.length) {
        throw new Error(`glyf: loca span ${start}..${end} for glyph ${glyphId} is outside glyf`);
    }
    return { offset: glyf.offset + start, length: end - start };
}

function readShortLoca(bytes: Uint8Array, offset: number, length: number, count: number): number[] {
    requireCount(count, 2, length, "loca");
    const cursor = openCursor(bytes, offset, count * 2, "loca");
    const offsets: number[] = [];
    let previous = 0;
    for (let index = 0; index < count; index++) {
        const value = readU16(cursor, "loca offset") * 2;
        if (value < previous) {
            throw new Error("loca: offsets are not monotonic");
        }
        offsets.push(value);
        previous = value;
    }
    return offsets;
}

function readLongLoca(bytes: Uint8Array, offset: number, length: number, count: number): number[] {
    requireCount(count, 4, length, "loca");
    const cursor = openCursor(bytes, offset, count * 4, "loca");
    const offsets: number[] = [];
    let previous = 0;
    for (let index = 0; index < count; index++) {
        const value = readU32Local(cursor, "loca offset");
        if (value < previous) {
            throw new Error("loca: offsets are not monotonic");
        }
        offsets.push(value);
        previous = value;
    }
    return offsets;
}

function readU32Local(cursor: { bytes: Uint8Array; offset: number; limit: number }, label: string): number {
    if (cursor.offset + 4 > cursor.limit) {
        throw new Error(`${label}: truncated`);
    }
    const bytes = cursor.bytes;
    const at = cursor.offset;
    cursor.offset += 4;
    return ((bytes[at] ?? 0) * 0x1000000 + (bytes[at + 1] ?? 0) * 0x10000 + (bytes[at + 2] ?? 0) * 0x100 + (bytes[at + 3] ?? 0)) >>> 0;
}

function parseGlyphData(bytes: Uint8Array, offset: number, length: number, glyphId: number): SimpleGlyph | CompositeSource {
    if (length < 10) {
        throw new Error(`glyf: glyph ${glyphId} header is truncated`);
    }
    const cursor = openCursor(bytes, offset, length, `glyf ${glyphId}`);
    const numberOfContours = readI16(cursor, `glyf ${glyphId} numberOfContours`);
    const xMin = readI16(cursor, `glyf ${glyphId} xMin`);
    const yMin = readI16(cursor, `glyf ${glyphId} yMin`);
    const xMax = readI16(cursor, `glyf ${glyphId} xMax`);
    const yMax = readI16(cursor, `glyf ${glyphId} yMax`);
    const bounds = { xMin, yMin, xMax, yMax };
    if (numberOfContours < -1) {
        throw new Error(`glyf: glyph ${glyphId} has invalid contour count ${numberOfContours}`);
    }
    if (numberOfContours === -1) {
        return { kind: "composite", bounds, components: parseComponents(cursor, glyphId) };
    }
    return { kind: "simple", bounds, contours: parseSimple(cursor, numberOfContours, glyphId) };
}

function parseSimple(cursor: ReturnType<typeof openCursor>, numberOfContours: number, glyphId: number): OutlinePoint[][] {
    if (numberOfContours === 0) {
        return [];
    }
    requireCount(numberOfContours, 2, cursor.limit - cursor.offset, `glyf ${glyphId} endPts`);
    const ends: number[] = [];
    let previous = -1;
    for (let index = 0; index < numberOfContours; index++) {
        const end = readU16(cursor, `glyf ${glyphId} endPts`);
        if (end <= previous) {
            throw new Error(`glyf: glyph ${glyphId} contour ends are not strictly increasing`);
        }
        ends.push(end);
        previous = end;
    }
    const pointCount = (ends[ends.length - 1] ?? -1) + 1;
    const instructionLength = readU16(cursor, `glyf ${glyphId} instructionLength`);
    requireCount(instructionLength, 1, cursor.limit - cursor.offset, `glyf ${glyphId} instructions`);
    cursor.offset += instructionLength;
    const flags: number[] = [];
    while (flags.length < pointCount) {
        const flag = readU8(cursor, `glyf ${glyphId} flag`);
        flags.push(flag);
        if (flag & REPEAT) {
            const repeat = readU8(cursor, `glyf ${glyphId} flag repeat`);
            if (flags.length + repeat > pointCount) {
                throw new Error(`glyf: glyph ${glyphId} flag repeat exceeds the point count`);
            }
            for (let index = 0; index < repeat; index++) {
                flags.push(flag);
            }
        }
    }
    const xs: number[] = [];
    let x = 0;
    for (const flag of flags) {
        if (flag & X_SHORT) {
            const delta = readU8(cursor, `glyf ${glyphId} x`);
            x += flag & X_SAME ? delta : -delta;
        } else if (!(flag & X_SAME)) {
            x += readI16(cursor, `glyf ${glyphId} x`);
        }
        xs.push(x);
    }
    const ys: number[] = [];
    let y = 0;
    for (const flag of flags) {
        if (flag & Y_SHORT) {
            const delta = readU8(cursor, `glyf ${glyphId} y`);
            y += flag & Y_SAME ? delta : -delta;
        } else if (!(flag & Y_SAME)) {
            y += readI16(cursor, `glyf ${glyphId} y`);
        }
        ys.push(y);
    }
    const contours: OutlinePoint[][] = [];
    let point = 0;
    for (const end of ends) {
        const contour: OutlinePoint[] = [];
        while (point <= end) {
            contour.push({ x: xs[point] ?? 0, y: ys[point] ?? 0, onCurve: ((flags[point] ?? 0) & ON_CURVE) !== 0 });
            point++;
        }
        contours.push(contour);
    }
    return contours;
}

function parseComponents(cursor: ReturnType<typeof openCursor>, glyphId: number): CompositeSource["components"] {
    const components: CompositeSource["components"] = [];
    let flags = 0;
    do {
        flags = readU16(cursor, `glyf ${glyphId} component flags`);
        const componentId = readU16(cursor, `glyf ${glyphId} component glyph`);
        let arg1: number;
        let arg2: number;
        if (flags & ARG_WORDS) {
            arg1 = flags & ARGS_XY ? readI16(cursor, `glyf ${glyphId} arg1`) : readU16(cursor, `glyf ${glyphId} arg1`);
            arg2 = flags & ARGS_XY ? readI16(cursor, `glyf ${glyphId} arg2`) : readU16(cursor, `glyf ${glyphId} arg2`);
        } else {
            arg1 = flags & ARGS_XY ? readI8(cursor, `glyf ${glyphId} arg1`) : readU8(cursor, `glyf ${glyphId} arg1`);
            arg2 = flags & ARGS_XY ? readI8(cursor, `glyf ${glyphId} arg2`) : readU8(cursor, `glyf ${glyphId} arg2`);
        }
        let m00 = 1;
        let m10 = 0;
        let m01 = 0;
        let m11 = 1;
        if (flags & WE_HAVE_SCALE) {
            m00 = m11 = readF2Dot14(cursor, `glyf ${glyphId} scale`);
        } else if (flags & X_AND_Y_SCALE) {
            m00 = readF2Dot14(cursor, `glyf ${glyphId} xScale`);
            m11 = readF2Dot14(cursor, `glyf ${glyphId} yScale`);
        } else if (flags & TWO_BY_TWO) {
            m00 = readF2Dot14(cursor, `glyf ${glyphId} m00`);
            m10 = readF2Dot14(cursor, `glyf ${glyphId} m10`);
            m01 = readF2Dot14(cursor, `glyf ${glyphId} m01`);
            m11 = readF2Dot14(cursor, `glyf ${glyphId} m11`);
        }
        components.push({ glyphId: componentId, flags, arg1, arg2, m00, m10, m01, m11 });
    } while (flags & MORE_COMPONENTS);
    if (flags & WE_HAVE_INSTRUCTIONS) {
        const instructionLength = readU16(cursor, `glyf ${glyphId} component instructions`);
        requireCount(instructionLength, 1, cursor.limit - cursor.offset, `glyf ${glyphId} component instructions`);
    }
    return components;
}

function componentDelta(
    component: CompositeSource["components"][number],
    points: readonly OutlinePoint[],
    placed: readonly OutlinePoint[],
    parentId: number
): { dx: number; dy: number } {
    if (component.flags & ARGS_XY) {
        let dx = component.arg1;
        let dy = component.arg2;
        if (scaleOffset(component)) {
            const scaledX = component.m00 * dx + component.m01 * dy;
            const scaledY = component.m10 * dx + component.m11 * dy;
            dx = scaledX;
            dy = scaledY;
        }
        if (component.flags & ROUND_XY) {
            dx = Math.round(dx);
            dy = Math.round(dy);
        }
        return { dx, dy };
    }
    const parent = placed[component.arg1];
    const child = points[component.arg2];
    if (!parent || !child) {
        throw new Error(`glyf: glyph ${parentId} point attachment ${component.arg1},${component.arg2} is outside the contours`);
    }
    return {
        dx: parent.x - (component.m00 * child.x + component.m01 * child.y),
        dy: parent.y - (component.m10 * child.x + component.m11 * child.y),
    };
}

function transformPoint(component: CompositeSource["components"][number], point: OutlinePoint, delta: { dx: number; dy: number }): OutlinePoint {
    return {
        x: Math.round(component.m00 * point.x + component.m01 * point.y + delta.dx),
        y: Math.round(component.m10 * point.x + component.m11 * point.y + delta.dy),
        onCurve: point.onCurve,
    };
}

function scaleOffset(component: CompositeSource["components"][number]): boolean {
    if (component.flags & UNSCALED_OFFSET) {
        return false;
    }
    if (component.flags & SCALED_OFFSET) {
        return true;
    }
    return component.m00 !== 1 || component.m10 !== 0 || component.m01 !== 0 || component.m11 !== 1;
}
