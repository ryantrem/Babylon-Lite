/** CFF1 CharString reader. Unsupported operators and CID fonts throw; they never return a partial outline.
 *
 *  Stack, call, and operator budgets are fixed. FontMatrix maps charstring space into ems,
 *  then into font units. This module does not import text-shaper. */

import type { GlyphBounds, GlyphCurves } from "../glyph-storage.js";
import type { SfntTable } from "./sfnt.js";
import { createOutlineRecorder, outlineClose, outlineCubicTo, outlineLineTo, outlineMoveTo } from "../outline-path.js";
import type { OutlineRecorder } from "../outline-path.js";

/** Type 2 transient stack limit. */
export const MAX_CFF_STACK = 48;
/** Type 2 subroutine call depth. */
export const MAX_CFF_CALL_DEPTH = 10;
/** Fixed operator budget, also capped by table size in the caller. */
export const MAX_CFF_OPERATIONS = 100_000;

const DEFAULT_MATRIX = [0.001, 0, 0, 0.001, 0, 0];

interface CffIndex {
    readonly count: number;
    readonly data: Uint8Array;
    readonly offsets: readonly number[];
}

interface CffFont {
    readonly matrix: readonly number[];
    readonly charStrings: CffIndex;
    readonly globalSubrs: CffIndex;
    readonly localSubrs: CffIndex;
    readonly defaultWidthX: number;
    readonly nominalWidthX: number;
    readonly operationBudget: number;
}

export function readCffGlyph(bytes: Uint8Array, table: SfntTable, glyphId: number, glyphCount: number, unitsPerEm: number): GlyphCurves | null {
    if (!Number.isInteger(glyphId) || glyphId < 0 || glyphId >= glyphCount) {
        throw new Error(`CFF: glyph ${glyphId} is outside 0..${glyphCount - 1}`);
    }
    const font = parseCff(bytes, table, glyphCount);
    if (glyphId >= font.charStrings.count) {
        throw new Error(`CFF: glyph ${glyphId} is outside the CharStrings INDEX`);
    }
    const program = indexBytes(font.charStrings, glyphId, "CFF CharString");
    if (program.length === 0) {
        throw new Error(`CFF: glyph ${glyphId} CharString is empty`);
    }
    const recorder = createOutlineRecorder();
    executeCharString(font, program, recorder, [], 0, { operations: 0, widthSeen: false });
    const bounds = boundsOf(recorder, unitsPerEm, font.matrix);
    return scaleRecorder(recorder, glyphId, unitsPerEm, font.matrix, bounds);
}

function parseCff(bytes: Uint8Array, table: SfntTable, glyphCount: number): CffFont {
    const start = table.offset;
    const limit = table.offset + table.length;
    if (table.length < 4) {
        throw new Error("CFF: truncated header");
    }
    const major = bytes[start] ?? 0;
    const headerSize = bytes[start + 2] ?? 0;
    if (major !== 1) {
        throw new Error(`CFF: unsupported major version ${major}`);
    }
    if (headerSize < 4 || start + headerSize > limit) {
        throw new Error("CFF: invalid header size");
    }
    let cursor = start + headerSize;
    const name = readIndex(bytes, cursor, limit, "CFF Name");
    cursor = name.end;
    const top = readIndex(bytes, cursor, limit, "CFF Top DICT");
    cursor = top.end;
    const strings = readIndex(bytes, cursor, limit, "CFF String");
    cursor = strings.end;
    const globalSubrs = readIndex(bytes, cursor, limit, "CFF Global Subr");
    if (top.count < 1) {
        throw new Error("CFF: Top DICT INDEX is empty");
    }
    const topDict = parseDict(indexBytes(top, 0, "CFF Top DICT"));
    if (topDict.operators.has(0x0c1e)) {
        throw new Error("CFF: CID fonts are unsupported");
    }
    const charStringsOffset = requiredOffset(topDict, 17, "CFF CharStrings");
    const charsetOffset = dictNumber(topDict, 15) ?? 0;
    const privateEntry = topDict.operators.get(18);
    const charStrings = readIndex(bytes, checked(start, charStringsOffset, limit, "CFF CharStrings"), limit, "CFF CharStrings");
    if (charStrings.count !== glyphCount) {
        throw new Error(`CFF: CharStrings count ${charStrings.count} does not match glyphCount ${glyphCount}`);
    }
    if (charsetOffset !== 0) {
        validateCharset(bytes, checked(start, charsetOffset, limit, "CFF charset"), limit, charStrings.count);
    }
    let localSubrs = emptyIndex();
    let defaultWidthX = 0;
    let nominalWidthX = 0;
    if (privateEntry && privateEntry.length >= 2) {
        const size = privateEntry[0] ?? 0;
        const offset = privateEntry[1] ?? 0;
        if (size !== 0) {
            const privateStart = checked(start, offset, limit, "CFF Private");
            if (privateStart + size > limit) {
                throw new Error("CFF: Private DICT is outside the table");
            }
            const privateDict = parseDict(bytes.subarray(privateStart, privateStart + size));
            defaultWidthX = dictNumber(privateDict, 20) ?? 0;
            nominalWidthX = dictNumber(privateDict, 21) ?? 0;
            const subrsOffset = dictNumber(privateDict, 19);
            if (subrsOffset) {
                localSubrs = readIndex(bytes, checked(privateStart, subrsOffset, limit, "CFF Subrs"), limit, "CFF Subrs");
            }
        }
    }
    const matrix = topDict.operators.get(0x0c07) ?? DEFAULT_MATRIX;
    if (matrix.length !== 6 || matrix.some((value) => !Number.isFinite(value))) {
        throw new Error("CFF: FontMatrix must contain six finite numbers");
    }
    return {
        matrix,
        charStrings,
        globalSubrs,
        localSubrs,
        defaultWidthX,
        nominalWidthX,
        operationBudget: Math.min(MAX_CFF_OPERATIONS, Math.max(table.length * 64, 64)),
    };
}

function executeCharString(font: CffFont, program: Uint8Array, recorder: OutlineRecorder, stack: number[], depth: number, state: { operations: number; widthSeen: boolean }): void {
    if (depth > MAX_CFF_CALL_DEPTH) {
        throw new Error(`CFF: subroutine depth exceeds ${MAX_CFF_CALL_DEPTH}`);
    }
    let hintCount = 0;
    let cursor = 0;
    const push = (value: number): void => {
        if (stack.length >= MAX_CFF_STACK) {
            throw new Error(`CFF: stack exceeds ${MAX_CFF_STACK}`);
        }
        stack.push(value);
    };
    while (cursor < program.length) {
        if (state.operations++ > font.operationBudget) {
            throw new Error(`CFF: operator budget ${font.operationBudget} exceeded`);
        }
        const byte = program[cursor] ?? 0;
        if (byte === 28 || byte === 255 || byte >= 32) {
            const operand = readOperand(program, cursor);
            push(operand.value);
            cursor = operand.next;
            continue;
        }
        cursor++;
        const operator = byte === 12 ? 0x0c00 + (program[cursor++] ?? 0) : byte;
        if (cursor > program.length) {
            throw new Error("CFF: truncated escape operator");
        }
        if (operator === 10 || operator === 29) {
            const index = pop(stack, "CFF callsubr") + bias(operator === 10 ? font.localSubrs.count : font.globalSubrs.count);
            const subrs = operator === 10 ? font.localSubrs : font.globalSubrs;
            executeCharString(font, indexBytes(subrs, index, "CFF subroutine"), recorder, stack, depth + 1, state);
            continue;
        }
        if (operator === 11 || operator === 14) {
            if (operator === 14) {
                applyOperator(font, operator, stack, recorder, state, (count) => {
                    hintCount += count;
                });
            }
            return;
        }
        if (operator === 19 || operator === 20) {
            consumeWidth(stack, operator, state, font);
            const stems = Math.floor(stack.length / 2);
            hintCount += stems;
            stack.length = 0;
            const maskBytes = Math.ceil(hintCount / 8);
            if (cursor + maskBytes > program.length) {
                throw new Error("CFF: hint mask is truncated");
            }
            cursor += maskBytes;
            continue;
        }
        applyOperator(font, operator, stack, recorder, state, (count) => {
            hintCount += count;
        });
    }
    throw new Error("CFF: CharString ended without endchar or return");
}

function applyOperator(font: CffFont, operator: number, stack: number[], recorder: OutlineRecorder, state: { widthSeen: boolean }, addHints: (count: number) => void): void {
    consumeWidth(stack, operator, state, font);
    switch (operator) {
        case 1:
        case 3:
        case 18:
        case 23:
            if (stack.length % 2 !== 0) {
                throw new Error(`CFF: stem operator ${operator} has an odd stack`);
            }
            addHints(stack.length / 2);
            stack.length = 0;
            return;
        case 4:
            outlineMoveTo(recorder, recorder.x, recorder.y + pop(stack, "CFF vmoveto"));
            stack.length = 0;
            return;
        case 22:
            outlineMoveTo(recorder, recorder.x + pop(stack, "CFF hmoveto"), recorder.y);
            stack.length = 0;
            return;
        case 21: {
            const dy = pop(stack, "CFF rmoveto dy");
            const dx = pop(stack, "CFF rmoveto dx");
            outlineMoveTo(recorder, recorder.x + dx, recorder.y + dy);
            stack.length = 0;
            return;
        }
        case 5:
            drawRelativeLines(recorder, stack, false, false);
            return;
        case 6:
        case 7:
            drawRelativeLines(recorder, stack, operator === 6, operator === 7);
            return;
        case 8:
            drawCurves(recorder, stack, 6, (values) => values);
            return;
        case 24:
            drawCurveThenLine(recorder, stack);
            return;
        case 25:
            drawLineThenCurve(recorder, stack);
            return;
        case 26:
        case 27:
            drawEvenCurves(recorder, stack, operator === 27);
            return;
        case 30:
        case 31:
            drawAlternatingCurves(recorder, stack, operator === 31);
            return;
        case 14:
            outlineClose(recorder);
            stack.length = 0;
            return;
        case 0x0c22:
            drawFlex(recorder, stack);
            return;
        case 0x0c23:
            drawFlex1(recorder, stack);
            return;
        case 0x0c24:
            drawHFlex1(recorder, stack);
            return;
        case 0x0c25:
            drawHFlex(recorder, stack);
            return;
        case 0x0c00:
            stack.length = 0;
            return;
        default:
            if (applyArithmetic(operator, stack)) {
                return;
            }
            throw new Error(`CFF: unsupported operator ${operator}`);
    }
}

function consumeWidth(stack: number[], operator: number, state: { widthSeen: boolean }, font: CffFont): void {
    if (state.widthSeen || !widthOperator(operator)) {
        return;
    }
    const extra = hasWidthOperand(operator, stack.length);
    if (extra) {
        const width = stack.shift();
        if (width === undefined || !Number.isFinite(width)) {
            throw new Error("CFF: invalid width operand");
        }
        if (Math.abs(width - font.nominalWidthX) > 0 && Math.abs(width - font.defaultWidthX) > 0xffff) {
            // Width is recorded only to consume the operand. Positioning does not use it.
        }
    }
    state.widthSeen = true;
}

function widthOperator(operator: number): boolean {
    return (
        operator === 1 ||
        operator === 3 ||
        operator === 4 ||
        operator === 14 ||
        operator === 18 ||
        operator === 19 ||
        operator === 20 ||
        operator === 21 ||
        operator === 22 ||
        operator === 23
    );
}

function hasWidthOperand(operator: number, stackLength: number): boolean {
    if (operator === 14) {
        return stackLength === 1;
    }
    if (operator === 4 || operator === 22) {
        return stackLength === 2;
    }
    if (operator === 21) {
        return stackLength === 3;
    }
    return stackLength % 2 === 1;
}

function drawRelativeLines(recorder: OutlineRecorder, stack: number[], horizontal: boolean, vertical: boolean): void {
    if (stack.length === 0 || (!horizontal && !vertical && stack.length % 2 !== 0)) {
        throw new Error("CFF: malformed line operands");
    }
    let horizontalNext = horizontal;
    for (let index = 0; index < stack.length;) {
        if (horizontal || vertical) {
            const delta = stack[index++] ?? 0;
            outlineLineTo(recorder, recorder.x + (horizontalNext ? delta : 0), recorder.y + (horizontalNext ? 0 : delta));
            horizontalNext = !horizontalNext;
        } else {
            const dx = stack[index++] ?? 0;
            const dy = stack[index++] ?? 0;
            outlineLineTo(recorder, recorder.x + dx, recorder.y + dy);
        }
    }
    stack.length = 0;
}

function drawCurves(recorder: OutlineRecorder, stack: number[], width: number, map: (values: number[]) => number[]): void {
    if (stack.length === 0 || stack.length % width !== 0) {
        throw new Error("CFF: malformed curve operands");
    }
    for (let index = 0; index < stack.length; index += width) {
        const values = map(stack.slice(index, index + width));
        curveFrom(recorder, values);
    }
    stack.length = 0;
}

function drawCurveThenLine(recorder: OutlineRecorder, stack: number[]): void {
    if (stack.length < 8 || (stack.length - 2) % 6 !== 0) {
        throw new Error("CFF: malformed rcurveline operands");
    }
    const line = stack.splice(stack.length - 2, 2);
    drawCurves(recorder, stack, 6, (values) => values);
    outlineLineTo(recorder, recorder.x + (line[0] ?? 0), recorder.y + (line[1] ?? 0));
}

function drawLineThenCurve(recorder: OutlineRecorder, stack: number[]): void {
    if (stack.length < 8 || (stack.length - 6) % 2 !== 0) {
        throw new Error("CFF: malformed rlinecurve operands");
    }
    const curve = stack.splice(stack.length - 6, 6);
    drawRelativeLines(recorder, stack, false, false);
    curveFrom(recorder, curve);
}

function drawEvenCurves(recorder: OutlineRecorder, stack: number[], horizontal: boolean): void {
    if (stack.length < 4) {
        throw new Error("CFF: malformed even-curve operands");
    }
    let index = 0;
    if (stack.length % 4 !== 0) {
        const first = stack[index++] ?? 0;
        if (horizontal) {
            curveFrom(recorder, [first, 0, stack[index++] ?? 0, stack[index++] ?? 0, stack[index++] ?? 0, 0]);
        } else {
            curveFrom(recorder, [0, first, stack[index++] ?? 0, stack[index++] ?? 0, 0, stack[index++] ?? 0]);
        }
    }
    while (index < stack.length) {
        if (horizontal) {
            curveFrom(recorder, [stack[index++] ?? 0, 0, stack[index++] ?? 0, stack[index++] ?? 0, stack[index++] ?? 0, 0]);
        } else {
            curveFrom(recorder, [0, stack[index++] ?? 0, stack[index++] ?? 0, stack[index++] ?? 0, 0, stack[index++] ?? 0]);
        }
    }
    stack.length = 0;
}

function drawAlternatingCurves(recorder: OutlineRecorder, stack: number[], startHorizontal: boolean): void {
    if (stack.length < 4 || stack.length % 4 !== 0) {
        throw new Error("CFF: malformed alternating-curve operands");
    }
    let horizontal = startHorizontal;
    for (let index = 0; index < stack.length; index += 4) {
        const a = stack[index] ?? 0;
        const b = stack[index + 1] ?? 0;
        const c = stack[index + 2] ?? 0;
        const d = stack[index + 3] ?? 0;
        if (horizontal) {
            curveFrom(recorder, [a, 0, b, c, 0, d]);
        } else {
            curveFrom(recorder, [0, a, b, c, d, 0]);
        }
        horizontal = !horizontal;
    }
    stack.length = 0;
}

function drawFlex(recorder: OutlineRecorder, stack: number[]): void {
    if (stack.length !== 13) {
        throw new Error("CFF: flex requires 13 operands");
    }
    curveFrom(recorder, stack.slice(0, 6));
    curveFrom(recorder, stack.slice(6, 12));
    stack.length = 0;
}

function drawHFlex(recorder: OutlineRecorder, stack: number[]): void {
    if (stack.length !== 7) {
        throw new Error("CFF: hflex requires 7 operands");
    }
    const [dx1, dx2, dy2, dx3, dx4, dx5, dx6] = stack;
    curveFrom(recorder, [dx1 ?? 0, 0, dx2 ?? 0, dy2 ?? 0, dx3 ?? 0, 0]);
    curveFrom(recorder, [dx4 ?? 0, 0, dx5 ?? 0, -(dy2 ?? 0), dx6 ?? 0, 0]);
    stack.length = 0;
}

function drawHFlex1(recorder: OutlineRecorder, stack: number[]): void {
    if (stack.length !== 9) {
        throw new Error("CFF: hflex1 requires 9 operands");
    }
    const [dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6] = stack;
    curveFrom(recorder, [dx1 ?? 0, dy1 ?? 0, dx2 ?? 0, dy2 ?? 0, dx3 ?? 0, 0]);
    curveFrom(recorder, [dx4 ?? 0, 0, dx5 ?? 0, dy5 ?? 0, dx6 ?? 0, -((dy1 ?? 0) + (dy2 ?? 0) + (dy5 ?? 0))]);
    stack.length = 0;
}

function drawFlex1(recorder: OutlineRecorder, stack: number[]): void {
    if (stack.length !== 11) {
        throw new Error("CFF: flex1 requires 11 operands");
    }
    const startX = recorder.x;
    const startY = recorder.y;
    curveFrom(recorder, stack.slice(0, 6));
    const dx = (stack[0] ?? 0) + (stack[2] ?? 0) + (stack[4] ?? 0) + (stack[6] ?? 0) + (stack[8] ?? 0);
    const dy = (stack[1] ?? 0) + (stack[3] ?? 0) + (stack[5] ?? 0) + (stack[7] ?? 0) + (stack[9] ?? 0);
    const last = stack[10] ?? 0;
    if (Math.abs(dx) > Math.abs(dy)) {
        curveFrom(recorder, [stack[6] ?? 0, stack[7] ?? 0, stack[8] ?? 0, stack[9] ?? 0, last, startY - (recorder.y + (stack[7] ?? 0) + (stack[9] ?? 0))]);
    } else {
        curveFrom(recorder, [stack[6] ?? 0, stack[7] ?? 0, stack[8] ?? 0, stack[9] ?? 0, startX - (recorder.x + (stack[6] ?? 0) + (stack[8] ?? 0)), last]);
    }
    stack.length = 0;
}

function curveFrom(recorder: OutlineRecorder, values: readonly number[]): void {
    outlineCubicTo(
        recorder,
        recorder.x + (values[0] ?? 0),
        recorder.y + (values[1] ?? 0),
        recorder.x + (values[0] ?? 0) + (values[2] ?? 0),
        recorder.y + (values[1] ?? 0) + (values[3] ?? 0),
        recorder.x + (values[0] ?? 0) + (values[2] ?? 0) + (values[4] ?? 0),
        recorder.y + (values[1] ?? 0) + (values[3] ?? 0) + (values[5] ?? 0)
    );
}

function applyArithmetic(operator: number, stack: number[]): boolean {
    const pop2 = (): [number, number] => [pop(stack, "CFF arithmetic"), pop(stack, "CFF arithmetic")];
    switch (operator) {
        case 0x0c03: {
            const [right, left] = pop2();
            stack.push(left && right ? 1 : 0);
            return true;
        }
        case 0x0c04: {
            const [right, left] = pop2();
            stack.push(left || right ? 1 : 0);
            return true;
        }
        case 0x0c05:
            stack.push(pop(stack, "CFF not") ? 0 : 1);
            return true;
        case 0x0c09:
            stack.push(Math.abs(pop(stack, "CFF abs")));
            return true;
        case 0x0c0a: {
            const [right, left] = pop2();
            stack.push(left + right);
            return true;
        }
        case 0x0c0b: {
            const [right, left] = pop2();
            stack.push(left - right);
            return true;
        }
        case 0x0c0c: {
            const [right, left] = pop2();
            if (right === 0) {
                throw new Error("CFF: division by zero");
            }
            stack.push(left / right);
            return true;
        }
        case 0x0c0e:
            stack.push(-pop(stack, "CFF neg"));
            return true;
        case 0x0c0f: {
            const [right, left] = pop2();
            stack.push(left === right ? 1 : 0);
            return true;
        }
        case 0x0c12:
            pop(stack, "CFF drop");
            return true;
        case 0x0c15: {
            const value = pop(stack, "CFF ifelse");
            const [right, left] = pop2();
            const test = pop(stack, "CFF ifelse");
            stack.push(test <= value ? left : right);
            return true;
        }
        case 0x0c17:
            throw new Error("CFF: random is unsupported");
        case 0x0c18: {
            const [right, left] = pop2();
            stack.push(left * right);
            return true;
        }
        case 0x0c1a: {
            const value = pop(stack, "CFF sqrt");
            if (value < 0) {
                throw new Error("CFF: square root of a negative number");
            }
            stack.push(Math.sqrt(value));
            return true;
        }
        case 0x0c1b: {
            const value = pop(stack, "CFF dup");
            stack.push(value, value);
            return true;
        }
        case 0x0c1c: {
            const right = pop(stack, "CFF exch");
            const left = pop(stack, "CFF exch");
            stack.push(right, left);
            return true;
        }
        case 0x0c1d: {
            const index = pop(stack, "CFF index");
            const value = stack[stack.length - 1 - index];
            if (value === undefined) {
                throw new Error("CFF: index is outside the stack");
            }
            stack.push(value);
            return true;
        }
        case 0x0c1e: {
            const count = pop(stack, "CFF roll");
            const shift = pop(stack, "CFF roll");
            if (count < 0 || count > stack.length) {
                throw new Error("CFF: roll count is outside the stack");
            }
            const group = stack.splice(stack.length - count, count);
            const amount = ((shift % count) + count) % count;
            stack.push(...group.slice(count - amount), ...group.slice(0, count - amount));
            return true;
        }
        default:
            return false;
    }
}

function scaleRecorder(recorder: OutlineRecorder, glyphId: number, unitsPerEm: number, matrix: readonly number[], bounds: GlyphBounds | null): GlyphCurves | null {
    const curves = recorder.curves.map((curve) => ({
        p0x: scaleX(curve.p0x, curve.p0y, matrix, unitsPerEm),
        p0y: scaleY(curve.p0x, curve.p0y, matrix, unitsPerEm),
        p1x: scaleX(curve.p1x, curve.p1y, matrix, unitsPerEm),
        p1y: scaleY(curve.p1x, curve.p1y, matrix, unitsPerEm),
        p2x: scaleX(curve.p2x, curve.p2y, matrix, unitsPerEm),
        p2y: scaleY(curve.p2x, curve.p2y, matrix, unitsPerEm),
    }));
    if (curves.length === 0 || !bounds) {
        return null;
    }
    return { glyphId, curves, bounds };
}

function boundsOf(recorder: OutlineRecorder, unitsPerEm: number, matrix: readonly number[]): GlyphBounds | null {
    if (recorder.curves.length === 0) {
        return null;
    }
    let xMin = Infinity;
    let yMin = Infinity;
    let xMax = -Infinity;
    let yMax = -Infinity;
    for (const curve of recorder.curves) {
        for (const [x, y] of [
            [curve.p0x, curve.p0y],
            [curve.p1x, curve.p1y],
            [curve.p2x, curve.p2y],
        ] as const) {
            const sx = scaleX(x, y, matrix, unitsPerEm);
            const sy = scaleY(x, y, matrix, unitsPerEm);
            xMin = Math.min(xMin, sx);
            yMin = Math.min(yMin, sy);
            xMax = Math.max(xMax, sx);
            yMax = Math.max(yMax, sy);
        }
    }
    return { xMin, yMin, xMax, yMax };
}

function scaleX(x: number, y: number, matrix: readonly number[], unitsPerEm: number): number {
    return ((matrix[0] ?? 0) * x + (matrix[2] ?? 0) * y + (matrix[4] ?? 0)) * unitsPerEm;
}

function scaleY(x: number, y: number, matrix: readonly number[], unitsPerEm: number): number {
    return ((matrix[1] ?? 0) * x + (matrix[3] ?? 0) * y + (matrix[5] ?? 0)) * unitsPerEm;
}

function pop(stack: number[], label: string): number {
    const value = stack.pop();
    if (value === undefined || !Number.isFinite(value)) {
        throw new Error(`${label}: stack underflow`);
    }
    return value;
}

function bias(count: number): number {
    if (count < 1240) {
        return 107;
    }
    if (count < 33900) {
        return 1131;
    }
    return 32768;
}

function readOperand(bytes: Uint8Array, offset: number): { value: number; next: number } {
    const b0 = bytes[offset];
    if (b0 === undefined) {
        throw new Error("CFF: truncated operand");
    }
    if (b0 >= 32 && b0 <= 246) {
        return { value: b0 - 139, next: offset + 1 };
    }
    const b1 = bytes[offset + 1];
    if (b0 >= 247 && b0 <= 250) {
        if (b1 === undefined) {
            throw new Error("CFF: truncated operand");
        }
        return { value: (b0 - 247) * 256 + b1 + 108, next: offset + 2 };
    }
    if (b0 >= 251 && b0 <= 254) {
        if (b1 === undefined) {
            throw new Error("CFF: truncated operand");
        }
        return { value: -(b0 - 251) * 256 - b1 - 108, next: offset + 2 };
    }
    if (b0 === 28) {
        if (offset + 3 > bytes.length) {
            throw new Error("CFF: truncated short operand");
        }
        const value = ((bytes[offset + 1] ?? 0) << 8) | (bytes[offset + 2] ?? 0);
        return { value: value >= 0x8000 ? value - 0x10000 : value, next: offset + 3 };
    }
    if (b0 === 255) {
        if (offset + 5 > bytes.length) {
            throw new Error("CFF: truncated fixed operand");
        }
        const raw = ((bytes[offset + 1] ?? 0) << 24) | ((bytes[offset + 2] ?? 0) << 16) | ((bytes[offset + 3] ?? 0) << 8) | (bytes[offset + 4] ?? 0);
        const signed = raw >= 0x80000000 ? raw - 0x100000000 : raw;
        return { value: signed / 65536, next: offset + 5 };
    }
    throw new Error(`CFF: invalid operand byte ${b0}`);
}

interface CffDict {
    readonly operators: Map<number, number[]>;
}

function parseDict(bytes: Uint8Array): CffDict {
    const operators = new Map<number, number[]>();
    const stack: number[] = [];
    let cursor = 0;
    while (cursor < bytes.length) {
        const byte = bytes[cursor] ?? 0;
        if (byte === 28 || byte === 255 || byte >= 32) {
            const operand = readOperand(bytes, cursor);
            stack.push(operand.value);
            cursor = operand.next;
            continue;
        }
        cursor++;
        const operator = byte === 12 ? 0x0c00 + (bytes[cursor++] ?? 0) : byte;
        if (operator === 0x0c07 || operator === 30 || operator === 0x0c1e) {
            operators.set(operator, stack.splice(0));
            continue;
        }
        operators.set(operator, stack.splice(0));
    }
    if (stack.length !== 0) {
        throw new Error("CFF: DICT ended with an unconsumed operand");
    }
    return { operators };
}

function dictNumber(dict: CffDict, operator: number): number | null {
    const values = dict.operators.get(operator);
    if (!values || values.length === 0) {
        return null;
    }
    return values[0] ?? null;
}

function requiredOffset(dict: CffDict, operator: number, label: string): number {
    const values = dict.operators.get(operator);
    const offset = values?.[0];
    if (!Number.isFinite(offset) || !offset || offset < 0) {
        throw new Error(`${label}: missing offset`);
    }
    return offset;
}

function readIndex(bytes: Uint8Array, offset: number, limit: number, label: string): CffIndex & { end: number } {
    if (offset + 2 > limit) {
        throw new Error(`${label}: truncated count`);
    }
    const count = ((bytes[offset] ?? 0) << 8) | (bytes[offset + 1] ?? 0);
    if (count === 0) {
        return { count: 0, data: bytes.subarray(offset + 2, offset + 2), offsets: [1], end: offset + 2 };
    }
    const offSize = bytes[offset + 2] ?? 0;
    if (offSize < 1 || offSize > 4) {
        throw new Error(`${label}: invalid offSize ${offSize}`);
    }
    const offsetBase = offset + 3;
    const offsetBytes = (count + 1) * offSize;
    if (offsetBase + offsetBytes > limit) {
        throw new Error(`${label}: truncated offsets`);
    }
    const offsets: number[] = [];
    for (let index = 0; index <= count; index++) {
        let value = 0;
        const at = offsetBase + index * offSize;
        for (let byte = 0; byte < offSize; byte++) {
            value = value * 256 + (bytes[at + byte] ?? 0);
        }
        if (index > 0 && value < (offsets[index - 1] ?? 0)) {
            throw new Error(`${label}: offsets are not monotonic`);
        }
        offsets.push(value);
    }
    const dataStart = offsetBase + offsetBytes;
    const dataEnd = dataStart + (offsets[count] ?? 1) - 1;
    if (dataEnd > limit) {
        throw new Error(`${label}: data is outside the table`);
    }
    return { count, data: bytes.subarray(dataStart, dataEnd), offsets, end: dataEnd };
}

function indexBytes(index: CffIndex, item: number, label: string): Uint8Array {
    if (!Number.isInteger(item) || item < 0 || item >= index.count) {
        throw new Error(`${label}: index ${item} is outside 0..${index.count - 1}`);
    }
    const start = (index.offsets[item] ?? 1) - 1;
    const end = (index.offsets[item + 1] ?? 1) - 1;
    if (start < 0 || end < start || end > index.data.length) {
        throw new Error(`${label}: index ${item} offset is outside the data`);
    }
    return index.data.subarray(start, end);
}

function validateCharset(bytes: Uint8Array, offset: number, limit: number, glyphCount: number): void {
    const format = bytes[offset];
    if (format === 0) {
        const needed = 1 + (glyphCount - 1) * 2;
        if (offset + needed > limit) {
            throw new Error("CFF: charset format 0 is truncated");
        }
        return;
    }
    if (format !== 1 && format !== 2) {
        throw new Error(`CFF: unsupported charset format ${format}`);
    }
    let seen = 1;
    let cursor = offset + 1;
    const rangeBytes = format === 1 ? 3 : 4;
    while (seen < glyphCount) {
        if (cursor + rangeBytes > limit) {
            throw new Error("CFF: charset range is truncated");
        }
        const nLeft = format === 1 ? (bytes[cursor + 2] ?? 0) : ((bytes[cursor + 2] ?? 0) << 8) | (bytes[cursor + 3] ?? 0);
        seen += nLeft + 1;
        cursor += rangeBytes;
    }
    if (seen !== glyphCount) {
        throw new Error(`CFF: charset covers ${seen} glyphs, expected ${glyphCount}`);
    }
}

function emptyIndex(): CffIndex {
    return { count: 0, data: new Uint8Array(0), offsets: [1] };
}

function checked(base: number, offset: number, limit: number, label: string): number {
    if (!Number.isFinite(offset) || offset < 0 || base + offset >= limit) {
        throw new Error(`${label}: offset ${offset} is outside the table`);
    }
    return base + offset;
}
