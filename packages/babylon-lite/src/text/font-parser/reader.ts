/** Bounded big-endian cursor. Offsets are absolute indexes into `bytes`. */

export interface ByteCursor {
    readonly bytes: Uint8Array;
    readonly limit: number;
    offset: number;
}

export function openCursor(bytes: Uint8Array, start: number, length: number, label: string): ByteCursor {
    if (!Number.isInteger(start) || !Number.isInteger(length) || start < 0 || length < 0 || start > bytes.length || length > bytes.length - start) {
        throw new Error(`${label}: extent ${start}+${length} exceeds font (${bytes.length} bytes)`);
    }
    return { bytes, offset: start, limit: start + length };
}

export function remaining(cursor: ByteCursor): number {
    return cursor.limit - cursor.offset;
}

export function skip(cursor: ByteCursor, count: number, label: string): void {
    if (!Number.isInteger(count) || count < 0 || count > remaining(cursor)) {
        throw new Error(`${label}: cannot skip ${count} bytes at ${cursor.offset}`);
    }
    cursor.offset += count;
}

function requireBytes(cursor: ByteCursor, count: number, label: string): number {
    const offset = cursor.offset;
    if (!Number.isInteger(count) || count < 0 || count > cursor.limit - offset) {
        throw new Error(`${label}: truncated read of ${count} bytes at ${offset}`);
    }
    cursor.offset = offset + count;
    return offset;
}

export function readU8(cursor: ByteCursor, label: string): number {
    const offset = requireBytes(cursor, 1, label);
    return cursor.bytes[offset] ?? 0;
}

export function readI8(cursor: ByteCursor, label: string): number {
    const value = readU8(cursor, label);
    return value >= 0x80 ? value - 0x100 : value;
}

/** Signed 2.14 fixed point used by composite glyf transforms. */
export function readF2Dot14(cursor: ByteCursor, label: string): number {
    return readI16(cursor, label) / 16384;
}

export function readU16(cursor: ByteCursor, label: string): number {
    const offset = requireBytes(cursor, 2, label);
    const bytes = cursor.bytes;
    return ((bytes[offset] ?? 0) << 8) | (bytes[offset + 1] ?? 0);
}

export function readI16(cursor: ByteCursor, label: string): number {
    const value = readU16(cursor, label);
    return value >= 0x8000 ? value - 0x10000 : value;
}

export function readU32(cursor: ByteCursor, label: string): number {
    const offset = requireBytes(cursor, 4, label);
    const bytes = cursor.bytes;
    return ((bytes[offset] ?? 0) * 0x1000000 + (bytes[offset + 1] ?? 0) * 0x10000 + (bytes[offset + 2] ?? 0) * 0x100 + (bytes[offset + 3] ?? 0)) >>> 0;
}

export function readTag(cursor: ByteCursor, label: string): string {
    const offset = requireBytes(cursor, 4, label);
    const bytes = cursor.bytes;
    return String.fromCharCode(bytes[offset] ?? 0, bytes[offset + 1] ?? 0, bytes[offset + 2] ?? 0, bytes[offset + 3] ?? 0);
}

/** Reject a font-supplied count before it is used as an allocation length. */
export function requireCount(count: number, recordBytes: number, availableBytes: number, label: string): void {
    if (!Number.isInteger(count) || count < 0) {
        throw new Error(`${label}: invalid count ${count}`);
    }
    if (!Number.isInteger(recordBytes) || recordBytes < 0) {
        throw new Error(`${label}: invalid record size`);
    }
    if (recordBytes !== 0 && count > Math.floor(availableBytes / recordBytes)) {
        throw new Error(`${label}: count ${count} exceeds ${availableBytes} bytes`);
    }
}

export function checkedOffset(base: number, offset: number, label: string): number {
    if (!Number.isInteger(base) || !Number.isInteger(offset) || base < 0 || offset < 0) {
        throw new Error(`${label}: invalid offset ${base}+${offset}`);
    }
    const sum = base + offset;
    if (!Number.isSafeInteger(sum)) {
        throw new Error(`${label}: offset overflow`);
    }
    return sum;
}
