/** Test-only CLI counterpart to the ctypes restricted nominal callback.
 * Rebuilds a temporary sfnt with exactly input scalars plus space in cmap.
 * All non-cmap table bytes stay identical, except head.checkSumAdjustment.
 * This never changes a fixture or a product font.
 */
import assert from "node:assert/strict";

function checksum(bytes) {
    const padded = Buffer.alloc(Math.ceil(bytes.length / 4) * 4);
    bytes.copy(padded);
    let sum = 0;
    for (let i = 0; i < padded.length; i += 4) {
        sum = (sum + padded.readUInt32BE(i)) >>> 0;
    }
    return sum;
}

export function restrictedCmapFont(original, font, text) {
    const scalars = [...new Set([0x20, ...[...text].map((cp) => cp.codePointAt(0))])].sort((a, b) => a - b);
    const groups = scalars.filter((scalar) => font.glyphId(scalar) !== 0);
    const cmap = Buffer.alloc(28 + groups.length * 12);
    cmap.writeUInt16BE(1, 2);
    cmap.writeUInt16BE(3, 4);
    cmap.writeUInt16BE(10, 6);
    cmap.writeUInt32BE(12, 8);
    cmap.writeUInt16BE(12, 12);
    cmap.writeUInt32BE(cmap.length - 12, 16);
    cmap.writeUInt32BE(groups.length, 24);
    groups.forEach((scalar, i) => {
        cmap.writeUInt32BE(scalar, 28 + i * 12);
        cmap.writeUInt32BE(scalar, 32 + i * 12);
        cmap.writeUInt32BE(font.glyphId(scalar), 36 + i * 12);
    });
    const count = original.readUInt16BE(4);
    const tables = Array.from({ length: count }, (_, i) => {
        const p = 12 + i * 16;
        const tag = original.toString("ascii", p, p + 4);
        const offset = original.readUInt32BE(p + 8);
        const bytes = Buffer.from(original.subarray(offset, offset + original.readUInt32BE(p + 12)));
        if (tag === "head") {
            bytes.writeUInt32BE(0, 8);
        }
        return { tag, bytes: tag === "cmap" ? cmap : bytes };
    });
    const result = Buffer.alloc(12 + count * 16 + tables.reduce((sum, entry) => sum + Math.ceil(entry.bytes.length / 4) * 4, 0));
    original.copy(result, 0, 0, 12);
    let cursor = 12 + count * 16;
    let head;
    for (const [i, { tag, bytes }] of tables.entries()) {
        const p = 12 + i * 16;
        result.write(tag, p, 4, "ascii");
        result.writeUInt32BE(checksum(bytes), p + 4);
        result.writeUInt32BE(cursor, p + 8);
        result.writeUInt32BE(bytes.length, p + 12);
        bytes.copy(result, cursor);
        if (tag === "head") {
            head = cursor;
        }
        // All positioning, substitution, outlines, widths and class data are exact.
        if (!["cmap", "head"].includes(tag)) {
            const oldOffset = original.readUInt32BE(p + 8);
            assert.deepEqual(result.subarray(cursor, cursor + bytes.length), original.subarray(oldOffset, oldOffset + bytes.length));
        }
        cursor += Math.ceil(bytes.length / 4) * 4;
    }
    assert.notEqual(head, undefined);
    result.writeUInt32BE((0xb1b0afba - checksum(result)) >>> 0, head + 8);
    assert.equal(checksum(result), 0xb1b0afba);
    return result;
}
