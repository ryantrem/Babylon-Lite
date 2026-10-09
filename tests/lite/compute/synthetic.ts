import { createComputeFontFromBuffer } from "../../../packages/babylon-lite/src/index";
import { Font } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import type { Fixture } from "./fixtures";

function words(...values: number[]): Uint8Array {
    const bytes = new Uint8Array(values.length * 2);
    const view = new DataView(bytes.buffer);
    values.forEach((value, at) => view.setUint16(at * 2, value));
    return bytes;
}
function concat(...chunks: Uint8Array[]): Uint8Array {
    const bytes = new Uint8Array(chunks.reduce((size, chunk) => size + chunk.length, 0));
    let cursor = 0;
    chunks.forEach((chunk) => {
        bytes.set(chunk, cursor);
        cursor += chunk.length;
    });
    return bytes;
}
function layout(type: number, subtable: Uint8Array, featureTag = "ccmp", extra?: Uint8Array): Uint8Array {
    return layoutFromLookups([{ type, subtable }, ...(extra ? [{ type, subtable: extra }] : [])], extra ? [0, 1] : [0], featureTag);
}

function layoutFromLookups(
    lookups: readonly { type: number; subtable: Uint8Array; extraSubtable?: Uint8Array; flag?: number }[],
    roots: readonly number[],
    featureTag = "ccmp"
): Uint8Array {
    const script = concat(words(1), new TextEncoder().encode("latn"), words(8, 4, 0, 0, 0xffff, 1, 0));
    const feature = concat(words(1), new TextEncoder().encode(featureTag), words(8, 0, roots.length, ...roots));
    const bodies = lookups.map(({ type, subtable, extraSubtable, flag = 0 }) =>
        extraSubtable ? concat(words(type, flag, 2, 10, 10 + subtable.length), subtable, extraSubtable) : concat(words(type, flag, 1, 8), subtable)
    );
    let offset = 2 + bodies.length * 2;
    const offsets = bodies.map((body) => {
        const at = offset;
        offset += body.length;
        return at;
    });
    const lookup = concat(words(bodies.length, ...offsets), ...bodies);
    return concat(words(1, 0, 10, 10 + script.length, 10 + script.length + feature.length), script, feature, lookup);
}

/** Valid small sfnt fixtures, independently decoded by text-shaper. No font internals are mutated. */
export function syntheticFont(
    kind:
        | "multiple"
        | "alternate"
        | "mark-ligature"
        | "mark-mark"
        | "point-anchor"
        | "adjust"
        | "transitive"
        | "context"
        | "backtrack"
        | "backtrack-reject"
        | "ligature-ignore-marks"
        | "pair-first"
        | "pair-second"
        | "pair-reject"
): Fixture {
    const head = new Uint8Array(54);
    new DataView(head.buffer).setUint32(0, 0x10000);
    new DataView(head.buffer).setUint32(12, 0x5f0f3cf5);
    new DataView(head.buffer).setUint16(18, 1000);
    const hhea = new Uint8Array(36);
    new DataView(hhea.buffer).setUint32(0, 0x10000);
    new DataView(hhea.buffer).setUint16(34, 4);
    const maxp = new Uint8Array(32);
    new DataView(maxp.buffer).setUint32(0, 0x10000);
    new DataView(maxp.buffer).setUint16(4, 4);
    const cmap = concat(words(0, 1, 3, 1), Uint8Array.of(0, 0, 0, 12), words(4, 32, 0, 4, 4, 1, 0, 0x43, 0xffff, 0, 0x41, 0xffff, -64, 1, 0, 0));
    const tables: Record<string, Uint8Array> = {
        head,
        hhea,
        cmap,
        maxp,
        hmtx: words(0, 0, 100, 0, 200, 0, 300, 0),
        loca: new Uint8Array(10),
        glyf: new Uint8Array(1),
    };
    if (kind === "multiple" || kind === "alternate") {
        tables.GSUB = layout(kind === "multiple" ? 2 : 3, words(1, 8, 1, 14, 1, 1, 1, 2, 2, 3));
    } else if (kind === "backtrack" || kind === "backtrack-reject") {
        tables.GSUB = layoutFromLookups(
            [
                { type: 6, subtable: words(3, 1, 20, 2, 26, 32, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1) },
                kind === "backtrack" ? { type: 1, subtable: words(1, 6, 1, 1, 1, 1) } : { type: 3, subtable: words(1, 8, 1, 14, 1, 1, 1, 2, 2, 3) },
            ],
            [0]
        );
    } else if (kind === "ligature-ignore-marks") {
        tables.GSUB = layoutFromLookups([{ type: 4, flag: 8, subtable: words(1, 8, 1, 14, 1, 1, 1, 1, 4, 3, 2, 3) }], [0]);
        tables.GDEF = concat(words(1, 0, 12, 0, 0, 0), words(1, 1, 3, 1, 3, 1));
    } else if (kind === "context") {
        tables.GSUB = layoutFromLookups(
            [
                { type: 6, subtable: words(3, 0, 2, 22, 28, 0, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 2) },
                { type: 1, subtable: words(1, 6, 1, 1, 1, 1) },
                { type: 1, subtable: words(1, 6, 1, 1, 1, 2) },
            ],
            [0]
        );
    } else if (kind === "transitive") {
        tables.GSUB = layout(1, words(1, 6, 1, 1, 1, 1), "ccmp", words(1, 6, 1, 1, 1, 2));
    } else if (kind === "adjust") {
        tables.GPOS = layout(1, words(1, 8, 4, 13, 1, 1, 1), "kern");
    } else if (kind === "mark-mark") {
        tables.GPOS = layout(
            6,
            concat(words(1, 12, 20, 1, 28, 50), words(1, 2, 2, 3), words(1, 2, 2, 3), words(2, 0, 10, 0, 16, 1, 5, 7, 1, 15, 17), words(2, 6, 12, 1, 25, 47, 1, 45, 67)),
            "mkmk"
        );
        tables.GDEF = concat(words(1, 0, 12, 0, 0, 0), words(1, 1, 3, 1, 3, 3));
    } else if (kind === "pair-reject") {
        tables.GPOS = layoutFromLookups(
            [
                { type: 2, subtable: words(1, 12, 0x40, 0, 1, 18, 1, 1, 2, 1, 3, 24, 0, 0, 0x8000) },
                { type: 2, subtable: words(1, 12, 0x40, 0, 1, 18, 1, 1, 1, 1, 2, 24, 0, 0, 0x8000) },
            ],
            [0, 1],
            "kern"
        );
        // Both pairs reference a one-region VariationIndex with a nonzero default delta.
        tables.GDEF = concat(words(1, 3, 0, 0, 0, 0, 0, 0, 18), words(1, 0, 12, 1, 0, 16, 0, 1, 1, 1, 1, 0, 7));
    } else if (kind === "pair-first" || kind === "pair-second") {
        const second = kind === "pair-second";
        const subtable = concat(words(1, 14, 4, second ? 4 : 0, 2, 22, second ? 30 : 28, 1, 2, 1, 2), second ? words(1, 2, 13, 17, 1, 3, 19, 23) : words(1, 2, 13, 1, 3, 19));
        tables.GPOS = layoutFromLookups([{ type: 2, subtable, extraSubtable: second ? subtable : undefined }], [0], "kern");
    } else {
        const markCoverage = words(1, 1, 2);
        const ligatureCoverage = words(1, 1, 1);
        const markArray = words(1, 0, 6, 1, 5, 7);
        const ligatureArray = kind === "point-anchor" ? words(1, 4, 2, 6, 14, 2, 25, 47, 0, 1, 45, 67) : words(1, 4, 2, 6, 12, 1, 25, 47, 1, 45, 67);
        tables.GPOS = layout(5, concat(words(1, 12, 18, 1, 24, 36), markCoverage, ligatureCoverage, markArray, ligatureArray), "mark");
        tables.GDEF = concat(words(1, 0, 12, 0, 0, 0), words(1, 1, 3, 2, 3, 1));
        if (kind === "point-anchor") {
            tables.GSUB = layout(1, words(1, 6, 0, 1, 1, 1));
        }
    }
    const tags = Object.keys(tables);
    const size = 12 + tags.length * 16 + Object.values(tables).reduce((total, table) => total + Math.ceil(table.length / 4) * 4, 0);
    const bytes = new Uint8Array(size);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x10000);
    view.setUint16(4, tags.length);
    let cursor = 12 + tags.length * 16;
    tags.forEach((tag, index) => {
        const at = 12 + index * 16;
        const table = tables[tag]!;
        bytes.set(new TextEncoder().encode(tag), at);
        view.setUint32(at + 8, cursor);
        view.setUint32(at + 12, table.length);
        bytes.set(table, cursor);
        cursor += Math.ceil(table.length / 4) * 4;
    });
    return { file: "synthetic", bytes: bytes.buffer, font: createComputeFontFromBuffer(bytes.buffer), oracle: Font.load(bytes.buffer) };
}
