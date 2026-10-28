import { createComputeFontFromBuffer } from "../../../packages/babylon-lite/src/text/compute-font";
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

export type SyntheticKind =
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
    | "refresh-roots"
    | "refresh-ligature"
    | "arabic-no-joining"
    | "segment-reject"
    | "contour-anchor"
    | CorrectionKind;

export type CorrectionKind =
    | "c1-mark-base"
    | "c1-mark-mark-chain"
    | "c2-filter-set"
    | "c3-pair-placement"
    | "c3-pair-value2"
    | "c4-nested-ligature"
    | "c4-nested-multiple"
    | "c5-context-format3"
    | "c6-context-pos"
    | "c7-ligature-across-mark"
    | "c8-mark-ligature"
    | "c8-mark-mark-components"
    | "reverse-chain"
    | "cursive"
    | "nested-context";

export const CORRECTION_KINDS: readonly CorrectionKind[] = [
    "c1-mark-base",
    "c1-mark-mark-chain",
    "c2-filter-set",
    "c3-pair-placement",
    "c3-pair-value2",
    "c4-nested-ligature",
    "c4-nested-multiple",
    "c5-context-format3",
    "c6-context-pos",
    "c7-ligature-across-mark",
    "c8-mark-ligature",
    "c8-mark-mark-components",
    "reverse-chain",
    "cursive",
    "nested-context",
];

/** Frozen pre-M4 kinds, retained byte-identically for the existing-input inventory. */
export const LEGACY_SYNTHETIC_KINDS: readonly SyntheticKind[] = [
    "multiple",
    "alternate",
    "mark-ligature",
    "mark-mark",
    "point-anchor",
    "adjust",
    "transitive",
    "context",
    "backtrack",
    "backtrack-reject",
    "ligature-ignore-marks",
    "pair-first",
    "pair-second",
    "pair-reject",
    "refresh-roots",
    "refresh-ligature",
];

/** Valid small sfnt fixtures, independently decoded by text-shaper. No font internals are mutated. */
export function syntheticFont(kind: SyntheticKind): Fixture {
    const bytes = syntheticFontBytes(kind);
    return { file: "synthetic", bytes, font: createComputeFontFromBuffer(bytes), oracle: Font.load(bytes) };
}

/** Raw fixture bytes for independent reference tools, without constructing a Lite font. */
export function syntheticFontBytes(kind: SyntheticKind): ArrayBuffer {
    if (CORRECTION_KINDS.includes(kind as CorrectionKind)) {
        return correctionFontBytes(kind as CorrectionKind);
    }
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
    if (kind === "arabic-no-joining" || kind === "segment-reject") {
        if (kind === "segment-reject") {
            // B has a real on-curve triangle. The rejecting-segment test must
            // suppress previously visible instances, not an empty-outline font.
            tables.glyf = concat(words(1, 0, 0, 100, 100, 2, 0), Uint8Array.of(1, 1, 1), words(0, 100, -100, 0, 0, 100), Uint8Array.of(0));
            tables.loca = words(0, 0, 0, 15, 15);
            new DataView(head.buffer).setInt16(40, 100);
            new DataView(head.buffer).setInt16(42, 100);
            new DataView(maxp.buffer).setUint16(6, 3);
            new DataView(maxp.buffer).setUint16(8, 1);
        }
        const mappings =
            kind === "arabic-no-joining"
                ? [
                      [0x20, 3],
                      [0x628, 1],
                      [0x62a, 2],
                  ]
                : [
                      [0x20, 3],
                      [0x41, 1],
                      [0x42, 2],
                  ];
        const mapping = new Uint8Array(28 + mappings.length * 12);
        const view = new DataView(mapping.buffer);
        view.setUint16(2, 1);
        view.setUint16(4, 3);
        view.setUint16(6, 10);
        view.setUint32(8, 12);
        view.setUint16(12, 12);
        view.setUint32(16, mapping.length - 12);
        view.setUint32(24, mappings.length);
        mappings.forEach(([scalar, glyph], i) => {
            view.setUint32(28 + i * 12, scalar!);
            view.setUint32(32 + i * 12, scalar!);
            view.setUint32(36 + i * 12, glyph!);
        });
        tables.cmap = mapping;
        tables.GDEF = concat(words(1, 0, 12, 0, 0, 0), words(1, 1, 3, 1, 1, 0));
        if (kind === "segment-reject") {
            tables.GSUB = layout(3, words(1, 8, 1, 14, 1, 1, 1, 2, 2, 3));
        }
        return sfnt(tables);
    }
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
    } else if (kind === "refresh-roots") {
        tables.GSUB = layoutFromLookups(
            [
                { type: 1, subtable: words(1, 6, 1, 1, 1, 1) },
                { type: 1, subtable: words(1, 6, 1, 1, 1, 2) },
                { type: 1, subtable: words(1, 6, 2, 1, 1, 1) },
            ],
            [0, 1, 2]
        );
    } else if (kind === "refresh-ligature") {
        tables.GSUB = layout(4, words(1, 8, 1, 14, 1, 1, 1, 1, 4, 3, 3, 2, 3));
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
        const ligatureArray = kind === "point-anchor" || kind === "contour-anchor" ? words(1, 4, 2, 6, 14, 2, 25, 47, 0, 1, 45, 67) : words(1, 4, 2, 6, 12, 1, 25, 47, 1, 45, 67);
        tables.GPOS = layout(5, concat(words(1, 12, 18, 1, 24, 36), markCoverage, ligatureCoverage, markArray, ligatureArray), "mark");
        tables.GDEF = concat(words(1, 0, 12, 0, 0, 0), words(1, 1, 3, 2, 3, 1));
        if (kind === "point-anchor") {
            tables.GSUB = layout(1, words(1, 6, 0, 1, 1, 1));
        } else if (kind === "contour-anchor") {
            // A + C ligates back to A, skipping B. B is tagged component 1,
            // which genuinely reaches the contour anchor rather than fallback component 2.
            tables.GSUB = layoutFromLookups([{ type: 4, flag: 8, subtable: words(1, 8, 1, 14, 1, 1, 1, 1, 4, 1, 2, 3) }], [0]);
        }
    }
    return sfnt(tables);
}

function sfnt(tables: Record<string, Uint8Array>): ArrayBuffer {
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
    return bytes.buffer;
}

function coverage(...glyphs: number[]): Uint8Array {
    return words(1, glyphs.length, ...glyphs);
}

function single(from: number, to: number): Uint8Array {
    return concat(words(2, 8, 1, to), coverage(from));
}

function ligature(first: number, second: number): Uint8Array {
    return concat(words(1, 8, 1, 14), coverage(first), words(1, 4, 6, 2, second));
}

function context3(chaining: boolean, inputs: number[], records: number[]): Uint8Array {
    const headerWords = (chaining ? 5 : 3) + inputs.length + records.length;
    const offsets = inputs.map((_, index) => (headerWords + index * 3) * 2);
    return concat(
        chaining ? words(3, 0, inputs.length, ...offsets, 0, records.length / 2, ...records) : words(3, inputs.length, records.length / 2, ...offsets, ...records),
        ...inputs.map((glyph) => coverage(glyph))
    );
}

function markAttachment(type: 4 | 5 | 6, markGlyphs?: number[]): Uint8Array {
    const marks = markGlyphs ?? (type === 6 ? [5] : type === 4 ? [4] : [4, 5]);
    const targets = type === 6 ? [4] : type === 4 ? [1, 2] : [6];
    const markCoverage = coverage(...marks);
    const targetCoverage = coverage(...targets);
    const markArray = concat(
        words(marks.length, ...marks.flatMap((_, index) => [0, 2 + marks.length * 4 + index * 6])),
        ...marks.map((_, index) => words(1, 5 + index * 10, 7 + index * 10))
    );
    const targetArray =
        type === 5
            ? words(1, 4, 2, 6, 12, 1, 25, 47, 1, 125, 147)
            : concat(
                  words(targets.length, ...targets.map((_, index) => 2 + targets.length * 2 + index * 6)),
                  ...targets.map((_, index) => words(1, 25 + index * 20, 47 + index * 20))
              );
    return concat(
        words(1, 12, 12 + markCoverage.length, 1, 12 + markCoverage.length + targetCoverage.length, 12 + markCoverage.length + targetCoverage.length + markArray.length),
        markCoverage,
        targetCoverage,
        markArray,
        targetArray
    );
}

function pairFormat1(first: number[], second: number[], format1: number, format2: number): Uint8Array {
    const cov = coverage(...first);
    const sets = first.map((_, index) => concat(words(1, second[index]!), words(...(format1 === 7 ? [11, 17, 13] : [13 + index * 6])), format2 ? words(19) : words()));
    let at = 10 + first.length * 2 + cov.length;
    const offsets = sets.map((set) => {
        const result = at;
        at += set.length;
        return result;
    });
    return concat(words(1, 10 + first.length * 2, format1, format2, first.length, ...offsets), cov, ...sets);
}

/** New witnesses use A/B/C and two PUA marks, so Unicode normalization is inert.
 * IDs: A=1 B=2 C=3 mark1=4 mark2=5 ligature=6, substitutions=7/8/9. */
function correctionFontBytes(kind: CorrectionKind): ArrayBuffer {
    const head = new Uint8Array(54);
    const hv = new DataView(head.buffer);
    hv.setUint32(0, 0x10000);
    hv.setUint32(12, 0x5f0f3cf5);
    hv.setUint16(18, 1000);
    const hhea = new Uint8Array(36);
    new DataView(hhea.buffer).setUint32(0, 0x10000);
    new DataView(hhea.buffer).setUint16(34, 10);
    const maxp = new Uint8Array(32);
    new DataView(maxp.buffer).setUint32(0, 0x10000);
    new DataView(maxp.buffer).setUint16(4, 10);
    const cmap = concat(words(0, 1, 3, 1), Uint8Array.of(0, 0, 0, 12), words(4, 40, 0, 6, 4, 1, 2, 0x43, 0xe001, 0xffff, 0, 0x41, 0xe000, 0xffff, -64, 4 - 0xe000, 1, 0, 0, 0));
    const classes = words(1, 1, 9, 1, 1, 1, 3, 3, 2, 1, 1, 1);
    const tables: Record<string, Uint8Array> = {
        head,
        hhea,
        cmap,
        maxp,
        hmtx: words(0, 0, 100, 0, 200, 0, 300, 0, 0, 0, 0, 0, 400, 0, 110, 0, 120, 0, 130, 0),
        loca: new Uint8Array(22),
        glyf: new Uint8Array(1),
        GDEF: concat(words(1, 0, 12, 0, 0, 0), classes),
    };
    const lookup = (type: number, subtable: Uint8Array, flag = 0) => ({ type, subtable, flag });
    if (kind === "c1-mark-base") {
        tables.GPOS = layoutFromLookups([lookup(4, markAttachment(4))], [0], "mark");
    } else if (kind === "c1-mark-mark-chain") {
        tables.GPOS = layoutFromLookups([lookup(4, markAttachment(4)), lookup(6, markAttachment(6)), lookup(2, pairFormat1([1], [2], 4, 0), 8)], [0, 1, 2], "kern");
    } else if (kind === "c2-filter-set") {
        // All marks have zero advances, and there is no MarkBase attachment:
        // neither pen-relative C1 arithmetic nor PairPos C3 can confound C2.
        const subtable = markAttachment(6, [4]);
        const body = concat(words(6, 0x10, 1, 10, 0), subtable);
        const skeleton = layoutFromLookups([lookup(6, words())], [0], "mkmk");
        const offset = new DataView(skeleton.buffer).getUint16(8);
        tables.GPOS = concat(skeleton.slice(0, offset), words(1, 4), body);
        tables.GDEF = concat(words(1, 2, 14, 0, 0, 0, 14 + classes.length), classes, words(1, 1, 0, 8), coverage(4));
    } else if (kind === "c3-pair-placement") {
        tables.GPOS = layoutFromLookups([lookup(2, pairFormat1([1, 2], [2, 3], 7, 0))], [0], "kern");
    } else if (kind === "c3-pair-value2") {
        // Class 1 contains A/B; class 2 contains B/C. Only the non-default cell
        // is nonzero. The orthogonal advance is retained in the source for the
        // review gate; HarfBuzz applies advance values only in writing direction.
        const cov = coverage(1, 2);
        const class1 = words(1, 1, 2, 1, 1);
        const class2 = words(1, 2, 2, 1, 1);
        const cells = concat(words(...new Array<number>(15).fill(0)), words(13, 19, 23, 29, 31));
        tables.GPOS = layout(2, concat(words(2, 56, 4, 15, 56 + cov.length, 56 + cov.length + class1.length, 2, 2), cells, cov, class1, class2), "kern");
    } else if (kind === "c4-nested-ligature") {
        tables.GSUB = layoutFromLookups([lookup(6, context3(true, [1, 2, 3], [1, 1]), 8), lookup(4, ligature(2, 3), 8)], [0]);
    } else if (kind === "c4-nested-multiple") {
        const rule = words(3, 2, 2, 3, 0, 1, 2, 2);
        const context = concat(words(1, 8, 1, 14), coverage(1), words(1, 4), rule);
        const multiple = concat(words(1, 8, 1, 14), coverage(1), words(2, 7, 8));
        const laterSingle = concat(words(2, 10, 2, 9, 9), coverage(2, 3));
        tables.GSUB = layoutFromLookups([lookup(5, context), lookup(2, multiple), lookup(1, laterSingle)], [0]);
    } else if (kind === "c5-context-format3") {
        tables.GSUB = layoutFromLookups([lookup(5, context3(false, [1, 2], [1, 1])), lookup(1, single(2, 7))], [0]);
    } else if (kind === "c6-context-pos") {
        tables.GPOS = layoutFromLookups([lookup(8, context3(true, [1, 2], [1, 1])), lookup(1, concat(words(1, 14, 15, 11, 17, 13, 19), coverage(2)))], [0], "kern");
    } else if (kind === "c7-ligature-across-mark" || kind.startsWith("c8-")) {
        tables.GSUB = layoutFromLookups([lookup(4, ligature(1, 2), 8)], [0]);
        if (kind === "c8-mark-ligature") {
            tables.GPOS = layout(5, markAttachment(5), "mark");
        } else if (kind === "c8-mark-mark-components") {
            tables.GPOS = layout(6, markAttachment(6), "mkmk");
        }
    } else if (kind === "reverse-chain") {
        tables.GSUB = layout(8, concat(words(1, 12, 0, 0, 1, 7), coverage(1)));
    } else if (kind === "cursive") {
        tables.GPOS = layout(3, concat(words(1, 10, 1, 16, 22), coverage(1), words(1, 25, 47, 1, 45, 67)), "kern");
    } else if (kind === "nested-context") {
        tables.GSUB = layoutFromLookups([lookup(6, context3(true, [1, 2], [0, 1])), lookup(5, context3(false, [1, 2], [1, 2])), lookup(1, single(2, 7))], [0]);
    }
    return sfnt(tables);
}
