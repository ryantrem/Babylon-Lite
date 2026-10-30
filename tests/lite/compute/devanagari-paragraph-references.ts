/** Typed access to the CTF-03 HarfBuzz paragraph references and reference-derived (never Lite-derived) expectations. */
import { readFileSync } from "node:fs";

/** `[glyph, cluster, xAdvance, yAdvance, xOffset, yOffset]` in font units. */
export type ShapedRecord = number[];

export interface ParagraphReferenceRow {
    readonly id: string;
    readonly tags: readonly string[];
    readonly text: string;
    readonly whole: ShapedRecord[];
    /** HarfBuzz shaping of `text[start, end)` keyed `start:end`, for line boundaries at most `lineDistance` apart. */
    readonly lines?: Record<string, ShapedRecord[]>;
    readonly lineDistance: number;
    readonly candidates: number[];
    /** U+0020 indexes the paragraph space proof must mark safe. */
    readonly proven: number[];
}

export const paragraphReferences = JSON.parse(readFileSync(new URL("./references/deva-paragraph-harfbuzz-11.0.1.json", import.meta.url), "utf8")) as {
    readonly cases: ParagraphReferenceRow[];
};

export function referenceRow(id: string): ParagraphReferenceRow {
    const row = paragraphReferences.cases.find((c) => c.id === id);
    if (!row) {
        throw new Error(`no paragraph reference row ${id}`);
    }
    return row;
}

/** Scalar boundaries a line may start or end at: paragraph edges and the scalar after each breakable space. */
export function referenceBoundaries(row: ParagraphReferenceRow): number[] {
    const spaces = row.tags.includes("mixed") ? row.candidates : row.proven;
    return [0, ...spaces.map((s) => s + 1), [...row.text].length];
}

/** HarfBuzz shaping of `text[start, end)` between two boundaries, composed from stored lines at proven-independent boundaries when farther apart than `lineDistance`. */
export function referenceLine(row: ParagraphReferenceRow, start: number, end: number): ShapedRecord[] | undefined {
    const stored = row.lines?.[`${start}:${end}`];
    if (stored) {
        return stored;
    }
    const boundaries = referenceBoundaries(row);
    const from = boundaries.indexOf(start);
    const to = boundaries.lastIndexOf(end);
    if (from < 0 || to <= from + row.lineDistance) {
        return undefined;
    }
    const middle = boundaries[from + row.lineDistance]!;
    const left = referenceLine(row, start, middle);
    const right = referenceLine(row, middle, end);
    return left && right ? [...left, ...right.map(([glyph, cluster, ...rest]) => [glyph!, cluster! + middle - start, ...rest])] : undefined;
}

const graphemes = new Intl.Segmenter("und", { granularity: "grapheme" });

/** Oracle paragraph flags from the reference rows: SPACE (1), proven SPACE|BREAK_AFTER (3), and NO_BREAK_BEFORE (4) off grapheme starts (Intl.Segmenter). */
export function oracleParagraphFlags(text: string, proven: readonly number[]): Uint8Array {
    const scalars = [...text];
    const starts = new Set<number>();
    let unit = 0;
    let scalar = 0;
    for (const { index } of graphemes.segment(text)) {
        while (unit < index) {
            unit += scalars[scalar++]!.length;
        }
        starts.add(scalar);
    }
    const safe = new Set(proven);
    return Uint8Array.from(scalars, (s, i) => (s === " " ? (safe.has(i) ? 3 : 1) : starts.has(i) ? 0 : 4));
}
