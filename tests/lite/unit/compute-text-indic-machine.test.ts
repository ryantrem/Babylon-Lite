/** Test-only transcription of the GPU table walker, not a CPU shaping path.
 * DFA is generated from the pinned HB11.0.1 Indic grammar (Old MIT).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { INDIC_SYLLABLE_MACHINE as words } from "../../../packages/babylon-lite/src/text/compute-text-indic-unicode";
import { indicProperties } from "../../../packages/babylon-lite/src/text/compute-text-indic-plan";

function syllables(cats: number[]): { ranges: number[][]; bytes: number[] } {
    const data = (slot: number, index: number) => words[words[slot]! + index]!;
    const ranges: number[][] = [],
        bytes: number[] = new Array(cats.length).fill(0);
    let state = words[0]!,
        p = 0,
        start = 0,
        end = 0,
        act = 0,
        serial = 1,
        iterations = 0;
    for (;;) {
        // Unit-only progress assertion, never a production run limit/fallback.
        if (++iterations > cats.length * 4 + 8) {
            throw new Error("DFA failed to make progress");
        }
        let trans: number;
        if (p === cats.length) {
            const eof = data(9, state);
            if (!eof) {
                break;
            }
            trans = eof - 1;
        } else {
            if (data(8, state) === 10) {
                start = p;
            }
            const lo = data(1, state * 2),
                hi = data(1, state * 2 + 1),
                span = data(2, state),
                cat = cats[p]!;
            trans = data(4, data(3, state) + (span > 0 && cat >= lo && cat <= hi ? cat - lo : span));
        }
        state = data(5, trans);
        const action = data(6, trans);
        let kind = -1;
        if (action === 2) {
            end = p + 1;
        } else if (action === 11) {
            end = p + 1;
            kind = 5;
        } else if ([14, 15, 18, 20, 16, 17].includes(action)) {
            end = p;
            p--;
            kind = ({ 14: 0, 15: 1, 18: 2, 20: 3, 16: 4, 17: 5 } as Record<number, number>)[action]!;
        } else if ([1, 3, 7, 8, 4].includes(action)) {
            p = end - 1;
            kind = ({ 1: 0, 3: 1, 7: 2, 8: 3, 4: 4 } as Record<number, number>)[action]!;
        } else if (action === 6) {
            p = end - 1;
            kind = act === 1 ? 0 : act === 6 ? 4 : 5;
        } else if ([19, 13, 5, 12].includes(action)) {
            end = p + 1;
            act = ({ 19: 1, 13: 5, 5: 6, 12: 7 } as Record<number, number>)[action]!;
        }
        if (kind >= 0) {
            ranges.push([start, end, kind]);
            bytes.fill((serial << 4) | kind, start, end);
            serial = serial === 15 ? 1 : serial + 1;
        }
        if (data(7, state) === 9) {
            start = 0;
        }
        p++;
        if (p > cats.length) {
            break;
        }
    }
    return { ranges, bytes };
}

describe("Indic syllable DFA and GPU transcription", () => {
    it("classifies all 122 independent single-scalar references, including every broken starter", () => {
        const rows = (
            JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: { id: string; text: string; records?: number[][] }[] }
        ).cases.filter((r) => r.id.startsWith("scalar-"));
        expect(rows).toHaveLength(122);
        for (const row of rows) {
            const cat = indicProperties(row.text.codePointAt(0)!) & 255;
            const result = syllables([cat]);
            expect(result.ranges, row.id).toHaveLength(1);
            expect(result.ranges[0]!.slice(0, 2)).toEqual([0, 1]);
            expect(result.ranges[0]![2] === 4, row.id).toBe(row.records!.some((glyph) => glyph[0] === 789));
        }
    });
    it("keeps conjuncts, explicit half forms and halant-ZWNJ termination per the documented grammar", () => {
        expect(syllables([1, 4, 1]).ranges).toEqual([[0, 3, 0]]);
        expect(syllables([1, 4, 6, 1]).ranges).toEqual([[0, 4, 0]]);
        expect(syllables([1, 4, 5, 1, 7]).ranges).toEqual([
            [0, 3, 0],
            [3, 5, 0],
        ]);
        expect(syllables([7, 1]).ranges).toEqual([
            [0, 1, 4],
            [1, 2, 0],
        ]);
        expect(syllables([1, 0, 1]).ranges).toEqual([
            [0, 1, 0],
            [1, 2, 5],
            [2, 3, 0],
        ]);
    });
    it("wraps serials 1..15 without joining adjacent independent syllables", () => {
        const result = syllables(new Array<number>(20).fill(1));
        expect(result.ranges).toHaveLength(20);
        expect(result.bytes).toEqual(Array.from({ length: 20 }, (_, i) => ((i % 15) + 1) << 4));
    });
});
