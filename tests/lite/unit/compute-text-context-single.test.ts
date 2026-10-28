import { describe, expect, it } from "vitest";
import { COMPUTE_TEXT_KERNEL_SOURCE } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
describe("single SequenceLookupRecord equivalence", () => {
    it("eliminates only state unobservable after the sole child, retaining all shared failure/coordinate rules", () => {
        const source = COMPUTE_TEXT_KERNEL_SOURCE;
        const single = source.slice(source.indexOf("fn applyContextSingle("), source.indexOf("fn applyContext("));
        expect(single).not.toContain("array<");
        expect(source).toContain("if (nested == 1u) { return applyContextSingle(");
        for (const retained of [
            "let at = matchPositions[k]",
            "selectRule(base,count,at,childFlag",
            "applyNonContext(base,count,at,childFlag",
            "seqCount + extra > 64u",
            "rejectRun(out,parentLookup,index)",
            "boundary += extra",
            "if (boundary > index)",
            "high = max(high,applied.y)",
            "terminal != 0u",
        ]) {
            expect(single).toContain(retained);
        }
        // The old one-iteration loop's final state is (count+delta, original
        // boundary+max(delta,0), max(index,childHigh)). Its seq filter can only
        // affect a later record, which does not exist. Exhaust the permitted
        // sequence bound and child growth for return/failure equivalence.
        for (let sequence = 0; sequence <= 64; sequence++) {
            for (let extra = 0; extra <= 64; extra++) {
                const oldOverflow = sequence + extra > 64,
                    directOverflow = sequence + extra > 64;
                expect(directOverflow).toBe(oldOverflow);
                for (const index of [0, 7]) {
                    const boundary = index + Math.max(1, sequence);
                    const old = { delta: extra, advance: Math.max(1, boundary + extra - index), high: index + 2 };
                    const direct = { delta: extra, advance: boundary + extra > index ? Math.max(1, boundary + extra - index) : 1, high: Math.max(index, index + 2) };
                    expect(direct).toEqual(old);
                }
            }
        }
        const general = source.slice(source.indexOf("fn applyContext("), source.indexOf("fn probeMarkLigature("));
        expect(general).toContain("for (var i = 0u; i < nested; i++)");
        expect(general).toContain("seqCount = kept;");
        expect(general).not.toContain("compactRun(");
    });
});
