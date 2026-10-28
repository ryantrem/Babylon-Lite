/** Structural guard only — NOT evidence of unchanged speed.
 * Entries are table/index / descriptor capability / ordered bits / compact-footer eligibility.
 * Advance-only kern stays capability 3; Hebrew/Arabic placement pairs use 2.
 * Root order includes repeat Arabic lookups in distinct masked stages. Arabic has
 * no simple executor. RTL and missing-space programs keep whole-run fallback.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { createComputeTextPlan, ensureComputeTextProfile } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { computeTextPlanVariant, kernelSource, prepareComputeTextShapeWords } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { fixture } from "../compute/fixtures";

const samples = [
    [
        "Inter.ttf",
        "A",
        "default",
        false,
        2,
        true,
        "S3/0/5/0 S5/0/5/0 S6/0/5/0 S8/0/5/0 S9/0/5/0 S11/0/5/0 S47/0/5/0 S48/0/5/0 S51/0/5/0 S52/0/1/0 P1/3/1/1 P2/3/1/1 P3/3/1/1 P4/0/1/0 P5/0/1/0 P6/0/1/0 P7/0/1/0 P8/0/1/0 P9/0/1/0",
    ],
    [
        "Roboto-Regular.ttf",
        "A",
        "default",
        false,
        2,
        true,
        "S3/0/5/0 S4/0/5/0 S5/0/5/0 S6/0/5/0 S7/0/1/0 S8/0/5/0 S14/0/5/0 S15/0/5/0 P1/0/1/0 P2/0/1/0 P3/0/1/0 P4/0/1/0 P5/0/1/0 P6/0/1/0 P7/0/1/0 P8/0/1/0 P9/0/1/0 P10/0/1/0 P11/0/1/0 P12/0/1/0 P13/3/1/1",
    ],
    ["Inter.ttf", "Α", "default", false, 2, true, "S3/0/5/0 S5/0/5/0 S6/0/5/0 S8/0/5/0 S9/0/5/0 S11/0/5/0 S47/0/5/0 S48/0/5/0 S51/0/5/0 S52/0/1/0 P1/3/1/1 P2/3/1/1 P3/3/1/1"],
    [
        "Roboto-Regular.ttf",
        "Α",
        "default",
        false,
        2,
        true,
        "S3/0/5/0 S4/0/5/0 S5/0/5/0 S6/0/5/0 S7/0/1/0 S8/0/5/0 P1/0/1/0 P2/0/1/0 P3/0/1/0 P4/0/1/0 P5/0/1/0 P6/0/1/0 P7/0/1/0 P8/0/1/0 P9/0/1/0 P10/0/1/0 P11/0/1/0 P12/0/1/0 P13/3/1/1",
    ],
    ["Inter.ttf", "Б", "default", false, 2, true, "S3/0/5/0 S5/0/5/0 S6/0/5/0 S8/0/5/0 S9/0/5/0 S11/0/5/0 S47/0/5/0 S48/0/5/0 S51/0/5/0 S52/0/1/0 P1/3/1/1 P2/3/1/1 P3/3/1/1"],
    [
        "Roboto-Regular.ttf",
        "Б",
        "default",
        false,
        2,
        true,
        "S3/0/5/0 S4/0/5/0 S5/0/5/0 S6/0/5/0 S7/0/1/0 S8/0/5/0 P1/0/1/0 P2/0/1/0 P3/0/1/0 P4/0/1/0 P5/0/1/0 P6/0/1/0 P7/0/1/0 P8/0/1/0 P9/0/1/0 P10/0/1/0 P11/0/1/0 P12/0/1/0 P13/3/1/1",
    ],
    [
        "NotoSansHebrew-Regular.ttf",
        "ש",
        "hebrew",
        false,
        2,
        false,
        "S2/0/5/0 S3/0/5/0 S5/0/5/0 P0/0/1/0 P2/2/1/0 P4/0/1/0 P5/0/1/0 P6/0/1/0 P7/0/1/0 P8/0/1/0 P9/0/1/0 P10/0/1/0 P11/0/1/0",
    ],
    [
        "NotoSansArabic-Regular.ttf",
        "ب",
        "arabic",
        true,
        7,
        false,
        "S11/0/1/0 S13/0/5/0 S14/0/6/0 S15/0/6/0 S41/0/5/0 S43/0/5/0 S22/0/6/0 S4/1/5/0 S21/0/6/0 S4/1/5/0 S20/0/6/0 S25/0/1/0 S26/0/6/0 S27/0/1/0 S29/0/5/0 S31/0/5/0 S33/0/1/0 S35/0/5/0 S36/0/5/0 S37/0/5/0 P0/2/1/0 P2/0/1/0 P3/0/1/0 P4/0/1/0 P5/0/1/0 P6/0/1/0 P7/0/1/0 P8/0/1/0 P9/0/1/0 P10/0/1/0 P11/0/1/0 P12/0/1/0 P13/0/1/0 P14/0/1/0 P15/0/1/0 P16/0/1/0 P17/0/1/0 P18/0/1/0 P19/0/1/0 P20/0/1/0 P21/0/1/0 P22/0/1/0",
    ],
    ["NotoSansSC-Regular-subset.ttf", "汉", "default", false, 2, false, "P0/3/1/1"],
    ["NotoSansGothic-Regular.ttf", "𐌰", "default", false, 2, true, "S0/0/5/0 S1/0/1/0 S3/0/1/0 S5/0/1/0 S7/0/5/0 S10/0/5/0 P0/3/1/1 P1/0/1/0"],
] as const;

describe("shared-core structural executor guard", () => {
    beforeAll(async () => ensureComputeTextProfile("arabic"));
    it.each(samples)("%s / %s", (file, text, id, masked, stages, segmented, expectedRoots) => {
        const face = fixture(file);
        const plan = createComputeTextPlan(face.font, selectComputeTextRun(text));
        const prepared = prepareComputeTextShapeWords(plan, []);
        const program = prepared.program;
        const variant = computeTextPlanVariant(plan);
        expect({ id: variant.id, masked: variant.masked, stages: plan._stages.length, segmented: !!analyzeComputeTextSegmentation(program) }).toEqual({
            id,
            masked,
            stages,
            segmented,
        });
        const footer = prepared.programWords.length - prepared.simpleWordCount;
        const roots = program.rootOrder
            .map((at, root) => {
                const lookup = program.lookups[at]!;
                const descriptor = program.words[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
                const compact = !variant.masked && prepared.simpleWordCount > 0 && prepared.programWords[footer + root]! > 0;
                return `${lookup.table[1]}${lookup.lookupIndex}/${program.words[descriptor + 2]}/${program.words[descriptor + 7]}/${+compact}`;
            })
            .join(" ");
        expect(roots).toBe(expectedRoots);
        expect(kernelSource(0, variant).includes("fn finishSimpleRun")).toBe(!masked);
    });
});
