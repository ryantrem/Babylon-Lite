import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { syntheticFont, syntheticFontBytes } from "../compute/synthetic";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { analyzeComputeTextSegmentation } from "../../../packages/babylon-lite/src/text/compute-text-segment";

describe("new negative fixture identities (all pre-existing bytes remain separately pinned)", () => {
    it.each([
        ["arabic-no-joining", "e42f90783e27566b8c94e319557c4a1cfe217292821482e745ad335bb1c4f30f"],
        ["segment-reject", "822ea2bffb78ad30ce98601528600606c7efb570b945e428d0301429139f79af"],
        ["contour-anchor", "2ff7e898883c70e7d0a77059a259b805e994ba3dd67f26d784369d5e54a39758"],
    ] as const)("pins %s", (kind, expected) => {
        expect(
            createHash("sha256")
                .update(new Uint8Array(syntheticFontBytes(kind)))
                .digest("hex")
        ).toBe(expected);
    });

    it("publishes exactly the pre-existing reverse-chain font for the browser negative", () => {
        expect(new Uint8Array(readFileSync("lab/public/fonts/test-unsupported-reverse-chain.ttf"))).toEqual(new Uint8Array(syntheticFontBytes("reverse-chain")));
    });

    it("proves the new segmented rejection, without relaxing the original safety proof", () => {
        const fixture = syntheticFont("segment-reject");
        const program = compileComputeTextProgram(createComputeTextPlan(fixture.font, selectComputeTextRun("A B")));
        expect(fixture.oracle.glyphId(32)).toBe(3);
        const proof = analyzeComputeTextSegmentation(program)!;
        expect({ span: proof.span, maxD: proof.maxD, crossings: proof.crossings }).toEqual({ span: 1, maxD: 1, crossings: 0 });
        expect(program.rejections).toEqual([{ table: "GSUB", lookupIndex: 0, lookupType: 3, reason: "GSUB alternate substitution is not executed (2 outputs)" }]);
    });
});
