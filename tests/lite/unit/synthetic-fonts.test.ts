import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ensureComputeFontLayout } from "../../../packages/babylon-lite/src/text/compute-font";
import { parseLookupActions } from "../../../packages/babylon-lite/src/text/font-parser/layout-actions";
import { syntheticFont, syntheticFontBytes, CORRECTION_KINDS } from "../compute/synthetic";
import type { SyntheticKind } from "../compute/synthetic";

// Frozen before changing the builder; protects all existing native witnesses.
const EXISTING_HASHES = {
    multiple: "ede722128eaa7c6fb1de8cf72bfc55b88ec0cb7c72ed21c56cfbd7801fbcbc84",
    alternate: "5271ee7844d34128875213bb21167b652ccdadec232f9631e73f87decab29d30",
    "mark-ligature": "b55e61f4ca172933f155e3189ee2dcc8072eaa92a5547b159cbeb4f2b5e8a0ca",
    "mark-mark": "a0942969c0d14bfece664a9cca0c99d1e4e6ce8049991faec94c494b5dc34ede",
    "point-anchor": "071b8f9a4ca2eeb4322c2a2c5e8d78d5622351243bdaf332a450270197e313ff",
    adjust: "5c079a760acb22f08b049caddfb0a512a3178e1cf70f329db5e58eb22c17e7eb",
    transitive: "e34fae4e9363f67884041997c1bc8d525ef6cb83052eb3129ca4427c5549f908",
    context: "3153323d447d3789f45a15cdc0ace70cd98d3403802d808fdb0a27605782566a",
    backtrack: "738d237470d84c8ba0026c6d50043dfc2be59b93c577fd78fd140e20ddc86147",
    "backtrack-reject": "312bc51785edf112497c83732c91a7136210a24d0217037a40fa69413db992c0",
    "ligature-ignore-marks": "bb9ae7ecb6bdcc39c72ed8eeda124c8f1ebf15264b45f51e1b068969a909fdc1",
    "pair-first": "fc39de46450c677e57f49b0f5d6d860455831a1205fb8e7e81ef77d642ff4e19",
    "pair-second": "2323a0901c0a919e5f98f8a3f804b00524fd8351758f5e75fb27c5551fdcc7e8",
    "pair-reject": "c9758ba6d92215afc20eafe9bb6f67e451892f0b77fa960dd887b59da5caf3d3",
    "refresh-roots": "472ad4693129145acc05862196b92222821a0df231c1aedd2285aaebe9cc5f81",
    "refresh-ligature": "083f991ce0a93832db470ee6fa399e075f733b35b99d223365f675177cef7118",
};
interface Witness {
    id: string;
    kind: string;
    correction: string | null;
    expected: string;
    records?: number[][];
    info_only?: number[][];
    oldTextShaperRecords: number[][];
    fontSha256: string;
    description: string;
    discriminates: string[];
    controls: string[];
    coExercised: string[];
}
const references = JSON.parse(readFileSync("tests/lite/compute/references/synthetic-harfbuzz-11.0.1.json", "utf8")) as { cases: Witness[] };

describe("independent synthetic correction fonts", () => {
    it("preserves the accepted font hashes and six-field records while annotating the witnesses", () => {
        const records = references.cases.map((entry) => [entry.id, entry.fontSha256, entry.records ?? entry.info_only, entry.oldTextShaperRecords]);
        expect(createHash("sha256").update(JSON.stringify(records)).digest("hex")).toBe("2191e712539fff9567d0ab2447b7e83134f62cdac1f7b8ed501b12ec4fe0e434");
    });

    it("identifies reviewed discriminator/control roles without narrowing any glyph-field assertion", () => {
        const witness = (id: string) => references.cases.find((entry) => entry.id === id)!;
        expect(witness("c8-mark-ligature-0").controls).toEqual(["C8"]);
        expect(witness("c8-mark-ligature-1").discriminates).toEqual(["C8"]);
        expect(witness("c8-mark-ligature-1").coExercised).toEqual(["C1", "C7"]);
        expect(witness("c8-mark-mark-components-0").discriminates).toEqual(["C8"]);
        expect(witness("c8-mark-mark-components-1").controls).toEqual(["C8"]);
        expect(witness("c8-mark-mark-components-0").coExercised).toEqual(["C7"]);
        expect(witness("c2-filter-set-0").discriminates).toEqual(["C2"]);
        expect(witness("c2-filter-set-1").controls).toEqual(["C2"]);
        expect(witness("c4-nested-ligature-0").coExercised).toEqual(["C7"]);
        expect(witness("c4-nested-multiple-0").description).not.toContain("require Lead clarification");
        expect(witness("c4-nested-multiple-0").records?.map((record) => record.slice(0, 2))).toEqual([
            [7, 0],
            [8, 0],
            [9, 1],
            [3, 2],
        ]);
    });
    it.each(Object.entries(EXISTING_HASHES))("preserves existing %s bytes", (kind, expected) => {
        expect(
            createHash("sha256")
                .update(new Uint8Array(syntheticFontBytes(kind as SyntheticKind)))
                .digest("hex")
        ).toBe(expected);
    });

    it.each(CORRECTION_KINDS)("decodes every layout action in %s without changing parser/runtime", (kind) => {
        const fixture = syntheticFont(kind);
        const layout = ensureComputeFontLayout(fixture.font);
        expect(layout.gdef).not.toBeNull();
        for (const table of ["GSUB", "GPOS"] as const) {
            const source = table === "GSUB" ? layout.gsub : layout.gpos;
            source?.lookups.forEach((_, index) => expect(parseLookupActions(fixture.font, table, index).subtables.length).toBeGreaterThan(0));
        }
        expect(fixture.oracle.glyphId(0x41)).toBe(1);
        expect(fixture.oracle.glyphId(0xe000)).toBe(4);
        expect(fixture.oracle.glyphId(0xe001)).toBe(5);
        expect(fixture.oracle.glyphId(0xe002)).toBe(0);
        expect(fixture.oracle.glyphId(0xc1)).toBe(0);
    });

    it("covers all eight corrections and retains deliberate negatives as information only", () => {
        expect([...new Set(references.cases.map((entry) => entry.correction).filter(Boolean))].sort()).toEqual(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]);
        for (const kind of CORRECTION_KINDS) {
            expect(
                references.cases.some((entry) => entry.kind === kind),
                kind
            ).toBe(true);
        }
        const negatives = references.cases.filter((entry) => entry.expected === "negative");
        expect(negatives).toHaveLength(3);
        for (const entry of negatives) {
            expect(entry.records).toBeUndefined();
            expect(entry.info_only!.length).toBeGreaterThan(0);
        }
    });

    it("isolates pen-relative propagation, filtering, continuation and component controls", () => {
        const records = (id: string) => references.cases.find((entry) => entry.id === id)!.records!;
        expect(records("c1-mark-mark-chain-0")).toEqual([
            [1, 0, 113, 0, 0, 0],
            [4, 1, 0, 0, -93, 40],
            [5, 2, 0, 0, -73, 80],
            [2, 3, 200, 0, 0, 0],
        ]);
        expect(records("c2-filter-set-0")[3]!.slice(4)).toEqual([20, 40]);
        expect(records("c2-filter-set-1")[2]!.slice(4)).toEqual([20, 40]);
        expect(references.cases.find((entry) => entry.id === "c2-filter-set-0")!.oldTextShaperRecords[3]!.slice(4)).toEqual([0, 0]);
        expect(references.cases.find((entry) => entry.id === "c2-filter-set-1")!.oldTextShaperRecords).toEqual(records("c2-filter-set-1"));
        expect(records("c3-pair-placement-0")[1]![4]).toBe(11);
        expect(records("c3-pair-value2-0")[2]).toEqual([3, 2, 300, 0, 0, 0]);
        expect(records("c7-ligature-across-mark-0")[1]![1]).toBe(0);
        expect(records("c8-mark-mark-components-0")[2]!.slice(4)).toEqual([0, 0]);
        expect(records("c8-mark-mark-components-1")[2]!.slice(4)).toEqual([20, 40]);
    });
});
