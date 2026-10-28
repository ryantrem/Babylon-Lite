import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { shapeComputeText } from "../../../packages/babylon-lite/src/index";
import { fields, fixture, oracleFields } from "./fixtures";
import { nativeHarness, assertNativeClean } from "./native";
import type { NativeHarness } from "./native";

interface Entry {
    id: string;
    text: string;
    expected: string;
    records?: number[][];
    profile: string;
    error?: string;
}
const corpus = (JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: Entry[] }).cases;

describe("actual public shared-core Devanagari corpus", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => {
        await native?.close();
    });

    it.each(corpus.filter((entry) => entry.expected === "positive"))("$id: exact count and all six fields", async (entry) => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        const actual = (await shapeComputeText(native.engine, face.font, entry.text)).map(fields);
        expect(actual, `${entry.id} ${JSON.stringify(entry.text)}`).toEqual(entry.records);
    });

    it("keeps weak-only DFLT selection and the unchanged legacy CPU difference explicit", async () => {
        const face = fixture("NotoSansDevanagari-Regular.ttf");
        for (const text of ["।", "॥", "।॥ 12"]) {
            expect((await shapeComputeText(native.engine, face.font, text)).map(fields)).toEqual(oracleFields(face, text));
        }
        const row = corpus.find((entry) => entry.text === "क्ष" && entry.expected === "positive")!;
        expect((await shapeComputeText(native.engine, face.font, row.text)).map(fields)).toEqual(row.records);
        expect(row.records!.map((record) => record[0])).toEqual([90]);
    });
});
