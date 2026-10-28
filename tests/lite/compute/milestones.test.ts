import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { shapeComputeText } from "../../../packages/babylon-lite/src/index";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";
import { compare, CYRILLIC, fixture, GREEK, mapped, seededStrings } from "./fixtures";
import { syntheticFont } from "./synthetic";

describe("native M1–3 production shaping", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => assertNativeClean(native));
    afterAll(async () => native?.close());

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("Latin repertoire and ordered witnesses: %s", async (file) => {
        const font = fixture(file);
        const texts = [
            "A",
            "AC",
            "5,",
            "To",
            "AV",
            "fi",
            "ffi",
            "office",
            "Café",
            "Æther",
            "Łódź",
            "1 < A",
            "A\u0326",
            "B\u0301",
            "B\u0301\u0307",
            "i\u0301",
            "abb",
            "aba",
            "ffif",
            "B\u0300\u0301",
        ];
        for (const text of texts) {
            compare(font, text, await shapeComputeText(native.engine, font.font, text));
        }
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("Latin-1/Extended scalar repertoire and seeded sequences: %s", async (file) => {
        const font = fixture(file);
        const texts = [
            ...mapped(font, 0x20, 0x7e),
            ...mapped(font, 0xa0, 0x24f),
            ...mapped(font, 0x1e00, 0x1eff),
            ...mapped(font, 0x2c60, 0x2c7f),
            ...mapped(font, 0xa720, 0xa7ff),
            ...mapped(font, 0xab30, 0xab6f),
            ...mapped(font, 0x2000, 0x200b),
            ...mapped(font, 0x2010, 0x2027),
            ...seededStrings("ABCDEFXYTabcfivxy 0123456789.,:;!?"),
            "A\u0308",
            "e\u0301",
            "a\u0301",
            "a\u0300",
            "a\u0302",
            "a\u0303",
            "a\u0304",
            "a\u0306",
            "a\u0307",
            "B\u0326",
            "A\u0327v",
            "B\u0301\u0300",
            "B\u0301\u0307",
            "a\u0301b",
            "1 < A",
        ];
        expect(texts).toHaveLength(file === "Inter.ttf" ? 897 : 1133);
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((run, index) => compare(font, texts[index]!, run));
    });

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("Greek and Cyrillic: %s", async (file) => {
        const font = fixture(file);
        expect(new Set(GREEK).size).toBe(53);
        expect(new Set(CYRILLIC).size).toBe(72);
        const texts = [...GREEK, ...CYRILLIC, GREEK, CYRILLIC];
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((run, index) => compare(font, texts[index]!, run));
    });

    it("CJK and supplementary Gothic", async () => {
        for (const [file, start, end, count] of [
            ["NotoSansSC-Regular-subset.ttf", 0x3400, 0x9fff, 237],
            ["NotoSansGothic-Regular.ttf", 0x10330, 0x1034a, 27],
        ] as const) {
            const font = fixture(file);
            const repertoire = mapped(font, start, end);
            expect(repertoire).toHaveLength(count);
            const texts = [...repertoire, repertoire.join("")];
            const result = await shapeComputeText(native.engine, font.font, texts);
            result.forEach((run, index) => compare(font, texts[index]!, run));
            expect(result.at(-1)!.map((glyph) => glyph.cluster)).toEqual(Array.from({ length: count }, (_, index) => index));
        }
    });

    it("Hebrew and Arabic", async () => {
        for (const [file, texts] of [
            ["NotoSansHebrew-Regular.ttf", ["שלום", "אבגד", "עברית", "שָׁלוֹם", "אב אב"]],
            ["NotoSansArabic-Regular.ttf", ["كتب", "سلام", "بب", "ببب", "ابت", "كُتُب", "العربية"]],
        ] as const) {
            const font = fixture(file);
            for (const text of texts) {
                compare(font, text, await shapeComputeText(native.engine, font.font, text));
            }
        }
    });

    it.each([
        ["NotoSansHebrew-Regular.ttf", 0x05d0, 0x05ea, "אבגדהוזחטיכלמנסעפצקרשת", ["אָ", "שׁ", "שָׁ", "בְ", "תּ"]],
        ["NotoSansArabic-Regular.ttf", 0x0620, 0x064a, "ابتجحخدذرزسشصضطظعغفقكلمنهوي", ["بَ", "بِ", "بُ", "بْ", "بّ", "بَّ", "بب ب", "لا"]],
    ] as const)("mapped RTL repertoire and seeded joining/marks: %s", async (file, start, end, alphabet, witnesses) => {
        const font = fixture(file);
        const texts = [...mapped(font, start, end), ...seededStrings(alphabet), ...witnesses];
        expect(texts).toHaveLength(file === "NotoSansHebrew-Regular.ttf" ? 96 : 115);
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((run, index) => compare(font, texts[index]!, run));
    });

    it("batch ordering and lengths", async () => {
        const font = fixture("Roboto-Regular.ttf");
        await expect(shapeComputeText(native.engine, font.font, [])).resolves.toEqual([]);
        const texts = ["", " ", "  ", "A", "Αθήνα", "Москва", ...seededStrings("ABCDfioxv ,.0123456789", 100), "A".repeat(512), "A".repeat(1024)];
        const result = await shapeComputeText(native.engine, font.font, texts);
        result.forEach((run, index) => compare(font, texts[index]!, run));
    });

    it.each([512, 1024, 2048])("completes every run and six fields in a natural Latin50 batch of %i", async (count) => {
        const font = fixture("Roboto-Regular.ttf");
        const phrases = ["The efficient flight lifts off at five. ", "WAVY typography: coffee, waffles and films. ", "A quick brown fox visits the office today. "];
        const texts = Array.from({ length: count }, (_, run) => {
            const phrase = phrases[run % phrases.length]!;
            return phrase.repeat(2).slice(0, 46) + String(run).padStart(4, "0");
        });
        const result = await shapeComputeText(native.engine, font.font, texts);
        expect(result).toHaveLength(count);
        result.forEach((run, index) => compare(font, texts[index]!, run));
    });

    it("fail-closed corpus", async () => {
        const font = fixture("Inter.ttf");
        for (const text of ["A\u20de", "A\u20dd", "<-"]) {
            compare(font, text, await shapeComputeText(native.engine, font.font, text));
        }
        for (const kind of ["alternate", "reverse-chain", "nested-context"] as const) {
            const negative = syntheticFont(kind);
            await expect(shapeComputeText(native.engine, negative.font, "AB")).rejects.toThrow(/unsupported GSUB 0/);
            await expect(shapeComputeText(native.engine, negative.font, ["", "C", "AB"])).rejects.toThrow(/run 2 unsupported GSUB 0/);
        }
        const contour = syntheticFont("contour-anchor");
        await expect(shapeComputeText(native.engine, contour.font, ["", "B", "ABC"])).rejects.toThrow(/run 2 unsupported GPOS 0/);
        for (const text of ["\ud800", "\udc00", "A\ud800B", "ΑЯ", "Aא", "अ", "한", "\n", "\r", "\t", "\u0000", "\u2028", "\u202e", "\u200d", "A\ufe0f", "A\u{e0100}"]) {
            await expect(shapeComputeText(native.engine, font.font, ["A", text])).rejects.toThrow(/run 1/);
        }
        const cjk = fixture("NotoSansSC-Regular-subset.ttf");
        await expect(shapeComputeText(native.engine, cjk.font, "A")).rejects.toThrow(/unmapped/);
    });
});
