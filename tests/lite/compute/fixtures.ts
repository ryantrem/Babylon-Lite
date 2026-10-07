import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect } from "vitest";
import { Direction, Font, GlyphBuffer, UnicodeBuffer, shapeInto } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { createComputeFontFromBuffer } from "../../../packages/babylon-lite/src/index";
import type { ComputeFont, ComputeTextGlyph } from "../../../packages/babylon-lite/src/index";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";

export const FONT_HASHES = {
    "Inter.ttf": "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031",
    "Roboto-Regular.ttf": "56a45233d29f11b4dfb86d248e921939d115778f87325e7ae8cc108383d6664d",
    "NotoSansArabic-Regular.ttf": "7ed3fe069312aceac454f17cf613a30f95271d6ed7ce58005ed4d016bd3823d7",
    "NotoSansGothic-Regular.ttf": "1977c6f245a6f1ae4b421370b0aba480238cc8a31744b7e3ef7f92569f662c07",
    "NotoSansHebrew-Regular.ttf": "671951828bd5c95db818e5bb12dcea2d0c0dda00311888522be061ee6835125e",
    "NotoSansSC-Regular-subset.ttf": "db2aee41c6b823a7487f1057e4f0f429603acd5a02949dbb9cc176810e4b31d6",
} as const;

export type FontFile = keyof typeof FONT_HASHES;
export interface Fixture {
    file: FontFile | "synthetic";
    bytes: ArrayBuffer;
    font: ComputeFont;
    oracle: Font;
}

export function fixture(file: FontFile): Fixture {
    const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", file));
    expect(createHash("sha256").update(bytes).digest("hex"), file).toBe(FONT_HASHES[file]);
    const version = JSON.parse(readFileSync(join(process.cwd(), "packages/babylon-lite/node_modules/text-shaper/package.json"), "utf8")).version;
    expect(version).toBe("0.1.28");
    const data = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    return { file, bytes: data, font: createComputeFontFromBuffer(data), oracle: Font.load(data) };
}

export function fields(glyph: ComputeTextGlyph): number[] {
    return [glyph.glyphId, glyph.cluster, glyph.xAdvance, glyph.yAdvance, glyph.xOffset, glyph.yOffset];
}

export function oracleFields(font: Fixture, text: string): number[][] {
    const selection = oracleSelection(text);
    expect(selectComputeTextRun(text), `${font.file} independent script/direction ${JSON.stringify(text)}`).toMatchObject({
        ...selection,
        language: null,
        scalarCount: [...text].length,
    });
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    input.addStr(text);
    input.setScript(selection.script);
    input.setDirection(selection.direction === "rtl" ? Direction.RTL : Direction.LTR);
    shapeInto(font.oracle, input, output, { script: selection.script, language: null, direction: selection.direction });
    return output.infos.slice(0, output.length).map((info, index) => {
        const position = output.positions[index]!;
        return [info.glyphId, info.cluster, position.xAdvance, position.yAdvance, position.xOffset, position.yOffset];
    });
}

export function label(font: Fixture, text: string): string {
    return `${font.file} ${oracleSelection(text).script} ${JSON.stringify(text)} [${[...text].map((cp) => cp.codePointAt(0)!.toString(16)).join(" ")}]`;
}

function oracleSelection(text: string): { script: string; direction: "ltr" | "rtl" } {
    const scripts = [
        ["latn", /\p{Script=Latin}/u],
        ["grek", /\p{Script=Greek}/u],
        ["cyrl", /\p{Script=Cyrillic}/u],
        ["hebr", /\p{Script=Hebrew}/u],
        ["arab", /\p{Script=Arabic}/u],
        ["hani", /\p{Script=Han}/u],
        ["goth", /\p{Script=Gothic}/u],
    ] as const;
    const matches = scripts.filter(([, pattern]) => pattern.test(text));
    expect(matches.length, `positive corpus must be homogeneous: ${JSON.stringify(text)}`).toBeLessThanOrEqual(1);
    const script = matches[0]?.[0] ?? "DFLT";
    return { script, direction: script === "hebr" || script === "arab" ? "rtl" : "ltr" };
}

export function compare(font: Fixture, text: string, actual: readonly ComputeTextGlyph[]): void {
    const expected = oracleFields(font, text);
    const message = label(font, text);
    expect(actual.length, `${message}: count`).toBe(expected.length);
    actual.forEach((glyph, index) => expect(fields(glyph), `${message}: record ${index} [id,cluster,xAdvance,yAdvance,xOffset,yOffset]`).toEqual(expected[index]));
}

export function mapped(font: Fixture, start: number, end: number): string[] {
    return Array.from({ length: end - start + 1 }, (_, index) => start + index)
        .filter((scalar) => font.oracle.glyphId(scalar) !== 0)
        .map((scalar) => String.fromCodePoint(scalar));
}

export function seededStrings(alphabet: string, count = 64): string[] {
    let state = 0x4d313233;
    return Array.from({ length: count }, (_, run) =>
        Array.from({ length: 2 + (run % 49) }, () => {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return alphabet[state % alphabet.length]!;
        }).join("")
    );
}

export const GREEK = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωΑθήναςϊϋΐ";
export const CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяМоскваІіЇїЄє";
