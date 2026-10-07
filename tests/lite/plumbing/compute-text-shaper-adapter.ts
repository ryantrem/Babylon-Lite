/** Test-only text-shaper 0.1.28 adapter. The compute page must not import this. */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Font, GlyphBuffer, UnicodeBuffer, shapeInto } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

export const TEXT_SHAPER_VERSION = JSON.parse(readFileSync(join(process.cwd(), "packages/babylon-lite/node_modules/text-shaper/package.json"), "utf8")).version as string;

export function shapeSixFields(file: string, text: string, script = "latn", direction: "ltr" | "rtl" = "ltr"): number[][] {
    const bytes = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    const font = Font.load(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    input.addStr(text);
    shapeInto(font, input, output, { script, language: null, direction });
    return output.infos.slice(0, output.length).map((info, index) => {
        const position = output.positions[index];
        return [info.glyphId, info.cluster, position?.xAdvance ?? 0, position?.yAdvance ?? 0, position?.xOffset ?? 0, position?.yOffset ?? 0];
    });
}
