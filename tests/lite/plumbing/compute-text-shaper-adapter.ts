/** Test-only text-shaper 0.1.28 adapter. The compute page must not import this. */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Font, GlyphBuffer, UnicodeBuffer, shapeInto } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { reviewedFontKey, reviewedRecordByInput } from "../compute/references/reviewed-records";

export const TEXT_SHAPER_VERSION = JSON.parse(readFileSync(join(process.cwd(), "packages/babylon-lite/node_modules/text-shaper/package.json"), "utf8")).version as string;

export function shapeSixFields(file: string, text: string, script = "latn", direction: "ltr" | "rtl" = "ltr"): number[][] {
    const bytes = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    const key = reviewedFontKey(bytes);
    const row = key ? reviewedRecordByInput(key, text, script, direction) : undefined;
    if (row && row.category !== "U") {
        if (!row.records) {
            throw new Error(`${row.id}: incomplete reviewed fields`);
        }
        return row.records.map((record) => record.slice());
    }
    if (script === "arab" && !row) {
        throw new Error(`Unreviewed Arabic browser expectation: ${JSON.stringify(text)}`);
    }
    const font = Font.load(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    input.addStr(text);
    input.setScript(script);
    shapeInto(font, input, output, { script, language: null, direction });
    return output.infos.slice(0, output.length).map((info, index) => {
        const position = output.positions[index];
        return [info.glyphId, info.cluster, position?.xAdvance ?? 0, position?.yAdvance ?? 0, position?.xOffset ?? 0, position?.yOffset ?? 0];
    });
}
