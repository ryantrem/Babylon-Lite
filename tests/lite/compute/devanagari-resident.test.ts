/** Full resident corpus through the normal lazy public profile loader. */
import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, updateComputeTextData, disposeComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { GlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fixture } from "./fixtures";
import { nativeHarness, assertNativeClean } from "./native";
import type { NativeHarness } from "./native";
interface Entry {
    id: string;
    text: string;
    expected: string;
    records?: number[][];
}
const corpus = (JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: Entry[] }).cases;
describe("actual resident shared-core Devanagari corpus", () => {
    let native: NativeHarness, storage: GlyphStorage, data: ComputeTextData;
    beforeAll(async () => {
        native = await nativeHarness();
        storage = createGlyphStorage();
        data = createComputeTextData(native.engine, fixture("NotoSansDevanagari-Regular.ttf").font, storage, "indic-resident");
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        disposeComputeTextData(data);
        disposeGlyphStorage(storage);
        await native?.close();
    });
    it.each(corpus.filter((entry) => entry.expected === "positive"))("$id: exact resident count, six fields, closure and pen placement", async (entry) => {
        const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
        await (
            await updateComputeTextData(data, [entry.text])
        ).completed;
        expect(maps.mock.calls.map((args) => args.slice(0, 3))).toEqual([[GPUMapMode.READ, 0, 4]]);
        maps.mockRestore();
        const shaped = new Int32Array(await readStorageBuffer(data._shaped));
        expect(Array.from(shaped.slice(0, 2))).toEqual([0, entry.records!.length]);
        expect(Array.from(shaped.slice(4, 4 + entry.records!.length * 6))).toEqual(entry.records!.flat());
        const upload = data._slots._data!;
        const slots = new Uint32Array(upload.buffer, upload.byteOffset, upload.byteLength / 4);
        const atlas = storage._curveSets.get("indic-resident")!._atlas!;
        const bytes = new DataView(await readStorageBuffer(data._instances));
        const scale = Math.fround(data.fontSizePx / data._font.unitsPerEm);
        let pen = 0,
            visible = 0;
        for (const record of entry.records!) {
            expect(slots[record[0]!]).not.toBe(0xffffffff);
            const slot = atlas._glyphSlots.get(record[0]!);
            if (slot) {
                const at = visible++ * 12;
                expect(bytes.getFloat32(at, true)).toBe(Math.fround(Math.fround(pen + record[4]!) * scale));
                expect(bytes.getFloat32(at + 4, true)).toBe(Math.fround(Math.fround(record[5]!) * scale));
                expect(bytes.getUint32(at + 8, true)).toBe(slot._index);
            }
            pen += record[2]!;
        }
        expect(Array.from(new Uint32Array(await readStorageBuffer(data._indirect)))).toEqual([6, visible, 0, 0]);
    });
});
