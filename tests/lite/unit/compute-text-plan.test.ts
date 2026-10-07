/** Default lookup plans. text-shaper is an oracle for lookup indexes only; these tests do not import the compute modules from it. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { createComputeFontFromBuffer, ensureComputeFontLayout } from "../../../packages/babylon-lite/src/text/compute-font";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { Font, getOrCreateShapePlan } from "../../../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

const INTER_GSUB = [3, 5, 6, 8, 9, 11, 47, 48, 51, 52];
const INTER_GPOS = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const ROBOTO_GSUB = [3, 4, 5, 6, 7, 8, 14, 15];
const ROBOTO_GPOS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

function load(file: string): { bytes: ArrayBuffer; font: ReturnType<typeof createComputeFontFromBuffer>; oracle: Font } {
    const buf = readFileSync(join(process.cwd(), "lab", "public", "fonts", file));
    const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
    return { bytes, font: createComputeFontFromBuffer(bytes), oracle: Font.load(bytes) };
}

function oracleClosure(lookups: readonly { subtables?: readonly object[] }[] | undefined, roots: readonly number[]): number[] {
    const seen = new Set<number>(roots);
    const extra: number[] = [];
    const queue = [...roots];
    for (let cursor = 0; cursor < queue.length; cursor++) {
        const lookup = lookups?.[queue[cursor] ?? -1];
        for (const nested of nestedLookupIndices(lookup)) {
            if (seen.has(nested)) {
                continue;
            }
            seen.add(nested);
            extra.push(nested);
            queue.push(nested);
        }
    }
    return extra.sort((left, right) => left - right);
}

function nestedLookupIndices(lookup: { subtables?: readonly object[] } | undefined): number[] {
    const found: number[] = [];
    const visit = (value: unknown): void => {
        if (!value || typeof value !== "object") {
            return;
        }
        if (Array.isArray(value)) {
            for (const item of value) {
                visit(item);
            }
            return;
        }
        const record = value as { lookupListIndex?: unknown; sequenceIndex?: unknown };
        if (typeof record.lookupListIndex === "number" && typeof record.sequenceIndex === "number") {
            found.push(record.lookupListIndex);
        }
        for (const [key, child] of Object.entries(value)) {
            if (key === "coverage" || key === "digest") {
                continue;
            }
            visit(child);
        }
    };
    visit(lookup?.subtables);
    return found;
}

describe("createComputeTextPlan", () => {
    it.each([
        ["Inter.ttf", INTER_GSUB, INTER_GPOS],
        ["Roboto-Regular.ttf", ROBOTO_GSUB, ROBOTO_GPOS],
    ] as const)("locks %s latn/und/ltr roots and nested dependencies", (file, gsub, gpos) => {
        const { font, oracle } = load(file);
        const selection = selectComputeTextRun("latin");
        const plan = createComputeTextPlan(font, selection);
        const oraclePlan = getOrCreateShapePlan(oracle, "latn", null, "ltr");
        expect(plan.script).toBe("latn");
        expect(plan.direction).toBe("ltr");
        expect(plan.language).toBeNull();
        expect(plan._gsubRoots).toEqual(gsub);
        expect(plan._gposRoots).toEqual(gpos);
        expect(plan._gsubRoots).toEqual(oraclePlan.gsubLookups.map((entry) => entry.index));
        expect(plan._gposRoots).toEqual(oraclePlan.gposLookups.map((entry) => entry.index));
        expect(plan._gsubDependencies).toEqual(oracleClosure(oracle.gsub?.lookups, plan._gsubRoots));
        expect(plan._gposDependencies).toEqual(oracleClosure(oracle.gpos?.lookups, plan._gposRoots));

        const layout = ensureComputeFontLayout(font);
        const latn = layout.gsub?.scripts.find((script) => script.tag === "latn");
        expect(latn?.defaultLangSys?.requiredFeatureIndex).toBe(0xffff);
    });

    it("rejects a forged script/direction pair and does not read extra option fields", () => {
        const { font } = load("Inter.ttf");
        const selection = selectComputeTextRun("A");
        const forged = { ...selection, direction: "rtl" as const, userFeatures: ["liga"], language: "ENG " } as unknown as typeof selection;
        expect(() => createComputeTextPlan(font, forged)).toThrow(/invalid script, direction, or language/);
    });

    it("includes a required feature that is not in the default set", () => {
        const font = createComputeFontFromBuffer(fontWithGsub(requiredFeatureGsub()));
        const plan = createComputeTextPlan(font, selectComputeTextRun("A"));
        expect(plan._gsubRoots).toEqual([0]);
    });

    it("deduplicates lookup references and closes an extension-wrapped dependency", () => {
        const duplicated = createComputeFontFromBuffer(fontWithGsub(duplicateFeatureGsub()));
        expect(createComputeTextPlan(duplicated, selectComputeTextRun("A"))._gsubRoots).toEqual([0]);

        const extended = createComputeFontFromBuffer(fontWithGsub(extensionDependencyGsub()));
        const plan = createComputeTextPlan(extended, selectComputeTextRun("A"));
        expect(plan._gsubRoots).toEqual([0]);
        expect(plan._gsubDependencies).toEqual([1]);
    });

    it("uses an empty plan for a missing layout table, an absent script, or a missing default LangSys", () => {
        const bare = createComputeFontFromBuffer(fontWithGsub(null));
        expect(createComputeTextPlan(bare, selectComputeTextRun("A"))._gsubRoots).toEqual([]);

        const arabOnly = createComputeFontFromBuffer(fontWithGsub(scriptOnlyGsub("arab")));
        expect(createComputeTextPlan(arabOnly, selectComputeTextRun("A"))._gsubRoots).toEqual([]);

        const noDefault = createComputeFontFromBuffer(fontWithGsub(langSysWithoutDefaultGsub()));
        expect(createComputeTextPlan(noDefault, selectComputeTextRun("A"))._gsubRoots).toEqual([]);
    });
});

function fontWithGsub(gsub: Uint8Array | null): ArrayBuffer {
    const tables: Record<string, Uint8Array> = {
        cmap: cmapFormat4(0x41, 1),
        glyf: new Uint8Array([0]),
        head: headTable(),
        hhea: hheaTable(),
        hmtx: hmtxTable(),
        loca: new Uint8Array(6),
        maxp: maxpTable(),
    };
    if (gsub) {
        tables.GSUB = gsub;
    }
    return buildSfnt(tables);
}

function requiredFeatureGsub(): Uint8Array {
    return assembleGsub(scriptTable("latn", 0, []), featureTable("smcp", [0]), singleLookupList());
}

function duplicateFeatureGsub(): Uint8Array {
    return assembleGsub(
        scriptTable("latn", 0xffff, [0, 1]),
        featureList([
            ["ccmp", [0]],
            ["liga", [0]],
        ]),
        singleLookupList()
    );
}

function scriptOnlyGsub(tag: string): Uint8Array {
    return assembleGsub(scriptTable(tag, 0xffff, [0]), featureTable("ccmp", [0]), singleLookupList());
}

function langSysWithoutDefaultGsub(): Uint8Array {
    const script = new Writer();
    script.u16(1);
    script.tag("latn");
    script.u16(8);
    script.u16(0);
    script.u16(1);
    script.tag("ENG ");
    script.u16(10);
    const lang = new Writer();
    lang.u16(0);
    lang.u16(0xffff);
    lang.u16(1);
    lang.u16(0);
    script.bytes.push(...lang.bytes);
    return assembleGsub(script.toUint8(), featureTable("ccmp", [0]), singleLookupList());
}

function extensionDependencyGsub(): Uint8Array {
    const lookups = new Writer();
    lookups.u16(2);
    lookups.u16(6);
    const extensionOffsetAt = lookups.offset();
    lookups.u16(0);

    lookups.u16(6);
    lookups.u16(0);
    lookups.u16(1);
    lookups.u16(8);
    const chainingSubtableAt = lookups.offset();
    lookups.u16(3);
    lookups.u16(0);
    lookups.u16(1);
    const coverageOffsetAt = lookups.offset();
    lookups.u16(0);
    lookups.u16(0);
    lookups.u16(1);
    lookups.u16(0);
    lookups.u16(1);
    lookups.patchU16(coverageOffsetAt, lookups.offset() - chainingSubtableAt);
    writeCoverage(lookups);

    const extensionAt = lookups.offset();
    lookups.patchU16(extensionOffsetAt, extensionAt);
    lookups.u16(7);
    lookups.u16(0);
    lookups.u16(1);
    lookups.u16(8);
    const extensionSubtableAt = lookups.offset();
    lookups.u16(1);
    lookups.u16(1);
    const innerOffsetAt = lookups.offset();
    lookups.u32(0);
    const innerAt = lookups.offset();
    lookups.patchU32(innerOffsetAt, innerAt - extensionSubtableAt);
    lookups.u16(1);
    const innerCoverageAt = lookups.offset();
    lookups.u16(0);
    lookups.i16(0);
    lookups.patchU16(innerCoverageAt, lookups.offset() - innerAt);
    writeCoverage(lookups);
    return assembleGsub(scriptTable("latn", 0xffff, [0]), featureTable("ccmp", [0]), lookups.toUint8());
}

function singleLookupList(): Uint8Array {
    const lookup = new Writer();
    lookup.u16(1);
    lookup.u16(4);
    lookup.u16(1);
    lookup.u16(0);
    lookup.u16(1);
    lookup.u16(8);
    lookup.u16(1);
    lookup.u16(6);
    lookup.i16(0);
    writeCoverage(lookup);
    return lookup.toUint8();
}

function scriptTable(tag: string, requiredFeatureIndex: number, featureIndices: number[]): Uint8Array {
    const writer = new Writer();
    writer.u16(1);
    writer.tag(tag);
    writer.u16(8);
    writer.u16(4);
    writer.u16(0);
    writer.u16(0);
    writer.u16(requiredFeatureIndex);
    writer.u16(featureIndices.length);
    for (const index of featureIndices) {
        writer.u16(index);
    }
    return writer.toUint8();
}

function featureTable(tag: string, lookups: number[]): Uint8Array {
    return featureList([[tag, lookups]]);
}

function featureList(features: [string, number[]][]): Uint8Array {
    const writer = new Writer();
    writer.u16(features.length);
    const recordAt = writer.offset();
    for (const [tag] of features) {
        writer.tag(tag);
        writer.u16(0);
    }
    features.forEach(([, lookups], index) => {
        const featureAt = writer.offset();
        writer.patchU16(recordAt + index * 6 + 4, featureAt);
        writer.u16(0);
        writer.u16(lookups.length);
        for (const lookup of lookups) {
            writer.u16(lookup);
        }
    });
    return writer.toUint8();
}

function assembleGsub(script: Uint8Array, feature: Uint8Array, lookup: Uint8Array): Uint8Array {
    const out = new Uint8Array(10 + script.length + feature.length + lookup.length);
    const view = new DataView(out.buffer);
    view.setUint16(0, 1);
    const scriptAt = 10;
    const featureAt = scriptAt + script.length;
    const lookupAt = featureAt + feature.length;
    view.setUint16(4, scriptAt);
    view.setUint16(6, featureAt);
    view.setUint16(8, lookupAt);
    out.set(script, scriptAt);
    out.set(feature, featureAt);
    out.set(lookup, lookupAt);
    return out;
}

function writeCoverage(writer: Writer): void {
    writer.u16(1);
    writer.u16(1);
    writer.u16(1);
}

class Writer {
    readonly bytes: number[] = [];

    offset(): number {
        return this.bytes.length;
    }

    u16(value: number): void {
        this.bytes.push((value >> 8) & 0xff, value & 0xff);
    }

    i16(value: number): void {
        this.u16(value < 0 ? value + 0x10000 : value);
    }

    u32(value: number): void {
        this.bytes.push((value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff);
    }

    tag(value: string): void {
        for (let index = 0; index < 4; index++) {
            this.bytes.push(value.charCodeAt(index) || 0x20);
        }
    }

    patchU16(at: number, value: number): void {
        this.bytes[at] = (value >> 8) & 0xff;
        this.bytes[at + 1] = value & 0xff;
    }

    patchU32(at: number, value: number): void {
        this.bytes[at] = (value >>> 24) & 0xff;
        this.bytes[at + 1] = (value >>> 16) & 0xff;
        this.bytes[at + 2] = (value >>> 8) & 0xff;
        this.bytes[at + 3] = value & 0xff;
    }

    toUint8(): Uint8Array {
        return Uint8Array.from(this.bytes);
    }
}

function buildSfnt(tables: Record<string, Uint8Array>): ArrayBuffer {
    const tags = Object.keys(tables);
    const headerSize = 12 + tags.length * 16;
    let offset = headerSize;
    const records = tags.map((tag) => {
        const body = tables[tag] ?? new Uint8Array();
        const record = { tag, offset, body };
        offset += body.length;
        return record;
    });
    const out = new Uint8Array(offset);
    const view = new DataView(out.buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, tags.length);
    records.forEach((record, index) => {
        const at = 12 + index * 16;
        for (let byte = 0; byte < 4; byte++) {
            view.setUint8(at + byte, record.tag.charCodeAt(byte) || 0x20);
        }
        view.setUint32(at + 8, record.offset);
        view.setUint32(at + 12, record.body.length);
        out.set(record.body, record.offset);
    });
    return out.buffer;
}

function headTable(): Uint8Array {
    const bytes = new Uint8Array(54);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint32(12, 0x5f0f3cf5);
    view.setUint16(18, 1000);
    return bytes;
}

function hheaTable(): Uint8Array {
    const bytes = new Uint8Array(36);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 1);
    view.setUint16(34, 2);
    return bytes;
}

function hmtxTable(): Uint8Array {
    const bytes = new Uint8Array(8);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, 100);
    view.setUint16(4, 200);
    return bytes;
}

function maxpTable(): Uint8Array {
    const bytes = new Uint8Array(32);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, 2);
    return bytes;
}

function cmapFormat4(codePoint: number, glyphId: number): Uint8Array {
    const bytes = new Uint8Array(4 + 8 + 12 + 32);
    const view = new DataView(bytes.buffer);
    view.setUint16(2, 1);
    view.setUint16(6, 3);
    view.setUint32(8, 12);
    const at = 12;
    view.setUint16(at, 4);
    view.setUint16(at + 2, 32);
    view.setUint16(at + 6, 4);
    view.setUint16(at + 8, 4);
    view.setUint16(at + 10, 1);
    view.setUint16(at + 12, 2);
    view.setUint16(at + 14, codePoint);
    view.setUint16(at + 16, 0xffff);
    view.setUint16(at + 20, codePoint);
    view.setUint16(at + 22, 0xffff);
    view.setInt16(at + 26, glyphId - codePoint);
    return bytes;
}
