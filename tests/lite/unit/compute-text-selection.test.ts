/** Script and direction selection. These tests do not import text-shaper. */

import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as unicodeScripts from "../../../packages/babylon-lite/src/text/unicode-script-data";
import { createComputeFontFromBuffer, mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";

import { selectComputeTextRun, scanComputeTextRun, computeTextRequiredProfiles, createComputeTextInputSlab } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import type { ComputeTextRunSelection, ComputeTextSelectionReuse } from "../../../packages/babylon-lite/src/text/compute-text-selection";

describe("selectComputeTextRun", () => {
    it("selects strong Devanagari including digits, keeps weak-only behavior, and mirrors the scanner", () => {
        for (const text of ["क", "१२३", "॑क", "क\u200dष", "क\u200cष"]) {
            expect(selectComputeTextRun(text)).toMatchObject({ script: "deva", direction: "ltr" });
            expect(scanComputeTextRun(text)).toMatchObject({ script: "deva", direction: "ltr", scalarCount: [...text].length });
            expect(computeTextRequiredProfiles([text])).toEqual(["indic"]);
        }
        for (const text of ["।", "॥", "।॥ 12"]) {
            expect(selectComputeTextRun(text).script).toBe("DFLT");
            expect(scanComputeTextRun(text).script).toBe("DFLT");
            expect(computeTextRequiredProfiles([text])).toEqual([]);
        }
        expect(() => selectComputeTextRun("॑")).toThrow("inherited mark without an admitted base script");
        expect(computeTextRequiredProfiles(["كتب", "क\u200dष"])).toEqual(["arabic", "indic"]);
    });
    it("retains conditional joiner and surrogate/unconditional error precedence in both selection paths", () => {
        for (const select of [selectComputeTextRun, scanComputeTextRun]) {
            expect(() => select("H\u200d\ud800")).toThrow("lone surrogate");
            expect(() => select("a\u200db")).toThrow("joining control at scalar 1");
            expect(() => select("\u200d")).toThrow("joining control at scalar 0");
            expect(() => select("\u200d\u2066क")).toThrow("bidi control at scalar 1");
            expect(() => select("कa")).toThrow('mixed scripts "deva" and "latn"');
            expect(() => select("a\u200dβ")).toThrow("joining control at scalar 1");
            expect(() => select("\u200d\u2066a")).toThrow("joining control at scalar 0");
            expect(() => select("aβ\u200d")).toThrow('mixed scripts "latn" and "grek"');
            expect(() => select("क\u200da")).toThrow('mixed scripts "deva" and "latn"');
        }
    });
    it("reuses immutable page directories within one decode while writing fresh scalar, offset and nominal vectors", () => {
        const bytes = readFileSync(join(process.cwd(), "lab/public/fonts/Inter.ttf"));
        const font = createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
        const text = "ABCD ".repeat(10);
        for (const scalar of new Set([...text].map((character) => character.codePointAt(0)!))) {
            mapComputeFontScalar(font, scalar);
        }
        let pageReads = 0;
        const pages = font._nominalPages!;
        font._nominalPages = new Proxy(pages, {
            get(target, key, receiver) {
                if (typeof key === "string" && /^\d+$/.test(key)) {
                    pageReads++;
                }
                return Reflect.get(target, key, receiver);
            },
        });
        const first = selectComputeTextRun(text, font);
        expect(pageReads).toBe(1);
        expect(first._scalars).toEqual(Uint32Array.from(text, (character) => character.codePointAt(0)!));
        expect(first._utf16Offsets).toEqual(Uint32Array.from({ length: text.length }, (_, index) => index));
        expect(first._nominal?.glyphIds).toEqual(Uint32Array.from(text, (character) => mapComputeFontScalar(font, character.codePointAt(0)!)));
        const second = selectComputeTextRun("DCBA", font);
        expect(second._scalars.buffer).not.toBe(first._scalars.buffer);
        expect(second._nominal?.glyphIds).toEqual(Uint32Array.of(...[..."DCBA"].map((character) => mapComputeFontScalar(font, character.codePointAt(0)!))));
        expect(() => selectComputeTextRun("ABβ\uD800", font)).toThrow(/lone surrogate at UTF-16 index 3/);
        expect(() => selectComputeTextRun("ABβ", font)).toThrow(/mixed scripts "latn" and "grek"/);
        expect(selectComputeTextRun("A\u0301B", font)).toMatchObject({ script: "latn", scalarCount: 3 });
    });

    it("reuses Unicode property lookup, not decoded inputs, while preserving scalar offsets and rejection order", () => {
        const tags = vi.spyOn(unicodeScripts, "scriptTagForScalar");
        try {
            const first = selectComputeTextRun("\u{10332}\u{10332}");
            const next = selectComputeTextRun("\u{10332}");
            expect(first._scalars).toEqual(Uint32Array.of(0x10332, 0x10332));
            expect(first._utf16Offsets).toEqual(Uint32Array.of(0, 2));
            expect(next._scalars).not.toBe(first._scalars);
            expect(next._scalars).toEqual(Uint32Array.of(0x10332));
            expect(tags.mock.calls.filter(([scalar]) => scalar === 0x10332)).toHaveLength(1);
            expect(() => selectComputeTextRun("\u{10332}A")).toThrow(/mixed scripts "goth" and "latn"/);
            expect(() => selectComputeTextRun("\u{10332}\nA")).toThrow(/control at scalar 1/);
            for (let index = 0; index < 2; index++) {
                expect(() => selectComputeTextRun("অ")).toThrow(/unsupported script "beng" at scalar 0/);
            }
        } finally {
            tags.mockRestore();
        }
    });

    it("allocates one owned scalar/offset slab per input with disjoint views and exact supplementary offsets", () => {
        const first = selectComputeTextRun("A\u0301");
        expect(first._scalars.buffer).toBe(first._utf16Offsets.buffer);
        expect(first._scalars.byteOffset + first._scalars.byteLength).toBeLessThanOrEqual(first._utf16Offsets.byteOffset);
        const next = selectComputeTextRun("\u{10330}\u{10332}");
        expect(next._scalars.buffer).toBe(next._utf16Offsets.buffer);
        expect(next._scalars.buffer).not.toBe(first._scalars.buffer);
        expect(next._scalars).toEqual(Uint32Array.of(0x10330, 0x10332));
        expect(next._utf16Offsets).toEqual(Uint32Array.of(0, 2));
        expect(next._scalars.byteOffset + next._scalars.byteLength).toBeLessThanOrEqual(next._utf16Offsets.byteOffset);
        first._scalars[0] = 0;
        expect(first._utf16Offsets).toEqual(Uint32Array.of(0, 1));
    });

    it("decodes fresh typed scalar and offset storage without intermediate numeric arrays", () => {
        const conversions = vi.spyOn(Uint32Array, "from");
        try {
            const selected = selectComputeTextRun("\u{10330}\u{10332}");
            expect(conversions).not.toHaveBeenCalled();
            expect(selected._scalars).toEqual(Uint32Array.of(0x10330, 0x10332));
            expect(selected._utf16Offsets).toEqual(Uint32Array.of(0, 2));
        } finally {
            conversions.mockRestore();
        }
    });

    it("resolves script during decoding without a second scalar traversal, preserving deferred rejection priority", () => {
        const source = readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/compute-text-selection.ts"), "utf8");
        expect(source).not.toContain("index < decoded.scalars.length");
        expect(() => selectComputeTextRun("\nA\uD800")).toThrow(/lone surrogate at UTF-16 index 2/);
        expect(() => selectComputeTextRun("Aβ\uD800")).toThrow(/lone surrogate at UTF-16 index 2/);
        expect(() => selectComputeTextRun("Aβ\n")).toThrow(/mixed scripts "latn" and "grek"/);
        expect(() => selectComputeTextRun("A\nβ")).toThrow(/control at scalar 1/);
    });

    it("caches complete numeric scalar admission rather than repeating rejection ranges and script dispatch in the warm decode loop", () => {
        const source = readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/compute-text-selection.ts"), "utf8");
        const decode = source.slice(source.indexOf("export function selectComputeTextRun"), source.indexOf("\nfunction "));
        expect(decode).not.toContain("rejectionReason(scalar)");
        expect(decode).not.toContain("admittedScript(");
        expect(source).toContain("new Uint8Array(256)");
        for (let revision = 0; revision < 2; revision++) {
            expect(selectComputeTextRun(revision ? "Η\u0301!" : "H\u0301!").script).toBe(revision ? "grek" : "latn");
            expect(() => selectComputeTextRun("H\u200D\uD800")).toThrow(/lone surrogate at UTF-16 index 2/);
            expect(() => selectComputeTextRun("H\u200D")).toThrow(/joining control at scalar 1 U\+200D/);
            expect(() => selectComputeTextRun("অ")).toThrow(/unsupported script "beng" at scalar 0 U\+0985/);
            expect(() => selectComputeTextRun("\u0378")).toThrow(/unknown script at scalar 0 U\+0378/);
        }
    });

    it("reuses adjacent scalar admission before script dispatch while decoding fresh offsets and preserving rejection priority", () => {
        const source = readFileSync(join(process.cwd(), "packages/babylon-lite/src/text/compute-text-selection.ts"), "utf8");
        const decode = source.slice(source.indexOf("export function selectComputeTextRun"), source.indexOf("\nfunction "));
        expect(decode).toMatch(/if \(repeated\) \{[\s\S]*?continue;\s*\}/);
        expect(decode).not.toContain("repeated ? lastClassification");
        const text = "A".repeat(50) + "\u0301".repeat(3) + "B".repeat(50);
        const first = selectComputeTextRun(text);
        expect(first.script).toBe("latn");
        expect(first._scalars).toEqual(Uint32Array.from(text, (char) => char.codePointAt(0)!));
        expect(first._utf16Offsets).toEqual(Uint32Array.from({ length: text.length }, (_, index) => index));
        expect(selectComputeTextRun(text)._scalars.buffer).not.toBe(first._scalars.buffer);
        expect(selectComputeTextRun("\u0301\u0301AA").script).toBe("latn");
        expect(() => selectComputeTextRun("\u0301\u0301")).toThrow(/inherited mark without an admitted base script/);
        expect(() => selectComputeTextRun("AAββ\uD800")).toThrow(/lone surrogate at UTF-16 index 4/);
        expect(() => selectComputeTextRun("AAββ\n")).toThrow(/mixed scripts "latn" and "grek"/);
        expect(() => selectComputeTextRun("AA\n\nββ")).toThrow(/control at scalar 2/);
        const gothic = selectComputeTextRun("\u{10330}".repeat(3));
        expect(gothic._utf16Offsets).toEqual(Uint32Array.of(0, 2, 4));
        expect(gothic.scalarCount).toBe(3);
    });

    it("selects admitted LTR and RTL scripts, including a supplementary Gothic scalar", () => {
        expect(selectComputeTextRun("")).toMatchObject({ script: "DFLT", direction: "ltr", language: null, scalarCount: 0 });
        expect(selectComputeTextRun("123, ")).toMatchObject({ script: "DFLT", direction: "ltr", scalarCount: 5 });
        expect(selectComputeTextRun("Café")).toMatchObject({ script: "latn", direction: "ltr", scalarCount: 4 });
        expect(selectComputeTextRun("Hello, world!")).toMatchObject({ script: "latn", direction: "ltr" });
        expect(selectComputeTextRun("Αθήνα")).toMatchObject({ script: "grek", direction: "ltr", scalarCount: 5 });
        expect(selectComputeTextRun("Москва")).toMatchObject({ script: "cyrl", direction: "ltr", scalarCount: 6 });
        expect(selectComputeTextRun("שלום")).toMatchObject({ script: "hebr", direction: "rtl", scalarCount: 4 });
        expect(selectComputeTextRun("كتب")).toMatchObject({ script: "arab", direction: "rtl", scalarCount: 3 });
        expect(selectComputeTextRun("一")).toMatchObject({ script: "hani", direction: "ltr", scalarCount: 1 });
        const gothic = selectComputeTextRun("\u{10330}");
        expect(gothic).toMatchObject({ script: "goth", direction: "ltr", scalarCount: 1 });
        expect(gothic._utf16Offsets[0]).toBe(0);
        expect("\u{10330}".length).toBe(2);
    });

    it("keeps a leading inherited mark with its following admitted base, and rejects an inherited-only string", () => {
        expect(selectComputeTextRun("\u0301A")).toMatchObject({ script: "latn", direction: "ltr", scalarCount: 2 });
        expect(() => selectComputeTextRun("\u0301")).toThrow(/inherited mark without an admitted base script/);
    });

    it("rejects lone surrogates without treating a valid supplementary pair as one", () => {
        expect(() => selectComputeTextRun("\uD800")).toThrow(/lone surrogate at UTF-16 index 0/);
        expect(() => selectComputeTextRun("\uDFFF")).toThrow(/lone surrogate/);
        expect(() => selectComputeTextRun("A\uD800B")).toThrow(/lone surrogate at UTF-16 index 1/);
    });

    it("rejects mixed scripts, unadmitted Indic scripts, and Hangul before any shaping", () => {
        expect(() => selectComputeTextRun("Aβ")).toThrow(/mixed scripts "latn" and "grek"/);
        expect(() => selectComputeTextRun("Hello, κόσμε")).toThrow(/mixed scripts "latn" and "grek"/);
        expect(() => selectComputeTextRun("Aكتب")).toThrow(/mixed scripts "latn" and "arab"/);
        expect(() => selectComputeTextRun("ক")).toThrow(/unsupported script "beng"/);
        expect(() => selectComputeTextRun("한")).toThrow(/unsupported script "hang"/);
        expect(() => selectComputeTextRun("\u1100")).toThrow(/unsupported script "hang"/);
    });

    it("rejects controls, paragraph separators, bidi controls, joining controls, and variation selectors", () => {
        expect(() => selectComputeTextRun("a\tb")).toThrow(/control/);
        expect(() => selectComputeTextRun("a\nb")).toThrow(/control/);
        expect(() => selectComputeTextRun("a\rb")).toThrow(/control/);
        expect(() => selectComputeTextRun("a\u2028b")).toThrow(/paragraph separator/);
        expect(() => selectComputeTextRun("a\u2029b")).toThrow(/paragraph separator/);
        expect(() => selectComputeTextRun("a\u202Ab")).toThrow(/bidi control/);
        expect(() => selectComputeTextRun("a\u2066b")).toThrow(/bidi control/);
        expect(() => selectComputeTextRun("a\u200Db")).toThrow(/joining control/);
        expect(() => selectComputeTextRun("a\uFE00")).toThrow(/variation selector/);
    });
    describe("incremental reuse of a prior plain decode", () => {
        function loadFont(name: string) {
            const bytes = readFileSync(join(process.cwd(), "lab/public/fonts", name));
            return createComputeFontFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
        }
        function random(seed: number): () => number {
            return () => {
                seed = (seed + 0x6d2b79f5) | 0;
                let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
                t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
                return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
            };
        }
        /** Production fills nominal pages after each decode; later decodes of those scalars hit. */
        function warm(font: ReturnType<typeof loadFont>, text: string): void {
            for (const character of text) {
                try {
                    mapComputeFontScalar(font, character.codePointAt(0)!);
                } catch {
                    // Unmapped scalars stay misses.
                }
            }
        }
        function firstOccurrences(glyphIds: Uint32Array): number[] {
            const seen = new Set<number>();
            const out: number[] = [];
            glyphIds.forEach((glyph, index) => {
                if (!seen.has(glyph)) {
                    seen.add(glyph);
                    out.push(index);
                }
            });
            return out;
        }
        type Outcome = { selection: ComputeTextRunSelection; seeds: number[]; byteOffset: number } | { error: string };
        function select(text: string, font: ReturnType<typeof loadFont>, prior: number[], previous?: ComputeTextSelectionReuse): Outcome {
            const seeds = new Set(prior);
            const slab = createComputeTextInputSlab(["pad", text], false);
            slab.byteOffset = 24;
            try {
                const selection = selectComputeTextRun(text, font, seeds, slab, previous);
                return { selection, seeds: [...seeds], byteOffset: slab.byteOffset };
            } catch (error) {
                return { error: (error as Error).message };
            }
        }
        function record(text: string, font: ReturnType<typeof loadFont>, selection: ComputeTextRunSelection): ComputeTextSelectionReuse | undefined {
            const strong = selection._reuseStrong;
            return strong === undefined
                ? undefined
                : { text, font, scalars: selection._scalars.slice(), glyphIds: selection._nominal!.glyphIds.slice(), strong, firsts: selection._firsts };
        }

        it("matches a full decode under random chained edits, stale records and pre-populated seed sets", () => {
            const latin = "The editor reviews a clear paragraph, checks its meaning, and shares the revised text.";
            const special = [" ", "  ", ".", "7", "\u0301", "\u200d", "β", "Ж", "😀", "\ud800", "\n", "क", "कि", "क्ष", "中", "\u2066", "ä"];
            const cases = [
                { font: loadFont("Inter.ttf"), base: latin },
                { font: loadFont("NotoSansDevanagari-Regular.ttf"), base: "नमस्ते दुनिया, यह एक परीक्षण वाक्य है। " },
            ];
            let reused = 0;
            let steps = 0;
            for (let trial = 0; trial < 60; trial++) {
                const rand = random(trial + 1);
                const { font, base } = cases[trial % cases.length]!;
                const piece = () => (rand() < 0.25 ? special[Math.floor(rand() * special.length)]! : base.slice(Math.floor(rand() * base.length)).slice(0, Math.floor(rand() * 6)));
                let text = base.repeat(3 + Math.floor(rand() * 8));
                if (trial % 3 === 0) {
                    text = " ".repeat(text.length);
                }
                const prior = trial % 2 ? [] : [...selectComputeTextRun("zyx ", font)._scalars].map((scalar) => scalar % 7);
                let previous: ComputeTextSelectionReuse | undefined;
                for (let edit = 0; edit < 30; edit++) {
                    const at = Math.floor(rand() * (text.length + 1));
                    const remove = rand() < 0.5 ? Math.floor(rand() * 4) : 0;
                    let insert = "";
                    for (let count = Math.floor(rand() * 3); count > 0; count--) {
                        insert += piece();
                    }
                    if (rand() < 0.05) {
                        insert = text.slice(0, Math.floor(text.length * 0.6));
                    }
                    text = text.slice(0, at) + insert + text.slice(at + remove);
                    if (rand() < 0.15 && previous) {
                        const roll = Math.floor(rand() * text.length);
                        text = text.slice(roll) + text.slice(0, roll);
                    }
                    const full = select(text, font, prior);
                    const incremental = select(text, font, prior, previous);
                    steps++;
                    if ("error" in full) {
                        expect(incremental).toEqual(full);
                        warm(font, text);
                        continue;
                    }
                    expect("error" in incremental ? incremental.error : null).toBeNull();
                    const result = incremental as Exclude<Outcome, { error: string }>;
                    const expected = full.selection;
                    const actual = result.selection;
                    expect([actual.script, actual.direction, actual.scalarCount, actual._reuseStrong]).toEqual([
                        expected.script,
                        expected.direction,
                        expected.scalarCount,
                        expected._reuseStrong,
                    ]);
                    expect(actual._scalars).toEqual(expected._scalars);
                    expect(actual._utf16Offsets.length).toBe(0);
                    expect(actual._nominal!.glyphIds).toEqual(expected._nominal!.glyphIds);
                    expect(actual._nominal!.misses).toBe(expected._nominal!.misses);
                    expect(result.seeds).toEqual(full.seeds);
                    expect(result.byteOffset).toBe(full.byteOffset);
                    if (actual._firsts) {
                        reused++;
                        const [prefix, suffix] = actual._unchanged!;
                        expect(actual._scalars.subarray(0, prefix)).toEqual(previous!.scalars.subarray(0, prefix));
                        expect(actual._scalars.subarray(actual.scalarCount - suffix)).toEqual(previous!.scalars.subarray(previous!.scalars.length - suffix));
                        expect(actual._nominal!.glyphIds.subarray(actual.scalarCount - suffix)).toEqual(previous!.glyphIds.subarray(previous!.glyphIds.length - suffix));
                        expect([...actual._firsts]).toEqual(firstOccurrences(expected._nominal!.glyphIds));
                    }
                    if (previous?.firsts) {
                        expect([...previous.firsts]).toEqual(firstOccurrences(previous.glyphIds));
                    }
                    previous = record(text, font, actual) ?? previous;
                    warm(font, text);
                }
            }
            expect(steps).toBe(1800);
            expect(reused).toBeGreaterThan(200);
        });

        it("keeps the prior strong script only while a strong scalar survives outside the replaced span", () => {
            const font = loadFont("Inter.ttf");
            const spaces = " ".repeat(300);
            warm(font, "aβ ");
            const latin = selectComputeTextRun("a" + spaces, font);
            const previous = record("a" + spaces, font, latin)!;
            expect(previous.strong).not.toBe(0);
            const slab = createComputeTextInputSlab([spaces + " "], false);
            const dflt = selectComputeTextRun(" " + spaces, font, undefined, slab, previous);
            expect([dflt.script, dflt._reuseStrong, dflt._firsts !== undefined]).toEqual(["DFLT", 0, true]);
            const kept = selectComputeTextRun("a" + spaces.slice(1) + "  ", font, undefined, createComputeTextInputSlab([spaces + "  "], false), previous);
            expect([kept.script, kept._firsts !== undefined]).toEqual(["latn", true]);
            const greek = selectComputeTextRun("β" + spaces, font, undefined, createComputeTextInputSlab([spaces + " "], false), record(" " + spaces, font, dflt)!);
            expect([greek.script, greek._firsts !== undefined]).toEqual(["grek", true]);
            const mixed = "a" + spaces + "β";
            expect(() => selectComputeTextRun(mixed, font, undefined, createComputeTextInputSlab([mixed], false), previous)).toThrow(/mixed scripts "latn" and "grek"/);
        });

        it("finds suffix first occurrences of glyphs whose prior first occurrence was replaced", () => {
            const font = loadFont("Inter.ttf");
            warm(font, "aQb");
            const old = "a".repeat(200) + "Q" + "b".repeat(100) + "Q";
            const text = "a".repeat(201) + "b".repeat(100) + "Q";
            const previous = record(old, font, selectComputeTextRun(old, font))!;
            const seeds = new Set<number>();
            const reused = selectComputeTextRun(text, font, seeds, createComputeTextInputSlab([text], false), previous);
            const full = new Set<number>();
            const expected = selectComputeTextRun(text, font, full, createComputeTextInputSlab([text], false));
            expect(reused._firsts && [...reused._firsts]).toEqual([0, 201, 301]);
            expect([...seeds]).toEqual([...full]);
            expect(reused._nominal!.glyphIds).toEqual(expected._nominal!.glyphIds);
        });

        it("ignores records for another font, short text or offset-bearing slabs", () => {
            const inter = loadFont("Inter.ttf");
            const roboto = loadFont("Roboto-Regular.ttf");
            const text = "abc ".repeat(80);
            warm(inter, text);
            warm(roboto, text);
            const previous = record(text, inter, selectComputeTextRun(text, inter))!;
            expect(selectComputeTextRun(text, roboto, undefined, createComputeTextInputSlab([text], false), previous)._firsts).toBeUndefined();
            expect(selectComputeTextRun(text, inter, undefined, createComputeTextInputSlab([text], true), previous)._firsts).toBeUndefined();
            expect(selectComputeTextRun(text, inter, undefined, createComputeTextInputSlab([text], false), previous)._firsts).toBeDefined();
            const short = "abc ".repeat(10);
            expect(
                selectComputeTextRun(short, inter, undefined, createComputeTextInputSlab([short], false), record(short, inter, selectComputeTextRun(short, inter)))._firsts
            ).toBeUndefined();
        });
    });
});
