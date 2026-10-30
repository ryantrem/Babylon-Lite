/** Native qualification of the normal public Indic profile and legitimate loader-failure fixtures. */
import { readFileSync } from "node:fs";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { fixture, fields, oracleFields } from "./fixtures";
import { nativeHarness, assertNativeClean } from "./native";
import type { NativeHarness } from "./native";
import {
    createComputeTextPlan,
    ensureComputeTextProfile,
    installComputeTextProfile,
    installComputeTextProfileLoader,
    profileSupport,
} from "../../../packages/babylon-lite/src/text/compute-text-plan";
import type { ComputeTextProfileSupport } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { compileComputeTextProgram } from "../../../packages/babylon-lite/src/text/compute-text-program";
import { analyzeComputeTextSegmentation, segmentComputeTextRuns } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import { createComputeTextShaper, disposeComputeTextShaper, shapeComputeText, shapeComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import { createComputeTextData, disposeComputeTextData, updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { computeTextShapeCapacity } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";

interface Row {
    id: string;
    text: string;
    expected: string;
    records?: number[][];
}
const cases = (JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: Row[] }).cases;
const expected = (text: string) => cases.find((row) => row.text === text && row.expected === "positive")!.records!;
interface ParagraphRow {
    id: string;
    text: string;
    length: number;
    tags: string[];
    proven: number[];
    whole: number[][];
}
const paragraphCases = (JSON.parse(readFileSync("tests/lite/compute/references/deva-paragraph-harfbuzz-11.0.1.json", "utf8")) as { cases: ParagraphRow[] }).cases;

function pipelineStarted(native: NativeHarness): Promise<void> {
    let notify!: () => void;
    const entered = new Promise<void>((resolve) => {
        notify = resolve;
    });
    const original = native.device.createComputePipelineAsync.bind(native.device);
    vi.spyOn(native.device, "createComputePipelineAsync").mockImplementation((descriptor) => {
        notify();
        return original(descriptor);
    });
    return entered;
}

describe("native Indic resident, proof and lifetime integration", () => {
    let original: ComputeTextProfileSupport;
    const engines: NativeHarness[] = [];
    async function engine() {
        const native = await nativeHarness();
        engines.push(native);
        return native;
    }
    beforeAll(async () => {
        await ensureComputeTextProfile("indic");
        await ensureComputeTextProfile("arabic");
        original = profileSupport("indic")!;
        expect(original.available).toBe(true);
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        installComputeTextProfile("indic", original);
        for (const native of engines.splice(0)) {
            await assertNativeClean(native);
            await native.close();
        }
    });
    afterAll(() => installComputeTextProfile("indic", original));

    it("segments long runs at proven Indic spaces and stitches HarfBuzz whole-paragraph records", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf");
        const storage = createGlyphStorage(),
            data = createComputeTextData(native.engine, face.font, storage, "indic-segments");
        try {
            const rows = paragraphCases.filter((row) => row.length > 64 && !row.tags.includes("reference-only"));
            expect(rows.map((row) => row.id)).toEqual(expect.arrayContaining(["plain-rain", "overlong-single", "multiseam-forced", "multiseam-default"]));
            for (const row of rows) {
                const selection = selectComputeTextRun(row.text),
                    plan = createComputeTextPlan(face.font, selection);
                const groups = [{ plan, runs: [{ scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => face.oracle.glyphId(cp)) }], outputRun: 0 }];
                const segments = segmentComputeTextRuns(groups, native.device.limits, { headerWords: 4, recordWords: 6 });
                const starts = segments ? [...segments.table.subarray(2)] : [0];
                expect(starts[0], row.id).toBe(0);
                for (const start of starts.slice(1)) {
                    expect(row.proven, `${row.id} split after ${start - 1}`).toContain(start - 1);
                }
                expect(starts.length > 1, row.id).toBe(row.proven.some((space) => space > 0 && space < row.length - 1));
                await (
                    await updateComputeTextData(data, [row.text])
                ).completed;
                const words = new Int32Array(await readStorageBuffer(data._shaped));
                expect([...words.slice(0, 2)], row.id).toEqual([0, row.whole.length]);
                expect([...words.slice(4, 4 + row.whole.length * 6)], row.id).toEqual(row.whole.flat());
            }
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("segmented and whole-run shaping agree on long runs built from the hard paragraph rows", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf");
        const storage = createGlyphStorage(),
            data = createComputeTextData(native.engine, face.font, storage, "indic-segments-hard");
        try {
            const hard = paragraphCases
                .filter((row) => row.length <= 64 && /\S/.test(row.text) && !row.tags.includes("mixed") && !row.tags.includes("reference-only"))
                .map((row) => row.text);
            const texts = [hard.join(" "), [...hard].reverse().join(" "), hard.map((text) => `${text} ${text}`).join(" ")];
            for (const text of texts) {
                const selection = selectComputeTextRun(text),
                    plan = createComputeTextPlan(face.font, selection);
                const groups = [{ plan, runs: [{ scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => face.oracle.glyphId(cp)) }], outputRun: 0 }];
                expect(segmentComputeTextRuns(groups, native.device.limits, { headerWords: 4, recordWords: 6 })?.count).toBeGreaterThan(1);
                const results: number[][] = [];
                for (const disabled of [false, true]) {
                    data._disableSegmentation = disabled;
                    await (
                        await updateComputeTextData(data, [text])
                    ).completed;
                    const words = new Int32Array(await readStorageBuffer(data._shaped));
                    results.push([...words.slice(0, 4 + words[1]! * 6)]);
                }
                expect(results[0]).toEqual(results[1]);
            }
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("text-block analysis stays null for Indic; disabling segmentation preserves long independent records", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf");
        const storage = createGlyphStorage(),
            data = createComputeTextData(native.engine, face.font, storage, "indic-proof");
        try {
            const row = cases.find((row) => row.id === "long-512")!;
            const selection = selectComputeTextRun(row.text),
                plan = createComputeTextPlan(face.font, selection);
            const groups = [{ plan, runs: [{ scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => face.oracle.glyphId(cp)) }], outputRun: 0 }];
            expect(analyzeComputeTextSegmentation(compileComputeTextProgram(plan))).toBeNull();
            expect(segmentComputeTextRuns(groups, native.device.limits, { headerWords: 4, recordWords: 6 })?.count).toBeGreaterThan(1);
            for (const disabled of [false, true]) {
                data._disableSegmentation = disabled;
                await (
                    await updateComputeTextData(data, [row.text])
                ).completed;
                const words = new Int32Array(await readStorageBuffer(data._shaped));
                expect([...words.slice(0, 2)]).toEqual([0, row.records!.length]);
                expect([...words.slice(4, 4 + row.records!.length * 6)]).toEqual(row.records!.flat());
            }
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("bounds fail before atlas mutation, preserve the atlas and clear previously drawn indirect words", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf");
        const storage = createGlyphStorage(),
            data = createComputeTextData(native.engine, face.font, storage, "indic-bounds");
        try {
            await (
                await updateComputeTextData(data, ["क्ष"])
            ).completed;
            const atlas = storage._curveSets.get("indic-bounds")!._atlas!;
            const slots = [...atlas._glyphSlots.entries()],
                shaped = data._shaped;
            vi.spyOn(native.device.limits, "maxStorageBufferBindingSize", "get").mockReturnValue(512);
            await expect(updateComputeTextData(data, ["ऩे"])).rejects.toThrow(/updateComputeTextData:.*exceeds device/);
            expect(storage._curveSets.get("indic-bounds")!._atlas).toBe(atlas);
            expect([...atlas._glyphSlots.entries()]).toEqual(slots);
            expect(data._shaped).toBe(shaped);
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("GPU normalization growth and pack overflow are explicit status 2 and publish no partial instances", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf");
        const storage = createGlyphStorage(),
            data = createComputeTextData(native.engine, face.font, storage, "indic-overflow");
        try {
            await (
                await updateComputeTextData(data, ["ि"])
            ).completed;
            const atlas = storage._curveSets.get("indic-overflow")!._atlas!,
                slots = [...atlas._glyphSlots.entries()];
            const selection = selectComputeTextRun("ि"),
                plan = createComputeTextPlan(face.font, selection);
            const runs = [{ scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => face.oracle.glyphId(cp)) }];
            const undersized = { ...plan, _profileData: { ...plan._profileData!, capacity: () => 1 } };
            for (const shapeOverflow of [true, false]) {
                const token = {},
                    instances = createStorageBuffer(native.engine, shapeOverflow ? 24 : 12, { writable: true });
                try {
                    const submission = await executeResidentComputeText(
                        token,
                        native.engine,
                        [{ plan: shapeOverflow ? undersized : plan, runs, outputRun: 0 }],
                        {
                            shaped: data._shaped,
                            instances,
                            indirect: data._indirect,
                            slots: data._slots,
                            status: data._statusBuffer,
                        },
                        shapeOverflow ? 1 : computeTextShapeCapacity(plan, runs),
                        1,
                        32,
                        face.font.unitsPerEm,
                        face.font.glyphCount,
                        () => undefined
                    );
                    expect(await submission.status).toBe(2);
                    expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
                    expect([...atlas._glyphSlots.entries()]).toEqual(slots);
                } finally {
                    releaseResidentComputeText(token);
                    disposeStorageBuffer(instances);
                }
            }
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("normal public/resident homogeneous-run batches retain empty indexes and fail closed on an unmapped later run", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf"),
            storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, face.font, storage, "indic-batch");
        const texts = ["12", "द्ध", "", "।"];
        const refs = [oracleFields(face, "12"), expected("द्ध"), [], oracleFields(face, "।")];
        try {
            expect((await shapeComputeText(native.engine, face.font, texts)).map((row) => row.map(fields))).toEqual(refs);
            await (
                await updateComputeTextData(data, texts)
            ).completed;
            const capacity = Math.max(
                ...texts.map((text) => {
                    const selection = selectComputeTextRun(text);
                    return computeTextShapeCapacity(createComputeTextPlan(face.font, selection), [{ scalars: selection._scalars }]);
                })
            );
            const raw = new Int32Array(await readStorageBuffer(data._shaped)),
                stride = 4 + capacity * 6;
            refs.forEach((records, run) => {
                expect([...raw.slice(run * stride, run * stride + 2)]).toEqual([0, records.length]);
                expect([...raw.slice(run * stride + 4, run * stride + 4 + records.length * 6)]).toEqual(records.flat());
            });
            await expect(shapeComputeText(native.engine, face.font, ["द्ध", "a"])).rejects.toThrow(/run 1.*unmapped/);
            await expect(updateComputeTextData(data, ["द्ध", "a"])).rejects.toThrow(/run 1.*unmapped/);
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("three actual mixed plan variants preserve source order and independent records", async () => {
        const native = await engine();
        const fonts = [fixture("Roboto-Regular.ttf"), fixture("NotoSansArabic-Regular.ttf"), fixture("NotoSansDevanagari-Regular.ttf")];
        const texts = ["HH", "كتب", "द्ध"];
        const groups = fonts.map((face, outputRun) => {
            const selection = selectComputeTextRun(texts[outputRun]!);
            return {
                plan: createComputeTextPlan(face.font, selection),
                runs: [{ scalars: selection._scalars, glyphIds: selection._scalars.map((cp) => face.oracle.glyphId(cp)) }],
                outputRun,
            };
        });

        expect(groups.map((group) => group.plan._profile)).toEqual(["default", "arabic", "indic"]);
        const arabic = (JSON.parse(readFileSync("tests/lite/compute/references/arabic-harfbuzz-11.0.1.json", "utf8")) as { cases: Row[] }).cases;
        const refs = [oracleFields(fonts[0]!, "HH"), arabic.find((row) => row.text === "كتب")!.records!, expected("द्ध")];
        const capacity = Math.max(...groups.map((group) => computeTextShapeCapacity(group.plan, group.runs))),
            stride = 4 + capacity * 6;
        const glyphCount = Math.max(...fonts.map((face) => face.font.glyphCount)),
            token = {};
        const buffers = {
            shaped: createStorageBuffer(native.engine, groups.length * stride * 4, { writable: true }),
            instances: createStorageBuffer(native.engine, groups.length * capacity * 12, { writable: true }),
            indirect: createStorageBuffer(native.engine, Uint32Array.of(6, 0, 0, 0), { writable: true }),
            slots: createStorageBuffer(native.engine, new Uint32Array(glyphCount).fill(0xfffffffe)),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
        try {
            expect(await (await executeResidentComputeText(token, native.engine, groups, buffers, capacity, 3, 32, 1000, glyphCount, () => undefined)).status).toBe(0);
            const raw = new Int32Array(await readStorageBuffer(buffers.shaped));
            refs.forEach((records, i) => {
                expect([...raw.slice(i * stride, i * stride + 2)]).toEqual([0, records.length]);
                expect([...raw.slice(i * stride + 4, i * stride + 4 + records.length * 6)]).toEqual(records.flat());
            });
        } finally {
            releaseResidentComputeText(token);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    });

    it("busy/disposal during real public pipeline preparation and recreation are terminal and exact", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf");
        const plan = createComputeTextPlan(face.font, selectComputeTextRun("द्ध")),
            shaper = createComputeTextShaper(native.engine);
        const entered = pipelineStarted(native);
        const pending = shapeComputeTextPlan(shaper, plan, ["द्ध"]),
            rejected = expect(pending).rejects.toThrow(/disposed/);
        await entered;
        expect(shaper.status).toBe("busy");
        await expect(shapeComputeTextPlan(shaper, plan, ["क्ष"])).rejects.toThrow(/busy/);
        disposeComputeTextShaper(shaper);
        await rejected;
        expect((await shapeComputeText(native.engine, face.font, "द्ध")).map(fields)).toEqual(expected("द्ध"));
    });

    it("resident disposal while a real pipeline is pending leaves an independently owned survivor valid", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf"),
            storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, face.font, storage, "indic-life"),
            survivor = createComputeTextData(native.engine, face.font, storage, "indic-life");
        try {
            const entered = pipelineStarted(native);
            const pending = updateComputeTextData(data, ["द्ध"]),
                rejected = expect(pending).rejects.toThrow(/disposed/);
            await entered;
            await expect(updateComputeTextData(data, ["क्ष"])).rejects.toThrow(/busy/);
            disposeComputeTextData(data);
            await rejected;
            await (
                await updateComputeTextData(survivor, ["द्ध"])
            ).completed;
            const words = new Int32Array(await readStorageBuffer(survivor._shaped));
            expect([...words.slice(0, 2)]).toEqual([0, 1]);
            expect([...words.slice(4, 10)]).toEqual(expected("द्ध")[0]);
        } finally {
            disposeComputeTextData(data);
            disposeComputeTextData(survivor);
            disposeGlyphStorage(storage);
        }
    });

    it("real device loss cancels both public and resident preparation; independent/fresh engines remain valid", async () => {
        const lost = await engine(),
            surviving = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf"),
            storage = createGlyphStorage();
        const data = createComputeTextData(lost.engine, face.font, storage, "indic-lost");
        try {
            const entered = pipelineStarted(lost);
            const publicPending = shapeComputeText(lost.engine, face.font, "द्ध");
            const publicRejected = expect(publicPending).rejects.toThrow(/lost|destroyed/i);
            const residentPending = updateComputeTextData(data, ["द्ध"]);
            const residentRejected = expect(residentPending).rejects.toThrow(/lost|destroyed/i);
            await entered;
            lost.destroyExpected = true;
            lost.device.destroy();
            await lost.device.lost;
            await publicRejected;
            await residentRejected;
            expect(data.status).toBe("lost");
            expect(data._indirect._destroyed).toBe(true);
            await expect(updateComputeTextData(data, ["द्ध"])).rejects.toThrow(/lost/);
            expect((await shapeComputeText(surviving.engine, face.font, "द्ध")).map(fields)).toEqual(expected("द्ध"));
            const fresh = await engine();
            expect((await shapeComputeText(fresh.engine, face.font, "क्ष")).map(fields)).toEqual(expected("क्ष"));
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("first use coalesces lazy loading before either busy transition; failure publishes nothing and retry succeeds", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf"),
            storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, face.font, storage, "indic-cold");
        let release!: () => void, entered!: () => void;
        const gate = new Promise<void>((resolve) => {
                release = resolve;
            }),
            started = new Promise<void>((resolve) => {
                entered = resolve;
            });
        let attempts = 0;
        try {
            installComputeTextProfileLoader("indic", async () => {
                attempts++;
                entered();
                await gate;
                installComputeTextProfile("indic", original);
            });
            const resident = updateComputeTextData(data, ["द्ध"]),
                publicRun = shapeComputeText(native.engine, face.font, "क्ष");
            await started;
            expect(data.status).toBe("ready");
            expect(attempts).toBe(1);
            release();
            await (
                await resident
            ).completed;
            expect((await publicRun).map(fields)).toEqual(expected("क्ष"));
            installComputeTextProfileLoader("indic", async () => {
                throw new Error("Indic import witness");
            });
            await expect(updateComputeTextData(data, ["द्ध"])).rejects.toThrow("Indic import witness");
            expect(data.status).toBe("ready");
            expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
            installComputeTextProfile("indic", original);
            await (
                await updateComputeTextData(data, ["द्ध"])
            ).completed;
        } finally {
            release();
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("all explicit negative corpus boundaries reject without atlas publication", async () => {
        const native = await engine(),
            face = fixture("NotoSansDevanagari-Regular.ttf"),
            storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, face.font, storage, "indic-negatives");
        try {
            await (
                await updateComputeTextData(data, ["द्ध"])
            ).completed;
            const atlas = storage._curveSets.get("indic-negatives")!._atlas!,
                slots = [...atlas._glyphSlots.entries()];
            for (const row of cases.filter((row) => row.expected.startsWith("negative:"))) {
                const weakOnly = ["negative-soft-hyphen", "negative-zero-width-space", "negative-dotted-circle"].includes(row.id);
                // REQ-INPUT-3 applies to Devanagari runs; REQ-INPUT-1a preserves
                // existing DFLT-only behavior. Exercise the excluded scalar in
                // its declared Deva context rather than changing admission.
                const text = weakOnly ? `क${row.text}` : row.text;
                if (weakOnly) {
                    expect(selectComputeTextRun(row.text).script).toBe("DFLT");
                    expect((await shapeComputeText(native.engine, face.font, row.text)).map(fields)).toEqual(oracleFields(face, row.text));
                }
                await expect(shapeComputeText(native.engine, face.font, text), row.id).rejects.toThrow();
                await expect(updateComputeTextData(data, [text]), row.id).rejects.toThrow();
                expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
                expect([...atlas._glyphSlots.entries()]).toEqual(slots);
            }
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });
});
