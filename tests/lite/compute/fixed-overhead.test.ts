import { Buffer } from "node:buffer";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createComputeTextData, disposeComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { awaitComputeTextUpdate as updateComputeTextData } from "./await-update";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import { createGlyphStorage, disposeGlyphStorage } from "../../../packages/babylon-lite/src/text/glyph-storage";
import { closeComputeTextGlyphs } from "../../../packages/babylon-lite/src/text/compute-text-closure";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { mapComputeFontScalar } from "../../../packages/babylon-lite/src/text/compute-font";
import * as extraction from "../../../packages/babylon-lite/src/text/compute-glyph-extraction";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fixture, GREEK, CYRILLIC } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

const SERIAL_PREFIX = `
fn prefixOffsets() {
  indirect[0] = 6u;
  indirect[1] = 0u;
  indirect[2] = 0u;
  indirect[3] = 0u;
  var total = 0u;
  for (var run = 0u; run < params.runCount; run++) {
    let result = offsets[run];
    if (result.x > arrayLength(&instances) - total) { status[0] = 2u; return; }
    if (result.y != 0u) { status[0] = result.y; return; }
    offsets[run].x = total;
    total += result.x;
  }
  indirect[1] = total;
  status[0] = 0u;
}
@compute @workgroup_size(256)
fn countRuns(@builtin(global_invocation_id) invocation:vec3u) {
  countRun(invocation.x);
  storageBarrier();
  if (invocation.x == 0u) { prefixOffsets(); }
}
`;

describe("fixed-overhead differential contracts", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => native?.close());

    function readableSlots(data: ComputeTextData): void {
        disposeStorageBuffer(data._slots);
        data._slots = createStorageBuffer(native.engine, data._slotWords!, { writable: true });
    }

    it.each(["Inter.ttf", "Roboto-Regular.ttf"] as const)("incremental installation equals full-copy installation through closure transitions: %s", async (file) => {
        const font = fixture(file).font;
        const storage = createGlyphStorage();
        const referenceStorage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font, storage, file);
        const reference = createComputeTextData(native.engine, font, referenceStorage, file);
        readableSlots(data);
        readableSlots(reference);
        const originalExtract = extraction.extractComputeGlyphCurves;
        try {
            const transitions = [
                [],
                [""],
                [" "],
                ["HH"],
                ["fi AV é"],
                [GREEK],
                [CYRILLIC],
                ["office fi AV é", GREEK, CYRILLIC, " "],
                ["H"],
                ["H"],
                [],
                ["AV"],
                [" "],
                [""],
                ...[255, 256, 257, 1, 256, 0, 257].map((count) => Array.from({ length: count }, (_, run) => (run % 3 ? "fi H" : " "))),
            ];
            for (const texts of transitions) {
                const plans = texts.map((text) => createComputeTextPlan(font, selectComputeTextRun(text)));
                const seeds = texts.flatMap((text) => [...text].map((char) => mapComputeFontScalar(font, char.codePointAt(0)!)));
                const closure = closeComputeTextGlyphs(font, plans, seeds).glyphIds;
                await updateComputeTextData(data, texts);
                // Substitute the former full-copy extraction input for the reference update.
                const fullCopy = vi.spyOn(extraction, "extractComputeGlyphCurves").mockImplementation((source, _ids, target) => {
                    for (const [id, curves] of referenceStorage._curveSets.get(file)?._curves ?? []) {
                        target.set(id, curves);
                    }
                    originalExtract(source, new Set(closure), target);
                });
                try {
                    await updateComputeTextData(reference, texts);
                } finally {
                    fullCopy.mockRestore();
                }
                const atlas = storage._curveSets.get(file)!._atlas;
                const oldAtlas = referenceStorage._curveSets.get(file)!._atlas;
                expect([...atlas._glyphSlots]).toEqual([...oldAtlas._glyphSlots]);
                for (const key of ["_curveTexelsUsed", "_bandTexelsUsed", "_slotCount", "_version"] as const) {
                    expect(atlas[key]).toBe(oldAtlas[key]);
                }
                for (const key of ["_curveTexData", "_bandTexData", "_metaData"] as const) {
                    expect(Buffer.from(atlas[key].buffer).equals(Buffer.from(oldAtlas[key].buffer)), key).toBe(true);
                }
                const oldSlots = new Uint32Array(font.glyphCount).fill(0xffffffff);
                const curves = referenceStorage._curveSets.get(file)!._curves;
                for (const id of closure) {
                    oldSlots[id] = curves.has(id) ? oldAtlas._glyphSlots.get(id)!._index : 0xfffffffe;
                }
                expect(new Uint32Array(await readStorageBuffer(data._slots))).toEqual(oldSlots);
                expect(data._slotWords).toEqual(oldSlots);
                for (const key of ["_slots", "_shaped", "_instances", "_indirect", "_statusBuffer"] as const) {
                    expect(Buffer.from(await readStorageBuffer(data[key])).equals(Buffer.from(await readStorageBuffer(reference[key]))), key).toBe(true);
                }
            }
            const words = data._slotWords;
            await updateComputeTextData(data, ["HH"]);
            expect(data._slotWords).toBe(words);
            data._font = { ...font };
            await updateComputeTextData(data, ["AV"]);
            expect(data._slotWords).not.toBe(words);
            const resized = data._slotWords;
            (data._font as { glyphCount: number }).glyphCount--;
            await updateComputeTextData(data, ["AV"]);
            expect(data._slotWords).not.toBe(resized);
            expect(data._slotWords?.length).toBe(font.glyphCount - 1);
            data._onDeviceLost();
            expect(data._slotWords).toBeNull();
            expect(data._slotGlyphIds).toEqual([]);
        } finally {
            disposeComputeTextData(data);
            disposeComputeTextData(reference);
            expect(reference._slotWords).toBeNull();
            disposeGlyphStorage(storage);
            disposeGlyphStorage(referenceStorage);
        }
    });

    it("clears the ids touched by a failed slot build before the next successful install", async () => {
        const font = fixture("Roboto-Regular.ttf").font;
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font, storage, "slot-failure");
        readableSlots(data);
        try {
            await updateComputeTextData(data, ["AB"]);
            const before = await readStorageBuffer(data._slots);
            const id = mapComputeFontScalar(font, 66);
            const slot = storage._curveSets.get(data._curveSet)!._atlas._glyphSlots.get(id)!;
            const oldIndex = slot._index;
            slot._index = 0xffff;
            await expect(updateComputeTextData(data, ["AB"])).rejects.toThrow(`updateComputeTextData: closure cannot see atlas slot for glyph ${id}`);
            expect(new Uint8Array(await readStorageBuffer(data._slots))).toEqual(new Uint8Array(before));
            slot._index = oldIndex;
            await updateComputeTextData(data, [" "]);
            const slots = new Uint32Array(await readStorageBuffer(data._slots));
            expect(slots[id]).toBe(0xffffffff);
            expect(slots[mapComputeFontScalar(font, 65)]).toBe(0xffffffff);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    it("keeps outline extraction failures transactional, with the same wrapped error and recovery", async () => {
        const font = fixture("Roboto-Regular.ttf").font;
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font, storage, "extraction-failure");
        readableSlots(data);
        try {
            await updateComputeTextData(data, ["H"]);
            const atlas = storage._curveSets.get(data._curveSet)!._atlas;
            const slots = Buffer.from(await readStorageBuffer(data._slots));
            const installed = [...atlas._glyphSlots];
            const version = atlas._version;
            const original = extraction.extractComputeGlyphCurves;
            const failure = vi.spyOn(extraction, "extractComputeGlyphCurves").mockImplementation((source, ids, target) => {
                original(source, ids, target);
                throw new Error("outline failure witness");
            });
            try {
                for (let repeat = 0; repeat < 2; repeat++) {
                    await expect(updateComputeTextData(data, ["AV"])).rejects.toThrow("updateComputeTextData: closure cannot install outlines (outline failure witness)");
                    expect([...atlas._glyphSlots]).toEqual(installed);
                    expect(atlas._version).toBe(version);
                    expect(slots.equals(Buffer.from(await readStorageBuffer(data._slots)))).toBe(true);
                    expect([...new Uint32Array(await readStorageBuffer(data._indirect))]).toEqual([6, 0, 0, 0]);
                }
            } finally {
                failure.mockRestore();
            }
            await updateComputeTextData(data, ["AV"]);
            expect(data.status).toBe("ready");
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });

    async function readGpu(buffer: GPUBuffer, size = buffer.size): Promise<ArrayBuffer> {
        const staging = native.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        try {
            const encoder = native.device.createCommandEncoder();
            encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
            native.device.queue.submit([encoder.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            return staging.getMappedRange().slice(0);
        } finally {
            staging.unmap();
            staging.destroy();
        }
    }

    async function pack(records: Int32Array, count: number, instanceCapacity: number, unfused: boolean) {
        const engine = native.engine;
        const buffers = {
            shaped: createStorageBuffer(engine, Math.max(4, records.byteLength), { writable: true }),
            slots: createStorageBuffer(engine, Uint32Array.of(0xfffffffe, 7, 0xffffffff)),
            instances: createStorageBuffer(engine, Math.max(12, instanceCapacity * 12), { writable: true }),
            indirect: createStorageBuffer(engine, Uint32Array.of(6, 123, 456, 789), { writable: true }),
            status: createStorageBuffer(engine, Uint32Array.of(99), { writable: true }),
        };
        const token = {};
        let offsets: GPUBuffer | undefined;
        const allocate = vi.spyOn(native.device, "createBuffer");
        const modules = vi.spyOn(native.device, "createShaderModule");
        const owned: GPUBuffer[] = [];
        try {
            if (records.length) {
                updateStorageBuffer(engine, buffers.shaped, records);
            }
            const status = await (await executeResidentComputeText(token, engine, [], buffers, 2, count, 32, 2048, 3, () => undefined)).status;
            const allocation = allocate.mock.calls.reduce((last, [descriptor], index) => (descriptor.label === "compute-text-pack-offsets" ? index : last), -1);
            offsets = allocate.mock.results[allocation]!.value as GPUBuffer;
            if (unfused) {
                const source = modules.mock.calls.find(([descriptor]) => descriptor.code.includes("fn packSmall("))![0].code + SERIAL_PREFIX;
                const params = native.device.createBuffer({ size: 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const oldOffsets = native.device.createBuffer({ size: Math.max(8, count * 8), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                owned.push(params, oldOffsets);
                offsets = oldOffsets;
                native.device.queue.writeBuffer(params, 0, Uint32Array.of(count, 2, 0x3c800000, 3, 0x42000000));
                // Remove production output before running the independent old schedule.
                updateStorageBuffer(engine, buffers.instances, new Uint32Array(buffers.instances.byteLength / 4));
                updateStorageBuffer(engine, buffers.indirect, Uint32Array.of(6, 123, 456, 789));
                updateStorageBuffer(engine, buffers.status, Uint32Array.of(99));
                const resources = [
                    params,
                    buffers.shaped._buffer!,
                    buffers.slots._buffer!,
                    buffers.instances._buffer!,
                    buffers.indirect._buffer!,
                    buffers.status._buffer!,
                    oldOffsets,
                ];
                const layout = native.device.createBindGroupLayout({
                    entries: resources.map((_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding < 3 ? "read-only-storage" : "storage" } })),
                });
                const pipelineLayout = native.device.createPipelineLayout({ bindGroupLayouts: [layout] });
                const group = native.device.createBindGroup({ layout, entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) });
                const module = native.device.createShaderModule({ code: source });
                const entries = count <= 256 ? ["countRuns", "main"] : ["countParallel", "prefixRuns", "main"];
                const encoder = native.device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                for (const entryPoint of entries) {
                    const pipeline = await native.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } });
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, group);
                    pass.dispatchWorkgroups(entryPoint === "countRuns" || entryPoint === "prefixRuns" ? 1 : Math.max(1, Math.ceil(count / 64)));
                }
                pass.end();
                native.device.queue.submit([encoder.finish()]);
            }
            return {
                status: unfused ? new Uint32Array(await readStorageBuffer(buffers.status))[0] : status,
                indirect: new Uint8Array(await readStorageBuffer(buffers.indirect)),
                instances: new Uint8Array(await readStorageBuffer(buffers.instances)),
                offsets: new Uint8Array(await readGpu(offsets, Math.max(8, count * 8))),
            };
        } finally {
            allocate.mockRestore();
            modules.mockRestore();
            releaseResidentComputeText(token);
            Object.values(buffers).forEach(disposeStorageBuffer);
            owned.forEach((buffer) => buffer.destroy());
        }
    }

    function records(count: number): Int32Array {
        const words = new Int32Array(count * 16);
        for (let run = 0; run < count; run++) {
            words[run * 16 + 1] = run % 7 === 0 ? 0 : 2;
            words.set([0, 0, -100, 25, 0, 0, 1, 1, 200, -50, 10, -5], run * 16 + 4);
        }
        return words;
    }

    it.each([0, 1, 2, 255, 256, 257])("matches the unfused schedule bit-for-bit at %i runs", async (count) => {
        const words = records(count);
        expect(await pack(words, count, count, false)).toEqual(await pack(words, count, count, true));
    });

    it.each(["overflow-first", "failure-first", "same-run", "missing", "glyph-range", "capacity", "unknown", "negative"] as const)(
        "matches unfused status precedence, partial offsets and untouched instances: %s",
        async (kind) => {
            const words = records(256);
            let capacity = 256;
            if (kind === "overflow-first") {
                capacity = 1;
                words[15 * 16] = 1;
            }
            if (kind === "failure-first") {
                capacity = 1;
                words[0] = 1;
            }
            if (kind === "same-run" || kind === "missing") {
                words[2 * 16 + 4] = 1;
                words[2 * 16 + 10] = 2;
                capacity = kind === "same-run" ? 1 : 256;
            }
            if (kind === "glyph-range") {
                words[3 * 16 + 10] = 3;
            }
            if (kind === "capacity") {
                words[4 * 16 + 1] = 3;
            }
            if (kind === "unknown") {
                words[0] = 7;
            }
            if (kind === "negative") {
                words[0] = -1;
            }
            expect(await pack(words, 256, capacity, false)).toEqual(await pack(words, 256, capacity, true));
        }
    );
});
