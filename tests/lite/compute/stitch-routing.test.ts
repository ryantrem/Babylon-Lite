import { createHash } from "node:crypto";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { COMPUTE_TEXT_STITCH_KERNEL, executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import type { ResidentPackStage } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import type { ComputeTextSegments } from "../../../packages/babylon-lite/src/text/compute-text-segment";
import type { ComputeShader } from "../../../packages/babylon-lite/src/compute/compute-shader";
import * as shaderApi from "../../../packages/babylon-lite/src/compute/compute-shader";
import * as taskApi from "../../../packages/babylon-lite/src/compute/compute-task";
import * as storageApi from "../../../packages/babylon-lite/src/resource/storage-buffer";
import type { StorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

const BASELINE_HASH = "6a4543bb7fda56958d3e1dd99b447ca0c9a8e78cc8d0f66e30d400c08f747412";
const TILED_HASH = "7e0f74852cab4245de30fb6b69539fc808df7d17d705b263d7e603b9d41e18ff";
const hash = (text: string) => createHash("sha256").update(text).digest("hex");

describe("native resident stitch routing", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => native?.close());

    function fixture(runCount = 1, capacity = 65, staged = true) {
        const token = {};
        const reservedSegments = runCount * 65;
        const seg = new Int32Array(reservedSegments * 10);
        const table = new Uint32Array(runCount * 2 + reservedSegments);
        for (let i = 0; i < reservedSegments; i++) {
            seg.set([0, 1, 111, -222, 20 + i, 3, i + 1, 0, -i, i], i * 10);
            table[runCount * 2 + i] = i * 11;
        }
        const initial = new Int32Array(runCount * (4 + capacity * 6) + 16).fill(0x13579bdf);
        const buffers = {
            shaped: storageApi.createStorageBuffer(native.engine, initial, { writable: true }),
            instances: storageApi.createStorageBuffer(native.engine, runCount * capacity * 12, { writable: true }),
            indirect: storageApi.createStorageBuffer(native.engine, Uint32Array.of(6, 0, 0, 0), { writable: true }),
            slots: storageApi.createStorageBuffer(native.engine, new Uint32Array(256)),
            status: storageApi.createStorageBuffer(native.engine, Uint32Array.of(0), { writable: true }),
        };
        let segmentOutput: StorageBuffer | undefined;
        const allocate = storageApi.createStorageBuffer;
        vi.spyOn(storageApi, "createStorageBuffer").mockImplementation((engine, data, options) => {
            const buffer = allocate(engine, data, options);
            if (typeof options !== "string" && options?.label === "compute-text-segment-output") {
                segmentOutput = buffer;
                if (buffer.byteLength >= seg.byteLength) {
                    storageApi.updateStorageBuffer(engine, buffer, seg);
                }
            }
            return buffer;
        });
        const shaders: ComputeShader[] = [];
        const create = shaderApi.createComputeShader;
        vi.spyOn(shaderApi, "createComputeShader").mockImplementation((engine, options) => {
            const shader = create(engine, options);
            if (shader.name.startsWith("compute-text-stitch")) {
                shaders.push(shader);
            }
            return shader;
        });
        const compile = native.device.createComputePipelineAsync.bind(native.device);
        const compiled = vi.spyOn(native.device, "createComputePipelineAsync");
        const scheduled = vi.spyOn(taskApi, "addComputeDispatch");
        const pipelineNames = new Map<GPUComputePipeline, string>();
        compiled.mockImplementation(async (descriptor) => {
            const pipeline = await compile(descriptor);
            pipelineNames.set(pipeline, descriptor.label ?? "");
            return pipeline;
        });
        const gpuDispatches: { name: string; x: number; y: number; z: number }[] = [];
        const encode = native.device.createCommandEncoder.bind(native.device);
        vi.spyOn(native.device, "createCommandEncoder").mockImplementation((descriptor) => {
            const encoder = encode(descriptor);
            const begin = encoder.beginComputePass.bind(encoder);
            vi.spyOn(encoder, "beginComputePass").mockImplementation((passDescriptor) => {
                const pass = begin(passDescriptor);
                let name = "";
                const set = pass.setPipeline.bind(pass);
                vi.spyOn(pass, "setPipeline").mockImplementation((pipeline) => {
                    name = pipelineNames.get(pipeline) ?? "";
                    set(pipeline);
                });
                const dispatch = pass.dispatchWorkgroups.bind(pass);
                vi.spyOn(pass, "dispatchWorkgroups").mockImplementation((x, y = 1, z = 1) => {
                    if (name.startsWith("compute-text-stitch")) {
                        gpuDispatches.push({ name, x, y, z });
                    }
                    dispatch(x, y, z);
                });
                return pass;
            });
            return encoder;
        });
        const stage: ResidentPackStage = {
            version: 1,
            prepare(engine) {
                expect(segmentOutput).toBeDefined();
                expect(engine).toBe(native.engine);
            },
            dispatches: () => [],
            release() {},
        };
        const segments: ComputeTextSegments = { groups: [], table, capacity: 1, count: reservedSegments };
        return {
            shaders,
            compiled,
            scheduled,
            gpuDispatches,
            async run(counts: number[], failure?: number) {
                expect(counts).toHaveLength(runCount);
                counts.forEach((count, i) => table.set([i * 65, count], i * 2));
                seg[0] = failure ?? 0;
                if (segmentOutput) {
                    storageApi.updateStorageBuffer(native.engine, segmentOutput, seg);
                }
                storageApi.updateStorageBuffer(native.engine, buffers.shaped, initial);
                const before = gpuDispatches.length;
                const result = await executeResidentComputeText(token, native.engine, [], buffers, capacity, runCount, 1, 1, 256, () => {}, segments, staged ? stage : undefined);
                native.engine._flushPendingWork?.(native.engine);
                expect(await result.status).toBe(0);
                expect(gpuDispatches.slice(before)).toHaveLength(1);
                const selected = gpuDispatches.at(-1)!;
                const chosen = shaders.find((shader) => shader.name === selected.name)!;
                expect(hash(chosen._source)).toBe(Math.max(...counts) <= 64 ? BASELINE_HASH : TILED_HASH);
                expect(selected).toEqual({ name: chosen.name, x: runCount, y: Math.max(1, Math.ceil(Math.max(...counts) / 64)), z: 1 });
                const expected = initial.slice();
                counts.forEach((count, run) => {
                    const at = run * (4 + capacity * 6);
                    const status = run === 0 && failure !== undefined ? failure : count > capacity ? 2 : 0;
                    expected.set(status ? [status, 0, status === failure ? 111 : 0, status === failure ? -222 : 0] : [0, count, 0, 0], at);
                    if (!status) {
                        for (let i = 0; i < count; i++) {
                            const source = run * 65 + i;
                            expected.set([20 + source, 3 + source * 11, source + 1, 0, -source, source], at + 4 + i * 6);
                        }
                    }
                });
                const actual = await storageApi.readStorageBuffer(buffers.shaped);
                expect(Buffer.compare(Buffer.from(actual), Buffer.from(expected.buffer))).toBe(0);
            },
            close() {
                releaseResidentComputeText(token);
                Object.values(buffers).forEach(storageApi.disposeStorageBuffer);
            },
        };
    }

    it.each([0, 1, 64, 65])("routes S=%i through one exact shader/output dispatch", async (count) => {
        const test = fixture();
        try {
            await test.run([count]);
        } finally {
            test.close();
        }
    });

    it("keeps the exported tiled kernel byte-identical for the separate paged executor", () => {
        expect(hash(COMPUTE_TEXT_STITCH_KERNEL)).toBe(TILED_HASH);
    });

    it("prepares both variants before first submission and flips 1→2→1 rows without cold compile", async () => {
        const test = fixture();
        const submit = native.device.queue.submit.bind(native.device.queue);
        let recorded = false;
        vi.spyOn(native.device.queue, "submit").mockImplementation((commands) => {
            expect(test.shaders).toHaveLength(2);
            expect(test.shaders.every((shader) => shader._pipeline !== null)).toBe(true);
            expect(new Set(test.shaders.map((shader) => shader._pipeline)).size).toBe(2);
            if (!recorded) {
                console.warn(
                    "Resident stitch first-submission compile proof:",
                    JSON.stringify(
                        test.shaders.map((shader) => ({
                            name: shader.name,
                            sha256: hash(shader._source),
                            ready: shader._pipeline !== null,
                            compilations: test.compiled.mock.calls.filter(([descriptor]) => descriptor.label === shader.name).length,
                        }))
                    )
                );
                recorded = true;
            }
            submit(commands);
        });
        try {
            await test.run([64]);
            expect(test.shaders).toHaveLength(2);
            expect(test.shaders.every((shader) => shader._pipeline !== null)).toBe(true);
            const compileCount = test.compiled.mock.calls.length;
            await test.run([65]);
            await test.run([64]);
            expect(test.compiled).toHaveBeenCalledTimes(compileCount);
            const stitch = test.scheduled.mock.calls.map(([, dispatch]) => dispatch).filter((dispatch) => dispatch.shader.name.startsWith("compute-text-stitch"));
            expect(stitch).toHaveLength(3);
            expect(stitch.map((dispatch) => dispatch._y)).toEqual([1, 2, 1]);
            expect(new Set(stitch.map((dispatch) => dispatch.bindings)).size).toBe(3);
            expect(stitch.every((dispatch) => dispatch.bindings.shader === dispatch.shader)).toBe(true);
            expect(test.shaders.map((shader) => hash(shader._source)).sort()).toEqual([BASELINE_HASH, TILED_HASH].sort());
        } finally {
            test.close();
        }
        expect(test.shaders.every((shader) => shader._destroyed)).toBe(true);
    });

    it("uses the same route changes without a custom pack stage", async () => {
        const test = fixture(1, 65, false);
        try {
            await test.run([64]);
            await test.run([65]);
            await test.run([64]);
            expect(test.gpuDispatches.map((dispatch) => dispatch.y)).toEqual([1, 2, 1]);
        } finally {
            test.close();
        }
    });

    it("does not publish readiness when preparation of the initially unselected variant fails", async () => {
        const test = fixture();
        const prepare = test.compiled.getMockImplementation()!;
        const failure = new Error("synthetic tiled pipeline preparation failure");
        let reject = true;
        test.compiled.mockImplementation((descriptor) => {
            if (descriptor.label === "compute-text-stitch" && reject) {
                reject = false;
                return Promise.reject(failure);
            }
            return prepare(descriptor);
        });
        try {
            await expect(test.run([64])).rejects.toBe(failure);
            expect(test.gpuDispatches).toHaveLength(0);
            await test.run([64]);
            expect(test.shaders.every((shader) => shader._pipeline !== null)).toBe(true);
            const count = test.compiled.mock.calls.length;
            await test.run([65]);
            expect(test.compiled).toHaveBeenCalledTimes(count);
            expect(test.compiled.mock.calls.filter(([descriptor]) => descriptor.label === "compute-text-stitch")).toHaveLength(2);
        } finally {
            test.close();
        }
    });

    it("uses the batch maximum for mixed runs and preserves signed failure and overflow headers", async () => {
        const mixed = fixture(2);
        try {
            await mixed.run([1, 65]);
        } finally {
            mixed.close();
            vi.restoreAllMocks();
        }
        for (const count of [64, 65]) {
            const failed = fixture(1, 64);
            try {
                await failed.run([count], 3);
                await failed.run([count], -7);
                await failed.run([count]);
            } finally {
                failed.close();
                vi.restoreAllMocks();
            }
        }
    });
});
