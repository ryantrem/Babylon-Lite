import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import * as computeTasks from "../../../packages/babylon-lite/src/compute/compute-task";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

const EMPTY = 0xfffffffe;
const MISSING = 0xffffffff;
const SLOTS = Uint32Array.of(EMPTY, 7, MISSING, 9);
const COUNTS = [0, 1, 63, 64, 65, 127, 128, 129, 480, 1000];
const RUNS = [1, 3, 255, 256, 257, 700];

function records(capacity: number, counts: readonly number[]): Int32Array {
    const stride = 4 + capacity * 6;
    const words = new Int32Array(stride * counts.length);
    counts.forEach((count, run) => {
        words[run * stride + 1] = count;
        for (let i = 0; i < count; i++) {
            const empty = i === 0 || i === 63 || i === 64 || i === count - 1;
            words.set([empty ? 0 : i % 2 ? 1 : 3, i, i % 3 ? -101 : 203, i % 2 ? 37 : -19, i - 100, 50 - i], run * stride + 4 + i * 6);
        }
    });
    return words;
}

/** Literal serial count/prefix/pack reference, including partial counts and wrapping pens. */
function reference(words: Int32Array, capacity: number, runCount: number, limit: number, initial: Uint8Array) {
    const stride = 4 + capacity * 6;
    const offsets: number[] = [];
    const failures: number[] = [];
    for (let run = 0; run < runCount; run++) {
        const out = run * stride;
        let failed = words[out]! >>> 0;
        let visible = 0;
        if (!failed) {
            const count = words[out + 1]! >>> 0;
            if (count > capacity) {
                failed = 2;
            } else {
                for (let i = 0; i < count; i++) {
                    const glyph = words[out + 4 + i * 6]! >>> 0;
                    if (glyph >= SLOTS.length || SLOTS[glyph] === MISSING) {
                        failed = 3;
                        break;
                    }
                    if (SLOTS[glyph] !== EMPTY) {
                        visible++;
                    }
                }
            }
        }
        offsets.push(visible);
        failures.push(failed);
    }
    let total = 0;
    let status = 0;
    for (let run = 0; run < runCount; run++) {
        const count = offsets[run]!;
        offsets[run] = total;
        if (count > limit - total) {
            status = 2;
            break;
        }
        total += count;
        if (failures[run]) {
            status = failures[run]!;
            break;
        }
    }
    const instances = initial.slice();
    if (!status) {
        const view = new DataView(instances.buffer);
        for (let run = 0; run < runCount; run++) {
            let penX = 0;
            let penY = 0;
            let at = offsets[run]!;
            for (let i = 0; i < words[run * stride + 1]!; i++) {
                const record = run * stride + 4 + i * 6;
                const slot = SLOTS[words[record]!]!;
                if (slot !== EMPTY) {
                    view.setFloat32(at * 12, Math.fround(Math.fround((penX + words[record + 4]!) | 0) * Math.fround(32 / 2048)), true);
                    const y = Math.fround(Math.fround((penY + words[record + 5]!) | 0) * Math.fround(32 / 2048));
                    view.setFloat32(at * 12 + 4, Math.fround(y - Math.fround(Math.fround(run) * 32)), true);
                    view.setUint32(at * 12 + 8, slot, true);
                    at++;
                }
                penX = (penX + words[record + 2]!) | 0;
                penY = (penY + words[record + 3]!) | 0;
            }
        }
    }
    return { status, indirect: [6, status ? 0 : total, 0, 0], instances };
}

describe("native wide resident pack exactness", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        await native?.close();
    });

    function buffers(wordBytes: number, limit: number) {
        return {
            shaped: createStorageBuffer(native.engine, Math.max(4, wordBytes), { writable: true }),
            slots: createStorageBuffer(native.engine, SLOTS),
            instances: createStorageBuffer(native.engine, Math.max(12, limit * 12), { writable: true }),
            indirect: createStorageBuffer(native.engine, 16, { writable: true }),
            status: createStorageBuffer(native.engine, 4, { writable: true }),
        };
    }

    async function verify(words: Int32Array, capacity: number, runCount: number, limit: number, token = {}, existing?: ReturnType<typeof buffers>) {
        const owned = !existing;
        const storage = existing ?? buffers(words.byteLength, limit);
        const initial = new Uint8Array(storage.instances.byteLength).fill(0xa5);
        const expected = reference(words, capacity, runCount, storage.instances.byteLength / 12, initial);
        try {
            if (words.length) {
                updateStorageBuffer(native.engine, storage.shaped, words);
            }
            updateStorageBuffer(native.engine, storage.instances, initial);
            updateStorageBuffer(native.engine, storage.indirect, Uint32Array.of(123, 456, 789, 101));
            updateStorageBuffer(native.engine, storage.status, Uint32Array.of(99));
            const result = await executeResidentComputeText(token, native.engine, [], storage, capacity, runCount, 32, 2048, SLOTS.length, () => undefined);
            expect(await result.status).toBe(expected.status);
            expect([...new Uint32Array(await readStorageBuffer(storage.status))]).toEqual([expected.status]);
            const indirect = [...new Uint32Array(await readStorageBuffer(storage.indirect))];
            expect(indirect).toEqual(expected.indirect);
            const instances = new Uint8Array(await readStorageBuffer(storage.instances));
            expect(Buffer.from(instances).equals(Buffer.from(expected.instances)), "full instance byte image").toBe(true);
            return { instances, indirect };
        } finally {
            if (owned) {
                releaseResidentComputeText(token);
                Object.values(storage).forEach(disposeStorageBuffer);
            }
        }
    }

    it.each(COUNTS.flatMap((count) => RUNS.map((runs) => [count, runs] as const)))("matches all bytes for %i glyphs × %i runs", async (count, runs) => {
        const capacity = Math.max(65, count);
        await verify(
            records(
                capacity,
                Array.from({ length: runs }, () => count)
            ),
            capacity,
            runs,
            Math.max(1, count * runs)
        );
    });

    it("keeps mixed lengths and EMPTY chunk boundaries in serial order", async () => {
        await verify(records(1000, COUNTS), 1000, COUNTS.length, 4000);
    });

    it.each([0, 63, 64, 128].flatMap((index) => [2, 4].map((glyph) => [index, glyph] as const)))(
        "keeps the partial count before failure at %i (glyph %i)",
        async (index, glyph) => {
            const words = records(129, [129, 129, 129]);
            words[4 + index * 6] = glyph;
            await verify(words, 129, 3, 400);
            // A later failure cannot change the first run's status or partial visible count.
            words[2 * (4 + 129 * 6)] = 7;
            await verify(words, 129, 3, 400);
        }
    );

    it("chooses the first of two invalid glyphs, not the lowest lane or failure kind", async () => {
        const words = records(129, [129]);
        words[4 + 63 * 6] = 2;
        words[4 + 64 * 6] = 4;
        await verify(words, 129, 1, 61);
        await verify(words, 129, 1, 62);
    });

    it.each([1, 7, -1])("preserves header status %i ahead of later failures", async (header) => {
        const words = records(129, [129, 129, 129]);
        words[4 + 129 * 6] = header;
        words[2 * (4 + 129 * 6) + 4] = 4;
        await verify(words, 129, 3, 400);
    });

    it.each([130, -1])("rejects invalid count %i before reading glyphs", async (count) => {
        const words = records(129, [129, 129]);
        words[1] = count;
        await verify(words, 129, 2, 400);
    });

    it("handles exact capacity, one excess instance and overflow-before-partial-failure precedence", async () => {
        const words = records(129, [129, 129, 129]);
        const total = reference(words, 129, 3, 400, new Uint8Array(4800)).indirect[1]!;
        await verify(words, 129, 3, total);
        await verify(words, 129, 3, total - 1);
        words[4 + 129 * 6 + 4 + 64 * 6] = 2;
        const partial = total / 3 + 62;
        await verify(words, 129, 3, partial - 1);
        await verify(words, 129, 3, partial);
        words[2 * (4 + 129 * 6)] = 7;
        await verify(words, 129, 3, partial);
    });

    it("preserves wrapping pen sums and offsets around both i32 limits", async () => {
        const words = records(1000, [1000, 480, 129]);
        for (let run = 0; run < 3; run++) {
            for (let i = 0; i < words[run * 6004 + 1]!; i++) {
                const at = run * 6004 + 4 + i * 6;
                words[at + 2] = i % 2 ? -2147483611 : 2147483637;
                words[at + 3] = i % 3 ? 2147483600 : -2147483637;
                words[at + 4] = i % 2 ? 2147483640 : -2147483640;
                words[at + 5] = i % 2 ? -2147483600 : 2147483600;
            }
        }
        await verify(words, 1000, 3, 1800);
    });

    it("matches capacity 64 and 65 for the same logical records", async () => {
        const small = await verify(records(64, [64, 1, 63]), 64, 3, 128);
        const wide = await verify(records(65, [64, 1, 63]), 65, 3, 128);
        expect(small).toEqual(wide);
    });

    it("selects small at capacity 32, fused up to four runs and wide above at the tuned threshold 33 with identical output", async () => {
        const dispatches = vi.spyOn(GPUComputePassEncoder.prototype, "dispatchWorkgroups");
        const tasks = vi.spyOn(computeTasks, "createComputeTask");
        const small = await verify(records(32, [32, 1, 31]), 32, 3, 64);
        expect(dispatches.mock.calls.map(([x]) => x)).toEqual([1]);
        dispatches.mockClear();
        const fused = await verify(records(33, [32, 1, 31]), 33, 3, 64);
        expect(dispatches.mock.calls.map(([x]) => x)).toEqual([1]);
        expect(small).toEqual(fused);
        const packShaders = tasks.mock.results.filter((result) => result.type === "return" && result.value.name === "compute-text-resident-pack");
        expect(packShaders).toHaveLength(2);
        dispatches.mockClear();
        const smallFive = await verify(records(32, [32, 1, 31, 0, 7]), 32, 5, 71);
        dispatches.mockClear();
        const wide = await verify(records(33, [32, 1, 31, 0, 7]), 33, 5, 71);
        expect(dispatches.mock.calls.map(([x]) => x)).toEqual([5, 1, 5]);
        expect(smallFive).toEqual(wide);
    });

    it("fused pack matches the reference across overflow and failure precedence", async () => {
        const words = records(129, [129, 64, 2, 100]);
        const total = reference(words, 129, 4, 400, new Uint8Array(4800)).indirect[1]!;
        const firstTwo = reference(records(129, [129, 64]), 129, 2, 400, new Uint8Array(4800)).indirect[1]!;
        await verify(words, 129, 4, total);
        await verify(words, 129, 4, total - 1);
        await verify(words, 129, 4, firstTwo);
        words[129 * 6 + 4 + 4 + 10 * 6] = 2;
        await verify(words, 129, 4, total);
        await verify(words, 129, 4, firstTwo - 60);
        await verify(words, 129, 4, firstTwo - 50);
        words[0] = 7;
        await verify(words, 129, 4, total);
    });

    it("re-records small → fused → parallel → wide and updates both wide sizes on a stable token", async () => {
        const token = {};
        const storage = buffers(700 * (4 + 129 * 6) * 4, 700 * 65);
        const tasks = vi.spyOn(computeTasks, "createComputeTask");
        const dispatches = vi.spyOn(GPUComputePassEncoder.prototype, "dispatchWorkgroups");
        try {
            for (const [capacity, runs, expected] of [
                [2, 3, [1]],
                [129, 3, [1]],
                [2, 700, [11, 1, 11]],
                [129, 257, [257, 1, 257]],
                [129, 255, [255, 1, 255]],
                [129, 0, [1]],
            ] as const) {
                dispatches.mockClear();
                await verify(
                    records(
                        capacity,
                        Array.from({ length: runs }, () => Math.min(capacity, 65))
                    ),
                    capacity,
                    runs,
                    700 * 65,
                    token,
                    storage
                );
                expect(dispatches.mock.calls.map(([x]) => x)).toEqual(expected);
            }
            const packs = tasks.mock.results.filter((result) => result.type === "return" && result.value.name === "compute-text-resident-pack");
            expect(packs).toHaveLength(5);
            expect(packs.slice(0, -1).every((result) => result.value._disposed)).toBe(true);
            expect(packs.at(-1)!.value._disposed).toBe(false);
        } finally {
            releaseResidentComputeText(token);
            Object.values(storage).forEach(disposeStorageBuffer);
        }
    });

    it("shares one pack pass across independent objects admitted together, each with its own exact output and status", async () => {
        const cases = [
            { words: records(129, [129, 64, 2]), capacity: 129, runs: 3, limit: 400 },
            { words: records(32, [32, 1, 31]), capacity: 32, runs: 3, limit: 64 },
            { words: records(65, [65, 3, 65, 1, 64]), capacity: 65, runs: 5, limit: 400 },
            { words: records(129, [129, 129]), capacity: 129, runs: 2, limit: 10 },
        ];
        const storages = cases.map(({ words, limit }) => buffers(words.byteLength, limit));
        const tokens = cases.map(() => ({}));
        try {
            const expected = cases.map(({ words, capacity, runs }, index) => {
                const initial = new Uint8Array(storages[index]!.instances.byteLength).fill(0xa5);
                updateStorageBuffer(native.engine, storages[index]!.shaped, words);
                updateStorageBuffer(native.engine, storages[index]!.instances, initial);
                return reference(words, capacity, runs, storages[index]!.instances.byteLength / 12, initial);
            });
            const run = () =>
                Promise.all(
                    cases.map(({ capacity, runs }, index) =>
                        executeResidentComputeText(tokens[index]!, native.engine, [], storages[index]!, capacity, runs, 32, 2048, SLOTS.length, () => undefined)
                    )
                );
            await Promise.all((await run()).map((result) => result.status));
            const encoders = vi.spyOn(native.device, "createCommandEncoder");
            const results = await run();
            native.engine._flushPendingWork?.(native.engine);
            expect(encoders.mock.calls.filter(([descriptor]) => descriptor?.label === "compute-text-resident-shared-pack")).toHaveLength(1);
            expect(await Promise.all(results.map((result) => result.status))).toEqual(expected.map(({ status }) => status));
            for (const [index, storage] of storages.entries()) {
                expect([...new Uint32Array(await readStorageBuffer(storage.indirect))]).toEqual(expected[index]!.indirect);
                const instances = new Uint8Array(await readStorageBuffer(storage.instances));
                expect(Buffer.from(instances).equals(Buffer.from(expected[index]!.instances)), `object ${index} instance bytes`).toBe(true);
            }
        } finally {
            tokens.forEach(releaseResidentComputeText);
            storages.forEach((storage) => Object.values(storage).forEach(disposeStorageBuffer));
        }
    });

    it("falls back above the device's wide workgroup limit", async () => {
        const runs = native.device.limits.maxComputeWorkgroupsPerDimension + 1;
        const dispatches = vi.spyOn(GPUComputePassEncoder.prototype, "dispatchWorkgroups");
        await verify(
            records(
                65,
                Array.from({ length: runs }, () => 1)
            ),
            65,
            runs,
            1
        );
        expect(dispatches.mock.calls.map(([x]) => x)).toEqual([Math.ceil(runs / 64), 1, Math.ceil(runs / 64)]);
    });
});
