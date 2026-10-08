import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { executeResidentComputeText, releaseResidentComputeText } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-resident";
import { createStorageBuffer, disposeStorageBuffer, readStorageBuffer, updateStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

describe("production resident prefix boundaries and ordered errors", () => {
    let native: NativeHarness;
    beforeAll(async () => {
        native = await nativeHarness();
    });
    afterEach(async () => {
        await assertNativeClean(native);
    });
    afterAll(async () => {
        await native?.close();
    });

    async function pack(records: Int32Array, runCount: number, instanceCapacity: number): Promise<{ status: number; indirect: number[]; instances: ArrayBuffer }> {
        const engine = native.engine;
        const buffers = {
            shaped: createStorageBuffer(engine, Math.max(4, records.byteLength), { writable: true }),
            slots: createStorageBuffer(engine, Uint32Array.of(0xfffffffe, 7, 0xffffffff)),
            instances: createStorageBuffer(engine, Math.max(12, instanceCapacity * 12), { writable: true }),
            indirect: createStorageBuffer(engine, 16, { writable: true }),
            status: createStorageBuffer(engine, 4, { writable: true }),
        };
        const token = {};
        try {
            if (records.length) {
                updateStorageBuffer(engine, buffers.shaped, records);
            }
            updateStorageBuffer(engine, buffers.indirect, Uint32Array.of(6, 123, 456, 789));
            updateStorageBuffer(engine, buffers.status, Uint32Array.of(99));
            const maps = vi.spyOn(GPUBuffer.prototype, "mapAsync");
            let status: number;
            try {
                status = await executeResidentComputeText(token, engine, [], buffers, 2, runCount, 32, 2048, 3, () => undefined);
                expect(maps.mock.calls.map((call) => call.slice(0, 3))).toEqual([[GPUMapMode.READ, 0, 4]]);
            } finally {
                maps.mockRestore();
            }
            return {
                status,
                indirect: [...new Uint32Array(await readStorageBuffer(buffers.indirect))],
                instances: await readStorageBuffer(buffers.instances),
            };
        } finally {
            releaseResidentComputeText(token);
            Object.values(buffers).forEach(disposeStorageBuffer);
        }
    }

    function records(count: number): Int32Array {
        const words = new Int32Array(count * 16);
        for (let run = 0; run < count; run++) {
            words[run * 16 + 1] = 2;
            // An empty outline still advances the pen before a drawable glyph.
            words.set([0, 0, -100, 25, 0, 0, 1, 1, 200, -50, 10, -5], run * 16 + 4);
        }
        return words;
    }

    it.each([0, 1, 255, 256, 257, 511, 512, 1023, 1024, 1025, 4097])("keeps every exact instance across %i runs, including negative advances", async (count) => {
        const shaped = records(count);
        for (let run = 0; run < count; run += 7) {
            shaped[run * 16 + 1] = 0;
        }
        const result = await pack(shaped, count, count);
        expect(result.status).toBe(0);
        const visible = count - Math.ceil(count / 7);
        expect(result.indirect).toEqual([6, visible, 0, 0]);
        const view = new DataView(result.instances);
        let at = 0;
        for (let run = 0; run < count; run++) {
            if (run % 7 === 0) {
                continue;
            }
            expect(view.getFloat32(at, true)).toBe(-90 / 64);
            expect(view.getFloat32(at + 4, true)).toBe(20 / 64 - run * 32);
            expect(view.getUint32(at + 8, true)).toBe(7);
            at += 12;
        }
    });

    it.each([1, 256, 257, 1025])("retains earliest status and capacity-before-status precedence at %i runs", async (count) => {
        const shaped = records(count);
        const last = (count - 1) * 16;
        shaped[last] = 1;
        expect((await pack(shaped, count, count)).status).toBe(1);
        shaped[last] = 0;
        shaped[last + 1] = 3;
        expect((await pack(shaped, count, count)).status).toBe(2);
        shaped[last + 1] = 2;
        shaped[last + 4] = 1;
        shaped[last + 10] = 2;
        const result = await pack(shaped, count, count);
        expect(result.status).toBe(3);
        expect(result.indirect).toEqual([6, 0, 0, 0]);
        if (count > 1) {
            // This run counts one drawable before encountering the missing slot.
            expect((await pack(shaped, count, count - 1)).status).toBe(2);
            shaped[0] = 1;
            expect((await pack(shaped, count, 1)).status).toBe(1);
            shaped[0] = 0;
            shaped[last + 10] = 3;
            expect((await pack(shaped, count, count)).status).toBe(3);
        }
        const recovered = await pack(records(count), count, count);
        expect(recovered.status).toBe(0);
        expect(recovered.indirect).toEqual([6, count, 0, 0]);
    });

    it.each([255, 256, 257, 1023, 1024])("orders failures and overflow around run %i without rewriting a failure", async (run) => {
        const count = 2049;
        const shaped = records(count);
        shaped[run * 16] = 7;
        shaped[(run + 1) * 16] = 1;
        let result = await pack(shaped, count, count);
        expect(result.status).toBe(7);
        expect(result.indirect).toEqual([6, 0, 0, 0]);
        shaped[run * 16] = -1;
        expect((await pack(shaped, count, count)).status).toBe(0xffffffff);
        shaped[run * 16] = 0;
        shaped[run * 16 + 4] = 1;
        shaped[run * 16 + 10] = 2;
        result = await pack(shaped, count, run);
        expect(result.status).toBe(2);
        expect(result.indirect).toEqual([6, 0, 0, 0]);
        expect((await pack(shaped, count, run + 1)).status).toBe(3);
        shaped[run * 16 + 1] = -1;
        expect((await pack(shaped, count, count)).status).toBe(2);
    });
});
