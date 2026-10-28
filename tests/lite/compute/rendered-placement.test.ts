/** Explicitly authorized, reference-positioned draw comparison. Never a golden update. */
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import { PNG } from "pngjs";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
    createComputeTextData,
    createComputeTextLayer,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    updateComputeTextData,
    createTextData,
    createTextLayer,
    disposeTextData,
} from "../../../packages/babylon-lite/src/index";
import { getOrCreateTextPipeline, clearTextPipelineCache } from "../../../packages/babylon-lite/src/text/_gpu/text-pipeline";
import type { CpuTextLayerDraw } from "../../../packages/babylon-lite/src/text/text-layer-cpu";
import { readStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";
import { fixture } from "./fixtures";
import type { FontFile } from "./fixtures";
import { reviewedRecordByInput } from "./references/reviewed-records";
import { nativeHarness, assertNativeClean } from "./native";
import type { NativeHarness } from "./native";

const deva = (JSON.parse(readFileSync("tests/lite/compute/references/deva-harfbuzz-11.0.1.json", "utf8")) as { cases: { text: string; records?: number[][] }[] }).cases;
const arabic = (JSON.parse(readFileSync("tests/lite/compute/references/arabic-harfbuzz-11.0.1.json", "utf8")) as { cases: { text: string; records: number[][] }[] }).cases;
const witnesses = [
    ...["क्ष", "र्कि", "कि", "प्र", "कृ", "किं", "ि"].map((text, i) => ({
        id: `deva-${i}`,
        font: "NotoSansDevanagari-Regular.ttf" as FontFile,
        text,
        records: deva.find((row) => row.text === text)!.records!,
    })),
    { id: "inter-c1", font: "Inter.ttf" as FontFile, text: "A\u0326", records: reviewedRecordByInput("Inter.ttf", "A\u0326", "latn", "ltr")!.records },
    ...["ج", "كتب", "بَّ"].map((text, i) => ({
        id: `arabic-${i}`,
        font: "NotoSansArabic-Regular.ttf" as FontFile,
        text,
        records: arabic.find((row) => row.text === text)!.records,
    })),
];

describe.runIf(process.env.LITE_RENDER_WITNESS === "1")("authorized independent resident rendered placements", () => {
    const width = 512,
        height = 192,
        rowBytes = width * 4;
    const out = process.env.LITE_RENDER_WITNESS_OUT ?? join(tmpdir(), "lite-rendered-placement");
    const metrics: object[] = [];
    let native: NativeHarness;
    beforeAll(async () => {
        mkdirSync(out);
        native = await nativeHarness();
    });
    afterEach(async () => {
        vi.restoreAllMocks();
        await assertNativeClean(native);
    });
    afterAll(async () => {
        await native?.close();
    });

    async function readBuffer(buffer: GPUBuffer, bytes: number): Promise<Uint8Array> {
        const device = native.device;
        const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]);
        try {
            await staging.mapAsync(GPUMapMode.READ);
            return new Uint8Array(staging.getMappedRange().slice(0));
        } finally {
            staging.unmap();
            staging.destroy();
        }
    }
    async function pixels(texture: GPUTexture): Promise<Uint8Array> {
        const device = native.device;
        const staging = device.createBuffer({ size: rowBytes * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: rowBytes }, [width, height]);
        device.queue.submit([encoder.finish()]);
        try {
            await staging.mapAsync(GPUMapMode.READ);
            return new Uint8Array(staging.getMappedRange().slice(0));
        } finally {
            staging.unmap();
            staging.destroy();
        }
    }

    it.each(witnesses)("$id: bit-identical independent instances and pixels", async ({ id, font, text, records }) => {
        const face = fixture(font),
            device = native.device,
            storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, face.font, storage, id, { fontSizePx: 48 });
        let cpu: ReturnType<typeof createTextData> | undefined;
        const draws: ReturnType<ReturnType<typeof createTextLayer>["_createDraw"]>[] = [];
        const textures: GPUTexture[] = [];
        const uniform = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const pipeline = getOrCreateTextPipeline(native.engine, "rgba8unorm", 1, null, false);
        try {
            await (
                await updateComputeTextData(data, [text])
            ).completed;
            // Expected positions come ONLY from the independent records. Neither
            // the shaper output nor GPU pack values construct this CPU layer.
            const scale = Math.fround(48 / face.font.unitsPerEm);
            let penX = 0,
                penY = 0;
            const glyphs = records.map(([glyphId, , xAdvance, yAdvance, xOffset, yOffset]) => {
                const placed = { glyphId: glyphId!, x: Math.fround(Math.fround(penX + xOffset!) * scale), y: Math.fround(Math.fround(penY + yOffset!) * scale) };
                penX = Math.fround(penX + xAdvance!);
                penY = Math.fround(penY + yAdvance!);
                return placed;
            });
            cpu = createTextData(storage, [{ curveSet: id, glyphs, pixelsPerFontUnit: 48 / face.font.unitsPerEm }]);
            const original = device.createBuffer.bind(device);
            const spy = vi.spyOn(device, "createBuffer").mockImplementation((descriptor) =>
                original({
                    ...descriptor,
                    usage: descriptor.usage | (descriptor.label === "text-layer-instances" ? GPUBufferUsage.COPY_SRC : 0),
                })
            );
            const cpuDraw = createTextLayer(cpu)._createDraw(native.engine, uniform) as CpuTextLayerDraw;
            spy.mockRestore();
            draws.push(createComputeTextLayer(data)._createDraw(native.engine, uniform), cpuDraw);
            const ubo = new Float32Array(24);
            ubo[0] = 2 / width;
            ubo[5] = 2 / height;
            ubo[10] = 1;
            ubo[12] = 64 / width - 1;
            ubo[13] = 1 - 240 / height;
            ubo[15] = 1;
            ubo[16] = width;
            ubo[17] = height;
            ubo.set([1, 1, 1, 1], 20);
            device.queue.writeBuffer(uniform, 0, ubo);
            for (const draw of draws) {
                draw._prepare(pipeline._cache._bindGroupLayout);
                const texture = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
                textures.push(texture);
                const encoder = device.createCommandEncoder();
                const bundle = device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
                bundle.setPipeline(pipeline._pipeline);
                expect(draw._active).toBe(true);
                expect(draw._record(bundle, pipeline._cache._quadVertexBuffer, pipeline._pipeline, pipeline._variantPipeline)).toBe(1);
                const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
                pass.executeBundles([bundle.finish()]);
                pass.end();
                device.queue.submit([encoder.finish()]);
            }
            const indirect = new Uint32Array(await readStorageBuffer(data._indirect)),
                visible = indirect[1]!;
            expect(visible).toBeGreaterThan(0);
            expect(cpu._groups.reduce((sum, group) => sum + group._liveCount, 0)).toBe(visible);
            const gpuInstances = new Uint8Array(await readStorageBuffer(data._instances)).slice(0, visible * 12);
            const cpuUploaded = await readBuffer(cpuDraw._instanceBuf, cpu._instanceCount * 12);
            // TextData retains dead allocator slots (documented 0xffffffff
            // sentinel); resident pack compacts live instances. Compare the
            // actual uploaded live slots, not allocator capacity or dead quads.
            const cpuWords = new Uint32Array(cpuUploaded.buffer, cpuUploaded.byteOffset, cpuUploaded.byteLength / 4);
            const liveSlots = Array.from({ length: cpu._instanceCount }, (_, i) => i).filter((i) => cpuWords[i * 3 + 2] !== 0xffffffff);
            expect(liveSlots.length).toBe(visible);
            const cpuInstances = new Uint8Array(liveSlots.flatMap((slot) => [...cpuUploaded.slice(slot * 12, slot * 12 + 12)]));
            expect([...gpuInstances], `first differing instance byte for ${text}`).toEqual([...cpuInstances]);
            const gpu = await pixels(textures[0]!),
                reference = await pixels(textures[1]!);
            let differences = 0,
                absolute = 0,
                max = 0,
                ink = 0;
            for (let i = 0; i < gpu.length; i++) {
                const delta = Math.abs(gpu[i]! - reference[i]!);
                if (delta) {
                    differences++;
                }
                absolute += delta;
                max = Math.max(max, delta);
                if (i % 4 === 3 && gpu[i]! > 0) {
                    ink++;
                }
            }
            metrics.push({
                id,
                font,
                text,
                visible,
                cpuDeadSlots: cpu._instanceCount - visible,
                instanceBytes: visible * 12,
                pixelBytes: gpu.length,
                differingChannels: differences,
                mad: absolute / gpu.length,
                max,
                ink,
            });
            writeFileSync(join(out, "metrics.json"), JSON.stringify(metrics, null, 2) + "\n");
            const pair = new PNG({ width: width * 2, height });
            for (let y = 0; y < height; y++) {
                pair.data.set(gpu.subarray(y * rowBytes, (y + 1) * rowBytes), y * rowBytes * 2);
                pair.data.set(reference.subarray(y * rowBytes, (y + 1) * rowBytes), y * rowBytes * 2 + rowBytes);
            }
            writeFileSync(join(out, `${id}-gpu.rgba`), gpu);
            writeFileSync(join(out, `${id}-reference.rgba`), reference);
            // Review images flatten the framebuffer's premultiplied RGB onto
            // black. Raw alpha and all equality metrics remain unmodified.
            for (let i = 3; i < pair.data.length; i += 4) {
                pair.data[i] = 255;
            }
            const png = join(out, `${id}-gpu-reference.png`),
                jpg = join(out, `${id}-gpu-reference.jpg`);
            writeFileSync(png, PNG.sync.write(pair));
            const conversion = spawnSync("/usr/bin/sips", ["-s", "format", "jpeg", "-s", "formatOptions", "55", png, "--out", jpg], { encoding: "utf8" });
            expect(conversion.status, conversion.stderr).toBe(0);
            expect(statSync(jpg).size).toBeLessThan(1024 * 1024);
            expect(ink).toBeGreaterThan(0);
            expect(differences).toBe(0);
            expect(max).toBe(0);
            expect(absolute).toBe(0);
        } finally {
            await device.queue.onSubmittedWorkDone();
            draws.forEach((draw) => draw._dispose());
            textures.forEach((texture) => texture.destroy());
            uniform.destroy();
            if (cpu) {
                disposeTextData(cpu);
            }
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
            pipeline._cache._quadVertexBuffer.destroy();
            clearTextPipelineCache(native.engine);
        }
    });
});
