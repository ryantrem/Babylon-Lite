import { afterEach, describe, expect, it, vi } from "vitest";
import {
    createComputeTextData,
    createGlyphStorage,
    disposeComputeTextData,
    disposeGlyphStorage,
    shapeComputeText,
    updateComputeTextData,
} from "../../../packages/babylon-lite/src/index";
import { createComputeTextShaper, disposeComputeTextShaper, shapeComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-shaper";
import { createComputeTextPlan } from "../../../packages/babylon-lite/src/text/compute-text-plan";
import { selectComputeTextRun } from "../../../packages/babylon-lite/src/text/compute-text-selection";
import { compare, fixture } from "./fixtures";
import { assertNativeClean, nativeHarness } from "./native";
import type { NativeHarness } from "./native";

function pipelineStarted(device: GPUDevice): Promise<void> {
    let notify: () => void = () => undefined;
    const started = new Promise<void>((resolve) => {
        notify = resolve;
    });
    const original = device.createComputePipelineAsync.bind(device);
    vi.spyOn(device, "createComputePipelineAsync").mockImplementation((descriptor) => {
        notify();
        return original(descriptor);
    });
    return started;
}

describe("native compute lifecycle and independently owned engines", () => {
    const engines: NativeHarness[] = [];
    async function engine(): Promise<NativeHarness> {
        const native = await nativeHarness();
        engines.push(native);
        return native;
    }
    afterEach(async () => {
        vi.restoreAllMocks();
        for (const native of engines.splice(0)) {
            await assertNativeClean(native);
            await native.close();
        }
    });

    it("busy rejection, disposal during an actual pipeline request, and fresh shaper recreation", async () => {
        const native = await engine();
        const font = fixture("Roboto-Regular.ttf");
        const plan = createComputeTextPlan(font.font, selectComputeTextRun("office"));
        const shaper = createComputeTextShaper(native.engine);
        const started = pipelineStarted(native.device);
        const pending = shapeComputeTextPlan(shaper, plan, ["office"]);
        const rejected = expect(pending).rejects.toThrow(/disposed/);
        await started;
        expect(shaper.status).toBe("busy");
        await expect(shapeComputeTextPlan(shaper, plan, ["A"])).rejects.toThrow(/busy/);
        disposeComputeTextShaper(shaper);
        disposeComputeTextShaper(shaper);
        await rejected;
        await expect(shapeComputeTextPlan(shaper, plan, ["A"])).rejects.toThrow(/disposed/);
        compare(font, "office", await shapeComputeText(native.engine, font.font, "office"));
    });

    it("device.destroy resolves real device.lost during pending work; independent engines survive", async () => {
        const lost = await engine();
        const surviving = await engine();
        const font = fixture("NotoSansGothic-Regular.ttf");
        const started = pipelineStarted(lost.device);
        const pending = shapeComputeText(lost.engine, font.font, "𐌰𐌱");
        const rejected = expect(pending).rejects.toThrow(/lost|destroyed/i);
        await started;
        lost.destroyExpected = true;
        lost.device.destroy();
        expect((await lost.device.lost).reason).toBe("destroyed");
        await rejected;
        compare(font, "𐌰𐌱", await shapeComputeText(surviving.engine, font.font, "𐌰𐌱"));
        const fresh = await engine();
        const results = await Promise.all([shapeComputeText(fresh.engine, font.font, "𐌰𐌱"), shapeComputeText(surviving.engine, font.font, "𐌰𐌱")]);
        results.forEach((result) => compare(font, "𐌰𐌱", result));
    });

    it("resident busy, disposal and survivor ownership while the real pipeline is pending", async () => {
        const native = await engine();
        const font = fixture("Roboto-Regular.ttf");
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "roboto");
        const survivor = createComputeTextData(native.engine, font.font, storage, "roboto");
        try {
            const started = pipelineStarted(native.device);
            const pending = updateComputeTextData(data, ["office"]);
            const rejected = expect(pending).rejects.toThrow(/disposed/);
            await started;
            await expect(updateComputeTextData(data, ["A"])).rejects.toThrow(/busy/);
            disposeComputeTextData(data);
            await rejected;
            await expect(updateComputeTextData(data, ["A"])).rejects.toThrow(/disposed/);
            await updateComputeTextData(survivor, ["A"]);
            expect(survivor.status).toBe("ready");
        } finally {
            disposeComputeTextData(data);
            disposeComputeTextData(survivor);
            disposeGlyphStorage(storage);
        }
    });

    it("resident device.lost is terminal during actual pipeline preparation", async () => {
        const native = await engine();
        const font = fixture("NotoSansGothic-Regular.ttf");
        const storage = createGlyphStorage();
        const data = createComputeTextData(native.engine, font.font, storage, "gothic");
        try {
            const started = pipelineStarted(native.device);
            const pending = updateComputeTextData(data, ["𐌰"]);
            const rejected = expect(pending).rejects.toThrow(/lost/i);
            await started;
            native.destroyExpected = true;
            native.device.destroy();
            expect((await native.device.lost).reason).toBe("destroyed");
            await rejected;
            expect(data.status).toBe("lost");
            expect(data._indirect._destroyed).toBe(true);
            await expect(updateComputeTextData(data, ["𐌰"])).rejects.toThrow(/lost/i);
        } finally {
            disposeComputeTextData(data);
            disposeGlyphStorage(storage);
        }
    });
});
