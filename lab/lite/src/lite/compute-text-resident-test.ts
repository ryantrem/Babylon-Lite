/** Real-WebGPU resident compute text. This entry does not import text-shaper and does not map instance bytes. */

import { createComputeTextData, createEngine, createGlyphStorage, disposeComputeTextData, loadComputeFont, updateComputeTextData } from "babylon-lite";

interface ComputeTextResidentResults {
    ready: boolean;
    error: string | null;
    robotoFiStatus: number | null;
    interAcStatus: number | null;
    sharedSurvived: boolean;
}

declare global {
    interface Window {
        __computeTextResidentTest: ComputeTextResidentResults;
    }
}

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const results: ComputeTextResidentResults = {
    ready: false,
    error: null,
    robotoFiStatus: null,
    interAcStatus: null,
    sharedSurvived: false,
};
window.__computeTextResidentTest = results;

async function run(): Promise<void> {
    const engine = await createEngine(canvas);
    try {
        const roboto = await loadComputeFont("/fonts/Roboto-Regular.ttf");
        const inter = await loadComputeFont("/fonts/Inter.ttf");
        const storage = createGlyphStorage();
        const robotoData = createComputeTextData(engine, roboto, storage, "roboto", { fontSizePx: 32 });
        const robotoAgain = createComputeTextData(engine, roboto, storage, "roboto", { fontSizePx: 32 });
        const interData = createComputeTextData(engine, inter, storage, "inter", { fontSizePx: 32 });
        await updateComputeTextData(robotoData, ["fi"]);
        results.robotoFiStatus = 0;
        disposeComputeTextData(robotoData);
        await updateComputeTextData(robotoAgain, ["fi"]);
        await updateComputeTextData(interData, ["AC"]);
        results.interAcStatus = 0;
        results.sharedSurvived = true;
    } catch (error) {
        results.error = error instanceof Error ? error.message : String(error);
    } finally {
        results.ready = true;
        canvas.dataset.ready = "true";
    }
}

void run();
