/** Timed workload page; run only with approval (see compute-text-shaping-core.md). */
import {
    createEngine,
    disposeEngine,
    createComputeFontFromBuffer,
    createGlyphStorage,
    disposeGlyphStorage,
    createComputeTextData,
    updateComputeTextData,
    createComputeTextLayer,
    disposeComputeTextData,
    createTextRenderer,
    registerTextRenderer,
    disposeTextRenderer,
    startEngine,
    readStorageBuffer,
} from "babylon-lite";

const quantile = (values: number[], q: number) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) * q)]!;
const statistics = (values: number[]) => ({ median: quantile(values, 0.5), p10: quantile(values, 0.1), p90: quantile(values, 0.9) });

export async function runComputeTextWorkload() {
    const query = new URLSearchParams(location.search);
    const script = query.get("script") === "arab" ? "arab" : query.get("script") === "latn" ? "latn" : "deva";
    const arabic = script === "arab",
        latin = script === "latn",
        single = query.get("workload") === "single";
    if (arabic && single) {
        throw new Error("The single-run qualification workload is Hindi; Arabic is the optional block mode.");
    }
    const fontFile = arabic ? "NotoSansArabic-Regular.ttf" : latin ? "Roboto-Regular.ttf" : "NotoSansDevanagari-Regular.ttf";
    const a = arabic
        ? "كتب ج بَّ كتب ج بَّ كتب ج بَّ كتب ج بَّ كتب ج بَّ كتب ج بَّ"
        : latin
          ? "Hello world. Plain English text written for the compute text bench."
          : "नमस्ते दुनिया। हिन्दी में लिखा हुआ पाठ नमस्ते दुनिया नमस्ते दुनिया।";
    const b = arabic ? a.replace("كتب", "ج") : latin ? a.replace("world", "earth") : a.replace("दुनिया", "हिन्दी");
    const long = (text: string) => {
        const scalars = [...text.repeat(Math.ceil(2000 / [...text].length) + 1)].slice(0, 2000);
        const boundary = scalars.lastIndexOf(" ");
        return scalars
            .slice(0, boundary + 1)
            .join("")
            .padEnd(2000, " ");
    };
    const variants = single ? [long(a), long(b)] : [a, b];
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const fontStarted = performance.now();
    const response = await fetch(`/fonts/${fontFile}`);
    if (!response.ok) {
        throw new Error(`Font request failed: ${response.status} ${fontFile}`);
    }
    const fontBytes = await response.arrayBuffer();
    const fontIdentity = [...new Uint8Array(await crypto.subtle.digest("SHA-256", fontBytes))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    const expectedIdentity = arabic
        ? "7ed3fe069312aceac454f17cf613a30f95271d6ed7ce58005ed4d016bd3823d7"
        : latin
          ? "56a45233d29f11b4dfb86d248e921939d115778f87325e7ae8cc108383d6664d"
          : "9c7d935139ea6a1e6ad9dbac4f6d27ece1e04bca8123c8888d00a0f9df4724cd";
    if (fontIdentity !== expectedIdentity) {
        throw new Error(`Unexpected workload font identity: ${fontIdentity}`);
    }
    const font = createComputeFontFromBuffer(fontBytes),
        fontLoadMs = performance.now() - fontStarted;
    const engine = await createEngine(canvas),
        storage = createGlyphStorage();
    const blocks = Array.from({ length: single ? 1 : 100 }, () => createComputeTextData(engine, font, storage, `bench-${fontFile}`, { fontSizePx: 32 }));
    const layers = blocks.map((data, i) => createComputeTextLayer(data, { positionPx: { x: 20 + (i % 10) * 155, y: 60 + Math.floor(i / 10) * 70 } }));
    const renderer = createTextRenderer(engine.surfaces[0], { layers });
    registerTextRenderer(renderer);
    let running = true;
    try {
        await startEngine(engine);
        // Render-task GPU timing only covers frame-graph tasks; compute text work is fused into the
        // frame submit outside them. Observe queue submits instead: first submit to all-work-done.
        const queue = engine._device.queue,
            submit = queue.submit.bind(queue);
        let submits: { at: number; done: Promise<number> }[] = [];
        queue.submit = (buffers: Iterable<GPUCommandBuffer>) => {
            const at = performance.now();
            submit(buffers);
            submits.push({ at, done: queue.onSubmittedWorkDone().then(() => performance.now()) });
        };
        const firstStarted = performance.now();
        await Promise.all(blocks.map(async (data) => (await updateComputeTextData(data, [variants[0]!])).completed));
        const firstUseMs = performance.now() - firstStarted;
        let resolution = Infinity,
            previous = performance.now();
        for (let i = 0; i < 1000; i++) {
            const now = performance.now();
            if (now > previous) {
                resolution = Math.min(resolution, now - previous);
            }
            previous = now;
        }
        const synchronousAdmissionMs: number[] = [],
            completionLatencyMs: number[] = [],
            submitToGpuDoneMs: number[] = [],
            submitsPerFrame: number[] = [];
        for (let frame = 0; frame < 150 && running; frame++) {
            await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            let active = 0;
            submits = [];
            const began = performance.now();
            const admitted = blocks.map((data, i) => {
                const start = performance.now();
                const promise = updateComputeTextData(data, [variants[(frame + i) % 2]!]);
                active += performance.now() - start;
                return promise;
            });
            const updates = await Promise.all(admitted);
            await Promise.all(updates.map((update) => update.completed));
            const completedAt = performance.now();
            const frameSubmits = submits;
            const lastDone = frameSubmits.length ? await frameSubmits[frameSubmits.length - 1]!.done : NaN;
            if (frame >= 30) {
                synchronousAdmissionMs.push(active);
                completionLatencyMs.push(completedAt - began);
                submitToGpuDoneMs.push(lastDone - frameSubmits[0]!.at);
                submitsPerFrame.push(frameSubmits.length);
            }
        }
        // Diagnostic validation occurs outside all measured intervals.
        const indirect = await Promise.all(blocks.map(async (data) => [...new Uint32Array(await readStorageBuffer(data._indirect))]));
        if (indirect.some((words) => words[0] !== 6 || words[1]! <= 0) || blocks.some((data) => data.status !== "ready")) {
            throw new Error("Workload produced invalid status or no resident instances; do not publish timings.");
        }
        queue.submit = submit;
        const result = {
            script,
            workload: single ? "single-2000-scalars" : "100-independent-blocks",
            fontFile,
            fontIdentity,
            blocks: blocks.length,
            scalarsPerVariant: variants.map((text) => [...text].length),
            texts: variants,
            warmupFrames: 30,
            sampleFrames: synchronousAdmissionMs.length,
            fontLoadMs,
            firstUseMs,
            synchronousAdmissionMs: statistics(synchronousAdmissionMs),
            completionLatencyMs: statistics(completionLatencyMs),
            mainThreadEndpoint: "Sum of synchronous updateComputeTextData call segments; excludes awaited continuations, not total async self CPU time.",
            completionEndpoint: "From first admission call until every submitted update.completed resolves.",
            clock: { crossOriginIsolated, observedResolutionMs: Number.isFinite(resolution) ? resolution : null, coarseClockWarning: !crossOriginIsolated },
            submitToGpuDoneMs: statistics(submitToGpuDoneMs),
            submitsPerFrame: statistics(submitsPerFrame),
            gpuEndpoint:
                "From the first queue.submit after admission to onSubmittedWorkDone of the last submit before completion; includes the frame's render pass and readback copies (compare against the Latin control).",
            validStatuses: true,
            visibleInstances: indirect.map((words) => words[1]),
            comparison: arabic
                ? "Absolute optional C9 block workload; RTL segmentation is not claimed."
                : latin
                  ? "Latin control with the same protocol and block shape."
                  : "Candidate absolute workload only: baseline 27c6ecb3 does not support Devanagari; no B/A ratio.",
            segmentation: arabic ? "CTF-14 whole RTL run" : latin ? "Latin segmentation as admitted" : "CTF-16 whole run; independent blocks are parallel",
        };
        (document.getElementById("result") as HTMLElement).textContent = JSON.stringify(result, null, 2);
        return result;
    } finally {
        running = false;
        disposeTextRenderer(renderer);
        blocks.forEach(disposeComputeTextData);
        disposeGlyphStorage(storage);
        disposeEngine(engine);
    }
}

declare global {
    interface Window {
        runComputeTextWorkload: typeof runComputeTextWorkload;
    }
}
window.runComputeTextWorkload = runComputeTextWorkload;
document.getElementById("run")!.addEventListener("click", () => {
    void runComputeTextWorkload().catch((error: unknown) => {
        document.getElementById("result")!.textContent = String(error);
    });
});
