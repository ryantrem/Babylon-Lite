/** Timed workload page; run only with approval (see compute-text-shaping-core.md). */
import {
    createEngine,
    disposeEngine,
    createComputeFontFromBuffer,
    createGlyphStorage,
    disposeGlyphStorage,
    createComputeTextData,
    updateComputeTextData,
    updateComputeTextParagraphs,
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

interface PassSample {
    readonly frame: number;
    readonly key: string;
    readonly ns: number;
}

interface PassProfiler {
    frame: number;
    readonly samples: PassSample[];
    readonly submitted: { frame: number; encoders: string[] }[];
    readonly pending: Promise<void>[];
    onSubmit(buffers: readonly GPUCommandBuffer[]): void;
    uninstall(): void;
}

const PASS_QUERY_CAPACITY = 128;

/** Diagnostic only: per-pass GPU timestamps, keyed by encoder label, pass label and pipeline labels. */
function installPassProfiler(device: GPUDevice): PassProfiler {
    interface Slot {
        readonly querySet: GPUQuerySet;
        readonly resolve: GPUBuffer;
        readonly read: GPUBuffer;
    }
    interface Recorded {
        readonly encoder: string;
        readonly keys: string[];
        readonly slot: Slot | null;
    }
    const pool: Slot[] = [];
    const byBuffer = new WeakMap<GPUCommandBuffer, Recorded>();
    const createCommandEncoder = device.createCommandEncoder.bind(device);
    const take = (): Slot =>
        pool.pop() ?? {
            querySet: device.createQuerySet({ type: "timestamp", count: PASS_QUERY_CAPACITY }),
            resolve: device.createBuffer({ size: PASS_QUERY_CAPACITY * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
            read: device.createBuffer({ size: PASS_QUERY_CAPACITY * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        };
    const profiler: PassProfiler = {
        frame: -1,
        samples: [],
        submitted: [],
        pending: [],
        onSubmit(buffers) {
            const frame = profiler.frame;
            profiler.submitted.push({ frame, encoders: buffers.map((buffer) => byBuffer.get(buffer)?.encoder ?? "(unlabeled)") });
            for (const buffer of buffers) {
                const recorded = byBuffer.get(buffer);
                const slot = recorded?.slot;
                if (!recorded || !slot) {
                    continue;
                }
                profiler.pending.push(
                    slot.read.mapAsync(GPUMapMode.READ).then(() => {
                        const times = new BigUint64Array(slot.read.getMappedRange(0, recorded.keys.length * 16));
                        recorded.keys.forEach((key, i) => {
                            const ns = Number(times[i * 2 + 1]! - times[i * 2]!);
                            profiler.samples.push({ frame, key: `${recorded.encoder} > ${key}`, ns });
                        });
                        slot.read.unmap();
                        pool.push(slot);
                    })
                );
            }
        },
        uninstall() {
            device.createCommandEncoder = createCommandEncoder;
        },
    };
    device.createCommandEncoder = (descriptor?: GPUCommandEncoderDescriptor) => {
        const encoder = createCommandEncoder(descriptor);
        const label = descriptor?.label || "(unlabeled)";
        const keys: string[] = [];
        let slot: Slot | null = null;
        const timestamps = () => {
            if (keys.length * 2 >= PASS_QUERY_CAPACITY) {
                return undefined;
            }
            slot ??= take();
            return { querySet: slot.querySet, beginningOfPassWriteIndex: keys.length * 2, endOfPassWriteIndex: keys.length * 2 + 1 };
        };
        const beginComputePass = encoder.beginComputePass.bind(encoder);
        encoder.beginComputePass = (pass?: GPUComputePassDescriptor) => {
            const timestampWrites = pass?.timestampWrites ? undefined : timestamps();
            if (!timestampWrites) {
                return beginComputePass(pass);
            }
            const index = keys.length;
            keys.push(`compute:${pass?.label || "(unlabeled)"}`);
            const computePass = beginComputePass({ ...pass, timestampWrites });
            const setPipeline = computePass.setPipeline.bind(computePass);
            const pipelines: string[] = [];
            computePass.setPipeline = (pipeline: GPUComputePipeline) => {
                if (!pipelines.includes(pipeline.label)) {
                    pipelines.push(pipeline.label);
                    keys[index] = `compute:${pass?.label || "(unlabeled)"} [${pipelines.join(" + ")}]`;
                }
                setPipeline(pipeline);
            };
            return computePass;
        };
        const beginRenderPass = encoder.beginRenderPass.bind(encoder);
        encoder.beginRenderPass = (pass: GPURenderPassDescriptor) => {
            const timestampWrites = pass.timestampWrites ? undefined : timestamps();
            if (!timestampWrites) {
                return beginRenderPass(pass);
            }
            keys.push(`render:${pass.label || "(unlabeled)"}`);
            return beginRenderPass({ ...pass, timestampWrites });
        };
        const finish = encoder.finish.bind(encoder);
        encoder.finish = (finishDescriptor?: GPUCommandBufferDescriptor) => {
            if (slot && keys.length) {
                encoder.resolveQuerySet(slot.querySet, 0, keys.length * 2, slot.resolve, 0);
                encoder.copyBufferToBuffer(slot.resolve, 0, slot.read, 0, keys.length * 16);
            }
            const buffer = finish(finishDescriptor);
            byBuffer.set(buffer, { encoder: label, keys, slot });
            return buffer;
        };
        return encoder;
    };
    return profiler;
}

function summarizePassProfile(profiler: PassProfiler, firstFrame: number) {
    const frames = new Set<number>();
    const byKey = new Map<string, Map<number, { ns: number; count: number }>>();
    for (const sample of profiler.samples) {
        if (sample.frame < firstFrame) {
            continue;
        }
        frames.add(sample.frame);
        const perFrame = byKey.get(sample.key) ?? new Map<number, { ns: number; count: number }>();
        byKey.set(sample.key, perFrame);
        const entry = perFrame.get(sample.frame) ?? { ns: 0, count: 0 };
        entry.ns += sample.ns;
        entry.count++;
        perFrame.set(sample.frame, entry);
    }
    const frameList = [...frames];
    const totals = frameList.map((frame) => [...byKey.values()].reduce((sum, perFrame) => sum + (perFrame.get(frame)?.ns ?? 0), 0) / 1e6);
    const passes = [...byKey.entries()]
        .map(([key, perFrame]) => {
            const ms = frameList.map((frame) => (perFrame.get(frame)?.ns ?? 0) / 1e6);
            const counts = frameList.map((frame) => perFrame.get(frame)?.count ?? 0);
            return { key, msPerFrame: statistics(ms), passesPerFrame: quantile(counts, 0.5) };
        })
        .sort((left, right) => right.msPerFrame.median - left.msPerFrame.median);
    const allNs = profiler.samples.map((sample) => sample.ns).filter((ns) => ns > 0);
    const submitShapes = new Map<string, number>();
    for (const submit of profiler.submitted) {
        if (submit.frame >= firstFrame) {
            const shape = submit.encoders.join(", ");
            submitShapes.set(shape, (submitShapes.get(shape) ?? 0) + 1);
        }
    }
    return {
        frames: frameList.length,
        summedPassMsPerFrame: statistics(totals),
        passes,
        submitsBySampledFrameCount: Object.fromEntries([...submitShapes.entries()].map(([shape, count]) => [shape, count / Math.max(1, frameList.length)])),
        quantizationCheck: { positiveSamples: allNs.length, multiplesOf100us: allNs.filter((ns) => ns % 100_000 === 0).length },
        endpoint: "GPU timestamp begin/end of each pass, summed per sampled frame. Profiling adds resolve/copy work, so do not compare this mode's latencies with unprofiled runs.",
    };
}

const WRAP_TEXT = "भारत एक विशाल देश है। यहाँ कर्म और धर्म का प्रकाश है। किताब और विज्ञान की शक्ति से स्त्री पुरुष सब मिलकर सुंदर भविष्य बनाएँगे।";
const WRAP_LINE_HEIGHT_PX = 48;

/** CTF-03 witness (`?wrap=1&widths=320,640`): one Hindi paragraph with a conjunct, reph, pre-base ि and danda, wrapped at each width and left rendering. Untimed. */
async function runWrappedParagraphWitness(query: URLSearchParams) {
    const widths = (query.get("widths") ?? "320,640").split(",").map(Number);
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const response = await fetch("/fonts/NotoSansDevanagari-Regular.ttf");
    if (!response.ok) {
        throw new Error(`Font request failed: ${response.status}`);
    }
    const font = createComputeFontFromBuffer(await response.arrayBuffer());
    const engine = await createEngine(canvas),
        storage = createGlyphStorage();
    const blocks = widths.map(() => createComputeTextData(engine, font, storage, "bench-wrap", { fontSizePx: 32 }));
    const lineCounts: number[] = [];
    let error: string | null = null;
    try {
        for (const [i, data] of blocks.entries()) {
            await (
                await updateComputeTextParagraphs(data, [WRAP_TEXT], { maxWidthPx: widths[i]!, lineHeightPx: WRAP_LINE_HEIGHT_PX })
            ).completed;
            // Instances are x, y, slot words; glyph y offsets stay well inside one line height of their baseline.
            const visible = new Uint32Array(await readStorageBuffer(data._indirect))[1]!;
            const instances = new Float32Array(await readStorageBuffer(data._instances, 0, visible * 12));
            lineCounts.push(new Set(Array.from({ length: visible }, (_, g) => Math.round(instances[g * 3 + 1]! / WRAP_LINE_HEIGHT_PX))).size);
        }
    } catch (caught) {
        error = caught instanceof Error ? caught.message : String(caught);
    }
    let y = 60;
    const layers = blocks.map((data, i) => {
        const layer = createComputeTextLayer(data, { positionPx: { x: 20, y } });
        y += (lineCounts[i] ?? 1) * WRAP_LINE_HEIGHT_PX + 40;
        return layer;
    });
    registerTextRenderer(createTextRenderer(engine.surfaces[0], { layers }));
    await startEngine(engine);
    const result = { mode: "wrap", text: WRAP_TEXT, widths, lineCounts, error };
    (document.getElementById("result") as HTMLElement).textContent = JSON.stringify(result, null, 2);
    return result;
}

/** `workload=paragraph`: one ~2000-scalar paragraph at 32 px; wall clock from `updateComputeTextParagraphs` to `completed`. */
async function runParagraphWorkload(script: "deva" | "latn") {
    const latin = script === "latn";
    const fontFile = latin ? "Roboto-Regular.ttf" : "NotoSansDevanagari-Regular.ttf";
    const sentence = latin ? "Plain English text written for the compute text bench wraps across many lines of a long paragraph." : WRAP_TEXT;
    const words = sentence.repeat(Math.ceil(2000 / [...sentence].length) + 1).split(" ");
    while ([...words.join(" ")].length > 2000) {
        words.pop();
    }
    const middle = words.length >> 1;
    const texts = [words.join(" "), words.map((word, i) => (i === middle ? (latin ? "planet" : "किताब") : word)).join(" ")];
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const response = await fetch(`/fonts/${fontFile}`);
    if (!response.ok) {
        throw new Error(`Font request failed: ${response.status} ${fontFile}`);
    }
    const font = createComputeFontFromBuffer(await response.arrayBuffer());
    const engine = await createEngine(canvas),
        storage = createGlyphStorage();
    const options = (maxWidthPx: number) => ({ maxWidthPx, lineHeightPx: WRAP_LINE_HEIGHT_PX });
    const timed = async (data: ReturnType<typeof createComputeTextData>, text: string, maxWidthPx: number) => {
        const start = performance.now();
        await (
            await updateComputeTextParagraphs(data, [text], options(maxWidthPx))
        ).completed;
        return performance.now() - start;
    };
    const fresh = () => createComputeTextData(engine, font, storage, `bench-paragraph-${fontFile}`, { fontSizePx: 32 });
    try {
        const coldData = fresh();
        const coldMs = await timed(coldData, texts[0]!, 640);
        disposeComputeTextData(coldData);
        const initialMs: number[] = [];
        for (let i = 0; i < 7; i++) {
            const data = fresh();
            initialMs.push(await timed(data, texts[0]!, 640));
            disposeComputeTextData(data);
        }
        const data = fresh();
        try {
            await timed(data, texts[0]!, 640);
            const editMs: number[] = [],
                widthMs: number[] = [];
            for (let i = 1; i <= 10; i++) {
                editMs.push(await timed(data, texts[i % 2]!, 640));
            }
            for (let i = 1; i <= 10; i++) {
                widthMs.push(await timed(data, texts[0]!, i % 2 ? 480 : 640));
            }
            const visible = new Uint32Array(await readStorageBuffer(data._indirect))[1]!;
            if (data.status !== "ready" || visible <= 0) {
                throw new Error("Paragraph workload produced invalid status or no resident instances; do not publish timings.");
            }
            const result = {
                script,
                workload: "paragraph-2000-scalars",
                fontFile,
                scalars: texts.map((text) => [...text].length),
                coldMs,
                initialMs: statistics(initialMs),
                editMs: statistics(editMs),
                widthMs: statistics(widthMs),
                visibleInstances: visible,
                endpoint:
                    "Wall clock from updateComputeTextParagraphs to completed; initial uses fresh data (n=7), edit swaps one middle word (n=10), width alternates 480/640 (n=10).",
            };
            (document.getElementById("result") as HTMLElement).textContent = JSON.stringify(result, null, 2);
            return result;
        } finally {
            disposeComputeTextData(data);
        }
    } finally {
        disposeGlyphStorage(storage);
        disposeEngine(engine);
    }
}

export async function runComputeTextWorkload() {
    const query = new URLSearchParams(location.search);
    if (query.has("wrap")) {
        return runWrappedParagraphWitness(query);
    }
    if (query.get("workload") === "paragraph") {
        if (query.get("script") === "arab") {
            throw new Error("Arabic paragraphs are not supported.");
        }
        return runParagraphWorkload(query.get("script") === "latn" ? "latn" : "deva");
    }
    const profilePasses = query.get("profile") === "passes";
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
        if (profilePasses && !engine._device.features.has("timestamp-query")) {
            throw new Error("Pass profiling requires the WebGPU timestamp-query feature.");
        }
        const profiler = profilePasses ? installPassProfiler(engine._device) : null;
        const queue = engine._device.queue,
            submit = queue.submit.bind(queue);
        let submits: { at: number; done: Promise<number> }[] = [];
        queue.submit = (buffers: Iterable<GPUCommandBuffer>) => {
            const list = [...buffers];
            const at = performance.now();
            submit(list);
            submits.push({ at, done: queue.onSubmittedWorkDone().then(() => performance.now()) });
            profiler?.onSubmit(list);
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
            if (frame === 30) {
                await window.__benchCpuProfile?.("start");
            }
            await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            let active = 0;
            submits = [];
            if (profiler) {
                profiler.frame = frame;
            }
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
        await window.__benchCpuProfile?.("stop");
        // Diagnostic validation occurs outside all measured intervals.
        const indirect = await Promise.all(blocks.map(async (data) => [...new Uint32Array(await readStorageBuffer(data._indirect))]));
        if (indirect.some((words) => words[0] !== 6 || words[1]! <= 0) || blocks.some((data) => data.status !== "ready")) {
            throw new Error("Workload produced invalid status or no resident instances; do not publish timings.");
        }
        queue.submit = submit;
        let passProfile: ReturnType<typeof summarizePassProfile> | null = null;
        if (profiler) {
            profiler.frame = -1;
            await Promise.all(profiler.pending);
            profiler.uninstall();
            passProfile = summarizePassProfile(profiler, 30);
        }
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
            passProfile,
            segmentation: arabic ? "CTF-14 whole RTL run" : latin ? "Latin segmentation as admitted" : "CTF-16 runs split at proven spaces; independent blocks are parallel",
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
        /** Installed by the runner only for `--cpu-profile`. */
        __benchCpuProfile?: (phase: "start" | "stop") => Promise<void>;
    }
}
window.runComputeTextWorkload = runComputeTextWorkload;
document.getElementById("run")!.addEventListener("click", () => {
    void runComputeTextWorkload().catch((error: unknown) => {
        document.getElementById("result")!.textContent = String(error);
    });
});
