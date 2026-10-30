/** Timed workload runner; run only with approval. Bounded; Prints JSON; writes no repository files. */
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { writeFileSync } from "node:fs";
import { basename } from "node:path";
const args = process.argv.slice(2);
for (const arg of args) {
    if (!/^--script=(?:deva|arab|latn)$|^--workload=(?:blocks|single)$|^--base=https?:\/\/|^--profile=passes$|^--cpu-profile=\/.+\.cpuprofile$/.test(arg)) {
        throw new Error(`Unknown workload argument: ${arg}`);
    }
}
const script = args.includes("--script=arab") ? "arab" : args.includes("--script=latn") ? "latn" : "deva";
const workload = args.includes("--workload=single") ? "single" : "blocks";
const profilePasses = args.includes("--profile=passes");
const cpuProfilePath = args.find((arg) => arg.startsWith("--cpu-profile="))?.slice(14);
const base = new URL(args.find((arg) => arg.startsWith("--base="))?.slice(7) ?? "http://localhost:5187");
let server;
try {
    let ready = false;
    try {
        ready = (await fetch(base, { signal: AbortSignal.timeout(2000) })).ok;
    } catch {
        /* Server absence is expected; errors after launch are explicit. */
    }
    if (!ready) {
        server = spawn("pnpm", ["--filter", "@babylon-lite/lab", "dev"], {
            env: { ...process.env, LAB_DEV_PORT: base.port },
            stdio: ["ignore", "pipe", "inherit"],
        });
        server.stdout.pipe(process.stderr);
        for (let i = 0; i < 30; i++) {
            await delay(1000);
            try {
                ready = (await fetch(base, { signal: AbortSignal.timeout(2000) })).ok;
            } catch {
                /* Bounded startup polling only. */
            }
            if (ready) {
                break;
            }
            if (server.exitCode !== null) {
                throw new Error(`Owned Vite server exited ${server.exitCode}`);
            }
        }
        if (!ready) {
            throw new Error("Owned Vite server did not become ready");
        }
    }
    const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
    try {
        const context = await browser.newContext(),
            page = await context.newPage();
        const errors = [];
        page.on("pageerror", (error) => errors.push(error.message));
        let cpuProfile = null;
        if (cpuProfilePath) {
            const cdp = await context.newCDPSession(page);
            await cdp.send("Profiler.enable");
            await cdp.send("Profiler.setSamplingInterval", { interval: 100 });
            await page.exposeFunction("__benchCpuProfile", async (phase) => {
                if (phase === "start") {
                    await cdp.send("Profiler.start");
                } else {
                    cpuProfile = (await cdp.send("Profiler.stop")).profile;
                }
            });
        }
        const pageQuery = `script=${script}&workload=${workload}${profilePasses ? "&profile=passes" : ""}`;
        await page.goto(new URL(`/lite/compute-text-devanagari-bench.html?${pageQuery}`, base).href);
        const adapter = await page.evaluate(async () => {
            const value = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
            return value ? { vendor: value.info.vendor, architecture: value.info.architecture, device: value.info.device, description: value.info.description } : null;
        });
        const results = await page.evaluate(async () => window.runComputeTextWorkload());
        if (errors.length) {
            throw new Error(`Browser errors: ${errors.join("; ")}`);
        }
        let cpuSelfTime = null;
        if (cpuProfilePath) {
            if (!cpuProfile) {
                throw new Error("CPU profile was not captured");
            }
            writeFileSync(cpuProfilePath, JSON.stringify(cpuProfile));
            cpuSelfTime = summarizeCpuProfile(cpuProfile);
        }
        process.stdout.write(JSON.stringify({ browser: browser.version(), adapter, results, cpuSelfTime }, null, 2) + "\n");
        await context.close();
    } finally {
        await browser.close();
    }
} finally {
    if (server?.pid && server.exitCode === null) {
        server.kill("SIGTERM");
    }
}

/** Self time by function and by module over the sampled frames; diagnostic only. */
function summarizeCpuProfile(profile) {
    const nodes = new Map(profile.nodes.map((node) => [node.id, node]));
    const byFunction = new Map(),
        byModule = new Map();
    let totalUs = 0;
    profile.samples.forEach((id, i) => {
        const us = profile.timeDeltas[i] ?? 0;
        const frame = nodes.get(id).callFrame;
        const module = frame.url ? basename(frame.url.split("?")[0]) : `(${frame.functionName || "native"})`;
        const name = `${frame.functionName || "(anonymous)"} ${module}:${frame.lineNumber + 1}`;
        byFunction.set(name, (byFunction.get(name) ?? 0) + us);
        byModule.set(module, (byModule.get(module) ?? 0) + us);
        totalUs += us;
    });
    const top = (map, count) =>
        [...map.entries()]
            .sort((left, right) => right[1] - left[1])
            .slice(0, count)
            .map(([name, us]) => ({ name, ms: Math.round(us) / 1000, share: Math.round((us / totalUs) * 1000) / 10 }));
    return { totalMs: totalUs / 1000, byModule: top(byModule, 25), byFunction: top(byFunction, 40) };
}
