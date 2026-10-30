/** Timed workload runner; run only with approval. Bounded; Prints JSON; writes no repository files. */
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
const args = process.argv.slice(2);
for (const arg of args) {
    if (!/^--script=(?:deva|arab)$|^--workload=(?:blocks|single)$|^--base=https?:\/\//.test(arg)) {
        throw new Error(`Unknown workload argument: ${arg}`);
    }
}
const script = args.includes("--script=arab") ? "arab" : "deva";
const workload = args.includes("--workload=single") ? "single" : "blocks";
const base = new URL(args.find((arg) => arg.startsWith("--base="))?.slice(7) ?? "http://localhost:5187");
let server;
try {
    let ready = false;
    try {
        ready = (await fetch(base)).ok;
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
                ready = (await fetch(base)).ok;
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
        await page.goto(new URL(`/lite/compute-text-devanagari-bench.html?script=${script}&workload=${workload}`, base).href);
        const adapter = await page.evaluate(async () => {
            const value = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
            return value ? { vendor: value.info.vendor, architecture: value.info.architecture, device: value.info.device, description: value.info.description } : null;
        });
        const results = await page.evaluate(async () => window.runComputeTextWorkload());
        if (errors.length) {
            throw new Error(`Browser errors: ${errors.join("; ")}`);
        }
        process.stdout.write(JSON.stringify({ browser: browser.version(), adapter, results }, null, 2) + "\n");
        await context.close();
    } finally {
        await browser.close();
    }
} finally {
    if (server?.pid && server.exitCode === null) {
        server.kill("SIGTERM");
    }
}
