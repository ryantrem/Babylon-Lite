import type { Page, TestInfo } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
const captures = new WeakMap<Page, { console: { type: string; text: string; url: string }[]; errors: string[]; responses: { status: number; url: string }[] }>();
export function captureComputeBrowser(page: Page) {
    const console: { type: string; text: string; url: string }[] = [],
        errors: string[] = [],
        responses: { status: number; url: string }[] = [];
    page.on("console", (message) => {
        if (["warning", "error"].includes(message.type())) {
            console.push({ type: message.type(), text: message.text(), url: message.location().url });
        }
    });
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("response", (response) => {
        if (response.status() >= 400) {
            responses.push({ status: response.status(), url: response.url() });
        }
    });
    const captured = { console, errors, responses };
    captures.set(page, captured);
    return captured;
}
export async function attachComputeBrowser(page: Page, info: TestInfo): Promise<void> {
    const captured = captures.get(page);
    const adapter = await page.evaluate(async () => {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        return adapter ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description } : null;
    });
    const body = JSON.stringify({ adapter, ...captured }, null, 2);
    if (process.env.LITE_BROWSER_EVIDENCE_DIR) {
        mkdirSync(process.env.LITE_BROWSER_EVIDENCE_DIR, { recursive: true });
        const path = join(process.env.LITE_BROWSER_EVIDENCE_DIR, `${basename(info.file)}.json`);
        writeFileSync(path, body + "\n");
        await info.attach("compute-webgpu-and-console", { path, contentType: "application/json" });
    } else {
        await info.attach("compute-webgpu-and-console", { body, contentType: "application/json" });
    }
}
