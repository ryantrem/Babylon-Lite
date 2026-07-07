import { createEngine, startEngine, loadFont, createDefaultTextData, createTextLayer, createTextRenderer, registerTextRenderer } from "babylon-lite";

/**
 * Babylon.js (WebGPU `babylon-lite`) reference for GL Scene 16 — TextRenderer (Slug).
 *
 * scene16 (lab/gl/src/scene16.ts) is the GL port of the WebGPU `scene180` text demo; it was
 * translated from this exact WebGPU Slug path. Both packages share the same font, the same CPU
 * layout, and the same Slug coverage shader (only the GPU backend — WGSL/WebGPU here vs GLSL ES
 * 3.00/WebGL2 there — differs), so a correct port is identical up to sub-pixel `fwidth` AA.
 *
 * NOT an automated parity target. Slug has no full-Babylon.js equivalent to diff against, so
 * scene16 ships with `skipParity: true` and no golden (matching the WebGPU text scenes
 * scene180/181). This file exists purely as a MANUAL visual reference: open scene16.html (GL) and
 * babylon-ref-scene16.html (WebGPU) side-by-side to confirm the hand-ported GLSL matches the
 * original WGSL. No parity harness consumes it.
 *
 * scene16 is interactive (live textarea + sliders + drag/wheel, like scene180). This reference
 * renders scene16's DEFAULT state — the initial textarea text, default white color, and the
 * initial `{x:360,y:380}` position at scale 1 / rotation 0 / opacity 1 — as a single STATIC frame
 * so the still comparison lines up. After `startEngine` resolves (first frame presented) we stamp
 * `dataset.ready` then `dataset.animationFrozen`; the loop keeps re-presenting the identical frame
 * so any later screenshot is stable.
 */

// Matches scene16.html's default #textInput content (the leading newline after the tag is ignored).
const TEXT = "Hello Slug\nWebGL2 text\nNo scene. No camera.";
const FONT_SIZE_PX = 48;
const POSITION_PX = { x: 360, y: 380 };
const TEXT_COLOR: readonly [number, number, number, number] = [1, 1, 1, 1];
const CLEAR_VALUE = { r: 0.05, g: 0.06, b: 0.09, a: 1 };

async function run(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const engine = await createEngine(canvas);
    const font = await loadFont("/fonts/Inter.ttf");

    const data = createDefaultTextData(font, FONT_SIZE_PX, TEXT, TEXT_COLOR, { maxWidth: 600, align: "left" });
    const layer = createTextLayer(data, { positionPx: POSITION_PX, scale: 1, rotationRad: 0, opacity: 1 });

    const tr = createTextRenderer(engine, { layers: [layer], clearValue: CLEAR_VALUE });
    registerTextRenderer(tr);

    await startEngine(engine);
    canvas.dataset.ready = "true";
    canvas.dataset.animationFrozen = "true";
}

void run();
