/** Scene 16 — TextRenderer demo (Slug, WebGL2). The GL analogue of the WebGPU `scene180`:
 *  a standalone 2D text path — no scene, no camera. A live-editable textarea + sliders +
 *  canvas drag/wheel drive a `TextLayer`.
 *
 *  NOT a parity scene: Slug SDF text has no full-Babylon.js equivalent to diff against, so this
 *  ships with `skipParity: true` (matching WebGPU scene180/181). The sibling
 *  `babylon-ref-scene16.ts` renders the same text with the WebGPU `babylon-lite` Slug renderer as
 *  a MANUAL side-by-side visual reference — no automated parity test consumes it. */

import {
    createGLEngine,
    resizeGLEngine,
    runRenderLoop,
    loadFont,
    createDefaultTextData,
    updateDefaultTextData,
    updateTextData,
    createTextLayer,
    createTextRenderer,
    registerTextRenderer,
    isTextRendererReady,
} from "babylon-lite-gl";

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const textarea = document.getElementById("textInput") as HTMLTextAreaElement;
const rot = document.getElementById("rot") as HTMLInputElement;
const opacity = document.getElementById("opacity") as HTMLInputElement;
const rotVal = document.getElementById("rotVal")!;
const opacityVal = document.getElementById("opacityVal")!;
const red = document.getElementById("red") as HTMLInputElement;
const green = document.getElementById("green") as HTMLInputElement;
const blue = document.getElementById("blue") as HTMLInputElement;
const redVal = document.getElementById("redVal")!;
const greenVal = document.getElementById("greenVal")!;
const blueVal = document.getElementById("blueVal")!;

function currentColor(): [number, number, number, number] {
    return [+red.value, +green.value, +blue.value, 1];
}

async function run(): Promise<void> {
    const engine = createGLEngine(canvas, { alpha: false });
    // Track the CSS box each frame (WebGPU's engine auto-resizes; GL is explicit).
    runRenderLoop(engine, () => resizeGLEngine(engine));

    const font = await loadFont("/fonts/Inter.ttf");

    const data = createDefaultTextData(font, 48, textarea.value, currentColor(), {
        maxWidth: 600,
        align: "left",
    });

    const layer = createTextLayer(data, {
        positionPx: { x: 360, y: 380 },
        scale: 1,
        rotationRad: 0,
        opacity: 1,
    });

    const tr = createTextRenderer(engine, {
        layers: [layer],
        clear: true,
        clearValue: { r: 0.05, g: 0.06, b: 0.09, a: 1 },
    });
    registerTextRenderer(tr);

    textarea.addEventListener("input", () => {
        // updateDefaultTextData preserves whatever defaultColor the live run currently has,
        // so a previous color-slider change carries through automatically.
        updateDefaultTextData(data, textarea.value);
    });

    const onColor = (): void => {
        // Color isn't part of updateDefaultTextData; overlay it via replaceRun.
        const r = data.runs[0]!;
        updateTextData(data, { update: "replaceRun", previous: r, run: { ...r, defaultColor: currentColor() } });
        redVal.textContent = (+red.value).toFixed(2);
        greenVal.textContent = (+green.value).toFixed(2);
        blueVal.textContent = (+blue.value).toFixed(2);
    };
    red.addEventListener("input", onColor);
    green.addEventListener("input", onColor);
    blue.addEventListener("input", onColor);

    rot.addEventListener("input", () => {
        layer.rotationRad = (+rot.value * Math.PI) / 180;
        rotVal.textContent = rot.value + "°";
    });
    opacity.addEventListener("input", () => {
        layer.opacity = +opacity.value;
        opacityVal.textContent = (+opacity.value).toFixed(2);
    });

    // Drag-to-move (in CSS pixels — matches layer.positionPx units).
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    canvas.addEventListener("pointerdown", (e) => {
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        canvas.classList.add("dragging");
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
        if (!dragging) {
            return;
        }
        layer.positionPx.x += e.clientX - lastX;
        layer.positionPx.y += e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
    });
    const endDrag = (e: PointerEvent): void => {
        if (!dragging) {
            return;
        }
        dragging = false;
        canvas.classList.remove("dragging");
        canvas.releasePointerCapture(e.pointerId);
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // Wheel-to-scale, anchored at the cursor so the point under the mouse stays put.
    canvas.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();
            const factor = Math.exp(-e.deltaY * 0.001);
            const newScale = layer.scale * factor;
            const k = newScale / layer.scale;
            layer.positionPx.x = e.clientX + (layer.positionPx.x - e.clientX) * k;
            layer.positionPx.y = e.clientY + (layer.positionPx.y - e.clientY) * k;
            layer.scale = newScale;
        },
        { passive: false }
    );

    // The registered renderer self-gates until the effect compiles; stamp `ready` for the loader
    // overlay + screenshot harness once it is drawing (mirrors scene180's post-startEngine stamp).
    let ready = false;
    runRenderLoop(engine, () => {
        if (!ready && isTextRendererReady(tr)) {
            ready = true;
            canvas.dataset.ready = "true";
        }
    });
}

void run();
