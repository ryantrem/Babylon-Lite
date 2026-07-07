import { describe, expect, it, beforeAll } from "vitest";
import { createGLEngine, registerTextRenderer, runRenderLoop, unregisterTextRenderer, type GLEngineContext, type TextRenderer } from "../../../packages/babylon-lite-gl/src/index";
import { createMockCanvas, createMockGL } from "./_lite-gl-mock";

beforeAll(() => {
    // Node has no rAF — stub it. registerTextRenderer installs a render-loop callback via
    // runRenderLoop, which schedules the first frame with requestAnimationFrame; these tests
    // never advance a frame, so the bound draw callback is stored but never actually invoked.
    const g = globalThis as { requestAnimationFrame?: (cb: FrameRequestCallback) => number; cancelAnimationFrame?: (h: number) => void };
    if (g.requestAnimationFrame === undefined) {
        g.requestAnimationFrame = () => 1;
    }
    if (g.cancelAnimationFrame === undefined) {
        g.cancelAnimationFrame = () => undefined;
    }
});

/** A minimal `TextRenderer` stand-in. `registerTextRenderer` / `unregisterTextRenderer` only
 *  read `_engine` / `_disposed` / `_loopFn`, so we avoid compiling a real Slug effect (which
 *  needs a live GL context). The stored callback is never invoked here (rAF is a no-op). */
function fakeRenderer(engine: GLEngineContext, disposed = false): TextRenderer {
    return { _engine: engine, _disposed: disposed, _loopFn: null } as unknown as TextRenderer;
}

describe("lite-gl text renderer — engine-loop registration", () => {
    it("registerTextRenderer installs exactly one render-loop callback (idempotent)", () => {
        const engine = createGLEngine(createMockCanvas(createMockGL()));
        const r = fakeRenderer(engine);
        registerTextRenderer(r);
        registerTextRenderer(r);
        registerTextRenderer(r);
        expect(engine._loops.length).toBe(1);
        expect(r._loopFn).not.toBeNull();
    });

    it("unregisterTextRenderer removes the callback and clears _loopFn (idempotent)", () => {
        const engine = createGLEngine(createMockCanvas(createMockGL()));
        const r = fakeRenderer(engine);
        registerTextRenderer(r);
        unregisterTextRenderer(r);
        expect(engine._loops.length).toBe(0);
        expect(r._loopFn).toBeNull();
        unregisterTextRenderer(r);
        expect(engine._loops.length).toBe(0);
    });

    it("registering one renderer leaves another engine callback untouched", () => {
        const engine = createGLEngine(createMockCanvas(createMockGL()));
        const r = fakeRenderer(engine);
        registerTextRenderer(r);
        const other = (): void => undefined;
        runRenderLoop(engine, other);
        unregisterTextRenderer(r);
        expect(engine._loops.length).toBe(1);
        expect(engine._loops[0]).toBe(other);
    });

    it("registerTextRenderer is a no-op on a disposed renderer", () => {
        const engine = createGLEngine(createMockCanvas(createMockGL()));
        const r = fakeRenderer(engine, true);
        registerTextRenderer(r);
        expect(engine._loops.length).toBe(0);
        expect(r._loopFn).toBeNull();
    });
});
