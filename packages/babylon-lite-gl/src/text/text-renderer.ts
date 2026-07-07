/** TextRenderer (WebGL2) — a standalone draw for one or more `TextLayer`s. The WebGL2 port
 *  of the WebGPU `text/text-renderer.ts`, but reshaped to the lite-gl idiom: it is NOT a
 *  scene `RenderingContext`. Draw it either by registering it with the engine's render loop
 *  ({@link registerTextRenderer} — the portable path that mirrors the WebGPU renderer's
 *  engine-driven auto-draw) or by calling {@link renderText} manually each frame (the sibling
 *  of `renderSprites`, for callers that drive their own loop / need per-frame control).
 *
 *  `TextLayer` is the pure-data 2D pixel-space placement record bound to a single `TextData`
 *  (position / rotation / scale / order / opacity / visible). Mutate its fields between
 *  frames; `renderText` rebuilds the per-layer MVP and re-uploads dirty instances on demand.
 *
 *  GPU model (no UBOs / bind groups in WebGL2):
 *    - one shared 6-corner quad VBO (the two-triangle unit quad, corner signs = `slugCorner`);
 *    - the per-`TextData` instance buffer lives on `TextData._gpu` (see `_gpu/text-instances`);
 *    - the curve/band `RGBA32F` atlas textures live on `SharedAtlas.gpu` (see `_gpu/text-textures`);
 *    - the 96-byte WebGPU UBO becomes three discrete uniforms: `mvp` / `uViewport` / `uColor`.
 *  Per draw group the instance attribute pointers are re-based at the group's byte offset —
 *  WebGL2 has no base-instance, so this stands in for the WebGPU `draw(6, count, 0, slotStart)`. */

import type { GLEngineContext } from "../context.js";
import { createEffect, disposeEffect, isEffectReady, setEffectFloat4, setEffectMatrix, setEffectTexture, useEffect, type GLEffect } from "../effect.js";
import { bindAttributes, createVertexBuffer, disposeBuffer, unbindInstanceAttributes, type GLAttributeDescriptor, type GLVertexBuffer } from "../mesh.js";
import { GLBlendMode, setBlendMode, setBlendState } from "../blend.js";
import { applyGLStates } from "../apply-states.js";
import { clearEngine } from "../depth-stencil.js";
import { setViewport } from "../effect-renderer.js";
import { runRenderLoop, stopRenderLoop } from "../render-loop.js";
import type { TextData } from "./text-data.js";
import { TEXT_INSTANCE_BYTES } from "./text-data.js";
import { ensureSharedAtlasGpu } from "./_gpu/text-textures.js";
import { ensureTextDataGpu } from "./_gpu/text-instances.js";

/* ─────────────────────────────  shaders (GLSL ES 3.00)  ───────────────────── */

/** Vertex shader — GLSL ES 3.00 translation of `slug.vert.wgsl`. Eric Lengyel's Slug
 *  dynamic dilation, byte-for-byte: the dead-slot collapse, the MVP-row extraction
 *  (`mvp[i].x` — GLSL mats are column-major, identical to WGSL) and the dilation math. */
const TEXT_VERTEX_SOURCE = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec2 slugCorner;
layout(location = 1) in vec4 slugBounds;
layout(location = 2) in vec4 slugAnchor;
layout(location = 3) in vec4 slugAtlas;
layout(location = 4) in vec4 slugBand;
layout(location = 5) in vec4 slugColor;
uniform mat4 mvp;
uniform vec4 uViewport;
uniform vec4 uColor;
out vec2 vTexcoord;
flat out vec4 vBanding;
flat out vec4 vGlyph;
flat out vec4 vColor;
void main(void) {
    // Dead-slot sentinel (slugAnchor.w == 1): collapse all 6 vertices off-screen.
    if (slugAnchor.w > 0.5) {
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
        vTexcoord = vec2(0.0);
        vBanding = vec4(0.0);
        vGlyph = vec4(0.0);
        vColor = vec4(0.0);
        return;
    }
    vec2 isMax = vec2(step(0.0, slugCorner.x), step(0.0, slugCorner.y));
    vec2 tex = mix(slugBounds.xy, slugBounds.zw, isMax);
    float invScale = slugAnchor.z;
    float scale = invScale != 0.0 ? 1.0 / invScale : 0.0;
    vec2 pos = slugAnchor.xy + tex * scale;
    vec2 normal = slugCorner;
    vec4 jac = vec4(invScale, 0.0, 0.0, invScale);
    // MVP rows from column-major storage (only .xy + translate column are used).
    vec2 row0 = vec2(mvp[0].x, mvp[1].x);
    vec2 row1 = vec2(mvp[0].y, mvp[1].y);
    vec2 row3 = vec2(mvp[0].w, mvp[1].w);
    float row0w = mvp[3].x;
    float row1w = mvp[3].y;
    float row3w = mvp[3].w;
    vec2 n = normalize(normal);
    float s = dot(row3, pos) + row3w;
    float t = dot(row3, n);
    float u = (s * dot(row0, n) - t * (dot(row0, pos) + row0w)) * uViewport.x;
    float v = (s * dot(row1, n) - t * (dot(row1, pos) + row1w)) * uViewport.y;
    float s2 = s * s;
    float st = s * t;
    float uv = u * u + v * v;
    vec2 d = normal * (s2 * (st + sqrt(uv)) / (uv - st * st));
    vec2 dilatedPos = pos + d;
    vec2 dilatedTex = vec2(tex.x + dot(d, jac.xy), tex.y + dot(d, jac.zw));
    gl_Position = mvp * vec4(dilatedPos, 0.0, 1.0);
    vTexcoord = dilatedTex;
    vBanding = slugBand;
    vGlyph = slugAtlas;
    vColor = vec4(slugColor.rgb, slugColor.a * uColor.a);
}`;

/** Fragment shader — GLSL ES 3.00 translation of `slug.frag.wgsl`. Per-pixel coverage from
 *  quadratic Bézier bands: the `texelFetch` curve/band lookups, the quadratic-root solvers,
 *  the root-code table and the dual-axis coverage accumulation are byte-for-byte. */
const TEXT_FRAGMENT_SOURCE = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D curveTex;
uniform sampler2D bandTex;
uniform vec4 uColor;
in vec2 vTexcoord;
flat in vec4 vBanding;
flat in vec4 vGlyph;
flat in vec4 vColor;
out vec4 glFragColor;

int calcRootCode(float y1, float y2, float y3) {
    int i1 = y1 < 0.0 ? 1 : 0;
    int i2 = y2 < 0.0 ? 2 : 0;
    int i3 = y3 < 0.0 ? 4 : 0;
    int shift = i1 + i2 + i3;
    return (0x2E74 >> shift) & 0x0101;
}
vec2 solveHorizPoly(vec4 p12, vec2 p3) {
    vec2 a = vec2(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    vec2 b = vec2(p12.x - p12.z, p12.y - p12.w);
    float ra = 1.0 / a.y;
    float rb = 0.5 / b.y;
    float disc = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - disc) * ra;
    float t2 = (b.y + disc) * ra;
    if (abs(a.y) <= max(abs(b.y), abs(p12.y)) * 1.0e-4) {
        t1 = p12.y * rb;
        t2 = p12.y * rb;
    }
    return vec2((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}
vec2 solveVertPoly(vec4 p12, vec2 p3) {
    vec2 a = vec2(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    vec2 b = vec2(p12.x - p12.z, p12.y - p12.w);
    float ra = 1.0 / a.x;
    float rb = 0.5 / b.x;
    float disc = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    float t1 = (b.x - disc) * ra;
    float t2 = (b.x + disc) * ra;
    if (abs(a.x) <= max(abs(b.x), abs(p12.x)) * 1.0e-4) {
        t1 = p12.x * rb;
        t2 = p12.x * rb;
    }
    return vec2((a.y * t1 - b.y * 2.0) * t1 + p12.y, (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}
ivec2 calcBandLoc(ivec2 glyphLoc, int offset) {
    ivec2 bandLoc = ivec2(glyphLoc.x + offset, glyphLoc.y);
    bandLoc.y = bandLoc.y + (bandLoc.x >> 12);
    bandLoc.x = bandLoc.x & 4095;
    return bandLoc;
}
void main(void) {
    // Cull back-facing fragments — one-sided text (matches the WebGPU @builtin(front_facing)
    // discard). renderText keeps gl.frontFace(CCW) so the readable side stays front-facing in
    // WebGL's Y-up window space; only the mirror (back) side is discarded.
    if (!gl_FrontFacing) {
        discard;
    }
    vec2 renderCoord = vTexcoord;
    vec2 emsPerPixel = fwidth(renderCoord);
    vec2 pixelsPerEm = 1.0 / emsPerPixel;

    ivec2 glyphLoc = ivec2(int(vGlyph.x + 0.5), int(vGlyph.y + 0.5));
    ivec2 bandMax = ivec2(int(vGlyph.z + 0.5), int(vGlyph.w + 0.5));
    vec4 bandTransform = vBanding;

    ivec2 bandIndex = clamp(ivec2(renderCoord * bandTransform.xy + bandTransform.zw), ivec2(0), bandMax);

    float xcov = 0.0;
    float xwgt = 0.0;
    vec4 hbandRaw = texelFetch(bandTex, ivec2(glyphLoc.x + bandIndex.y, glyphLoc.y), 0);
    int hbandCount = int(hbandRaw.x + 0.5);
    int hbandOffset = int(hbandRaw.y + 0.5);
    ivec2 hbandLoc = calcBandLoc(glyphLoc, hbandOffset);
    for (int ci = 0; ci < hbandCount; ci++) {
        vec4 locRaw = texelFetch(bandTex, ivec2(hbandLoc.x + ci, hbandLoc.y), 0);
        ivec2 curveLoc = ivec2(int(locRaw.x + 0.5), int(locRaw.y + 0.5));
        vec4 p12 = texelFetch(curveTex, curveLoc, 0) - vec4(renderCoord, renderCoord);
        vec2 p3 = texelFetch(curveTex, ivec2(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) { break; }
        int code = calcRootCode(p12.y, p12.w, p3.y);
        if (code != 0) {
            vec2 r = solveHorizPoly(p12, p3) * pixelsPerEm.x;
            if ((code & 1) != 0) {
                xcov = xcov + clamp(r.x + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1) {
                xcov = xcov - clamp(r.y + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    float ycov = 0.0;
    float ywgt = 0.0;
    vec4 vbandRaw = texelFetch(bandTex, ivec2(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y), 0);
    int vbandCount = int(vbandRaw.x + 0.5);
    int vbandOffset = int(vbandRaw.y + 0.5);
    ivec2 vbandLoc = calcBandLoc(glyphLoc, vbandOffset);
    for (int ci = 0; ci < vbandCount; ci++) {
        vec4 locRaw = texelFetch(bandTex, ivec2(vbandLoc.x + ci, vbandLoc.y), 0);
        ivec2 curveLoc = ivec2(int(locRaw.x + 0.5), int(locRaw.y + 0.5));
        vec4 p12 = texelFetch(curveTex, curveLoc, 0) - vec4(renderCoord, renderCoord);
        vec2 p3 = texelFetch(curveTex, ivec2(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) { break; }
        int code = calcRootCode(p12.x, p12.z, p3.x);
        if (code != 0) {
            vec2 r = solveVertPoly(p12, p3) * pixelsPerEm.y;
            if ((code & 1) != 0) {
                ycov = ycov - clamp(r.x + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1) {
                ycov = ycov + clamp(r.y + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    float coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0), min(abs(xcov), abs(ycov)));
    coverage = clamp(coverage, 0.0, 1.0);
    // Coverage gamma: raise edge coverage to 1/coverageGamma so anti-aliased edges composite
    // heavier (mimics gamma-space stem darkening). uColor.x is 1 by default (no-op).
    coverage = pow(coverage, uColor.x);
    // Premultiplied-alpha output: scale rgb by the effective alpha (per-glyph alpha ×
    // layer opacity, carried in vColor.a) so the premult blend composites correctly. At
    // full opacity vColor.a is 1, so this is a no-op; below 1 it fades rgb toward the
    // destination instead of leaving it full-bright (which made the opacity slider inert).
    glFragColor = vec4(vColor.rgb * vColor.a, vColor.a) * coverage;
}`;

/** Attribute names in location order (0..5) — `slugCorner` then the five per-instance vec4s. */
const TEXT_ATTRIBUTES = ["slugCorner", "slugBounds", "slugAnchor", "slugAtlas", "slugBand", "slugColor"] as const;

/** Shared 6-corner unit quad (two triangles): corner signs feed `slugCorner`. Matches the
 *  WebGPU pipeline's quad vertex buffer exactly. */
const QUAD_CORNERS = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);

/** `gl.TRIANGLES`. */
const TRIANGLES = 0x0004;
/** `gl.CCW` — counter-clockwise front face. The Slug quad's readable triangles are CCW in
 *  WebGL's y-up window space (the WebGPU reference's `frontFace:"ccw"` selected the same
 *  readable side in its y-down framebuffer space), so this keeps them front-facing and the
 *  `gl_FrontFacing` discard culls only the mirror (back) side. It is also the GL default. */
const CCW = 0x0901;

// ─── TextLayer ────────────────────────────────────────────────────────────

export interface TextLayerOptions {
    /** Top-left origin (in canvas pixels) for the layer's local coordinate frame. Default (0, 0). */
    readonly positionPx?: { readonly x: number; readonly y: number };
    /** Z-axis rotation about `positionPx`, in radians. Default 0. */
    readonly rotationRad?: number;
    /** Uniform scale applied to the laid-out text. Default 1. */
    readonly scale?: number;
    /** Sort order within a renderer (lower draws first). Default 0. */
    readonly order?: number;
    /** Alpha multiplier in [0, 1]. Default 1. */
    readonly opacity?: number;
    /** Coverage gamma for anti-aliased edges. Raises per-pixel coverage to `1/coverageGamma`
     *  before compositing, darkening/thickening anti-aliased edges to mimic the gamma-space
     *  blending used by native text renderers (DirectWrite/CoreText "stem darkening"). Only
     *  meaningful when rendering into an sRGB (linear-blended) surface, where correct linear
     *  AA otherwise makes text look lighter/thinner than gamma-space rasterizers. Values \>1
     *  thicken; 1 (default) is a no-op. Typical text values are ~1.8–2.2. */
    readonly coverageGamma?: number;
    /** Default true. */
    readonly visible?: boolean;
}

/** Pure-data 2D text layer. Mutate fields directly between frames. */
export interface TextLayer {
    /** @internal */
    readonly _kind: "text-layer";
    readonly data: TextData;
    positionPx: { x: number; y: number };
    rotationRad: number;
    scale: number;
    order: number;
    opacity: number;
    /** Coverage gamma for anti-aliased edges (see `TextLayerOptions.coverageGamma`). Default 1. */
    coverageGamma: number;
    visible: boolean;
}

/** Create a 2D text layer bound to `data`. The layer is pure data — add it to a
 *  {@link TextRenderer} (via the `layers` option or {@link addTextRendererLayer}). */
export function createTextLayer(data: TextData, options?: TextLayerOptions): TextLayer {
    return {
        _kind: "text-layer",
        data,
        positionPx: { x: options?.positionPx?.x ?? 0, y: options?.positionPx?.y ?? 0 },
        rotationRad: options?.rotationRad ?? 0,
        scale: options?.scale ?? 1,
        order: options?.order ?? 0,
        opacity: options?.opacity ?? 1,
        coverageGamma: options?.coverageGamma ?? 1,
        visible: options?.visible ?? true,
    };
}

/** Update the layer's pixel position. Convenience wrapper. */
export function setTextLayerPosition(layer: TextLayer, x: number, y: number): void {
    layer.positionPx.x = x;
    layer.positionPx.y = y;
}

// ─── TextRenderer ─────────────────────────────────────────────────────────

export interface TextRendererOptions {
    /** The layers to draw (copied — later mutate via {@link addTextRendererLayer} /
     *  {@link removeTextRendererLayer}, or mutate the returned `layers` entries in place). */
    layers: readonly TextLayer[];
    /** Clear the canvas before drawing. Default true (set false for HUD overlays). */
    clear?: boolean;
    /** Clear color when `clear` is true. Default `{ r: 0, g: 0, b: 0, a: 1 }`. */
    clearValue?: { r: number; g: number; b: number; a?: number };
}

/** A text renderer owning its compiled effect + shared quad VBO. Created by
 *  {@link createTextRenderer}; drive it with {@link renderText}; release it with
 *  {@link disposeTextRenderer}. The per-block instance buffers and atlas textures are owned by
 *  the `TextData` / `GlyphStorage` the layers reference, not by the renderer. */
export interface TextRenderer {
    /** @internal */
    readonly _kind: "text-renderer";
    /** Read-only view of the layers drawn (in `order`, then insertion, order). */
    readonly layers: readonly TextLayer[];
    /** @internal Mutable alias of {@link layers} (same array reference). */
    _layers: TextLayer[];
    /** @internal Clear the canvas before drawing. */
    _clear: boolean;
    /** @internal Clear color used when {@link _clear} is true. */
    _clearValue: { r: number; g: number; b: number; a?: number };
    /** @internal The engine the renderer was created for. */
    _engine: GLEngineContext;
    /** @internal The compiled Slug text effect this renderer owns. */
    _effect: GLEffect;
    /** @internal Bound per-frame draw callback installed by {@link registerTextRenderer};
     *  null while the renderer is not registered with the engine's render loop. */
    _loopFn: ((dt: number) => void) | null;
    /** @internal Shared 6-corner quad VBO (per-vertex `slugCorner`). */
    _quad: GLVertexBuffer;
    /** @internal Reused per-group instance attribute descriptors (offsets re-based per draw). */
    _instDesc: GLAttributeDescriptor[];
    /** @internal True once disposed; subsequent calls are no-ops. */
    _disposed: boolean;
}

const _mvpScratch = new Float32Array(16);

/** Build the column-major affine MVP for `layer` — ported verbatim from the WebGPU
 *  `text-renderer.ts` `buildLayerMvp`. Maps glyph-local (font Y-up) coords through
 *  (scale, flip-Y) → rotate → translate → ortho(W, H, Y-down). */
function buildLayerMvp(layer: TextLayer, targetW: number, targetH: number, out: Float32Array): void {
    const s = layer.scale;
    const r = layer.rotationRad;
    const cr = Math.cos(r);
    const sr = Math.sin(r);
    const px = layer.positionPx.x;
    const py = layer.positionPx.y;
    const cx = (2 * s) / targetW;
    const cy = (2 * s) / targetH;
    out.fill(0);
    out[0] = cx * cr; // col 0, row 0
    out[1] = -cy * sr; // col 0, row 1
    out[4] = cx * sr; // col 1, row 0
    out[5] = cy * cr; // col 1, row 1
    out[10] = 1; // depth pass-through (we don't write depth)
    out[12] = (2 * px) / targetW - 1;
    out[13] = 1 - (2 * py) / targetH;
    out[15] = 1;
}

function compareLayers(a: TextLayer, b: TextLayer): number {
    return a.order - b.order;
}

/** Create a text renderer: compiles the Slug effect and builds the shared quad VBO. Both
 *  GPU resources auto-restore on `webglcontextrestored` (effect + buffer registries). */
export function createTextRenderer(engine: GLEngineContext, options: TextRendererOptions): TextRenderer {
    const effect = createEffect(engine, {
        name: "text-slug",
        vertexSource: TEXT_VERTEX_SOURCE,
        fragmentSource: TEXT_FRAGMENT_SOURCE,
        uniformNames: ["mvp", "uViewport", "uColor"],
        samplerNames: ["curveTex", "bandTex"],
        attributeNames: TEXT_ATTRIBUTES,
    });
    const layers = options.layers.slice();
    return {
        _kind: "text-renderer",
        layers,
        _layers: layers,
        _clear: options.clear ?? true,
        _clearValue: options.clearValue ?? { r: 0, g: 0, b: 0, a: 1 },
        _engine: engine,
        _effect: effect,
        _loopFn: null,
        _quad: createVertexBuffer(engine, QUAD_CORNERS, false),
        _instDesc: [
            { index: 1, size: 4, offset: 0, divisor: 1 },
            { index: 2, size: 4, offset: 16, divisor: 1 },
            { index: 3, size: 4, offset: 32, divisor: 1 },
            { index: 4, size: 4, offset: 48, divisor: 1 },
            { index: 5, size: 4, offset: 64, divisor: 1 },
        ],
        _disposed: false,
    };
}

/** Per-vertex quad descriptor (location 0, divisor 0). Module-scoped literal — pure. */
const QUAD_DESC: readonly GLAttributeDescriptor[] = [{ index: 0, size: 2, offset: 0, divisor: 0 }];

/**
 * Draw every visible layer of `renderer` to the bound framebuffer (the canvas). Uploads each
 * layer's dirty instances + its atlas, sets the MVP / viewport / opacity uniforms, and issues
 * one instanced `drawArraysInstanced(TRIANGLES, 0, 6, slotCount)` per draw group.
 *
 * No-op (returns 0) when the context is lost/disposed, the renderer is disposed, or the effect
 * has not finished compiling yet — poll {@link isTextRendererReady} (or just call every frame;
 * it self-gates) before stamping a "ready" flag.
 *
 * @returns The number of draw calls issued this frame.
 */
export function renderText(renderer: TextRenderer): number {
    const engine = renderer._engine;
    if (engine._isLost || engine._disposed || renderer._disposed) {
        return 0;
    }
    const effect = renderer._effect;
    if (!isEffectReady(engine, effect)) {
        return 0;
    }
    const gl = engine.gl;
    const W = engine.canvas.width;
    const H = engine.canvas.height;

    setViewport(engine);
    if (renderer._clear) {
        clearEngine(engine, { color: renderer._clearValue });
    }

    if (renderer._layers.length > 1) {
        renderer._layers.sort(compareLayers);
    }

    useEffect(engine, effect);
    // Premultiplied-alpha blend — matches the WebGPU pipeline exactly. The Slug fragment
    // outputs `vColor * coverage` (RGB already premultiplied by coverage, alpha = coverage),
    // so the color/alpha blend must use srcFactor `one` (NOT `src-alpha`, which would multiply
    // by coverage a second time and render anti-aliased edges as coverage², too dark).
    setBlendState(engine, { srcRGB: gl.ONE, dstRGB: gl.ONE_MINUS_SRC_ALPHA, srcAlpha: gl.ONE, dstAlpha: gl.ONE_MINUS_SRC_ALPHA });
    // The Slug quad's readable triangles are CCW in WebGL window space; keep the front face CCW
    // so the shader's `gl_FrontFacing` discard culls only the back (mirror) side, not everything.
    gl.frontFace(CCW);

    // Bind the shared quad once (location 0, divisor 0); instance attrs are re-bound per group.
    bindAttributes(engine, renderer._quad, QUAD_DESC, effect, true);

    let drawCalls = 0;
    for (const layer of renderer._layers) {
        if (!layer.visible) {
            continue;
        }
        const data = layer.data;
        if (data._instanceCount === 0) {
            continue;
        }
        const gpu = ensureTextDataGpu(engine, data);
        if (!gpu) {
            continue;
        }

        buildLayerMvp(layer, W, H, _mvpScratch);
        setEffectMatrix(engine, effect, "mvp", _mvpScratch);
        setEffectFloat4(engine, effect, "uViewport", W, H, 0, 0);
        // uColor.x = coverage-gamma reciprocal (1/coverageGamma) applied to anti-aliased edge
        // coverage in the fragment shader (1 = no-op). G/B unused. Alpha carries layer opacity;
        // per-glyph color comes from the slugColor attribute.
        const gammaInv = layer.coverageGamma > 0 ? 1 / layer.coverageGamma : 1;
        setEffectFloat4(engine, effect, "uColor", gammaInv, 1, 1, layer.opacity);

        for (const g of data._groups) {
            if (g.slotCount === 0) {
                continue;
            }
            const atlasGpu = ensureSharedAtlasGpu(engine, g.curveSet.atlas);
            setEffectTexture(engine, effect, "curveTex", atlasGpu.curveTex);
            setEffectTexture(engine, effect, "bandTex", atlasGpu.bandTex);

            // Re-base the 5 instance attributes at this group's first slot (no base-instance in GL2).
            const base = g.slotStart * TEXT_INSTANCE_BYTES;
            const d = renderer._instDesc;
            d[0]!.offset = base;
            d[1]!.offset = base + 16;
            d[2]!.offset = base + 32;
            d[3]!.offset = base + 48;
            d[4]!.offset = base + 64;
            bindAttributes(engine, gpu.instanceBuf, d, effect, true);

            // Flush the deferred GL state (premultiplied blend set above, plus any
            // depth/cull/color-mask) to GL before the direct draw — mirrors the flush
            // sites in effect-renderer/sprites/mesh. Without this the premult blend is
            // never enabled and empty quad texels overwrite the target with black.
            applyGLStates(engine);
            gl.drawArraysInstanced(TRIANGLES, 0, 6, g.slotCount);
            drawCalls++;
        }
    }

    unbindInstanceAttributes(engine);
    // Leave blending disabled so a subsequent opaque pass renders predictably (Babylon parity).
    setBlendMode(engine, GLBlendMode.DISABLE);
    return drawCalls;
}

/** Returns true once the renderer's effect has finished compiling (so {@link renderText} will
 *  actually draw). Handy for gating a "first frame drawn" flag in a demo / capture harness. */
export function isTextRendererReady(renderer: TextRenderer): boolean {
    if (renderer._disposed) {
        return false;
    }
    return isEffectReady(renderer._engine, renderer._effect);
}

/** Append `layer` to the renderer (no-op if already present). */
export function addTextRendererLayer(renderer: TextRenderer, layer: TextLayer): void {
    if (renderer._disposed || renderer._layers.includes(layer)) {
        return;
    }
    renderer._layers.push(layer);
}

/** Remove `layer` from the renderer. Returns true when it was present. Does NOT dispose the
 *  layer's `TextData` (the caller owns it). */
export function removeTextRendererLayer(renderer: TextRenderer, layer: TextLayer): boolean {
    const i = renderer._layers.indexOf(layer);
    if (i < 0) {
        return false;
    }
    renderer._layers.splice(i, 1);
    return true;
}

/** Register `renderer` with its engine's render loop so it draws automatically every frame —
 *  the portable counterpart of the WebGPU `registerTextRenderer` (whose engine auto-draws
 *  registered renderers). Idempotent. Installs a per-frame callback via `runRenderLoop`, which
 *  starts the engine's RAF loop if it is not already running; the draw self-gates until the
 *  effect has compiled (see {@link renderText}). Pair with {@link unregisterTextRenderer}.
 *
 *  Callers that drive their own loop (or need per-frame control, e.g. to freeze on a seek time)
 *  can skip this and invoke {@link renderText} manually instead. */
export function registerTextRenderer(renderer: TextRenderer): void {
    if (renderer._disposed || renderer._loopFn !== null) {
        return;
    }
    const fn = (): void => {
        renderText(renderer);
    };
    renderer._loopFn = fn;
    runRenderLoop(renderer._engine, fn);
}

/** Remove `renderer` from its engine's render loop, stopping the per-frame auto-draw installed
 *  by {@link registerTextRenderer}. Idempotent (no-op if the renderer was never registered). */
export function unregisterTextRenderer(renderer: TextRenderer): void {
    const fn = renderer._loopFn;
    if (fn === null) {
        return;
    }
    renderer._loopFn = null;
    stopRenderLoop(renderer._engine, fn);
}

/** Release the renderer's effect + quad VBO. Idempotent. Does NOT dispose the `TextData` /
 *  `GlyphStorage` the layers reference — the caller owns those (dispose them separately). */
export function disposeTextRenderer(renderer: TextRenderer): void {
    if (renderer._disposed) {
        return;
    }
    unregisterTextRenderer(renderer);
    renderer._disposed = true;
    disposeEffect(renderer._engine, renderer._effect);
    disposeBuffer(renderer._engine, renderer._quad);
    renderer._layers.length = 0;
}
