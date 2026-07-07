/** Owns the curve + band `RGBA32F` GPU textures shared across a `SharedAtlas`'s lifetime —
 *  the WebGL2 port of the WebGPU `_gpu/text-textures.ts` (`ensureSharedAtlasGpu`).
 *
 *  Both textures are 4096 wide (`TEX_WIDTH`); their height is the number of used rows. The
 *  fragment shader addresses them exclusively with `texelFetch` (integer texel coords, no
 *  interpolation), so they are uploaded as **full 32-bit float** (curve control points need
 *  the precision) with **NEAREST** min/mag — that means no `OES_texture_float_linear`
 *  dependency, and `RGBA32F` is sampleable in core WebGL2.
 *
 *  Lazy-init: created on first bind, re-uploaded only when the atlas `version` advances,
 *  resized (re-`texImage2D`) when the used-row count grows. */

import type { GLEngineContext } from "../../context.js";
import { createFloatTexture, disposeTexture, updateRawTexture, type GLTexture } from "../../texture.js";
import type { SharedAtlas, SharedAtlasGpu } from "../glyph-storage.js";

/** Width of the curve / band textures (texels). Must match `TEX_WIDTH` in `glyph-storage.ts`;
 *  duplicated as a literal so this module never imports the data layer just for a constant. */
const TEX_WIDTH = 4096;

/** Rows needed to hold `texels` at `TEX_WIDTH` per row (≥ 1 so the texture is never 0-height). */
function rowsForTexels(texels: number): number {
    if (texels <= 0) {
        return 1;
    }
    return Math.ceil(texels / TEX_WIDTH);
}

/** Create one `RGBA32F` / NEAREST atlas texture sized `TEX_WIDTH × rows`, uploading `data`. */
function createAtlasTexture(engine: GLEngineContext, data: Float32Array, rows: number): GLTexture {
    const gl = engine.gl;
    return createFloatTexture(engine, data, TEX_WIDTH, rows, {
        type: gl.FLOAT,
        format: gl.RGBA,
        minFilter: gl.NEAREST,
        magFilter: gl.NEAREST,
        wrapS: gl.CLAMP_TO_EDGE,
        wrapT: gl.CLAMP_TO_EDGE,
    });
}

/** Ensure `atlas.gpu` matches `engine` and reflects the atlas's current contents.
 *  Creates the textures on first call, re-uploads when `atlas.version` advances (a glyph
 *  was appended), and resizes when the used-row count grows. Returns the live companion. */
export function ensureSharedAtlasGpu(engine: GLEngineContext, atlas: SharedAtlas): SharedAtlasGpu {
    const curveRows = rowsForTexels(atlas.curveTexelsUsed);
    const bandRows = rowsForTexels(atlas.bandTexelsUsed);

    let gpu = atlas.gpu;
    if (gpu && gpu.engine !== engine) {
        disposeTexture(gpu.engine, gpu.curveTex);
        disposeTexture(gpu.engine, gpu.bandTex);
        gpu = null;
        atlas.gpu = null;
    }

    if (!gpu) {
        gpu = {
            engine,
            curveTex: createAtlasTexture(engine, atlas.curveTexData, curveRows),
            bandTex: createAtlasTexture(engine, atlas.bandTexData, bandRows),
            curveTexRows: curveRows,
            bandTexRows: bandRows,
            uploadedVersion: atlas.version,
        };
        atlas.gpu = gpu;
        return gpu;
    }

    // A version bump always accompanies an append (which is the only thing that can change the
    // used-row count), so the version compare alone is sufficient; the row checks are kept as a
    // defensive belt so a resize never silently uploads at the wrong height.
    let reupload = gpu.uploadedVersion !== atlas.version;
    if (curveRows !== gpu.curveTexRows || bandRows !== gpu.bandTexRows) {
        reupload = true;
    }
    if (reupload) {
        updateRawTexture(engine, gpu.curveTex, atlas.curveTexData, { width: TEX_WIDTH, height: curveRows });
        updateRawTexture(engine, gpu.bandTex, atlas.bandTexData, { width: TEX_WIDTH, height: bandRows });
        gpu.curveTexRows = curveRows;
        gpu.bandTexRows = bandRows;
        gpu.uploadedVersion = atlas.version;
    }
    return gpu;
}
