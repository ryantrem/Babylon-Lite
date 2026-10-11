# Module: Text

> Package path: `packages/babylon-lite/src/text/`
>
> Slug-style GPU font rendering for Lite. Glyphs are stored as quadratic Bézier
> outlines with a spatial-band index packed into two `rgba32float` textures, and
> drawn as instanced unit quads where the fragment shader resolves analytic
> coverage from the curves intersecting each pixel's bands. The module is layered
> so the lowest level (`GlyphStorage`) holds only outlines and atlases, the
> middle level (`TextData`) layers per-instance slot allocation on top, and the
> top levels (`TextRenderable`, `TextRenderer`) attach `TextData` to either a 3D
> scene or a standalone 2D pass.

## Purpose

The text feature exists to draw resolution-independent glyphs anywhere — inside a
3D scene as a world-space `TextRenderable`, or in pure 2D as a standalone
`TextRenderer` (no scene, no camera). Both paths share one CPU layout
(`TextData` per-instance buffer + draw groups) and one GPU pipeline (Slug
fragment shader against curve+band textures). They differ only in how the MVP
matrix is composed and which render pass owns the draw.

The module is organized as four lifetime tiers, longest-lived first:

1. **`GlyphStorage`** — outlines + GPU atlases. One storage can hold many
   curve-sets (one per font face) and back many `TextData`s. Caller-owned.
2. **`TextData`** — per-text-block instance buffer + slot allocator + draw
   groups. Borrows a `GlyphStorage`. One `TextData` per logical text block.
3. **`TextRenderable`** (3D) / **`TextRenderer` + `TextLayer`** (2D) — the
   thing that gets registered with an engine/scene and actually draws.
4. **`DefaultTextData`** + helpers — convenience layer that does default LTR
   layout via `text-shaper` and ships its own private `GlyphStorage`.

Each tier depends only on the tier below it, so a caller that hand-rolls its own
layout (e.g. an external rich-text engine) imports only `GlyphStorage` +
`TextData` + a renderer and pays zero bytes for the default layout / text-shaper
glue.

## Public API Surface

### Tier 1 — `GlyphStorage` (longest-lived)

```typescript
export interface GlyphStorage {
    /* opaque */
}
export type CurveSetId = string;

export function createGlyphStorage(initial?: Map<CurveSetId, Map<number, GlyphCurves>>): GlyphStorage;
export function updateGlyphStorage(storage: GlyphStorage, curveSetId: CurveSetId, curves: ReadonlyMap<number, GlyphCurves>): void;
export function disposeGlyphStorage(storage: GlyphStorage): void;
```

A `GlyphStorage` is an opaque bundle of `(curveSetId → glyph outlines + packed
atlas)`. Each `CurveSetId` (a string, typically the font family name) maps to
exactly one atlas; one `GlyphStorage` holds an arbitrary number of curve-sets.
Glyph ids inside a curve-set are dense small integers (font glyph indices), and
once packed into the atlas a glyph's slot is never moved.

`updateGlyphStorage` is idempotent per glyph id — already-present ids are
skipped, so callers can pass the union of every glyph they might draw without
re-rasterizing. Lifetime is caller-owned (matches `Texture2D` semantics): the
caller must outlive any `TextData` that borrows the storage, then call
`disposeGlyphStorage` exactly once to release every atlas's GPU textures.

The same `GlyphStorage` can be shared by reference across any number of
`TextData`s — that is the whole reason it is a separate tier. Two text blocks
in the same font pay one atlas upload.

### Tier 2 — `TextData`

```typescript
export interface TextData {
    readonly runs: readonly GlyphRun[];
    // ... opaque internals
}

export type GlyphRun = {
    readonly curveSet: CurveSetId;
    readonly glyphs: readonly PlacedGlyph[];
    readonly pixelsPerFontUnit: number;
    readonly defaultColor?: readonly [number, number, number, number];
};
export type PlacedGlyph = {
    readonly glyphId: number;
    readonly x: number; // pixels, glyph origin
    readonly y: number; // pixels, baseline up
    readonly color?: readonly [number, number, number, number]; // overrides run defaultColor
};

export function createTextData(storage: GlyphStorage, runs?: readonly GlyphRun[]): TextData;
export function updateTextData(data: TextData, update: TextDataUpdate): void;
export function disposeTextData(data: TextData): void;

export type TextDataUpdate =
    | { update: "reset"; runs?: GlyphRun[]; storage?: GlyphStorage }
    | { update: "addRun"; run: GlyphRun; insertBefore?: number }
    | { update: "removeRun"; run: GlyphRun | number }
    | { update: "replaceRun"; previous: GlyphRun | number; run: GlyphRun };
```

A `TextData` represents one logical text block as an ordered list of
`GlyphRun`s. Each run carries the glyphs in a single font (one `curveSet`) at a
single pixels-per-font-unit scale; mixed-font / mixed-size content is just
multiple runs in the same `TextData`. The `update` API is a small discriminated
union driving the slot allocator (see §Implementation).

`disposeTextData` releases only the per-block GPU resources (instance buffer +
bind groups). It does **not** touch the borrowed `GlyphStorage` — the caller
owns that lifetime.

### Tier 3a — `TextRenderable` (3D, attached to a scene)

```typescript
export interface TextRenderableOptions {
    readonly position?: Vec3;
    readonly rotationQuaternion?: { x: number; y: number; z: number; w: number };
    readonly scaling?: Vec3;
    readonly opacity?: number; // whole-block fade. default 1
    readonly ignoreDepth?: boolean; // skip depth test/write. default false
    readonly order?: number; // sort order. default 200
}

export interface TextRenderable extends Renderable {
    readonly position: ObservableVec3;
    readonly rotation: EulerProxy;
    readonly rotationQuaternion: ObservableQuat;
    readonly scaling: ObservableVec3;
    opacity: number;
    ignoreDepth: boolean;
    order: number;
}

export function createTextRenderable(data: TextData, options?: TextRenderableOptions): TextRenderable;
export function addTextRenderable(scene: SceneContext, renderable: TextRenderable): void;
export function disposeTextRenderable(renderable: TextRenderable): void;
```

A `TextRenderable` mirrors `Mesh`'s TRS surface (`position`, `rotation`,
`rotationQuaternion`, `scaling`) and implements the standard `Renderable`
interface, so it sorts and binds like any other scene entity. It is `isTransparent`
by default (text always uses src-over blending) and consumes its `TextData`
read-only — the data and its underlying `GlyphStorage` may be shared across
many `TextRenderable`s.

Scene-attached text can opt into alpha-to-coverage with
`setAlphaToCoverage(renderable, true)` before it is added/registered. The option
is effective only when `ignoreDepth === false` and the scene target is
multisampled. That variant uses straight RGB, analytic glyph coverage as alpha,
replacement color (no blend), and per-sample depth writes. Standalone
`TextRenderer` layers draw to a 1x swapchain and retain ordinary straight-alpha
blending.

### Tier 3b — `TextRenderer` + CPU/resident layers (standalone 2D)

```typescript
export interface TextLayerOptions {
    readonly positionPx?: { x: number; y: number }; // canvas pixel origin
    readonly rotationRad?: number; // z-axis rotation
    readonly scale?: number; // uniform
    readonly order?: number; // within renderer
    readonly opacity?: number;
    readonly coverageGamma?: number;
    readonly visible?: boolean;
}
export interface TextLayerPlacement {
    positionPx: { x: number; y: number };
    rotationRad: number;
    scale: number;
    order: number;
    opacity: number;
    coverageGamma: number;
    visible: boolean;
}
export interface TextLayer extends TextLayerPlacement {
    readonly data: TextData;
}
export interface ComputeTextLayer extends TextLayerPlacement {
    readonly data: ComputeTextData;
}
export type TextRendererLayer = TextLayer | ComputeTextLayer;
export function createTextLayer(data: TextData, options?: TextLayerOptions): TextLayer;
export function createComputeTextLayer(data: ComputeTextData, options?: TextLayerOptions): ComputeTextLayer;
export function setTextLayerPosition(layer: TextLayerPlacement, x: number, y: number): void;

export interface TextRendererOptions {
    layers: readonly TextRendererLayer[];
    clear?: boolean; // default true
    clearValue?: GPUColorDict;
}
export function createTextRenderer(surface: SurfaceContext, opts: TextRendererOptions): TextRenderer;
export function addTextRendererLayer(tr: TextRenderer, layer: TextRendererLayer): void;
export function removeTextRendererLayer(tr: TextRenderer, layer: TextRendererLayer): boolean;
export function registerTextRenderer(tr: TextRenderer): void;
export function unregisterTextRenderer(tr: TextRenderer): void;
export function disposeTextRenderer(tr: TextRenderer): void;
```

`TextRenderer` is a standalone `RenderingContext` (sibling of `SpriteRenderer`):
no scene, no camera. It opens its own swapchain render pass and draws each
visible CPU or resident layer in `order` order. The MVP is a pure CPU-built 2D affine
(layer position/rotation/scale + ortho projection) — there is no view matrix.
Layers can be added/removed at any time; their pixel position/rotation/scale/
opacity may be mutated directly between frames.

A scene that wants 2D HUD text on top of 3D uses `registerScene` first, then
`createTextRenderer` + `registerTextRenderer` so the text pass runs after the
scene's frame graph.

### Tier 4 — Default helpers (depend on `text-shaper`)

```typescript
export interface Font {
    /* opaque, wraps text-shaper.Font */
}
export function loadFont(url: string): Promise<Font>;
export function createFontFromBuffer(data: ArrayBuffer): Font;

export function extractGlyphCurves(font: Font, glyphIds: ReadonlySet<number>, target: Map<number, GlyphCurves>): void;
export function cubicToQuadratics(/* control points */): [QuadCurve, QuadCurve];

export interface DefaultTextData extends TextData {
    readonly width: number; // pixel-space laid-out width
    readonly height: number; // pixel-space laid-out height
}
export function createDefaultTextData(font: Font, fontSizePx: number, text: string, textColor?: [number, number, number, number], options?: TextLayoutOptions): DefaultTextData;
export function updateDefaultTextData(data: DefaultTextData, text: string, textColor?: [number, number, number, number]): void;
export function disposeDefaultTextData(data: DefaultTextData): void;
```

`createDefaultTextData` runs the default LTR + word-wrap + align layout
(`layout.ts`, built on `text-shaper`'s HarfBuzz-style shaping), extracts the
required glyph outlines from the font (`glyph-extraction.ts`, using `text-shaper`'s glyph
path provider), packs them into a fresh single-curve-set `GlyphStorage`, and
wraps the result in a `TextData` with one `GlyphRun`. The branded
`DefaultTextData` carries pixel-space `width` / `height` so callers can size
their `TextRenderable.scaling` or place a `TextLayer` precisely.

`updateDefaultTextData` re-shapes the text, appends any newly-needed glyph
outlines to the existing `GlyphStorage` (existing ids no-op via
`updateGlyphStorage`), and applies the new run via `updateTextData(replaceRun)`
— which hits the in-place rewrite fast path whenever the glyph count is
unchanged.

`disposeDefaultTextData` releases both the per-block resources and the
owned `GlyphStorage` (because this helper allocated both).

Callers driving their own text layout import only Tiers 1–3 and pay zero bytes
for `layout.ts`, `glyph-extraction.ts`, `default-text-data.ts`, or `text-shaper`'s shaping
codepath.

### Minimal example — `createDefaultTextData` + `TextRenderer`

The shortest path from font URL to rendered text on a canvas:

```typescript
import { createEngine, startEngine, loadFont, createDefaultTextData, createTextLayer, createTextRenderer, registerTextRenderer } from "@babylonjs/lite";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const engine = await createEngine(canvas);
const font = await loadFont("/fonts/Inter.ttf");

// Shape + extract curves + pack atlas in one call (a fresh GlyphStorage is
// allocated under the hood and released by `disposeDefaultTextData`).
const data = createDefaultTextData(font, 48, "Hello, world!");

// Place the laid-out block at pixel (32, 64) on the canvas.
const layer = createTextLayer(data, { positionPx: { x: 32, y: 64 } });

// Standalone 2D renderer — no scene, no camera.
const renderer = createTextRenderer(engine, { layers: [layer] });
registerTextRenderer(renderer);

await startEngine(engine);
```

For a 3D scene, swap the last three lines for `createTextRenderable(data)` +
`addTextRenderable(scene, …)`; everything else is identical.

## Implementation

### CPU data layout

Each `GlyphStorage` curve-set owns a `SharedAtlas`:

```typescript
type SharedAtlas = {
    curveTexData: Float32Array; // rgba32float, width 4096, grows by row doubling
    curveTexelsUsed: number;
    bandTexData: Float32Array; // rgba32float, width 4096, grows by row doubling
    bandTexelsUsed: number;
    glyphSlots: Map<number, AtlasSlot>; // glyphId → curve start + band header location + band counts
    version: number; // monotonic, bumped per packAppendGlyph
    gpu: SharedAtlasGpu | null; // lazy
};
```

The two textures are fixed-width `4096` and grow in row-doubling steps. Curve
texels store quadratic control points (`p0`, `p1`, `p2` packed as two
`vec4`s per curve); band texels store per-band headers (count + offset) followed
by curve-index lists. Per-glyph metadata lives in `AtlasSlot` (curve start
texel, band header location, `(vBandCount, hBandCount)` for the fragment
shader's transform).

Each `TextData` owns a contiguous packed instance buffer (`Float32Array`, 20
floats = 5 `vec4`s = 80 bytes per instance), an `instanceCount`, and a list of
draw groups:

```typescript
type TextDataDrawGroup = {
    groupKey: TextGroupKey; // draw-group identity; === curveSetId unless a styling feature interned another
    curveSetId: CurveSetId;
    curveSet: GlyphStorageCurveSet; // cached pointer into _storage._curveSets
    slotStart: number;
    slotCount: number; // live + dead
    liveCount: number;
    freeSlots: number[]; // LIFO stack of dead slot indices
    bindGroup: GPUBindGroup | null;
    bindGroupVersion: number; // last-seen atlas.uploadedVersion
};
```

One draw group per unique `groupKey` used by the live runs — which is one per unique
`curveSetId` unless an opt-in styling feature splits a curve set further (see
"Draw-group keys"). Each group owns a
contiguous `[slotStart, slotStart + slotCount)` slot range in the shared
instance buffer; **live and dead slots intermix within that range**. The vertex
shader detects dead slots and emits a degenerate off-screen quad, so they cost
only a vertex-shader invocation. A per-run `RunRecord` tracks which absolute
slot indices each `GlyphRun` currently occupies, so add/remove/replace are O(touched glyphs).

### Per-instance layout (5 × `vec4`, 80 bytes)

| Field        | Floats | Contents                                                                                             |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------- |
| `slugBounds` | 4      | `(xMin, yMin, xMax, yMax)` in font units — the quad's extent                                         |
| `slugAnchor` | 4      | `(xPx, yPx, 1/pixelsPerFontUnit, deadSentinel)` — pixel origin + scale; `.w = 1` marks the slot dead |
| `slugAtlas`  | 4      | `(glyphLocX, glyphLocY, bandMaxX, bandMaxY)` — band texture lookup base + max band indices           |
| `slugBand`   | 4      | `(bandScaleX, bandScaleY, bandOffsetX, bandOffsetY)` — derived from glyph bounds + band counts       |
| `slugColor`  | 4      | linear RGBA per glyph (falls back to run `defaultColor`, then white)                                 |

The vertex shader reads `slugAnchor.w` first; when non-zero it emits a
clip-space point at `(-2, -2, -2, 1)` so all six quad vertices collapse to one
off-screen point and rasterization culls the zero-area triangles.

### Slot allocator (fast incremental updates)

The allocator is the heart of the module — its purpose is to make
`updateTextData` cheap enough that a typing user can drive it on every
keystroke without rebuilding GPU buffers.

- **`addRun`** allocates one slot per glyph, first by popping from the owning
  group's `freeSlots` LIFO (a dead slot vacated by a prior `removeRun`), then
  by extending the group's range. Extending requires `copyWithin`-shifting
  every later group right by `extraSlots`; same-group runs are never shifted
  because slots within a group are unordered.
- **`removeRun`** writes the dead-slot sentinel into every slot the run
  occupied and pushes those slot indices into the group's `freeSlots`. If the
  group becomes empty (no live runs) it is dropped and later groups shift left
  to close the gap. Removed slots dirty the buffer range `[minSlot, maxSlot+1]`.
- **`replaceRun`** has a fast path: same curveSet + same glyph count → rewrite
  the existing slots in place (no allocator work, no buffer shift). This is
  the path `updateDefaultTextData` hits when the user types a single character
  into a line that didn't word-wrap. Different size or different curveSet
  falls back to `removeRun` + `addRun` at the same `runs` index.
- **`reset` (full or compaction)** rebuilds groups + slot ranges from scratch,
  re-using existing group records when the curveSet matches (preserves
  `bindGroup` identity when the atlas pointer is unchanged) and packing every
  live slot contiguously without gaps. Calling `updateTextData(data, { update: "reset" })`
  with both `runs` and `storage` omitted is therefore a pure compaction pass.

**Automatic vs manual compaction.** The allocator does compact automatically
in two cases: (a) dead slots within a group are always reclaimed by the next
`addRun` / `replaceRun` via the `freeSlots` LIFO, so freed slots are not lost,
they just sit dormant in-place; and (b) a group whose `liveCount` reaches zero
is dropped wholesale by `dropEmptyGroup`, shifting later groups left to close
the gap. What is **not** automatic is intra-group hole closing while the group
is still live: if a 200-glyph run shrinks to 5 glyphs the other 195 slots
remain dead until a future run reuses them or the caller invokes `reset`. The
vertex shader collapses dead slots to a single off-screen point (a cheap
vertex invocation, no fragment work), so the steady-state cost is bounded —
an explicit `reset` is only needed if a workload spends a long time with a
large dead-slot fraction and the caller wants the GPU instance buffer to
shrink. Adding an internal heuristic (e.g. auto-reset when `dead / total > 0.5`)
is a future option but not currently implemented.

### Dirty range + version-based GPU upload

`TextData` carries a `_version` (bumped per mutation) and a `[_dirtyStart, _dirtyEnd)`
half-open dirty range. Every allocator path that writes the instance buffer
calls `markDirty(data, minSlot, maxSlot + 1)`. The GPU side
(`TextRenderable` / `TextRenderer`) caches `uploadedDataVersion` and at frame
upload time:

1. If `data._version === uploadedDataVersion` → skip entirely.
2. Else if a partial upload is safe (`uploadedDataVersion !== -1` and the
   dirty range is non-empty) → `writeBuffer` only the dirty subrange.
3. Else (post-reset or post-resize) → upload the whole prefix
   `[0, _instanceCount)`.

The instance GPU buffer doubles capacity when needed; on resize the next upload
falls into branch 3.

`SharedAtlas.version` plays the same role for the curve+band textures.
`ensureSharedAtlasGpu(device, atlas)` lazily allocates the textures, grows them
(power-of-two rows) when capacity needs change, and re-uploads only when
`uploadedVersion !== atlas.version`. It returns `{ rebuilt, gpu }`; `rebuilt =
true` (texture object identity changed) is the signal for the renderer to drop
every draw group's `bindGroup` so it gets re-created against the new texture
views. Same-version polls during steady-state are a single integer compare.

### Spatial-band index (internal to `glyph-storage.ts`)

For each glyph, `buildGlyphBands(glyph)` partitions the curves into up to 8
horizontal and 8 vertical bands by bounding-box overlap. Curves within an
h-band are sorted by descending `max(p0x, p1x, p2x)` (and v-bands by
descending `max(p0y, p1y, p2y)`) so the fragment shader can early-exit a band
as soon as a curve is entirely to the left/below the pixel. The result is
memoized on the `GlyphCurves` object via a `@internal _bands?` field so a
glyph re-used by a second `GlyphStorage` (e.g. another text block re-extracting
from the same `Font`) pays the band-build cost only once.

### Pipeline (`_gpu/text-pipeline.ts`)

One bind group layout + one pair of base WGSL modules cached per `GPUDevice` (a
`WeakMap<GPUDevice, TextPipelineDeviceCache>`). The render pipeline itself is
cached per fixed-arity
`(colorFormat, sampleCount, depthStencilFormat, depthWrite, alphaToCoverage, variantId)`
key — so a `TextRenderable` with `ignoreDepth=true` and a `TextRenderable`
with `ignoreDepth=false` share modules + bind group layout but get separate
pipelines. Optional shader variants are supplied as already-compiled opaque module
pairs by an opt-in feature; see "Variant module lifecycle" below.

Blend is fixed src-over:

```
color: (src.a * src.rgb) + (1 - src.a) * dst.rgb
alpha: src.a + (1 - src.a) * dst.a
```

Topology is `triangle-list` (two triangles per glyph from a shared 6-vertex
unit-quad buffer). The bind group has three entries: the `TextU` UBO (mvp +
viewport + color) on binding 0, and the curve / band textures on bindings 1
and 2 (both `texture_2d<f32>` as `unfilterable-float`).

### WGSL outline

Both shader stages are direct WGSL ports of Eric Lengyel's Slug algorithm
([github.com/EricLengyel/Slug](https://github.com/EricLengyel/Slug)) — the
curve+band atlas layout, the per-pixel band lookup, the quadratic root-code
table, and the screen-space dilation math all come from that reference
implementation. The Babylon Lite shaders are the same algorithm reshaped to
fit Lite's instance-buffer + bind-group plumbing.

The vertex stage (the vertex half of `shaders/slug-shader.ts`):

1. **Dead-slot detection.** `if (in.pk == 0xffffffffu) → emit (-2,-2,-2,1) point`.
   Dead slots use an all-ones packed glyph/style index.
2. **Quad corner expansion.** `tex = mix(slugBounds.xy, slugBounds.zw, isMax)`
   maps the unit corner sign to the glyph's font-unit bounds; `pos = slugAnchor.xy + tex * scale`
   puts the corner in object-space pixels.
3. **Slug dilation** (Eric Lengyel's analytic AA expansion) — extracts MVP
   rows, computes a screen-space dilation vector `d` proportional to
   `(1 / viewport, 1 / viewport)` so glyph edges always cover one fragment.
   Also dilates the texcoord by the inverse glyph Jacobian.
4. Outputs the dilated clip position + dilated `vTexcoord` + flat
   `vBanding` / `vGlyph` / `vColor`.

The fragment stage (the fragment half of `shaders/slug-shader.ts`):

1. From `vTexcoord`, derive `(hBandIndex, vBandIndex)` via
   `bandScale * tex + bandOffset` clamped to `[0, bandMax]`.
2. Read the band's header (count + curve-list offset) from `bandTex` at
   `glyphLoc + bandIndex`.
3. Walk the curve-index list; for each curve, fetch its two `vec4`s from
   `curveTex` and solve the horizontal (and vertical for v-bands) polynomial
   to count signed crossings using the standard Loop-Blinn 3-bit root-code
   table (`0x2E74`). Early-exit when sorted-curve x/y is past the pixel.
4. Accumulate signed coverage from h-bands and v-bands and average; clamp
   to `[0, 1]`; multiply by `vColor`. Output the result.

The shader is independent of the layout / curve-extraction path — it consumes
only the packed atlas + instance buffer.

### `TextRenderable` (3D) per-frame

`TextRenderable` is a `Renderable` with `isTransparent = true`. Its `bind` is
called by the scene's frame graph; `update` does:

0. **Late-installed variant refresh.** `bind` resolves both pipelines, but scene
   bindings are built once — not per frame — so a styling feature enabled _after_
   a binding exists would otherwise leave its weighted groups drawing with the base
   pipeline until an unrelated scene mutation rebuilt the binding. The binding's
   `update` therefore re-resolves the variant pipeline exactly once, when
   `gpu._variantPipeline === gpu._pipeline && _textVariantResolver` — an identity
   comparison that a base consumer fails on its first term and that stops matching
   after the single refresh, so no cache key is ever built per frame. The
   re-resolution reuses the _bind-time_ target arguments captured in the binding
   closure (`colorFormat`, `sampleCount`, `depthStencilFormat`, `depthWrite`, and the
   renderable itself as the alpha-to-coverage owner), so the refreshed variant is the
   exact sibling of the base pipeline the binding already declares.
1. For each draw group: `ensureSharedAtlasGpu` (uploads / regrows curve+band
   textures); rebuild `bindGroup` when atlas was rebuilt or `bindGroupVersion`
   is stale.
2. Resize / re-upload the instance buffer per the version + dirty-range
   protocol above.
3. Compose MVP into a 16-float scratch from the active camera's view-projection
   × the renderable's world matrix; `writeBuffer` to the `TextU` UBO offset 0.
   Skip the recompute + upload when the world matrix is clean **and** the
   camera's `worldMatrixVersion` and effective aspect are both unchanged.
4. Write viewport size at UBO offset 64 (16 bytes) when target size changed.
   Write `(1, 1, 1, opacity)` at UBO offset 80 when opacity changed.

`draw` then iterates the draw groups: bind the group's pipeline (base, or the
pre-resolved variant when `g.groupKey !== g.curveSetId`, deduped against the
currently bound one); `setBindGroup(0, g.bindGroup)`;
`pass.draw(6, g.slotCount, 0, g.slotStart)`. There is one draw call per
non-empty group, and `slotCount` includes dead slots (which collapse in the
vertex shader). If any variant group was bound, the base pipeline is re-bound
before returning so the draw list's `setPipeline` dedupe stays valid for the next
renderable. Crucially, `TextRenderable` does **not** bind the scene's
shared scene-UBO at group 0 — it composes its own MVP so the same pipeline can
run from a `TextRenderer` with no scene at all.

### `TextRenderer` (2D) per-frame

`TextRenderer` is a `RenderingContext` registered via `registerTextRenderer`.
`startEngine` calls its `_update` (per-layer GPU sync) and `_record` (opens a
swapchain render pass and emits per-layer draws) once per frame.

The MVP for a 2D layer is built directly:

```
[ cos·(2s/W)   sin·(2s/W)   0   (2·px/W - 1)  ]
[ -sin·(2s/H)  cos·(2s/H)   0   (1 - 2·py/H)  ]
[ 0            0            1   0             ]
[ 0            0            0   1             ]
```

A 6-float `lastMvpInputs` cache (px, py, rot, scale, W, H) gates the MVP
writeBuffer so a static layer pays zero per frame after the first.
The color vec4 at UBO offset 80 is uploaded once per GPU buffer, then only when
its effective coverage-gamma reciprocal or opacity changes. Compare the
computed values with `Object.is` (including signed zero and NaN); the other
two channels remain 1. Mutations while hidden are picked up on the next
visible update. A removed/re-added layer gets a new buffer and an initial
upload; device-lost recovery replaces the buffer and invalidates this cache.
Cache values advance only after `queue.writeBuffer` succeeds.

The renderer's pipeline is cached at `(swapchain format, sampleCount=1, no
depth, depthWrite=false, flipY=false)`; depth-less means depth-hosted text-on-3D
must use `TextRenderable` instead. Layers are sorted by `order` once per frame
when there are >1; per-layer GPU records (`LayerGpu`) hold one bind group per
draw group plus an `instanceBuf` and a `textU` UBO.

### Default layout (`layout.ts`)

`layoutText(font, text, fontSizePx, options)` runs LTR shaping per paragraph
(`text-shaper.shape` returns positioned glyph clusters), greedy word wrap at
`maxWidth`, optional alignment, then bakes a flat `PlacedGlyph[]` with
pixel-space positions:

- Y is up: line 0 sits at `y=0`, subsequent lines at `y = -lineIdx * lineHeightPx`.
  This pairs naturally with the font's em-space Y-up bounds so a 3D scene with
  a Y-up camera renders text upright with no extra transform.
- `pixelsPerFontUnit = font.scaleForSize(fontSizePx)` is returned alongside;
  the `GlyphRun` consumes it and the per-instance `slugAnchor.z = 1 / scale`
  drives the vertex shader's em→pixel transform.

### Default curve extraction (`glyph-extraction.ts`)

`extractGlyphCurves(font, glyphIds, target)` walks `text-shaper.getGlyphPath`
for each requested id, converts every command (`M`, `L`, `Q`, `C`, `Z`) into
quadratic Bézier segments, and stores the result in `target`. `L` is split into
a degenerate quadratic (control point at midpoint); `C` is split into two
quadratics via the "3/4 rule" (exposed publicly as `cubicToQuadratics` for
callers that ingest their own cubic outlines from DirectWrite / FreeType / etc).

`Font._curvesCache` memoizes per-glyph extraction across calls, so a second
`createDefaultTextData` with the same font re-uses the already-rasterized
outlines.

## Dependencies

- `text-shaper` (npm) — used by `font.ts` (font loading), `layout.ts`
  (shaping), and `glyph-extraction.ts` (glyph path extraction). Only the default-layout
  / default-curves modules import it; callers using hand-rolled layout pay zero
  bytes for it.
- `engine/engine.ts` — `EngineContext`, `RenderingContext` (TextRenderer
  registration), `getRenderTargetSize`.
- `engine/render-target.ts` — `RenderTargetSignature` (TextRenderable
  pipeline key).
- `render/renderable.ts` — `Renderable`, `DrawBinding`, `DrawUpdateContext`
  (TextRenderable contract).
- `scene/scene-core.ts` — `addDeferredSceneRenderables` (`addTextRenderable`
  attachment).
- `camera/camera.ts` — `getViewProjectionMatrix`,
  `getEffectiveAspectRatio` (TextRenderable per-frame MVP composition).
- `math/observable-vec3.ts`, `math/observable-quat.ts`,
  `math/compose-mat4.ts`, `math/multiply-mat4-into-buffer.ts`,
  `scene/world-matrix-state.ts`, `scene/scene-node.ts` — TRS + world
  matrix plumbing reused from Mesh.
- `resource/gpu-buffers.ts` — `createEmptyUniformBuffer` for the `TextU`
  UBO.

## Reference scenes

- **Scene 180 — `scene180-text-renderer`**: standalone 2D `TextRenderer` +
  `TextLayer` driven by a `<textarea>` + sliders. Demonstrates pure-2D path
  (no scene, no camera), live `updateDefaultTextData`, color slider routed
  through a `replaceRun` op, drag-to-move, wheel-to-scale.
- **Scene 181 — `scene181-text-editor`**: 3D `TextRenderable` attached to an
  arc-rotate scene, driven by a `<textarea>` calling `updateDefaultTextData`
  on every keystroke. Demonstrates in-place atlas growth and instance-buffer
  reuse — typing a new character extends the atlas with that glyph's outlines
  and rewrites the run via `replaceRun`'s in-place fast path when the glyph
  count is unchanged.

Both scenes are tagged `skipParity` and `skipPerf` (interactive demos, no
golden-image oracle).

## Test specification

Unit tests live in `tests/lite/unit/`:

- `text-glyph-storage.test.ts` — covers `GlyphStorage` ownership semantics:
  `disposeTextData` does not touch the borrowed storage; one `GlyphStorage`
  backs multiple `TextData`s; `disposeGlyphStorage` is idempotent and tears
  down every curve-set's atlas; `updateGlyphStorage` extends an existing
  curve-set and creates new ones on demand; `reset` with no args performs
  compaction (re-lays-out slots and frees dead-slot gaps).
- `text-color.test.ts` — covers per-glyph `PlacedGlyph.color` overriding the
  run's `defaultColor`, and `defaultColor` propagating to glyphs that omit
  their own color.

## File inventory

| File                                         | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/text/font.ts`                           | `Font` (branded type) + `loadFont` + `createFontFromBuffer`; text-shaper boundary.                                                                                                                                                                                                                                                                                                                                                  |
| `src/text/glyph-extraction.ts`               | `extractGlyphCurves` + `cubicToQuadratics`; default text-shaper-backed glyph path extraction. Outline value types live in `glyph-storage.ts`.                                                                                                                                                                                                                                                                                       |
| `src/text/glyph-storage.ts`                  | `GlyphStorage` (branded) + `CurveSetId` + outline value types (`QuadCurve`, `GlyphBounds`, `GlyphCurves`) + the supporting `SharedAtlas` / `AtlasSlot` / `GlyphBands` types and their packers (`packAppendGlyph`, `buildGlyphBands`). Public API: `createGlyphStorage` / `updateGlyphStorage` / `disposeGlyphStorage`.                                                                                                              |
| `src/text/layout.ts`                         | `TextLayoutOptions` + `layoutText` — default LTR + word-wrap + align layout via `text-shaper.shape`.                                                                                                                                                                                                                                                                                                                                |
| `src/text/text-data.ts`                      | `TextData` (branded) + `GlyphRun` + `PlacedGlyph` + `TextDataUpdate` + slot-allocator types. Public API: `createTextData` / `updateTextData` / `disposeTextData` and the per-instance allocator (addRun / removeRun / replaceRun / reset+compaction).                                                                                                                                                                               |
| `src/text/default-text-data.ts`              | `DefaultTextData` (branded) + `createDefaultTextData` / `updateDefaultTextData` / `disposeDefaultTextData`; convenience layer composing layout + curve extraction + a private `GlyphStorage`.                                                                                                                                                                                                                                       |
| `src/text/text-renderable.ts`                | `TextRenderable` + `createTextRenderable` / `addTextRenderable` / `disposeTextRenderable`; 3D `Renderable` implementation.                                                                                                                                                                                                                                                                                                          |
| `src/text/text-renderer.ts`                  | Shared `TextRenderer` core: sorted CPU/resident layers, placement uniforms, pipelines, bundle preparation/replay and attachment lifecycle. No runtime adapter imports.                                                                                                                                                                                                                                                              |
| `src/text/text-layer.ts`                     | Common placement/options, internal prepared-draw protocol and placement helper.                                                                                                                                                                                                                                                                                                                                                     |
| `src/text/text-layer-cpu.ts`                 | `TextLayer` and its factory; per-attachment CPU uploads, palette synchronization and ordinary draw adapter.                                                                                                                                                                                                                                                                                                                         |
| `src/text/compute-text-layer.ts`             | `ComputeTextLayer` and its factory; per-attachment resident bindings and indirect draw adapter.                                                                                                                                                                                                                                                                                                                                     |
| `src/text/_gpu/text-textures.ts`             | `ensureSharedAtlasGpu`; lazy `rgba32float` texture create + version-gated upload + capacity grow. (Atlas teardown is inlined in `disposeGlyphStorage` to avoid a circular import.)                                                                                                                                                                                                                                                  |
| `src/text/_gpu/text-pipeline.ts`             | `getOrCreateTextPipeline` / `getTextPipelineCache` / `clearTextPipelineCache` + the `TextPipelineVariant` seam; per-device bind group layout + base WGSL modules + per-target-key pipeline cache. Knows nothing about shader fragments and composes no WGSL.                                                                                                                                                                        |
| `src/text/shaders/slug-shader.ts`            | **The single authoritative copy of the Slug WGSL.** Inline TypeScript template + `composeSlugShader(fragment)` — a deterministic builder that interpolates an optional `TextShaderFragment` into the base vertex/fragment source at named slots. Called with `null` for the base variant.                                                                                                                                           |
| `src/text/shaders/text-shader-fragment.ts`   | Type-only module: `TextShaderFragment`, `TextVertexSlot`, `TextFragmentSlot`. Feature-agnostic — no font-weight (or any other feature's) semantics. Erased at build time.                                                                                                                                                                                                                                                           |
| `src/text/set-font-weight-offset.ts`         | Opt-in feature entry point: `setFontWeightOffset(data, run, offset)`. Owns the per-run offset map, the interned draw-group keys, and the per-`GPUDevice` composed+compiled variant module pair; validates and resolves the run against its `TextData`, installs the `text-data` styling seam and the `text-pipeline` variant resolver on first effective call, then repacks the data through `updateTextData({ update: "reset" })`. |
| `src/text/load-font-weight-offset.ts`        | Root-exported lazy loader for the exact `set-font-weight-offset.ts` implementation module. Application code can keep the feature behind a dynamic boundary without importing the package barrel as a namespace.                                                                                                                                                                                                                     |
| `src/text/shaders/weight-shader-fragment.ts` | The weight-only `TextShaderFragment`: distance-to-quadratic helper, bounded nearest-contour band scan, weight varying, quad inflation, and unsigned-distance coverage expansion. Contains **no** copy of any base Slug logic. Reachable only from `set-font-weight-offset.ts`.                                                                                                                                                      |

---

## Slug shader composition

### One template, many variants

There is exactly **one** copy of the base Slug vertex and fragment logic, and it lives
in `src/text/shaders/slug-shader.ts` as an inline TypeScript template. There are no
`slug.vert.wgsl` / `slug.frag.wgsl` `?raw` files, and there is no second "weighted"
copy of the shader: optional features contribute _incremental_ WGSL through a typed
fragment that the template interpolates at named slots.

```typescript
export function composeSlugShader(fragment: TextShaderFragment | null): ComposedSlugShader;

export interface ComposedSlugShader {
    /** @internal Composed vertex WGSL. */ readonly _vert: string;
    /** @internal Composed fragment WGSL. */ readonly _frag: string;
    /** @internal Fragment id ("" for the base variant) — folded into the pipeline cache key. */
    readonly _key: string;
}
```

This mirrors the material stack's `ShaderTemplate` + `ShaderFragment` + `composeShader`
model (`src/shader/fragment-types.ts`, `src/shader/shader-composer.ts`), but is a
compact text-specific builder rather than a second instance of the generic material
composer: text owns a fixed custom `@group(0)` layout (uniform + curve texture + band
texture + glyph-metadata storage + style storage) with no mesh/material UBO, no shadow
group, no vertex-attribute negotiation and no UBO layout computation, so none of the
generic composer's machinery applies.

Interpolation is **direct** (`${...}` into the template literal), not marker
substitution and not a regex pass over emitted WGSL. Production bundles minify inline
WGSL template literals (`scripts/wgsl-minify-plugin.ts` → `renderChunk`), which strips
`//` comments and collapses whitespace; anything that depended on comment markers or on
parsing the emitted string would break there (see GUIDANCE "Never parse emitted WGSL
strings"). A builder that concatenates known strings is immune.

### Injection slots

`TextShaderFragment` is a pure-type description (`src/text/shaders/text-shader-fragment.ts`):

```typescript
export type TextVertexSlot = "VO" | "VD" | "VB" | "VA";
export type TextFragmentSlot = "FI" | "FH" | "CO";

export interface TextShaderFragment {
    /** @internal Stable id — part of the pipeline cache key and the shader module label. */
    readonly _id: string;
    /** @internal */ readonly _vertexSlots?: Partial<Record<TextVertexSlot, string>>;
    /** @internal */ readonly _fragmentSlots?: Partial<Record<TextFragmentSlot, string>>;
}
```

| Slot | Stage    | Location                                                                     | Contract                                                                                                                                                                                                                                                                                     |
| ---- | -------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VO` | vertex   | End of the `VOut` struct body                                                | Comma-terminated member declarations. Locations `0..3` are taken by the base varyings, so fragments start at `@location(4)`.                                                                                                                                                                 |
| `VD` | vertex   | Dead-slot early-return block                                                 | Statements assigning a defined value to every member the fragment added, on the local `d: VOut`.                                                                                                                                                                                             |
| `VB` | vertex   | After `md` (glyph metadata) and `sy` (style) are in scope, before quad setup | Statements that **must** declare `let sb: vec4<f32>` — the shaped em-space glyph bounds. When the slot is present the template uses `sb` in place of `md.b`; when absent the template names `md.b` directly, so the base variant pays nothing. Values declared here stay in scope for `VA`.  |
| `VA` | vertex   | Just before `return out;`                                                    | Statements assigning the fragment's added varyings on `out`.                                                                                                                                                                                                                                 |
| `FI` | fragment | End of the `FIn` struct body (before `@builtin(front_facing)`)               | Comma-terminated member declarations mirroring `VO`.                                                                                                                                                                                                                                         |
| `FH` | fragment | Module scope, after the base helpers, before `@fragment fn main`             | Helper function / constant declarations. The base helpers (`rcode`, `solveH`, `solveV`, `ccov`, `cwgt`, `bloc`) and the `ct` / `bt` textures are already in scope, so a fragment that needs its own scan of the glyph's band lists writes it here as a self-contained function.              |
| `CO` | fragment | After base `cov` is computed and clamped, before the coverage-gamma `pow`    | Statements that may read and reassign `var cov: f32` (coverage in `[0,1]`). In scope: `rc` (em-space render coord), `pe` (pixels-per-em), `gp` (glyph texel origin), `bm` (max band indices), `in`. The gamma, the `a2c` premultiply select, and the final color write stay in the template. |

There are deliberately **no** per-curve slots inside the base band loops. Those loops
`break` as soon as a curve is behind the pixel along the band's sort axis, and they only
ever visit the single band that contains the pixel — both correct for coverage, both
wrong for anything that needs _nearest-contour_ information (see "Why the weight fragment
owns its own scan"). A feature that needs a different traversal writes its own bounded
scan in `FH` rather than piggy-backing on a loop whose exit condition it does not control.

Slot ids are short because they are WGSL-adjacent identifiers in a size-sensitive path;
they are enumerated as a union type, so a typo is a compile error rather than a silently
ignored injection.

### Base variant is free

`composeSlugShader(null)` emits exactly the base shader: no extra varying, no helper, no
coverage override — the slot expressions collapse to `""` and the bounds expression names
`md.b` directly. Unweighted (base) draw groups therefore execute behaviourally identical
shader code to the pre-feature engine, with zero extra fragment-shader work.

### Variant module lifecycle

`src/text/_gpu/text-pipeline.ts` knows nothing about shader fragments and never composes
WGSL. It accepts an already-compiled, opaque module pair:

```typescript
/** @internal Opaque compiled shader-module pair for one composed text shader variant. */
export interface TextPipelineVariant {
    /** @internal Stable id — the variant field of the pipeline cache key. */
    readonly _id: string;
    /** @internal */ readonly _vertModule: GPUShaderModule;
    /** @internal */ readonly _fragModule: GPUShaderModule;
}

/** @internal Installed by an opt-in text styling feature; resolves that feature's compiled
 *  module pair for a device. Null until a feature installs one. */
export let _textVariantResolver: ((device: GPUDevice) => TextPipelineVariant) | null;
/** @internal */ export function _installTextVariantResolver(resolve: (device: GPUDevice) => TextPipelineVariant): void;
```

- **The feature owns composition and compilation.** `set-font-weight-offset.ts` calls
  `composeSlugShader(WEIGHT_SHADER_FRAGMENT)` and `device.createShaderModule` itself, and
  memoizes the resulting `TextPipelineVariant` in a lazily created
  `WeakMap<GPUDevice, TextPipelineVariant>` (no module-level allocation, auto-invalidating
  per device — GUIDANCE §4). The core cache holds no variant module fields at all, so a
  second feature can never overwrite or inherit another's stale modules.
- **One styling feature at a time.** The resolver is a single nullable slot; installing a
  second feature replaces the first. The group-key seam (below) and the variant resolver
  are installed together by the same feature, so they cannot disagree.
- The pipeline cache key is **fixed arity**, six `:`-separated fields, every field always
  present:
  `format : sampleCount : depthStencilFormat|"-" : "w"|"r" : "a"|"-" : variantId|"-"`.
  There is no optional field and no delimiter alias, so a base A2C pipeline
  (`…:a:-`) can never collide with a variant whose id happens to be `"a"` (`…:-:a`).
- Alpha-to-coverage stays a **pipeline-overridable constant** (`@id(0) override a2c`) on
  whichever module the pipeline uses — one module still serves both the blended and the
  A2C pipeline for each variant.

### Pipeline selection per draw group

`getOrCreateTextPipeline` returns both pipelines for a target signature:

```typescript
export interface TextPipelineSet {
    /** @internal Base Slug pipeline. */ _pipeline: GPURenderPipeline;
    /** @internal Composed-variant pipeline; **aliases `_pipeline`** when no styling feature
     *  is installed, so draw paths can bind it unconditionally. */ _variantPipeline: GPURenderPipeline;
    /** @internal */ _cache: TextPipelineDeviceCache;
}
```

Resolution happens once per bind (`TextRenderable`) / once per frame update
(`TextRenderer`) and is stored on the per-renderable / per-layer GPU record — never inside
the draw loop, which does no map lookups and builds no key strings. A `TextRenderable`
binding additionally re-resolves its variant pipeline once if a styling feature was
installed _after_ the binding was built (see "`TextRenderable` (3D) per-frame" step 0);
the `TextRenderer` path needs no equivalent because its `_update` already resolves both
pipelines every frame. Draw paths that only need the shared quad buffer or bind-group
layout call `getTextPipelineCache(engine)` instead, which is a single `WeakMap` get.

A draw group selects between the two with one identity comparison,
`g._groupKey === g._curveSetId` (see "Draw-group keys"), tracks the currently bound
pipeline locally, and — in the `TextRenderable` path, which shares an encoder with the
scene's draw list — restores the base pipeline before returning so the caller's
`setPipeline` dedupe stays valid for the next renderable. The `TextRenderer` path records
into a render bundle whose state does not outlive `executeBundles`, so it needs no restore.

### Tree-shaking

- `text-shader-fragment.ts` is types only → zero runtime bytes.
- `slug-shader.ts` (base template + builder) is reached from `text-pipeline.ts`, so every
  text consumer carries it — exactly one copy, as before.
- `weight-shader-fragment.ts` is imported **only** by `set-font-weight-offset.ts`. A
  consumer that does not import `setFontWeightOffset` never reaches it, so the
  distance-to-quadratic WGSL, the band scan, the weight varying and the coverage override
  are absent from the bundle. `tests/lite/build/text-shader-fragment-treeshake.test.ts`
  asserts both directions against the real shipped `build/lib` output.
- `text-data.ts` carries no font-weight semantics. It knows only an optional per-run style
  parameter and an opaque draw-group key whose default is the run's own `CurveSetId`.

---

## Opt-in feature: synthetic font-weight offset

### Purpose

A synthetic contour offset (emboldening) per `GlyphRun`, evaluated
analytically in the Slug fragment shader. This is **not** CSS font-weight face selection
(100–900); it is a non-negative contour offset in font design units that thickens the
rendered outlines by expanding them outward. Round
contour joins fall out of the distance field automatically — no miter/bevel logic.

The feature owns all of its cost at its enabler: importing `setFontWeightOffset` pulls in
the weight shader fragment; calling it with an offset that actually changes a run installs
the two core seams and composes the variant shader. Consumers that never import the setter
pay zero shader bytes and zero shader work.

### Public API

```typescript
export function setFontWeightOffset(data: TextData, run: GlyphRun | number, offset: number): void;
export function loadFontWeightOffset(): Promise<typeof setFontWeightOffset>;
```

`loadFontWeightOffset()` is the application-owned lazy-boundary form. Its implementation
dynamically imports only `text/set-font-weight-offset`, never the package root namespace,
so bundlers do not conservatively retain unrelated root exports in the lazy chunk.

| Parameter | Type                 | Description                                                                                                                                                                                       |
| --------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`    | `TextData`           | The `TextData` that owns the run. The setter mutates live data: it repacks and regroups `data` immediately, exactly like a setter on a live material.                                             |
| `run`     | `GlyphRun \| number` | The run to weight, as either its `GlyphRun` reference or its current index in `data.runs`. Must be a live run of `data`; anything else throws with the same message shape `updateTextData` uses.  |
| `offset`  | `number`             | Contour offset in **font design units** (the space of `GlyphCurves.bounds`). Positive = bolder. Must be finite; clamped to `[0, 100]` with a `console.warn`. Zero clears a previously set offset. |

```typescript
import { setFontWeightOffset, createDefaultTextData, loadFont } from "@babylonjs/lite";

const font = await loadFont("/fonts/Inter.ttf");
const data = createDefaultTextData(font, 48, "Bold text");
setFontWeightOffset(data, data.runs[0]!, 10); // +10 font units
setFontWeightOffset(data, 0, 10); // identical: runs may be addressed by index
setFontWeightOffset(data, 0, 0); // back to the base pipeline
```

It takes the owning `TextData` — not a bare `GlyphRun` — because runs are packed
**synchronously**. `createTextData` / `createDefaultTextData` pack their runs before they
return, and `updateTextData` packs each edit as it is applied, so a setter that only
recorded state on a detached run descriptor could never affect data that already existed;
on a `DefaultTextData` (whose single run is created and packed inside the factory) there is
no moment at which such a call could be made at all. Owning the `TextData` lets the setter
repack and regroup on the spot, so the call works at any point in the data's life,
including after it is already bound and rendering.

### Setter lifecycle

1. **Validate the offset.** Non-finite → `console.error`, return. Nothing is installed and
   nothing is repacked.
2. **Resolve the run against `data`.** `_resolveRunRef(data, run, "setFontWeightOffset")`
   (`text-data.ts`) accepts a `GlyphRun` reference that is live in `data` or an in-range
   index into `data.runs`, and throws otherwise — same wording as the `updateTextData`
   errors (`… : run index N out of range (0..M).` / `… : GlyphRun reference is not in this
TextData.`). `updateTextData`'s own index path shares this helper, so the two can never
   drift.
3. **Clamp** to `[0, 100]` with a `console.warn` when out of range.
4. **No-op detection.** If the resolved run's current effective offset already equals the
   clamped value, return before touching anything: no seam install, no shader composition,
   no `_version` / `_layoutVersion` churn. A zero call on a run that never had an offset is
   therefore a _true_ no-op — it does not install the feature or compile its shader — while
   the first nonzero call always installs.
5. **Capture** the `WeakMap`'s exact prior entry for this run — whether it had one at all,
   and its value — then **store** the clamped offset (zero deletes the entry), then
   **install the seams** (`_installTextStyleSeam` + `_installTextVariantResolver`),
   idempotently.
6. **Repack** via `updateTextData(data, { update: "reset" })` — the narrowest existing
   TextData operation that both re-reads every run's draw-group key and rewrites every
   style entry's `params.y`. `reset` with no `runs` keeps the current run objects (it
   defensively copies `data._runs`), compacts the slot allocator, and reuses the previous
   draw-group objects per key, so a run moving between the base and weighted groups is a
   regrouping — not a rebuild of the caller's descriptors. The feature deliberately
   re-implements none of the allocator, grouping or style-packing logic.
7. **Roll back on failure.** If step 6 throws (e.g. the run's curve set is no longer in
   `data`'s storage), the `WeakMap` entry written in step 5 is restored to its exact prior
   state — deleted if the run had no entry before this call, set back to the previous value
   if it did — before the error is rethrown unchanged. Without this, a failed repack would
   leave the map at the _new_ offset while the data itself was never actually repacked; a
   caller that fixes the failure and retries with the same offset would then hit the no-op
   guard in step 4 and get silent non-repair instead of a retry. The seams installed in step
   5 are **not** rolled back — they are idempotent, side-effect-free until a run actually
   uses them, and re-running `ensureInstalled` on a retry is a no-op, so leaving them
   installed after a failed call is harmless.

Zero therefore both clears the offset and returns the run's group to the base key (and the
base pipeline) in the same call.

### `DefaultTextData` and run replacement

`createDefaultTextData` packs its single run before returning, so the setter is usable
immediately: `setFontWeightOffset(data, data.runs[0]!, 10)` (or `setFontWeightOffset(data, 0, 10)`).

`updateDefaultTextData` re-shapes the text into a **new** `GlyphRun` descriptor and applies
it with `replaceRun`. Offsets are keyed by run identity, so they do **not** transfer to the
replacement run: the block returns to the base pipeline until the caller re-applies
`setFontWeightOffset(data, 0, offset)` on the new run (index form is the convenient one
here, since the reference changed). This is deliberate — a generic "carry per-run feature
state across replaceRun" seam would put feature semantics back into `text-data.ts` and cost
every text consumer bytes for a transfer no base consumer can use. The same holds for any
caller-driven `updateTextData({ update: "replaceRun" | "reset", runs })` that swaps run
descriptors.

### The core styling seam

`text-data.ts` exposes exactly one nullable seam object and one opaque type:

```typescript
/** @internal Opaque draw-group identity, compared by `===` only. The default is the run's
 *  own `CurveSetId`; a styling feature may substitute an interned non-string token, which
 *  by construction can never equal a curve-set id. */
export type TextGroupKey = CurveSetId | object;

/** @internal */
export interface TextStyleSeam {
    /** @internal Draw-group key for a run — `run.curveSet` unless the run needs a different
     *  pipeline variant. Must never return a value that equals another curve set's key. */
    _key(run: GlyphRun): TextGroupKey;
    /** @internal Extra style float, packed verbatim into `TextStyle.params.y`. */
    _param(run: GlyphRun): number;
}

/** @internal */ export let _textStyleSeam: TextStyleSeam | null;
/** @internal */ export function _installTextStyleSeam(seam: TextStyleSeam): void;
```

`text-data.ts` attaches **no** meaning to either result: `_param` is a float it copies into
`params.y`, `_key` is a token it compares with `===`. Both call sites are single
optional-chained calls with a neutral default (`_textStyleSeam?._key(run) ?? run.curveSet`,
`_textStyleSeam?._param(run) ?? 0`), and only scalars / opaque values cross the boundary.
All font-weight semantics live in `set-font-weight-offset.ts`.

`text-data.ts` additionally exports one feature-agnostic helper so per-run feature setters
resolve and validate their run argument exactly as `updateTextData` does:

```typescript
/** @internal Resolve a `GlyphRun | number` reference against `data`'s live runs. `op`
 *  prefixes the thrown message. */
export function _resolveRunRef(data: TextData, ref: GlyphRun | number, op: string): GlyphRun;
```

It is used by `updateTextData`'s own index path, so the two cannot drift, and it is
tree-shaken away for consumers that import no feature setter.

### Style packing

`writeRunToSlots` calls `_param` once per run and writes the result into
`TextStyle.params.y` (previously reserved/zero; `params.x` still carries `invScale`). No
struct or stride change. The value is compared through `Math.fround` like every other style
float, so an unchanged offset does not bump `_styleVersion`.

### Draw-group keys

`TextDataDrawGroup._groupKey: TextGroupKey` replaces the pre-feature implicit "one group per
curve set" identity. Grouping, previous-group reuse across a `reset`, `addRun` lookup and
`replaceRun`'s in-place fast path all compare **that one field by identity** — the flat
`Map<TextGroupKey, GlyphRun[]>` / `Map<TextGroupKey, TextDataDrawGroup>` shape, allocation
profile and iteration order of the pre-feature code are preserved exactly (one array per
group, no nested maps, no per-curve-set array pairs). `_curveSetId` remains on the group for
atlas lookup, bind-group labelling and error messages.

- With no styling feature installed, `_groupKey === _curveSetId` for every group and the
  grouping algorithm is byte-for-byte the pre-feature one.
- `set-font-weight-offset.ts` returns `run.curveSet` for an unweighted run and an interned
  **object** token (one per `CurveSetId`, from a lazily created `Map<CurveSetId, object>`)
  for a weighted one. An object can never `===` a string, so a curve set whose id happens to
  look like another group's variant key cannot collide — the pathological `"X"` / `"X:w"`
  pair that a delimited composite string key would conflate stays in distinct groups
  (regression-tested).
- Base and weighted runs on the same curve set land in **separate** draw groups, because
  they need different pipelines. Weighted runs with **different** nonzero offsets batch into
  **one** group — the offset travels per style entry in `params.y`.
- `g._groupKey !== g._curveSetId` is exactly "this group needs the variant pipeline", so the
  draw paths need no extra per-group state.

### Weight shader fragment

`src/text/shaders/weight-shader-fragment.ts` exports one `TextShaderFragment`
(`_id: "w"`), containing **only** incremental code:

| Slot | Contribution                                                                                                                                                            |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VO` | `@location(4) @interpolate(flat) wo:f32,` — the per-instance weight offset varying.                                                                                     |
| `VD` | `d.wo=0.0;` — dead-slot default.                                                                                                                                        |
| `VB` | `let wo=sy.p.y; let sb=vec4<f32>(md.b.xy-vec2<f32>(wo),md.b.zw+vec2<f32>(wo));` — inflates the font-space bounds symmetrically so the quad covers the expanded contour. |
| `VA` | `out.wo=wo;`                                                                                                                                                            |
| `FI` | `@location(4) @interpolate(flat) wo:f32,`                                                                                                                               |
| `FH` | `dot2` + `dq` (exact distance to a quadratic Bézier) + `wdst` (the bounded nearest-contour band scan).                                                                  |
| `CO` | Zero-offset guard, unsigned distance, weighted coverage, and monotone `max` finalization.                                                                               |

### Why the weight fragment owns its own scan

The base band walk is a _winding_ query, not a _nearest-contour_ query, and its two
optimizations are only sound for winding:

1. It reads exactly **one** h-band (the band containing the pixel's `y`) and one v-band.
   That is complete for a `+x` / `+y` ray cast, because a ray from the pixel stays inside
   its own band. It is **not** complete for distance: the nearest contour to a pixel near a
   band boundary is frequently in the adjacent band.
2. Inside a band it `break`s as soon as a curve's `max x` is more than half a pixel behind
   the pixel (curves are sorted by descending `max x`). Every remaining curve is behind the
   ray, so it cannot cross it — but curves to the _left_ of the pixel are exactly the ones
   that can be nearest to it.

Accumulating a running minimum distance inside those loops therefore silently skips curves
left of and below the pixel, which clips positive (emboldening) offsets on the right and
top edges of a glyph and on all four corners. The weight fragment consequently declares its
own complete, bounded scan in `FH` and reads nothing from the base loops:

```wgsl
fn wdst(rc:vec2<f32>,gp:vec2<i32>,bm:vec2<i32>,bn:vec4<f32>,rad:f32)->f32
```

- **Search radius.** `rad = wo + 1/aaScale` font units, where `aaScale = max(pe.x, pe.y)` is
  pixels-per-font-unit. Beyond that radius the weighted coverage saturates to 0 (outside) or is
  dominated by the base coverage (inside), so a curve farther than `rad` cannot change the
  result.
- **Complete band range.** Every h-band spans the full glyph width in `x` and partitions
  `y` (`glyph-storage.ts` → `buildBandsInternal`), and a band holds every curve whose
  `y`-extent intersects it. `wdst` therefore iterates **all** h-bands intersecting
  `[rc.y - rad, rc.y + rad]`:
  `y0 = clamp(i32((rc.y - rad) * bn.y + bn.w), 0, bm.y)`,
  `y1 = clamp(i32((rc.y + rad) * bn.y + bn.w), 0, bm.y)`.
  Any curve with a point within `rad` of the pixel has that point inside the glyph's
  `y`-bounds, so its band index is inside `[y0, y1]` and the curve is in that band's list.
  _Validated band-transform invariant:_ `bandScaleY = hBandCount / heightFu` when
  `heightFu > 0` and `0` otherwise, and `bandOffsetY = -yMin * bandScaleY`, so the transform
  is monotone non-decreasing and never negative. A zero scale collapses the range to band 0,
  which is the only band a zero-height glyph has. No sign guard is required; a unit test
  pins the invariant in the builder.
- **Radius-aware break only.** Within a band the scan still exploits the descending-`max x`
  sort, but with the radius-aware bound `curveMaxX < rc.x - rad`. A curve with a point
  within `rad` has `maxX >= rc.x - rad`, and every curve before it in sort order has a
  `maxX` at least as large, so no in-range curve is ever skipped.
- `dq` is the exact point-to-quadratic distance: a degenerate straight-segment fast path
  when `dot(b,b) < 1e-7`, otherwise the closest-point depressed-cubic solve with `t` clamped
  to `[0,1]` (Cardano for one real root, trigonometric form for three).

### Shader math (the `CO` slot)

```wgsl
if(in.wo!=0.0){
  let aas=max(max(pe.x,pe.y),1.0e-8);
  let d=wdst(rc,gp,bm,in.bn,in.wo+1.0/aas);
  let wc=clamp((in.wo-d)*aas+0.5,0.0,1.0);
  cov=max(cov,wc);
}
```

1. **Zero-offset guard.** A run whose offset is zero — including a stale style entry left in
   a variant group — takes no distance scan at all and keeps the base coverage bit-exactly.
2. **Base Slug coverage runs unchanged** and remains authoritative inside the original fill.
3. **Unsigned contour distance.** Positive-only emboldening needs no inside/outside
   classification: `wdst` is the distance to the nearest outline in either direction.
4. **Offset threshold.** Coverage decreases with distance, and the offset pushes
   the boundary outward: `wc = saturate((wo - d) * aaScale + 0.5)`, a ~1px screen-space
   transition.
5. **Monotone finalization.** Emboldening may only add coverage:
   `cov = max(cov, wc)`. This keeps the analytic base coverage authoritative inside the
   original fill, while the distance field contributes only its outward expansion. An
   overestimated `wdst` can never punch holes into the original glyph.
6. The template then applies the coverage gamma and the `a2c` premultiply select exactly as
   for the base variant.

`aaScale` is floored at `1e-8` so a degenerate derivative (`fwidth == inf`) cannot produce a
non-finite radius; the coverage transfer function is unaffected for any real glyph.

### Limits

Offsets are clamped to `[0, 100]` font design units; out-of-range values are clamped with a
`console.warn`, and non-finite values are rejected with a `console.error`. The proportional
effect depends on the font's `unitsPerEm`: for a 2048-unit font, `100` is about `0.049 em`.
The bound exists because large offsets expand the glyph quad into its neighbours and
lengthen the per-pixel band scan (its cost is proportional to the number of bands the
radius spans).

### Test specification

`tests/lite/unit/text-font-weight-offset.test.ts`:

- **Setter semantics** — attach / overwrite / clear-with-zero on a live `TextData`; `NaN`
  and `Infinity` rejected with `console.error` and no repack; values clamped to `[0, 100]` with
  `console.warn`; setting the value a run already has is a no-op (`_version` and
  `_layoutVersion` unchanged), and a zero call on a never-weighted run installs nothing.
- **Run resolution** — the index form (`setFontWeightOffset(data, 0, …)`) is equivalent to
  the reference form; an out-of-range index and a `GlyphRun` that belongs to another
  `TextData` both throw with the `updateTextData`-shaped message.
- **Post-create mutation** — a setter call made _after_ `createTextData` (i.e. after the
  runs are already packed) changes the run's draw-group key and its `params.y` immediately;
  a following zero restores both to the base key and `0`.
- **Style packing** — a weighted run's `params.y` holds `Math.fround(offset)`; a base run's
  is `0`.
- **`DefaultTextData` flow** — the setter works on `data.runs[0]` immediately after
  `createDefaultTextData`, and after `updateDefaultTextData` replaces the run the offset
  does not transfer but re-applying it by index does.
- **Variant grouping** — one base + one weighted run on the same curve set produce two groups
  with the same `_curveSetId`, one keyed by the curve set and one by an interned token; two
  weighted runs with different offsets share one group; `addRun` / `replaceRun` / `removeRun`
  respect the key.
- **Group-key collision regression** — the `"X:w"` / `"X"` pair stays in distinct groups with
  the correct curve set, on create and across a `reset` (group object identity is reused per
  key).
- **Band-scan completeness (algorithmic)** — a JS replica of `wdst`'s candidate selection,
  driven by the real `buildGlyphBands` output for a synthetic square glyph, must consider the
  true nearest curve for sample points on all four exterior sides and all four corners. The
  pre-fix "accumulate inside the base loop" candidate set is asserted to _fail_ the same
  check, so the regression cannot silently return.
- **Band-transform invariant** — `buildGlyphBands` + the packed metadata always produce a
  non-negative, monotone band transform (what `wdst`'s range clamp relies on).
- **Composition** — `composeSlugShader(null)` contains no weight varying, `dq`/`wdst` helper
  or coverage override, and its band loops contain no injected statements; the composed
  variant contains all of them; the base source is a subsequence of the variant at every
  significant line, proving composition rather than duplication. `_key` is `""` for the base
  and the fragment id for the variant.
- **Coverage direction** — a JS replica of the `CO` transfer function proves positive offsets
  expand and that `max` finalization can never remove analytic base coverage.
- **Pipeline cache key** — the exported key builder is fixed arity and a base A2C pipeline
  cannot collide with a variant whose id is `"a"`.

`tests/lite/unit/text-renderer-variant-pipeline.test.ts`: a mocked device + render-bundle
encoder records a layer holding base, weighted and base draw groups and asserts the exact
`setPipeline` / `setBindGroup` / `draw` command sequence (base → variant → base), that a
base-only layer records exactly the pre-feature command sequence, and that the variant
pipeline is resolved during `_update`, not per draw. The same file covers the
`TextRenderable` late-install path in module isolation (`vi.resetModules()` + dynamic
import, so no earlier test has installed the resolver): a renderable is **bound first**,
its binding `update` runs with no styling feature installed (variant pipeline aliases the
base), the setter is then called on its live `TextData`, and the next binding `update`
must refresh `gpu._variantPipeline` to a distinct pipeline built from the same target
signature — after which the recorded draw sequence is base → variant → base.

`tests/lite/build/text-shader-fragment-treeshake.test.ts` (Rollup over the shipped
`build/lib`): a consumer importing text rendering but not `setFontWeightOffset` retains the
base Slug source and **not** the weight fragment's WGSL; a text consumer that also imports
`setFontWeightOffset` retains both, with the base Slug logic still declared exactly once.

## Opt-in compute shaper support and lifetime

The public compute path is `loadComputeFont` / `createComputeFontFromBuffer`, `extractComputeGlyphCurves`, and `shapeComputeText(engine, font, text)`. `shapeComputeText` takes characters and returns glyph ids plus font-unit positions. Script selection, lookup plans, and the GPU shaper stay internal. This does not replace `DefaultTextData`, `layoutText`, or the text-shaper-backed renderer. A consumer that never calls these functions retains neither the new parser nor the compute shader. The compute path does not import text-shaper.

Shaping matches text-shaper 0.1.28 for the admitted one-direction runs the compiler can execute, including Inter Greek `Αθήνα`, Inter Cyrillic `Москва`, Hebrew `שלום` (visual clusters, vav advance 291, zero xOffset), and Arabic `كتب` (five records, including mark offsets). A lookup that matches but is not executable fails the run. It is not skipped. Device loss rejects the pending job and does not publish stale records. The caller does not own a shaper object. Engine teardown releases the internal one, and an already returned glyph list or `TextData` stays valid. One batch is active per engine. Shaping never mutates `TextData`.

## GPU-resident compute text

### Production ordered-index and immutable-preparation contract (2026-10-06)

The compiler's eight-word lookup descriptor retains every selected root and
dependency, flags, action order and fail-if-matched rejection. Word 6 now points
to a directory containing two words per font glyph: an absolute candidate-list
origin followed by its count. Entries are original sixteen-word rule indexes,
strictly in original subtable/rule order. Enumerate the complete first predicate:
glyph lists include all listed font glyphs; class predicates enumerate the full
font domain, including the class-zero complement, intersecting any coverage.
Neither output repertoire nor a particular input text restricts this directory.
Empty lookups still have an empty directory. Rejections remain candidates in
their original position. Nested Single execution uses its own descriptor and
the current target glyph, without adopting the parent's lookup flags.

Root traversal reads only the current glyph's candidates, so predicate zero is
already proved. Matching still checks all other input predicates, filtered live
successors, backtrack and lookahead. Nested execution remains the compiler's
admitted direct Single semantics. GSUB output mutation, deletion/compaction,
multiple expansion and cursor advancement, GPOS application, original-lookup
failure attribution and direction reversal remain unchanged.

The separate metrics/GDEF/class buffer has its existing six-word header and
class-origin directory. Each class origin addresses exactly `font.glyphCount`
class IDs, with class zero for absent definitions/ranges. `classOf` is one
indexed read, not a format/range loop. Advance and GDEF tables retain their ABI.

Font construction copies its bytes; the snapshot and default plans are immutable
internal inputs. Lazy weak caches own decoded lookup actions per font/table/index,
default plans per font/script/direction, and prepared immutable words per plan
identity. Validation occurs before plan-cache lookup. A different font, script,
direction or explicit internal plan identity cannot reuse another program.
The immutable preparation includes the compiled semantic program, table buffer
and appended table-tagged root order. Every update independently packs the
current glyph IDs, scalar clusters, run count, capacity and shared output origin.
No cache contains shaped output or memoizes strings.

Each private public-readback schedule retains successful shader preparation
in its cache's `taskPrepared` flag. Prepare a new shader variant once; warm submissions
reuse it without another async preparation turn. Shader/scratch variant
replacement resets preparation. Buffer growth retaining a prepared shader
does not need another pipeline preparation; the new binding set is validated
and built synchronously during recording. The latch belongs
to the device/plan/owner-bound cache, not to general compute tasks. Submission
still validates the active task, pipeline and binding resources, and generation/
device checks remain before submission and after completion. A failed or
cancelled initial preparation never marks the schedule prepared.

Readback and resident execution share immutable preparation, never a CPU shaper.
GPU caches remain device/owner-bound and are released on disposal or terminal
loss. Resident program/table upload is skipped only when the currently uploaded
plan identity and storage-buffer identities are unchanged; multi-selection
updates upload each changed group in submission order. Buffer growth invalidates
upload identity. Closure borrows cached decoded actions but computes the complete
current conservative repertoire and outline components independently.

Regression tests must prove directory exhaustiveness/order and class-zero
semantics, dense class equality, one compile on stable warm revisions,
font/selection isolation and current mutable words. Native tests execute actual
production WGSL through the root APIs, preserving exact six-field oracle output,
resident f32/slot/style/indirect ABI and four-byte-only status reads. The observed
Roboto natural-Latin50 N512 incomplete-output regression must pass for every run;
N1024/N2048 admission and resident growth/shrink are additional bounded checks.
Performance acceptance uses matched public and resident endpoints separately,
not this structural contract as a surrogate for measured PoC parity.

The bounded 32-lane kernel can finish compiler-proven nominal/first-only pair
runs before initializing mutable glyph scratch. It remains one pipeline,
dispatch and submission; there is no pending status or extra preflight kernel.
The global-scratch variant executes the complete program. Each bounded invocation
initially sets private `nominalSource = true`, so shared `readGlyph` and `live`
read immutable nominal inputs. The leader checks overflow and computes the
first/witness root intersection. More than 32 roots always uses the complete
executor.

For each potentially active GSUB root, prove that no original input/context
predicate matches using the original ordered directory, membership, GDEF
filtering and traversal. An actual match, including an identity substitution
or matched rejection, requires the complete executor. Root bits are visited in
ascending order using `countTrailingZeros`. Exclude only roots proven not to
match; a failed proof returns the original candidate mask. Potentially active
GPOS roots must all have compiler capability 3 and an available compact pair
directory. Otherwise every invocation switches `nominalSource` off, initializes
the original mutable scratch, and executes the complete kernel unchanged.

Admitted runs retain nominal IDs, count, scalar-index clusters and zero offsets.
Lane `l` positions source indices `l, l+32, ...`; each advance is the nominal
advance plus supported first-only pair deltas in ascending original root order.
Compact operations preserve candidate/subtable ordering, class-zero coverage,
absent cells and matched rejection cells. Traverse successors using only
implemented GDEF filtering bits `0xff0e`: bit 16 alone does not filter under
text-shaper 0.1.28. Original flags and `pairAppliesSecond` compiler policy remain
unchanged. `skipped` returns before GDEF loads when no implemented filtering
bits are present. RTL reverses output positions, not clusters.

An atomic minimum of `(root << 8) | sourceIndex` preserves root-first rejection
priority across lanes. This encoding is exact only for bounded capacity <=256.
After a storage/workgroup barrier the leader publishes the original table,
lookup and glyph error, or status 0 and the complete count. Overflow retains
status 2. Partial records from a failed run are never published as success.

The original version-2 semantic program, table-tagged roots, first masks and
witness masks remain intact. Append optional GPU-only pair metadata at
`program.words.length + rootCount*2 + glyphCount*2`: `rootCount` absolute directory
origins, zero for non-first-only GPOS roots, followed by compact directories.
Each directory has two words/glyph `[vectorOrigin, operationCount]`. Each
five-word operation is either `[0, secondPredicateOrigin, 0, deltaBits, status]`
or `[1, secondClassTableIndex, rowLength, rowOrigin, 0]`. Class row cells are
`[deltaBits, status]`, with status 0 absent, 1 supported match, 2 rejection.
Immutable vectors and rows share original identities. The builder rejects any
unexpected non-pair/non-rejection action in a compiler-proven root.

Kernel assembly remains in `src/text/_gpu/compute-text-shape.ts`.
`compute-text-predicates.ts` supplies shared advances, GDEF filtering,
membership, pair selection and input/context traversal WGSL.
`compute-text-simple-wgsl.ts` supplies exact nominal proof and lane-striped
pair output. `compute-text-pair-program.ts` constructs the optional immutable
compact tables from the existing semantic program. CPU shaping imports none
of these GPU-only modules.

Preparation retains both the full words and a base view excluding this optional
footer. Readback and resident pass the real device buffer/binding limits; when
only the footer would exceed either limit, upload the base view and use complete
semantics rather than narrowing admission. Oversized semantic resources still
reject through existing bounds checks. Shader proof checks footer bounds and
nonzero directory availability before reading compact data. Owner/device/plan
caches, preparation, disposal and terminal-loss behavior remain unchanged.
No CPU-shaped result, input memoization, font/string exception or relaxed
validation is permitted.

The bounded leader retains each glyph's first-root mask in workgroup
`nominalMasks` during the initial nominal presence scan, then skips GSUB proof
candidates whose cached mask lacks that root. These masks are necessary
conditions, not independent shaping decisions; all potentially matching
predicates still receive exact traversal.

Packed program version 2 shares class coverage: a class predicate is three
words `[0x80000000 | classId, classTableIndex, coverageOrigin]`. A zero origin
means unrestricted coverage; otherwise the origin addresses `[count, ...glyphIds]`
packed once per parsed coverage object. Glyph predicates keep `[count, ...ids]`.
All logical rule descriptors and exhaustive ordered candidate entries remain
intact, so the independent semantic executor and original rejection witnesses
do not need a second compiler or a rewritten shaping contract.

A contiguous class-pair row also has an acceleration matrix in descriptor
words 10–13: `[rowOrigin, secondClassTableIndex, rowLength, rowCandidateCount]`.
The row is indexed by the second glyph's dense class; entries are original
rule index plus one, or zero for an absent/zero-adjustment cell. Every supported
or rejected cell retains its original mode/action and descriptor. The kernel
can visit the row once, resolve the exact original cell, and advance past that
row's candidate segment. Other subtables and explicit pairs keep original
precedence. A missing second glyph, absent class or zero cell does not consume
the lookup; matching continues with later subtables. Rejection cells still fail
with the original lookup/table/run attribution.

Resident warm outline admission reuses the caller curve-set's already installed
curves. A lazy font-keyed glyph-reader cache retains immutable TrueType contours,
bounds and component IDs (including valid empty glyphs), shared by extraction
and conservative closure. New glyph IDs still validate/decode transactionally
before target mutation, and atlas growth/slot upload retain their existing
ownership and resource-identity handshake. This caches font data, not run output.

Cooperative execution uses **32 lanes on one independent run**, not 32
independent runs per workgroup. Descriptor word 2 is a compiler capability:
1 for GSUB rules containing only input-one Single/rejection actions with no
backtrack/lookahead; 2 for GPOS input-two Pair/class-pair/rejection actions with
no contextual predicates; 3 for those same pair roots when every executable
action has an unapplied or exactly zero second advance; otherwise 0. Empty
lookups may inherit their table's pure capability. All other roots, nested
actions, marks, ligatures and multiple
substitutions retain the same ordered leader executor.

Initialization with nominal advances and final six-field output writes
are lane-striped. Glyph replacement also updates its nominal advance; newly
inserted glyphs receive their own nominal advance. GSUB never observes or
positions these fields, so every live glyph enters GPOS with the same current-id
advance and zero offsets as the former whole-run transition. Ligature and
multiple outputs maintain this invariant, including nested substitutions.
There is no redundant first-GPOS initialization pass or its extra fence.
For a pure Single root, lanes apply disjoint substitutions and
retain any matched rejection descriptor. For a pure Pair root, lanes select each
live first glyph's earliest original candidate and filtered second glyph from
unchanged glyph IDs. The leader then commits actions or the first rejection in
source order, preserving second-advance accumulation exactly. Capability 3
instead commits only each lane's own first advance during selection; no lane
writes the second glyph, even an addition of zero. The leader still publishes
the earliest rejection in source order. Already committed Single/first-only
Pair successes clear their selection slot, so the leader does not re-fetch
their global rule descriptors. This is a compiler proof over every original
rule, not an inference from a particular string or a format flag.
Any possible nonzero applied second advance retains capability 2 and its
ordered commit. Two bounded workgroup arrays hold selected rules and successor
indexes. Above 256 expansion slots these arrays remain bounded to 256 entries:
cooperative roots select a contiguous tile in parallel, synchronize, and let
the leader commit that tile in source order before reusing its slots. A
mode-5 rejection publishes the first matching source index and prevents the
next tile; capability-2 second advances retain ordered integer additions.
Single and first-only Pair successes retain their disjoint lane commits.
Capability-0 roots retain the complete ordered executor. Barriers and
workgroup-uniform length/failure loads separate
all mutation, selection and commit phases; compaction/reversal remain ordered.
The same production kernel serves readback and resident endpoints.

Immutable GPU preparation appends one `u32` per font glyph after the two-word
root records. Bit `r` proves a nonempty original candidate directory for root
`r`, for the first 32 roots only. Enumeration includes rejection rules and
class-zero complements. The leader ORs these words over current live glyph ids.
A zero bit permits bypassing that root's candidate traversal and fences;
roots beyond 31 always retain the complete executor. Nominal advances already
hold before any GPOS bypass. Actual GSUB glyph-id changes,
insertions and ligature tombstones atomically mark presence dirty, including
nested actions. After ordered compaction, the leader recomputes dirty presence
before the next root and clears the marker; substitutions therefore activate
their subsequent roots. Unmatched or identity-preserving roots retain presence.
No input-specific program, feature omission or root reordering is introduced.

The compiled program retains `rootWitnesses: Uint32Array`. A second
font-glyph-indexed mask follows the first-glyph masks in GPU preparation. For each GSUB
rule, choose the smallest exhaustive glyph domain among its non-leading input,
backtrack and lookahead predicates. Its presence somewhere in the live run is
necessary (not sufficient) for that rule to match. Union the chosen domains
over all original rules, including rejection rules. If any rule has no such
predicate, the root's witness bit is unconditional; GPOS witness bits are also
unconditional. Class predicates use the same exhaustive class/coverage
enumeration as the candidate compiler, including class-zero complements.
The root mask is `OR(firstMasks[currentIds]) & OR(witnessMasks[currentIds])`.
Test its bit before fetching any root record or lookup descriptor.
This avoids traversing composition/ligature candidates when their required
combining or contextual glyphs are absent, without dropping any rule. Roots
above 31 remain unconditionally executable. The existing GSUB identity/live
membership dirty marker refreshes both masks after nested substitution,
insertion and compaction, so introduced witnesses activate later roots.
Readback and resident use identical immutable words and unchanged scheduling.

The reference program ABI stays version 2; this is immutable GPU preparation
metadata at `params.programWords + params.rootCount * 2`.

Unicode admission caches the complete immutable scalar classification, not only
its Unicode Script tag. A lazily allocated sparse directory indexes 256-entry
`Uint8Array` pages by `scalar >>> 8`: zero is uncached, 1 is Common, 2 is Inherited,
3–9 are the admitted scripts in Latin/Greek/Cyrillic/Hebrew/Arabic/Han/Gothic
order, 10 is unknown, 11 is unsupported, and 12–16 are control/paragraph/bidi/
joining/variation rejections. Cold classification tests the existing rejection
ranges before reading Unicode 16.0.0 Script. Warm decoding compares numeric
classes and converts the resolved strong script once per run. Rejection wording,
scalar attribution and deferred UTF-16 priority are unchanged; unsupported tags
are recovered only for their error message. This bounded Unicode property cache
contains neither input strings nor decoded runs or shaped outputs.

The selector has a package-internal font/seed overload, removed from the public
declaration by `@internal`. In the default offset-preserving mode it reserves a
third disjoint slab view for nominal glyph ids. During decoding it copies only positive checked entries below that
font's glyph count and collects distinct adjacent nominal seeds into the current
update's set. Missing entries remain zero and are counted. UTF-16 and script
validation still finishes first; the existing mapping stage then resolves those
misses with original caller error attribution, after the same plan-validation
boundary as before. A vector carries its immutable font identity, so a different
font cannot reuse it. All-warm vectors require neither a second scalar-mapping
loop nor a separate resident seed traversal. Nothing caches a run or its output.
After the first cold miss, decoding defers all remaining seed insertion.
The miss-resolution loop resumes insertion from that position, preserving
first-seen seed order, closure witnesses and cold atlas slot installation order.
Within a single decode, an adjacent equal scalar reuses its preceding immutable
Unicode admission and checked nominal page entry before script dispatch and
font-bound checks, rather than repeating already-proven admission for that
scalar. Missing nominal entries still increment the miss count on every
occurrence; all misses are resolved with the original error/seed-order policy.
Every UTF-16 unit is still
decoded/validated and every current scalar, requested offset, glyph and cluster is written
to fresh input/scratch storage. This scalar-property reuse applies uniformly to the
entire Unicode domain; there is no character-specific path or string/output
memoization. The preceding property values never survive the call.

Within a decode, retain the current 256-scalar page index and the immutable
nominal/admission page references. Refresh directory references only when
the decoded scalar enters a different page. Each different scalar still reads
its own cell; uncached admission still uses the full Unicode classifier and
refreshes the page reference after lazy allocation. Missing nominal cells,
adjacent repetition, cold seed order, mixed-script/control diagnostics and
deferred lone-surrogate priority retain their existing behavior. No decoded
input, glyph vector or string is retained across calls.

The public executor's mutable input buffer reserves `runCapacity *
(glyphCapacity + 2) * 4` bytes: one run-directory word, one count word and at
most `glyphCapacity` nominal glyph words per run. No redundant initial-cluster
word is reserved. Shorter warm runs reuse this full retained capacity, while
input revisions and counts are always repacked.

Root readback batches, direct-plan batches and resident updates allocate one
fresh backing slab sized to the validated snapshot's summed UTF-16 lengths
times eight bytes. These shaping-only callers explicitly omit offset metadata:
they consume scalar-index clusters and never read UTF-16 offset vectors.
The internal selector overload reserves disjoint scalar/nominal regions in
source order and uses a shared zero-length `_utf16Offsets` sentinel for this
mode. It still decodes every UTF-16 unit and computes exact error positions;
lone-surrogate and deferred script/control error priorities are unchanged.
Run views never overlap and a subsequent
update never reuses the backing buffer. Standalone public selection still owns
its separate scalar/offset slab with complete offsets. Internal selection and
batch slabs default to including offsets unless the caller opts out; that
default font-aware layout retains twelve bytes per source UTF-16 unit.
This removes unconsumed metadata, not input validation or mutable scalar/
nominal writes; the cursor is owned only by that batch.
Resident grouping compares the checked script/direction fields directly and
collects the fresh prepared records themselves. Their scalar/nominal views are
already the GPU input contract, so grouping does not allocate replacement
two-field descriptors or per-run compound key strings. Groups remain
contiguous and retain source-order output offsets, including empty DFLT runs.

Resident visible-count validation and ordered prefixing share one 256-lane
workgroup, the WebGPU minimum guaranteed invocation/X-size capacity:
for at most 256 runs, lane `l` validates/counts run `l` into the retained
offset buffer, then all lanes execute a storage barrier. Lane zero runs the
unchanged source-ordered failure/capacity checks and prefix, publishing status
and indirect count. A following parallel packing dispatch consumes the offsets.
Above 256 runs, the shared count helper executes over 64-lane workgroups,
followed by a 256-lane, contiguous-tiled prefix dispatch and parallel packing. This
preserves count-stage concurrency rather than serializing a large batch on
one workgroup. The retained packing schedule rebuilds only on buffer identity
or a crossing of this count tier; dispatch dimensions still refresh each update.
Counts remain complete for arbitrary admitted run counts. Every fused lane
reaches its fence, including lanes with no runs. Failed batches keep zero
indirect instances and the original first-run failure. Shape, packing stages
and four-byte completion remain one submission for a single plan group.

The public batch wrapper selects each input once, retaining that checked
selection through grouping and glyph mapping. The private execution helper
accepts these selections only from that validated wrapper; direct internal-plan
calls still validate their own strings. Every executor batch must contain
exactly one result for each dispatched nonempty run, or the wrapper rejects
instead of substituting an empty output. Production decode allocates fresh owned
records in an array reserved to the checked published count, filling by index
without repeated backing-array growth. The public wrapper may publish them without a redundant object/array
clone. The test-executor seam and direct plan API still snapshot foreign arrays,
preserving isolation. All glyph IDs, clusters and signed safe-integer fields are
validated before publication; no check is skipped to reduce CPU work.
The four signed position fields use direct safe-integer checks, not a newly
allocated four-number array for each published glyph.
The dynamically imported executor function is cached only after successful
module loading. Warm calls use that function directly rather than resolving
the already-loaded module and yielding again before input submission. The
test-executor seam always takes precedence. This lazy cache owns no device,
font, plan or buffers: all resource/loss checks remain in the executing owner.
An import failure leaves it empty so a later call may retry.

For capacity at most 256, the same source factory rounds required capacity up
to the next power of two and sizes glyph scratch to eight words per slot and
selected-rule/successor/nominal-mask arrays to one word per slot. The run-local base is zero.
Thus a 50-glyph dispatch reserves 2,816 bytes plus state, not the maximum
11,264 bytes. Together with state, this fits the
WebGPU minimum workgroup-storage limit. Larger capacities use the original
per-run global scratch binding and storage fences. The factory emits the
correct declaration/base/fences directly from a typed capacity decision; it
does not parse or rewrite emitted WGSL. All remaining shader logic is shared.
Shape pipeline identity changes only when growing beyond its current bounded
variant or crossing the bounded/global boundary. Shrinking reuses a sufficient
variant. Readback retains immutable program/table/parameter buffers and a
compatible shader while replacing mutable buffers on batch/capacity growth.
Readback and resident use one common shape-shader constructor.
Existing buffer growth, output stride, device replacement and terminal-loss
ownership remain unchanged.

**Length-independent fast-path contract (specified before implementation,
2026-10-08):** Global scratch does not disable nominal admission, simple-run
proof or fused first-only positioning. During proof `readGlyph` reads immutable
packed input and `live` applies the same skip predicates without scratch reads.
Bounded variants retain their glyph-mask cache; global variants read the
immutable per-glyph program mask directly. Successful proof writes the same
six fields directly, including scalar-index clusters and RTL reversal, without
initializing scratch. The compact Pair directory/capability-3 requirement and
root-count limit remain unchanged. Any GSUB match or inadmissible GPOS root
restores the **entire original candidate mask**, never a partially filtered one.

For long input runs, 32 lanes OR first-glyph presence and witness masks into
separate workgroup atomics and intersect the results. They then stripe each
candidate GSUB root's no-match proof over input indexes; a per-root atomic
failure flag forces the same all-or-nothing fallback. Short input runs retain
the serial leader proof even when a long neighbor forces global scratch.
The input-length threshold is performance-validated, not a semantic boundary.
Dirty root presence after ordered GSUB mutation retains its exact existing
leader recomputation. Successful simple-run rejection reduction first selects
the earliest root, then the earliest matching index for that root using
separate atomics; no eight-bit glyph-index encoding is valid for long runs.
Bounded variants retain the original `(root << 8) | index` reduction and do
not emit long-run atomics, proof helpers or extra tile barriers. Only global
scratch input lengths above 256 use parallel admission; this threshold retains
the measured short-run schedule.
For parallel proof, the leader resets the per-root failure atomic, lanes
stripe candidate matching, then synchronize before the leader publishes a
uniform proof result: 0 for a GSUB no-match, 1 for fallback, 2 for admitted
GPOS. Only result 0 clears that root's bit; result 1 returns the original
candidate mask. Presence atomics are initialized before any lane scans input.

Tiled cooperative fallback uses uniform length and terminal loads so every
lane reaches every tile barrier. Global scratch uses storage fences between
selection, ordered commit and the next tile/root. Local scratch continues to
use its power-of-two arrays and the single-tile schedule. The largest bounded
variant, including eight-word scratch, mask/selection/successor arrays and
scalar/atomic state, stays below the default 16-KiB workgroup-storage limit.
There is no public API, program ABI, output stride, overflow/rejection encoding,
placement, ownership or prefix change. Resident shared-owner capacities and
readback both use the same constructor across growth, shrinking reuse and
bounded/global transitions.

**Parallel-selection / ordered-commit contract (specified before implementation,
2026-10-08):** Capability-0 roots share two functions with their serial fallback.
`selectRule(base, count, index, flag, ruleOrigin, directory) -> u32` is read-only:
zero means no terminating candidate; otherwise the result is the original
sixteen-word rule address plus one. It preserves candidate-directory order,
class-pair row indirection and row-candidate skipping. Rejections select their
original rule, not a successful no-op. Multiple substitution selects only a
nonempty output. Pair selection requires the original second-glyph existence
and explicit-pair ID test. Mark selection requires a present base and anchor;
mark-ligature probing also preserves the contour-anchor rejection outcome.
Context, Single, SinglePos and ligature select after the complete input,
backtrack and lookahead predicate succeeds. A ligature cursor cannot fail after
that predicate: both walks use the same `nextLive` sequence, and commitment
only tombstones previously visited components, never its future components.

`applySelected(base, count, index, flag, ruleAt, out, lookup) -> vec2u` returns
consumed input-cursor steps and the highest written index. It does not re-match
predicates. Attachment probes may be repeated to recover base/anchor addresses;
the actual offsets are read at commitment time, including offsets written by
an earlier attachment in this same lookup. Rejection/overflow still publish
the unchanged terminal status, lookup word and first source index. Single,
SinglePos and attachment writes end at their current index; pairs end at their
second index; ligatures end at their last component; context ends at the
maximum in-range nested action index. Multiple insertion reports
`0xffffffff` because it shifts all later positions. Serial execution uses
exactly these functions, advances past inserted glyphs, and retains ordered
compaction and root-presence refresh.

Descriptor word 2 retains capabilities 0–3 unchanged. The formerly reserved
word 7 encodes compiler-proved ordered eligibility: 0 means serial-only,
1 means tile-safe, 2 means insertion-containing (bounded scratch only).
Bit 2 additionally proves a GSUB root has no backtrack predicates: its selection
cannot read below its current index. Eligibility is in the low two bits.
GSUB admits only modes 1/2/5/9/11; GPOS only 3/4/5/7/8/10/12. Every rule must
be admitted, and only capability-0 roots use this new schedule. No runtime
rule scan infers eligibility. The retained threshold is run length greater than
32: always and greater-than-32 improve marked50 timestamps, whereas
greater-than-64 misses that gain. Two-character controls stay on the existing
leader schedule. The naive parallel schedule with a complete leader traversal
did not pay; the bitmap schedule is the retained implementation.

For an eligible root above the threshold, all 32 lanes select from unchanged
scratch, storing rule addresses in `selected` and the lowest inspected index
in `seconds`. Selection-only private tracking starts at the current index and
records every inspected position for backtracking GSUB, including failed
candidates and skipped or tombstoned glyphs in `nextLive`. Forward-only GSUB
has exact lowest read equal to its current index and needs no tracking;
GPOS needs no tracking because the field-disjoint proof below is sufficient.
Attachment probes are read-only, including complete mark-ligature scans. A zero
candidate directory or non-live index still records its current-index read.
After a barrier, lane 0 commits indexes ascending and preserves the serial
consumed cursor. It tracks a write frontier: none initially, otherwise the
maximum index written since this snapshot. A GSUB selection is reusable iff
there is no frontier, or both its current index and its lowest read exceed
the frontier; otherwise the leader re-selects from current state. Bounding
all writes above by the frontier and all selection reads below by the lowest
read proves disjointness, even for failed selections and contextual backtrack.
Insertion makes the frontier unbounded, forcing every later re-selection.
Bounded insertion can increase the cursor past the original snapshot; indexes
outside the snapshot also always re-select. Eight bounded atomic words record
nonzero selections, one bit per snapshot index. For GPOS, or a forward-only GSUB
cursor strictly beyond the write frontier, the leader can skip zero-selection
spans via `countTrailingZeros`: those selections are proved valid and cannot
have become matches. For backtracking GSUB, positions at/below the frontier,
or positions outside an insertion snapshot, it instead visits every position.
The bitmap is reset before each tile selection; it is never a replacement for
the lowest-read validity proof. This removes serial empty-directory traversal
without omitting any index whose selection could have changed.

GPOS selections are always reusable: modes 3/4 read IDs/live flags and write
only advances (fields 2/3); mode 10 writes only fields 2–5; modes 7/8/12 select
using IDs, live flags and stored glyph classes (fields 0/6/7), program anchors
and immutable GDEF, never positioning fields. Mode 5 has no writes. Mark-mark
inherited offsets and mark-ligature/base offsets are deliberately read only
at commitment. Thus every admitted GPOS mode has disjoint selection-read and
mutation-field sets, including overlapping pairs and stacked marks.

Above 256 capacity, selection uses contiguous 256-slot tiles and storage
fences. Each tile takes its fresh snapshot after previous commitment; its
frontier resets, while the serial consumed cursor carries across tiles.
Cross-boundary ligature tombstones are visible to the next snapshot. Roots
containing mode 11 retain the serial executor in this scratch mode; no shifted
index is mapped to a stale tile. Workgroup-uniform length/terminal publication
separates all phases, and rejection prevents the next tile. Bounded scratch
uses the same selection and commitment logic with one snapshot and the same
eight-word bitmap. No output record, resident packing, RTL reversal or API changes.

Mode-8 base discovery scans backward from `index-1`, stopping at the first
stored class other than 3. This is exactly the last such position found by
the former complete forward scan. Mode 7 still requires the immediately
preceding stored mark class. Stable upper-bound anchor lookup, including
last-duplicate precedence, remains unchanged. Root refresh and compaction
remain serial. Their substantial Roboto attribution is a remaining bottleneck,
not a claimed gain of this change; parallel refresh/compaction was not included
in this bounded implementation. Native and counterbalanced actual indirect-draw
measurements retain the ordered path for repeatable marked-run improvements,
without a repeatable greater-than-0.1-ms median penalty on short controls.

**Parallel post-GSUB refresh contract (specified before implementation, 2026-10-08):**
After commitment, lane 0 publishes a plain workgroup `refreshFlags` word:
bit 0 is exactly `!isGpos && !cooperative && removed`; bit 1 is exactly
`!isGpos && atomicLoad(&idsChanged) != 0u`; bit 2 snapshots `terminal != 0u`.
It resets `idsChanged` there. All lanes uniformly load the flags after the
existing root storage fence, replacing the former terminal uniform load
without adding a barrier on unaffected roots. `refreshRun(base:u32, lane:u32,
flags:u32)` skips work when bits 0–1 are zero, then the caller honors bit 2.
Cooperative substitutions rescan but never compact. Earlier global
cooperative/ordered terminal exits remain before publication; local and
global serial terminal exits remain after refresh, exactly as before.
Initial nominal-source root discovery and nominal masks are unchanged.

Above a measured, uniformly selected count threshold, stable compaction uses
eight atomic bitmap words, reused from ordered selection under the neutral
name `indexBits` only after commitment has finished. In each 256-index tile,
clear the words, barrier, and atomically set one bit per live record (field 6
equals zero), then barrier. Lane 0 publishes `refreshStart` (the number of
unshifted, fully live leading chunks), uniformly loaded before movement.
Every lane derives the same running word prefix from popcounts plus the
destination base across tiles; no private eight-element prefix/bitmap arrays
are needed. Process 32-index
chunks ascending. Read each live record's complete eight words into private
registers before any lane writes its destination. Pipeline the next chunk's
reads before the current chunk's writes, with a workgroup barrier and a
storage barrier on global scratch between chunks. Destination is the word
prefix plus the count of lower live bits. Skip unchanged destinations and
unshifted, fully live leading chunks. Lane 0 publishes the final live count.
All reads precede potentially overlapping writes within a chunk; destinations
never exceed source indexes, and earlier-chunk destinations precede
later-chunk sources. Thus neither in-chunk nor cross-chunk/tile overwrite
can destroy an unread record. Ascending live-bit ranks preserve exact order.

Rescan strides live scratch records by 32 lanes, ORing the same immutable
first/witness masks into two workgroup atomics, declared in both variants.
Lane 0 resets these before the accumulation barrier, then publishes their
intersection after accumulation. Uniform root-presence publication orders
the following root. Serial refresh remains at counts up to 32, with a storage fence after its
leader writes even when a short run shares oversized global scratch.
The retained parallel threshold is greater than 32: greater-than-64 loses
marked50 benefits, while always adds work to two-character controls.
Measured fusion is rejected: Roboto medians were flat and Inter mixed.
Only the separate exact compaction and rescan ship; every survivor, including
unmoved records, contributes through the rescan's original liveness test.
Native/oracle and eight-word synthetic differentials cover both scratch
variants, chunk/tile boundaries, dense and whole-chunk tombstones, gained/lost
presence, RTL and terminal paths. Measured on Apple M2 Max native Dawn,
Roboto 480-glyph English shaping passes fell from about 1.9 ms to 1.0 ms and
Roboto 4096×50 marked from 2.69 ms to 2.36 ms, with two-character controls
unchanged.

**Mark-anchor lookup contract (specified before implementation, 2026-10-08):**
The ordered fallback's mark-base/mark-mark action keeps its existing
`[markClass, markX, markY, baseCount, ...[glyphId, x, y]]` ABI. Pack base
triples in stable ascending glyph-ID order. This is immutable program
preparation, not CPU shaping or run-dependent work. The shader uses an
upper-bound binary search and checks the preceding triple for equality.
For repeated glyph IDs the last original triple wins, exactly as in the
previous exhaustive scan; missing bases still leave the action unapplied.
Nearest-base selection, mark-to-mark ordering, signed anchor arithmetic,
inherited offsets, advance clearing, lookup/candidate precedence, rejection
and complete bounded/global execution are unchanged. Both resident and
returned-record endpoints use this same compiler/kernel contract.
Acceptance requires six-field/native and actual resident draw-fence comparisons
against committed `519e4aa6`, including marked 50-scalar and short controls.
Controlled shader omissions are attribution proxies, not additive timings or
correctness-preserving candidates.

**Bounded scheduling experiments (2026-10-08; not production contracts):**
Compare 4 independent eight-lane teams and 2 independent sixteen-lane teams
inside a 32-invocation workgroup. Only nominal runs proven by the existing
exact GSUB/first-only-pair proof finish there. Each team owns its masks,
count/proof and atomic root/index failure; all 32 invocations traverse the
same barriers, including inactive partial-final teams. A subsequent unchanged
ordered interpreter dispatch consumes only GPU-marked pending runs. No CPU
readback or run-content shaping selects the path. Count/prefix/pack execute
once after completion. The extra preflight/fallback dispatch overhead is part
of the comparison and disqualifies a nominal GPU-only improvement if the
ordinary update-through-draw endpoint regresses.

Separately assess contiguous capacity buckets within each selected-plan group:
split only caller-defined runs at the <=256/>256 expansion-capacity boundary,
never split text. Retain shared output stride and original outputRun coordinates,
but choose sufficient bounded scratch for a short chunk rather than forcing it
through its long neighbor's global variant. Each chunk uploads/prepares/dispatches
its own inputs before those shared buffers are overwritten; packing still runs
once in original source order. Compare preparation/submission overhead, a moving
long run among short runs, and all-short controls. These isolated experiment
executors must not be described as the normal production import. Promote only
after exact-field/lifecycle tests and repeatable actual-draw gains; otherwise
preserve source and evidence outside the shipped library.

**Scheduling decision and validation (2026-10-08):** Retain only stable-sorted
mark-anchor binary lookup. Two ordinary source-pinned AB/BA screens show
marked 50-scalar update-through-draw medians of Inter 100 runs
1.8→1.2/1.1 ms and 4096 runs 15.4/15.7→9.8/10.1 ms; Roboto 100 runs
1.5→1.1 ms and 4096 runs 13.9/14.1→10.3/10.4 ms. This is not a universal
latency improvement: short and mixed controls are mostly flat, and a late
short-only bracket regresses by 0.1/0.2 ms at 4096 runs. Preserve those tails
and counterexamples rather than claiming all workloads improve.
Neither independent-team variant establishes repeatable end-to-end gains;
the contiguous-bucket candidate regresses mixed cases by roughly 0.1–0.3 ms,
while the unavoidable global long run still dominates. Both remain outside
production. `tests/lite/compute/mark-anchor.test.ts` executes the production
action at empty, absent, first, last and duplicate anchor boundaries, with
negative inherited offsets and both mark modes. Compiler/native and emitted
root consumer tests retain all six record fields, complete fallback and
resident ownership/packing. Exact evidence is in session
`files/shaping-efficiency-20261008/`; no public API or prefix-executor change
belongs to this round.

All packed glyph-set predicates and shared coverage lists are sorted by glyph
ID (predicate order is set membership, not action precedence). Remaining input,
backtrack and lookahead membership uses bounded lower-bound binary search;
singleton glyph tests and shared class coverage use the same search helper.
This eliminates linear scans of wide contextual coverage without removing any
predicate or weakening first-match/rejection order. Full native accepted/rejected
corpora remain the semantic oracle for this membership change.

Warm GPU ownership includes the binding set, direct dispatch and recorded
compute task, not just uploaded font words and a pipeline. Readback keeps one
schedule per existing owner/device/plan/buffer cache and updates its direct run
count. Resident keeps separate shape and pack schedules while their input and
borrowed draw-buffer identities remain stable. Shader variant changes or any
referenced buffer replacement rebuild only the affected schedule. Preparation
still validates current resources through the compute framework on every
submission. Disposal/loss disposes cached tasks as well as owned shader/buffer
resources; no new methods are attached to public state.

Private shaping readback records prepared compute tasks and their completion
copy in one encoder/submission. Its opt-in internal compute helper reuses the
owned storage allocation's staging buffer and synchronously materializes the
result while mapped, unmapping in `finally`; no mapped view escapes. General
`readStorageBuffer` retains its independent-copy/coalescing contract unchanged.
Readback copies only the active record stride/count. Resident submits the last
shape group, pack and four-byte status copy together. Earlier groups still
submit before overwriting shared input words, preserving multi-script order.
All resource validation, active-frame exclusion, disposal/loss checks and
original rejection attribution remain effective.

Root synchronization follows data dependencies rather than repeating fences:
the prior root's uniform terminal load publishes leader commits and run length.
Only the GSUB→GPOS transition adds a fence after parallel advance assignment;
parallel selection has its own fence before ordered leader commitment. Global
scratch additionally uses storage fences. Only non-independent GSUB execution
can introduce tombstones/expansion and therefore compacts. Independent Single
roots, the GPOS transition and final publication do not rescan an already
compact run. All 32 lanes still participate in every required fence, including
rejection paths.
Ordered GSUB roots additionally compact only when a successful ligature has
created tombstones. The leader resets a private `removed` flag on entering an
ordered root; ligature commitment sets it. Multiple substitution inserts live
records directly and does not require a redundant full-run copy. Empty
candidate lists are checked before live/flag traversal, matching the complete
directory's proved admission without changing any eligible candidate/rejection.

**Production prefix transfer contract (specified before implementation, 2026-10-07):**
Apply the experiment's integer-prefix pattern to the existing resident run
metadata, not on guessed word boundaries. Public APIs, shaping, grouping,
glyph ordering, integer pen accumulation and f32 placement remain unchanged.
Keep the existing fused serial prefix for at most 256 runs. For larger batches,
use one 256-lane prefix workgroup: lane `l` owns the contiguous range
`[l * ceil(runCount/256), min((l+1) * ceil(runCount/256), runCount))`.
Each lane reduces its current visible counts and earliest failing run. Scan
these 256 `vec2u` summaries with private pending values and barriers before and
after overwrites; combine counts with saturation at instance capacity plus one
and failing indexes with `min`. Saturation is solely an overflow diagnostic:
successful offsets/counts are exact integers, never saturated.
The 12-byte instance stride bounds instance capacity below `2^32/12`, so
adding two saturated summaries cannot overflow `u32`. The admitted shape
binding similarly bounds each run's count; no widened floating-point sum or
unchecked count is used.

Before any metadata rewrite, the leader compares the earliest shape/slot error
with the earliest instance overflow. Locate the first overflowing summary by
lower-bound search, then walk only that lane's original range to locate the
first overflowing run. Capacity failure precedes another failure at the same
run, exactly as in the serial loop; any earlier run's original status wins.
Publish `[6,0,0,0]` and that unchanged status on failure. Broadcast the decision
with a uniform workgroup load before any lane returns or writes offsets. On
success, each lane uses the previous inclusive summary as its exclusive base
and rewrites its own original counts to compact offsets in source order;
publish `[6,totalVisible,0,0]` and status zero. Workgroup storage is fixed at
2 KiB plus one status word, independent of admitted batch size. This is
scalable contiguous tiling, not a 1,024-run limit or cross-workgroup fence.
There is no added dispatch, host readback, allocation or public option.
Adoption requires complete hardware ABI/error tests and a bounded matched
production before/after comparison; page experiment ratios are not evidence
for this transfer. The two independent accepted production comparisons and
separate pack-stage timestamp diagnostic are recorded in
`text-blocks-profile-results.md`. Small batches keep their unchanged serial
prefix; no evidence supports replacing that tier.

Resident packing shares `countRun`, which validates shape status/count and atlas
slots and writes visible count plus first failure to a private `vec2u` metadata
buffer, and `prefixOffsets`, which traverses runs in source order, computes
compact offsets, checks instance capacity, and publishes the first failure or
`[6,totalVisible,0,0]`. For at most 256 runs, `countRuns` fuses these helpers in
one 256-lane workgroup. Larger batches dispatch `countParallel` over 64-lane
workgroups, then `prefixRuns` with the single 256-lane workgroup specified above.
`main` (64 invocations/workgroup, one
invocation per run) places its records at the assigned offset with the existing
integer pen accumulation and exact f32 scaling/line/style ABI. Empty outlines
advance the pen but create no instance. A nonzero status disables every packing
invocation and leaves indirect instance count zero. Metadata is owned by the
resident shader cache and grows/invalidate bindings like other owned buffers.
There is still one normal four-byte map after the packing stages, no shaped
record map or CPU glyph positioning.
Shared resident capacity is computed from immutable plan expansion and current
scalar lengths without building mutable input words. Those words are packed
exactly once per actual plan group, after the shared stride is known.
The mutable input ABI begins with one word offset per run. Each offset points
to `[scalarCount, ...nominalGlyphIds]`; the current contiguous `Uint32Array`
glyph vector is bulk-copied into fresh packed storage. Initial clusters are
always source scalar indices, so the shader writes cluster `i` while reading
glyph `runs[inputOrigin + 1 + i]`, rather than uploading redundant cluster words.
Supplementary scalars therefore retain scalar-index, not UTF-16-index clusters.
Empty runs contain only the zero count. Both readback and resident paths share
this packing/initialization; glyph scratch, subsequent GSUB cluster operations,
output ABI, immutable program ABI version 2, and all bounds checks are unchanged.

Warm ingestion also reuses immutable per-scalar data, never a shaped result.
Each font lazily owns 256-scalar `Uint32Array` nominal-mapping pages, with
`0xffffffff` meaning not computed and zero retaining the normal unmapped
failure. The Unicode range-table result is lazily memoized in 256-scalar pages
by selection. Both are bounded by the Unicode scalar domain (4,352 pages);
page directories are lazy sparse arrays indexed by `scalar >>> 8`, not
per-scalar hash-map lookups. Unallocated pages remain absent.
invalid scalar arguments never address nominal pages. Run decoding, UTF-16
offsets, control checks, mixed-script checks and unmapped/font-bound diagnostics
are recomputed in their original order. Font ownership isolates nominal pages;
Unicode properties are immutable and device-independent. No import-time
allocation or font/GPU back-reference is introduced.
Run mapping copies already checked nominal page entries directly into fresh
glyph-id arrays. Typed entries prove integer representation; positive glyph
ids still must be below the current immutable font's glyph count. Missing,
zero, sentinel or out-of-font entries use the full scalar mapper and original
caller-specific error attribution. No input or shaped output is memoized.
Decoding owns one fresh buffer of `text.length * 8` bytes. Scalars occupy its
first UTF-16-upper-bound-sized half and UTF-16 offsets its second half; the
disjoint `Uint32Array` views cannot mutate each other's elements. Supplementary
input returns length-bounded views, and subsequent inputs own different buffers.
Script/control classification is performed as scalars are decoded, without a
second traversal. The first classification error is deferred until all UTF-16
units have been checked, so any lone-surrogate error still outranks an earlier
control/script error exactly as before. After the first classification failure,
decoding continues but later classification cannot replace that failure.
Resident repertoire collection inserts mapped glyph ids directly into a
current-update set; closure receives each seed once, with no intermediate
per-run numeric-array copies. Closure and all mutable run records still reflect
the current input.

### Native M1–3 correctness acceptance matrix

`pnpm test:compute-text-gpu` opts into Node/Dawn (`webgpu` 0.6.1), independently
of mocked `unit` tests. It installs native globals only in that Vitest project;
an unavailable binary, adapter, compilation failure, validation error or unexpected
device loss is a failure, never a skip or interpreter fallback. The default native
backend is platform-selected (Metal on macOS); `LITE_GPU_BACKEND=vulkan` selects a
real Vulkan backend for software-driver CI. `null` is forbidden. Test devices and
all block/storage allocations are released at teardown. Tests run serially with
bounded timeouts and reuse a device within each suite.

Version 0.6.1 is pinned because 0.6.2 was younger than the repository's seven-day
dependency release-age gate when this project was added; the gate is unchanged.

The oracle is installed `text-shaper` **0.1.28**, using the exact SHA-256-checked
Inter, Roboto, Noto Sans SC subset, Gothic, Hebrew and Arabic font bytes. Each
accepted string compares count and every `[glyphId, scalarCluster, xAdvance,
yAdvance, xOffset, yOffset]` record exactly. Positive/negative expectations are
declared before dispatch; compiler rejection cannot turn a positive into a skip.
Seeded sequences supplement explicit repertoire and contextual witnesses rather
than substituting case counts for behavior coverage.
Oracle script/direction configuration comes from independent Unicode Script
property witnesses and explicit admitted-script directions, not from the Lite
selector's output; every accepted run also checks that selector against those
independent expectations. Common-only input is explicitly `DFLT`/LTR.

| Milestone               | Native test family                                                                                                                                                                                                                                                                                                                                                                                        | Current API / assertion                                                                                                                                                                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1                      | `milestones.test.ts`: `Latin repertoire and ordered witnesses`, `batch ordering and lengths`; `resident.test.ts`: `resident records, packing and indirect`                                                                                                                                                                                                                                                | Root `shapeComputeText`, single and batch; font metrics, ordered substitutions/ligatures/compositions and pairs; empty strings/batches; 100 independent strings and 100 independent resident blocks; 2/4/8/16/32/50 and 49↔50 counts; 512/1024-scalar public runs (512 resident)                                                                                |
| M2                      | `milestones.test.ts`: `Latin-1/Extended scalar repertoire and seeded sequences`, `fail-closed corpus`; `kernel.test.ts`: `storage/buffer/workgroup exact boundaries use real device limits without near-OOM allocation`                                                                                                                                                                                   | Latin-1/Extended, digits/punctuation, combining/base/stack/filter witnesses on original Inter/Roboto; changing counts; selected default plans and guarded unsupported operations; deterministic seeded sequence corpus                                                                                                                                          |
| M3                      | `milestones.test.ts`: `Greek and Cyrillic`, `CJK and supplementary Gothic`, `Hebrew and Arabic`, `mapped RTL repertoire and seeded joining/marks`                                                                                                                                                                                                                                                         | 53 unique Greek and 72 unique Cyrillic scalars (individual characters plus complete strings); all 237 mapped CJK / 27 Gothic scalars; scalar clusters independent of UTF-16; visual RTL/joining/mark witnesses; homogeneous strings in mixed-script batches                                                                                                     |
| M1–3 resident           | `resident.test.ts`: `resident records, packing and indirect`, `shared atlas and independent blocks`, `transitive selected GSUB closure installs empty-outline outputs, never phantom instances`, `nonvisual actual indirect draws reuse shared native atlas resources and invalidate sibling bindings on growth`                                                                                          | Root creation/update/disposal, test-only six-field shaped-buffer readbacks, exact f32 instance ABI, atlas slot/style words, all four indirect words, empty outlines, cold closure/growth and stable slots, cross-font curve sets/concurrent updates, surviving siblings; production text pipeline/bundles draw into an offscreen texture without pixel readback |
| M1–3 failure/lifetime   | `lifecycle.test.ts` (four named busy/disposal/loss/recreation cases); `resident.test.ts`: `rejection clears every indirect word and recovery keeps the caller atlas`, `kernel failure statuses`, `kernel overflow clears draw count`; `kernel.test.ts`: `shape overflow status 2 executes production WGSL`, `GPOS point-anchor rejection identifies its table and original run after empty-run filtering` | Actual pipeline-entry signals, queue fences and device destruction/lost promises; independent engines; rejection clears stale draws; targeted internal capacity/missing-slot injections are labeled separately from public admission tests                                                                                                                      |
| Parser/outline boundary | Focused existing compute font/extraction/selection/plan/program unit tests                                                                                                                                                                                                                                                                                                                                | Malformed containers, fetch failure, glyf composites/cycles/truncation, CFF recursion/truncation and table classification; these are CPU-reader evidence, not WGSL execution evidence                                                                                                                                                                           |

No test installs the reference executor. Native production compute scheduling,
storage helpers, pipeline preparation and queue submission execute the shipped
WGSL. Resident updates are observed through pass-through native `mapAsync` calls:
successful updates may read exactly one four-byte status, never glyph/instance
records. Additional readbacks are explicitly test-only and outside that window.
Packing expectations use the shader's single-precision operations, not a loose
epsilon. Status failures must keep `[6, 0, 0, 0]`.

This matrix does **not** transfer the lab's 12,048-case HarfBuzz policy corpus or
historical wrapping/paragraph-layout claims to Lite. Current resident input is
independent single-line runs with baseline spacing, not a wrapping API. Inter
`A`+U+20DE, `A`+U+20DD and `<-` stay explicit unsupported nested-GSUB witnesses.
Full Unicode bidi, mixed strong-script strings, Indic/Hangul preprocessing,
feature/language/variation selection (M4/5), pixels/MAD, performance and exhaustive
arbitrary-font conformance remain outside this local correctness project.

Internal native failure fixtures also exercise pack allocation overflow and
missing atlas slots without modifying font bytes. An undersized instance binding
must report status 2 and leave all indirect words `[6, 0, 0, 0]`; WebGPU's robust
out-of-bounds write suppression is not a successful pack. Admission checks use
the actual device's storage binding, buffer and workgroup limits before allocation,
with arithmetic-only threshold fixtures rather than near-OOM GPU buffers.

Native Arabic regression `سلام` exposed a compatibility boundary: text-shaper
0.1.28 gives an all-format-3 **GSUB Context (type 5)** lookup an empty originating
digest, so that root never executes. The compiler retains its selected descriptor
but emits no executable candidate rules for that precise oracle-inert structure.
Fail-closed rejection predicates are retained, including Inter's nested-ligature
boundaries (`A` + U+20DE/U+20DD and `<-`). This does
not apply to Chaining Context type 6, mixed-format lookups, or generic unsupported
mode 5, which must still fail if matched. Arabic lookup 35 must therefore leave
the lam/alef glyphs at 72/9, not promote them to 73/10.

Native point-anchor rejection also exposed ambiguous diagnostics when GSUB and
GPOS shared a lookup index. Failure-header word 2 now packs the table bit in bit
16 and the lookup index in bits 0–15; word 3 remains the matched glyph index.
Readback diagnostics preserve the original batch run index through script grouping
and empty-run filtering. Successful six-field records and the public API are
unchanged. The readback cache is lazily allocated, not an import-time allocation.

### Native coverage execution evidence

The full opt-in project contains **37 test families**, not a count inflated by
unrelated engine unit tests. `milestones.test.ts` checks the following **2,941
accepted font/string cases** (repeated alphabet characters are counted as cases,
not additional unique scalars). Empty public calls correctly return without a
dispatch; every nonempty accepted call follows the production Dawn compute path.

| Pinned font                   | Latin/common accepted | Greek accepted | Cyrillic accepted | Other accepted |                                                    Predeclared rejected calls |
| ----------------------------- | --------------------: | -------------: | ----------------: | -------------: | ----------------------------------------------------------------------------: |
| Inter.ttf                     |                   917 |             58 |                79 |              0 | 23 (guarded selections and three unsupported witnesses, single/batch/grouped) |
| Roboto-Regular.ttf            |                  1259 |             59 |                80 |              0 |                                                                             0 |
| NotoSansHebrew-Regular.ttf    |                     0 |              0 |                 0 |     101 Hebrew |                                                                             0 |
| NotoSansArabic-Regular.ttf    |                     0 |              0 |                 0 |     122 Arabic |                                                                             0 |
| NotoSansSC-Regular-subset.ttf |                     0 |              0 |                 0 |        238 CJK |                                                        1 (unmapped Latin `A`) |
| NotoSansGothic-Regular.ttf    |                     0 |              0 |                 0 |      28 Gothic |                                                                             0 |

Font hashes are pinned in `tests/lite/compute/fixtures.ts`, not inferred from
compute output. Two original-font Latin corpus lengths are asserted before
dispatch (897 Inter / 1133 Roboto, beyond their 20 explicit witnesses), including
mapped Latin Extended-Additional and Extended-C/D/E scalars; RTL
corpora similarly assert 96 Hebrew / 115 Arabic cases beyond 5/7 explicit words.
The Hebrew/Arabic coverage is a bounded mapped-letter repertoire with declared
joining and mark witnesses, not exhaustive Unicode or arbitrary-font support.

`kernel.test.ts` adds source-generated sfnt fixtures, read independently by the
same oracle, to execute multiple substitution (mode 11), mark-to-ligature
component selection (12), single adjustment (10), and chained-context nested
single actions (2), including explicit literal
output records. Original-font witnesses execute single/context/pair/class-pair,
mark-to-mark/base and ligature paths (1/2/3/4/7/8/9). Understood alternates and
unsupported point anchors reject, and mode 5 is never a generic no-op. These
branch witnesses are correctness evidence, **not** an asserted WGSL line/branch
coverage percentage.

Resident checks verify 17 length/run-list transitions per original font, all
six fonts, 100 independently owned blocks, shared and cold-growing atlases,
outline components and a literal A→B→C transitive closure. Each instance is
checked with exact `Math.fround` ABI arithmetic, each glyph record with the
independent six-field oracle, and every indirect word with `[6, visible, 0, 0]`.
Successful original-font updates observe exactly `mapAsync(READ, 0, 4)`. Their
shaped/instance CPU mirrors remain null. Status-only production reads and extra
test readbacks are separate. The read-only slot upload mirror is inspected only
for the synthetic empty-outline closure; it is not a GPU glyph/instance map.

Execution used Node **22.19.0**, **webgpu 0.6.1**, native **Metal / Apple M2 Max**
on macOS 27.0.1. No executor seam, mock command method, self-skip, null backend
or CPU/reference fallback was installed. Native validation/device errors are
assertion failures. Lifecycle spies signal actual pipeline requests and forward
unchanged to Dawn; they do not replace dispatch or hold a fake device-loss promise.

Regression fixes discovered by this coverage:

1. `سلام` originally returned lam/alef 73/10 instead of the pinned oracle's
   72/9. The precise oracle-inert Context-3 compatibility correction above fixes
   it while preserving Inter's unsupported nested-ligature predicates.
2. A real undersized instance allocation originally returned success and an
   overlarge draw count because robust GPU writes were discarded. Packing now
   checks the runtime instance-array length and returns overflow status 2.
3. A matched GPOS point anchor originally reported `unsupported GSUB 0` and
   omitted its input run. Table-tagged failure headers and original-run metadata
   fix the fixed native regression after empty/script-group filtering.
4. Shape/storage/dispatch arithmetic is checked against actual supported device
   limits; the mutable readback cache is lazy. No near-OOM fixture is allocated.
5. The eight existing compute source/test type diagnostics are corrected with
   erased typing/import changes. No public signature, package subpath, font
   bytes, ceiling or golden is changed.

Focused validation is separate from native corpus counts: **61** related
compute/shared-GPU mocked unit tests and **31** root emission/public-type tests.
Library and test TypeScript checks, scoped ESLint/Prettier and `pnpm build:lib`
pass. Root emission tests exclude `webgpu`, `dawn.node` and native setup modules
from both compute-only and mixed production import graphs.

Final native rerun: **37/37 passed across four files in 99.53 seconds**. Scoped
commands executed successfully:

```bash
pnpm test:compute-text-gpu
pnpm exec vitest run --project unit tests/lite/unit/compute-{font,glyph-extraction,text-selection,text-plan,text-program,text-shaper,text-placement}.test.ts tests/lite/unit/text-shared-gpu.test.ts
pnpm exec tsc -p packages/babylon-lite/tsconfig.json --noEmit
pnpm exec tsc -p tests/lite/tsconfig.json --noEmit
pnpm build:lib
pnpm exec vitest run --project build tests/lite/build/compute-text-shaper-treeshake.test.ts tests/lite/build/public-api-types.test.ts
pnpm exec tsx scripts/build-bundle-scenes.ts --scenes scene180,scene181,scene275
BUNDLE_ANALYSIS_DIR=node_modules/.cache/compute-m123-analysis pnpm analyze-bundle scene180
```

ESLint and Prettier were scoped to native tests, changed compute source/units
and emission tests. The existing ESLint configuration ignores `vitest.config.ts`;
it was separately checked by TypeScript and Prettier. An initial test-typecheck
invocation used a nonexistent root `tsconfig.test.json`; the corrected command
above passes, with no remaining source/test type diagnostics.

Filtered live bundle-size builds cover only scene180 / scene181 / scene275:
**30,138 / 30,720 B**, **45,478 / 47,104 B**, **39,463 / 40,448 B** respectively.
These match the immediately preceding layer-separation evidence above/below;
all three ceilings remain unchanged. Scene180 bundle attribution was also
inspected with `analyze-bundle`; generated but unfetched dynamic chunks are
not confused with measured runtime bytes. Pixels/MAD/goldens, performance,
wrapping, the historical HarfBuzz policy corpus and M4/5 remain excluded.

### Neutral implementation boundaries

CPU and resident text share outline primitives (`outline-curves.ts` plus
`cubic-to-quadratics.ts`), the
12-byte instance / 32-byte style ABI (`text-buffer-layout.ts`) and narrow GPU resource constructors
(`_gpu/text-bind-group.ts`, `_gpu/text-style-buffer.ts`). These modules have no
runtime dependency on either font/shaping pipeline, layout, CPU slot packing,
or compute selection/plans/execution; imports of state types are erased.
`text-data-state.ts` initializes legitimate CPU blocks only. Resident blocks have
no placeholder `TextData` or private placeholder atlas.

### Distinct layers and the prepared-draw protocol

```typescript
export interface TextLayer extends TextLayerPlacement {
    readonly data: TextData;
}
export interface ComputeTextLayer extends TextLayerPlacement {
    readonly data: ComputeTextData;
}
export type TextRendererLayer = TextLayer | ComputeTextLayer;
export function createTextLayer(data: TextData, options?: TextLayerOptions): TextLayer;
export function createComputeTextLayer(data: ComputeTextData, options?: TextLayerOptions): ComputeTextLayer;
export function setTextLayerPosition(layer: TextLayerPlacement, x: number, y: number): void;
export interface TextRendererOptions {
    layers: readonly TextRendererLayer[];
    clear?: boolean;
    clearValue?: GPUColorDict;
}
export function addTextRendererLayer(renderer: TextRenderer, layer: TextRendererLayer): void;
export function removeTextRendererLayer(renderer: TextRenderer, layer: TextRendererLayer): boolean;
```

`TextLayerPlacement` contains mutable `positionPx`, `rotationRad`, `scale`, `order`,
`opacity`, `coverageGamma`, and `visible`. Both types are pure public state and
are exported from the sole package root; GPU resources and callbacks are
`@internal` and trimmed from published declarations. No public/internal companion
interfaces or widened CPU `data` type are introduced.

Each factory installs an internal `_createDraw(engine, uniform)` function.
The renderer calls it separately for each layer/renderer attachment. It returns
an internal prepared-draw adapter with `_prepare(layout)`, `_record(encoder,
quad, basePipeline, variantPipeline): number`, `_dispose()`, `_active`, and
`_version`. Preparation synchronizes resources and advances the version only
when a baked resource identity, binding layout, or draw structure changes.
Recording consumes prepared resources and returns the actual recorded draw count.
The shared core imports neither adapter at runtime: `text-layer.ts` owns common
placement/protocol types, `text-layer-cpu.ts` owns CPU allocation/uploads and
ordinary draws, and `compute-text-layer.ts` owns borrowed resident buffers and
indirect draws. The root re-exports each factory from its own module.

The renderer owns the 96-byte per-attachment uniform, pipelines, cached bundles,
ordering and compositing. The CPU adapter owns its instance/style buffers and
per-group bindings. The compute adapter owns only its per-attachment bindings;
the block owns shaped/instance/indirect/status/style buffers, and the caller owns
the shared atlas. Two layers referencing one resident block never share their
placement uniform or bindings. Removal, re-addition and renderer disposal destroy
only attachment-owned resources. Data disposal/loss is terminal, releases only
block resources, rejects pending work, and prevents cached bundles from replaying.
Recovery destroys/recreates attachments through the same factory seam: CPU data
is re-uploaded, resident data stays terminal and is not silently recreated.

Preparation runs before bundle replay as well as during update, including cached
draws. Both adapters compare actual curve texture, band texture, metadata buffer,
style buffer and layout identities (not a transient atlas `_rebuilt` result).
Sibling atlas growth therefore replaces every affected binding before replay,
even when a sibling already performed the upload or growth occurs between update
and record. Content-only writes preserve bundles; CPU slot-range changes and
resident instance/indirect replacement re-record. Invisible layers do no work.
Busy/failed resident updates keep indirect instance count zero; missing live
buffers or an invalid owning engine produce diagnostics, not success-shaped draws.

Example mixed use: `createTextRenderer(surface, { layers: [createTextLayer(cpu),
createComputeTextLayer(resident, { positionPx: { x: 40, y: 80 } })] })`.

### Layer-separation emission and validation evidence

Before and after fixtures use the same shipped `build/lib/index.js` root, named
imports and `console.log` references. Fixture strings were checked for exact
equality. `bundler-harness.ts` runs Rollup ESM unminified, the identical Rollup
entry with Terser `{ module: true }`, and actual Vite production emission with
minification disabled and vendors included. Counts below are emitted UTF-8 bytes,
not source sizes or externalized-vendor estimates.

The shared import list is `createGlyphStorage, createTextRenderer`. The resident
list adds `loadComputeFont, createComputeTextData, updateComputeTextData,
createComputeTextLayer, disposeComputeTextData`. CPU adds `loadFont,
createDefaultTextData, updateDefaultTextData, createTextLayer`; BYO adds
`createTextData, updateTextData, createTextLayer`. Combined includes the shared,
CPU and resident lists once. The measured core-only list is `createTextRenderer`;
the emission regression test additionally references `setTextLayerPosition`.

| Consumer | Emission | Before (B) | After (B) | Delta (B) |
| -------- | -------- | ---------: | --------: | --------: |
| Core     | Rollup   |     27,925 |    17,509 |   −10,416 |
| Core     | Terser   |     15,007 |     9,929 |    −5,078 |
| Core     | Vite     |    268,784 |   258,172 |   −10,612 |
| Resident | Rollup   |    270,098 |   263,067 |    −7,031 |
| Resident | Terser   |    139,127 |   135,386 |    −3,741 |
| Resident | Vite     |    509,817 |   502,786 |    −7,031 |
| CPU      | Rollup   |     74,376 |    73,390 |      −986 |
| CPU      | Terser   |     30,012 |    29,492 |      −520 |
| CPU      | Vite     |  1,478,216 | 1,477,310 |      −906 |
| BYO      | Rollup   |     57,692 |    56,706 |      −986 |
| BYO      | Terser   |     25,650 |    25,136 |      −514 |
| BYO      | Vite     |    296,552 |   295,646 |      −906 |
| Combined | Rollup   |    303,830 |   302,536 |    −1,294 |
| Combined | Terser   |    149,987 |   149,064 |      −923 |
| Combined | Vite     |  1,708,593 | 1,707,419 |    −1,174 |

Root-consumer tests inspect emitted code and per-module rendered bytes across
these bundlers. Resident-only rendering retains no CPU packing/state/palette
synchronization, default shaping/font/layout/extraction, readback executor or
reference interpreter. CPU/BYO retains no compute font/plan/compiler/shaper or
resident shaders. Core-only retains neither adapter; mixed retains the shared
binding/style constructors once.

Final focused verification:

- `pnpm build:lib` passes, including API Extractor's trimmed declarations.
- `vitest run --project unit` with the 20 affected renderer/recovery, shared GPU,
  CPU data/layout/outline/storage/color/weight and compute font/extraction/
  selection/program/plan/placement/shaper test files: **739 tests pass**.
  Shared GPU mocks verify actual draws, instance/style writes and uniform bindings,
  two layers sharing a block, one layer attached to two renderers, removal/re-addition,
  content-only caching, instance/indirect/style/layout replacement, sibling atlas
  growth between update and record, terminal loss with and without device replacement,
  admission failure, missing live curve sets and foreign-engine attachment cleanup.
- `vitest run --project build` scoped to `public-api-types.test.ts`,
  `compute-text-shaper-treeshake.test.ts` and `bundling-rollup.test.ts`:
  **36 tests pass**, including exact CPU/resident data types, mixed rendering and
  positioning APIs, mutation and trimmed private protocol/resource fields.
- `node --test scripts/text-compute-live-input.test.mjs
scripts/text-compute-ordered-program.test.mjs`: **45 tests pass**.
  The three experimental private-renderer scripts pass `node --check`; their
  attachment accesses now use `_draw`. Cold-frame checks validate actual cached
  atlas handles and require rebuilding on replacement, not on mere content writes;
  `coldBundlesRebuilt` counts actual re-recordings and `coldBindingsValidated`
  proves every cold attachment was checked. No profiling run was performed.
- Scoped TypeScript ESLint and source/test/script Prettier checks pass.
- Headless nonvisual `compute-text-resident.spec.ts` and
  `compute-text-shaper.spec.ts`: **2 tests pass**. No visual, MAD, screenshot,
  golden or performance tests were run.
- `pnpm build:bundle-scenes -- --scenes scene180,scene181,scene275` passes:
  **30,138 / 30,720 B**, **45,478 / 47,104 B**, **39,463 / 40,448 B**.
  All original ceilings remain unchanged; no affected scene regression.
- Library `tsc --noEmit` diagnostics exactly match the recorded pre-change
  baseline: the existing `GlyphRun` import in `compute-text-placement.ts`,
  brand conversion in `compute-text-plan.ts`, and narrowing/brand conversion in
  `compute-text-shaper.ts`. Test-project checking has only those four plus the
  existing language argument in `compute-text-plan.test.ts` and three typed-array
  arguments in `compute-text-shaper.test.ts`. This layer change introduces no
  additional diagnostics; these pre-existing failures are not fixed here.

The shared binding helper receives the selected uniform, style buffer, layout,
and an already-selected atlas resource subset (`_curveTex`, `_bandTex`, `_metaBuf`).
The resource list follows the contiguous group-0 ABI: uniform, curves, bands,
metadata, styles; list indices supply bindings 0–4. It does not select an atlas, cache
bindings, upload, invalidate, or own resources. Each renderer/recorder retains
its existing cache and lifetime rules. The style-buffer constructor owns only
allocation in **bytes**, so palette growth and device recovery pass their existing
byte lengths without converting through an entry count. Initial buffers use
`TEXT_STYLE_BYTES`; caller labels remain unchanged. CPU palette growth/version synchronization stays in
`_gpu/text-style-gpu.ts`, while resident base-style initialization stays with
its block.

Line segments are emitted for either-axis deltas **≥ 0.1**; explicit closing
segments require **> 0.1**. Cubics retain the two-quadratic 3/4 subdivision.
Parser contours auto-close on another move and at finish via `outline-path.ts`;
the text-shaper command adapter deliberately does neither, preserving open paths
and next-M behavior. Each adapter owns its contour state; CPU-only consumers do
not retain the parser recorder/TrueType contour walker. Bounds remain supplied
by each outline source, and empty/bounds-less outlines remain absent. TrueType
implied points/composites stay parser-owned.

Root-export emitted-bundle tests enforce these boundaries: resident-only
consumers exclude the vendor CPU font, layout/extraction/packing, glyph-record
readback executor, and CPU reference interpreter. CPU and BYO render consumers
exclude compute fonts, plans/compiler/shaping and resident pack shaders.
The shared renderer has no CPU upload arm or compute import. Resident-only
render consumers exclude CPU state initialization and palette synchronization;
core-only consumers exclude both adapters.
Combined consumers retain one copy of each neutral helper.

Focused regression coverage is in `text-outline-path.test.ts` (epsilon,
closing policy, bounds, cubics and implied points), `text-shared-gpu.test.ts`
(state aliases, resource descriptors, CPU palette synchronization, cached
CPU/resident draws and caller-owned atlas lifetime), and
`compute-text-shaper-treeshake.test.ts` (shipped root-export Rollup and Vite
consumers, including vendor-inclusive Vite output). Renderer-only consumers
retain no text initialization/packing; resident factories retain no CPU palette
synchronization.

`createComputeTextData` / `updateComputeTextData` / `disposeComputeTextData` / `createComputeTextLayer` are a sibling of `shapeComputeText`. They do not return positions and do not build `PlacedGlyph[]`. The caller owns the `GlyphStorage` and passes it in, so several blocks share one atlas. Each block keeps its own instance and indirect buffers. Disposing a block does not dispose that storage. Different fonts in one storage use different curve-set ids. One block has one base style. The shape kernel writes the shaped-record buffer; a following pack kernel writes the 12-byte instance buffer and the 16-byte indirect args (`6`, instance count, `0`, `0`). Positions are pixels (`fontSizePx / unitsPerEm`) on a Y-up baseline at `(0, 0)`. RTL is reversed once, in the shape kernel. A missing atlas slot, overflow, or unsupported lookup zeros the indirect instance count and writes a nonzero status. JavaScript reads that 4-byte status and nothing else. Outlines come from a cmap-seeded closure plus transitive selected GSUB outputs and outline components, installed before dispatch. Atlas slots are append-only; adapters detect replacement resources before replay. Device loss and dispose drop pending work and prevent replay. There is no external draw override on CPU layers.

`createComputeTextData(engine, font, storage, curveSet, options?)` accepts
`options.defaultColor?: readonly [number, number, number, number]` alongside
`fontSizePx`. This is the block's base linear RGBA in `[0, 1]`, matching
`GlyphRun.defaultColor`; omission preserves opaque white. Validate exactly four
finite channels in range before allocating any GPU resources. Copy the channels
into the first four floats of the existing eight-float style record, without
premultiplication; font scale and reserved floats are unchanged. Caller mutation
of the supplied tuple cannot change the uploaded style. Layer opacity still
multiplies the base alpha. The color is fixed for the block's lifetime; this adds
neither per-run styling nor a palette-update API.

`updateComputeTextParagraphs(data, paragraphs, layout)` is the opt-in
[GPU paragraph layout](compute-text-paragraph-layout.md) sibling of
`updateComputeTextData`: LTR paragraphs wrap at shaping-safe ASCII spaces (with
emergency cluster breaks and hanging trailing spaces), align left/center/right,
and use font line metrics or a custom line height plus paragraph spacing.
`getComputeFontLineMetrics` reads scaled hhea metrics without GPU work. Layout
and placement remain GPU-resident with only the existing four-byte status read;
plain updates do not retain the layout kernels. The same block can alternate
between plain and paragraph updates.

### Resident fixed-overhead contract (2026-10-09)

This optimization changes neither the admission/error contract nor any shaped,
atlas, slots, instance or indirect bytes. `updateComputeTextData(data, texts)`
still selects every current input, installs its conservative closure, submits
shaping and packing, and awaits only the ordinary four-byte status before
resolving. No cache is keyed on text, seeds, revision or an input hash.

**Text-independent closure memo.** `closeComputeTextGlyphs(font, plans, seeds)`
keeps a lazily allocated `WeakMap<ComputeFont, Map<string,
Map<number, readonly number[]>>>`. Its key is the joined **insertion-ordered**
union of each plan's `_gsubRoots` followed by `_gsubDependencies`, in plan-array
order. A sorted-set-only key is incorrect: reversing plans can change the
order of reachable glyph insertion. Each glyph entry is exactly the
concatenation of the existing `outputsOf` results, in selected-lookup order
and then subtable-index order. Lookup parsing and its wrapped errors still
precede seed validation. Entries are published only after every subtable
succeeds; throwing glyphs are not cached. Seed validation, LIFO work queue,
convergence bound, closure insertion order and the subsequent component walk
are unchanged. Components use `readComputeGlyfGlyph`'s existing immutable
font/glyph cache. The returned `glyphIds` is identical element for element,
including repeated seeds, empty seeds and calls with different plan orders.

**Incremental outline installation.** `installClosure` creates only a pending
curve map, extracting the insertion-ordered `Set(glyphIds)` minus ids already
in the installed curve map. Empty outlines remain absent and are not
remembered as curves. Extraction remains transactional: failures leave the
installed catalog/atlas unchanged and retain the existing wrapped error.
`updateGlyphStorage` receives only pending non-null outlines, in exactly the
order the former full map would append them. Its append-only atlas indices,
curve/band texels, metadata, versions and growth behavior are unchanged;
`ensureSharedAtlasGpu` is still called on every successful install.

Each data object owns an internal slots staging `Uint32Array` plus the ids
touched on the preceding attempt. On reuse, those ids alone are reset to
`0xffffffff` (MISSING), then current closure ids are written as the exact
atlas index or `0xfffffffe` (EMPTY). The **entire array** is uploaded, including
all MISSING entries outside the closure. A partially failed slot build must
record its touched ids so the next attempt cannot expose stale slots.
The staging array is invalidated on terminal dispose/device loss and rebuilt
when font identity or glyph count changes. Missing curve-set and atlas-slot
errors (including index greater than `0xfffe`) retain their messages and
ordering relative to extraction, atlas installation and upload. No new public
fields, entry points or module-level allocations are introduced.

**One-workgroup small-batch packing.** For `runCount <= 256`, one 256-lane
`packSmall` dispatch replaces the former count/serial-prefix dispatch and
separate pack dispatch. Every lane counts one run using the unchanged
`countRun` checks: shaped status; shaped count exceeding capacity; glyph
outside the font; MISSING slot; count of non-EMPTY slots before a failure.
Inactive lanes contribute `(0, 0xffffffff)`. Workgroup inclusive scan combines
`(visible, firstFailingRun)` by saturated count addition at instance capacity
plus one, and minimum failing-run index. Barriers separate every scan read
and write. Lane zero finds the first inclusive prefix exceeding instance
capacity and selects status in this exact order:

1. Overflow at or before the first failing run → `2`.
2. Otherwise first failing run's original code (including unknown codes).
3. Otherwise `0`.

Lane zero always writes indirect `[6, 0, 0, 0]`, replacing word one with total
visible count only on success, and writes the status. A
`workgroupUniformLoad` broadcasts that status before any instance packing.
Offsets before the first overflow/failure become exclusive visible prefixes;
offsets at and after that run retain their counts, as in the former serial
prefix, including on errors. Successful lanes pack the same six-field records
in run/glyph order, using integer pen accumulation, the same f32 scaling and
line advance, and the unchanged twelve-byte `(x, y, slot)` instance layout.
Zero runs still dispatch one workgroup and write a zero-instance successful
draw. For more than 256 runs, `countParallel → prefixRuns → main` is unchanged.
Schedule caching distinguishes these tiers and rebuilds when the tier or any
bound buffer changes; the small dispatch always has exactly one workgroup.

**Validation.** Unit tests compare closure arrays against the pre-memo
algorithm for Inter/Roboto, multiple repertoires, seed sets and plan orders,
and verify repeated throwing calls. Native differential tests compare old
full-copy installation against growing/shrinking/disjoint/repeated closures,
including slot failures followed by recovery and lifecycle invalidation.
Native pack comparisons cover 0/1/2/255/256/257 runs, empty/invisible runs,
all error precedence cases, raw offsets and complete instance/indirect/status
bytes, plus repeated tier switches on the same resident token.

## Asynchronous status contract (2026-10-09)

```typescript
export interface ComputeTextUpdate {
    readonly completed: Promise<void>;
}
export function updateComputeTextData(data: ComputeTextData, texts: readonly string[]): Promise<ComputeTextUpdate>;
```

The update promise resolves after CPU admission, closure installation, buffer
growth, task preparation and submission of shape/pack plus the four-byte status
copy. It never waits for mapping, GPU completion or a queue fence. The data is
`busy` only during preparation/submission and becomes `ready` before resolution.
The caller may immediately render or submit another update: queue order makes
the indirect draw consume the submitted instances. No text-keyed cache or
shape-skipping path is introduced; all kernel and buffer layouts are unchanged.

`completed` belongs to that individual submission. It resolves on status zero
and rejects with the former `residentStatusMessage` error for GPU-detected
unsupported lookup, reserved-capacity overflow, missing atlas slot or unknown
kernel status. Such an update has already written indirect instance count zero
on the GPU and draws nothing. Its asynchronous failure never clears indirect,
changes data status, or writes any GPU state: a newer update may own the draw.
An internal no-op rejection handler suppresses unhandled rejection events if
the caller ignores `completed`, without changing what callers observe when
awaiting it. Applications should observe `completed` outside frame timing.

CPU-detected failures retain their existing messages, witness text, admission
order and indirect clearing. Generation/device checks remain around every
asynchronous preparation boundary and immediately before submission. Disposal
or device loss during preparation rejects the update promise; neither can
submit after its terminal check.

Each data object owns a free list of unmapped four-byte MAP_READ/COPY_DST
staging buffers and a set of all allocations. Submission takes a free buffer
or allocates one if every existing buffer is in flight, records the ordinary
shape/pack tasks and status copy in the same encoder, submits once, preserves
`engine._gpuTaskTimerResolve?.(encoder)`, and starts `mapAsync` without awaiting
it. Each copied word is private to its submission. After consumption, unmap
returns the buffer to the free list; typical sequential use needs one buffer,
while allocation is bounded by peak overlapping reads. The existing
`readComputeTaskResult` helper remains unchanged for CPU-output shaping.

Terminal disposal/device loss rejects every pending completion respectively
with `updateComputeTextData: disposed` / `updateComputeTextData: device lost`,
destroys all staging buffers, and empties the pool. Mapping failures caused by
destroy/loss receive those same terminal messages, not low-level map errors.
Late handlers do not recycle destroyed buffers or mutate resident draw state.
The sole new public symbol is the root-exported `ComputeTextUpdate` type.

Focused native tests cover resolution while mapping is pending, immediately
queued draw/read correctness for success and all GPU failure kinds, independent
overlapping statuses in both success/failure orders, three-plus in-flight reads
and pool reuse, terminal rejection/destruction, ignored completion rejection,
and unchanged CPU admission failures. Native API timing separates submission
latency from `completed` latency; browser crossover timing uses the supported
update → real indirect draw → queue-fence path and validates status afterward,
without submit-hook draw injection.

### 2026-10-09 — Step 6: two-glyph frame overhead experiments (rejected)

Shipping behavior is unchanged from the asynchronous status contract above.
Cross-origin-isolated timing (≈5 µs clock) at 256 two-glyph blocks put the GPU
path only ≈0.08–0.20 ms above a draw-only floor (≈0.6 ms), versus ≈0.30–0.42 ms
for the CPU path. Two candidates were measured and rejected because neither gained
≥0.03 ms (paired median, same session) on both fonts:

- **Frame-coalesced submission** — record shape/pack/status copy and submit it
  together with the next frame (engine `_prependCbs` seam, macrotask fallback
  flush, flush before conflicting writes/reads). Inter −0.0475 ms, Roboto
  +0.0275 ms.
- **Closure/selection allocation trimming** — lazy GSUB subtable metadata and
  no temporary decoded-selection or nominal-error objects on warm paths. Inter
  −0.0200 ms, Roboto +0.0225 ms.

A fused shape+pack kernel was not attempted: the measured inter-pass gap was
≈0.001 ms. Timestamp-query instrumentation inflated both floor and fence time,
so it must not be used to predict uninstrumented gains.

### 2026-10-09 — Step 7: wide per-run pack

**ACCEPTED after architect override and confirmation.** No bindings, buffers or uploads added.
Capacity ≥33 and `0 < runCount <= maxComputeWorkgroupsPerDimension` select wide:
`countWide(runCount)` → existing `prefixRuns(1)` → `placeWide(runCount)`.
Otherwise ≤256 runs use small, larger batches parallel; both kernels are unchanged.
Mode changes re-record the task; both run-sized wide dispatches update every revision.
Each 64-lane count workgroup finds the first invalid glyph atomically and counts only
preceding visible glyphs, preserving ordered overflow-before-failure precedence.
Placement scans 64-record vec3i chunks with carry. Associative/commutative wrapping
i32 sums make `carry + inclusive - own` bit-identical to serial pens, including overflow.
Instances keep `f32(pen + off) * scale` and `... - f32(run) * lineAdvance`.
Zero status proves validation/capacity; base plus visible rank preserves serial order; rank≤capacity≤`(maxStorageBufferBindingSize / 4 - 4) / 6`<2³¹.

Paired HEAD−candidate ms [95% bootstrap interval], 40 same-session real-draw pairs:
| Cell | Inter | Roboto |
|---|---|---|
| english480×1 | +0.6950 [0.5900,0.7400] | +0.3225 [0.2600,0.3500] |
| english480×4 | +0.5150 [0.3450,0.5725] | +0.5925 [0.4925,0.7950] |
| english480×16 | +0.4975 [0.4725,0.5350] | +0.4300 [0.3225,0.5075] |
| natural50×16 | +0.0800 [0.0600,0.1075] | +0.0875 [0.0275,0.1425] |
| two2×256 confirmation | +0.0125 [-0.0800,0.1025] | +0.0575 [-0.0050,0.1475] |
| two2×256 HEAD A/A | +0.0400 [-0.0350,0.1100] | +0.0275 [-0.0175,0.0900] |
Two2 uses the unchanged capacity-2 path; confirmation passes the revised interval gate.
Timing stops at the real-draw queue fence, never status; all measured output bytes match.

### 2026-10-10 — Step 8: safe segmentation of long runs

**Correctness validated; measured target T=64.** A pure CPU analysis in
`text/compute-text-segment.ts` proves individual boundaries after U+0020 safe; it
does not shape text, cache input strings, or change the logical-run API. Analysis
is lazy on the first run longer than `SEGMENT_TARGET` (64, measured among
32/64/128), cached by compiled program, including ineligible results. Short-only
updates do not analyze a program.

Eligibility requires LTR, a mapped space, known rule modes, all GSUB context
dependencies present, at most one expanding multiple-substitution lookup, and a
space substitution/multiple closure B* entirely in GDEF class 0. No ligature tail
may accept B*. Producer indexing by glyph predicates plus a class-producer list
computes substitution/multiple/ligature closures. Nominal spans are bounded
separately in GSUB root application order (`program.rootOrder`, not the appended
dependency list). Each root rule reads pre-root spans: a ligature sums input
predicate maxima, while single/multiple outputs inherit the input maximum.
Class predicates inspect matching glyph spans, including class-0 defaults.
Rules in the same lookup never consume that lookup's own outputs. Each context
rule composes its nested lookups in record order at that root, not at a separate
dependency pass; a nested lookup that is also a root still runs at its later
root position. Because non-leading targets can be revisited by overlapping
context matches, span-preserving nested substitutions are closed within the
context root. That closure only transfers existing bounds and terminates.
Nested ligatures, cardinality-changing multiple substitutions, or deeper
contexts fail closed: repeated child applications within one root can consume
prior child outputs, so a once-per-root span bound does not prove them safe.
The compiler currently emits executable contexts only for depth-1 singles.
The maximum S must be ≤8 (individual spans capped at 64);
maximum crossing distance d must satisfy d·S≤64; at most 4096 crossings are allowed.

For each non-mark-attachment rule window `[reverse(backtrack), input, lookahead]`,
each B*-accepting position other than the last defines a crossing. Other slots
retain side, distance, predicate and a global bit. An anchor minimizes hits in
closures of A–Z/a–z/0–9, then accepted glyph count, then distance. Lazy glyph info
records closure, whether any closure glyph can be skipped by any lookup (`free`),
whether any can be a mark, anchor hits, and a lazy complete slot-hit bitset.
`safeAfter(glyphs,s)` requires a nominal space and a following glyph. Independent
left/right walks visit up to maxD·S non-free glyphs and at most 64 steps each,
recording the non-free count before each glyph. An anchor hit within its d·S range
touches a crossing; if every slot can hit its side within its own d·S range the
boundary is unsafe. If any mark-base/mark-mark base accepts B*, a possibly-mark
first nominal glyph on the right also makes the boundary unsafe.

Proof sketch: B* is class 0, never skipped/deleted or a ligature tail. A lookup
reading across a cut must include a B*-derived glyph in its rule window. The d-th
live glyph is conservatively represented within d·S non-free nominal positions;
free glyphs cover skipped glyphs and ligature-interior leftovers. Absence of a
fully possible crossing therefore proves independence by induction over GSUB
compaction and GPOS. S is itself proved by induction over ordered GSUB roots,
using pre-pass spans and the span-preserving context closure described above;
decompose/recompose cycles across different roots are finite, not a global
span fixpoint. A right-side first live non-mark prevents attachment across
the space. Rejects are local because crossing rejects cannot match. With only
one expanding lookup, `length·maxExpansion` bounds both whole and segment outputs.
Successful stitched headers 0–1 and every six-word record, including clusters,
must be bit-identical to whole-run shaping. Diagnostics 2–3 are zero on success.

`segmentComputeTextRuns` first searches downward from target−1 to start+1, then
upward from target to the penultimate glyph, cutting after the first safe space.
Scalars and nominal glyphs use subarray views. Ineligible/short/unsplittable runs
stay whole. No splits, excess per-group workgroups, or segment output/table bytes
above storage/buffer limits return null and retain the existing GPU path.

Logical shape capacity, shaped slots, instance sizing and pack remain unchanged.
Segment groups preserve group/run order with global segment `outputRun`; segment
capacity is the maximum shape capacity over these groups. A table maps each
logical shaped slot to `[firstSegment, segmentCount]`, followed by per-segment
scalar starts. Resident grow-only buffers are `segmentOutput` (header plus
segmentCapacity six-word records), `segmentTable` (read storage), `stitchParams`
(read storage vec4u: runCount, logical capacity, segment capacity, zero).
Shape binds segment output and selects the segment-capacity scratch tier. Output
identity invalidates shape bindings; segmentation mode and all stitch buffer
identities invalidate pack schedules. Release disposes all new buffers/shaders.

A 64-lane stitch dispatch, one workgroup per logical run, precedes existing pack
dispatches. Every lane sums segment counts and selects the first failure. Lane 0
writes failure status/count zero and copies failing diagnostics (overflow uses
zero diagnostics), or success/count/zero diagnostics. Lanes copy segment records
in order, adding scalarStart only to the cluster word. Total above logical
capacity yields status 2. Async status aggregation/readback remains unchanged and
untimed; measurement ends at a queue fence after a real draw.

Tests cover probe-equivalent boundaries, eligibility/fallback/lazy caching,
search order/limits, native exact shaped records and oracle comparisons at 480
and 1000 glyphs, mixed groups, rejection, cluster offsets, mode transitions and
release. Validation is scoped to compute-text unit/native tests, package
typecheck, changed-file lint and scene1/169/180/181/188 bundle-size checks only.

**Validation result:** all 266 native compute tests pass, including successful
480/1000-glyph corpus records, clusters, mixed/RTL groups, rejection, transitions
and buffer release. Package typecheck, changed-file lint and all five scoped
bundle ceilings pass. Scoped unit tests pass 87/87. The architect's ruling
corrected §6's isolated-string examples: tests assert complete corpus safe counts
and exact deduplicated unsafe-context sets, plus safe spaces after `The` and
unsafe period-before-`To`/`The` boundaries. Counts match exactly: Inter 101/101,
54/54, 32/45; Roboto 96/101, 51/54, 45/45; zero probe disagreements across
400 boundaries. Context snippets are witnesses, not context-free classifiers.

**Measured result:** warm fresh-analysis median of five (caches cleared):
Inter 4.615 ms, Roboto 6.386 ms. Initial Inter analysis exceeded 20 ms; indexed,
weighted anchor scoring removed repeated predicate membership construction.
First-update analysis cache misses in the browser full run cost Inter 9.570 ms,
Roboto 6.605 ms (analysis only, not total update time).

The 15-round real-draw gate passed. The 31-round sweep's equal-weight mean
percentage changes across 1/16×english480 in both fonts were T32 −30.54%,
T64 −33.36%, T128 −28.60%; T64 was selected and the source constant remains 64.
Full results below are 31 same-session interleaved baseline/candidate pairs per
cell, COOP/COEP-isolated 5 µs timers, update through the real-draw queue fence,
with ordinary status validation outside timing. Baseline is pinned c3ac8eba.

| Cell | Inter baseline → candidate ms (% change) | Roboto baseline → candidate ms (% change) |
|---|---|---|
| 1×english480 | 1.655 → 1.040 (−37.16%) | 1.840 → 0.945 (−48.64%) |
| 4×english480 | 1.555 → 1.125 (−27.65%) | 1.955 → 1.060 (−45.78%) |
| 16×english480 | 1.655 → 1.495 (−9.67%) | 2.260 → 1.400 (−38.05%) |
| 16×natural50 | 1.230 → 1.190 (−3.25%) | 1.415 → 1.415 (0.00%, tie) |
| 256×two2 | 0.720 → 0.730 (+1.39%, tie) | 0.705 → 0.700 (−0.71%, tie) |

Tie means absolute median difference below 0.02 ms. Full preflight/final checks
verified 782,166 shaped records and 661,510 GPU instances, with no GPU/browser
errors; byte comparisons of visible instances, indirect and status all matched.
Native tests independently compare stitched versus whole-run headers/records.
Timestamp diagnostics at 1×480 (separate instrumented run) measured shape GPU
Inter 0.946 → 0.281 ms, Roboto 1.611 → 0.370 ms.

A/A baseline-versus-baseline controls showed ordering noise: 31-round apparent
changes at 1×480/256×two2 were Inter −0.150/−0.075 ms and Roboto −0.035/−0.070 ms.
A 61-round reversed confirmation gave Inter +0.015/+0.035 ms and Roboto
+0.010/+0.025 ms. These are not engine improvements. Small short-run differences
and Inter's 16×480 result are not confidently attributable; single-run long-run
gains exceed observed control shifts. No visual/MAD tests or golden changes were
performed; CI remains responsible for those checks.

### 2026-10-10 — Step 8a: segmentation analysis cost

The one-time segmentation cost includes `analyze()` and first-use lazy per-glyph
work in `safeAfter`. The architect's prototype run measured base cold Chrome
timings of Inter 9.8 + 15.9 = 25.5 ms and Roboto 8.0 + 0.6 = 8.6 ms (analysis +
first segmentation = total); the implementation run's own base is in the table.

Four output-identical changes reduce this cost:
- Lazy per-(glyph,slot) tri-state memo replaces eager all-slot bitsets.
- A u16 B* membership mask avoids per-rule window allocation until a crossing is found.
- `markBaseHit` is lazy, evaluated only when the first right glyph can become a mark
  (or when the analysis getter is read).
- Class counts traverse coverage or reuse per-ClassDef histograms rather than
  scanning the glyph domain for every predicate.

Anchors and counts only order proof work; the `markBaseHit` boolean is unchanged.
Parsed coverage is strictly ascending, so coverage counting visits each glyph once.
Cold headless Chrome 155 medians (ms; nine fresh contexts per cell, first dropped):

| Font | Base analyze / first / total | New analyze / first / total |
|---|---|---|
| Inter | 9.2 / 14.8 / 24.0 | 7.3 / 2.5 / 9.9 |
| Roboto | 7.7 / 0.6 / 8.3 | 2.2 / 0.6 / 2.8 |

Equivalence passed 18,720 `safeAfter` positions each for Inter, Roboto,
NotoSansHebrew and NotoSansGothic (74,880 total), plus all metadata comparisons.
All 18 eligibility checks matched across six fonts and three script samples;
NotoSansArabic and NotoSansSC subset were ineligible for all three samples.
The same deterministic 61-text corpus was checked at every index of each eligible plan.
Program compile remains a larger separate one-time cost (~67–75 ms cold) and is
out of scope.

### 2026-10-10 — Step 8b: program compile cost

`compileComputeTextProgram` is the CPU-side compilation of parsed GSUB/GPOS lookups
into the immutable rule-program buffer. It runs once per plan on first use, not
per update, and is not WGSL compilation. Acceptance requires a byte-identical
program, so the rule-program layout, WGSL and runtime path are unchanged.

Profiled base costs: on Inter the main costs were per-rule class-glyph filtering
(`firstGlyphs`), spread-based word packing and `ValueRecord` parsing. On Roboto they
were the per-mark copying and packing of the shared mark-base base triples.

Output-identical changes:
- `packProgram` writes into a growable `Uint32Array`. The append order is the same
  as before, and the result is returned as one tight slice (no `number[]` and no
  `Uint32Array.from`).
- Class predicate first-glyph sets come from per-(ClassDef, coverage) class buckets,
  built once in domain order, instead of a full domain filter for every rule.
- Per-lookup candidate directories use counting placement (counts, prefix sums,
  then fill). Lists are in glyph order and rule order, with the same out-of-range guard.
- Mark-base and mark-mark rules share one base list per mark class within a subtable.
  Their packed base triples are cached per list, and sorting is skipped when the
  list is already ascending.
- Spread-free maxima, a per-list cache for the maximum anchor magnitude, and a
  closure-free `readValue` with the same read order, error labels and result shape.

Equivalence: the base and new compilers produce byte-identical words and identical
root witnesses and metadata. This covers 42 programs (six fonts × seven samples),
and `parseLookupActions` results deep-equal for all 268 lookups. A unit test pins
the tight buffer and shared per-class base lists.

Cold headless Chrome 155 medians (ms; nine fresh contexts per cell, first dropped).
"Rest" is the remainder of first-use immutable preparation:

| Font | Base compile / rest / total | New compile / rest / total |
|---|---|---|
| Inter | 73.1 / 5.9 / 79.0 | 36.4 / 8.5 / 44.9 (−43%) |
| Roboto | 63.7 / 3.2 / 66.9 | 15.2 / 4.0 / 19.2 (−71%) |
| NotoSansArabic | 29.0 / 1.6 / 30.6 | 9.9 / 1.9 / 11.8 (−61%) |

The rest phase does no new work; its inputs are byte-identical. Its small increase
comes from V8 heap state. The base compile's garbage grew the young generation,
which made the push-heavy `prepareSimplePairWords` cheaper afterwards. With untimed
synthetic churn inserted before the rest phase, the new rest returns to 6.4 / 3.6 /
1.5 ms. In fresh Node processes, compile drops from 121 to 65 ms on Inter and rest
moves only from 10.3 to 11.1 ms.

Remaining larger reductions would change the program layout. Options are one
indirect base list per mark class (Roboto spends 850k of 1.13M words on repeated
base triples) and compact candidate headers; moving the compile off the main
thread is another option. All of these are out of scope for this byte-identical step.
