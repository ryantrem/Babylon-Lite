# Module: GPU-resident paragraph layout

## Purpose

Move paragraph wrapping and line placement onto the GPU, after resident
shaping, so a host such as LineLayoutPad can draw plain-text documents without
any LLX WASM shaping or layout. The work is **incremental**: each step ships a
working, explicitly bounded subset and the next step restores lost
functionality. Earlier steps may drop editor features (caret, selection, hit
testing) as long as the loss is explicit and the record formats do not have to be
redesigned later.

### Incremental roadmap

| Step              | Babylon Lite                                                                                                                                                                                                                                                    | Pad                                                                                                                                                               | Functionality restored                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **1 (this spec)** | `updateComputeTextParagraphs`: shape each paragraph once, GPU greedy wrap at shaping-safe spaces, emergency breaks, hanging trailing spaces, left/center/right alignment, paragraph spacing, font line metrics. Per-line records written on the GPU (internal). | "Compute Text (GPU layout)" mode: no LLX shaping/layout on the path; plain-text documents drawn from GPU layout. Caret, selection, hit testing, markers disabled. | Wrapped, aligned plain-text drawing with no LLX. |
| 2                 | Paragraph slots with stable identities; reshape only dirty paragraphs; width/alignment changes re-run layout without reshaping.                                                                                                                                 | Per-keystroke dirty-paragraph updates.                                                                                                                            | Edit latency independent of document size.       |
| 3                 | Revision-bound async queries over line/cluster records: line extents, cluster↔x, caret stops, hit test.                                                                                                                                                         | Caret, click-to-place, arrow keys, selection highlight.                                                                                                           | Editing.                                         |
| 4                 | GPU caret/selection rendering from cluster records (no per-keystroke readback).                                                                                                                                                                                 | Remove query round-trip from typing.                                                                                                                              | Latency parity.                                  |
| 5                 | Within-paragraph parallel wrap (prefix sums + break search) for very long paragraphs; device-level pipeline sharing.                                                                                                                                            | Large documents, many panes.                                                                                                                                      | Scale.                                           |
| 6                 | Multiple styles per paragraph (style runs, per-run fonts/sizes/colors, line metrics from the tallest run).                                                                                                                                                      | Bold/italic/size/color, decorations.                                                                                                                              | Rich text.                                       |
| 7                 | Line-local bidi reordering, more scripts, UAX #14 classes beyond spaces, tabs, justification.                                                                                                                                                                   | Remaining eligibility gates removed.                                                                                                                              | Feature parity with LLX.                         |

LLX remains available as a **test oracle** (line breaks, metrics, positions) and
as a user-selectable backend; it is never consulted by the GPU layout path.

## Public API Surface

```ts
/** Line box and paragraph placement for {@link updateComputeTextParagraphs}. */
export interface ComputeTextParagraphLayout {
    /** Line box width in CSS px. Positive; `Infinity` disables wrapping. */
    readonly maxWidthPx: number;
    /** Baseline-to-baseline distance in CSS px. Defaults to the font's natural line height at the block's size. */
    readonly lineHeightPx?: number;
    /** Extra distance in CSS px added between consecutive paragraphs. Default 0; must be finite and >= 0. */
    readonly paragraphSpacingPx?: number;
    /** Horizontal alignment inside the line box. Default "left". "center"/"right" require a finite `maxWidthPx`. */
    readonly align?: "left" | "center" | "right";
}

/** Shape, wrap and place `paragraphs` on the GPU. Same lifecycle, status and failure contract as `updateComputeTextData`. */
export function updateComputeTextParagraphs(data: ComputeTextData, paragraphs: readonly string[], layout: ComputeTextParagraphLayout): Promise<ComputeTextUpdate>;

/** Font vertical metrics scaled to `fontSizePx`. `descentPx` is positive below the baseline. */
export interface ComputeFontLineMetrics {
    readonly ascentPx: number;
    readonly descentPx: number;
    readonly lineGapPx: number;
    /** `ascentPx + descentPx + lineGapPx`. */
    readonly lineHeightPx: number;
}

/** Read `hhea` ascender/descender/lineGap. Pure; no GPU work. */
export function getComputeFontLineMetrics(font: ComputeFont, fontSizePx: number): ComputeFontLineMetrics;
```

`updateComputeTextData` is unchanged. A block may alternate between the two update
functions; the last submitted update defines what the layer draws.

### Coordinates

Same space as the existing resident pack (CSS px, Y-up, layer position is the
origin): the first line's baseline is at `y = 0`, the line box's left edge at
`x = 0`. Line `k` of the block (counted across paragraphs) has its baseline at
`y = -(k * lineHeightPx + paragraphIndex * paragraphSpacingPx)`. A host that
wants the top of the first line box at its origin offsets the layer by
`ascentPx` from `getComputeFontLineMetrics` (plus half-leading if it uses
CSS-style line boxes).

### Input contract

- Each paragraph is one string with no `\n`, `\r`, U+2028, U+2029 or `\t`;
  these reject synchronously. Hosts split documents into paragraphs.
- Each paragraph must select an LTR script (existing selection rules). RTL
  paragraphs reject synchronously (bidi is step 7).
- Whitespace is preserved exactly: no trimming, collapsing or tab expansion.
  This deliberately differs from the CPU `layoutText` path.
- Empty paragraphs produce one empty line.
- Layout values and the font scale are uploaded as f32. Finite widths, line
  heights and the scale must remain positive and finite after f32 conversion;
  paragraph spacing must remain finite (underflow to zero is allowed). Reject
  unrepresentable values synchronously rather than silently turning finite
  aligned boxes into infinity or producing non-finite coordinates.
- Existing selection gates still apply, including rejection of joining
  controls (U+200C/U+200D). The U+200D flag is specified and tested independently
  so the emergency-break format is ready for later admission expansion.

## Line-breaking rules (step 1)

All widths are advance-based in font units, compared in px after the same f32
scale the pack uses (`f32(units) * scale`).

1. **Hanging whitespace.** A record whose first source scalar is U+0020 is a
   space record. Space records never cause overflow and are excluded from a
   line's measured width when they end the line. Other space characters
   (NBSP, U+2000–U+200A, …) are ordinary content in step 1.
2. **Break opportunities.** A break is allowed after the last U+0020 of a run of
   spaces that is followed by another scalar, **only if** the resident
   segmentation analysis proves that shaping text before and after that space
   independently gives identical records (`SegmentAnalysis.safeAfter`). Such a
   break is therefore exactly equivalent to reshaping each line separately. A
   space that cannot be proven independent is not a break opportunity. If the
   analysis is unavailable for a paragraph's plan and the paragraph contains a
   candidate space, the update rejects synchronously; it never guesses.
3. **Greedy fit.** For each non-space record, if the line already holds at least
   one record and `(end of this record - line start) * scale > maxWidthPx`, the
   line breaks at the most recent break opportunity within the line.
4. **Emergency breaks.** When the overflowing line has no opportunity, it breaks
   immediately before the overflowing record if that record begins a new
   cluster (its cluster differs from the previous record's) and its first scalar
   is not a combining mark (`\p{M}`) or U+200D. Otherwise the line keeps
   overflowing until the next allowed position. Emergency breaks do **not**
   reshape; glyphs keep whole-paragraph context. A line always holds at least
   one record, so an unbreakable cluster wider than the box overflows.
5. **Alignment.** Line width = end of the last non-space record minus line start
   (0 for empty or all-space lines). Offset is `0`, `(maxWidthPx - width) / 2`
   or `maxWidthPx - width`, clamped to `>= 0` (an overflowing line starts at the
   left edge, as in CSS).

## Internal Architecture

`updateComputeTextData` and `updateComputeTextParagraphs` share one internal
submission routine in `compute-text-data.ts`. It accepts an optional
**pack-stage factory**; the paragraph module supplies one that replaces the
default pack dispatches with the layout kernels. The resident executor binds the
stage after shaping (and segment stitching) in the same pack task, so there is
still exactly one status readback per update and no other mapping.

The shared routine's selection/grouping and stage factory run synchronously,
before its async GPU continuation and before any closure/atlas installation.
The factory receives the original (unsegmented) groups and shared record
capacity; nominal glyph vectors and scalar arrays are already decoded. The plain
public update retains its Promise-rejection contract. Paragraph updates prefix
both admission and completion diagnostics with `updateComputeTextParagraphs:`.
Admission failures on a ready block clear its indirect arguments without changing
the atlas. Busy rejections do not interfere with the already pending update.

Paragraph-specific code lives in `compute-text-paragraphs.ts` (validation,
CPU break flags) and `_gpu/compute-text-paragraph-layout.ts` (kernels, buffers),
which are reached only from `updateComputeTextParagraphs`. The scenes that use
only `updateComputeTextData` do not pull in the layout kernels.

### CPU preparation (per update)

Existing selection, nominal mapping, grouping, segmentation and closure run
unchanged with one run per paragraph. Additionally, for each paragraph the CPU
writes one flag byte per source scalar:

The shared submission owns the only string-array snapshot. Paragraph preparation
uses the already decoded scalar/nominal views, without decoding again or doing
CPU wrapping, measurement, visible-instance counting, or placement. A cheap
paragraph-length pass sizes the input; one scalar pass packs flags directly into
the reusable word buffer (no per-paragraph flag arrays or flag-copy pass).
Analysis is resolved lazily at the first candidate space of each plan group;
the program/segmentation caches retain the analysis per immutable plan/program.
ASCII non-spaces bypass Unicode mark tests. Non-ASCII mark classification is
cached lazily by scalar in 256-scalar pages, independent of font or document.
Natural line height is cached per block's immutable font/size.

CPU input words and their float/upload views are kept with the per-token stage.
Input capacity grows geometrically within device limits; same-sized updates reuse
all views. Only the active word range is uploaded after a shrink. Each active flag
word is rebuilt, including zero padding, so stale flags cannot survive reuse.

| Bit | Meaning                                               |
| --- | ----------------------------------------------------- |
| 0   | `SPACE`: scalar is U+0020 (hangs).                    |
| 1   | `BREAK_AFTER`: rule 2 opportunity after this scalar.  |
| 2   | `NO_BREAK_BEFORE`: combining mark or U+200D (rule 4). |

### Layout input buffer (`u32` words, read-only)

```
[0] paragraphCount   [1] capacity (records per paragraph)
[2] scaleBits (f32)  [3] glyphCount
[4] maxWidthPx (f32, +Inf allowed)  [5] lineHeightPx (f32)
[6] paragraphSpacingPx (f32)        [7] align (0 left, 1 center, 2 right)
[8 .. 8+P)  word offset (from word 0) of paragraph p's flag bytes
[...]       flag bytes, 4 per word little-endian, each paragraph word-aligned
```

### Per-paragraph info (`vec4u`, read-write)

`x` visible instance count → exclusive instance base after the prefix pass;
`y` failure status (0 ok); `z` line count; `w` global first-line index after
the prefix pass.

### Line records (`vec4u` per line, read-write, `capacity` lines per paragraph)

`x` first shaped record, `y` record count (including hanging spaces), `z` width
in px (f32 bits, excluding hanging spaces), `w` source scalar index of the first
record (reserved for step-3 queries). Line `l` of paragraph `p` is at
`p * capacity + l`. Lines ≤ max(1, records) ≤ capacity, so this never overflows.

## Pipeline Configuration

Three compute entry points in one WGSL module, each with the same 8 storage
bindings (`input`, `shaped`, `slots`, `instances`, `indirect`, `status`,
`info`, `lines`) to stay within the default 8-storage-buffers-per-stage limit:

1. `measure` — workgroup 64, one invocation per paragraph.
2. `prefix` — one workgroup of 256 lanes.
3. `place` — workgroup 64, one invocation per paragraph.

Dispatch order inside the pack task: [stitch] → measure → prefix → place.

## Shader Logic

**measure(p):** read the shaped header; validate status, count ≤ capacity,
glyph < glyphCount, slot ≠ MISSING (same codes as the default pack: 1
unsupported lookup, 2 capacity, 3 missing slot) and count visible slots. On
failure write `info[p] = (0, status, 1, 0)`. Otherwise run rules 1–4 in one
sequential pass:

```
lineStart = 0; startPos = 0; pos = 0; contentEnd = 0
lastBreak = NONE; breakPos = 0; breakContentEnd = 0
for i in 0..count:
  c = cluster[i]; adv = xAdvance[i]
  if !SPACE(c):
    end = pos + adv
    loop:
      if i == lineStart or f32(end - startPos) * scale <= maxWidth: break
      if lastBreak != NONE:
        emit(lineStart, lastBreak + 1 - lineStart, breakContentEnd - startPos)
        lineStart = lastBreak + 1; startPos = breakPos; lastBreak = NONE; continue
      if cluster[i] != cluster[i-1] and !NO_BREAK_BEFORE(c):
        emit(lineStart, i - lineStart, contentEnd - startPos)
        lineStart = i; startPos = pos
      break
    contentEnd = end
  pos += adv
  if i + 1 < count and cluster[i+1] > c and BREAK_AFTER(cluster[i+1] - 1):
    lastBreak = i; breakPos = pos; breakContentEnd = contentEnd
emit(lineStart, count - lineStart, contentEnd - startPos)
```

`emit` writes a line record with width `max(0, f32(w) * scale)`.

**prefix:** exclusive scans of visible counts (instance bases) and line counts
(global line bases) using a 256-lane span scan; the lowest failing paragraph's
status is written to `status[0]`; `indirect = [6, total visible or 0, 0, 0]`.

**place(p):** if `status[0] != 0` return. Walk the paragraph's lines, keeping a
running pen `(penX, penY)` in font units across the whole paragraph. At each
line start, `lineX = penX`; for each visible record:

```
baseline = fma(f32(globalLine), lineHeight, f32(p) * paragraphSpacing)
x = fma(f32(penX - lineX + xOffset), scale, alignOffset(line))
y = fma(f32(penY + yOffset), scale, -baseline)
```

`fma` makes the multiply-add association explicit. This preserves the coordinate
math while avoiding reassociation across the baseline and position expressions
(observable at one ULP for long blocks). WGSL permits both fused (one rounding)
and non-fused fma implementations; bitwise coordinates across different adapters
are not guaranteed. Native tests discover the executing adapter's fma mode in
test-only setup, then compare exact results with a pure-TS reference using that
mode. Widths and alignment offsets use ordinary f32 arithmetic; left alignment
never subtracts a width from infinity.

## State Machine / Lifecycle

Identical to `updateComputeTextData`: `ready → busy → ready`, generation
guarded, rejects while busy, terminal on dispose/device loss. Layout buffers and
shaders are owned per `ComputeTextData` and released with the resident shaders.
The lazy per-token stage cache retains buffers across plain/paragraph alternation;
the active pack schedule is rebuilt in both directions. Device changes and input,
info or lines buffer growth bump the stage version. Stable bindings reuse the
schedule, with measure/place dispatch sizes updated by stage preparation.
Input, info and lines binding sizes must fit both `maxBufferSize` and
`maxStorageBufferBindingSize`, and dispatch dimensions must fit device limits.

## Babylon.js Equivalence Map

None. Babylon.js has no GPU paragraph layout; the CPU `layoutText` path is the
nearest Lite analogue but has different whitespace semantics.

## Dependencies

`compute-text-data.ts` (shared submission), `_gpu/compute-text-resident.ts`
(pack-stage hook), `compute-text-segment.ts` (`analyzeComputeTextSegmentation`,
`safeAfter`), `compute-text-program.ts`, `compute-font.ts` (hhea).

## Test Specification

- **Unit (no GPU):** flag bytes (spaces, break-after only at the end of space
  runs, unsafe spaces, marks/ZWJ); synchronous rejection of line breaks, tabs,
  RTL, invalid layout options; `getComputeFontLineMetrics` against hhea.
- **Native Dawn (`compute-text-gpu`):** for Inter and Roboto, a pure-TS reference
  implementation of rules 1–5 over oracle shaped records must match the GPU's
  line records, instances and indirect count exactly, across: no wrap
  (`Infinity`), narrow widths forcing every-word and emergency breaks, leading
  / multiple / trailing spaces, empty paragraphs, many paragraphs, long
  paragraphs that exercise segmented shaping, alignment variants, paragraph
  spacing, and alternation with `updateComputeTextData`. Updates read back only
  the 4-byte status.
- **Build:** a resident-only bundle must not contain the layout kernels.

## File Manifest

- `packages/babylon-lite/src/text/compute-text-paragraphs.ts` (new)
- `packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout.ts` (new)
- `packages/babylon-lite/src/text/compute-text-data.ts` (shared submission)
- `packages/babylon-lite/src/text/_gpu/compute-text-resident.ts` (pack-stage hook)
- `packages/babylon-lite/src/text/compute-font.ts` (`getComputeFontLineMetrics`)
- `packages/babylon-lite/src/index.ts` (root re-exports)
- `tests/lite/unit/compute-text-paragraphs.test.ts`, `tests/lite/compute/paragraph-layout.test.ts`
