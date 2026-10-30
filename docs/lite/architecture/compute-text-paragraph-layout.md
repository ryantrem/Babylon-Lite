# Module: GPU-resident paragraph layout

## Purpose

The production admission, fused submission, status completion and fallback
contract is specified in [resident text submission](compute-text-submission.md).

## Paged paragraph storage (Stage 2)

The fitting Stage 1 kernels below remain the fast path. Required uniform storage
that cannot fit, or exceeds both 4 MiB and twice independently sized live storage,
selects persistent post-shape chunks. Paged owners remain paged until a plain
resident update. Internal test budgets can force small pages; there is no public
configuration or device-limit increase.

Chunks group proven-independent shaping segments (target 64 scalars) up to 16,384
conservatively expanded records. Storage seams are never wrapping or paragraph
boundaries. A whole-paragraph validation/selection scan preserves Common/Inherited
resolution and rejection precedence; mapping, proof windows and uploads are bounded.
The unchanged `safeAfter` proof receives 65 neighbors on either side, the candidate
and immediate successor, retaining true paragraph ends. Unsafe spans extend to a
proven seam or the true end. A fitting indivisible span has dedicated storage;
an over-budget span rejects before atlas installation with its exact resource bytes.

Devanagari pages use the same planner. Chunk capacity, closure seeds and the batch
workspace bound come from the profile (`capacity(scalars)`, `closureSeeds`, the
kernel variant's per-record bytes), and the per-chunk run validation is the
profile's `validateRun`. Proven seams are exactly the
[indic paragraph space proof](#indic-paragraph-space-proof)'s safe spaces, and chunk
flags come from the same producer as the monolithic path, so they are byte-identical.
An indivisible span between proven spaces that exceeds the binding limit gives the
existing binding-limit error.

Persistent shaped records retain the existing header/six-i32 ABI with chunk-local
clusters. Flags, compact instances, metadata/checkpoints and indirect records
occupy independently sized windows in lazily allocated arenas (normal target
1 MiB). Offsets align to `minStorageBufferOffsetAlignment`; bound windows fit the
storage binding limit even when the containing allocation is larger. Arenas fit
`maxBufferSize`. Optional growth is at most 25%, budget-aware. No longest-paragraph
stride or empty paragraph pool is allocated. Normal transient shaping/local
stitch/replay storage is bounded to 8 MiB; dedicated indivisible fallback is checked
separately. Source snapshot/chunk descriptors are explicitly persistent O(N).

Paged layout is exact ordered streaming serial wrap, not independent chunk wrap.
Forward dispatches save 128-byte entry/exit checkpoints: glyph/line/start ordinals,
first cluster, i32 start/pen/content/break positions, last break/cluster and visible
count, status and validity. Summaries propagate the next nonempty record across
empty chunks for exact cross-seam break lookahead. Only real paragraph ends close
the final line. Paragraph line totals prefix in logical document order.

Reverse traversal replays from each saved entry checkpoint into bounded scratch:
16 bytes per record for integer pen/visible prefixes and 32 bytes per local line
event (at most records + 2). It merges up to two future line events, places actual
records into the chunk's instance window and passes backward only events covering
earlier records. At a forward checkpoint there is one open line and at most one
remembered break: future input can close through that break and then its remaining
suffix, after which every line starts in future input. Thus at most two future
events cover earlier records; exceeding two is an error, never event truncation.
Empty chunks preserve this carry. Integer wraparound/subtraction, f32 conversion,
alignment and the original placement `fma` order are retained; there is no rounded
floating chunk origin. Negative advances use these same serial transitions.

Unchanged chunk shape windows/identities survive edits only with exact source,
plan and boundary-proof dependencies. Source prefix/suffix reuse does not globally
retile a shifted suffix. Layout reuse compares complete old/candidate checkpoints;
reverse placement also compares incoming events and paragraph placement/atlas
dependencies. A count/hash match alone is not convergence. Worst-case changed
paragraph selection and GPU reflow remain linear; no constant-time edit guarantee.

Every real instance window has an ordered indirect draw range. Admission zeros
all old counts; candidate counts stay zero until a final GPU global-success
publication. An error in any chunk leaves the entire owner invisible. One logical
owner, one normal four-byte status readback and existing deferred frame fusion
remain. Immutable dispatch parameter windows prevent queued work observing later
writes. Intermediate shaping submissions consume bounded scratch before reuse,
without per-chunk GPU waits. Generation/revision/ancestor-failure rules, busy
rejection, plain-mode switching, atlas invalidation, disposal and device loss apply
to all windows. Terminal release covers stages created before executor preparation.

Exact tests force small windows/arenas, compare all six shaped fields and instance
bits with serial/oracle output, stress reverse carry and empty/deleted records,
exercise reuse/split/merge/atomic failure/lifecycle, and verify a real default-limit
paragraph whose required Stage 1 line bytes exceed 128 MiB. Untimed diagnostics
iterate every actual ordered range and encoded command, not just aggregate counts.

### Paged implementation data and dispatch ABI

`planComputeTextParagraphChunks(font, paragraphs, limits, budgets?, previous?)`
returns source chunks, per-paragraph plans, closure seeds, maximum local capacity,
effective binding/arena budgets and peak transient workspace bytes. Each source
chunk retains exact text, UTF-16/scalar starts and counts, segment end indices,
plan identity and 66-scalar boundary contexts. Persistent chunk IDs and resource
windows belong to `ComputeTextData._chunkState`; they are not public exports.
`submitComputeTextPrepared(data, prepare)` shares the normal owner admission,
cancellation and completion machinery without allocating a document-sized slab.

The layout shader binds eight windows in this order: immutable params, shaped
records, scalar flags, shared atlas slots (read-only), local instances, local
metadata, ordered carry, reusable replay scratch (read-write). Summary, measure,
replay and placement are distinct ordered dispatches. The final four-binding
publication kernel reads metadata/carry and writes one local indirect record and
the owner's status. Parameters are 16 words per chunk:

| Word  | Meaning                                                                                  |
| ----- | ---------------------------------------------------------------------------------------- |
| 0–5   | Required capacity, paragraph scalar start/count, paragraph index, first/last chunk flags |
| 6–9   | Width, line height, spacing and scale f32 bits                                           |
| 10–13 | Font glyph count, logical chunk index, dirty-shape flag, alignment                       |
| 14–15 | Reserved zero                                                                            |

Every metadata window is 1,024 bytes. Checkpoints at words 0/32 are the current
entry/exit; 64/96 retain the previous generation. A checkpoint's words 0–13 are
glyph ordinal, line ordinal, line start ordinal/cluster, line-start X, pen X/Y,
content end, remembered break ordinal/X/content end, previous cluster, visible
count and cluster after the remembered break; 14–31 are zero. Integer positions
are i32 bit patterns. Metadata words 128–133 hold actual count, visible count,
next nonempty cluster, its preceding scalar flag, status and document line base.
Words 134–138 retain width/scale/successor dependencies and validity; 139/140
report actual measure/placement execution. Words 160–176 retain the complete
incoming reverse carry and 180–186 retain width/height/spacing/scale/line-base/
paragraph-index/alignment placement dependencies.

The 512-byte owner carry contains status/first-failing chunk at words 0/1,
document line base at 4, next nonempty cluster/preceding flag at 6/7, the current
32-word forward checkpoint at 16 and reverse event count/two complete events
at 48–64. Replay scratch starts with four words (event count and zero padding),
then four words per capacity record (integer X/Y, local visible prefix, zero),
then eight words per event, reserving capacity + 4 events including future carry.
Event fields are first/end glyph ordinals, line ordinal, start X, final f32 width,
first source cluster and two zero words.
For M local records, each ordinary closure consumes a distinct local record;
one remembered incoming break can add one earlier closure, and true paragraph
end adds at most one final closure. Local events are therefore at most M + 2,
including empty chunks; appending the two incoming future events fits capacity + 4.
This bound does not depend on signs or monotonicity of advances.

Reverse carry includes events ending **entirely before** the current chunk when
they still close earlier records. Filtering by intersection with the current
chunk would silently lose a retroactive closure; the adversarial two-event test
checks this exact case with right alignment. Placement binary-searches ordered
event intervals without changing any wrapping or floating-point arithmetic.
Placement grids linearize X/Y and stride over records when a small dispatch
budget cannot cover a fitting dedicated span in one invocation per record.

Scalar selection is scanned without retaining a paragraph-sized scalar vector.
Normal source/proof materializations and shape inputs are chunk-bounded. An
unavailable space-independence analysis or an indivisible required workspace
failure rejects before closure installation. Required counters reserve the u32
sentinels and signed scalar-cluster ABI. Normal workspace excludes immutable font
program/tables, whose independent existing device bounds remain enforced.
Their uploads retain the resident executor's plan-identity cache rule rather
than re-uploading the same immutable font program for every chunk.
The normal 8 MiB reservation allows up to 4 MiB for a legacy admission decode
probe, 1 MiB for bounded chunk decode/upload views and 3 MiB for reusable GPU
workspace. A dirty batch larger than the admission probe budget is checked with
the bounded paragraph scan before any legacy slab, even when an existing fitting
owner grows into paged mode. Uneven safe segments split into additional storage
chunks before their shared shaping stride can exceed the workspace reservation;
optional grouping never rejects required independently fitting segments.
General JavaScript object/source metadata and immutable font data are not GPU
workspace. Dedicated indivisible spans retain their explicitly checked exception.
Arena windows have optional budget-aware growth of at most 25%; freed compatible
capacity windows are first-fit reused. Extra empty arenas beyond one per class
are retired after safe detachment and status settlement; an empty dedicated arena is never an ordinary
spare. Projected retained transient bytes are checked against the current plan,
not merely new requested sizes. Obsolete dedicated workspaces are retired when
their high-water allocations would exceed the current normal reservation (or
25% bounded dedicated headroom). Resource retirement preserves privately mapping
status buffers until their completion callback, including legacy↔paged switches.

Retirement is revision-owned, not an arbitrary delayed destruction callback.
Before changing a published chunk list, replace the owner's draw view with an
empty range list. The renderer always rechecks that list before bundle replay;
its changed identity invalidates the cached bundle, so a preparation-gap frame
cannot execute retired views. Already submitted commands remain queue-ordered.
Removed arenas/workspace buffers are detached from the live allocator immediately,
but retained in that revision's retirement batch until its private status settles.
Freed windows are quarantined in the same batch and cannot be zeroed/reused by a
newer allocation before settlement. Only after detachment and settlement may
buffers be destroyed and windows enter surviving arenas' free lists. Later
revisions cannot reuse pending buffers; an ancestor callback settles only its own
batch. Failed admission detaches all candidate views before terminal resource
release. Disposal/loss drains every outstanding batch exactly once after the
existing deferred-work flush/abandon barrier and makes layer replay inactive.
No additional GPU waits/readbacks are introduced.
Pending retirement bytes are in-flight generation storage, reported separately
from the active 3 MiB GPU workspace reservation; delaying their destruction does
not raise the normal active workspace budget.

Paged shaping resolves program/analysis metadata once per consecutive selected
plan and decodes/maps each bounded context once. Chunk input/flag preparation
shares views of that context, including the same complete proof neighbors; it
does not shape the halo. Both storage paths use one lazy scalar mark classifier.
Stable seam starts are traversed with an ordered cursor rather than rescanning
all retained boundaries per segment. Routing retries retain the admitted source
and layout snapshot, never the caller's later-mutated arrays/options.

Move paragraph wrapping and line placement onto the GPU, after resident
shaping, so a host such as LineLayoutPad can draw plain-text documents without
any LLX WASM shaping or layout. The work is **incremental**: each step ships a
working, explicitly bounded subset and the next step restores lost
functionality. Earlier steps may drop editor features (caret, selection, hit
testing) as long as the loss is explicit and the record formats do not have to be
redesigned later.

### Incremental roadmap

| Step                           | Babylon Lite                                                                                                                                                                                                                                                                     | Pad                                                                                                                                                               | Functionality restored                                                        |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **1 (this spec)**              | `updateComputeTextParagraphs`: shape each paragraph once, GPU greedy wrap at shaping-safe spaces, emergency breaks, hanging trailing spaces, left/center/right alignment, paragraph spacing, font line metrics. Per-line records written on the GPU (internal).                  | "Compute Text (GPU layout)" mode: no LLX shaping/layout on the path; plain-text documents drawn from GPU layout. Caret, selection, hit testing, markers disabled. | Wrapped, aligned plain-text drawing with no LLX.                              |
| **2 (implemented)**            | Paragraph slots with stable identities; reshape only dirty paragraphs; width/alignment changes re-run layout without reshaping.                                                                                                                                                  | Per-keystroke dirty-paragraph updates.                                                                                                                            | Clean paragraphs avoid reshaping; dirty preparation/reflow can remain linear. |
| 3                              | Revision-bound async queries over line/cluster records: line extents, cluster↔x, caret stops, hit test.                                                                                                                                                                          | Caret, click-to-place, arrow keys, selection highlight.                                                                                                           | Editing.                                                                      |
| 4                              | GPU caret/selection rendering from cluster records (no per-keystroke readback).                                                                                                                                                                                                  | Remove query round-trip from typing.                                                                                                                              | Latency parity.                                                               |
| **5 (wrap/place implemented)** | Within-paragraph parallel wrap (blocked scans + candidate break search) and parallel placement; negative-advance paragraphs retain serial wrapping. Same-task frame submission fusion is implemented. Dirty safe-break-chunk reshaping and device-level pipeline sharing remain. | Large documents, many panes.                                                                                                                                      | Scale.                                                                        |
| 6                              | Multiple styles per paragraph (style runs, per-run fonts/sizes/colors, line metrics from the tallest run).                                                                                                                                                                       | Bold/italic/size/color, decorations.                                                                                                                              | Rich text.                                                                    |
| 7                              | Line-local bidi reordering, more scripts, UAX #14 classes beyond spaces, tabs, justification.                                                                                                                                                                                    | Remaining eligibility gates removed.                                                                                                                              | Feature parity with LLX.                                                      |

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
- Latin, Common-only and Devanagari paragraphs are accepted, including Devanagari
  and Latin/Common paragraphs in the same call. A Devanagari paragraph may contain
  the Common scalars the indic plan admits (digits, punctuation, danda, NBSP).
  Latin letters inside a Devanagari paragraph, or Devanagari mixed with another
  complex script, still reject synchronously (`mixed scripts`). Other complex
  scripts stay rejected.
- The indic profile lazy-loads after synchronous validation, the same contract as
  `updateComputeTextData`. In a first-load call, errors detected after the load
  trigger (another paragraph's selection error, proof unavailable, limits)
  surface as a rejected promise with the same wording and order; once the profile
  is loaded they are synchronous.
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
   Indic plans use the profile's
   [paragraph space proof](#indic-paragraph-space-proof) instead of
   `SegmentAnalysis.safeAfter`, with the same meaning and the same unavailable
   error.
3. **Greedy fit.** For each non-space record, if the line already holds at least
   one record and `(end of this record - line start) * scale > maxWidthPx`, the
   line breaks at the most recent break opportunity within the line.
4. **Emergency breaks.** When the overflowing line has no opportunity, it breaks
   immediately before the overflowing record if that record begins a new
   cluster (its cluster differs from the previous record's) and its first scalar
   is not a combining mark (`\p{M}`) or U+200D. Otherwise the line keeps
   overflowing until the next allowed position. For indic plans the test is
   instead "the record's first scalar starts an extended grapheme cluster"
   (Unicode 16 GCB/InCB rules, including GB9c conjuncts, over the admitted
   repertoire), so an emergency break never splits a grapheme. Emergency breaks do **not**
   reshape; glyphs keep whole-paragraph context. A line always holds at least
   one record, so an unbreakable cluster wider than the box overflows.
5. **Alignment.** Line width = end of the last non-space record minus line start
   (0 for empty or all-space lines). Offset is `0`, `(maxWidthPx - width) / 2`
   or `maxWidthPx - width`, clamped to `>= 0` (an overflowing line starts at the
   left edge, as in CSS).

### Indic paragraph space proof

`computeTextParagraphSpaceProof` (in `compute-text-segment.ts`) dispatches indic
plans to the profile seam, which is implemented in the lazy
`compute-text-indic-paragraph.ts` (`proveIndicParagraphSpaces`).

**Claim.** For `T = L · U+0020 · R` with `R[0] ≠ U+0020`, a proven space satisfies
`shape(T) = shape(L·␠) ++ rebase(shape(R))` on glyph id, cluster, advance and
offset. A line ending at a proven space therefore reproduces whole-paragraph shaping
exactly. The proof keeps the buffer as `A · σ' · B` after every pipeline step,
where σ' is in the space glyph's closure B\*.

Static obligations (all must hold, or the proof is `null`):

- **O1** U+0020 is in no vowel-constraint pattern, so no dotted circle is inserted across it.
- **O2** U+0020 has ccc 0, does not decompose and starts no composition pair.
- **O3** U+0020 is indic category X and, in the syllable DFA, always forms its own one-scalar syllable.
- **O4** Reordering, broken-cluster dotted circles, masks and cluster merges are per-syllable (structural, given O3).
- **O5** No GSUB/GPOS lookup, root or nested, can match across σ': B\* is GDEF class 0, never skipped, and no crossing rule both reaches σ' and matches on both sides.
- **O6** Attachment propagation and hidden-glyph rewriting stay on one side (follows from O5).
- **O7** The plan's applied feature tags are a subset of the modeled set; an unmodeled stage makes the proof unavailable.

Dynamic conditions per space (a space that fails one is simply not a break):

- **D1** The scalar after the space starts an extended grapheme cluster.
- **D2** The words on both sides have at most E = 64 scalars, or reach the true paragraph end.
- **D3** No crossing rule has every left slot satisfiable by the left word's glyph
  closure and every right slot by the right word's.

D2 makes the existing windows exact: the decode window is 65 scalars on either
side, so a word of at most 64 scalars is delimited inside it and a longer word
reaches the slice edge and is unsafe exactly as on the full text. Incremental flag
reuse keeps its ±66 radius and, for indic, extends forward to the end of the
grapheme chain (`chainEnd`). Chunk contexts of 66 cover the space reach, and grapheme
flags never need context across a seam, because every non-initial chunk starts
after a proven space (D1).

A `null` proof gives the existing unavailable error for paragraphs with break
candidates. The same proof also drives run segmentation for indic plans (CTF-16):
`segmentComputeTextRuns` splits long runs and monolithic paragraphs at proven spaces,
reusing cached paragraph flags where present. `analyzeComputeTextSegmentation` itself
stays null for indic plans.

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

### Dirty-range preparation and physical slots

The stage retains a string-array snapshot **only after successful admission**,
plus the font, size, device, generation, shaped-buffer identity, pool size,
record capacity, flag stride, logical order and last uploaded layout parameters.
It is internal state, not a new public API. Compare strings by `===`, finding
the common prefix `a` and then common suffix `b` without overlap. Replace
previous logical `[a, previousN-b)` with new `[a, nextN-b)`. Duplicate strings
are positional; disjoint edits deliberately produce one larger dirty middle.
An unchanged repeat and a pure deletion have no shaping work.

Each logical paragraph owns a physical shaped run slot. Release the old
middle's slots, find the first contiguous free range of `k` slots, and allocate
the dirty batch there. A one-paragraph edit usually reuses its own slot; queue
ordering makes immediate reuse safe. If no contiguous range exists, or growth
is needed, compact with a full rebuild: logical `i` maps to slot `i`.
The preferred pool is `N + max(8, ceil(N/4))` and preferred record capacity is
`ceil(maxNeeded * 1.25)`, but both allowances are optional. Required storage
(`pool = N`, capacity = `maxNeeded`, at least one) must fit first. A deterministic
monotone search first selects the largest capacity up to the preferred capacity
that fits the required live pool, then the largest pool up to the preferred pool
that fits that capacity. This favors growth headroom over empty insertion slots
without rejecting content merely because the preferred reservation does not fit.
There remains exactly one physical slot per logical paragraph and one uniform
record stride; this is sizing policy, not post-shape chunking.

Preflight every pool/capacity-dependent allocation against both actual device
limits (`maxBufferSize` and `maxStorageBufferBindingSize`) and safe integer
arithmetic, before closure/atlas mutation:

| Buffer                          | Bytes                                       |
| ------------------------------- | ------------------------------------------- |
| Layout input                    | `(16 + pool + pool * ceil(capacity/4)) * 4` |
| Layout info                     | `max(1, pool * 2) * 16`                     |
| Layout lines and scratch tables | `max(1, pool) * capacity * 48`              |
| Resident shaped records         | `max(1, pool) * (4 + capacity * 6) * 4`     |
| Resident draw instances         | `max(1, pool * capacity) * 12`              |

Unsegmented shaping scratch uses at most `pool * capacity * 32` bytes, bounded
by the line/scratch allocation; segmented shaping retains its existing independent
safe-boundary sizing and bounds checks. The draw allocation formulas are shared
with ordinary resident updates. All sizes are already four-byte aligned.
Reusable CPU input words may grow by doubling, clamped to the smaller device
limit in whole words; admission also checks that actual high-water byte length
before prepare binds it. GPU draw/layout allocations grow to exactly the checked
size, or retain an already admitted high-water buffer. Failed required storage
throws the existing paragraph-prefixed `Error`, naming the buffer, exact bytes,
both device limits, pool and capacity. A ready block draws nothing after rejection
(`indirect = [6, 0, 0, 0]`) and its atlas is unchanged.

Reduced optional headroom can cause more capacity-growth rebuilds or earlier
pool exhaustion/fragmentation compaction; cache stamps and first-fit reuse rules
are unchanged, and every pool/capacity change still rebuilds all live paragraphs.
Flag stride is `ceil(scalarCapacity/4)`; the current
shape expansion bound is at least the scalar count, so record capacity is also
the scalar high-water bound. Check dirty scalar lengths explicitly; overflow
rebuilds. All buffer arithmetic and dispatches are checked against device limits.

Only the dirty strings enter selection, mapping, grouping, segmentation and
closure. Unsegmented group output indices are `runBase + relative`; segmented
groups and their table remain batch-relative. Stitch dispatches x=`k` and
y=`max(1, max ceil(segmentCount/64))`, checks `wg.x < params.x`
(`params.x = k`) and writes absolute
`run = params.w + wg.x`, with `params.w = runBase`. Draw buffer sizing uses the
pool, not `k` or logical N. Unsegmented shaping uses the pool record stride,
including its scratch-capacity shader bucket; headroom reduces recompilations.
Resident execution computes this y value once per batch. When y=1, it selects
the exact baseline stitch shader and explicitly dispatches `{x: k, y: 1}`.
When y>1, it selects the unchanged tiled shader. These are separate shader
objects/pipelines with separate binding sets over the same buffers, never a
uniform branch inside one shader. Both pipelines are explicitly prepared on
first segmented admission, before the existing warm draws complete; a cached
per-device resident-ready flag is set only after both preparations succeed.
Later admissions do not repeat that preparation or add an await for it.
The selected route is part of the pack schedule rebuild key, so 1-to-2-to-1
row transitions re-record exactly one stitch dispatch in its original position
without compiling a previously unused pipeline. Release disposes both variants.
For the tiled route, each row reduces all run headers across its 64 lanes, then copies only
its consecutive 64-segment tile using the unchanged ordered record-copy loop.
The separate paged executor is unchanged: it still imports the tiled
`COMPUTE_TEXT_STITCH_KERNEL`, caches its own shader, and dispatches x=1/y=1.
The final tiled row consumes any remainder, so even a high-segment-count paged
chunk retains the whole-run loop; it does not use the resident route predicate.
All tiled rows use the same run-global failure/capacity decision;
only row zero lane zero publishes the original header. Added shared memory is
24 bytes, with no new storage buffers/bindings or status/lifetime changes.
Global-y ragged over-dispatch and repeated header reads are costs, not proven
neutral. See `33-text.md` for the existing u32-addressing contract/proof scope.
The one-row baseline route is a conservative fused-pipeline control, not proof
that the tiled one-row kernel is pure overhead: that kernel still parallelizes
header reduction and removes the old serial per-lane header scan.
Zero dirty runs omit both shape and stitch work entirely.
Both shape paths initialize all four header words before processing each run,
so successful slot reuse cannot retain old failure witnesses or record data in
the diagnostic header.

Dirty preparation uses the already decoded scalar/nominal views, without
decoding clean text, CPU wrapping, measurement, visible-instance counting or
placement. One scalar pass packs flags directly into the dirty slots' reusable
word range, including zero padding.
The fitting path's optional incremental flag cache retains one entry per live
paragraph: selected plan identity, exact scalar/glyph ID vectors and one flag byte
per scalar. The current decode uses a fresh shared batch slab (not a reusable
pool), but retaining a view can pin unrelated old dirty paragraphs and UTF-16
slack. Cache entries therefore copy scalar/glyph subviews into exact-sized backing
allocations; flag bytes are also exact-sized. Retained backing bytes are 9 per
scalar, excluding small objects/arrays and current in-flight candidates. This is
source flag reuse only: decode, nominal mapping, GPU shaping/layout, full dirty
flag upload ranges and little-endian u32 packing remain unchanged and linear.

Only a non-full valid legacy candidate with unchanged dirty paragraph count and
the same selected plan can reuse flags. Compare both scalar and glyph IDs for a
non-overlapping common prefix `p`/suffix `s`. Recompute `[max(0,p-66),
min(N,N-s+66))`, extending either boundary to the paragraph edge when within 66
scalars. If that interval covers at least half the paragraph, recompute all flags.
Outside it, copy exact previous prefix/shifted suffix bytes. The radius includes
65 **visited** proof positions (including ignored/free glyphs and walk exhaustion)
plus the flag's immediate-right-scalar test. Never shorten it to meaningful glyph
count or UTF-16 distance. Resolve analysis lazily only for recomputed candidate
spaces. A failed evaluation does not publish a partial entry.

The entry array travels on the existing revision/epoch-bound candidate: preserve
clean paragraph prefix/suffix entries, replace dirty entries, publish only at the
existing successful admission epoch. Every normal cache invalidation, failed
completion/ancestor rule, plain/paged mode switch, terminal loss/disposal and
capacity rebuild drops or fully regenerates it. Paragraph insertion/deletion/split
does not reuse dirty-entry correspondence. No old revision history is retained.
Exact seeded differential tests compare bytes/packed words to full recomputation;
separate untimed counters demonstrate evaluations avoided rather than cache-ID
matches. Render/status/upload and Stage 2 resource lifetime behavior are unchanged.
Analysis is resolved lazily at the first candidate space of each plan group;
the program/segmentation caches retain the analysis per immutable plan/program.
ASCII non-spaces bypass Unicode mark tests. Non-ASCII mark classification is
cached lazily by scalar in 256-scalar pages, independent of font or document.
Natural line height is cached per block's immutable font/size.

Input words and float views are retained per stage. Upload the 16-word parameter
header every update, the changed logical-order interval only, and the dirty
contiguous flag range only. A rebuild uploads the complete input. Clean flags,
shaped records, slot info and line records remain GPU-resident.

Closure is additive between rebuilds: close only dirty seeds/plans and union new
glyph IDs into the installed slot set. Existing slot indices are append-only in
the caller-owned atlas. Skip outline extraction, atlas uploads and slot uploads
when no glyph is new; upload only new slot entries otherwise. Atlas GPU
capacity growth or a non-additive slot-table change requests a full rebuild.
Full rebuilds reset the block's slot table exactly as the plain update does.
Deleted glyphs may remain until compaction. `glyphCount` is always the immutable
font's count, not closure size. Instance slot words are atlas indices: bitwise
comparison uses the same caller-owned storage/curve set for fresh and incremental
blocks; independent atlases can assign different indices, despite identical
glyph geometry. Only defined output is compared: headers and active shaped
records of live slots, active line records, draw-order instances, indirect
arguments and status. Unused capacity, free slots and stale extra closure
entries are not outputs.
Line records are defined only for successfully measured slots. A failed slot
defines its header/status and slot info, not its reserved line range. On an
overall failure, logical bases and instance contents are undefined and the
indirect instance count is zero; successful slots still define their line records.

Atlas suffix uploads are opt-in through internal
`appendSharedAtlasGpu(device, atlas, [curveTexels, bandTexels, slotCount, version])`.
It refuses device/version/capacity mismatches before writing. Within capacity,
upload only newly appended texels (split at 4096-texel row boundaries) and the
new metadata suffix at `oldSlotCount * 48` bytes. Update the uploaded version.
The ordinary `ensureSharedAtlasGpu` path is unchanged and does not retain append
code in CPU-only bundles. A refused append retries full paragraph preparation;
already extracted outlines remain safely in the append-only CPU catalog.

| Bit | Meaning                                               |
| --- | ----------------------------------------------------- |
| 0   | `SPACE`: scalar is U+0020 (hangs).                    |
| 1   | `BREAK_AFTER`: rule 2 opportunity after this scalar.  |
| 2   | `NO_BREAK_BEFORE`: combining mark or U+200D (rule 4). |

For indic plans bit 1 is set only on proven spaces, and bit 2 marks scalars that
do not start an extended grapheme cluster.

### Layout input buffer (`u32` words, read-only)

```
[0] logicalCount     [1] capacity (records per physical slot)
[2] scaleBits (f32)  [3] glyphCount
[4] maxWidthPx (f32, +Inf allowed)  [5] lineHeightPx (f32)
[6] paragraphSpacingPx (f32)        [7] align (0 left, 1 center, 2 right)
[8] measureBase      [9] measureCount
[10] measureMode (0 contiguous dirty slots, 1 all logical paragraphs)
[11] pool            [12] flagStride (words per slot)
[13 .. 16) reserved (zero)
[16 .. 16+pool) logical-to-physical order table (only first logicalCount live)
[16+pool ..) flags: pool * flagStride words, four scalar bytes per word
```

Flags for slot `s`, scalar `c` are at word
`16 + pool + s * flagStride + floor(c/4)`, byte `c % 4`.

### Slot info and logical bases (`vec4u`, read-write)

First `pool` entries are persistent slot measurements:
`(visibleCount, failureStatus, lineCount, 0)`. Prefix never modifies these.
Following `pool` entries are logical bases, of which first `N` are active:
`(exclusiveInstanceBase, 0, lineCount, globalFirstLineIndex)`.
Keeping capacity for pool logical bases avoids reallocating on insertion.

### Line records (`vec4u` per line, read-write, `capacity` lines per paragraph)

`x` first shaped record, `y` record count (including hanging spaces), `z` width
in px (f32 bits, excluding hanging spaces), `w` source scalar index of the first
record (reserved for step-3 queries). Line `l` of physical slot `s` is at
`s * capacity + l`. Lines ≤ max(1, records) ≤ capacity, so this never overflows.

The same `lines` storage binding also holds persistent per-record tables after
the `pool * capacity` line records. Record `i` of physical slot `s` uses two
`vec4u` at `pool * capacity + 2 * (s * capacity + i)`:

- A: `(startX, startY, visibleBefore, contentEnd)`; signed sums use i32 bitcasts.
- B: `(lineIndex, nextNW, firstEvent, next)`; B.x is `prevBreak` until the final
  measurement phase replaces it with the record's line index.

Allocate/check `max(1, pool) * capacity * 48` bytes instead of 16 bytes per slot
record: **+32 * max(1, pool) * capacity bytes**, three times the previous lines
allocation. Tables for clean slots persist across edits, just like their lines.
Only active records' A and B.x are placement inputs.

## Pipeline Configuration

Three compute entry points in one WGSL module, each with the same 8 storage
bindings (`input`, `shaped`, `slots`, `instances`, `indirect`, `status`,
`info`, `lines`) to stay within the default 8-storage-buffers-per-stage limit:

1. `measure` — workgroup 256, one workgroup per dirty slot or logical paragraph;
   disabled when measurement is unnecessary (zero direct dimensions).
2. `prefix` — one workgroup of 256 lanes.
3. `place` — workgroup 256, one workgroup per display paragraph, lanes stride records.
   Disabled for an empty document; prefix still resets all indirect/status words.

Measure/place dispatch `count` workgroups, using `(min(count, limit),
ceil(count / limit), 1)` when needed, with linear paragraph index
`workgroup.x + workgroup.y * numWorkgroups.x`. Bounds retain the previously
supported logical paragraph count (`ceil(N/64) <= limit`) and also check the
new two-dimensional dispatches.

Dispatch order inside the pack task: [stitch] → measure → prefix → place.

## Shader Logic

**measure(i):** mode 0 selects slot `measureBase+i` for `i < measureCount`;
mode 1 selects `order[i]` for `i < N`. Read that slot's shaped header; validate status, count ≤ capacity,
glyph < glyphCount, slot ≠ MISSING (same codes as the default pack: 1
unsupported lookup, 2 capacity, 3 missing slot) and count visible slots. On
failure write `info[p] = (0, status, 1, 0)`. The following serial pseudo-code
remains the **normative reference semantics** (also the negative-x-advance
fallback, executed verbatim by lane 0):

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

**Parallel measure:** each lane owns a contiguous block of
`span = ceil(count/256)` records. Validate glyph/slot indices in parallel, count
visible slots, and detect any negative x advance with workgroup atomics.
Header failure takes precedence, then count overflow (2), then missing glyph/slot
(3). Lane 0 publishes failure info; uniform workgroup loads guard all returns.
Blocked scans use local accumulation, a 256-lane Hillis–Steele summary scan,
then local application. Workgroup and storage barriers separate producer and
consumer phases:

1. Exclusive i32 sums `startX[i]`, `startY[i]`, and u32 `visibleBefore[i]`.
2. Forward last-value scans: `contentEnd[i] = end(k)` for the last non-space
   `k <= i` (0 if none); `prevBreak[i]` is the last break opportunity `j <= i`
   (NONE if none). `end(i) = startX[i] + xAdvance[i]`.
   `breakOpp(j)` means `j+1 < count`, increasing clusters and BREAK_AFTER on
   `cluster[j+1]-1`.
3. Reverse min scans: `nextNW[i]` is the first non-space `k >= i`, else NONE;
   `nextNW[count] = NONE`. `emergency(i)` means `i >= 1`, non-space, a different
   cluster than `i-1`, and no NO_BREAK_BEFORE.
   `selfEvent(i)` is `i` if emergency, otherwise `nextNW[i+1]` if breakOpp,
   otherwise NONE. `firstEvent[i] = min(selfEvent(k), k >= i)`.
4. Parallel candidate-start search: only `s == 0 || breakOpp(s-1) ||
emergency(s)` can start a line. For each candidate compute:

    ```
    i0 = first i in (s, count) with !(f32(end(i) - startX[s]) * scale <= maxWidth)
         // monotone-hinted gallop, then binary search on non-decreasing end(i)
    if none: next(s) = FINAL
    else i = nextNW[i0]
    if i == NONE: next(s) = FINAL
    else if prevBreak[i-1] != NONE && prevBreak[i-1] >= s:
      next(s) = prevBreak[i-1] + 1
    else:
      k = firstEvent[i]
      if k == NONE: next(s) = FINAL
      else if prevBreak[k-1] != NONE && prevBreak[k-1] >= s:
        next(s) = prevBreak[k-1] + 1
      else next(s) = k
    ```

    Each lane visits candidate `s` in ascending order and retains only its prior
    first-false index `hint`, initially zero. Start at `low = max(s+1, hint)`;
    `count` is a sentinel and is never passed to the fit predicate. If `low`
    already fails, keep it. Otherwise bracket the first failure by probing
    `low+1`, `low+3`, `low+7`, ... clamped to `count`, then use the original
    first-false binary-search update inside that bracket. Saturate the gallop
    step at `count-low` to avoid overflow. Preserve the identical predicate:
    load `startPos = bitcast<i32>(A[s].x)`. For `i+1 < count`, load
    `endPos = bitcast<i32>(A[i+1].x)`; for the final record, retain
    `endPos = bitcast<i32>(A[i].x) + shapedAdvance[i]`. Use a real conditional,
    not `select`: there is no valid successor at `count`. In both cases evaluate
    `f32(endPos-startPos)*scale <= maxWidth` in the identical order.
    This successor-prefix equivalence is valid only on the guarded non-serial
    P7 path, with `i < count`. The scanned u32 prefixes are congruent to the
    true prefixes modulo 2^32 even across lane boundaries. When no advance is
    negative and every signed-overflow check passes, induction in global record
    order bounds each true prefix, including the final sum, by `0x7fffffff`.
    Congruence therefore gives exact nonnegative i32 prefixes, so `A[i+1].x`
    equals the original prefix-plus-advance bits. The preceding storage barriers
    make cross-lane successor prefixes visible; A.x remains immutable during P7.
    Counts zero and one make no predicate calls; count two uses the original
    final-record expression. The non-final probe removes one static storage load
    but adds a branch; compiler, cache, register, occupancy and latency effects
    are not guaranteed.
    Nonnegative advances and no signed-prefix overflow make this predicate
    monotone both in `i` and candidate `s`, because admission guarantees positive
    finite scale. The previous first-false cannot move left for a later candidate.
    Negative/overflow paths skip this search entirely and retain serialMeasure.
    The gallop is logarithmically bounded even after a huge advance followed by
    a long zero-advance run; it is not an unbounded linear two-pointer walk.
    `+Infinity` eventually reaches the count sentinel without reading it.

5. Lane 0 follows `0 -> next(0) -> ...`, emitting `[s,next)` with width
   `contentEnd[next-1] - startX[s]`, or final `[s,count)` with width
   `contentEnd[count-1] - startX[s]`. Empty paragraphs emit one empty line.
   `contentEnd` is global and never reset at line starts. For nonnegative
   advances this orbit is exactly the normative greedy walk: skipped records
   after a normal break are either whitespace or non-overflowing and contain no
   later break opportunity. A negative advance (or signed prefix overflow,
   which also invalidates monotonicity) selects the serial fallback instead.
6. Publish slot info. Each nonempty lane binary-searches only its first record's
   containing line. Sweep that lane's ascending records with a cached next line
   start, advancing while `nextLineFirst <= record`. Emitted line starts are
   strictly increasing on both chain and serial paths, so the sweep returns
   the same largest line index whose start is no greater than the record.
   Empty lanes write nothing; one-line paragraphs use the count sentinel.
   Replace B.x with each record's line index. A and B.x are filled even on
   fallback. No other scan, table, line-chain, failure or placement math changes;
   hints and sweep state are lane-local registers, with no new storage or dispatch.

**prefix:** read `info[order[p]]`; exclusive scans of visible counts (instance bases) and line counts
(global line bases) using a 256-lane span scan; the lowest failing paragraph's
status is written to `status[0]`; `indirect = [6, total visible or 0, 0, 0]`.
Write bases to `info[pool+p]`, not persistent slot info. Failure selection and
overflow precedence use logical order, including cached clean failures.

**place(p):** if `status[0] != 0` return. Select `s = order[p]` and logical
bases at `info[pool+p]`. Lanes stride shaped records. Record `i` selects line
`l = B[s,i].x`, loads its line record and `lineX = A[s,line.first].startX`.
For every visible glyph, write instance `paragraph.instanceBase +
A[s,i].visibleBefore` directly. The normative serial placement sequence below
is unchanged mathematically: `penX = A[s,i].startX`, `penY = A[s,i].startY`,
both global across lines, and alignment uses the same line width:

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

Width changes remeasure **all logical paragraphs without reshaping**. Measurement
depends only on shaped records, slots, flags, width and immutable font scale.
Alignment, line height and paragraph spacing changes need only prefix/place.
Text edits measure dirty slots; combined width/text edits measure all logical
slots after dirty shaping. Instances are fully rewritten by place and may grow
without preserving old contents.

Invalidate the snapshot on synchronous admission/pre-submission errors, async
status failures, plain resident updates, disposal/device loss/change, font/size
changes, unexpected shaped-buffer replacement, and any rebuild. Generic
submission plumbing uses a revision to distinguish admitted updates without importing
paragraph code. A late completion failure invalidates only its own snapshot;
it must not discard a subsequently admitted full rebuild. Busy rejections
invalidate reuse but never mutate the in-flight input. Commit the candidate
snapshot only when admission succeeds, not when preparing words/flags. Any
failure during incremental preparation abandons the candidate. Pool/capacity
changes perform a full preparation and install before committing new state.
The snapshot also retains atlas identity/version after admission. External
mutation conservatively rebuilds, because callers can install outlines for a
previously empty glyph (which changes cached visibility and slot words). Ordinary
append work from this submission records the new version on commit and does not
invalidate the next edit.

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
- **Parallel native:** retain the pre-parallel measure/place WGSL only as a test
  fixture. Compare complete reserved line ranges, info, instance bytes, indirect
  and status for fixed-seed records at 0/1/256/257/2049/4097 lengths, small/large/
  infinite widths, all alignments, space/break/protected flags, repeated clusters,
  empty glyph slots, y advances, negative x and signed-prefix overflow. Exercise
  2D-linearized workgroups, permuted order, clean-table reuse, dirty remeasurement,
  width remeasurement and failure recovery. Unit bounds include the extra tables.
- **Incremental unit:** positional prefix/suffix diff (duplicates, empty,
  replace-all, split/merge), first-fit contiguous slots, fragmentation fallback,
  pool sizing and snapshot invalidation decisions.
- **Incremental native:** compare the defined outputs bit-for-bit with a fresh
  block sharing the atlas after typing/deleting a character, split/merge,
  multi-paragraph paste/deletion, append, disjoint edits, width/placement changes,
  unchanged calls, capacity/atlas growth, failures/fixes and plain alternation.
  Assert one edited paragraph in a 100-paragraph document uploads only dirty
  run/flags/new closure entries, dispatches shape/stitch for that batch, and
  keeps clean shaped/info/line buffers. Include nonzero-base segmented and
  unsegmented updates, zero-run updates, and lowest-logical-failure selection
  with a permuted physical order.

- **Devanagari (CTF-03).** Unit: `compute-text-indic-paragraph.test.ts` (O1–O7
  and D1–D3 mutations, slice/full equivalence at word lengths 63–66),
  `compute-text-paragraph-flags-indic.test.ts` (grapheme bits against
  `Intl.Segmenter`), `compute-text-paragraphs-indic.test.ts` (lazy load, synchronous
  mixed-script rejection, seam routing, chain extension),
  `compute-text-paragraph-chunks-indic.test.ts` (paged capacity, seams, workspace
  bound), `compute-text-paragraph-references.test.ts` and
  `compute-text-indic-unicode.test.ts` (grapheme table). Native:
  `devanagari-paragraphs.test.ts` compares every admitted row of
  `deva-paragraph-harfbuzz-11.0.1.json` at 12 widths and three alignments with
  HarfBuzz lines, checks GPU output bit-for-bit against the serial WGSL oracle on
  reference records, and checks paged and incremental output against monolithic and
  fresh layouts. Build: the Latin-only paragraph consumer fetches no indic module, and
  `compute-text-indic-paragraph` is in the lazy indic chunk.

## File Manifest

- `packages/babylon-lite/src/text/compute-text-paragraphs.ts` (dirty-range admission and flag preparation)
- `packages/babylon-lite/src/text/_gpu/compute-text-paragraph-layout.ts` (persistent slot layout)
- `packages/babylon-lite/src/text/compute-text-data.ts` (shared submission)
- `packages/babylon-lite/src/text/_gpu/compute-text-resident.ts` (pack-stage hook)
- `packages/babylon-lite/src/text/_gpu/compute-text-shape.ts` (deterministic run headers)
- `packages/babylon-lite/src/text/_gpu/text-textures.ts` (opt-in atlas suffix uploads)
- `packages/babylon-lite/src/text/compute-font.ts` (`getComputeFontLineMetrics`)
- `packages/babylon-lite/src/index.ts` (root re-exports)
- `tests/lite/unit/compute-text-paragraphs.test.ts`, `tests/lite/compute/paragraph-layout.test.ts`
- `tests/lite/unit/text-atlas-append.test.ts` (suffix upload ranges and fallback conditions)
- `packages/babylon-lite/src/text/compute-text-paragraph-flags.ts` (shared monolithic/paged flag producer)
- `packages/babylon-lite/src/text/compute-text-indic-paragraph.ts` (lazy indic proof, grapheme flags, crossings)
- `tests/lite/compute/devanagari-paragraphs.test.ts`, `tests/lite/compute/devanagari-paragraph-references.ts`
- `tests/lite/compute/references/deva-paragraph-harfbuzz-11.0.1.json`, `tests/lite/compute/references/devanagari-paragraph-corpus.mjs`, `scripts/generate-devanagari-paragraph-references.mjs`
