# Module: Compute text shaping core

> Package path: `packages/babylon-lite/src/text/`
>
> Milestone 4 slice 1: **Devanagari, pinned Noto Sans Devanagari 2.007**.
> Milestone 4 and the broader user task remain open. See the durable
> [follow-up boundaries](compute-text-follow-ups.md), especially CTF-01/02/03/04,
> CTF-14 (whole RTL runs), CTF-16 (whole Indic runs), and CTF-17 (multi-record
> contextual Ligature admission). No arbitrary-font conformance or measured
> performance claim follows from admitting a font.

## Purpose

Shape homogeneous horizontal scalar runs into visual-order glyph records on WebGPU,
then optionally keep those records resident for atlas-backed rendering. There is
**one OpenType compiler/kernel family**, not a text-shaper engine and a HarfBuzz
engine. Independent references establish semantics and diagnose defects; they are
not selectable product modes.

The legacy text-shaper-backed CPU renderer is unchanged. In particular, its old
mark-offset and Arabic joining behavior can differ from corrected GPU output.
Use the corrected records—not a legacy CPU shaped string—as the placement oracle
for GPU rendering comparisons.

## Public API surface

All public imports use the single package root `@babylonjs/lite` (workspace
`babylon-lite`); there are no subpath exports or public WebGPU handles.

```ts
interface ComputeFont {
    readonly glyphCount: number;
    readonly unitsPerEm: number;
    // Branded font state; internal sfnt/cmap/layout caches are declaration-trimmed.
}
function createComputeFontFromBuffer(data: ArrayBuffer): ComputeFont;
function loadComputeFont(url: string): Promise<ComputeFont>;
interface ComputeTextGlyph {
    readonly glyphId: number;
    readonly cluster: number; // original scalar index, not UTF-16 byte/unit index
    readonly xAdvance: number;
    readonly yAdvance: number;
    readonly xOffset: number;
    readonly yOffset: number; // signed font units
}
function shapeComputeText(engine: EngineContext, font: ComputeFont, text: string): Promise<readonly ComputeTextGlyph[]>;
function shapeComputeText(engine: EngineContext, font: ComputeFont, texts: readonly string[]): Promise<readonly (readonly ComputeTextGlyph[])[]>;

interface ComputeTextData {
    readonly status: "ready" | "busy" | "lost" | "disposed";
    readonly fontSizePx: number;
}
interface ComputeTextUpdate {
    readonly completed: Promise<void>;
}
function createComputeTextData(
    engine: EngineContext,
    font: ComputeFont,
    storage: GlyphStorage,
    curveSet: CurveSetId,
    options?: { readonly fontSizePx?: number; readonly defaultColor?: readonly [number, number, number, number] }
): ComputeTextData;
function updateComputeTextData(data: ComputeTextData, texts: readonly string[]): Promise<ComputeTextUpdate>;
function disposeComputeTextData(data: ComputeTextData): void;
function createComputeTextLayer(data: ComputeTextData, options?: TextLayerOptions): ComputeTextLayer;
function updateComputeTextParagraphs(data: ComputeTextData, paragraphs: readonly string[], layout: ComputeTextParagraphLayout): Promise<ComputeTextUpdate>;
interface ComputeTextParagraphLayout {
    readonly maxWidthPx: number;
    readonly lineHeightPx?: number;
    readonly paragraphSpacingPx?: number;
    readonly align?: "left" | "center" | "right";
}
```

Batches preserve order and empty-run indexes, and use one supplied font. Different
font calls can be independently concurrent; this is not automatic multi-font
fallback. A run with mixed strong scripts is rejected. Common-only text keeps the
existing DFLT contract. Inherited-only and joiner-only strings reject. ZWJ/ZWNJ
are admitted for Devanagari only; Arabic joining controls remain unsupported.
The paragraph API accepts Devanagari paragraphs (CTF-03); see
[paragraph layout](compute-text-paragraph-layout.md#indic-paragraph-space-proof).

## Internal architecture and CPU/GPU ownership

CPU work is font/input preparation: sfnt/cmap parsing, nominal mapping, selected
feature/dependency compilation, Unicode property packing, capacity/binding checks,
proof construction, atlas closure and buffer scheduling. Immutable font/program
preparation is cached by plan; mutable scalar vectors are packed once per update.

GPU work is every per-text normalization, joining/mark order, syllable decision,
mask/reorder, GSUB/GPOS action, cluster merge, attachment propagation and pen/instance
placement. There is no CPU shaping fallback or memoized shaped-record replay.
JavaScript reads public shaped records on the readback API, and only the four-byte
status in the production resident completion window.

`compute-text-plan.ts` owns retryable lazy profile loading and shared stage plans.
Arabic loads `compute-text-arabic.ts` / generated joining properties / WGSL fragments.
Indic loads `compute-text-indic.ts`, `compute-text-indic-plan.ts`, generated Unicode
data and its fragments. Importing unused text or rendering features does not
register profiles or execute module-level cache allocations.

Profile support may also install a `paragraph` seam (`ComputeTextParagraphProfile`
in `compute-text-plan.ts`): `spaces` (the paragraph space proof), `flags` (paragraph
flag bytes for a scalar range) and `chainEnd` (how far an edit's flag dependency
extends forward). Only the indic profile installs it. Its implementation, including
the shared `collectComputeTextCrossings` helper (B\*, closure and crossing rules
without the generic span induction), lives in `compute-text-indic-paragraph.ts`,
which is loaded only in the lazy indic chunk. The base paragraph code keeps only the
dispatch (`computeTextParagraphSpaceProof`) and the shared flag producer.

### Program v3

All program/table words are little-endian u32; signed positioning values are
bit-preserved i32. Program header (16 words):

| Word  | Meaning                                     |
| ----- | ------------------------------------------- |
| 0 / 1 | magic `0x43545031` / version 3              |
| 2 / 3 | GSUB / GPOS root counts                     |
| 4 / 5 | GSUB / GPOS lookup descriptor origins       |
| 6 / 7 | descriptor capacities                       |
| 8 / 9 | four-word root origin / total root count    |
| 10    | four-word stage origin                      |
| 11    | reserved                                    |
| 12    | stage count                                 |
| 13    | root-presence width (up to eight u32 words) |
| 14    | optional profile data origin                |
| 15    | reserved                                    |

Root: `[tableBit<<16 | lookupIndex, mask, flags, stageIndex]`.
Stage: `[tableBit, firstRoot, rootCount, pauseId]`. Pause IDs are
0 none, 1 setupSyllables, 2 initialReorder, 3 finalReorder, 4 clearSyllables.
Root flags encode auto-ZWJ, auto-ZWNJ and per-syllable matching. Lookup descriptors
have eight words and rules sixteen; the shared compiler owns predicate directories,
class tables, nested SequenceLookupRecords and capability/ordered-selection proofs.
Extension subtables are resolved there, not interpreted as a second executor.

Descriptor words 0–7 are kind, lookup flag/filtering-set word, cooperative membership
capability, rule origin, rule count, expansion, first-glyph candidate directory and
ordered eligibility/forward proof. Directory entries are `(candidate origin,count)`
per glyph; candidates retain rule/subtable order. Predicate blobs are sorted glyph
sets or tagged class predicates with optional coverage and shared class-table index.
Rule words 0–9 are input origin/count, backtrack origin/count, lookahead origin/count,
action origin/length, mode and rejection marker; 10–13 optionally hold class-pair
row origin, class table, row length and grouped-rule count; 14–15 are reserved.
Modes: 1 Single, 2 Context, 3 Pair, 4 ClassPair, 5 reject, 7 MarkMark, 8 MarkBase,
9 Ligature, 10 SinglePos adjustment, 11 Multiple, 12 MarkLigature. Context actions
pack a record count followed by `(sequenceIndex, tableBit<<16|lookupIndex)` pairs.

The dispatch parameter ABI is eight words:
`[runCount, capacity, direction, glyphCount, rootCount, programWords, outputRun, profileId]`.
`profileId` is identity only: 0 default, 1 Arabic, 2 Hebrew, 3 Indic. Production has
no high-bit stage selector, counter, capture clock or stop hook. Stage snapshots are
test-only source replacements, keyed by memory tier and requested pause.

### Records, memory and output

Default/Arabic records have eight words:
`[glyph, cluster, xAdv, yAdv, xOff, yOff, flags, classAndAttachment]`.
Flags include tombstone bit 0; ligature id bits 8–15, component bits 16–19,
ligated bit 20, multiplied bit 21 and substituted bit 22.
Word 7 is **GDEF class bits 0–2** and backward attachment distance `<< 3`.
Classes above four normalize to zero. Mark-attachment classes remain in font tables;
there is no 16-bit run-capacity ceiling. Capacity is limited by real device bindings
and safe u32 record addressing.

Indic uses sixteen words: common 0–7; scalar 8; category/position/syllable/ccc 9;
mask 10; Unicode flags 11; sort key 12; reserved 13–15. Two record planes support
GPU scatter/normalization; four auxiliary words per capacity hold syllable/reorder
state. Per-run scratch is `capacity × (16 × 2 + 4) × 4` bytes. A 128-capacity Indic
allocation is **18,432 bytes**, above a 16 KiB workgroup limit, so the real factory
selects global scratch. Both allocator values and actual declarations are tested.

Variants are profile/word-width/mask/plane/memory-tier choices, justified by GPU
storage and execution needs. Default fast paths remain; source eligibility is not
speed evidence. Each independent run uses one workgroup. Proven generic LTR runs
can segment and stitch; Arabic remains whole (CTF-14), Indic remains whole (CTF-16).
Neither boundary lowers the public device-binding run maximum.

Each public/resident shape output has four header words (status/count/failure
lookup/failure index), then six i32 words per visual glyph. Resident pack compacts
visible atlas instances into 12-byte `[x:f32,y:f32,packedGlyphStyle:u32]` records and
uses indirect words `[6, visibleCount, 0, 0]`. CPU `TextData` may retain documented
dead allocator slots; the renderer skips their `0xffffffff` sentinel.

## Shader logic and corrected semantics

### Cached nominal proof and general state

`kernelSource` selects `splitFastState = simpleEnabled && localScratch`. Only
non-masked cached tiers (1, 2, 4, 8, 16, 32, 64, 128, 256) use this split;
cacheless and masked Arabic/Indic source remains unchanged.

Two pure, module-private TypeScript generators share the actual scan/matching
logic, with no parsing of emitted WGSL:

- `presenceSource(name: "activeRoots" | "activeRootsProbe", mode: "all-words" | "word0-probe", localScratch: boolean): string`.
  The original `activeRoots` still scans and stores **every** presence word.
  `activeRootsProbe` scans word zero and caches the same nominal masks, but does
  not touch `rootPresence`.
- `matchInputSource(name: "matchInput" | "matchInputProbe", record: boolean, masked: boolean, profileGates: boolean): string`.
  The original recording matcher stays unchanged. The probe omits exactly the
  first-input and subsequent-input recording stores.

The cached prologue calls `activeRootsProbe`; `proveSimpleRun` alone calls
`matchInputProbe`. The simple/terminal returns occur before general-state
initialization. After the workgroup-uniform simple return, while input is still
nominal, lane zero calls the **original** `activeRoots(inputOrigin, scalarCount)`
and initializes `nextLigature = 0u`. One uniform workgroup barrier follows the
lane-zero block, before `nominalSource = false` and scratch initialization. This
initializes presence words one and higher even when no GSUB edit ever refreshes
the run. `rootCount > 32` always declines the simple proof.

The fast function closure references none of `rootPresence`, `nextLigature`,
`matchPositions` or `recordMatches`; the full kernel retains every declaration,
binding and general executor. Serial refresh still depends on the original
`activeRoots` side effects, not its returned first word. Parallel refresh still
rebuilds all words using its original atomic scans. The general matcher and
recording callers are unchanged. This source-level isolation is not performance
acceptance or a claim about native resource allocation.

Within the changed positional range, paragraph admission runs the line-break check
and then the unchanged Devanagari helper, which has its own non-ASCII early return.
The check order, diagnostics and first-offender semantics are unchanged.

### Shared lookup semantics

The same cooperative, ordered-parallel and serial executors share actions:

- **C1:** positioning offsets are pen-relative. Only actually attached marks have
  advances zeroed; unattached spacing widths are retained.
- **C2:** mark filtering-set membership and attachment-class skips are applied.
- **C3:** ordered subtables/coverage are searched correctly; a nonmatching first
  subtable does not make the whole lookup inert.
- **C4:** child Single/Multiple/Ligature/positioning actions use current logical
  context sequence semantics. Multiple physically inserts/shifts; Ligature retains
  tombstones until root refresh. Cluster groups and ligature components survive.
- **C5:** Context formats and first matching rule use the same shared predicates.
- **C6:** contextual GPOS executes matched selected actions rather than silently
  treating contexts as unsupported/inert.
- **C7:** ligature components, skipped records and attachment ownership are retained.
- **C8:** cluster merging and attachment/pen arithmetic are shared and exact.

These replace the historical draft rule that Arabic lookup 35 must leave lam/alef
at 72/9 or that matched contexts must stay inert. Expected values come from cleared
independent semantics, not reproduction of text-shaper bugs.

For one SequenceLookupRecord, the generic single-record executor omits only local
sequence remapping that no subsequent record can observe. The child, flags,
descriptor, boundary, real high, delta, failure handling and 64-entry expansion
guard are unchanged. The native exact context state is `[0,3,3,3,2,0]`.
Multi-record Single/Multiple contexts remain shared. **F1 admission rejects selected
Indic compiled contexts with two or more records if an actual GSUB child contains
a Ligature rule**, with:
`createComputeTextPlan: font uses multi-record contextual ligatures not yet supported by the GPU shaper`.
Unselected features are not rejected. The general 16-word masked Ligature pattern
is unqualified (CTF-17); there is no production cap/skip/fallback.

### Arabic C9/C10 and font policy

Pinned UCD16 Joining_Type, UnicodeData/ArabicShaping and modifier-mark properties
are generated with Unicode License v3 provenance. GPU chunk summaries/carries
compute joining forms and feature masks; GPU bounded modified-ccc/MCM reordering
and cluster merges precede shared lookup stages. The 220 MCM block is retained
before 230 using an advancing insertion cursor. Unicode properties are input data,
not pre-shaped CPU output.

The first increment supports `arab` fonts with OpenType joining features
isol/fina/medi/init; absence gets an explicit unsupported-font error after nominal
unmapped precedence. This is an implementation boundary, not an invalid-font claim.
There is no presentation-form/AAT fallback, Syriac/N'Ko admission, Arabic joiner
admission or bidi mirroring. Tested Arabic conformance is the pinned Noto fixture.

### Indic preparation and stages

Pinned UCD16 and HarfBuzz11.0.1 behavioral source (annotated tag `9f3deba7…`,
verified peeled commit `43a778492219032b9d747112e412dec2201d2667`) generate
category/position/modified-ccc/flags, ten decompositions, three allowed compositions,
22 vowel constraints and the syllable DFA. Source/license hashes are in generated
data headers. This is reimplemented behavior, not a runtime HarfBuzz dependency.

GPU load → vowel-constraint dotted circles → decomposition/mark order/composition →
properties/syllables → locl/ccmp → initial reorder → nukt, akhn, rphf, rkrf, blwf,
half, cjct → final reorder → presentation/positioning stages → clear syllables /
hidden-joiner handling. Input masks, per-syllable input/backtrack gates and MAYBE
joiner matching are shared with the lookup predicates; lookahead is not restricted
to the input syllable.

Always recompose the three allowed mapped pairs. HB's encoding-dependent
`all_simple` shortcut is not copied. Before nukt only, the reference comparison
projects adjacent same-cluster nominal pairs through those three compositions;
the independent generator validates against HB's decomposed-input path with scalar
cluster renumbering. Post-nukt and all final fields are exact. Raw snapshots remain
unchanged. Pre-nukt coverage indistinguishability is guarded, with a negative test.

## Pipeline configuration and rendering

Shape bindings: read-only params/program/tables/runs and writable scratch/output.
Resident tasks reuse per-plan/profile/tier programs, shaders and bind groups;
device-relative resources are never shared across replacement devices. Pack follows
the final shape group; unchanged F1 stitch routing selects its proved schedule.

Atlas closure includes nominal seeds, normalization parts/composites, space,
virama, dotted circle, selected GSUB dependencies and outline components. Numeric
shape/draw size preflight happens **before atlas mutation**, using immutable
preparation without a second mutable input pack; the run-word size uses the group's
unsegmented scalar total. Failure publishes no partial batch.
GPU overflow/status failures clear indirect words; missing atlas slots are explicit.

`createComputeTextLayer` and CPU `createTextLayer` share the existing curve atlas
and Slug renderer. At run zero, instance coordinates are independently specified as
f32 `(pen + offset) × (fontSize / unitsPerEm)` in pack order. Layer MVP maps font
Y-up into canvas placement. The authorized witness compares actual uploaded live
instance bytes and separately rendered RGBA8 pixels exactly, not against new goldens.

## State machine / lifecycle

Profile imports coalesce before busy; failures clear pending loader state and can
retry. Font plans cache successes, not failed admission. Updates copy input before
async admission; empty indexes/source order are preserved. Status progresses ready
→ busy → ready, or terminal lost/disposed. Generation/device checks invalidate
late pipeline, mapping, status and publish continuations. Disposal releases owned
buffers/tasks/attachments, not caller-owned glyph storage or sibling blocks.

Independent engines and blocks stay independently owned. Warm variants and buffers
can be reused until growth/device/plan/profile identity requires recreation. A failed
update suppresses old drawing; it never silently returns nominal/partial success.

## Equivalence and reference policy

HarfBuzz11.0.1 CLI and ctypes must agree under pinned font/configuration and scalar
clusters. Synthetic fixtures isolate corrected primitives; real Arabic evidence
adds joining/mark-order checks. The test adapter keeps unchanged U cases live and
uses compact, identity-bound non-U expectations for reviewed corrections. It does
not import the ~47 MB diagnostic report. CPU audit digests are checked against the
independent live adapter; no Lite-derived expected records or blanket tolerances.

The legacy CPU path remains unchanged, including Devanagari's missing script
preprocessing and its mark/joining differences. The CPU layer used by the rendered
witness receives **independently positioned glyph IDs**, not legacy CPU shaping.
Representation differences and defects are recorded/reviewed; they do not create
product modes. See the test divergence register and the follow-up backlog.

## Dependencies and attribution

Production: Lite font parsers, compute/storage/scheduling primitives, generated
Unicode properties, shared curve atlas and text renderer. No text-shaper/HarfBuzz
runtime is retained by compute-only consumers. Legacy CPU consumers may still use
their existing text-shaper dependency.

Behavioral references: HarfBuzz authors, HarfBuzz11.0.1, Old MIT; Unicode16 UCD,
Unicode License v3; Noto Sans Devanagari2.007, OFL. License texts/provenance are
retained with fixtures/generated data. References are evidence, not copied engines.

## Test specification and file manifest

- `compute-text-program*.test.ts`: ABI, directories, flags and nested actions.
- `compute-text-indic-plan/unicode/machine/projection.test.ts`: admission,
  selected-child F1 guard, source tables/DFA and B1 safety.
- `devanagari.test.ts`, `devanagari-resident.test.ts`, `devanagari-stages.test.ts`:
  normal public loader, exact independent six fields, slots/f32 placement/status
  window and stage snapshots. No availability override is needed.
- `devanagari-integration.test.ts`, `lifecycle.test.ts`: limits/overflow/atomicity,
  actual null proof, mixed variants, import/busy/dispose/loss/recreation.
- `shared-core-corrections`, `arabic-joining`, `segment`, `resident`, `kernel`,
  `long-run`, `fused-submit`, `stitch-routing`, `paragraph-chunks`: shared regressions.
- Browser specs: `compute-text-shaper.spec.ts`, `compute-text-resident.spec.ts`.
  The owned M4 sidecar leaves the protected historical fixture source untouched.
- `rendered-placement.test.ts`: **only** `LITE_RENDER_WITNESS=1`, exact live uploaded
  instance bits, nonempty ink and pixel equality; no golden writes.
- `compute-text-shaper-treeshake.test.ts`: literal lazy edges, fail-closed opaque
  import/code-URL detection, actual reachable static/dynamic graphs, reference
  import boundaries and unchanged non-compute consumers.

Core files: `compute-text-{selection,plan,program,shaper,data,closure,segment}.ts`,
`compute-text-indic{,-plan,-unicode,-paragraph}.ts`, `compute-text-arabic{,-joining}.ts`,
`_gpu/compute-text-{shape,predicates,indic-wgsl,arabic-wgsl,resident}.ts`.
Reference generators/fixtures and compact test data are test-only. The ignored
`scripts/shaping-references-ctypes.py` is a required future publication deliverable.

## Performance validation recipe

Timed runs need explicit approval because they are long. Correctness, source
identity and structural executor eligibility are not speed measurements.

1. Rerun the existing **unchanged** F1 P20k1p/X50k1p/G1/G2 normal protocol:
   6 ABBA quartets × 4 cohorts = 96 runs, same device/clock/viewport/fonts/endpoints,
   candidate versus `27c6ecb3`. Record exact status, font hashes, adapter/backend and
   source SHA. Investigate any median throughput ratio below 0.97; do not tune or
   raise a ceiling to hide it. The historical F1 result was P +25.4%, X +27.3%,
   G1 +8.9%, G2 −0.17%, with two negative G2 blocks; those are historical results,
   **not measurements of this candidate**.
2. Run `node scripts/measure-compute-text-devanagari.mjs`: 100 independent Hindi
   blocks, fixed closed repertoire, one changed word per block, 30 warm frames and
   120 sample frames. Report candidate **absolute** numbers, first use (profile,
   plan/variant/atlas work) separately, synchronous admission-call segments and
   completion latency. Await statuses; invalid/missing output is a failure, not a
   timing. The clock can be coarse without cross-origin isolation; record observed
   resolution.
3. **Required CTF-16 workload:**
   `node scripts/measure-compute-text-devanagari.mjs --workload=single`.
   This is one exactly 2,000-scalar Hindi run, padded only at a word boundary.
   It measures the honest whole-run limitation. It is not 100 small blocks or
   claimed Indic segmentation.
4. CTF-14/C9 cost:
   `node scripts/measure-compute-text-devanagari.mjs --script=arab`.
   Record absolute Arabic block results with the same bounded protocol.
   Arabic whole-run/RTL behavior is not an independent split proof.
5. Latin control: rerun steps 2 and 3 with `--script=latn` (Roboto, 67 scalars per
   block; the single workload uses the same 2,000-scalar padding rule). This gives
   the same protocol, block shape and frame pacing, so complex-script costs can be
   expressed relative to Latin on the same device and run session. Interleave the
   scripts across repeats.

The workload page/runner are approval-gated files; existing user profile pages,
Pad1342, workbooks, Vite configuration and F1 scripts are unchanged. Render-task GPU
timing is **not** used: it times only frame-graph tasks, while compute text shaping
and packing are submitted outside any task, so its snapshots stay `pending` and never
contain a shape/pack interval. Instead the page wraps `queue.submit` during sampled
frames and reports `submitToGpuDoneMs`, from the first submit after admission to
`onSubmittedWorkDone` of the last submit before completion, plus `submitsPerFrame`.
That endpoint is a wall-clock GPU-queue interval, not isolated shader time.
Synchronous admission time excludes awaited continuations and is not called total
async self CPU time. If the runner starts its own Vite server, its readiness probe
uses a bounded fetch timeout; an already running lab server can be passed with
`--base`.

Two diagnostic options attribute cost; use them for relative attribution, not as the
headline timings above:

- `--profile=passes` adds `?profile=passes` to the page, which wraps command-encoder
  pass creation with unique timestamp-query slots per pass and reports
  `passProfile`: GPU milliseconds per sampled frame for each
  `encoder > pass [pipeline]` key, passes per frame, submits per encoder label per
  frame, and a quantization check (timestamps must not be multiples of 100 µs).
  Summed pass time can exceed `submitToGpuDoneMs` because passes from different
  command buffers overlap on the device.
- `--cpu-profile=/absolute/path.cpuprofile` samples the main thread at 100 µs
  through CDP over the sampled frames only, writes the raw profile, and adds a
  `cpuSelfTime` summary by module and by function.

**Never compute a Hindi B/A ratio against baseline rejection or incorrect shaping:**
`27c6ecb3` has no Devanagari implementation. Use candidate absolute results until a
correct, supported Hindi baseline exists. Only the existing comparable Latin/F1
cases receive baseline ratios. Browser/render/native checks, bundle ceilings and
final independent review are current delivery gates, not optional backlog items.
