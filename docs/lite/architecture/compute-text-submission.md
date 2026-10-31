# Module: resident text submission

## Purpose and public contract

Resident text admission prepares pipelines, uploads inputs, records shaping and
packing/layout, and copies the four-byte status to private staging. The returned
`ComputeTextUpdate.completed` still observes that exact update's GPU status.
Admission does not wait for mapping or GPU execution.

The finished compute command buffer is deferred until the next frame submit in
the same task. The frame calls one `queue.submit([...pending, ...frameCbs])`,
with compute before rendering. Mapping starts only after submission. Status
codes, fail-closed indirect draws, asynchronous paragraph-cache invalidation,
and generation/disposal checks are unchanged. No public option or export is
added; ordinary `submitComputeTaskResult` remains immediate.

Paragraph paging preserves this contract. Bounded shaping batches may submit
before scratch/parameter reuse, without completion waits. Ordered wrap, reverse
replay, placement, global publication and the single status copy remain in the
final deferred sequence. All old/candidate indirect ranges are zero until global
success; no partial prefix is visible. Per-dispatch immutable parameter windows
are retained through submission. Range identity/order/window changes version the
existing layer bundle; counts alone do not. Terminal owner release also releases
paragraph resources allocated before executor preparation. See
[paged paragraph storage](compute-text-paragraph-layout.md#paged-paragraph-storage-stage-2).

## Internal architecture

`compute/deferred-submit.ts` exports these internal functions:

```ts
_deferCommandBuffer(
    engine: EngineContext,
    commandBuffer: GPUCommandBuffer | (() => GPUCommandBuffer),
    afterSubmit?: () => void,
    onAbandon?: (error: Error) => void,
    owner?: object
): void;
_addDeferredOwner(engine: EngineContext, owner: object): void;
_flushDeferredWorkFor(engine: EngineContext, owner: object): void;
```

State is stored on `EngineContext._pendingWork`: the owning device, ordered
command-buffer/callback entries, the set of owners whose resources pending
commands use, an `unowned` flag, and an optional fallback timer. There are no
module-level allocations. A single device-loss listener is installed per state,
not per edit. The optional `_flushPendingWork(engine, frameCbs?)` hook exists
only while work is pending. A command-buffer array means fuse with that frame;
omitting it means submit pending work alone; `null` means abandon without
submitting.

An entry is either a finished command buffer or a lazy builder. Builders run in
queue order at flush time, so a command can absorb work admitted after it was
queued (see [cross-object shape batching](#cross-object-shape-batching)). A
frame flush runs after the frame encoder is finished, so `flush` parks
`engine._currentEncoder` while builders run and restores it afterwards; builders
may then record compute tasks directly.

Before submission, flushing detaches the entries, removes the hook and cancels
the timer. Submission precedes every callback. Callbacks run in insertion order;
resident callbacks start their private maps and then resolve optional task
timing against the original compute encoder. New work admitted by a callback
belongs to a new batch. Submission exceptions abandon all detached entries.

Abandonment on device loss or change rejects deferred readback promises without
starting maps. If a frame accompanies work owned by a lost or changed device,
abandon only that pending work and submit the frame command buffers on the
engine's current device. Device loss and disposal also retain the existing
terminal rejection paths.

## Fallback and lifecycle

Exactly one `setTimeout(0)` fallback is scheduled per nonempty pending batch.
It runs after the current task's microtasks, allowing `await update; renderFrame`
to fuse. It is canceled when a frame or an ordering barrier flushes the batch.
There is no rAF dependency: awaiting `completed` without rendering completes
through the fallback one task later.

An empty frame, `stopEngine` and `waitForGpuIdle` flush pending work. Stopping
the render loop does not cancel admitted updates on a live device. Idle waiting
flushes before calling `queue.onSubmittedWorkDone()`, so its fence includes
already-admitted deferred updates even without rendering. Normal engine disposal
also flushes through `stopEngine` before destroying resources. Only device loss
or change abandons pending work without submitting it. State can be reused after
stop on the same device, and is replaced on device change; stale loss listeners
cannot cancel a replacement device's work.

## Ordering barriers

Flush pending work before a resident update writes or reallocates its GPU
resources, before direct resident execution prepares/uploads stage resources,
and before normal text disposal destroys borrowed buffers. Resident updates and
direct execution flush only through `_flushDeferredWorkFor(engine, owner)`: it
submits the queue when some pending command names that owner, or when any
pending command is unowned; otherwise other objects' work keeps accumulating.
Factories, paragraph updates (`updateComputeTextParagraphs`) and paging, disposal
and other generic sites still flush everything, so only ordinary updates batch
across objects. A lost text object
instead abandons pending work. Resident shader release also flushes before
destroying resources when called independently.

`waitForGpuIdle`, `stopEngine`, `submitComputeTasks`, both immediate compute
readback functions, storage-buffer readback, and storage-buffer
clear/range-update/frame-readback operations flush before fencing, submitting or
recording dependent operations. They use only the optional engine seam; they do
not import the deferred implementation.

Repeated updates to one object before a frame flush their predecessor before
uploading into that object's reused params/runs/paragraph buffers. Updates to
different objects do not flush each other: one shape batch per plan and
kernel, the shared pack pass, and paragraph pack/status command buffers all fuse
with the next frame in a single queue submit. CPU paragraph input mutation is safe only after this barrier;
asynchronous preparation cannot retain mutable GPU inputs across another
owner's writes without preserving their submission order.

## Cross-object shape batching

`text/_gpu/compute-text-shape-batch.ts` shapes the runs of many independent text
objects in one dispatch. Before it, each object recorded its own shape task: one
dispatch of one workgroup per run, serialized per object. One hundred Hindi
blocks took 104 submits per frame and left most of the GPU idle.

```ts
prepareComputeTextShapeBatch(engine, plan, prepared): Promise<ComputeTextShapeBatch>;
addComputeTextShapeSlice(batch, owner, prepared, shaped: StorageBuffer): void;
```

Both functions are `@internal`. Resident execution (`executeResidentComputeText`)
is the only caller and has no other shaping path.

Ownership and caching:

- Per-engine state lives in a lazily created `WeakMap<EngineContext, …>` and
  holds the device, the shape kernels, and a `WeakMap` from plan to batches.
- A kernel is keyed by the variant family (`computeTextVariantKey` without the
  tier) and its workgroup-scratch tier. As before, a retained larger tier serves
  smaller capacities. Kernel source depends only on the variant and tier; the
  record stride comes from `params.capacity` at runtime. Kernels are therefore
  shared across plans and objects.
- Each plan holds one batch per kernel. The batch owns its params, program,
  tables, runs, scratch and arena buffers, plus the shape task bound to them.
  It also caches a single-slice task bound to the last lone slice's target, and
  rebuilds it when the target or the batch buffers change.
  Program and tables are uploaded once at creation. Batch buffers keep their
  high-water size until the plan, and so its font, is garbage-collected.
- A device change disposes the old kernels and starts new state.

Admission and recording:

1. `prepareComputeTextShapeBatch` finds or creates the kernel and batch, awaits
   the kernel pipeline, and reserves buffers for the open slices plus this one.
   Reservation at admission, with power-of-two growth, keeps steady-state flushes
   from reallocating. The arena grows only once a batch holds two slices, so a
   lone large document doesn't double its shaped memory. Any storage-buffer create or dispose bumps the engine
   resource epoch and invalidates every cached bind group.
2. After the object's pack task is prepared, `addComputeTextShapeSlice` appends
   its slice (packed runs, output run, capacity, target `shaped` buffer). The
   first slice of an open batch queues a lazy builder, owned by that object;
   later slices add their owner with `_addDeferredOwner`. A slice that would
   exceed the dispatch, binding or buffer limits flushes the queue first and
   opens a new batch.
3. The object then queues its pack into the shared pack pass (see below), or a
   paragraph stage defers its own pack/status command buffer. No `await` may
   occur between slice admission and this step, or the fallback timer could
   split them into separate submits.
4. At flush, the builder closes the batch, packs every slice's run words with
   rebased origins, uploads params and runs, and dispatches one workgroup per
   run into the arena at the largest slice capacity. It then copies each run's
   row prefix to its target at that object's exact stride: one copy per slice
   when strides match, otherwise one per run. Targets destroyed since admission
   are skipped.
5. A batch holding one slice skips the arena: it dispatches straight into the
   target at the slice's own capacity and output run, with the slice's run words
   unchanged. Small single-object edits are GPU-latency-bound, and the copy
   between the shape and pack passes cost about 25 µs per edit.

The builder was queued before every participant's pack buffer, so the batch
submits before all its readers in the same `queue.submit`. Segmented updates
target the object's segment output, which the stitch pass in the pack task then
reads. Status codes, record contents and overflow reporting are unchanged: a
header count larger than the capacity is preserved, and the pack still reports
it.

Single-slice measurements (Apple M2 Max, Chrome, LLX Pad `runPadPipelineCapacity`
throughput, edits/s, before batching → arena with copy → direct): 8 words
7,263 → 6,150 → 7,260; 64 words 6,667 → 6,080 → 6,481; 256 words in 4
paragraphs 5,786 → 5,531 → 5,756; 1,000 words in 8 paragraphs 4,569 → 4,650 →
4,555; 4,000 words in 1 paragraph 1,630 → 1,828 → 1,629; 4,000 words in 32
paragraphs about 3,380 → 4,280 → 3,550. The direct path matches pre-batching at
every size. The arena's large-document gain wasn't batching work: on Metal, any
blit after the shape pass, even an unrelated 16-byte copy, reproduces both the
small-document loss and the GPU-saturated gain. With pass timestamps enabled,
both variants show the same GPU timeline, so the cause is likely Metal
scheduling between command buffers. See CTF-18 in the
[follow-ups](compute-text-follow-ups.md).

## Shared pack pass

After batching, each independent object still recorded its pack (stitch,
count, prefix and place) in its own compute pass and command buffer. Dawn's
Metal backend runs passes serially with a fixed cost of about 30 µs each, so 100
Hindi blocks spent 4.0 ms of GPU time packing. Two changes remove most of it.

`deferResidentPack` (in `compute-text-resident.ts`) coalesces packs:

- Per-engine state, in a lazily created `WeakMap`, holds the device, one
  recorded combined task (`compute-text-resident-shared-pack`), and the open
  pack queue.
- An object appends `{task, status, staging}` to the open queue only while the
  queue's builder is the newest deferred command. Otherwise it defers a new
  queue. Every command queued before the pack, including the object's shape
  slices, therefore submits before it.
- At flush, the builder moves every entry's dispatches into the combined task,
  records one compute pass, then copies each object's status word to its own
  staging buffer. After submit, each staging buffer maps and resolves its own
  update. Abandonment rejects every entry.
- Within the pass, WebGPU usage-scope ordering keeps each object's stitch, count
  and place in order. Objects write disjoint buffers.
- Recording is lazy, so the flush reads each object's dispatch sizes, params and
  bind groups. `executeResidentComputeText` flushes the object's pending work
  before rewriting any of them, and release flushes first.
- A shape batch opened for a new kernel after the open queue closes that queue.
  Passes per flush are therefore bounded by the number of distinct shaping
  kernels, not objects.

Paragraph stages (`updateComputeTextParagraphs`) keep their per-object
`submitDeferredComputeTaskResult` submission.

The fused pack mode (`packFused`) replaces the wide mode's three serial
dispatches when capacity is at least 33 and the object has at most
`FUSED_PACK_MAX_RUNS = 4` runs, which covers typical blocks. One 64-lane
workgroup finds each run's first invalid glyph, counts visible glyphs before it,
prefixes runs and places instances. Overflow (status 2) takes precedence over a
later missing glyph (status 3), exactly as the serial reference does. More runs
use the existing wide or parallel modes.

Measured on Apple M2 Max / Chrome 156 (100 Hindi blocks, one changed word per
block, medians of back-to-back A/B runs), before → after: pack 4.0 ms over 100
passes → 1.9 ms in one pass; GPU 6.1–6.4 → 4.8–5.0 ms; completion 13.3–13.9 →
12.4–12.6 ms. Without the fused mode the shared pass alone measures 3.1 ms of
pack. Latin blocks are within run-to-run noise.

## Pipeline and shader configuration

No workgroup, binding layout, storage ABI, render bundle, or indirect buffer
format changes. The pack shader gains the `packFused` entry point; shape
batching reuses the existing shape kernel. Rows
past a run's record count in `shaped` were already unspecified, and the prefix
copies keep them so. Each update retains its own unmapped status staging
allocation until its map completes. A pending staging allocation must not map
before the command buffer that writes it has been submitted.

## Tests and file manifest

- `compute/deferred-submit.ts`: generic queue, fallback, cancellation and device
  ownership; unit tests cover callback ordering/reentrancy and timer behavior.
- `engine/engine.ts`: internal state types; final frame submit, empty frame and
  stop/idle-wait seams only. No runtime import of the deferred module.
- `compute/compute-task-readback.ts`: opt-in deferred status helper alongside
  unchanged immediate submission semantics. The deferred path does not flush
  earlier work (queue order already follows it) and passes its owner through.
- `text/_gpu/compute-text-shape-batch.ts`: cross-object shape batches and shared
  shape kernels. `tests/lite/compute/shape-batch.test.ts` checks that concurrent
  mixed-script, mixed-capacity objects produce the same shaped records,
  instances and indirect bytes as serial updates, that two updates to one object
  in one task stay ordered, and that disposing an object with a pending slice is
  safe. Fused-submit, resident and paragraph tests assert the batched dispatch
  shape and command-buffer counts.
- `text/_gpu/compute-text-resident.ts`: shared pack pass and fused pack mode.
  `tests/lite/compute/wide-pack.test.ts` checks fused output and failure
  precedence against the serial reference, mode selection, and that independent
  small, fused, wide and overflowing objects admitted together share one pass
  with exact per-object instances, indirect args and status.
- Resident data/paragraph/execution lifecycle and compute/storage submit sites:
  ordering barriers.
- Native GPU tests compare complete instance/indirect/status bytes against
  immediate execution, count N submits, check fallback/empty frame, readback
  ordering, idle-wait/stop completion, failure/cache invalidation, disposal and
  device loss. Unit tests ensure device-mismatched pending work cannot drop the
  current device's frame submission.
- Build graph tests exclude the deferred module from nonresident consumers.
  Focused scene bundle ceilings remain immutable.

This scheduling policy has no Babylon.js shader/math equivalent: it changes
only when already-recorded WebGPU work is submitted, not its rendered result.
