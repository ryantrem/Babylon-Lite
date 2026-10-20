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

`compute/deferred-submit.ts` exports the internal function:

```ts
_deferCommandBuffer(
    engine: EngineContext,
    commandBuffer: GPUCommandBuffer,
    afterSubmit?: () => void,
    onAbandon?: (error: Error) => void
): void;
```

State is stored on `EngineContext._pendingWork`: the owning device, ordered
command-buffer/callback entries, and an optional fallback timer. There are no
module-level allocations. A single device-loss listener is installed per state,
not per edit. The optional `_flushPendingWork(engine, frameCbs?)` hook exists
only while work is pending. A command-buffer array means fuse with that frame;
omitting it means submit pending work alone; `null` means abandon without
submitting.

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

Flush pending work before a resident factory/update writes or reallocates GPU
resources, before direct resident execution prepares/uploads stage resources,
and before normal text disposal destroys borrowed buffers. A lost text object
instead abandons pending work. Resident shader release also flushes before
destroying resources when called independently.

`waitForGpuIdle`, `stopEngine`, `submitComputeTasks`, both immediate compute
readback functions, storage-buffer readback, and storage-buffer
clear/range-update/frame-readback operations flush before fencing, submitting or
recording dependent operations. They use only the optional engine seam; they do
not import the deferred implementation.

For N updates before one frame, each subsequent update flushes its predecessor
before uploading into reused params/runs/paragraph buffers. Only the last update
fuses with rendering: N queue submits, instead of N compute submits plus a frame
submit. Updates are never coalesced. CPU paragraph input mutation is safe only
after this barrier; asynchronous preparation cannot retain mutable GPU inputs
across another owner's writes without preserving their submission order.

## Pipeline and shader configuration

No shader, workgroup, binding layout, storage ABI, render bundle, or indirect
buffer format changes. Each update retains its own unmapped status staging
allocation until its map completes. A pending staging allocation must not map
before the command buffer that writes it has been submitted.

## Tests and file manifest

- `compute/deferred-submit.ts`: generic queue, fallback, cancellation and device
  ownership; unit tests cover callback ordering/reentrancy and timer behavior.
- `engine/engine.ts`: internal state types; final frame submit, empty frame and
  stop/idle-wait seams only. No runtime import of the deferred module.
- `compute/compute-task-readback.ts`: opt-in deferred status helper alongside
  unchanged immediate submission semantics.
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
