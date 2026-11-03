# Module: Device Context

> Package path: `packages/babylon-lite/src/engine/`

## Purpose

Separate canvas-independent WebGPU acquisition from engine initialization. A small
bootstrap bundle can call `createDevice` while downloading application code, then
pass the resulting context to one or more engines. No native GPU handles are public.

## Public API Surface

```typescript
interface DeviceOptions {
    powerPreference?: GPUPowerPreference;
    requiredFeatures?: readonly GPUFeatureName[];
    requiredLimits?: Record<string, GPUSize64 | undefined>;
}
interface DeviceContext {
    readonly features: GPUSupportedFeatures;
    readonly limits: GPUSupportedLimits;
}
function createDevice(options?: DeviceOptions): Promise<DeviceContext>;
function disposeDevice(device: DeviceContext): void;
interface EngineDeviceOptions extends Omit<EngineOptions, "requiredLimits" | "device"> {
    device: DeviceContext;
    requiredLimits?: never;
    requiredFeatures?: never;
}
function createEngine(canvas: RenderCanvas, options: EngineDeviceOptions): Promise<EngineContext>;
function createEngine(canvas: RenderCanvas, options?: EngineOptions): Promise<EngineContext>;
```

`EngineContext.device` exposes its device context. Legacy `EngineOptions` forbids a
device property and retains its existing device limits and surface/precision options.
`createEngineWithFeatures` retains its existing signature; explicit features are now
forwarded by device acquisition rather than a process-global feature installer.

## Internal Architecture

The device module has no runtime dependency on engines, canvases, rendering, or recovery.
It contains adapter-option installation (retaining the existing XR enabler), the
opportunistic feature list, feature merging, acquisition, and disposal.

Internal fields hold the native device (including its enabled features) and snapshots
of adapter options and requested limits. Contexts from `createDevice` additionally
carry an engine array, `_attach`/`_detach` seams, and disposal/loss flags; an optional
recovery promise is installed by opt-in recovery. Engine-private contexts acquired by
`createEngine` are plain state without these fields, so convenience-only bundles never
include sharing, attachment validation, or loss-watching code. All internal
fields use `@internal` and disappear from
emitted public declarations. An erased unique-symbol brand prevents native devices
and structural lookalikes from satisfying the public context type. Engine `_device` remains an internal alias used by
existing GPU subsystems.

Features are Lite's existing seven opportunistic features plus deduplicated explicitly
required features. Lite does not pre-validate explicit features: WebGPU's
`requestDevice` rejects unsupported ones with a `TypeError`, keeping the bootstrap small.
Default power preference is `high-performance`. Adapter options are captured at
creation, so later XR installer changes do not change a context's recovery policy.
Limits and caller arrays are copied rather than retained as mutable configuration.

## Pipeline Configuration

Unchanged. Surface creation configures its canvas with the context's native device.

## Shader Logic

Unchanged.

## State Machine / Lifecycle

An explicitly created device is caller-owned. Engines attach after building their
surface, through the `createDevice` attach seam, which checks disposal, loss, and
recovery state and rejects engine-level feature/limit options. A rejected attachment
unconfigures the canvas and leaves no registry entry.
Native loss is tracked even before an engine attaches, so a device lost while the
application bundle downloads cannot be used to initialize a new engine.

`disposeEngine` releases engine-owned resources and detaches without destroying an
explicit device. An engine-private context from the convenience API has exactly one
implicit engine and is destroyed by `disposeEngine`; it must not be passed to other
engines. `disposeDevice` is idempotent but rejects private contexts, attached engines,
and active recovery.

Device loss is coordinated only when recovery is enabled. The optional coordinator
stops every attached engine (or the single owner of a private context), including
engines without recovery. It snapshots all
participating registrations, validates context kinds, drains retirements, and requests
one replacement using the enabled features and the maximum required limits across
participating engines. It rebuilds engines serially against that device before
restarting any render loop. Engines without recovery remain stopped on the lost
device and must be disposed/recreated. Lite adds no render-loop guards for this state:
WebGPU rendering on a lost device is a silent no-op. A failure leaves every participant stopped,
reports failure, and does not re-arm an endless retry. Disposal while recovery is
pending prevents rebuilding/restarting the disposed engine and releases a private
replacement that its engine never received.

Shared native-device identity does not grant arbitrary cross-engine sharing of
mutable Lite wrappers; engine-specific ownership and recovery contracts still apply.
URL texture caches are keyed by engine and replaced when the engine's native device
changes, preserving same-engine
promise reuse and automatic invalidation on device replacement without sharing mutable
wrappers between independently owned engines. When textures are shared explicitly,
recovery defers ownership settlement until every engine's handlers have finished.
Independent engines share a queue, so GPU-idle waits observe all previously submitted
work on that queue. Multiple surfaces on one engine remain preferable for coordinated
rendering.

## Babylon.js Equivalence Map

`createEngine` remains the convenient full initialization path. `createDevice` is a
Lite-specific plain-state ownership boundary.

## Dependencies

Device acquisition: browser WebGPU and erased engine type imports only.
Recovery coordination: existing opt-in recovery, storage recovery, and engine lifecycle.

## Test Specification

- Device-only creation never touches a canvas or imports engine runtime code.
- Opportunistic/required feature selection, unsupported features, absent adapter,
  adapter/device rejection, power preference, limits, and immutable snapshots.
- Two engines attach without additional adapter/device requests; disposing one
  preserves the other. Explicit device disposal rejects while attached and succeeds
  after detachment. Convenience devices are private: `disposeEngine` destroys them and
  `disposeDevice` rejects them.
- Invalid option combinations reject at runtime and in emitted type probes; rejected
  attachment unconfigures the canvas.
- Async disposal races do not leak devices/attachments.
- Shared recovery requests exactly one replacement and restarts only after all
  rebuilds complete; unrecoverable engines remain stopped. Failures are reported.
- Standalone minified bundles importing the package root compare `createDevice()`
  with `createEngine(canvas)` using identical build settings and raw/gzip byte counts
  (about 1.2 KB / 0.6 KB versus 4.8 KB / 2.3 KB).
- `createEngine`-only bundles grow by at most about 140 B raw / 60 B gzip over the
  pre-device-context baseline.

## File Manifest

- `device.ts`: context, acquisition, adapter options.
- `device-dispose.ts`: explicit device teardown.
- `engine.ts`, `engine-dispose.ts`: attachment and ownership.
- `device-lost-device-recovery.ts`: optional shared recovery coordinator.
- `device-lost-recovery.ts`, `device-lost-recovery-run.ts`: recovery integration.
- `compute/compute-engine-features.ts`, `compute/compute-uniform-f16.ts`,
  `xr/xr-adapter.ts`: feature/adapter selection integration.
- `texture/texture-2d.ts`: engine-local mutable URL wrapper caches.
- `index.ts`: root-only exports.
- Focused unit/build tests and the engine architecture document.
