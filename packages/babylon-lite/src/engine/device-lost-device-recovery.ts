import type { DeviceContext } from "./device.js";
import { _watchDeviceLoss } from "./device.js";
import type { EngineContext } from "./engine.js";
import type { DeviceLostRecoveryRegistration } from "./device-lost-recovery.js";
import { assertEveryActiveContextKindIsRecoverable, requestRecoveryDevice, runDeviceLostRecovery } from "./device-lost-recovery-run.js";
import { startEngine, stopEngine } from "./engine.js";
import { disposeGpuResourceRetirements } from "./gpu-resource-retirement.js";

function notify(registrations: readonly DeviceLostRecoveryRegistration[], callback: (registration: DeviceLostRecoveryRegistration) => void): void {
    for (const registration of registrations) {
        try {
            callback(registration);
        } catch (error) {
            console.error("Device-lost recovery callback failed.", error);
        }
    }
}

/** @internal Coalesce all engine loss subscribers into one device-level recovery. */
export function recoverDeviceContext(context: DeviceContext, lostDevice: GPUDevice, info: GPUDeviceLostInfo, lostEngine: EngineContext): Promise<void> {
    if (context._device !== lostDevice || context._disposed) {
        return Promise.resolve();
    }
    return (context._recovery ??= Promise.resolve()
        .then(async () => {
            context._lost = true;
            // Engine-private contexts have no shared registry and belong to exactly one engine.
            const engines = [...(context._engines ?? [lostEngine])];
            const participants = engines.flatMap((engine) => {
                const wasRunning = engine._renderFn !== null;
                stopEngine(engine);
                engine._deviceUnavailable = true;
                const state = engine._deviceLostRecovery;
                if (!state?._registrations.length || engine._device !== lostDevice) {
                    return [];
                }
                state._recovering = true;
                state._forceNextLoss = false;
                const disposed = new Promise<void>((resolve) => {
                    state._onDisposed = resolve;
                });
                return [{ engine, state, registrations: [...state._registrations], wasRunning, disposed }];
            });
            const ownershipSettlements: Array<() => void> = [];
            try {
                for (const { engine, registrations } of participants) {
                    notify(registrations, (registration) => registration._onLost?.(info));
                    if (!engine._disposed) {
                        assertEveryActiveContextKindIsRecoverable(engine, new Map(registrations.map((registration) => [registration._kind, registration])));
                    }
                }
                // Drain all outgoing ownership before any shared wrapper is rebuilt.
                for (const engine of engines) {
                    disposeGpuResourceRetirements(engine);
                }
                if (!participants.some(({ engine }) => !engine._disposed)) {
                    return;
                }
                const replacement = await requestRecoveryDevice(participants.map(({ engine }) => engine));
                context._device = replacement;
                context._lost = false;
                Object.assign(context, { features: replacement.features, limits: replacement.limits });
                _watchDeviceLoss(context);
                for (const { engine, state, registrations } of participants) {
                    if (!engine._disposed) {
                        await runDeviceLostRecovery(engine, state, registrations, replacement, false, (settle) => ownershipSettlements.push(settle));
                    }
                }
                for (const settle of ownershipSettlements.splice(0)) {
                    settle();
                }
                const starts: Promise<void>[] = [];
                for (const { engine, wasRunning, disposed } of participants) {
                    if (!engine._disposed) {
                        engine._deviceUnavailable = false;
                        if (wasRunning) {
                            starts.push(Promise.race([startEngine(engine), disposed]));
                        }
                    }
                }
                await Promise.all(starts);
                for (const { engine, registrations } of participants) {
                    if (!engine._disposed) {
                        notify(registrations, (registration) => registration._onRecovered?.());
                    }
                }
            } catch (error) {
                if (!participants.some(({ registrations }) => registrations.some((registration) => registration._onRecoveryFailed))) {
                    console.error("Device-lost recovery failed.", error);
                }
                for (const { engine, registrations } of participants) {
                    stopEngine(engine);
                    engine._deviceUnavailable = true;
                    if (!engine._disposed) {
                        notify(registrations, (registration) => registration._onRecoveryFailed?.(error));
                    }
                }
            } finally {
                for (const settle of ownershipSettlements.splice(0)) {
                    settle();
                }
                for (const { state } of participants) {
                    state._recovering = false;
                    state._onDisposed = undefined;
                }
            }
        })
        .finally(() => {
            context._recovery = undefined;
            if (!context._engines && lostEngine._disposed) {
                // The private owner was disposed mid-recovery; release a replacement it never received.
                if (lostEngine._device !== context._device) {
                    context._device.destroy();
                }
            } else {
                for (const engine of context._engines ?? [lostEngine]) {
                    if (!engine._deviceUnavailable) {
                        engine._armDeviceRecovery?.();
                    }
                }
            }
        }));
}
