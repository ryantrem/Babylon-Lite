import type { EngineContext } from "../engine/engine.js";

interface PendingCommand {
    readonly command: GPUCommandBuffer;
    readonly afterSubmit?: () => void;
    readonly onAbandon?: (error: Error) => void;
}

/** @internal Device-owned deferred queue. Allocated only by its first user. */
export interface DeferredSubmitState {
    readonly device: GPUDevice;
    pending: PendingCommand[];
    timer?: ReturnType<typeof setTimeout>;
    lost: boolean;
}

function flush(engine: EngineContext, state: DeferredSubmitState, frameCbs?: readonly GPUCommandBuffer[] | null): void {
    const pending = state.pending;
    state.pending = [];
    engine._flushPendingWork = undefined;
    clearTimeout(state.timer);
    state.timer = undefined;
    if (frameCbs === null || state.lost || state.device !== engine._device) {
        const error = new Error("Deferred GPU work was abandoned before submission.");
        for (const item of pending) {
            item.onAbandon?.(error);
        }
        if (frameCbs) {
            engine._device.queue.submit(frameCbs);
        }
        return;
    }
    try {
        state.device.queue.submit([...pending.map((item) => item.command), ...(frameCbs ?? [])]);
    } catch (error) {
        for (const item of pending) {
            item.onAbandon?.(error instanceof Error ? error : new Error(String(error)));
        }
        throw error;
    }
    for (const item of pending) {
        item.afterSubmit?.();
    }
}

/** @internal Queue work ahead of this task's frame, or submit it in the next macrotask. */
export function _deferCommandBuffer(engine: EngineContext, commandBuffer: GPUCommandBuffer, afterSubmit?: () => void, onAbandon?: (error: Error) => void): void {
    let state = engine._pendingWork;
    if (state && state.device !== engine._device) {
        flush(engine, state, null);
        state = undefined;
    }
    if (!state) {
        state = engine._pendingWork = { device: engine._device, pending: [], lost: false };
        const owned = state;
        void owned.device.lost.then(() => {
            owned.lost = true;
            if (engine._pendingWork === owned && owned.pending.length) {
                flush(engine, owned, null);
            }
        });
    }
    if (state.lost) {
        throw new Error("Cannot defer GPU work on a lost device.");
    }
    state.pending.push({ command: commandBuffer, afterSubmit, onAbandon });
    if (!engine._flushPendingWork) {
        const owned = state;
        engine._flushPendingWork = (engine, frameCbs) => flush(engine, owned, frameCbs);
        state.timer = setTimeout(() => {
            if (engine._pendingWork === owned && owned.pending.length) {
                try {
                    flush(engine, owned);
                } catch {
                    // Submission failure has already rejected the detached owners.
                }
            }
        }, 0);
    }
}
