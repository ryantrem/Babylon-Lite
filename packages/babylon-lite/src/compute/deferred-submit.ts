import type { EngineContext } from "../engine/engine.js";

interface PendingCommand {
    /** A finished buffer, or a builder invoked in queue order at flush time. */
    readonly command: GPUCommandBuffer | (() => GPUCommandBuffer);
    readonly afterSubmit?: () => void;
    readonly onAbandon?: (error: Error) => void;
}

/** @internal Device-owned deferred queue. Allocated only by its first user. */
export interface DeferredSubmitState {
    readonly device: GPUDevice;
    pending: PendingCommand[];
    /** Owners whose resources some pending command reads or writes. */
    owners: Set<object>;
    /** Some pending command did not name an owner; owner-scoped flushes must flush everything. */
    unowned: boolean;
    timer?: ReturnType<typeof setTimeout>;
    lost: boolean;
}

function abandon(pending: readonly PendingCommand[], error: Error): void {
    for (const item of pending) {
        item.onAbandon?.(error);
    }
}

function flush(engine: EngineContext, state: DeferredSubmitState, frameCbs?: readonly GPUCommandBuffer[] | null): void {
    const pending = state.pending;
    state.pending = [];
    state.owners = new Set();
    state.unowned = false;
    engine._flushPendingWork = undefined;
    clearTimeout(state.timer);
    state.timer = undefined;
    if (frameCbs === null || state.lost || state.device !== engine._device) {
        abandon(pending, new Error("Deferred GPU work was abandoned before submission."));
        if (frameCbs) {
            engine._device.queue.submit(frameCbs);
        }
        return;
    }
    // A frame flush runs after its encoder is finished; park it so lazy builders may record direct tasks.
    const frameEncoder = engine._currentEncoder;
    try {
        const commands: GPUCommandBuffer[] = [];
        engine._currentEncoder = undefined!;
        try {
            for (const item of pending) {
                commands.push(typeof item.command === "function" ? item.command() : item.command);
            }
        } finally {
            engine._currentEncoder = frameEncoder;
        }
        state.device.queue.submit(frameCbs ? [...commands, ...frameCbs] : commands);
    } catch (error) {
        abandon(pending, error instanceof Error ? error : new Error(String(error)));
        throw error;
    }
    for (const item of pending) {
        item.afterSubmit?.();
    }
}

/** @internal Queue work ahead of this task's frame, or submit it in the next macrotask.
 *  `owner` names the object whose resources the command uses, enabling {@link _flushDeferredWorkFor}. */
export function _deferCommandBuffer(
    engine: EngineContext,
    commandBuffer: GPUCommandBuffer | (() => GPUCommandBuffer),
    afterSubmit?: () => void,
    onAbandon?: (error: Error) => void,
    owner?: object
): void {
    let state = engine._pendingWork;
    if (state && state.device !== engine._device) {
        flush(engine, state, null);
        state = undefined;
    }
    if (!state) {
        state = engine._pendingWork = { device: engine._device, pending: [], owners: new Set(), unowned: false, lost: false };
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
    if (owner) {
        state.owners.add(owner);
    } else {
        state.unowned = true;
    }
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

/** @internal Record that queued work also uses `owner`'s resources. Requires pending deferred work. */
export function _addDeferredOwner(engine: EngineContext, owner: object): void {
    engine._pendingWork?.owners.add(owner);
}

/** @internal Whether `command` is the most recently deferred command, so work it records lazily still
 *  follows everything queued before it. */
export function _isLastDeferred(engine: EngineContext, command: () => GPUCommandBuffer): boolean {
    const pending = engine._pendingWork?.pending;
    return !!pending && pending[pending.length - 1]?.command === command;
}

/** @internal Submit queued work only if some of it may use `owner`'s resources.
 *  Call before writing or replacing those resources; unrelated owners keep accumulating. */
export function _flushDeferredWorkFor(engine: EngineContext, owner: object): void {
    const state = engine._pendingWork;
    if (engine._flushPendingWork && (!state || state.unowned || state.device !== engine._device || state.owners.has(owner))) {
        engine._flushPendingWork(engine);
    }
}
