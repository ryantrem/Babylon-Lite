import { createActor } from "./actors.js";
import type { Actor } from "./actors.js";
import type { Frame, Menu, Runtime } from "./runtime.js";
import { createRuntime } from "./runtime.js";

function record(value: unknown): Record<string, unknown> {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("Invalid save record.");
    }
    return value as Record<string, unknown>;
}

function array(value: unknown): unknown[] {
    if (!Array.isArray(value)) {
        throw new Error("Invalid save array.");
    }
    return value;
}

function number(value: unknown, maximum = Number.MAX_SAFE_INTEGER): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < -1 || value > maximum) {
        throw new Error("Invalid numeric save value.");
    }
    return value;
}

function string(value: unknown): string {
    if (typeof value !== "string" || value.length > 65536) {
        throw new Error("Invalid save string.");
    }
    return value;
}

function bytes(value: unknown, length: number, maximum = 255): number[] {
    const values = array(value);
    if (values.length !== length || values.some((item) => typeof item !== "number" || !Number.isInteger(item) || item < 0 || item > maximum)) {
        throw new Error("Invalid byte array in save.");
    }
    return values as number[];
}

function actors(value: unknown, legacy: boolean): Actor[] {
    const values = array(value);
    if (values.length !== 256) {
        throw new Error("Invalid saved actor count.");
    }
    return values.map((entry) => {
        const source = record(entry);
        const actor = createActor();
        if (legacy) {
            const hadBaselines = source.previousY !== undefined;
            const additions = {
                previousX: source.x,
                previousY: source.y,
                stationary: false,
                newlyPositioned: false,
                cycleDelay: false,
                followDelay: 255,
            };
            for (const [key, initial] of Object.entries(additions)) {
                if (source[key] === undefined) {
                    source[key] = initial;
                }
            }
            if (!hadBaselines) {
                source.stepClock = Math.max(0, number(source.stepTime, 255) - number(source.stepClock, 255));
                source.cycleClock = Math.max(0, number(source.cycleTime, 255) - number(source.cycleClock, 255));
            }
        }
        for (const [key, initial] of Object.entries(actor)) {
            const item = source[key];
            if (typeof initial === "boolean") {
                if (typeof item !== "boolean") {
                    throw new Error("Invalid actor flag in save.");
                }
            } else {
                number(item, 65535);
            }
        }
        return Object.assign(actor, Object.fromEntries(Object.keys(actor).map((key) => [key, source[key]])));
    });
}

function frames(value: unknown): Frame[] {
    const values = array(value);
    if (values.length > 32) {
        throw new Error("Invalid saved call depth.");
    }
    return values.map((item) => {
        const entry = record(item);
        return { logic: number(entry.logic, 255), pc: number(entry.pc, 65535) };
    });
}

function menus(value: unknown): Menu[] {
    return array(value).map((item) => {
        const entry = record(item);
        return {
            title: string(entry.title),
            items: array(entry.items).map((member) => {
                const data = record(member);
                if (typeof data.enabled !== "boolean") {
                    throw new Error("Invalid saved menu flag.");
                }
                return { text: string(data.text), controller: number(data.controller, 255), enabled: data.enabled };
            }),
        };
    });
}

export function serializeSnapshot(r: Runtime): string {
    const { game, host: _host, random: _random, modal: _modal, ...state } = r;
    return JSON.stringify({ version: 2, fingerprint: game.fingerprint, state }, (_key, value: unknown) => {
        if (value instanceof Uint8Array || value instanceof Uint16Array) {
            return Array.from(value);
        }
        if (value instanceof Map) {
            return [...value.entries()];
        }
        return value;
    });
}

export function deserializeSnapshot(r: Runtime, source: string): void {
    if (source.length > 2 * 1024 * 1024) {
        throw new Error("Save is too large.");
    }
    const envelope = record(JSON.parse(source) as unknown);
    if ((envelope.version !== 1 && envelope.version !== 2) || envelope.fingerprint !== r.game.fingerprint) {
        throw new Error("This save belongs to a different game-data revision or save format.");
    }
    const state = record(envelope.state);
    const restored = createRuntime(r.game, r.host);
    restored.random = r.random;
    const legacy = envelope.version === 1;
    if (legacy) {
        const additions = { displayRow: 1, inputRow: 22, statusRow: 0, inputWidth: 0, keys: [], controllerQueue: [] };
        for (const [key, initial] of Object.entries(additions)) {
            if (state[key] === undefined) {
                state[key] = initial;
            }
        }
    }
    for (const [key, initial] of Object.entries(restored)) {
        if (typeof initial === "boolean" || typeof initial === "string" || typeof initial === "number") {
            const value = state[key];
            if (typeof value !== typeof initial) {
                throw new Error(`Invalid saved ${key}.`);
            }
            if (typeof value === "number") {
                number(value);
            }
            if (typeof value === "string") {
                string(value);
            }
            Object.assign(restored, { [key]: value });
        }
    }
    restored.variables.set(bytes(state.variables, 256));
    restored.flags.set(bytes(state.flags, 256, 1));
    restored.inventory.set(bytes(state.inventory, r.inventory.length));
    restored.picture.set(bytes(state.picture, r.picture.length, 15));
    restored.priority.set(bytes(state.priority, r.priority.length, 15));
    restored.shownPicture.set(bytes(state.shownPicture, r.shownPicture.length, 15));
    restored.shownPriority.set(bytes(state.shownPriority, r.shownPriority.length, 15));
    restored.scanStarts.set(bytes(state.scanStarts, 256, 65535));
    restored.controllers.set(bytes(state.controllers, 256, 1));
    restored.keys = bytes(state.keys, array(state.keys).length, 65535);
    restored.controllerQueue = bytes(state.controllerQueue, array(state.controllerQueue).length);
    if (restored.keys.length + restored.controllerQueue.length > 19) {
        throw new Error("Invalid saved input queue.");
    }
    restored.actors = actors(state.actors, legacy);
    restored.stack = frames(state.stack);
    restored.strings = array(state.strings).map(string);
    if (restored.strings.length !== 24) {
        throw new Error("Invalid saved string slots.");
    }
    restored.words = array(state.words).map((value) => number(value, 65535));
    restored.wordText = array(state.wordText).map(string);
    restored.keyMap = new Map(
        array(state.keyMap).map((value) => {
            const pair = array(value);
            return [number(pair[0], 65535), number(pair[1], 255)];
        })
    );
    restored.menus = menus(state.menus);
    restored.text = array(state.text).map((value) => {
        const entry = record(value);
        return {
            row: number(entry.row, 24),
            column: number(entry.column, 39),
            value: string(entry.value),
            foreground: number(entry.foreground, 15),
            background: number(entry.background, 15),
        };
    });
    if (state.block !== null) {
        const block = bytes(state.block, 4);
        restored.block = [block[0]!, block[1]!, block[2]!, block[3]!];
    }
    if (restored.soundFlag >= 0) {
        restored.flags[restored.soundFlag] = 1;
    }
    restored.soundFlag = -1;
    restored.soundRemaining = 0;
    restored.flags[12] = 1;
    r.host.stopSound();
    Object.assign(r, restored);
}

export function saveSnapshot(r: Runtime): void {
    localStorage.setItem(`lite-agi-save-${r.game.fingerprint}`, serializeSnapshot(r));
}

export function restoreSnapshot(r: Runtime): void {
    const source = localStorage.getItem(`lite-agi-save-${r.game.fingerprint}`);
    if (source === null) {
        throw new Error("No saved game exists for these game files in this browser.");
    }
    deserializeSnapshot(r, source);
}
