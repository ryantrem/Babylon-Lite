import { actorCel, createActor, prepareActors, prepareMotion, priorityFor, repositionActor, stampCel, updateActors } from "./actors.js";
import type { Actor } from "./actors.js";
import { drawPicture, HEIGHT, WIDTH } from "./picture.js";
import { byte, getLogic, getResource, getView, word } from "./resources.js";
import type { GameResources, Logic } from "./resources.js";
import { decodeSound } from "./sound.js";
import type { Sound } from "./sound.js";

export interface MenuItem {
    text: string;
    controller: number;
    enabled: boolean;
}

export interface Menu {
    title: string;
    items: MenuItem[];
}

export interface Modal {
    kind: "message" | "string" | "number" | "inventory" | "menu" | "quit";
    text: string;
    destination?: number;
    maxLength?: number;
    deadline?: number;
    view?: number;
}

export interface RuntimeHost {
    sound: (sound: Sound, enabled: boolean) => void;
    stopSound: () => void;
    save: (runtime: Runtime) => void;
    restore: (runtime: Runtime) => void;
    notify: (text: string) => void;
    editCommand: (text: string) => void;
}

export interface Frame {
    logic: number;
    pc: number;
}

export interface Runtime {
    game: GameResources;
    host: RuntimeHost;
    variables: Uint8Array;
    flags: Uint8Array;
    strings: string[];
    inventory: Uint8Array;
    actors: Actor[];
    picture: Uint8Array;
    priority: Uint8Array;
    shownPicture: Uint8Array;
    shownPriority: Uint8Array;
    scanStarts: Uint16Array;
    stack: Frame[];
    words: number[];
    wordText: string[];
    controllers: Uint8Array;
    keyMap: Map<number, number>;
    keys: number[];
    controllerQueue: number[];
    keyPoll: Frame | null;
    menus: Menu[];
    menusSubmitted: boolean;
    text: { row: number; column: number; value: string; foreground: number; background: number }[];
    foreground: number;
    background: number;
    displayRow: number;
    inputRow: number;
    statusRow: number;
    inputWidth: number;
    graphics: boolean;
    inputEnabled: boolean;
    statusEnabled: boolean;
    playerControl: boolean;
    holdKey: boolean;
    horizon: number;
    block: [number, number, number, number] | null;
    modal: Modal | null;
    window: string;
    stopped: boolean;
    elapsed: number;
    cycleElapsed: number;
    soundRemaining: number;
    soundFlag: number;
    roomChanged: boolean;
    lastCommand: string;
    gameId: string;
    random: () => number;
}

export function createRuntime(game: GameResources, host: RuntimeHost): Runtime {
    const variables = new Uint8Array(256);
    variables[8] = 255;
    variables[10] = 2;
    variables[22] = 3;
    variables[24] = 41;
    variables[26] = 3;
    const flags = new Uint8Array(256);
    flags[5] = flags[9] = flags[11] = 1;
    return {
        game,
        host,
        variables,
        flags,
        strings: Array.from({ length: 24 }, () => ""),
        inventory: new Uint8Array(game.inventory.map((item) => item.room)),
        actors: Array.from({ length: 256 }, createActor),
        picture: new Uint8Array(WIDTH * HEIGHT).fill(15),
        priority: new Uint8Array(WIDTH * HEIGHT).fill(4),
        shownPicture: new Uint8Array(WIDTH * HEIGHT),
        shownPriority: new Uint8Array(WIDTH * HEIGHT).fill(4),
        scanStarts: new Uint16Array(256),
        stack: [],
        words: [],
        wordText: [],
        controllers: new Uint8Array(256),
        keyMap: new Map(),
        keys: [],
        controllerQueue: [],
        keyPoll: null,
        menus: [],
        menusSubmitted: false,
        text: [],
        foreground: 15,
        background: 0,
        displayRow: 1,
        inputRow: 22,
        statusRow: 0,
        inputWidth: 0,
        graphics: true,
        inputEnabled: true,
        statusEnabled: true,
        playerControl: true,
        holdKey: false,
        horizon: 36,
        block: null,
        modal: null,
        window: "",
        stopped: false,
        elapsed: 0,
        cycleElapsed: 0,
        soundRemaining: 0,
        soundFlag: -1,
        roomChanged: false,
        lastCommand: "",
        gameId: "",
        random: Math.random,
    };
}

export function submitCommand(r: Runtime, source: string): void {
    if (!r.inputEnabled || r.modal || r.stopped) {
        return;
    }
    r.lastCommand = source;
    parse(r, source);
}

function parse(r: Runtime, source: string): void {
    r.flags[2] = r.flags[4] = 0;
    r.variables[9] = 0;
    r.words = [];
    r.wordText = [];
    const tokens = source
        .toLowerCase()
        .replace(/['`"-]/g, "")
        .replace(/[,.?!();:[\]{}]/g, " ")
        .trim()
        .split(/\s+/)
        .filter(Boolean);
    for (let i = 0; i < tokens.length && r.words.length < 10;) {
        let id: number | undefined;
        let end = tokens.length;
        let token = "";
        for (; end > i; end--) {
            token = tokens.slice(i, end).join(" ");
            id = r.game.words.get(token);
            if (id !== undefined) {
                break;
            }
        }
        if (id === undefined) {
            r.wordText.push(tokens[i]!);
            r.variables[9] = r.words.length + 1;
            r.flags[2] = 1;
            return;
        }
        if (id !== 0) {
            r.words.push(id);
            r.wordText.push(token);
        }
        i = end;
    }
    r.flags[2] = Number(r.words.length > 0);
}

function queueKey(r: Runtime, value: number, mapped: boolean): void {
    if (r.keys.length + r.controllerQueue.length >= 19) {
        r.host.notify("AGI input queue is full; key was not accepted.");
    } else {
        (mapped ? r.controllerQueue : r.keys).push(value);
    }
}

export function sendKey(r: Runtime, key: number): void {
    const controller = r.keyMap.get(key);
    queueKey(r, controller ?? key, controller !== undefined);
}

export function sendDirection(r: Runtime, direction: number, pressed = true): void {
    if (!Number.isInteger(direction) || direction < 0 || direction > 8) {
        throw new Error("Invalid AGI direction input.");
    }
    if (!r.playerControl || r.modal || r.stopped) {
        return;
    }
    if (pressed) {
        r.variables[6] = !r.holdKey && r.variables[6] === direction ? 0 : direction;
    } else if (r.holdKey && r.variables[6] === direction) {
        r.variables[6] = 0;
    }
}

export function dismissModal(r: Runtime, result = ""): void {
    const modal = r.modal;
    if (!modal) {
        return;
    }
    switch (modal.kind) {
        case "string":
            r.strings[modal.destination!] = result.slice(0, modal.maxLength);
            break;
        case "number":
            r.variables[modal.destination!] = Number.parseInt(result || "0", 10) & 255;
            break;
        case "inventory":
            if (r.flags[13]) {
                r.variables[25] = result === "" ? 255 : Number(result);
            }
            break;
        case "menu":
            if (result !== "") {
                const id = Number(result);
                const item = r.menus.flatMap((menu) => menu.items).find((entry) => entry.controller === id && entry.enabled);
                if (item) {
                    queueKey(r, id, true);
                }
            }
            break;
        case "quit":
            r.stopped = result === "quit";
            break;
        case "message":
            break;
    }
    r.modal = null;
    r.window = "";
}

function message(r: Runtime, logicId: number, id: number, depth = 0): string {
    if (depth > 8) {
        throw new Error("Recursive AGI message substitution.");
    }
    const value = getLogic(r.game, logicId).messages[id];
    if (value === undefined) {
        throw new Error(`Missing message ${id} in LOGIC ${logicId}.`);
    }
    return value
        .replace(/%([vgmsow])(\d+)(?:\|(\d+))?/g, (_match: string, type: string, index: string, width: string | undefined) => {
            const n = Number(index);
            switch (type) {
                case "v":
                    return String(r.variables[n] ?? 0).padStart(Number(width ?? 0), "0");
                case "g":
                    return message(r, 0, n, depth + 1);
                case "m":
                    return message(r, logicId, n, depth + 1);
                case "s":
                    return r.strings[n] ?? "";
                case "o":
                    return r.game.inventory[n]?.name ?? "";
                case "w":
                    return r.wordText[n - 1] ?? "";
                default:
                    throw new Error(`Unknown AGI message format ${type}.`);
            }
        })
        .replace(/\\n/g, "\n");
}

function showMessage(r: Runtime, text: string, view?: number): void {
    r.window = text;
    if (!r.flags[15]) {
        r.modal = { kind: "message", text, view, deadline: r.variables[21] ? r.elapsed + r.variables[21]! * 500 : undefined };
    }
}

function stopRuntimeSound(r: Runtime): void {
    r.host.stopSound();
    if (r.soundFlag >= 0) {
        r.flags[r.soundFlag] = 1;
    }
    r.soundFlag = -1;
    r.soundRemaining = 0;
}

function unanimateActors(r: Runtime): void {
    for (const actor of r.actors) {
        actor.animated = actor.drawn = actor.updating = actor.cycling = false;
        actor.fixedPriority = actor.fixedLoop = actor.ignoreHorizon = actor.ignoreObjects = actor.ignoreBlocks = false;
        actor.terrain = 0;
    }
}

function changeRoom(r: Runtime, room: number): void {
    stopRuntimeSound(r);
    const ego = r.actors[0]!;
    r.variables[16] = ego.view;
    const edge = r.variables[2];
    if (edge === 1) {
        ego.y = HEIGHT - 1;
    } else if (edge === 2) {
        ego.x = 0;
    } else if (edge === 3) {
        ego.y = 37;
    } else if (edge === 4) {
        ego.x = WIDTH - actorCel(r, ego).width;
    }
    unanimateActors(r);
    for (const actor of r.actors) {
        actor.stepSize = actor.stepTime = actor.cycleTime = 1;
        actor.stepClock = actor.cycleClock = 1;
        actor.newlyPositioned = actor.cycleDelay = false;
    }
    r.variables[1] = r.variables[0]!;
    r.variables[0] = room;
    r.variables[2] = r.variables[4] = r.variables[5] = r.variables[9] = 0;
    r.flags[2] = r.flags[4] = 0;
    r.flags[5] = 1;
    r.horizon = 36;
    r.block = null;
    r.playerControl = true;
    r.stack.length = 0;
    r.scanStarts.fill(0);
    r.text.length = 0;
    r.window = "";
    r.roomChanged = true;
    getLogic(r.game, room);
}

function next(frame: Frame, logic: Logic): number {
    return byte(logic.code, frame.pc++);
}

function displacement(frame: Frame, logic: Logic): number {
    const value = word(logic.code, frame.pc);
    frame.pc += 2;
    return value & 0x8000 ? value - 65536 : value;
}

function predicate(r: Runtime, frame: Frame, logic: Logic, opcode: number, evaluate: boolean): boolean {
    const counts = [0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 5, 1, 0, -1, 2, 5, 5, 5];
    const count = counts[opcode];
    if (count === undefined) {
        throw new Error(`Unsupported AGI predicate 0x${opcode.toString(16)}.`);
    }
    if (opcode === 14) {
        const length = next(frame, logic);
        const pattern: number[] = [];
        for (let n = 0; n < length; n++) {
            pattern.push(word(logic.code, frame.pc));
            frame.pc += 2;
        }
        if (!evaluate || !r.flags[2] || r.flags[4] || (r.words.length === 0 && r.variables[9] === 0)) {
            return false;
        }
        for (let n = 0; n < pattern.length; n++) {
            if (pattern[n] === 9999) {
                r.flags[4] = 1;
                return true;
            }
            if (n >= r.words.length || (pattern[n] !== 1 && pattern[n] !== r.words[n])) {
                return false;
            }
        }
        if (pattern.length !== r.words.length) {
            return false;
        }
        r.flags[4] = 1;
        return true;
    }
    const args = Array.from({ length: count! }, () => next(frame, logic));
    if (!evaluate) {
        return false;
    }
    const a = args[0]!;
    const b = args[1]!;
    const v = r.variables;
    switch (opcode) {
        case 0:
            return false;
        case 1:
            return v[a] === b;
        case 2:
            return v[a] === v[b];
        case 3:
            return v[a]! < b;
        case 4:
            return v[a]! < v[b]!;
        case 5:
            return v[a]! > b;
        case 6:
            return v[a]! > v[b]!;
        case 7:
            return r.flags[a] !== 0;
        case 8:
            return r.flags[v[a]!] !== 0;
        case 9:
            return r.inventory[a] === 255;
        case 10:
            return r.inventory[a] === v[b];
        case 12:
            return r.controllers[a] !== 0;
        case 13:
            if (v[19]) {
                return true;
            }
            if (r.keys.length) {
                const key = r.keys.shift()!;
                v[19] = key & 255;
                return key !== 0;
            }
            r.keyPoll = { logic: frame.logic, pc: frame.pc - 1 };
            return false;
        case 15: {
            const normalize = (s: string): string => s.toLowerCase().replace(/[\s.,;:'!-]/g, "");
            return normalize(r.strings[a] ?? "") === normalize(r.strings[b] ?? "");
        }
        case 11:
        case 16:
        case 17:
        case 18: {
            const actor = r.actors[a]!;
            let x = actor.x;
            let right = x;
            if (opcode !== 11) {
                const width = actorCel(r, actor).width;
                if (opcode === 16) {
                    right += width - 1;
                } else {
                    x += opcode === 17 ? Math.floor(width / 2) : width - 1;
                    right = x;
                }
            }
            return x >= args[1]! && right <= args[3]! && actor.y >= args[2]! && actor.y <= args[4]!;
        }
        default:
            throw new Error(`Unsupported AGI predicate ${opcode}.`);
    }
}

function conditions(r: Runtime, frame: Frame, logic: Logic): boolean {
    let result = true;
    let group = false;
    let groupResult = false;
    let invert = false;
    for (;;) {
        const opcode = next(frame, logic);
        if (opcode === 255) {
            if (group || invert) {
                throw new Error("Malformed AGI condition group.");
            }
            return result;
        }
        if (opcode === 252) {
            if (group) {
                result &&= groupResult;
                group = false;
            } else {
                group = true;
                groupResult = false;
            }
            continue;
        }
        if (opcode === 253) {
            invert = !invert;
            continue;
        }
        const evaluate = result && (!group || !groupResult);
        const value = predicate(r, frame, logic, opcode, evaluate);
        if (evaluate) {
            if (group) {
                groupResult ||= invert ? !value : value;
            } else {
                result &&= invert ? !value : value;
            }
        }
        invert = false;
    }
}

function call(r: Runtime, id: number): void {
    if (r.stack.length >= 32) {
        throw new Error("AGI call depth exceeds 32.");
    }
    getLogic(r.game, id);
    r.stack.push({ logic: id, pc: r.scanStarts[id]! });
}

function isKeyPollJump(r: Runtime, frame: Frame, start: number): boolean {
    const poll = r.keyPoll;
    return poll !== null && poll.logic === frame.logic && frame.pc <= poll.pc && poll.pc < start;
}

function execute(r: Runtime, frame: Frame, logic: Logic, opcode: number): void {
    const n = (): number => next(frame, logic);
    const v = r.variables;
    const f = r.flags;
    const variable = (): number => v[n()]!;
    const actor = (): Actor => r.actors[n()]!;
    const msg = (): string => message(r, frame.logic, n());
    const signed = (value: number): number => (value > 127 ? value - 256 : value);
    switch (opcode) {
        case 0x01: {
            const a = n();
            v[a] = Math.min(255, v[a]! + 1);
            break;
        }
        case 0x02: {
            const a = n();
            v[a] = Math.max(0, v[a]! - 1);
            break;
        }
        case 0x03: {
            const a = n();
            v[a] = n();
            break;
        }
        case 0x04: {
            const a = n();
            v[a] = variable();
            break;
        }
        case 0x05: {
            const a = n();
            v[a] = v[a]! + n();
            break;
        }
        case 0x06: {
            const a = n();
            v[a] = v[a]! + variable();
            break;
        }
        case 0x07: {
            const a = n();
            v[a] = v[a]! - n();
            break;
        }
        case 0x08: {
            const a = n();
            v[a] = v[a]! - variable();
            break;
        }
        case 0x09: {
            const a = variable();
            v[a] = variable();
            break;
        }
        case 0x0a: {
            const a = n();
            v[a] = v[variable()]!;
            break;
        }
        case 0x0b: {
            const a = variable();
            v[a] = n();
            break;
        }
        case 0x0c:
            f[n()] = 1;
            break;
        case 0x0d:
            f[n()] = 0;
            break;
        case 0x0e: {
            const a = n();
            f[a] = Number(!f[a]);
            break;
        }
        case 0x0f:
            f[variable()] = 1;
            break;
        case 0x10:
            f[variable()] = 0;
            break;
        case 0x11: {
            const a = variable();
            f[a] = Number(!f[a]);
            break;
        }
        case 0x12:
            changeRoom(r, n());
            break;
        case 0x13:
            changeRoom(r, variable());
            break;
        case 0x14:
            getLogic(r.game, n());
            break;
        case 0x15:
            getLogic(r.game, variable());
            break;
        case 0x16:
            call(r, n());
            break;
        case 0x17:
            call(r, variable());
            break;
        case 0x18:
            getResource(r.game, "pic", variable());
            break;
        case 0x19:
            drawPicture(getResource(r.game, "pic", variable()), r.picture, r.priority);
            break;
        case 0x1a:
            r.shownPicture.set(r.picture);
            r.shownPriority.set(r.priority);
            r.text.length = 0;
            f[15] = 0;
            break;
        case 0x1b:
            getResource(r.game, "pic", variable());
            break;
        case 0x1c:
            drawPicture(getResource(r.game, "pic", variable()), r.picture, r.priority, false);
            break;
        case 0x1d:
            throw new Error("Priority-screen debugger is not implemented.");
        case 0x1e:
            getView(r.game, n());
            break;
        case 0x1f:
            getView(r.game, variable());
            break;
        case 0x20:
            r.game.views.delete(n());
            break;
        case 0x21: {
            const a = actor();
            if (!a.animated) {
                a.animated = a.updating = a.cycling = true;
                a.direction = a.motion = a.cycleMode = 0;
            }
            break;
        }
        case 0x22:
            unanimateActors(r);
            break;
        case 0x23: {
            const a = actor();
            if (a.drawn) {
                break;
            }
            repositionActor(r, a, a.x, a.y);
            a.drawn = true;
            a.updating = true;
            a.previousX = a.x;
            a.previousY = a.y;
            a.priority = a.fixedPriority ? a.priority : priorityFor(a.y);
            break;
        }
        case 0x24:
            actor().drawn = false;
            break;
        case 0x25: {
            const a = actor();
            a.x = n();
            a.y = n();
            a.previousX = a.x;
            a.previousY = a.y;
            break;
        }
        case 0x26: {
            const a = actor();
            a.x = variable();
            a.y = variable();
            a.previousX = a.x;
            a.previousY = a.y;
            break;
        }
        case 0x27: {
            const a = actor();
            v[n()] = a.x;
            v[n()] = a.y;
            break;
        }
        case 0x28: {
            const a = actor();
            const x = a.x + signed(variable());
            repositionActor(r, a, x, a.y + signed(variable()));
            a.newlyPositioned = true;
            break;
        }
        case 0x29:
        case 0x2a: {
            const a = actor();
            a.view = opcode === 0x29 ? n() : variable();
            a.cycleMode = 0;
            a.cycleDelay = false;
            const view = getView(r.game, a.view);
            if (a.loop >= view.loops.length) {
                a.loop = 0;
            }
            if (a.cel >= view.loops[a.loop]!.length) {
                a.cel = 0;
            }
            break;
        }
        case 0x2b:
        case 0x2c: {
            const a = actor();
            a.loop = opcode === 0x2b ? n() : variable();
            const loop = getView(r.game, a.view).loops[a.loop];
            if (loop) {
                if (a.cel >= loop.length) {
                    a.cel = 0;
                }
            } else if (a.drawn) {
                actorCel(r, a);
            }
            break;
        }
        case 0x2d:
            actor().fixedLoop = true;
            break;
        case 0x2e:
            actor().fixedLoop = false;
            break;
        case 0x2f:
        case 0x30: {
            const a = actor();
            a.cel = opcode === 0x2f ? n() : variable();
            a.cycleDelay = false;
            actorCel(r, a);
            break;
        }
        case 0x31: {
            const a = actor();
            v[n()] = getView(r.game, a.view).loops[a.loop]!.length - 1;
            break;
        }
        case 0x32: {
            const a = actor();
            v[n()] = a.cel;
            break;
        }
        case 0x33: {
            const a = actor();
            v[n()] = a.loop;
            break;
        }
        case 0x34: {
            const a = actor();
            v[n()] = a.view;
            break;
        }
        case 0x35: {
            const a = actor();
            v[n()] = getView(r.game, a.view).loops.length;
            break;
        }
        case 0x36:
        case 0x37: {
            const a = actor();
            a.priority = opcode === 0x36 ? n() : variable();
            a.fixedPriority = true;
            break;
        }
        case 0x38:
            actor().fixedPriority = false;
            break;
        case 0x39: {
            const a = actor();
            v[n()] = a.priority;
            break;
        }
        case 0x3a:
            actor().updating = false;
            break;
        case 0x3b:
            actor().updating = true;
            break;
        case 0x3c:
            n();
            break;
        case 0x3d:
            actor().ignoreHorizon = true;
            break;
        case 0x3e:
            actor().ignoreHorizon = false;
            break;
        case 0x3f:
            r.horizon = n();
            break;
        case 0x40:
            actor().terrain = 1;
            break;
        case 0x41:
            actor().terrain = 2;
            break;
        case 0x42:
            actor().terrain = 0;
            break;
        case 0x43:
            actor().ignoreObjects = true;
            break;
        case 0x44:
            actor().ignoreObjects = false;
            break;
        case 0x45: {
            const a = actor();
            const b = actor();
            const destination = n();
            v[destination] = a.drawn && b.drawn ? Math.min(254, Math.abs(a.x + actorCel(r, a).width / 2 - b.x - actorCel(r, b).width / 2) + Math.abs(a.y - b.y)) : 255;
            break;
        }
        case 0x46:
            actor().cycling = false;
            break;
        case 0x47:
            actor().cycling = true;
            break;
        case 0x48: {
            const a = actor();
            a.cycleMode = 0;
            a.cycling = true;
            break;
        }
        case 0x49:
        case 0x4b: {
            const a = actor();
            a.cycleFlag = n();
            f[a.cycleFlag] = 0;
            a.cycleMode = opcode === 0x49 ? 1 : 3;
            a.cycling = true;
            a.cycleDelay = true;
            break;
        }
        case 0x4a: {
            const a = actor();
            a.cycleMode = 2;
            a.cycling = true;
            break;
        }
        case 0x4c: {
            const a = actor();
            a.cycleTime = Math.max(1, variable());
            a.cycleClock = a.cycleTime;
            break;
        }
        case 0x4d:
        case 0x4e: {
            const a = actor();
            a.motion = 0;
            if (opcode === 0x4d) {
                a.direction = 0;
            }
            if (a === r.actors[0]) {
                v[6] = 0;
                r.playerControl = opcode === 0x4e;
            }
            break;
        }
        case 0x4f: {
            const a = actor();
            a.stepSize = Math.max(1, variable());
            break;
        }
        case 0x50: {
            const a = actor();
            a.stepTime = Math.max(1, variable());
            a.stepClock = a.stepTime;
            break;
        }
        case 0x51:
        case 0x52: {
            const a = actor();
            a.targetX = opcode === 0x51 ? n() : variable();
            a.targetY = opcode === 0x51 ? n() : variable();
            const step = opcode === 0x51 ? n() : variable();
            a.motionFlag = n();
            f[a.motionFlag] = 0;
            a.oldStepSize = a.stepSize;
            a.stepSize = step || a.stepSize;
            a.motion = 1;
            if (a === r.actors[0]) {
                r.playerControl = false;
            }
            prepareMotion(r, a);
            break;
        }
        case 0x53: {
            const a = actor();
            a.followDistance = Math.max(a.stepSize, n());
            a.motionFlag = n();
            f[a.motionFlag] = 0;
            a.oldStepSize = a.stepSize;
            a.motion = 2;
            a.followDelay = 255;
            break;
        }
        case 0x54: {
            const a = actor();
            a.motion = 3;
            a.wanderClock = 0;
            if (a === r.actors[0]) {
                r.playerControl = false;
            }
            break;
        }
        case 0x55:
            actor().motion = 0;
            break;
        case 0x56: {
            const a = actor();
            a.direction = variable();
            if (a.direction > 8) {
                throw new Error("Invalid AGI direction.");
            }
            break;
        }
        case 0x57: {
            const a = actor();
            v[n()] = a.direction;
            break;
        }
        case 0x58:
            actor().ignoreBlocks = true;
            break;
        case 0x59:
            actor().ignoreBlocks = false;
            break;
        case 0x5a:
            r.block = [n(), n(), n(), n()];
            break;
        case 0x5b:
            r.block = null;
            break;
        case 0x5c:
            r.inventory[n()] = 255;
            break;
        case 0x5d:
            r.inventory[variable()] = 255;
            break;
        case 0x5e:
            r.inventory[n()] = 0;
            break;
        case 0x5f: {
            const item = n();
            r.inventory[item] = variable();
            break;
        }
        case 0x60: {
            const item = variable();
            r.inventory[item] = variable();
            break;
        }
        case 0x61: {
            const item = variable();
            v[n()] = r.inventory[item]!;
            break;
        }
        case 0x62:
            decodeSound(getResource(r.game, "sound", n()));
            break;
        case 0x63: {
            const sound = decodeSound(getResource(r.game, "sound", n()));
            stopRuntimeSound(r);
            r.soundFlag = n();
            f[r.soundFlag] = 0;
            r.soundRemaining = f[9] ? sound.duration * 1000 : 0;
            r.host.sound(sound, Boolean(f[9]));
            if (!r.soundRemaining) {
                f[r.soundFlag] = 1;
                r.soundFlag = -1;
            }
            break;
        }
        case 0x64:
            stopRuntimeSound(r);
            break;
        case 0x65:
            showMessage(r, msg());
            break;
        case 0x66:
            showMessage(r, message(r, frame.logic, variable()));
            break;
        case 0x67:
        case 0x68: {
            const row = opcode === 0x67 ? n() : variable();
            const column = opcode === 0x67 ? n() : variable();
            const value = message(r, frame.logic, opcode === 0x67 ? n() : variable());
            r.text = r.text.filter((entry) => entry.row !== row || entry.column !== column);
            r.text.push({ row, column, value, foreground: r.foreground, background: r.background });
            break;
        }
        case 0x69: {
            const top = n();
            const bottom = n();
            n();
            r.text = r.text.filter((entry) => entry.row < top || entry.row > bottom);
            break;
        }
        case 0x6a:
            r.graphics = false;
            r.text.length = 0;
            break;
        case 0x6b:
            r.graphics = true;
            r.text.length = 0;
            break;
        case 0x6c:
            r.strings[0] = msg().slice(0, 1);
            break;
        case 0x6d:
            r.foreground = n();
            r.background = n();
            break;
        case 0x6e:
            r.host.notify(`Screen shake requested (${n()} pulses); presentation effect is not reproduced.`);
            break;
        case 0x6f:
            r.displayRow = n();
            r.inputRow = n();
            r.statusRow = n();
            break;
        case 0x70:
            r.statusEnabled = true;
            break;
        case 0x71:
            r.statusEnabled = false;
            break;
        case 0x72: {
            const destination = n();
            r.strings[destination] = msg().slice(0, 39);
            break;
        }
        case 0x73: {
            const destination = n();
            const text = msg();
            n();
            n();
            const maxLength = Math.min(39, n());
            r.modal = { kind: "string", text, destination, maxLength };
            break;
        }
        case 0x74: {
            const index = n();
            r.strings[n()] = (r.wordText[index - 1] ?? "").slice(0, 39);
            break;
        }
        case 0x75:
            parse(r, r.strings[n()] ?? "");
            break;
        case 0x76: {
            const text = msg();
            r.modal = { kind: "number", text, destination: n(), maxLength: 4 };
            break;
        }
        case 0x77:
            r.inputEnabled = false;
            break;
        case 0x78:
            r.inputEnabled = true;
            break;
        case 0x79: {
            const ascii = n();
            const scan = n();
            r.keyMap.set(ascii || scan << 8, n());
            break;
        }
        case 0x7a:
        case 0x7b: {
            const args = Array.from({ length: 7 }, () => (opcode === 0x7a ? n() : variable()));
            const cel = getView(r.game, args[0]!).loops[args[1]!]![args[2]!]!;
            stampCel(r.picture, r.priority, cel, args[3]!, args[4]!, args[5] || priorityFor(args[4]!));
            const control = args[6]!;
            if (control < 4) {
                const y = args[4]!;
                const x = args[3]!;
                for (let px = Math.max(0, x); px < Math.min(WIDTH, x + cel.width); px++) {
                    r.priority[y * WIDTH + px] = control;
                }
            }
            break;
        }
        case 0x7c:
            r.modal = { kind: "inventory", text: "Inventory" };
            break;
        case 0x7d:
            r.host.save(r);
            break;
        case 0x7e:
            r.host.restore(r);
            break;
        case 0x7f:
            break;
        case 0x80: {
            const fresh = createRuntime(r.game, r.host);
            fresh.flags[6] = 1;
            Object.assign(r, fresh);
            r.roomChanged = true;
            break;
        }
        case 0x81:
        case 0xa2: {
            const view = opcode === 0x81 ? n() : variable();
            showMessage(r, getView(r.game, view).description, view);
            break;
        }
        case 0x82: {
            const low = n();
            const high = n();
            v[n()] = low + Math.floor(r.random() * (high - low + 1));
            break;
        }
        case 0x83:
            r.playerControl = false;
            break;
        case 0x84:
            r.playerControl = true;
            r.actors[0]!.motion = 0;
            break;
        case 0x85: {
            const a = r.actors[variable()]!;
            showMessage(r, `Object: ${a.view}/${a.loop}/${a.cel} at ${a.x},${a.y}`);
            break;
        }
        case 0x86:
            if (n()) {
                r.stopped = true;
            } else {
                r.modal = { kind: "quit", text: "Quit the game?" };
            }
            break;
        case 0x87:
            showMessage(r, "Resources are held in browser memory.");
            break;
        case 0x88:
            stopRuntimeSound(r);
            showMessage(r, "Game paused.");
            break;
        case 0x89:
            if (r.inputEnabled) {
                r.host.editCommand(r.lastCommand);
            }
            break;
        case 0x8a:
            r.host.editCommand("");
            break;
        case 0x8b:
            break;
        case 0x8c:
            r.host.notify("Monitor switching is unavailable in the WebGPU demo.");
            break;
        case 0x8d:
            showMessage(r, "Standalone Lite AGI v2 compatibility runtime (not the Sierra executable).");
            break;
        case 0x8e:
            n();
            break;
        case 0x8f:
            r.gameId = msg().slice(0, 7);
            break;
        case 0x90:
            r.host.notify(msg());
            break;
        case 0x91:
            r.scanStarts[frame.logic] = frame.pc;
            break;
        case 0x92:
            r.scanStarts[frame.logic] = 0;
            break;
        case 0x93:
        case 0x94: {
            const a = actor();
            const x = opcode === 0x93 ? n() : variable();
            const y = opcode === 0x93 ? n() : variable();
            repositionActor(r, a, x, y);
            a.newlyPositioned = true;
            break;
        }
        case 0x95:
        case 0x96:
            throw new Error("AGI debugger tracing is not implemented.");
        case 0x97:
        case 0x98: {
            const text = message(r, frame.logic, opcode === 0x97 ? n() : variable());
            n();
            n();
            n();
            showMessage(r, text);
            break;
        }
        case 0x99:
            r.game.views.delete(variable());
            break;
        case 0x9a: {
            const top = n();
            const left = n();
            const bottom = n();
            const right = n();
            n();
            r.text = r.text.filter((entry) => entry.row < top || entry.row > bottom || entry.column < left || entry.column > right);
            break;
        }
        case 0x9b:
            n();
            n();
            break;
        case 0x9c: {
            const title = msg();
            if (!r.menusSubmitted) {
                r.menus.push({ title, items: [] });
            }
            break;
        }
        case 0x9d: {
            const text = msg();
            const controller = n();
            const menu = r.menus.at(-1);
            if (!menu) {
                throw new Error("Menu member has no menu heading.");
            }
            if (!r.menusSubmitted) {
                menu.items.push({ text, controller, enabled: true });
            }
            break;
        }
        case 0x9e:
            r.menusSubmitted = true;
            break;
        case 0x9f:
        case 0xa0: {
            const controller = n();
            for (const menu of r.menus) {
                for (const item of menu.items) {
                    if (item.controller === controller) {
                        item.enabled = opcode === 0x9f;
                    }
                }
            }
            break;
        }
        case 0xa1:
            if (f[14]) {
                r.modal = { kind: "menu", text: "Game menu" };
            }
            break;
        case 0xa3:
            r.inputWidth = 36;
            break;
        case 0xa4:
            r.inputWidth = 0;
            break;
        case 0xa5: {
            const a = n();
            v[a] = v[a]! * n();
            break;
        }
        case 0xa6: {
            const a = n();
            v[a] = v[a]! * variable();
            break;
        }
        case 0xa7:
        case 0xa8: {
            const a = n();
            const divisor = opcode === 0xa7 ? n() : variable();
            if (!divisor) {
                throw new Error("AGI division by zero.");
            }
            v[a] = Math.floor(v[a]! / divisor);
            break;
        }
        case 0xa9:
            r.window = "";
            r.modal = null;
            r.inputWidth = 0;
            break;
        case 0xad:
            r.holdKey = true;
            break;
        default:
            throw new Error(`Unsupported AGI action 0x${opcode.toString(16)} in LOGIC ${frame.logic} at ${frame.pc - 1}.`);
    }
}

export function tickRuntime(r: Runtime, elapsedMs: number): void {
    if (r.stopped) {
        return;
    }
    r.elapsed += elapsedMs;
    const seconds = Math.floor(r.elapsed / 1000);
    r.variables[11] = seconds % 60;
    r.variables[12] = Math.floor(seconds / 60) % 60;
    r.variables[13] = Math.floor(seconds / 3600) % 24;
    r.variables[14] = Math.floor(seconds / 86400) % 256;
    if (r.soundFlag >= 0) {
        r.soundRemaining -= elapsedMs;
        if (r.soundRemaining <= 0) {
            r.flags[r.soundFlag] = 1;
            r.soundFlag = -1;
        }
    }
    if (r.modal?.deadline && r.elapsed >= r.modal.deadline) {
        dismissModal(r);
    }
    if (r.modal) {
        return;
    }
    r.keyPoll = null;
    if (!r.stack.length) {
        r.cycleElapsed += elapsedMs;
        const cycle = Math.max(1, r.variables[10]!) * 50;
        if (r.cycleElapsed < cycle) {
            return;
        }
        r.cycleElapsed %= cycle;
        r.roomChanged = false;
        for (const key of r.keys) {
            r.variables[19] = key & 255;
        }
        r.keys.length = 0;
        r.controllers.fill(0);
        for (const controller of r.controllerQueue) {
            r.controllers[controller] = 1;
        }
        r.controllerQueue.length = 0;
        prepareActors(r);
        call(r, 0);
    }
    let budget = 50000;
    while (r.stack.length && !r.modal && !r.stopped) {
        if (--budget <= 0) {
            throw new Error("AGI scan exceeded 50,000 instructions; execution stopped to prevent a browser hang.");
        }
        const frame = r.stack.at(-1)!;
        const logic = getLogic(r.game, frame.logic);
        const opcode = next(frame, logic);
        if (opcode === 0) {
            r.stack.pop();
        } else if (opcode === 254) {
            const start = frame.pc - 1;
            const delta = displacement(frame, logic);
            frame.pc += delta;
            if (isKeyPollJump(r, frame, start)) {
                return;
            }
        } else if (opcode === 255) {
            const result = conditions(r, frame, logic);
            const delta = displacement(frame, logic);
            if (!result) {
                frame.pc += delta;
            }
        } else {
            execute(r, frame, logic, opcode);
        }
    }
    if (!r.stack.length && !r.modal && !r.roomChanged) {
        r.variables[2] = r.variables[4] = r.variables[5] = 0;
        r.flags[5] = r.flags[6] = r.flags[11] = r.flags[12] = 0;
        if (r.graphics) {
            updateActors(r);
        }
        r.controllers.fill(0);
        r.variables[19] = 0;
        r.keys.length = 0;
        r.flags[2] = r.flags[4] = 0;
    }
}
