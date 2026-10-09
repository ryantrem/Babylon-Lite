import { describe, expect, it } from "vitest";
import { canPlace, createActor, prepareActors, renderRuntime, repositionActor, updateActors } from "../../../lab/lite/src/demos/kings-quest/actors.js";
import { deserializeSnapshot, serializeSnapshot } from "../../../lab/lite/src/demos/kings-quest/persistence.js";
import { drawPicture, HEIGHT, WIDTH } from "../../../lab/lite/src/demos/kings-quest/picture.js";
import { decodeInventory, decodeLogic, decodeView, decodeWords, getResource, loadGame } from "../../../lab/lite/src/demos/kings-quest/resources.js";
import type { GameResources } from "../../../lab/lite/src/demos/kings-quest/resources.js";
import { createRuntime, dismissModal, sendKey, submitCommand, tickRuntime } from "../../../lab/lite/src/demos/kings-quest/runtime.js";
import { decodeSound, noiseNotes } from "../../../lab/lite/src/demos/kings-quest/sound.js";

function logicPayload(code: number[], messages: string[] = []): Uint8Array {
    const table = new Uint8Array(3 + messages.length * 2);
    table[0] = messages.length;
    const text = messages.map((message) => `${message}\0`).join("");
    let offset = table.length - 1;
    messages.forEach((message, index) => {
        table[3 + index * 2] = offset & 255;
        table[4 + index * 2] = offset >> 8;
        offset += message.length + 1;
    });
    table[1] = offset & 255;
    table[2] = offset >> 8;
    return new Uint8Array([code.length & 255, code.length >> 8, ...code, ...table, ...[...text].map((c, i) => c.charCodeAt(0) ^ "Avis Durgan".charCodeAt(i % 11))]);
}

function dictionary(entries: [string, number][]): Uint8Array {
    const bytes = new Array<number>(52).fill(0);
    let previous = "";
    for (const [text, id] of entries.sort(([a], [b]) => a.localeCompare(b))) {
        const letter = text.charCodeAt(0) - 97;
        if (bytes[letter * 2] === 0 && bytes[letter * 2 + 1] === 0) {
            bytes[letter * 2] = bytes.length >> 8;
            bytes[letter * 2 + 1] = bytes.length & 255;
        }
        let prefix = 0;
        while (prefix < Math.min(previous.length, text.length) && previous[prefix] === text[prefix]) {
            prefix++;
        }
        bytes.push(prefix);
        for (let p = prefix; p < text.length; p++) {
            bytes.push((text.charCodeAt(p) ^ 127) | (p === text.length - 1 ? 128 : 0));
        }
        bytes.push(id >> 8, id & 255);
        previous = text;
    }
    bytes.push(0);
    return new Uint8Array(bytes);
}

function game(code: number[], messages: string[] = []): GameResources {
    return {
        files: new Map(),
        words: new Map([
            ["look", 5],
            ["tree", 6],
            ["at", 0],
            ["the", 0],
        ]),
        inventory: [{ name: "key", room: 255 }],
        logics: new Map([[0, decodeLogic(logicPayload(code, messages))]]),
        views: new Map([[0, { loops: [[{ width: 2, height: 2, transparent: 0, pixels: new Uint8Array([12, 12, 12, 12]) }]], description: "" }]]),
        fingerprint: "synthetic",
    };
}

function runtime(code: number[], messages: string[] = []) {
    return createRuntime(game(code, messages), { sound: () => {}, stopSound: () => {}, save: () => {}, restore: () => {}, notify: () => {}, editCommand: () => {} });
}

describe("AGI room lifecycle", () => {
    it("replaces an old completing animation when rebinding a view without cancelling target motion", () => {
        const r = runtime([0x21, 0, 0x29, 0, 0, 0x25, 0, 100, 100, 0x23, 0, 0x49, 0, 42, 0x29, 0, 1, 0x51, 0, 110, 70, 2, 43, 0]);
        r.game.views.set(1, r.game.views.get(0)!);
        r.priority.fill(15);
        tickRuntime(r, 100);
        r.game.logics.set(0, decodeLogic(logicPayload([0])));
        for (let i = 0; i < 20; i++) {
            tickRuntime(r, 100);
        }
        expect(r.flags[42]).toBe(0);
        expect(r.flags[43]).toBe(1);
        expect(r.actors[0]).toMatchObject({ view: 1, cycleMode: 0, cycling: true, motion: 0 });
    });

    it("clears stale water/trigger classifications when priority 15 bypasses the control scan", () => {
        const r = runtime([0]);
        const a = r.actors[0]!;
        a.x = 81;
        a.y = 113;
        a.fixedPriority = true;
        a.priority = 15;
        r.flags[0] = r.flags[3] = 1;
        r.priority.fill(3);
        expect(canPlace(r, a, 81, 113)).toBe(true);
        expect([r.flags[0], r.flags[3]]).toEqual([0, 0]);
    });

    it("clears object-control flags without losing bindings or coordinates on unanimation", () => {
        const r = runtime([0x22, 0]);
        const a = r.actors[0]!;
        a.x = 75;
        a.y = 26;
        a.animated = a.drawn = a.updating = a.cycling = true;
        a.fixedPriority = a.fixedLoop = a.ignoreHorizon = a.ignoreObjects = a.ignoreBlocks = true;
        a.terrain = 1;
        tickRuntime(r, 100);
        expect(a).toMatchObject({
            x: 75,
            y: 26,
            view: 0,
            terrain: 0,
            animated: false,
            drawn: false,
            updating: false,
            cycling: false,
            fixedPriority: false,
            fixedLoop: false,
            ignoreHorizon: false,
            ignoreObjects: false,
            ignoreBlocks: false,
        });
    });

    it("removes swimming restrictions on room entry and keeps destination logic deferred", () => {
        const r = runtime([0x12, 1, 0]);
        r.game.logics.set(1, decodeLogic(logicPayload([0])));
        r.actors[0]!.terrain = 1;
        r.actors[0]!.stepSize = 3;
        tickRuntime(r, 100);
        expect(r.variables[0]).toBe(1);
        expect(r.flags[5]).toBe(1);
        expect(r.actors[0]).toMatchObject({ terrain: 0, stepSize: 1, updating: false });
    });
});

describe("standalone AGI resource decoding", () => {
    it("decodes prefix-compressed dictionary words, synonyms, phrases and the trailing terminator", () => {
        const words = decodeWords(
            dictionary([
                ["at", 0],
                ["look", 5],
                ["look at", 5],
                ["tree", 6],
            ])
        );
        expect([...words]).toEqual([
            ["at", 0],
            ["look", 5],
            ["look at", 5],
            ["tree", 6],
        ]);
        expect(() => decodeWords(new Uint8Array(4))).toThrow("alphabet index");
    });

    it("decodes encrypted inventory without mutating source bytes", () => {
        const plain = new Uint8Array([6, 0, 17, 6, 0, 255, 10, 0, 2, ...[..."key\0rock\0"].map((c) => c.charCodeAt(0))]);
        const encrypted = plain.map((b, i) => b ^ "Avis Durgan".charCodeAt(i % 11));
        const copy = encrypted.slice();
        expect(decodeInventory(encrypted)).toEqual([
            { name: "key", room: 255 },
            { name: "rock", room: 2 },
        ]);
        expect(encrypted).toEqual(copy);
        expect(decodeInventory(plain)).toEqual(decodeInventory(encrypted));
    });

    it("decodes logic messages and validates their offsets", () => {
        const decoded = decodeLogic(logicPayload([3, 30, 8, 0], ["Synthetic greeting", "Value %v30"]));
        expect([...decoded.code]).toEqual([3, 30, 8, 0]);
        expect(decoded.messages).toEqual(["", "Synthetic greeting", "Value %v30"]);
        expect(() => decodeLogic(new Uint8Array([255, 255]))).toThrow("Truncated");
    });

    it("loads a complete synthetic v2 resource set and rejects wrong editions/headers", () => {
        const payload = logicPayload([0]);
        const volume = new Uint8Array([18, 52, 0, payload.length, 0, ...payload]);
        const files = new Map([
            ["LOGDIR", new Uint8Array([0, 0, 0])],
            ["PICDIR", new Uint8Array([255, 255, 255])],
            ["VIEWDIR", new Uint8Array([255, 255, 255])],
            ["SNDDIR", new Uint8Array([255, 255, 255])],
            ["WORDS.TOK", dictionary([["look", 5]])],
            ["OBJECT", new Uint8Array([3, 0, 17, 3, 0, 255, 107, 101, 121, 0])],
            ["VOL.0", volume],
        ]);
        const result = loadGame(files);
        expect(getResource(result, "logic", 0)).toEqual(payload);
        expect(() => getResource(result, "pic", 0)).toThrow("Missing pic");
        expect(() => getResource(result, "logic", 1)).toThrow("Missing logic");
        expect(() => loadGame(new Map())).toThrow("DOS AGI v2");
        files.set("VOL.0", new Uint8Array([0, 0, 0, 0, 0]));
        expect(() => loadGame(files)).toThrow("header");
    });

    it("decodes shared, mirrored VIEW loops and rejects overlong RLE runs", () => {
        const bytes = new Uint8Array([1, 1, 2, 0, 0, 9, 0, 9, 0, 1, 3, 0, 3, 1, 0x80, 0x11, 0x22, 0]);
        const view = decodeView(bytes);
        expect([...view.loops[0]![0]!.pixels]).toEqual([1, 2, 2]);
        expect([...view.loops[1]![0]!.pixels]).toEqual([2, 2, 1]);
        bytes[15] = 0x14;
        expect(() => decodeView(bytes)).toThrow("RLE");
    });

    it("decodes all four original sound channels with 60-Hz timing", () => {
        const bytes = new Uint8Array([8, 0, 15, 0, 17, 0, 19, 0, 6, 0, 6, 0x8a, 0x92, 255, 255, 255, 255, 255, 255, 255, 255]);
        const sound = decodeSound(bytes);
        expect(sound.channels).toHaveLength(4);
        expect(sound.channels[0]![0]).toEqual({ start: 0, duration: 0.1, divisor: 106, attenuation: 2, noise: 2 });
        expect(sound.duration).toBe(0.1);
        expect(() => decodeSound(bytes.subarray(0, 12))).toThrow();
    });

    it("clocks noise mode three from the third tone voice, including mid-note changes", () => {
        const noise = { start: 0.25, duration: 1, divisor: 999, attenuation: 2, noise: 7 };
        const tone = { ...noise, start: 0, duration: 0.5, divisor: 100 };
        const sound = { channels: [[], [], [tone, { ...tone, start: 0.5, divisor: 200 }], [noise]], duration: 1.25 };
        expect(noiseNotes(sound, noise)).toEqual([
            { ...noise, duration: 0.25, divisor: 100 },
            { ...noise, start: 0.5, duration: 0.75, divisor: 200 },
        ]);
        expect(noiseNotes(sound, { ...noise, noise: 4 })).toEqual([{ ...noise, noise: 4 }]);
    });
});

describe("standalone AGI picture and actor planes", () => {
    it("draws exact diagonal cells and fills only within visual boundaries", () => {
        const picture = new Uint8Array(WIDTH * HEIGHT);
        const priority = new Uint8Array(picture.length);
        drawPicture(new Uint8Array([0xf0, 0, 0xf6, 0, 0, 3, 1, 0xff]), picture, priority);
        expect([picture[0], picture[1], picture[WIDTH + 2], picture[WIDTH + 3]]).toEqual([0, 0, 0, 0]);
        expect(picture[2]).toBe(15);
        drawPicture(new Uint8Array([0xf0, 0, 0xf6, 10, 10, 20, 10, 20, 20, 10, 20, 10, 10, 0xf0, 2, 0xf2, 8, 0xf8, 15, 15, 0xff]), picture, priority);
        expect(picture[15 * WIDTH + 15]).toBe(2);
        expect(priority[15 * WIDTH + 15]).toBe(8);
        expect(picture[9 * WIDTH + 15]).toBe(15);
        expect(() => drawPicture(new Uint8Array([0xfc]), picture, priority)).toThrow("Unsupported");
    });

    it("moves actors, blocks control-zero baselines and emits screen-border events", () => {
        const r = runtime([0]);
        const ego = r.actors[0]!;
        Object.assign(ego, createActor(), { animated: true, drawn: true, x: 157, y: 80, cycling: false });
        r.variables[6] = 3;
        updateActors(r);
        expect(ego.x).toBe(158);
        updateActors(r);
        expect(r.variables[2]).toBe(2);
        r.variables[6] = 5;
        r.priority.fill(0, 81 * WIDTH, 82 * WIDTH);
        updateActors(r);
        expect(ego.y).toBe(80);
    });

    it("uses the last baseline control cell and exempts fixed-priority fifteen", () => {
        const r = runtime([0]);
        const ego = r.actors[0]!;
        Object.assign(ego, { x: 10, y: 80 });
        r.priority[80 * WIDTH + 10] = 2;
        r.priority[80 * WIDTH + 11] = 3;
        expect(canPlace(r, ego, 10, 80)).toBe(true);
        expect([r.flags[0], r.flags[3]]).toEqual([1, 0]);
        r.priority[80 * WIDTH + 11] = 4;
        expect(canPlace(r, ego, 10, 80)).toBe(true);
        expect([r.flags[0], r.flags[3]]).toEqual([0, 0]);
        r.priority.fill(0);
        Object.assign(ego, { priority: 15, fixedPriority: true });
        expect(canPlace(r, ego, 10, 80)).toBe(true);
        expect(canPlace(r, ego, 10, 0)).toBe(false);
    });

    it("repositions in AGI spiral order and stops completing cycles", () => {
        const r = runtime([0]);
        const ego = r.actors[0]!;
        r.priority[80 * WIDTH + 10] = 0;
        repositionActor(r, ego, 10, 80);
        expect([ego.x, ego.y]).toEqual([9, 81]);
        Object.assign(ego, { animated: true, drawn: true, cycleMode: 1, cycleFlag: 40 });
        r.variables[6] = 3;
        updateActors(r);
        expect([ego.cycling, ego.direction, ego.cycleMode, r.flags[40]]).toEqual([false, 0, 0, 1]);
    });

    it("prepares autonomous direction and completion before running the next logic", () => {
        const r = runtime([0]);
        const actor = r.actors[1]!;
        Object.assign(actor, { animated: true, drawn: true, x: 10, y: 80, motion: 1, targetX: 12, targetY: 80, stepSize: 2, motionFlag: 40 });
        prepareActors(r);
        expect(actor.direction).toBe(3);
        updateActors(r);
        expect(actor.x).toBe(12);
        expect(r.flags[40]).toBe(0);
        prepareActors(r);
        expect([actor.motion, actor.direction, r.flags[40]]).toEqual([0, 0, 1]);
    });

    it("repairs a stationary footprint after a wider view or cel is selected", () => {
        const r = runtime([0]);
        const ego = r.actors[0]!;
        Object.assign(ego, { animated: true, drawn: true, cycling: false, x: 10, y: 80 });
        r.game.views.set(0, { loops: [[{ width: 4, height: 2, transparent: 0, pixels: new Uint8Array(8) }]], description: "" });
        r.priority[80 * WIDTH + 13] = 0;
        updateActors(r);
        expect([ego.x, ego.y, ego.direction]).toEqual([9, 80, 0]);
        expect(canPlace(r, ego, ego.x, ego.y)).toBe(true);
    });

    it("uses movement countdowns and delays the first completing-cycle callback", () => {
        const r = runtime([0]);
        const ego = r.actors[0]!;
        Object.assign(ego, { animated: true, drawn: true, cycling: false, x: 10, y: 80, stepTime: 3 });
        r.variables[6] = 3;
        updateActors(r);
        expect([ego.x, ego.stepClock]).toEqual([11, 3]);
        updateActors(r);
        updateActors(r);
        expect([ego.x, ego.stepClock]).toEqual([11, 1]);
        updateActors(r);
        expect(ego.x).toBe(12);
        const cel = r.game.views.get(0)!.loops[0]![0]!;
        r.game.views.set(0, { loops: [[cel, cel]], description: "" });
        Object.assign(ego, { cycling: true, cycleMode: 1, cycleFlag: 40, cycleDelay: true });
        updateActors(r);
        expect([ego.cel, r.flags[40]]).toEqual([0, 0]);
        updateActors(r);
        expect([ego.cel, ego.cycling, r.flags[40]]).toEqual([1, false, 1]);
    });

    it("checks earlier-partition actors and saved baseline crossings", () => {
        const r = runtime([0]);
        const ego = r.actors[0]!;
        Object.assign(ego, { x: 10, y: 80, previousY: 80 });
        Object.assign(r.actors[1]!, { animated: true, drawn: true, updating: false, x: 10, y: 82, previousY: 82 });
        expect(canPlace(r, ego, 10, 83)).toBe(false);
        expect(canPlace(r, ego, 13, 83)).toBe(true);
        ego.previousY = 82;
        expect(canPlace(r, ego, 10, 83)).toBe(true);
    });

    it("enforces strict rectangle crossings before the logic phase", () => {
        const r = runtime([0]);
        Object.assign(r.actors[0]!, { animated: true, drawn: true, x: 20, y: 80 });
        r.block = [19, 79, 22, 82];
        r.variables[6] = 3;
        prepareActors(r);
        expect(r.variables[6]).toBe(3);
        r.actors[0]!.x = 21;
        prepareActors(r);
        expect(r.variables[6]).toBe(0);
        r.actors[0]!.ignoreBlocks = true;
        r.variables[6] = 3;
        prepareActors(r);
        expect(r.variables[6]).toBe(3);
    });

    it("composites palette pixels through priority and doubles AGI pixels horizontally", () => {
        const r = runtime([0]);
        Object.assign(r.actors[0]!, { drawn: true, x: 5, y: 70, priority: 6 });
        const pixels = new Uint8ClampedArray(WIDTH * 2 * HEIGHT * 4);
        renderRuntime(r, pixels);
        const offset = (69 * WIDTH + 5) * 8;
        expect([...pixels.subarray(offset, offset + 8)]).toEqual([255, 85, 85, 255, 255, 85, 85, 255]);
        r.shownPriority.fill(15);
        renderRuntime(r, pixels);
        expect([...pixels.subarray(offset, offset + 4)]).toEqual([0, 0, 0, 255]);
    });
});

describe("standalone AGI logic, parser and persistence", () => {
    it("wraps arithmetic, saturates increments, takes branches and shares called-logic variables", () => {
        const r = runtime([3, 30, 255, 1, 30, 5, 30, 2, 0xff, 1, 30, 1, 0xff, 3, 0, 3, 31, 9, 0x16, 1, 0]);
        r.game.logics.set(1, decodeLogic(logicPayload([3, 32, 7, 0])));
        tickRuntime(r, 100);
        expect([...r.variables.subarray(30, 33)]).toEqual([1, 9, 7]);
    });

    it("resumes after modal print rather than restarting or repeating it", () => {
        const r = runtime([3, 30, 1, 0x65, 1, 1, 30, 0], ["Synthetic message"]);
        tickRuntime(r, 100);
        expect(r.modal?.text).toBe("Synthetic message");
        expect(r.variables[30]).toBe(1);
        tickRuntime(r, 100);
        expect(r.variables[30]).toBe(1);
        dismissModal(r);
        tickRuntime(r, 0);
        expect(r.variables[30]).toBe(2);
        expect(r.stack).toHaveLength(0);
    });

    it("uses original parser IDs, ignored words, unknown-word indices and said consumption", () => {
        const r = runtime([0xff, 14, 2, 5, 0, 6, 0, 0xff, 3, 0, 3, 40, 1, 0]);
        submitCommand(r, "LOOK at the TREE");
        expect(r.words).toEqual([5, 6]);
        tickRuntime(r, 100);
        expect(r.variables[40]).toBe(1);
        submitCommand(r, "look mysterious thing");
        expect(r.variables[9]).toBe(2);
        expect(r.wordText).toEqual(["look", "mysterious"]);
    });

    it("does not consume said in short-circuited conditions and maps function-key controllers", () => {
        const r = runtime([0xff, 1, 30, 1, 14, 1, 5, 0, 0xff, 3, 0, 3, 31, 1, 0]);
        submitCommand(r, "look");
        tickRuntime(r, 100);
        expect(r.variables[31]).toBe(0);
        r.keyMap.set(0x3f00, 9);
        sendKey(r, 0x3f00);
        expect(r.controllerQueue).toEqual([9]);
        tickRuntime(r, 100);
        expect(r.controllerQueue).toHaveLength(0);
    });

    it("recognizes ordinary title keys and enables the hold-key profile", () => {
        for (const key of [32, 97]) {
            const r = runtime([0xff, 13, 0xff, 3, 0, 3, 40, 1, 0xad, 0]);
            sendKey(r, key);
            tickRuntime(r, 100);
            expect(r.variables[40]).toBe(1);
            expect(r.holdKey).toBe(true);
        }
    });

    it("yields backward raw-key polls and resumes only on a new unmapped key", () => {
        const r = runtime([3, 19, 0, 0xff, 0xfd, 13, 0xff, 3, 0, 0xfe, 0xf7, 0xff, 3, 30, 42, 0]);
        sendKey(r, 97);
        tickRuntime(r, 100);
        expect(r.stack).toHaveLength(1);
        expect(r.variables[30]).toBe(0);
        tickRuntime(r, 100);
        expect(r.stack).toHaveLength(1);
        r.keyMap.set(0x3b00, 9);
        sendKey(r, 0x3b00);
        tickRuntime(r, 0);
        expect(r.stack).toHaveLength(1);
        expect(r.keys).toHaveLength(0);
        sendKey(r, 32);
        tickRuntime(r, 0);
        expect(r.variables[30]).toBe(42);
        expect(r.stack).toHaveLength(0);
    });

    it("stages erased-object loop setup and resets invalid selections on a new view", () => {
        const r = runtime([0x2b, 0, 3, 0x29, 0, 1, 0]);
        const cel = r.game.views.get(0)!.loops[0]![0]!;
        r.game.views.set(1, { loops: [[cel], [cel]], description: "" });
        tickRuntime(r, 100);
        expect([r.actors[0]!.view, r.actors[0]!.loop, r.actors[0]!.cel]).toEqual([1, 0, 0]);
        const active = runtime([0x2b, 0, 1, 0]);
        active.actors[0]!.drawn = true;
        expect(() => tickRuntime(active, 100)).toThrow("loop");
    });

    it("echoes and cancels the live editor without erasing accepted command history", () => {
        const r = runtime([0x89, 0x8a, 0]);
        const edits: string[] = [];
        r.host.editCommand = (text) => edits.push(text);
        r.lastCommand = "look";
        tickRuntime(r, 100);
        expect(edits).toEqual(["look", ""]);
        expect(r.lastCommand).toBe("look");
    });

    it("retains mapped input arriving after modal dismissal for the next scan", () => {
        const r = runtime([0xff, 12, 9, 0xff, 3, 0, 3, 30, 42, 0x65, 1, 0], ["Message"]);
        r.keyMap.set(0x3d00, 9);
        tickRuntime(r, 100);
        dismissModal(r);
        sendKey(r, 0x3d00);
        tickRuntime(r, 0);
        expect(r.variables[30]).toBe(0);
        expect(r.controllerQueue).toEqual([9]);
        tickRuntime(r, 100);
        expect(r.variables[30]).toBe(42);
        expect(r.controllerQueue).toHaveLength(0);
    });

    it("does not change the object draw partition for force.update", () => {
        const r = runtime([0x3c, 1, 0]);
        r.actors[1]!.updating = false;
        tickRuntime(r, 100);
        expect(r.actors[1]!.updating).toBe(false);
    });

    it("keeps nonblocking script windows live until close.window", () => {
        const r = runtime([0x65, 1, 3, 30, 42, 0], ["Nonblocking message"]);
        r.flags[15] = 1;
        tickRuntime(r, 100);
        expect([r.window, r.variables[30], r.modal]).toEqual(["Nonblocking message", 42, null]);
        r.game.logics.set(0, decodeLogic(logicPayload([0xa9, 0])));
        tickRuntime(r, 100);
        expect(r.window).toBe("");
    });

    it("completes the interrupted sound on stop and pause", () => {
        for (const opcode of [0x64, 0x88]) {
            const r = runtime([opcode, 0]);
            let stops = 0;
            r.host.stopSound = () => stops++;
            r.soundFlag = 40;
            r.soundRemaining = 10000;
            tickRuntime(r, 100);
            expect([r.flags[40], r.soundFlag, r.soundRemaining, stops]).toEqual([1, -1, 0, 1]);
        }
    });

    it("reads all four variable-selected move.obj.v operands from variables", () => {
        const r = runtime([0x52, 1, 30, 31, 32, 40, 0]);
        r.variables[30] = 50;
        r.variables[31] = 90;
        r.variables[32] = 2;
        tickRuntime(r, 100);
        expect([r.actors[1]!.targetX, r.actors[1]!.targetY, r.actors[1]!.stepSize, r.actors[1]!.motionFlag]).toEqual([50, 90, 2, 40]);
    });

    it("restores ego control when immediate, moving and border targets complete", () => {
        const immediate = runtime([0x51, 0, 10, 80, 0, 40, 0]);
        Object.assign(immediate.actors[0]!, { x: 10, y: 80 });
        tickRuntime(immediate, 100);
        expect([immediate.playerControl, immediate.flags[40], immediate.actors[0]!.motion]).toEqual([true, 1, 0]);
        const moving = runtime([0x51, 0, 12, 80, 0, 40, 0x91, 0]);
        Object.assign(moving.actors[0]!, { animated: true, drawn: true, cycling: false, x: 10, y: 80 });
        tickRuntime(moving, 100);
        expect(moving.playerControl).toBe(false);
        tickRuntime(moving, 100);
        tickRuntime(moving, 100);
        expect([moving.playerControl, moving.flags[40], moving.actors[0]!.x]).toEqual([true, 1, 12]);
        const border = runtime([0x51, 0, 159, 80, 2, 40, 0]);
        Object.assign(border.actors[0]!, { animated: true, drawn: true, cycling: false, x: 157, y: 80 });
        tickRuntime(border, 100);
        expect([border.playerControl, border.flags[40], border.actors[0]!.stepSize]).toEqual([true, 1, 1]);
    });

    it("round-trips game state without embedding resource bytes and rejects other revisions/corruption", () => {
        const r = runtime([0]);
        r.variables[3] = 42;
        r.actors[0]!.x = 77;
        const snapshot = serializeSnapshot(r);
        expect(snapshot).not.toContain('"files"');
        r.variables[3] = 0;
        deserializeSnapshot(r, snapshot);
        expect(r.variables[3]).toBe(42);
        expect(r.actors[0]!.x).toBe(77);
        expect(r.flags[12]).toBe(1);
        expect(() => deserializeSnapshot(r, snapshot.replace('"synthetic"', '"another"'))).toThrow("revision");
        expect(() => deserializeSnapshot(r, "{}")).toThrow("revision");
    });

    it("preserves the injected random provider and its live sequence across restore", () => {
        const r = runtime([0]);
        let count = 0;
        const random = () => ++count / 10;
        r.random = random;
        expect(r.random()).toBe(0.1);
        const snapshot = serializeSnapshot(r);
        expect(snapshot).not.toContain('"random"');
        expect(r.random()).toBe(0.2);
        deserializeSnapshot(r, snapshot);
        expect(r.random).toBe(random);
        expect(r.random()).toBe(0.3);
    });

    it.each(["keys", "controllerQueue"])("rejects malformed saved %s identifiers without modifying live state", (field) => {
        const r = runtime([0]);
        r.variables[3] = 42;
        const snapshot = JSON.parse(serializeSnapshot(r)) as { state: Record<string, unknown> };
        for (const invalid of [-1, 0.5, field === "keys" ? 65536 : 256]) {
            snapshot.state[field] = [invalid];
            expect(() => deserializeSnapshot(r, JSON.stringify(snapshot))).toThrow("byte array");
            expect(r.variables[3]).toBe(42);
        }
        snapshot.state[field] = new Array<number>(20).fill(1);
        expect(() => deserializeSnapshot(r, JSON.stringify(snapshot))).toThrow("input queue");
        expect(r.variables[3]).toBe(42);
    });

    it("fails loudly on unsupported opcodes and protects the browser from runaway bytecode", () => {
        expect(() => tickRuntime(runtime([0xfa, 0]), 100)).toThrow("Unsupported AGI action");
        expect(() => tickRuntime(runtime([0xfe, 0xfd, 0xff]), 100)).toThrow("50,000");
    });

    it("migrates version-one fields without accepting missing original state", () => {
        const r = runtime([0]);
        r.actors[0]!.x = 77;
        r.actors[0]!.y = 80;
        r.actors[0]!.stepTime = 3;
        r.actors[0]!.stepClock = 1;
        const snapshot = JSON.parse(serializeSnapshot(r)) as { version: number; state: Record<string, unknown> & { actors: Record<string, unknown>[] } };
        snapshot.version = 1;
        for (const key of ["displayRow", "inputRow", "statusRow", "inputWidth", "keys", "controllerQueue"]) {
            delete snapshot.state[key];
        }
        for (const actor of snapshot.state.actors) {
            for (const key of ["previousX", "previousY", "stationary", "newlyPositioned", "cycleDelay", "followDelay"]) {
                delete actor[key];
            }
        }
        deserializeSnapshot(r, JSON.stringify(snapshot));
        expect([r.actors[0]!.previousX, r.actors[0]!.previousY, r.actors[0]!.stepClock, r.inputRow, r.keys.length]).toEqual([77, 80, 2, 22, 0]);
        delete snapshot.state.actors[0]!.view;
        expect(() => deserializeSnapshot(r, JSON.stringify(snapshot))).toThrow("numeric");
    });
});
