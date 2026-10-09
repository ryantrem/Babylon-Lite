import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { actorCel, canPlace } from "../../../lab/lite/src/demos/kings-quest/actors.js";
import { deserializeSnapshot, serializeSnapshot } from "../../../lab/lite/src/demos/kings-quest/persistence.js";
import { getView, loadGame } from "../../../lab/lite/src/demos/kings-quest/resources.js";
import type { GameResources } from "../../../lab/lite/src/demos/kings-quest/resources.js";
import { createRuntime, dismissModal, sendDirection, sendKey, submitCommand, tickRuntime } from "../../../lab/lite/src/demos/kings-quest/runtime.js";
import type { Runtime } from "../../../lab/lite/src/demos/kings-quest/runtime.js";

export const kingsQuestAssetNames = ["LOGDIR", "OBJECT", "PICDIR", "SNDDIR", "VIEWDIR", "WORDS.TOK", "VOL.0", "VOL.1", "VOL.2"] as const;

export interface QuestState {
    room: number;
    x: number;
    y: number;
    view: number;
    score: number;
    inventory: string[];
    dead: boolean;
}

export type QuestInput =
    | { kind: "tick"; milliseconds: number; count: number }
    | { kind: "direction"; value: number }
    | { kind: "key"; value: number }
    | { kind: "command"; text: string }
    | { kind: "acknowledge" };

export interface QuestEvent {
    kind: "room" | "modal" | "command" | "walk" | "notice" | "checkpoint" | "death";
    detail: string;
    state: QuestState;
}

export interface QuestResult {
    runtime: Runtime;
    inputs: QuestInput[];
    trace: QuestEvent[];
    checkpoint: string;
}

export function loadKingsQuestGame(): GameResources {
    return loadGame(new Map(kingsQuestAssetNames.map((name) => [name, new Uint8Array(readFileSync(resolve("lab", "lite", "src", "demos", "kings-quest", "assets", name)))])));
}

export function questState(r: Runtime): QuestState {
    const a = r.actors[0]!;
    return {
        room: r.variables[0]!,
        x: a.x,
        y: a.y,
        view: a.view,
        score: r.variables[3]!,
        inventory: r.game.inventory.filter((_, i) => r.inventory[i] === 255).map((item) => item.name),
        dead: Boolean(r.flags[63]),
    };
}

function freshRuntime(game: GameResources, notify: (text: string) => void): Runtime {
    let saved = "";
    let seed = 23;
    const r = createRuntime(game, {
        sound: () => {},
        stopSound: () => {},
        editCommand: () => {},
        notify,
        save: (current) => {
            saved = serializeSnapshot(current);
        },
        restore: (current) => {
            if (!saved) {
                throw new Error("No legitimately earned save.");
            }
            deserializeSnapshot(current, saved);
        },
    });
    r.random = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 4294967296;
    };
    return r;
}

export function replayQuestInputs(game: GameResources, inputs: readonly QuestInput[]): Runtime {
    const r = freshRuntime(game, () => {});
    for (const input of inputs) {
        switch (input.kind) {
            case "tick":
                for (let i = 0; i < input.count; i++) {
                    tickRuntime(r, input.milliseconds);
                }
                break;
            case "direction":
                sendDirection(r, input.value);
                break;
            case "key":
                sendKey(r, input.value);
                break;
            case "command":
                submitCommand(r, input.text);
                break;
            case "acknowledge":
                dismissModal(r);
                break;
        }
    }
    return r;
}

export function playQuest(game: GameResources, actions: readonly string[], allowDeath = false): QuestResult {
    const inputs: QuestInput[] = [];
    const trace: QuestEvent[] = [];
    const r = freshRuntime(game, (text) => {
        trace.push({ kind: "notice", detail: text, state: questState(r) });
    });
    let checkpoint = "";
    const dx = [0, 0, 1, 1, 1, 0, -1, -1, -1];
    const dy = [0, -1, -1, 0, 1, 1, 1, 0, -1];
    const event = (kind: QuestEvent["kind"], detail: string): void => {
        trace.push({ kind, detail, state: questState(r) });
    };
    const steer = (value: number): void => {
        inputs.push({ kind: "direction", value });
        sendDirection(r, value);
    };
    const key = (value: number): void => {
        inputs.push({ kind: "key", value });
        sendKey(r, value);
    };
    function advance(milliseconds: number): void {
        const last = inputs.at(-1);
        if (last?.kind === "tick" && last.milliseconds === milliseconds) {
            last.count++;
        } else {
            inputs.push({ kind: "tick", milliseconds, count: 1 });
        }
        const room = r.variables[0];
        const dead = r.flags[63];
        tickRuntime(r, milliseconds);
        if (room !== r.variables[0]) {
            event("room", `${room} -> ${r.variables[0]}`);
        }
        if (!dead && r.flags[63]) {
            event("death", "Original game death flag.");
        }
    }
    function tick(): void {
        advance(Math.max(1, r.variables[10]!) * 50);
        let count = 0;
        while (r.modal) {
            if (++count > 100) {
                throw new Error("Quest modal loop.");
            }
            event("modal", r.modal.text);
            if (r.modal.kind === "quit") {
                throw new Error("Unexpected quest quit.");
            }
            inputs.push({ kind: "acknowledge" });
            dismissModal(r);
            advance(0);
        }
        if (r.keyPoll && r.stack.length) {
            key(32);
            advance(0);
        }
        if (r.flags[63] && !allowDeath) {
            throw new Error(`Quest death: ${JSON.stringify(questState(r))}`);
        }
    }
    function wait(count = 1): void {
        for (let i = 0; i < count; i++) {
            tick();
            if (allowDeath && r.flags[63]) {
                break;
            }
        }
    }
    function stop(): void {
        if (r.playerControl && r.variables[6]) {
            steer(0);
        }
    }
    function ready(): void {
        for (let i = 0; i < 500 && (!r.playerControl || !r.actors[0]!.drawn || r.flags[5]); i++) {
            tick();
        }
        if (!r.playerControl || !r.actors[0]!.drawn || r.flags[5]) {
            throw new Error(`Quest navigation unavailable: ${JSON.stringify(questState(r))}`);
        }
    }
    function command(text: string): void {
        stop();
        event("command", text);
        if (!r.inputEnabled) {
            throw new Error(`Quest command input unavailable: ${text}`);
        }
        inputs.push({ kind: "command", text });
        submitCommand(r, text);
        wait();
    }
    function plan(tx: number, ty: number, water = false, near = 0): number[] {
        const ego = r.actors[0]!;
        const a = { ...ego, terrain: water ? ego.terrain : 2 };
        const planning = { ...r, flags: r.flags.slice(), actors: [a, ...r.actors.slice(1)] };
        const width = actorCel(r, a).width;
        const step = a.stepSize;
        const view = getView(r.game, a.view);
        const start = a.y * 160 + a.x;
        let goal = near ? -1 : ty * 160 + tx;
        const from = new Int32Array(160 * 168).fill(-1);
        const via = new Uint8Array(from.length);
        const queue = new Int32Array(from.length);
        from[start] = start;
        queue[0] = start;
        let head = 0;
        let tail = 1;
        while (head < tail && (goal === -1 || from[goal] === -1)) {
            const p = queue[head++]!;
            const x = p % 160;
            const y = Math.floor(p / 160);
            a.x = a.previousX = x;
            a.y = a.previousY = y;
            if (near && Math.abs(tx - x) + Math.abs(ty - y) < near) {
                goal = p;
                break;
            }
            for (let d = 1; d < 9; d++) {
                a.loop = ego.loop;
                a.cel = ego.cel;
                if (!a.fixedLoop) {
                    const loop = d >= 2 && d <= 4 ? 0 : d >= 6 ? 1 : d === 5 ? 2 : 3;
                    if (view.loops.length === 4 || ((view.loops.length === 2 || view.loops.length === 3) && loop < 2)) {
                        a.loop = loop;
                        a.cel = Math.min(a.cel, view.loops[loop]!.length - 1);
                    }
                }
                const cel = actorCel(r, a);
                const nx = x + dx[d]! * step;
                const ny = y + dy[d]! * step;
                if (nx < 0 || nx + cel.width > 160 || ny >= 168 || ny < cel.height - 1 || (!a.ignoreHorizon && ny <= r.horizon)) {
                    continue;
                }
                const next = ny * 160 + nx;
                if (from[next] !== -1 || !canPlace(planning, a, nx, ny)) {
                    continue;
                }
                if (!a.ignoreBlocks && r.block) {
                    const [l, t, rr, b] = r.block;
                    const inside = (px: number, py: number): boolean => px > l && px < rr && py > t && py < b;
                    if (inside(x, y) !== inside(nx, ny)) {
                        continue;
                    }
                }
                const room = r.variables[0]!;
                const trigger = r.priority[ny * 160 + nx + cel.width - 1] === 2;
                if (!water && (room === 1 || room === 2 || room === 63) && trigger) {
                    continue;
                }
                if (r.flags[157] && trigger && !(room === 72 && nx >= 65 && nx <= 82 && ny >= 100 && ny <= 120)) {
                    continue;
                }
                if (room >= 56 && room <= 62 && trigger) {
                    continue;
                }
                if (room === 72 && !r.flags[157] && trigger) {
                    continue;
                }
                if (room >= 66 && room <= 69 && trigger) {
                    continue;
                }
                from[next] = p;
                via[next] = d;
                queue[tail++] = next;
            }
        }
        if (goal === -1 || from[goal] === -1) {
            throw new Error(`No path to ${tx},${ty} from ${JSON.stringify(questState(r))}; visited ${tail}, width ${width}.`);
        }
        const path: number[] = [];
        for (let p = goal; p !== start; p = from[p]!) {
            path.push(via[p]!);
        }
        return path.reverse();
    }
    function reachable(x: number, y: number): boolean {
        try {
            plan(x, y);
            return true;
        } catch (error) {
            if (error instanceof Error && error.message.startsWith("No path")) {
                return false;
            }
            throw error;
        }
    }
    function walk(x: number, y: number, water = false, near = 0, landing = false): void {
        ready();
        const room = r.variables[0];
        event("walk", `${x},${y}`);
        for (let attempt = 0; attempt < 300; attempt++) {
            if (allowDeath && r.flags[63]) {
                return;
            }
            if (landing && !r.flags[157]) {
                stop();
                return;
            }
            if (r.variables[0] !== room) {
                stop();
                return;
            }
            if (!r.playerControl) {
                wait();
                continue;
            }
            const a = r.actors[0]!;
            if (near ? Math.abs(a.x - x) + Math.abs(a.y - y) < near : a.x === x && a.y === y) {
                stop();
                return;
            }
            for (const direction of plan(x, y, water, near)) {
                const bx = a.x;
                const by = a.y;
                const step = a.stepSize;
                if (r.variables[6] !== direction) {
                    steer(direction);
                }
                wait();
                if (r.flags[63] || r.variables[0] !== room || !r.playerControl || a.x !== bx + dx[direction]! * step || a.y !== by + dy[direction]! * step) {
                    break;
                }
            }
        }
        throw new Error(`Quest walking stalled toward ${x},${y}: ${JSON.stringify(questState(r))}`);
    }
    function sorted(points: number[][]): number[][] {
        const a = r.actors[0]!;
        return points.sort((p, q) => Math.abs(p[0]! - a.x) + Math.abs(p[1]! - a.y) - Math.abs(q[0]! - a.x) - Math.abs(q[1]! - a.y));
    }
    function exit(direction: number): void {
        ready();
        const a = r.actors[0]!;
        const cel = actorCel(r, a);
        const room = r.variables[0];
        const points: number[][] = [];
        if (direction === 3 || direction === 7) {
            for (let y = r.horizon + 1; y < 168; y++) {
                points.push([direction === 3 ? 160 - cel.width : 0, y]);
            }
        } else {
            for (let x = 0; x <= 160 - cel.width; x++) {
                points.push([x, direction === 5 ? 167 : Math.max(cel.height - 1, a.ignoreHorizon ? 0 : r.horizon + 1)]);
            }
        }
        const target = sorted(points).find(([x, y]) => reachable(x!, y!));
        if (!target) {
            throw new Error(`No quest exit ${direction}: ${JSON.stringify(questState(r))}`);
        }
        walk(target[0]!, target[1]!);
        if (r.variables[0] !== room) {
            return;
        }
        if (r.variables[6] !== direction) {
            steer(direction);
        }
        for (let i = 0; i < 10 && r.variables[0] === room; i++) {
            wait();
        }
        stop();
        wait(2);
        if (r.variables[0] === room) {
            throw new Error(`Quest exit failed ${direction}.`);
        }
    }
    function region(left: number, top: number, right: number, bottom: number): void {
        ready();
        const points: number[][] = [];
        for (let y = top; y <= bottom; y++) {
            for (let x = left; x <= right; x++) {
                points.push([x, y]);
            }
        }
        const target = sorted(points).find(([x, y]) => reachable(x!, y!));
        if (!target) {
            throw new Error(`No reachable quest region ${left},${top}..${right},${bottom}.`);
        }
        walk(target[0]!, target[1]!);
    }
    function condor(): void {
        ready();
        for (let attempt = 0; attempt < 10000 && r.variables[0] === 22; attempt++) {
            const a = r.actors[0]!;
            const bird = r.actors[1]!;
            if (!bird.animated && attempt % 500 === 499) {
                exit(3);
                exit(7);
                continue;
            }
            if (!bird.drawn) {
                wait();
                continue;
            }
            const distance = Math.abs(a.x + actorCel(r, a).width / 2 - bird.x - actorCel(r, bird).width / 2) + Math.abs(a.y - bird.y);
            if (a.y === bird.y + 20 && distance > 20 && distance < 35) {
                command("jump");
                if (r.variables[0] !== 22 || r.actors[0]!.view === 89) {
                    for (let i = 0; i < 500 && r.variables.at(0) !== 48; i++) {
                        wait();
                    }
                    wait(60);
                    break;
                }
                wait(12);
                continue;
            }
            if (bird.y + 20 <= r.horizon || bird.y + 20 >= 168) {
                stop();
                wait();
                continue;
            }
            let path: number[];
            try {
                path = plan(bird.x, bird.y + 20);
            } catch (error) {
                if (error instanceof Error && error.message.startsWith("No path")) {
                    stop();
                    wait();
                    continue;
                }
                throw error;
            }
            if (path.length) {
                if (r.variables[6] !== path[0]) {
                    steer(path[0]!);
                }
            } else {
                stop();
            }
            wait();
        }
        for (let i = 0; i < 500 && r.variables[0] === 80; i++) {
            wait();
        }
        if (r.variables[0] === 48) {
            wait(60);
        }
        ready();
        if (r.variables[0] !== 48) {
            throw new Error(`Quest condor ride failed: ${JSON.stringify(questState(r))}`);
        }
    }
    wait(4);
    key(32);
    wait(4);
    for (const action of actions) {
        const [kind, ...args] = action.split(":");
        const n = (i: number): number => Number(args[i]);
        switch (kind) {
            case "walk":
                walk(n(0), n(1), args[2] === "water");
                break;
            case "landing":
                walk(n(0), n(1), false, 0, true);
                break;
            case "region":
                region(n(0), n(1), n(2), n(3));
                break;
            case "exit":
                exit(n(0));
                break;
            case "wait":
                wait(n(0));
                break;
            case "command":
                command(args.join(":"));
                break;
            case "direction":
                steer(n(0));
                wait(n(1));
                stop();
                break;
            case "key":
                key(n(0));
                break;
            case "checkpoint":
                command("save game");
                checkpoint = serializeSnapshot(r);
                event("checkpoint", "Earned original save.");
                break;
            case "house": {
                let absent = false;
                for (let i = 0; i < 20; i++) {
                    region(40, 127, 70, 135);
                    const before = trace.length;
                    command("eat house");
                    if (trace.slice(before).some((entry) => entry.kind === "modal" && entry.detail.startsWith("Yum!"))) {
                        absent = true;
                        break;
                    }
                    exit(3);
                    exit(7);
                }
                if (!absent) {
                    throw new Error("Quest witch did not leave her house.");
                }
                break;
            }
            case "trigger": {
                ready();
                const width = actorCel(r, r.actors[0]!).width;
                const points: number[][] = [];
                for (let y = r.horizon + 1; y < 168; y++) {
                    for (let x = 0; x <= 160 - width; x++) {
                        if (r.priority[y * 160 + x + width - 1] === 2) {
                            points.push([x, y]);
                        }
                    }
                }
                const target = sorted(points).find(([x, y]) => reachable(x!, y!));
                if (!target) {
                    throw new Error("No reachable quest doorway.");
                }
                walk(target[0]!, target[1]!);
                wait(2);
                break;
            }
            case "condor":
                condor();
                break;
            case "approach": {
                ready();
                for (let i = 0; i < 1000; i++) {
                    const a = r.actors[0]!;
                    const b = r.actors[n(0)]!;
                    const distance = Math.abs(a.x + actorCel(r, a).width / 2 - b.x - actorCel(r, b).width / 2) + Math.abs(a.y - b.y);
                    if (distance < n(1)) {
                        stop();
                        break;
                    }
                    const path = plan(b.x, b.y, false, n(1));
                    if (path.length && r.variables[6] !== path[0]) {
                        steer(path[0]!);
                    }
                    wait();
                    if (i === 999) {
                        throw new Error("Quest approach timed out.");
                    }
                }
                break;
            }
            case "near": {
                ready();
                const a = r.actors[n(0)]!;
                walk(a.x + Math.floor(actorCel(r, a).width / 2) - Math.floor(actorCel(r, r.actors[0]!).width / 2), a.y, false, n(1));
                break;
            }
            default:
                throw new Error(`Unknown quest action ${action}.`);
        }
    }
    return { runtime: r, inputs, trace, checkpoint };
}
