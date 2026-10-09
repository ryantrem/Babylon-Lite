import { getView } from "./resources.js";
import type { Cel } from "./resources.js";
import type { Runtime } from "./runtime.js";
import { HEIGHT, WIDTH } from "./picture.js";

export interface Actor {
    x: number;
    y: number;
    view: number;
    loop: number;
    cel: number;
    direction: number;
    priority: number;
    fixedPriority: boolean;
    fixedLoop: boolean;
    animated: boolean;
    drawn: boolean;
    updating: boolean;
    cycling: boolean;
    cycleMode: number;
    cycleFlag: number;
    cycleTime: number;
    cycleClock: number;
    stepSize: number;
    stepTime: number;
    stepClock: number;
    ignoreHorizon: boolean;
    ignoreObjects: boolean;
    ignoreBlocks: boolean;
    terrain: number;
    motion: number;
    targetX: number;
    targetY: number;
    motionFlag: number;
    oldStepSize: number;
    followDistance: number;
    wanderClock: number;
    followDelay: number;
    previousX: number;
    previousY: number;
    stationary: boolean;
    newlyPositioned: boolean;
    cycleDelay: boolean;
}

export function createActor(): Actor {
    return {
        x: 0,
        y: 0,
        view: 0,
        loop: 0,
        cel: 0,
        direction: 0,
        priority: 4,
        fixedPriority: false,
        fixedLoop: false,
        animated: false,
        drawn: false,
        updating: true,
        cycling: true,
        cycleMode: 0,
        cycleFlag: 0,
        cycleTime: 1,
        cycleClock: 1,
        stepSize: 1,
        stepTime: 1,
        stepClock: 1,
        ignoreHorizon: false,
        ignoreObjects: false,
        ignoreBlocks: false,
        terrain: 0,
        motion: 0,
        targetX: 0,
        targetY: 0,
        motionFlag: 0,
        oldStepSize: 1,
        followDistance: 1,
        wanderClock: 0,
        followDelay: 255,
        previousX: 0,
        previousY: 0,
        stationary: false,
        newlyPositioned: false,
        cycleDelay: false,
    };
}

export function actorCel(r: Runtime, a: Actor): Cel {
    const loops = getView(r.game, a.view).loops;
    const loop = loops[a.loop];
    const cel = loop?.[a.cel];
    if (!cel) {
        throw new Error(`Invalid VIEW ${a.view}, loop ${a.loop}, cel ${a.cel}.`);
    }
    return cel;
}

export function priorityFor(y: number): number {
    return y < 48 ? 4 : Math.min(15, Math.floor(y / 12) + 1);
}

function direction(dx: number, dy: number): number {
    if (!dx && !dy) {
        return 0;
    }
    return dx > 0 ? (dy < 0 ? 2 : dy > 0 ? 4 : 3) : dx < 0 ? (dy < 0 ? 8 : dy > 0 ? 6 : 7) : dy < 0 ? 1 : 5;
}

export function canPlace(r: Runtime, a: Actor, x: number, y: number): boolean {
    const cel = actorCel(r, a);
    if (x < 0 || x + cel.width > WIDTH || y < cel.height - 1 || y >= HEIGHT || (!a.ignoreHorizon && y <= r.horizon)) {
        return false;
    }
    if (!a.ignoreObjects) {
        for (const other of r.actors) {
            if (other === a || !other.drawn || !other.animated || other.ignoreObjects) {
                continue;
            }
            const otherCel = actorCel(r, other);
            const overlaps = x <= other.x + otherCel.width && x + cel.width >= other.x;
            const crosses = (a.previousY < other.previousY && y > other.y) || (a.previousY > other.previousY && y < other.y);
            if (overlaps && (y === other.y || crosses)) {
                return false;
            }
        }
    }
    if (a.fixedPriority && a.priority === 15) {
        if (a === r.actors[0]) {
            r.flags[0] = r.flags[3] = 0;
        }
        return true;
    }
    let water = false;
    let trigger = false;
    for (let px = x; px < x + cel.width; px++) {
        const control = r.priority[y * WIDTH + px];
        if (control === 0 || (!a.ignoreBlocks && control === 1)) {
            return false;
        }
        water = control === 3;
        trigger = control === 2;
    }
    if ((a.terrain === 1 && !water) || (a.terrain === 2 && water)) {
        return false;
    }
    if (a === r.actors[0]) {
        r.flags[0] = Number(water);
        r.flags[3] = Number(trigger);
    }
    return true;
}

export function repositionActor(r: Runtime, a: Actor, x: number, y: number): void {
    const cel = actorCel(r, a);
    x = Math.max(0, Math.min(WIDTH - cel.width, x));
    y = Math.max(cel.height - 1, a.ignoreHorizon ? 0 : r.horizon + 1, Math.min(HEIGHT - 1, y));
    if (canPlace(r, a, x, y)) {
        a.x = x;
        a.y = y;
        return;
    }
    let nx = x;
    let ny = y;
    const turns = [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ];
    for (let leg = 0; leg < (WIDTH + HEIGHT) * 2; leg++) {
        const delta = turns[leg % 4]!;
        const distance = Math.floor(leg / 2) + 1;
        for (let step = 0; step < distance; step++) {
            nx += delta[0]!;
            ny += delta[1]!;
            if (canPlace(r, a, nx, ny)) {
                a.x = nx;
                a.y = ny;
                return;
            }
        }
    }
    throw new Error("No legal position for AGI actor.");
}

export function prepareActors(r: Runtime): void {
    const ego = r.actors[0]!;
    if (r.playerControl) {
        ego.direction = r.variables[6]!;
    } else {
        r.variables[6] = ego.direction;
    }
    for (let id = 0; id < r.actors.length; id++) {
        const a = r.actors[id]!;
        if (!a.animated || !a.drawn || !a.updating || a.stepClock !== 1) {
            continue;
        }
        prepareMotion(r, a);
        if (a.direction && !a.ignoreBlocks && r.block) {
            const [left, top, right, bottom] = r.block;
            const inside = (x: number, y: number): boolean => x > left && x < right && y > top && y < bottom;
            const nx = a.x + [0, 0, 1, 1, 1, 0, -1, -1, -1][a.direction]! * a.stepSize;
            const ny = a.y + [0, -1, -1, 0, 1, 1, 1, 0, -1][a.direction]! * a.stepSize;
            if (inside(a.x, a.y) !== inside(nx, ny)) {
                a.direction = 0;
            }
        }
        if (id === 0) {
            r.variables[6] = a.direction;
        }
    }
}

function completeMotion(r: Runtime, a: Actor): void {
    if (a.motion === 1) {
        a.stepSize = a.oldStepSize;
        if (a === r.actors[0]) {
            r.playerControl = true;
        }
    }
    a.motion = a.direction = 0;
    r.flags[a.motionFlag] = 1;
}

export function prepareMotion(r: Runtime, a: Actor): void {
    const randomWord = (): number => Math.floor(r.random() * 65536);
    if (a.motion === 1 || a.motion === 2) {
        const target = a.motion === 1 ? [a.targetX, a.targetY] : [r.actors[0]!.x, r.actors[0]!.y];
        const dx = target[0]! - a.x + (a.motion === 2 ? Math.floor(actorCel(r, r.actors[0]!).width / 2) - Math.floor(actorCel(r, a).width / 2) : 0);
        const dy = target[1]! - a.y;
        const range = a.motion === 1 ? a.stepSize : a.followDistance;
        const direct = direction(Math.abs(dx) < range ? 0 : dx, Math.abs(dy) < range ? 0 : dy);
        if (!direct) {
            completeMotion(r, a);
        } else if (a.motion === 1) {
            a.direction = direct;
        } else if (a.followDelay === 255) {
            a.followDelay = 0;
            a.direction = direct;
        } else if (a.stationary) {
            do {
                a.direction = randomWord() % 9;
            } while (!a.direction);
            const distance = Math.floor((Math.abs(dx) + Math.abs(dy)) / 2) + 1;
            if (distance <= a.stepSize) {
                a.followDelay = a.stepSize;
            } else {
                do {
                    a.followDelay = randomWord() % distance;
                } while (a.followDelay < a.stepSize);
            }
        } else if (a.followDelay) {
            const remaining = (a.followDelay - a.stepSize) & 255;
            a.followDelay = remaining < 128 ? remaining : 0;
        } else {
            a.direction = direct;
        }
    } else if (a.motion === 3) {
        const old = a.wanderClock;
        a.wanderClock = (old - 1) & 255;
        if (!old || a.stationary) {
            a.direction = randomWord() % 9;
            do {
                a.wanderClock = randomWord() % 51;
            } while (a.wanderClock < 6);
        }
    }
    if (a === r.actors[0]) {
        r.variables[6] = a.direction;
    }
}

export function updateActors(r: Runtime): void {
    const dxs = [0, 0, 1, 1, 1, 0, -1, -1, -1];
    const dys = [0, -1, -1, 0, 1, 1, 1, 0, -1];
    for (const a of r.actors) {
        a.previousX = a.x;
        a.previousY = a.y;
    }
    for (let id = 0; id < r.actors.length; id++) {
        const a = r.actors[id]!;
        if (!a.animated || !a.drawn || !a.updating) {
            continue;
        }
        if (id === 0) {
            a.direction = r.variables[6]!;
        }
        const view = getView(r.game, a.view);
        if (!a.fixedLoop && a.direction !== 0 && a.stepClock === 1) {
            const loop = a.direction >= 2 && a.direction <= 4 ? 0 : a.direction >= 6 ? 1 : a.direction === 5 ? 2 : 3;
            if (view.loops.length === 4 || ((view.loops.length === 2 || view.loops.length === 3) && loop < 2)) {
                a.loop = loop;
                a.cel = Math.min(a.cel, view.loops[loop]!.length - 1);
            }
        }
        if (a.cycling && (!a.cycleClock || --a.cycleClock === 0)) {
            a.cycleClock = a.cycleTime;
            if (a.cycleDelay) {
                a.cycleDelay = false;
            } else {
                const count = view.loops[a.loop]!.length;
                const reverse = a.cycleMode >= 2;
                const next = a.cel + (reverse ? -1 : 1);
                if (a.cycleMode === 1 || a.cycleMode === 3) {
                    a.cel = Math.max(0, Math.min(count - 1, next));
                    if ((!reverse && a.cel === count - 1) || (reverse && a.cel === 0)) {
                        a.cycling = false;
                        a.direction = a.cycleMode = 0;
                        r.flags[a.cycleFlag] = 1;
                    }
                } else {
                    a.cel = (next + count) % count;
                }
            }
        }
        if (!a.stepClock || --a.stepClock === 0) {
            a.stepClock = a.stepTime;
            if (!a.newlyPositioned) {
                const cel = actorCel(r, a);
                let nx = a.x + dxs[a.direction]! * a.stepSize;
                let ny = a.y + dys[a.direction]! * a.stepSize;
                let border = 0;
                if (nx < 0) {
                    nx = 0;
                    border = 4;
                } else if (nx + cel.width > WIDTH) {
                    nx = WIDTH - cel.width;
                    border = 2;
                }
                if (ny >= HEIGHT) {
                    ny = HEIGHT - 1;
                    border = 3;
                } else if (ny < cel.height - 1 || (!a.ignoreHorizon && ny <= r.horizon)) {
                    ny = Math.max(cel.height - 1, a.ignoreHorizon ? 0 : r.horizon + 1);
                    border = 1;
                }
                if (canPlace(r, a, nx, ny)) {
                    a.x = nx;
                    a.y = ny;
                    if (border) {
                        if (a.motion === 1) {
                            completeMotion(r, a);
                        }
                        if (id === 0) {
                            r.variables[2] = border;
                        } else {
                            r.variables[4] = id;
                            r.variables[5] = border;
                        }
                    }
                } else {
                    repositionActor(r, a, a.x, a.y);
                }
            }
            a.newlyPositioned = false;
        }
        if (!a.fixedPriority) {
            a.priority = priorityFor(a.y);
        }
        a.stationary = a.x === a.previousX && a.y === a.previousY;
        if (id === 0) {
            r.variables[6] = a.direction;
            canPlace(r, a, a.x, a.y);
        }
    }
}

export function stampCel(picture: Uint8Array, priorities: Uint8Array, cel: Cel, x: number, y: number, priority: number): void {
    const top = y - cel.height + 1;
    if (top < 0) {
        x += top;
        y -= top;
    }
    x = Math.max(0, Math.min(WIDTH - cel.width, x));
    for (let cy = 0; cy < cel.height; cy++) {
        const py = y - cel.height + 1 + cy;
        if (py < 0 || py >= HEIGHT) {
            continue;
        }
        for (let cx = 0; cx < cel.width; cx++) {
            const px = x + cx;
            if (px < 0 || px >= WIDTH) {
                continue;
            }
            const color = cel.pixels[cy * cel.width + cx]!;
            const index = py * WIDTH + px;
            let backgroundPriority = priorities[index]!;
            if (backgroundPriority < 3) {
                let below = py + 1;
                while (below < HEIGHT && priorities[below * WIDTH + px]! < 3) {
                    below++;
                }
                backgroundPriority = below < HEIGHT ? priorities[below * WIDTH + px]! : 0;
            }
            if (color !== cel.transparent && priority >= backgroundPriority) {
                picture[index] = color;
                priorities[index] = priority;
            }
        }
    }
}

export function renderRuntime(r: Runtime, pixels: Uint8ClampedArray): void {
    if (pixels.length !== WIDTH * 2 * HEIGHT * 4) {
        throw new Error("Incorrect AGI framebuffer length.");
    }
    const colors = r.shownPicture.slice();
    const priorities = r.shownPriority.slice();
    if (r.graphics) {
        const actors = r.actors.filter((a) => a.drawn).sort((a, b) => Number(a.updating) - Number(b.updating) || a.priority - b.priority || a.y - b.y);
        for (const actor of actors) {
            stampCel(colors, priorities, actorCel(r, actor), actor.x, actor.y, actor.priority);
        }
    } else {
        colors.fill(r.background & 15);
    }
    for (let i = 0; i < colors.length; i++) {
        const index = colors[i]!;
        const bright = index & 8 ? 85 : 0;
        const red = index === 6 ? 170 : (index & 4 ? 170 : 0) + bright;
        const green = index === 6 ? 85 : (index & 2 ? 170 : 0) + bright;
        const blue = (index & 1 ? 170 : 0) + bright;
        const offset = i * 8;
        pixels[offset] = pixels[offset + 4] = red;
        pixels[offset + 1] = pixels[offset + 5] = green;
        pixels[offset + 2] = pixels[offset + 6] = blue;
        pixels[offset + 3] = pixels[offset + 7] = 255;
    }
}
