import { byte } from "./resources.js";

export const WIDTH = 160;
export const HEIGHT = 168;

export function drawPicture(data: Uint8Array, picture: Uint8Array, priority: Uint8Array, clear = true): void {
    if (picture.length !== WIDTH * HEIGHT || priority.length !== picture.length) {
        throw new Error("AGI picture planes must be 160x168.");
    }
    if (clear) {
        picture.fill(15);
        priority.fill(4);
    }
    let color = -1;
    let pri = -1;
    let pen = 0;
    let p = 0;
    function put(x: number, y: number): void {
        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
            return;
        }
        const index = y * WIDTH + x;
        if (color >= 0) {
            picture[index] = color;
        }
        if (pri >= 0) {
            priority[index] = pri;
        }
    }
    function line(x: number, y: number, nx: number, ny: number): void {
        const dx = Math.abs(nx - x);
        const dy = Math.abs(ny - y);
        const major = Math.max(dx, dy);
        let xerror = dx >= dy ? 0 : Math.floor(dy / 2);
        let yerror = dx >= dy ? Math.floor(dx / 2) : 0;
        const xstep = nx >= x ? 1 : -1;
        const ystep = ny >= y ? 1 : -1;
        put(x, y);
        for (let i = 0; i < major; i++) {
            yerror = (yerror + dy) & 255;
            if (yerror >= major) {
                yerror = (yerror - major) & 255;
                y += ystep;
            }
            xerror = (xerror + dx) & 255;
            if (xerror >= major) {
                xerror = (xerror - major) & 255;
                x += xstep;
            }
            put(x, y);
        }
    }
    function fill(x: number, y: number): void {
        if ((color < 0 && pri < 0) || color === 15 || (color < 0 && pri === 4) || x >= WIDTH || y >= HEIGHT) {
            return;
        }
        const stack = [y * WIDTH + x];
        const visited = new Uint8Array(picture.length);
        while (stack.length) {
            const index = stack.pop()!;
            if (visited[index] || (color >= 0 ? picture[index] !== 15 : priority[index] !== 4)) {
                continue;
            }
            visited[index] = 1;
            const px = index % WIDTH;
            const py = Math.floor(index / WIDTH);
            put(px, py);
            if (px > 0) {
                stack.push(index - 1);
            }
            if (px < WIDTH - 1) {
                stack.push(index + 1);
            }
            if (py > 0) {
                stack.push(index - WIDTH);
            }
            if (py < HEIGHT - 1) {
                stack.push(index + WIDTH);
            }
        }
    }
    function argument(): number {
        const value = byte(data, p++);
        if (value >= 0xf0) {
            throw new Error("Truncated PIC drawing arguments.");
        }
        return value;
    }
    while (p < data.length) {
        const opcode = byte(data, p++);
        if (opcode === 0xff) {
            return;
        }
        switch (opcode) {
            case 0xf0:
                color = byte(data, p++) & 15;
                break;
            case 0xf1:
                color = -1;
                break;
            case 0xf2:
                pri = byte(data, p++) & 15;
                break;
            case 0xf3:
                pri = -1;
                break;
            case 0xf4:
            case 0xf5:
            case 0xf6:
            case 0xf7: {
                let x = Math.min(WIDTH - 1, argument());
                let y = Math.min(HEIGHT - 1, argument());
                put(x, y);
                let vertical = opcode === 0xf4;
                while (byte(data, p) < 0xf0) {
                    let nx = x;
                    let ny = y;
                    if (opcode === 0xf6) {
                        nx = Math.min(WIDTH - 1, argument());
                        ny = Math.min(HEIGHT - 1, argument());
                    } else if (opcode === 0xf7) {
                        const d = argument();
                        nx += ((d >> 4) & 7) * (d & 128 ? -1 : 1);
                        ny += (d & 7) * (d & 8 ? -1 : 1);
                        nx = Math.min(WIDTH - 1, nx & 255);
                        ny = Math.min(HEIGHT - 1, ny & 255);
                    } else if (vertical) {
                        ny = Math.min(HEIGHT - 1, argument());
                    } else {
                        nx = Math.min(WIDTH - 1, argument());
                    }
                    line(x, y, nx, ny);
                    x = nx;
                    y = ny;
                    vertical = !vertical;
                }
                break;
            }
            case 0xf8:
                while (byte(data, p) < 0xf0) {
                    const x = argument();
                    fill(x, argument());
                }
                break;
            case 0xf9:
                pen = byte(data, p++);
                break;
            case 0xfa: {
                const masks = [
                    [0x8000],
                    [0xe000, 0xe000, 0xe000],
                    [0x7000, 0xf800, 0xf800, 0xf800, 0x7000],
                    [0x3800, 0x7c00, 0xfe00, 0xfe00, 0xfe00, 0x7c00, 0x3800],
                    [0x1c00, 0x7f00, 0xff80, 0xff80, 0xff80, 0xff80, 0xff80, 0x7f00, 0x1c00],
                    [0x0e00, 0x3f80, 0x7fc0, 0x7fc0, 0xffe0, 0xffe0, 0xffe0, 0x7fc0, 0x7fc0, 0x3f80, 0x1f00],
                    [0x0f80, 0x3fe0, 0x7ff0, 0x7ff0, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0x7ff0, 0x7ff0, 0x3fe0, 0x0f80],
                    [0x07c0, 0x1ff0, 0x3ff8, 0x7ffc, 0x7ffc, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0x7ffc, 0x7ffc, 0x3ff8, 0x1ff0, 0x07c0],
                ];
                const radius = pen & 7;
                while (byte(data, p) < 0xf0) {
                    let stipple = pen & 32 ? argument() | 1 : 0;
                    const x = Math.min(WIDTH - 1, argument());
                    const y = Math.min(HEIGHT - 1, argument());
                    const sx = Math.floor(Math.max(0, Math.min(320 - radius * 2, x * 2 - radius)) / 2);
                    const sy = Math.max(0, Math.min(167 - radius * 2, y - radius));
                    for (let row = 0; row < radius * 2 + 1; row++) {
                        for (let column = 0; column <= radius; column++) {
                            if (!(pen & 16) && !(masks[radius]![row]! & (0x8000 >> (column * 2)))) {
                                continue;
                            }
                            if (pen & 32) {
                                const carry = stipple & 1;
                                stipple >>= 1;
                                if (carry) {
                                    stipple ^= 0xb8;
                                }
                                if ((stipple & 3) !== 2) {
                                    continue;
                                }
                            }
                            const index = (sy + row) * WIDTH + sx + column;
                            if (index < picture.length) {
                                put(index % WIDTH, Math.floor(index / WIDTH));
                            }
                        }
                    }
                }
                break;
            }
            default:
                throw new Error(`Unsupported PIC opcode 0x${opcode.toString(16)} at ${p - 1}.`);
        }
    }
    throw new Error("PIC resource has no end marker.");
}
