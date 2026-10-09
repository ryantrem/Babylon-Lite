export type ResourceKind = "logic" | "pic" | "view" | "sound";

export interface Logic {
    code: Uint8Array;
    messages: string[];
}

export interface Cel {
    width: number;
    height: number;
    transparent: number;
    pixels: Uint8Array;
}

export interface View {
    loops: Cel[][];
    description: string;
}

export interface InventoryItem {
    name: string;
    room: number;
}

export interface GameResources {
    files: ReadonlyMap<string, Uint8Array>;
    words: Map<string, number>;
    inventory: InventoryItem[];
    logics: Map<number, Logic>;
    views: Map<number, View>;
    fingerprint: string;
}

export function byte(data: Uint8Array, offset: number): number {
    if (!Number.isInteger(offset) || offset < 0 || offset >= data.length) {
        throw new Error(`Truncated AGI resource at byte ${offset} (length ${data.length}).`);
    }
    return data[offset]!;
}

export function word(data: Uint8Array, offset: number): number {
    return byte(data, offset) | (byte(data, offset + 1) << 8);
}

function text(data: Uint8Array, offset: number): string {
    let result = "";
    for (let p = offset; ; p++) {
        const c = byte(data, p);
        if (c === 0) {
            return result;
        }
        result += String.fromCharCode(c);
    }
}

function decrypt(data: Uint8Array, start = 0): Uint8Array {
    const result = data.slice();
    const key = "Avis Durgan";
    for (let p = start; p < result.length; p++) {
        result[p] = result[p]! ^ key.charCodeAt((p - start) % key.length);
    }
    return result;
}

export function decodeWords(data: Uint8Array): Map<string, number> {
    if (data.length < 52) {
        throw new Error("WORDS.TOK is missing its alphabet index.");
    }
    const result = new Map<string, number>();
    let previous = "";
    let p = 52;
    while (p < data.length) {
        if (data.subarray(p).every((value) => value === 0)) {
            break;
        }
        const prefix = byte(data, p++);
        if (prefix > previous.length) {
            throw new Error("Invalid WORDS.TOK prefix length.");
        }
        let current = previous.slice(0, prefix);
        let c: number;
        do {
            c = byte(data, p++);
            current += String.fromCharCode((c ^ 0x7f) & 0x7f);
        } while ((c & 0x80) === 0);
        const id = (byte(data, p) << 8) | byte(data, p + 1);
        p += 2;
        result.set(current, id);
        previous = current;
    }
    return result;
}

export function decodeInventory(source: Uint8Array): InventoryItem[] {
    const plainOffset = word(source, 0);
    const data = plainOffset % 3 === 0 && plainOffset + 3 < source.length ? source : decrypt(source);
    const namesOffset = word(data, 0);
    if (namesOffset % 3 !== 0 || namesOffset + 3 >= data.length) {
        throw new Error("Unsupported or malformed OBJECT inventory file.");
    }
    const result: InventoryItem[] = [];
    for (let p = 3; p < namesOffset + 3; p += 3) {
        const nameOffset = word(data, p) + 3;
        if (nameOffset < namesOffset + 3) {
            throw new Error("Invalid inventory name offset.");
        }
        result.push({ name: text(data, nameOffset), room: byte(data, p + 2) });
    }
    return result;
}

export function loadGame(input: ReadonlyMap<string, Uint8Array>): GameResources {
    const files = new Map<string, Uint8Array>();
    let hash = 2166136261;
    for (const [path, data] of [...input.entries()].sort(([a], [b]) => a.localeCompare(b))) {
        const name = path.replace(/\\/g, "/").split("/").pop()!.toLowerCase();
        if (!/^(logdir|picdir|viewdir|snddir|words\.tok|object|vol\.\d+)$/.test(name)) {
            continue;
        }
        if (files.has(name)) {
            throw new Error(`Multiple copies of ${name}. Select one extracted game directory.`);
        }
        files.set(name, data);
        for (const c of name) {
            hash = Math.imul(hash ^ c.charCodeAt(0), 16777619) >>> 0;
        }
        for (const value of data) {
            hash = Math.imul(hash ^ value, 16777619) >>> 0;
        }
    }
    for (const name of ["logdir", "picdir", "viewdir", "snddir", "words.tok", "object"]) {
        if (!files.has(name)) {
            throw new Error(`Missing ${name.toUpperCase()}. Use an extracted DOS AGI v2 edition, not a boot disk, AGI v3 or SCI remake.`);
        }
    }
    const game: GameResources = {
        files,
        words: decodeWords(files.get("words.tok")!),
        inventory: decodeInventory(files.get("object")!),
        logics: new Map(),
        views: new Map(),
        fingerprint: hash.toString(16).padStart(8, "0"),
    };
    for (const kind of ["logic", "pic", "view", "sound"] as const) {
        const directory = files.get(kind === "sound" ? "snddir" : `${kind === "logic" ? "log" : kind}dir`)!;
        if (directory.length % 3 !== 0 || directory.length > 256 * 3) {
            throw new Error(`Malformed ${kind} directory.`);
        }
    }
    getLogic(game, 0);
    return game;
}

export function getResource(game: GameResources, kind: ResourceKind, id: number): Uint8Array {
    const name = kind === "sound" ? "snddir" : `${kind === "logic" ? "log" : kind}dir`;
    const directory = game.files.get(name);
    if (!directory || !Number.isInteger(id) || id < 0 || id * 3 + 2 >= directory.length) {
        throw new Error(`Missing ${kind} resource ${id}.`);
    }
    const entry = (byte(directory, id * 3) << 16) | (byte(directory, id * 3 + 1) << 8) | byte(directory, id * 3 + 2);
    if (entry === 0xffffff) {
        throw new Error(`Missing ${kind} resource ${id}.`);
    }
    const volumeId = entry >>> 20;
    const offset = entry & 0xfffff;
    const volume = game.files.get(`vol.${volumeId}`);
    if (!volume || byte(volume, offset) !== 0x12 || byte(volume, offset + 1) !== 0x34 || (byte(volume, offset + 2) & 0x7f) !== volumeId) {
        throw new Error(`Invalid ${kind} ${id} header in VOL.${volumeId}.`);
    }
    const length = word(volume, offset + 3);
    if (offset + 5 + length > volume.length) {
        throw new Error(`Truncated ${kind} ${id} in VOL.${volumeId}.`);
    }
    return volume.subarray(offset + 5, offset + 5 + length);
}

export function decodeLogic(source: Uint8Array): Logic {
    const codeLength = word(source, 0);
    const start = codeLength + 2;
    const count = byte(source, start);
    const textStart = start + 3 + count * 2;
    if (textStart > source.length) {
        throw new Error("Truncated LOGIC message table.");
    }
    const data = decrypt(source, textStart);
    const messages = [""];
    for (let n = 0; n < count; n++) {
        const offset = word(data, start + 3 + n * 2);
        if (offset !== 0 && start + 1 + offset < textStart) {
            throw new Error("Invalid LOGIC message offset.");
        }
        messages.push(offset === 0 ? "" : text(data, start + 1 + offset));
    }
    return { code: source.slice(2, start), messages };
}

export function getLogic(game: GameResources, id: number): Logic {
    let logic = game.logics.get(id);
    if (!logic) {
        logic = decodeLogic(getResource(game, "logic", id));
        game.logics.set(id, logic);
    }
    return logic;
}

export function decodeView(data: Uint8Array): View {
    const loops: Cel[][] = [];
    const loopCount = byte(data, 2);
    for (let loop = 0; loop < loopCount; loop++) {
        const start = word(data, 5 + loop * 2);
        const cels: Cel[] = [];
        for (let cel = 0; cel < byte(data, start); cel++) {
            const offset = start + word(data, start + 1 + cel * 2);
            const width = byte(data, offset);
            const height = byte(data, offset + 1);
            const flags = byte(data, offset + 2);
            if (width < 1 || width > 160 || height < 1 || height > 168) {
                throw new Error("Invalid VIEW cel dimensions.");
            }
            const transparent = flags & 15;
            const mirrored = (flags & 128) !== 0 && ((flags >> 4) & 7) !== loop;
            const pixels = new Uint8Array(width * height).fill(transparent);
            let p = offset + 3;
            for (let y = 0; y < height; y++) {
                let x = 0;
                let run: number;
                while ((run = byte(data, p++)) !== 0) {
                    const count = run & 15;
                    if (count === 0 || x + count > width) {
                        throw new Error("Invalid VIEW RLE run.");
                    }
                    for (let n = 0; n < count; n++, x++) {
                        pixels[y * width + (mirrored ? width - x - 1 : x)] = run >> 4;
                    }
                }
            }
            cels.push({ width, height, transparent, pixels });
        }
        if (cels.length === 0) {
            throw new Error("Empty VIEW loop.");
        }
        loops.push(cels);
    }
    const descriptionOffset = word(data, 3);
    return { loops, description: descriptionOffset ? text(data, descriptionOffset) : "" };
}

export function getView(game: GameResources, id: number): View {
    let view = game.views.get(id);
    if (!view) {
        view = decodeView(getResource(game, "view", id));
        game.views.set(id, view);
    }
    return view;
}
