import type { ComputeTextProgram } from "../compute-text-program.js";

/** Compact acceleration only; original rules remain the complete semantic program. */
export function prepareSimplePairWords(program: ComputeTextProgram, origin: number): Uint32Array {
    const source = program.words;
    const words = new Array<number>(program.rootOrder.length).fill(0);
    if (program.rootOrder.length > 32) {
        return new Uint32Array();
    }
    const cell = (rule: number): readonly [number, number] => {
        const mode = source[rule + 8]!;
        if (mode === 5) {
            return [0, 2];
        }
        if (mode !== 3 && mode !== 4) {
            throw new Error("compute-text: non-pair action in a proven first-only lookup");
        }
        return [source[source[rule + 6]! + (mode === 4 ? 2 : 1)]!, 1];
    };
    const rows = new Map<number, number>();
    for (let root = 0; root < program.rootOrder.length; root++) {
        const lookup = program.lookups[program.rootOrder[root]!]!;
        const descriptor = source[lookup.table === "GSUB" ? 4 : 5]! + lookup.lookupIndex * 8;
        if (lookup.table !== "GPOS" || source[descriptor + 2] !== 3) {
            continue;
        }
        const directory = words.length;
        words[root] = origin + directory;
        words.length += program.plan.font.glyphCount * 2;
        words.fill(0, directory);
        const originalDirectory = source[descriptor + 6]!;
        const ruleOrigin = source[descriptor + 3]!;
        const vectors = new Map<number, readonly [number, number]>();
        for (let glyph = 0; glyph < program.plan.font.glyphCount; glyph++) {
            const entry = originalDirectory + glyph * 2;
            const count = source[entry + 1]!;
            if (count === 0 || originalDirectory === 0) {
                continue;
            }
            const candidates = source[entry]!;
            let vector = vectors.get(candidates);
            if (!vector) {
                const ops: number[] = [];
                for (let candidate = 0; candidate < count; candidate++) {
                    const rule = ruleOrigin + source[candidates + candidate]! * 16;
                    const row = source[rule + 10]!;
                    if (row !== 0) {
                        let compact = rows.get(row);
                        const length = source[rule + 12]!;
                        if (compact === undefined) {
                            compact = origin + words.length;
                            for (let cls = 0; cls < length; cls++) {
                                const selected = source[row + cls]!;
                                words.push(...(selected === 0 ? [0, 0] : cell(ruleOrigin + (selected - 1) * 16)));
                            }
                            rows.set(row, compact);
                        }
                        ops.push(1, source[rule + 11]!, length, compact, 0);
                        candidate += source[rule + 13]! - 1;
                    } else {
                        const input = source[rule]!;
                        const predicate = input + ((source[input]! & 0x80000000) !== 0 ? 3 : 1 + source[input]!);
                        const [delta, status] = cell(rule);
                        ops.push(0, predicate, 0, delta, status);
                    }
                }
                vector = [origin + words.length, ops.length / 5];
                for (const word of ops) {
                    words.push(word);
                }
                vectors.set(candidates, vector);
            }
            words[directory + glyph * 2] = vector[0];
            words[directory + glyph * 2 + 1] = vector[1];
        }
    }
    return Uint32Array.from(words);
}
