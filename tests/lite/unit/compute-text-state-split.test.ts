import { describe, expect, it } from "vitest";
import { computeTextVariantTier, kernelSource } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import type { ComputeTextKernelVariant } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-shape";
import { computeTextInputPredicates } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-predicates";
import { ARABIC_KERNEL_FRAGMENTS } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-arabic-wgsl";
import { INDIC_KERNEL_FRAGMENTS } from "../../../packages/babylon-lite/src/text/_gpu/compute-text-indic-wgsl";

/** Test-only function walker. Production generation never parses emitted WGSL. */
function functions(source: string): Map<string, string> {
    const result = new Map<string, string>();
    for (const match of source.matchAll(/\bfn (\w+)\(/g)) {
        const start = match.index;
        const open = source.indexOf("{", start);
        let end = open + 1;
        let depth = 1;
        while (depth && end < source.length) {
            if (source[end] === "{") {
                depth++;
            } else if (source[end] === "}") {
                depth--;
            }
            end++;
        }
        expect(depth).toBe(0);
        expect(result.has(match[1]!)).toBe(false);
        result.set(match[1]!, source.slice(start, end));
    }
    return result;
}

function callers(fns: Map<string, string>, target: string): string[] {
    return [...fns]
        .filter(([, text]) => new RegExp(`\\b${target}\\(`).test(text.slice(text.indexOf("{"))))
        .map(([name]) => name)
        .sort();
}

function closure(fns: Map<string, string>, seeds: string[]): string {
    const seen = new Set<string>();
    const pending = seeds.slice();
    while (pending.length) {
        const name = pending.pop()!;
        if (seen.has(name)) {
            continue;
        }
        const text = fns.get(name);
        expect(text, name).toBeDefined();
        seen.add(name);
        for (const call of text!.slice(text!.indexOf("{")).matchAll(/\b(\w+)\(/g)) {
            if (fns.has(call[1]!) && !seen.has(call[1]!)) {
                pending.push(call[1]!);
            }
        }
    }
    return [...seen].map((name) => fns.get(name)).join("\n");
}

const tiers = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const generalEntry = `if (lane == 0u) {
let generalPresence = activeRoots(inputOrigin, scalarCount);
nextLigature = 0u;
}
workgroupBarrier();`;
const state = ["rootPresence", "nextLigature", "matchPositions", "recordMatches"];

describe("cached fast-path state isolation", () => {
    it.each(tiers)("keeps recording and all-word presence on the general path at tier %i", (tier) => {
        const source = kernelSource(tier);
        const fns = functions(source);
        const main = fns.get("main")!;
        const probe = fns.get("activeRootsProbe");
        expect(probe).toBeDefined();
        expect(probe).not.toMatch(/rootPresence/);
        expect(probe).toContain("if (nominalSource) { nominalMasks[index] = first; }");
        expect(fns.get("activeRoots")).toContain("for (var word = 0u; word < width; word++)");
        expect(fns.get("activeRoots")).toContain("rootPresence[word] = bits & witnesses;");
        expect(fns.get("activeRoots")).not.toContain("Barrier(");
        const fastReturn = main.indexOf("if (simple.y != 0u)");
        const entry = main.indexOf(generalEntry);
        expect(entry).toBeGreaterThan(fastReturn);
        expect(main.indexOf(generalEntry, entry + 1)).toBe(-1);
        expect(entry).toBeLessThan(main.indexOf("nominalSource = false;"));
        expect(main.slice(0, fastReturn)).toContain("let present = activeRootsProbe(inputOrigin, length);");
        expect(main.slice(0, fastReturn)).not.toContain("nextLigature =");
        expect(main.slice(0, fastReturn)).toContain("let simple = workgroupUniformLoad(&simpleRoots);");
        expect(callers(fns, "activeRootsProbe")).toEqual(["main"]);
        expect(callers(fns, "activeRoots")).toEqual(["main", "refreshRun"]);
        expect(callers(fns, "matchInputProbe")).toEqual(["proveSimpleRun"]);
        expect(callers(fns, "matchInput")).toEqual(["applySelected", "selectCandidate"]);
        // Serial refresh ignores the return and depends on the ORIGINAL helper's stores.
        expect(fns.get("refreshRun")).toContain("if ((flags & 2u) != 0u) { let present = activeRoots(base, length); }");
        expect(fns.get("refreshRun")).not.toContain("activeRootsProbe");
        expect(fns.get("refreshRun")).toContain("atomicOr(&presenceBits[word], bits)");
        const fast = closure(fns, ["activeRootsProbe", "proveSimpleRun", "finishSimpleRun", "finishSimpleRunLean"]);
        const full = closure(fns, ["main"]);
        for (const name of state) {
            expect(fast, name).not.toMatch(new RegExp(`\\b${name}\\b`));
            expect(full, name).toMatch(new RegExp(`\\b${name}\\b`));
        }
    });

    const variants: (ComputeTextKernelVariant | undefined)[] = [
        undefined,
        { id: "arabic", glyphWords: 8, planes: 1, masked: true, fragments: ARABIC_KERNEL_FRAGMENTS },
        { id: "indic", glyphWords: 16, planes: 2, masked: true, fragments: INDIC_KERNEL_FRAGMENTS, auxBytes: (capacity) => capacity * 16 },
    ];
    for (const variant of variants) {
        for (const tier of [0, ...tiers]) {
            if (tier && (!variant || computeTextVariantTier(tier, variant, 16384) !== tier)) {
                continue;
            }
            it(`retains the old path for ${variant?.id ?? "default"}/${tier}`, () => {
                const source = kernelSource(tier, variant);
                const main = functions(source).get("main")!;
                expect(source).not.toMatch(/activeRootsProbe|matchInputProbe|generalPresence/);
                expect(main.slice(0, main.indexOf("let scalarCount"))).toContain("nextLigature = 0u;");
                expect(main).toContain("activeRoots(");
                if (!variant) {
                    expect(main).toContain("let present = activeRoots(inputOrigin, length);");
                }
            });
        }
    }

    it.each([false, true])("shares the exact recording matcher with probe omissions (masked=%s)", (masked) => {
        for (const gates of [false, true]) {
            const fns = functions(computeTextInputPredicates(masked, gates, true));
            const expected = fns
                .get("matchInput")!
                .replace("fn matchInput(", "fn matchInputProbe(")
                .replace("if (recordMatches && inputCount > 0u) { matchPositions[0u] = index; }\n", "")
                .replace("if (recordMatches) { matchPositions[step] = cursor; }\n", "");
            expect(fns.get("matchInputProbe")).toBe(expected);
            expect(fns.get("matchInput")).toBe(functions(computeTextInputPredicates(masked, gates)).get("matchInput"));
        }
    });
});
