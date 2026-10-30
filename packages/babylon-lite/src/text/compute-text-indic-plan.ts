/** Lazy dev2 font/input preparation. No per-text shaping.
 * Follow-up boundaries: docs/lite/architecture/compute-text-follow-ups.md.
 * Feature policy/zero-context font queries follow the OpenType Indic model,
 * with HarfBuzz11.0.1 (Old MIT) as independently tested behavioral evidence.
 */
import type { ComputeFont } from "./compute-font.js";
import { ensureComputeFontLayout } from "./compute-font.js";
import type { ComputeTextRunSelection } from "./compute-text-selection.js";
import type { ComputeTextPlan, ComputeTextStage } from "./compute-text-plan.js";
import { createComputeTextProfilePlan } from "./compute-text-plan.js";
import { compileComputeTextProgram } from "./compute-text-program.js";
import type { ComputeTextProgram } from "./compute-text-program.js";
import { mapCmapScalar } from "./font-parser/cmap.js";
import { findLayoutScript, coverageGlyphs } from "./font-parser/opentype-layout.js";
import type { LayoutTable, LayoutFeature } from "./font-parser/opentype-layout.js";
import { parseLookupActions } from "./font-parser/layout-actions.js";
import { INDIC_DENSE, INDIC_SPARSE, INDIC_DECOMPOSITIONS, INDIC_COMPOSITIONS, INDIC_VOWEL_CONSTRAINTS, INDIC_SYLLABLE_MACHINE } from "./compute-text-indic-unicode.js";

export function indicProperties(cp: number): number {
    if (cp >= 0x900 && cp <= 0x97f) {
        return INDIC_DENSE[cp - 0x900]!;
    }
    let lo = 0,
        hi = INDIC_SPARSE.length / 2;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1,
            value = INDIC_SPARSE[mid * 2]!;
        if (value < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return INDIC_SPARSE[lo * 2] === cp ? INDIC_SPARSE[lo * 2 + 1]! : 0;
}

export function validateIndicRun(selection: ComputeTextRunSelection): void {
    for (let i = 0; i < selection._scalars.length; i++) {
        const cp = selection._scalars[i]!;
        if ((indicProperties(cp) & 0x10000000) === 0) {
            throw new Error(`unsupported scalar in Devanagari run at scalar ${i} U+${cp.toString(16).toUpperCase().padStart(4, "0")}`);
        }
    }
}

const GSUB = ["ccmp", "locl", "nukt", "akhn", "rphf", "rkrf", "blwf", "half", "cjct", "pres", "abvs", "blws", "psts", "haln", "rlig", "calt", "clig", "rclt"];
const GPOS = ["abvm", "blwm", "dist", "mark", "mkmk", "kern"];
const UNVERIFIED = ["pref", "pstf", "vatu", "abvf", "init", "rand", "frac", "numr", "dnom", "rvrn", "ltra", "ltrm", "curs"];

function unsupported(reason: string): never {
    throw new Error(`Devanagari font is unsupported: ${reason}`);
}

/** @internal Only selected, compiled rules/dependencies are inspected. Extension
 * subtables have already been resolved by the common compiler. */
export function validateIndicCompiledProgram(program: Pick<ComputeTextProgram, "lookups">): void {
    for (const lookup of program.lookups) {
        if (lookup.table !== "GSUB") {
            continue;
        }
        for (const rule of lookup.rules) {
            if (rule.action.kind !== "context" || rule.action.lookups.length < 2) {
                continue;
            }
            for (const record of rule.action.lookups) {
                const child = program.lookups.find((entry) => entry.table === "GSUB" && entry.lookupIndex === record.lookupIndex);
                if (child?.rules.some((entry) => entry.action.kind === "ligature")) {
                    throw new Error("createComputeTextPlan: font uses multi-record contextual ligatures not yet supported by the GPU shaper");
                }
            }
        }
    }
}

function features(table: LayoutTable): LayoutFeature[] {
    const lang = findLayoutScript(table, "dev2")?.defaultLangSys;
    if (!lang) {
        return unsupported("dev2 GSUB and GPOS scripts are required");
    }
    const indexes = [...lang.featureIndices, ...(lang.requiredFeatureIndex === 0xffff ? [] : [lang.requiredFeatureIndex])];
    return indexes.map((i) => table.features[i]!);
}

/** Per-font only; the shared plan registry memoizes successes, never failures. */
export function buildIndicPlan(font: ComputeFont): ComputeTextPlan {
    if (font._hasVariableAxes) {
        unsupported("variable fonts are unsupported");
    }
    if (["morx", "kerx", "trak"].some((tag) => font._directory.tables.has(tag))) {
        unsupported("AAT tables are unsupported");
    }
    const layout = ensureComputeFontLayout(font),
        gsub = layout.gsub,
        gpos = layout.gpos;
    if (!gsub || !gpos || !findLayoutScript(gsub, "dev2")?.defaultLangSys || !findLayoutScript(gpos, "dev2")?.defaultLangSys) {
        unsupported("dev2 GSUB and GPOS scripts are required");
    }
    if (!layout.gdef?.glyphClassDef) {
        unsupported("GDEF glyph classes are required");
    }
    const subFeatures = features(gsub),
        posFeatures = features(gpos);
    if (font._directory.tables.has("kern") && !posFeatures.some((f) => f.tag === "kern")) {
        unsupported("legacy kern fallback is unsupported");
    }
    for (const f of [...subFeatures, ...posFeatures]) {
        if (UNVERIFIED.includes(f.tag)) {
            unsupported(`unverified Devanagari feature '${f.tag}'`);
        }
    }
    const nominal = (cp: number): number => mapCmapScalar(font._cmap, cp);
    for (const cp of [0x20, 0x25cc, 0x94d]) {
        if (!nominal(cp)) {
            unsupported(`required glyph U+${cp.toString(16).toUpperCase().padStart(4, "0")} is unmapped`);
        }
    }
    const masks = new Map<string, number>();
    for (const tag of ["rphf", "blwf", "half"]) {
        if (subFeatures.some((f) => f.tag === tag)) {
            masks.set(tag, 1 << masks.size);
        }
    }
    const stages: ComputeTextStage[] = [];
    const manual = ["nukt", "akhn", "rphf", "rkrf", "blwf", "half", "cjct", "pres", "abvs", "blws", "psts", "haln"];
    const add = (table: "GSUB" | "GPOS", tags: readonly string[], pause: ComputeTextStage["pause"]): void => {
        const roots = new Map<number, { lookup: number; mask: number; flags: number }>();
        for (const f of table === "GSUB" ? subFeatures : posFeatures) {
            if (!tags.includes(f.tag)) {
                continue;
            }
            const flags = manual.includes(f.tag) ? 4 : f.tag === "locl" || f.tag === "ccmp" ? 7 : 3;
            for (const lookup of f.lookupListIndices) {
                const mask = masks.get(f.tag) ?? 0x80000000,
                    old = roots.get(lookup);
                roots.set(lookup, old ? { lookup, mask: (old.mask | mask) >>> 0, flags: (old.flags & flags & 3) | ((old.flags | flags) & 4) } : { lookup, mask, flags });
            }
        }
        if (roots.size || pause !== "none") {
            stages.push({ table, roots: [...roots.values()].sort((a, b) => a.lookup - b.lookup), pause });
        }
    };
    stages.push({ table: "GSUB", roots: [], pause: "setupSyllables" });
    add("GSUB", ["locl", "ccmp"], "initialReorder");
    for (const tag of ["nukt", "akhn", "rphf", "rkrf", "blwf", "half", "cjct"]) {
        add("GSUB", [tag], tag === "cjct" ? "finalReorder" : "none");
    }
    if (!subFeatures.some((f) => f.tag === "cjct")) {
        stages.push({ table: "GSUB", roots: [], pause: "finalReorder" });
    }
    add(
        "GSUB",
        GSUB.filter((tag) => !["locl", "ccmp", ...manual.slice(0, 7)].includes(tag)),
        "clearSyllables"
    );
    add("GPOS", GPOS, "none");

    const pairs = (tags: readonly string[], strict: boolean): number[] => {
        const pairs: number[] = [];
        for (const f of subFeatures.filter((f) => tags.includes(f.tag))) {
            for (const index of f.lookupListIndices) {
                for (const sub of parseLookupActions(font, "GSUB", index).subtables) {
                    if (sub.kind === "single" && !strict) {
                        continue;
                    }
                    if (sub.kind !== "ligature") {
                        unsupported(`GSUB lookup ${index} ${sub.kind} cannot be enumerated for ${f.tag}`);
                    }
                    coverageGlyphs(sub.coverage, font.glyphCount).forEach((first, i) => {
                        for (const ligature of sub.sets[i] ?? []) {
                            if (ligature.components.length === 1) {
                                pairs.push(((first << 16) | ligature.components[0]!) >>> 0);
                            }
                        }
                    });
                }
            }
        }
        return [...new Set(pairs)].sort((a, b) => a - b);
    };
    const virama = nominal(0x94d),
        dotted = nominal(0x25cc),
        space = nominal(0x20);
    const below = new Set(pairs(["blwf", "vatu"], false)),
        post = new Set(pairs(["pstf", "pref"], false));
    const positions = Uint32Array.from({ length: font.glyphCount }, (_, glyph) => {
        const a = ((virama << 16) | glyph) >>> 0,
            b = ((glyph << 16) | virama) >>> 0;
        return below.has(a) || below.has(b) ? 1 : post.has(a) || post.has(b) ? 2 : 0;
    });
    const reph = pairs(["rphf"], true);
    const internalScalars = [...new Set([0x20, 0x25cc, 0x94d, ...INDIC_DECOMPOSITIONS, ...INDIC_COMPOSITIONS])].sort((a, b) => a - b);
    const internal = internalScalars.filter((cp) => nominal(cp) !== 0).flatMap((cp) => [cp, nominal(cp)]);
    const closureSeeds = [...new Set(internal.filter((_, i) => i % 2 === 1))];
    const normalizationClosure: [number, number][] = [];
    for (let i = 0; i < INDIC_DECOMPOSITIONS.length; i += 3) {
        const [cp, a, b] = Array.from(INDIC_DECOMPOSITIONS.slice(i, i + 3));
        for (const part of [a!, b!]) {
            if (nominal(cp!) && nominal(part)) {
                normalizationClosure.push([nominal(cp!), nominal(part)]);
            }
        }
    }
    for (let i = 0; i < INDIC_COMPOSITIONS.length; i += 3) {
        const [a, b, cp] = Array.from(INDIC_COMPOSITIONS.slice(i, i + 3));
        for (const part of [a!, b!]) {
            if (nominal(cp!) && nominal(part)) {
                normalizationClosure.push([nominal(part), nominal(cp!)]);
            }
        }
    }
    // Profile section offsets are relative to this 24-word header.
    const words: number[] = new Array<number>(24).fill(0);
    const append = (slot: number, values: Iterable<number>, count: number): void => {
        words[slot] = words.length;
        words[slot + 1] = count;
        words.push(...values);
    };
    append(0, INDIC_DENSE, 128);
    append(2, INDIC_SPARSE, INDIC_SPARSE.length / 2);
    append(4, INDIC_DECOMPOSITIONS, INDIC_DECOMPOSITIONS.length / 3);
    append(6, INDIC_COMPOSITIONS, INDIC_COMPOSITIONS.length / 3);
    append(8, INDIC_VOWEL_CONSTRAINTS, INDIC_VOWEL_CONSTRAINTS.length / 4);
    append(10, internal, internal.length / 2);
    words[12] = words.length;
    words.push(...positions);
    append(13, reph, reph.length);
    words[15] = virama;
    words[16] = dotted;
    words[17] = space;
    words[18] = masks.get("rphf") ?? 0;
    words[19] = masks.get("half") ?? 0;
    words[20] = masks.get("blwf") ?? 0;
    words[21] = words.length;
    words.push(...INDIC_SYLLABLE_MACHINE);
    const packed = Uint32Array.from(words);
    let expansion = 1;
    const plan = createComputeTextProfilePlan(font, "deva", "indic", stages, {
        tables: () => packed,
        closureSeeds,
        normalizationClosure,
        features: [...subFeatures.filter((f) => GSUB.includes(f.tag)), ...posFeatures.filter((f) => GPOS.includes(f.tag))].map((f) => f.tag),
        spaceLocal: (indicProperties(0x20) & 255) === 0 && ((indicProperties(0x20) >>> 16) & 255) === 0 && !INDIC_VOWEL_CONSTRAINTS.includes(0x20),
        capacity: (scalars) => {
            let extra = 0;
            for (const cp of scalars) {
                const props = indicProperties(cp),
                    cat = props & 255;
                if ((props & 0x20000000) !== 0) {
                    extra++;
                }
                if (cp !== 0x20 && ![0, 1, 2, 10, 15, 17, 18].includes(cat)) {
                    extra++;
                }
            }
            return (scalars.length + extra) * expansion;
        },
    });
    const program = compileComputeTextProgram(plan);
    validateIndicCompiledProgram(program);
    if (!program.fullyExecutable) {
        const reason = program.rejections[0]!;
        unsupported(`${reason.table} lookup ${reason.lookupIndex} ${reason.reason} is not executable`);
    }
    expansion = program.maxExpansion;
    return plan;
}
