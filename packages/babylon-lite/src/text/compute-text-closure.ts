/** Conservative glyph closure for one compute-text object.

 *  Seeds are cmap glyphs for the input repertoire. Selected GSUB roots and nested
 *  lookups then contribute every output they can produce from the growing set,
 *  including nested Single/Multiple/Ligature/Alternate outputs. Outline components
 *  are added afterwards. The caller installs those ids before dispatch and must
 *  not discover ids by reading shaped output. */

import type { ComputeFont } from "./compute-font.js";
import type { ComputeTextPlan } from "./compute-text-plan.js";
import { parseLookupActions } from "./font-parser/layout-actions.js";
import type { ParsedSubtable } from "./font-parser/layout-actions.js";
import { readComputeGlyfGlyph } from "./compute-glyph-extraction.js";
import { coverageIndex } from "./font-parser/opentype-layout.js";

export interface ComputeTextClosure {
    readonly glyphIds: readonly number[];
}

let outputsByFont: WeakMap<ComputeFont, Map<string, Map<number, readonly number[]>>> | null = null;

/** Glyph ids that must be installed before a resident shape/pack dispatch. */
export function closeComputeTextGlyphs(font: ComputeFont, plans: readonly ComputeTextPlan[], seeds: readonly number[]): ComputeTextClosure {
    const selected = new Set<number>();
    for (const plan of plans) {
        if (plan.font !== font) {
            throw new Error("updateComputeTextData: closure plan belongs to a different font");
        }
        for (const index of plan._gsubRoots) {
            selected.add(index);
        }
        for (const index of plan._gsubDependencies) {
            selected.add(index);
        }
    }
    const lookups = new Map<number, readonly ParsedSubtable[]>();
    for (const index of selected) {
        try {
            lookups.set(index, parseLookupActions(font, "GSUB", index).subtables);
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new Error(`updateComputeTextData: closure cannot see GSUB lookup ${index} (${message})`, { cause: error });
        }
    }
    let bySelection = outputsByFont?.get(font);
    if (!bySelection) {
        bySelection = new Map();
        (outputsByFont ??= new WeakMap()).set(font, bySelection);
    }
    // The selected set's insertion order is observable in the returned closure.
    const key = [...selected].join(",");
    let outputs = bySelection.get(key);
    if (!outputs) {
        outputs = new Map();
        bySelection.set(key, outputs);
    }

    const closure = new Set<number>();
    const queue: number[] = [];
    for (const seed of seeds) {
        requireGlyph(font, seed, "cmap seed");
        if (!closure.has(seed)) {
            closure.add(seed);
            queue.push(seed);
        }
    }
    let steps = 0;
    const stepLimit = font.glyphCount * Math.max(1, selected.size) + 1;
    while (queue.length > 0) {
        if (steps++ > stepLimit) {
            throw new Error("updateComputeTextData: closure did not converge");
        }
        const glyphId = queue.pop() ?? 0;
        let produced = outputs.get(glyphId);
        if (!produced) {
            const pending: number[] = [];
            for (const [lookupIndex, subtables] of lookups) {
                for (let subtableIndex = 0; subtableIndex < subtables.length; subtableIndex++) {
                    for (const output of outputsOf(font, lookupIndex, subtableIndex, subtables[subtableIndex]!, glyphId)) {
                        pending.push(output);
                    }
                }
            }
            outputs.set(glyphId, (produced = pending));
        }
        for (const output of produced) {
            if (closure.has(output)) {
                continue;
            }
            closure.add(output);
            queue.push(output);
        }
    }

    const withComponents = new Set(closure);
    const components = [...closure];
    while (components.length > 0) {
        const glyphId = components.pop() ?? 0;
        for (const component of outlineComponents(font, glyphId)) {
            if (withComponents.has(component)) {
                continue;
            }
            withComponents.add(component);
            components.push(component);
        }
    }
    return { glyphIds: [...withComponents] };
}

function outputsOf(font: ComputeFont, lookupIndex: number, subtableIndex: number, subtable: ParsedSubtable, glyphId: number): number[] {
    const witness = `GSUB lookup ${lookupIndex} subtable ${subtableIndex}`;
    switch (subtable.kind) {
        case "context":
            return [];
        case "single": {
            const index = coverageIndex(subtable.coverage, glyphId);
            if (index === null) {
                return [];
            }
            if (subtable.format === 1) {
                return [requireGlyph(font, glyphId + subtable.delta, witness)];
            }
            const substitute = subtable.substitutes?.[index];
            if (substitute === undefined) {
                throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
            }
            return [requireGlyph(font, substitute, witness)];
        }
        case "multiple": {
            const index = coverageIndex(subtable.coverage, glyphId);
            if (index === null) {
                return [];
            }
            const sequence = subtable.sequences[index];
            if (!sequence) {
                throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
            }
            return sequence.map((output) => requireGlyph(font, output, witness));
        }
        case "alternate": {
            const index = coverageIndex(subtable.coverage, glyphId);
            if (index === null) {
                return [];
            }
            const alternatives = subtable.alternatives[index];
            if (!alternatives) {
                throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
            }
            return alternatives.map((output) => requireGlyph(font, output, witness));
        }
        case "ligature": {
            const index = coverageIndex(subtable.coverage, glyphId);
            if (index === null) {
                return [];
            }
            const set = subtable.sets[index];
            if (!set) {
                throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
            }
            return set.map((entry) => requireGlyph(font, entry.glyph, witness));
        }
        case "unimplemented": {
            if (subtable.coverage && coverageIndex(subtable.coverage, glyphId) === null) {
                return [];
            }
            throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId} (${subtable.reason})`);
        }
        default:
            throw new Error(`updateComputeTextData: closure cannot see ${witness} output for glyph ${glyphId}`);
    }
}

function outlineComponents(font: ComputeFont, glyphId: number): readonly number[] {
    if (font.outlineFormat !== "truetype") {
        return [];
    }
    try {
        return readComputeGlyfGlyph(font, glyphId).componentGlyphIds;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`updateComputeTextData: closure cannot see outline components of glyph ${glyphId} (${message})`, { cause: error });
    }
}

function requireGlyph(font: ComputeFont, glyphId: number, witness: string): number {
    if (!Number.isInteger(glyphId) || glyphId < 0 || glyphId >= font.glyphCount) {
        throw new Error(`updateComputeTextData: closure cannot see ${witness} output glyph ${glyphId}`);
    }
    return glyphId;
}
