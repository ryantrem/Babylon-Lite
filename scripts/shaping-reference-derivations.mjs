/** Test-only, independently specified reference configurations and derivations.
 * No Lite import, runtime switch or output-fitted Arabic form selection. */
import assert from "node:assert/strict";
import { getCombiningClass, getOrCreateShapePlan } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

const tagString = (tag) => String.fromCharCode(tag >>> 24, (tag >>> 16) & 255, (tag >>> 8) & 255, tag & 255);
const tagNumber = (tag) => [...tag].reduce((n, c) => n * 256 + c.charCodeAt(0), 0);
const joiningTags = ["isol", "init", "medi", "fina"];
const selectedGsub = ["ccmp", "locl", "rlig", "rclt", "calt", "liga"];
const selectedGpos = ["kern", "mark", "mkmk"];

function languageFeatures(table, script) {
    if (!table) {
        return [];
    }
    const scripts = table.scriptList.scripts;
    const selected = [script, "DFLT", "latn"].map((name) => scripts.find((entry) => entry.scriptTag === tagNumber(name))?.script).find(Boolean);
    const lang = selected?.defaultLangSys;
    if (!lang) {
        return [];
    }
    return [...new Set([...(lang.requiredFeatureIndex === 0xffff ? [] : [lang.requiredFeatureIndex]), ...lang.featureIndices])].map((index) => table.featureList.features[index]);
}

/** HarfBuzz 11.0.1 hb-ot-shape.cc collect_features, plus Arabic collect_features.
 * Include directional, automatic-fraction, random and private default features;
 * these are explicitly disabled unless requested by the existing profile.
 * Tags not present in the selected LangSys cannot affect these font runs. */
export function featureConfiguration(font, script, direction) {
    const defaults = [
        "rvrn",
        "frac",
        "numr",
        "dnom",
        "rand",
        "Harf",
        "HARF",
        "Buzz",
        "BUZZ",
        "abvm",
        "blwm",
        "ccmp",
        "locl",
        "mark",
        "mkmk",
        "rlig",
        "calt",
        "clig",
        "curs",
        "dist",
        "kern",
        "liga",
        "rclt",
        ...(direction === "rtl" ? ["rtla", "rtlm"] : ["ltra", "ltrm"]),
        ...(script === "arab" ? ["stch", "isol", "fina", "fin2", "fin3", "medi", "med2", "init", "mset"] : []),
    ];
    const requested = [...selectedGsub, ...selectedGpos, ...(script === "arab" ? joiningTags : [])];
    const disabled = [...new Set(defaults)].filter((tag) => !requested.includes(tag)).sort();
    const available = [...languageFeatures(font.gsub, script), ...languageFeatures(font.gpos, script)].map((entry) => tagString(entry.featureTag));
    return {
        requested: requested.slice().sort(),
        requestedInFont: [...new Set(available.filter((tag) => requested.includes(tag)))].sort(),
        harfbuzzDefaults: [...new Set(defaults)].sort(),
        disabled,
        features: Object.fromEntries(disabled.map((tag) => [tag, 0])),
        source: "HarfBuzz 11.0.1 hb-ot-shape.cc::hb_ot_shape_collect_features; hb-ot-shaper-arabic.cc::collect_features_arabic; text-shaper 0.1.28 CB/PB/aE",
    };
}

/** Independent data from HarfBuzz 11.0.1 hb-unicode.hh CCC10..CCC35 macros.
 * Canonical classes themselves come from the pinned text-shaper Unicode data. */
const modifiedClasses = {
    10: 22,
    11: 15,
    12: 16,
    13: 17,
    14: 23,
    15: 18,
    16: 19,
    17: 20,
    18: 21,
    19: 14,
    20: 24,
    21: 12,
    22: 25,
    23: 13,
    24: 10,
    25: 11,
    26: 26,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 33,
    33: 27,
    34: 34,
    35: 35,
};
export function modifiedClass(scalar) {
    const canonical = getCombiningClass(scalar);
    return modifiedClasses[canonical] ?? canonical;
}

/** Lead G1/G2 criterion, with a per-criterion explanation for every row. */
export function normalizationEvidence(font, text, script) {
    const scalars = [...text];
    const singleCharacterClusters = scalars.every((cp, i) => cp === cp.normalize("NFD") || (font.glyphId(cp.codePointAt(0)) !== 0 && !/\p{M}/u.test(scalars[i + 1] ?? "")));
    const nfc = text.normalize("NFC");
    const noMappedNewComposition = [...nfc].every((cp) => scalars.includes(cp) || font.glyphId(cp.codePointAt(0)) === 0);
    const noCanonicalReordering = text.normalize("NFD") === scalars.map((cp) => cp.normalize("NFD")).join("");
    let last = 0;
    const usesModifiedClasses = script === "hebr" || script === "arab";
    const classEntries = [];
    let noModifiedReordering = true;
    for (const cp of scalars) {
        const scalar = cp.codePointAt(0);
        const canonical = getCombiningClass(scalar);
        const cls = usesModifiedClasses ? modifiedClass(scalar) : canonical;
        if (cls !== 0 && last > cls) {
            noModifiedReordering = false;
        }
        if (canonical !== 0) {
            classEntries.push({ scalar, canonical, modified: cls });
        }
        last = cls;
    }
    return {
        inert: singleCharacterClusters && noMappedNewComposition && noCanonicalReordering && noModifiedReordering,
        singleCharacterClusters,
        noMappedNewComposition,
        noCanonicalReordering,
        noModifiedReordering,
        usesModifiedClasses,
        classEntries,
        source: "Phase 1 Lead review G1/G2; modified classes: HarfBuzz 11.0.1 src/hb-unicode.hh CCC10..CCC35",
    };
}

/** Lead G3: stable mark-order mapping, guarded against MarkMark interactions.
 * All six fields are asserted: old order/clusters, reference positions/advances.
 * No offset is taken from old output and no field is silently left unasserted. */
export function markOrderKeyed(font, entry, oldRecords, referenceRecords) {
    if (oldRecords.length !== referenceRecords.length) {
        return { ok: false, reason: "glyph count differs" };
    }
    const old = entry.direction === "rtl" ? oldRecords.slice().reverse() : oldRecords;
    const reference = entry.direction === "rtl" ? referenceRecords.slice().reverse() : referenceRecords;
    const scalars = [...entry.text];
    const permutation = old.map((_, i) => i);
    const parents = old.map((_, i) => i);
    const find = (i) => (parents[i] === i ? i : (parents[i] = find(parents[i])));
    const merge = (a, b) => {
        parents[find(b)] = find(a);
    };
    const reordered = [];
    for (let start = 0; start < old.length;) {
        if (!/\p{M}/u.test(scalars[old[start][1]] ?? "")) {
            start++;
            continue;
        }
        let end = start + 1;
        while (end < old.length && /\p{M}/u.test(scalars[old[end][1]] ?? "")) {
            end++;
        }
        const run = permutation.slice(start, end).sort((a, b) => modifiedClass(scalars[old[a][1]].codePointAt(0)) - modifiedClass(scalars[old[b][1]].codePointAt(0)));
        const changed = run.some((index, j) => index !== start + j);
        if (changed) {
            reordered.push({ start, end, original: permutation.slice(start, end), sorted: run });
            for (let j = 0; j < run.length; j++) {
                const lo = Math.min(start + j, run[j]);
                const hi = Math.max(start + j, run[j]);
                for (let i = lo; i < hi; i++) {
                    merge(i, i + 1);
                }
            }
        }
        permutation.splice(start, end - start, ...run);
        start = end;
    }
    const marks = new Map();
    for (const run of reordered) {
        for (const i of run.original) {
            const glyph = old[i][0];
            marks.set(glyph, (marks.get(glyph) ?? 0) + 1);
        }
    }
    const plan = getOrCreateShapePlan(font, entry.script, null, entry.direction);
    for (const { index, lookup } of plan.gposLookups) {
        if (lookup.type !== 6) {
            continue;
        }
        for (const table of lookup.subtables) {
            for (const [a, count] of marks) {
                for (const b of marks.keys()) {
                    if (a === b && count < 2) {
                        continue;
                    }
                    if (table.mark1Coverage.covers(a) && table.mark2Coverage.covers(b)) {
                        return { ok: false, reason: `MarkMark guard: lookup ${index} covers reordered marks ${a}/${b}` };
                    }
                }
            }
        }
    }
    const minima = new Map();
    old.forEach((record, i) => minima.set(find(i), Math.min(minima.get(find(i)) ?? Infinity, record[1])));
    for (let i = 0; i < old.length; i++) {
        const original = permutation[i];
        if (reference[i][0] !== old[original][0]) {
            return { ok: false, reason: `glyph sequence is not the modified-class permutation at logical index ${i}` };
        }
        if (reference[i][1] !== minima.get(find(original))) {
            return { ok: false, reason: `clusters are not the mark-reordering merge at logical index ${i}` };
        }
    }
    const expected = old.map((record) => record.slice(0, 2));
    permutation.forEach((original, i) => expected[original].push(...reference[i].slice(2)));
    assert.ok(expected.every((record) => record.length === 6));
    return {
        ok: true,
        records: entry.direction === "rtl" ? expected.reverse() : expected,
        reordered: reordered.length > 0,
        permutation,
        reorderedRuns: reordered,
        markMarkGuard: "passed",
        source: "Phase 1 Lead review G3; HarfBuzz 11.0.1 src/hb-unicode.hh modified combining-class entries",
    };
}
