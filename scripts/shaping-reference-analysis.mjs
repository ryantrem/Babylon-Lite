/** Independent impact analysis: text-shaper tables only, never a Lite import.
 * Trigger tests are conservative; nominal and final glyph streams are inspected.
 * A trigger is not a claim that the final output necessarily changes. */
import { getOrCreateShapePlan } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

const has = (coverage, glyph) => coverage?.covers(glyph) ?? false;
const glyphClass = (font, glyph) => font.gdef?.glyphClassDef.get(glyph) ?? 0;
const firstCoverage = (subtable) => subtable.coverage ?? subtable.coverages?.[0] ?? subtable.inputCoverages?.[0] ?? subtable.markCoverage ?? subtable.mark1Coverage;

function skip(font, lookup, glyph) {
    const cls = glyphClass(font, glyph);
    if ((cls === 1 && lookup.flag & 2) || (cls === 2 && lookup.flag & 4) || (cls === 3 && lookup.flag & 8)) {
        return true;
    }
    if (cls === 3 && lookup.flag & 0x10) {
        return !font.gdef?.markGlyphSets?.has(lookup.markFilteringSet, glyph);
    }
    return cls === 3 && (lookup.flag & 0xff00) !== 0 && font.gdef?.markAttachClassDef.get(glyph) !== lookup.flag >>> 8;
}

function contextCandidates(subtable, glyph) {
    const records = (input, lookupRecords) => ({ input, lookupRecords });
    if (subtable.format === 3) {
        return [
            records(
                (subtable.inputCoverages ?? subtable.coverages).map((coverage) => (g) => has(coverage, g)),
                subtable.lookupRecords
            ),
        ];
    }
    if (!has(subtable.coverage, glyph)) {
        return [];
    }
    if (subtable.format === 1) {
        const rules = (subtable.chainRuleSets ?? subtable.ruleSets)[subtable.coverage.get(glyph)] ?? [];
        return rules.map((rule) => records([(g) => g === glyph, ...rule.inputSequence.map((id) => (g) => g === id)], rule.lookupRecords));
    }
    if (subtable.format === 2) {
        const classes = subtable.inputClassDef ?? subtable.classDef;
        const rules = (subtable.chainClassRuleSets ?? subtable.classRuleSets)[classes.get(glyph)] ?? [];
        return rules.map((rule) =>
            records(
                [(g) => has(subtable.coverage, g) && classes.get(g) === classes.get(glyph), ...rule.inputClasses.map((cls) => (g) => classes.get(g) === cls)],
                rule.lookupRecords
            )
        );
    }
    throw new Error(`Unrecognized independent context format: ${subtable.format}`);
}

function matchInput(font, lookup, sequence, start, predicates) {
    const positions = [];
    let cursor = start;
    for (const predicate of predicates) {
        while (cursor < sequence.length && skip(font, lookup, sequence[cursor])) {
            cursor++;
        }
        if (cursor >= sequence.length || !predicate(sequence[cursor])) {
            return null;
        }
        positions.push(cursor++);
    }
    return positions;
}

function pairValue(subtable, first, second) {
    if (!has(subtable.coverage, first)) {
        return null;
    }
    if (subtable.format === 1) {
        return subtable.pairSets[subtable.coverage.get(first)]?.pairValueRecords.find((record) => record.secondGlyph === second);
    }
    return subtable.class1Records[subtable.classDef1.get(first)]?.class2Records[subtable.classDef2.get(second)];
}
function extraPairValues(value) {
    return value && (value.value1.xPlacement || value.value1.yPlacement || value.value1.yAdvance || Object.values(value.value2).some(Boolean));
}

export function analyzeImpact(font, entry, oldRecords) {
    const plan = getOrCreateShapePlan(font, entry.script, null, entry.direction);
    const logical = entry.direction === "rtl" ? oldRecords.slice().reverse() : oldRecords;
    const finalIds = logical.map((record) => record[0]);
    const nominal = [...entry.text].map((cp) => font.glyphId(cp.codePointAt(0)));
    const sequences = [nominal, finalIds];
    const allIds = new Set(sequences.flat());
    const flags = new Set();
    const evidence = [];
    const parents = new Map();
    function flag(correction, table, index, detail) {
        flags.add(correction);
        evidence.push({ correction, table, lookup: index, detail });
    }
    for (const table of ["GSUB", "GPOS"]) {
        for (const { index, lookup } of table === "GSUB" ? plan.gsubLookups : plan.gposLookups) {
            const intersects = lookup.subtables.some((subtable) => [...allIds].some((glyph) => has(firstCoverage(subtable), glyph)));
            if (!intersects) {
                continue;
            }
            if (lookup.flag & 0x10) {
                flag("C2", table, index, "Filtering-set lookup coverage intersects the nominal/final stream.");
            }
            if (table === "GSUB" && lookup.type === 5 && lookup.subtables.every((subtable) => subtable.format === 3)) {
                flag("C5", table, index, "All-format-3 Context coverage intersects the stream.");
            }
            if (table === "GPOS" && (lookup.type === 7 || lookup.type === 8)) {
                flag("C6", table, index, "Contextual GPOS coverage intersects the stream.");
            }
            if ((table === "GSUB" && [5, 6].includes(lookup.type)) || (table === "GPOS" && [7, 8].includes(lookup.type))) {
                for (const sequence of sequences) {
                    for (let start = 0; start < sequence.length; start++) {
                        for (const candidate of lookup.subtables.flatMap((subtable) => contextCandidates(subtable, sequence[start]))) {
                            const positions = matchInput(font, lookup, sequence, start, candidate.input);
                            if (!positions) {
                                continue;
                            }
                            for (const record of candidate.lookupRecords) {
                                const child = (table === "GSUB" ? font.gsub : font.gpos).lookups[record.lookupListIndex];
                                if (child.type !== 1 || positions[record.sequenceIndex] !== start + record.sequenceIndex) {
                                    flag("C4", table, index, `Nested lookup ${record.lookupListIndex} type ${child.type}; matched positions ${positions.join(",")}.`);
                                }
                            }
                            // Context dependencies are part of the selected plan even when not roots.
                            const roots = new Set(plan.gposLookups.map((entry) => entry.index));
                            const queue = [...roots];
                            const visited = new Set(queue);
                            for (let cursor = 0; cursor < queue.length; cursor++) {
                                const index = queue[cursor];
                                const lookup = font.gpos.lookups[index];
                                if (lookup.type === 7 || lookup.type === 8) {
                                    for (const subtable of lookup.subtables) {
                                        const records = [
                                            ...(subtable.lookupRecords ?? []),
                                            ...[subtable.ruleSets, subtable.chainRuleSets, subtable.classRuleSets, subtable.chainClassRuleSets]
                                                .filter(Boolean)
                                                .flatMap((sets) => sets.flatMap((rules) => (rules ?? []).flatMap((rule) => rule.lookupRecords))),
                                        ];
                                        for (const record of records) {
                                            if (!visited.has(record.lookupListIndex)) {
                                                visited.add(record.lookupListIndex);
                                                queue.push(record.lookupListIndex);
                                            }
                                        }
                                    }
                                } else if (lookup.type === 1 && !roots.has(index)) {
                                    for (const subtable of lookup.subtables) {
                                        for (const glyph of allIds) {
                                            if (!subtable.coverage.covers(glyph)) {
                                                continue;
                                            }
                                            const value = subtable.value ?? subtable.values[subtable.coverage.get(glyph)];
                                            if ((value.yAdvance ?? 0) !== 0) {
                                                flag("C3", "GPOS", index, "Nested-only SinglePos yAdvance must be ignored in horizontal layout.");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (table === "GSUB" && lookup.type === 4 && lookup.flag & 0xff0e) {
                for (const sequence of sequences) {
                    for (let start = 0; start < sequence.length; start++) {
                        for (const subtable of lookup.subtables) {
                            if (!has(subtable.coverage, sequence[start])) {
                                continue;
                            }
                            for (const lig of subtable.ligatureSets[subtable.coverage.get(sequence[start])].ligatures) {
                                const matched = matchInput(
                                    font,
                                    lookup,
                                    sequence,
                                    start,
                                    [sequence[start], ...lig.componentGlyphIds].map((id) => (g) => g === id)
                                );
                                if (matched && matched.at(-1) - start >= matched.length) {
                                    flag("C7", table, index, "A skipped glyph lies inside a matching ligature span.");
                                }
                            }
                        }
                    }
                }
            }
            if (table === "GPOS" && lookup.type === 2) {
                for (const sequence of sequences) {
                    const filtered = sequence.filter((glyph) => !skip(font, lookup, glyph));
                    for (let i = 0; i + 1 < filtered.length; i++) {
                        if (lookup.subtables.some((subtable) => extraPairValues(pairValue(subtable, filtered[i], filtered[i + 1])))) {
                            flag("C3", table, index, `Pair ${filtered[i]},${filtered[i + 1]} has non-advance-only values.`);
                        }
                    }
                    if (
                        table === "GPOS" &&
                        lookup.type === 1 &&
                        lookup.subtables.some((subtable) => (subtable.value?.yAdvance ?? 0) !== 0 || subtable.values?.some((value) => (value.yAdvance ?? 0) !== 0))
                    ) {
                        flag("C3", table, index, "SinglePos yAdvance must be ignored in horizontal layout.");
                    }
                }
            }
            if (table !== "GPOS" || ![4, 5, 6].includes(lookup.type)) {
                continue;
            }
            for (const subtable of lookup.subtables) {
                if (lookup.type === 5 || (lookup.type === 6 && [...allIds].some((glyph) => glyphClass(font, glyph) === 2))) {
                    flag("C8", table, index, "MarkLig or MarkMark with a ligature intersects the stream.");
                }
                for (let at = 0; at < logical.length; at++) {
                    const glyph = finalIds[at];
                    const markCoverage = subtable.markCoverage ?? subtable.mark1Coverage;
                    if (glyphClass(font, glyph) !== 3 || !has(markCoverage, glyph)) {
                        continue;
                    }
                    let target = at - 1;
                    if (lookup.type !== 6) {
                        while (target >= 0 && glyphClass(font, finalIds[target]) === 3) {
                            target--;
                        }
                    }
                    if (target < 0) {
                        continue;
                    }
                    const targetCoverage = subtable.baseCoverage ?? subtable.ligatureCoverage ?? subtable.mark2Coverage;
                    if (!has(targetCoverage, finalIds[target]) || (lookup.type === 6 && glyphClass(font, finalIds[target]) !== 3)) {
                        continue;
                    }
                    const mark = (subtable.markArray ?? subtable.mark1Array).markRecords[markCoverage.get(glyph)];
                    const targetIndex = targetCoverage.get(finalIds[target]);
                    const targetAnchors =
                        lookup.type === 4 ? subtable.baseArray[targetIndex].baseAnchors : lookup.type === 6 ? subtable.mark2Array[targetIndex].mark2Anchors : null;
                    if (targetAnchors && !targetAnchors[mark.markClass]) {
                        continue;
                    }
                    parents.set(at, target);
                    flag("C1", table, index, `Output logical glyph ${at} attaches to ${target}.`);
                }
            }
        }
    }
    const arithmetic = logical.map((record) => record.slice());
    for (const [at, target] of parents) {
        let root = target;
        while (parents.has(root)) {
            root = parents.get(root);
        }
        const first = entry.direction === "rtl" ? root + 1 : root;
        const last = entry.direction === "rtl" ? at + 1 : at;
        const sign = entry.direction === "rtl" ? 1 : -1;
        for (let i = first; i < last; i++) {
            arithmetic[at][4] += sign * logical[i][2];
            arithmetic[at][5] += sign * logical[i][3];
        }
    }
    return {
        flags: [...flags].sort(),
        evidence: [...new Map(evidence.map((item) => [JSON.stringify(item), item])).values()],
        arithmeticRecords: entry.direction === "rtl" ? arithmetic.reverse() : arithmetic,
        attachmentParentsLogical: [...parents],
    };
}

/** Complete table scan, including lookups reachable only as context dependencies.
 * Explicitly counted rather than hidden by a matched-coverage fast path. */
export function scanPositionValues(font) {
    const counts = { singleRecords: 0, pairRecords: 0, nonzeroSingleYAdvance: 0, nonzeroPairYAdvance: 0, nonzeroPairXAdvance: 0 };
    for (const lookup of font.gpos?.lookups ?? []) {
        for (const subtable of lookup.subtables) {
            if (lookup.type === 1) {
                for (const value of subtable.value ? [subtable.value] : (subtable.values ?? [])) {
                    counts.singleRecords++;
                    counts.nonzeroSingleYAdvance += Number((value.yAdvance ?? 0) !== 0);
                }
            } else if (lookup.type === 2) {
                const cells = subtable.format === 1 ? subtable.pairSets.flatMap((set) => set.pairValueRecords) : subtable.class1Records.flatMap((row) => row.class2Records);
                for (const cell of cells) {
                    for (const value of [cell.value1, cell.value2]) {
                        counts.pairRecords++;
                        counts.nonzeroPairYAdvance += Number((value.yAdvance ?? 0) !== 0);
                        counts.nonzeroPairXAdvance += Number((value.xAdvance ?? 0) !== 0);
                    }
                }
            }
        }
    }
    return counts;
}
