/** R3 independent attachment geometry. Test tooling only; no Lite outputs/imports. */
import assert from "node:assert/strict";
import { getOrCreateShapePlan } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { analyzeImpact } from "./shaping-reference-analysis.mjs";

function absolutePositions(records) {
    let x = 0;
    let y = 0;
    return records.map((record) => {
        const position = [x + record[4], y + record[5]];
        x += record[2];
        y += record[3];
        assert.ok(Number.isSafeInteger(x) && Number.isSafeInteger(y));
        return { pen: [x - record[2], y - record[3]], position };
    });
}

/** Applies only the Lead-approved three legacy MarkMark rows.
 * Expected fields come exclusively from old reference records and font hmtx.
 * HarfBuzz is used for cross-checking, never as a source of fitted expected fields.
 */
export function deriveAttachedAdvances(font, entry, oldRecords, analysis, harfbuzzRecords) {
    assert.equal(entry.font, "synthetic:mark-mark");
    const plan = getOrCreateShapePlan(font, entry.script, null, entry.direction);
    assert.ok(plan.gposLookups.length > 0 && plan.gposLookups.every(({ lookup }) => lookup.type === 6), "A-attach derivation requires the isolated MarkMark fixture");
    assert.equal(plan.gsubLookups.length, 0);
    assert.deepEqual(
        oldRecords.map((record) => record.slice(0, 2)),
        harfbuzzRecords.map((record) => record.slice(0, 2)),
        `${entry.id}: glyph/cluster identity`
    );
    const logical = entry.direction === "rtl" ? oldRecords.slice().reverse() : oldRecords;
    const parents = new Map(analysis.attachmentParentsLogical);
    const derived = logical.map((record, index) => {
        assert.equal(record[2], font.advanceWidth(record[0]), `${entry.id}: old advance must be independently backed by hmtx`);
        assert.equal(record[3], 0);
        return [record[0], record[1], parents.has(index) ? 0 : record[2], parents.has(index) ? 0 : record[3], record[4], record[5]];
    });
    const roots = new Map();
    for (const [index, parent] of parents) {
        assert.ok(parent < index, `${entry.id}: attachment must point backward`);
        let root = parent;
        while (parents.has(root)) {
            assert.ok(parents.get(root) < root, "Acyclic attachment graph");
            root = parents.get(root);
        }
        roots.set(index, root);
        const first = entry.direction === "rtl" ? root + 1 : root;
        const end = entry.direction === "rtl" ? index + 1 : index;
        const sign = entry.direction === "rtl" ? 1 : -1;
        for (let i = first; i < end; i++) {
            derived[index][4] += sign * derived[i][2];
            derived[index][5] += sign * derived[i][3];
        }
    }
    const records = entry.direction === "rtl" ? derived.reverse() : derived;
    const toVisual = (index) => (entry.direction === "rtl" ? records.length - index - 1 : index);
    const positions = absolutePositions(records);
    const hbPositions = absolutePositions(harfbuzzRecords);
    const geometryChecks = [];
    const unattachedAdvanceDifferences = [];
    const downstreamTranslations = [];
    let deltaX = 0;
    let deltaY = 0;
    for (let visual = 0; visual < records.length; visual++) {
        const logicalIndex = entry.direction === "rtl" ? records.length - visual - 1 : visual;
        const record = records[visual];
        const hb = harfbuzzRecords[visual];
        assert.ok(record.length === 6 && record.every(Number.isInteger));
        assert.deepEqual(record.slice(0, 2), hb.slice(0, 2));
        const penTranslation = positions[visual].pen.map((value, axis) => value - hbPositions[visual].pen[axis]);
        assert.deepEqual(penTranslation, [deltaX, deltaY], `${entry.id}: every downstream pen shift must be explained`);
        let absoluteTranslation;
        if (parents.has(logicalIndex)) {
            assert.deepEqual(record.slice(2, 4), [0, 0]);
            assert.deepEqual(hb.slice(2, 4), [0, 0]);
            const root = toVisual(roots.get(logicalIndex));
            const relative = positions[visual].position.map((value, axis) => value - positions[root].position[axis]);
            const hbRelative = hbPositions[visual].position.map((value, axis) => value - hbPositions[root].position[axis]);
            assert.deepEqual(relative, hbRelative, `${entry.id}: attached glyph ${visual} root-relative geometry`);
            geometryChecks.push({ index: visual, root, rootRelative: relative, harfbuzzRootRelative: hbRelative, passed: true });
            absoluteTranslation = positions[root].position.map((value, axis) => value - hbPositions[root].position[axis]);
        } else {
            assert.deepEqual(record.slice(4), hb.slice(4), `${entry.id}: unattached offsets must not differ`);
            absoluteTranslation = [deltaX, deltaY];
            if (record[2] !== hb[2] || record[3] !== hb[3]) {
                const hmtx = font.advanceWidth(record[0]);
                assert.equal(font.gdef.glyphClassDef.get(record[0]), 3);
                assert.notEqual(hmtx, 0);
                assert.deepEqual(record.slice(2, 4), [hmtx, 0]);
                assert.deepEqual(hb.slice(2, 4), [0, 0]);
                unattachedAdvanceDifferences.push({ index: visual, glyph: record[0], hmtx, expectedAdvance: record.slice(2, 4), harfbuzzAdvance: hb.slice(2, 4) });
            }
        }
        assert.deepEqual(
            positions[visual].position.map((value, axis) => value - hbPositions[visual].position[axis]),
            absoluteTranslation,
            `${entry.id}: every absolute-position difference must be the documented root/downstream translation`
        );
        downstreamTranslations.push({ index: visual, penTranslation, absoluteTranslation });
        deltaX += record[2] - hb[2];
        deltaY += record[3] - hb[3];
    }
    return {
        records,
        crossCheck: {
            passed: true,
            attachedGlyphs: geometryChecks.length,
            geometryChecks,
            unattachedAdvanceDifferences,
            downstreamTranslations,
            totalAdvanceDifference: [deltaX, deltaY],
            allSixFieldsDerived: true,
            rule: "Old independent records plus hmtx; zero attached advances, then pen-relative C1 transform; exact root-relative HarfBuzz geometry",
        },
    };
}

/** Rejects prospective H expectations that would adopt GDEF-wide width zeroing.
 * Match/anchor presence is read independently from the reference font/records.
 */
export function unattachedNonzeroMarkGuard(font, entry, records) {
    const logical = entry.direction === "rtl" ? records.slice().reverse() : records;
    const candidates = logical.flatMap((record, index) =>
        font.gdef?.glyphClassDef.get(record[0]) === 3 && font.advanceWidth(record[0]) !== 0
            ? [{ logicalIndex: index, glyph: record[0], hmtx: font.advanceWidth(record[0]), referenceAdvance: record[2] }]
            : []
    );
    if (!candidates.length) {
        return [];
    }
    const parents = new Map(analyzeImpact(font, entry, records).attachmentParentsLogical);
    return candidates.filter((candidate) => !parents.has(candidate.logicalIndex));
}

// These six reviewed IDs are the entire scope; no general H-guard exemption.
const kernel = "tests/lite/compute/kernel.test.ts#";
const ordered = "tests/lite/compute/ordered-parallel.test.ts#keeps failed selections, skipped components, insertion cursors and inherited attachment fields exact";
export const N_PROOF_IDS = [
    `${kernel}actual WGSL mode mark-ligature:synthetic:mark-ligature:2`,
    `${kernel}actual WGSL mode mark-ligature:synthetic:mark-ligature:3`,
    `${kernel}GPOS point-anchor rejection identifies its table and original run after empty-run filtering:synthetic:point-anchor:0`,
];
export const G0_IDS = [0, 1, 2].map((i) => `${ordered}: ligature-ignore-marks:synthetic:ligature-ignore-marks:${i}`);

export function reviewedFontPositions(font, entry, oldRecords, hb) {
    const nominal = [...entry.text].map((cp, i) => [font.glyphId(cp.codePointAt(0)), i]);
    const cls = (glyph) => font.gdef?.glyphClassDef.get(glyph) ?? 0;
    assert.equal(entry.direction, "ltr");
    if (N_PROOF_IDS.includes(entry.id)) {
        assert.ok(["synthetic:mark-ligature", "synthetic:point-anchor"].includes(entry.font));
        for (const lookup of font.gsub?.lookups ?? []) {
            assert.equal(lookup.type, 1);
            for (const subtable of lookup.subtables) {
                assert.equal(subtable.format, 1);
                assert.equal(subtable.deltaGlyphId, 0);
                assert.ok(
                    nominal.every(([glyph]) => !subtable.coverage.covers(glyph)),
                    "No GSUB input coverage intersects this reviewed nominal row"
                );
            }
        }
        assert.equal(font.gpos.lookups.length, 1);
        const lookup = font.gpos.lookups[0];
        assert.equal(lookup.type, 5);
        assert.equal(lookup.flag, 0);
        assert.equal(lookup.subtables.length, 1);
        const subtable = lookup.subtables[0];
        assert.equal(subtable.markCoverage.size, 1);
        assert.ok(subtable.markCoverage.covers(2));
        assert.equal(subtable.ligatureCoverage.size, 1);
        assert.ok(subtable.ligatureCoverage.covers(1));
        const unattached = [];
        nominal.forEach(([glyph], i) => {
            if (!subtable.markCoverage.covers(glyph)) {
                return;
            }
            let previous = i - 1;
            while (previous >= 0 && cls(nominal[previous][0]) === 3) {
                previous--;
            }
            assert.ok(previous < 0 || !subtable.ligatureCoverage.covers(nominal[previous][0]));
            unattached.push({ index: i, nearestNonMark: previous < 0 ? null : nominal[previous][0] });
        });
        const records = nominal.map(([glyph, cluster]) => [glyph, cluster, font.advanceWidth(glyph), 0, 0, 0]);
        assert.deepEqual(oldRecords, records, `${entry.id}: N-proof old records`);
        assert.deepEqual(
            hb,
            records.map((record) => record.map((value, field) => (field === 2 && cls(record[0]) === 3 ? 0 : value))),
            `${entry.id}: exact HB width-only difference`
        );
        return { category: "N-proof", treatment: "N-proof", records, fontPositionCrossCheck: { passed: true, noGsubEffect: true, unattached, allSixFieldsChecked: true } };
    }
    assert.ok(G0_IDS.includes(entry.id), "No other row may use font-intrinsic positions");
    assert.equal(entry.font, "synthetic:ligature-ignore-marks");
    assert.equal(font.gpos, null);
    assert.ok(!font.directory.tables.has(0x47504f53), "No GPOS table, not merely no selected positioning roots");
    assert.equal(font.gsub.lookups.length, 1);
    const lookup = font.gsub.lookups[0];
    assert.equal(lookup.type, 4);
    assert.equal(lookup.flag, 8);
    assert.equal(lookup.subtables.length, 1);
    const subtable = lookup.subtables[0];
    assert.equal(subtable.coverage.size, 1);
    assert.ok(subtable.coverage.covers(1));
    assert.deepEqual(
        subtable.ligatureSets.map((set) => set.ligatures),
        [[{ ligatureGlyph: 3, componentGlyphIds: [3] }]]
    );
    const proved = [];
    const spans = [];
    for (let i = 0; i < nominal.length; i++) {
        let end = i + 1;
        if (nominal[i][0] === 1) {
            while (end < nominal.length && cls(nominal[end][0]) === 3) {
                end++;
            }
        }
        if (nominal[i][0] === 1 && end < nominal.length && nominal[end][0] === 3) {
            proved.push({ glyph: 3, oldCluster: i, cluster: i });
            const skipped = [];
            for (let mark = i + 1; mark < end; mark++) {
                assert.equal(cls(nominal[mark][0]), 3);
                proved.push({ glyph: nominal[mark][0], oldCluster: mark, cluster: i });
                skipped.push(mark);
            }
            spans.push({ start: i, end, minimum: i, skipped });
            i = end;
        } else {
            proved.push({ glyph: nominal[i][0], oldCluster: i, cluster: i });
        }
    }
    assert.deepEqual(
        oldRecords.map((r) => r.slice(0, 2)),
        proved.map((p) => [p.glyph, p.oldCluster]),
        `${entry.id}: independently proved old GSUB stream`
    );
    assert.deepEqual(
        hb.map((r) => r.slice(0, 2)),
        proved.map((p) => [p.glyph, p.cluster]),
        `${entry.id}: HB changes clusters only for skipped marks inside proved C7 spans`
    );
    const records = hb.map(([glyph, cluster]) => [glyph, cluster, font.advanceWidth(glyph), 0, 0, 0]);
    assert.deepEqual(
        oldRecords.map((r) => r.slice(2)),
        records.map((r) => r.slice(2)),
        `${entry.id}: font-intrinsic old positions`
    );
    assert.deepEqual(
        hb,
        records.map((r) => (cls(r[0]) === 3 ? [r[0], r[1], 0, 0, -font.advanceWidth(r[0]), 0] : r)),
        `${entry.id}: exact no-GPOS HB width/offset difference`
    );
    return { category: "G0", treatment: "G0", records, fontPositionCrossCheck: { passed: true, noGposTable: true, spans, allSixFieldsChecked: true } };
}
