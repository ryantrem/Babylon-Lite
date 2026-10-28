/** Narrow B1 intermediate-buffer relation, independently validated by HB itself.
 * No Lite imports, no raw snapshot mutation, no general canonical tolerance.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

export const PRE_NUKT_MESSAGES = ["start reordering indic initial", "end reordering indic initial"];
const unicodeSha256 = "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f";

export function indicCompositionEvidence(font) {
    const bytes = readFileSync(resolve(homedir(), ".cache/babylon-lite/unicode/16.0.0/UnicodeData.txt"));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), unicodeSha256);
    const entries = new Map(
        bytes
            .toString()
            .trim()
            .split("\n")
            .map((row) => {
                const columns = row.split(";");
                return [parseInt(columns[0], 16), columns];
            })
    );
    const compositions = [0x929, 0x931, 0x934].map((composite) => {
        const parts = entries
            .get(composite)[5]
            .split(" ")
            .map((part) => parseInt(part, 16));
        assert.equal(parts.length, 2);
        return { scalars: [...parts, composite], glyphs: [...parts, composite].map((cp) => font.glyphId(cp)) };
    });
    assert.deepEqual(
        compositions.map((entry) => entry.scalars),
        [
            [0x928, 0x93c, 0x929],
            [0x930, 0x93c, 0x931],
            [0x933, 0x93c, 0x934],
        ]
    );
    assert.ok(compositions.every((entry) => entry.glyphs.every((glyph) => glyph > 0)));
    const tag = (text) => Buffer.from(text).readUInt32BE();
    const script = font.gsub.scriptList.scripts.find((entry) => entry.scriptTag === tag("dev2"));
    assert.ok(script, "Independent dev2 script missing");
    const features = script.script.defaultLangSys.featureIndices.map((index) => font.gsub.featureList.features[index]);
    const roots = [...new Set(features.filter((entry) => ["locl", "ccmp"].some((name) => entry.featureTag === tag(name))).flatMap((entry) => entry.feature.lookupListIndices))];
    const checks = roots.flatMap((index) =>
        compositions.map((entry) => {
            const lookup = font.gsub.lookups[index];
            const covered = entry.glyphs.map((glyph) =>
                lookup.subtables.some((table) => (table.coverage ?? table.coverages?.[0] ?? table.inputCoverages?.[0])?.covers(glyph) ?? false)
            );
            assert.ok(
                covered.every((value) => value === covered[0]),
                `B1 pre-nukt lookup ${index} distinguishes ${entry.scalars.join("/")}`
            );
            return { lookup: index, scalars: entry.scalars, covered };
        })
    );
    return { unicodeSha256, compositions, preNuktCoverage: { script: "dev2", language: "default", roots, checks, passed: true } };
}

export function canonicalIndicInput(text, compositions) {
    const clusterMap = [],
        scalars = [];
    [...text].forEach((character, cluster) => {
        const cp = character.codePointAt(0);
        const entry = compositions.find((entry) => entry.scalars[2] === cp);
        for (const scalar of entry ? entry.scalars.slice(0, 2) : [cp]) {
            scalars.push(scalar);
            clusterMap.push(cluster);
        }
    });
    return { text: String.fromCodePoint(...scalars), clusterMap };
}

export function projectIndicSnapshot(glyphs, compositions) {
    const projected = [];
    for (let i = 0; i < glyphs.length; i++) {
        const a = glyphs[i],
            b = glyphs[i + 1];
        const entry = b && a[1] === b[1] && compositions.find((entry) => entry.glyphs[0] === a[0] && entry.glyphs[1] === b[0]);
        if (entry) {
            projected.push([entry.glyphs[2], a[1]]);
            i++;
        } else {
            projected.push([...a]);
        }
    }
    return projected;
}

export function validateIndicProjection(original, decomposed, clusterMap, evidence) {
    const renumber = (cluster) => {
        assert.ok(Number.isInteger(cluster) && cluster >= 0 && cluster < clusterMap.length, `Invalid independent scalar cluster ${cluster}`);
        return clusterMap[cluster];
    };
    assert.deepEqual(
        original.records,
        decomposed.records.map((row) => row.map((value, i) => (i === 1 ? renumber(value) : value))),
        `${original.id}: canonical-path exact final count/six-field disagreement`
    );
    const pauses = original.pauseSnapshots.map((raw) => {
        const other = decomposed.pauseSnapshots.find((snapshot) => snapshot.message === raw.message);
        assert.ok(other, `${original.id}: missing independent decomposed-path pause ${raw.message}`);
        const expected = other.glyphs.map(([glyph, cluster]) => [glyph, renumber(cluster)]);
        const projected = PRE_NUKT_MESSAGES.includes(raw.message) ? projectIndicSnapshot(raw.glyphs, evidence.compositions) : raw.glyphs;
        assert.deepEqual(projected, expected, `${original.id}: B1 HB decomposed-path snapshot ${raw.message}`);
        return {
            message: raw.message,
            glyphs: projected,
            rawGlyphs: raw.glyphs,
            decomposedGlyphs: expected,
            relation: PRE_NUKT_MESSAGES.includes(raw.message) ? "three-pair-pre-nukt-projection" : "exact",
        };
    });
    return { id: original.id, clusterMap, pauseSnapshots: pauses, finalSixFieldsExact: true, passed: true };
}
