/** Independent Unicode 16 data for reference tooling only. No runtime imports. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";

export const UCD_SOURCES = {
    "DerivedJoiningType.txt": {
        url: "https://www.unicode.org/Public/16.0.0/ucd/extracted/DerivedJoiningType.txt",
        sha256: "6bd08b97da66b70ccfdab105a352de2984e02625239ec5695422c99b33d854f0",
    },
    "ArabicShaping.txt": { url: "https://www.unicode.org/Public/16.0.0/ucd/ArabicShaping.txt", sha256: "764f420cedfc8b43d9fec251c957a5d55fc45d40f6573f162990ed1dce7e36e0" },
    "UnicodeData.txt": { url: "https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt", sha256: "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f" },
    "LICENSE.txt": { url: "https://www.unicode.org/license.txt", sha256: "e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96" },
};
const TYPES = ["U", "R", "D", "C", "L", "T"];
// UTR #53 (AMTRA), https://www.unicode.org/reports/tr53/#AMTRA.
export const ARABIC_MCM = [0x654, 0x655, 0x658, 0x6dc, 0x6e3, 0x6e7, 0x6e8, 0x8ca, 0x8cb, 0x8cd, 0x8ce, 0x8cf, 0x8d3, 0x8f3];

export function loadJoiningData() {
    const cache = process.env.LITE_UNICODE_CACHE ?? resolve(homedir(), ".cache/babylon-lite/unicode/16.0.0");
    const source = Object.fromEntries(
        Object.entries(UCD_SOURCES).map(([name, identity]) => {
            const bytes = readFileSync(resolve(cache, name));
            assert.equal(createHash("sha256").update(bytes).digest("hex"), identity.sha256, `Unicode identity mismatch: ${name}`);
            return [name, bytes.toString("utf8")];
        })
    );
    assert.match(source["LICENSE.txt"], /UNICODE LICENSE V3/);
    const derived = new Uint8Array(0x110000);
    const crossCheck = new Uint8Array(0x110000);
    const mirrored = new Uint8Array(0x110000);
    const combining = new Uint8Array(0x110000);
    for (const line of source["DerivedJoiningType.txt"].split("\n")) {
        const match = line.match(/^([A-F0-9]+)(?:\.\.([A-F0-9]+))?\s*;\s*([URDCLT])\b/);
        if (match) {
            derived.fill(TYPES.indexOf(match[3]), parseInt(match[1], 16), parseInt(match[2] ?? match[1], 16) + 1);
        }
    }
    let rangeStart = 0;
    for (const line of source["UnicodeData.txt"].split("\n")) {
        const fields = line.split(";");
        if (fields.length !== 15) {
            continue;
        }
        const cp = parseInt(fields[0], 16);
        if (fields[1].endsWith(", First>")) {
            rangeStart = cp;
            continue;
        }
        const start = fields[1].endsWith(", Last>") ? rangeStart : cp;
        combining.fill(Number(fields[3]), start, cp + 1);
        if (["Mn", "Me", "Cf"].includes(fields[2])) {
            crossCheck.fill(TYPES.indexOf("T"), start, cp + 1);
        }
        if (fields[9] === "Y") {
            mirrored.fill(1, start, cp + 1);
        }
    }
    for (const line of source["ArabicShaping.txt"].split("\n")) {
        const fields = line
            .split("#")[0]
            .split(";")
            .map((field) => field.trim());
        if (fields.length >= 4 && /^[A-F0-9]+$/.test(fields[0])) {
            assert.ok(TYPES.includes(fields[2]), `Unknown joining type: ${line}`);
            crossCheck[parseInt(fields[0], 16)] = TYPES.indexOf(fields[2]);
        }
    }
    let checked = 0;
    for (let cp = 0; cp <= 0x10ffff; cp++) {
        if (cp >= 0xd800 && cp <= 0xdfff) {
            continue;
        }
        assert.equal(derived[cp], crossCheck[cp], `Joining_Type cross-check U+${cp.toString(16)}`);
        checked++;
    }
    return {
        identities: { version: "16.0.0", license: "Unicode License v3", sources: UCD_SOURCES, crossCheckedScalars: checked },
        joiningType(scalar) {
            const type = TYPES[derived[scalar]];
            return type === "C" ? "D" : type;
        },
        isMirrored: (scalar) => mirrored[scalar] === 1,
        canonicalClass: (scalar) => combining[scalar],
        // Independently transcribed Arabic CCC27..35 convention, HarfBuzz 11.0.1
        // hb-unicode.hh (MIT, Copyright the HarfBuzz authors). Other classes unchanged.
        markClass: (scalar) => (combining[scalar] === 33 ? 27 : combining[scalar] >= 27 && combining[scalar] <= 32 ? combining[scalar] + 1 : combining[scalar]),
        isMcm: (scalar) => ARABIC_MCM.includes(scalar),
    };
}

/** C10 pre-lookup evidence from original input and pinned UCD only, not HB output. */
export function arabicPreLookup(data, text) {
    const original = [...text].map((cp, cluster) => {
        const scalar = cp.codePointAt(0);
        return { scalar, cluster, originalIndex: cluster, markClass: data.markClass(scalar), isMcm: data.isMcm(scalar) };
    });
    const records = original.map((entry) => ({ ...entry }));
    const moves = [];
    function merge(start, end) {
        while (start > 0 && records[start - 1].cluster === records[start].cluster) {
            start--;
        }
        while (end < records.length && records[end - 1].cluster === records[end].cluster) {
            end++;
        }
        const minimum = Math.min(...records.slice(start, end).map((entry) => entry.cluster));
        for (let i = start; i < end; i++) {
            records[i].cluster = minimum;
        }
    }
    for (let start = 0; start < records.length;) {
        if (records[start].markClass === 0) {
            start++;
            continue;
        }
        let end = start + 1;
        while (end < records.length && records[end].markClass !== 0) {
            end++;
        }
        if (end - start <= 32) {
            for (let j = start + 1; j < end; j++) {
                let k = j;
                while (k > start && records[j].markClass < records[k - 1].markClass) {
                    k--;
                }
                if (k !== j) {
                    merge(k, j + 1);
                    records.splice(k, 0, records.splice(j, 1)[0]);
                    moves.push({ rule: "modified-class", from: j, to: k });
                }
            }
            let ins = start;
            for (const cc of [220, 230]) {
                let p = ins;
                while (p < end && records[p].markClass < cc) {
                    p++;
                }
                let q = p;
                while (q < end && records[q].markClass === cc && records[q].isMcm) {
                    q++;
                }
                if (q > p) {
                    merge(ins, q);
                    records.splice(ins, 0, ...records.splice(p, q - p));
                    moves.push({ rule: "MCM", cc, from: p, to: ins, count: q - p });
                    ins += q - p;
                }
            }
        }
        start = end;
    }
    const expected = records.map(({ scalar, cluster }) => [scalar, cluster]);
    return {
        keys: original,
        expectedLogical: expected,
        changed: JSON.stringify(expected) !== JSON.stringify(original.map(({ scalar, cluster }) => [scalar, cluster])),
        moves,
    };
}

/** Unicode neighbour rule, independently transcribed before native C9 exists.
 * Only Unicode properties and input scalars are read; no font/output fitting. */
export function correctJoiningForms(data, text) {
    const scalars = [...text].map((cp) => cp.codePointAt(0));
    const types = scalars.map((cp) => data.joiningType(cp));
    const previous = [];
    const next = [];
    let carry = "U";
    for (let i = 0; i < types.length; i++) {
        previous[i] = carry;
        if (types[i] !== "T") {
            carry = types[i];
        }
    }
    carry = "U";
    for (let i = types.length - 1; i >= 0; i--) {
        next[i] = carry;
        if (types[i] !== "T") {
            carry = types[i];
        }
    }
    return types.map((type, cluster) => {
        const backward = ["R", "D"].includes(type) && ["D", "L"].includes(previous[cluster]);
        const forward = ["D", "L"].includes(type) && ["R", "D"].includes(next[cluster]);
        return {
            scalar: scalars[cluster],
            cluster,
            joiningType: type,
            form: ["U", "T"].includes(type) ? null : backward && forward ? "medi" : backward ? "fina" : forward ? "init" : "isol",
        };
    });
}

export function correctJoiningFeatureRanges(forms) {
    return {
        features: { isol: 0, fina: 0, medi: 0, init: 0 },
        featureRanges: forms.filter((entry) => entry.form !== null).map((entry) => ({ tag: entry.form, value: 1, start: entry.cluster, end: entry.cluster + 1 })),
    };
}
