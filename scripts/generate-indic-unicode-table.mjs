/** Font/input property generation only. No shaping or reference-output replay. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { format } from "prettier";
import { Font } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";

const args = process.argv.slice(2);
const sourceAt = args.indexOf("--source-dir");
const cache = sourceAt < 0 ? resolve(homedir(), ".cache/babylon-lite") : resolve(args[sourceAt + 1]);
const write = args.includes("--write");
assert.equal(Number(write) + Number(args.includes("--check")), 1, "Specify --write or --check");
const commit = "43a778492219032b9d747112e412dec2201d2667";
// 9f3deba7d0661ca99f1229b1710752483a321981 is the annotated 11.0.1
// tag object, not a commit. GitHub's signed tag object peels to `commit`.
const pins = {
    "unicode/16.0.0/UnicodeData.txt": "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f",
    "unicode/16.0.0/CompositionExclusions.txt": "89e83cf9cc8bef6c1f8bf77e42cf6f0341dfa42e66261f4dbe9b492e7a23c8ee",
    "unicode/16.0.0/DerivedCoreProperties.txt": "39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd",
    "unicode/16.0.0/LICENSE.txt": "e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96",
    "indic/harfbuzz-11.0.1/hb-ot-shaper-indic-table.cc": "b7a455c29a425a237f2ffeff9c6a44ef5a57de1eb5ff73a63e5c5c81a24702d1",
    "indic/harfbuzz-11.0.1/hb-ot-shaper-indic.hh": "3c6ddb7bc92ac5d641ba1b0206b091e40b0e8f41569f97704a7a6d8eac47d405",
    "indic/harfbuzz-11.0.1/hb-ot-shaper-indic-machine.rl": "85330b07abacb3aee2243f5c1ace4bf4a257c2d32dd5863949e2f4869371e988",
    "indic/harfbuzz-11.0.1/hb-ot-shaper-indic-machine.hh": "67da2e512ba8717c497ee53b06c29b4d8a92a565b89c7946e089862f3d42b264",
    "indic/harfbuzz-11.0.1/COPYING": "ba8f810f2455c2f08e2d56bb49b72f37fcf68f1f4fade38977cfd7372050ad64",
};
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sources = {};
for (const [name, expected] of Object.entries(pins)) {
    const bytes = readFileSync(resolve(cache, name));
    assert.equal(hash(bytes), expected, `Source identity mismatch: ${name}`);
    sources[name.split("/").at(-1)] = bytes.toString("utf8");
}
const table = sources["hb-ot-shaper-indic-table.cc"];
assert.match(table, /IndicSyllabicCategory-16\.0\.0/);
const category = Object.fromEntries([...sources["hb-ot-shaper-indic-machine.rl"].matchAll(/export\s+(\w+)\s*=\s*(\d+);/g)].map((m) => [m[1], Number(m[2])]));
const position = Object.fromEntries([...sources["hb-ot-shaper-indic.hh"].matchAll(/POS_(\w+)\s*=\s*(\d+)/g)].map((m) => [m[1], Number(m[2])]));
const catAliases = Object.fromEntries([...table.matchAll(/#define _OT_(\w+)\s+OT_(\w+)/g)].map((m) => [m[1], m[2]]));
const posAliases = Object.fromEntries([...table.matchAll(/#define _POS_(\w+)\s+POS_(\w+)/g)].map((m) => [m[1], m[2]]));
assert.equal(category.Ra, 15);
assert.equal(position.END, 14);
const entries = new Map();
for (const line of table.split("\n")) {
    const start = line.match(/\/\* ([0-9A-F]{4,6}) \*\//);
    if (!start) {
        continue;
    }
    const tokens = [...line.matchAll(/_\((\w+),(\w+)\)/g)];
    assert.equal(tokens.length, 8, `Ambiguous generated row: ${line}`);
    tokens.forEach((m, i) => entries.set(parseInt(start[1], 16) + i, [m[1], m[2]]));
}
const unicode = new Map(
    sources["UnicodeData.txt"]
        .trim()
        .split("\n")
        .map((line) => {
            const fields = line.split(";");
            return [parseInt(fields[0], 16), fields];
        })
);
const di = [...sources["DerivedCoreProperties.txt"].matchAll(/^([0-9A-F]+)(?:\.\.([0-9A-F]+))?\s*;\s*Default_Ignorable_Code_Point\b/gm)].map((m) => [
    parseInt(m[1], 16),
    parseInt(m[2] ?? m[1], 16),
]);
const exclusions = new Set([...sources["CompositionExclusions.txt"].matchAll(/^([0-9A-F]+)\s*(?:#|$)/gm)].map((m) => parseInt(m[1], 16)));
const mapped = [];
const ranges = [
    [13, 13],
    [0x20, 0x23],
    [0x25, 0x25],
    [0x27, 0x3f],
    [0x5b, 0x5f],
    [0x7b, 0x7e],
    [0xa0, 0xa0],
    [0xad, 0xad],
    [0xd7, 0xd7],
    [0xf7, 0xf7],
    [0x2bc, 0x2bc],
    [0x900, 0x97f],
    [0x1cd0, 0x1cf6],
    [0x1cf8, 0x1cf9],
    [0x200b, 0x200d],
    [0x2010, 0x2010],
    [0x2013, 0x2014],
    [0x2018, 0x2019],
    [0x201c, 0x201d],
    [0x2026, 0x2026],
    [0x20b9, 0x20b9],
    [0x20f0, 0x20f0],
    [0x2212, 0x2212],
    [0x25cc, 0x25cc],
    [0xa830, 0xa839],
    [0xa8e0, 0xa8ff],
    [0x11b00, 0x11b09],
];
for (const [a, b] of ranges) {
    for (let cp = a; cp <= b; cp++) {
        mapped.push(cp);
    }
}
const fontBytes = readFileSync(new URL("../lab/public/fonts/NotoSansDevanagari-Regular.ttf", import.meta.url));
assert.equal(hash(fontBytes), "9c7d935139ea6a1e6ad9dbac4f6d27ece1e04bca8123c8888d00a0f9df4724cd");
const font = Font.load(fontBytes.buffer.slice(fontBytes.byteOffset, fontBytes.byteOffset + fontBytes.byteLength));
mapped.forEach((cp) => assert.notEqual(font.glyphId(cp), 0, `Pinned mapped set U+${cp.toString(16)}`));
const rejected = (cp) =>
    cp === 13 || cp === 0xad || cp === 0x200b || cp === 0x25cc || (cp >= 0x1cd0 && cp <= 0x1cff) || (cp >= 0xa8e0 && cp <= 0xa8ff) || (cp >= 0x11b00 && cp <= 0x11b5f);
const decomposition = [];
for (let cp = 0x900; cp <= 0x97f; cp++) {
    const raw = unicode.get(cp)?.[5];
    if (cp !== 0x931 && raw && !raw.startsWith("<")) {
        const parts = raw.split(" ").map((part) => parseInt(part, 16));
        assert.equal(parts.length, 2);
        decomposition.push(cp, ...parts);
    }
}
assert.equal(decomposition.length, 30);
const compositions = [];
for (let cp = 0x900; cp <= 0x97f; cp++) {
    const raw = unicode.get(cp)?.[5];
    if (!raw || raw.startsWith("<") || exclusions.has(cp)) {
        continue;
    }
    const [a, b] = raw.split(" ").map((part) => parseInt(part, 16));
    if (!unicode.get(a)[2].startsWith("M")) {
        compositions.push([a, b, cp]);
    }
}
compositions.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
assert.deepEqual(compositions, [
    [0x928, 0x93c, 0x929],
    [0x930, 0x93c, 0x931],
    [0x933, 0x93c, 0x934],
]);
const props = (cp) => {
    let aliases = entries.get(cp) ?? ["X", "X"];
    if (cp === 0xa0) {
        aliases = ["GB", "C"];
    }
    if (cp === 0x25cc) {
        aliases = ["DC", "C"];
    }
    const cat = category[catAliases[aliases[0]]],
        pos = position[posAliases[aliases[1]]];
    assert.ok(Number.isInteger(cat) && Number.isInteger(pos), `Ambiguous Indic property U+${cp.toString(16)}`);
    const u = unicode.get(cp);
    assert.ok(u);
    const mark = u[2].startsWith("M"),
        ignorable = di.some(([a, b]) => cp >= a && cp <= b);
    const admissible = !rejected(cp);
    const decomposable = decomposition.some((value, i) => i % 3 === 0 && value === cp);
    // Modified CCC overrides apply to Arabic/Hebrew, not this admitted set.
    const flags = Number(mark) | (Number(ignorable) << 1) | (Number(cp === 0x200d) << 2) | (Number(cp === 0x200c) << 3) | (Number(admissible) << 4) | (Number(decomposable) << 5);
    return (cat | (pos << 8) | (Number(u[3]) << 16) | (flags << 24)) >>> 0;
};
const dense = Array.from({ length: 128 }, (_, i) => props(0x900 + i));
const sparse = mapped.filter((cp) => (cp < 0x900 || cp > 0x97f) && (!rejected(cp) || cp === 0x25cc)).flatMap((cp) => [cp, props(cp)]);
const vowels = [];
for (const [a, bs] of [
    [0x905, [0x93a, 0x93b, 0x93e, 0x945, 0x946, 0x949, 0x94a, 0x94b, 0x94c, 0x94f, 0x956, 0x957]],
    [0x906, [0x93a, 0x945, 0x946, 0x947, 0x948]],
    [0x909, [0x941]],
    [0x90f, [0x945, 0x946, 0x947]],
]) {
    bs.forEach((b) => vowels.push(a, b, 0, 1));
}
vowels.push(0x930, 0x94d, 0x907, 2);
assert.equal(vowels.length, 22 * 4);
const licence = sources["hb-ot-shaper-indic-machine.rl"].slice(0, sources["hb-ot-shaper-indic-machine.rl"].indexOf("#ifndef")).trim();
const version = hash(readFileSync(fileURLToPath(import.meta.url)));
const machineSource = sources["hb-ot-shaper-indic-machine.hh"];
const machineStart = Number(machineSource.match(/indic_syllable_machine_start = (\d+);/)[1]);
const machine = new Array(10).fill(0);
machine[0] = machineStart;
for (const [i, name] of ["trans_keys", "key_spans", "index_offsets", "indicies", "trans_targs", "trans_actions", "to_state_actions", "from_state_actions", "eof_trans"].entries()) {
    const match = machineSource.match(new RegExp(`static const [\\w ]+ _indic_syllable_machine_${name}\\[\\] = \\{([\\s\\S]*?)\\};`));
    assert.ok(match, `Ambiguous syllable DFA ${name}`);
    const values = [...match[1].matchAll(/-?\d+/g)].map((m) => Number(m[0]));
    assert.ok(values.every((value) => value >= 0));
    machine[i + 1] = machine.length;
    machine.push(...values);
}
const arrays = {
    INDIC_DENSE: dense,
    INDIC_SPARSE: sparse,
    INDIC_DECOMPOSITIONS: decomposition,
    INDIC_COMPOSITIONS: compositions.flat(),
    INDIC_VOWEL_CONSTRAINTS: vowels,
    INDIC_SYLLABLE_MACHINE: machine,
};
const sourceHeader = Object.entries(pins)
    .map(([name, sha]) => `${name}: ${sha}`)
    .join("\n * ");
const text =
    `${licence}\n/** GENERATED; do not edit. Node22: node scripts/generate-indic-unicode-table.mjs --source-dir ~/.cache/babylon-lite --write|--check.
 * Unicode16 UCD: https://www.unicode.org/Public/16.0.0/ucd/ ; Unicode License v3.
 * HarfBuzz11.0.1 annotated tag9f3deba7; peeled commit${commit}.
 * Source base: https://raw.githubusercontent.com/harfbuzz/harfbuzz/${commit}/src/
 * Source identities:\n * ${sourceHeader}
 * Props: category8 | position8 | modifiedCCC8 | flags8 (mark, DI, ZWJ, ZWNJ, admissible, decomposition).
 * Vowel entries: first, second, third-or-zero, insertion index. No modified CCC override is needed for the admitted set.
 */\nexport const INDIC_TABLE_VERSION = "${version}";\nexport const INDIC_DENSE_BASE = 0x900;\n` +
    Object.entries(arrays)
        .map(([name, values]) => `export const ${name} = /* @__PURE__ */ new Uint32Array([${values.join(",")}]);`)
        .join("\n") +
    "\n";
const output = await format(text, { parser: "typescript", tabWidth: 4, printWidth: 180, trailingComma: "es5" });
const destination = new URL("../packages/babylon-lite/src/text/compute-text-indic-unicode.ts", import.meta.url);
if (write) {
    writeFileSync(destination, output);
} else {
    assert.equal(readFileSync(destination, "utf8"), output, "Generated Indic Unicode data is stale");
}
process.stdout.write(
    `Indic Unicode table ${write ? "written" : "verified"}: ${dense.length} dense, ${sparse.length / 2} sparse, ${decomposition.length / 3} decompositions, version ${version}\n`
);
