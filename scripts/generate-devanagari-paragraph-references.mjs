/** Standalone HarfBuzz 11.0.1 paragraph references for CTF-03 Devanagari wrapping.
 *
 * Shapes every corpus paragraph, the left/right split at candidate spaces and per-line texts
 * through both the pinned hb-shape CLI and the unchanged ctypes helper, which must agree exactly.
 * Identity constants and checks are duplicated from generate-shaping-references.mjs on purpose:
 * that generator hashes its own sources, so it must not be modified or imported.
 *
 * node scripts/generate-devanagari-paragraph-references.mjs --write | --check
 */
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Direction, Font, GlyphBuffer, UnicodeBuffer, shapeInto } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { PARAGRAPH_WORD_CAP, devanagariParagraphCorpus, devanagariParagraphDfltRows } from "../tests/lite/compute/references/devanagari-paragraph-corpus.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const corpusPath = resolve(root, "tests/lite/compute/references/devanagari-paragraph-corpus.mjs");
const outputPath = resolve(root, "tests/lite/compute/references/deva-paragraph-harfbuzz-11.0.1.json");
const generatorPath = fileURLToPath(import.meta.url);
const binary = "/opt/homebrew/Cellar/harfbuzz/11.0.1/bin/hb-shape";
const library = "/opt/homebrew/Cellar/harfbuzz/11.0.1/lib/libharfbuzz.0.dylib";
const python = "/opt/homebrew/bin/python3.12";
const pins = {
    binary: "75ff65d0aee0d59a0baade13620f7effac5ed12f2710b7aabd87c148315fff5c",
    library: "75cd955a1a28be95a789b529974690e7017d747f363937cc2680eccdc12fe8a4",
    font: "9c7d935139ea6a1e6ad9dbac4f6d27ece1e04bca8123c8888d00a0f9df4724cd",
    license: "a216f6f8d85c7228093e0ee5e258d9d377e6671f68acb4db1930b29583d0f331",
    textShaper: "6c844a3133e6816b00be742ee0d9ae4b4298a3db63c1f5975ed0c7b4ae253943",
};
/** Line table bound: at most K proven spaces strictly inside a line. */
const LINE_INNER_SPACES = 12;
/** Rows longer than this store sampled splits only and lines of boundary distance <= 3. */
const LONG_ROW = 2000;
const LONG_SPLIT_SAMPLES = 24;
const LONG_LINE_DISTANCE = 3;
/** Rows whose line table would dominate the file keep only a consistency summary (see README). */
const SUMMARY_ONLY_LINES = new Set(["multiseam-default"]);
const CLI_CONCURRENCY = 8;
const env = { ...process.env, LC_ALL: "C", LANG: "C" };
const configuration = {
    bufferFlags: "HB_BUFFER_FLAG_DEFAULT (no BOT/EOT, no pre/post context)",
    clusterLevel: 1,
    direction: "ltr",
    features: "script defaults (no overrides)",
    fontFuncs: "ot",
    fontScale: "upem",
    language: "und",
    ppem: 0,
    script: "Deva",
    shapers: "ot",
    variations: "none",
    dflt: { adapter: "text-shaper 0.1.28", direction: "ltr", language: null, script: "DFLT" },
};

const hash = (data) => createHash("sha256").update(data).digest("hex");
const fileHash = (path) => hash(readFileSync(path));
const args = process.argv.slice(2);
for (const arg of args) {
    assert.ok(arg === "--write" || arg === "--check", `Unknown argument ${arg}`);
}
assert.equal(args.length, 1, "Specify exactly one of --write / --check");
const write = args[0] === "--write";

function run(command, argv, input) {
    const result = spawnSync(command, argv, { cwd: root, encoding: "utf8", input, env, maxBuffer: 512 * 1024 * 1024 });
    assert.equal(result.status, 0, `${command} failed: ${result.error ?? result.stderr}`);
    return result.stdout;
}
function verify(path, expected) {
    assert.equal(fileHash(path), expected, `Identity mismatch: ${path}`);
}
verify(binary, pins.binary);
verify(library, pins.library);
assert.match(run(binary, ["--version"]), /HarfBuzz\) 11\.0\.1\b/);
assert.ok(run("/usr/bin/otool", ["-L", binary]).includes(library), "hb-shape must link to the pinned library");
assert.equal(run(python, ["-c", "import platform;print(platform.machine())"]).trim(), "arm64");
const fontPath = resolve(root, "lab/public/fonts/NotoSansDevanagari-Regular.ttf");
verify(fontPath, pins.font);
verify(resolve(root, "lab/public/fonts/OFL-NotoSansDevanagari.txt"), pins.license);
assert.equal(JSON.parse(readFileSync(resolve(root, "packages/babylon-lite/node_modules/text-shaper/package.json"))).version, "0.1.28");
verify(resolve(root, "packages/babylon-lite/node_modules/text-shaper/dist/index.js"), pins.textShaper);
const fontData = readFileSync(fontPath);
const oracleFont = Font.load(fontData.buffer.slice(fontData.byteOffset, fontData.byteOffset + fontData.byteLength));

function adapterRecords(text) {
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    input.addStr(text);
    input.setScript("DFLT");
    input.setDirection(Direction.LTR);
    shapeInto(oracleFont, input, output, { script: "DFLT", language: null, direction: "ltr" });
    return output.infos.slice(0, output.length).map((info, i) => {
        const p = output.positions[i];
        return [info.glyphId, info.cluster, p.xAdvance, p.yAdvance, p.xOffset, p.yOffset];
    });
}

function cliArgs(text) {
    return [
        "--shapers=ot",
        "--font-funcs=ot",
        "--font-size=upem",
        "--font-ppem=0",
        "--variations=",
        "--script=Deva",
        "--language=und",
        "--direction=ltr",
        "--cluster-level=1",
        "--no-glyph-names",
        "--output-format=json",
        "--features=",
        fontPath,
        `--unicodes=${[...text].map((cp) => cp.codePointAt(0).toString(16)).join(",")}`,
    ];
}

function cli(text) {
    return new Promise((done, fail) => {
        const child = spawn(binary, cliArgs(text), { cwd: root, env });
        const out = [];
        const err = [];
        child.stdout.on("data", (chunk) => out.push(chunk));
        child.stderr.on("data", (chunk) => err.push(chunk));
        child.on("error", fail);
        child.on("close", (status) => {
            if (status !== 0) {
                fail(new Error(`hb-shape failed (${status}): ${Buffer.concat(err).toString()}`));
                return;
            }
            const output = Buffer.concat(out).toString();
            done(JSON.parse(output).map((g) => [g.g, g.cl, g.ax, g.ay, g.dx, g.dy]));
        });
    });
}

/** Shapes unique non-empty Deva texts through ctypes (one batch) and the CLI; both must agree. */
async function shapeDeva(texts) {
    const unique = [...new Set(texts)];
    for (const text of unique) {
        assert.ok(text.length > 0, "Empty shaping request");
    }
    const requests = unique.map((text, i) => ({ id: `t${i}`, text, fontPath, sha256: pins.font, script: "Deva", direction: "ltr", captureEvidence: false }));
    const ctypes = JSON.parse(run(python, [resolve(root, "scripts/shaping-references-ctypes.py"), library], JSON.stringify(requests)));
    assert.equal(ctypes.version, "11.0.1");
    assert.equal(ctypes.results.length, unique.length);
    const shaped = new Map();
    ctypes.results.forEach((result, i) => {
        assert.equal(result.id, requests[i].id);
        shaped.set(unique[i], result.records);
    });
    let next = 0;
    const worker = async () => {
        while (next < unique.length) {
            const text = unique[next++];
            assert.deepEqual(await cli(text), shaped.get(text), `CLI/ctypes disagreement for ${JSON.stringify(text.slice(0, 40))}`);
        }
    };
    await Promise.all(Array.from({ length: CLI_CONCURRENCY }, worker));
    return shaped;
}

function candidates(scalars) {
    const out = [];
    for (let s = 0; s + 1 < scalars.length; s++) {
        if (scalars[s] === " " && scalars[s + 1] !== " ") {
            out.push(s);
        }
    }
    return out;
}

function sample(values, count) {
    if (values.length <= count) {
        return values;
    }
    return [...new Set(Array.from({ length: count }, (_, k) => values[Math.round((k * (values.length - 1)) / (count - 1))]))];
}

function independent(whole, left, right, s) {
    if (whole.length !== left.length + right.length) {
        return false;
    }
    const joined = [...left, ...right.map(([g, c, ...rest]) => [g, c + s + 1, ...rest])];
    return joined.every((record, i) => record.every((value, k) => value === whole[i][k]));
}

/** Records of `whole` whose clusters fall in [a, b), rebased to a. */
function slice(whole, a, b) {
    return whole.filter(([, c]) => c >= a && c < b).map(([g, c, ...rest]) => [g, c - a, ...rest]);
}

function sameRecords(x, y) {
    return x.length === y.length && x.every((record, i) => record.every((value, k) => value === y[i][k]));
}

function linePairs(boundaries, maxDistance) {
    const pairs = [];
    for (let i = 0; i < boundaries.length; i++) {
        for (let j = i + 1; j < boundaries.length && j - i <= maxDistance; j++) {
            pairs.push([boundaries[i], boundaries[j]]);
        }
    }
    return pairs;
}

const rows = [...devanagariParagraphCorpus.map((row) => ({ ...row, script: "Deva" })), ...devanagariParagraphDfltRows.map((row) => ({ ...row, script: "DFLT" }))];
assert.equal(new Set(rows.map((row) => row.id)).size, rows.length, "Duplicate corpus ids");

const plans = rows.map((row) => {
    const scalars = [...row.text];
    const n = scalars.length;
    const all = candidates(scalars);
    for (const s of row.proven) {
        assert.ok(all.includes(s), `${row.id}: proven ${s} is not a candidate space`);
    }
    const long = n > LONG_ROW;
    const splitAt = long ? sample(all, LONG_SPLIT_SAMPLES) : all;
    const breaks = row.script === "DFLT" ? all : row.proven;
    const boundaries = [...new Set([0, ...breaks.map((s) => s + 1), n])].sort((a, b) => a - b);
    const pairs = linePairs(boundaries, long ? LONG_LINE_DISTANCE : LINE_INNER_SPACES + 1);
    const sub = (a, b) => scalars.slice(a, b).join("");
    const texts = [row.text, ...splitAt.flatMap((s) => [sub(0, s + 1), sub(s + 1, n)]), ...pairs.map(([a, b]) => sub(a, b))];
    return { row, scalars, n, all, long, splitAt, pairs, sub, texts };
});

const devaTexts = plans.filter((plan) => plan.row.script === "Deva").flatMap((plan) => plan.texts);
const shaped = await shapeDeva(devaTexts);
const shape = (script, text) => (script === "DFLT" ? adapterRecords(text) : shaped.get(text));

const cases = plans.map(({ row, n, all, long, splitAt, pairs, sub }) => {
    const whole = shape(row.script, row.text);
    if (row.script === "Deva" && !row.tags.includes("reference-only")) {
        assert.ok(
            whole.every(([g]) => g !== 0),
            `${row.id}: .notdef in whole-paragraph records`
        );
    }
    const splits = splitAt.map((s) => {
        const left = shape(row.script, sub(0, s + 1));
        const right = shape(row.script, sub(s + 1, n));
        const result = { s, independent: independent(whole, left, right, s) };
        return long ? result : { ...result, left, right };
    });
    const lines = {};
    let consistent = 0;
    for (const [a, b] of pairs) {
        const records = shape(row.script, sub(a, b));
        if (sameRecords(records, slice(whole, a, b))) {
            consistent++;
        }
        if (!SUMMARY_ONLY_LINES.has(row.id)) {
            lines[`${a}:${b}`] = records;
        }
    }
    return {
        id: row.id,
        tags: [...row.tags],
        script: row.script,
        text: row.text,
        ...(row.note ? { note: row.note } : {}),
        length: n,
        candidates: all,
        proven: [...row.proven],
        whole,
        splitsSampled: long,
        splits,
        lineDistance: long ? LONG_LINE_DISTANCE : LINE_INNER_SPACES + 1,
        ...(SUMMARY_ONLY_LINES.has(row.id) ? { lineSummary: { pairs: pairs.length, consistent } } : { lines }),
    };
});

function stable(value) {
    if (Array.isArray(value)) {
        return value.map(stable);
    }
    if (value !== null && typeof value === "object") {
        return Object.fromEntries(
            Object.keys(value)
                .sort()
                .map((key) => [key, stable(value[key])])
        );
    }
    return value;
}
function serialize(value) {
    // Keep numeric record arrays on one line (same convention as generate-shaping-references.mjs).
    return JSON.stringify(stable(value), null, 2).replace(/\[\n\s+-?\d+(?:,\n\s+-?\d+)*\n\s*\]/g, (row) => `[${row.slice(1, -1).trim().split(/,\s*/).join(", ")}]`) + "\n";
}

const bytes = serialize({
    identity: {
        harfbuzz: { version: "11.0.1", binary, binarySha256: pins.binary, library, librarySha256: pins.library, python: `${python} (arm64)` },
        font: { path: "lab/public/fonts/NotoSansDevanagari-Regular.ttf", sha256: pins.font },
        license: { path: "lab/public/fonts/OFL-NotoSansDevanagari.txt", sha256: pins.license },
        textShaper: { version: "0.1.28", sha256: pins.textShaper },
        corpusSha256: fileHash(corpusPath),
        generatorSha256: fileHash(generatorPath),
    },
    configuration,
    wordCap: PARAGRAPH_WORD_CAP,
    lineInnerSpaces: LINE_INNER_SPACES,
    longRow: LONG_ROW,
    cases,
});

if (write) {
    writeFileSync(outputPath, bytes);
    console.warn(`wrote ${outputPath} (${bytes.length} bytes, ${cases.length} cases)`);
} else {
    assert.equal(readFileSync(outputPath, "utf8"), bytes, "Checked-in paragraph references differ from regenerated output");
    console.warn(`checked ${outputPath} (${bytes.length} bytes)`);
}
