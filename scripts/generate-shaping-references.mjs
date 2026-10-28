import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Direction, Font, GlyphBuffer, UnicodeBuffer, getOrCreateShapePlan, shapeInto } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { devanagariCorpus } from "../tests/lite/compute/references/devanagari-corpus.mjs";
import { SYNTHETIC_CMAP, SYNTHETIC_WITNESSES } from "../tests/lite/compute/references/synthetic-corpus.mjs";
import { analyzeImpact, scanPositionValues } from "./shaping-reference-analysis.mjs";
import { featureConfiguration, markOrderKeyed, normalizationEvidence } from "./shaping-reference-derivations.mjs";
import { ARABIC_MCM, arabicPreLookup, correctJoiningFeatureRanges, correctJoiningForms, loadJoiningData } from "./shaping-reference-unicode.mjs";
import { arabicCorpus } from "../tests/lite/compute/references/arabic-corpus.mjs";
import { G0_IDS, N_PROOF_IDS, deriveAttachedAdvances, reviewedFontPositions, unattachedNonzeroMarkGuard } from "./shaping-reference-attachments.mjs";
import { restrictedCmapFont } from "./shaping-reference-cmap.mjs";
import { indicCompositionEvidence, canonicalIndicInput, validateIndicProjection } from "./shaping-reference-indic-projection.mjs";
import { checkReferenceOutputs } from "./shaping-reference-outputs.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const destination = resolve(root, "tests/lite/compute/references");
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
const existingPins = {
    "Inter.ttf": "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031",
    "Roboto-Regular.ttf": "56a45233d29f11b4dfb86d248e921939d115778f87325e7ae8cc108383d6664d",
    "NotoSansArabic-Regular.ttf": "7ed3fe069312aceac454f17cf613a30f95271d6ed7ce58005ed4d016bd3823d7",
    "NotoSansGothic-Regular.ttf": "1977c6f245a6f1ae4b421370b0aba480238cc8a31744b7e3ef7f92569f662c07",
    "NotoSansHebrew-Regular.ttf": "671951828bd5c95db818e5bb12dcea2d0c0dda00311888522be061ee6835125e",
    "NotoSansSC-Regular-subset.ttf": "db2aee41c6b823a7487f1057e4f0f429603acd5a02949dbb9cc176810e4b31d6",
};
const legacyPins = {
    multiple: "ede722128eaa7c6fb1de8cf72bfc55b88ec0cb7c72ed21c56cfbd7801fbcbc84",
    alternate: "5271ee7844d34128875213bb21167b652ccdadec232f9631e73f87decab29d30",
    "mark-ligature": "b55e61f4ca172933f155e3189ee2dcc8072eaa92a5547b159cbeb4f2b5e8a0ca",
    "mark-mark": "a0942969c0d14bfece664a9cca0c99d1e4e6ce8049991faec94c494b5dc34ede",
    "point-anchor": "071b8f9a4ca2eeb4322c2a2c5e8d78d5622351243bdaf332a450270197e313ff",
    adjust: "5c079a760acb22f08b049caddfb0a512a3178e1cf70f329db5e58eb22c17e7eb",
    transitive: "e34fae4e9363f67884041997c1bc8d525ef6cb83052eb3129ca4427c5549f908",
    context: "3153323d447d3789f45a15cdc0ace70cd98d3403802d808fdb0a27605782566a",
    backtrack: "738d237470d84c8ba0026c6d50043dfc2be59b93c577fd78fd140e20ddc86147",
    "backtrack-reject": "312bc51785edf112497c83732c91a7136210a24d0217037a40fa69413db992c0",
    "ligature-ignore-marks": "bb9ae7ecb6bdcc39c72ed8eeda124c8f1ebf15264b45f51e1b068969a909fdc1",
    "pair-first": "fc39de46450c677e57f49b0f5d6d860455831a1205fb8e7e81ef77d642ff4e19",
    "pair-second": "2323a0901c0a919e5f98f8a3f804b00524fd8351758f5e75fb27c5551fdcc7e8",
    "pair-reject": "c9758ba6d92215afc20eafe9bb6f67e451892f0b77fa960dd887b59da5caf3d3",
    "refresh-roots": "472ad4693129145acc05862196b92222821a0df231c1aedd2285aaebe9cc5f81",
    "refresh-ligature": "083f991ce0a93832db470ee6fa399e075f733b35b99d223365f675177cef7118",
};
const hash = (data) => createHash("sha256").update(data).digest("hex");
const fileHash = (path) => hash(readFileSync(path));
const args = process.argv.slice(2);
assert.equal(args.filter((arg) => arg === "--write" || arg === "--check").length, 1, "Specify exactly one of --write / --check");
const write = args.includes("--write");
const override = args.find((arg) => arg.startsWith("--expected-binary-sha256="))?.split("=")[1];
for (const arg of args) {
    assert.ok(["--write", "--check"].includes(arg) || arg.startsWith("--expected-binary-sha256="), `Unknown argument ${arg}`);
}
function run(command, argv, input) {
    const result = spawnSync(command, argv, { cwd: root, encoding: "utf8", input, env: { ...process.env, LC_ALL: "C", LANG: "C" }, maxBuffer: 128 * 1024 * 1024 });
    assert.equal(result.status, 0, `${command} failed: ${result.error ?? result.stderr}`);
    return result.stdout;
}
function verify(path, expected) {
    assert.equal(fileHash(path), expected, `Identity mismatch: ${path}`);
}
verify(binary, override ?? pins.binary);
verify(library, pins.library);
assert.match(run(binary, ["--version"]), /HarfBuzz\) 11\.0\.1\b/);
assert.ok(run("/usr/bin/otool", ["-L", binary]).includes(library), "hb-shape must link to the pinned library");
assert.equal(run(python, ["-c", "import platform;print(platform.machine())"]).trim(), "arm64");
const fontPath = resolve(root, "lab/public/fonts/NotoSansDevanagari-Regular.ttf");
verify(fontPath, pins.font);
verify(resolve(root, "lab/public/fonts/OFL-NotoSansDevanagari.txt"), pins.license);
assert.equal(JSON.parse(readFileSync(resolve(root, "packages/babylon-lite/node_modules/text-shaper/package.json"))).version, "0.1.28");
verify(resolve(root, "packages/babylon-lite/node_modules/text-shaper/dist/index.js"), pins.textShaper);
for (const [file, sha256] of Object.entries(existingPins)) {
    verify(resolve(root, "lab/public/fonts", file), sha256);
}
const fontData = readFileSync(fontPath);
const oracleFont = Font.load(fontData.buffer.slice(fontData.byteOffset, fontData.byteOffset + fontData.byteLength));

export function adapterRecords(font, text, script, direction) {
    const input = new UnicodeBuffer();
    const output = new GlyphBuffer();
    input.addStr(text);
    input.setScript(script);
    input.setDirection(direction === "rtl" ? Direction.RTL : Direction.LTR);
    shapeInto(font, input, output, { script, language: null, direction });
    return output.infos.slice(0, output.length).map((info, i) => {
        const p = output.positions[i];
        return [info.glyphId, info.cluster, p.xAdvance, p.yAdvance, p.xOffset, p.yOffset];
    });
}

function cli(request) {
    const features = [
        ...Object.entries(request.features ?? {}).map(([key, value]) => `${key}=${value}`),
        ...(request.featureRanges ?? []).map((feature) => `${feature.tag}[${feature.start}:${feature.end}]=${feature.value}`),
    ].join(",");
    const output = run(binary, [
        "--shapers=ot",
        "--font-funcs=ot",
        "--font-size=upem",
        "--font-ppem=0",
        "--variations=",
        `--script=${request.script}`,
        "--language=und",
        `--direction=${request.direction}`,
        "--cluster-level=1",
        "--no-glyph-names",
        "--output-format=json",
        `--features=${features}`,
        request.cliRestrictedPath ?? request.fontPath,
        `--unicodes=${[...request.text].map((cp) => cp.codePointAt(0).toString(16)).join(",")}`,
    ]);
    if (request.text.length === 0) {
        // hb-shape serializes no paragraph for empty scalar input. This is an
        // explicit empty-input boundary, not a fallback for missing output.
        assert.equal(output, "", "Unexpected CLI output for empty scalar input");
        return [];
    }
    return JSON.parse(output).map((g) => [g.g, g.cl, g.ax, g.ay, g.dx, g.dy]);
}

function shapeRequests(requests) {
    const ctypes = JSON.parse(run(python, [resolve(root, "scripts/shaping-references-ctypes.py"), library], JSON.stringify(requests)));
    assert.equal(ctypes.version, "11.0.1");
    return ctypes.results.map((result, i) => {
        assert.equal(result.id, requests[i].id);
        if (!requests[i].restrictedCmap || requests[i].cliRestrictedPath) {
            assert.deepEqual(cli(requests[i]), result.records, `CLI/ctypes disagreement: ${result.id}`);
            if (requests[i].cliRestrictedPath) {
                result.cliCtypesAgreed = true;
            }
        } else if (requests[i].restrictedCrossCheck) {
            const unrestricted = cli(requests[i]);
            result.restrictedCrossCheck = {
                passed: JSON.stringify(unrestricted) === JSON.stringify(result.records),
                unrestrictedRecords: unrestricted,
            };
        }
        return result;
    });
}
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
const pendingOutputs = new Map();
function emit(file, value) {
    // Keep numeric record arrays on one line; full six-field evidence otherwise
    // repeats millions of indentation bytes. No data or precision is discarded.
    const bytes = JSON.stringify(stable(value), null, 2).replace(/\[\n\s+-?\d+(?:,\n\s+-?\d+)*\n\s*\]/g, (row) => `[${row.slice(1, -1).trim().split(/,\s*/).join(", ")}]`) + "\n";
    emitText(file, bytes);
}
function emitText(file, bytes) {
    pendingOutputs.set(file, bytes);
}
const allSyntheticFonts = JSON.parse(run(process.execPath, ["--import", "tsx", resolve(root, "scripts/shaping-references-fixtures.ts")]));
for (const entry of allSyntheticFonts.filter((font) => font.legacy)) {
    assert.equal(hash(Buffer.from(entry.bytes, "base64")), legacyPins[entry.kind], `Legacy font identity mismatch: ${entry.kind}`);
}
const corpus = devanagariCorpus();
const requests = corpus
    .filter((entry) => entry.expected === "positive" && entry.reference !== "text-shaper")
    .map((entry) => ({ ...entry, fontPath, sha256: pins.font, script: "Deva", direction: "ltr" }));
const results = new Map(shapeRequests(requests).map((result) => [result.id, result]));
const indicEvidence = indicCompositionEvidence(oracleFont);
const b1Inputs = requests.filter((request) => [...request.text].some((character) => indicEvidence.compositions.some((entry) => entry.scalars[2] === character.codePointAt(0))));
const b1Requests = b1Inputs.map((request) => ({ ...request, id: `${request.id}/canonical-decomposed`, text: canonicalIndicInput(request.text, indicEvidence.compositions).text }));
const b1Results = shapeRequests(b1Requests);
emit("deva-projected-pauses.json", {
    fontSha256: pins.font,
    ...indicEvidence,
    note: "Raw independent snapshots are unchanged. Only three same-cluster nominal pairs before nukt are projected; later pauses and all final fields are exact.",
    cases: b1Inputs.map((request, i) =>
        validateIndicProjection(results.get(request.id), b1Results[i], canonicalIndicInput(request.text, indicEvidence.compositions).clusterMap, indicEvidence)
    ),
});
const cases = corpus.map((entry) => {
    const base = { ...entry, scalars: [...entry.text].map((cp) => cp.codePointAt(0).toString(16).toUpperCase()) };
    if (entry.expected !== "positive") {
        return base;
    }
    if (entry.reference === "text-shaper") {
        return { ...base, records: adapterRecords(oracleFont, entry.text, "DFLT", "ltr"), lookupTrace: [], pauseSnapshots: [] };
    }
    const result = results.get(entry.id);
    return { ...base, id: result.id, records: result.records, lookupTrace: result.lookupTrace, pauseSnapshots: result.pauseSnapshots };
});
emit("deva-harfbuzz-11.0.1.json", { fontSha256: pins.font, cases });
const disabledFeatures = Object.fromEntries(["clig", "dist", "abvm", "blwm", "curs", "rvrn", "ltra", "ltrm", "rclt", "rlig", "locl"].map((tag) => [tag, 0]));
// This exact declared cmap is also asserted against both parsers in the unit test.
assert.deepEqual(SYNTHETIC_CMAP, [0x41, 0x42, 0x43, 0xe000, 0xe001]);
assert.ok(
    SYNTHETIC_CMAP.every((cp) => String.fromCodePoint(cp).normalize("NFD") === String.fromCodePoint(cp)),
    "Synthetic cmap decomposition"
);
const temporary = mkdtempSync(resolve(tmpdir(), "lite-shaping-references-"));
let syntheticCount;
let legacyFontData;
try {
    const fonts = allSyntheticFonts;
    legacyFontData = fonts.filter((font) => font.legacy);
    assert.equal(legacyFontData.length, 16);
    const paths = new Map(
        fonts.map(({ kind, bytes }) => {
            const data = Buffer.from(bytes, "base64");
            const path = resolve(temporary, `${kind}.ttf`);
            writeFileSync(path, data);
            return [kind, { path, sha256: hash(data), oracle: Font.load(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)) }];
        })
    );
    const syntheticRequests = SYNTHETIC_WITNESSES.map((entry) => ({
        ...entry,
        fontPath: paths.get(entry.kind).path,
        sha256: paths.get(entry.kind).sha256,
        script: "Latn",
        direction: "ltr",
        features: disabledFeatures,
    }));
    const shaped = shapeRequests(syntheticRequests);
    syntheticCount = shaped.length;
    emit("synthetic-harfbuzz-11.0.1.json", {
        configurationMatched: true,
        script: "Latn",
        direction: "ltr",
        language: "und",
        clusterLevel: 1,
        disabledFeatures,
        cmap: SYNTHETIC_CMAP,
        cases: SYNTHETIC_WITNESSES.map((entry, index) => ({
            ...entry,
            fontSha256: paths.get(entry.kind).sha256,
            oldTextShaperRecords: adapterRecords(paths.get(entry.kind).oracle, entry.text, "latn", "ltr"),
            ...(entry.expected === "positive" ? { records: shaped[index].records } : { info_only: shaped[index].records }),
            lookupTrace: shaped[index].lookupTrace,
        })),
    });
} finally {
    rmSync(temporary, { recursive: true, force: true });
}
const existingFonts = new Map(
    Object.entries(existingPins).map(([file, sha256]) => {
        const path = resolve(root, "lab/public/fonts", file);
        verify(path, sha256);
        const bytes = readFileSync(path);
        return [file, Font.load(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))];
    })
);
const fixtureFiles = Object.keys(existingPins);
const fontPaths = Object.fromEntries(fixtureFiles.map((file) => [file, resolve(root, "lab/public/fonts", file)]));
const legacyTemporary = mkdtempSync(resolve(tmpdir(), "lite-legacy-shaping-fonts-"));
process.once("exit", () => rmSync(legacyTemporary, { recursive: true, force: true }));
for (const { kind, bytes } of legacyFontData) {
    const data = Buffer.from(bytes, "base64");
    const key = `synthetic:${kind}`;
    const path = resolve(legacyTemporary, `${kind}.ttf`);
    writeFileSync(path, data);
    const font = Font.load(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
    assert.ok([0x41, 0x42, 0x43].every((cp) => String.fromCodePoint(cp).normalize("NFD") === String.fromCodePoint(cp)));
    existingFonts.set(key, font);
    existingPins[key] = hash(data);
    fontPaths[key] = path;
}
const inventory = JSON.parse(run(process.execPath, ["--import", "tsx", resolve(root, "scripts/shaping-references-corpus.ts"), JSON.stringify(fontPaths)]));
const existingCases = inventory.cases;
const planIndex = inventory.planIndex.map((entry) => {
    const font = existingFonts.get(entry.font);
    const plan = getOrCreateShapePlan(font, entry.script, null, entry.direction);
    const flags = [];
    if (plan.gsubLookups.some(({ lookup }) => lookup.type === 5 && lookup.subtables.every((subtable) => subtable.format === 3))) {
        flags.push("C5");
    }
    if (plan.gsubLookups.some(({ lookup }) => [5, 6].includes(lookup.type))) {
        flags.push("C4");
    }
    if (plan.gposLookups.some(({ lookup }) => [7, 8].includes(lookup.type))) {
        flags.push("C6");
    }
    return { ...entry, flags, gsubRoots: plan.gsubLookups.map((lookup) => lookup.index), gposRoots: plan.gposLookups.map((lookup) => lookup.index) };
});
const joiningData = loadJoiningData();
const arabicWitnesses = arabicCorpus().map((entry) => ({
    ...entry,
    configuration: featureConfiguration(existingFonts.get(entry.font), entry.script, entry.direction),
    normalization: normalizationEvidence(existingFonts.get(entry.font), entry.text, entry.script),
}));
const analyzed = existingCases.map((entry) => {
    const font = existingFonts.get(entry.font);
    const oldRecords = adapterRecords(font, entry.text, entry.script, entry.direction);
    const analysis = analyzeImpact(font, entry, oldRecords);
    const flags = analysis.flags;
    if (entry.script === "arab") {
        flags.push("C9");
        analysis.evidence.push({
            correction: "C9",
            table: "GSUB",
            lookup: null,
            detail: "User-approved Unicode joining masks and Arabic stages replace global joining substitutions.",
        });
    }
    const normalization = normalizationEvidence(font, entry.text, entry.script);
    const configuration = featureConfiguration(font, entry.script, entry.direction);
    const category =
        flags.length === 0
            ? "U"
            : flags.length === 1 && flags[0] === "C1"
              ? "A"
              : (["latn", "grek", "cyrl", "hani", "goth", "DFLT"].includes(entry.script) || (entry.script === "hebr" && !/\p{M}/u.test(entry.text))) && normalization.inert
                ? "H"
                : "X";
    return { ...entry, category, oldRecords, ...analysis, normalization, configuration };
});
// Repeated inputs in separate source families share only reference-tool work,
// never product output or a runtime shaping cache.
const requestKey = (entry) => JSON.stringify([entry.font, entry.script, entry.direction, entry.text]);
const existingRequests = [
    ...new Map(
        [...analyzed, ...arabicWitnesses].map((entry) => [
            requestKey(entry),
            {
                id: requestKey(entry),
                fontPath: fontPaths[entry.font],
                sha256: existingPins[entry.font],
                text: entry.text,
                script: entry.script === "DFLT" ? "DFLT" : entry.script[0].toUpperCase() + entry.script.slice(1),
                direction: entry.direction,
                features: entry.configuration.features,
                captureEvidence: false,
            },
        ])
    ).values(),
];
const existingResults = new Map(shapeRequests(existingRequests).map((entry) => [entry.id, entry]));
const existingShaped = new Map([...existingResults].map(([id, entry]) => [id, entry.records]));
const restrictedRequests = existingRequests.map((request) => ({
    ...request,
    id: `restricted:${request.id}`,
    restrictedCmap: true,
    restrictedCrossCheck: normalizationEvidence(existingFonts.get(JSON.parse(request.id)[0]), request.text, JSON.parse(request.id)[1]).inert,
}));
const restrictedResults = shapeRequests(restrictedRequests);
const restrictedShaped = new Map(restrictedResults.map((entry, i) => [existingRequests[i].id, entry.records]));
const restrictedFailures = restrictedResults.filter((entry) => entry.restrictedCrossCheck?.passed === false);
emit("restricted-cmap-validation.json", {
    tested: restrictedRequests.filter((request) => request.restrictedCrossCheck).length,
    failed: restrictedFailures.length,
    failures: restrictedFailures,
    contract: "Subfont maps input scalars plus U+0020 only; every refined-inert case is cross-checked against unrestricted CLI.",
});
const preLookupRequests = existingRequests
    .filter((request) => request.script === "Arab")
    .map((request, i) => {
        const font = existingFonts.get(JSON.parse(request.id)[0]);
        const defaults = featureConfiguration(font, "arab", request.direction).harfbuzzDefaults;
        const tags = [font.gsub, font.gpos].flatMap((table) =>
            (table?.featureList.features ?? []).map(({ featureTag }) =>
                String.fromCharCode(featureTag >>> 24, (featureTag >>> 16) & 255, (featureTag >>> 8) & 255, featureTag & 255)
            )
        );
        const cliRestrictedPath = resolve(legacyTemporary, `c10-restricted-${i}.ttf`);
        const bytes = restrictedCmapFont(readFileSync(request.fontPath), font, request.text);
        writeFileSync(cliRestrictedPath, bytes);
        const clone = Font.load(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        for (const scalar of new Set([0x20, ...[...request.text].map((cp) => cp.codePointAt(0))])) {
            assert.equal(clone.glyphId(scalar), font.glyphId(scalar));
        }
        return {
            ...request,
            id: `prelookup:${request.id}`,
            restrictedCmap: true,
            cliRestrictedPath,
            features: Object.fromEntries([...new Set([...tags, ...defaults])].sort().map((tag) => [tag, 0])),
            captureEvidence: true,
        };
    });
const preLookupResults = shapeRequests(preLookupRequests);
const preLookupValidation = new Map(
    preLookupResults.map((result, i) => {
        const request = preLookupRequests[i];
        const key = request.id.slice("prelookup:".length);
        const font = existingFonts.get(JSON.parse(key)[0]);
        const expected = arabicPreLookup(joiningData, request.text);
        const inverse = new Map();
        for (const cp of new Set([...request.text].map((cp) => cp.codePointAt(0)))) {
            const glyph = font.glyphId(cp);
            inverse.set(glyph, [...(inverse.get(glyph) ?? []), cp]);
        }
        const logicalRecords = result.records.slice().reverse();
        const ambiguous = logicalRecords.filter(([glyph]) => inverse.get(glyph)?.length !== 1).map(([glyph]) => ({ glyph, scalars: inverse.get(glyph) ?? [] }));
        const observedLogical = logicalRecords.map(([glyph, cluster]) => [inverse.get(glyph)?.length === 1 ? inverse.get(glyph)[0] : null, cluster]);
        return [
            key,
            {
                text: request.text,
                ...expected,
                observedLogical,
                ambiguous,
                passed:
                    result.cliCtypesAgreed &&
                    result.lookupTrace.length === 0 &&
                    ambiguous.length === 0 &&
                    JSON.stringify(expected.expectedLogical) === JSON.stringify(observedLogical),
                cliCtypesAgreed: result.cliCtypesAgreed,
                lookupsOffRecords: result.records,
                disabledFeatures: Object.keys(request.features),
                lookupsOffTrace: result.lookupTrace,
                restrictedCliFontSha256: fileHash(request.cliRestrictedPath),
            },
        ];
    })
);
const preLookupFailures = [...preLookupValidation].filter(([, entry]) => !entry.passed);
emit("arabic-prelookup-validation.json", {
    rule: "Original input; pinned UCD16 Arabic modified CCC + UTR53 MCM; stable sort with cluster spans; 32-mark cap",
    mcmScalars: ARABIC_MCM,
    tested: preLookupValidation.size,
    failed: preLookupFailures.length,
    cases: [...preLookupValidation].map(([id, entry]) => ({ id, ...entry })),
});
const joiningValidationRequests = existingRequests
    .filter((request) => request.script === "Arab")
    .map((request) => {
        const forms = correctJoiningForms(joiningData, request.text);
        const ranges = correctJoiningFeatureRanges(forms);
        return { ...request, id: `correct-joining:${request.id}`, features: { ...request.features, ...ranges.features }, featureRanges: ranges.featureRanges };
    });
const joiningValidationResults = shapeRequests(joiningValidationRequests);
const joiningValidation = new Map(
    joiningValidationResults.map((result, i) => {
        const request = joiningValidationRequests[i];
        const key = request.id.slice("correct-joining:".length);
        const records = existingShaped.get(key);
        return [
            key,
            {
                text: request.text,
                joiningForms: correctJoiningForms(joiningData, request.text),
                passed: JSON.stringify(result.records) === JSON.stringify(records),
                harfbuzzRecords: records,
                explicitCorrectFormRecords: result.records,
            },
        ];
    })
);
const yAdvanceScan = Object.fromEntries(fixtureFiles.map((file) => [file, scanPositionValues(existingFonts.get(file))]));
for (const [file, counts] of Object.entries(yAdvanceScan)) {
    assert.equal(counts.nonzeroSingleYAdvance + counts.nonzeroPairYAdvance, 0, `Fixture orthogonal-advance drift: ${file}`);
}
const joiningFailures = [...joiningValidation].filter(([, entry]) => !entry.passed);
emit("arabic-joining-validation.json", {
    rule: "Unicode16 nearest non-transparent neighbours; correct forms, never legacy-output pins",
    tested: joiningValidation.size,
    failed: joiningFailures.length,
    cases: [...joiningValidation].map(([id, entry]) => ({ id, ...entry })),
});
function arabicTreatment(entry) {
    const validation = joiningValidation.get(requestKey(entry));
    const preLookup = preLookupValidation.get(requestKey(entry));
    const base = {
        joiningForms: validation.joiningForms,
        joiningValidated: validation.passed,
        preLookup,
        flags: [...new Set([...(entry.flags ?? []), "C9", ...(preLookup.changed ? ["C10"] : [])])],
    };
    if (!entry.configuration.requestedInFont.some((tag) => ["isol", "fina", "medi", "init"].includes(tag))) {
        return { ...base, category: "X", pendingPolicy: "D10", failedDerivation: "D10 font policy pending: no joining features; neither acceptance nor rejection invented" };
    }
    if (!validation.passed) {
        return { ...base, category: "X", failedDerivation: "Independent Unicode joining analysis differs from HarfBuzz's own joining" };
    }
    if ([...entry.text].some((cp) => joiningData.isMirrored(cp.codePointAt(0)))) {
        return { ...base, category: "X", failedDerivation: "bidi mirroring out of scope" };
    }
    if (!preLookup.passed) {
        return { ...base, category: "X", failedDerivation: "C10 original-input pre-lookup scalar order/cluster validation failed" };
    }
    return { ...base, category: "H", treatment: preLookup.changed ? "H-C10" : "H-C9", records: existingShaped.get(requestKey(entry)), referenceConfiguration: "matched" };
}
const arabicCases = arabicWitnesses.map((entry) => ({
    ...entry,
    ...arabicTreatment(entry),
    harfbuzzRecords: existingShaped.get(requestKey(entry)),
    count: existingShaped.get(requestKey(entry)).length,
    lookupTrace: existingResults.get(requestKey(entry)).lookupTrace,
}));
emit("arabic-harfbuzz-11.0.1.json", {
    fontSha256: existingPins["NotoSansArabic-Regular.ttf"],
    cases: arabicCases,
});
let impactCases = analyzed.map(({ arithmeticRecords, ...entry }) => {
    if (entry.script === "arab") {
        const treatment = arabicTreatment(entry, entry.oldRecords);
        return {
            ...entry,
            ...treatment,
            info_only: existingShaped.get(requestKey(entry)),
            ...(treatment.category === "X"
                ? {
                      syntheticWitnesses: SYNTHETIC_WITNESSES.filter((witness) => entry.flags.includes(witness.correction)).map((witness) => witness.id),
                      proposedTreatment: { status: "requires individual Lead approval; no fields relaxed", reason: treatment.failedDerivation },
                  }
                : {}),
        };
    }
    if (entry.category === "U") {
        return { ...entry, records: entry.oldRecords };
    }
    const harfbuzz = existingShaped.get(requestKey(entry));
    if ([...N_PROOF_IDS, ...G0_IDS].includes(entry.id)) {
        const derived = reviewedFontPositions(existingFonts.get(entry.font), entry, entry.oldRecords, harfbuzz);
        return { ...entry, ...derived, info_only: harfbuzz };
    }
    if (entry.font === "synthetic:mark-mark") {
        const derived = deriveAttachedAdvances(existingFonts.get(entry.font), entry, entry.oldRecords, entry, harfbuzz);
        return { ...entry, category: "A-attach", treatment: "A-attach", records: derived.records, attachmentCrossCheck: derived.crossCheck, info_only: harfbuzz };
    }
    if (entry.font === "Inter.ttf" && entry.text === "A\u0326") {
        assert.deepEqual(arithmeticRecords, harfbuzz, "Inter comma-below C1 arithmetic cross-check");
    }
    if (entry.category === "A") {
        const matchedGlyphsAndClusters = JSON.stringify(entry.oldRecords.map((row) => row.slice(0, 2))) === JSON.stringify(harfbuzz.map((row) => row.slice(0, 2)));
        if (matchedGlyphsAndClusters) {
            if (JSON.stringify(arithmeticRecords) !== JSON.stringify(harfbuzz)) {
                return {
                    ...entry,
                    category: "X",
                    info_only: harfbuzz,
                    c1ArithmeticInfo: arithmeticRecords,
                    failedDerivation: "C1 arithmetic/HarfBuzz disagreement (including legacy synthetic mark-width policy); exact treatment requires individual review",
                    syntheticWitnesses: SYNTHETIC_WITNESSES.filter((witness) => witness.correction === "C1").map((witness) => witness.id),
                    proposedTreatment: { status: "requires individual Lead approval; no fields relaxed", reason: "All differing fields preserved; no automatic expected records." },
                };
            }
        }
        return { ...entry, records: arithmeticRecords, harfbuzz, arithmeticCrossCheck: matchedGlyphsAndClusters ? "equal" : "glyphs-or-clusters-confounded" };
    }
    if (entry.category === "H") {
        return { ...entry, records: harfbuzz, ...(entry.flags.includes("C1") ? { c1ArithmeticInfo: arithmeticRecords } : {}), configurationMatched: true };
    }
    const font = existingFonts.get(entry.font);
    const restricted = restrictedShaped.get(requestKey(entry));
    const candidate = entry.normalization.inert ? harfbuzz : restricted;
    const keyed = markOrderKeyed(font, entry, entry.oldRecords, candidate);
    if (keyed.ok) {
        return {
            ...entry,
            category: "H",
            treatment: entry.normalization.inert ? "H-matched" : "H-restricted",
            ...(keyed.reordered ? { markOrderKeyed: keyed } : {}),
            records: keyed.records,
            configurationMatched: true,
            info_only: harfbuzz,
        };
    }
    return {
        ...entry,
        info_only: harfbuzz,
        syntheticWitnesses: SYNTHETIC_WITNESSES.filter((witness) => entry.flags.includes(witness.correction)).map((witness) => witness.id),
        failedDerivation: keyed.reason,
        restrictedInfoOnly: restricted,
        proposedTreatment: { status: "requires individual Lead approval; no fields relaxed", reason: "No automatic expectation assigned after failed independent derivation." },
    };
});
const hGuardViolations = [];
let hGuardChecked = 0;
for (const entry of [...impactCases, ...arabicCases].filter((entry) => entry.category === "H")) {
    hGuardChecked++;
    const violations = unattachedNonzeroMarkGuard(existingFonts.get(entry.font), entry, entry.records);
    if (violations.length) {
        hGuardViolations.push({ id: entry.id, source: entry.source ?? "new Arabic witness", font: entry.font, text: entry.text, violations });
    }
}
const invalidHIds = new Set(hGuardViolations.map((entry) => entry.id));
// Quarantine invalid prospective expectations; never silently adopt zero widths
// or invent a treatment for cases outside the three approved A-attach rows.
impactCases = impactCases.map((entry) => {
    if (!invalidHIds.has(entry.id)) {
        return entry;
    }
    const { records, ...diagnostic } = entry;
    return {
        ...diagnostic,
        category: "X",
        candidateReferenceRecords: records,
        guardFailure: "R3_UNATTACHED_NONZERO_HMTX",
        failedDerivation: "R3 accepted-H guard failed: unattached class-3 nonzero-hmtx glyph; no new expectation accepted",
        syntheticWitnesses: SYNTHETIC_WITNESSES.filter((witness) => entry.flags.includes(witness.correction)).map((witness) => witness.id),
        proposedTreatment: {
            status: "requires individual Lead approval; no fields relaxed",
            reason: "Mandatory guard rejected the prospective H expectation; no field values fitted.",
        },
    };
});
const devaNonzeroMarks = Array.from({ length: oracleFont.numGlyphs }, (_, glyph) => glyph).filter(
    (glyph) => oracleFont.gdef?.glyphClassDef.get(glyph) === 3 && oracleFont.advanceWidth(glyph) !== 0
);
const devaGuardViolations = cases.flatMap((entry) =>
    (entry.records ?? []).flatMap((record, index) => (devaNonzeroMarks.includes(record[0]) ? [{ id: entry.id, index, glyph: record[0] }] : []))
);
const guards = {
    acceptedHUnattachedNonzeroHmtx: {
        passed: hGuardViolations.length === 0,
        checkedRows: hGuardChecked,
        violatingRows: hGuardViolations.length,
        violatingGlyphs: hGuardViolations.reduce((sum, entry) => sum + entry.violations.length, 0),
        violations: hGuardViolations,
        note: "Prospective H/H-restricted/H-C9 expectations are checked before quarantine; rejection is not a guard pass.",
    },
    devanagariNonzeroClass3: {
        passed: devaGuardViolations.length === 0 && JSON.stringify(devaNonzeroMarks) === JSON.stringify([149, 150, 151, 152, 153, 154]),
        nonzeroClass3Glyphs: devaNonzeroMarks,
        checkedCases: cases.filter((entry) => entry.records).length,
        violations: devaGuardViolations,
    },
};
const totals = { U: 0, A: 0, "A-attach": 0, "N-proof": 0, G0: 0, H: 0, X: 0 };
const perSource = {};
const treatments = {};
for (const entry of impactCases) {
    totals[entry.category]++;
    (perSource[entry.source] ??= { U: 0, A: 0, "A-attach": 0, "N-proof": 0, G0: 0, H: 0, X: 0 })[entry.category]++;
    const treatment = entry.markOrderKeyed ? `${entry.treatment}+mark-order-keyed` : (entry.treatment ?? entry.category);
    treatments[treatment] = (treatments[treatment] ?? 0) + 1;
}
const sourceLabels = new Map();
for (const entry of [...impactCases, ...inventory.nonReference, ...planIndex]) {
    const file = entry.source.split("#")[0];
    if (!sourceLabels.has(file)) {
        sourceLabels.set(file, new Set());
    }
    sourceLabels.get(file).add(entry.source);
}
const sourceAccounting = Object.fromEntries(
    [...sourceLabels]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([file, labels]) => {
            const sorted = [...labels].sort();
            return [file, { labels: sorted.length, sha256: hash(JSON.stringify(sorted)) }];
        })
);
assert.deepEqual(sourceAccounting, inventory.fixedSourceAccounting, "Fixed manual source accounting changed; review explicit source-label set before updating pins");
const coverageComplete = inventory.pendingInputs.length === 0;
const residuals = [
    ...impactCases
        .filter((entry) => entry.category === "X")
        .map((entry) => ({ kind: "existing", id: entry.id, text: entry.text, pendingIntegration: entry.pendingIntegration ?? null, reason: entry.failedDerivation })),
    ...arabicCases
        .filter((entry) => entry.category === "X")
        .map((entry) => ({ kind: "new-arabic", id: entry.id, text: entry.text, pendingIntegration: entry.pendingIntegration ?? null, reason: entry.failedDerivation })),
];
const attachedRows = impactCases.filter((entry) => entry.category === "A-attach");
const fontPositionRows = impactCases.filter((entry) => ["N-proof", "G0"].includes(entry.category));
const allArabicRows = [...impactCases.filter((entry) => entry.script === "arab"), ...arabicCases];
const c10Impact = allArabicRows
    .filter((entry) => entry.preLookup.changed)
    .map((entry) => ({
        kind: entry.source ? "existing" : "new-arabic",
        id: entry.id,
        text: entry.text,
        treatment: entry.treatment,
        expectedPreLookup: entry.preLookup.expectedLogical,
        records: entry.records,
        exactMatchedHarfbuzz: JSON.stringify(entry.records) === JSON.stringify(existingShaped.get(requestKey(entry))),
    }));
emit("arabic-c10-impact.json", { arabicRows: allArabicRows.length, reorderedRows: c10Impact.length, cases: c10Impact });
function exactReviewedPositions(category, ids) {
    const rows = fontPositionRows.filter((entry) => entry.category === category);
    return (
        JSON.stringify(rows.map((entry) => entry.id).sort()) === JSON.stringify(ids.slice().sort()) &&
        rows.every((entry) => entry.fontPositionCrossCheck.passed && entry.fontPositionCrossCheck.allSixFieldsChecked)
    );
}
const gateConditions = {
    fixedEnumerationProof: true,
    coverageComplete,
    noPendingInputs: inventory.pendingInputs.length === 0,
    zeroExistingAndArabicResiduals: residuals.length === 0,
    exactlyThreeNProofWithExactCrossChecks: exactReviewedPositions("N-proof", N_PROOF_IDS),
    exactlyThreeG0WithExactCrossChecks: exactReviewedPositions("G0", G0_IDS),
    exactlyThreeAAttachWithExactCrossChecks:
        attachedRows.length === 3 && attachedRows.every((entry) => entry.attachmentCrossCheck.passed && entry.attachmentCrossCheck.allSixFieldsDerived),
    acceptedHUnattachedMarkGuard: guards.acceptedHUnattachedNonzeroHmtx.passed,
    devanagariNonzeroMarkGuard: guards.devanagariNonzeroClass3.passed,
    independentJoiningValidation: joiningFailures.length === 0 && allArabicRows.every((entry) => entry.joiningValidated),
    c10PreLookupEveryArabicRow: preLookupFailures.length === 0 && allArabicRows.every((entry) => entry.preLookup.passed && entry.preLookup.cliCtypesAgreed),
    completeC10ImpactWithExactReferences:
        c10Impact.every((entry) => entry.exactMatchedHarfbuzz && entry.treatment === "H-C10") &&
        allArabicRows.every((entry) => entry.flags.includes("C10") === entry.preLookup.changed && !entry.markOrderKeyed),
    restrictedCmapValidation: restrictedFailures.length === 0,
};
const gateResults = {
    conditions: gateConditions,
    eligibleForCoordinatorClearance: Object.values(gateConditions).every(Boolean),
    productionAuthorized: false,
    conditionalTasks: ["T-95", "T-96", "T-99", "T-108 except D10 section 8"],
    externallyRequired: ["two successful --check runs", "focused CPU tests pass", "production diff empty, user baseline unchanged and index empty", "Coordinator authorization"],
    stillGated: { "T-108 section 8": ["D10 admission policy (undecided)"], "T-97": ["T-108 completion"] },
    decisions: { C10: "user-approved; independent reference integration in this checkpoint", D10: "undecided" },
    residuals,
    guards,
    attachmentDerivations: attachedRows.map((entry) => ({ id: entry.id, passed: entry.attachmentCrossCheck.passed, attachedGlyphs: entry.attachmentCrossCheck.attachedGlyphs })),
    fontPositionDerivations: fontPositionRows.map((entry) => ({ id: entry.id, treatment: entry.treatment, passed: entry.fontPositionCrossCheck.passed })),
    arabicValidation: {
        rows: allArabicRows.length,
        uniquePreLookupRequests: preLookupValidation.size,
        preLookupFailures: preLookupFailures.length,
        joiningRequests: joiningValidation.size,
        joiningFailures: joiningFailures.length,
        c10ImpactIds: c10Impact.map((entry) => entry.id),
    },
};
emit("phase1-gate-results.json", gateResults);
emit("existing-impact-report.json", {
    status: "Phase1 C1-C10 independent evidence; Coordinator verification and production authorization required",
    sourceAccountingComplete: true,
    coverageComplete,
    coverage: "Fixed source accounting complete; differential-only titles need no instantiated reference records",
    sourceAccounting,
    fontHashes: existingPins,
    yAdvanceScan,
    correctJoiningValidation: { tested: joiningValidation.size, failed: joiningFailures.length },
    c10PreLookupValidation: { tested: preLookupValidation.size, failed: preLookupFailures.length, rows: allArabicRows.length },
    restrictedCmapValidation: { tested: restrictedRequests.filter((request) => request.restrictedCrossCheck).length, failed: restrictedFailures.length },
    totals,
    treatments,
    perSource,
    cases: impactCases,
    nonReference: inventory.nonReference,
    planIndex,
    pendingInputs: inventory.pendingInputs,
    guards,
    gateResults,
});
emit("coverage-enumeration.json", {
    sourceAccountingComplete: true,
    sourceAccounting,
    coverageComplete,
    status: "Fixed source accounting complete; R1 differential contracts include all runtime-derived edits",
    registered: Object.entries(perSource).map(([source, categories]) => ({ source, shapedRows: Object.values(categories).reduce((sum, count) => sum + count, 0), categories })),
    remaining: [...inventory.pendingInputs.map((entry) => `${entry.file}#${entry.label}: ${entry.reason}`)],
    nonReference: inventory.nonReference,
    planIndex,
    pendingInputs: inventory.pendingInputs,
});
const xCases = impactCases.filter((entry) => entry.category === "X");
emitText(
    "existing-impact-report.md",
    [
        "# Existing corpus impact — Phase 1 review gate",
        "",
        "No Lite selector, plan, compiler, kernel or shaped output is used by the classifier. Font tables and default plans are read using text-shaper 0.1.28.",
        "Nominal and final streams are inspected conservatively; trigger flags do not themselves assert a changed final record.",
        "",
        `Registered shaping cases: **${impactCases.length}**. ${Object.entries(totals)
            .map(([key, value]) => `${key}=${value}`)
            .join(", ")}.`,
        "Repeated identical inputs within a source family are collapsed. Each JSON row includes source, old records, triggers and independent evidence.",
        "",
        "## Per-source classification",
        "",
        "| Source | U | A | A-attach | N-proof | G0 | H | X |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ...Object.entries(perSource).map(
            ([source, counts]) => `| ${source} | ${counts.U} | ${counts.A} | ${counts["A-attach"]} | ${counts["N-proof"]} | ${counts.G0} | ${counts.H} | ${counts.X} |`
        ),
        "",
        "## Coverage limitations — gate remains closed",
        "",
        `Fixed manual source accounting covers ${Object.keys(sourceAccounting).length} files and ${Object.values(sourceAccounting).reduce((n, entry) => n + entry.labels, 0)} exact source labels, including plan-index and reasoned non-reference rows.`,
        `coverageComplete=${coverageComplete}; pendingInputs=${inventory.pendingInputs.length}. R1 differential-only titles compare Lite paths, not independent glyph records, including every capacity-derived edit.`,
        ...inventory.pendingInputs.map((entry) => `- ${entry.file}#${entry.label}: ${entry.expression}`),
        "Coverage status is separate from treatment/gate acceptance. See phase1-gate-results.json for mechanically checkable conditions. Production authorization remains with the Coordinator.",
        "",
        "## Reviewed contracts — resolved, not runtime changes",
        "",
        "- Per the Lead review and OpenType ValueRecord, horizontal runs ignore yAdvance. The complete six-font table scan finds zero nonzero yAdvance records, including all dependency lookups.",
        "- Context records index the sequence modified by prior actions. C4 yields 7/8/9/C, with inherited clusters and no extra context merge.",
        "- Synthetic witness annotations identify discriminators, controls and co-exercised corrections; font bytes and all six-field records remain unchanged.",
        "- Inter A+U+0326 remains H (C1/C2/C5); the generator independently asserts 1309−1413=−104 against HarfBuzz.",
        "- R3 A-attach rows zero only actually attached advances, preserve every other old/hmtx advance and derive every offset independently. Exact root-relative HarfBuzz checks cover every attached glyph; only documented unattached-width/downstream translations may differ.",
        `- R3 prospective H guard: passed=${guards.acceptedHUnattachedNonzeroHmtx.passed}, violating rows=${hGuardViolations.length}. Devanagari guard passed=${guards.devanagariNonzeroClass3.passed}.`,
        ...hGuardViolations.map((entry) => `  - REJECTED prospective H ${entry.id}: ${JSON.stringify(entry.text)}; ${entry.violations.length} unattached nonzero-hmtx marks.`),
        "",
        "## Independent reference validation",
        "",
        `C9: **${joiningFailures.length}/${joiningValidation.size}** independent correct-joining requests differ from HarfBuzz's own joining. The legacy G5 pinning approach is retired and preserved outside the repository.`,
        `C10: **${preLookupFailures.length}/${preLookupValidation.size}** original-input pre-lookup requests fail; every Arabic row is validated. ${c10Impact.length} affected rows are listed with exact references in arabic-c10-impact.json.`,
        `G4: **${restrictedFailures.length}** of ${restrictedRequests.filter((request) => request.restrictedCrossCheck).length} refined-inert cross-checks failed. The subfont maps input scalars plus U+0020 only, as approved.`,
        "See arabic-joining-validation.json, restricted-cmap-validation.json and the per-case normalization evidence. D10 remains pending; no public font rejection is implemented.",
        "",
        "### Every correct-joining validation failure",
        ...joiningFailures.map(([id, entry]) => `- ${id}: ${JSON.stringify(entry.text)}`),
        "",
        "### C9 witness residuals",
        ...arabicWitnesses
            .map((entry) => ({ ...entry, ...arabicTreatment(entry) }))
            .filter((entry) => entry.category === "X")
            .map((entry) => `- ${entry.id}: ${JSON.stringify(entry.text)} — ${entry.failedDerivation}`),
        "",
        "### Every G4 inert cross-check failure",
        ...restrictedFailures.map(
            (entry) => `- ${entry.id}: unrestricted ${JSON.stringify(entry.restrictedCrossCheck.unrestrictedRecords)}, restricted ${JSON.stringify(entry.records)}.`
        ),
        "",
        "## Every X case (proposals only; none applied)",
        "",
        ...xCases.flatMap((entry) => [
            `### ${entry.id}`,
            "",
            `Text: ${JSON.stringify(entry.text)}. Flags: **${entry.flags.join(", ")}**.`,
            `Source: ${entry.source}.`,
            `Synthetic replacements: ${entry.syntheticWitnesses.join(", ")}.`,
            `Failed derivation: ${entry.failedDerivation}. No blanket field split is proposed.`,
            `Reason: ${entry.proposedTreatment.reason}`,
            "Old and info-only HarfBuzz six-field records are preserved in the matching JSON row.",
            "",
        ]),
    ].join("\n") + "\n"
);
const expectedMessages = [
    "start reordering indic initial",
    "end reordering indic initial",
    "start reordering indic final",
    "end reordering indic final",
    "start table GPOS",
    "end table GPOS",
];
const observed = new Set([...results.values()].flatMap((result) => result.pauseMessages));
const sources = [
    "scripts/generate-shaping-references.mjs",
    "scripts/shaping-references-ctypes.py",
    "scripts/shaping-references-fixtures.ts",
    "tests/lite/compute/synthetic.ts",
    "tests/lite/compute/references/devanagari-corpus.mjs",
    "tests/lite/compute/references/synthetic-corpus.mjs",
    "tests/lite/compute/corpus.ts",
    "tests/lite/compute/corpus-inventory.ts",
    "scripts/shaping-references-corpus.ts",
    "scripts/shaping-reference-analysis.mjs",
    "scripts/shaping-reference-derivations.mjs",
    "scripts/shaping-reference-unicode.mjs",
    "scripts/shaping-reference-attachments.mjs",
    "scripts/shaping-reference-cmap.mjs",
    "scripts/shaping-reference-indic-projection.mjs",
    "scripts/shaping-reference-outputs.mjs",
    "tests/lite/compute/references/arabic-corpus.mjs",
];
const identityPath = resolve(destination, "reference-identity.json");
const timestamp = write ? new Date().toISOString() : JSON.parse(readFileSync(identityPath, "utf8")).generatedAt;
emit("reference-identity.json", {
    generatedAt: timestamp,
    existingImpactReportSha256: hash(pendingOutputs.get("existing-impact-report.json")),
    existingImpactReportMarkdownSha256: hash(pendingOutputs.get("existing-impact-report.md")),
    harfbuzz: { version: "11.0.1", binary, library, binarySha256: pins.binary, librarySha256: pins.library, sourceCommit: "9f3deba7" },
    textShaper: { version: "0.1.28", entrySha256: fileHash(resolve(root, "packages/babylon-lite/node_modules/text-shaper/dist/index.js")) },
    font: { file: "NotoSansDevanagari-Regular.ttf", sha256: pins.font, licenseSha256: pins.license },
    configuration: {
        shaper: "ot",
        fontFuncs: "ot",
        scale: "upem",
        ppem: 0,
        variations: [],
        script: "Deva",
        language: "und",
        direction: "ltr",
        clusterLevel: 1,
        flags: 0,
        features: [],
    },
    generatorSources: Object.fromEntries(sources.map((path) => [path, fileHash(resolve(root, path))])),
    corpusSourceHashes: Object.fromEntries(
        [...new Set([...existingCases, ...inventory.nonReference, ...planIndex].map((entry) => entry.source.split("#")[0]))]
            .sort()
            .map((path) => [path, fileHash(resolve(root, path))])
    ),
    unavailablePauseMessages: expectedMessages.filter((message) => !observed.has(message)),
    observedPauseMessages: [...observed].sort(),
    existingFontHashes: existingPins,
    unicode: joiningData.identities,
    cliCtypesAgreement: {
        checked: requests.length + b1Requests.length + syntheticCount + existingRequests.length + joiningValidationRequests.length + preLookupRequests.length,
        total: requests.length + b1Requests.length + syntheticCount + existingRequests.length + joiningValidationRequests.length + preLookupRequests.length,
        indicCanonicalPathChecks: b1Requests.length,
        c10RestrictedCliCtypes: preLookupRequests.length,
        restrictedCmapInertCrossChecks: restrictedRequests.filter((request) => request.restrictedCrossCheck).length,
        restrictedCmapNonInertRequests: restrictedRequests.filter((request) => !request.restrictedCrossCheck).length,
    },
});
const impactBytes = pendingOutputs.get("existing-impact-report.json");
const impactForCompact = JSON.parse(impactBytes);
assert.ok(
    impactForCompact.cases.every((entry) => entry.category !== "X" && entry.records?.every((record) => record.length === 6 && record.every(Number.isInteger))),
    "Unaccepted/incomplete diagnostic row cannot be hidden by compact expectation filtering"
);
emit("reviewed-expectations.json", {
    schemaVersion: 1,
    sourceReportSha256: hash(impactBytes),
    referenceIdentitySha256: hash(pendingOutputs.get("reference-identity.json")),
    fontHashes: impactForCompact.fontHashes,
    cases: impactForCompact.cases
        .filter((entry) => entry.category !== "U")
        .map(({ id, font, text, script, direction, category, flags, records }) => ({ id, font, text, script, direction, category, flags, records })),
});
// CPU-only audit invariants are not runtime expectations. Preserve the old
// assertions without importing the diagnostic report into every test worker.
// Unchanged arrays are represented by two independent digests and are checked
// against the live pinned adapter by the audit test, not reconstructed from Lite.
const auditKeys = [
    "id",
    "font",
    "text",
    "source",
    "script",
    "direction",
    "currentlyRejected",
    "category",
    "flags",
    "records",
    "oldRecords",
    "configuration",
    "c1ArithmeticInfo",
    "syntheticWitnesses",
    "treatment",
    "failedDerivation",
    "proposedTreatment",
    "fontPositionCrossCheck",
    "attachmentParentsLogical",
    "attachmentCrossCheck",
    "preLookup",
    "markOrderKeyed",
    "harfbuzzRecords",
];
const auditConfigurations = [];
const auditConfigurationIds = new Map();
emit("impact-validation.json", {
    sourceReportSha256: hash(impactBytes),
    referenceIdentitySha256: hash(pendingOutputs.get("reference-identity.json")),
    coverageComplete: impactForCompact.coverageComplete,
    sourceAccountingComplete: impactForCompact.sourceAccountingComplete,
    totals: impactForCompact.totals,
    nonReference: impactForCompact.nonReference,
    planIndex: impactForCompact.planIndex,
    pendingInputs: impactForCompact.pendingInputs,
    guards: impactForCompact.guards,
    yAdvanceScan: impactForCompact.yAdvanceScan,
    configurations: auditConfigurations,
    cases: impactForCompact.cases.map((entry) => {
        const audit = Object.fromEntries(auditKeys.filter((key) => key in entry).map((key) => [key, entry[key]]));
        const configurationKey = JSON.stringify(stable(entry.configuration));
        if (!auditConfigurationIds.has(configurationKey)) {
            auditConfigurationIds.set(configurationKey, auditConfigurations.length);
            auditConfigurations.push(entry.configuration);
        }
        audit.configurationIndex = auditConfigurationIds.get(configurationKey);
        delete audit.configuration;
        if (entry.category === "U") {
            assert.deepEqual(entry.records, entry.oldRecords, `${entry.id}: unchanged audit is not unchanged`);
            audit.recordsSha256 = hash(JSON.stringify(entry.records));
            audit.oldRecordsSha256 = hash(JSON.stringify(entry.oldRecords));
            delete audit.records;
            delete audit.oldRecords;
        } else if (["A-attach", "N-proof", "G0"].includes(entry.category) || entry.script === "arab") {
            audit.info_only = entry.info_only;
        }
        return audit;
    }),
});
if (write) {
    for (const [file, bytes] of pendingOutputs) {
        writeFileSync(resolve(destination, file), bytes);
    }
} else {
    checkReferenceOutputs(destination, pendingOutputs);
}
console.warn(
    `Reference bytes ${write ? "written" : "verified"}; C9 joining failures ${joiningFailures.length}/${joiningValidation.size}; C10 pre-lookup failures ${preLookupFailures.length}/${preLookupValidation.size}; G4 failures ${restrictedFailures.length}. Impact: ${JSON.stringify(totals)}.`
);
assert.equal(preLookupFailures.length, 0, `C10 original-input pre-lookup validation failed: ${preLookupFailures.map(([id]) => id).join(", ")}; no tuning or weakened fields`);
assert.equal(
    joiningFailures.length,
    0,
    `C9 joining validation failed on ${joiningFailures.length}/${joiningValidation.size} requests; report Unicode/transcription mismatches without fitting forms.`
);
assert.equal(Object.values(gateConditions).every(Boolean), true, "Phase1 source gate failed; see phase1-gate-results.json");
assert.equal(
    guards.acceptedHUnattachedNonzeroHmtx.passed,
    true,
    `R3 mandatory accepted-H guard failed on ${hGuardViolations.length} rows; see phase1-gate-results.json. No fitted expectations or field weakening.`
);
assert.equal(guards.devanagariNonzeroClass3.passed, true, "R3 Devanagari nonzero-class3 guard failed");
assert.equal(
    restrictedFailures.length,
    0,
    `G4 restricted-cmap inert cross-check failed on ${restrictedFailures.length} requests; STOP for review. See restricted-cmap-validation.json.`
);
