/** Declarative scalar-index corpus. Never normalize these strings. */
export const SEED = 0x4d34;
export const ALPHABET = [..."कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूृेैोौ़्ं‍‌ ।"];

export function seededSyllables() {
    let state = SEED;
    const next = () => {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        return state >>> 0;
    };
    return Array.from({ length: 200 }, () => Array.from({ length: 1 + (next() % 12) }, () => ALPHABET[next() % ALPHABET.length]).join(""));
}

const families = [
    ["matra", ["कि", "की", "कु", "कृ", "के", "कै", "को", "कौ"]],
    ["conjunct", ["क्ष", "द्ध", "श्र", "त्र", "ह्म", "स्त्री", "क्त"]],
    ["rakar", ["प्र", "ट्र"]],
    ["reph", ["र्क", "र्कि", "र्कं", "कर्म", "धर्मि", "र् क", "र्अ"]],
    ["joiner", ["क्‍ष", "क्‍", "क्‌ष", "क्‌", "र्‍य", "र्‍"]],
    ["nukta", ["क़", "क़", "ऩ", "ऩ", "क़्त", "क़्त"]],
    ["broken", ["ि", "ँ", "क ि", "अा", "आॅ", "उु", "एॅ", "र्इ"]],
    ["stack", ["किं", "शृंगार"]],
    ["digits", ["१२३।", "12।"]],
    ["sentence", ["नमस्ते दुनिया", "हिन्दी में लिखा हुआ पाठ"]],
    ["normalization-bridge", ["ऩे", "ऩ्न", "ऴ्य", "ऱ्", "ऴ", "ऱ"]],
    ["context-return", ["नद्ध", "द्ध द्ध"]],
    ["browser", ["हिन्दी"]],
];

// All 22 sequences, independently transcribed from the Devanagari branch of
// HarfBuzz 11.0.1 hb-ot-shaper-vowel-constraints.cc. Five lead-scalar families.
export const VOWEL_CONSTRAINTS = [
    ...[0x93a, 0x93b, 0x93e, 0x945, 0x946, 0x949, 0x94a, 0x94b, 0x94c, 0x94f, 0x956, 0x957].map((cp) => `अ${String.fromCodePoint(cp)}`),
    ...[0x93a, 0x945, 0x946, 0x947, 0x948].map((cp) => `आ${String.fromCodePoint(cp)}`),
    "उु",
    ...[0x945, 0x946, 0x947].map((cp) => `ए${String.fromCodePoint(cp)}`),
    "र्इ",
];

function longRun(length) {
    const sentence = "नमस्ते दुनिया हिन्दी में लिखा हुआ पाठ ";
    const full = sentence.repeat(Math.ceil(length / [...sentence].length));
    const prefix = [...full].slice(0, length).join("");
    // Exact capacity witnesses, without cutting a word. Remaining scalars are spaces.
    const boundary = prefix.lastIndexOf(" ");
    return prefix.slice(0, boundary + 1).padEnd(length, " ");
}

export function devanagariCorpus() {
    const positive = (id, text, extra = {}) => ({ id, text, expected: "positive", profile: "deva", ...extra });
    const standalone = Array.from({ length: 128 }, (_, i) => i + 0x900)
        .filter((cp) => ![0x951, 0x952, 0x953, 0x954, 0x964, 0x965].includes(cp))
        .map((cp) => positive(`scalar-${cp.toString(16)}`, String.fromCodePoint(cp)));
    return [
        ...standalone,
        ...[0x951, 0x952, 0x953, 0x954, 0x964, 0x965].map((cp) => positive(`weak-after-base-${cp.toString(16)}`, `क${String.fromCodePoint(cp)}`)),
        positive("inherited-before-base", "॑क"),
        ...families.flatMap(([family, texts]) => texts.map((text, i) => positive(`${family}-${i}`, text))),
        ...VOWEL_CONSTRAINTS.map((text, i) => positive(`vowel-constraint-${i}`, text)),
        ...seededSyllables().map((text, i) => positive(`seeded-${i}`, text)),
        ...["।", "॥", "।॥ 12"].map((text, i) => positive(`weak-only-${i}`, text, { profile: "DFLT", reference: "text-shaper" })),
        ...[512, 1024, 2048].map((n) => positive(`long-${n}`, longRun(n))),
        ...[
            ["vedic", "कᳲ", "unsupported-scalar"],
            ["extended", "ꣲ", "unsupported-script"],
            ["soft-hyphen", "\u00ad", "default-ignorable"],
            ["zero-width-space", "\u200b", "default-ignorable"],
            ["dotted-circle", "\u25cc", "user-dotted-circle"],
            ["latin-mixed", "aक", "mixed-script"],
            ["bengali-mixed", "কक", "mixed-script"],
            ["joiner-only", "\u200d", "joining-control"],
            ["inherited-only", "\u0951", "inherited-only"],
        ].map(([id, text, category]) => ({ id: `negative-${id}`, text, expected: `negative:${category}`, profile: "deva" })),
    ];
}
