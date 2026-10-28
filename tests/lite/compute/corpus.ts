/** CPU-only registry of the M1–3 comparison inputs. No Lite selector, plan or output.
 * Sources identify test families; repeated identical rows within a family are collapsed.
 * The Coordinator-owned feature board, not this registry, tracks task status. */
import { appendReviewedInputs, DIFFERENTIAL_CASES } from "./corpus-inventory";
export const GREEK = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωΑθήναςϊϋΐ";
export const CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяМоскваІіЇїЄє";

/** Moved byte-for-byte from fixtures.ts; seed and UTF-16 alphabet indexing unchanged. */
export function seededStrings(alphabet: string, count = 64): string[] {
    let state = 0x4d313233;
    return Array.from({ length: count }, (_, run) =>
        Array.from({ length: 2 + (run % 49) }, () => {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return alphabet[state % alphabet.length]!;
        }).join("")
    );
}

export interface ExistingCase {
    id: string;
    font: string;
    text: string;
    source: string;
    direction: "ltr" | "rtl";
    script: string;
    currentlyRejected?: boolean;
}
export interface CorpusFont {
    glyphId: (scalar: number) => number;
}

/** Independent script selection for the homogeneous existing corpus. */
export function corpusSelection(text: string): { script: string; direction: "ltr" | "rtl" } {
    const scripts = [
        ["latn", /\p{Script=Latin}/u],
        ["grek", /\p{Script=Greek}/u],
        ["cyrl", /\p{Script=Cyrillic}/u],
        ["hebr", /\p{Script=Hebrew}/u],
        ["arab", /\p{Script=Arabic}/u],
        ["hani", /\p{Script=Han}/u],
        ["goth", /\p{Script=Gothic}/u],
    ] as const;
    const matches = scripts.filter(([, pattern]) => pattern.test(text));
    if (matches.length > 1) {
        throw new Error(`Mixed-script comparison input: ${JSON.stringify(text)}`);
    }
    const script = matches[0]?.[0] ?? "DFLT";
    return { script, direction: script === "hebr" || script === "arab" ? "rtl" : "ltr" };
}

/** Font mapping comes only from the independent reference parser. */
export function existingCorpus(fonts: ReadonlyMap<string, CorpusFont>): ExistingCase[] {
    const result: ExistingCase[] = [];
    const seen = new Set<string>();
    const ordinals = new Map<string, number>();
    function add(file: string, family: string, font: string, texts: readonly string[], currentlyRejected = false): void {
        if (family === "resident updates preserve status-only mapping across long/short transitions: %s") {
            family = "resident shared owner grows, shrinks, reuses and disposes: %s";
        }
        if (family === "preserves RTL ordering and stacked-mark offsets") {
            family = file.includes("parallel-refresh")
                ? "retains RTL order and mark offsets in both scratch variants"
                : "preserves RTL ordering and stacked-mark offsets with bounded and global scratch";
        }
        const source = `${file}#${family.replace("%s", font).replace("%i", "2048")}`;
        if (DIFFERENTIAL_CASES.some((entry) => entry.source === source)) {
            return;
        }
        for (const text of texts) {
            const key = JSON.stringify([source, font, text]);
            if (seen.has(key)) {
                continue;
            }
            seen.add(key);
            const group = `${source}:${font}`;
            const index = ordinals.get(group) ?? 0;
            ordinals.set(group, index + 1);
            // These unit helpers explicitly pass Latn even for a space or an
            // Inherited-only mark pair; they do not use public auto-selection.
            const selection =
                file === "tests/lite/unit/compute-text-shaper.test.ts" || file === "tests/lite/unit/compute-text-program.test.ts"
                    ? { script: "latn", direction: "ltr" as const }
                    : corpusSelection(text);
            result.push({
                id: `${source}:${font}:${index}`,
                font,
                text,
                source,
                ...selection,
                ...(currentlyRejected ? { currentlyRejected: true } : {}),
            });
        }
    }
    const native = (name: string) => `tests/lite/compute/${name}.test.ts`;
    const mapped = (font: string, start: number, end: number) =>
        Array.from({ length: end - start + 1 }, (_, i) => start + i)
            .filter((cp) => fonts.get(font)!.glyphId(cp) !== 0)
            .map((cp) => String.fromCodePoint(cp));
    for (const font of ["Inter.ttf", "Roboto-Regular.ttf"]) {
        add(native("milestones"), "Latin repertoire and ordered witnesses: %s", font, [
            "A",
            "AC",
            "5,",
            "To",
            "AV",
            "fi",
            "ffi",
            "office",
            "Café",
            "Æther",
            "Łódź",
            "1 < A",
            "A\u0326",
            "B\u0301",
            "B\u0301\u0307",
            "i\u0301",
            "abb",
            "aba",
            "ffif",
            "B\u0300\u0301",
        ]);
        add(native("milestones"), "Latin-1/Extended scalar repertoire and seeded sequences: %s", font, [
            ...mapped(font, 0x20, 0x7e),
            ...mapped(font, 0xa0, 0x24f),
            ...mapped(font, 0x1e00, 0x1eff),
            ...mapped(font, 0x2c60, 0x2c7f),
            ...mapped(font, 0xa720, 0xa7ff),
            ...mapped(font, 0xab30, 0xab6f),
            ...mapped(font, 0x2000, 0x200b),
            ...mapped(font, 0x2010, 0x2027),
            ...seededStrings("ABCDEFXYTabcfivxy 0123456789.,:;!?"),
            "A\u0308",
            "e\u0301",
            "a\u0301",
            "a\u0300",
            "a\u0302",
            "a\u0303",
            "a\u0304",
            "a\u0306",
            "a\u0307",
            "B\u0326",
            "A\u0327v",
            "B\u0301\u0300",
            "B\u0301\u0307",
            "a\u0301b",
            "1 < A",
        ]);
        add(native("milestones"), "Greek and Cyrillic: %s", font, [...GREEK, ...CYRILLIC, GREEK, CYRILLIC]);
        add(native("kernel"), "keeps exact marked records across anchor search boundaries and global scratch: %s", font, [
            "A\u0326",
            "H\u0326\u0301",
            "A \u0326",
            "office fi To A\u0326 HH ".repeat(3),
            "A\u0326 ".repeat(100),
        ]);
        add(native("long-run"), "finishes proven long nominal runs without scratch fallback: %s", font, ["HH ".repeat(100), "AV To A", "", "H".repeat(1025)]);
        const boundaryTexts: string[] = [];
        for (const length of [255, 256, 257, 300, 600, 1025]) {
            boundaryTexts.push(
                ...["HH ", "AV To ", "office fi ", "A\u0326 ", GREEK, CYRILLIC].map((seed) => seed.repeat(Math.ceil(length / seed.length)).slice(0, length)),
                "",
                "HH ".repeat(100),
                "HH HH H",
                "AV To A",
                "office ",
                " A fi H"
            );
        }
        add(native("long-run"), "matches all six oracle fields across scratch/tile boundaries: %s", font, boundaryTexts);
        add(
            native("long-run"),
            "resident updates preserve status-only mapping across long/short transitions: %s",
            font,
            [256, 257, 7, 600, 255, 1025].map((n) => "H".repeat(n)).concat(["AV To A", "A\u0326", GREEK, CYRILLIC, ""])
        );
        const ordered: string[] = [];
        for (const length of [64, 256, 257, 600]) {
            ordered.push(
                ...["ffi ffl fi fffi ffli office affinity ", "A\u0326\u0301\u0300 H\u0308\u0301 ", "fi\u0326\u0301 ffl\u0308 "].map((seed) =>
                    seed.repeat(Math.ceil(length / seed.length)).slice(0, length)
                )
            );
            if (length > 256) {
                ordered.push("H".repeat(255) + "fi" + "H".repeat(length - 257));
            }
        }
        add(native("ordered-parallel"), "retains overlapping ffi/ffl/fi precedence, marks and boundary-crossing ligatures: %s", font, ordered);
        add(
            native("parallel-refresh"),
            "compacts real boundary-crossing ligatures exactly: %s",
            font,
            [31, 32, 33, 64, 255, 256, 257, 480, 768].map((count) => {
                const text = Array<string>(count).fill("H");
                for (const at of [29, 31, 62, 254, 256, 510, 600]) {
                    if (at + 1 < count) {
                        text[at] = "f";
                        text[at + 1] = "i";
                    }
                }
                return text.join("");
            })
        );
        add(native("resident"), "resident records, packing and indirect: %s", font, [
            "",
            " ",
            "  ",
            "fi",
            "A\u0308",
            "B\u0301\u0307",
            "1 < A",
            ...[2, 4, 8, 16, 32, 50, 49, 50, 2].flatMap((n) => ["A".repeat(n), "fi ".repeat(Math.floor(n / 2))]),
            GREEK,
            CYRILLIC,
            "A",
            "A".repeat(512),
        ]);
    }
    for (const [font, start, end] of [
        ["NotoSansSC-Regular-subset.ttf", 0x3400, 0x9fff],
        ["NotoSansGothic-Regular.ttf", 0x10330, 0x1034a],
    ] as const) {
        const repertoire = mapped(font, start, end);
        add(native("milestones"), "CJK and supplementary Gothic", font, [...repertoire, repertoire.join("")]);
    }
    for (const [font, start, end, alphabet, witnesses, words] of [
        ["NotoSansHebrew-Regular.ttf", 0x5d0, 0x5ea, "אבגדהוזחטיכלמנסעפצקרשת", ["אָ", "שׁ", "שָׁ", "בְ", "תּ"], ["שלום", "אבגד", "עברית", "שָׁלוֹם", "אב אב"]],
        [
            "NotoSansArabic-Regular.ttf",
            0x620,
            0x64a,
            "ابتجحخدذرزسشصضطظعغفقكلمنهوي",
            ["بَ", "بِ", "بُ", "بْ", "بّ", "بَّ", "بب ب", "لا"],
            ["كتب", "سلام", "بب", "ببب", "ابت", "كُتُب", "العربية"],
        ],
    ] as const) {
        add(native("milestones"), "Hebrew and Arabic", font, words);
        add(native("milestones"), "mapped RTL repertoire and seeded joining/marks: %s", font, [...mapped(font, start, end), ...seededStrings(alphabet), ...witnesses]);
    }
    add(native("milestones"), "batch ordering and lengths", "Roboto-Regular.ttf", [
        "",
        " ",
        "  ",
        "A",
        "Αθήνα",
        "Москва",
        ...seededStrings("ABCDfioxv ,.0123456789", 100),
        "A".repeat(512),
        "A".repeat(1024),
    ]);
    const phrases = ["The efficient flight lifts off at five. ", "WAVY typography: coffee, waffles and films. ", "A quick brown fox visits the office today. "];
    add(
        native("milestones"),
        "completes every run and six fields in a natural Latin50 batch of %i",
        "Roboto-Regular.ttf",
        Array.from({ length: 2048 }, (_, run) => phrases[run % phrases.length]!.repeat(2).slice(0, 46) + String(run).padStart(4, "0"))
    );
    add(native("milestones"), "fail-closed corpus", "Inter.ttf", ["A\u20de", "A\u20dd", "<-"], true);
    add(native("kernel"), "preserves original independent blocks and activates non-leading composition/context witnesses", "Inter.ttf", [
        ...Array.from({ length: 100 }, (_, run) => [`Block ${String(run).padStart(2, "0")} office AVATAR`, "quick fox 0140 ffi fl To"]).flat(),
        "A\u0301",
        "a\u0306",
        "i\u0307",
        "1 < A",
        "H\u0301",
    ]);
    add(
        native("kernel"),
        "finishes proven nominal/pair runs without executing the complete mutation kernel and retains contextual fallback",
        "Inter.ttf",
        Array.from({ length: 40 }, (_, run) => [`Block ${String(run).padStart(2, "0")} office AVATAR`, "quick fox 0140 ffi fl To"]).flat()
    );
    add(native("kernel"), "refreshes root presence after a nested Arabic substitution activates positioning", "NotoSansArabic-Regular.ttf", ["ج", "جج"]);
    add(native("kernel"), "derives initial scalar-index clusters from compact bulk input including supplementary and empty runs", "NotoSansGothic-Regular.ttf", ["𐌰𐌱", "𐌲", ""]);
    add(native("long-run"), "preserves long RTL clusters and ordered mark attachment", "NotoSansHebrew-Regular.ttf", ["אבגד".repeat(150), "שָׁלוֹם ".repeat(80)]);
    for (const [file, repeats] of [
        ["ordered-parallel", [10, 60]],
        ["parallel-refresh", [10, 80]],
    ] as const) {
        add(
            native(file),
            "preserves RTL ordering and stacked-mark offsets",
            "NotoSansHebrew-Regular.ttf",
            repeats.map((n) => "שָׁלוֹם ".repeat(n))
        );
    }
    for (const [font, texts] of [
        ["NotoSansHebrew-Regular.ttf", ["שלום", "שָׁלוֹם", "אבגד"]],
        ["NotoSansArabic-Regular.ttf", ["كتب", "سلام", "بَّ", "العربية"]],
        ["NotoSansGothic-Regular.ttf", ["𐌰𐌱𐌲", "𐌳"]],
        ["NotoSansSC-Regular-subset.ttf", ["中文", "中"]],
    ] as const) {
        add(native("resident"), "resident RTL/scalar repertoire: %s", font, texts);
    }
    add(native("resident"), "rejection clears every indirect word and recovery keeps the caller atlas", "Inter.ttf", ["B"]);
    for (const [file, texts] of [
        ["Roboto-Regular.ttf", ["fi", "A", "To", " "]],
        ["Inter.ttf", ["A", "AC", "5,", "A\u0326", "\u0301\u0300"]],
    ] as const) {
        add("tests/lite/unit/compute-text-shaper.test.ts", "matches text-shaper for Roboto Latin and executable Inter kerning and marks", file, texts);
    }
    for (const [font, texts] of [
        ["Roboto-Regular.ttf", ["fi"]],
        ["Inter.ttf", ["AC", "A\u0326", "Αθήνα", "Москва"]],
        ["NotoSansHebrew-Regular.ttf", ["שלום"]],
        ["NotoSansArabic-Regular.ttf", ["كتب"]],
        ["NotoSansSC-Regular-subset.ttf", ["汉字"]],
        ["NotoSansGothic-Regular.ttf", ["𐌰𐌱𐌲"]],
    ] as const) {
        add("lab/lite/src/lite/compute-text-shaper-test.ts", "run", font, texts);
    }
    appendReviewedInputs(add, GREEK, CYRILLIC);
    return result;
}
