/** Declarative Devanagari paragraph corpus (CTF-03). Scalar-index rows; never normalize these strings.
 *
 * `proven` lists the U+0020 indexes the paragraph space proof is expected to mark safe:
 * a candidate space s (T[s] = U+0020, T[s + 1] exists and is not U+0020) whose right
 * neighbour starts an extended grapheme cluster and whose left and right words each have
 * at most 64 scalars. Test-only evidence: production and lab modules must never import it.
 */

export const PARAGRAPH_WORD_CAP = 64;

/** Deterministic repetition of a unit sentence, joined by single spaces. */
export function repeat(unit, count) {
    return Array.from({ length: count }, () => unit).join(" ");
}

export const MULTISEAM_UNIT = "भारत एक विशाल देश है और यहाँ कई भाषाएँ बोली जाती हैं";
export const MULTISEAM_FORCED_COUNT = 6;
export const MULTISEAM_DEFAULT_COUNT = 900;

const MULTISEAM_UNIT_PROVEN = [4, 7, 13, 17, 20, 23, 28, 31, 38, 43, 48];

/** Unit spaces repeat per instance; the joining spaces sit between two short plain words. */
function repeatProven(count) {
    const stride = [...MULTISEAM_UNIT].length + 1;
    return Array.from({ length: count }, (_, i) => [...MULTISEAM_UNIT_PROVEN.map((s) => s + i * stride), ...(i + 1 < count ? [i * stride + stride - 1] : [])]).flat();
}

const OVERLONG_WORD = "कमल".repeat(24);

function row(id, tags, text, proven, note) {
    return Object.freeze({ id, tags: Object.freeze(tags), text, proven: Object.freeze(proven), ...(note ? { note } : {}) });
}

export const devanagariParagraphCorpus = Object.freeze([
    row("plain-short", ["plain"], "नमस्ते दुनिया", [6]),
    row("plain-country", ["plain"], "भारत एक विशाल देश है और यहाँ कई भाषाएँ बोली जाती हैं", [4, 7, 13, 17, 20, 23, 28, 31, 38, 43, 48]),
    row("plain-village", ["plain"], "हम सब मिलकर अपने गाँव को साफ और सुंदर बनाएँगे", [2, 5, 11, 16, 21, 24, 28, 31, 37]),
    row(
        "plain-rain",
        ["plain"],
        "आज सुबह से ही आसमान में काले बादल छाए हुए थे और ठंडी हवा चल रही थी इसलिए बच्चे खिड़की के पास बैठकर बारिश का इंतजार कर रहे थे",
        [2, 7, 10, 13, 19, 23, 28, 33, 37, 41, 44, 47, 52, 56, 59, 63, 66, 72, 78, 85, 88, 92, 98, 104, 107, 114, 117, 121]
    ),
    row("plain-numbers", ["plain", "digits"], "कक्षा 5 में 32 बच्चे और १२ शिक्षक हैं", [5, 7, 11, 14, 20, 23, 26, 33]),
    row("conjunct-ksha", ["conjunct"], "क्षमा रक्षा पक्ष", [5, 11]),
    row("conjunct-tra", ["conjunct"], "त्रिकोण पत्र मित्र", [7, 12]),
    row("conjunct-jna", ["conjunct"], "ज्ञान विज्ञान यज्ञ", [5, 13]),
    row("conjunct-sta", ["conjunct"], "स्तर नमस्ते अस्त व्यस्त स्त्री", [4, 11, 16, 23]),
    row("reph", ["reph"], "कर्म ही धर्म है और सूर्य का प्रकाश सर्वत्र है पूर्व में दर्शन", [4, 7, 12, 15, 18, 24, 27, 34, 42, 45, 51, 55]),
    row("prebase-edge", ["prebase-edge"], "कि शक्ति हानि किताब निधि दिन", [2, 8, 13, 19, 24]),
    row("nukta", ["nukta"], "\u0958लम \u095Bमीन \u0915\u093Cलम फ़ायदा", [3, 8, 13]),
    row("joiner-inside", ["joiner"], "पक्\u200Dषी अक्\u200Cषर", [6]),
    row("joiner-space-before", ["joiner"], "पक्षी \u200Dक अक्षर \u200Cक", [8], "D1: U+200D (GCB ZWJ) and U+200C (GCB Extend) after the spaces at 5 and 14 are not grapheme starts"),
    row("digits-punctuation", ["digits"], "₹500 (पाँच सौ), कीमत: 12.5? हाँ! ठीक - सही", [4, 10, 15, 21, 27, 32, 36, 38]),
    row("digits-devanagari", ["digits"], "०१२३४५६७८९ और 0123456789", [10, 13]),
    row("danda", ["danda"], "राम घर गया। सीता भी गई॥ फिर सब खुश थे।", [3, 6, 11, 16, 19, 23, 27, 30, 34]),
    row("nbsp", ["nbsp"], "श्री\u00A0राम का मंदिर १०\u00A0किलो दूर है", [8, 11, 17, 25, 29]),
    row("overlong-between", ["overlong"], `नमस्ते ${OVERLONG_WORD} दुनिया हिन्दी`, [86], "D2: the spaces at 6 and 79 border a 72-scalar word"),
    row("overlong-single", ["overlong"], `${OVERLONG_WORD}कमलकमल`, [], "D2: one 78-scalar word with no spaces"),
    row(
        "overlong-boundary",
        ["overlong"],
        `${"क".repeat(64)} ${"ख".repeat(64)} ${"ग".repeat(65)} घ`,
        [64],
        "D2: words of exactly 64 scalars are provable; the spaces at 129 and 195 border a 65-scalar word"
    ),
    row("spaces", ["spaces"], "  नमस्ते  दुनिया   हिन्दी  ", [1, 9, 18], "Only the last space of each run is a candidate; the trailing spaces have no right neighbour"),
    row("spaces-only", ["spaces"], "    ", [], "No candidate spaces"),
    row(
        "space-before-mark",
        ["spaces", "prebase-edge"],
        "क \u093Fक ख \u094Dग घ \u200Dच छ \u20F0ज झ",
        [4, 9, 14, 19],
        "D1: the spaces at 1, 6, 11 and 16 are followed by U+093F, U+094D, U+200D and U+20F0, which are not grapheme starts"
    ),
    row("vedic-crossing", ["reference-only"], "क \u1CD0ख ग", [4], "Exercises GPOS lookup 3 (Vedic tone after space); U+1CD0 is not admitted by validateIndicRun"),
    row("multiseam-unit", ["plain", "multiseam"], MULTISEAM_UNIT, MULTISEAM_UNIT_PROVEN),
    row("multiseam-forced", ["multiseam"], repeat(MULTISEAM_UNIT, MULTISEAM_FORCED_COUNT), repeatProven(MULTISEAM_FORCED_COUNT)),
    row("multiseam-default", ["multiseam"], repeat(MULTISEAM_UNIT, MULTISEAM_DEFAULT_COUNT), repeatProven(MULTISEAM_DEFAULT_COUNT)),
]);

/** Common-only paragraph intended for the DFLT selection in mixed calls (REQ-MIX-2). */
export const devanagariParagraphDfltRows = Object.freeze([
    row("mixed-common", ["mixed"], "12, 34. (56) 7-8? 9! 10", [], "DFLT selection: the indic proof does not apply; line keys use every candidate boundary"),
]);
