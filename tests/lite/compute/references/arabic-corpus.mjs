/** Correct-joining C9 witnesses; no legacy-form or Lite-output pinning. */
export function arabicCorpus() {
    const groups = [
        ["isolated", ["ب", "ج", "ت", "ا", "د"]],
        ["joining", ["بب", "ببب", "كتب"]],
        ["right-break", ["ابت", "دار"]],
        ["transparent", ["بَب", "كُتُب", `ب${"\u064e".repeat(100)}ب`]],
        ["tatweel", ["بـب", "ـب"]],
        ["rlig", ["لا", "سلام"]],
        ["stacked", ["بَّ"]],
        ["space-break", ["ب ب"]],
        ["digits", ["١٢"]],
        ["c10", ["بِّ", "بُّ", "بّْ", "بَِ", "بَٔ", "بَّٔ", `ب\u064e${"\u0651".repeat(32)}`]],
        ["c10-both", ["\u0628\u0654\u0655", "\u0628\u0650\u0655\u064e\u0654"]],
    ];
    const seed = "كتب سلام دار ";
    return [
        ...groups.flatMap(([group, texts]) =>
            texts.map((text, i) => ({ id: `arabic-${group}-${i}`, text, font: "NotoSansArabic-Regular.ttf", script: "arab", direction: "rtl", expected: "positive" }))
        ),
        {
            id: "arabic-long-600",
            text: seed.repeat(Math.ceil(600 / seed.length)).slice(0, 600),
            font: "NotoSansArabic-Regular.ttf",
            script: "arab",
            direction: "rtl",
            expected: "positive",
        },
    ];
}
