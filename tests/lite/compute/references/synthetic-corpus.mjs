/** PUA marks have no canonical decomposition or composition partners. */
export const SYNTHETIC_CMAP = [0x41, 0x42, 0x43, 0xe000, 0xe001];
export const SYNTHETIC_WITNESSES = [
    ["c1-mark-base", "C1", ["A\uE000", "AB\uE000"], "Mark attachment offsets are relative to the current pen, not the base origin."],
    ["c1-mark-mark-chain", "C1", ["A\uE000\uE001B"], "Attachment chains propagate after the final pair adjustment changes the root advance."],
    ["c2-filter-set", "C2", ["A\uE000\uE001\uE000", "A\uE000\uE000"], "MarkMark skips an excluded mark to an included target; zero-advance marks avoid C1/C3 confounding."],
    ["c3-pair-placement", "C3", ["ABC"], "Full value1 placement is applied and valueFormat2=0 permits the overlapping next pair."],
    ["c3-pair-value2", "C3", ["ABC"], "The source contains all four value2 fields; HarfBuzz applies horizontal advances only, and continuation skips the second glyph."],
    ["c4-nested-ligature", "C4", ["A\uE000BC"], "A nested ligature acts at the matched B after the skipped mark, not at raw index+1."],
    [
        "c4-nested-multiple",
        "C4",
        ["ABC"],
        "After A expands to 7/8, the later sequence-index-2 Single applies to B in the modified sequence, not original C; clusters are inherited without an extra context merge.",
    ],
    ["c5-context-format3", "C5", ["AB"], "An all-format-3 Context lookup executes its nested Single."],
    ["c6-context-pos", "C6", ["AB"], "Contextual GPOS executes a nested full SinglePos."],
    ["c7-ligature-across-mark", "C7", ["A\uE000B"], "The skipped mark joins the entire ligature span's minimum cluster."],
    ["c8-mark-ligature", "C8", ["A\uE000B\uE001", "AB\uE000\uE001"], "Ligature component metadata selects anchors, including a last-component control."],
    ["c8-mark-mark-components", "C8", ["A\uE000B\uE001", "AB\uE000\uE001"], "MarkMark rejects different components and accepts a same-component control."],
    ["reverse-chain", null, ["AB"], "ReverseChain GSUB remains deliberately unsupported.", "unsupported GSUB 0"],
    ["cursive", null, ["AA"], "Cursive GPOS remains deliberately unsupported.", "unsupported GPOS 0"],
    ["nested-context", null, ["AB"], "Nested contextual lookups remain deliberately unsupported.", "not executable"],
].flatMap(([kind, correction, texts, description, errorCategory]) =>
    texts.map((text, index) => ({
        id: `${kind}-${index}`,
        kind,
        correction,
        text,
        description,
        expected: errorCategory ? "negative" : "positive",
        discriminates:
            errorCategory || (kind === "c2-filter-set" && index === 1) || (kind === "c8-mark-ligature" && index === 0) || (kind === "c8-mark-mark-components" && index === 1)
                ? []
                : [correction],
        controls:
            (kind === "c2-filter-set" && index === 1) || (kind === "c8-mark-ligature" && index === 0) || (kind === "c8-mark-mark-components" && index === 1) ? [correction] : [],
        coExercised: kind === "c8-mark-ligature" ? ["C1", "C7"] : kind === "c8-mark-mark-components" || kind === "c4-nested-ligature" ? ["C7"] : [],
        ...(errorCategory ? { errorCategory } : {}),
    }))
);
