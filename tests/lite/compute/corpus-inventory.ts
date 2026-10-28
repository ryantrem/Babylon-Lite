/** Manually enumerated source/input recipes. No test execution or discovery.
 * Injection-only models are separately accounted for and never assigned glyph records. */
export type AddCorpus = (file: string, title: string, font: string, texts: readonly string[], currentlyRejected?: boolean) => void;
const native = (file: string) => `tests/lite/compute/${file}.test.ts`;
const roboto = "Roboto-Regular.ttf";
const inter = "Inter.ttf";
const synthetic = (kind: string) => `synthetic:${kind}`;

export const SEGMENT_TEXT = {
    english:
        "The quick brown fox jumps over the lazy dog. Every letter belongs to a flowing English paragraph with ordinary spaces and punctuation. AV To are adjacent pairs; office and affinity remain ordered shaping controls. ",
    natural:
        "Office affinity offers fine AVATAR kerning. The efficient flight lifts off at five. WAVY typography: coffee, waffles and films. A quick brown fox visits the office today. Friendly staff review lovely typography. Affluent traffic flows over the river. Vast valleys offer fresh coffee at noon. To verify fonts, try AV WA To fi fl ffi. ",
    extra: "Hello (world) - it's a test: x = y + 2; [a] {b} @home <tag> ~tilde* \"quoted\" 'single' 12:30 -> => ... end. ",
};

export function appendReviewedInputs(add: AddCorpus, greek: string, cyrillic: string): void {
    const group = (file: string, title: string, font: string, texts: readonly string[], rejected = false) => add(native(file), title, font, texts, rejected);
    for (const kind of ["pair-first", "pair-second"]) {
        const texts = ["AB", "ABC", "AABC", "BC", "ABA", "ABC".repeat(8)];
        group("kernel", `proves disjoint pair commits but preserves nonzero second-advance ordering: ${kind}`, synthetic(kind), texts);
        group("resident", `packs correct resident advances for disjoint and overlapping pair commits: ${kind}`, synthetic(kind), texts);
    }
    group("kernel", "retains original root-first rejection priority across independently positioned glyphs", synthetic("pair-reject"), ["A", "B", "C", "BA", "CB", "AC"]);
    group("kernel", "executes complete bounded shaping when optional pair metadata would exceed the device program limit", synthetic("pair-first"), ["ABC"]);
    for (const kind of ["multiple", "mark-ligature", "adjust", "context"]) {
        const texts = kind === "mark-ligature" ? ["AB", "ABB", "ACB", "B"] : kind === "context" ? ["AB", "AA", "BA", "BB"] : ["A", "AA", "AB", "BA"];
        group("kernel", `actual WGSL mode ${kind}`, synthetic(kind), texts);
    }
    group("kernel", "understood alternate is rejected when matched, never a mode-5 noop", synthetic("alternate"), ["B"]);
    group("kernel", "GPOS point-anchor rejection identifies its table and original run after empty-run filtering", synthetic("point-anchor"), ["B"]);
    group("kernel", "shares bounded cooperative scratch within a run and executes the complete global fallback for larger runs", roboto, [
        "office ".repeat(7),
        "office ".repeat(50),
    ]);
    group(
        "kernel",
        "sizes cooperative storage to power-of-two capacity and reuses the warm variant until growth",
        roboto,
        [50, 10, 60, 100].map((n) => "H".repeat(n))
    );
    for (const title of [
        "reuses warm public bind groups while repacking changed glyphs and counts",
        "submits public shaping and its completion copy together without publishing mapped views",
        "reuses the loaded executor before warm input submission without another module-resolution yield",
    ]) {
        group("kernel", title, roboto, ["affinity", title.startsWith("reuses warm") ? "H".repeat(49) : "H"]);
    }
    group("kernel", "materializes the published glyph count without repeated output-array growth", roboto, ["office affinity ".repeat(3)]);
    group("kernel", "submits an already prepared owned schedule without an extra asynchronous preparation turn", roboto, ["affinity", "H", "office ".repeat(40), "H".repeat(300)]);
    group("kernel", "allocates compact mutable input capacity and reuses it for shorter and changed revisions", roboto, [
        "office".repeat(10),
        "H".repeat(50),
        "H".repeat(59),
        "affinity",
    ]);
    group("kernel", "storage/buffer/workgroup exact boundaries use real device limits without near-OOM allocation", "NotoSansGothic-Regular.ttf", ["𐌰"]);
    for (const kind of ["pair-first", "pair-second", "multiple", "context", "transitive", "mark-ligature"]) {
        group("long-run", `retains cooperative/ordered actions and the complete failed-proof candidate mask: ${kind}`, synthetic(kind), [
            "ABC".repeat(100),
            "C".repeat(257) + "AB",
            "AB".repeat(512),
        ]);
    }
    for (const kind of ["backtrack", "ligature-ignore-marks", "multiple", "mark-ligature", "mark-mark", "context"]) {
        const seed = kind === "backtrack" ? "ABAABC" : kind === "mark-mark" ? "ABBBCA" : "ABC";
        group(
            "ordered-parallel",
            `keeps failed selections, skipped components, insertion cursors and inherited attachment fields exact: ${kind}`,
            synthetic(kind),
            [64, 300, 600].map((n) => seed.repeat(Math.ceil(n / seed.length)).slice(0, n))
        );
    }
    group("ordered-parallel", "alternates snapshot reuse and forced re-selection after contextual writes", synthetic("backtrack"), ["ABAABC".repeat(20), "ABAABC".repeat(100)]);
    group(
        "parallel-refresh",
        "compacts dense real GSUB ligatures at chunk and tile boundaries",
        synthetic("refresh-ligature"),
        [0, 31, 255, 511].map((prefix) => "C".repeat(prefix) + "ABC".repeat(40) + "C".repeat(35))
    );
    group(
        "parallel-refresh",
        "rescans cooperative id changes, gaining and losing later roots in both scratch variants",
        synthetic("refresh-roots"),
        [1, 2, 31, 32, 33, 64, 255, 256, 257, 480, 768].map((n) => "A".repeat(n))
    );
    group(
        "parallel-refresh",
        "fences serial short-run refresh in oversized global scratch",
        synthetic("refresh-roots"),
        [1, 2, 31, 32, 33, 64, 255, 256].map((n) => "A".repeat(n))
    );
    group(
        "parallel-refresh",
        "fences serial short-run refresh in oversized global scratch",
        synthetic("refresh-ligature"),
        [1, 2, 10, 11].map((n) => "ABC".repeat(n))
    );
    for (const font of [inter, roboto]) {
        for (const length of [480, 1000]) {
            for (const [name, corpus] of Object.entries(SEGMENT_TEXT)) {
                group("segment", `matches whole-run output and oracle at 480/1000 glyphs for all corpora: ${font}#${length}/${name}`, font, [corpus.repeat(12).slice(0, length)]);
            }
        }
    }
    group("segment", "preserves logical slots across mixed Latin/Greek groups and unsplittable/short runs", roboto, [
        "fi",
        SEGMENT_TEXT.english.repeat(3).slice(0, 480),
        "ΑΒΓΔ ".repeat(60),
        "",
        "A".repeat(300),
        SEGMENT_TEXT.natural.repeat(3).slice(0, 480),
        "HH",
    ]);
    group("segment", "includes an RTL fallback group alongside long Latin segments", "NotoSansHebrew-Regular.ttf", ["a b ".repeat(120), "שלום ".repeat(30), "a b"]);
    group("segment", "tiles actual non-stage segmented callers above 64 segments through overlapping varying-size updates", roboto, [
        "HH",
        SEGMENT_TEXT.english.repeat(32),
        SEGMENT_TEXT.english.repeat(32) + " A",
        SEGMENT_TEXT.english.repeat(24),
        SEGMENT_TEXT.english.repeat(25),
    ]);
    group("segment", "propagates a rejecting segment to logical status 1 and suppresses all instances", inter, ["a ".repeat(60) + "<-" + " b".repeat(100)], true);
    group("segment", "reuses warm segmented bindings, rebuilds transitions and disposes all new buffers", roboto, [
        "HH",
        SEGMENT_TEXT.english.repeat(3).slice(0, 480),
        "A".repeat(600),
        SEGMENT_TEXT.english.repeat(5).slice(0, 1000),
    ]);

    const residentGroups: [string, string[]][] = [
        ["groups decoded input records directly without allocating a second per-run descriptor", ["office", "", greek, cyrillic, "A"]],
        ["keeps immutable compilation and pipelines warm while updating words, counts, selections and capacities", ["office", "A\u0326", "", "H".repeat(50), greek, "HH", "H"]],
        ["retains source order across prefix tiers, multiple plan groups and empty-run transitions", ["office ", greek, cyrillic, ""]],
        ["reuses admitted immutable outlines and component lists across warm closure revisions, including empty glyphs", ["HH", "H ".repeat(25), "é é", "é", "H"]],
        ["reuses resident shape and pack bind groups until buffer identities grow", ["affinity", "H".repeat(49), "H".repeat(512)]],
        ["copies checked warm nominal mappings directly while decoding current inputs and verifying all records", ["H".repeat(49)]],
        ["submits resident shaping, useful packing and its four-byte completion copy together", ["affinity", "H"]],
        [
            "decodes checked warm nominal vectors into the scalar slab and collects closure seeds without retraversing each glyph vector",
            ["office fi To A\u0326 ".repeat(4).slice(1, 50), "H".repeat(49)],
        ],
        ["owns one disjoint batch input slab per update instead of one backing allocation per run", ["office A\u0326".slice(1), "HHHH".slice(1)]],
        [
            "packs independent runs in parallel after ordered visible-count prefixing, preserving the exact output ABI",
            ["office fi To A\u0326 ".repeat(4).slice(0, 50), "H".repeat(50)],
        ],
        ["fuses lane-striped resident counts, prefixing and packing into one small-batch dispatch", ["A  ", "office A\u0326 ", "HH"]],
        ["retains the guaranteed 256-lane fused count entry while keeping every large-batch record and packed instance", ["office A\u0326 ".repeat(4), "H".repeat(50)]],
        ["scales counting over multiple workgroups above the fused tier without truncating any resident run", ["office A\u0326 ".repeat(4), "H".repeat(50)]],
        ["packs mutable resident input words once per plan group, not again to calculate capacity", ["affinity", "H".repeat(49)]],
        ["passes the current unique repertoire to closure, not one copied seed per input glyph", ["H".repeat(50)]],
        [
            "shared atlas and independent blocks: closure, cold growth, cross-font ownership and 100 blocks",
            [...Array.from({ length: 49 }, (_, i) => "A".repeat(i + 2)), greek, cyrillic, "é", "ffi", "AAA"],
        ],
    ];
    for (const [title, texts] of residentGroups) {
        group("resident", title, roboto, texts);
    }
    group("resident", "shared atlas and independent blocks: closure, cold growth, cross-font ownership and 100 blocks", inter, ["AC"]);
    for (const count of [512, 1024, 2048]) {
        group("resident", `packs a complete resident batch of ${count} with exact counts and ABI`, roboto, ["office fi To A\u0326 ".repeat(4).slice(0, 50), "H".repeat(50), "H"]);
    }
    group("resident", "transitive selected GSUB closure installs empty-outline outputs, never phantom instances", synthetic("transitive"), ["A"]);
    for (const count of [512, 1024]) {
        const phrases = ["The efficient flight lifts off at five. ", "WAVY typography: coffee, waffles and films. ", "A quick brown fox visits the office today. "];
        group(
            "milestones",
            `completes every run and six fields in a natural Latin50 batch of ${count}`,
            roboto,
            Array.from({ length: count }, (_, run) => phrases[run % phrases.length]!.repeat(2).slice(0, 46) + String(run).padStart(4, "0"))
        );
    }
    group("lifecycle", "busy rejection, disposal during an actual pipeline request, and fresh shaper recreation", roboto, ["office"]);
    group("lifecycle", "device.destroy resolves real device.lost during pending work; independent engines survive", "NotoSansGothic-Regular.ttf", ["𐌰𐌱"]);
    appendParagraphInputs(add);
    for (const file of ["tests/lite/plumbing/compute-text-shaper.spec.ts", "lab/lite/src/lite/compute-text-shaper-test.ts"]) {
        const title = file.endsWith(".spec.ts") ? "compute text shaper matches text-shaper on a real WebGPU device" : "run";
        for (const [font, texts] of [
            [roboto, ["fi"]],
            [inter, ["AC", "A\u0326", "Αθήνα", "Москва"]],
            ["NotoSansHebrew-Regular.ttf", ["שלום"]],
            ["NotoSansArabic-Regular.ttf", ["كتب"]],
            ["NotoSansSC-Regular-subset.ttf", ["汉字"]],
            ["NotoSansGothic-Regular.ttf", ["𐌰𐌱𐌲"]],
        ] as [string, string[]][]) {
            add(file, title, font, texts);
        }
        add(file, `${title}#replacement-required rejection witness`, inter, ["A\u20de"], true);
    }
    add("tests/lite/unit/compute-text-shaper.test.ts", "matches text-shaper for Roboto Latin and executable Inter kerning and marks", inter, ["A\u20de", "A\u20dd", "<-"], true);
    add("tests/lite/unit/compute-text-shaper.test.ts", "returns empty results without dispatching and publishes reference records", roboto, ["fi"]);
    add("tests/lite/unit/compute-text-shaper.test.ts", "shapes a character string without a caller-built plan or shaper", roboto, ["fi"]);
    add("tests/lite/unit/compute-text-program.test.ts", "admits Inter design values for zero default device deltas and keeps nested ligatures fail-closed", inter, [
        "AC",
        "5,",
        "A\u0326",
        "\u0301\u0300",
        "i\u0301",
        "e\u0301",
        "A\u20de",
        "A\u20dd",
        "<-",
    ]);
    group("resident", "kernel failure statuses (internal slot injection, not public admission)", roboto, ["A"]);
    group("resident", "rejection clears every indirect word and recovery keeps the caller atlas", inter, ["A\u20de", "A\u20dd", "<-"], true);
}

function appendParagraphInputs(add: AddCorpus): void {
    const layout = native("paragraph-layout");
    const chunks = native("paragraph-chunks");
    for (const font of [inter, roboto, "NotoSansGothic-Regular.ttf"]) {
        const gothic = font.includes("Gothic");
        const token = gothic ? "𐌰 𐌱 𐌲 " : "office affinity AV To A\u0308 ";
        const original = token.repeat(80);
        const at = Math.floor(original.length / 2);
        const scalarAt = gothic && original.charCodeAt(at) >= 0xdc00 && original.charCodeAt(at) <= 0xdfff ? at - 1 : at;
        add(layout, `reuses bounded flag windows while native output and exact backing ownership match fresh layout: ${font}`, font, [
            original,
            token.repeat(40),
            "  ",
            "",
            original.slice(0, scalarAt) + (gothic ? "𐌰" : "A") + original.slice(scalarAt),
            original + " ",
            " " + original,
        ]);
    }
    add(
        layout,
        "budgets optional capacity, counts growth rebuilds, and rejects only required overflow without drawing",
        synthetic("multiple"),
        [20, 21, 25, 26, 31].map((n) => "A".repeat(n))
    );
    add(layout, "retains clean multi-paragraph slots with a budgeted pool and compacts when insertion headroom is exhausted", synthetic("multiple"), [
        "A".repeat(10),
        "B".repeat(10),
        "C".repeat(10),
        "B".repeat(9),
    ]);
    for (const font of [inter, roboto]) {
        const medium = "office affinity AV To A\u0326 ".repeat(40);
        let paragraphs = [medium, medium + "A", medium + "B", medium + "C"];
        const all = paragraphs.slice();
        const edit = (next: string[]) => {
            paragraphs = next;
            all.push(...next);
        };
        edit([paragraphs[0]!, paragraphs[1]! + "x", paragraphs[2]!, paragraphs[3]!]);
        edit([paragraphs[0]!, paragraphs[1]!.slice(0, -1), paragraphs[2]!, paragraphs[3]!]);
        edit([paragraphs[0]!, paragraphs[1]!.slice(0, 480), paragraphs[1]!.slice(480), ...paragraphs.slice(2)]);
        edit([paragraphs[0]!, paragraphs[1]! + paragraphs[2]!, ...paragraphs.slice(3)]);
        edit([paragraphs[0]!, "pasted office", "", "affinity AV To", ...paragraphs.slice(2)]);
        edit([paragraphs[0]!, ...paragraphs.slice(3)]);
        edit([...paragraphs, "appended office"]);
        edit([paragraphs[0]! + "A", paragraphs[2]!, paragraphs[1]!, ...paragraphs.slice(3)]);
        add(layout, `is bit-identical to fresh updates through a four-medium-paragraph editing sequence: ${font}`, font, [...all, "", "A"]);
        add(layout, `matches oracle records, lines, instances and status-only readback: ${font}`, font, [
            "office affinity To A\u0326",
            "AV To",
            "  A   B C   ",
            "",
            "   ",
            "ABCDEFGHIJKLMNO",
            "B\u0301\u0307B\u0301\u0307B",
            "A\u00a0B",
            "A",
        ]);
        add(layout, `preserves whole-paragraph clusters after segmented shaping: ${font}`, font, ["office  affinity To A\u0326 ".repeat(12), "  ABC ".repeat(20), ""]);
        add(layout, `aligns hanging and overflowing lines, with custom height and paragraph spacing: ${font}`, font, ["A   B ", " ", "", "office affinity", "WWW"]);
        add(chunks, `matches six-field records and f32 instance bits across small arena seams: ${font}`, font, [
            "office affinity AV To A\u0326 ".repeat(8),
            "",
            "   ",
            "office   ".repeat(20),
        ]);
    }
    for (const [count, whole] of [
        [100, false],
        [100, true],
        [500, false],
    ] as const) {
        const text = "office affinity AV To A\u0326 ".repeat(12);
        add(layout, `prepares only one dirty paragraph in a ${count}-paragraph document (whole-run=${whole})`, roboto, [text + "A", text + "B", text + "AA"]);
    }
    const ordinary: [string, string, string[]][] = [
        [
            "shapes/stitches a nonzero-base dirty batch across several script groups",
            roboto,
            ["office affinity ".repeat(20), "α βγ ".repeat(40), "Москва Іі ".repeat(30), "A", "α βγ ".repeat(40) + "α", "Москва Іі ".repeat(30) + "І"],
        ],
        ["does no preparation/shape/stitch work for placement, width, deletion, or unchanged updates", roboto, ["office affinity ".repeat(25), "AV To ".repeat(30), "A"]],
        ["compacts on pool exhaustion and retains scalar flags when a dirty slot shrinks", roboto, ["A\u0308 A\u0326 ".repeat(16), "A", "B", "C", "D"]],
        ["plain updates and synchronous errors invalidate reuse, then recover through a full rebuild", roboto, ["office affinity", "AV To", "A"]],
        ["adds new glyph outlines/slot entries only and rebuilds on atlas capacity growth", inter, ["ABCDE", "ABCDF", "A", ..."GHIJKLMNOPQRSTUVWXYZ".match(/.{1,5}/gu)!]],
        [
            "keeps logical failure ordering in cached slot info and invalidates after a failed incremental submission",
            roboto,
            ["office A", "office B", "office BA", "office C", "office D"],
        ],
        ["matches fresh failing paragraphs and their later fixes through real unsupported lookups", synthetic("pair-reject"), ["A", "BA", "C"]],
        ["rebuilds after external atlas mutation, including a formerly empty glyph becoming visible", inter, ["A A", "B B"]],
        [
            "uses actual tiled stitch Y for long paragraph stages while shaped/layout/placement stay exact",
            roboto,
            ["The office reviews a clear paragraph with ordinary spaces. ".repeat(100), "Hello world"],
        ],
        ["scans spans of more than 256 paragraphs, preserving line and instance bases across plan groups", roboto, ["", "α βγ", "office   affinity ", "Москва Іі"]],
        ["uses scalar-indexed flags and line source indices for supplementary LTR text", "NotoSansGothic-Regular.ttf", ["𐌰𐌱 𐌲𐌳", "𐌰𐌱𐌲"]],
        [
            "reuses warm buffers and bindings, resizes dispatches without rebinding, and alternates both update APIs",
            roboto,
            ["office A\u0326", "AB", "office   ".repeat(20), "", "A"],
        ],
        ["rejects busy and disposed data synchronously with paragraph-specific diagnostics", inter, ["office"]],
        ["reports GPU-detected failures through paragraph-prefixed completion and recovers on the next update", roboto, ["A"]],
        ["clears all indirect words on synchronous admission failure without changing the atlas", inter, ["A"]],
        ["retains original paragraph completion semantics across a later plain update", roboto, ["B B"]],
        ["releases the cached stage on the block's terminal device-loss path", inter, ["A A"]],
    ];
    for (const [title, font, texts] of ordinary) {
        add(layout, title, font, texts);
    }
    for (const full of [false, true]) {
        add(layout, `late completion failure invalidates descendants but not a newer full rebuild (rebuild=${full})`, roboto, ["office AA", "office BA", "office C"]);
    }
    const chunkGroups: [string, string, string[]][] = [
        ["places a fitting indivisible span through a one-workgroup grid without truncation", inter, ["A".repeat(100)]],
        ["bounds uneven segment workspace without rejecting a fitting indivisible span", inter, ["a ".repeat(80) + "A".repeat(15000) + " " + "a ".repeat(200)]],
        ["retires obsolete dedicated arenas and transient high-water storage after a smaller replacement", inter, ["office ".repeat(30)]],
        [
            "retains paragraph-level script selection and scalar clusters across Common-only and supplementary chunks",
            "NotoSansGothic-Regular.ttf",
            [" ".repeat(96) + "𐌰 𐌱 𐌲𐌰 ".repeat(40), "𐌰 " + " ".repeat(96) + "𐌰 𐌱 𐌲𐌰 ".repeat(40), "𐌱𐌲"],
        ],
        ["uploads immutable font data once per selected plan rather than once per chunk", inter, ["office ".repeat(24), "ΑΒΓ ".repeat(24)]],
        ["skips clean shape dispatches and releases paged state when switching to plain updates", inter, ["office affinity ".repeat(20), "A", "B"]],
        [
            "re-places a clean prefix when a future chunk changes the width of their shared open line",
            inter,
            ["A " + "office affinity ".repeat(20), "A " + "office affinity ".repeat(20) + "A"],
        ],
        ["matches serial signed wrapping, deleted chunks and cross-boundary pen arithmetic", inter, ["A A A A A A A A A A A A A A A A A A A A A A A A A A"]],
        ["carries both retroactive line closures over many chunks without losing hanging-space events", inter, ["A ".repeat(80)]],
        [
            "automatically pages materially wasteful mixed-length storage while preserving clean paragraph shapes",
            inter,
            ["office ".repeat(180), "A", "affinity " + "office ".repeat(180)],
        ],
        ["rebuilds paged placement/slot dependencies after external atlas mutation", inter, ["A A ".repeat(25), "B B"]],
        ["retains admitted immutable source/layout snapshots while chunk preparation is pending", inter, ["office affinity ".repeat(10)]],
        ["renders a real paragraph beyond required Stage1 line storage with default device limits", inter, ["a a a "]],
    ];
    for (const [title, font, texts] of chunkGroups) {
        add(chunks, title, font, texts);
    }
    const original = "office affinity AV To ".repeat(24);
    let paragraphs = [original, "office", original + "A"];
    const edits = paragraphs.slice();
    paragraphs = ["A " + original, ...paragraphs.slice(1)];
    edits.push(...paragraphs);
    for (const next of [
        [paragraphs[0]!, paragraphs[0]!.slice(0, 200), paragraphs[0]!.slice(200), ...paragraphs.slice(1)],
        [paragraphs[0]!, "office " + original],
        ["", "office", original],
        [original, "office", ""],
        [],
    ]) {
        edits.push(...next);
    }
    add(chunks, "reuses proven shifted suffix shapes, converges exactly, and handles paragraph split/merge/reorder edits", roboto, edits);
    for (const kind of ["multiple", "context", "backtrack", "ligature-ignore-marks", "adjust"]) {
        add(chunks, `preserves fitting indivisible synthetic shaping: ${kind}`, synthetic(kind), ["ABCABC", "AABC", "BC"]);
    }
}

/** R1 resolved the runtime-dependent edits as differential; no concrete input is pending. */
export const PENDING_INPUTS: { file: string; label: string; font: string; expression: string; reason: string }[] = [];

export interface NonReferenceCase {
    source: string;
    reason: "negative" | "mock" | "internal-model" | "structural" | "status-only" | "differential";
    inputSource: string;
    font?: string;
}

/** R1: only already-enumerated titles, confirmed from their actual assertions.
 * compareFresh compares residentOutputs(incremental) with residentOutputs(fresh).
 * paragraph-chunks compare compares outputs(paged) with outputs(monolithic).
 * None of these titles calls oracleFields/verifyLayout. All edits, including
 * live-capacity-derived edits, belong to the differential contract, not a
 * concrete reference record. Reference-backed titles are intentionally absent.
 */
export const DIFFERENTIAL_CASES: NonReferenceCase[] = [
    ...[inter, roboto].map((font) => ({
        source: `${native("paragraph-layout")}#is bit-identical to fresh updates through a four-medium-paragraph editing sequence: ${font}`,
        reason: "differential" as const,
        font,
        inputSource:
            "Every edit, including office.repeat(live cache capacity): compareFresh -> residentOutputs equality plus capacity/order assertions; no reference shaping expectation",
    })),
    ...[
        [100, false],
        [100, true],
        [500, false],
    ].map(([count, whole]) => ({
        source: `${native("paragraph-layout")}#prepares only one dirty paragraph in a ${count}-paragraph document (whole-run=${whole})`,
        reason: "differential" as const,
        font: roboto,
        inputSource: "Before/after clean-slot residentOutputs and compareFresh equality; mapping/dispatch counts are structural only",
    })),
    ...[
        ["does no preparation/shape/stitch work for placement, width, deletion, or unchanged updates", roboto],
        ["compacts on pool exhaustion and retains scalar flags when a dirty slot shrinks", roboto],
        ["plain updates and synchronous errors invalidate reuse, then recover through a full rebuild", roboto],
        ["adds new glyph outlines/slot entries only and rebuilds on atlas capacity growth", inter],
        ["keeps logical failure ordering in cached slot info and invalidates after a failed incremental submission", roboto],
        ["matches fresh failing paragraphs and their later fixes through real unsupported lookups", synthetic("pair-reject")],
        ["rebuilds after external atlas mutation, including a formerly empty glyph becoming visible", inter],
    ].map(([title, font]) => ({
        source: `${native("paragraph-layout")}#${title}`,
        reason: "differential" as const,
        font,
        inputSource: "compareFresh -> residentOutputs(incremental) == residentOutputs(fresh), plus structural/status/error assertions; no oracleFields/verifyLayout",
    })),
    ...[false, true].map((full) => ({
        source: `${native("paragraph-layout")}#late completion failure invalidates descendants but not a newer full rebuild (rebuild=${full})`,
        reason: "differential" as const,
        font: roboto,
        inputSource: "Completion/cache lifetime checks and compareFresh output equality; no reference shaping values",
    })),
    ...[inter, roboto].map((font) => ({
        source: `${native("paragraph-chunks")}#matches six-field records and f32 instance bits across small arena seams: ${font}`,
        reason: "differential" as const,
        font,
        inputSource: "All widths/alignments compare outputs(paged) == outputs(monolithic), not an independent shaper",
    })),
    ...[
        ["places a fitting indivisible span through a one-workgroup grid without truncation", inter],
        ["bounds uneven segment workspace without rejecting a fitting indivisible span", inter],
        ["retires obsolete dedicated arenas and transient high-water storage after a smaller replacement", inter],
        ["retains paragraph-level script selection and scalar clusters across Common-only and supplementary chunks", "NotoSansGothic-Regular.ttf"],
        ["uploads immutable font data once per selected plan rather than once per chunk", inter],
        ["skips clean shape dispatches and releases paged state when switching to plain updates", inter],
        ["reuses proven shifted suffix shapes, converges exactly, and handles paragraph split/merge/reorder edits", roboto],
        ["re-places a clean prefix when a future chunk changes the width of their shared open line", inter],
        ["matches serial signed wrapping, deleted chunks and cross-boundary pen arithmetic", inter],
        ["carries both retroactive line closures over many chunks without losing hanging-space events", inter],
        ["automatically pages materially wasteful mixed-length storage while preserving clean paragraph shapes", inter],
        ["rebuilds paged placement/slot dependencies after external atlas mutation", inter],
        ["retains admitted immutable source/layout snapshots while chunk preparation is pending", inter],
    ].map(([title, font]) => ({
        source: `${native("paragraph-chunks")}#${title}`,
        reason: "differential" as const,
        font,
        inputSource: "All source inputs/edits compare paged and monolithic Lite outputs; other assertions concern workspace, counters, injected layout words or resource ownership",
    })),
    ...["multiple", "context", "backtrack", "ligature-ignore-marks", "adjust"].map((kind) => ({
        source: `${native("paragraph-chunks")}#preserves fitting indivisible synthetic shaping: ${kind}`,
        reason: "differential" as const,
        font: synthetic(kind),
        inputSource: "ABCABC/AABC/BC compare outputs of two Lite paths; no independent six-field expectation",
    })),
];

/** Fixed non-shaping accounting. No glyph records are synthesized for these rows. */
export function nonReferenceCorpus(): NonReferenceCase[] {
    const rows: NonReferenceCase[] = [];
    const add = (file: string, reason: NonReferenceCase["reason"], titles: readonly string[], inputSource: string, font?: string) => {
        rows.push(...titles.map((title) => ({ source: `${file}#${title}`, reason, inputSource, ...(font ? { font } : {}) })));
    };
    add(
        native("kernel"),
        "negative",
        ["shape overflow status 2 executes production WGSL (internal capacity injection)"],
        "Production action invoked with deliberately undersized capacity; compares terminal header, not shaped expectations",
        synthetic("multiple")
    );
    add(
        native("long-run"),
        "negative",
        [
            "reports root-first, then full source-index rejection beyond the first tile",
            "tiled pair fallback preserves rejection priority when compact simple metadata is unavailable",
        ],
        "Explicit injected overflow or matched unsupported lookup; no positive six-field record comparison",
        synthetic("pair-reject")
    );
    add(
        native("long-run"),
        "negative",
        ["keeps long multiple-substitution overflow header exact"],
        "Multiple substitution with deliberately undersized capacity, terminal header only",
        synthetic("multiple")
    );
    add(
        native("ordered-parallel"),
        "negative",
        ["keeps the first of two ordered rejections and the original lookup word in both scratch modes"],
        "C-prefix + ABA + C-prefix + ABA; index 40/255/256/300, raw unsupported header",
        synthetic("backtrack-reject")
    );
    add(
        native("ordered-parallel"),
        "negative",
        ["rejects the first contour mark-ligature anchor before a later rejection in the same root"],
        "C-prefix + AB at index40/255/300; point-anchor rejection",
        synthetic("point-anchor")
    );
    for (const count of [1, 2, 31, 32, 33, 64, 255, 256, 257, 480, 768]) {
        add(
            native("parallel-refresh"),
            "internal-model",
            [`preserves all eight words across arbitrary tombstones at length ${count}`],
            "Record word j = index*101+j, selected deleted indices; arithmetic survivors, not text-shaper data",
            synthetic("refresh-roots")
        );
    }
    add(
        native("parallel-refresh"),
        "negative",
        ["preserves the first unsupported status and source cursor"],
        "C-prefix+ABA, raw unsupported headers",
        synthetic("backtrack-reject")
    );
    add(native("parallel-refresh"), "negative", ["preserves capacity overflow before any refresh"], "Multiple action with forced capacity overflow", synthetic("multiple"));
    add(
        native("resident"),
        "structural",
        ["style ABI uses each font's unitsPerEm and requested pixel size"],
        "Style words [1,1,1,1,unitsPerEm/48,0,0,0], no shaped data",
        "NotoSansGothic-Regular.ttf"
    );
    add(
        native("resident"),
        "negative",
        ["kernel overflow clears draw count (internal undersized instance allocation)"],
        "AB plus undersized instance buffer; expects status2 and [6,0,0,0]"
    );
    add(
        native("resident"),
        "structural",
        ["nonvisual actual indirect draws reuse shared native atlas resources and invalidate sibling bindings on growth"],
        "Draw/bind-group command ownership for A and atlas growth; no reference field equality"
    );
    add(
        native("lifecycle"),
        "status-only",
        ["resident busy, disposal and survivor ownership while the real pipeline is pending", "resident device.lost is terminal during actual pipeline preparation"],
        "Pending actual pipeline, dispose/loss, survivor status and indirect resource ownership"
    );
    const layout = native("paragraph-layout");
    add(
        layout,
        "status-only",
        [
            "invalidates flag entries with ready admission errors and never publishes a busy-rejected candidate",
            "clears a previously visible draw on required budget overflow without installing new glyphs",
        ],
        "Cache invalidation/required allocation errors; no shaped output expectation"
    );
    add(
        layout,
        "structural",
        [
            "performs bounded flag work on eligible edits and fully regenerates every ineligible dirty paragraph",
            "recreates stage resources and bumps its version on a device change",
        ],
        "Flag-work counts and resource version/lifetime, not shaped values"
    );
    add(
        layout,
        "negative",
        ["matches fresh failing paragraphs and their later fixes through real unsupported lookups"],
        "BC matches a nonzero default variation delta and rejects; positive A/BA/C recovery inputs are registered separately",
        synthetic("pair-reject")
    );
    add(
        layout,
        "internal-model",
        [
            "preserves cached slot failures and selects the first failing logical paragraph after reordering or deletion",
            "matches serial WGSL bit-for-bit for blocked scans, candidate events, fallback, failures and incremental reuse (seed 0x31415926)",
            "matches the serial oracle for hinted gallop and lane line sweeps through 120001 records (seed 0x6a09e667)",
            "keeps final-record decisions exact at counts 1/2/256/257/512/513 and one f32 step below",
            "preserves rounded boundaries for sampled starts and cross-lane successors",
            "handles equal prefixes across lane/span boundaries and paragraph tails with emergency-only breaks",
            "keeps the exact signed-prefix limit parallel and overflow or negative advances serial",
            "matches fixed-seed mixed batches including adjacent final-paragraph bounds",
            "counts visible slots and keeps y advances across lines, including visible hanging spaces and protected clusters",
        ],
        "Manually injected records/advances/flags (source literals and seeds), CPU referenceLines or unchanged serial WGSL; no text-shaper input"
    );
    for (const data of [
        { failureAt: 299, failure: 1, limit: 300, expected: 1 },
        { failureAt: 299, failure: 3, limit: 1, expected: 2 },
        { failureAt: 0, failure: 3, limit: 1, expected: 3 },
        { failureAt: 3, failure: 1, limit: 2, expected: 2 },
        { failureAt: 0, failure: 2, limit: 300, expected: 2 },
        { failureAt: 0, failure: -3, limit: 300, expected: 3 },
    ]) {
        add(
            layout,
            "internal-model",
            [`reports the lowest failing paragraph or capacity overflow and draws nothing: ${JSON.stringify(data)}`],
            "300 records with injected status/capacity/slot failures; expected first logical failure"
        );
    }
    const chunks = native("paragraph-chunks");
    add(
        chunks,
        "structural",
        [
            "skips measure/place only at exact checkpoint and reverse-event convergence",
            "retains the original fitting-path render command order",
            "encodes every actual ordered window and invalidates bundles only when the range list changes",
            "binds bounded windows inside arenas larger than the effective test binding limit",
        ],
        "Metadata counters, render command order or binding window bounds; no reference shaping values"
    );
    add(
        chunks,
        "status-only",
        [
            "preserves busy admission, multiple pending results and terminal resource release",
            "preserves status results while switching between submitted legacy and paged resources",
            "releases early paged allocations when disposal interrupts pipeline preparation",
            "retires a failed admitted candidate before retrying pipeline preparation",
            "forgets all paged resources on actual device loss and never resurrects the owner",
        ],
        "Busy/dispose/preparation/loss status and resource lifetime checks"
    );
    for (const independent of [false, true]) {
        add(
            chunks,
            "status-only",
            [`invalidates a failed paged ancestor without clearing a newer independent generation: ${independent}`],
            "Injected completion status1, subsequent generation invalidation and visible-range status"
        );
    }
    for (const status of [1, 2, 3]) {
        add(
            chunks,
            "negative",
            [`keeps every old/candidate range invisible after synchronous or later-chunk GPU failure ${status}`],
            "Injected shaped status, count overflow or missing atlas slot; indirect draw must be suppressed"
        );
    }
    add(
        chunks,
        "negative",
        ["rejects an indivisible required span before installing atlas or paged resources"],
        "A.repeat(200) required workspace exceeds effective binding budget, no published output"
    );
    add(
        chunks,
        "internal-model",
        ["renders a real paragraph beyond required Stage1 line storage with default device limits#periodic-large-run"],
        'Model repeats the oracleFields("a a a ") two-record period for a device-limit-derived count; the source six-field period is registered separately',
        inter
    );
    for (const count of [0, 1, 64, 65]) {
        add(
            native("stitch-routing"),
            "internal-model",
            [`routes S=${count} through one exact shader/output dispatch`],
            "Source buildCase() produces controlled segment headers/records; CPU stitch model, no text-shaper records"
        );
    }
    add(
        native("stitch-routing"),
        "internal-model",
        [
            "keeps the exported tiled kernel byte-identical for the separate paged executor",
            "prepares both variants before first submission and flips 1→2→1 rows without cold compile",
            "uses the same route changes without a custom pack stage",
            "does not publish readiness when preparation of the initially unselected variant fails",
            "uses the batch maximum for mixed runs and preserves signed failure and overflow headers",
        ],
        "Controlled segment words from buildCase(); routing/lifecycle or CPU byte model"
    );
    for (const count of [0, 1, 63, 64, 65, 128, 129, 1961]) {
        add(
            native("stitch-tiling"),
            "internal-model",
            [`preserves every output/header/sentinel byte for S=${count} at Y=1/formula/overprovided, twice`],
            "Source compare(StitchCase) constructs synthetic segment record bytes and sentinels; CPU stitch expected bytes"
        );
    }
    add(
        native("stitch-tiling"),
        "internal-model",
        [
            "preserves mixed runs, nonzero first/runBase, exact fit and capacity overflow by one",
            "preserves the earliest failure's exact signed status and diagnostics across all tiles",
            "keeps ragged 300-run global-Y overdispatch and high run indices exact, including failures and overflow",
        ],
        "Source StitchCase counts, failures and offsets feed the CPU byte model, not a font shaper"
    );
    const plan = "tests/lite/unit/compute-text-plan.test.ts";
    add(plan, "negative", ["rejects a forged script/direction pair and does not read extra option fields"], "Forged selection object");
    add(
        plan,
        "structural",
        [
            "includes a required feature that is not in the default set",
            "deduplicates lookup references and closes an extension-wrapped dependency",
            "uses an empty plan for a missing layout table, an absent script, or a missing default LangSys",
        ],
        "Local sfnt table construction and expected lookup-index structure; no shaped records"
    );
    const program = "tests/lite/unit/compute-text-program.test.ts";
    add(
        program,
        "structural",
        [
            "packs a tight program buffer and shares mark-class base lists without changing per-rule anchors",
            "reuses immutable parsed actions without aliasing a different font snapshot or lookup",
            "shares class coverage and indexes every nonzero or rejected class-pair cell without changing logical rule order",
            "classifies only dependency-free single and pair roots for cooperative predicate selection",
            "compiles Roboto's Latin plan as executable text-shaper actions",
            "keeps Arabic Context-3 roots oracle-inert without weakening Inter rejection predicates",
            "compiles a single substitution and does not emit a no-op mode",
        ],
        "Compiler/parser rule directories, immutable storage or lookup digest; no six-field expectation fabricated"
    );
    add(
        program,
        "negative",
        ["rejects an unknown selected lookup type instead of skipping it", "keeps an understood alternate substitution as a fail-if-matched rejection"],
        "Local unsupported sfnt lookup rejection assertions"
    );
    for (const font of [inter, roboto, "NotoSansArabic-Regular.ttf"]) {
        add(program, "structural", [`packs stable sorted mark anchors with exact signed positions: ${font}`], "Parsed anchors vs packed words", font);
    }
    for (const font of [inter, roboto]) {
        add(
            program,
            "structural",
            [
                `indexes every first predicate in original order, including rejection and class zero: ${font}`,
                `encodes ordered eligibility and forward-only proof without changing cooperative capabilities: ${font}`,
            ],
            "Compiled predicate/capability/ordering metadata, not oracle glyph output",
            font
        );
    }
    const shaper = "tests/lite/unit/compute-text-shaper.test.ts";
    add(
        shaper,
        "mock",
        [
            "omits unused offset vectors in shaping batches without changing standalone offsets or UTF-16 rejection priority",
            "selects each public input once and preserves executor snapshot isolation",
            "rejects an incomplete executor batch instead of publishing missing nonempty runs as empty",
            "checks every signed publication field without allocating an intermediate numeric array per glyph",
            "rejects a second batch, dispose, and device loss without publishing",
        ],
        "Installed mock executor/device supplies records or deliberately invalid results"
    );
    add(
        shaper,
        "structural",
        [
            "reuses immutable preparation and compiles once while repacking every warm revision and capacity",
            "bulk-packs current nominal glyph vectors without transmitting redundant initial scalar-index clusters",
            "packs every class definition as a constant-time glyph-indexed table including the class-zero complement",
            "compacts first-only pair rows without losing original cell deltas or rejection modes",
            "requires a non-leading predicate witness before visiting dormant Inter composition roots",
            "bypasses absent roots before fetching their global lookup descriptors",
            "does not re-fetch successfully committed disjoint actions in the serial rejection scan",
            "uses bounded binary membership for remaining glyph-set predicates, not linear coverage scans",
            "does not load GDEF classes for flags that have no filtering effect",
            "compacts only ordered substitution roots, never independent roots or an already compact GPOS transition",
            "compacts ordered substitution scratch only after a successful ligature creates tombstones",
            "refreshes root presence only after glyph identities or live membership actually change",
            "maintains nominal advances during initialization and glyph replacement without a redundant whole-run GPOS transition",
            "keeps the GPU kernel behind a dynamic import and fails closed on mode 5",
        ],
        "Packed preparation words and source/code-string structure, not independently shaped glyph values"
    );
    for (const font of [inter, roboto]) {
        add(shaper, "structural", [`packs exhaustive first-32-root presence masks from candidate directories: ${font}`], "Exhaustive candidate-bit construction", font);
    }
    add(
        native("milestones"),
        "negative",
        ["fail-closed corpus"],
        "Original UTF16/control/script/joiner/variation/unmapped cases; obsolete C4 rejections are shaped rows separately"
    );
    add(
        native("kernel"),
        "negative",
        ["retains original root-first rejection priority across independently positioned glyphs"],
        "Batch ['', 'A', 'ABC'] matches nonzero default variation delta; positive non-matching pairs are registered separately",
        synthetic("pair-reject")
    );
    add(
        native("kernel"),
        "negative",
        ["understood alternate is rejected when matched, never a mode-5 noop"],
        "A matches Alternate; B is a separately registered positive row",
        synthetic("alternate")
    );
    add(
        native("kernel"),
        "negative",
        ["GPOS point-anchor rejection identifies its table and original run after empty-run filtering"],
        "Batch ['', 'B', 'AB'] matches a contour anchor; B is registered separately",
        synthetic("point-anchor")
    );
    add(
        native("kernel"),
        "structural",
        ["storage/buffer/workgroup exact boundaries use real device limits without near-OOM allocation"],
        "CPU exact storage/workgroup limit checks and capacity0 error; the positive Gothic record is registered separately"
    );
    add(
        native("resident"),
        "negative",
        ["rejection clears every indirect word and recovery keeps the caller atlas"],
        "Newline, mixed script, lone surrogate and unmapped emoji; obsolete C4 rejections are shaped rows separately"
    );
    add(
        "lab/lite/src/lite/compute-text-shaper-test.ts",
        "negative",
        ["run"],
        "NotoSansSC subset with unmapped Latin A; no expected glyph records",
        "NotoSansSC-Regular-subset.ttf"
    );
    add(
        "tests/lite/plumbing/compute-text-shaper.spec.ts",
        "negative",
        ["compute text shaper matches text-shaper on a real WebGPU device"],
        "NotoSansSC subset with unmapped Latin A; no expected glyph records",
        "NotoSansSC-Regular-subset.ttf"
    );
    return [...rows, ...DIFFERENTIAL_CASES];
}

export const PLAN_INDEX_CASES = [inter, roboto].map((font) => ({
    source: `tests/lite/unit/compute-text-plan.test.ts#locks ${font} latn/und/ltr roots and nested dependencies`,
    font,
    script: "latn",
    direction: "ltr" as const,
    feature: "default",
    kind: "plan-index",
}));

/** Frozen fixed enumeration of the manually audited source-label sets.
 * Hash = SHA256(JSON.stringify(sorted distinct full source labels)).
 * The readable labels are in the recipes/ledger above and coverage-enumeration.json.
 * These pins are NOT regenerated by the reference generator. They lock exact set
 * equality without discovering or executing native tests. R1 differential-only
 * titles retain their labels without instantiating runtime-derived edits. */
export const FIXED_SOURCE_ACCOUNTING = {
    "lab/lite/src/lite/compute-text-shaper-test.ts": { labels: 2, sha256: "be28bf2d2f8186c33dd17c88f0fb4c7482302e5c6b9a8b93db4cc0fc99296791" },
    "tests/lite/compute/kernel.test.ts": { labels: 26, sha256: "52ccae9dd4b807b6b7dfa7075cf6c462643db8336bacf3a2baa326168198710f" },
    "tests/lite/compute/lifecycle.test.ts": { labels: 4, sha256: "d55f3e480c3ac426b3ae19a1df00b7719c5bfbd2c6ccadbe8fc31397bd617eea" },
    "tests/lite/compute/long-run.test.ts": { labels: 16, sha256: "4003ad47d18ab848e757505ac13b36dbb7c9a9ac6bf585a981b87c8ef137d98f" },
    "tests/lite/compute/milestones.test.ts": { labels: 15, sha256: "5e4d636807a311681604fbe2df66c4cb03cd8288438ab5d851edcafb44a576b2" },
    "tests/lite/compute/ordered-parallel.test.ts": { labels: 12, sha256: "2451910ba2da8b6074a28f2fb5ca520aedfb16b8d7a1c95e6f92d2fd403e9341" },
    "tests/lite/compute/paragraph-chunks.test.ts": { labels: 37, sha256: "d82b97be878c4a2106430de21221223c49bbaf8dc3c73441b3a90f1c1c49918c" },
    "tests/lite/compute/paragraph-layout.test.ts": { labels: 54, sha256: "2895d4607511089b0c7d015526abd37da1aee467737572a78da8f294d75986bc" },
    "tests/lite/compute/parallel-refresh.test.ts": { labels: 19, sha256: "e57fe34a5dd219430872a9fed732938ef70ab73551abd1726ed406e9579fb22a" },
    "tests/lite/compute/resident.test.ts": { labels: 33, sha256: "a88358fa4c56f600c8811d276803d859d31330fd82bf5393981dd23c343e925d" },
    "tests/lite/compute/segment.test.ts": { labels: 17, sha256: "7dd6da048e26dd1ba36601d261bb3618fe4d3f18871abac82ef1478f058c06b7" },
    "tests/lite/compute/stitch-routing.test.ts": { labels: 9, sha256: "8c0e5944258bd75ae2609cf8759de3ef7195bd3773acb644470f266ed3e98a36" },
    "tests/lite/compute/stitch-tiling.test.ts": { labels: 11, sha256: "5c9bb168e0b44ad49f76c1f5c1e292c2620b677ab4b35f87f946e0ac00492020" },
    "tests/lite/plumbing/compute-text-shaper.spec.ts": { labels: 2, sha256: "86c4eca09eafbd664c99e07174508f372363bdde57f6bb697a86048d85e90314" },
    "tests/lite/unit/compute-text-plan.test.ts": { labels: 6, sha256: "41dfc7e68da81e0ad38c9754c3feb351d1fc084b284a3256a64d3c08119aa342" },
    "tests/lite/unit/compute-text-program.test.ts": { labels: 17, sha256: "75856cfce2dc8766238acdb9ffc469f81d0d6a3e22bac7c93d5b4766692f0ccf" },
    "tests/lite/unit/compute-text-shaper.test.ts": { labels: 24, sha256: "97513d2fbe5b155d0682fdaea86e023d1a3fe1c96a871d2069060e0dee7d3e92" },
} as const;
