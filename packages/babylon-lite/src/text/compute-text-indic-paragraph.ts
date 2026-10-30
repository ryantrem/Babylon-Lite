/** Lazy Devanagari paragraph rules: the barrier space proof (architecture §4 of CTF-03), UAX #29
 * extended-grapheme flag bytes and their dependency extent. Loaded only with the indic profile;
 * the only module state is the lazily created per-program proof cache.
 */
import type { ComputeTextProgram } from "./compute-text-program.js";
import { COMPUTE_RULE_MODE_MARK_BASE, COMPUTE_RULE_MODE_MARK_LIGATURE, COMPUTE_RULE_MODE_MARK_MARK } from "./compute-text-program.js";
import type { ComputeTextCrossingRule, ComputeTextCrossings, ComputeTextSpaceProof } from "./compute-text-segment.js";
import { enumerateCrossings, matches, spaceContext } from "./compute-text-segment.js";
import { indicProperties } from "./compute-text-indic-plan.js";
import { INDIC_GRAPHEME } from "./compute-text-indic-unicode.js";

const SPACE = 0x20;
const ADMITTED = 0x10000000;
/** GCB/InCB code of an admitted scalar outside every rule: it resets all grapheme context. */
const RESET = 0;
const UNMODELED = -1;
const GCB_PREPEND = 4;
const INCB_CONSONANT = 1;
const INCB_EXTEND = 2;
const INCB_LINKER = 3;
/** D2 word-extent bound E: paragraph context windows (65 back, 66 forward) are exact for it. */
const MAX_WORD = 64;
/** Stage semantics the barrier obligations cover; any other applied feature makes the proof unavailable (O7). */
const MODELED_FEATURES = "ccmp locl nukt akhn rphf rkrf blwf half cjct pres abvs blws psts haln rlig calt clig rclt abvm blwm dist mark mkmk kern".split(" ");
/** Syllable-DFA actions that end the pending token at or before the current scalar and rescan it from the start state. */
const DFA_BACKOFF = [1, 3, 4, 6, 7, 8, 14, 15, 16, 17, 18, 20];
/** Syllable-DFA action that accepts a one-scalar token ending at the current scalar. */
const DFA_ACCEPT_ONE = 11;

let proofs: WeakMap<ComputeTextProgram, { proof: ComputeTextSpaceProof | null; reason: string | null }> | undefined;

/** @internal Shared B*, closure and crossing facts without the generic span induction or expanding-lookup bound.
 * Uncached and profile-agnostic; lives in this lazy module so Latin-only consumers never fetch it. */
export function collectComputeTextCrossings(program: ComputeTextProgram): ComputeTextCrossings | null {
    const context = spaceContext(program, false);
    if (!context) {
        return null;
    }
    const rules: ComputeTextCrossingRule[] = [];
    const set = enumerateCrossings(program, context.bStar, rules);
    if (!set) {
        return null;
    }
    const { closure, bStar } = context;
    const inB = set.inB;
    const attached = program.lookups.some((l) =>
        l.rules.some(
            (r) =>
                (r.mode === COMPUTE_RULE_MODE_MARK_BASE || r.mode === COMPUTE_RULE_MODE_MARK_MARK || r.mode === COMPUTE_RULE_MODE_MARK_LIGATURE) &&
                ((!!r.input[0] && bStar.some((g) => matches(r.input[0]!, g))) ||
                    (r.action.kind === "mark-ligature"
                        ? r.action.ligatures.some((b) => inB[b.glyphId] === 1)
                        : (r.action.kind === "mark-base" || r.action.kind === "mark-mark") && r.action.bases.some((b) => inB[b.glyphId] === 1)))
        )
    );
    return {
        space: context.space,
        bStar,
        closure: (glyph) => closure(glyph),
        glyphClass: context.cls,
        skipped: context.skipped,
        matches,
        attached,
        crossings: rules,
    };
}

/** `GCB << 2 | InCB` for admitted scalars; `UNMODELED` for anything `validateIndicRun` would reject. */
function graphemeClass(cp: number): number {
    if ((indicProperties(cp) & ADMITTED) === 0) {
        return UNMODELED;
    }
    let lo = 0,
        hi = INDIC_GRAPHEME.length - 1;
    while (lo < hi) {
        const mid = (lo + hi + 1) >>> 1;
        if (INDIC_GRAPHEME[mid]! >>> 5 <= cp) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return INDIC_GRAPHEME[lo]! & 31;
}

/** True when bit 4 at `index` and beyond cannot depend on any scalar before `index - 1`. */
function resetsAt(scalars: Uint32Array, index: number): boolean {
    return index === 0 || graphemeClass(scalars[index - 1]!) === RESET;
}

/** @internal Writes indic paragraph flag bytes for `[lo, hi)`: spaces are 1, or 3 when proven breakable;
 * other scalars are 4 when they do not start an extended grapheme cluster (GB9, GB9a, GB9b, GB9c), else 0.
 * Unmodeled scalars conservatively forbid every boundary until the next resetting admitted scalar.
 */
export function indicParagraphFlags(
    scalars: Uint32Array,
    glyphIds: Uint32Array,
    proof: ComputeTextSpaceProof | null,
    lo: number,
    hi: number,
    out: Uint8Array,
    outOffset: number,
    unavailable: string
): void {
    let start = lo;
    while (!resetsAt(scalars, start)) {
        start--;
    }
    let prependBefore = false,
        unmodeled = false,
        // GB9c: 0 = outside a conjunct, 1 = after a consonant, 2 = after a consonant and a linker.
        conjunct = 0;
    for (let i = start; i < hi; i++) {
        const cp = scalars[i]!,
            code = graphemeClass(cp),
            gcb = code >> 2,
            incb = code & 3;
        if (i >= lo) {
            let flag: number;
            if (cp === SPACE) {
                flag = 1;
                if (i + 1 < scalars.length && scalars[i + 1] !== SPACE) {
                    if (!proof) {
                        throw new Error(unavailable);
                    }
                    flag = proof.safeAfter(glyphIds, i, scalars) ? 3 : 1;
                }
            } else {
                const joins = gcb >= 1 && gcb <= 3;
                const boundary = i === 0 || (!unmodeled && code !== UNMODELED && !joins && !prependBefore && !(incb === INCB_CONSONANT && conjunct === 2));
                flag = boundary ? 0 : 4;
            }
            out[outOffset + i - lo] = flag;
        }
        if (code === UNMODELED) {
            unmodeled = true;
            continue;
        }
        if (code === RESET) {
            unmodeled = false;
        }
        prependBefore = gcb === GCB_PREPEND;
        conjunct = incb === INCB_CONSONANT ? 1 : incb === INCB_LINKER ? (conjunct ? 2 : 0) : incb === INCB_EXTEND ? conjunct : 0;
    }
}

/** @internal Smallest `h >= hi` with `h === scalars.length` or a resetting scalar at `h - 1`:
 * indic flag bytes at indexes `>= h` never depend on scalars before `h - 1`.
 */
export function indicGraphemeChainEnd(scalars: Uint32Array, hi: number): number {
    let end = hi;
    while (end < scalars.length && !resetsAt(scalars, end)) {
        end++;
    }
    return end;
}

/** Category-X isolation in the packed Ragel scanner: X from the start state is a one-scalar token that returns to
 * the start state, and X from any other state ends the pending token before it and returns to the start state. */
function isolatesSpace(machine: Uint32Array): boolean {
    const data = (slot: number, index: number) => machine[machine[slot]! + index]!;
    const start = machine[0]!;
    const states = machine[3]! - machine[2]!;
    for (let state = 0; state < states; state++) {
        // Category 0 lies in the state's key range [lo, hi] exactly when lo is 0.
        const span = data(2, state);
        const trans = data(4, data(3, state) + (span > 0 && data(1, state * 2) === 0 ? 0 : span));
        const action = data(6, trans);
        if (data(5, trans) !== start || (state === start ? action !== DFA_ACCEPT_ONE : !DFA_BACKOFF.includes(action))) {
            return false;
        }
    }
    return true;
}

function buildProof(program: ComputeTextProgram): ComputeTextSpaceProof | string {
    const data = program.plan._profileData;
    if (program.plan._profile !== "indic" || !data?.normalizationClosure || !data.features) {
        return "not an indic plan with paragraph profile data";
    }
    const tables = data.tables();
    const section = (slot: number, width: number) => tables.subarray(tables[slot]!, tables[slot]! + tables[slot + 1]! * width);
    if (section(8, 4).includes(SPACE)) {
        return "O1: U+0020 is in a vowel constraint";
    }
    const sparse = section(2, 2);
    let props = 0;
    for (let i = 0; i < sparse.length; i += 2) {
        if (sparse[i] === SPACE) {
            props = sparse[i + 1]!;
        }
    }
    // Flags (bits 24+): mark, DI/hidden, ZWJ, ZWNJ, admitted, decomposable.
    const decompositions = section(4, 3),
        compositions = section(6, 3);
    if ((props & 0x2fff0000) !== 0 || decompositions.includes(SPACE) || compositions.includes(SPACE)) {
        return "O2: U+0020 can reorder, decompose, compose or hide";
    }
    // Category X (bits 0-7). Its position (POS_END) is read only by per-syllable kernels, which isolation confines to the space.
    if (!data.spaceLocal || (props & 0xff) !== 0 || !isolatesSpace(tables.subarray(tables[21]!))) {
        return "O3: U+0020 is not an isolated category-X syllable";
    }
    if (data.features.includes("init")) {
        return "O4: init reads across syllables";
    }
    const crossings = collectComputeTextCrossings(program);
    if (!crossings || crossings.attached || crossings.bStar.some((g) => crossings.glyphClass(g) !== 0 || crossings.skipped(g))) {
        return "O5: the space closure can be skipped, attached or merged";
    }
    if (crossings.crossings.some((rule) => rule.spaceSlots !== 1)) {
        return "O5: a crossing window has more than one space slot";
    }
    // O6 (attachment propagation, hidden-glyph finish) follows from O5: no attachment involves the space closure.
    if (data.features.some((tag) => !MODELED_FEATURES.includes(tag))) {
        return "O7: unmodeled feature";
    }
    const rules = crossings.crossings;
    const slotCount = rules.reduce((sum, rule) => sum + rule.window.length - 1, 0);
    const maskWords = (slotCount + 31) >>> 5;
    const partners = new Map<number, number[]>();
    for (const [a, b] of data.normalizationClosure) {
        for (const [x, y] of [
            [a, b],
            [b, a],
        ] as const) {
            let list = partners.get(x);
            if (!list) {
                partners.set(x, (list = []));
            }
            list.push(y);
        }
    }
    const masks = new Map<number, Uint32Array>();
    /** Crossing slots some glyph of the seed's closure (normalization partners included, transitively) can match. */
    const seedMask = (seed: number): Uint32Array => {
        let mask = masks.get(seed);
        if (mask) {
            return mask;
        }
        mask = new Uint32Array(maskWords);
        const seeds = new Set([seed]);
        for (const g of seeds) {
            for (const partner of partners.get(g) ?? []) {
                seeds.add(partner);
            }
        }
        const glyphs = new Set<number>();
        for (const g of seeds) {
            for (const out of crossings.closure(g)) {
                glyphs.add(out);
            }
        }
        let bit = 0;
        for (const rule of rules) {
            for (let j = 0; j < rule.window.length; j++) {
                if (j === rule.slot) {
                    continue;
                }
                for (const g of glyphs) {
                    if (crossings.matches(rule.window[j]!, g)) {
                        mask[bit >>> 5]! |= 1 << (bit & 31);
                        break;
                    }
                }
                bit++;
            }
        }
        masks.set(seed, mask);
        return mask;
    };
    const dotted = tables[16]!;
    /** Word extent from `from` stepping by `step`, or -1 above `MAX_WORD`; bounded by U+0020 or the array end. */
    const extent = (scalars: Uint32Array, from: number, step: number): number => {
        let length = 0;
        for (let i = from; i >= 0 && i < scalars.length && scalars[i] !== SPACE; i += step) {
            if (++length > MAX_WORD) {
                return -1;
            }
        }
        return length;
    };
    const wordMask = (glyphIds: Uint32Array, from: number, to: number): Uint32Array => {
        const mask = seedMask(dotted).slice();
        for (let i = from; i < to; i++) {
            const seed = seedMask(glyphIds[i]!);
            for (let w = 0; w < maskWords; w++) {
                mask[w]! |= seed[w]!;
            }
        }
        return mask;
    };
    return {
        safeAfter(glyphIds, s, scalars) {
            if (!scalars) {
                throw new Error("indic paragraph proof requires source scalars");
            }
            const next = scalars[s + 1];
            if (scalars[s] !== SPACE || next === undefined || next === SPACE) {
                return false;
            }
            // D1: the right line starts an extended grapheme cluster (no leading mark or joiner).
            const code = graphemeClass(next),
                gcb = code >> 2;
            if (code === UNMODELED || (gcb >= 1 && gcb <= 3) || (indicProperties(next) & 0x1000000) !== 0) {
                return false;
            }
            // D2: both words are bounded, so every window that can see them is exact.
            const left = extent(scalars, s - 1, -1),
                right = extent(scalars, s + 1, 1);
            if (left < 0 || right < 0) {
                return false;
            }
            // D3: no crossing rule can match every slot on both sides of its space slot.
            if (!rules.length) {
                return true;
            }
            const leftMask = wordMask(glyphIds, s - left, s),
                rightMask = wordMask(glyphIds, s + 1, s + 1 + right);
            let bit = 0;
            for (const rule of rules) {
                let matched = true;
                for (let j = 0; j < rule.window.length; j++) {
                    if (j === rule.slot) {
                        continue;
                    }
                    const mask = j < rule.slot ? leftMask : rightMask;
                    if ((mask[bit >>> 5]! & (1 << (bit & 31))) === 0) {
                        matched = false;
                    }
                    bit++;
                }
                if (matched) {
                    return false;
                }
            }
            return true;
        },
    };
}

function provenFor(program: ComputeTextProgram): { proof: ComputeTextSpaceProof | null; reason: string | null } {
    let entry = proofs?.get(program);
    if (!entry) {
        const result = buildProof(program);
        entry = typeof result === "string" ? { proof: null, reason: result } : { proof: result, reason: null };
        (proofs ??= new WeakMap()).set(program, entry);
    }
    return entry;
}

/** @internal Indic paragraph space proof: barrier obligations O1–O7 decide availability, D1–D3 decide each space. */
export function proveIndicParagraphSpaces(program: ComputeTextProgram): ComputeTextSpaceProof | null {
    return provenFor(program).proof;
}

/** @internal Test diagnostic: the first failed obligation for `program`, or null when the proof is available. */
export function indicParagraphProofReason(program: ComputeTextProgram): string | null {
    return provenFor(program).reason;
}
