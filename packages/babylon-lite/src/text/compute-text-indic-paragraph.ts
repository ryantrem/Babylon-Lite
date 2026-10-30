/** Lazy Devanagari paragraph rules: UAX #29 extended-grapheme flag bytes and their dependency extent.
 * Pure functions over scalar arrays; no module state. Loaded only with the indic profile.
 */
import type { ComputeTextSpaceProof } from "./compute-text-segment.js";
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
