import { SAFE_SPACE_FLAG } from "./compute-text-segment.js";
import type { ComputeTextSpaceProof } from "./compute-text-segment.js";
import type { ComputeTextParagraphProfile } from "./compute-text-plan.js";
import type { ComputeTextParagraphFlagEntry } from "./_gpu/compute-text-paragraph-layout.js";
import { computeTextNoBreakBefore } from "./compute-text-selection.js";

/** @internal Scalar-indexed bytes, matching shape clusters and paragraph-level stitched clusters. */
export function computeTextParagraphFlags(scalars: Uint32Array, glyphIds: Uint32Array, proof: ComputeTextSpaceProof | null): Uint8Array {
    const flags = new Uint8Array(scalars.length);
    const readProof = () => proof;
    for (let i = 0; i < scalars.length; i++) {
        flags[i] = paragraphFlag(scalars, glyphIds, i, readProof);
    }
    return flags;
}

const UNAVAILABLE = "updateComputeTextParagraphs: shaping-safe space analysis is unavailable for a paragraph with break candidates";

function paragraphFlag(scalars: Uint32Array, glyphIds: Uint32Array, i: number, readProof: () => ComputeTextSpaceProof | null): number {
    const cp = scalars[i]!;
    if (cp !== 0x20) {
        return computeTextNoBreakBefore(cp);
    }
    if (i + 1 < scalars.length && scalars[i + 1] !== 0x20) {
        const proof = readProof();
        if (!proof) {
            throw new Error(UNAVAILABLE);
        }
        if (proof.safeAfter(glyphIds, i, scalars)) {
            return SAFE_SPACE_FLAG;
        }
    }
    return 1;
}

/** @internal Reuse only proven unchanged scalar/glyph neighborhoods; callback injection is for exact untimed tests.
 *  `known` lower-bounds the unchanged prefix and suffix lengths, skipping their comparison.
 *  A `profile` seam replaces the per-scalar rule and extends the recomputed range to the end of its grapheme chain. */
export function computeTextParagraphFlagsIncremental<P extends ComputeTextSpaceProof>(
    previous: Pick<ComputeTextParagraphFlagEntry, "scalars" | "glyphIds" | "flags"> | undefined,
    scalars: Uint32Array,
    glyphIds: Uint32Array,
    readProof: () => P | null,
    evaluate: (scalars: Uint32Array, glyphIds: Uint32Array, i: number, readProof: () => P | null) => number = paragraphFlag,
    known?: readonly [prefix: number, suffix: number],
    profile?: ComputeTextParagraphProfile
): Uint8Array {
    const flags = new Uint8Array(scalars.length);
    let lo = 0;
    let hi = scalars.length;
    if (previous) {
        const oldN = previous.scalars.length;
        const limit = Math.min(oldN, scalars.length);
        let prefix = 0;
        let suffix = 0;
        if (known) {
            [prefix, suffix] = known;
        } else {
            while (prefix < limit && previous.scalars[prefix] === scalars[prefix] && previous.glyphIds[prefix] === glyphIds[prefix]) {
                prefix++;
            }
            while (
                suffix < limit - prefix &&
                previous.scalars[oldN - suffix - 1] === scalars[scalars.length - suffix - 1] &&
                previous.glyphIds[oldN - suffix - 1] === glyphIds[glyphIds.length - suffix - 1]
            ) {
                suffix++;
            }
        }
        lo = Math.max(0, prefix - 66);
        hi = Math.min(scalars.length, scalars.length - suffix + 66);
        if (lo < 66) {
            lo = 0;
        }
        if (scalars.length - hi < 66) {
            hi = scalars.length;
        }
        if (profile) {
            hi = profile.chainEnd(scalars, hi);
        }
        if (hi - lo >= scalars.length / 2) {
            lo = 0;
            hi = scalars.length;
        }
        flags.set(previous.flags.subarray(0, lo));
        flags.set(previous.flags.subarray(hi - scalars.length + oldN), hi);
    }
    if (profile) {
        profile.flags(scalars, glyphIds, readProof(), lo, hi, flags, lo, UNAVAILABLE);
    } else {
        for (let i = lo; i < hi; i++) {
            flags[i] = evaluate(scalars, glyphIds, i, readProof);
        }
    }
    return flags;
}
