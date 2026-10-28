/** Lazy Arabic property/variant support. Contextual decisions run only on the GPU.
 * Explicit first-increment font boundary; no presentation-form fallback.
 * See docs/lite/architecture/compute-text-follow-ups.md CTF-01/02/06/07/08.
 */
import { installComputeTextProfile } from "./compute-text-plan.js";
import { ARABIC_JOINING_RANGES, ARABIC_MARK_RANGES } from "./compute-text-arabic-joining.js";
import { ARABIC_KERNEL_FRAGMENTS } from "./_gpu/compute-text-arabic-wgsl.js";

function property(ranges: Uint32Array, scalar: number): number {
    let low = 0;
    let high = ranges.length / 3;
    while (low < high) {
        const middle = (low + high) >>> 1;
        if (ranges[middle * 3 + 1]! < scalar) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low * 3 < ranges.length && ranges[low * 3]! <= scalar ? ranges[low * 3 + 2]! : 0;
}

/** @internal U=0, R=1, D=2, L=3, T=4 (Join_Causing stored as D). */
export function arabicJoiningType(scalar: number): number {
    return property(ARABIC_JOINING_RANGES, scalar);
}

/** @internal Arabic modified combining class; independent of surrounding text. */
export function arabicMarkClass(scalar: number): number {
    return property(ARABIC_MARK_RANGES, scalar) & 255;
}

/** @internal UTR #53 modifier combining mark. */
export function arabicIsMcm(scalar: number): boolean {
    return (property(ARABIC_MARK_RANGES, scalar) & 256) !== 0;
}

/** @internal No eager registration; called by the lazy loader. */
export function installComputeTextArabicProfile(): void {
    installComputeTextProfile("arabic", {
        glyphWords: 8,
        masked: true,
        planes: 1,
        joiningType: arabicJoiningType,
        markClass: arabicMarkClass,
        isMcm: arabicIsMcm,
        inputProperties: (scalar) => {
            const mark = property(ARABIC_MARK_RANGES, scalar);
            return (arabicJoiningType(scalar) << 16) | ((mark & 255) << 24) | ((mark & 256) << 11);
        },
        fragments: ARABIC_KERNEL_FRAGMENTS,
    });
}
