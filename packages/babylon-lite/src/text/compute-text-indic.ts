/** Lazy Devanagari profile. No import-time registration.
 * Boundaries and scheduled follow-ups: docs/lite/architecture/compute-text-follow-ups.md.
 */
import { installComputeTextProfile } from "./compute-text-plan.js";
import { buildIndicPlan, validateIndicRun } from "./compute-text-indic-plan.js";
import { indicGraphemeChainEnd, indicParagraphFlags, proveIndicParagraphSpaces } from "./compute-text-indic-paragraph.js";
import { INDIC_KERNEL_FRAGMENTS } from "./_gpu/compute-text-indic-wgsl.js";

export function installComputeTextIndicProfile(): void {
    installComputeTextProfile("indic", {
        glyphWords: 16,
        masked: true,
        planes: 2,
        auxBytes: (capacity) => capacity * 16,
        fragments: INDIC_KERNEL_FRAGMENTS,
        buildPlan: buildIndicPlan,
        validateRun: validateIndicRun,
        available: true,
        paragraph: { spaces: proveIndicParagraphSpaces, flags: indicParagraphFlags, chainEnd: indicGraphemeChainEnd },
    });
}
