/** One-direction script selection for the bounded compute shaper.
 *
 *  Unicode Script values come from the committed Unicode 16.0.0 range table.
 *  Common and Inherited characters take the single surrounding strong script.
 *  They never hide a second strong script, and a leading inherited mark is not
 *  relabeled Latin when no admitted base script exists. */

import { scriptTagForScalar } from "./unicode-script-data.js";
import type { ComputeFont, ComputeNominalVector } from "./compute-font.js";
import { mapComputeFontScalars } from "./compute-font.js";
import type { ComputeTextPlan, ComputeTextProfileId } from "./compute-text-plan.js";

let scriptPages: (Uint8Array | undefined)[] | null = null;
let markPages: (Uint8Array | undefined)[] | null = null;

/** @internal Shared scalar flag classification for both paragraph storage paths. */
export function computeTextNoBreakBefore(cp: number): number {
    if (cp < 0x80) {
        return 0;
    }
    if (cp === 0x200d) {
        return 4;
    }
    const pages = (markPages ??= []);
    const index = cp >>> 8;
    const page = (pages[index] ??= new Uint8Array(256));
    const offset = cp & 255;
    let value = page[offset]!;
    if (value === 0) {
        value = /\p{M}/u.test(String.fromCodePoint(cp)) ? 5 : 1;
        page[offset] = value;
    }
    return value - 1;
}

declare const computeTextRunSelectionBrand: unique symbol;

/** Admitted strong scripts, plus `DFLT` for empty or Common-only input. */
export type ComputeTextScript = "latn" | "grek" | "cyrl" | "hebr" | "arab" | "hani" | "goth" | "deva" | "DFLT";

const ADMITTED_SCRIPTS: readonly ComputeTextScript[] = ["latn", "grek", "cyrl", "hebr", "arab", "hani", "goth", "deva"];
const EMPTY_OFFSETS = /* @__PURE__ */ new Uint32Array(0);

/** Horizontal direction of one admitted run. */
export type ComputeTextDirection = "ltr" | "rtl";

/** Validated scalar run. Scalar arrays and UTF-16 offsets stay internal. */
export interface ComputeTextRunSelection {
    readonly [computeTextRunSelectionBrand]: true;
    readonly script: ComputeTextScript;
    readonly direction: ComputeTextDirection;
    readonly language: null;
    readonly scalarCount: number;
    /** @internal Unicode scalars in source order. */
    readonly _scalars: Uint32Array;
    /** @internal UTF-16 index of each scalar; empty when internal shaping omits this metadata. */
    readonly _utf16Offsets: Uint32Array;
    /** @internal Checked nominal input for this freshly decoded run. */
    readonly _nominal?: ComputeNominalVector;
}

/** @internal Fresh source-ordered batch input regions. */
export interface ComputeTextInputSlab {
    readonly buffer: ArrayBuffer;
    readonly includeUtf16Offsets: boolean;
    byteOffset: number;
}

/** @internal Input strings have already passed the caller's snapshot/type validation. */
export function createComputeTextInputSlab(texts: readonly string[], includeUtf16Offsets = true): ComputeTextInputSlab {
    let length = 0;
    for (const text of texts) {
        length += text.length;
    }
    return { buffer: new ArrayBuffer(length * (includeUtf16Offsets ? 12 : 8)), includeUtf16Offsets, byteOffset: 0 };
}

/** Decode `text`, reject unsupported input, and resolve one script and direction. */
export function selectComputeTextRun(text: string): ComputeTextRunSelection;
/** @internal Copy checked nominal entries and current closure seeds while decoding. */
export function selectComputeTextRun(text: string, font: ComputeFont, seeds?: Set<number>, slab?: ComputeTextInputSlab): ComputeTextRunSelection;
export function selectComputeTextRun(text: string, font?: ComputeFont, seeds?: Set<number>, slab?: ComputeTextInputSlab): ComputeTextRunSelection {
    if (typeof text !== "string") {
        throw new Error("selectComputeTextRun: text must be a string");
    }
    const includeUtf16Offsets = slab?.includeUtf16Offsets ?? true;
    const bytes = text.length * ((font ? 8 : 4) + (includeUtf16Offsets ? 4 : 0));
    const buffer = slab?.buffer ?? new ArrayBuffer(bytes);
    const byteOffset = slab?.byteOffset ?? 0;
    if (slab) {
        slab.byteOffset += bytes;
    }
    const scalars = new Uint32Array(buffer, byteOffset, text.length);
    const utf16Offsets = includeUtf16Offsets ? new Uint32Array(buffer, byteOffset + text.length * 4, text.length) : EMPTY_OFFSETS;
    const glyphIds = font ? new Uint32Array(buffer, byteOffset + text.length * (includeUtf16Offsets ? 8 : 4), text.length) : undefined;
    const nominalPages = font?._nominalPages;
    const glyphCount = font?.glyphCount ?? 0;
    let misses = 0;
    let lastSeed = -1;
    let lastScalar = -1;
    let lastGlyph: number | undefined;
    let pageIndex = -1;
    let nominalPage: Uint32Array | undefined;
    let classificationPage: Uint8Array | undefined;
    let count = 0;
    let strong = 0;
    let sawInherited = false;
    let deferred: Error | null = null;
    let joinerIndex = -1;
    for (let offset = 0; offset < text.length; offset++) {
        const start = offset;
        const unit = text.charCodeAt(offset);
        let scalar = unit;
        if (unit >= 0xd800 && unit <= 0xdbff) {
            const next = text.charCodeAt(offset + 1);
            if (!(next >= 0xdc00 && next <= 0xdfff)) {
                throw new Error(`selectComputeTextRun: lone surrogate at UTF-16 index ${offset}`);
            }
            scalar = (unit - 0xd800) * 0x400 + (next - 0xdc00) + 0x10000;
            offset++;
        } else if (unit >= 0xdc00 && unit <= 0xdfff) {
            throw new Error(`selectComputeTextRun: lone surrogate at UTF-16 index ${offset}`);
        }
        const index = count++;
        scalars[index] = scalar;
        if (includeUtf16Offsets) {
            utf16Offsets[index] = start;
        }
        const repeated = scalar === lastScalar;
        if (repeated) {
            if (glyphIds) {
                if (lastGlyph !== undefined) {
                    glyphIds[index] = lastGlyph;
                } else {
                    misses++;
                }
            }
            continue;
        }
        lastScalar = scalar;
        const scalarPage = scalar >>> 8;
        const pageOffset = scalar & 255;
        if (scalarPage !== pageIndex) {
            pageIndex = scalarPage;
            nominalPage = nominalPages?.[pageIndex];
            classificationPage = scriptPages?.[pageIndex];
        }
        if (glyphIds) {
            const glyph = nominalPage?.[pageOffset];
            if (glyph !== undefined && glyph > 0 && glyph < glyphCount) {
                lastGlyph = glyph;
                glyphIds[index] = glyph;
                if (seeds && misses === 0 && glyph !== lastSeed) {
                    seeds.add(glyph);
                    lastSeed = glyph;
                }
            } else {
                lastGlyph = undefined;
                misses++;
            }
        }
        if (deferred) {
            if (joinerIndex >= 0 && strong === 0) {
                const later = classifyScalar(scalar);
                if (later >= 3 && later <= 10) {
                    strong = later;
                }
            }
            continue;
        }
        let classification = classificationPage?.[pageOffset] ?? 0;
        if (classification === 0) {
            classification = classifyScalar(scalar);
            classificationPage = scriptPages?.[pageIndex];
        }
        if (classification === 1) {
            continue;
        }
        if (classification === 2) {
            sawInherited = true;
            continue;
        }
        // Rare classes nest here so the common strong-script path stays one compare.
        if (classification >= 10) {
            if (classification === 15) {
                if (joinerIndex < 0) {
                    joinerIndex = index;
                }
                continue;
            }
            if (classification !== 10) {
                deferred = new Error(`selectComputeTextRun: ${classificationRejection(classification, scalar)} at scalar ${index} ${formatScalar(scalar)}`);
                continue;
            }
        }
        if (strong === 0) {
            strong = classification;
        } else if (strong !== classification) {
            deferred = new Error(`selectComputeTextRun: mixed scripts "${ADMITTED_SCRIPTS[strong - 3]}" and "${ADMITTED_SCRIPTS[classification - 3]}"`);
        }
    }
    // Joiners are recorded only before deferral, so they always precede any deferred error.
    if (joinerIndex >= 0 && strong !== 10) {
        throw new Error(`selectComputeTextRun: joining control at scalar ${joinerIndex} ${formatScalar(scalars[joinerIndex]!)}`);
    }
    if (deferred) {
        throw deferred;
    }
    const decoded = {
        scalars: count === text.length ? scalars : scalars.subarray(0, count),
        utf16Offsets: count === text.length || !includeUtf16Offsets ? utf16Offsets : utf16Offsets.subarray(0, count),
        nominal: glyphIds ? { font: font!, glyphIds: count === text.length ? glyphIds : glyphIds.subarray(0, count), misses, seeds } : undefined,
    };
    if (strong === 0) {
        if (sawInherited) {
            throw new Error("selectComputeTextRun: inherited mark without an admitted base script");
        }
        return finish("DFLT", "ltr", decoded);
    }
    const script = ADMITTED_SCRIPTS[strong - 3]!;
    return finish(script, directionFor(script), decoded);
}

function classifyScalar(scalar: number): number {
    const pages = (scriptPages ??= []);
    const index = scalar >>> 8;
    let page = pages[index];
    if (!page) {
        page = new Uint8Array(256);
        pages[index] = page;
    }
    const offset = scalar & 255;
    let classification = page[offset]!;
    if (classification === 0) {
        const rejection = rejectionClass(scalar);
        if (rejection) {
            classification = rejection;
        } else {
            const tag = scriptTagForScalar(scalar);
            const admitted = ADMITTED_SCRIPTS.findIndex((script) => script === tag);
            // Devanagari appends strong code10. Unknown moves to private code17;
            // existing scripts/rejection messages and codes12..16 stay intact.
            classification = tag === "Zyyy" ? 1 : tag === "Zinh" ? 2 : tag === "Zzzz" ? 17 : admitted >= 0 ? admitted + 3 : 11;
        }
        page[offset] = classification;
    }
    return classification;
}

/** @internal Non-throwing cold-profile pre-scan. Warm callers need not scan again.
 * Reuses Script classification pages, never nominal mapping or contextual shaping.
 */
export function computeTextRequiredProfiles(texts: readonly string[]): ComputeTextProfileId[] {
    let arabic = false;
    let indic = false;
    for (const text of texts) {
        if (typeof text !== "string" || !/[\u0080-\u{10ffff}]/u.test(text)) {
            continue;
        }
        let strong = 0;
        let invalid = false;
        let joiner = false;
        for (const character of text) {
            const scalar = character.codePointAt(0)!;
            if (scalar >= 0xd800 && scalar <= 0xdfff) {
                invalid = true;
                break;
            }
            const classification = classifyScalar(scalar);
            if (classification === 15) {
                joiner = true;
                continue;
            }
            if (classification > 10) {
                invalid = true;
                break;
            }
            if (classification >= 3) {
                if (strong !== 0 && strong !== classification) {
                    invalid = true;
                    break;
                }
                strong = classification;
            }
        }
        if (!invalid && (!joiner || strong === 10)) {
            arabic ||= ADMITTED_SCRIPTS[strong - 3] === "arab";
            indic ||= strong === 10;
        }
    }
    return [...(arabic ? ["arabic" as const] : []), ...(indic ? ["indic" as const] : [])];
}

/** @internal Non-throwing Script scan; Common/Inherited block members are not strong. */
export function computeTextHasDevanagari(text: string): boolean {
    if (!/[\u0080-\u{10ffff}]/u.test(text)) {
        return false;
    }
    for (const character of text) {
        if (classifyScalar(character.codePointAt(0)!) === 10) {
            return true;
        }
    }
    return false;
}
/** @internal Paragraph-level admission without a paragraph-sized decode allocation. */
export function scanComputeTextRun(text: string): Pick<ComputeTextRunSelection, "script" | "direction" | "language" | "scalarCount"> {
    let strong = 0;
    let inherited = false;
    let count = 0;
    let deferred: Error | undefined;
    let joinerIndex = -1;
    let joinerScalar = 0;
    for (let offset = 0; offset < text.length; offset++) {
        const unit = text.charCodeAt(offset);
        let scalar = unit;
        if (unit >= 0xd800 && unit <= 0xdbff) {
            const next = text.charCodeAt(offset + 1);
            if (!(next >= 0xdc00 && next <= 0xdfff)) {
                throw new Error(`selectComputeTextRun: lone surrogate at UTF-16 index ${offset}`);
            }
            scalar = (unit - 0xd800) * 0x400 + next - 0xdc00 + 0x10000;
            offset++;
        } else if (unit >= 0xdc00 && unit <= 0xdfff) {
            throw new Error(`selectComputeTextRun: lone surrogate at UTF-16 index ${offset}`);
        }
        const index = count++;
        if (deferred) {
            if (joinerIndex >= 0 && strong === 0) {
                const later = classifyScalar(scalar);
                if (later >= 3 && later <= 10) {
                    strong = later;
                }
            }
            continue;
        }
        const classification = classifyScalar(scalar);
        if (classification === 2) {
            inherited = true;
        } else if (classification === 15) {
            if (joinerIndex < 0) {
                joinerIndex = index;
                joinerScalar = scalar;
            }
        } else if (classification > 10) {
            deferred = new Error(`selectComputeTextRun: ${classificationRejection(classification, scalar)} at scalar ${index} ${formatScalar(scalar)}`);
        } else if (classification >= 3) {
            if (!strong) {
                strong = classification;
            } else if (strong !== classification) {
                deferred = new Error(`selectComputeTextRun: mixed scripts "${ADMITTED_SCRIPTS[strong - 3]}" and "${ADMITTED_SCRIPTS[classification - 3]}"`);
            }
        }
    }
    // Joiners are recorded only before deferral, so they always precede any deferred error.
    if (joinerIndex >= 0 && strong !== 10) {
        throw new Error(`selectComputeTextRun: joining control at scalar ${joinerIndex} ${formatScalar(joinerScalar)}`);
    }
    if (deferred) {
        throw deferred;
    }
    if (!strong && inherited) {
        throw new Error("selectComputeTextRun: inherited mark without an admitted base script");
    }
    const script = strong ? ADMITTED_SCRIPTS[strong - 3]! : "DFLT";
    return { script, direction: directionFor(script), language: null, scalarCount: count };
}

/** @internal Decode a bounded, previously validated source region. */
export function decodeComputeTextChunk(text: string): { scalars: Uint32Array; offsets: number[] } {
    const values: number[] = [];
    const offsets: number[] = [];
    for (let i = 0; i < text.length; i++) {
        offsets.push(i);
        const cp = text.codePointAt(i)!;
        values.push(cp);
        if (cp > 0xffff) {
            i++;
        }
    }
    offsets.push(text.length);
    return { scalars: Uint32Array.from(values), offsets };
}

/** @internal Materialize one bounded region under its paragraph-level selected plan. */
export function prepareComputeTextChunk(source: { readonly text: string; readonly plan: ComputeTextPlan; readonly segments: readonly number[] }) {
    const decoded = decodeComputeTextChunk(source.text);
    const glyphIds = mapComputeFontScalars(source.plan.font, decoded.scalars, (_i, error) => {
        throw error;
    });
    let first = 0;
    const runs = source.segments.map((end) => {
        const run = { scalars: decoded.scalars.subarray(first, end), glyphIds: glyphIds.subarray(first, end) };
        first = end;
        return run;
    });
    return { ...decoded, glyphIds, runs };
}

/** Direction required by an admitted script. `DFLT` is LTR. */
export function directionForComputeTextScript(script: ComputeTextScript): ComputeTextDirection {
    return directionFor(script);
}

function finish(
    script: ComputeTextScript,
    direction: ComputeTextDirection,
    decoded: { scalars: Uint32Array; utf16Offsets: Uint32Array; nominal?: ComputeNominalVector }
): ComputeTextRunSelection {
    return {
        script,
        direction,
        language: null,
        scalarCount: decoded.scalars.length,
        _scalars: decoded.scalars,
        _utf16Offsets: decoded.utf16Offsets,
        _nominal: decoded.nominal,
    } as ComputeTextRunSelection;
}

function rejectionClass(scalar: number): number {
    if (scalar <= 0x1f || (scalar >= 0x7f && scalar <= 0x9f)) {
        return 12;
    }
    if (scalar === 0x2028 || scalar === 0x2029) {
        return 13;
    }
    if ((scalar >= 0x202a && scalar <= 0x202e) || (scalar >= 0x2066 && scalar <= 0x2069) || scalar === 0x61c) {
        return 14;
    }
    if (scalar === 0x200c || scalar === 0x200d) {
        return 15;
    }
    if ((scalar >= 0xfe00 && scalar <= 0xfe0f) || (scalar >= 0xe0100 && scalar <= 0xe01ef)) {
        return 16;
    }
    return 0;
}

function classificationRejection(classification: number, scalar: number): string {
    switch (classification) {
        case 17:
            return "unknown script";
        case 11:
            return `unsupported script "${scriptTagForScalar(scalar)}"`;
        case 12:
            return "control";
        case 13:
            return "paragraph separator";
        case 14:
            return "bidi control";
        case 15:
            return "joining control";
        default:
            return "variation selector";
    }
}

function directionFor(script: ComputeTextScript): ComputeTextDirection {
    return script === "hebr" || script === "arab" ? "rtl" : "ltr";
}

function formatScalar(scalar: number): string {
    return `U+${scalar.toString(16).toUpperCase().padStart(4, "0")}`;
}
