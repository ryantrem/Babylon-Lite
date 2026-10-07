/** Shared vertex/storage ABI. No shaping, packing, or GPU resource ownership. */

/** Anchor x/y floats followed by `(atlas slot | style index << 16)` as a u32. */
export const TEXT_INSTANCE_FLOATS = 3;
export const TEXT_INSTANCE_BYTES = TEXT_INSTANCE_FLOATS * 4;

/** WGSL TextStyle: color vec4, then params vec4 (inverse scale, optional style parameter, padding). */
export const TEXT_STYLE_FLOATS = 8;
export const TEXT_STYLE_BYTES = TEXT_STYLE_FLOATS * 4;
