/** Empty text block state, shared by CPU data and resident layer compatibility data. */

import type { GlyphStorage } from "./glyph-storage.js";
import type { GlyphRun, TextData } from "./text-data.js";
import { TEXT_INSTANCE_FLOATS, TEXT_STYLE_FLOATS } from "./text-buffer-layout.js";

export function createEmptyTextData(storage: GlyphStorage): TextData {
    const runs: GlyphRun[] = [];
    const instances = new Float32Array(TEXT_INSTANCE_FLOATS);
    return {
        runs,
        _runs: runs,
        _groups: [],
        _runRecords: new Map(),
        _instances: instances,
        _instancesU32: new Uint32Array(instances.buffer),
        _instanceCount: 0,
        _styles: new Float32Array(TEXT_STYLE_FLOATS),
        _styleCount: 0,
        _freeStyleSlots: [],
        _styleVersion: 1,
        _storage: storage,
        _version: 1,
        _layoutVersion: 0,
        _dirtyStart: 0,
        _dirtyEnd: 0,
    } as unknown as TextData;
}
