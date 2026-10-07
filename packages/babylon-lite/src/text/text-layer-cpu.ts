import type { EngineContext } from "../engine/engine.js";
import type { TextData } from "./text-data.js";
import type { CurveSetId } from "./glyph-storage.js";
import { TEXT_INSTANCE_BYTES, TEXT_STYLE_BYTES } from "./text-buffer-layout.js";
import { createTextLayerPlacement } from "./text-layer.js";
import type { TextLayerDraw, TextLayerOptions, TextLayerPlacement } from "./text-layer.js";
import { ensureSharedAtlasGpu } from "./_gpu/text-textures.js";
import { ensureStyleGpu } from "./_gpu/text-style-gpu.js";
import { createStyleBuffer } from "./_gpu/text-style-buffer.js";
import { createTextBindGroup } from "./_gpu/text-bind-group.js";

/** Pure-state CPU text layer. Its TextData remains caller-owned. */
export interface TextLayer extends TextLayerPlacement {
    readonly data: TextData;
    /** @internal */
    readonly _kind: "text-layer";
}

/** Place a CPU TextData block in canvas pixel space. */
export function createTextLayer(data: TextData, options?: TextLayerOptions): TextLayer {
    return {
        ...createTextLayerPlacement(options),
        _kind: "text-layer",
        data,
        _createDraw: (engine, uniform) => createCpuDraw(engine, uniform, data),
    };
}

interface BindGroupCacheEntry {
    _bindGroup: GPUBindGroup;
    _curveSetId: CurveSetId;
    _curveTex: GPUTexture;
    _bandTex: GPUTexture;
    _metaBuf: GPUBuffer;
}

/** @internal Concrete CPU attachment state, separate from public layer/data state. */
export interface CpuTextLayerDraw extends TextLayerDraw {
    /** @internal */
    _instanceBuf: GPUBuffer;
    /** @internal */
    _instanceCap: number;
    /** @internal */
    _styleBuf: GPUBuffer;
    /** @internal */
    _uploadedStyleVersion: number;
    /** @internal */
    _uploadedDataVersion: number;
    /** @internal */
    _bindGroupCache: BindGroupCacheEntry[];
}

function createCpuDraw(engine: EngineContext, uniform: GPUBuffer, data: TextData): CpuTextLayerDraw {
    const device = engine._device;
    let bindGroupLayout: GPUBindGroupLayout | null = null;
    let layoutVersion = -1;
    const createInstances = (capacity: number) =>
        device.createBuffer({ label: "text-layer-instances", size: capacity * TEXT_INSTANCE_BYTES, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    const cap = Math.max(data._instanceCount, 8);
    const draw: CpuTextLayerDraw = {
        _active: false,
        _version: 0,
        _instanceBuf: createInstances(cap),
        _instanceCap: cap,
        _styleBuf: createStyleBuffer(device, TEXT_STYLE_BYTES),
        _uploadedStyleVersion: -1,
        _uploadedDataVersion: -1,
        _bindGroupCache: [],
        _prepare(layout) {
            const styleRecreated = ensureStyleGpu(device, data, draw);
            if (bindGroupLayout !== layout || styleRecreated) {
                bindGroupLayout = layout;
                draw._bindGroupCache.length = 0;
                draw._version++;
            }
            for (let i = 0; i < data._groups.length; i++) {
                const group = data._groups[i]!;
                const { _gpu: atlas } = ensureSharedAtlasGpu(device, group._curveSet._atlas);
                const cached = draw._bindGroupCache[i];
                if (
                    !cached ||
                    cached._curveSetId !== group._curveSetId ||
                    cached._curveTex !== atlas._curveTex ||
                    cached._bandTex !== atlas._bandTex ||
                    cached._metaBuf !== atlas._metaBuf
                ) {
                    draw._bindGroupCache[i] = {
                        _bindGroup: createTextBindGroup(device, layout, uniform, atlas, draw._styleBuf, "text-renderer-bg0-" + group._curveSetId),
                        _curveSetId: group._curveSetId,
                        _curveTex: atlas._curveTex,
                        _bandTex: atlas._bandTex,
                        _metaBuf: atlas._metaBuf,
                    };
                    draw._version++;
                }
            }
            if (draw._bindGroupCache.length > data._groups.length) {
                draw._bindGroupCache.length = data._groups.length;
                draw._version++;
            }
            if (data._instanceCount > draw._instanceCap) {
                while (data._instanceCount > draw._instanceCap) {
                    draw._instanceCap *= 2;
                }
                draw._instanceBuf.destroy();
                draw._instanceBuf = createInstances(draw._instanceCap);
                draw._uploadedDataVersion = -1;
                draw._version++;
            }
            if (draw._uploadedDataVersion !== data._version && data._instanceCount > 0) {
                // Dirty ranges describe only the latest edit. A lagging/shared attachment
                // must upload the whole block, not assume it observed intervening edits.
                const dirtyValid = draw._uploadedDataVersion === data._version - 1 && data._dirtyEnd > data._dirtyStart;
                const start = dirtyValid ? data._dirtyStart : 0;
                const end = dirtyValid ? data._dirtyEnd : data._instanceCount;
                const view = data._instances.subarray(start * (TEXT_INSTANCE_BYTES / 4), end * (TEXT_INSTANCE_BYTES / 4));
                device.queue.writeBuffer(draw._instanceBuf, start * TEXT_INSTANCE_BYTES, view.buffer as ArrayBuffer, view.byteOffset, view.byteLength);
                draw._uploadedDataVersion = data._version;
                data._dirtyStart = 0;
                data._dirtyEnd = 0;
            }
            if (layoutVersion !== data._layoutVersion) {
                layoutVersion = data._layoutVersion;
                draw._version++;
            }
            draw._active = data._instanceCount > 0;
        },
        _record(encoder, quad, base, variant) {
            encoder.setVertexBuffer(0, quad);
            encoder.setVertexBuffer(1, draw._instanceBuf);
            let bound = base;
            let count = 0;
            for (let i = 0; i < data._groups.length; i++) {
                const group = data._groups[i]!;
                if (group._slotCount === 0) {
                    continue;
                }
                const pipeline = group._groupKey === group._curveSetId ? base : variant;
                if (pipeline !== bound) {
                    encoder.setPipeline(pipeline);
                    bound = pipeline;
                }
                encoder.setBindGroup(0, draw._bindGroupCache[i]!._bindGroup);
                encoder.draw(6, group._slotCount, 0, group._slotStart);
                count++;
            }
            return count;
        },
        _dispose() {
            draw._instanceBuf.destroy();
            draw._styleBuf.destroy();
            draw._bindGroupCache.length = 0;
            draw._active = false;
        },
    };
    return draw;
}
