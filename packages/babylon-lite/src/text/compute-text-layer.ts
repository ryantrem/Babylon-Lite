import type { EngineContext } from "../engine/engine.js";
import type { ComputeTextData } from "./compute-text-data.js";
import { createTextLayerPlacement } from "./text-layer.js";
import type { TextLayerDraw, TextLayerOptions, TextLayerPlacement } from "./text-layer.js";
import { ensureSharedAtlasGpu } from "./_gpu/text-textures.js";
import { createTextBindGroup } from "./_gpu/text-bind-group.js";

/** Pure-state resident text layer. Its ComputeTextData remains caller-owned. */
export interface ComputeTextLayer extends TextLayerPlacement {
    readonly data: ComputeTextData;
    /** @internal */
    readonly _kind: "compute-text-layer";
}

/** Place a resident ComputeTextData block without CPU uploads or glyph readback. */
export function createComputeTextLayer(data: ComputeTextData, options?: TextLayerOptions): ComputeTextLayer {
    if (!data || data.status === "disposed" || data._dead) {
        throw new Error("createComputeTextLayer: data is not live");
    }
    return {
        ...createTextLayerPlacement(options),
        _kind: "compute-text-layer",
        data,
        _createDraw: (engine, uniform) => createResidentDraw(engine, uniform, data),
    };
}

function createResidentDraw(engine: EngineContext, uniform: GPUBuffer, data: ComputeTextData): TextLayerDraw {
    if (engine !== data._engine) {
        throw new Error("createComputeTextLayer: renderer must use the data's engine");
    }
    let bindGroup: GPUBindGroup | null = null;
    let bindGroupLayout: GPUBindGroupLayout | null = null;
    let curveTexture: GPUTexture | null = null;
    let bandTexture: GPUTexture | null = null;
    let metadata: GPUBuffer | null = null;
    let styles: GPUBuffer | null = null;
    let instance: GPUBuffer | null = null;
    let indirect: GPUBuffer | null = null;
    const draw: TextLayerDraw = {
        _active: false,
        _version: 0,
        _prepare(layout) {
            if (data._device !== engine._device && !data._dead) {
                data._onDeviceLost();
            }
            if (data._dead || data.status === "lost" || data.status === "disposed") {
                draw._active = false;
                bindGroup = null;
                return;
            }
            const nextInstance = data._instances._buffer;
            const nextIndirect = data._indirect._buffer;
            if (!nextInstance || !nextIndirect) {
                throw new Error("createComputeTextLayer: live data has no draw buffers");
            }
            const curveSet = data._storage._curveSets.get(data._curveSet);
            if (!curveSet && bindGroup) {
                throw new Error("createComputeTextLayer: live data's curve set is missing");
            }
            // A new block has no outlines until its first successful update.
            draw._active = curveSet !== undefined;
            if (!curveSet) {
                return;
            }
            const { _gpu: atlas } = ensureSharedAtlasGpu(engine._device, curveSet._atlas);
            if (
                !bindGroup ||
                bindGroupLayout !== layout ||
                curveTexture !== atlas._curveTex ||
                bandTexture !== atlas._bandTex ||
                metadata !== atlas._metaBuf ||
                styles !== data._style
            ) {
                bindGroup = createTextBindGroup(engine._device, layout, uniform, atlas, data._style, "compute-text-bg0");
                bindGroupLayout = layout;
                curveTexture = atlas._curveTex;
                bandTexture = atlas._bandTex;
                metadata = atlas._metaBuf;
                styles = data._style;
                draw._version++;
            }
            if (instance !== nextInstance || indirect !== nextIndirect) {
                instance = nextInstance;
                indirect = nextIndirect;
                draw._version++;
            }
        },
        _record(encoder, quad) {
            encoder.setVertexBuffer(0, quad);
            encoder.setVertexBuffer(1, instance!);
            encoder.setBindGroup(0, bindGroup!);
            encoder.drawIndirect(indirect!, 0);
            return 1;
        },
        _dispose() {
            bindGroup = null;
            instance = indirect = null;
            curveTexture = bandTexture = null;
            metadata = styles = null;
            draw._active = false;
        },
    };
    return draw;
}
