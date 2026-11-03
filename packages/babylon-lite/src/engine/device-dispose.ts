import type { DeviceContext } from "./device.js";

/** Destroy a device from createDevice after disposing all its engines. Repeated disposal is harmless. */
export function disposeDevice(device: DeviceContext): void {
    if (!device._engines || device._engines.length || device._recovery) {
        throw new Error("disposeDevice requires a createDevice context whose engines are disposed and whose recovery has finished.");
    }
    if (!device._disposed) {
        device._disposed = true;
        device._device.destroy();
    }
}
