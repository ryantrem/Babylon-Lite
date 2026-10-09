import { updateComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";
import type { ComputeTextData } from "../../../packages/babylon-lite/src/text/compute-text-data";

/** Validate GPU status outside any measured submission path. */
export async function awaitComputeTextUpdate(data: ComputeTextData, texts: readonly string[]): Promise<void> {
    await (
        await updateComputeTextData(data, texts)
    ).completed;
}
