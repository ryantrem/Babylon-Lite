import { globals } from "webgpu";

// This file is exclusive to the opted-in project; never install an executor.
Object.assign(globalThis, globals);
