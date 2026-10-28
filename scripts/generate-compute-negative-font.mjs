/** Deterministic test-only browser fixture. Run with Node's existing tsx loader. */
import { deepStrictEqual, strictEqual } from "node:assert";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { syntheticFontBytes } = require("../tests/lite/compute/synthetic.ts");
const bytes = new Uint8Array(syntheticFontBytes("reverse-chain"));
strictEqual(createHash("sha256").update(bytes).digest("hex"), "e3236a4a6a220646407088f07fb94450b6e7323270a4eea0e2a1b988748c97a2");
const path = new URL("../lab/public/fonts/test-unsupported-reverse-chain.ttf", import.meta.url);
const mode = process.argv[2];
if (mode === "--write") {
    writeFileSync(path, bytes);
} else if (mode === "--check") {
    deepStrictEqual(new Uint8Array(readFileSync(path)), bytes);
} else {
    throw new Error("Expected --write or --check (node --import tsx scripts/generate-compute-negative-font.mjs)");
}
