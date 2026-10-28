/** Independent registry serialization; deliberately no Lite imports or test fixtures. */
import { readFileSync } from "node:fs";
import { Font } from "../packages/babylon-lite/node_modules/text-shaper/dist/index.js";
import { existingCorpus } from "../tests/lite/compute/corpus";
import { FIXED_SOURCE_ACCOUNTING, nonReferenceCorpus, PENDING_INPUTS, PLAN_INDEX_CASES } from "../tests/lite/compute/corpus-inventory";

const files = JSON.parse(process.argv[2]!) as Record<string, string>;
const fonts = new Map(
    Object.entries(files).map(([file, path]) => {
        const bytes = readFileSync(path);
        return [file, Font.load(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer)] as const;
    })
);
process.stdout.write(
    JSON.stringify({
        cases: existingCorpus(fonts),
        nonReference: nonReferenceCorpus(),
        pendingInputs: PENDING_INPUTS,
        planIndex: PLAN_INDEX_CASES,
        fixedSourceAccounting: FIXED_SOURCE_ACCOUNTING,
    })
);
