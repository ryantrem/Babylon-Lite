/** Serialization bridge only: builds fonts, never shapes or imports a Lite plan/kernel. */
import { syntheticFontBytes, CORRECTION_KINDS, LEGACY_SYNTHETIC_KINDS } from "../tests/lite/compute/synthetic";

process.stdout.write(
    JSON.stringify(
        [...CORRECTION_KINDS, ...LEGACY_SYNTHETIC_KINDS].map((kind) => ({
            kind,
            legacy: LEGACY_SYNTHETIC_KINDS.includes(kind),
            bytes: Buffer.from(syntheticFontBytes(kind)).toString("base64"),
        }))
    )
);
