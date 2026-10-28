/**
 * Shared harness for the bundler build tests (`bundling-*.test.ts`, `treeshake-*.test.ts`).
 *
 * All four tests exercise the SHIPPED module-granular `build/lib` output (the tree a real npm
 * consumer imports via the package's `main`/`module` -> `./lib/index.js`). Keeping every helper
 * here means the vendor-external list, module enumeration, and bundler invocation live in one place
 * and never drift between the Rollup and webpack ports.
 *
 * Why `build/lib` (not `src`): it is exactly what consumers bundle, and it is plain ESM `.js` (one
 * file per source module), so webpack can bundle it with no TypeScript loader and every module —
 * including dynamic-`import()`-only ones like `mesh/csg2.js` — is individually reachable for the
 * "import every module" enumeration.
 */
import { spawnSync } from "child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "fs";
import { randomUUID } from "node:crypto";
import { join, resolve } from "path";
import { rollup, type Plugin as RollupPlugin } from "rollup";
import { minify as terserMinify } from "terser";
import { build as viteBuild } from "vite";

const ROOT = resolve(__dirname, "../../..");
export const PACKAGE_DIR = resolve(ROOT, "packages/babylon-lite");
export const BUILD_LIB_DIR = resolve(PACKAGE_DIR, "build/lib");
export const LIB_ENTRY = resolve(BUILD_LIB_DIR, "index.js");
const PACKAGE_JSON = resolve(PACKAGE_DIR, "package.json");
const VITE_JS = resolve(PACKAGE_DIR, "node_modules/vite/bin/vite.js");

/** A real side effect a bundler must always keep, so "nothing survived" reads as a recognisable
 *  non-empty string rather than "" (which a misconfigured bundler could emit for the wrong reason). */
export const SENTINEL = 'console.log("Babylon Lite bundler harness sentinel");';

/** Bundled third-party runtimes, derived from the package's declared `dependencies` so this list
 *  never drifts from what actually ships. In `build/lib` these live in `_chunks/vendor/*.js` and are
 *  reached only by relative import; consumers get them as isolated chunks. The bundler tests treat
 *  them as external so they validate only OUR first-party module graph (fast, and vendor init is out
 *  of our control). The wasm assets they reference are external for the same reason. */
const DECLARED_DEPENDENCIES = Object.keys((JSON.parse(readFileSync(PACKAGE_JSON, "utf-8")) as { dependencies?: Record<string, string> }).dependencies ?? {});

const NODE_BUILTIN = /^(node:)?(module|fs|path|url|os|crypto|worker_threads|util|stream|events|buffer)$/;

/** True for a request the bundler tests treat as external: a bundled vendor chunk, a
 *  Vite-generated inline-worker wrapper, a wasm/`?url` asset, a bare vendor specifier,
 *  or a Node builtin. Worker wrappers intentionally allocate a Blob at module scope and
 *  are framework output rather than a hand-written module-side-effect contract. */
export function isExternalRequest(request: string): boolean {
    if (!request) {
        return false;
    }
    if (/[\\/]_chunks[\\/]vendor[\\/]/.test(request)) {
        return true;
    }
    if (/[\\/]_chunks[\\/][^/\\]*-worker-[^/\\]+\.js$/.test(request)) {
        return true;
    }
    if (/\.wasm(\?|$)/.test(request) || /\?url(&|$)/.test(request) || /\?worker(&|$)/.test(request)) {
        return true;
    }
    if (NODE_BUILTIN.test(request)) {
        return true;
    }
    return DECLARED_DEPENDENCIES.some((dep) => request === dep || request.startsWith(`${dep}/`));
}

/** Warnings that are inherent to bundling a large library and say nothing about our code health.
 *  Everything else must be zero for the bundling tests to pass. */
export const BENIGN_WARNING_RE = /asset size limit|entrypoint size limit|performance recommendations|exceed the recommended size limit|bigger than 244 KiB|Circular dependenc/i;

let _libBuilt = false;

/** Newest mtime (ms) among the package's TypeScript sources — used to decide whether the existing
 *  `build/lib` output is already up to date. */
function newestSrcMtime(): number {
    let newest = 0;
    const walk = (dir: string): void => {
        for (const e of readdirSync(dir, { withFileTypes: true })) {
            const p = join(dir, e.name);
            if (e.isDirectory()) {
                walk(p);
            } else if (/\.ts$/.test(e.name)) {
                const m = statSync(p).mtimeMs;
                if (m > newest) {
                    newest = m;
                }
            }
        }
    };
    walk(resolve(PACKAGE_DIR, "src"));
    return newest;
}

/** Build the package's `build/lib` output once per test process, and only when it is missing or
 *  stale relative to `src`. Skipping a fresh build both saves time and avoids clobbering `build/lib`
 *  while a sibling build-test file is reading it (the build-test project runs files sequentially —
 *  see `vitest.config.ts` — so this check means only the first file actually rebuilds). Mirrors
 *  `public-api-types.test.ts`: invoked via the current node executable + vite's JS entry so it does
 *  not depend on PATH. */
export function ensureLibBuilt(): void {
    if (_libBuilt) {
        return;
    }
    const fresh = existsSync(LIB_ENTRY) && statSync(LIB_ENTRY).mtimeMs >= newestSrcMtime();
    if (!fresh) {
        const built = spawnSync(process.execPath, [VITE_JS, "build", "--mode", "lib"], {
            cwd: PACKAGE_DIR,
            encoding: "utf-8",
        });
        if (built.status !== 0) {
            throw new Error(`babylon-lite build (--mode lib) failed:\n${built.stdout ?? ""}${built.stderr ?? ""}`);
        }
    }
    _libBuilt = true;
}

/** Every first-party `build/lib` module a consumer could reach (statically OR via dynamic
 *  `import()`). Excludes the vendor chunks, sourcemaps, and `*-worker.js` entries (which carry a
 *  top-level `self.onmessage = …` by design and are only ever loaded via `?worker`). */
export function enumerateLibModules(): string[] {
    const out: string[] = [];
    const walk = (dir: string): void => {
        for (const e of readdirSync(dir, { withFileTypes: true })) {
            const p = join(dir, e.name);
            if (e.isDirectory()) {
                if (e.name === "_chunks") {
                    continue;
                }
                walk(p);
            } else if (/\.js$/.test(e.name) && !/-worker\.js$/.test(e.name)) {
                out.push(p);
            }
        }
    };
    walk(BUILD_LIB_DIR);
    return out;
}

// ─── Temp entry management ─────────────────────────────────────────────────

const _tempDirs: string[] = [];

/** Create a temp dir with an entry file containing `source`; returns the entry path. */
export function makeTempEntry(source: string): string {
    const dir = join(ROOT, "node_modules/.cache", `lite-bundler-${randomUUID()}`);
    mkdirSync(dir, { recursive: true });
    _tempDirs.push(dir);
    const entry = join(dir, "entry.mjs");
    writeFileSync(entry, source);
    return entry;
}

/** Remove all temp dirs created during the run. Call from `afterAll`. */
export function cleanupTempDirs(): void {
    for (const dir of _tempDirs.splice(0)) {
        rmSync(dir, { recursive: true, force: true });
    }
}

/** An entry that bare-imports the built package barrel plus the sentinel. */
export const barrelEntrySource = (): string => `import ${JSON.stringify(LIB_ENTRY)};\n${SENTINEL}\n`;

/** An entry that statically imports EVERY first-party module (forces dynamic-only modules such as
 *  `mesh/csg2.js` into the parsed graph) plus the sentinel. */
export const everyModuleEntrySource = (): string =>
    enumerateLibModules()
        .map((m) => `import ${JSON.stringify(m)};`)
        .join("\n") + `\n${SENTINEL}\n`;

// ─── Bundler results ───────────────────────────────────────────────────────

export interface BundleResult {
    errors: string[];
    /** All warnings, verbatim (first line each). */
    warnings: string[];
    /** Warnings after removing `BENIGN_WARNING_RE` (size/perf noise). */
    significantWarnings: string[];
    /** Emitted bundle code (concatenated across chunks), for tree-shaking content assertions. */
    code: string;
    /** Rollup/Vite chunks, when the selected harness exposes them. */
    chunks?: { code: string; isEntry: boolean }[];
    /** Modules with emitted runtime code, not merely parsed inputs. */
    modules?: { id: string; renderedLength: number; renderedExports: string[] }[];
    /** External imports that actually survive emission. */
    imports?: string[];
}

function classify(warnings: string[]): string[] {
    return warnings.filter((w) => !BENIGN_WARNING_RE.test(w));
}

/** Rolldown annotates bundled modules with source-region comments. Temp entry paths
 * differ between otherwise identical builds, so normalize only that generated path. */
function normalizeGeneratedRegions(code: string): string {
    return code.replace(/^\/\/#region .*[/\\]lite-bundler-[^/\\]+[/\\]entry\.mjs$/gm, "//#region <bundler-entry>");
}

// ─── Webpack ───────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-require-imports
const webpack = require("webpack") as typeof import("webpack").webpack;

export interface WebpackOpts {
    entrySource: string;
    /** "production" (Terser + full tree-shaking) or "development". */
    mode: "production" | "development";
    /** When true, disable webpack tree-shaking (usedExports/sideEffects/providedExports) to
     *  reproduce pipelines that defer optimization to a later stage (e.g. Google Closure Compiler,
     *  as Word Web does). This forces every barrel-reachable module to be parsed. */
    disableTreeShaking: boolean;
}

/** Run a webpack build in memory and return errors + warnings + emitted code. */
export function runWebpack(opts: WebpackOpts): Promise<BundleResult> {
    const entry = makeTempEntry(opts.entrySource);
    const optimization = opts.disableTreeShaking
        ? { usedExports: false, sideEffects: false, providedExports: false, minimize: false, concatenateModules: false }
        : { minimize: opts.mode === "production" };
    return new Promise((res, rej) => {
        const compiler = webpack({
            mode: opts.mode,
            entry,
            output: { path: join(entry, "..", "dist"), filename: "out.js" },
            target: ["web", "es2020"],
            performance: { hints: false },
            optimization,
            externals: [
                ({ request }: { request?: string }, cb: (err?: Error | null, result?: string) => void) => (request && isExternalRequest(request) ? cb(null, `var {}`) : cb()),
            ],
            stats: "errors-warnings",
        });
        compiler.run((err, stats) => {
            if (err) {
                compiler.close(() => rej(err));
                return;
            }
            const info = stats!.toJson({ all: false, warnings: true, errors: true, assets: true });
            const warnings = (info.warnings ?? []).map((w) => (typeof w === "string" ? w : w.message));
            const errors = (info.errors ?? []).map((e) => (typeof e === "string" ? e : e.message));
            // Concatenate emitted assets for content assertions.
            const outDir = join(entry, "..", "dist");
            let code = "";
            try {
                for (const f of readdirSync(outDir)) {
                    if (f.endsWith(".js")) {
                        code += readFileSync(join(outDir, f), "utf-8") + "\n";
                    }
                }
            } catch {
                // No output emitted (hard error build); leave code empty.
            }
            compiler.close(() => res({ errors, warnings, significantWarnings: classify(warnings), code }));
        });
    });
}

// ─── Rollup ────────────────────────────────────────────────────────────────

export interface RollupOpts {
    entrySource: string;
    format: "es" | "cjs";
    minify: boolean;
    /** Force first-party side-effect retention (ignore `sideEffects:false`) so a genuine module-init
     *  side effect survives tree-shaking instead of the whole graph being dropped on trust. */
    forceModuleSideEffects?: boolean;
}

/** Run a real downstream Rollup build and return errors + warnings + emitted code.
 * Vite 8 uses Rolldown internally, so calling Vite here would no longer test the
 * Rollup consumer compatibility this project promises. */
export async function runRollup(opts: RollupOpts): Promise<BundleResult> {
    const entry = makeTempEntry(opts.entrySource);
    const warnings: string[] = [];
    const errors: string[] = [];
    let code = "";
    const chunks: { code: string; isEntry: boolean }[] = [];
    const modules: NonNullable<BundleResult["modules"]> = [];
    const imports: string[] = [];
    let bundle: Awaited<ReturnType<typeof rollup>> | null = null;
    try {
        const plugins: RollupPlugin[] = opts.minify
            ? [
                  {
                      name: "test-terser-minify",
                      async renderChunk(chunkCode) {
                          const result = await terserMinify(chunkCode, { module: opts.format === "es" });
                          return result.code ? { code: result.code, map: null } : null;
                      },
                  },
              ]
            : [];
        bundle = await rollup({
            input: entry,
            external: (id: string) => isExternalRequest(id),
            treeshake: {
                // Vendors are externalized for first-party measurements, not imported for side effects.
                moduleSideEffects: (_id: string, external: boolean) => !external,
                ...(opts.forceModuleSideEffects ? { propertyReadSideEffects: false } : {}),
            },
            onwarn: (warning) => warnings.push(`[${warning.code}] ${warning.message}`),
            plugins,
        });
        const { output } = await bundle.generate({
            format: opts.format,
            entryFileNames: "entry.mjs",
            chunkFileNames: "[name]-[hash].mjs",
        });
        for (const chunk of output) {
            if (chunk.type === "chunk" && chunk.code) {
                const normalizedCode = normalizeGeneratedRegions(chunk.code);
                code += normalizedCode + "\n";
                chunks.push({ code: normalizedCode, isEntry: chunk.isEntry === true });
                modules.push(
                    ...Object.entries(chunk.modules)
                        .filter(([, module]) => module.renderedLength > 0)
                        .map(([id, module]) => ({ id, renderedLength: module.renderedLength, renderedExports: module.renderedExports }))
                );
                imports.push(...chunk.imports);
            }
        }
    } catch (e) {
        errors.push(e instanceof Error ? e.message : String(e));
    } finally {
        await bundle?.close();
    }
    return { errors, warnings, significantWarnings: classify(warnings), code, chunks, modules, imports };
}

/** Bundle the shipped root with Vite's actual package resolver and vendor code included.
 *  Complements first-party Rollup measurements: no external-vendor side-effect assumptions. */
export async function runVite(entrySource: string, reachableOnly = false): Promise<BundleResult> {
    const entry = makeTempEntry(entrySource);
    const result: BundleResult = { errors: [], warnings: [], significantWarnings: [], code: "", chunks: [], modules: [], imports: [] };
    try {
        const built = await viteBuild({
            configFile: false,
            publicDir: false,
            logLevel: "silent",
            build: {
                write: false,
                minify: false,
                sourcemap: false,
                modulePreload: false,
                rollupOptions: { input: entry },
            },
        });
        for (const output of Array.isArray(built) ? built : [built]) {
            if (!("output" in output)) {
                throw new Error("Vite consumer build unexpectedly returned a watcher");
            }
            // Rolldown can emit orphan dynamic chunks after eliminating their
            // import expressions. An opt-in consumer graph follows every static
            // AND dynamic edge from every entry; it does not drop lazy features
            // that the consumer can actually request.
            const chunks = output.output.filter((chunk) => chunk.type === "chunk");
            const byName = new Map(chunks.map((chunk) => [chunk.fileName, chunk]));
            const reachable = new Set<string>();
            const queue = chunks.filter((chunk) => chunk.isEntry).map((chunk) => chunk.fileName);
            while (queue.length) {
                const name = queue.pop()!;
                if (reachable.has(name)) {
                    continue;
                }
                reachable.add(name);
                const chunk = byName.get(name);
                if (chunk) {
                    queue.push(...chunk.imports, ...chunk.dynamicImports);
                }
            }
            for (const chunk of output.output) {
                if (chunk.type !== "chunk") {
                    continue;
                }
                if (reachableOnly && !reachable.has(chunk.fileName)) {
                    continue;
                }
                result.code += chunk.code + "\n";
                result.chunks!.push({ code: chunk.code, isEntry: chunk.isEntry });
                result.imports!.push(...chunk.imports);
                result.modules!.push(
                    ...Object.entries(chunk.modules)
                        .filter(([, module]) => module.renderedLength > 0)
                        .map(([id, module]) => ({ id, renderedLength: module.renderedLength, renderedExports: module.renderedExports }))
                );
            }
        }
    } catch (error) {
        result.errors.push(error instanceof Error ? error.message : String(error));
    }
    return result;
}

/** The recognisable sentinel content, for asserting "nothing but the sentinel survived" regardless
 *  of each bundler's runtime wrapper (webpack wraps in an IIFE + `__webpack_require__` scaffold;
 *  Rollup emits the bare statements). We assert on the sentinel's marker string plus the absence of
 *  any first-party package identifier, rather than exact-string equality. */
export const SENTINEL_MARKER = "Babylon Lite bundler harness sentinel";

/** First-party identifiers that must NOT appear in a fully tree-shaken bundle of a side-effect-only
 *  barrel import. If any survives, module-level code ran on import (a side effect) or `sideEffects`
 *  boundaries are wrong. Spans several subsystems so a leak anywhere is caught. */
const SURVIVING_PACKAGE_IDENTIFIERS: readonly (readonly [string, RegExp])[] = [
    ["engine", /createEngine|createSurface/],
    ["text", /createTextRenderer|createTextLayer|TextRenderer/],
    ["sprite", /createSpriteRenderer|createSprite2DLayer/],
    ["csg2/manifold", /reserveIDs|manifoldModule|CSG2 is not initialized/],
    ["havok", /createHavokWorld|HP_World_Create/],
    ["material", /createPbrMaterial|createStandardMaterial/],
];

/** Names of any first-party subsystems whose identifiers survived in `code` (should be empty for a
 *  side-effect-only import of a properly tree-shaking package). */
export function survivingPackageSubsystems(code: string): string[] {
    return SURVIVING_PACKAGE_IDENTIFIERS.filter(([, re]) => re.test(code)).map(([name]) => name);
}

/** True when a bundle contains the sentinel and no surviving first-party package code — i.e. the
 *  bundler tree-shook the whole side-effect-free graph away. */
export function isSentinelOnly(code: string): boolean {
    return code.includes(SENTINEL_MARKER) && survivingPackageSubsystems(code).length === 0;
}
