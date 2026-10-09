# Demo: King's Quest I / standalone AGI

> Demo path: `lab/lite/src/demos/kings-quest.ts`

## Purpose and dependencies

Run the original DOS AGI v2 game resources through an independently implemented,
browser-native runtime and Babylon Lite's WebGPU sprite renderer. No interpreter
package, vendored interpreter, emulator, WASM, or added runtime dependency.
TypeScript is compiled to standalone JavaScript by the existing Vite tooling.
The implementation is covered by this repository's Apache-2.0 license; Sierra's
game resources remain separately owned. At the user's explicit direction, the
nine original resource files are included in the demo's `assets/` directory
for eventual Git inclusion after testing/review. The user confirmed permission
for this port. Building locally does not publish anything; public deployment is
a separate action requiring review of that permission's distribution scope.

The original three-treasure winning route and ending are verified. This remains
a compatibility implementation, not a claim of exhaustive original-engine
parity or a maximum-score playthrough. Unsupported resource formats and
instructions must stop execution with a diagnostic, never silently succeed.

## Resource provenance and setup

Target: the DOS AGI v2 edition, not the 1984 PCjr boot disk or the SCI remake.
Verified Archive source:
<https://archive.org/details/msdos_Kings_Quest_I_-_Quest_for_the_Crown_1987>.
File: `Kings_Quest_I_-_Quest_for_the_Crown_1987.zip`, 271,962 bytes;
MD5 `75e461588720e9710bdeee2f91235456`,
SHA-1 `42939ec9c41b762d0d5aa1086f281e81a5b9650c`.
The metadata marks the item restricted; use only the ordinary download endpoint
when it permits access. No authentication workarounds or stream extraction.

All graphics, scripts, vocabulary, inventory and sound come from the supplied
game, not substitute artwork or reconstructed rooms. The bundled game loads
locally; an extracted compatible game directory can also be selected. Imported
files are held in browser memory, not uploaded.
The ZIP was obtained through the ordinary permitted download endpoint and
verified against the recorded size and SHA-1. Only `LOGDIR`, `PICDIR`, `VIEWDIR`,
`SNDDIR`, `WORDS.TOK`, `OBJECT`, `VOL.0`, `VOL.1`, and `VOL.2` are included under
`lab/lite/src/demos/kings-quest/assets/`: 344,614 bytes total (336.5 KiB).
The original executable and interpreter binaries are not included or executed.
`assets/NOTICE.txt` records the separate ownership and provenance.

No fetch script, Archive connection, manifest download, ZIP library or external
runtime is needed to build or play the supplied copy. The existing single-demo
build copies the included assets beside the bundle under `kings-quest/`, like
other demos' included asset subsets. The public-site assembler would include
that folder if explicitly run for deployment. No upload is part of building.
Imported alternative folders are read in memory and never uploaded.

Build only this demo with `corepack pnpm build:bundle-demo kings-quest`.
Launch the preview with
`corepack pnpm --dir lab dev --host 127.0.0.1 --port 5176 --strictPort`.
Open `http://127.0.0.1:5176/lite/demo-kings-quest.html` and click
**Play King's Quest**. Source development is available at
`http://127.0.0.1:5176/lite/demo-kings-quest-source.html`.
Both source and production resolve assets locally; production uses the shared
`demoAssetUrl` convention so the flat site also works under a URL subdirectory.
Extract ZIP files before selection; the runtime does not depend on a ZIP library.

## Specification references

Format and observable-behavior references, not interpreter source:

- <https://wiki.scummvm.org/index.php/AGI/Specifications> (v3.2, 17 July 2007).
- <https://www.agidev.com/articles/agispec/agispecs.html>.
- <https://peterkelly.github.io/agi-re/spec/>.

## Public demo-module surface

- `loadGame(files: ReadonlyMap<string, Uint8Array>): GameResources`: validates
  v2 directories and volumes, decodes vocabulary and inventory, initializes
  resource caches. Names are case-insensitive basenames; duplicates are errors.
- `getResource(game, kind, id): Uint8Array`: directory index (three bytes),
  high-nibble volume number and low-20-bit offset; validates the `12 34`
  signature and five-byte v2 volume header before returning resource payload.
- `decodeLogic(data): Logic`: little-endian code length, bytecode, message
  offset table, XOR-decoded message text with the AGI format key.
- `decodeView(data): View`: loop-relative cel offsets, RLE rows, transparency
  and per-loop mirroring; no mutation of source bytes.
- `drawPicture(data: Uint8Array, picture: Uint8Array, priority: Uint8Array, clear = true): void`: 160x168 palette-indexed
  picture and control/priority planes, vector lines and bounded flood fills.
- `createRuntime(game, host): Runtime`: pure state containing byte variables,
  flags, strings, inventory locations, actors, scan-start offsets and call frames.
- `tickRuntime(runtime, elapsedMs): void`: game clock, bounded LOGIC 0 execution,
  room transitions, animation/motion, controller and parser event lifetimes.
- `submitCommand(runtime, text): void`: dictionary longest-phrase matching,
  ignored words, unknown-word index, `said` wildcard and rest-of-line semantics.
- `sendKey(runtime: Runtime, key: number): void`: raw ASCII or DOS scan-code key
  delivery and original controller mappings.
- `prepareActors(runtime: Runtime): void`: pre-logic autonomous directions,
  target completion flags and player/program direction coupling.
- `updateActors(runtime: Runtime): void`: post-logic movement, screen-edge events,
  cel cycling and priority calculation.
- `renderRuntime(runtime, pixels): void`: background plus priority-composited
  actors into a 320x168 RGBA framebuffer (each AGI pixel is doubled horizontally).
- `decodeSound(data: Uint8Array): Sound`: four timed arrays of notes.
- `noiseNotes(sound: Sound, note: Note): Note[]`: splits tone-clocked noise at
  third-tone frequency changes.
- `createAudio(): AudioState`, `playSound(audio: AudioState, sound: Sound): void`,
  `stopSound(audio: AudioState): void`, `setMuted(audio: AudioState, muted: boolean): void`:
  user-gesture-unlocked browser synthesis.
- `serializeSnapshot(runtime: Runtime): string`,
  `deserializeSnapshot(runtime: Runtime, source: string): void`,
  `saveSnapshot(runtime: Runtime): void`, `restoreSnapshot(runtime: Runtime): void`:
  validated versioned state, associated with a game-data fingerprint.

## Data layout and execution contract

`GameResources` owns case-normalized resource-file bytes, dictionary IDs,
inventory names/initial locations, decoded LOGIC/VIEW caches and a stable
content fingerprint. Resource headers are checked on access rather than
eagerly rejecting unused stale directory entries.

`Runtime` owns 256 byte variables, 256 flags/controllers, 24 strings, 256 actors,
256 16-bit scan starts, four 160x168 byte picture/priority surfaces, inventory
location bytes, parsed IDs/text, key mappings, menus, text runs and resumable
`{logic, pc}` call frames. Host callbacks supply sound, save/restore, notices and
`editCommand(text: string): void` for explicit script-requested editor changes.
No game state is global. A scan has at most 50,000 instructions and 32 call frames.

LOGIC uses signed little-endian displacements for `fe` jumps and `ff` conditional
blocks; `fc` groups OR terms and `fd` negates one predicate. Short-circuited
predicates consume their encoded operands without mutating parser state.
Calls share variables but switch the active message table. Modal operations
suspend the following instruction until acknowledgement, without replaying print.

Actors use integer 160x168 coordinates with baseline Y, selected view/loop/cel,
direction 0..8, automatic/fixed priority and loop, cadence/cycle counters,
terrain/horizon/object/block exemptions and normal/target/follow/wander motion.
Placement uses the last scanned baseline control cell, strict block-rectangle
membership, inclusive object crossings and AGI's left/down/right/up spiral.
Fixed priority 15 bypasses control-plane rejection and clears ego's footprint
classifications, so climbing out of water cannot retain a stale water flag.
Ordinary priorities are
4 above row 48 and twelve-row bands thereafter. Target motion reads all
variable-selected operands, completes inside the strict step-size band and
restores its previous step size. Non-updating objects draw before updating ones.

Full-game validation must start with `createRuntime`, start the title with a key,
and use only parser commands, direction inputs and elapsed game time. Navigation
may plan paths from the decoded control plane, but must never reposition actors,
grant items, change quest flags, warp rooms or edit game resources. Assert the
original three-treasure inventory and scripted royal ending, retaining a route
trace with puzzle commands and actual room transitions.
The focused original-data suite loads all nine unchanged files. Its seed is a
deterministic random-source input, not a quest-state override. The driver logs
every requested direction, parser command, raw key, modal acknowledgement and
elapsed-time advance. A second fresh runtime replays that input tape without
pathfinding and must reach the same original ending. Saved checkpoints are
serialized only after the original save action at legitimately earned positions;
restore uses the original command or mapped F7. Alternative regressions cover the
gnome's wrong-name key, the dagger/dragon solution, and death followed by restoring
an earned checkpoint. Resource SHA-256 assertions prevent puzzle-data edits.

Actor motion uses independent saved baseline coordinates and a stationary result
from the previous refresh. Prepare directions/block crossings before logic;
select automatic loops and advance cel timers before moving. Countdown one is
the pre-logic motion refresh; zero is also due for movement. Target start computes
direction immediately and completion restores its old step size. Completed
ego targets also restore player control, including zero-distance
targets used to attach Graham to the beanstalk. Newly
positioned actors suppress one movement delta. Rejected steps restore position
and perform normal legal-position search. Follow motion retries a stationary
approach using a random nonzero direction and distance-derived delay; wander uses
byte countdown wrap and rejection-sampled intervals. Explicit cel selection
cancels the completing-cycle startup delay.

The original bow-restoration script erases ego, selects loop 1 while its bow
view has only loop 0, then binds the ordinary four-loop view. Loop setup for an
erased object is staged until its next view/cel use; active drawing and every
cel-consuming operation still validate the final binding. View changes reset
out-of-range loop/cel selections to zero, not to the last available index.
Rebinding a view replaces the old view's completing-cycle mode without satisfying
its old completion flag. It preserves motion and cycling eligibility. In the
original condor puzzle, the airborne carrying view must replace the jumping
animation without completing the old jump and cancelling the flight target.
Backward key-poll loops yield a resumable continuation to the browser event loop;
they are not runaway bytecode. Pending raw keys are separate from `v19`, allowing
the script to clear that variable and wait for a genuinely new key.
Mapped keys and menu selections remain queued until a new top-level scan, so
input arriving just after modal dismissal is not lost when the old scan ends.
They never satisfy raw-key polling.

Nonblocking script windows remain visible until closed without suspending input
or logic. Sound replacement, stop, room transitions and pause stop the host sound
and satisfy the previous sound's completion flag. Direction input is a runtime
API shared by the HTML keyboard handler and command-only full-game tests.

## State, input and lifecycle

Interpreter state is owned by the demo. Every resource access is bounded.
Byte arithmetic wraps except saturating increment/decrement. Calls share state
and have an explicit resumable stack so modal dialogs do not repeat commands.
Instruction and call-depth budgets guard malformed or unsupported scripts.
Room changes abort the old call stack, preserve global variables/inventory,
reset room-local actors and motion settings, and restart LOGIC 0 with flag 5.
Unanimation clears participation and transient object-control flags (fixed
priority/loop, horizon/object/block exemptions and water/land gates), retaining
the selected resource binding and coordinates. Room changes additionally reset
cadence defaults. A swimmer's water-only gate must not survive entry into the
underwater chamber, whose background is not classified as surface water.

Arrow keys and the numeric keypad drive the eight AGI directions; keypad 5 stops.
Ordinary keys, including Space and letters, reach `have.key` even while the
HTML parser is enabled; ordinary command entry is not suppressed. `hold.key`
enables release-gated movement, otherwise repeating a direction toggles it.
Registered printable bindings and Tab are consumed before browser editing;
Alt Z uses its original DOS scan word. Ctrl-letter interception is limited to
registered bindings, preserving unbound browser editing and paste shortcuts.
`echo.line` replaces the live edit with the accepted command; `cancel.line`
clears only the live edit, retaining accepted-command history. Ordinary room
transitions do not clear partial input. Commands go
through the original vocabulary/parser; Enter submits, Escape/Enter dismisses
dialogs, function keys map to original `set.key` controllers. Original messages,
menus and string prompts are presented as accessible HTML controls. Modal UI
pauses the game scan, not the wall clock. Audio is unlocked by a user gesture.
Enter acknowledges raw-key waits even when a disabled editor retains text.
The original sound flag `f9`, toolbar and HUD share the same enabled state;
original F2 bindings and the toolbar both update browser audio muting.

Saving uses a versioned browser-local snapshot associated with the imported game
fingerprint. It is not Sierra save-file interchange. No game bytes are persisted
in saves. Version 2 records the updated actor countdowns and screen/editor state.
Version 1 snapshots migrate only explicitly introduced fields: missing raw keys
start empty, screen geometry uses the original defaults, saved actor baselines
start at their saved coordinates, and old elapsed counters become countdowns.
Existing fields remain mandatory and validated. Import errors, storage errors
and interpreter errors are visible.
Pending raw/mapped input identifiers must be nonnegative integers in their
16-bit/8-bit ranges, with at most 19 combined pending events. Invalid saves fail
before replacing the live runtime.
The injected random provider is host-owned and remains unchanged on restore;
functions and the provider's external state are not serialized or rewound.

## Rendering and sound

The CPU decoder creates palette pixels, not a visible Canvas2D game renderer.
A hidden canvas is only the upload staging source. `createDynamicTexture` and
`updateDynamicTexture` upload it using Lite's existing upload Y-flip convention.
A single full-frame `createGridSpriteAtlas` / `Sprite2DLayer` is drawn by
`createSpriteRenderer`, registered on the actual Lite engine. Sampling is
nearest; no raw GPU handles or custom WebGPU pipeline are exposed by demo code.
CSS keeps the presentation approximately 4:3 with the command/status chrome.
Native HTML supplies text/dialogs; the small inventory-object thumbnail is
Canvas2D UI, distinct from the Lite-rendered game framebuffer.

PCjr SOUND resources have four little-endian channel offsets and five-byte note
records. Web Audio square oscillators reproduce the three tone voices from
frequency divisors; a generated noise source supplies the fourth voice.
Durations use the format's 60-Hz units; completion sets the requested AGI flag.
Tone frequency is `111860/divisor`; attenuation is `10^(-attenuation/10)`.
Noise mode 3 follows the third tone's divisor, including mid-note changes.
This is functional synthesis, not a cycle-exact SN76489 emulation.

When WebGPU is missing or no adapter is available, show a durable error suggesting
a WebGPU-capable external Chrome/Edge browser. Do not substitute WebGL or an iframe.

## Test specification and limitations

Focused unit tests use original synthetic binary fixtures, not proprietary data:
resource bounds/signatures; vocabulary compression and synonyms; encrypted
inventory/messages; VIEW mirroring/RLE; PIC vectors/fills; bytecode branches and
calls; parser event consumption; movement/control lines; SOUND note decoding.
Browser checks with a legally available local game must establish boot, visible
Lite rendering, movement, original parser replies and an actual room transition.
Synthetic fixtures do not establish King's Quest compatibility.

AGI v1 boot disks, AGI v3 compression, SCI, original save-file interchange and
cycle-exact graphics/audio hardware behavior are outside this v2 demo's scope.
HTML uses a browser monospace font, not Sierra's original bitmap glyphs.
Configured-screen rows position the browser surfaces; explicit print-window
positions still use the accessible layout instead of exact DOS geometry. Screen shake is reported but
not reproduced. Joystick/monitor/heap diagnostics use browser equivalents.
Interpreter tracing and the priority-screen debugger stop with explicit errors.
Autonomous obstacle recovery and original hardware cadence are not claimed
to be engine-identical. This is a whole-resource interpreter, not a one-room
remake. A complete original three-treasure winning route is verified; exhaustive
coverage of every optional puzzle, random encounter and AGI interpreter profile
is not claimed.

### Observed original-data evidence

The verified ZIP decodes 90 LOGICs, 82 PICs, 118 VIEWs and 22 SOUNDs. Static
bytecode inspection found every action used by the original scripts implemented
except the priority-screen debugger (`0x1d`) in diagnostic LOGIC 99.
Sound directory IDs 34..37 contain offsets beyond VOL.2 in this source revision;
none is referenced by its sound-load/play instructions. Every referenced sound
resource decodes. Invalid resources still fail explicitly if accessed.

Native Edge with isolated `--headless=new --enable-unsafe-webgpu
--ignore-gpu-blocklist` flags established actual Lite device/pipeline/indexed
draw calls and repeated texture uploads; title room 83 to room 1 through Space
and a letter; arrow movement; the original parser reply; room 1 to initialized
room 8; save/restore; folder import; and continued parser gameplay with network
requests blocked after loading. The original title music scheduled native
oscillators in a running AudioContext and produced measured nonzero samples.
A null WebGPU adapter produces the explicit external-browser diagnostic.
Default embedded-browser adapter availability must not be assumed.
The same included resources boot in the source preview and in the flat production
output served under `/preview/`; all nine asset requests stay under that subpath.

The persistent focused original-data route starts a fresh game and earns the
chest by the goat/gnome/beanstalk and sleeping-giant branch, the mirror by the
bucket/well/swim/dive/water-on-dragon branch, and the shield by the
food/fiddle/witch/condor/mushroom/rat/leprechaun branch. It returns through the
shrinking passage to the castle with all three treasures. An earned save at
room 53, `(70,105)`, score 137 is restored after ordinary eastward movement;
the original bow then completes the royal succession and credits at score 140,
room 53, `(30,85)`, crown view 142 and original ending flag 195.
The recorded direction/command/key/acknowledgement/time tape also wins in a
second fresh runtime without the navigation planner. No actor coordinates,
inventory items, quest flags or game resources are injected or edited.
The route is not a maximum-score claim.

An independent native WebGPU production-browser check restored the unmodified
earned pre-king save, confirmed all three treasures through the original
inventory, and completed the original bow and all ending windows at score 140,
with no page exceptions or caught interpreter errors. This is native ending
evidence from an earned checkpoint, not a separate native fresh-start route.
The repeatable alternate coverage includes the gnome's three-wrong-name gold
key, dagger/dragon mirror route at its original lower score, and original dragon
death followed by F7 restoration of an earned save and successful water solution.
All nine resource SHA-256 fingerprints are checked before/after gameplay.

Focused commands:

```powershell
corepack pnpm exec vitest run --project unit tests\lite\unit\kings-quest.test.ts tests\lite\unit\kings-quest-original.test.ts
corepack pnpm typecheck:lab
corepack pnpm exec eslint "lab/lite/src/demos/kings-quest.ts" "lab/lite/src/demos/kings-quest/*.ts" "tests/lite/unit/kings-quest*.ts"
corepack pnpm build:bundle-demo kings-quest
```

No all-scene, parity, performance, golden-reference or ceiling changes are needed.
The gallery thumbnail captures only the authored import UI, not original artwork.

## File manifest

- `lab/lite/demo-kings-quest.html`: production-bundle page.
- `lab/lite/demo-kings-quest-source.html`: source preview.
- `lab/lite/src/demos/kings-quest.ts`: import, input, modal UI, rendering lifecycle.
- `lab/lite/src/demos/kings-quest/resources.ts`: binary resource decoding.
- `lab/lite/src/demos/kings-quest/picture.ts`: picture/control-plane decoding.
- `lab/lite/src/demos/kings-quest/runtime.ts`: AGI state and bytecode execution.
- `lab/lite/src/demos/kings-quest/actors.ts`: motion, animation and compositing.
- `lab/lite/src/demos/kings-quest/sound.ts`: SOUND decoder and browser synthesis.
- `lab/lite/src/demos/kings-quest/persistence.ts`: browser-local snapshots.
- `lab/lite/src/demos/kings-quest/style.ts`: source/bundle-safe inline styles.
- `lab/lite/src/demos/kings-quest/assets/`: nine original resource files and notice.
- `scripts/bundle-demos-core.ts`: filtered demo asset copying.
- `demos-config.json`: gallery/build registration.
- `lab/public/thumbnails/demo-kings-quest.jpg`: authored import-UI thumbnail.
- `tests/lite/unit/kings-quest.test.ts`: synthetic resource/runtime coverage.
- `tests/lite/unit/kings-quest-original.test.ts`: original-data victory, replay, alternate puzzles, death and earned-save coverage.
- `tests/lite/unit/kings-quest-playthrough.ts`: input-only driver, read-only path planning and complete input tape.
- `tests/lite/unit/kings-quest-route.ts`: original-puzzle winning-route actions.
