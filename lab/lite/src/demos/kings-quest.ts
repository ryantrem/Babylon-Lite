import {
    addSprite2DIndex,
    createDynamicTexture,
    createEngine,
    createGridSpriteAtlas,
    createSprite2DLayer,
    createSpriteRenderer,
    disposeEngine,
    disposeSpriteRenderer,
    registerSpriteRenderer,
    startEngine,
    updateDynamicTexture,
    updateSprite2DIndex,
} from "babylon-lite";
import { renderRuntime } from "./kings-quest/actors.js";
import { getView, loadGame } from "./kings-quest/resources.js";
import { createRuntime, dismissModal, sendDirection, sendKey, submitCommand, tickRuntime } from "./kings-quest/runtime.js";
import type { Modal, Runtime } from "./kings-quest/runtime.js";
import { createAudio, playSound, setMuted, stopSound } from "./kings-quest/sound.js";
import type { AudioState } from "./kings-quest/sound.js";
import { restoreSnapshot, saveSnapshot } from "./kings-quest/persistence.js";
import { styles } from "./kings-quest/style.js";
import { demoAssetUrl } from "./demo-asset-url.js";

function element<T extends HTMLElement>(id: string, constructor: { new (): T }): T {
    const value = document.getElementById(id);
    if (!(value instanceof constructor)) {
        throw new Error(`Missing demo element ${id}.`);
    }
    return value;
}

async function main(): Promise<void> {
    const style = document.createElement("style");
    style.textContent = styles;
    document.head.append(style);
    document.body.innerHTML = `
<main>
  <header><h1>King's Quest I</h1><span class="badge">Standalone AGI JavaScript + Babylon Lite WebGPU</span></header>
  <div class="toolbar">
    <button id="local" disabled>Play King's Quest</button>
    <button id="import" disabled>Choose game folder</button>
    <input id="files" type="file" webkitdirectory multiple hidden aria-label="Select extracted AGI game directory" />
    <button id="save" disabled>Save</button><button id="restore" disabled>Restore</button>
    <button id="restart" disabled>Restart</button><button id="sound" disabled>Sound on</button>
  </div>
  <div class="screen" id="screen">
    <canvas id="renderCanvas" tabindex="0" aria-label="Original King's Quest graphics, rendered by Babylon Lite"></canvas>
    <div id="score" hidden></div><div id="gameText"></div><div id="scriptWindow" role="status" aria-live="polite" hidden></div>
    <form class="command" id="commandForm"><label for="command">&gt;</label><input id="command" autocomplete="off" spellcheck="false" maxlength="40" disabled aria-label="Game command" /><button id="submit" disabled>Enter</button></form>
  </div>
  <div id="status" role="status" aria-live="polite">Starting WebGPU...</div>
  <div class="help">
    <strong>Original DOS AGI v2 assets included.</strong> Graphics, game logic and sound are loaded from this demo; no external interpreter or game service.
    Arrow keys / numeric keypad: move (press the same direction again to stop). Enter: submit commands or close a message. Escape: game menu.
    Original function keys work; F5/F7 are commonly save/restore. Save files stay in this browser.
    <br />Use Chrome or Edge with working WebGPU; the embedded app browser may have no adapter.
    Three-treasure winning route and original ending verified. Exhaustive AGI compatibility, original bitmap-font and hardware-perfect audio parity are not claimed.
    <br /><a href="https://archive.org/details/msdos_Kings_Quest_I_-_Quest_for_the_Crown_1987" target="_blank" rel="noopener noreferrer">Archive source: 1987 DOS edition</a>.
    The runtime has no third-party interpreter, emulator, WASM or added runtime dependencies.
  </div>
  <dialog id="modal" aria-label="Game dialog"><div id="modalBody"></div></dialog>
</main>`;
    const canvas = element("renderCanvas", HTMLCanvasElement);
    const screen = element("screen", HTMLDivElement);
    const status = element("status", HTMLDivElement);
    const textLayer = element("gameText", HTMLDivElement);
    const score = element("score", HTMLDivElement);
    const scriptWindow = element("scriptWindow", HTMLDivElement);
    const commandForm = element("commandForm", HTMLFormElement);
    const input = element("command", HTMLInputElement);
    const fileInput = element("files", HTMLInputElement);
    const dialog = element("modal", HTMLDialogElement);
    const modalBody = element("modalBody", HTMLDivElement);
    const localButton = element("local", HTMLButtonElement);
    const importButton = element("import", HTMLButtonElement);
    const saveButton = element("save", HTMLButtonElement);
    const restoreButton = element("restore", HTMLButtonElement);
    const restartButton = element("restart", HTMLButtonElement);
    const soundButton = element("sound", HTMLButtonElement);
    const submitButton = element("submit", HTMLButtonElement);
    let runtime: Runtime | null = null;
    let audio: AudioState | null = null;
    let shownModal: Modal | null = null;
    let textSignature = "";
    let frameId = 0;
    let lastTime = performance.now();
    let loading = false;
    function report(text: string, error = false): void {
        status.textContent = text;
        status.dataset.error = String(error);
    }
    function failure(error: unknown): void {
        const text = error instanceof Error ? error.message : String(error);
        report(text, true);
        canvas.dataset.error = text;
        if (runtime) {
            runtime.stopped = true;
        }
        if (audio) {
            stopSound(audio);
        }
        input.disabled = submitButton.disabled = true;
        console.error(error);
    }
    function unlockAudio(): void {
        audio ??= createAudio();
        void audio.context.resume().catch(failure);
    }
    function finishModal(value = ""): void {
        if (runtime) {
            dismissModal(runtime, value);
        }
        if (dialog.open) {
            dialog.close();
        }
        shownModal = null;
        input.focus();
    }
    function button(label: string, value: string): HTMLButtonElement {
        const result = document.createElement("button");
        result.type = "button";
        result.textContent = label;
        result.onclick = () => finishModal(value);
        return result;
    }
    function updateModal(r: Runtime): void {
        if (r.modal === shownModal) {
            return;
        }
        if (!r.modal) {
            if (dialog.open) {
                dialog.close();
            }
            shownModal = null;
            return;
        }
        shownModal = r.modal;
        modalBody.replaceChildren();
        const message = document.createElement("p");
        message.textContent = r.modal.text;
        modalBody.append(message);
        if (r.modal.kind === "string" || r.modal.kind === "number") {
            const form = document.createElement("form");
            const editor = document.createElement("input");
            editor.setAttribute("aria-label", r.modal.text || "Game response");
            editor.maxLength = r.modal.maxLength ?? 39;
            if (r.modal.kind === "number") {
                editor.inputMode = "numeric";
                editor.pattern = "[0-9]{0,4}";
            }
            form.append(editor);
            const accept = document.createElement("button");
            accept.textContent = "Enter";
            form.append(accept);
            form.onsubmit = (event) => {
                event.preventDefault();
                if (form.reportValidity()) {
                    finishModal(editor.value);
                }
            };
            modalBody.append(form);
        } else if (r.modal.kind === "menu") {
            for (const menu of r.menus) {
                const heading = document.createElement("h2");
                heading.textContent = menu.title;
                modalBody.append(heading);
                for (const item of menu.items) {
                    const member = button(item.text, String(item.controller));
                    member.disabled = !item.enabled;
                    modalBody.append(member);
                }
            }
            modalBody.append(button("Cancel", ""));
        } else if (r.modal.kind === "inventory") {
            let count = 0;
            r.game.inventory.forEach((item, index) => {
                if (r.inventory[index] === 255) {
                    count++;
                    if (r.flags[13]) {
                        modalBody.append(button(item.name, String(index)));
                    } else {
                        const name = document.createElement("p");
                        name.textContent = item.name;
                        modalBody.append(name);
                    }
                }
            });
            if (!count) {
                const empty = document.createElement("p");
                empty.textContent = "You are carrying nothing.";
                modalBody.append(empty);
            }
            modalBody.append(button("Close", ""));
        } else if (r.modal.kind === "quit") {
            modalBody.append(button("Quit", "quit"), button("Keep playing", ""));
        } else {
            if (r.modal.view !== undefined) {
                const cel = getView(r.game, r.modal.view).loops[0]![0]!;
                const preview = document.createElement("canvas");
                preview.className = "preview";
                preview.width = cel.width * 2;
                preview.height = cel.height;
                const context = preview.getContext("2d");
                if (!context) {
                    throw new Error("Cannot create inventory preview staging context.");
                }
                const pixels = context.createImageData(preview.width, preview.height);
                for (let i = 0; i < cel.pixels.length; i++) {
                    const c = cel.pixels[i]!;
                    const bright = c & 8 ? 85 : 0;
                    for (let px = 0; px < 2; px++) {
                        const offset = i * 8 + px * 4;
                        pixels.data[offset] = c === 6 ? 170 : (c & 4 ? 170 : 0) + bright;
                        pixels.data[offset + 1] = c === 6 ? 85 : (c & 2 ? 170 : 0) + bright;
                        pixels.data[offset + 2] = (c & 1 ? 170 : 0) + bright;
                        pixels.data[offset + 3] = c === cel.transparent ? 0 : 255;
                    }
                }
                context.putImageData(pixels, 0, 0);
                modalBody.append(preview);
            }
            modalBody.append(button("Enter", ""));
        }
        if (!dialog.open) {
            dialog.showModal();
        }
        modalBody.querySelector("input")?.focus();
    }
    dialog.addEventListener("cancel", (event) => {
        event.preventDefault();
        finishModal();
    });
    dialog.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && runtime?.modal?.kind === "message") {
            event.preventDefault();
            event.stopPropagation();
            finishModal();
        }
    });
    if (!navigator.gpu) {
        throw new Error("WebGPU is unavailable. Open this demo in an external Chrome/Edge browser with WebGPU and hardware acceleration enabled.");
    }
    const engine = await createEngine(canvas).catch((error: unknown) => {
        throw new Error(
            `No usable WebGPU adapter/device. Open this URL in an external WebGPU-capable Chrome/Edge browser. ${error instanceof Error ? error.message : String(error)}`
        );
    });
    const staging = document.createElement("canvas");
    staging.width = 320;
    staging.height = 168;
    const context = staging.getContext("2d");
    if (!context) {
        disposeEngine(engine);
        throw new Error("Cannot create the AGI framebuffer upload staging context.");
    }
    const image = context.createImageData(320, 168);
    const texture = createDynamicTexture(engine, 320, 168, { minFilter: "nearest", magFilter: "nearest" });
    const atlas = createGridSpriteAtlas(texture, { cellWidthPx: 320, cellHeightPx: 168, pivot: [0, 0] });
    const layer = createSprite2DLayer(atlas, { capacity: 1, pivot: [0, 0] });
    const sprite = addSprite2DIndex(layer, { frame: 0, positionPx: [0, 0], sizePx: [320, 168], flipY: true });
    const renderer = createSpriteRenderer(engine, { layers: [layer] });
    registerSpriteRenderer(renderer);
    function resize(): void {
        const scale = window.devicePixelRatio || 1;
        updateSprite2DIndex(layer, sprite, { sizePx: [canvas.clientWidth * scale, canvas.clientHeight * scale] });
        screen.style.setProperty("--cell", `${screen.clientWidth / 40}px`);
    }
    const observer = new ResizeObserver(resize);
    observer.observe(screen);
    resize();
    await startEngine(engine);
    canvas.dataset.ready = "true";
    canvas.dataset.renderer = "babylon-lite-webgpu";
    canvas.dataset.state = "awaiting-import";
    report("Play King's Quest with the included original graphics and sound, or choose an alternative extracted AGI v2 game folder.");
    importButton.disabled = localButton.disabled = soundButton.disabled = false;

    function boot(files: ReadonlyMap<string, Uint8Array>): void {
        const game = loadGame(files);
        if (audio) {
            stopSound(audio);
        }
        if (dialog.open) {
            dialog.close();
        }
        shownModal = null;
        textSignature = "";
        input.value = "";
        delete canvas.dataset.error;
        runtime = createRuntime(game, {
            sound: (sound, enabled) => {
                if (audio && enabled) {
                    playSound(audio, sound);
                } else if (audio) {
                    stopSound(audio);
                }
            },
            stopSound: () => {
                if (audio) {
                    stopSound(audio);
                }
            },
            save: (r) => {
                try {
                    saveSnapshot(r);
                    report("Game saved in this browser (Lite save format).");
                } catch (error) {
                    report(`Save failed: ${error instanceof Error ? error.message : String(error)}`, true);
                }
            },
            restore: (r) => {
                try {
                    restoreSnapshot(r);
                    report("Saved game restored.");
                } catch (error) {
                    report(`Restore failed: ${error instanceof Error ? error.message : String(error)}`, true);
                }
            },
            notify: (text) => report(text),
            editCommand: (text) => {
                input.value = text;
                input.focus();
                input.setSelectionRange(text.length, text.length);
            },
        });
        runtime.flags[9] = Number(!audio?.muted);
        saveButton.disabled = restoreButton.disabled = restartButton.disabled = false;
        canvas.dataset.state = "running";
        report(`Local game loaded: ${game.files.size} resource files, including graphics and audio. Press any ordinary key on the title screen.`);
        input.focus();
    }
    importButton.onclick = () => {
        unlockAudio();
        fileInput.click();
    };
    fileInput.onchange = () => {
        if (!fileInput.files || loading) {
            return;
        }
        const selected = [...fileInput.files];
        loading = true;
        report("Reading local game resources...");
        void Promise.all(
            selected
                .filter((file) => /^(logdir|picdir|viewdir|snddir|words\.tok|object|vol\.\d+)$/i.test(file.name))
                .map(async (file) => {
                    if (file.size > 4 * 1024 * 1024) {
                        throw new Error(`Unexpectedly large AGI resource file ${file.name}.`);
                    }
                    return [file.webkitRelativePath || file.name, new Uint8Array(await file.arrayBuffer())] as const;
                })
        )
            .then((files) => boot(new Map(files)))
            .catch(failure)
            .finally(() => {
                loading = false;
                fileInput.value = "";
            });
    };
    localButton.onclick = () => {
        if (loading) {
            return;
        }
        unlockAudio();
        loading = true;
        report("Loading original graphics, audio and game logic...");
        void (async () => {
            const base = demoAssetUrl(import.meta.env.DEV ? "./kings-quest/assets/" : "./kings-quest/", import.meta.url);
            const names = ["LOGDIR", "PICDIR", "VIEWDIR", "SNDDIR", "WORDS.TOK", "OBJECT", "VOL.0", "VOL.1", "VOL.2"];
            const files = await Promise.all(
                names.map(async (name) => {
                    const resource = await fetch(new URL(encodeURIComponent(name), base));
                    if (!resource.ok) {
                        throw new Error(`Local resource ${name} could not be read (${resource.status}).`);
                    }
                    if (resource.headers.get("content-type")?.includes("text/html")) {
                        throw new Error(`Game asset ${name} returned a web page instead of binary data. Rebuild this demo or choose your extracted game folder.`);
                    }
                    return [name, new Uint8Array(await resource.arrayBuffer())] as const;
                })
            );
            boot(new Map(files));
        })()
            .catch(failure)
            .finally(() => {
                loading = false;
            });
    };
    saveButton.onclick = () => {
        if (runtime) {
            runtime.host.save(runtime);
        }
    };
    restoreButton.onclick = () => {
        if (runtime) {
            runtime.host.restore(runtime);
        }
    };
    restartButton.onclick = () => {
        if (runtime) {
            boot(runtime.game.files);
        }
    };
    soundButton.onclick = () => {
        unlockAudio();
        if (audio) {
            const muted = runtime ? Boolean(runtime.flags[9]) : !audio.muted;
            setMuted(audio, muted);
            if (runtime) {
                runtime.flags[9] = Number(!muted);
            }
            soundButton.textContent = audio.muted ? "Sound off" : "Sound on";
        }
    };
    commandForm.onsubmit = (event) => {
        event.preventDefault();
        unlockAudio();
        if (runtime && input.value.trim()) {
            submitCommand(runtime, input.value.slice(0, Math.max(0, runtime.variables[24]! - 1)));
            input.value = "";
        }
    };
    const directions: Readonly<Record<string, number>> = {
        ArrowUp: 1,
        Numpad8: 1,
        Numpad9: 2,
        PageUp: 2,
        ArrowRight: 3,
        Numpad6: 3,
        Numpad3: 4,
        PageDown: 4,
        ArrowDown: 5,
        Numpad2: 5,
        Numpad1: 6,
        End: 6,
        ArrowLeft: 7,
        Numpad4: 7,
        Numpad7: 8,
        Home: 8,
        Numpad5: 0,
    };
    const scanCodes: Readonly<Record<string, number>> = {
        F1: 0x3b,
        F2: 0x3c,
        F3: 0x3d,
        F4: 0x3e,
        F5: 0x3f,
        F6: 0x40,
        F7: 0x41,
        F8: 0x42,
        F9: 0x43,
        F10: 0x44,
    };
    document.addEventListener("keydown", (event) => {
        if (!runtime || dialog.open || loading || event.repeat || event.metaKey) {
            return;
        }
        if (event.ctrlKey || event.altKey) {
            const key = event.ctrlKey && /^[a-z]$/i.test(event.key) ? event.key.toUpperCase().charCodeAt(0) - 64 : event.altKey && event.code === "KeyZ" ? 0x2c00 : 0;
            if (key && runtime.keyMap.has(key)) {
                event.preventDefault();
                sendKey(runtime, key);
                unlockAudio();
            }
            return;
        }
        const rawKey =
            event.key.length === 1
                ? event.key.charCodeAt(0)
                : event.key === "Tab"
                  ? 9
                  : event.key === "Escape"
                    ? 27
                    : event.key === "Enter"
                      ? 13
                      : (scanCodes[event.code] ?? 0) << 8;
        if (rawKey && runtime.keyMap.has(rawKey)) {
            event.preventDefault();
            sendKey(runtime, rawKey);
            unlockAudio();
            return;
        }
        if (event.key === "Enter" && (!runtime.inputEnabled || !runtime.graphics || (runtime.keyPoll && runtime.stack.length))) {
            event.preventDefault();
            sendKey(runtime, 13);
            unlockAudio();
            return;
        }
        const direction = directions[event.code];
        if (direction !== undefined && runtime.playerControl) {
            event.preventDefault();
            sendDirection(runtime, direction);
            unlockAudio();
        } else if (scanCodes[event.code] !== undefined) {
            event.preventDefault();
            sendKey(runtime, scanCodes[event.code]! << 8);
            unlockAudio();
        } else if (event.key === "Escape") {
            event.preventDefault();
            sendKey(runtime, 27);
        } else if (event.key === "Enter" && !input.value.trim()) {
            event.preventDefault();
            sendKey(runtime, 13);
            unlockAudio();
        } else if (event.key.length === 1) {
            sendKey(runtime, event.key.charCodeAt(0));
            unlockAudio();
            if (!runtime.inputEnabled || runtime.variables[0] === 83 || (runtime.keyPoll && runtime.stack.length)) {
                event.preventDefault();
            } else if (event.target !== input) {
                event.preventDefault();
                input.focus();
                input.setRangeText(event.key, input.selectionStart ?? input.value.length, input.selectionEnd ?? input.value.length, "end");
            }
        }
    });
    document.addEventListener("keyup", (event) => {
        if (runtime && directions[event.code] !== undefined) {
            sendDirection(runtime, directions[event.code]!, false);
        }
    });
    window.addEventListener("blur", () => {
        if (runtime?.playerControl) {
            sendDirection(runtime, 0);
        }
    });
    function updateText(r: Runtime): void {
        const signature = JSON.stringify(r.text);
        if (signature !== textSignature) {
            textSignature = signature;
            textLayer.replaceChildren();
            for (const entry of r.text) {
                const line = document.createElement("div");
                line.className = "text-line";
                line.style.top = `${entry.row * 4}%`;
                line.style.left = `${entry.column * 2.5}%`;
                line.style.width = `${(40 - entry.column) * 2.5}%`;
                line.style.color = ["#000", "#00a", "#0a0", "#0aa", "#a00", "#a0a", "#a50", "#aaa", "#555", "#55f", "#5f5", "#5ff", "#f55", "#f5f", "#ff5", "#fff"][
                    entry.foreground & 15
                ]!;
                line.textContent = entry.value;
                textLayer.append(line);
            }
        }
        canvas.style.top = `${r.graphics ? r.displayRow * 4 : 0}%`;
        canvas.style.height = `${r.graphics ? 84 : 100}%`;
        score.style.top = `${r.statusRow * 4}%`;
        commandForm.style.top = `${r.inputRow * 4}%`;
        commandForm.style.bottom = "auto";
        commandForm.hidden = !r.inputEnabled || !r.graphics;
        score.hidden = !r.statusEnabled || !r.graphics;
        scriptWindow.hidden = !r.window || Boolean(r.modal);
        scriptWindow.textContent = r.window;
        input.maxLength = r.inputWidth || Math.max(0, r.variables[24]! - 1);
        if (audio && audio.muted === Boolean(r.flags[9])) {
            setMuted(audio, !r.flags[9]);
        }
        soundButton.textContent = r.flags[9] ? "Sound on" : "Sound off";
        score.textContent = `Score: ${r.variables[3]} of ${r.variables[7]}                 ${r.flags[9] ? "Sound: on" : "Sound: off"}`;
        input.disabled = submitButton.disabled = !r.inputEnabled || r.stopped || Boolean(r.modal);
        canvas.dataset.room = String(r.variables[0]);
        canvas.dataset.egoX = String(r.actors[0]!.x);
        canvas.dataset.egoY = String(r.actors[0]!.y);
        canvas.dataset.gameId = r.gameId;
        canvas.dataset.gameReady = String(r.actors[0]!.drawn);
    }
    function tick(time: number): void {
        const elapsed = Math.min(250, Math.max(0, time - lastTime));
        lastTime = time;
        try {
            if (runtime && !runtime.stopped && !loading) {
                tickRuntime(runtime, elapsed);
                renderRuntime(runtime, image.data);
                context!.putImageData(image, 0, 0);
                updateDynamicTexture(engine, texture, staging);
                updateText(runtime);
                updateModal(runtime);
                if (runtime.stopped) {
                    canvas.dataset.state = "stopped";
                    report("Game stopped. Restart to play again.");
                }
            }
        } catch (error) {
            failure(error);
        }
        frameId = requestAnimationFrame(tick);
    }
    frameId = requestAnimationFrame(tick);
    window.addEventListener(
        "pagehide",
        () => {
            cancelAnimationFrame(frameId);
            observer.disconnect();
            disposeSpriteRenderer(renderer);
            disposeEngine(engine);
            if (audio) {
                stopSound(audio);
                void audio.context.close().catch((error: unknown) => console.error(error));
            }
        },
        { once: true }
    );
}

void main().catch((error: unknown) => {
    const status = document.getElementById("status");
    const canvas = document.getElementById("renderCanvas");
    const text = error instanceof Error ? error.message : String(error);
    if (status) {
        status.textContent = text;
        status.dataset.error = "true";
    }
    if (canvas) {
        canvas.dataset.error = text;
    }
    console.error(error);
});
