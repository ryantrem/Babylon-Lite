export const styles = `
:root {
    color-scheme: dark;
    font-family: system-ui, sans-serif;
    color: #ece8d9;
    background: #10151b;
}
* {
    box-sizing: border-box;
}
body {
    margin: 0;
    min-height: 100vh;
}
main {
    width: min(960px, 100%);
    margin: auto;
    padding: 20px;
}
header {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 12px;
}
h1 {
    font-size: 1.5rem;
    margin: 0;
}
.badge {
    color: #a9cbd4;
    font-size: 0.8rem;
}
.toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 16px 0;
}
button,
input {
    font: inherit;
}
button {
    border: 1px solid #617883;
    border-radius: 4px;
    background: #24333e;
    color: inherit;
    padding: 8px 12px;
    cursor: pointer;
}
button:hover {
    background: #354a58;
}
button:disabled {
    opacity: 0.45;
    cursor: default;
}
button:focus-visible,
input:focus-visible,
canvas:focus-visible {
    outline: 2px solid #e6c57b;
    outline-offset: 2px;
}
.screen {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    background: #000;
    border: 2px solid #657580;
    overflow: hidden;
    --cell: 16px;
}
#renderCanvas {
    position: absolute;
    top: 4%;
    left: 0;
    width: 100%;
    height: 84%;
    image-rendering: pixelated;
}
#gameText {
    position: absolute;
    inset: 0;
    pointer-events: none;
    font: var(--cell)/1 monospace;
}
.text-line {
    position: absolute;
    white-space: pre-wrap;
    margin: 0;
    overflow-wrap: anywhere;
}
#scriptWindow {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    max-width: 85%;
    padding: 1em;
    white-space: pre-wrap;
    font-family: monospace;
    background: #eee9db;
    color: #111;
    border: 3px double #252525;
}
.command[hidden] {
    display: none;
}
#score {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4%;
    padding: 0 1%;
    font: var(--cell)/1 monospace;
    white-space: pre;
    background: #fff;
    color: #000;
}
.command {
    position: absolute;
    bottom: 4%;
    left: 0;
    width: 100%;
    height: 8%;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0 8px;
    font: var(--cell)/1 monospace;
    background: #000;
}
.command input {
    flex: 1;
    min-width: 0;
    background: transparent;
    color: #fff;
    border: 0;
    outline-offset: 0;
}
.command button {
    padding: 3px 6px;
    font-size: 0.65em;
}
#status {
    min-height: 2.5em;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    padding: 12px 0;
    color: #bfd1d8;
}
#status[data-error="true"] {
    color: #ffb7a4;
}
.help {
    color: #aebac1;
    font-size: 0.85rem;
    line-height: 1.6;
}
.help a {
    color: #a9cbd4;
}
dialog {
    width: min(650px, 90vw);
    max-height: 85vh;
    overflow: auto;
    background: #eee9db;
    color: #111;
    border: 3px double #252525;
}
dialog::backdrop {
    background: #0006;
}
dialog p {
    white-space: pre-wrap;
    font-family: monospace;
}
dialog input {
    color: #111;
    background: #fff;
    width: 100%;
    margin: 12px 0;
    padding: 8px;
}
dialog button {
    background: #e2dfd3;
    color: #111;
    border-color: #444;
    margin: 4px;
}
dialog h2 {
    font-size: 1rem;
}
dialog .preview {
    display: block;
    image-rendering: pixelated;
    width: auto;
    max-width: 100%;
    margin: auto;
}
@media (max-width: 600px) {
    main {
        padding: 10px;
    }
    h1 {
        font-size: 1.2rem;
    }
    button {
        padding: 6px 8px;
    }
}
`;
