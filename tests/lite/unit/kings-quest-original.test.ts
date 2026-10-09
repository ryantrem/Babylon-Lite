import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import type { GameResources } from "../../../lab/lite/src/demos/kings-quest/resources.js";
import { loadKingsQuestGame, playQuest, questState, replayQuestInputs } from "./kings-quest-playthrough.js";
import { kingsQuestWinningRoute } from "./kings-quest-route.js";

const originalHashes: Record<string, string> = {
    LOGDIR: "18D53AC02A5457494F03352C36A4A4C2EB8251DF70FFF8DAE6DC3D5B41D55B38",
    OBJECT: "2D1B7A75BB443B4A340CB8D9A2842FF5C834D7D597EA2A1365B6A7A2DFB57985",
    PICDIR: "7A2181429C473C5983547F115869C1A948249D986D0176F5FA30B605510CC3C9",
    SNDDIR: "FD5D50FD0EAC57168E3CD070B7CF0D7040A26D303415F5318142A0695959563D",
    VIEWDIR: "1B36AB1CCAA118870EE7D6D651359912E79F35F3ACC71858E4B668E3B34E2A04",
    "VOL.0": "7F10C63737B4DA8CF1633BA7C317FFC84A4BBC4ECA10B39D2E349B18636BB5E2",
    "VOL.1": "7C3A0C86C8AD1B028DC3D56CAFB30A925690F243D70F2E7B2AC6F0DC802B00D0",
    "VOL.2": "B6E321821A0D90809D985F57B279C2F155A6972482C4811C4A737987042A53CF",
    "WORDS.TOK": "41D863172326C712C0AEBADF12FC63B049FF5D892743F4EE990004C344EB3780",
};

function originalAssets(game: GameResources): void {
    expect(game.fingerprint).toBe("f5cdbd20");
    for (const [name, expected] of Object.entries(originalHashes)) {
        const bytes = game.files.get(name.toLowerCase());
        expect(bytes, name).toBeDefined();
        expect(createHash("sha256").update(bytes!).digest("hex").toUpperCase(), name).toBe(expected);
    }
}

function through(action: string, inclusive = false): string[] {
    const index = kingsQuestWinningRoute.indexOf(action);
    if (index < 0) {
        throw new Error(`Missing route checkpoint ${action}.`);
    }
    return kingsQuestWinningRoute.slice(0, index + Number(inclusive));
}

describe("Original King's Quest I resources", () => {
    it("earns all three treasures and the original ending from a fresh start, then replays every actual input without navigation", () => {
        const game = loadKingsQuestGame();
        originalAssets(game);
        const result = playQuest(game, kingsQuestWinningRoute);
        const r = result.runtime;
        expect(questState(r)).toMatchObject({ room: 53, x: 30, y: 85, view: 142, score: 140, dead: false });
        expect([r.variables[74], r.flags[195], r.inputEnabled]).toEqual([3, 1, false]);
        expect([r.inventory[2], r.inventory[19], r.inventory[21]]).toEqual([255, 255, 255]);
        const commands = result.trace.filter((entry) => entry.kind === "command").map((entry) => entry.detail);
        expect(commands).toEqual(
            expect.arrayContaining([
                "IFNKOVHGROGHPRM",
                "get chest",
                "throw water",
                "get mirror",
                "fill",
                "push witch",
                "jump",
                "give cheese",
                "play fiddle",
                "get shield",
                "eat mushroom",
                "restore game",
                "bow",
            ])
        );
        expect(result.trace.some((entry) => entry.kind === "modal" && entry.detail.includes("Thank you for playing"))).toBe(true);
        expect(result.inputs.some((input) => input.kind === "direction")).toBe(true);
        const checkpoint = JSON.parse(result.checkpoint) as { state: { variables: number[]; inventory: number[]; flags: number[]; actors: { x: number; y: number }[] } };
        expect([checkpoint.state.variables[0], checkpoint.state.variables[3], checkpoint.state.flags[195]]).toEqual([53, 137, 0]);
        expect(checkpoint.state.actors[0]).toMatchObject({ x: 70, y: 105 });
        expect([checkpoint.state.inventory[2], checkpoint.state.inventory[19], checkpoint.state.inventory[21]]).toEqual([255, 255, 255]);
        const replay = replayQuestInputs(loadKingsQuestGame(), result.inputs);
        expect(questState(replay)).toEqual(questState(r));
        expect(replay.flags[195]).toBe(1);
        originalAssets(game);
        originalAssets(replay.game);
    }, 60000);

    it("awards the original gold-key alternative after three wrong gnome names", () => {
        const result = playQuest(loadKingsQuestGame(), [
            ...through("command:IFNKOVHGROGHPRM"),
            "command:rumpelstiltskin",
            "command:nikstlitselpmur",
            "command:graham",
            "near:3:15",
            "command:get key",
        ]);
        expect(result.runtime.variables[83]).toBe(2);
        expect(result.runtime.inventory[5]).toBe(255);
        expect([result.runtime.variables[84], result.runtime.variables[85], result.runtime.variables[3]]).toEqual([3, 1, 33]);
        expect(result.runtime.inventory[25]).not.toBe(255);
        expect(result.trace.some((entry) => entry.detail.includes("gold key anyway"))).toBe(true);
        originalAssets(result.runtime.game);
    }, 30000);

    it("supports the original dagger/dragon alternative and mirror pickup with its original point cost", () => {
        const result = playQuest(loadKingsQuestGame(), [...through("command:throw water"), "command:throw dagger", "wait:100", "region:26:104:59:127", "command:get mirror"]);
        expect(result.runtime.variables[75]).toBe(1);
        expect([result.runtime.inventory[1], result.runtime.inventory[2], result.runtime.inventory[19]]).toEqual([0, 255, 255]);
        expect(questState(result.runtime)).toMatchObject({ room: 51, score: 72, dead: false });
        expect(result.trace.some((entry) => entry.detail.includes("strike it in the heart"))).toBe(true);
        originalAssets(result.runtime.game);
    }, 30000);

    it("enforces original dragon death and restores an earned checkpoint through the original F7 binding before solving the puzzle", () => {
        const result = playQuest(
            loadKingsQuestGame(),
            [
                ...through("command:throw water"),
                "command:save game",
                "walk:40:135",
                "key:16640",
                "wait:3",
                "command:throw water",
                "wait:100",
                "region:26:104:59:127",
                "command:get mirror",
            ],
            true
        );
        expect(result.trace.some((entry) => entry.state.dead)).toBe(true);
        expect(result.runtime.flags[63]).toBe(0);
        expect([result.runtime.inventory[2], result.runtime.inventory[19]]).toEqual([255, 255]);
        expect(questState(result.runtime)).toMatchObject({ room: 51, score: 79, dead: false });
        originalAssets(result.runtime.game);
    }, 30000);
});
