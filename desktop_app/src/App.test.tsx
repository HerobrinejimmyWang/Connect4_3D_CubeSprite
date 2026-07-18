import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { App } from "./App";
import { emptyState, FakeBackend, replayAnalysis, replayOpen } from "./test/fixtures";
import type { GameState, ReplayOpenResult, WinRateAnalysis } from "./types";

class DeferredAiBackend extends FakeBackend {
  resolveAi: (() => void) | null = null;

  override async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (command !== "game.ai_move") return super.request<T>(command, params);
    this.calls.push({ command, params });
    return new Promise<T>((resolve) => {
      this.resolveAi = () => {
        const move = this.state.legal_moves.find((candidate) => candidate.action === 12)!;
        const board = this.state.board.map((layer) => layer.map((row) => [...row]));
        board[move.layer][move.row][move.col] = 1;
        this.state = emptyState({
          ...this.state,
          board,
          revision: this.state.revision + 1,
          current_player: -1,
          move_count: 1,
          last_move: { ...move, player: 1 },
          legal_moves: this.state.legal_moves.filter((candidate) => candidate.action !== move.action),
        });
        resolve(this.state as T);
      };
    });
  }
}

class DeferredAnalysisBackend extends FakeBackend {
  resolveAnalysis: ((analysis: WinRateAnalysis) => void) | null = null;

  override async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (command !== "replay.analyze") return super.request<T>(command, params);
    this.calls.push({ command, params });
    return new Promise<T>((resolve) => {
      this.resolveAnalysis = (analysis) => resolve(analysis as T);
    });
  }
}

class DeferredReplayOpenBackend extends FakeBackend {
  resolveOpen: ((opened: ReplayOpenResult) => void) | null = null;

  override async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (command !== "replay.open") return super.request<T>(command, params);
    this.calls.push({ command, params });
    return new Promise<T>((resolve) => {
      this.resolveOpen = (opened) => resolve(opened as T);
    });
  }
}

class DeferredContinueBackend extends FakeBackend {
  resolveContinue: ((state: GameState) => void) | null = null;

  override async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (command !== "replay.continue") return super.request<T>(command, params);
    this.calls.push({ command, params });
    return new Promise<T>((resolve) => {
      this.resolveContinue = (state) => resolve(state as T);
    });
  }
}

class DeferredMoveBackend extends FakeBackend {
  resolveMove: ((state: GameState) => void) | null = null;

  override async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (command !== "game.move") return super.request<T>(command, params);
    this.calls.push({ command, params });
    return new Promise<T>((resolve) => {
      this.resolveMove = (state) => resolve(state as T);
    });
  }
}

class DeferredNewGameBackend extends FakeBackend {
  resolveNewGame: ((state: GameState) => void) | null = null;

  override async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (command !== "game.new") return super.request<T>(command, params);
    this.calls.push({ command, params });
    return new Promise<T>((resolve) => {
      this.resolveNewGame = (state) => resolve(state as T);
    });
  }
}

describe("CubeSprite app shell", () => {
  it("loads the menu and switches all copy instantly", async () => {
    const user = userEvent.setup();
    render(<App backend={new FakeBackend()} />);

    expect(await screen.findByRole("button", { name: "玩家 vs 玩家" })).toBeVisible();
    expect(screen.queryByText("离线 · 本地 AI · 6 × 5 × 5")).not.toBeInTheDocument();
    expect(screen.queryByText("在六层立方棋盘上，连成属于你的四子路线。")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "EN" }));
    expect(screen.getByRole("button", { name: "Player vs Player" })).toBeVisible();
    expect(screen.queryByText("OFFLINE · LOCAL AI · 6 × 5 × 5")).not.toBeInTheDocument();
    expect(screen.queryByText("Build your line of four across a six-floor cube.")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Instructions" }));
    expect(screen.getByRole("heading", { name: "Instructions" })).toBeVisible();
    expect(screen.getByText(/six floors, F1–F6/i)).toBeVisible();
    const instructions = screen.getByRole("region", { name: "Instructions" });
    expect(instructions).toHaveClass("instruction-list");
    expect(within(instructions).getAllByRole("listitem")).toHaveLength(9);
    expect(document.querySelectorAll(".instruction-card")).toHaveLength(0);
  });

  it("lets a PvAI blue player choose second and automatically asks the combat AI to open", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);

    await user.click(await screen.findByRole("button", { name: "玩家 vs AI" }));
    await user.click(screen.getByRole("button", { name: /蓝方 · 后手/ }));

    expect(await screen.findByText("蓝方", { selector: ".turn-state strong" })).toBeVisible();
    expect(backend.calls.some((call) => call.command === "game.new" && call.params.human_player === -1)).toBe(true);
    const aiCall = backend.calls.find((call) => call.command === "game.ai_move");
    expect(aiCall?.params).toMatchObject({ expected_revision: 2 });
    expect(aiCall?.params.ai).toMatchObject({ model_id: "v2.2_balance", mcts_sims: 128, temperature: 1 });
  });

  it("only sends a backend move for a legal board cell", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "玩家 vs 玩家" }));

    const legal = await screen.findByRole("button", { name: "F1, 1, 1: 合法落点" });
    const illegal = screen.getByRole("button", { name: "F2, 1, 1: 暂不可落" });
    expect(illegal).toBeDisabled();
    await user.click(legal);

    await waitFor(() => expect(backend.calls.some((call) => call.command === "game.move")).toBe(true));
    expect(backend.calls.find((call) => call.command === "game.move")?.params).toMatchObject({ layer: 0, row: 0, col: 0, expected_revision: 2 });
  });

  it("keeps the three AI roles independent", async () => {
    const user = userEvent.setup();
    render(<App backend={new FakeBackend()} />);
    await user.click(await screen.findByRole("button", { name: "AI 设置" }));

    const columns = screen.getAllByRole("article");
    const combat = columns[0];
    const hint = columns[1];
    expect(within(combat).getByText("v2.2_balance")).toBeVisible();
    expect(within(combat).queryByText("均衡模型")).not.toBeInTheDocument();
    expect(within(combat).getByText("128")).toBeVisible();
    expect(within(hint).getByText("128")).toBeVisible();
    await user.click(within(combat).getByRole("button", { name: "combat MCTS plus" }));
    expect(within(combat).getByText("256")).toBeVisible();
    expect(within(hint).getByText("128")).toBeVisible();
  });

  it("keeps the replay autoplay interval as an independent app setting", async () => {
    const user = userEvent.setup();
    render(<App backend={new FakeBackend()} />);
    await user.click(await screen.findByRole("button", { name: "设置" }));

    const group = screen.getByRole("radiogroup", { name: "回放自动播放速度" });
    expect(within(group).getByRole("radio", { name: "1.0 s" })).toHaveAttribute("aria-checked", "true");
    await user.click(within(group).getByRole("radio", { name: "0.25 s" }));
    expect(within(group).getByRole("radio", { name: "0.25 s" })).toHaveAttribute("aria-checked", "true");
    expect(within(group).getByRole("radio", { name: "1.0 s" })).toHaveAttribute("aria-checked", "false");
  });

  it("preloads a hint in the background only when enabled", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "设置" }));
    const toggle = screen.getByRole("switch", { name: "预加载提示" });
    expect(toggle).toHaveAttribute("aria-checked", "false");
    await user.click(toggle);
    await user.click(screen.getByRole("button", { name: /返回主菜单/ }));
    await user.click(screen.getByRole("button", { name: "玩家 vs 玩家" }));

    await waitFor(() => expect(backend.calls.some((call) => call.command === "analysis.hint")).toBe(true));
    expect(await screen.findByText(/提示已预加载/)).toBeVisible();
  });

  it("waits for the combat AI before preloading the next human-turn hint", async () => {
    const user = userEvent.setup();
    const backend = new DeferredAiBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "设置" }));
    await user.click(screen.getByRole("switch", { name: "预加载提示" }));
    await user.click(screen.getByRole("button", { name: /返回主菜单/ }));
    await user.click(screen.getByRole("button", { name: "玩家 vs AI" }));
    await user.click(screen.getByRole("button", { name: /蓝方 · 后手/ }));

    await waitFor(() => expect(backend.calls.some((call) => call.command === "game.ai_move")).toBe(true));
    expect(backend.calls.some((call) => call.command === "analysis.hint")).toBe(false);

    act(() => backend.resolveAi?.());
    await waitFor(() => expect(backend.calls.some((call) => call.command === "analysis.hint")).toBe(true));
  });

  it("keeps menu navigation locked while a new game is being created", async () => {
    const user = userEvent.setup();
    const backend = new DeferredNewGameBackend();
    render(<App backend={backend} />);

    await user.click(await screen.findByRole("button", { name: "玩家 vs 玩家" }));
    await waitFor(() => expect(backend.calls.filter((call) => call.command === "game.new")).toHaveLength(1));
    expect(screen.getByRole("button", { name: "设置" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "设置" }));
    expect(screen.queryByRole("heading", { name: "设置" })).not.toBeInTheDocument();

    act(() => backend.resolveNewGame?.(emptyState({ session_id: "new-game-session" })));
    expect(await screen.findByText("总落子")).toBeVisible();
  });

  it("ignores a move response that arrives after exiting into a newer game", async () => {
    const user = userEvent.setup();
    const backend = new DeferredMoveBackend();
    render(<App backend={backend} />);

    await user.click(await screen.findByRole("button", { name: "玩家 vs 玩家" }));
    const firstSession = backend.state.session_id;
    await user.click(await screen.findByRole("button", { name: "F1, 1, 1: 合法落点" }));
    await waitFor(() => expect(backend.calls.some((call) => call.command === "game.move")).toBe(true));
    await user.click(screen.getByRole("button", { name: "退出对局" }));
    await user.click(await screen.findByRole("button", { name: "玩家 vs 玩家" }));
    const secondSession = backend.state.session_id;
    expect(secondSession).not.toBe(firstSession);

    act(() => backend.resolveMove?.(emptyState({
      session_id: firstSession,
      revision: 99,
      move_count: 1,
    })));

    const totalMoves = await screen.findByText("总落子");
    const totalMovesPanel = totalMoves.parentElement!;
    await waitFor(() => expect(within(totalMovesPanel).getByText("0")).toBeVisible());
    expect(within(totalMovesPanel).queryByText("1")).not.toBeInTheDocument();
  });

  it("opens the local replay library and enters a replay at step zero", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);

    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.list")).toBe(true));
    expect(await screen.findByText("测试回放 01")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "进入 测试回放 01" }));

    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.open")).toBe(true));
    expect(backend.calls.find((call) => call.command === "replay.open")?.params).toMatchObject({
      id: "replay-1",
      expected_fingerprint: "sha256:test-replay-1",
    });
    expect(await screen.findByText("回放进度")).toBeVisible();
    const progress = screen.getByText("回放进度").parentElement!;
    expect(within(progress).getByText("0")).toBeVisible();
    expect(within(progress).getByText("/2")).toBeVisible();
  });

  it("does not enter a replay when its library panel was closed before open completed", async () => {
    const user = userEvent.setup();
    const backend = new DeferredReplayOpenBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    const library = await screen.findByRole("complementary", { name: "回放列表" });
    await user.click(within(library).getByRole("button", { name: "进入 测试回放 01" }));
    await user.click(within(library).getByRole("button", { name: "关闭" }));

    act(() => backend.resolveOpen?.(replayOpen()));

    await waitFor(() => expect(screen.queryByRole("complementary", { name: "回放列表" })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "对局回放" })).toBeVisible();
    expect(screen.queryByText("回放进度")).not.toBeInTheDocument();
  });

  it("does not enter a replay after navigating from its pending open to settings", async () => {
    const user = userEvent.setup();
    const backend = new DeferredReplayOpenBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    await user.click(await screen.findByRole("button", { name: "进入 测试回放 01" }));
    await user.click(screen.getByRole("button", { name: "设置" }));

    act(() => backend.resolveOpen?.(replayOpen()));

    expect(await screen.findByRole("heading", { name: "设置" })).toBeVisible();
    expect(screen.queryByText("回放进度")).not.toBeInTheDocument();
  });

  it("saves the exact current revision from the live game", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "玩家 vs 玩家" }));
    await user.click(await screen.findByRole("button", { name: "F1, 1, 1: 合法落点" }));
    await user.click(screen.getByRole("button", { name: "保存回放" }));

    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.save")).toBe(true));
    const save = backend.calls.find((call) => call.command === "replay.save");
    expect(save?.params).toMatchObject({
      session_id: backend.state.session_id,
      expected_revision: backend.state.revision,
    });
    expect(await screen.findByText(/已保存回放/)).toBeVisible();
  });

  it("imports and deletes replay files from the menu panel", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    expect(await screen.findByText("测试回放 01")).toBeVisible();

    const input = document.querySelector<HTMLInputElement>('.replay-import input[type="file"]')!;
    const file = new File(['{"format":"cubesprite.replay"}'], "shared-replay.json", { type: "application/json" });
    await user.upload(input, file);
    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.import")).toBe(true));
    const imported = backend.calls.find((call) => call.command === "replay.import");
    expect(imported?.params.filename).toBe("shared-replay.json");
    expect(imported?.params.content).toContain("cubesprite.replay");

    await user.click(screen.getByRole("button", { name: "删除 测试回放 01" }));
    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.delete")).toBe(true));
    expect(backend.calls.find((call) => call.command === "replay.delete")?.params).toMatchObject({
      id: "replay-1",
      expected_fingerprint: "sha256:test-replay-1",
    });
    expect(confirm).toHaveBeenCalled();
    expect(screen.queryByText("测试回放 01")).not.toBeInTheDocument();
    confirm.mockRestore();
  });

  it("rejects replay files containing malformed UTF-8 before calling the backend", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));

    const input = document.querySelector<HTMLInputElement>('.replay-import input[type="file"]')!;
    const file = new File(
      [new Uint8Array([0x7b, 0x22, 0xff, 0x22, 0x7d])],
      "invalid-utf8.c4replay.json",
      { type: "application/json" },
    );
    await user.upload(input, file);

    expect(await screen.findByText("回放文件必须是严格有效的 UTF-8 文本。")).toBeVisible();
    expect(backend.calls.some((call) => call.command === "replay.import")).toBe(false);
  });

  it("exports a replay as a downloadable file with optimistic fingerprint protection", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    const originalCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
    const originalRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
    const createObjectURL = vi.fn(() => "blob:cubesprite-replay");
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectURL });
    let downloadedFilename = "";
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      downloadedFilename = this.download;
    });

    try {
      render(<App backend={backend} />);
      await user.click(await screen.findByRole("button", { name: "对局回放" }));
      await user.click(await screen.findByRole("button", { name: "导出 测试回放 01" }));

      await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.export")).toBe(true));
      expect(backend.calls.find((call) => call.command === "replay.export")?.params).toMatchObject({
        id: "replay-1",
        expected_fingerprint: "sha256:test-replay-1",
      });
      expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
      expect(downloadedFilename).toBe("test-replay.c4replay.json");
      expect(revokeObjectURL).toHaveBeenCalledWith("blob:cubesprite-replay");
      expect(await screen.findByText("已导出回放：测试回放 01")).toBeVisible();
    } finally {
      click.mockRestore();
      if (originalCreate) Object.defineProperty(URL, "createObjectURL", originalCreate);
      else Reflect.deleteProperty(URL, "createObjectURL");
      if (originalRevoke) Object.defineProperty(URL, "revokeObjectURL", originalRevoke);
      else Reflect.deleteProperty(URL, "revokeObjectURL");
    }
  });

  it("continues a replay position as a new independent PvAI game", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    await user.click(await screen.findByRole("button", { name: "进入 测试回放 01" }));
    await user.click(await screen.findByRole("button", { name: "下一步" }));
    await user.click(screen.getByRole("button", { name: "从这里继续" }));
    await user.click(screen.getByRole("button", { name: /执蓝对抗 AI/ }));

    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.continue")).toBe(true));
    expect(backend.calls.find((call) => call.command === "replay.continue")?.params).toMatchObject({
      id: "replay-1",
      expected_fingerprint: "sha256:test-replay-1",
      step: 1,
      mode: "pvai",
      human_player: -1,
    });
    await waitFor(() => expect(backend.calls.some((call) => call.command === "game.ai_move")).toBe(true));
    expect(await screen.findByText("PVAI")).toBeVisible();
  });

  it("keeps replay exit disabled until an authoritative continuation completes", async () => {
    const user = userEvent.setup();
    const backend = new DeferredContinueBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    await user.click(await screen.findByRole("button", { name: "进入 测试回放 01" }));
    await user.click(await screen.findByRole("button", { name: "从这里继续" }));
    await user.click(screen.getByRole("button", { name: /玩家 vs 玩家.*双方在本机继续残局/ }));
    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.continue")).toBe(true));
    expect(screen.getByRole("button", { name: "退出回放" })).toBeDisabled();

    act(() => backend.resolveContinue?.(emptyState({ session_id: "late-continuation" })));

    expect(await screen.findByText("总落子")).toBeVisible();
    expect(screen.queryByRole("button", { name: "退出回放" })).not.toBeInTheDocument();
  });

  it("ignores a late replay analysis after leaving the replay screen", async () => {
    const user = userEvent.setup();
    const backend = new DeferredAnalysisBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    await user.click(await screen.findByRole("button", { name: "进入 测试回放 01" }));
    await user.click(await screen.findByRole("button", { name: "胜率计算" }));
    expect(await screen.findByText(/正在后台计算全局胜率/)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "退出回放" }));
    expect(await screen.findByRole("button", { name: "对局回放" })).toBeVisible();

    act(() => backend.resolveAnalysis?.(replayAnalysis()));
    await waitFor(() => expect(screen.queryByRole("img", { name: /胜率随回放步数变化/ })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "对局回放" })).toBeVisible();
  });

  it("rejects an analysis response for a different replay fingerprint", async () => {
    const user = userEvent.setup();
    const backend = new DeferredAnalysisBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "对局回放" }));
    await user.click(await screen.findByRole("button", { name: "进入 测试回放 01" }));
    await user.click(await screen.findByRole("button", { name: "胜率计算" }));

    act(() => backend.resolveAnalysis?.(replayAnalysis({ replay_fingerprint: "sha256:different-replay" })));

    expect(await screen.findByText("回放内容已变化，请返回列表后重新打开。")).toBeVisible();
    expect(screen.queryByRole("img", { name: /胜率随回放步数变化/ })).not.toBeInTheDocument();
  });

  it("recalculates a replay with the current win-rate AI settings", async () => {
    const user = userEvent.setup();
    const backend = new FakeBackend();
    render(<App backend={backend} />);
    await user.click(await screen.findByRole("button", { name: "AI 设置" }));
    const winRateRole = screen.getAllByRole("article")[2];
    await user.click(within(winRateRole).getByRole("button", { name: "winRate MCTS plus" }));
    await user.click(screen.getByRole("button", { name: /返回主菜单/ }));
    await user.click(screen.getByRole("button", { name: "对局回放" }));
    await user.click(await screen.findByRole("button", { name: "进入 测试回放 01" }));
    await user.click(await screen.findByRole("button", { name: "胜率计算" }));

    await waitFor(() => expect(backend.calls.some((call) => call.command === "replay.analyze")).toBe(true));
    expect(backend.calls.find((call) => call.command === "replay.analyze")?.params.ai).toMatchObject({
      model_id: "v2.2_balance",
      mcts_sims: 256,
      temperature: 1,
    });
    expect(backend.calls.find((call) => call.command === "replay.analyze")?.params).toMatchObject({
      id: "replay-1",
      expected_fingerprint: "sha256:test-replay-1",
    });
    expect(await screen.findByRole("img", { name: /胜率随回放步数变化/ })).toBeVisible();
  });
});
