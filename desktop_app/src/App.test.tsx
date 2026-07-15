import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "./App";
import { emptyState, FakeBackend } from "./test/fixtures";

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
});
