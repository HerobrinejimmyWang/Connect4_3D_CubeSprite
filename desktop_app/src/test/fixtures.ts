import type { BackendApi, GameState, InitializationResult, Move } from "../types";

export function emptyState(overrides: Partial<GameState> = {}): GameState {
  const legal_moves: Move[] = Array.from({ length: 25 }, (_, action) => ({
    action,
    layer: 0,
    row: Math.floor(action / 5),
    col: action % 5,
  }));
  return {
    session_id: "test-session",
    revision: 1,
    mode: "pvp",
    human_player: 1,
    board: Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => Array(5).fill(0))),
    current_player: 1,
    move_count: 0,
    status: "playing",
    winner: null,
    last_move: null,
    winning_line: [],
    legal_moves,
    can_undo: false,
    ...overrides,
  };
}

export const initialization: InitializationResult = {
  backend_version: "0.1.0",
  protocol_version: 1,
  board: { layers: 6, size: 5, connect_n: 4 },
  mcts_options: [32, 64, 128, 256, 512, 1024],
  models: [
    {
      id: "cubesprite-v3",
      display_name: "CubeSprite V3",
      model_path: null,
      architecture: "gravity_resnet_v1",
      default_mcts_sims: 128,
      default_temperature: 1,
      description: { zh: "训练中", en: "Training in progress" },
      available: false,
      unavailable_reason: "model_not_ready",
    },
    {
      id: "v2.2_balance",
      display_name: "v2.2_balance",
      model_path: "models/v2.2_balance.onnx",
      architecture: "modern",
      default_mcts_sims: 128,
      default_temperature: 1,
      description: { zh: "均衡模型", en: "Balanced model" },
      available: true,
      unavailable_reason: null,
    },
    {
      id: "v2.1_high",
      display_name: "v2.1_high",
      model_path: "models/v2.1_high.onnx",
      architecture: "legacy-v21",
      default_mcts_sims: 128,
      default_temperature: 1,
      description: { zh: "旧版模型", en: "Legacy model" },
      available: true,
      unavailable_reason: null,
    },
  ],
  state: emptyState(),
};

export class FakeBackend implements BackendApi {
  readonly calls: Array<{ command: string; params: Record<string, unknown> }> = [];
  state = emptyState();
  handler?: (command: string, params: Record<string, unknown>) => unknown | Promise<unknown>;

  async start(): Promise<InitializationResult> { return initialization; }
  async close(): Promise<void> { /* no-op */ }

  async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    this.calls.push({ command, params });
    if (this.handler) return await this.handler(command, params) as T;
    if (command === "game.new") {
      this.state = emptyState({
        session_id: `session-${this.calls.length}`,
        revision: this.state.revision + 1,
        mode: params.mode as "pvp" | "pvai",
        human_player: params.human_player as 1 | -1,
      });
      return this.state as T;
    }
    if (command === "game.ai_move") {
      const board = this.state.board.map((layer) => layer.map((row) => [...row]));
      board[0][0][0] = 1;
      this.state = emptyState({ ...this.state, board, revision: this.state.revision + 1, current_player: -1, move_count: 1, last_move: { action: 0, layer: 0, row: 0, col: 0, player: 1 }, can_undo: true, legal_moves: this.state.legal_moves.filter((move) => move.action !== 0) });
      return this.state as T;
    }
    if (command === "game.move") {
      const move = this.state.legal_moves.find((candidate) => candidate.layer === params.layer && candidate.row === params.row && candidate.col === params.col)!;
      const board = this.state.board.map((layer) => layer.map((row) => [...row]));
      board[move.layer][move.row][move.col] = this.state.current_player;
      this.state = emptyState({ ...this.state, board, revision: this.state.revision + 1, current_player: this.state.current_player === 1 ? -1 : 1, move_count: this.state.move_count + 1, last_move: { ...move, player: this.state.current_player }, can_undo: true, legal_moves: this.state.legal_moves.filter((candidate) => candidate.action !== move.action) });
      return this.state as T;
    }
    if (command === "game.restart" || command === "game.undo") return this.state as T;
    if (command === "analysis.hint") return { for_revision: this.state.revision, move: this.state.legal_moves[0], value: 0.1 } as T;
    if (command === "analysis.win_rate") return { for_revision: this.state.revision, red: 0.6, blue: 0.4, estimate: "model_mcts" } as T;
    throw new Error(`Unhandled fake command: ${command}`);
  }
}
