import type {
  BackendApi,
  GameState,
  InitializationResult,
  Move,
  ReplayFrame,
  ReplayOpenResult,
  ReplaySummary,
  WinRateAnalysis,
} from "../types";

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
      id: "cubesprite_v3",
      display_name: "CubeSprite V3",
      model_path: "models/cubesprite_v3.onnx",
      architecture: "gravity_resnet_v1",
      artifact_sha256: "61f4619d4b46daba149667697fcc9ffbf28171cef9b03d1b659a07395403814e",
      source_iteration: 240,
      default_mcts_sims: 256,
      default_temperature: 0.4,
      description: { zh: "CubeSprite V3 旗舰版", en: "CubeSprite V3 flagship" },
      available: true,
      unavailable_reason: null,
    },
    {
      id: "cubesprite_v3_mini",
      display_name: "CubeSprite V3 mini",
      model_path: "models/cubesprite_v3_mini.onnx",
      architecture: "gravity_resnet_v1",
      artifact_sha256: "31143a556257708b2363b3e280988c1bf00fb15df49b7bc842de015fd6a6b8a9",
      source_iteration: 260,
      default_mcts_sims: 256,
      default_temperature: 0.4,
      description: { zh: "CubeSprite V3 mini", en: "CubeSprite V3 mini" },
      available: true,
      unavailable_reason: null,
    },
    {
      id: "v2.2_balance",
      display_name: "v2.2_balance",
      model_path: "models/v2.2_balance.onnx",
      architecture: "modern",
      artifact_sha256: "bb8cc0c6042276dfa3954e67b71f1fd43f603f9d6d9a0492412726cc41d30712",
      source_iteration: null,
      default_mcts_sims: 256,
      default_temperature: 0.4,
      description: { zh: "均衡模型", en: "Balanced model" },
      available: true,
      unavailable_reason: null,
    },
    {
      id: "v2.1_high",
      display_name: "v2.1_high",
      model_path: "models/v2.1_high.onnx",
      architecture: "legacy-v21",
      artifact_sha256: "d2b761e40bdccc40e8745589605dc46951cfb240ff357439a98c11035892bfa1",
      source_iteration: null,
      default_mcts_sims: 256,
      default_temperature: 0.4,
      description: { zh: "旧版模型", en: "Legacy model" },
      available: true,
      unavailable_reason: null,
    },
  ],
  state: emptyState(),
};

export function replaySummary(overrides: Partial<ReplaySummary> = {}): ReplaySummary {
  return {
    id: "replay-1",
    name: "测试回放 01",
    saved_at: "2026-07-18T10:20:30Z",
    move_count: 2,
    status: "unfinished",
    winner: null,
    fingerprint: "sha256:test-replay-1",
    ...overrides,
  };
}

export function replayAnalysis(overrides: Partial<WinRateAnalysis> = {}): WinRateAnalysis {
  return {
    format: "cubesprite.win-rate-analysis",
    protocol_version: 1,
    replay_id: "replay-1",
    replay_fingerprint: "sha256:test-replay-1",
    model: {
      id: "v2.2_balance",
      display_name: "v2.2_balance",
      architecture: "modern",
      artifact_sha256: "bb8cc0c6042276dfa3954e67b71f1fd43f603f9d6d9a0492412726cc41d30712",
      source_iteration: null,
    },
    config: { model_id: "v2.2_balance", mcts_sims: 128, temperature: 1 },
    request_generation: 1,
    started_at: "2026-07-18T10:21:00Z",
    completed_at: "2026-07-18T10:21:02Z",
    duration_ms: 2000,
    points: [
      { step: 0, red: 0.5, blue: 0.5, estimate: "model_mcts" },
      { step: 1, red: 0.6, blue: 0.4, estimate: "model_mcts" },
      { step: 2, red: 0.45, blue: 0.55, estimate: "model_mcts" },
    ],
    ...overrides,
  };
}

export function replayOpen(overrides: Partial<ReplayOpenResult> = {}): ReplayOpenResult {
  const summary = replaySummary();
  const frame0 = emptyState({
    session_id: "replay:replay-1",
    revision: 0,
    mode: "replay",
    move_count: 0,
    can_undo: false,
  }) as ReplayFrame;
  Object.assign(frame0, { replay_id: summary.id, replay_step: 0, replay_total_steps: summary.move_count });

  const board1 = frame0.board.map((layer) => layer.map((row) => [...row]));
  board1[0][0][0] = 1;
  const frame1 = emptyState({
    ...frame0,
    board: board1,
    revision: 1,
    mode: "replay",
    move_count: 1,
    current_player: -1,
    last_move: { action: 0, layer: 0, row: 0, col: 0, player: 1 },
    legal_moves: frame0.legal_moves.filter((move) => move.action !== 0),
  }) as ReplayFrame;
  Object.assign(frame1, { replay_id: summary.id, replay_step: 1, replay_total_steps: summary.move_count });

  const board2 = frame1.board.map((layer) => layer.map((row) => [...row]));
  board2[0][0][1] = -1;
  const frame2 = emptyState({
    ...frame1,
    board: board2,
    revision: 2,
    mode: "replay",
    move_count: 2,
    current_player: 1,
    last_move: { action: 1, layer: 0, row: 0, col: 1, player: -1 },
    legal_moves: frame1.legal_moves.filter((move) => move.action !== 1),
  }) as ReplayFrame;
  Object.assign(frame2, { replay_id: summary.id, replay_step: 2, replay_total_steps: summary.move_count });

  return {
    replay: {
      ...summary,
      format: "cubesprite.replay",
      protocol_version: 1,
      rules: {
        format: "connect4-3d-gravity",
        version: 1,
        board_layers: 6,
        board_size: 5,
        connect_n: 4,
        gravity: "layer_ascending",
        starting_player: 1,
      },
      moves: [
        { ply: 1, action: 0, layer: 0, row: 0, col: 0, player: 1 },
        { ply: 2, action: 1, layer: 0, row: 0, col: 1, player: -1 },
      ],
    },
    frames: [frame0, frame1, frame2],
    analysis: null,
    ...overrides,
  };
}

export class FakeBackend implements BackendApi {
  readonly calls: Array<{ command: string; params: Record<string, unknown> }> = [];
  state = emptyState();
  replays: ReplaySummary[] = [replaySummary()];
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
    if (command === "analysis.tactical_hint") return null as T;
    if (command === "analysis.win_rate") return { for_revision: this.state.revision, red: 0.6, blue: 0.4, estimate: "model_mcts" } as T;
    if (command === "replay.list") return { replays: this.replays } as T;
    if (command === "replay.save") {
      const replay = replaySummary({ id: `replay-${this.replays.length + 1}`, name: `保存回放 ${this.replays.length + 1}`, move_count: this.state.move_count });
      this.replays = [replay, ...this.replays];
      return { replay } as T;
    }
    if (command === "replay.import") {
      const replay = replaySummary({ id: `import-${this.replays.length + 1}`, name: params.filename as string });
      this.replays = [replay, ...this.replays];
      return { replay } as T;
    }
    if (command === "replay.open") return replayOpen() as T;
    if (command === "replay.hint") {
      const opened = replayOpen();
      const step = params.step as number;
      return {
        replay_id: opened.replay.id,
        replay_fingerprint: opened.replay.fingerprint,
        for_step: step,
        for_revision: step,
        move: opened.frames[step].legal_moves[0],
        value: 0.1,
      } as T;
    }
    if (command === "replay.export") {
      return { filename: "test-replay.c4replay.json", content: JSON.stringify(replayOpen().replay, null, 2) } as T;
    }
    if (command === "replay.delete") {
      this.replays = this.replays.filter((replay) => replay.id !== params.id);
      return { deleted: true } as T;
    }
    if (command === "replay.analyze") return replayAnalysis() as T;
    if (command === "replay.continue") {
      this.state = emptyState({
        session_id: "continued-session",
        revision: 1,
        mode: params.mode as "pvp" | "pvai",
        human_player: params.human_player as 1 | -1,
        move_count: params.step as number,
      });
      return this.state as T;
    }
    throw new Error(`Unhandled fake command: ${command}`);
  }
}
