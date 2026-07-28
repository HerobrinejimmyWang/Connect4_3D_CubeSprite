export type Language = "zh" | "en";
export type Screen = "loading" | "menu" | "ai-settings" | "settings" | "instructions" | "game" | "replay";
export type MenuPanel = "side" | "replays" | null;
export type AutoplayInterval = 2000 | 1000 | 500 | 250;
export type TacticalHintDelay = "off" | 0 | 5000;
export type Player = 1 | -1;
export type GameMode = "pvp" | "pvai";
export type GameStateMode = GameMode | "replay";
export type AiRole = "combat" | "hint" | "winRate";
export type BoardViewMode = "2d" | "3d";
export type LayerViewMode = "sliding4" | "all6";
export type PieceFocus = "all" | "red" | "blue";
export type LayerSpacing = "standard" | "expanded";
export type CameraPreset = "isometric" | "front" | "top";
export type SliceAxis = "col" | "row" | "layer";

export interface SliceSelection {
  axis: SliceAxis;
  index: number;
}

export interface CameraCommand {
  preset: CameraPreset;
  serial: number;
}

export interface Move {
  action: number;
  layer: number;
  row: number;
  col: number;
  player?: Player;
}

export interface GameState {
  session_id: string;
  revision: number;
  mode: GameStateMode;
  human_player: Player;
  board: number[][][];
  current_player: Player;
  move_count: number;
  status: "playing" | "won" | "draw";
  winner: Player | 0 | null;
  last_move: Move | null;
  winning_line: Move[];
  legal_moves: Move[];
  can_undo: boolean;
  analysis?: {
    value: number;
    policy: number[];
  };
}

export interface ModelInfo {
  id: string;
  display_name: string;
  model_path: string | null;
  architecture: string;
  artifact_sha256: string;
  source_iteration: number | null;
  default_mcts_sims: number;
  default_temperature: number;
  description: Record<Language, string>;
  available: boolean;
  unavailable_reason: string | null;
}

export interface InitializationResult {
  backend_version: string;
  protocol_version: number;
  board: { layers: number; size: number; connect_n: number };
  mcts_options: number[];
  models: ModelInfo[];
  capabilities?: {
    replay?: boolean;
  };
  state: GameState;
}

export interface AiConfig {
  model_id: string;
  mcts_sims: number;
  temperature: number;
}

export type AiSettings = Record<AiRole, AiConfig>;

export interface HintResult {
  for_revision: number;
  move: Move;
  value: number;
  kind?: "win" | "block";
}

export interface ReplayHintResult extends HintResult {
  replay_id: string;
  replay_fingerprint: string;
  for_step: number;
}

export interface WinRateResult {
  for_revision: number;
  red: number;
  blue: number;
  estimate: string;
}

export type ReplayStatus = "playing" | "unfinished" | "won" | "draw";

export interface ReplaySummary {
  id: string;
  name: string;
  saved_at: string;
  move_count: number;
  status: ReplayStatus;
  winner: Player | 0 | null;
  fingerprint: string;
}

export interface ReplayMove extends Move {
  ply: number;
  player: Player;
}

export interface ReplayDocument extends ReplaySummary {
  format: string;
  protocol_version: number;
  rules: {
    format: "connect4-3d-gravity";
    version: number;
    board_layers: number;
    board_size: number;
    connect_n: number;
    gravity: "layer_ascending";
    starting_player: 1;
  };
  moves: ReplayMove[];
}

export interface ReplayFrame extends GameState {
  mode: "replay";
  replay_id: string;
  replay_step: number;
  replay_total_steps: number;
}

export interface WinRateAnalysisPoint {
  step: number;
  red: number;
  blue: number;
  estimate: string;
}

export interface WinRateAnalysis {
  format: string;
  protocol_version: number;
  replay_id: string;
  replay_fingerprint: string;
  model: {
    id: string;
    display_name: string;
    architecture: string;
    artifact_sha256: string;
    source_iteration: number | null;
  };
  config: AiConfig;
  request_generation: number;
  started_at: string;
  completed_at: string;
  duration_ms: number;
  points: WinRateAnalysisPoint[];
}

export interface ReplayOpenResult {
  replay: ReplayDocument;
  frames: ReplayFrame[];
  analysis: WinRateAnalysis | null;
}

export interface ReplayListResult {
  replays: ReplaySummary[];
}

export interface ReplayMutationResult {
  replay: ReplaySummary;
}

export interface ReplayExportResult {
  filename: string;
  content: string;
}

export interface BackendApi {
  start(): Promise<InitializationResult>;
  request<T>(command: string, params?: Record<string, unknown>): Promise<T>;
  close(): Promise<void>;
}

export interface BackendErrorShape {
  code: string;
  message: string;
  details?: unknown;
}
