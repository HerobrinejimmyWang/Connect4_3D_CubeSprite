export type Language = "zh" | "en";
export type Screen = "loading" | "menu" | "ai-settings" | "settings" | "instructions" | "game";
export type Player = 1 | -1;
export type GameMode = "pvp" | "pvai";
export type AiRole = "combat" | "hint" | "winRate";
export type BoardViewMode = "2d" | "3d";
export type PieceFocus = "all" | "red" | "blue";
export type LayerSpacing = "standard" | "expanded";
export type CameraPreset = "isometric" | "front" | "top";

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
  mode: GameMode;
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
}

export interface WinRateResult {
  for_revision: number;
  red: number;
  blue: number;
  estimate: string;
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
