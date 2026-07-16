import type { LayerSpacing, Move, PieceFocus, Player } from "./types";

export const BOARD_SIZE = 5;
export const MAX_LAYERS = 6;
export const CELL_GAP = 1.22;
export const STANDARD_LAYER_GAP = 1.04;
export const EXPANDED_LAYER_GAP = 1.46;
export const MAX_CLICK_DELTA = 4;

export interface ColumnGuide {
  row: number;
  col: number;
  key: string;
}

export type PieceRenderMode = "solid" | "outline" | "winning";

export function coordinateKey(move: Pick<Move, "layer" | "row" | "col">): string {
  return `${move.layer}:${move.row}:${move.col}`;
}

export function columnKey(row: number, col: number): string {
  return `${row}:${col}`;
}

export function layerGap(spacing: LayerSpacing): number {
  return spacing === "expanded" ? EXPANDED_LAYER_GAP : STANDARD_LAYER_GAP;
}

export function boardToWorld(
  layer: number,
  row: number,
  col: number,
  spacing: LayerSpacing,
): [number, number, number] {
  if (layer < 0 || layer >= MAX_LAYERS || row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
    throw new RangeError(`Invalid board coordinate: ${layer}, ${row}, ${col}`);
  }
  return [
    (col - (BOARD_SIZE - 1) / 2) * CELL_GAP,
    (layer - (MAX_LAYERS - 1) / 2) * layerGap(spacing),
    (row - (BOARD_SIZE - 1) / 2) * CELL_GAP,
  ];
}

export function createColumnGuides(): ColumnGuide[] {
  return Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, index) => {
    const row = Math.floor(index / BOARD_SIZE);
    const col = index % BOARD_SIZE;
    return { row, col, key: columnKey(row, col) };
  });
}

export function legalMovesByColumn(moves: Move[]): Map<string, Move> {
  return new Map(moves.map((move) => [columnKey(move.row, move.col), move]));
}

export function findLegalMoveForColumn(moves: Move[], row: number, col: number): Move | null {
  return legalMovesByColumn(moves).get(columnKey(row, col)) ?? null;
}

export function pieceRenderMode(player: Player, focus: PieceFocus, winning: boolean): PieceRenderMode {
  if (winning) return "winning";
  if (focus === "all") return "solid";
  const focusedPlayer: Player = focus === "red" ? 1 : -1;
  return player === focusedPlayer ? "solid" : "outline";
}

export function isIntentionalBoardClick(delta: number): boolean {
  return Number.isFinite(delta) && delta <= MAX_CLICK_DELTA;
}
