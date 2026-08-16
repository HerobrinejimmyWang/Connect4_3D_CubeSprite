import type { CameraPreset, GameState, LayerSpacing, Move, PieceFocus, Player, SliceSelection } from "./types";

export const BOARD_SIZE = 5;
export const MAX_LAYERS = 6;
export const CELL_GAP = 1.22;
export const STANDARD_LAYER_GAP = 1.04;
export const EXPANDED_LAYER_GAP = 1.46;
export const MAX_CLICK_DELTA = 4;
export const BOARD_WORLD_UP = [0, 1, 0] as const;

const CAMERA_POSITIONS: Record<CameraPreset, readonly [number, number, number]> = {
  isometric: [8.6, 7.6, 8.6],
  front: [0, 1.2, 12],
  top: [0, 12, 0.001],
};

export interface ColumnGuide {
  row: number;
  col: number;
  key: string;
}

export interface SliceCell {
  layer: number;
  row: number;
  col: number;
  value: number;
}

export type PieceRenderMode = "solid" | "outline" | "winning";

export function cameraPosition(preset: CameraPreset): readonly [number, number, number] {
  return CAMERA_POSITIONS[preset];
}

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

export function sliceSelectionLabel(selection: SliceSelection): string {
  validateSliceSelection(selection);
  const prefix = selection.axis === "col" ? "C" : selection.axis === "row" ? "R" : "F";
  return `${prefix}${selection.index + 1}`;
}

export function moveMatchesSlice(
  move: Pick<Move, "layer" | "row" | "col">,
  selection: SliceSelection,
): boolean {
  validateSliceSelection(selection);
  return move[selection.axis] === selection.index;
}

export function sliceCells(board: GameState["board"], selection: SliceSelection): SliceCell[] {
  validateSliceSelection(selection);
  const cells: SliceCell[] = [];

  if (selection.axis === "layer") {
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        cells.push({ layer: selection.index, row, col, value: board[selection.index]?.[row]?.[col] ?? 0 });
      }
    }
    return cells;
  }

  for (let layer = MAX_LAYERS - 1; layer >= 0; layer -= 1) {
    for (let offset = 0; offset < BOARD_SIZE; offset += 1) {
      const row = selection.axis === "row" ? selection.index : offset;
      const col = selection.axis === "col" ? selection.index : offset;
      cells.push({ layer, row, col, value: board[layer]?.[row]?.[col] ?? 0 });
    }
  }
  return cells;
}

function validateSliceSelection(selection: SliceSelection): void {
  const limit = selection.axis === "layer" ? MAX_LAYERS : BOARD_SIZE;
  if (!Number.isInteger(selection.index) || selection.index < 0 || selection.index >= limit) {
    throw new RangeError(`Invalid ${selection.axis} slice index: ${selection.index}`);
  }
}

export function isIntentionalBoardClick(delta: number): boolean {
  return Number.isFinite(delta) && delta <= MAX_CLICK_DELTA;
}
