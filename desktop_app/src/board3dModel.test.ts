import {
  BOARD_SIZE,
  MAX_LAYERS,
  boardToWorld,
  createColumnGuides,
  findLegalMoveForColumn,
  isIntentionalBoardClick,
  layerGap,
  moveMatchesSlice,
  pieceRenderMode,
  sliceCells,
  sliceSelectionLabel,
} from "./board3dModel";
import type { Move } from "./types";

describe("3D board model", () => {
  it("maps board coordinates around a stable center and only expands the layer axis", () => {
    const standard = boardToWorld(2, 2, 2, "standard");
    const expanded = boardToWorld(2, 2, 2, "expanded");
    expect(standard[0]).toBe(0);
    expect(standard[2]).toBe(0);
    expect(expanded[0]).toBe(standard[0]);
    expect(expanded[2]).toBe(standard[2]);
    expect(Math.abs(expanded[1])).toBeGreaterThan(Math.abs(standard[1]));
    expect(boardToWorld(5, 2, 2, "standard")[1] - boardToWorld(4, 2, 2, "standard")[1]).toBeCloseTo(layerGap("standard"));
  });

  it("rejects coordinates outside the 6 x 5 x 5 board", () => {
    expect(() => boardToWorld(-1, 0, 0, "standard")).toThrow(RangeError);
    expect(() => boardToWorld(MAX_LAYERS, 0, 0, "standard")).toThrow(RangeError);
    expect(() => boardToWorld(0, BOARD_SIZE, 0, "standard")).toThrow(RangeError);
    expect(() => boardToWorld(0, 0, BOARD_SIZE, "standard")).toThrow(RangeError);
  });

  it("creates exactly 25 unique row and column guides", () => {
    const guides = createColumnGuides();
    expect(guides).toHaveLength(25);
    expect(new Set(guides.map((guide) => guide.key)).size).toBe(25);
    expect(guides[0]).toMatchObject({ row: 0, col: 0 });
    expect(guides[24]).toMatchObject({ row: 4, col: 4 });
  });

  it("uses the backend-provided legal layer instead of recomputing gravity", () => {
    const move: Move = { action: 87, layer: 3, row: 2, col: 2 };
    expect(findLegalMoveForColumn([move], 2, 2)).toBe(move);
    expect(findLegalMoveForColumn([move], 1, 2)).toBeNull();
  });

  it("keeps the other side as an outline and never hides winning pieces", () => {
    expect(pieceRenderMode(1, "all", false)).toBe("solid");
    expect(pieceRenderMode(1, "red", false)).toBe("solid");
    expect(pieceRenderMode(-1, "red", false)).toBe("outline");
    expect(pieceRenderMode(1, "blue", false)).toBe("outline");
    expect(pieceRenderMode(-1, "blue", false)).toBe("solid");
    expect(pieceRenderMode(-1, "red", true)).toBe("winning");
  });

  it("accepts a click but rejects a camera drag release", () => {
    expect(isIntentionalBoardClick(0)).toBe(true);
    expect(isIntentionalBoardClick(4)).toBe(true);
    expect(isIntentionalBoardClick(4.01)).toBe(false);
    expect(isIntentionalBoardClick(Number.NaN)).toBe(false);
  });

  it("extracts column, row, and layer slices in their face-on display order", () => {
    const board = Array.from({ length: 6 }, (_, layer) => (
      Array.from({ length: 5 }, (_, row) => Array.from({ length: 5 }, (_, col) => layer * 100 + row * 10 + col))
    ));

    const column = sliceCells(board, { axis: "col", index: 2 });
    expect(column).toHaveLength(30);
    expect(column[0]).toMatchObject({ layer: 5, row: 0, col: 2, value: 502 });
    expect(column[29]).toMatchObject({ layer: 0, row: 4, col: 2, value: 42 });

    const row = sliceCells(board, { axis: "row", index: 3 });
    expect(row[0]).toMatchObject({ layer: 5, row: 3, col: 0, value: 530 });
    expect(row[29]).toMatchObject({ layer: 0, row: 3, col: 4, value: 34 });

    const layer = sliceCells(board, { axis: "layer", index: 4 });
    expect(layer).toHaveLength(25);
    expect(layer[0]).toMatchObject({ layer: 4, row: 0, col: 0, value: 400 });
    expect(layer[24]).toMatchObject({ layer: 4, row: 4, col: 4, value: 444 });
  });

  it("labels and filters slices with zero-based board indices", () => {
    expect(sliceSelectionLabel({ axis: "col", index: 2 })).toBe("C3");
    expect(sliceSelectionLabel({ axis: "row", index: 0 })).toBe("R1");
    expect(sliceSelectionLabel({ axis: "layer", index: 5 })).toBe("F6");
    expect(moveMatchesSlice({ layer: 2, row: 4, col: 1 }, { axis: "layer", index: 2 })).toBe(true);
    expect(moveMatchesSlice({ layer: 2, row: 4, col: 1 }, { axis: "col", index: 2 })).toBe(false);
    expect(() => sliceSelectionLabel({ axis: "layer", index: 6 })).toThrow(RangeError);
  });
});
