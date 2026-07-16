import {
  BOARD_SIZE,
  MAX_LAYERS,
  boardToWorld,
  createColumnGuides,
  findLegalMoveForColumn,
  isIntentionalBoardClick,
  layerGap,
  pieceRenderMode,
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
});
