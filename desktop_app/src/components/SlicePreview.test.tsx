import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { translations } from "../i18n";
import { emptyState } from "../test/fixtures";
import type { GameState, Move } from "../types";
import { SlicePreview } from "./SlicePreview";

function stateWithPieces(): GameState {
  const state = emptyState();
  state.board[0][0][2] = 1;
  state.board[1][1][2] = -1;
  state.board[5][4][2] = 1;
  state.last_move = { action: 7, layer: 1, row: 1, col: 2, player: -1 };
  state.winning_line = [{ action: 2, layer: 0, row: 0, col: 2, player: 1 }];
  state.legal_moves = [{ action: 12, layer: 2, row: 2, col: 2 }];
  return state;
}

it("renders a six-by-five vertical slice with focus, last move, win, hint, and legal state", () => {
  const hint: { for_revision: number; move: Move; value: number } = {
    for_revision: 1,
    move: { action: 12, layer: 2, row: 2, col: 2 },
    value: 0.4,
  };
  const { container } = render(
    <SlicePreview
      copy={translations.zh}
      state={stateWithPieces()}
      hint={hint}
      pieceFocus="red"
      selection={{ axis: "col", index: 2 }}
      onClear={() => undefined}
    />,
  );

  expect(screen.getByLabelText(`${translations.zh.game.view3d.selectedSlice}: C3`)).toBeVisible();
  expect(screen.getAllByRole("gridcell")).toHaveLength(30);
  expect(container.querySelector(".slice-piece.blue.outline")).toBeInTheDocument();
  expect(container.querySelector(".slice-cell.last")).toBeInTheDocument();
  expect(container.querySelector(".slice-cell.winning")).toBeInTheDocument();
  expect(container.querySelector(".slice-cell.hint.legal")).toBeInTheDocument();
});

it("renders a five-by-five floor slice and exposes a keyboard-safe clear action", async () => {
  const user = userEvent.setup();
  const onClear = vi.fn();
  render(
    <SlicePreview
      copy={translations.en}
      state={stateWithPieces()}
      hint={null}
      pieceFocus="all"
      selection={{ axis: "layer", index: 5 }}
      onClear={onClear}
    />,
  );

  expect(screen.getByLabelText(`${translations.en.game.view3d.selectedSlice}: F6`)).toBeVisible();
  expect(screen.getAllByRole("gridcell")).toHaveLength(25);
  await user.click(screen.getByRole("button", { name: translations.en.game.view3d.clearSlice }));
  expect(onClear).toHaveBeenCalledOnce();
});
