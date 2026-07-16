import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { translations } from "../i18n";
import { emptyState } from "../test/fixtures";
import type { CameraCommand, GameState, LayerSpacing, Move, PieceFocus, SliceSelection } from "../types";
import { GameScreen } from "./GameScreen";

interface CanvasStubProps {
  state: GameState;
  moveLocked: boolean;
  pieceFocus: PieceFocus;
  showColumnGuides: boolean;
  slicePickerEnabled: boolean;
  sliceSelection: SliceSelection | null;
  layerSpacing: LayerSpacing;
  cameraCommand: CameraCommand;
  onMove: (move: Move) => void;
  onSliceSelection: (selection: SliceSelection) => void;
}

vi.mock("./Board3DCanvas", () => ({
  Board3DCanvas: (props: CanvasStubProps) => (
    <div
      data-testid="board-3d-canvas"
      data-focus={props.pieceFocus}
      data-guides={String(props.showColumnGuides)}
      data-slice-picker={String(props.slicePickerEnabled)}
      data-slice={props.sliceSelection ? `${props.sliceSelection.axis}:${props.sliceSelection.index}` : "none"}
      data-spacing={props.layerSpacing}
      data-camera={props.cameraCommand.preset}
    >
      <button
        type="button"
        aria-label="3D legal column"
        disabled={props.moveLocked}
        onClick={() => props.state.legal_moves[0] && props.onMove(props.state.legal_moves[0])}
      />
    </div>
  ),
}));

const noop = () => undefined;

function renderGame(overrides: Partial<React.ComponentProps<typeof GameScreen>> = {}) {
  const props: React.ComponentProps<typeof GameScreen> = {
    copy: translations.zh,
    state: emptyState(),
    combatThinking: false,
    hintThinking: false,
    winRateThinking: false,
    mutationBusy: false,
    hint: null,
    hintPreloaded: false,
    winRate: null,
    onMove: noop,
    onUndo: noop,
    onRestart: noop,
    onHint: noop,
    onWinRate: noop,
    onExit: noop,
    ...overrides,
  };
  return render(<GameScreen {...props} />);
}

it("trusts backend can_undo after a PvAI human move even before the AI responds", () => {
  renderGame({
    state: emptyState({ mode: "pvai", human_player: 1, current_player: -1, move_count: 1, can_undo: true }),
  });

  expect(screen.getByRole("button", { name: translations.zh.game.undo })).toBeEnabled();
});

it("switches between the same 2D and 3D game state without a coming-soon placeholder", async () => {
  const user = userEvent.setup();
  renderGame();

  expect(screen.getByLabelText("6 × 5 × 5 board")).toBeVisible();
  await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));
  expect(screen.queryByLabelText("6 × 5 × 5 board")).not.toBeInTheDocument();
  expect(await screen.findByTestId("board-3d-canvas")).toBeVisible();
  expect(screen.getByRole("button", { name: translations.zh.game.switch2d })).toBeVisible();

  await user.click(screen.getByRole("button", { name: translations.zh.game.switch2d }));
  expect(screen.getByLabelText("6 × 5 × 5 board")).toBeVisible();
});

it("keeps all observation controls live and passes them to the 3D scene", async () => {
  const user = userEvent.setup();
  renderGame();
  await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));
  const scene = await screen.findByTestId("board-3d-canvas");

  await user.click(screen.getByRole("radio", { name: translations.zh.game.view3d.focusRed }));
  expect(scene).toHaveAttribute("data-focus", "red");

  await user.click(screen.getByRole("switch", { name: translations.zh.game.view3d.showColumnGuides }));
  expect(scene).toHaveAttribute("data-guides", "false");

  await user.click(screen.getByRole("radio", { name: translations.zh.game.view3d.expandedSpacing }));
  expect(scene).toHaveAttribute("data-spacing", "expanded");

  await user.click(screen.getByRole("button", { name: translations.zh.game.view3d.top }));
  expect(scene).toHaveAttribute("data-camera", "top");
});

it("locks only 3D move submission while AI thinking and keeps observation controls enabled", async () => {
  const user = userEvent.setup();
  renderGame({ combatThinking: true });
  await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));

  expect(await screen.findByRole("button", { name: "3D legal column" })).toBeDisabled();
  expect(screen.getByRole("radio", { name: translations.zh.game.view3d.focusBlue })).toBeEnabled();
  expect(screen.getByRole("switch", { name: translations.zh.game.view3d.showColumnGuides })).toBeEnabled();
});

it("keeps slice labels protected until enabled and clears the selection when disabled", async () => {
  const user = userEvent.setup();
  const onMove = vi.fn();
  renderGame({ onMove });
  await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));
  const scene = await screen.findByTestId("board-3d-canvas");
  const sliceSwitch = screen.getByRole("switch", { name: translations.zh.game.view3d.enableSliceSelection });

  expect(sliceSwitch).not.toBeChecked();
  expect(scene).toHaveAttribute("data-slice-picker", "false");
  const edgeLabel = `${translations.zh.game.view3d.boardEdgeSlice}: C3`;
  expect(screen.queryByRole("button", { name: edgeLabel })).not.toBeInTheDocument();

  await user.click(sliceSwitch);
  expect(scene).toHaveAttribute("data-slice-picker", "true");
  await user.click(screen.getByRole("button", { name: edgeLabel }));
  expect(scene).toHaveAttribute("data-slice", "col:2");
  expect(screen.getByLabelText(`${translations.zh.game.view3d.selectedSlice}: C3`)).toBeVisible();
  expect(onMove).not.toHaveBeenCalled();

  await user.click(sliceSwitch);
  expect(scene).toHaveAttribute("data-slice-picker", "false");
  expect(scene).toHaveAttribute("data-slice", "none");
  expect(screen.queryByLabelText(`${translations.zh.game.view3d.selectedSlice}: C3`)).not.toBeInTheDocument();
});

it("selects C, R, and F slices from accessible drawer buttons independently of column guides", async () => {
  const user = userEvent.setup();
  renderGame();
  await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));
  const scene = await screen.findByTestId("board-3d-canvas");

  await user.click(screen.getByRole("switch", { name: translations.zh.game.view3d.showColumnGuides }));
  await user.click(screen.getByRole("switch", { name: translations.zh.game.view3d.enableSliceSelection }));
  await user.click(screen.getByRole("button", { name: `${translations.zh.game.view3d.selectColumnSlice} C3` }));
  expect(scene).toHaveAttribute("data-guides", "false");
  expect(scene).toHaveAttribute("data-slice", "col:2");

  await user.click(screen.getByRole("button", { name: `${translations.zh.game.view3d.selectRowSlice} R1` }));
  expect(scene).toHaveAttribute("data-slice", "row:0");

  await user.click(screen.getByRole("button", { name: `${translations.zh.game.view3d.selectLayerSlice} F6` }));
  expect(scene).toHaveAttribute("data-slice", "layer:5");
  expect(screen.getByRole("button", { name: `${translations.zh.game.view3d.selectLayerSlice} F6` })).toHaveAttribute("aria-pressed", "true");
});

it("submits the exact backend-provided legal move selected in 3D", async () => {
  const user = userEvent.setup();
  const move: Move = { action: 87, layer: 3, row: 2, col: 2 };
  const onMove = vi.fn();
  renderGame({ state: emptyState({ legal_moves: [move] }), onMove });
  await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));
  await user.click(await screen.findByRole("button", { name: "3D legal column" }));

  expect(onMove).toHaveBeenCalledWith(move);
});
