import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { translations } from "../i18n";
import { emptyState } from "../test/fixtures";
import type { CameraCommand, GameState, LayerSpacing, Move, PieceFocus, SliceSelection } from "../types";
import { GameScreen } from "./GameScreen";

interface CanvasStubProps {
  state: GameState;
  moveLocked: boolean;
  compactLayout: boolean;
  showCoordinateLabels: boolean;
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
      data-compact={String(props.compactLayout)}
      data-focus={props.pieceFocus}
      data-guides={String(props.showColumnGuides)}
      data-coordinate-labels={String(props.showCoordinateLabels)}
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
    saveReplayThinking: false,
    mutationBusy: false,
    hint: null,
    hintPreloaded: false,
    winRate: null,
    onMove: noop,
    onUndo: noop,
    onRestart: noop,
    onHint: noop,
    onWinRate: noop,
    onSaveReplay: noop,
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

it("uses the original four-plus-two grid on desktop without mobile layer controls", () => {
  renderGame();

  const board = screen.getByLabelText("6 × 5 × 5 board");
  expect(board).toHaveClass("desktop-six");
  expect(board).not.toHaveAttribute("data-window-start");
  expect(screen.getByLabelText("F1")).toBeVisible();
  expect(screen.getByLabelText("F6")).toBeVisible();
  expect(screen.queryByRole("radiogroup", { name: translations.zh.game.view2d.layout })).not.toBeInTheDocument();
});

it("defaults to consecutive four-layer windows and keeps the original layer index when moving", async () => {
  const user = userEvent.setup();
  const move: Move = { action: 112, layer: 4, row: 2, col: 2 };
  const onMove = vi.fn();
  renderGame({ state: emptyState({ legal_moves: [move] }), mobileLayout: true, onMove });

  const board = screen.getByLabelText("6 × 5 × 5 board");
  expect(board).toHaveClass("sliding-four");
  expect(board).toHaveAttribute("data-window-start", "0");
  expect(screen.getByLabelText("F1")).toBeVisible();
  expect(screen.getByLabelText("F4")).toBeVisible();
  expect(screen.queryByLabelText("F5")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: translations.zh.game.view2d.nextWindow }));
  expect(board).toHaveAttribute("data-window-start", "1");
  expect(screen.queryByLabelText("F1")).not.toBeInTheDocument();
  expect(screen.getByLabelText("F2")).toBeVisible();
  expect(screen.getByLabelText("F5")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "F5, 3, 3: 合法落点" }));
  expect(onMove).toHaveBeenCalledWith(move);

  await user.click(screen.getByRole("button", { name: translations.zh.game.view2d.nextWindow }));
  expect(board).toHaveAttribute("data-window-start", "2");
  expect(screen.getByLabelText("F3")).toBeVisible();
  expect(screen.getByLabelText("F6")).toBeVisible();
  expect(screen.getByRole("button", { name: translations.zh.game.view2d.nextWindow })).toBeDisabled();

  const slidingMode = screen.getByRole("radio", { name: translations.zh.game.view2d.slidingFour });
  const allSixMode = screen.getByRole("radio", { name: translations.zh.game.view2d.allSix });
  expect(slidingMode).toBeChecked();
  slidingMode.focus();
  await user.keyboard("{ArrowRight}");
  expect(allSixMode).toBeChecked();
  expect(allSixMode).toHaveFocus();
  expect(board).toHaveClass("all-six");
  expect(screen.getByLabelText("F1")).toBeVisible();
  expect(screen.getByLabelText("F6")).toBeVisible();
  expect(screen.queryByRole("button", { name: translations.zh.game.view2d.nextWindow })).not.toBeInTheDocument();
});

it("swipes by one layer without submitting the cell under the released pointer", () => {
  const move: Move = { action: 27, layer: 1, row: 0, col: 2 };
  const onMove = vi.fn();
  renderGame({ state: emptyState({ legal_moves: [move] }), mobileLayout: true, onMove });

  const board = screen.getByLabelText("6 × 5 × 5 board");
  const pointerEvent = (type: string, clientX: number, clientY: number) => {
    const event = new MouseEvent(type, { bubbles: true, button: 0, clientX, clientY });
    Object.defineProperty(event, "pointerId", { value: 7 });
    return event;
  };
  fireEvent(board, pointerEvent("pointerdown", 620, 120));
  fireEvent(board, pointerEvent("pointerup", 480, 124));

  expect(board).toHaveAttribute("data-window-start", "1");
  const releasedCell = screen.getByRole("button", { name: "F2, 1, 3: 合法落点" });
  fireEvent.click(releasedCell);
  expect(onMove).not.toHaveBeenCalled();

  fireEvent.click(releasedCell);
  expect(onMove).toHaveBeenCalledOnce();
  expect(onMove).toHaveBeenCalledWith(move);
});

it("follows a winning line before an out-of-window last move or hint", async () => {
  const winningLine: Move[] = [1, 2, 3, 4].map((layer) => ({
    action: layer * 25,
    layer,
    row: 0,
    col: 0,
    player: 1,
  }));
  renderGame({
    state: emptyState({
      revision: 9,
      winning_line: winningLine,
      last_move: { action: 125, layer: 5, row: 0, col: 0, player: 1 },
    }),
    mobileLayout: true,
    hint: { for_revision: 9, move: { action: 0, layer: 0, row: 0, col: 0 }, value: 0.2 },
  });

  const board = screen.getByLabelText("6 × 5 × 5 board");
  await waitFor(() => expect(board).toHaveAttribute("data-window-start", "1"));
  expect(screen.getByLabelText("F2")).toBeVisible();
  expect(screen.getByLabelText("F5")).toBeVisible();
  expect(screen.queryByLabelText("F6")).not.toBeInTheDocument();
});

it("follows an out-of-window hint when higher-priority markers are already visible", async () => {
  renderGame({
    state: emptyState({
      revision: 4,
      last_move: { action: 0, layer: 0, row: 0, col: 0, player: 1 },
    }),
    mobileLayout: true,
    hint: { for_revision: 4, move: { action: 125, layer: 5, row: 0, col: 0 }, value: 0.1 },
  });

  const board = screen.getByLabelText("6 × 5 × 5 board");
  await waitFor(() => expect(board).toHaveAttribute("data-window-start", "2"));
  expect(screen.getByLabelText("F6")).toBeVisible();
});

it("follows a new last move once without fighting later manual window navigation", async () => {
  const user = userEvent.setup();
  renderGame({
    state: emptyState({
      revision: 3,
      last_move: { action: 125, layer: 5, row: 0, col: 0, player: -1 },
    }),
    mobileLayout: true,
    hint: { for_revision: 3, move: { action: 0, layer: 0, row: 0, col: 0 }, value: -0.1 },
  });

  const board = screen.getByLabelText("6 × 5 × 5 board");
  await waitFor(() => expect(board).toHaveAttribute("data-window-start", "2"));
  await user.click(screen.getByRole("button", { name: translations.zh.game.view2d.previousWindow }));
  expect(board).toHaveAttribute("data-window-start", "1");
  await waitFor(() => expect(board).toHaveAttribute("data-window-start", "1"));
});

it("lays out the 2D win-rate card after the board instead of over its cells", () => {
  renderGame({ winRate: { for_revision: 1, red: 0.6, blue: 0.4, estimate: "model_mcts" } });

  const board = screen.getByLabelText("6 × 5 × 5 board");
  const winRate = screen.getByRole("complementary", { name: translations.zh.game.winRate });
  expect(winRate.parentElement).toHaveClass("board-area");
  expect(board.nextElementSibling).toBe(winRate);
});

it("starts the compact landscape 3D view with a clear board and collapsed tools", async () => {
  const originalMatchMedia = window.matchMedia;
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query === "(orientation: landscape) and (max-height: 600px)",
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));

  try {
    const user = userEvent.setup();
    renderGame({ mobileLayout: true });
    await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));

    const scene = await screen.findByTestId("board-3d-canvas");
    expect(scene).toHaveAttribute("data-compact", "true");
    expect(scene).toHaveAttribute("data-guides", "false");
    expect(scene).toHaveAttribute("data-coordinate-labels", "false");
    const layout = scene.closest(".board-view-layout");
    expect(layout).toHaveClass("drawer-closed");
    const openTools = screen.getByRole("button", { name: translations.zh.game.view3d.openTools });
    expect(openTools).toHaveAttribute("aria-expanded", "false");
    await user.click(openTools);
    expect(layout).toHaveClass("drawer-open");
    expect(screen.getByRole("button", { name: translations.zh.game.view3d.closeTools })).toHaveAttribute("aria-expanded", "true");
  } finally {
    window.matchMedia = originalMatchMedia;
  }
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

it("places save replay immediately before the view toggle and delegates the snapshot action", async () => {
  const user = userEvent.setup();
  const onSaveReplay = vi.fn();
  renderGame({ onSaveReplay });

  const save = screen.getByRole("button", { name: translations.zh.game.saveReplay });
  const switch3d = screen.getByRole("button", { name: translations.zh.game.switch3d });
  expect(save.nextElementSibling).toBe(switch3d);
  await user.click(save);
  expect(onSaveReplay).toHaveBeenCalledOnce();
});
