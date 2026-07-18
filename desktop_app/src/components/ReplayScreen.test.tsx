import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { translations } from "../i18n";
import { replayAnalysis, replayOpen } from "../test/fixtures";
import type { GameState, Move } from "../types";
import { ReplayScreen } from "./ReplayScreen";

const noop = () => undefined;

vi.mock("./Board3DCanvas", () => ({
  Board3DCanvas: ({ state, moveLocked, onMove }: { state: GameState; moveLocked: boolean; onMove: (move: Move) => void }) => (
    <button
      aria-label="Replay 3D legal column"
      disabled={moveLocked}
      onClick={() => state.legal_moves[0] && onMove(state.legal_moves[0])}
    />
  ),
}));

function renderReplay(overrides: Partial<React.ComponentProps<typeof ReplayScreen>> = {}) {
  const props: React.ComponentProps<typeof ReplayScreen> = {
    copy: translations.zh,
    replay: replayOpen(),
    autoplayIntervalMs: 1000,
    analysisThinking: false,
    continueBusy: false,
    onAnalyze: noop,
    onContinue: noop,
    onExit: noop,
    ...overrides,
  };
  return render(<ReplayScreen {...props} />);
}

afterEach(() => {
  vi.useRealTimers();
});

describe("ReplayScreen", () => {
  it("plays the authoritative frames from step zero and keeps the board read-only", async () => {
    const user = userEvent.setup();
    const onContinue = vi.fn();
    renderReplay({ onContinue });

    const progress = screen.getByText(translations.zh.replay.progress).parentElement!;
    expect(within(progress).getByText("0")).toBeVisible();
    expect(within(progress).getByText("/2")).toBeVisible();
    expect(screen.getByRole("button", { name: translations.zh.replay.previous })).toBeDisabled();
    expect(screen.getByRole("button", { name: "F1, 1, 1: 合法落点" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: translations.zh.replay.next }));
    expect(within(progress).getByText("1")).toBeVisible();
    expect(screen.getByRole("button", { name: "F1, 1, 1: 红方" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: translations.zh.replay.fromStart }));
    expect(within(progress).getByText("0")).toBeVisible();
  });

  it("keeps the shared 3D board read-only while observation remains available", async () => {
    const user = userEvent.setup();
    renderReplay();

    await user.click(screen.getByRole("button", { name: translations.zh.game.switch3d }));
    expect(await screen.findByRole("button", { name: "Replay 3D legal column" })).toBeDisabled();
    expect(screen.getByRole("button", { name: translations.zh.game.switch2d })).toBeVisible();
    expect(screen.getByRole("radio", { name: translations.zh.game.view3d.focusRed })).toBeEnabled();
  });

  it("autoplays at the configured interval and stops at the saved limit", () => {
    vi.useFakeTimers();
    renderReplay({ autoplayIntervalMs: 250 });
    const progress = screen.getByText(translations.zh.replay.progress).parentElement!;

    fireEvent.click(screen.getByRole("button", { name: translations.zh.replay.autoplay }));
    act(() => vi.advanceTimersByTime(250));
    expect(within(progress).getByText("1")).toBeVisible();
    act(() => vi.advanceTimersByTime(250));
    expect(within(progress).getByText("2")).toBeVisible();
    act(() => vi.runOnlyPendingTimers());
    expect(screen.getByRole("button", { name: translations.zh.replay.autoplay })).toBeDisabled();
  });

  it("offers all three continuation modes at the current non-terminal step", async () => {
    const user = userEvent.setup();
    const onContinue = vi.fn();
    renderReplay({ onContinue });

    await user.click(screen.getByRole("button", { name: translations.zh.replay.next }));
    await user.click(screen.getByRole("button", { name: translations.zh.replay.continueFromHere }));
    expect(screen.getByRole("dialog", { name: translations.zh.replay.continueTitle })).toBeVisible();
    expect(screen.getByRole("button", { name: /玩家 vs 玩家/ })).toBeVisible();
    expect(screen.getByRole("button", { name: /执红对抗 AI/ })).toBeVisible();
    expect(screen.getByRole("button", { name: /执蓝对抗 AI/ })).toBeVisible();

    await user.click(screen.getByRole("button", { name: /执红对抗 AI/ }));
    expect(onContinue).toHaveBeenCalledWith(1, "pvai", 1);
  });

  it("shows the no-data monitor and delegates global analysis", async () => {
    const user = userEvent.setup();
    const onAnalyze = vi.fn();
    renderReplay({ onAnalyze });

    await user.click(screen.getByRole("button", { name: translations.zh.replay.monitor }));
    expect(screen.getByText(translations.zh.replay.noAnalysis)).toBeVisible();
    expect(screen.getByText(translations.zh.replay.noAnalysisDetail)).toBeVisible();
    await user.click(screen.getByRole("button", { name: translations.zh.replay.calculate }));
    expect(onAnalyze).toHaveBeenCalledOnce();
  });

  it("draws cached red/blue curves and analysis metadata", () => {
    renderReplay({ replay: replayOpen({ analysis: replayAnalysis() }) });

    expect(screen.getByRole("img", { name: translations.zh.replay.chartLabel })).toBeVisible();
    expect(document.querySelector("polyline.red-line")).toHaveAttribute("points");
    expect(document.querySelector("polyline.blue-line")).toHaveAttribute("points");
    expect(screen.getByText("v2.2_balance")).toBeVisible();
    expect(screen.getByText(/bb8cc0c60422/)).toBeVisible();
    expect(screen.getByText(/128 MCTS/)).toBeVisible();
    expect(screen.getByText("分析耗时")).toBeVisible();
    expect(screen.getByText("2.0 s")).toBeVisible();
  });
});
