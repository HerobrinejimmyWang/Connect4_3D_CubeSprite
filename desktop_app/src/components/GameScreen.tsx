import { useState } from "react";

import type { Copy } from "../i18n";
import type { BoardViewMode, GameState, HintResult, Move, WinRateResult } from "../types";
import { BoardWorkspace } from "./BoardWorkspace";

interface Props {
  copy: Copy;
  state: GameState;
  combatThinking: boolean;
  hintThinking: boolean;
  winRateThinking: boolean;
  saveReplayThinking: boolean;
  replayEnabled?: boolean;
  mobileLayout?: boolean;
  mutationBusy: boolean;
  hint: HintResult | null;
  hintPreloaded: boolean;
  winRate: WinRateResult | null;
  onMove: (move: Move) => void;
  onUndo: () => void;
  onRestart: () => void;
  onHint: () => void;
  onWinRate: () => void;
  onSaveReplay: () => void;
  onExit: () => void;
}

export function GameScreen(props: Props) {
  const { copy: t, state } = props;
  const [viewMode, setViewMode] = useState<BoardViewMode>("2d");
  const finished = state.status !== "playing";
  const aiTurn = state.mode === "pvai" && state.current_player !== state.human_player;
  const moveLocked = props.mutationBusy || props.combatThinking || finished || aiTurn;
  const undoDisabled = props.mutationBusy || props.combatThinking || !state.can_undo;
  const statusText = finished
    ? state.winner === 1 ? t.game.redWins : state.winner === -1 ? t.game.blueWins : t.game.draw
    : state.current_player === 1 ? t.game.red : t.game.blue;
  const statusClass = state.winner === -1 || (!finished && state.current_player === -1) ? "blue-text" : state.winner === 0 ? "neutral-text" : "red-text";
  const thinkingText = props.combatThinking
    ? t.game.aiThinking
    : props.hintThinking
      ? t.game.hintThinking
      : props.winRateThinking
        ? t.game.winRateThinking
        : props.saveReplayThinking
          ? t.game.saveReplayThinking
          : "";
  const tacticalHintText = props.hint?.kind === "win"
    ? t.game.tacticalWinHint
    : props.hint?.kind === "block"
      ? t.game.tacticalBlockHint
      : "";

  return (
    <main className="game-screen">
      <header className="game-topbar">
        <div className="compact-brand"><span className="mini-cube">◆</span><strong>CubeSprite</strong><small>{state.mode === "pvp" ? "PVP" : "PVAI"}</small></div>
        <div className="game-state-panel">
          <div className="turn-state">
            <small>{finished ? t.game.result : t.game.currentPlayer}</small>
            <strong className={statusClass}><i />{statusText}</strong>
          </div>
          <div className="move-count"><small>{t.game.totalMoves}</small><strong>{state.move_count}<span>/150</span></strong></div>
          <div className={`thinking-state ${thinkingText ? "visible" : ""}`} aria-live="polite">
            {thinkingText && <><span className="thinking-dots"><i /><i /><i /></span>{thinkingText}</>}
          </div>
          {tacticalHintText && !thinkingText && state.status === "playing"
            ? <div className="tactical-hint-ready">✦ {tacticalHintText}</div>
            : props.hintPreloaded && !thinkingText && state.status === "playing"
              ? <div className="preload-ready">✓ {t.game.preloadReady}</div>
              : null}
        </div>
      </header>

      <BoardWorkspace
        copy={t}
        state={state}
        viewMode={viewMode}
        hint={props.hint}
        winRate={props.winRate}
        locked={moveLocked}
        mobileLayout={props.mobileLayout === true}
        onMove={props.onMove}
        onViewModeChange={setViewMode}
      />

      <footer className="game-functions">
        <div className="function-group main-functions">
          <button aria-label={t.game.undo} disabled={undoDisabled} onClick={props.onUndo}><span aria-hidden="true">↶</span>{t.game.undo}</button>
          <button aria-label={t.game.restart} disabled={props.mutationBusy || props.combatThinking} onClick={props.onRestart}><span aria-hidden="true">↻</span>{t.game.restart}</button>
          <button aria-label={t.game.hint} className={props.hintThinking ? "working" : ""} disabled={finished || props.hintThinking || props.combatThinking} onClick={props.onHint}><span aria-hidden="true">✦</span>{t.game.hint}</button>
          <button aria-label={t.game.winRate} className={props.winRateThinking ? "working" : ""} disabled={finished || props.winRateThinking || props.combatThinking} onClick={props.onWinRate}><span aria-hidden="true">▰</span>{t.game.winRate}</button>
        </div>
        <div className="function-group edge-functions">
          {props.replayEnabled !== false && (
            <button
              aria-label={t.game.saveReplay}
              className={props.saveReplayThinking ? "working save-replay-button" : "save-replay-button"}
              disabled={props.saveReplayThinking || props.mutationBusy}
              onClick={props.onSaveReplay}
            >
              <span aria-hidden="true">⇩</span>{t.game.saveReplay}
            </button>
          )}
          <button
            aria-label={viewMode === "2d" ? t.game.switch3d : t.game.switch2d}
            className="view-toggle-button"
            onClick={() => setViewMode((mode) => mode === "2d" ? "3d" : "2d")}
          >
            <span aria-hidden="true">◇</span>{viewMode === "2d" ? t.game.switch3d : t.game.switch2d}
          </button>
          <button aria-label={t.game.exit} className="exit-button" onClick={props.onExit}><span aria-hidden="true">⌂</span>{t.game.exit}</button>
        </div>
      </footer>
    </main>
  );
}
