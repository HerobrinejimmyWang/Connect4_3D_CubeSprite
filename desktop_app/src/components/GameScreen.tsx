import { useMemo } from "react";

import type { Copy } from "../i18n";
import type { GameState, HintResult, Move, WinRateResult } from "../types";

function moveKey(move: Pick<Move, "layer" | "row" | "col">): string {
  return `${move.layer}:${move.row}:${move.col}`;
}

interface BoardProps {
  copy: Copy;
  state: GameState;
  hint: HintResult | null;
  locked: boolean;
  onMove: (move: Move) => void;
}

function LayerBoards({ copy: t, state, hint, locked, onMove }: BoardProps) {
  const legalByCell = useMemo(() => new Map(state.legal_moves.map((move) => [moveKey(move), move])), [state.legal_moves]);
  const winning = useMemo(() => new Set(state.winning_line.map(moveKey)), [state.winning_line]);
  const last = state.last_move ? moveKey(state.last_move) : "";
  const hintCell = hint ? moveKey(hint.move) : "";

  return (
    <div className={`layer-grid ${locked ? "board-locked" : ""}`} aria-label="6 × 5 × 5 board">
      {state.board.map((layer, layerIndex) => (
        <section className="layer-board" key={layerIndex} aria-label={`${t.game.floor}${layerIndex + 1}`}>
          <header>
            <strong>{t.game.floor}{layerIndex + 1}</strong>
            <span>{layerIndex + 1}/6</span>
          </header>
          <div className="cell-grid">
            {layer.map((row, rowIndex) => row.map((value, colIndex) => {
              const key = `${layerIndex}:${rowIndex}:${colIndex}`;
              const legalMove = legalByCell.get(key);
              const isEmpty = value === 0;
              const isWinning = winning.has(key);
              const className = [
                "board-cell",
                isEmpty ? "empty" : "occupied",
                legalMove ? "legal" : "illegal",
                key === last ? "last-move" : "",
                isWinning ? "winning" : "",
                key === hintCell ? "hint-cell" : "",
              ].filter(Boolean).join(" ");
              const cellState = value === 1 ? t.game.red : value === -1 ? t.game.blue : legalMove ? t.game.legal : t.game.illegal;
              return (
                <button
                  className={className}
                  key={key}
                  disabled={locked || !legalMove}
                  onClick={() => legalMove && onMove(legalMove)}
                  aria-label={`${t.game.floor}${layerIndex + 1}, ${rowIndex + 1}, ${colIndex + 1}: ${cellState}`}
                  title={cellState}
                >
                  {!isEmpty && <span className={`game-piece ${value === 1 ? "red" : "blue"} ${isWinning ? "winner" : ""}`} />}
                  {key === hintCell && isEmpty && <span className="hint-pulse" />}
                </button>
              );
            }))}
          </div>
        </section>
      ))}
    </div>
  );
}

interface Props {
  copy: Copy;
  state: GameState;
  combatThinking: boolean;
  hintThinking: boolean;
  winRateThinking: boolean;
  mutationBusy: boolean;
  hint: HintResult | null;
  hintPreloaded: boolean;
  winRate: WinRateResult | null;
  onMove: (move: Move) => void;
  onUndo: () => void;
  onRestart: () => void;
  onHint: () => void;
  onWinRate: () => void;
  onExit: () => void;
  onSwitch3d: () => void;
}

export function GameScreen(props: Props) {
  const { copy: t, state } = props;
  const finished = state.status !== "playing";
  const aiTurn = state.mode === "pvai" && state.current_player !== state.human_player;
  const boardLocked = props.mutationBusy || props.combatThinking || finished || aiTurn;
  const undoDisabled = props.mutationBusy || props.combatThinking || !state.can_undo;
  const statusText = finished
    ? state.winner === 1 ? t.game.redWins : state.winner === -1 ? t.game.blueWins : t.game.draw
    : state.current_player === 1 ? t.game.red : t.game.blue;
  const statusClass = state.winner === -1 || (!finished && state.current_player === -1) ? "blue-text" : state.winner === 0 ? "neutral-text" : "red-text";
  const thinkingText = props.combatThinking ? t.game.aiThinking : props.hintThinking ? t.game.hintThinking : props.winRateThinking ? t.game.winRateThinking : "";

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
          {props.hintPreloaded && !thinkingText && state.status === "playing" && <div className="preload-ready">✓ {t.game.preloadReady}</div>}
        </div>
      </header>

      <section className="game-workspace">
        <div className="board-area">
          <LayerBoards copy={t} state={state} hint={props.hint} locked={boardLocked} onMove={props.onMove} />
          <div className="board-legend">
            <span><i className="legend-cell legal" />{t.game.legal}</span>
            <span><i className="legend-cell illegal" />{t.game.illegal}</span>
            <span><i className="legend-piece red" />{t.game.red}</span>
            <span><i className="legend-piece blue" />{t.game.blue}</span>
          </div>
        </div>

        {props.winRate && (
          <aside className="win-rate-card" aria-label={t.game.winRate}>
            <div className="rate-labels">
              <strong className="red-text">{t.game.redRate} {(props.winRate.red * 100).toFixed(1)}%</strong>
              <strong className="blue-text">{t.game.blueRate} {(props.winRate.blue * 100).toFixed(1)}%</strong>
            </div>
            <div className="rate-track">
              <div className="red-rate" style={{ width: `${props.winRate.red * 100}%` }} />
              <div className="blue-rate" style={{ width: `${props.winRate.blue * 100}%` }} />
              <i style={{ left: `${props.winRate.red * 100}%` }} />
            </div>
          </aside>
        )}
      </section>

      <footer className="game-functions">
        <div className="function-group main-functions">
          <button aria-label={t.game.undo} disabled={undoDisabled} onClick={props.onUndo}><span aria-hidden="true">↶</span>{t.game.undo}</button>
          <button aria-label={t.game.restart} disabled={props.mutationBusy || props.combatThinking} onClick={props.onRestart}><span aria-hidden="true">↻</span>{t.game.restart}</button>
          <button aria-label={t.game.hint} className={props.hintThinking ? "working" : ""} disabled={finished || props.hintThinking || props.combatThinking} onClick={props.onHint}><span aria-hidden="true">✦</span>{t.game.hint}</button>
          <button aria-label={t.game.winRate} className={props.winRateThinking ? "working" : ""} disabled={finished || props.winRateThinking || props.combatThinking} onClick={props.onWinRate}><span aria-hidden="true">▰</span>{t.game.winRate}</button>
        </div>
        <div className="function-group edge-functions">
          <button aria-label={t.game.switch3d} className="placeholder-button" onClick={props.onSwitch3d}><span aria-hidden="true">◇</span>{t.game.switch3d}<small>{t.common.soon}</small></button>
          <button aria-label={t.game.exit} className="exit-button" onClick={props.onExit}><span aria-hidden="true">⌂</span>{t.game.exit}</button>
        </div>
      </footer>
    </main>
  );
}
