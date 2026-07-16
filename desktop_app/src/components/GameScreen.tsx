import { lazy, Suspense, useMemo, useState } from "react";

import type { Copy } from "../i18n";
import type {
  BoardViewMode,
  CameraCommand,
  CameraPreset,
  GameState,
  HintResult,
  LayerSpacing,
  Move,
  PieceFocus,
  SliceSelection,
  WinRateResult,
} from "../types";
import { ObservationDrawer } from "./ObservationDrawer";

const Board3D = lazy(async () => {
  const module = await import("./Board3D");
  return { default: module.Board3D };
});

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
}

function WinRateCard({ copy: t, result }: { copy: Copy; result: WinRateResult | null }) {
  if (!result) return null;
  return (
    <aside className="win-rate-card" aria-label={t.game.winRate}>
      <div className="rate-labels">
        <strong className="red-text">{t.game.redRate} {(result.red * 100).toFixed(1)}%</strong>
        <strong className="blue-text">{t.game.blueRate} {(result.blue * 100).toFixed(1)}%</strong>
      </div>
      <div className="rate-track">
        <div className="red-rate" style={{ width: `${result.red * 100}%` }} />
        <div className="blue-rate" style={{ width: `${result.blue * 100}%` }} />
        <i style={{ left: `${result.red * 100}%` }} />
      </div>
    </aside>
  );
}

export function GameScreen(props: Props) {
  const { copy: t, state } = props;
  const [viewMode, setViewMode] = useState<BoardViewMode>("2d");
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [pieceFocus, setPieceFocus] = useState<PieceFocus>("all");
  const [showColumnGuides, setShowColumnGuides] = useState(true);
  const [slicePickerEnabled, setSlicePickerEnabled] = useState(false);
  const [sliceSelection, setSliceSelection] = useState<SliceSelection | null>(null);
  const [layerSpacing, setLayerSpacing] = useState<LayerSpacing>("standard");
  const [cameraCommand, setCameraCommand] = useState<CameraCommand>({ preset: "isometric", serial: 0 });
  const finished = state.status !== "playing";
  const aiTurn = state.mode === "pvai" && state.current_player !== state.human_player;
  const moveLocked = props.mutationBusy || props.combatThinking || finished || aiTurn;
  const undoDisabled = props.mutationBusy || props.combatThinking || !state.can_undo;
  const statusText = finished
    ? state.winner === 1 ? t.game.redWins : state.winner === -1 ? t.game.blueWins : t.game.draw
    : state.current_player === 1 ? t.game.red : t.game.blue;
  const statusClass = state.winner === -1 || (!finished && state.current_player === -1) ? "blue-text" : state.winner === 0 ? "neutral-text" : "red-text";
  const thinkingText = props.combatThinking ? t.game.aiThinking : props.hintThinking ? t.game.hintThinking : props.winRateThinking ? t.game.winRateThinking : "";
  const selectCamera = (preset: CameraPreset) => {
    setCameraCommand((current) => ({ preset, serial: current.serial + 1 }));
  };
  const setSlicePicker = (enabled: boolean) => {
    setSlicePickerEnabled(enabled);
    if (!enabled) setSliceSelection(null);
  };

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

      <section className={`game-workspace ${viewMode === "3d" ? "three-d-workspace" : ""}`}>
        <div className={`board-view-layout ${viewMode === "3d" ? `three-d ${drawerOpen ? "drawer-open" : "drawer-closed"}` : "two-d"}`}>
          <div className={`board-canvas-pane ${sliceSelection ? "slice-active" : ""} ${props.winRate ? "has-win-rate" : ""}`}>
            {viewMode === "2d" ? (
              <div className="board-area">
                <LayerBoards copy={t} state={state} hint={props.hint} locked={moveLocked} onMove={props.onMove} />
                <div className="board-legend">
                  <span><i className="legend-cell legal" />{t.game.legal}</span>
                  <span><i className="legend-cell illegal" />{t.game.illegal}</span>
                  <span><i className="legend-piece red" />{t.game.red}</span>
                  <span><i className="legend-piece blue" />{t.game.blue}</span>
                </div>
              </div>
            ) : (
              <Suspense fallback={<div className="board-3d-loading">{t.game.view3d.loading}</div>}>
                <Board3D
                  copy={t}
                  state={state}
                  hint={props.hint}
                  moveLocked={moveLocked}
                  pieceFocus={pieceFocus}
                  showColumnGuides={showColumnGuides}
                  slicePickerEnabled={slicePickerEnabled}
                  sliceSelection={sliceSelection}
                  layerSpacing={layerSpacing}
                  cameraCommand={cameraCommand}
                  onMove={props.onMove}
                  onSliceSelection={setSliceSelection}
                  onClearSlice={() => setSliceSelection(null)}
                  onFallbackTo2d={() => setViewMode("2d")}
                />
              </Suspense>
            )}
            <WinRateCard copy={t} result={props.winRate} />
          </div>
          {viewMode === "3d" && (
            <ObservationDrawer
              copy={t}
              open={drawerOpen}
              pieceFocus={pieceFocus}
              showColumnGuides={showColumnGuides}
              slicePickerEnabled={slicePickerEnabled}
              sliceSelection={sliceSelection}
              layerSpacing={layerSpacing}
              onToggleOpen={() => setDrawerOpen((open) => !open)}
              onPieceFocus={setPieceFocus}
              onShowColumnGuides={setShowColumnGuides}
              onSlicePickerEnabled={setSlicePicker}
              onSliceSelection={setSliceSelection}
              onLayerSpacing={setLayerSpacing}
              onCameraPreset={selectCamera}
              onResetCamera={() => selectCamera("isometric")}
            />
          )}
        </div>
      </section>

      <footer className="game-functions">
        <div className="function-group main-functions">
          <button aria-label={t.game.undo} disabled={undoDisabled} onClick={props.onUndo}><span aria-hidden="true">↶</span>{t.game.undo}</button>
          <button aria-label={t.game.restart} disabled={props.mutationBusy || props.combatThinking} onClick={props.onRestart}><span aria-hidden="true">↻</span>{t.game.restart}</button>
          <button aria-label={t.game.hint} className={props.hintThinking ? "working" : ""} disabled={finished || props.hintThinking || props.combatThinking} onClick={props.onHint}><span aria-hidden="true">✦</span>{t.game.hint}</button>
          <button aria-label={t.game.winRate} className={props.winRateThinking ? "working" : ""} disabled={finished || props.winRateThinking || props.combatThinking} onClick={props.onWinRate}><span aria-hidden="true">▰</span>{t.game.winRate}</button>
        </div>
        <div className="function-group edge-functions">
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
