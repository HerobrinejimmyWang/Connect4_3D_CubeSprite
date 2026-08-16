import { lazy, Suspense, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import type { Copy } from "../i18n";
import type {
  BoardViewMode,
  CameraCommand,
  CameraPreset,
  GameState,
  HintResult,
  LayerSpacing,
  LayerViewMode,
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

const LAYER_WINDOW_SIZE = 4;
const SWIPE_THRESHOLD_PX = 44;
const COMPACT_LANDSCAPE_QUERY = "(orientation: landscape) and (max-height: 600px)";

function isCompactLandscapeViewport(): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia(COMPACT_LANDSCAPE_QUERY).matches;
}

function useCompactLandscapeViewport(): boolean {
  const [compact, setCompact] = useState(isCompactLandscapeViewport);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(COMPACT_LANDSCAPE_QUERY);
    const update = () => setCompact(query.matches);
    update();
    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", update);
      return () => query.removeEventListener("change", update);
    }
    query.addListener(update);
    return () => query.removeListener(update);
  }, []);

  return compact;
}

function moveKey(move: Pick<Move, "layer" | "row" | "col">): string {
  return `${move.layer}:${move.row}:${move.col}`;
}

function nextTrackedWindowStart(
  current: number,
  maxWindowStart: number,
  ranges: ReadonlyArray<readonly [minimum: number, maximum: number]>,
): number {
  for (const [minimum, maximum] of ranges) {
    const alreadyVisible = minimum >= current && maximum < current + LAYER_WINDOW_SIZE;
    if (alreadyVisible) continue;

    const firstContainingWindow = Math.max(0, maximum - LAYER_WINDOW_SIZE + 1);
    const lastContainingWindow = Math.min(maxWindowStart, minimum);
    return Math.max(firstContainingWindow, Math.min(lastContainingWindow, current));
  }
  return current;
}

interface LayerBoardsProps {
  copy: Copy;
  state: GameState;
  hint: HintResult | null;
  locked: boolean;
  onMove?: (move: Move) => void;
  layerViewMode: LayerViewMode;
  windowStart: number;
  onLayerViewModeChange: (mode: LayerViewMode) => void;
  onWindowStartChange: (start: number) => void;
  mobileLayout: boolean;
}

function LayerBoards({
  copy: t,
  state,
  hint,
  locked,
  onMove,
  layerViewMode,
  windowStart,
  onLayerViewModeChange,
  onWindowStartChange,
  mobileLayout,
}: LayerBoardsProps) {
  const legalByCell = useMemo(() => new Map(state.legal_moves.map((move) => [moveKey(move), move])), [state.legal_moves]);
  const winning = useMemo(() => new Set(state.winning_line.map(moveKey)), [state.winning_line]);
  const last = state.last_move ? moveKey(state.last_move) : "";
  const hintCell = hint ? moveKey(hint.move) : "";
  const maxWindowStart = Math.max(0, state.board.length - LAYER_WINDOW_SIZE);
  const safeWindowStart = Math.max(0, Math.min(maxWindowStart, windowStart));
  const visibleLayers = useMemo(
    () => state.board
      .map((layer, layerIndex) => ({ layer, layerIndex }))
      .slice(
        mobileLayout && layerViewMode === "sliding4" ? safeWindowStart : 0,
        mobileLayout && layerViewMode === "sliding4" ? safeWindowStart + LAYER_WINDOW_SIZE : state.board.length,
      ),
    [layerViewMode, mobileLayout, safeWindowStart, state.board],
  );
  const pointerGesture = useRef<{ pointerId: number; startX: number; startY: number } | null>(null);
  const suppressNextClick = useRef(false);
  const windowLabel = `${t.game.view2d.showingLayers} F${safeWindowStart + 1}–F${Math.min(
    state.board.length,
    safeWindowStart + LAYER_WINDOW_SIZE,
  )}`;

  const selectWindow = (start: number) => {
    onWindowStartChange(Math.max(0, Math.min(maxWindowStart, start)));
  };
  const beginSwipe = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!mobileLayout || layerViewMode !== "sliding4" || event.button !== 0) return;
    pointerGesture.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
    };
  };
  const cancelSwipe = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (pointerGesture.current?.pointerId === event.pointerId) pointerGesture.current = null;
  };
  const finishSwipe = (event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = pointerGesture.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    pointerGesture.current = null;
    const deltaX = event.clientX - gesture.startX;
    const deltaY = event.clientY - gesture.startY;
    if (Math.abs(deltaX) < SWIPE_THRESHOLD_PX || Math.abs(deltaX) <= Math.abs(deltaY)) return;

    suppressNextClick.current = true;
    window.setTimeout(() => {
      suppressNextClick.current = false;
    }, 0);
    selectWindow(safeWindowStart + (deltaX < 0 ? 1 : -1));
    event.preventDefault();
  };

  return (
    <>
      {mobileLayout && <div className="layer-layout-toolbar">
        <div className="layer-layout-switch" role="radiogroup" aria-label={t.game.view2d.layout}>
          <label>
            <input
              type="radio"
              name="layer-view-mode"
              value="sliding4"
              checked={layerViewMode === "sliding4"}
              onChange={() => onLayerViewModeChange("sliding4")}
            />
            <span>{t.game.view2d.slidingFour}</span>
          </label>
          <label>
            <input
              type="radio"
              name="layer-view-mode"
              value="all6"
              checked={layerViewMode === "all6"}
              onChange={() => onLayerViewModeChange("all6")}
            />
            <span>{t.game.view2d.allSix}</span>
          </label>
        </div>

        {layerViewMode === "sliding4" && (
          <div className="layer-window-nav">
            <button
              type="button"
              className="layer-window-arrow"
              aria-label={t.game.view2d.previousWindow}
              disabled={safeWindowStart === 0}
              onClick={() => selectWindow(safeWindowStart - 1)}
            >
              ‹
            </button>
            <div className="layer-window-status">
              <strong aria-live="polite">{windowLabel}</strong>
              <div className="layer-window-indicators">
                {Array.from({ length: maxWindowStart + 1 }, (_, start) => (
                  <button
                    type="button"
                    key={start}
                    aria-label={`${t.game.view2d.showingLayers} F${start + 1}–F${start + LAYER_WINDOW_SIZE}`}
                    aria-pressed={safeWindowStart === start}
                    className={safeWindowStart === start ? "active" : ""}
                    onClick={() => selectWindow(start)}
                  />
                ))}
              </div>
              <small>{t.game.view2d.swipeHint}</small>
            </div>
            <button
              type="button"
              className="layer-window-arrow"
              aria-label={t.game.view2d.nextWindow}
              disabled={safeWindowStart === maxWindowStart}
              onClick={() => selectWindow(safeWindowStart + 1)}
            >
              ›
            </button>
          </div>
        )}
      </div>}

      <div
        className={`layer-grid ${mobileLayout ? (layerViewMode === "sliding4" ? "sliding-four" : "all-six") : "desktop-six"} ${locked ? "board-locked" : ""}`}
        aria-label="6 × 5 × 5 board"
        data-window-start={mobileLayout ? safeWindowStart : undefined}
        onPointerDown={beginSwipe}
        onPointerUp={finishSwipe}
        onPointerCancel={cancelSwipe}
        onClickCapture={(event) => {
          if (!suppressNextClick.current) return;
          suppressNextClick.current = false;
          event.preventDefault();
          event.stopPropagation();
        }}
      >
      {visibleLayers.map(({ layer, layerIndex }) => (
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
                  disabled={locked || !legalMove || !onMove}
                  onClick={() => legalMove && onMove?.(legalMove)}
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
    </>
  );
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

interface Props {
  copy: Copy;
  state: GameState;
  viewMode: BoardViewMode;
  hint?: HintResult | null;
  winRate?: WinRateResult | null;
  locked: boolean;
  mobileLayout?: boolean;
  onMove?: (move: Move) => void;
  onViewModeChange: (mode: BoardViewMode) => void;
}

export function BoardWorkspace({
  copy: t,
  state,
  viewMode,
  hint = null,
  winRate = null,
  locked,
  mobileLayout = false,
  onMove,
  onViewModeChange,
}: Props) {
  const compactViewport = useCompactLandscapeViewport();
  const compactLandscape = mobileLayout && compactViewport;
  const [drawerOpen, setDrawerOpen] = useState(() => !compactLandscape);
  const [pieceFocus, setPieceFocus] = useState<PieceFocus>("all");
  const [showColumnGuides, setShowColumnGuides] = useState(() => !compactLandscape);
  const [slicePickerEnabled, setSlicePickerEnabled] = useState(false);
  const [sliceSelection, setSliceSelection] = useState<SliceSelection | null>(null);
  const [layerSpacing, setLayerSpacing] = useState<LayerSpacing>("standard");
  const [layerViewMode, setLayerViewMode] = useState<LayerViewMode>("sliding4");
  const [layerWindowStart, setLayerWindowStart] = useState(0);
  const [cameraCommand, setCameraCommand] = useState<CameraCommand>({ preset: "isometric", serial: 0 });
  const maxLayerWindowStart = Math.max(0, state.board.length - LAYER_WINDOW_SIZE);
  const winningLayers = state.winning_line.map((move) => move.layer);
  const winningMinimum = winningLayers.length > 0 ? Math.min(...winningLayers) : null;
  const winningMaximum = winningLayers.length > 0 ? Math.max(...winningLayers) : null;
  const winningMarkerKey = state.winning_line.map(moveKey).join("|");
  const lastLayer = state.last_move?.layer ?? null;
  const lastMarkerKey = state.last_move ? `${state.revision}:${moveKey(state.last_move)}` : "";
  const hintLayer = hint?.move.layer ?? null;
  const hintMarkerKey = hint ? `${hint.for_revision}:${moveKey(hint.move)}` : "";
  const previousCompactLandscape = useRef(compactLandscape);

  useEffect(() => {
    if (previousCompactLandscape.current === compactLandscape) return;
    setDrawerOpen(!compactLandscape);
    setShowColumnGuides(!compactLandscape);
    previousCompactLandscape.current = compactLandscape;
  }, [compactLandscape]);

  useEffect(() => {
    if (!mobileLayout || layerViewMode !== "sliding4") return;
    const ranges: Array<readonly [number, number]> = [];
    if (winningMinimum !== null && winningMaximum !== null) ranges.push([winningMinimum, winningMaximum]);
    if (lastLayer !== null) ranges.push([lastLayer, lastLayer]);
    if (hintLayer !== null) ranges.push([hintLayer, hintLayer]);
    if (ranges.length === 0) return;

    setLayerWindowStart((current) => nextTrackedWindowStart(current, maxLayerWindowStart, ranges));
  }, [
    hintLayer,
    hintMarkerKey,
    lastLayer,
    lastMarkerKey,
    layerViewMode,
    maxLayerWindowStart,
    mobileLayout,
    winningMarkerKey,
    winningMaximum,
    winningMinimum,
  ]);

  const selectCamera = (preset: CameraPreset) => {
    setCameraCommand((current) => ({ preset, serial: current.serial + 1 }));
  };
  const setSlicePicker = (enabled: boolean) => {
    setSlicePickerEnabled(enabled);
    if (!enabled) setSliceSelection(null);
  };

  return (
    <section className={`game-workspace ${viewMode === "3d" ? "three-d-workspace" : ""}`}>
      <div className={`board-view-layout ${viewMode === "3d" ? `three-d ${drawerOpen ? "drawer-open" : "drawer-closed"}` : "two-d"}`}>
        <div className={`board-canvas-pane ${sliceSelection ? "slice-active" : ""} ${winRate ? "has-win-rate" : ""}`}>
          {viewMode === "2d" ? (
            <div className="board-area">
              <LayerBoards
                copy={t}
                state={state}
                hint={hint}
                locked={locked}
                onMove={onMove}
                layerViewMode={layerViewMode}
                windowStart={layerWindowStart}
                onLayerViewModeChange={setLayerViewMode}
                onWindowStartChange={setLayerWindowStart}
                mobileLayout={mobileLayout}
              />
              {mobileLayout && <WinRateCard copy={t} result={winRate} />}
              <div className="board-legend">
                <span><i className="legend-cell legal" />{t.game.legal}</span>
                <span><i className="legend-cell illegal" />{t.game.illegal}</span>
                <span><i className="legend-piece red" />{t.game.red}</span>
                <span><i className="legend-piece blue" />{t.game.blue}</span>
              </div>
            </div>
          ) : (
            <>
              <Suspense fallback={<div className="board-3d-loading">{t.game.view3d.loading}</div>}>
                <Board3D
                  copy={t}
                  state={state}
                  hint={hint}
                  moveLocked={locked}
                  compactLayout={compactLandscape}
                  pieceFocus={pieceFocus}
                  showColumnGuides={showColumnGuides}
                  slicePickerEnabled={slicePickerEnabled}
                  sliceSelection={sliceSelection}
                  layerSpacing={layerSpacing}
                  cameraCommand={cameraCommand}
                  onMove={(move) => onMove?.(move)}
                  onSliceSelection={setSliceSelection}
                  onClearSlice={() => setSliceSelection(null)}
                  onFallbackTo2d={() => onViewModeChange("2d")}
                />
              </Suspense>
              <WinRateCard copy={t} result={winRate} />
            </>
          )}
          {viewMode === "2d" && !mobileLayout && <WinRateCard copy={t} result={winRate} />}
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
  );
}
