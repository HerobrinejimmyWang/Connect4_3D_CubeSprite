import { useEffect, useMemo, useState } from "react";

import type { Copy } from "../i18n";
import type { BoardViewMode, GameMode, Player, ReplayOpenResult } from "../types";
import { BoardWorkspace } from "./BoardWorkspace";
import { ContinueGameDialog } from "./ContinueGameDialog";
import { WinRateMonitor } from "./WinRateMonitor";

interface Props {
  copy: Copy;
  replay: ReplayOpenResult;
  autoplayIntervalMs: number;
  analysisThinking: boolean;
  continueBusy: boolean;
  onAnalyze: () => void;
  onContinue: (step: number, mode: GameMode, humanPlayer?: Player) => void;
  onExit: () => void;
}

export function ReplayScreen(props: Props) {
  const { copy: t, replay } = props;
  const maxSteps = replay.replay.move_count;
  const [cursor, setCursor] = useState(0);
  const [autoplay, setAutoplay] = useState(false);
  const [viewMode, setViewMode] = useState<BoardViewMode>("2d");
  const [monitorOpen, setMonitorOpen] = useState(Boolean(replay.analysis));
  const [continueOpen, setContinueOpen] = useState(false);

  useEffect(() => {
    setCursor(0);
    setAutoplay(false);
    setViewMode("2d");
    setMonitorOpen(Boolean(replay.analysis));
    setContinueOpen(false);
  }, [replay.replay.id]);

  useEffect(() => {
    if (replay.analysis) setMonitorOpen(true);
  }, [replay.analysis]);

  useEffect(() => {
    if (!autoplay) return;
    if (cursor >= maxSteps) {
      setAutoplay(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setCursor((current) => Math.min(maxSteps, current + 1));
    }, props.autoplayIntervalMs);
    return () => window.clearTimeout(timer);
  }, [autoplay, cursor, maxSteps, props.autoplayIntervalMs]);

  const frame = replay.frames[Math.min(cursor, replay.frames.length - 1)];
  const finished = frame.status !== "playing";
  const statusText = finished
    ? frame.winner === 1 ? t.game.redWins : frame.winner === -1 ? t.game.blueWins : t.game.draw
    : frame.current_player === 1 ? t.game.red : t.game.blue;
  const statusClass = frame.winner === -1 || (!finished && frame.current_player === -1) ? "blue-text" : frame.winner === 0 ? "neutral-text" : "red-text";

  const stepTo = (step: number) => {
    setAutoplay(false);
    setCursor(Math.max(0, Math.min(maxSteps, step)));
  };

  const functionLabel = useMemo(
    () => autoplay ? t.replay.pause : t.replay.autoplay,
    [autoplay, t.replay.autoplay, t.replay.pause],
  );

  return (
    <main className={`game-screen replay-screen ${monitorOpen ? "monitor-open" : "monitor-closed"}`}>
      <header className="game-topbar">
        <div className="compact-brand"><span className="mini-cube">◆</span><strong>CubeSprite</strong><small>REPLAY</small></div>
        <div className="game-state-panel replay-state-panel">
          <div className="turn-state">
            <small>{finished ? t.game.result : t.game.currentPlayer}</small>
            <strong className={statusClass}><i />{statusText}</strong>
          </div>
          <div className="move-count"><small>{t.replay.progress}</small><strong>{cursor}<span>/{maxSteps}</span></strong></div>
          <div className="replay-name" title={replay.replay.name}>
            <small>{t.replay.nowPlaying}</small>
            <strong>{replay.replay.name}</strong>
          </div>
          <div className={`thinking-state ${props.analysisThinking ? "visible" : ""}`} aria-live="polite">
            {props.analysisThinking && <><span className="thinking-dots"><i /><i /><i /></span>{t.replay.analysisThinking}</>}
          </div>
        </div>
      </header>

      <div className="replay-workspace-shell">
        <BoardWorkspace
          copy={t}
          state={frame}
          viewMode={viewMode}
          locked
          onViewModeChange={setViewMode}
        />
        <WinRateMonitor
          copy={t}
          analysis={replay.analysis}
          cursor={cursor}
          maxSteps={maxSteps}
          open={monitorOpen}
          onToggle={() => setMonitorOpen((open) => !open)}
        />
      </div>

      <footer className="game-functions replay-functions">
        <div className="function-group main-functions">
          <button aria-label={t.replay.previous} disabled={cursor <= 0} onClick={() => stepTo(cursor - 1)}><span aria-hidden="true">←</span>{t.replay.previous}</button>
          <button aria-label={t.replay.next} disabled={cursor >= maxSteps} onClick={() => stepTo(cursor + 1)}><span aria-hidden="true">→</span>{t.replay.next}</button>
          <button
            aria-label={functionLabel}
            className={autoplay ? "working" : ""}
            aria-pressed={autoplay}
            disabled={maxSteps === 0 || (!autoplay && cursor >= maxSteps)}
            onClick={() => setAutoplay((playing) => !playing)}
          >
            <span aria-hidden="true">{autoplay ? "Ⅱ" : "▶"}</span>{functionLabel}
          </button>
          <button aria-label={t.replay.fromStart} disabled={cursor === 0} onClick={() => stepTo(0)}><span aria-hidden="true">↤</span>{t.replay.fromStart}</button>
        </div>
        <div className="function-group edge-functions">
          <button
            aria-label={t.replay.continueFromHere}
            disabled={finished || props.continueBusy}
            onClick={() => {
              setAutoplay(false);
              setContinueOpen(true);
            }}
          >
            <span aria-hidden="true">↗</span>{t.replay.continueFromHere}
          </button>
          <button
            aria-label={t.replay.monitor}
            aria-expanded={monitorOpen}
            onClick={() => setMonitorOpen((open) => !open)}
          >
            <span aria-hidden="true">⌁</span>{t.replay.monitor}
          </button>
          <button
            aria-label={t.replay.calculate}
            className={props.analysisThinking ? "working" : ""}
            disabled={props.analysisThinking}
            onClick={props.onAnalyze}
          >
            <span aria-hidden="true">▰</span>{t.replay.calculate}
          </button>
          <button
            aria-label={viewMode === "2d" ? t.game.switch3d : t.game.switch2d}
            className="view-toggle-button"
            onClick={() => setViewMode((mode) => mode === "2d" ? "3d" : "2d")}
          >
            <span aria-hidden="true">◇</span>{viewMode === "2d" ? t.game.switch3d : t.game.switch2d}
          </button>
          <button
            aria-label={t.replay.exit}
            className="exit-button"
            disabled={props.continueBusy}
            onClick={props.onExit}
          >
            <span aria-hidden="true">⌂</span>{t.replay.exit}
          </button>
        </div>
      </footer>

      {continueOpen && (
        <ContinueGameDialog
          copy={t}
          step={cursor}
          busy={props.continueBusy}
          onChoose={(mode, humanPlayer) => props.onContinue(cursor, mode, humanPlayer)}
          onClose={() => setContinueOpen(false)}
        />
      )}
    </main>
  );
}
