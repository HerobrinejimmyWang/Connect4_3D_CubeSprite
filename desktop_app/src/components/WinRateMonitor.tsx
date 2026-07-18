import { useMemo } from "react";

import type { Copy } from "../i18n";
import type { WinRateAnalysis, WinRateAnalysisPoint } from "../types";

interface Props {
  copy: Copy;
  analysis: WinRateAnalysis | null;
  cursor: number;
  maxSteps: number;
  open: boolean;
  onToggle: () => void;
}

const CHART = { width: 320, height: 196, left: 38, right: 12, top: 14, bottom: 28 };

function pointPosition(point: WinRateAnalysisPoint, maxSteps: number, value: number): [number, number] {
  const plotWidth = CHART.width - CHART.left - CHART.right;
  const plotHeight = CHART.height - CHART.top - CHART.bottom;
  const x = CHART.left + (maxSteps > 0 ? point.step / maxSteps : 0) * plotWidth;
  const y = CHART.top + (1 - Math.max(0, Math.min(1, value))) * plotHeight;
  return [x, y];
}

function linePoints(points: WinRateAnalysisPoint[], maxSteps: number, side: "red" | "blue"): string {
  return points
    .map((point) => pointPosition(point, maxSteps, point[side]).map((value) => value.toFixed(2)).join(","))
    .join(" ");
}

function formatTimestamp(value: string): string {
  return value.replace("T", " ").replace(/Z$/, "").slice(0, 19);
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${Math.max(0, Math.round(durationMs))} ms`;
  const seconds = durationMs / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60);
  return `${minutes} min ${remaining} s`;
}

export function WinRateMonitor({ copy: t, analysis, cursor, maxSteps, open, onToggle }: Props) {
  const points = useMemo(
    () => analysis?.points.filter((point) => Number.isFinite(point.step) && Number.isFinite(point.red) && Number.isFinite(point.blue)) ?? [],
    [analysis],
  );
  const cursorX = pointPosition({ step: cursor, red: 0, blue: 0, estimate: "" }, maxSteps, 0)[0];
  const current = points.reduce<WinRateAnalysisPoint | null>((nearest, point) => (
    !nearest || Math.abs(point.step - cursor) < Math.abs(nearest.step - cursor) ? point : nearest
  ), null);

  return (
    <aside className={`win-rate-monitor ${open ? "open" : "closed"}`} aria-label={t.replay.monitor}>
      <button
        className="monitor-toggle"
        aria-label={open ? t.replay.collapseMonitor : t.replay.expandMonitor}
        aria-expanded={open}
        onClick={onToggle}
      >
        <span aria-hidden="true">⌁</span>{open ? "›" : "‹"}
      </button>
      {open && (
        <div className="monitor-content">
          <header>
            <div><small>WIN RATE / STEPS</small><h2>{t.replay.monitor}</h2></div>
            <span>{cursor}/{maxSteps}</span>
          </header>
          {!analysis || points.length === 0 ? (
            <div className="monitor-empty">
              <span aria-hidden="true">⌁</span>
              <strong>{t.replay.noAnalysis}</strong>
              <p>{t.replay.noAnalysisDetail}</p>
            </div>
          ) : (
            <>
              <div className="chart-legend">
                <span className="red-text"><i />{t.game.red}</span>
                <span className="blue-text"><i />{t.game.blue}</span>
                {current && <b>{t.replay.currentRate}: {(current.red * 100).toFixed(1)}% / {(current.blue * 100).toFixed(1)}%</b>}
              </div>
              <svg
                className="win-rate-chart"
                role="img"
                aria-label={t.replay.chartLabel}
                viewBox={`0 0 ${CHART.width} ${CHART.height}`}
              >
                <title>{t.replay.chartLabel}</title>
                {[0, 0.25, 0.5, 0.75, 1].map((value) => {
                  const y = pointPosition({ step: 0, red: value, blue: 0, estimate: "" }, maxSteps, value)[1];
                  return (
                    <g key={value}>
                      <line className="chart-grid-line" x1={CHART.left} x2={CHART.width - CHART.right} y1={y} y2={y} />
                      <text className="chart-axis-label" x={CHART.left - 7} y={y + 4} textAnchor="end">{Math.round(value * 100)}%</text>
                    </g>
                  );
                })}
                <line className="chart-axis-line" x1={CHART.left} x2={CHART.width - CHART.right} y1={CHART.height - CHART.bottom} y2={CHART.height - CHART.bottom} />
                <text className="chart-axis-label" x={CHART.left} y={CHART.height - 7}>0</text>
                <text className="chart-axis-label" x={CHART.width - CHART.right} y={CHART.height - 7} textAnchor="end">{maxSteps}</text>
                <polyline className="chart-line red-line" points={linePoints(points, maxSteps, "red")} />
                <polyline className="chart-line blue-line" points={linePoints(points, maxSteps, "blue")} />
                <line className="chart-cursor" x1={cursorX} x2={cursorX} y1={CHART.top} y2={CHART.height - CHART.bottom} />
                {current && (
                  <>
                    <circle className="chart-point red-point" cx={pointPosition(current, maxSteps, current.red)[0]} cy={pointPosition(current, maxSteps, current.red)[1]} r="4" />
                    <circle className="chart-point blue-point" cx={pointPosition(current, maxSteps, current.blue)[0]} cy={pointPosition(current, maxSteps, current.blue)[1]} r="4" />
                  </>
                )}
              </svg>
              <dl className="analysis-meta">
                <div><dt>{t.replay.analysisModel}</dt><dd>{analysis.model.display_name}</dd></div>
                <div>
                  <dt>{t.replay.analysisArtifact}</dt>
                  <dd title={`SHA-256 ${analysis.model.artifact_sha256}`}>
                    {analysis.model.source_iteration ? `iter ${analysis.model.source_iteration} · ` : ""}
                    {analysis.model.artifact_sha256.slice(0, 12)}…
                  </dd>
                </div>
                <div><dt>{t.replay.analysisConfig}</dt><dd>{analysis.config.mcts_sims} MCTS · T {analysis.config.temperature.toFixed(1)}</dd></div>
                <div><dt>{t.replay.analysisTime}</dt><dd>{formatTimestamp(analysis.completed_at)}</dd></div>
                <div><dt>{t.replay.analysisDuration}</dt><dd>{formatDuration(analysis.duration_ms)}</dd></div>
              </dl>
            </>
          )}
        </div>
      )}
    </aside>
  );
}
