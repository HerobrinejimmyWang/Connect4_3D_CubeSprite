import { Fragment, useMemo } from "react";

import {
  coordinateKey,
  pieceRenderMode,
  sliceCells,
  sliceSelectionLabel,
} from "../board3dModel";
import type { Copy } from "../i18n";
import type { GameState, HintResult, PieceFocus, Player, SliceSelection } from "../types";

interface Props {
  copy: Copy;
  state: GameState;
  hint: HintResult | null;
  pieceFocus: PieceFocus;
  selection: SliceSelection;
  onClear: () => void;
}

export function SlicePreview({ copy, state, hint, pieceFocus, selection, onClear }: Props) {
  const t = copy.game.view3d;
  const cells = useMemo(() => sliceCells(state.board, selection), [selection, state.board]);
  const winning = useMemo(() => new Set(state.winning_line.map(coordinateKey)), [state.winning_line]);
  const legal = useMemo(() => new Set(state.legal_moves.map(coordinateKey)), [state.legal_moves]);
  const last = state.last_move ? coordinateKey(state.last_move) : "";
  const hintKey = hint ? coordinateKey(hint.move) : "";
  const label = sliceSelectionLabel(selection);
  const topLabels = Array.from({ length: 5 }, (_, index) => (
    selection.axis === "col" ? `R${index + 1}` : `C${index + 1}`
  ));
  const sideLabels = selection.axis === "layer"
    ? Array.from({ length: 5 }, (_, index) => `R${index + 1}`)
    : Array.from({ length: 6 }, (_, index) => `F${6 - index}`);
  const rows = Array.from({ length: sideLabels.length }, (_, index) => cells.slice(index * 5, index * 5 + 5));
  const kind = selection.axis === "col" ? t.columnSlice : selection.axis === "row" ? t.rowSlice : t.layerSlice;

  return (
    <aside className="slice-preview" aria-label={`${t.selectedSlice}: ${label}`}>
      <header>
        <div>
          <small>{t.selectedSlice}</small>
          <strong>{label} · {kind}</strong>
        </div>
        <button type="button" onClick={onClear} aria-label={t.clearSlice}>×</button>
      </header>

      <div className="slice-orientation">
        <span>{selection.axis === "col" ? "F × R" : selection.axis === "row" ? "F × C" : "R × C"}</span>
        <small>{t.sliceReadOnly}</small>
      </div>

      <div className={`slice-matrix axis-${selection.axis}`} role="grid" aria-label={`${label} ${kind}`}>
        <span className="slice-axis-corner" aria-hidden="true" />
        {topLabels.map((topLabel) => <b className="slice-axis-top" key={topLabel}>{topLabel}</b>)}
        {rows.map((row, rowIndex) => (
          <Fragment key={sideLabels[rowIndex]}>
            <b className="slice-axis-side">{sideLabels[rowIndex]}</b>
            {row.map((cell) => {
              const key = coordinateKey(cell);
              const player = cell.value === 1 || cell.value === -1 ? cell.value as Player : null;
              const mode = player ? pieceRenderMode(player, pieceFocus, winning.has(key)) : null;
              const cellState = player === 1 ? copy.game.red : player === -1 ? copy.game.blue : t.emptySliceCell;
              const className = [
                "slice-cell",
                legal.has(key) ? "legal" : "",
                key === last ? "last" : "",
                key === hintKey && !player ? "hint" : "",
                winning.has(key) ? "winning" : "",
              ].filter(Boolean).join(" ");
              return (
                <div
                  className={className}
                  role="gridcell"
                  aria-label={`F${cell.layer + 1}, R${cell.row + 1}, C${cell.col + 1}: ${cellState}`}
                  key={key}
                >
                  {player && <i className={`slice-piece ${player === 1 ? "red" : "blue"} ${mode ?? ""}`} />}
                  {key === hintKey && !player && <i className="slice-hint" />}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>

      <footer>{t.sliceContextHint}</footer>
    </aside>
  );
}
