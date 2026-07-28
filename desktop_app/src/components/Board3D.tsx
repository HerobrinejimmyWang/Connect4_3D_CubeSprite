import { Component, useState, type ErrorInfo, type ReactNode } from "react";

import { sliceSelectionLabel } from "../board3dModel";
import type { Copy } from "../i18n";
import type { CameraCommand, GameState, HintResult, LayerSpacing, Move, PieceFocus, SliceAxis, SliceSelection } from "../types";
import { Board3DCanvas } from "./Board3DCanvas";
import { SlicePreview } from "./SlicePreview";

interface BoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
}

class ThreeCanvasBoundary extends Component<BoundaryProps, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("CubeSprite 3D canvas failed", error, info.componentStack);
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

interface Props {
  copy: Copy;
  state: GameState;
  hint: HintResult | null;
  moveLocked: boolean;
  compactLayout: boolean;
  pieceFocus: PieceFocus;
  showColumnGuides: boolean;
  slicePickerEnabled: boolean;
  sliceSelection: SliceSelection | null;
  layerSpacing: LayerSpacing;
  cameraCommand: CameraCommand;
  onMove: (move: Move) => void;
  onSliceSelection: (selection: SliceSelection) => void;
  onClearSlice: () => void;
  onFallbackTo2d: () => void;
}

const EDGE_LABEL_GROUPS: Array<{ axis: SliceAxis; count: number; prefix: string; className: string }> = [
  { axis: "col", count: 5, prefix: "C", className: "columns" },
  { axis: "row", count: 5, prefix: "R", className: "rows" },
  { axis: "layer", count: 6, prefix: "F", className: "floors" },
];

export function Board3D(props: Props) {
  const [hoveredMove, setHoveredMove] = useState<Move | null>(null);
  const t = props.copy.game.view3d;

  const fallback = (
    <div className="board-3d-fallback" role="alert">
      <strong>{t.unavailable}</strong>
      <button type="button" onClick={props.onFallbackTo2d}>{t.return2d}</button>
    </div>
  );

  return (
    <div
      className={`board-3d-stage ${props.sliceSelection ? "slice-active" : ""}`}
      role="region"
      aria-label={t.boardLabel}
    >
      <div className={`board-3d-view ${props.moveLocked ? "move-locked" : ""}`}>
        <ThreeCanvasBoundary fallback={fallback}>
          <Board3DCanvas
            state={props.state}
            hint={props.hint}
            moveLocked={props.moveLocked}
            compactLayout={props.compactLayout}
            showCoordinateLabels={!props.compactLayout}
            pieceFocus={props.pieceFocus}
            showColumnGuides={props.showColumnGuides}
            slicePickerEnabled={props.slicePickerEnabled}
            sliceSelection={props.sliceSelection}
            layerSpacing={props.layerSpacing}
            cameraCommand={props.cameraCommand}
            onMove={props.onMove}
            onHoverMove={setHoveredMove}
            onSliceSelection={props.onSliceSelection}
          />
        </ThreeCanvasBoundary>

        {props.slicePickerEnabled && EDGE_LABEL_GROUPS.map((group) => (
          <div className={`slice-edge-labels ${group.className}`} role="group" aria-label={`${t.boardEdgeSlice} ${group.prefix}`} key={group.axis}>
            {Array.from({ length: group.count }, (_, index) => {
              const label = `${group.prefix}${index + 1}`;
              const selected = props.sliceSelection?.axis === group.axis && props.sliceSelection.index === index;
              return (
                <button
                  type="button"
                  aria-label={`${t.boardEdgeSlice}: ${label}`}
                  aria-pressed={selected}
                  className={selected ? "selected" : ""}
                  key={label}
                  onClick={() => props.onSliceSelection({ axis: group.axis, index })}
                >
                  {label}
                </button>
              );
            })}
          </div>
        ))}

        {(hoveredMove || props.sliceSelection) && (
          <div className="board-3d-coordinate visible" aria-live="polite">
            {hoveredMove
              ? `F${hoveredMove.layer + 1} · R${hoveredMove.row + 1} · C${hoveredMove.col + 1}`
              : `${t.selectedSlice}: ${sliceSelectionLabel(props.sliceSelection!)}`}
          </div>
        )}
        <div className="board-3d-help">{t.controlsHint}</div>
      </div>

      {props.sliceSelection && (
        <SlicePreview
          copy={props.copy}
          state={props.state}
          hint={props.hint}
          pieceFocus={props.pieceFocus}
          selection={props.sliceSelection}
          onClear={props.onClearSlice}
        />
      )}
    </div>
  );
}
