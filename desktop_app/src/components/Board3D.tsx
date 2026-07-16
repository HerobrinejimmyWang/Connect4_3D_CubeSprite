import { Component, useState, type ErrorInfo, type ReactNode } from "react";

import type { Copy } from "../i18n";
import type { CameraCommand, GameState, HintResult, LayerSpacing, Move, PieceFocus } from "../types";
import { Board3DCanvas } from "./Board3DCanvas";

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
  pieceFocus: PieceFocus;
  showColumnGuides: boolean;
  layerSpacing: LayerSpacing;
  cameraCommand: CameraCommand;
  onMove: (move: Move) => void;
  onFallbackTo2d: () => void;
}

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
      className={`board-3d-view ${props.moveLocked ? "move-locked" : ""}`}
      role="region"
      aria-label={t.boardLabel}
    >
      <ThreeCanvasBoundary fallback={fallback}>
        <Board3DCanvas
          state={props.state}
          hint={props.hint}
          moveLocked={props.moveLocked}
          pieceFocus={props.pieceFocus}
          showColumnGuides={props.showColumnGuides}
          layerSpacing={props.layerSpacing}
          cameraCommand={props.cameraCommand}
          onMove={props.onMove}
          onHoverMove={setHoveredMove}
        />
      </ThreeCanvasBoundary>

      <div className={`board-3d-coordinate ${hoveredMove ? "visible" : ""}`} aria-live="polite">
        {hoveredMove ? `F${hoveredMove.layer + 1} · R${hoveredMove.row + 1} · C${hoveredMove.col + 1}` : "F– · R– · C–"}
      </div>
      <div className="board-3d-help">{t.controlsHint}</div>
    </div>
  );
}
