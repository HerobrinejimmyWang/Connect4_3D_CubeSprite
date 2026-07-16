import type { Copy } from "../i18n";
import type { CameraPreset, LayerSpacing, PieceFocus } from "../types";

interface Props {
  copy: Copy;
  open: boolean;
  pieceFocus: PieceFocus;
  showColumnGuides: boolean;
  layerSpacing: LayerSpacing;
  onToggleOpen: () => void;
  onPieceFocus: (focus: PieceFocus) => void;
  onShowColumnGuides: (show: boolean) => void;
  onLayerSpacing: (spacing: LayerSpacing) => void;
  onCameraPreset: (preset: CameraPreset) => void;
  onResetCamera: () => void;
}

const FOCUS_OPTIONS: Array<[PieceFocus, "showAll" | "focusRed" | "focusBlue"]> = [
  ["all", "showAll"],
  ["red", "focusRed"],
  ["blue", "focusBlue"],
];

const CAMERA_OPTIONS: Array<[CameraPreset, "isometric" | "front" | "top"]> = [
  ["isometric", "isometric"],
  ["front", "front"],
  ["top", "top"],
];

export function ObservationDrawer(props: Props) {
  const t = props.copy.game.view3d;

  return (
    <aside className={`observation-drawer ${props.open ? "open" : "closed"}`} aria-label={t.observationTools}>
      <button
        type="button"
        className="drawer-toggle"
        aria-expanded={props.open}
        aria-label={props.open ? t.closeTools : t.openTools}
        onClick={props.onToggleOpen}
      >
        <span aria-hidden="true">◫</span>
        {props.open && <strong>{t.observationTools}</strong>}
        <i aria-hidden="true">{props.open ? "›" : "‹"}</i>
      </button>

      {props.open && (
        <div className="drawer-controls">
          <fieldset>
            <legend>{t.pieceDisplay}</legend>
            <div className="segmented-control focus-control">
              {FOCUS_OPTIONS.map(([value, label]) => (
                <label key={value} className={value}>
                  <input
                    type="radio"
                    name="piece-focus"
                    value={value}
                    checked={props.pieceFocus === value}
                    onChange={() => props.onPieceFocus(value)}
                  />
                  <span>{t[label]}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <section className="drawer-control-block">
            <h3>{t.columnGuides}</h3>
            <label className="drawer-switch">
              <span>{t.showColumnGuides}</span>
              <input
                type="checkbox"
                role="switch"
                checked={props.showColumnGuides}
                onChange={(event) => props.onShowColumnGuides(event.target.checked)}
              />
              <i aria-hidden="true" />
            </label>
            {props.showColumnGuides && (
              <div className="column-key" aria-label={t.columnCoordinate}>
                <span />
                {[1, 2, 3, 4, 5].map((col) => <b key={`c${col}`}>C{col}</b>)}
                {[1, 2, 3, 4, 5].map((row) => (
                  <div className="column-key-row" key={`r${row}`}>
                    <b>R{row}</b>
                    {[1, 2, 3, 4, 5].map((col) => <i key={`${row}:${col}`} />)}
                  </div>
                ))}
              </div>
            )}
          </section>

          <fieldset>
            <legend>{t.layerSpacing}</legend>
            <div className="segmented-control two-up">
              <label>
                <input
                  type="radio"
                  name="layer-spacing"
                  checked={props.layerSpacing === "standard"}
                  onChange={() => props.onLayerSpacing("standard")}
                />
                <span>{t.standardSpacing}</span>
              </label>
              <label>
                <input
                  type="radio"
                  name="layer-spacing"
                  checked={props.layerSpacing === "expanded"}
                  onChange={() => props.onLayerSpacing("expanded")}
                />
                <span>{t.expandedSpacing}</span>
              </label>
            </div>
            <p>{t.displayOnly}</p>
          </fieldset>

          <section className="drawer-control-block">
            <h3>{t.camera}</h3>
            <div className="camera-presets">
              {CAMERA_OPTIONS.map(([preset, label]) => (
                <button
                  type="button"
                  key={preset}
                  onClick={() => props.onCameraPreset(preset)}
                >
                  {t[label]}
                </button>
              ))}
            </div>
            <button type="button" className="reset-camera" onClick={props.onResetCamera}>↺ {t.resetCamera}</button>
          </section>
        </div>
      )}
    </aside>
  );
}
