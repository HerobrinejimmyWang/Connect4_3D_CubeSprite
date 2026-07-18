import type { Copy } from "../i18n";
import type { MenuPanel, Player, ReplaySummary } from "../types";
import { ReplayLibraryPanel } from "./ReplayLibraryPanel";

interface Props {
  copy: Copy;
  activePanel: MenuPanel;
  busy: boolean;
  replays: ReplaySummary[];
  replayListBusy: boolean;
  replayImportBusy: boolean;
  replayDeleteBusyId: string | null;
  replayExportBusyId: string | null;
  onPvp: () => void;
  onOpenPvai: () => void;
  onOpenReplays: () => void;
  onChooseSide: (player: Player) => void;
  onClosePanel: () => void;
  onOpenReplay: (replay: ReplaySummary) => void;
  onDeleteReplay: (replay: ReplaySummary) => void;
  onExportReplay: (replay: ReplaySummary) => void;
  onImportReplay: (file: File) => void;
  onAiSettings: () => void;
  onSettings: () => void;
  onInstructions: () => void;
}

export function MenuScreen(props: Props) {
  const { copy: t } = props;
  return (
    <main className="menu-screen">
      <section className="brand-panel">
        <div className="cube-mark" aria-hidden="true">
          <i /><i /><i /><i />
        </div>
        <h1>Connect4 3D <span>CubeSprite</span></h1>
        <small>{t.version}</small>
      </section>

      <section className={`menu-actions ${props.activePanel ? "menu-panel-open" : ""}`} aria-label="Main menu">
        <div className="primary-menu">
          <button className="menu-card red-accent" aria-label={t.menu.pvp} disabled={props.busy} onClick={props.onPvp}>
            <span className="menu-icon">●●</span>
            <span><strong>{t.menu.pvp}</strong><small>{t.menu.pvpDetail}</small></span>
            <b aria-hidden="true">›</b>
          </button>
          <button className="menu-card blue-accent" aria-label={t.menu.pvai} disabled={props.busy} onClick={props.onOpenPvai} aria-expanded={props.activePanel === "side"}>
            <span className="menu-icon">◆</span>
            <span><strong>{t.menu.pvai}</strong><small>{t.menu.pvaiDetail}</small></span>
            <b aria-hidden="true">›</b>
          </button>
          <button className="menu-card replay-accent" aria-label={t.menu.replay} disabled={props.busy} onClick={props.onOpenReplays} aria-expanded={props.activePanel === "replays"}>
            <span className="menu-icon">▶</span>
            <span><strong>{t.menu.replay}</strong><small>{t.menu.replayDetail}</small></span>
            <b aria-hidden="true">›</b>
          </button>
          <div className="secondary-menu">
            <button aria-label={t.menu.aiSettings} disabled={props.busy} onClick={props.onAiSettings}><span aria-hidden="true">⌁</span>{t.menu.aiSettings}</button>
            <button aria-label={t.menu.settings} disabled={props.busy} onClick={props.onSettings}><span aria-hidden="true">⚙</span>{t.menu.settings}</button>
            <button aria-label={t.menu.instructions} disabled={props.busy} onClick={props.onInstructions}><span aria-hidden="true">?</span>{t.menu.instructions}</button>
          </div>
        </div>

        {props.activePanel === "side" && <aside className="side-picker">
          <div className="side-picker-heading">
            <span>{t.menu.chooseSide}</span>
            <button aria-label={t.menu.cancel} disabled={props.busy} onClick={props.onClosePanel}>×</button>
          </div>
          <button className="side-option red-side" aria-label={t.menu.redFirst} disabled={props.busy} onClick={() => props.onChooseSide(1)}>
            <i className="piece-preview red" />
            <span><strong>{t.menu.redFirst}</strong><small>{t.menu.redFirstDetail}</small></span>
          </button>
          <button className="side-option blue-side" aria-label={t.menu.blueSecond} disabled={props.busy} onClick={() => props.onChooseSide(-1)}>
            <i className="piece-preview blue" />
            <span><strong>{t.menu.blueSecond}</strong><small>{t.menu.blueSecondDetail}</small></span>
          </button>
          {props.busy && <div className="mini-loader"><i /> <span>{t.loadingDetail}</span></div>}
        </aside>}
        {props.activePanel === "replays" && (
          <ReplayLibraryPanel
            copy={t}
            replays={props.replays}
            loading={props.replayListBusy}
            importBusy={props.replayImportBusy}
            deleteBusyId={props.replayDeleteBusyId}
            exportBusyId={props.replayExportBusyId}
            onOpen={props.onOpenReplay}
            onDelete={props.onDeleteReplay}
            onExport={props.onExportReplay}
            onImport={props.onImportReplay}
            onClose={props.onClosePanel}
          />
        )}
      </section>
    </main>
  );
}
