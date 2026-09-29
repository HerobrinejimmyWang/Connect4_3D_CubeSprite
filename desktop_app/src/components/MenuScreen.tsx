import type { CSSProperties } from "react";

import type { Copy } from "../i18n";
import type { MenuPanel, Player, ReplaySummary, RuleId } from "../types";
import { ReplayLibraryPanel } from "./ReplayLibraryPanel";

interface Props {
  copy: Copy;
  activePanel: MenuPanel;
  busy: boolean;
  replayEnabled?: boolean;
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
  ruleId: RuleId;
  onRuleChange: (ruleId: RuleId) => void;
  intelligence: number;
  onIntelligenceChange: (index: number) => void;
}

const RULE_IDS: RuleId[] = ["classic", "p1_vertical_ignored", "p1_vertical_forbidden", "p1_layer0_ignored", "p1_vertical_and_layer0_ignored"];
const SIMS = [16, 32, 64, 128, 256, 512, 1024];

export function MenuScreen(props: Props) {
  const { copy: t } = props;
  return (
    <main className={`menu-screen ${props.activePanel ? "panel-open" : ""}`}>
      <section className="brand-panel">
        <div className="cube-mark" aria-hidden="true">
          <i /><i /><i /><i />
        </div>
        <h1>Connect4 3D <span>CubeSprite</span></h1>
      </section>

      <section className={`menu-actions ${props.activePanel ? "menu-panel-open" : ""}`} aria-label="Main menu">
        <div className="primary-menu">
          <fieldset className="rule-selector">
            <legend>{t.menu.ruleSelection}</legend>
            <div className="rule-selector-options">
              {RULE_IDS.map((id, index) => <label key={id} title={t.menu.rules[index]} className={props.ruleId === id ? "selected" : ""}>
                <input type="radio" name="game-rule" value={id} checked={props.ruleId === id} disabled={props.busy} onChange={() => props.onRuleChange(id)} />
                <span>{t.menu.ruleShort[index]}</span>
              </label>)}
            </div>
          </fieldset>
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
          {props.replayEnabled !== false && (
            <button className="menu-card replay-accent" aria-label={t.menu.replay} disabled={props.busy} onClick={props.onOpenReplays} aria-expanded={props.activePanel === "replays"}>
              <span className="menu-icon">▶</span>
              <span><strong>{t.menu.replay}</strong><small>{t.menu.replayDetail}</small></span>
              <b aria-hidden="true">›</b>
            </button>
          )}
          <div className="secondary-menu">
            <button aria-label={t.menu.aiSettings} disabled={props.busy} onClick={props.onAiSettings}><span aria-hidden="true">⌁</span>{t.menu.aiSettings}</button>
            <button aria-label={t.menu.settings} disabled={props.busy} onClick={props.onSettings}><span aria-hidden="true">⚙</span>{t.menu.settings}</button>
            <button aria-label={t.menu.instructions} disabled={props.busy} onClick={props.onInstructions}><span aria-hidden="true">?</span>{t.menu.instructions}</button>
          </div>
          <section className="intelligence-box" aria-label={t.menu.intelligence}>
            <div className="intelligence-heading"><strong>{t.menu.intelligence}: {t.menu.effortNames[props.intelligence]}</strong><button onClick={props.onAiSettings} disabled={props.busy}>{t.menu.advance}</button></div>
            <div className="intelligence-slider" style={{ "--effort-progress": `${props.intelligence / (SIMS.length - 1) * 100}%` } as CSSProperties}>
              <div className="intelligence-rail" aria-hidden="true">
                <div className="intelligence-ticks">{SIMS.map((sim, index) => <span className={index <= props.intelligence ? "active" : ""} key={sim} />)}</div>
              </div>
              <input type="range" min="0" max="6" step="1" value={props.intelligence} onChange={(event) => props.onIntelligenceChange(Number(event.target.value))} aria-label={t.menu.intelligence} aria-valuetext={`${t.menu.effortNames[props.intelligence]}, ${SIMS[props.intelligence]} MCTS`} disabled={props.busy} />
            </div>
          </section>
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
        {props.replayEnabled !== false && props.activePanel === "replays" && (
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
