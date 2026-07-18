import { useRef } from "react";

import type { Copy } from "../i18n";
import type { ReplaySummary } from "../types";

interface Props {
  copy: Copy;
  replays: ReplaySummary[];
  loading: boolean;
  importBusy: boolean;
  deleteBusyId: string | null;
  exportBusyId: string | null;
  onOpen: (replay: ReplaySummary) => void;
  onDelete: (replay: ReplaySummary) => void;
  onExport: (replay: ReplaySummary) => void;
  onImport: (file: File) => void;
  onClose: () => void;
}

function replayStatus(copy: Copy, replay: ReplaySummary): string {
  if (replay.status === "won") {
    if (replay.winner === 1) return copy.game.redWins;
    if (replay.winner === -1) return copy.game.blueWins;
  }
  if (replay.status === "draw") return copy.game.draw;
  return copy.replay.unfinished;
}

function savedTime(value: string): string {
  return value.replace("T", " ").replace(/Z$/, "").slice(0, 16);
}

export function ReplayLibraryPanel(props: Props) {
  const { copy: t } = props;
  const inputRef = useRef<HTMLInputElement>(null);

  const requestDelete = (replay: ReplaySummary) => {
    if (window.confirm(t.replay.deleteConfirm.replace("{name}", replay.name))) props.onDelete(replay);
  };

  return (
    <aside className="side-picker replay-picker" aria-label={t.replay.library}>
      <div className="side-picker-heading">
        <span>{t.replay.library}</span>
        <button aria-label={t.common.close} onClick={props.onClose}>×</button>
      </div>

      <div className="replay-list" aria-live="polite">
        {props.loading ? (
          <div className="replay-list-message"><i className="inline-spinner" />{t.replay.loading}</div>
        ) : props.replays.length === 0 ? (
          <div className="replay-list-message empty">{t.replay.empty}</div>
        ) : props.replays.map((replay) => (
          <article className="replay-list-item" key={replay.id}>
            <div className="replay-list-copy">
              <strong title={replay.name}>{replay.name}</strong>
              <small>{replay.move_count} {t.replay.steps} · {replayStatus(t, replay)}</small>
              <time dateTime={replay.saved_at}>{savedTime(replay.saved_at)}</time>
            </div>
            <div className="replay-list-actions">
              <button
                className="replay-enter"
                aria-label={`${t.replay.enter} ${replay.name}`}
                disabled={props.loading || props.deleteBusyId === replay.id}
                onClick={() => props.onOpen(replay)}
              >
                {t.replay.enter}
              </button>
              <button
                className="replay-export"
                aria-label={`${t.replay.export} ${replay.name}`}
                disabled={props.loading || props.deleteBusyId === replay.id || props.exportBusyId === replay.id}
                onClick={() => props.onExport(replay)}
              >
                {props.exportBusyId === replay.id ? "…" : t.replay.export}
              </button>
              <button
                className="replay-delete"
                aria-label={`${t.replay.delete} ${replay.name}`}
                disabled={props.deleteBusyId === replay.id}
                onClick={() => requestDelete(replay)}
              >
                {props.deleteBusyId === replay.id ? "…" : "×"}
              </button>
            </div>
          </article>
        ))}
      </div>

      <div className="replay-import">
        <input
          ref={inputRef}
          type="file"
          accept=".json,application/json"
          tabIndex={-1}
          aria-hidden="true"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (file) props.onImport(file);
          }}
        />
        <button
          disabled={props.importBusy}
          onClick={() => inputRef.current?.click()}
        >
          <span aria-hidden="true">＋</span>{props.importBusy ? t.replay.importing : t.replay.import}
        </button>
      </div>
    </aside>
  );
}
