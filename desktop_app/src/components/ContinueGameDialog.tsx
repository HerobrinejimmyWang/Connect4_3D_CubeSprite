import { useEffect, useRef } from "react";

import type { Copy } from "../i18n";
import type { GameMode, Player } from "../types";

interface Props {
  copy: Copy;
  step: number;
  busy: boolean;
  onChoose: (mode: GameMode, humanPlayer?: Player) => void;
  onClose: () => void;
}

export function ContinueGameDialog({ copy: t, step, busy, onChoose, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onClose]);

  return (
    <div className="dialog-backdrop">
      <section className="continue-dialog" role="dialog" aria-modal="true" aria-labelledby="continue-title">
        <header>
          <div>
            <small>{t.replay.stepLabel.replace("{step}", String(step))}</small>
            <h2 id="continue-title">{t.replay.continueTitle}</h2>
          </div>
          <button ref={closeRef} aria-label={t.common.close} disabled={busy} onClick={onClose}>×</button>
        </header>
        <p>{t.replay.continueDetail}</p>
        <div className="continue-options">
          <button disabled={busy} onClick={() => onChoose("pvp")}>
            <span aria-hidden="true">●●</span>
            <strong>{t.replay.continuePvp}</strong>
            <small>{t.replay.continuePvpDetail}</small>
          </button>
          <button className="red-choice" disabled={busy} onClick={() => onChoose("pvai", 1)}>
            <i className="piece-preview red" />
            <strong>{t.replay.continueRed}</strong>
            <small>{t.replay.continueRedDetail}</small>
          </button>
          <button className="blue-choice" disabled={busy} onClick={() => onChoose("pvai", -1)}>
            <i className="piece-preview blue" />
            <strong>{t.replay.continueBlue}</strong>
            <small>{t.replay.continueBlueDetail}</small>
          </button>
        </div>
        {busy && <div className="continue-busy" role="status"><i className="inline-spinner" />{t.replay.continuing}</div>}
      </section>
    </div>
  );
}
