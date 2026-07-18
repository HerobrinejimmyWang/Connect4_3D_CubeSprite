import type { Copy } from "../i18n";
import type { AutoplayInterval } from "../types";
import { PageShell } from "./PageShell";

interface Props {
  copy: Copy;
  preloadHint: boolean;
  onPreloadHint: (enabled: boolean) => void;
  autoplayIntervalMs: AutoplayInterval;
  onAutoplayInterval: (interval: AutoplayInterval) => void;
  onBack: () => void;
}

const AUTOPLAY_OPTIONS: Array<{ value: AutoplayInterval; label: string }> = [
  { value: 2000, label: "2.0 s" },
  { value: 1000, label: "1.0 s" },
  { value: 500, label: "0.5 s" },
  { value: 250, label: "0.25 s" },
];

export function SettingsScreen({ copy: t, preloadHint, onPreloadHint, autoplayIntervalMs, onAutoplayInterval, onBack }: Props) {
  return (
    <PageShell copy={t} title={t.settings.title} subtitle={t.settings.subtitle} onBack={onBack}>
      <div className="settings-stack">
      <div className="settings-card">
        <div className="settings-icon" aria-hidden="true">✦</div>
        <div className="settings-copy">
          <h2>{t.settings.preload}</h2>
          <p>{t.settings.preloadDetail}</p>
        </div>
        <button
          className={`toggle-switch ${preloadHint ? "on" : ""}`}
          role="switch"
          aria-checked={preloadHint}
          aria-label={t.settings.preload}
          onClick={() => onPreloadHint(!preloadHint)}
        >
          <span />
          <b>{preloadHint ? t.settings.preloadOn : t.settings.preloadOff}</b>
        </button>
      </div>
      <div className="settings-card replay-speed-setting">
        <div className="settings-icon" aria-hidden="true">▶</div>
        <div className="settings-copy">
          <h2>{t.settings.autoplaySpeed}</h2>
          <p>{t.settings.autoplaySpeedDetail}</p>
        </div>
        <div className="autoplay-options" role="radiogroup" aria-label={t.settings.autoplaySpeed}>
          {AUTOPLAY_OPTIONS.map((option) => (
            <button
              key={option.value}
              role="radio"
              aria-checked={autoplayIntervalMs === option.value}
              className={autoplayIntervalMs === option.value ? "selected" : ""}
              onClick={() => onAutoplayInterval(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      </div>
      <div className="future-settings"><span>＋</span>{t.settings.future}</div>
    </PageShell>
  );
}
