import type { Copy } from "../i18n";
import { PageShell } from "./PageShell";

interface Props {
  copy: Copy;
  preloadHint: boolean;
  onPreloadHint: (enabled: boolean) => void;
  onBack: () => void;
}

export function SettingsScreen({ copy: t, preloadHint, onPreloadHint, onBack }: Props) {
  return (
    <PageShell copy={t} title={t.settings.title} subtitle={t.settings.subtitle} onBack={onBack}>
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
      <div className="future-settings"><span>＋</span>{t.settings.future}</div>
    </PageShell>
  );
}
