import { useEffect, useState } from "react";

import type { Copy } from "../i18n";
import type { AiConfig, AiRole, AiSettings, ModelInfo } from "../types";
import { PageShell } from "./PageShell";

const ROLE_ORDER: AiRole[] = ["combat", "hint", "winRate"];

interface Props {
  copy: Copy;
  models: ModelInfo[];
  mctsOptions: number[];
  settings: AiSettings;
  onChange: (role: AiRole, next: AiConfig) => void;
  onBack: () => void;
}

function TemperatureControl({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const [input, setInput] = useState(value.toFixed(1));
  useEffect(() => setInput(value.toFixed(1)), [value]);

  const commit = (raw: string) => {
    const parsed = Number(raw);
    const clamped = Number.isFinite(parsed) ? Math.min(5, Math.max(0, parsed)) : value;
    onChange(Math.round(clamped * 10) / 10);
    setInput(clamped.toFixed(1));
  };

  return (
    <div className="number-stepper temperature-stepper">
      <button disabled={value <= 0} onClick={() => onChange(Math.max(0, Math.round((value - 0.1) * 10) / 10))}>−</button>
      <input
        aria-label="Temperature"
        type="number"
        min="0"
        max="5"
        step="0.1"
        value={input}
        onChange={(event) => {
          setInput(event.target.value);
          const parsed = Number(event.target.value);
          if (event.target.value !== "" && Number.isFinite(parsed) && parsed >= 0 && parsed <= 5) onChange(Math.round(parsed * 10) / 10);
        }}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") (event.currentTarget as HTMLInputElement).blur();
        }}
      />
      <button disabled={value >= 5} onClick={() => onChange(Math.min(5, Math.round((value + 0.1) * 10) / 10))}>+</button>
    </div>
  );
}

export function AiSettingsScreen({ copy: t, models, mctsOptions, settings, onChange, onBack }: Props) {
  const roleNames: Record<AiRole, [string, string]> = {
    combat: [t.ai.combat, t.ai.combatDetail],
    hint: [t.ai.hint, t.ai.hintDetail],
    winRate: [t.ai.winRate, t.ai.winRateDetail],
  };

  return (
    <PageShell copy={t} title={t.ai.title} subtitle={t.ai.subtitle} onBack={onBack} wide>
      <div className="ai-columns">
        {ROLE_ORDER.map((role, roleIndex) => {
          const config = settings[role];
          const mctsIndex = Math.max(0, mctsOptions.indexOf(config.mcts_sims));
          return (
            <article className={`ai-role role-${role}`} key={role}>
              <header>
                <span className="role-number">0{roleIndex + 1}</span>
                <div><h2>{roleNames[role][0]}</h2><p>{roleNames[role][1]}</p></div>
              </header>
              <fieldset className="model-list">
                <legend>{t.ai.model}</legend>
                {models.map((model) => (
                  <label className={`model-option ${!model.available ? "unavailable" : ""}`} key={model.id}>
                    <input
                      type="radio"
                      name={`model-${role}`}
                      value={model.id}
                      checked={config.model_id === model.id}
                      disabled={!model.available}
                      onChange={() => onChange(role, { ...config, model_id: model.id })}
                    />
                    <span className="radio-dot" />
                    <span className="model-copy">
                      <strong>{model.display_name}</strong>
                    </span>
                    {!model.available && <em>{model.model_path ? t.common.unavailable : t.ai.placeholder}</em>}
                  </label>
                ))}
              </fieldset>
              <div className="parameter-block">
                <label>{t.ai.mcts}</label>
                <div className="number-stepper">
                  <button
                    aria-label={`${role} MCTS minus`}
                    disabled={mctsIndex <= 0}
                    onClick={() => onChange(role, { ...config, mcts_sims: mctsOptions[mctsIndex - 1] })}
                  >−</button>
                  <output>{config.mcts_sims}</output>
                  <button
                    aria-label={`${role} MCTS plus`}
                    disabled={mctsIndex >= mctsOptions.length - 1}
                    onClick={() => onChange(role, { ...config, mcts_sims: mctsOptions[mctsIndex + 1] })}
                  >+</button>
                </div>
              </div>
              <div className="parameter-block">
                <label>{t.ai.temperature}</label>
                <TemperatureControl value={config.temperature} onChange={(temperature) => onChange(role, { ...config, temperature })} />
              </div>
            </article>
          );
        })}
      </div>
      <p className="session-note"><span>●</span>{t.ai.currentSession}</p>
    </PageShell>
  );
}
