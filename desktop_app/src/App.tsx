import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AiSettingsScreen } from "./components/AiSettingsScreen";
import { GameScreen } from "./components/GameScreen";
import { InstructionsScreen } from "./components/InstructionsScreen";
import { LanguageToggle } from "./components/LanguageToggle";
import { MenuScreen } from "./components/MenuScreen";
import { SettingsScreen } from "./components/SettingsScreen";
import { translations } from "./i18n";
import { BackendRequestError, sidecarBackend } from "./lib/backend";
import type {
  AiConfig,
  AiRole,
  AiSettings,
  BackendApi,
  GameMode,
  GameState,
  HintResult,
  InitializationResult,
  Language,
  ModelInfo,
  Move,
  Player,
  Screen,
  WinRateResult,
} from "./types";

const DEFAULT_AI: AiSettings = {
  combat: { model_id: "v2.2_balance", mcts_sims: 128, temperature: 1.0 },
  hint: { model_id: "v2.2_balance", mcts_sims: 128, temperature: 1.0 },
  winRate: { model_id: "v2.2_balance", mcts_sims: 128, temperature: 1.0 },
};

function analysisKey(state: GameState, config: AiConfig): string {
  return `${state.session_id}:${state.revision}:${config.model_id}:${config.mcts_sims}:${config.temperature.toFixed(1)}`;
}

function isStaleError(error: unknown): boolean {
  return error instanceof BackendRequestError && (error.code === "STALE_REVISION" || error.code === "STALE_SESSION");
}

interface AppProps { backend?: BackendApi }

export function App({ backend = sidecarBackend }: AppProps) {
  const [language, setLanguage] = useState<Language>("zh");
  const [screen, setScreen] = useState<Screen>("loading");
  const screenRef = useRef<Screen>("loading");
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [initError, setInitError] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [mctsOptions, setMctsOptions] = useState([32, 64, 128, 256, 512, 1024]);
  const [aiSettings, setAiSettings] = useState<AiSettings>(DEFAULT_AI);
  const [preloadHint, setPreloadHint] = useState(false);
  const [sideChoiceOpen, setSideChoiceOpen] = useState(false);
  const [game, setGame] = useState<GameState | null>(null);
  const gameRef = useRef<GameState | null>(null);
  const [menuBusy, setMenuBusy] = useState(false);
  const [mutationBusy, setMutationBusy] = useState(false);
  const [combatThinking, setCombatThinking] = useState(false);
  const combatToken = useRef(0);
  const [visibleHint, setVisibleHint] = useState<HintResult | null>(null);
  const hintCache = useRef(new Map<string, HintResult>());
  const hintInflight = useRef(new Map<string, Promise<HintResult>>());
  const [hintPending, setHintPending] = useState(new Set<string>());
  const [hintCacheVersion, setHintCacheVersion] = useState(0);
  const [winRate, setWinRate] = useState<WinRateResult | null>(null);
  const [winRateThinking, setWinRateThinking] = useState(false);
  const winRateToken = useRef(0);
  const [toast, setToast] = useState("");
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const t = translations[language];

  const navigate = useCallback((next: Screen) => {
    screenRef.current = next;
    setScreen(next);
  }, []);

  const showToast = useCallback((message: string) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToast(message);
    toastTimer.current = setTimeout(() => setToast(""), 4500);
  }, []);

  const reportError = useCallback((error: unknown) => {
    if (isStaleError(error)) return;
    if (error instanceof BackendRequestError && error.code === "MODEL_UNAVAILABLE") showToast(translations[language].errors.modelUnavailable);
    else showToast(error instanceof Error && error.message ? error.message : translations[language].errors.generic);
  }, [language, showToast]);

  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);

  useEffect(() => {
    let cancelled = false;
    setInitError("");
    navigate("loading");
    backend.start().then((result: InitializationResult) => {
      if (cancelled) return;
      setModels(result.models);
      setMctsOptions(result.mcts_options);
      const preferred = result.models.find((model) => model.id === "v2.2_balance" && model.available)
        ?? result.models.find((model) => model.available);
      if (preferred) {
        setAiSettings((current) => Object.fromEntries(
          (Object.keys(current) as AiRole[]).map((role) => {
            const selected = result.models.find((model) => model.id === current[role].model_id);
            return [role, selected?.available ? current[role] : { ...current[role], model_id: preferred.id }];
          }),
        ) as AiSettings);
      }
      navigate("menu");
    }).catch((error: unknown) => {
      if (cancelled) return;
      setInitError(error instanceof Error ? error.message : String(error));
    });
    return () => {
      cancelled = true;
      void backend.close();
    };
  }, [backend, loadAttempt, navigate]);

  useEffect(() => () => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
  }, []);

  const commitState = useCallback((next: GameState) => {
    gameRef.current = next;
    setGame(next);
    setVisibleHint(null);
    setWinRate(null);
    winRateToken.current += 1;
    setWinRateThinking(false);
  }, []);

  const runCombatAi = useCallback(async (position: GameState) => {
    if (position.mode !== "pvai" || position.status !== "playing" || position.current_player === position.human_player) return;
    const token = ++combatToken.current;
    setCombatThinking(true);
    try {
      const next = await backend.request<GameState>("game.ai_move", {
        session_id: position.session_id,
        expected_revision: position.revision,
        ai: aiSettings.combat,
      });
      if (token === combatToken.current && screenRef.current === "game") commitState(next);
    } catch (error) {
      if (token === combatToken.current) reportError(error);
    } finally {
      if (token === combatToken.current) setCombatThinking(false);
    }
  }, [aiSettings.combat, backend, commitState, reportError]);

  const startGame = useCallback(async (mode: GameMode, humanPlayer: Player = 1) => {
    setMenuBusy(true);
    combatToken.current += 1;
    try {
      const next = await backend.request<GameState>("game.new", { mode, human_player: humanPlayer });
      commitState(next);
      setSideChoiceOpen(false);
      navigate("game");
      await runCombatAi(next);
    } catch (error) {
      reportError(error);
    } finally {
      setMenuBusy(false);
    }
  }, [backend, commitState, navigate, reportError, runCombatAi]);

  const handleMove = useCallback(async (move: Move) => {
    const position = gameRef.current;
    if (!position || mutationBusy || combatThinking) return;
    setMutationBusy(true);
    try {
      const next = await backend.request<GameState>("game.move", {
        session_id: position.session_id,
        expected_revision: position.revision,
        layer: move.layer,
        row: move.row,
        col: move.col,
      });
      commitState(next);
      await runCombatAi(next);
    } catch (error) {
      reportError(error);
    } finally {
      setMutationBusy(false);
    }
  }, [backend, combatThinking, commitState, mutationBusy, reportError, runCombatAi]);

  const mutateGame = useCallback(async (command: "game.undo" | "game.restart", autoAi: boolean) => {
    const position = gameRef.current;
    if (!position || mutationBusy || combatThinking) return;
    combatToken.current += 1;
    setMutationBusy(true);
    try {
      const next = await backend.request<GameState>(command, {
        session_id: position.session_id,
        expected_revision: position.revision,
      });
      commitState(next);
      if (autoAi) await runCombatAi(next);
    } catch (error) {
      reportError(error);
    } finally {
      setMutationBusy(false);
    }
  }, [backend, combatThinking, commitState, mutationBusy, reportError, runCombatAi]);

  const getHint = useCallback((position: GameState, reveal: boolean): Promise<HintResult> => {
    const config = aiSettings.hint;
    const key = analysisKey(position, config);
    const cached = hintCache.current.get(key);
    if (cached) {
      if (reveal && gameRef.current?.session_id === position.session_id && gameRef.current.revision === position.revision) setVisibleHint(cached);
      return Promise.resolve(cached);
    }
    let request = hintInflight.current.get(key);
    if (!request) {
      setHintPending((current) => new Set(current).add(key));
      request = backend.request<HintResult>("analysis.hint", {
        session_id: position.session_id,
        expected_revision: position.revision,
        ai: config,
      }).then((result) => {
        if (result.for_revision === position.revision) {
          hintCache.current.set(key, result);
          setHintCacheVersion((version) => version + 1);
        }
        return result;
      }).finally(() => {
        hintInflight.current.delete(key);
        setHintPending((current) => {
          const next = new Set(current);
          next.delete(key);
          return next;
        });
      });
      hintInflight.current.set(key, request);
    }
    if (reveal) {
      return request.then((result) => {
        const current = gameRef.current;
        if (current?.session_id === position.session_id && current.revision === position.revision && analysisKey(current, aiSettings.hint) === key) setVisibleHint(result);
        return result;
      });
    }
    return request;
  }, [aiSettings.hint, backend, reportError]);

  useEffect(() => {
    hintCache.current.clear();
    setVisibleHint(null);
    setHintCacheVersion((version) => version + 1);
  }, [aiSettings.hint.model_id, aiSettings.hint.mcts_sims, aiSettings.hint.temperature]);

  useEffect(() => {
    const isHumanTurn = game?.mode !== "pvai" || game.current_player === game.human_player;
    if (!preloadHint || screen !== "game" || !game || game.status !== "playing" || !isHumanTurn) return;
    void getHint(game, false).catch((error) => {
      if (!isStaleError(error)) console.error("Hint preload failed", error);
    });
  }, [game, getHint, preloadHint, screen]);

  const requestVisibleHint = useCallback(() => {
    const position = gameRef.current;
    if (!position) return;
    void getHint(position, true).catch(reportError);
  }, [getHint, reportError]);

  const requestWinRate = useCallback(async () => {
    const position = gameRef.current;
    if (!position) return;
    const token = ++winRateToken.current;
    setWinRateThinking(true);
    try {
      const result = await backend.request<WinRateResult>("analysis.win_rate", {
        session_id: position.session_id,
        expected_revision: position.revision,
        ai: aiSettings.winRate,
      });
      const current = gameRef.current;
      if (token === winRateToken.current && current?.session_id === position.session_id && current.revision === result.for_revision) setWinRate(result);
    } catch (error) {
      if (token === winRateToken.current) reportError(error);
    } finally {
      if (token === winRateToken.current) setWinRateThinking(false);
    }
  }, [aiSettings.winRate, backend, reportError]);

  const exitGame = useCallback(() => {
    combatToken.current += 1;
    winRateToken.current += 1;
    setCombatThinking(false);
    setWinRateThinking(false);
    setVisibleHint(null);
    setWinRate(null);
    navigate("menu");
  }, [navigate]);

  const updateAi = useCallback((role: AiRole, next: AiConfig) => {
    setAiSettings((current) => ({ ...current, [role]: next }));
    if (role === "winRate") {
      winRateToken.current += 1;
      setWinRate(null);
      setWinRateThinking(false);
    }
  }, []);

  const currentHintKey = game ? analysisKey(game, aiSettings.hint) : "";
  const hintThinking = hintPending.has(currentHintKey);
  const hintPreloaded = Boolean(currentHintKey && hintCache.current.has(currentHintKey));
  void hintCacheVersion;

  const screenContent = useMemo(() => {
    if (screen === "loading") {
      return (
        <main className="loading-screen">
          <div className="loading-cube"><i /><i /><i /></div>
          <h1>{t.loading}</h1>
          <p>{initError ? `${t.errors.backend}: ${initError}` : t.loadingDetail}</p>
          {initError && <button className="retry-button" onClick={() => setLoadAttempt((attempt) => attempt + 1)}>{t.retry}</button>}
        </main>
      );
    }
    if (screen === "menu") {
      return <MenuScreen copy={t} sideChoiceOpen={sideChoiceOpen} busy={menuBusy} onPvp={() => void startGame("pvp")} onOpenPvai={() => setSideChoiceOpen(true)} onChooseSide={(player) => void startGame("pvai", player)} onCancelSide={() => setSideChoiceOpen(false)} onAiSettings={() => navigate("ai-settings")} onSettings={() => navigate("settings")} onInstructions={() => navigate("instructions")} />;
    }
    if (screen === "ai-settings") return <AiSettingsScreen copy={t} models={models} mctsOptions={mctsOptions} settings={aiSettings} onChange={updateAi} onBack={() => navigate("menu")} />;
    if (screen === "settings") return <SettingsScreen copy={t} preloadHint={preloadHint} onPreloadHint={setPreloadHint} onBack={() => navigate("menu")} />;
    if (screen === "instructions") return <InstructionsScreen copy={t} onBack={() => navigate("menu")} />;
    if (screen === "game" && game) {
      return <GameScreen copy={t} state={game} combatThinking={combatThinking} hintThinking={hintThinking} winRateThinking={winRateThinking} mutationBusy={mutationBusy} hint={visibleHint} hintPreloaded={preloadHint && hintPreloaded} winRate={winRate} onMove={(move) => void handleMove(move)} onUndo={() => void mutateGame("game.undo", false)} onRestart={() => void mutateGame("game.restart", true)} onHint={requestVisibleHint} onWinRate={() => void requestWinRate()} onExit={exitGame} />;
    }
    return null;
  }, [aiSettings, combatThinking, exitGame, game, handleMove, hintPreloaded, hintThinking, initError, language, menuBusy, mctsOptions, models, mutateGame, mutationBusy, navigate, preloadHint, requestVisibleHint, requestWinRate, screen, showToast, sideChoiceOpen, startGame, t, updateAi, visibleHint, winRate, winRateThinking]);

  return (
    <div className="app-frame">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" />
      <LanguageToggle language={language} onChange={setLanguage} />
      {screenContent}
      <div className={`toast ${toast ? "visible" : ""}`} role="status" aria-live="polite">
        <span>!</span>{toast}<button aria-label={t.common.close} onClick={() => setToast("")}>×</button>
      </div>
    </div>
  );
}
