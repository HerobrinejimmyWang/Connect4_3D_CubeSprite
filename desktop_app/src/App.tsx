import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AiSettingsScreen } from "./components/AiSettingsScreen";
import { GameScreen } from "./components/GameScreen";
import { InstructionsScreen } from "./components/InstructionsScreen";
import { LanguageToggle } from "./components/LanguageToggle";
import { MenuScreen } from "./components/MenuScreen";
import { ReplayScreen } from "./components/ReplayScreen";
import { SettingsScreen } from "./components/SettingsScreen";
import { translations } from "./i18n";
import { BackendRequestError, sidecarBackend } from "./lib/backend";
import type {
  AiConfig,
  AiRole,
  AiSettings,
  AutoplayInterval,
  BackendApi,
  GameMode,
  GameState,
  HintResult,
  InitializationResult,
  Language,
  MenuPanel,
  ModelInfo,
  Move,
  Player,
  ReplayListResult,
  ReplayMutationResult,
  ReplayExportResult,
  ReplayOpenResult,
  ReplaySummary,
  Screen,
  WinRateAnalysis,
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

function readFileBytes(file: File): Promise<ArrayBuffer> {
  if (typeof file.arrayBuffer === "function") return file.arrayBuffer();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Unable to read replay file."));
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) resolve(reader.result);
      else reject(new Error("Unable to read replay file."));
    };
    reader.readAsArrayBuffer(file);
  });
}

async function readFileText(file: File, invalidEncodingMessage: string): Promise<string> {
  const bytes = await readFileBytes(file);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(invalidEncodingMessage);
  }
}

function downloadReplayFile(filename: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: "application/json;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.append(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(url);
  }
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
  const [autoplayIntervalMs, setAutoplayIntervalMs] = useState<AutoplayInterval>(1000);
  const [activeMenuPanel, setActiveMenuPanel] = useState<MenuPanel>(null);
  const [game, setGame] = useState<GameState | null>(null);
  const gameRef = useRef<GameState | null>(null);
  const [menuBusy, setMenuBusy] = useState(false);
  const gameStartBusy = useRef(false);
  const gameStartToken = useRef(0);
  const [mutationBusy, setMutationBusy] = useState(false);
  const mutationToken = useRef(0);
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
  const [saveReplayThinking, setSaveReplayThinking] = useState(false);
  const [replays, setReplays] = useState<ReplaySummary[]>([]);
  const [replayListBusy, setReplayListBusy] = useState(false);
  const [replayImportBusy, setReplayImportBusy] = useState(false);
  const [replayDeleteBusyId, setReplayDeleteBusyId] = useState<string | null>(null);
  const [replayExportBusyId, setReplayExportBusyId] = useState<string | null>(null);
  const [activeReplay, setActiveReplay] = useState<ReplayOpenResult | null>(null);
  const activeReplayRef = useRef<ReplayOpenResult | null>(null);
  const replayOpenToken = useRef(0);
  const replayAnalysisToken = useRef(0);
  const replayContinueToken = useRef(0);
  const [replayAnalysisThinking, setReplayAnalysisThinking] = useState(false);
  const [replayContinueBusy, setReplayContinueBusy] = useState(false);
  const [toast, setToast] = useState("");
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const t = translations[language];

  const navigate = useCallback((next: Screen) => {
    if (next !== "replay") replayOpenToken.current += 1;
    if (next !== "game") {
      gameStartToken.current += 1;
      gameStartBusy.current = false;
      setMenuBusy(false);
    }
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

  const commitReplay = useCallback((next: ReplayOpenResult | null) => {
    activeReplayRef.current = next;
    setActiveReplay(next);
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
    if (gameStartBusy.current) return;
    gameStartBusy.current = true;
    const token = ++gameStartToken.current;
    replayOpenToken.current += 1;
    mutationToken.current += 1;
    setMenuBusy(true);
    combatToken.current += 1;
    try {
      const next = await backend.request<GameState>("game.new", { mode, human_player: humanPlayer });
      if (token !== gameStartToken.current || screenRef.current !== "menu") return;
      commitState(next);
      setActiveMenuPanel(null);
      navigate("game");
      await runCombatAi(next);
    } catch (error) {
      if (token === gameStartToken.current && screenRef.current === "menu") reportError(error);
    } finally {
      if (token === gameStartToken.current) {
        gameStartBusy.current = false;
        setMenuBusy(false);
      }
    }
  }, [backend, commitState, navigate, reportError, runCombatAi]);

  const upsertReplay = useCallback((replay: ReplaySummary) => {
    setReplays((current) => [replay, ...current.filter((item) => item.id !== replay.id)]);
  }, []);

  const refreshReplays = useCallback(async () => {
    setReplayListBusy(true);
    try {
      const result = await backend.request<ReplayListResult>("replay.list");
      setReplays(result.replays);
    } catch (error) {
      reportError(error);
    } finally {
      setReplayListBusy(false);
    }
  }, [backend, reportError]);

  const openReplayPanel = useCallback(() => {
    replayOpenToken.current += 1;
    setActiveMenuPanel("replays");
    void refreshReplays();
  }, [refreshReplays]);

  const openPvaiPanel = useCallback(() => {
    replayOpenToken.current += 1;
    setReplayListBusy(false);
    setActiveMenuPanel("side");
  }, []);

  const closeMenuPanel = useCallback(() => {
    replayOpenToken.current += 1;
    setReplayListBusy(false);
    setActiveMenuPanel(null);
  }, []);

  const saveReplay = useCallback(async () => {
    const position = gameRef.current;
    if (!position || saveReplayThinking) return;
    setSaveReplayThinking(true);
    try {
      const result = await backend.request<ReplayMutationResult>("replay.save", {
        session_id: position.session_id,
        expected_revision: position.revision,
      });
      upsertReplay(result.replay);
      showToast(translations[language].replay.saved.replace("{name}", result.replay.name));
    } catch (error) {
      reportError(error);
    } finally {
      setSaveReplayThinking(false);
    }
  }, [backend, language, reportError, saveReplayThinking, showToast, upsertReplay]);

  const importReplay = useCallback(async (file: File) => {
    if (replayImportBusy) return;
    if (file.size > 512 * 1024) {
      showToast(translations[language].replay.fileTooLarge);
      return;
    }
    setReplayImportBusy(true);
    try {
      const content = await readFileText(file, translations[language].replay.invalidUtf8);
      const result = await backend.request<ReplayMutationResult>("replay.import", {
        content,
        filename: file.name,
      });
      upsertReplay(result.replay);
      showToast(translations[language].replay.imported.replace("{name}", result.replay.name));
    } catch (error) {
      reportError(error);
    } finally {
      setReplayImportBusy(false);
    }
  }, [backend, language, replayImportBusy, reportError, showToast, upsertReplay]);

  const deleteReplay = useCallback(async (replay: ReplaySummary) => {
    if (replayDeleteBusyId) return;
    setReplayDeleteBusyId(replay.id);
    try {
      await backend.request<{ deleted: boolean }>("replay.delete", {
        id: replay.id,
        expected_fingerprint: replay.fingerprint,
      });
      setReplays((current) => current.filter((item) => item.id !== replay.id));
      showToast(translations[language].replay.deleted.replace("{name}", replay.name));
    } catch (error) {
      reportError(error);
    } finally {
      setReplayDeleteBusyId(null);
    }
  }, [backend, language, replayDeleteBusyId, reportError, showToast]);

  const exportReplay = useCallback(async (replay: ReplaySummary) => {
    if (replayExportBusyId) return;
    setReplayExportBusyId(replay.id);
    try {
      const result = await backend.request<ReplayExportResult>("replay.export", {
        id: replay.id,
        expected_fingerprint: replay.fingerprint,
      });
      downloadReplayFile(result.filename, result.content);
      showToast(translations[language].replay.exported.replace("{name}", replay.name));
    } catch (error) {
      reportError(error);
    } finally {
      setReplayExportBusyId(null);
    }
  }, [backend, language, replayExportBusyId, reportError, showToast]);

  const openReplay = useCallback(async (replay: ReplaySummary) => {
    const token = ++replayOpenToken.current;
    replayAnalysisToken.current += 1;
    replayContinueToken.current += 1;
    setReplayListBusy(true);
    try {
      const result = await backend.request<ReplayOpenResult>("replay.open", {
        id: replay.id,
        expected_fingerprint: replay.fingerprint,
      });
      if (token !== replayOpenToken.current) return;
      commitReplay(result);
      setActiveMenuPanel(null);
      navigate("replay");
    } catch (error) {
      if (token === replayOpenToken.current) reportError(error);
    } finally {
      if (token === replayOpenToken.current) setReplayListBusy(false);
    }
  }, [backend, commitReplay, navigate, reportError]);

  const analyzeReplay = useCallback(async () => {
    const opened = activeReplayRef.current;
    if (!opened || replayAnalysisThinking) return;
    const token = ++replayAnalysisToken.current;
    const replayId = opened.replay.id;
    setReplayAnalysisThinking(true);
    try {
      const analysis = await backend.request<WinRateAnalysis>("replay.analyze", {
        id: replayId,
        expected_fingerprint: opened.replay.fingerprint,
        ai: aiSettings.winRate,
      });
      const current = activeReplayRef.current;
      if (token !== replayAnalysisToken.current || screenRef.current !== "replay" || current?.replay.id !== replayId) return;
      if (analysis.replay_fingerprint !== current.replay.fingerprint) {
        throw new Error(translations[language].errors.replayChanged);
      }
      commitReplay({ ...current, analysis });
    } catch (error) {
      if (token === replayAnalysisToken.current) reportError(error);
    } finally {
      if (token === replayAnalysisToken.current) setReplayAnalysisThinking(false);
    }
  }, [aiSettings.winRate, backend, commitReplay, language, replayAnalysisThinking, reportError]);

  const continueReplay = useCallback(async (step: number, mode: GameMode, humanPlayer: Player = 1) => {
    const opened = activeReplayRef.current;
    if (!opened || replayContinueBusy) return;
    const token = ++replayContinueToken.current;
    const replayId = opened.replay.id;
    replayAnalysisToken.current += 1;
    setReplayAnalysisThinking(false);
    setReplayContinueBusy(true);
    try {
      const next = await backend.request<GameState>("replay.continue", {
        id: replayId,
        expected_fingerprint: opened.replay.fingerprint,
        step,
        mode,
        human_player: humanPlayer,
      });
      if (
        token !== replayContinueToken.current
        || screenRef.current !== "replay"
        || activeReplayRef.current?.replay.id !== replayId
      ) return;
      commitState(next);
      commitReplay(null);
      navigate("game");
      await runCombatAi(next);
    } catch (error) {
      if (token === replayContinueToken.current && screenRef.current === "replay") reportError(error);
    } finally {
      if (token === replayContinueToken.current) setReplayContinueBusy(false);
    }
  }, [backend, commitReplay, commitState, navigate, replayContinueBusy, reportError, runCombatAi]);

  const exitReplay = useCallback(() => {
    replayOpenToken.current += 1;
    replayAnalysisToken.current += 1;
    replayContinueToken.current += 1;
    setReplayAnalysisThinking(false);
    setReplayContinueBusy(false);
    commitReplay(null);
    navigate("menu");
  }, [commitReplay, navigate]);

  const handleMove = useCallback(async (move: Move) => {
    const position = gameRef.current;
    if (!position || mutationBusy || combatThinking) return;
    const token = ++mutationToken.current;
    const sessionId = position.session_id;
    setMutationBusy(true);
    try {
      const next = await backend.request<GameState>("game.move", {
        session_id: position.session_id,
        expected_revision: position.revision,
        layer: move.layer,
        row: move.row,
        col: move.col,
      });
      if (
        token !== mutationToken.current
        || screenRef.current !== "game"
        || gameRef.current?.session_id !== sessionId
      ) return;
      commitState(next);
      await runCombatAi(next);
    } catch (error) {
      if (token === mutationToken.current && screenRef.current === "game") reportError(error);
    } finally {
      if (token === mutationToken.current) setMutationBusy(false);
    }
  }, [backend, combatThinking, commitState, mutationBusy, reportError, runCombatAi]);

  const mutateGame = useCallback(async (command: "game.undo" | "game.restart", autoAi: boolean) => {
    const position = gameRef.current;
    if (!position || mutationBusy || combatThinking) return;
    const token = ++mutationToken.current;
    const sessionId = position.session_id;
    combatToken.current += 1;
    setMutationBusy(true);
    try {
      const next = await backend.request<GameState>(command, {
        session_id: position.session_id,
        expected_revision: position.revision,
      });
      if (
        token !== mutationToken.current
        || screenRef.current !== "game"
        || gameRef.current?.session_id !== sessionId
      ) return;
      commitState(next);
      if (autoAi) await runCombatAi(next);
    } catch (error) {
      if (token === mutationToken.current && screenRef.current === "game") reportError(error);
    } finally {
      if (token === mutationToken.current) setMutationBusy(false);
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
    mutationToken.current += 1;
    combatToken.current += 1;
    winRateToken.current += 1;
    setCombatThinking(false);
    setMutationBusy(false);
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
      return (
        <MenuScreen
          copy={t}
          activePanel={activeMenuPanel}
          busy={menuBusy}
          replays={replays}
          replayListBusy={replayListBusy}
          replayImportBusy={replayImportBusy}
          replayDeleteBusyId={replayDeleteBusyId}
          replayExportBusyId={replayExportBusyId}
          onPvp={() => void startGame("pvp")}
          onOpenPvai={openPvaiPanel}
          onOpenReplays={openReplayPanel}
          onChooseSide={(player) => void startGame("pvai", player)}
          onClosePanel={closeMenuPanel}
          onOpenReplay={(replay) => void openReplay(replay)}
          onDeleteReplay={(replay) => void deleteReplay(replay)}
          onExportReplay={(replay) => void exportReplay(replay)}
          onImportReplay={(file) => void importReplay(file)}
          onAiSettings={() => navigate("ai-settings")}
          onSettings={() => navigate("settings")}
          onInstructions={() => navigate("instructions")}
        />
      );
    }
    if (screen === "ai-settings") return <AiSettingsScreen copy={t} models={models} mctsOptions={mctsOptions} settings={aiSettings} onChange={updateAi} onBack={() => navigate("menu")} />;
    if (screen === "settings") return <SettingsScreen copy={t} preloadHint={preloadHint} onPreloadHint={setPreloadHint} autoplayIntervalMs={autoplayIntervalMs} onAutoplayInterval={setAutoplayIntervalMs} onBack={() => navigate("menu")} />;
    if (screen === "instructions") return <InstructionsScreen copy={t} onBack={() => navigate("menu")} />;
    if (screen === "game" && game) {
      return <GameScreen copy={t} state={game} combatThinking={combatThinking} hintThinking={hintThinking} winRateThinking={winRateThinking} saveReplayThinking={saveReplayThinking} mutationBusy={mutationBusy} hint={visibleHint} hintPreloaded={preloadHint && hintPreloaded} winRate={winRate} onMove={(move) => void handleMove(move)} onUndo={() => void mutateGame("game.undo", false)} onRestart={() => void mutateGame("game.restart", true)} onHint={requestVisibleHint} onWinRate={() => void requestWinRate()} onSaveReplay={() => void saveReplay()} onExit={exitGame} />;
    }
    if (screen === "replay" && activeReplay) {
      return <ReplayScreen key={activeReplay.replay.id} copy={t} replay={activeReplay} autoplayIntervalMs={autoplayIntervalMs} analysisThinking={replayAnalysisThinking} continueBusy={replayContinueBusy} onAnalyze={() => void analyzeReplay()} onContinue={(step, mode, humanPlayer) => void continueReplay(step, mode, humanPlayer)} onExit={exitReplay} />;
    }
    return null;
  }, [activeMenuPanel, activeReplay, aiSettings, analyzeReplay, autoplayIntervalMs, closeMenuPanel, combatThinking, continueReplay, deleteReplay, exitGame, exitReplay, exportReplay, game, handleMove, hintPreloaded, hintThinking, importReplay, initError, menuBusy, mctsOptions, models, mutateGame, mutationBusy, navigate, openPvaiPanel, openReplay, openReplayPanel, preloadHint, replayAnalysisThinking, replayContinueBusy, replayDeleteBusyId, replayExportBusyId, replayImportBusy, replayListBusy, replays, requestVisibleHint, requestWinRate, saveReplay, saveReplayThinking, screen, startGame, t, updateAi, visibleHint, winRate, winRateThinking]);

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
