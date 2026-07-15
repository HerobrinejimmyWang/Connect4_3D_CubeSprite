import type { Language } from "./types";

export interface Copy {
  appName: string;
  version: string;
  loading: string;
  loadingDetail: string;
  retry: string;
  menu: {
    pvp: string;
    pvpDetail: string;
    pvai: string;
    pvaiDetail: string;
    aiSettings: string;
    settings: string;
    instructions: string;
    chooseSide: string;
    redFirst: string;
    redFirstDetail: string;
    blueSecond: string;
    blueSecondDetail: string;
    cancel: string;
  };
  common: {
    back: string;
    unavailable: string;
    comingSoon: string;
    soon: string;
    close: string;
  };
  ai: {
    title: string;
    subtitle: string;
    combat: string;
    combatDetail: string;
    hint: string;
    hintDetail: string;
    winRate: string;
    winRateDetail: string;
    model: string;
    mcts: string;
    temperature: string;
    currentSession: string;
    placeholder: string;
  };
  settings: {
    title: string;
    subtitle: string;
    preload: string;
    preloadOn: string;
    preloadOff: string;
    preloadDetail: string;
    future: string;
  };
  instructions: {
    title: string;
    subtitle: string;
    rulesTitle: string;
    rules: string[];
    controlsTitle: string;
    controls: string[];
    quickTitle: string;
    quick: string[];
  };
  game: {
    currentPlayer: string;
    result: string;
    red: string;
    blue: string;
    redWins: string;
    blueWins: string;
    draw: string;
    totalMoves: string;
    aiThinking: string;
    hintThinking: string;
    winRateThinking: string;
    floor: string;
    undo: string;
    restart: string;
    hint: string;
    winRate: string;
    exit: string;
    switch3d: string;
    legal: string;
    illegal: string;
    redRate: string;
    blueRate: string;
    preloadReady: string;
  };
  errors: {
    backend: string;
    generic: string;
    modelUnavailable: string;
  };
}

export const translations: Record<Language, Copy> = {
  zh: {
    appName: "Connect4 3D CubeSprite",
    version: "版本 0.1.0",
    loading: "正在唤醒 CubeSprite",
    loadingDetail: "正在启动本地规则与 AI 引擎…",
    retry: "重试",
    menu: {
      pvp: "玩家 vs 玩家",
      pvpDetail: "两位玩家在同一台电脑上轮流落子",
      pvai: "玩家 vs AI",
      pvaiDetail: "挑战本地运行的 CubeSprite AI",
      aiSettings: "AI 设置",
      settings: "设置",
      instructions: "游戏说明",
      chooseSide: "选择你的阵营",
      redFirst: "红方 · 先手",
      redFirstDetail: "你执红棋，并完成第一步",
      blueSecond: "蓝方 · 后手",
      blueSecondDetail: "你执蓝棋，AI 会自动先行",
      cancel: "取消",
    },
    common: { back: "返回主菜单", unavailable: "暂不可用", comingSoon: "3D 棋盘将在后续版本开放", soon: "后续开放", close: "关闭" },
    ai: {
      title: "AI 设置",
      subtitle: "三种任务可独立选择模型和搜索参数，修改会立刻应用。",
      combat: "对战 AI",
      combatDetail: "在人机对局中负责落子",
      hint: "提示 AI",
      hintDetail: "分析当前棋盘并推荐落点",
      winRate: "胜率 AI",
      winRateDetail: "估计红蓝双方当前胜率",
      model: "模型",
      mcts: "MCTS 模拟次数",
      temperature: "温度",
      currentSession: "设置仅在本次 App 运行期间保留，无需保存。",
      placeholder: "预占位",
    },
    settings: {
      title: "设置",
      subtitle: "调整棋局辅助功能。",
      preload: "预加载提示",
      preloadOn: "已开启",
      preloadOff: "已关闭",
      preloadDetail: "棋盘变化后在后台提前计算提示；点击“获取提示”时可直接显示已完成的结果。",
      future: "更多设置将在后续版本加入。",
    },
    instructions: {
      title: "游戏说明",
      subtitle: "一分钟认识 6 × 5 × 5 三维四字棋。",
      rulesTitle: "游戏规则",
      rules: [
        "棋盘由 F1 至 F6 六层组成，每层是 5 × 5 方格；红方先手，双方轮流落子。",
        "棋子受重力约束：同一行列位置必须从 F1 开始逐层堆叠，深色空格表示当前合法落点。",
        "率先在任意方向连成四子即获胜，包括同层直线、跨层直线和空间对角线；棋盘填满仍无人连四则为平局。",
      ],
      controlsTitle: "操作说明",
      controls: [
        "点击合法格落子。绿色预览表示鼠标当前指向的合法位置，黄色外圈表示上一步。",
        "悔棋在双人模式撤销一步；在人机模式会一起撤销最近的人类与 AI 回合。重新开始会保留当前模式和阵营。",
        "获取提示会用闪烁黄圈标出建议位置；胜率会以红蓝比例条展示本地 AI 的估计。",
      ],
      quickTitle: "迅速上手",
      quick: [
        "先观察每个小棋盘左上角的 F 层号，再寻找深色合法格。",
        "不要只盯住单层：竖直跨层和空间斜线同样能形成连四。",
        "黄色获胜棋子会标出终局路线；你可以立即重新开始或返回主菜单调整 AI。",
      ],
    },
    game: {
      currentPlayer: "当前玩家",
      result: "对局结果",
      red: "红方",
      blue: "蓝方",
      redWins: "红方获胜",
      blueWins: "蓝方获胜",
      draw: "平局",
      totalMoves: "总落子",
      aiThinking: "AI 正在思考…",
      hintThinking: "正在计算提示…",
      winRateThinking: "正在估算胜率…",
      floor: "F",
      undo: "悔棋",
      restart: "重新开始",
      hint: "获取提示",
      winRate: "胜率",
      exit: "退出对局",
      switch3d: "切换至 3D",
      legal: "合法落点",
      illegal: "暂不可落",
      redRate: "红方",
      blueRate: "蓝方",
      preloadReady: "提示已预加载",
    },
    errors: { backend: "本地游戏引擎未能启动", generic: "操作失败，请重试。", modelUnavailable: "所选 AI 模型暂不可用。" },
  },
  en: {
    appName: "Connect4 3D CubeSprite",
    version: "Version 0.1.0",
    loading: "Waking CubeSprite",
    loadingDetail: "Starting the local rules and AI engine…",
    retry: "Retry",
    menu: {
      pvp: "Player vs Player",
      pvpDetail: "Take turns on the same computer",
      pvai: "Player vs AI",
      pvaiDetail: "Challenge the locally running CubeSprite AI",
      aiSettings: "AI Settings",
      settings: "Settings",
      instructions: "Instructions",
      chooseSide: "Choose your side",
      redFirst: "Red · First",
      redFirstDetail: "Play red and make the opening move",
      blueSecond: "Blue · Second",
      blueSecondDetail: "Play blue while the AI opens",
      cancel: "Cancel",
    },
    common: { back: "Back to main menu", unavailable: "Unavailable", comingSoon: "The 3D board is coming in a future release", soon: "Coming soon", close: "Close" },
    ai: {
      title: "AI Settings",
      subtitle: "Choose models and search parameters independently for all three roles. Changes apply instantly.",
      combat: "Combat AI",
      combatDetail: "Makes moves during Player vs AI games",
      hint: "Hint AI",
      hintDetail: "Analyzes the board and recommends a move",
      winRate: "Win Rate AI",
      winRateDetail: "Estimates the current red/blue chances",
      model: "Model",
      mcts: "MCTS simulations",
      temperature: "Temperature",
      currentSession: "Settings last for this app session only. There is no save step.",
      placeholder: "Placeholder",
    },
    settings: {
      title: "Settings",
      subtitle: "Tune game assistance features.",
      preload: "Preload Hint",
      preloadOn: "On",
      preloadOff: "Off",
      preloadDetail: "Compute a hint in the background after every board change so a ready result appears immediately when requested.",
      future: "More settings will arrive in future versions.",
    },
    instructions: {
      title: "Instructions",
      subtitle: "Learn 6 × 5 × 5 Connect4 in one minute.",
      rulesTitle: "Game rules",
      rules: [
        "The board has six floors, F1–F6, each containing a 5 × 5 grid. Red moves first and players alternate turns.",
        "Gravity applies: pieces in one row/column stack upward from F1. Dark empty cells are the currently legal moves.",
        "The first player to connect four in any direction wins—within a floor, between floors, or on a spatial diagonal. A full board without four is a draw.",
      ],
      controlsTitle: "Controls",
      controls: [
        "Click a legal cell to move. A green preview marks the hovered legal move; a yellow ring marks the latest move.",
        "Undo removes one move in PvP, or the latest human-and-AI pair in PvAI. Restart keeps the current mode and side.",
        "Get Hint shows a blinking gold suggestion. Win Rate displays the local AI estimate as a red/blue bar.",
      ],
      quickTitle: "Quick start",
      quick: [
        "Check the F-number on each board, then look for the darker legal cells.",
        "Think beyond one floor: vertical stacks and spatial diagonals can both connect four.",
        "Gold winning pieces reveal the final line. Restart immediately, or exit and tune the AI settings.",
      ],
    },
    game: {
      currentPlayer: "Current player",
      result: "Game result",
      red: "Red",
      blue: "Blue",
      redWins: "Red wins",
      blueWins: "Blue wins",
      draw: "Draw",
      totalMoves: "Total moves",
      aiThinking: "AI is thinking…",
      hintThinking: "Computing hint…",
      winRateThinking: "Estimating win rate…",
      floor: "F",
      undo: "Undo",
      restart: "Restart",
      hint: "Get Hint",
      winRate: "Win Rate",
      exit: "Exit",
      switch3d: "Switch to 3D",
      legal: "Legal move",
      illegal: "Not playable",
      redRate: "Red",
      blueRate: "Blue",
      preloadReady: "Hint preloaded",
    },
    errors: { backend: "The local game engine could not start", generic: "The action failed. Please try again.", modelUnavailable: "The selected AI model is unavailable." },
  },
};
