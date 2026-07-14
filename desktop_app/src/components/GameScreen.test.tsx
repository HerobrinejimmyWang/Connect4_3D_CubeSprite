import { render, screen } from "@testing-library/react";

import { translations } from "../i18n";
import { emptyState } from "../test/fixtures";
import { GameScreen } from "./GameScreen";

it("trusts backend can_undo after a PvAI human move even before the AI responds", () => {
  render(
    <GameScreen
      copy={translations.zh}
      state={emptyState({ mode: "pvai", human_player: 1, current_player: -1, move_count: 1, can_undo: true })}
      combatThinking={false}
      hintThinking={false}
      winRateThinking={false}
      mutationBusy={false}
      hint={null}
      hintPreloaded={false}
      winRate={null}
      onMove={() => undefined}
      onUndo={() => undefined}
      onRestart={() => undefined}
      onHint={() => undefined}
      onWinRate={() => undefined}
      onExit={() => undefined}
      onSwitch3d={() => undefined}
    />,
  );

  expect(screen.getByRole("button", { name: "悔棋" })).toBeEnabled();
});
