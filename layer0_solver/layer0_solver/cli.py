from __future__ import annotations

import argparse
import json
import random
import sys

from .game import BLUE, DRAW, ONGOING, RED, Layer0State
from .native import NativeSolver
from .solver import StrongSolver


KNOWN_PATH: tuple[int | str, ...] = (
    13,
    9,
    8,
    18,
    14,
    12,
    20,
    2,
    7,
    "pass",
    19,
    1,
    25,
)


def parse_moves(text: str) -> tuple[int | str, ...]:
    parsed: list[int | str] = []
    for token in text.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append("pass" if token.lower() in {"pass", "p", "skip"} else int(token))
    return tuple(parsed)


def state_payload(state: Layer0State) -> dict[str, object]:
    outcome = state.outcome()
    return {
        "rows": state.rows(),
        "to_move": "red" if state.to_move == RED else "blue",
        "ply": state.ply,
        "invisible_turns": state.invisible_turns,
        "outcome": {
            ONGOING: "ongoing",
            DRAW: "draw",
            RED: "red_win",
            BLUE: "blue_win",
        }[outcome],
    }


def analyze(args: argparse.Namespace) -> int:
    state = Layer0State.from_moves(parse_moves(args.moves))
    payload = state_payload(state)
    if state.outcome() == ONGOING:
        with StrongSolver(seed=args.seed, timeout=args.root_timeout) as solver:
            result = solver.analyze(state)
        payload["analysis"] = {
            "value_for_side_to_move": result.label,
            "distance": result.distance,
            "optimal_moves": result.optimal_moves,
            "principal_move": result.principal_move,
            "nodes": result.nodes,
            "cache_hits": result.cache_hits,
            "proven": result.proven,
            "note": result.note,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def replay_known_path(_: argparse.Namespace) -> int:
    state = Layer0State()
    for turn, move in enumerate(KNOWN_PATH, start=1):
        state = state.pass_invisible() if move == "pass" else state.play(int(move))
        print(f"{turn:02d}. {str(move):>4} -> {state.outcome()}")
    if state.outcome() != RED:
        print("Known path did not end in a red win", file=sys.stderr)
        return 1
    print("Verified: red wins on 7-13-19-25.")
    return 0


def self_play(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    games: list[dict[str, object]] = []
    with StrongSolver(seed=args.seed, timeout=args.root_timeout) as solver:
        for game_index in range(args.games):
            state = Layer0State()
            history: list[int] = []
            proof_steps = 0
            while state.occupied.bit_count() < 25 and state.outcome(stop_when_dead=False) == ONGOING:
                if state.outcome(stop_when_dead=True) == DRAW:
                    move = rng.choice(state.legal_positions)
                else:
                    analysis = solver.analyze(state)
                    if analysis.outcome != DRAW:
                        raise RuntimeError(
                            f"exact self-play left draw value at game {game_index + 1}, "
                            f"ply {state.ply}: {analysis.label}"
                        )
                    assert analysis.principal_move is not None
                    move = analysis.principal_move
                    proof_steps += 1
                history.append(move)
                state = state.play(move, allow_after_dead_draw=True)
            games.append(
                {
                    "game": game_index + 1,
                    "moves": history,
                    "proof_steps": proof_steps,
                    **state_payload(state),
                }
            )
            print(f"game={game_index + 1} plies={len(history)} outcome={state_payload(state)['outcome']}")
    print(json.dumps({"seed": args.seed, "games": games}, ensure_ascii=False, indent=2))
    return 0 if all(game["outcome"] == "draw" for game in games) else 2


def solve_root(args: argparse.Namespace) -> int:
    solver = NativeSolver(timeout=args.root_timeout)
    result = solver.analyze(Layer0State())
    print(
        json.dumps(
            {
                "value": {1: "win", 0: "draw", -1: "loss"}[result.outcome],
                "optimal_moves": result.optimal_moves,
                "nodes": result.nodes,
                "cache_hits": result.cache_hits,
                "cache_size": result.cache_size,
                "seconds": result.seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.outcome == DRAW else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5x5 Layer0 exact solver")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root-timeout", type=float, default=120.0)
    parser.add_argument("--games", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--moves", default="")
    analyze_parser.set_defaults(handler=analyze)

    commands.add_parser("replay-known-path").set_defaults(handler=replay_known_path)
    commands.add_parser("self-play").set_defaults(handler=self_play)
    commands.add_parser("solve-root").set_defaults(handler=solve_root)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
