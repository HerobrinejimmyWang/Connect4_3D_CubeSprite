# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python 3D Connect4 AI training and evaluation stack.
The isolated foundation for all new training work is `training/v3/`. Match and
replay tooling lives in `arena/`, including `main_arena.py`, UI launch code,
model discovery, and history handling. Distillation workflows live in
`distillation/`; compact feature-based policy training lives in
`train_features/`; human evaluation scripts and saved play histories live in
`test/`. Utility scripts are in `tools/`.

The repository also contains a substantial legacy training stack and historical
artifacts under `training/`, `training/checkpoints/`, `save_model/`,
`other_models/`, and old benchmark/result paths. Their presence does not make
them current training inputs or defaults.

## Training Lineage and Source of Truth

These rules are mandatory for future training work:

- Use `training/v3/` and its explicit JSON configs for every new self-play,
  learner, replay, gate, checkpoint, retention, or hardware-topology change.
- Treat `training/main_train.py`, `training/mcts.py`, `training/model.py`,
  `training/parallel_games.py`, their implicit configs, and historical
  checkpoints as **Legacy**. They may be used only for an explicitly requested
  reproduction, compatibility test, Arena opponent, migration test, or
  historical comparison. They are not fallback implementations for V3.
- `tools/benchmark_selfplay_topology.py` and
  `tools/benchmark_thread_match.py` exercise the Legacy stack. Any future V3
  benchmark must import only `training/v3/` and record explicit V3 evidence;
  do not use these Legacy tools to justify V3 settings.
- Do not infer V3 settings from a similarly named old file, an old cloud
  command, a checkpoint directory, a TODO, or an unlabelled JSON/log. A value
  enters V3 only through an explicit V3 config or documented code contract with
  a focused test and evidence.
- Do not load Legacy checkpoint weights into a V3 model or continue a Legacy
  run as a V3 lineage. A cross-lineage conversion requires a dedicated,
  validated converter and explicit user approval. A Legacy model may still be
  evaluated as an external opponent without becoming the V3 accepted model.
- Do not modify Legacy training code to solve a V3 problem. Put shared game
  rules in `connect4_core/` when genuinely shared; put new training behavior in
  `training/v3/`.
- Do not delete Legacy yet merely because it is not the formal path. Arena,
  distillation, feature training, exports, or compatibility tools may still
  import it. First inventory dependents, migrate them, and run their focused
  regressions. Existing backups do not replace dependency verification.

When documentation conflicts, prefer executable V3 contracts and verified V3
evidence in this order: current tests and code, current V3 config, current
`training/v3/README.md` and `STATIC_AUDIT.md`, then dated benchmark artifacts.
Legacy runbooks and old experiment logs are historical context only.

## Current V3 Status and Guardrails

- The deterministic CPU `smoke` path is executable. The `run` command is still
  plan-only and must not be presented as starting formal training.
- Do not enable unattended/formal training until the executable multi-generation
  scheduler is connected to cumulative replay and candidate/gate state, safe
  drain/shutdown is integrated, a generation draft/reconcile journal and
  single-coordinator lock exist, and archive receipts/disk-watermark handling
  are connected and tested.
- Self-play may use only the committed accepted champion. Candidate and rejected
  models must never produce replay. An inconclusive gate extends the same paired
  opening evidence; it does not silently accept a candidate or restart the gate.
- AMP overflow is a skipped optimizer step. It must not consume replay tokens,
  advance the sample/global cursor, or advance the LR scheduler. Preserve the
  existing retry semantics and regression coverage.
- `inference_batch_size` is a hard shared-service limit. Do not reintroduce
  whole-request overflow. MCTS lane count is semantic because virtual-loss
  batching changes search targets; actor count and other operational topology
  values must remain recorded but excluded from semantic model lineage.
- The 2026-08-16 CUDA run validates a 64x4 short learning loop, not final
  playing strength or a production schedule. Its random-bootstrap game lengths
  are not comparable to the historical 27-29-ply trained CubeSprite champion.
- For the exact 64x4 pilot on one RTX 3080 Ti with a 20-vCPU/30-GiB cgroup quota
  at 128/32 full/fast simulations, the provisional topology is 18 actors x 6
  lanes with batch 32. Treat 18-20 actors and 4-6 lanes as the local plateau.
  Do not copy this topology to the 128x6 Mini, a different GPU/CPU allocation,
  longer search, or a two-GPU machine without recalibration.
- Eight or more lanes increased raw throughput but lost serial-MCTS top-action
  agreement in the short fixed-position check. Do not optimize GPU utilization
  by increasing lanes past the quality-validated range without paired strength
  tests on a stable accepted champion.
- On the exact 2x RTX 3080 Ti, 40-vCPU, 60-GiB target, the bounded 2026-08-23
  calibration supports 24 actors x 4 lanes, batch 32, and a 1 ms batching
  timeout for one-card B4/B6 self-play while the other card is reserved for the
  learner. Paired random-weight checks found no result regression versus serial
  search, but repeat lane fidelity on a stable accepted champion before raising
  lanes. B8 remains uncalibrated. Keep DDP disabled unless measurements show
  the learner, rather than self-play, is the bottleneck.

The 2026-08-16 short-pilot conclusions above are summarized in the tracked V3
documentation and config. Raw benchmark outputs and copied cloud logs are local
evidence only and are intentionally excluded from source commits unless
explicitly requested. Their presence never promotes Legacy settings into V3.

## Build, Test, and Development Commands

Use the local virtual environment when available:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run lightweight syntax validation before committing:

```powershell
python -m compileall training arena distillation train_features test tools
```

Start current V3 and compatibility workflows from the repository root:

```powershell
python -B -m training.v3 print-config --config training\v3\configs\smoke_cpu.json
python -B -m training.v3 smoke --config training\v3\configs\smoke_cpu.json
python -B -m training.v3 run --config training\v3\configs\pilot_gpu_64x4.json
python distillation\main_distill.py --config distillation\distill_config.json --print-config
python arena\main_arena.py --black-random --white-random --games 1
python test\main_human_eval.py
python tools\export_model_pth.py save_model\v2.2_large\best.pth.tar
```

The V3 `run` command above prints a guarded plan; it does not authorize or start
formal training. `python training\main_train.py` is intentionally omitted
because it is a Legacy entry point. Run it only when the requested task is
explicitly scoped to Legacy reproduction or compatibility.
Distillation and historical checkpoint export may also depend on Legacy model
contracts; completing those compatibility workflows does not create a V3 model
or validate the V3 training chain.

Many scripts are GPU/CPU intensive; reduce iteration counts or use dry
configuration flags when checking changes. For cloud measurements, verify the
actual cgroup CPU and memory quotas, ensure the machine is uncontended, and
record exact model/search/topology settings. Host-wide `lscpu`, `free`, or
`vmstat` values can misrepresent a rented container's allocation.

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase constants such as `BOARD_SIZE`. Keep script entry points guarded with `if __name__ == "__main__":`, especially where multiprocessing is used. Prefer `pathlib.Path` for new path handling and keep imports explicit. Preserve existing local-import patterns unless converting a full package boundary.

## Testing Guidelines

There is no formal pytest suite configured. Treat `compileall` as the minimum
check. For V3 changes, also run:

```powershell
python -m unittest discover -s test -p "test_training_v3_*.py"
```

Then run the smallest relevant workflow: V3 CPU smoke for pipeline changes,
fixed-position search comparison for lane/virtual-loss changes, paired openings
with colors swapped for strength claims, random Arena games for shared game-rule
changes, `--print-config` for distillation config changes, and targeted
human-history commands for `train_features/`.

Do not label a configuration, checkpoint, throughput number, or strength claim
as verified merely because a script exited successfully. Record the exact code,
config, hardware/quota, seeds, sample size, and result artifact. Separate
functional effectiveness, playing strength, and hardware efficiency; none is a
proxy for the others. Avoid committing generated `__pycache__/`, large
experimental checkpoints, or ad hoc logs unless they are intentional,
documented artifacts.

## Commit & Pull Request Guidelines

Git history currently contains only `Initial code-only import`, so use concise imperative commits going forward, for example `Add tiny policy cache validation`. Pull requests should describe the affected workflow, list commands run, note hardware assumptions such as CUDA availability, and include screenshots or saved history paths for UI/evaluation changes.

## Agent-Specific Instructions

Keep changes scoped. Do not rewrite checkpoint directories or historical JSON
results unless asked. Preserve user changes in a dirty worktree. When adding
new generated outputs, document where they are produced, which lineage produced
them, whether they are verified, and whether they should be tracked. Never
silently convert archived Legacy evidence into a current V3 default.
