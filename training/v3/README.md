# V3.1 training foundation

`training.v3` is the isolated foundation for new Connect4 3D training runs.
Legacy `training/main_train.py` remains available for old experiments, but its
configuration and checkpoints are intentionally not imported into V3.

The deterministic CPU smoke and a bounded synchronous multi-generation runner
are implemented. The latter connects cumulative active-window replay,
consumed-position candidate cadence, committed-champion-only self-play,
generation-boundary signal drain, a checksum-bound pre-commit journal,
single-coordinator locking, and receipt-gated archive pruning. `run` remains
plan-only unless `--execute` and an absolute `--max-train-positions` bound are
both supplied. Stage 1 configs contain the reviewed P6 loss and occupancy
weights; target execution still requires the explicit bound and preflight.

## Commands

Use Python 3.11-3.13 with `training/requirements-v3.txt`.

```powershell
python -B -m training.v3 print-config --config training/v3/configs/smoke_cpu.json
python -B -m training.v3 smoke --config training/v3/configs/smoke_cpu.json
python -B -m training.v3 smoke --config training/v3/configs/smoke_cpu.json --resume
python -B -m training.v3 run --config training/v3/configs/pilot_gpu_64x4.json
python -B -m training.v3 run --config training/v3/configs/v3_1_mini_128x6.json
python -B -m training.v3 run --config training/v3/configs/stage1_scale_screen_b4c64_2x3080ti.json --execute --max-train-positions 61000
python -B -m training.v3 validate-local --config training/v3/configs/smoke_cpu.json
```

`run --execute` additionally requires `--max-train-positions`; optional
`--max-generations` bounds one maintenance/canary invocation. `validate-local`
accepts diagnostic output paths and an optional Replay V2 dataset path.
Unknown JSON sections, fields, and scalar types are errors. Operational runtime
topology and storage policy are recorded but excluded from the model-lineage
hash, so a run can move machines without pretending to be a new learning
experiment. MCTS lane count remains semantic because virtual-loss batching can
change visit targets; shard size remains operational.

## Presets

| Preset | Purpose | Model | Search schedule | Status |
|---|---|---:|---|---|
| `smoke_cpu.json` | Correctness and recovery | 16 channels, 1 block | 4 games, 8/2 full/fast sims | Executable |
| `pilot_gpu_64x4.json` | Historical cloud throughput and learning-signal check | 64 channels, 4 blocks | 64 games, 128/32 sims | Pilot evidence |
| `stage1_scale_screen_b4c64_2x3080ti.json` | Stage 1 canary on the target host | 64 channels, 4 blocks | 64 games, 128/32 sims | Executable with an explicit bound |
| `stage1_scale_screen_b6c128_2x3080ti.json` | Stage 1 medium-capacity target | 128 channels, 6 blocks | 160 games, 128/32 sims | Executable with an explicit bound |

The old ambiguous `v3_1_small.json` was removed. A 64x4 network is a pilot, not
a full-run target. The 128x6 Mini is about 1.82 million parameters, close to the
previous approximately 1.94 million-parameter CubeSprite V3 Mini. Model growth
requires a new run; it is not a hidden stage inside one config.

The Stage 1 plan now has two explicitly separate scaling tracks. The research
track starts every size from random initialization and imports neither weights
nor replay. The production track will allow a new, randomly initialized larger
lineage to offline-bootstrap from a provenance-complete donor Replay V2 bundle,
then requires a paired donor qualification gate before creating its first
accepted champion. Cross-scale replay import remains separate from same-lineage
formal replay; no formal command silently relaxes the replay/config checks.

The executable plan/bundle foundation and complete runbook are in
[`DUAL_TRACK_SCALING.md`](DUAL_TRACK_SCALING.md). `tools/run_v3_scaling.py`
prints the strict dual-track plan, builds and verifies authenticated data-only
donor bundles, inspects deterministic donor/own sampling, and writes
non-promoting donor qualification evidence. None of these operations registers
a champion or authorizes self-play.

One JSON covers bootstrap, early, and main training through explicit
`selfplay.search_schedule`, `selfplay.exploration_phases`,
`learner.lr_schedule`, and `gate.search_schedule`. There are no Python config
inheritance chains or script-local schedules.

The Mini exploration phases intentionally start from the last stable V3
schedule recorded in the historical runbook:

| Ply range | Temperature | Dirichlet alpha | Epsilon |
|---|---:|---:|---:|
| 1-28 | 1.0 | 0.24 | 0.060 |
| 29-50 | 0.5 | 0.5 | 0.005 |
| 51+ | 0.0 | 0.0 | 0.0 |

The Stage 1 research line deliberately includes the opening in the 28-turn
high-exploration phase. Earlier low-exploration openings were vulnerable to
human moves outside the model's narrow self-play distribution. This setting is
an explicit research hypothesis, not an optimality claim: per-phase selected
top-1 rate, selected-visit probability, entropy, opening diversity, short-game
rate, and committed-champion stability are monitored. The later 0.5 and greedy
phases remain, so one temperature/noise pair is not used for the whole game.

The optional `selfplay.dynamic_exploration` controller is a separate, semantic
training recipe. It changes only the end of the opening high-exploration phase
at committed generation boundaries. The Stage 1 B6 preset starts at 12 plies
and may advance through 20 to 28 after at least 16 generations in a stage and
an eight-generation window whose mean game length, short-game rate, and full
search policy entropy all remain within predeclared ranges. It never decreases
the window, skips a stage, reacts to a partially written generation, or changes
search simulations. Checkpoints persist both the controller stage and its
unreset observation history; metrics, replay manifests, and generation commits
record every effective phase and transition decision. The 2.4M-position run is
a recipe-level comparison with the historical from-scratch G343 checkpoint,
not a single-variable exploration ablation.

The optional `selfplay.opening_temperature_mixture` is another explicit
semantic fork. In its V1 contract, consecutive game IDs alternate between the
unchanged exploration schedule and a route that multiplies temperature by
0.5 only for plies 0-7; root-noise alpha/epsilon, search simulations, and the
schedule from ply 8 onward remain unchanged. Every search stage must contain an
even number of games, so each committed generation produces exactly half of
each route independent of actor completion order. The learner does not sample
the resulting raw pool uniformly: its checkpointed absolute sample cursor
alternates two position pools, giving each route exactly half of consumed
training positions even when their game-length distributions differ. Shuffle
manifests, learner metrics, self-play health, and replay manifests record both
the available raw-position skew and the consumed 50/50 position contract. The
G487 B10 preset uses a fresh replay child so pre-fork positions cannot dilute
the comparison.

## Data and learner contract

- A deterministic seed is derived from `run_seed + game_id`, independent of
  actor completion order.
- Every game receives at least one full-search position selected from its first
  seven plies; the empty board is not systematically over-sampled.
- Full and fast positions both enter immutable replay. WDL loss uses every
  position, while policy loss has weight one only on full-search rows.
- Raw NPZ shards are split on game boundaries according to `shard_games` and
  committed as NPZ, manifest, then a final ready marker. Manifests record SHA256,
  compressed bytes, and measured bytes per position.
- Train/validation split is a stable game-level hash. Selection manifests store
  bounds, counts, and digests instead of expanding every sample ID.
- The active set uses KataGo's growing recent-window formula. Raw history is
  retained independently from the active training window.
- A train-token bucket limits consumed positions per newly generated position.
  `max_optimizer_steps_per_cycle` is only a safety cap, not a promise that every
  cycle performs that many updates.
- D4 augmentation is online and deterministic by sample cursor. CUDA learners
  use FP32 policy/WDL losses under autocast, pinned memory, non-blocking transfer,
  and worker prefetch when applicable. If GradScaler skips an optimizer step,
  replay budget, sample cursor, scheduler, and global step remain unchanged so
  the same deterministic batch can be retried at the reduced scale.
- Validation reports policy CE, WDL CE, Brier score, calibration error, WDL
  accuracy, and throughput when the active window contains validation games.

## Gate contract

Evaluation uses a versioned deterministic opening manifest. Openings are D4
deduplicated, non-terminal, and every opening is played exactly twice with model
colors swapped. The decision reports overall, candidate-as-first, and
candidate-as-second W/D/L.

The config declares an initial pair count, an increment, and a maximum. A formal
loop should append pairs only after an inconclusive result and reuse earlier
results. Smoke executes only the initial batch. A one-look gate uses 95%; a
multi-look gate applies a Bonferroni confidence of
`1 - (1 - confidence) / sequential_looks` at every promotion decision, while
retaining ordinary 95% intervals for descriptive reporting. Acceptance also
uses an explicit role guard. `absolute_floor` preserves historical lineages and
requires both role scores to clear their configured floor.
`relative_noninferiority` is intended for rules with a large structural
first-player advantage: the paired-score confidence interval remains the
promotion criterion, while the candidate's second-player score is compared
with an accepted-champion self-match control on the identical openings, seeds,
search budget, and colors. It vetoes promotion only when the upper confidence
bound of candidate-minus-control performance is below the configured negative
margin; uncertain role evidence is not treated as regression. Candidate and
control games, the margin, interval, and runtime topology are persisted in the
gate artifact. Before a cold-start lineage has a committed champion, the first
scheduled candidate is evaluated against the explicit random bootstrap
incumbent and its role control is the matching random-vs-random baseline. The
artifact labels this as `random_bootstrap`; later gates require and label the
committed `accepted_champion` control. This bootstrap exception is explicit at
the formal-runner call site and cannot silently turn a missing incumbent into a
random control. Diagnostic matches against milestones or tactical suites must
not silently change this promotion protocol.

Self-play accepts only the committed accepted champion. Inconclusive
intermediate looks append the next disjoint pair increment. If the maximum pair
budget is exhausted without establishing promotion, the candidate is rejected
for `insufficient_evidence_at_max_pairs`; this is a no-promotion disposition,
not a claim that the candidate was proven weaker. Training continues with the
unchanged accepted champion. A candidate is moved to accepted/rejected before
checkpoint and audit artifacts are written, and a generation commit is
published last. Orphan model files are therefore ignored after an interrupted
commit. Runs created under the former operator-review contract resolve an
already committed terminal-inconclusive pending candidate through an explicit
checksum-bound audit artifact on resume, while preserving its historical
candidate path so the old generation commit remains immutable.

All formal V3 head-to-head paths use the same operational evaluation runtime:
same-lineage candidate gates, Historical Anchored Elo (including anchor-scale
calibration), and cross-scale donor qualification. Parallelism is only across
independent opening/color games. Every game keeps the frozen simulation count,
`cpuct`, deterministic seed, and single-lane MCTS semantics. The adopted target-
host mode uses device-local replicated serial workers: every process owns its
two models, dynamically receives complete opening pairs, and performs no per-
move IPC or cross-game inference batching. Immutable evidence records worker
devices, replica counts, model-load/play/wall time, and exact pair coverage.
These topology controls are operational and excluded from the model-lineage
hash.

Formal resumable checkpoints are not accepted-model artifacts. At a scheduled
learning milestone, project the checkpoint weights into an immutable,
evaluation-only artifact before running Anchored Elo or policy-target audits:

```powershell
python tools\export_v3_evaluation_snapshot.py `
  --checkpoint path\to\checkpoints\g000012-s00000398.pt `
  --output path\to\evaluation\snapshots\b6c128-100k.pt `
  --model-id b6c128-research-100k
```

The snapshot records the source checkpoint SHA-256, config hash, generation,
step, and consumed positions. It is explicitly ineligible for acceptance and
self-play and does not modify the formal run state. This distinction matters
when the latest accepted champion predates the milestone checkpoint.

GPU presets set `runtime.evaluation_devices` and
`runtime.evaluation_replicas_per_device` only after a fixed-opening serial-
versus-replicated check. The check requires identical opening/role results and
safe GPU-memory headroom. The older central-batched experiment remains available
through `evaluation_parallel_games`/`evaluation_inference_batch_size`, but is not
a target preset: it failed to provide stable cross-opponent speedups and one
tested topology changed a game outcome. Do not raise MCTS lanes to improve
evaluation utilization. Exact benchmark rows remain in ignored artifacts.

The target dual-GPU replicated evaluator was accepted only after fixed serial
comparisons and frozen anchor calibration reproduced the same game outcomes.
It keeps single-lane search semantics and uses host-wide CPU utilization only
as supporting operational evidence, never as a proxy for search quality.

For utilization checks, discard model loading and the final under-filled tail.
Use a workload with at least several times as many opening pairs as replicas and
report the highest sustained 10-30 second steady-state window, not a single
one-second spike or the whole-run average. Low aggregate CPU use can be correct
when the configured search replicas keep the GPU inference services saturated.
Retain exact utilization, power, memory, and timing samples only in ignored run
artifacts and their local archive bundles.

Use the common fixed-profile sweep before changing a GPU preset:

```powershell
python tools\benchmark_v3_evaluation_runtime.py `
  --device cuda:0 --pair-count 4 --topologies 4x8,8x16,12x32,16x32 `
  --output training\runs\evaluation_runtime_benchmark.json
```

The central-batch diagnostic accepts `anchor:<id>` and `v3:<checkpoint>` model specs. It saves
the serial games, every parallel game, exact-result mismatch count, effective
inference batches, wall-time speedup, cgroup CPU quota/use, and sampled GPU
utilization, memory, and power. Run it only in an uncontended hardware window.
For the adopted replicated path, use
`match --devices cuda:0,cuda:1 --replicas-per-device 4`; a single immutable
batch dynamically balances whole opening pairs across eight workers.

## Historical anchored Elo

Anchored Elo is a diagnostic scaling measure, not a promotion gate. The frozen
v1 registry uses `v2.2 Balance` as zero Elo plus CubeSprite V3 iter240 and V3
Mini iter260. Exact paths and SHA-256 digests live in
`configs/anchored_elo_historical_v1.json`. All three are external Legacy
opponents: they never enter V3 lineage and never produce replay.

Every model uses the common V3 classic rule engine and MCTS. The compatibility
adapter extracts only the 25 current gravity-legal actions from a Legacy
150-action policy. The primary `primary_256` profile runs at early, middle, and
final milestones; the separate `final_512` profile runs only at final. Results
and anchor scales never mix across profiles.

An independent third ruler lives in
`configs/anchored_pressure_256v512_historical_v1.json`. In
`pressure_256v512`, the V3 target always receives 256 simulations and the
historical anchor always receives 512; the budget follows model identity when
colors swap. It reports paired score/W-D-L, role splits, confidence intervals,
and saturation against each anchor, but does not fit or reuse an Elo scale.
Pressure results must not be numerically mixed with either symmetric profile
and cannot affect promotion or replay. Keeping this in a separate config also
preserves the frozen symmetric-config hash and all prior evidence.

The 200-opening manifest is D4-deduplicated, immutable, and checksum-bound.
Each opening is played with roles swapped. A profile begins at 50 pairs per
opponent and may append disjoint 50-pair batches up to 200 until saturation or
the configured 100-Elo descriptive interval-width target is reached. For each
symmetric Elo profile, historical anchors first play a one-time round robin;
the resulting scale is
frozen before target ratings are fitted. A saturated matchup reports a finite
Wilson score/Elo lower bound instead of infinite point Elo. Reports retain
target-as-FIRST/SECOND W/D/L and cannot feed promotion or self-play.

```powershell
python tools\run_v3_anchored_eval.py `
  --config training\v3\configs\anchored_elo_historical_v1.json plan
python tools\run_v3_anchored_eval.py `
  --config training\v3\configs\anchored_elo_historical_v1.json verify
```

Use `match --help`, `calibrate --help`, `report --help`, and
`pressure-report --help` for the explicit
batch workflow. Match artifacts use `*.match.json`, are immutable and
content-hashed, record an evaluator-source hash and runtime versions, and reject
duplicate opening/role evidence. Anchor calibration
uses `--milestone calibration`; target batches use `early`, `middle`, or
`final` according to the selected profile.

### Anchored Elo v2

The immutable v2 registry is
`configs/anchored_elo_historical_v2.json`. It preserves all three v1 anchors
and adds the independent V3 Research B6C128 dynamic G150 accepted checkpoint
as `b6_dynamic_g150`. The checkpoint is frozen by SHA-256 and installed at
`.tmp/anchored-models-v2/b6_dynamic_g150.pt`; it remains evaluation-only and
cannot produce replay or become an accepted training model. The corresponding
asymmetric ruler is
`configs/anchored_pressure_256v512_historical_v2.json`.

V1 match batches and frozen scales cannot be reused as v2 evidence because the
registry hash is different. Calibrate a fresh four-anchor round robin for each
symmetric profile before rating targets. At 50 opening pairs this is six
matchups and 600 games per profile. Existing G150 target-vs-v1 matches remain
historical diagnostics; they do not silently become anchor-calibration batches.
Keep v1 configs and reports unchanged so old ratings remain reproducible.

```powershell
python tools\run_v3_anchored_eval.py `
  --config training\v3\configs\anchored_elo_historical_v2.json plan
python tools\run_v3_anchored_eval.py `
  --config training\v3\configs\anchored_elo_historical_v2.json verify
```

Use `anchor:b6_dynamic_g150` in `match` commands. The evaluator selects the V3
artifact loader from the frozen anchor registry while continuing to load the
three historical checkpoints through the Legacy compatibility boundary.

### Anchored Elo v3

The Stage 2 ruler is an additive, immutable successor to v2. The symmetric
registry is `configs/anchored_elo_historical_v3.json`; the independent
asymmetric pressure ruler is
`configs/anchored_pressure_256v512_historical_v3.json`. V3 preserves the four
v2 anchors and adds the final accepted Stage 1 B8C192 G268 model artifact plus the
final accepted B10C256 mixed-opening-temperature G258 champion. Both additions
are checksum-frozen V3 artifacts under `.tmp/anchored-models-v3/`, are
evaluation-only, and must never produce replay or enter a training lineage.

Changing the registry changes its canonical hash. Consequently, no v1 or v2
match batch, calibration scale, or report is valid v3 evidence. A symmetric v3
profile starts with all 15 pairwise matchups among the six anchors (1,500 games
at 50 paired openings); `primary_256` and `final_512` receive separate frozen
scales. `pressure_256v512` remains an independent asymmetric report and is not
converted into an Elo rating. Extensions must use disjoint opening ranges up
to the configured 200-pair ceiling.

```powershell
python tools\run_v3_anchored_eval.py --config training\v3\configs\anchored_elo_historical_v3.json plan
python tools\run_v3_anchored_eval.py --config training\v3\configs\anchored_elo_historical_v3.json verify
python tools\run_v3_anchored_eval.py --config training\v3\configs\anchored_pressure_256v512_historical_v3.json verify
```

For a Stage 2 checkpoint, run fresh paired batches against every registered
anchor with `--model-a v3:<checkpoint>` and `--model-b anchor:<id>`. Rate the
`primary_256` or `final_512` batches with the matching frozen v3 scale; never
use a v1/v2 scale. For `pressure_256v512`, use `pressure-report` on fresh
target-at-256 versus anchor-at-512 batches. Pressure results are directional
stress evidence and must not be merged into the symmetric Elo scale.

## Hardware plan

The reference MCTS batches the virtual-loss lanes of one game through
`predict_batch`. `actor_processes` now creates a bounded process pool, and
accepted-model actors send inference requests to one batching owner per active
self-play device. The shared service treats `inference_batch_size` as a hard
limit, deferring a complete actor request when adding it would overflow the
current batch.

An uncontended single-card short pilot calibrated the 64x4 model at the explicit
full/fast simulation contract. The provisional topology and bounded local sweep
range are encoded in `pilot_gpu_64x4.json`. Higher lane counts changed
fixed-position search targets, so the larger 128x6 Mini must be recalibrated
rather than inheriting the pilot blindly. Exact pilot measurements stay outside
Git.

The Stage 1 capacity screen routes B4/B6 to the calibrated target presets
`stage1_scale_screen_b4c64_2x3080ti.json` and
`stage1_scale_screen_b6c128_2x3080ti.json`; B8 remains on its uncalibrated
generic preset. All three freeze self-play at 128/32 sims,
the same phased exploration schedule, replay ratio, learner semantics, and a
256-sim gate. B4 is a one-seed canary whose explicit budget closes its final
candidate threshold. B6 and B8 use staged primary and confirmation seeds. B8 actor counts and inference batches remain
calibration starting points, not verified target-machine defaults.

Stage 1 keeps this bounded independent Research line; it does not switch to a
Production-only path. The main V3 Base is selected only after the staged
B6/B8/(optional Larger) evidence. It is the last scale with supported stability,
strength, learning slope, data quality, and compute efficiency—not necessarily
B6 and not automatically the largest model. Replay-transfer checkpoints remain
eligible as practical high-performance candidates only with their Production
lineage stated separately from the independent Research reference.

For the `connect4_gpu_2608` class of host, the bounded calibration supports the
staged two-service B4/B6 presets committed in the explicit configs: both GPUs
produce self-play, those services fully drain, and only then does `cuda:0` run
the learner. The chosen actor count sits at the measured throughput plateau;
adding actors increased throttling without useful throughput. Low aggregate CPU
utilization at a plateau is therefore not a reason to add search threads.

Lane count remains semantic. Fixed-position and paired checks support lane 4 as
the conservative B4/B6 target-machine default, but this is bounded
no-regression evidence rather than final strength evidence. Any increase must
be repeated against a stable accepted champion. B8 still requires its own
target-hardware preset. Exact sweep measurements remain outside Git.

The two-service preset is valid only because the formal runner is synchronous:
it joins all actors and inference services before learning begins. A future
overlapped scheduler must return to a role split and must not silently move
`cuda:0` away from an active learner. Continue selecting topology by useful
simulations or games per wall time together with inference-batch formation,
GPU gaps, cgroup CPU use/throttling, memory headroom, and fixed-search quality;
CPU or GPU utilization alone is not an acceptance metric.

- One GPU: use one CUDA context and one shared inference owner in staged phases:
  self-play, learner, then gate. Do not keep separate learner and inference CUDA
  processes resident on the same card.
- Two or more GPUs: the explicitly configured learner device (default
  `cuda:0`) is the sole learner; every device in `selfplay_devices` owns one
  accepted-model inference service and an actor group. Including the learner
  device means staged execution and forbids overlap. Excluding it is an
  explicit role split that may support overlap in a future scheduler. Gate
  waits for current games to finish and switches models only at a new-game
  boundary.
- DDP is disabled. The network is too small to assume that gradient all-reduce
  will beat dedicating extra GPUs to self-play inference.
- All proposed game, inference, completed-game, and checkpoint queues are
  bounded. The static planner warns about actor-process CPU oversubscription and
  batch limits that cannot be filled by actor/lane request concurrency. Lanes
  are vectorized search semantics inside an actor, not additional OS threads.
- Self-play inference remains FP32 in this static baseline. A separate
  `fp32`/`bf16`/`fp16` choice should only be added after the first GPU pilot;
  it must not be conflated with `learner_amp`.

The historical RTX 4090 run used 24 actors x 4 search lanes with one shared
inference service. It remains historical context, not a portable default.

For the measured two-GPU staged topology, keep one preset and change only the
runtime controls:

```json
"device": "cuda:0",
"selfplay_devices": ["cuda:0", "cuda:1"],
"actor_processes": 40
```

`run` preserves the explicitly configured learner device and marks the CUDA
inventory unverified during offline planning. Single-GPU presets leave
`selfplay_devices` empty, so `--device cuda:N` moves the whole staged topology;
multi-GPU topology is intentionally explicit in JSON.
Every formal invocation appends its exact runtime controls and Git commit to
`run_manifest.json`; this preserves topology history when a lineage resumes
with a different operational placement.

## Storage and cloud retention

`keep_all` never proposes pruning. `archive_ack_prune` separates immutable
bundle creation, local verification/materialization, receipt ingestion, and an
explicit prune command. The Stage 1 target presets use a 70% soft watermark, a
10 GiB hard reserve, a 1.25x active-window margin, and approximately 4 GiB
archive bundles. They keep the newest three resumable checkpoints, two accepted
models, one rejected model, every unresolved candidate, all small
manifests/metrics, and all audit replays.

An artifact is never eligible merely because an upload command succeeded. A
verified receipt must match its run-relative path, byte size, and SHA256. Raw
shards inside the expanded active window or beyond the learner cursor remain
pinned. The trainer never uploads or deletes: operators invoke
`tools/sync_v3_run.py` explicitly, and cloud pruning occurs only after the local
copy has been fully verified and its receipt has been returned and ingested.

## Human audit replay

Each smoke checkpoint selects representative completed games deterministically,
avoiding the shortest and longest anchors when enough games exist and filling
available P1/P2/draw outcomes. Files are written to:

```text
training/runs/<run_id>/samples/g<generation>/*.c4replay.json
training/runs/<run_id>/samples/g<generation>/audit_index.json
```

The replay files use CubeSprite protocol V2 and can be imported directly by the
app backend. They preserve the rule ID/version, tagged placement/forced-pass
turns, and FIRST/SECOND participant provenance. Mutable display names are not
part of the participant-provenance hash. V1 imports remain readable through a
separate exact-key validation path. Seed, search counts, selection reason, and
checkpoint checksum remain in the adjacent audit index.

## Pre-Stage-1 model and replay contract

The learned action remains 25 gravity columns. The program maps a selected
column to the current cell in the 150-coordinate application representation;
forced pass is a tagged deterministic rule transition, never a policy logit.

Every model call carries a separate FIRST/SECOND one-hot and the registered
32-element semantic rule vector. The training forward returns named policy,
WDL, opponent-reply, future-occupancy, and moves-left logits. The search-only
forward computes only policy and WDL. Human/model/controller identity is
recorded for provenance but never enters the network.

Replay schema V2 stores rule code, absolute player, tagged turn kind, placement
and turn indices, stored policy weight, reply target/mask, absolute terminal
board, and remaining turns. Full placements have policy weight 1; fast
placements and forced-pass rows have policy weight 0. Pass rows retain WDL and
auxiliary supervision with zero visits. Replay V1 and V2 are rejected if mixed.

Stage 1 scale configs use `classic`, auxiliary loss weights
`1.0/1.0/0.15/0.15/0.05`, and P6-frozen future-occupancy class weights
`5.0/5.0/0.35561042132416815`. Smoke and historical pilot/Mini configs retain
neutral occupancy weights because they are correctness or historical evidence,
not formal Stage 1 lineages.

### Bounded local P6/P7 validation

The local validator freezes the five-way auxiliary-head matrix and audits the
safety foundations without starting self-play or formal training:

```powershell
python -B -m training.v3 validate-local `
  --config training\v3\configs\smoke_cpu.json `
  --output training\runs\local_validation\report.json `
  --write-ablation-configs training\runs\local_validation\p6_configs
```

The frozen P6 screening floor is 768 games and 12,000 samples. Collect the pool
without learner promotion, then run the same-position two-seed five-way screen:

```powershell
python tools\collect_v3_p6_replay.py --config training\v3\configs\stage1_scale_screen_b4c64_2x3080ti.json --output training\runs\p6_aux_calibration_896_v1 --games 896
python -B -m training.v3 validate-local --config training\v3\configs\stage1_scale_screen_b4c64_2x3080ti.json --replay-dir training\runs\p6_aux_calibration_896_v1 --minimum-replay-games 768 --minimum-replay-samples 12000
python tools\run_v3_p6_screen.py --config training\v3\configs\stage1_scale_screen_b4c64_2x3080ti.json --replay-dir training\runs\p6_aux_calibration_896_v1 --output training\runs\p6_aux_screen_896_v2
```

Add `--replay-dir <directory>` once a Replay V2 dataset has been selected. The
validator checks every NPZ/manifest/ready triplet, records a stable dataset
fingerprint and target coverage, and suggests inverse-frequency future
occupancy weights. The suggestion is evidence, not an automatic config change.
Dataset integrity and P6 screening readiness are separate. To assert the latter,
pass both `--minimum-replay-games` and `--minimum-replay-samples`; the validator
does not invent those experiment-size thresholds.

The 2026-08-23 screen used 896 games/13,255 positions (dataset fingerprint
`d117c989c6d6c588e62687d667095a5b234882371920375d386e68b453458e82`) and two
seeds at 40,000 positions per variant. All-head policy validation improved for
both seeds; WDL evidence was mixed, so this freezes a learning-system starting
point rather than claiming playing-strength improvement. P7 scheduler/archive
connections are executable and regression-tested.

A two-generation target-GPU canary consumed exactly 256 then 512 cumulative
train positions, restored checkpoint/RNG/optimizer/scaler/replay state, and
committed the second generation before a correlated random-bootstrap stability
pause. Its 36-file archive was fully materialized and verified locally; prune
kept both current resume generations and removed only the acknowledged staging
copy. P6 raw pools, all ten successful screen weights, the partial failed screen,
reports, and receipts were likewise retained locally before eligible cloud raw
shards were removed.

### Stage 1 stability and 256-sim target audit

`stability.py` evaluates correlated behavioral symptoms over consecutive
generations. Repeated absolute short-game/low-variance watch thresholds do not
alone pause a frozen champion under the Stage 1 high-exploration schedule: the
second correlated generation must also show a joint material adverse trend (at
least a 10% mean-length drop and a 10-point short-game-rate rise), or collapsed
value loss. The regression fixture in
`test/fixtures/v3_historical_stability_trace_v1.json` is aggregate Legacy
evidence only: it stops the high-temperature 253-264 branch at 254 while not
stopping the controlled 249-260 recovery. A single spike or a stable exploratory
short-game distribution remains a watch. The scheduler records the health
decision; escalating correlated instability still requires a safe-boundary
operator pause rather than automatic parameter changes.

Every new self-play health artifact includes full/fast visit-target summaries:
visit-budget integrity, entropy, effective action count, support, top-1 mass,
and top-1/top-2 gap. At selected accepted checkpoints, generate the same fixed
positions at 256, repeated 256, and 512 sims, then run:

```powershell
python tools\run_v3_policy_target_quality.py generate `
  --checkpoint path\to\accepted.pt `
  --openings training\v3\evaluation\anchored_openings_classic_v1.json `
  --position-count 50 --search-sims 256 --audit-seed 314159 `
  --mcts-lanes 4 --device cuda:0 --run-id milestone-33k-primary `
  --output path\to\fixed_256.npz
python tools\run_v3_policy_target_quality.py generate `
  --checkpoint path\to\accepted.pt `
  --openings training\v3\evaluation\anchored_openings_classic_v1.json `
  --position-count 50 --search-sims 256 --audit-seed 271828 `
  --mcts-lanes 4 --device cuda:0 --run-id milestone-33k-repeat `
  --output path\to\fixed_256_repeat.npz
python tools\run_v3_policy_target_quality.py generate `
  --checkpoint path\to\accepted.pt `
  --openings training\v3\evaluation\anchored_openings_classic_v1.json `
  --position-count 50 --search-sims 512 --audit-seed 314159 `
  --mcts-lanes 4 --device cuda:0 --run-id milestone-33k-reference `
  --output path\to\fixed_512.npz
python tools\run_v3_policy_target_quality.py compare `
  --primary path\to\fixed_256.npz `
  --reference path\to\fixed_512.npz `
  --output path\to\policy_target_quality.json
```

Primary and reference use the same audit seed so each position receives the
same Dirichlet draw; the repeated primary uses a different seed to measure
stochastic target variability. The generator records model/opening checksums,
rule registry, lane semantics, noise, exact budget, and inference batching in
an immutable diagnostic-only replay triplet. The comparison verifies triplets
and exact paired-position identity, then reports top-action agreement, total
variation, Jensen-Shannon divergence, and reference top-1 regret. It never
changes a search budget. Raise 256 only when its 512 delta exceeds repeated-256
variability and the 512 targets also improve held-out policy fit or paired
strength.

The fixed-position audits show that the frozen 128/32 Research target is a
deliberately coarse controlled scale-screen setting, not a quality-saturated
target. Keep it frozen across B4 and B6. Treat 256 as the Production candidate;
when its delta from 512 exceeds independent repeated-256 variability, still
require improved held-out policy fit or paired strength before paying for 512.
Fast-search positions have policy weight zero, so this decision concerns the
full-search policy targets rather than the fast action-selection path.

Milestone Anchored Elo and policy audits may use evaluation-only checkpoint
snapshots when the accepted producer predates the checkpoint. These observations
remain descriptive and never change the gate or self-play producer. Exact
milestone scores, intervals, policy metrics, and checksums stay in ignored run
artifacts and verified local archives rather than Git.

## Reusable experiment lessons

- Separate functional correctness, search-target equivalence, playing strength,
  and hardware efficiency. Passing one does not establish another.
- Split hardware observations into load, steady full-queue, and under-filled tail
  phases. Choose topology from useful work per wall time, queue formation,
  throttling, and memory headroom rather than CPU occupancy alone.
- Keep opening pairs as the scheduling unit. Parallelize complete paired games
  across model replicas while preserving per-game simulation count, seed, and
  lane semantics.
- Allow early short-game behavior time to evolve through a materially larger
  self-play pool. Treat game length together with diversity, role splits, losses,
  calibration, and strength; it is not a standalone stop rule.
- A formal checkpoint and an accepted producer are different states. Use an
  immutable evaluation-only projection for checkpoint learning curves, and never
  let that projection silently produce replay.
- Archive and verify checkpoints, replay, logs, and evaluation artifacts before
  retention or pruning. A disk watermark should stop safely rather than risk a
  partial checkpoint.
- Treat each generation's active-window DataLoader as one-shot. Do not enable
  persistent workers on a loader that is rebuilt every generation, especially
  with CUDA pin-memory; monitor parent-process file descriptors in long canaries
  as well as worker exit status.
- Store exact experimental measurements under ignored run directories and local
  receipt-verified bundles. Commit only executable contracts, chosen presets,
  generalized conclusions, and reproduction commands.

## Smoke artifacts

```text
training/runs/<run_id>/
  resolved_config.json
  run_manifest.json
  manifests/generations/
  manifests/generation_drafts/
  manifests/coordinator.lock  # present only while held
  replay/raw/
  replay/shuffle/
  metrics/
  checkpoints/
  candidates/
  accepted/
  rejected/
  samples/
  archive_staging/
  archive_receipts/
```

The CPU smoke completes random bootstrap self-play, immutable sharding, active
window selection, two AdamW updates, validation when available, paired gate,
atomic checkpoint, CubeSprite replay export, generation commit, and exact
checkpoint-resume comparison. `smoke --resume` remains a non-mutating in-memory
probe and deliberately does not advance the committed checkpoint.

Committed generations bind the checkpoint, replay NPZ/manifest/ready triplets,
accepted/candidate models, audit index, and every portable audit replay by hash;
resume scans backward to the newest complete generation. The formal runner
wraps every generation with the no-clobber coordinator lock and checksum-bound
pre-commit draft. A fully staged commit is publishable after restart; any other
partial state blocks for explicit recovery. File and directory
`fsync` provides the intended
power-loss durability on the Linux cloud target. Windows smoke validates atomic
replacement and checksums, but does not claim directory-metadata durability
across sudden power loss.

## Cloud-to-local archive and pruning

Formal target presets use a 70% soft watermark, a 10-GiB absolute hard reserve,
and 4-GiB bundle staging. At either the soft watermark or loss of staging
headroom, the scheduler finishes and commits the active generation, then stops
with `archive_required`. From the local machine run:

```powershell
python tools\sync_v3_run.py `
  --remote connect4_gpu_2608 `
  --run-dir training/runs/<run_id> `
  --local-root D:\path\to\v3-archives\<run_id> `
  --bundle-target-gib 4 `
  --max-bundles 1 `
  --prune
```

Each local archive, manifest, receipt, and materialized latest file tree is
retained. Logs, resolved configs, generation manifests, replay triplets,
checkpoints, model artifacts, metrics, and audit games are included
incrementally. The cloud only removes an archived staging tar or a superseded
training artifact after the returned receipt matches every entry. The latest
three resumable generations, current active replay margin, two accepted models,
and one rejected model remain protected. A failed transfer or checksum never
authorizes deletion.

See `STATIC_AUDIT.md` for the review decisions, on-machine pilot matrix, health
watch levels, and remaining production blockers.
