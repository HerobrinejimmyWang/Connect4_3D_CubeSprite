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
both supplied. Current Stage 1 configs are still rejected by the execution path
because their P6 auxiliary weights are explicitly provisional.

## Commands

Use Python 3.11-3.13 with `training/requirements-v3.txt`.

```powershell
python -B -m training.v3 print-config --config training/v3/configs/smoke_cpu.json
python -B -m training.v3 smoke --config training/v3/configs/smoke_cpu.json
python -B -m training.v3 smoke --config training/v3/configs/smoke_cpu.json --resume
python -B -m training.v3 run --config training/v3/configs/pilot_gpu_64x4.json
python -B -m training.v3 run --config training/v3/configs/v3_1_mini_128x6.json
python -B -m training.v3 run --config training/v3/configs/stage1_scale_screen_b4c64_2x3080ti.json --execute --max-train-positions 60000
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
| `stage1_scale_screen_b4c64_2x3080ti.json` | Stage 1 canary on the target host | 64 channels, 4 blocks | 64 games, 128/32 sims | Executable after P6 freeze |
| `stage1_scale_screen_b6c128_2x3080ti.json` | Stage 1 medium-capacity target | 128 channels, 6 blocks | 160 games, 128/32 sims | Executable after P6 freeze |

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
| 1-8 | 0.5 | 0.5 | 0.006 |
| 9-28 | 1.0 | 0.24 | 0.060 |
| 29-50 | 0.5 | 0.5 | 0.005 |
| 51+ | 0.0 | 0.0 | 0.0 |

These are conservative starting values, not a claim of V3.1 optimality. The
previous high-temperature branch collapsed to very short games, so a single
temperature/noise pair is no longer a supported production configuration.

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
requires both role scores to be at least 0.45. Diagnostic matches against
milestones or tactical suites must not silently change this promotion protocol.

Self-play accepts only the committed accepted champion. A candidate is moved to
accepted/rejected or remains inconclusive, then checkpoint and audit artifacts
are written, and a generation commit is published last. Orphan model files are
therefore ignored after an interrupted commit.

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

The 200-opening manifest is D4-deduplicated, immutable, and checksum-bound.
Each opening is played with roles swapped. A profile begins at 50 pairs per
opponent and may append disjoint 50-pair batches up to 200 until saturation or
the configured 100-Elo descriptive interval-width target is reached. Historical anchors
first play a one-time round robin for each profile; the resulting scale is
frozen before target ratings are fitted. A saturated matchup reports a finite
Wilson score/Elo lower bound instead of infinite point Elo. Reports retain
target-as-FIRST/SECOND W/D/L and cannot feed promotion or self-play.

```powershell
python tools\run_v3_anchored_eval.py `
  --config training\v3\configs\anchored_elo_historical_v1.json plan
python tools\run_v3_anchored_eval.py `
  --config training\v3\configs\anchored_elo_historical_v1.json verify
```

Use `match --help`, `calibrate --help`, and `report --help` for the explicit
batch workflow. Match artifacts use `*.match.json`, are immutable and
content-hashed, record an evaluator-source hash and runtime versions, and reject
duplicate opening/role evidence. Anchor calibration
uses `--milestone calibration`; target batches use `early`, `middle`, or
`final` according to the selected profile.

## Hardware plan

The reference MCTS batches the virtual-loss lanes of one game through
`predict_batch`. `actor_processes` now creates a bounded process pool, and
accepted-model actors send inference requests to one batching owner per active
self-play device. The shared service treats `inference_batch_size` as a hard
limit, deferring a complete actor request when adding it would overflow the
current batch.

An uncontended RTX 3080 Ti + 20-vCPU/30-GiB short pilot calibrated the 64x4
model at 128/32 full/fast simulations. The provisional single-card topology is
18 actors x 6 lanes with inference batch 32; 18-20 actors and 4-6 lanes are the
reasonable local ranges. Fixed-position comparison retained 100% serial-MCTS
top-action agreement through 6 lanes, while 8 and 10 lanes fell to 96.9%.
These values are encoded only in `pilot_gpu_64x4.json`; the larger 128x6 Mini
must be recalibrated rather than inheriting them blindly.

The Stage 1 capacity screen routes B4/B6 to the calibrated target presets
`stage1_scale_screen_b4c64_2x3080ti.json` and
`stage1_scale_screen_b6c128_2x3080ti.json`; B8 remains on its uncalibrated
generic preset. All three freeze self-play at 128/32 sims,
the same phased exploration schedule, replay ratio, learner semantics, and a
256-sim gate. B4 is a one-seed 60k-position canary; B6 and B8 use staged primary
and confirmation seeds. B8 actor counts and inference batches remain
calibration starting points, not verified target-machine defaults.

Stage 1 keeps this bounded independent Research line; it does not switch to a
Production-only path. The main V3 Base is selected only after the staged
B6/B8/(optional Larger) evidence. It is the last scale with supported stability,
strength, learning slope, data quality, and compute efficiency—not necessarily
B6 and not automatically the largest model. Replay-transfer checkpoints remain
eligible as practical high-performance candidates only with their Production
lineage stated separately from the independent Research reference.

For the `connect4_gpu_2608` class of host (2x RTX 3080 Ti, 40-vCPU cgroup,
60-GiB memory cgroup), the bounded 2026-08-23 calibration supports the B4 and
B6 role-split presets `stage1_scale_screen_b4c64_2x3080ti.json` and
`stage1_scale_screen_b6c128_2x3080ti.json`: `cuda:0` is the sole learner and
`cuda:1` owns accepted-model self-play inference. Both use 24 actors x 4 lanes,
batch 32, and the actor-pool default 1 ms batch timeout. B4 fine points at
20/24/28 actors produced about 5266/5586/5496 simulations/s; B6 coarse points
at 20/24/28 produced about 4502/4696/4613 simulations/s. Batch 64 and a 0.5 ms
timeout were slower.

Lane count remains semantic. With fixed random V3 weights, lane 4 versus serial
lane 1 scored 17:15 over 32 paired B4 games and 16:16 over 32 paired B6 games.
This is a bounded no-regression signal, not final playing-strength evidence.
Fixed-position top-action agreement was 98.4% for B4 lane 4 and 96.9% for B6
lane 4; lane 6 drifted further in B4. Therefore lane 4 is the conservative
target-machine default, and it must be repeated against a stable accepted
champion before any increase. B8 still requires its own target-hardware preset.

When the learner is intentionally idle, two self-play services at 40 actors x
4 lanes reached about 8027 simulations/s for B4 and 7146 simulations/s for the
B6 single point. This is an opportunistic data-production mode, not the formal
role-split preset: do not silently move `cuda:0` away from an active learner.

- One GPU: use one CUDA context and one shared inference owner in staged phases:
  self-play, learner, then gate. Do not keep separate learner and inference CUDA
  processes resident on the same card.
- Two or more GPUs: the explicitly configured learner device (default
  `cuda:0`) is the sole learner; every device in `selfplay_devices` owns one
  accepted-model inference service and an actor group. Self-play may overlap
  the learner. Gate waits for current games to finish and switches models only
  at a new-game boundary.
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

For a two-GPU role split, keep one preset and change only the runtime topology:

```json
"device": "cuda:0",
"selfplay_devices": ["cuda:1"]
```

`run` preserves the explicitly configured learner device and marks the CUDA
inventory unverified during offline planning. Single-GPU presets leave
`selfplay_devices` empty, so `--device cuda:N` moves the whole staged topology;
multi-GPU topology is intentionally explicit in JSON.

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

All explicit V3 configs currently use `classic` and provisional auxiliary loss
weights `1.0/1.0/0.15/0.15/0.05`. Occupancy class weights are explicitly
`1.0/1.0/1.0` only as a pre-calibration placeholder. They must be calibrated
and frozen by the pre-Stage-1 ablation before formal training.

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
python tools\collect_v3_p6_replay.py --config training\v3\configs\stage1_scale_screen_b4c64_2x3080ti.json --output training\runs\p6_aux_calibration_768_v1
python -B -m training.v3 validate-local --config training\v3\configs\stage1_scale_screen_b4c64_2x3080ti.json --replay-dir training\runs\p6_aux_calibration_768_v1\replay\raw --minimum-replay-games 768 --minimum-replay-samples 12000
python tools\run_v3_p6_screen.py --config training\v3\configs\stage1_scale_screen_b4c64_2x3080ti.json --replay-dir training\runs\p6_aux_calibration_768_v1\replay\raw --output training\runs\p6_aux_screen_v1
```

Add `--replay-dir <directory>` once a Replay V2 dataset has been selected. The
validator checks every NPZ/manifest/ready triplet, records a stable dataset
fingerprint and target coverage, and suggests inverse-frequency future
occupancy weights. The suggestion is evidence, not an automatic config change.
Dataset integrity and P6 screening readiness are separate. To assert the latter,
pass both `--minimum-replay-games` and `--minimum-replay-samples`; the validator
does not invent those experiment-size thresholds.

The report distinguishes `local_contract_passed`, dataset integrity, explicit
P6 screening readiness, and `stage1_ready`. P7 scheduler/archive connections
are now executable and regression-tested; Stage 1 remains blocked until the P6
screen is reviewed and its loss/class weights are frozen in the explicit configs.

### Stage 1 stability and 256-sim target audit

`stability.py` evaluates correlated behavioral symptoms over consecutive
generations. The regression fixture in
`test/fixtures/v3_historical_stability_trace_v1.json` is aggregate Legacy
evidence only: it stops the high-temperature 253-264 branch at 254 while not
stopping the controlled 249-260 recovery. A single short-game spike remains a
watch. The scheduler records the health decision; correlated instability still
requires a safe-boundary operator pause rather than automatic parameter changes.

Every new self-play health artifact includes full/fast visit-target summaries:
visit-budget integrity, entropy, effective action count, support, top-1 mass,
and top-1/top-2 gap. At selected accepted checkpoints, generate the same fixed
positions at 256, repeated 256, and 512 sims, then run:

```powershell
python tools\run_v3_policy_target_quality.py compare `
  --primary path\to\fixed_256.npz `
  --reference path\to\fixed_512.npz `
  --output path\to\policy_target_quality.json
```

The tool verifies replay triplets and exact paired-position identity, then
reports top-action agreement, total variation, Jensen-Shannon divergence, and
reference top-1 regret. It never changes a search budget. Raise 256 only when
its 512 delta exceeds repeated-256 variability and the 512 targets also improve
held-out policy fit or paired strength.

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
