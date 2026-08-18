# V3.1 training foundation

`training.v3` is the isolated foundation for new Connect4 3D training runs.
Legacy `training/main_train.py` remains available for old experiments, but its
configuration and checkpoints are intentionally not imported into V3.

The current supported endpoint is a deterministic CPU smoke workflow. A bounded
cross-process actor pool and one shared inference owner per configured self-play
device are implemented and covered by CPU regression tests. The formal
generation loop remains guarded pending its cumulative replay/candidate state
machine, crash journal, and uncontended on-machine CUDA efficiency calibration.
A tiny CUDA effectiveness loop has already exercised shared inference,
self-play, replay, AMP learning, validation, and paired gating. `run` prints the
reviewed hardware, stage, and retention plan; it does not start training.

## Commands

Use Python 3.11-3.13 with `training/requirements-v3.txt`.

```powershell
python -B -m training.v3 print-config --config training/v3/configs/smoke_cpu.json
python -B -m training.v3 smoke --config training/v3/configs/smoke_cpu.json
python -B -m training.v3 smoke --config training/v3/configs/smoke_cpu.json --resume
python -B -m training.v3 run --config training/v3/configs/pilot_gpu_64x4.json
python -B -m training.v3 run --config training/v3/configs/v3_1_mini_128x6.json
```

Only `--config`, `--run-dir`, `--resume`, and `--device` are accepted overrides.
Unknown JSON sections, fields, and scalar types are errors. Operational runtime
topology and storage policy are recorded but excluded from the model-lineage
hash, so a run can move machines without pretending to be a new learning
experiment. MCTS lane count remains semantic because virtual-loss batching can
change visit targets; shard size remains operational.

## Presets

| Preset | Purpose | Model | Search schedule | Status |
|---|---|---:|---|---|
| `smoke_cpu.json` | Correctness and recovery | 16 channels, 1 block | 4 games, 8/2 full/fast sims | Executable |
| `pilot_gpu_64x4.json` | First cloud throughput and learning-signal check | 64 channels, 4 blocks | 64 games, 128/32 sims | Plan only |
| `v3_1_mini_128x6.json` | First intended baseline after pilot | 128 channels, 6 blocks | 160 games at 128/32, then 200 at 256/64, then 240 at 512/128 | Plan only |

The old ambiguous `v3_1_small.json` was removed. A 64x4 network is a pilot, not
a full-run target. The 128x6 Mini is about 1.82 million parameters, close to the
previous approximately 1.94 million-parameter CubeSprite V3 Mini. Model growth
requires a new run; it is not a hidden stage inside one config.

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
  bounded. The static planner warns about CPU/lane oversubscription and batch
  limits that cannot be filled by configured concurrency.
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

`keep_all` never proposes pruning. `archive_ack_prune` is still non-destructive:
`retention.py` only computes eligibility. A future explicit cleanup command must
revalidate the plan before deleting anything.

The Mini plan uses an 80% soft watermark, a 20 GiB hard reserve, a 1.25x active
window margin, and approximately 8 GiB archive bundles. It keeps the newest
three resumable checkpoints, two accepted models, one rejected model, every
unresolved candidate, all small manifests/metrics, and all audit replays.

An artifact is never eligible merely because an upload command succeeded. A
verified receipt must match its run-relative path, byte size, and SHA256. Raw
shards inside the expanded active window or beyond the learner cursor remain
pinned. No deletion or upload is performed by `smoke` or `run`.

## Human audit replay

Each smoke checkpoint selects representative completed games deterministically,
avoiding the shortest and longest anchors when enough games exist and filling
available P1/P2/draw outcomes. Files are written to:

```text
training/runs/<run_id>/samples/g<generation>/*.c4replay.json
training/runs/<run_id>/samples/g<generation>/audit_index.json
```

The replay files use the exact CubeSprite desktop protocol v1 and can be
imported directly by the app. They contain no training-only fields. Seed,
producer model, search counts, selection reason, and checkpoint checksum live in
the adjacent audit index.

## Smoke artifacts

```text
training/runs/<run_id>/
  resolved_config.json
  run_manifest.json
  manifests/generations/
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
resume scans backward to the newest complete generation. Pre-commit crash
reconciliation and an OS-level single-coordinator lock are deliberately still
formal-run blockers. File and directory `fsync` provides the intended
power-loss durability on the Linux cloud target. Windows smoke validates atomic
replacement and checksums, but does not claim directory-metadata durability
across sudden power loss.

See `STATIC_AUDIT.md` for the review decisions, on-machine pilot matrix, health
watch levels, and remaining production blockers.
