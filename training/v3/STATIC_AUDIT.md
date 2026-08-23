# V3.1 static audit and cloud pilot plan

This document records the conclusions of three static review rounds originally
performed without a CUDA machine, followed by a minimal CUDA effectiveness
check. It separates implemented contracts from performance assumptions so an
uncontended cloud session can be reserved for meaningful measurements.

## Round 1: audit findings

### Hardware and concurrency

- `actor_processes` previously had no effect and MCTS "threads" were sequential
  virtual-loss lanes. CUDA inference was batch one with a synchronization back
  to CPU for every leaf.
- The former CUDA preset therefore advertised parallelism it did not provide.
- The learner's AMP path used `-1e9` in a possible FP16 tensor, which overflows
  before training starts.
- Gate games and checkpoint/replay writes are synchronous and can dominate or
  pause the loop. Any future asynchronous design needs bounded queues and
  backpressure before it starts background producers.

### Training effectiveness

- Fast-search positions were discarded, wasting completed-game WDL targets.
- `optimizer_steps=1000` was mathematically incompatible with the replay token
  budget of a 64-game batch.
- A single temperature/noise setting contradicted the stable historical V3
  schedule and the recorded collapse of a high-temperature branch.
- The 64x4 model has about 0.31 million parameters and should not be presented
  as the successor to the approximately 1.94 million-parameter V3 Mini.
- Validation was split but never evaluated. Health metrics lacked game-length
  variance, short-game rate, first/second-player results, opening diversity, and
  search-kind policy entropy.
- The fixed gate cost could exceed the self-play cost of a small generation,
  and several generated openings were duplicate empty or D4-equivalent states.

### Storage and auditability

- `shard_games` was unused; one whole cycle became a single shard.
- Long-run retention, archive receipts, disk watermarks, representative games,
  and a committed accepted-model pointer did not exist.
- Selection JSON expanded every sample ID and would grow unnecessarily.
- V3 retained complete moves in memory but discarded them after replay packing,
  so no checkpoint game could be inspected in CubeSprite.

## Round 2: implemented corrections

- Strict nested schedules replaced static search, exploration, gate-search, and
  learning-rate fields. Generation moved from semantic config into run state.
- Actor/search hardware controls moved to `runtime`; a bounded actor process
  pool and shared accepted-model inference owners now implement that topology.
- MCTS virtual lanes now use a batch predictor. Torch inference supports
  `[N,6,5,5]` input in one forward pass and reports inference batch metrics.
- FP32 losses under autocast, `-inf` legal masking, pinned memory, non-blocking
  H2D, and conditional prefetch removed the known AMP failure and obvious loader
  stalls.
- CUDA effectiveness testing exposed another AMP state bug: GradScaler could
  skip an overflowing optimizer step while replay tokens and resumable cursors
  still advanced. Skipped steps now stop the current learner call without
  consuming tokens, advancing the LR scheduler, or changing deterministic sample
  state; the next call retries at the reduced scale.
- Every position is persisted. Fast rows train WDL only; full rows train WDL and
  policy. A stable early forced-full ply avoids empty-board over-sampling.
- Replay is split by game count and committed with checksum, measured storage,
  and a ready marker. Selection uses compact digests.
- Validation and self-play health metrics were added. Opening generation is D4
  deduplicated. Gate schedules declare initial/increment/max pair counts.
- A pure no-DDP hardware planner resolves single-GPU staging or multi-GPU role
  split with bounded queue sizes and oversubscription warnings.
- A pure retention planner requires verified archive receipts and never deletes.
- Checkpoints export strict CubeSprite protocol-v1 representative games with a
  separate provenance index. A generation commit is published last.

## Round 3: static acceptance state

The framework supports CPU correctness smoke and an explicitly bounded,
synchronous multi-generation runner. The Stage 1 configs now contain reviewed
P6 auxiliary and occupancy weights; execution remains opt-in and bounded, with
CUDA, archive mode, and disk-reserve preflight.

The third review also closed issues found only after integration:

- single-coordinator smoke checkpoints and generation commits are immutable;
  the latest pointer carries a commit checksum, and resume verifies the
  checkpoint cursor, replay triplets, accepted/candidate models, audit index,
  and every portable replay before restoring;
- position-based LR stages are applied at learner batch boundaries and remain
  exact across checkpoint restore;
- sequential gate looks use Bonferroni-adjusted decision confidence;
- maximum opening suites skip terminal and immediate-win prefixes instead of
  aborting generation;
- MCTS lane count is part of the semantic hash, while shard size is operational;
- executable smoke rejects GPU presets, and offline hardware plans preserve the
  explicitly configured learner while flagging the CUDA inventory as unverified.

Completed P7 connections:

1. The bounded scheduler restores cumulative replay/learner/candidate state and
   uses only the committed champion for self-play.
2. Each generation is protected by an OS-level coordinator lock and a staged,
   checksum-bound commit journal. Signals drain at a committed generation boundary.
3. Incremental archives are verified locally entry-by-entry; cloud pruning is an
   explicit receipt-gated command. Target presets pause around 70% use and keep a
   10-GiB hard reserve plus archive staging headroom.
4. B4/B6 target GPU batch fill, throughput, learner throughput, and lane quality
   were measured on 2026-08-23.

P6 completed on 2026-08-23 with 896 games/13,255 Replay V2 positions, dataset
fingerprint `d117c989c6d6c588e62687d667095a5b234882371920375d386e68b453458e82`, and
two seeds x five variants x 40,000 train positions. The frozen Stage 1 losses
remain `reply=0.15`, `future_occupancy=0.15`, `moves_left=0.05`; occupancy class
weights are `5.0/5.0/0.35561042132416815`. All-head policy loss improved under
both seeds, while WDL was mixed, so this is learning-system calibration rather
than strength evidence. The next admission evidence is the bounded target-GPU
scheduler canary and its verified local archive receipt.

## Minimal CUDA effectiveness result

An isolated, deliberately tiny 4-actor run on an RTX 3080 Ti completed the
accepted-model shared-inference path, four self-play games, replay construction,
one real optimizer update, a non-empty validation split, and a paired gate. The
learner required two AMP scale reductions before the successful update; the
accepted update had finite loss and gradient norm and changed model weights.
Shared inference merged 362 requested positions into 140 batches, with a maximum
batch of four. The gate was inconclusive after one pair, as expected for a tiny
random-model check.

This establishes functional connectivity only. Another workload occupied the
machine, and the search/model sizes were intentionally tiny, so wall time,
throughput, CPU/GPU utilization, and the observed batch-size distribution are
not efficiency evidence.

## Uncontended 3080 Ti short-pilot result

The later uncontended window exposed a 20-core CPU quota and 30-GiB memory quota
despite the container seeing 80 host threads and 125 GiB. All CPU percentages
below therefore use cgroup accounting rather than host-wide `vmstat`.

For the 64x4 model at 128/32 full/fast simulations and batch limit 32:

- 18, 20, and 22 actors at 4 lanes produced about 4214, 4207, and 4145
  simulations/s. Eighteen is the conservative default; 18-20 is the plateau.
- At 18 actors, 2/4/6/8/10 lanes produced about
  2622/4236/5218/5949/6434 simulations/s.
- On 32 fixed positions at 128 simulations, top-action agreement with serial
  MCTS was 100% through 6 lanes and 96.9% at 8 and 10. Mean root-value absolute
  error was 0.005/0.026/0.050/0.076/0.101 for 2/4/6/8/10 lanes.
- The selected provisional topology is therefore 18 actors x 6 lanes, batch 32;
  4-6 lanes is the quality-conscious range. This calibration applies to the
  64x4 pilot, not automatically to the 128x6 Mini.
- Shared-inference testing revealed that whole actor requests could previously
  exceed the nominal batch limit (observed maximum 39 for a limit of 32). The
  service now defers the request and enforces a hard limit, with a regression
  test covering the boundary.

At 18 x 6 the shared-inference batch averaged 28.2 and reached the hard limit of
32, but active GPU utilization was still only about 6.4% and VRAM about 338 MiB.
The 64x4 self-play model is far too small to saturate a 3080 Ti; raising lanes
beyond 6 buys throughput by changing search semantics rather than by solving
GPU starvation cleanly. Learner samples briefly reached roughly 15-18% GPU in
the tiny closed-loop runs. Final hardware economics require the 128x6 Mini and
longer searches, after the formal scheduler produces a stable champion.

## 2026-08-23 dual-3080-Ti target calibration

On the uncontended `connect4_gpu_2608` 40-vCPU/60-GiB cgroup, single-service
role-split self-play converged to 24 actors x 4 lanes, batch 32, and a 1 ms
timeout for both B4C64 and B6C128. The B4 192-game fine scan measured about
5266/5586/5496 simulations/s at 20/24/28 actors. The B6 96-game scan measured
about 4502/4696/4613 at the same points. Batch 64 reduced throughput for both.

Random-weight paired lane checks did not expose a playing-result regression:
B4 lane 4 versus serial lane 1 scored 17:15 and B6 scored 16:16, with colors
swapped over 16 openings. Fixed-position targets still drifted, so this is not
a stable-champion strength claim and does not authorize lanes above four.

With no concurrent learner, two self-play inference owners and 40 actors x 4
lanes reached about 8027 simulations/s for B4; the B6 single point reached about
7146. The formal plan retains `cuda:0` for the learner and `cuda:1` for
self-play. A bounded B6 batch-256 AMP loop completed four optimizer steps at
about 786 positions/s and changed weights, but it does not validate the missing
multi-generation scheduler or a production schedule.

## Minimal cloud pilot matrix

Use `pilot_gpu_64x4.json`; do not start with the Mini long-run preset.

### A. Search and inference calibration

Use a fixed accepted/random producer and fixed seeds. Test only enough games to
stabilize throughput:

| Case | Full/fast sims | Actors x lanes | Batch limit | Purpose |
|---|---:|---:|---:|---|
| A1 | 128/32 | 8 x 2 | 16 | conservative baseline |
| A2 | 256/64 | 8 x 2 | 16 | search-quality/throughput tradeoff |
| A3 | 512/128 | 24 x 4 | 32 | historical-strength comparison |

Record games/s, sims/s, average and p95 inference batch, fill ratio, queue wait,
forward time, actor blocked/idle time, GPU utilization, VRAM, RAM, and wall-time
variance. Do not add multiple CUDA inference processes to one GPU merely to
raise utilization.

On a fixed set of positions, also compare 128/256/512 full search by top-action
agreement, policy KL/entropy, and root-value stability. Select the lowest budget
that preserves the 512-sim reference signal sufficiently for the pilot.

### B. Learning-signal pilot

Run only a few candidate intervals. Require:

- finite policy and WDL losses;
- nonzero full-policy rows in every learner chunk;
- stable train/data ratio at or below the configured token budget;
- validation WDL/Brier/calibration not degrading persistently;
- no rapid collapse in game length or opening diversity;
- paired gate artifacts reproducible from the same opening manifest.

Only then switch to `v3_1_mini_128x6.json`. Do not tune model size, exploration,
search budget, LR, and gate threshold simultaneously.

## Health watch levels from the previous run

The historical runbook treated these as correlated warning signals, not
independent stop rules:

- mean game length below 18 and falling;
- game-length variance below 50, especially below 30;
- more than 10 games of length at most 12 per 100 games;
- low value loss combined with short or repetitive games;
- weak anchor/teacher score, non-finite loss, worker death, or checkpoint error.

V3 emits the first three as watch warnings. `stability.py` now combines at
least two behavioral symptoms across at least two consecutive generations and
is regression-checked against the archived collapse/recovery traces. The bounded
formal scheduler persists that history and, when the assessment requests a
pause, commits the active generation before stopping at its boundary. It does
not change parameters automatically; the operator still combines this evidence
with validation, gate, historical-milestone, and tactical-suite results.

## Two-GPU plan (P2)

```text
configured learner GPU: AdamW/AMP, candidate snapshot, checkpoint snapshot
each self-play GPU:     one accepted-champion inference service and actor group
CPU:                    MCTS actors, shard writer, replay loader, coordinator
```

Each self-play GPU changes networks only between games and acknowledges the
accepted model ID. Candidate models never produce replay. Gate waits for
in-flight accepted games to drain, then evaluates candidate and incumbent.
Learner and self-play GPUs may overlap during ordinary self-play/learning. With
more GPUs, add one inference service and actor group per extra GPU; keep one
learner and no DDP until evidence shows the learner is the bottleneck.

## Disk and archive procedure

At the soft watermark, finish the active generation and create an immutable
archive bundle through `archive_staging/<id>.partial`. The cloud verifies each
source before and after packing, then publishes the tar plus a path/size/SHA256
manifest. After transfer, the receiving machine verifies the tar and every
entry, materializes the trace tree, and returns a recorded receipt containing
the archive-manifest checksum.

The cloud side may plan pruning only after receipt verification. It retains at
least 1.25 times the active replay window, current and previous accepted models,
the latest three resumable checkpoints, unresolved candidates, metrics,
manifests, and audit replays. At the 10-GiB hard reserve, if a verified archive cannot
be produced, stop safely instead of deleting unacknowledged data.

`retention.py` remains the pure eligibility calculator. `archive.py` and
`tools/manage_v3_archive.py` add an explicitly invoked deletion path that
revalidates every file against a locally verified receipt immediately before
unlinking it; there is still no automatic deletion inside the trainer.

The committed-generation validator now authenticates the checkpoint cursor,
every replay NPZ/manifest/ready triplet, model artifacts, audit index, and each
CubeSprite replay, and it can fall back to an older complete generation.
`formal_journal.py` provides the single-coordinator no-clobber lock and a
checksum-bound pre-commit draft/reconcile journal; the bounded writer is wrapped
by both. A fully staged commit is recoverable, while any other partial state
blocks for explicit operator recovery. Linux is the
durability target because its
directory renames are followed by `fsync`; Windows is supported for smoke and
logical recovery checks, not a sudden-power-loss durability guarantee.
