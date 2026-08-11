# V3.1 static audit and cloud pilot plan

This document records the conclusions of three static review rounds performed
without a CUDA machine. It separates implemented contracts from GPU-dependent
assumptions so the first cloud session can be used for measurements rather than
rediscovering configuration errors.

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
- No-op actor/search hardware controls moved to `runtime`; the formal entry is
  still guarded until their process scheduler exists.
- MCTS virtual lanes now use a batch predictor. Torch inference supports
  `[N,6,5,5]` input in one forward pass and reports inference batch metrics.
- FP32 losses under autocast, `-inf` legal masking, pinned memory, non-blocking
  H2D, and conditional prefetch removed the known AMP failure and obvious loader
  stalls.
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

The framework is suitable for CPU correctness smoke and for producing a
reviewed cloud pilot plan. It is not yet suitable for an unattended formal run.
The guarded status is intentional, not a missing flag.

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

Remaining blockers:

1. Implement the multiprocess actor pool and one bounded shared inference
   service per self-play GPU.
2. Implement the formal generation state machine, including append-only replay
   cursor, candidate trigger by consumed positions, inconclusive gate extension,
   and safe shutdown/drain.
3. Implement a generation draft/reconcile journal and an OS-level
   single-coordinator no-clobber lock.
4. Integrate archive catalog creation, verified transfer receipts, and a
   separately invoked prune command with disk-watermark backpressure.
5. Benchmark actual GPU batch fill, queue wait, forward latency, game throughput,
   learner throughput, and peak VRAM/RAM.
6. Run a short learning-signal pilot before accepting the Mini schedule for an
   unattended run.

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

V3 emits the first three as watch warnings. A formal scheduler should combine
them with validation, gate, historical milestone, and tactical-suite evidence
before stopping or changing parameters.

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

At the soft watermark, stop admitting new games at a shard boundary and create
an immutable archive bundle in `archive_staging/<id>.partial`. The bundle
manifest records path, size, and SHA256. Rename only after complete local
verification. After transfer, the receiving machine verifies every entry and
returns a signed/recorded receipt containing the archive-manifest checksum.

The cloud side may plan pruning only after receipt verification. It retains at
least 1.25 times the active replay window, current and previous accepted models,
the latest three resumable checkpoints, unresolved candidates, metrics,
manifests, and audit replays. At the hard reserve, if a verified archive cannot
be produced, stop safely instead of deleting unacknowledged data.

`retention.py` implements only the deterministic eligibility calculation. There
is deliberately no automatic deletion command in this revision.

The committed-generation validator now authenticates the checkpoint cursor,
every replay NPZ/manifest/ready triplet, model artifacts, audit index, and each
CubeSprite replay, and it can fall back to an older complete generation. Before
formal training is enabled, add a single-coordinator no-clobber lock and a
pre-commit draft/reconcile journal so a crash between individual artifact
publishes can be retried safely. Linux is the durability target because its
directory renames are followed by `fsync`; Windows is supported for smoke and
logical recovery checks, not a sudden-power-loss durability guarantee.
