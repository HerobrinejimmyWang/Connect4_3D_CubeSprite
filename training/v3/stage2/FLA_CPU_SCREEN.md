# FLA CPU efficiency screen

This screen starts at the BAL-2 B8 capacity boundary (about 5.63M-5.68M
parameters). FLA remains a separate evidence view: BAL-2 run IDs, configs,
weights, and results retain their original lineage. The 6M point is a design
boundary, not a Flash qualification.

## Existing evidence and interpretation

- At about 2M parameters and 256 simulations, archived searched-state CPU
  means were 0.920 s for `gravity_resnet`, 1.131 s for
  `column3d_fusion_resnet`, and 2.086 s for `raw3d_resnet` on the same
  measurement machine. This supports a cost concern for dense 3D paths but
  does not isolate the cost of the `Conv3d` operator. Multiview encoders can
  also be slower without a 3D convolution.
- BAL-2's five B8 models have single-seed 1M and 3M offline and paired-match
  evidence. Their roughly 133M/240M search MAC estimates are descriptive,
  not CPU latency measurements.
- The BAL-3 CPU sweep used larger B10 models. It must not be reported as a
  BAL-2 B8 CPU measurement.

## Phase 1: latency and matched-capacity design

Measure the five original BAL-2 3M artifacts on one idle CPU with the native
Stage 2 PyTorch predictor and Classic-rule `NumpyMCTS`:

```powershell
python -B tools/stage2_fla_cpu_screen.py --sims 512 --repeats 3 --idle-s 10 --group-idle-s 60
```

Results go to `training/runs/stage2/fla/cpu_boundary_6m/idle10_group60/`.
The 10-second response interval and 60-second group interval follow the later
BAL-3 CPU cooling procedure; this screen uses three repeats rather than the
BAL-3 sweep's one. The older FLA Round 1 sweep recorded `idle_s=0`, so its
absolute times are contextual rather than directly matched. Outputs produced
without the specified intervals are preliminary calibration only and cannot
enter the new comparison table. Every per-model
JSON records the config/model SHA-256, CPU/runtime/thread information,
searched-state counts, and mean/median/p90/p95 latency. The summary is a
relative latency screen until a deployment CPU and hard limit are frozen.

The completed 512-simulation, 3-repeat screen on the recorded Windows CPU
(14 PyTorch threads, 45 searched responses per model) gives these means:

| 3M archived BAL-2 model | Mean s | P95 s |
|---|---:|---:|
| gravity B8 | 2.859 | 3.785 |
| raw3d-to2d B8 | 3.436 | 4.413 |
| column3d-v2 B8 | 3.945 | 5.216 |
| multiview3d B8 | 4.205 | 5.513 |
| winning3d B8 | 5.161 | 6.832 |

The separate 1M offline checkpoints for the three new designs, measured on
the same CPU and protocol, give column 2D 3.186/4.670 s, thin raw3d-to2d
3.309/4.173 s, and thin column3d-v2 3.369/4.365 s (mean/P95). The thinner
3D branches reduce mean latency by 3.7% and 14.6% relative to their archived
structural counterparts. These are different training checkpoints, so use
this as an architecture screen rather than an isolated operator-cost claim.
The per-model results and hashes are joined in the FLA evidence table.

Prepare the three implemented-model efficiency candidates:

```powershell
python -B tools/stage2_fla_efficiency_design.py
```

The generated configs and design manifest go to
`training/runs/stage2/fla/efficiency_6m/`. Each is within 3% of the archived
5,630,340-parameter BAL-2 gravity control. `column_2d_b8c192` uses only a
column encoder and a 2D trunk. `raw3d_to2d_thin_b8c192` and
`column3d_v2_thin_b8c192` each use one 48-channel 3D block, down from the
BAL-2 two-block 88-channel branch, and place most parameters in the 2D trunk.
The design command checks the actual parameter count and finite search output.
Generated configs are new V3 offline lineages and must never be described as
BAL-2 checkpoints or as trained models before their artifacts exist.
After 1M offline completion and donor SHA-256 verification, measure them with:

```powershell
D:\Users\24233\anaconda3\envs\pytorch\python.exe -B tools/stage2_fla_cpu_screen.py --source new --sims 512 --repeats 3 --idle-s 10 --group-idle-s 60
```

The new-donor output is under
`training/runs/stage2/fla/cpu_new_6m/idle10_group60/`.

## Phase 2: training and strength gates

The three new efficiency configs enter the same `standard_late`, seed-271828,
1M-position offline screen. In parallel with the slow CPU protocol, a bounded
serial Classic self-play screen may run across selected archived BAL-2 models and
the new designs. It uses each model's own 1M offline artifact as a
model-only warm start; the queue waits until all new offline workers exit and
validates every config, report, model format, lineage, and SHA-256. This early
self-play is screening evidence, not a speed-based promotion. The queue stops
at exactly 1M training positions per model and halts on an incomplete run or
archive boundary. Use:

```bash
/root/miniconda3/bin/python -B tools/stage2_fla_selfplay_queue.py
/root/miniconda3/bin/python -u -B tools/stage2_fla_selfplay_queue.py --execute --wait-minutes 120
```

Results go to `training/runs/stage2/fla/selfplay/r1/`. Continue candidates
that survive the eventual CPU and strength screens from their own
1M checkpoints to 3M; do not train a fresh 3M run. Keep Classic-rule
fixed-opening, color-swapped, fixed-time matches against the B8 gravity
control as the strength floor; report fixed-simulation matches separately.
Accepted and terminal self-play artifacts require separate match lines if a
candidate later enters online training. Offline loss, parameters, or MACs do
not establish playing strength or Flash qualification.

After the formal CPU screen, the live serial queue was revised while
`raw3d_to2d_b8` kept running. The original controller was stopped by PID and
a replacement controller took over the same worker without restarting its
run. Three not-yet-started B8 models with measured mean above 3.8 s were
removed: `column3d_v2_b8` (3.945 s), `multiview3d_b8` (4.205 s), and
`winning3d_b8` (5.161 s). The retained queue is gravity B8, column 2D B8,
raw3d-to2d B8, thin raw3d-to2d B8, and thin column3d-v2 B8. The first two
were already completed before this change. The prior queue state is saved as
`queue_state_before_trim.json`; the revised plan is recorded in
`queue_state.json` with `plan_revision=cpu_mean_le_3p8_plus_b6_screen_v1`.

The additional smaller-capacity candidate is **thin raw3d-to2d B6C128**:
six 2D trunk blocks and 128 trunk channels, with the existing one-block,
48-channel thin 3D branch. It has 2,082,052 parameters. It uses the same
`standard_late`, seed-271828, 1M offline data contract, but its strength
must not be treated as parameter-matched to the B8 models. Offline training
is armed to start only after the retained self-play queue drains. Then the
trained donor must pass the same 512-simulation, 3-repeat, idle-10/group-60
CPU screen before B6 self-play is launched. A result faster than 3.0 s is
acceptable; 3.8 s is the CPU screen upper bound for this addition. Earlier
B6C208 and B6C192 design-only drafts were superseded before training. The
user clarified that B6 means both blocks and trunk channels decrease to
B6C128; the preliminary random-weight probe is not formal latency evidence.

The remote prestart on `connect4_gpu_2608` uses the existing Stage 2 offline
data pools and 1M bounds. Treat the generated configs as the source of truth
for each candidate architecture; verify the live process, committed
checkpoint/model/report, and hashes before marking a run complete. Generated
logs and runs are local experiment artifacts and should not be committed to Git.

Refresh the provenance-aware local comparison table after CPU results arrive:

```powershell
python -B tools/stage2_fla_evidence_table.py
```

It writes `training/runs/stage2/fla/evidence_table/{table.json,table.md}` and
keeps archived 3M Elo separate from new 1M design evidence. Missing CPU
results remain explicitly pending; the table never labels a model qualified.
