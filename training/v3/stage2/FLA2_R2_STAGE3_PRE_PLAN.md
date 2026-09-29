# FLA2-R2: multi-rule finalist and training-system preflight

Status: two provisional finalists selected by the stable top-three CPU rerun;
the first five-rule V3 canary is running on physical GPU0.
This is the last Stage 2 FLA architecture screen and a
Stage 3 preflight, separate from the Classic-only CPU screen in
`FLA_CPU_SCREEN.md`. The working name is **stage2-FLA2-R2
(stage3-FLA-pre)**; it is not a deployment or Flash qualification label.

## Questions

1. Which of the two Classic finalists is the better **five-rule FLA model**
   at comparable training exposure and CPU search cost? The Classic 3M screen
   chooses entrants, not the multi-rule winner. Compare both at the same
   multi-rule budgets, then use color-swapped, rule-stratified matches and
   CPU latency before selecting a final architecture.
2. Does the V3 multi-rule training system preserve useful learning across
   rules? Audit equal-game generation, per-rule replay and learner exposure,
   candidate acceptance, retained per-rule peaks, and the hard regression
   floor. A single macro score cannot hide a rule collapse.

## R1 handoff and evidence boundaries

- The four Classic candidates have a 1M parent plus a **semantic-fork 2M
  child**, with fresh optimizer and replay. Call this 3M cumulative exposure,
  not one continuous 3M run.
- The formal four-model direct round robin uses physical GPU1 only, six edges,
  64 paired Classic openings per edge, colors swapped, and 256 simulations.
  The earlier three-edge GPU0 run remains pilot evidence. Four terminal
  evaluation snapshots and a separate, same-machine 512-simulation CPU screen
  determine provisional entrants: mean response time at most 3.3 s, with
  the documented strict 95% CI strength exception.
- Preserve terminal and accepted weights separately. FLA2-R2 warm starts from
  each entrant's committed accepted weights, as BAL5-R2 does. If the accepted
  root has no candidate metadata because no gate passed, V3 requires a prior
  compatible warm-start artifact instead; verify that every tensor equals
  the accepted root. The prior source can be an accepted V3 candidate from
  the parent 1M run or an original `standard_late` offline artifact. Preserve
  its corresponding V3 warm-start mode.
  Record both source SHA-256 values and the terminal checkpoint SHA-256 used
  for the R1 matches. A lower-capacity entrant is not parameter matched to a
  B8 entrant.

## First experiment: two five-rule V3 runs to 2M

Start no earlier than **2026-09-27 07:30 Asia/Shanghai**, after the two
entrants are recorded and physical GPU0 is idle. Run the entrants serially
on physical GPU0, each as an independent bounded V3 lineage with its own run directory,
optimizer, replay, gate, and accepted champion. Do not resume the Classic
child or load a Legacy checkpoint.

Freeze the executable BAL5-R2 rules and gate baseline:

- Rule order: Classic, p1 vertical ignored, p1 vertical forbidden, p1 layer-0
  ignored, p1 vertical plus layer-0 ignored. Use the registry hash from
  `BAL5_R2_RULE_REGISTRY`.
- 800 games per generation, **160 per rule**, with the BAL5-R2 256/32
  full/fast search split and its two opening-temperature variants.
- Use the BAL5-R2 R2 opening mixture: first eight plies in half the games
  use 0.5 times the normal temperature, starting at this fresh run's 1M
  training-position mark.
- Keep the existing 32-dimensional rule features and `role_rule_v1`
  global encoder for this controlled baseline. Use BAL5-R2's candidate
  cadence and sequential five-rule gate: equal-rule macro paired evidence,
  the same openings against the incumbent and each rule's retained peak,
  and a **5 percentage point maximum point-score regression** against each
  rule's peak. Inconclusive gates extend paired evidence; rejected models
  never produce replay.
- Use a one-card persistent actor/inference pool per run. Start with 16
  actors, four MCTS lanes each, batch limit 32 on GPU0. Record physical GPU,
  cgroup quota, process count,
  memory, throughput, and inference queue data. This is an operational
  topology, not an assumed strength optimum.
- First stop each run at a bounded canary, verify generation commit, all
  five 160-game buckets, replay rule codes, learner samples, checkpoint, and
  accepted hashes, then safely resume each to exactly 2M training positions.
  Start the second run only after the first run's 2M boundary is verified.
  Never restart a finished run or prune its evidence.

## Evidence and decisions after 2M

For each architecture report: model/config/registry hashes; per-generation
games and raw positions by rule; active replay-window share and learner
sample share by rule; per-rule validation losses; candidate/gate chronology;
macro score and CI; each rule's incumbent and peak W/D/L, first/second split,
point score, CI, and hard-regression verdict; terminal and accepted SHA-256;
and same-protocol CPU latency. A macro win with a breached per-rule floor is
not a promotion.

The baseline has **equal games, not necessarily equal training positions**:
rule-specific game lengths and forced-pass samples can skew the cumulative
position pool. Measure the skew before changing replay sampling. If it is
material, compare a rule-balanced learner sampler against the baseline at
matched generation, raw-position, and optimizer-token budgets. Keep the
split by game and preserve the five rule codes. Record how every rule
contributes to the active window, learner batches, and validation.

The current model injects role plus 32 rule features once before the 2D
trunk through a scale/bias global encoder. If a finalist fails on certain
rules, test a rule-specific injection alternative (for example, per-block
conditioning) in a new V3 model lineage, matched for parameter budget and
CPU response latency. Do not change the injection inside the first two
baseline runs or attribute a difference to architecture when the rule
encoding also changed.

Evaluate final multi-rule strength with five-rule paired openings and color
swaps at the same search budget. Include fixed-time CPU play before a Flash
promotion decision. CPU deployment hardware and a formal Flash latency
threshold remain open; the 3.3 s screen is a selection rule, not qualification.

## 2026-09-27 R1 handoff result and CPU control

All four Classic child runs reached their separate 2M boundaries, and the
physical-GPU1 six-edge round robin completed. The formal local terminal CPU
screen used 512 simulations, three repeats, 10 s between responses, 60 s
between model groups, and 45 searched responses per model. Its means were
4.857 s (gravity B8), 5.789 s (raw3d-to2d B8), 4.990 s (thin raw B8), and
4.205 s (thin raw B6). All exceed the 3.3 s selection threshold. Raw3d-to2d
B8 led the fixed-simulation round robin, but its Elo 95% CI did not strictly
clear every other model's CI. The immutable R1 selection therefore has no
finalists. It remains the record for that host state; a separate R2 selection
uses the stable rerun below.

An archived gravity control with **the identical weight SHA-256** was rerun
under the same CPU protocol. Its searched-response mean changed from 2.859 s
  in the original screen to 4.601 s on 2026-09-27, despite matching recorded
CPU model, thread counts, search corpus, and simulation count. The benchmark's
timed search loop was unchanged. This establishes a large machine-state drift
in the absolute latency measurements; its cause has not yet been isolated.
Keep both raw results. Re-establish a stable CPU baseline and repeat the four
terminal measurements before applying the frozen 3.3 s rule or changing the
entrant set. Do not rescale the terminal latencies by the control ratio or
silently relax the threshold.

The paired parent-1M versus child-2M fixed-point MCTS/NN diagnosis and its
tree concentration metrics are recorded in
`training/runs/stage2/fla/latency_1m_vs_2m/diagnosis.md`. It finds small,
architecture-dependent search-shape effects at one point and also confirms
large same-model host/runtime drift; it is not a substitute for a full-corpus
CPU rerun or a playing-strength match.

## 2026-09-27 stable top-three rerun and R2 handoff

The archived gravity control measured 2.519 s before and 2.455 s after the
three leading models (2.6% within-bracket difference). Both were within the
predeclared 15% bound relative to its original 2.859 s result. Under the same
512-simulation, three-repeat, 10 s/60 s protocol, the terminal means were
3.149 s for raw3d-to2d B8, 2.716 s for thin raw B8C192, and 2.070 s for
thin raw B6C128. Each passed the 3.3 s architecture-selection screen.
The old R1 CPU measurements remain unchanged. This bracket is provisional
because the control itself is about 12-14% faster than its original value;
it does not establish deployment qualification.

The official GPU1 Elo order among these passing candidates selects
`raw3d_to2d_b8` and `raw3d_to2d_thin_b8c192`. The selection, bracket,
CPU summary, match inputs, terminal snapshots, and hashes are recorded in
`training/runs/stage2/fla/selection_3m/r2_stable_top3/` and
`training/runs/stage2/fla/cpu_terminal_3m/recheck_top3_20260927/`.
The first FLA2-R2 V3 canary uses GPU0; the second architecture is queued
behind its verified 2M boundary. Physical GPU1 remains free for other work.

## 2026-09-28 temporary GPU1 pool fork

The first line's generation 16 committed 1,025,352 training positions and
accepted `candidate-g000016-s00004013-d00256338`. Use that accepted artifact
as a fixed near-1M source for a separate `pool120x2x5_g16_gpu1` V3 lineage.
The parent checkpoint also contains optimizer and replay state, but V3 changes
to the self-play pool require a semantic fork with fresh optimizer and replay.
Do not call this an exact 1M checkpoint or a continuous parent resume.

The fork produces 1,200 games per generation: for each of five rules, 120
games retain temperature 1.0 through ply 27, and 120 use 0.5 temperature
through ply 7 then 1.0 through ply 27. The V3 alternating game-ID route and
position-balanced learner sampler implement the equal halves. The mixture is
active from the first fork generation. Keep the 256/32 search split, loss
weights, replay contract, candidate-position cadence, and five-rule gate
unchanged. Set 12 actors, four lanes, and batch 32 on physical GPU1 to share
the 40-vCPU quota with the original GPU0 line. This actor count is an
operational setting; record throughput and contention before drawing speed
conclusions.

Bound the fork at 80k for a canary, verify 240 games per rule, 600 games per
temperature route, learner exposure to both routes, replay, commit, checkpoint,
and accepted hashes, then resume to 1M child positions. Compare to the
original line at matched *child* positions while explicitly noting its
retained replay and optimizer history. The fork tests this pool schedule;
any measured strength difference is not an isolated pool causal estimate.

### Revised temperature control

The 120 x 2 x 5 proposal changes both generation size and temperature, so it
cannot isolate the timing or usefulness of the two-temperature mixture. It
was signalled to drain at the next V3 generation boundary; retain its raw
evidence and do not use it as the formal control.

The replacement `no_mixture_g16_gpu1` starts from the same g16 accepted model
at 1,025,352 parent training positions. Keep **160 games per rule**, five rules,
800 games per generation, the 256/32 search split, 28-ply high-exploration
window, learner/loss/replay settings, and position-based gate cadence. Disable
the opening-temperature mixture for the entire fork, including after its own
1M position mark. Use 12 actors on physical GPU1 as an operational sharing
choice while the original GPU0 line continues. Bound the new V3 lineage at
80k for validation and then 1M child positions; verify each generation has
160 games per rule and no lowered-temperature route.

Compare it with the original line, which enables the 50/50 temperature
mixture after 1M positions at 160 games per rule. Both use the same g16
accepted starting weights, but the new control has fresh optimizer and replay
while the original line continues its old optimizer and replay. Report that
lineage difference alongside gate, per-rule strength, replay composition, and
game-length evidence. If results are close or contradictory, a matched
mixture-on fork from g16 with the same fresh-state protocol is needed before
claiming a causal effect of temperature timing.

## Five-rule terminal comparisons and pool-quality audit

Run the same-architecture pool comparison on physical GPU1 once its two
terminal weights are verified. After thin B8 reaches its independently
verified 2M boundary, run the architecture comparison on physical GPU0.
The two comparisons may overlap because each uses one physical card:

1. `raw3d_to2d_b8` versus `raw3d_to2d_thin_b8c192` at their independent
   five-rule 2M terminal checkpoints.
2. Normal-mixture `raw3d_to2d_b8` at 2M versus the same architecture's
   `no_mixture_g16_gpu1` 1M-child terminal checkpoint. The latter starts from
   the normal line's g16 accepted weight at 1,025,352 positions, but has fresh
   optimizer and replay history. Treat this result as directional, not as an
   isolated temperature-policy effect.

For **each comparison**, use the same frozen five-rule opening set: 100
opening pairs per rule, with colors swapped within each pair. Thus each
comparison has 500 pairs and 1,000 games. Use 256 simulations and the same
search settings for both models, report per-rule and aggregate W/D/L, first
and second player scores, paired 95% confidence intervals, terminal checkpoint
and evaluation-snapshot SHA-256, opening-manifest SHA-256, and physical GPU.
Preserve each rule result so an interrupted comparison can resume only the
missing rule. Do not interpret a pooled win as sufficient if a rule regresses.

Before changing replay design or rule conditioning, audit the actual 2M data
pool. Count generated games and raw positions by rule and temperature route;
identify unique board/rule/player states and duplicate frequency in the active
replay window; measure the available train and validation positions by rule and
the recorded learner sampling by route. Historical learner metrics do not log
actual sampled positions by rule, so report that gap instead of substituting
the active-window distribution. Inspect gate coverage by rule separately.
The Stage 2 frozen `standard_late` and `mixed_late` artifacts each contain 1M
train positions and 50k validation positions sampled from long Classic source
trajectories; the user's roughly 15M source exposure is not 15M distinct
frozen training examples. Compare their source generation span and frozen
sample count with the new pool, while marking the rule-distribution mismatch.
Equal games per rule need not mean equal positions or learner exposure. The
five-rule matches test playing strength; pool statistics test coverage and
reuse. Specify a matched pool-size or sampling experiment only after both
sets of evidence are available.
