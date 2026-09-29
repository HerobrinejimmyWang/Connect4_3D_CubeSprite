# FLA3-pre raw B8 five-rule pool extension

This experiment starts from the committed FLA2-R2 raw3d-to2d B8 **accepted**
model at generation 32 (the last acceptance was generation 31, SHA-256
`b969a7c53f0f7197c99f2c27e81d7830f39e93426ab913d33baecb807117d305`).
The parent consumed 2M learner positions. Changing the five-rule gate or
self-play search budget changes V3 semantics, so the extension is a separate
warm-start lineage with fresh optimizer and replay. Its 10M target is **child**
positions; parent and child must not be called one continuous 12M run.

## Search-budget decision

Use exactly that accepted weight under the V3 rule engine and serial MCTS.
First compare 256 and 512 simulations, with root noise disabled and cpuct 1.5,
on 10 deterministic positions per rule using 0/8/16/24-move opening prefixes.
Record both visit vectors, top actions, root values, inference counts, seeds,
and model hash. This diagnoses changed decisions; it does not establish which
budget plays better.

Then play 512 versus 256 with the same weight, 50 paired openings per rule,
colors swapped in each pair, for 250 pairs and 500 games. Reuse the frozen
five-rule opening manifests from the terminal comparison. Four evaluation
workers per physical GPU run serial one-lane search. Save each rule's games,
paired bootstrap 95% CI, opening hash, worker timing, and the aggregate result.
Use 512 for the new run's `full_search_sims` only if its equal-rule point score
is above 50%, its 95% CI lower bound from pair resampling **within each rule**
is above 50%, and no individual rule has a 95% CI entirely below 50%.
Otherwise keep 256 and label the strength difference unproven. The 32-simulation fast branch
and 256-simulation gate search are separate contracts and stay unchanged.
Inspect per-rule results before interpreting the aggregate as a multi-rule gain.

## New V3 child

The child keeps raw3d-to2d B8, five rules, 800 games per generation (160 per
rule), the existing 50/50 opening-temperature mixture, replay sampling, loss
weights, 4 MCTS lanes per actor, and batch limit 32. The mixture starts at
child position zero because the parent had already entered its mixed phase.
The new gate uses `incumbent_first`: complete candidate-versus-incumbent
evidence first, then peak comparisons on the same openings only when the
macro test warrants promotion. Keep 50 initial paired openings per rule,
25-pair increments, 200-pair cap, 95% confidence, and a 5% per-rule peak
regression tolerance. An accepted candidate still requires all five peak
comparisons.

Start with 24 total actors across `cuda:0` and `cuda:1`; this is a conservative
two-card topology hypothesis, not a measured B8 optimum. Use both cards for
gate evaluation with four replicas per card, following the measured fixed-gate
topology preflight. Start the warm model at learning rate 5e-5 and decay to
2.5e-5 at child position 5M. Preserve exact config, code, input artifact,
opening, generation commit, replay, checkpoint, accepted, and gate hashes.

Run to 80k first. Verify 160 games per rule, replay rule coverage and SHA-256,
learner metrics, committed checkpoint and accepted weights, and actual device
usage before continuing. Because the child starts with an accepted model,
`candidate_due` uses the 300k regular gate interval. Stop at 320k to verify
the first committed optimized gate; then continue with an explicit 10M child bound. The V3
archive/prune contract and 10-GiB hard reserve remain active; never manually
prune unreceipted artifacts. The watcher pauses on a failed check rather than
silently restarting or altering an existing lineage.

## 2026-09-29 search-budget outcome and launch

The fixed-position diagnostic completed 50 states (10 per rule): 44/50
top actions agreed between 256 and 512 simulations. This is a decision
agreement measure only. Its summary SHA-256 is
`838639665f2addb884e3813649d2227e94e8b1274f32dea53f42e89540106566`.

The swapped-color same-weight match completed 50 opening pairs per rule,
500 games total, with 512 winning 298 and losing 202 (59.6% point score).
The equal-rule, within-rule pair bootstrap 95% CI is 57.0%-62.4%.
Per-rule 512 point scores are Classic 56%, vertical ignored 57%, vertical
forbidden 60%, layer0 ignored 64%, and vertical-and-layer0 ignored 61%.
Each rule's paired 95% CI lower bound exceeds 50%. The aggregate match
summary SHA-256 is
`dd43889bf17e608c802395cfbce41d1d4370d196abc6c495e7dead45dfdfc963`.
The local raw evidence is under
`training/runs/stage2/fla2_r2/search_budget_256_vs_512_accepted_g31/`;
its two summary hashes match the remote originals.

The conditional decision therefore sets **512** for the child's high-search
self-play branch, while retaining 32 for fast search and 256 for gate
evaluation. The new child config SHA-256 is
`20c022b8f2b03637b82e2e97682621e59972a2187602d5203991845950ded433`.
Its 80k canary started on both physical GPUs after the optimized-gate tests
passed. This paragraph records a launch, not a completed canary or 10M pool.

## 2026-09-29 80k canary and first-gate boundary correction

The 80k canary completed with two generations, five-rule replay coverage,
learner metrics, and verified generation/checkpoint/accepted hashes; receipt
`training/runs/stage2/fla3_pre/raw_b8_pool_10m/receipts/to80000.json` was
published remotely. A follow-up run safely committed 160k at generation 2,
with `gate_verdict=not_run`. The initial launcher incorrectly expected a gate
at 160k and stopped after checking that committed boundary; it did not fail
inside V3 training. Since the warm-start already has an accepted champion,
V3 uses `gate.candidate_train_positions=300000`, rather than the 150k
no-champion bootstrap interval. The original plan and failed launcher state
remain preserved. An explicit amendment records the 320k check, without
changing the config or lineage hash. The bounded V3 run resumed from the
verified 160k commit toward 320k, and will continue toward 10M only after
the first gate passes its evidence check.

## 2026-09-29 first gate and recovery

The 320k boundary committed as generation 4 (`f5e15e43203406a22167b33e7c7ea73939668aedba51830d183fe054359055c4`).
The optimized gate completed 75 swapped-color opening pairs per rule, including
all five incumbent and peak comparisons, and accepted the candidate. Its gate
artifact SHA-256 is `84e9e93b2ab51edcde1d5b037d1654f7cef99aca1d91eee46acfcbb56742a402`;
the committed accepted weight SHA-256 is
`d1bf2bfde9898423de9c5ce29dee50b566bceaf922eb67ff558256e629f4d3d0`.
The launcher briefly marked this successful V3 boundary as failed because its
verifier looked for gate rows in `metrics.jsonl`; formal gate evidence resides
in `metrics/gate_g000004.json`. The failed launcher state was preserved as
`watcher_state_failed_at320k.json`. The verifier now checks the gate artifact
and generation commit directly. Receipt `receipts/to320000.json` was published,
and the bounded V3 run resumed from 320k toward the separate 10M child target.
This is an active child run, not a completed 10M pool.

## 2026-09-29 raw B8 actor topology screen

At 24 actors the recent 800-game generations took roughly 988–1056 seconds,
with mean inference batches near 19/32 on both cards. A five-second cgroup
sample used about 12.6 of 40 allotted CPU cores; both GPU utilization samples
were near 30%. These observations motivate a controlled actor-count screen,
but do not establish a bottleneck by themselves.

The V3 process received a generation-boundary drain request after the last
verified generation 9 commit. The independent watcher
`tools/stage2_fla_raw_topology_screen.py` will verify the next committed
generation, then use its single checkpoint for two 128-game repeats at 36,
40, and 24 actors, in that order within each repeat. All points keep two GPUs,
512/32 search, four lanes, batch 32, and 1 ms inference timeout. The 24-actor
point is a matched control; the screen does not change model lineage or select
a new production topology. Outputs and hashes go under
`training/runs/stage2/fla3_pre/raw_b8_pool_10m/topology_screen_24_36_40/`.
After all points pass evidence checks, the watcher resumes the bounded 10M V3
run at the original 24 actors. If a check fails, it leaves the run at the
verified committed boundary for investigation.

## 2026-09-29 actor36 operational switch

The two-repeat topology screen completed. On the same generation-10 checkpoint
and 128-game workload per point, mean simulation throughput was 6,673.9/s at
24 actors, 8,185.6/s at 36, and 7,975.3/s at 40. The 36-actor point was fastest
in both repeats. The complete result SHA-256 is
`c9b551808d3761cc749c73ef700015354f93553c3c43edbbb23d5a51f6f186fe`.

At the user's request, the V3 process was force-stopped during generation 12.
Generation 11 remained the last committed boundary (commit SHA-256
`c1be04be79da44bc060e043d217a35e0daed03d6cc3fe4e58dddb750dc5d2ebc`).
The interrupted generation-12 draft was empty and in `started` phase. Its
original bytes and the dead coordinator lock were moved to
`actor36_switch/recovery_originals/` with SHA-256 recorded in
`actor36_switch/recovery_receipt.json`; no committed generation was removed.
The generation-11 checkpoint, accepted artifact, and all 240 committed replay
shards passed SHA-256 checks before resume.

The new runtime config changes only `actor_processes` from 24 to 36; its
SHA-256 is `61f1fb443c54515f48d2469d09c6c593ccf2ce3e340a3be840e322eef6296cf3`,
while the semantic config hash remains
`4b6c16f17ce0ae89084b5de43867c8ab0c79e54653d8dddbeb72eb0adb1bc6f0`.
The formal V3 process resumed from generation 11 with the original 10M bound.
The actor36 controller records and validates its first new generation commit;
process launch alone is not evidence that generation 12 completed.
