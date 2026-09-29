# Stage 3 five-rule gate protocol

Use this protocol only for a new V3 single-model five-rule lineage. Existing
BAL5-R2 and FLA2-R2 runs retain `gate.multirule_evaluation_mode=complete`.
The mode is part of the semantic config hash; switching a live run is not a
topology change.

For the Stage 3 config, set `gate.multirule_evaluation_mode` to
`incumbent_first`. Keep the five-rule registry, equal opening-pair count per
rule, paired colors, 50 initial pairs, 25-pair increments, 200-pair maximum,
95% family confidence, 256 search simulations, and 0.05 point-score peak
regression tolerance from the accepted five-rule baseline. The macro check
uses incumbent games first. A macro rejection or exhausted inconclusive gate
records `peak_evidence_status=not_evaluated`, with null peak statistics and
peak games. If the macro lower confidence bound exceeds 0.5, evaluate every
rule against its retained peak on those same openings before accepting. An
accepted candidate always has `peak_evidence_status=complete` and all five
peak result sets. `complete` remains available for later protocols requiring
peak evidence even on rejected candidates.

On an uncontended two-card host, start topology calibration with
`runtime.evaluation_devices=["cuda:0", "cuda:1"]` and
`runtime.evaluation_replicas_per_device=4`. These runtime fields are recorded
but excluded from semantic lineage. Use
`tools/benchmark_v3_multirule_gate_topology.py` on an existing committed
gate's fixed opening manifest to compare four workers on GPU0 with four per
card. The command prints a plan until `--execute` is supplied. Record its
JSON report outside source-controlled result paths, including model and
opening hashes, per-game results, wall times, and worker metrics. Use two
cards only when both are idle, game results match, and dual-card wall time is
lower without contention; otherwise use the one-card topology. Do not apply
an unmeasured speed ratio from another model or host.

The gate result is published with the generation commit. If a process stops
before that commit, the existing V3 draft-recovery procedure still applies;
the new mode does not persist partial gate matches for pair-level resumption.

## 2026-09-29 two-card calibration

On `connect4_gpu_2608` (two RTX 3080 Ti, cgroup 40 CPUs and 60 GiB), a fixed
`raw3d_to2d_b8` candidate-versus-incumbent comparison used 20 paired Classic
openings, 40 games, 256 simulations, and cpuct 1.5. Four replicas on GPU0
finished in 253.10 s; four replicas on each card finished in 140.05 s, a
1.81x wall-time speedup. Every per-game result and opening ID matched. Both
cards were idle before each case; sampled peak GPU memory was 1,530 MiB per
active card. The complete report is local, ignored evidence at
`training/runs/stage2/fla2_r2/gate_topology_stage3_preflight_20260929/report.json`
(SHA-256 `646a714aa0a078372ac4093e8793e3193148f8562446e2499fc074fcfc9b4981`).
This supports the two-card topology for this host and model under idle
conditions; it does not measure a complete five-rule gate or another model.
