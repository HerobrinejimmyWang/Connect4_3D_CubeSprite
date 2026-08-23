# V3 Dual-Track Scaling V1

Status: foundation implemented; formal scheduler integration disabled
Contract: `dual_track_scaling_v1` + `cross_scale_replay_bundle_v1`

## 1. Why there are two tracks

The two tracks answer different questions and never share a strength curve.

| Track | Question | Initialization | Imported replay | Primary output |
|---|---|---|---|---|
| Research | Does capacity independently improve stable self-play and the attainable ceiling? | Random | None | Clean scaling curve |
| Production | How quickly can an auditable stronger model be produced? | Random | Frozen donor bundle | Time/compute to strength |

The shared model/rule/replay/evaluation semantics remain fixed. Historical
Anchored Elo stays descriptive and never promotes a candidate.

## 1.1 Stage 1 execution decision

Stage 1 retains the minimum Research line needed to support causal scaling and
Stage 2 stability claims:

1. B4C64 runs one bounded 60k-position canary for pipeline and health checks.
2. B6C128 runs one independent primary seed to the declared milestones; a
   250k confirmation seed is admitted only after the primary seed is stable.
3. B8C192 runs one independent primary seed after B6 qualifies; its confirmation
   seed is admitted only after direct B8-vs-B6 and stability evidence supports
   continuation.
4. A Larger scale is opened only when B8 still has a supported
   compute-normalized learning/strength slope.

This is a staged Research funnel, not a Production-only shortcut. Production
replay transfer may run beside it to produce a strong model faster, but its
learning curve never substitutes for an independent Research curve.

## 2. Research track: complete flow

Before spending the full three-seed Research budget, run the staged contract in
`configs/stage1_scale_screen_v1.json`: B4C64 is only a one-seed 60k-position
pipeline/health canary; B6C128 is the stability reference; B8C192 is the first
capacity probe. B6/B8 start with one primary seed and add a 250k confirmation
seed only after the declared stability/continuation condition passes. This
screen is a resource-allocation funnel, not the final multi-seed scaling claim.

For every declared scale and seed:

1. Resolve a scale-specific V3 config and record its semantic hash.
2. Start a new run with random weights and random-bootstrap self-play.
3. Admit only candidates promoted by the ordinary same-lineage paired gate.
4. Allow only the committed champion to generate subsequent replay.
5. Retain raw history and train from the normal growing active window.
6. Freeze accepted `early`, `middle`, and `final` checkpoints by declared
   training-position boundaries, not by a favorable evaluation result.
7. At each milestone run `primary_256` Anchored Elo; at final also run
   `final_512`.
8. Report matched train positions, matched total compute including self-play
   search, and terminal strength after the predeclared plateau/stop rule.
9. Aggregate seeds only after preserving every seed-level learning curve,
   FIRST/SECOND split, gate history, failures, and hardware evidence.

No replay, checkpoint, optimizer, token bucket, or accepted-model state may be
copied from another size. Reusing opening manifests and external evaluation
anchors is required and is not lineage inheritance.

### Base-model selection

The V3 Base is not preassigned to B6C128 or to the largest configured network.
After the staged Research line, select the last supported scale that satisfies
all of the following:

- stable closed-loop self-play without a repeated behavioral pause;
- supported direct and Anchored-Elo strength at the frozen search profile;
- positive or still-useful learning slope at the final milestone;
- acceptable compute-normalized gain and target-hardware efficiency;
- policy-target and archived-data quality sufficient for Stage 2;
- at least the declared primary/confirmation evidence for the claimed result.

This may select B6, B8, or a later Larger model. If the next scale is unstable,
inconclusive after its one extension, or too inefficient, the previous stable
scale remains the Base. A replay-transfer model may be the practical high-
performance Base candidate, but its Production lineage must remain explicit;
the matching independent Research checkpoint remains the scaling reference.

Freeze the selected Base checkpoint and the versioned Reference Training
Distribution separately. Stage 2 must not silently replace either artifact
when a later Production checkpoint becomes stronger.

## 3. Production track: complete flow

For every adjacent donor-to-target transition:

1. Select accepted donor checkpoints from `early`, `middle`, `late`, and
   `strong`. Selection boundaries and the provisional 10/20/30/40 sampling
   weights are declared before target training.
2. Create a source catalog associating every replay shard with the exact
   accepted V3 artifact that produced it.
3. Build `cross_scale_replay_bundle_v1`. Construction rejects random producers,
   mixed source runs/configs/rules, duplicate shards, producer/artifact mismatch,
   incomplete Replay V2 triplets, or artifact/config mismatch.
4. Copy only replay triplets into the portable bundle. Model weights are hashed
   for producer attestation but never copied or initialized into the target.
5. Create the target lineage with random weights. Before qualification it runs
   an offline learner phase only; it is not an accepted champion and cannot
   generate replay.
6. Spend at most the declared donor token budget (V1 provisional value: four
   consumed training positions per donor position), checkpointing learner,
   optimizer, scaler, cursor, and donor-only token ledger.
7. Run common V3-MCTS qualification against the bundle's frozen strong donor:
   50 paired openings at 256 simulations, extending to 100/150/200 only when
   inconclusive. Accept only when the paired score interval lower bound exceeds
   0.5 and both role scores meet the 0.45 floor.
8. Passing evidence still does not mutate training state. The future formal
   coordinator must verify it and atomically commit the target as its lineage's
   first champion. Failed candidates may consume additional unused offline
   donor tokens and retry on disjoint evidence; they may not self-play.
9. After the first champion commit, self-play starts from that champion. Donor
   and own replay remain separate pools with separate cursors and receipts.
10. Sample donor/own replay deterministically by own generated positions:
    50/50, 25/75 at 250k, 10/90 at 1M, and 5/95 at 4M. These absolute boundaries
    are provisional target-hardware settings and must be frozen before the first
    Production run.
11. Record donor consumed, own consumed, own generated, sample cursor, bundle
    hash, and active schedule stage in every checkpoint. AMP-skipped optimizer
    steps advance none of these counters.
12. Evaluate early/middle/final exactly as in Research, while separately
    reporting time and total inherited compute required to match and exceed the
    donor.

## 4. Artifact and state graph

```text
accepted donor checkpoints
        + immutable Replay V2 shards
        + accepted-artifact attestations
                     |
                     v
       cross_scale_replay_bundle_v1
                     |
                     v
 random target -> offline bootstrap checkpoint
                     |
                     v
 paired donor qualification evidence
                     |
          formal coordinator commit (not implemented)
                     |
                     v
 first target champion -> target self-play -> own replay
                     |                         |
                     +---- deterministic -----+
                           donor/own sampler
```

Required immutable identities are the experiment hash, target V3 config hash,
bundle content hash, donor and candidate artifact hashes, opening-manifest hash,
qualification content hash, and subsequent generation commits.

## 5. Failure and recovery rules

- A missing/corrupt bundle triplet blocks offline bootstrap.
- An unknown producer, random producer, or producer/artifact mismatch requires
  rebuilding the source catalog; it is never waived.
- An inconclusive qualification extends evidence; it neither accepts nor
  rejects implicitly.
- A rejected target remains a non-producing candidate.
- A crash during offline training resumes from the exact donor cursor and token
  ledger. A crash during online training also restores own generated/consumed
  counters and the frozen transfer schedule.
- A changed bundle or schedule starts a new Production lineage/config hash.
- Donor replay reaching 5% is a regression/diversity sentinel, not permission
  to hide lack of fresh target replay.

## 6. Compute and reporting contract

Research reports raw/self-play/train positions, optimizer updates and skipped
AMP attempts, search simulations and inference positions, GPU/CPU hours, wall
time, estimated cost, ordinary gate history, Anchored Elo milestones,
FIRST/SECOND W/D/L, and learning/validation curves per seed and in aggregate.

Production additionally reports donor positions by stratum/checkpoint, donor
data generation compute, donor/own positions consumed at every milestone,
time/compute to donor parity and first supported superiority, qualification
attempts/pairs, and final strength delta against the same-size Research run.
Headline cost is shown with donor generation fully charged and amortized as an
existing asset.

## 7. Commands available now

```powershell
python tools\run_v3_scaling.py screen-plan
python tools\run_v3_scaling.py plan
python tools\run_v3_scaling.py build-bundle `
  --sources path\to\bundle_sources.json `
  --output training\runs\transfer\starter-to-mini\donor_bundle
python tools\run_v3_scaling.py verify-bundle `
  --bundle training\runs\transfer\starter-to-mini\donor_bundle
python tools\run_v3_scaling.py sample-plan `
  --donor-size 1000000 --own-size 500000 `
  --own-positions-generated 250000 --count 10000
```

`qualify` runs a bounded prefix qualification between two V3 model artifacts.
It writes immutable evidence with `automatic_promotion=false` and
`replay_generation_authorized=false`.

The formal coordinator does not yet consume these artifacts. `python -m
training.v3 run` remains plan-only until the existing P7 blockers plus offline
bootstrap, cumulative disjoint qualification extension, dual-pool checkpoint
state, and first-champion commit are connected and fault-tested.

The screen freezes self-play target generation at 128/32 sims so capacity is
not confounded with search. Primary cross-scale and Anchored-Elo diagnostics use
256 sims. At accepted milestones, a separate fixed-position audit compares
repeated 256 against 512 using `tools/run_v3_policy_target_quality.py`. The
budget may be raised only when the 256-to-512 delta exceeds the repeated-256
variability envelope and higher-budget targets improve held-out policy fit or
paired strength; entropy or target sharpness alone is insufficient.

The tool's `generate` command materializes those targets from the immutable
anchored-opening manifest. Primary/reference runs share an audit seed and root
noise, while the repeated-primary run changes only that seed. Every diagnostic
replay records the checkpoint and opening checksums, exact MCTS/lane/noise
contract, and is stored outside formal self-play replay so it cannot become a
training producer by directory discovery.
