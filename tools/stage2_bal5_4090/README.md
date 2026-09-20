# BAL-5 on 1x4090 + 20 vCPU: migration and comparison workflow

Operational scripts for the Part A machine comparison: run the BAL-5 three-phase
screen (selfplay, learner, paired gate) on a rented `1x RTX 4090 + 20 vCPU`
container and decide whether that machine should host the next training stage
instead of the incumbent `2x RTX 3080 Ti` host.

These files were authored and executed on the cloud hosts, verified byte for
byte against the copies that actually ran, and are collected here so the
comparison is reproducible instead of living only in a session transcript.

## Machines

| Role | SSH alias | Hardware | Notes |
|---|---|---|---|
| Incumbent | `connect4_gpu_2608` | 2x RTX 3080 Ti 12 GiB, 128 vCPU visible | source of all BAL-5 R1 evidence |
| Candidate | `exp_260921` | 1x RTX 4090 24 GiB, cgroup `cpu.max=2000000 100000` (20 vCPU), 90 GiB memory | migration target |

Everything below lives under `/root` on the candidate:

- repository: `/root/autodl-tmp/Connect4_3D_game_refactor`, reached through the
  symlink `/root/bal5_4090_migration_20260920/repo` so the absolute `run_dir`
  recorded in `resolved_config.json` keeps resolving
- benchmark output: `/root/bal5_4090_benchmarks/20260921`
- resume control: `/root/bal5_4090_resume`

The repository sits on the data volume (`/dev/md0`, 50 GiB) rather than the
30 GiB container system disk, because the extracted resume snapshot plus the
remaining training budget does not fit on the system disk.

## Order of operations

1. `stage_bal5_4090_materials.sh` — run on the incumbent. Copies the executable
   tree plus the three pinned evidence cases onto the system disk, then writes
   `MIGRATION_INFO.txt`, a full `SHA256SUMS`, and verifies it.
2. Transfer the staged directory with the AutoDL image-migration feature.
3. Restore a real `.git` on the candidate from a `git bundle` of the source
   commit and commit the single-GPU CUDA RNG resume compatibility patch. The
   staged tree deliberately excludes `.git/`, so the migration alone leaves the
   run without code provenance.
4. Move the repository to the data volume and symlink it back (see above).
5. `run_bal5_4090_test_queue.sh` — one launch for all three phases. It also
   runs syntax and focused regression checks first, and writes `QUEUE_SUCCESS`
   or `QUEUE_FAILED` next to the benchmark output.
6. In parallel, transfer the resume snapshot as 18 verified parts; then start
   `watch_bal5_resume_snapshot.sh` and `watch_bal5_test_then_resume.sh`.

Only step 6's second watcher starts training, and only after both the test queue
and the snapshot report success, because the two would otherwise contend for the
single GPU and corrupt the Phase 1 and Phase 2 measurements.

## Phase protocol

| Phase | Tool | Configuration |
|---|---|---|
| selfplay | `tools/benchmark_bal5_machine.py` | adapted production topology, then 128 games per case |
| learner | same tool, same run | 256 optimizer steps, batch 256, FP32 |
| paired gate | `tools/validate_bal5_gate_single_gpu.py` | `evaluation_parallel_games=8`, batch 32, 1 ms timeout, committed role-control reuse |

`prepare_bal5_resume_config.py` refuses to change `mcts_lanes_per_actor`, because
lane count changes search targets. The candidate's higher-throughput 6-lane
topology therefore cannot be adopted for a resume without paired strength
evidence on a stable accepted champion.

## Launching without a persistent session

The first attempt lost its watchers when the driving agent session ended. The
working pattern is:

```bash
setsid nohup bash /root/bal5_watch_snapshot.sh > /root/bal5_watch_snapshot.log 2>&1 < /dev/null &
setsid nohup bash /root/bal5_watch_resume.sh   > /root/bal5_watch_resume.log   2>&1 < /dev/null &
```

Verify with `ps -eo pid,etime,args | grep bal5_` and by reading the status files
under `/root/bal5_4090_resume/`.
