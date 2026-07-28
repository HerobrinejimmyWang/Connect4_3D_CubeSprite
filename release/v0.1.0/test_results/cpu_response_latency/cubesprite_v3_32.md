# Targeted desktop fixed-state latency

- Model: `cubesprite_v3`; MCTS: `32`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **0.198s**.
- Shortcut states: **0**; mean excluding shortcuts: **0.198s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 0.108s | 12 |
| 1 | -1 | no | 0.151s | 8 |
| 2 | 1 | no | 0.160s | 11 |
| 3 | -1 | no | 0.180s | 6 |
| 4 | 1 | no | 0.172s | 17 |
| 5 | -1 | no | 0.200s | 17 |
| 6 | 1 | no | 0.158s | 11 |
| 7 | -1 | no | 0.163s | 13 |
| 8 | 1 | no | 0.198s | 32 |
| 10 | 1 | no | 0.212s | 5 |
| 11 | -1 | no | 0.187s | 5 |
| 12 | 1 | no | 0.207s | 21 |
| 13 | -1 | no | 0.190s | 57 |
| 14 | 1 | no | 0.209s | 57 |
| 15 | -1 | no | 0.213s | 37 |
| 16 | 1 | no | 0.222s | 38 |
| 17 | -1 | no | 0.216s | 62 |
| 18 | 1 | no | 0.236s | 33 |
| 19 | -1 | no | 0.222s | 87 |
| 20 | 1 | no | 0.211s | 57 |
| 22 | 1 | no | 0.259s | 9 |
| 24 | 1 | no | 0.280s | 9 |
