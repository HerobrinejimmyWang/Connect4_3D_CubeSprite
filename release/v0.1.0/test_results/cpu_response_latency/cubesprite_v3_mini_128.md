# Targeted desktop fixed-state latency

- Model: `cubesprite_v3_mini`; MCTS: `128`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **0.615s**.
- Shortcut states: **0**; mean excluding shortcuts: **0.615s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 0.184s | 12 |
| 1 | -1 | no | 0.325s | 8 |
| 2 | 1 | no | 0.379s | 11 |
| 3 | -1 | no | 0.456s | 6 |
| 4 | 1 | no | 0.450s | 7 |
| 5 | -1 | no | 0.435s | 17 |
| 6 | 1 | no | 0.633s | 11 |
| 7 | -1 | no | 0.613s | 13 |
| 8 | 1 | no | 0.644s | 3 |
| 10 | 1 | no | 0.585s | 36 |
| 11 | -1 | no | 0.603s | 5 |
| 12 | 1 | no | 0.623s | 36 |
| 13 | -1 | no | 0.629s | 36 |
| 14 | 1 | no | 0.676s | 31 |
| 15 | -1 | no | 0.749s | 37 |
| 16 | 1 | no | 0.670s | 57 |
| 17 | -1 | no | 0.759s | 62 |
| 18 | 1 | no | 0.667s | 33 |
| 19 | -1 | no | 0.753s | 57 |
| 20 | 1 | no | 0.858s | 31 |
| 22 | 1 | no | 0.863s | 57 |
| 24 | 1 | no | 0.975s | 56 |
