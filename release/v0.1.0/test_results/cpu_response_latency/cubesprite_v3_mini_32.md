# Targeted desktop fixed-state latency

- Model: `cubesprite_v3_mini`; MCTS: `32`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **0.125s**.
- Shortcut states: **0**; mean excluding shortcuts: **0.125s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 0.045s | 12 |
| 1 | -1 | no | 0.070s | 8 |
| 2 | 1 | no | 0.089s | 11 |
| 3 | -1 | no | 0.081s | 6 |
| 4 | 1 | no | 0.098s | 7 |
| 5 | -1 | no | 0.088s | 22 |
| 6 | 1 | no | 0.111s | 11 |
| 7 | -1 | no | 0.107s | 13 |
| 8 | 1 | no | 0.120s | 3 |
| 10 | 1 | no | 0.122s | 36 |
| 11 | -1 | no | 0.122s | 5 |
| 12 | 1 | no | 0.142s | 36 |
| 13 | -1 | no | 0.131s | 36 |
| 14 | 1 | no | 0.172s | 57 |
| 15 | -1 | no | 0.152s | 37 |
| 16 | 1 | no | 0.125s | 38 |
| 17 | -1 | no | 0.156s | 62 |
| 18 | 1 | no | 0.132s | 28 |
| 19 | -1 | no | 0.137s | 57 |
| 20 | 1 | no | 0.165s | 31 |
| 22 | 1 | no | 0.190s | 57 |
| 24 | 1 | no | 0.194s | 56 |
