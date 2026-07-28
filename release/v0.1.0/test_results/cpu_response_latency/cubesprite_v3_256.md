# Targeted desktop fixed-state latency

- Model: `cubesprite_v3`; MCTS: `256`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **1.904s**.
- Shortcut states: **0**; mean excluding shortcuts: **1.904s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 0.739s | 12 |
| 1 | -1 | no | 1.144s | 16 |
| 2 | 1 | no | 1.435s | 11 |
| 3 | -1 | no | 1.633s | 6 |
| 4 | 1 | no | 1.704s | 7 |
| 5 | -1 | no | 1.846s | 17 |
| 6 | 1 | no | 1.784s | 13 |
| 7 | -1 | no | 2.036s | 13 |
| 8 | 1 | no | 1.980s | 32 |
| 10 | 1 | no | 1.919s | 5 |
| 11 | -1 | no | 1.822s | 5 |
| 12 | 1 | no | 1.647s | 38 |
| 13 | -1 | no | 1.827s | 38 |
| 14 | 1 | no | 2.284s | 57 |
| 15 | -1 | no | 2.226s | 37 |
| 16 | 1 | no | 2.180s | 62 |
| 17 | -1 | no | 2.215s | 62 |
| 18 | 1 | no | 2.133s | 87 |
| 19 | -1 | no | 2.397s | 87 |
| 20 | 1 | no | 2.319s | 57 |
| 22 | 1 | no | 2.306s | 87 |
| 24 | 1 | no | 2.302s | 87 |
