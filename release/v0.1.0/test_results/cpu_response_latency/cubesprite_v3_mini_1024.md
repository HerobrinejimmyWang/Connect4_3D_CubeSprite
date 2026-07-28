# Targeted desktop fixed-state latency

- Model: `cubesprite_v3_mini`; MCTS: `1024`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **7.863s**.
- Shortcut states: **0**; mean excluding shortcuts: **7.863s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 4.884s | 12 |
| 1 | -1 | no | 4.651s | 6 |
| 2 | 1 | no | 7.685s | 11 |
| 3 | -1 | no | 9.152s | 6 |
| 4 | 1 | no | 8.944s | 7 |
| 5 | -1 | no | 9.052s | 17 |
| 6 | 1 | no | 9.459s | 11 |
| 7 | -1 | no | 9.305s | 13 |
| 8 | 1 | no | 9.089s | 3 |
| 10 | 1 | no | 7.686s | 36 |
| 11 | -1 | no | 7.798s | 5 |
| 12 | 1 | no | 7.412s | 36 |
| 13 | -1 | no | 6.937s | 36 |
| 14 | 1 | no | 6.491s | 37 |
| 15 | -1 | no | 8.785s | 37 |
| 16 | 1 | no | 8.837s | 57 |
| 17 | -1 | no | 7.432s | 62 |
| 18 | 1 | no | 7.138s | 33 |
| 19 | -1 | no | 6.674s | 57 |
| 20 | 1 | no | 9.528s | 31 |
| 22 | 1 | no | 8.452s | 57 |
| 24 | 1 | no | 7.594s | 56 |
