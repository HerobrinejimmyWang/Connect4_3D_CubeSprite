# Targeted desktop fixed-state latency

- Model: `cubesprite_v3_mini`; MCTS: `512`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **3.313s**.
- Shortcut states: **0**; mean excluding shortcuts: **3.313s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 0.905s | 12 |
| 1 | -1 | no | 1.519s | 8 |
| 2 | 1 | no | 2.848s | 11 |
| 3 | -1 | no | 3.106s | 6 |
| 4 | 1 | no | 3.393s | 7 |
| 5 | -1 | no | 3.185s | 17 |
| 6 | 1 | no | 3.848s | 11 |
| 7 | -1 | no | 3.636s | 13 |
| 8 | 1 | no | 3.235s | 3 |
| 10 | 1 | no | 2.911s | 36 |
| 11 | -1 | no | 3.599s | 5 |
| 12 | 1 | no | 3.598s | 36 |
| 13 | -1 | no | 3.210s | 36 |
| 14 | 1 | no | 3.127s | 31 |
| 15 | -1 | no | 3.691s | 37 |
| 16 | 1 | no | 4.118s | 57 |
| 17 | -1 | no | 3.559s | 62 |
| 18 | 1 | no | 3.552s | 33 |
| 19 | -1 | no | 3.006s | 57 |
| 20 | 1 | no | 5.057s | 31 |
| 22 | 1 | no | 4.132s | 57 |
| 24 | 1 | no | 3.652s | 56 |
