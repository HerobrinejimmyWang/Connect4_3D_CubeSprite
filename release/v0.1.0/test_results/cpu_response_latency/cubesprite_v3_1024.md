# Targeted desktop fixed-state latency

- Model: `cubesprite_v3`; MCTS: `1024`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **9.279s**.
- Shortcut states: **0**; mean excluding shortcuts: **9.279s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 3.911s | 12 |
| 1 | -1 | no | 5.518s | 8 |
| 2 | 1 | no | 9.443s | 17 |
| 3 | -1 | no | 8.876s | 6 |
| 4 | 1 | no | 8.538s | 7 |
| 5 | -1 | no | 8.210s | 17 |
| 6 | 1 | no | 8.415s | 11 |
| 7 | -1 | no | 10.290s | 13 |
| 8 | 1 | no | 9.960s | 32 |
| 10 | 1 | no | 7.961s | 32 |
| 11 | -1 | no | 7.266s | 5 |
| 12 | 1 | no | 7.926s | 36 |
| 13 | -1 | no | 8.136s | 38 |
| 14 | 1 | no | 8.735s | 57 |
| 15 | -1 | no | 10.488s | 37 |
| 16 | 1 | no | 10.412s | 62 |
| 17 | -1 | no | 10.997s | 62 |
| 18 | 1 | no | 10.036s | 87 |
| 19 | -1 | no | 12.050s | 87 |
| 20 | 1 | no | 13.082s | 57 |
| 22 | 1 | no | 10.944s | 9 |
| 24 | 1 | no | 12.949s | 87 |
