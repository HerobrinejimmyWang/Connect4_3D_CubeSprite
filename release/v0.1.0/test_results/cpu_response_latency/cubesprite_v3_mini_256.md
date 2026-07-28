# Targeted desktop fixed-state latency

- Model: `cubesprite_v3_mini`; MCTS: `256`; temperature: `0.4`.
- 22 replay states; `10s` idle interval after each non-final response (excluded from latency).
- Mean including shortcuts: **1.384s**.
- Shortcut states: **0**; mean excluding shortcuts: **1.384s** across 22 states.

| State | Side to move | Shortcut | Latency | Returned action |
|---:|---:|:---:|---:|---:|
| 0 | 1 | no | 0.393s | 12 |
| 1 | -1 | no | 0.646s | 8 |
| 2 | 1 | no | 0.984s | 11 |
| 3 | -1 | no | 1.020s | 6 |
| 4 | 1 | no | 1.206s | 7 |
| 5 | -1 | no | 1.164s | 17 |
| 6 | 1 | no | 1.221s | 11 |
| 7 | -1 | no | 1.392s | 13 |
| 8 | 1 | no | 1.417s | 3 |
| 10 | 1 | no | 1.031s | 36 |
| 11 | -1 | no | 1.441s | 5 |
| 12 | 1 | no | 1.489s | 36 |
| 13 | -1 | no | 1.407s | 36 |
| 14 | 1 | no | 1.412s | 31 |
| 15 | -1 | no | 1.412s | 37 |
| 16 | 1 | no | 1.478s | 57 |
| 17 | -1 | no | 1.775s | 62 |
| 18 | 1 | no | 1.597s | 33 |
| 19 | -1 | no | 1.351s | 57 |
| 20 | 1 | no | 2.339s | 31 |
| 22 | 1 | no | 2.224s | 57 |
| 24 | 1 | no | 2.047s | 56 |
