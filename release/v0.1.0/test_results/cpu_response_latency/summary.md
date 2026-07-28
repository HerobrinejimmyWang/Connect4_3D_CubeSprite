# Desktop CPU response-latency matrix retest

- States: replay states 0–24, excluding shortcut states 9, 21, and 23.
- Temperature: 0.4. Each response is followed by 10 seconds of idle time; groups are separated by 60 seconds.
- A blank cell means at least one measured response exceeded 60 seconds; that group was stopped.

| Model \ MCTS_sims | 32 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| CubeSprite V3 | 0.198s | 0.884s | 1.904s | 4.728s | 9.279s |
| CubeSprite V3 mini | 0.125s | 0.615s | 1.384s | 3.313s | 7.863s |
| V2.2 Balance | 1.040s | 4.044s | 8.035s | 16.172s | 33.245s |
| V2.1 High | 2.858s | 10.971s | 22.349s | 44.640s |  |

## Raw result files

- `cubesprite_v3@128`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\targeted_v3_128_excluding_shortcuts_idle10s.json`
- `cubesprite_v3@512`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\targeted_v3_512_idle10s.json`
- `cubesprite_v3@32`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_32.json`
- `cubesprite_v3@256`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_256.json`
- `cubesprite_v3@1024`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_1024.json`
- `cubesprite_v3_mini@32`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_mini_32.json`
- `cubesprite_v3_mini@128`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_mini_128.json`
- `cubesprite_v3_mini@256`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_mini_256.json`
- `cubesprite_v3_mini@512`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_mini_512.json`
- `cubesprite_v3_mini@1024`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\cubesprite_v3_mini_1024.json`
- `v2.2_balance@32`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.2_balance_32.json`
- `v2.2_balance@128`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.2_balance_128.json`
- `v2.2_balance@256`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.2_balance_256.json`
- `v2.2_balance@512`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.2_balance_512.json`
- `v2.2_balance@1024`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.2_balance_1024.json`
- `v2.1_high@32`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.1_high_32.json`
- `v2.1_high@128`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.1_high_128.json`
- `v2.1_high@256`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.1_high_256.json`
- `v2.1_high@512`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.1_high_512.json`
- `v2.1_high@1024`: `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\cpu_response_latency\matrix_retest_idle10s\v2.1_high_1024.json`
