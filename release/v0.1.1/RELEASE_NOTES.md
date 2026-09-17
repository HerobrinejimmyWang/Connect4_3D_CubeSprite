# Connect4 3D CubeSprite v0.1.1

- Fixes desktop win-rate prediction placement and prevents the 3D view from rotating below the board.
- Replaces the retired `v2.1_high` model with the completed Stage 1 models: **V3 B6C128**, **V3 B8C192**, and **V3 B10C256**.
- Windows desktop edition only. No `.pth` model files are included in this release.

## CPU response latency

Mean response latency excluding forced tactical shortcuts, in seconds:

| Model \\ MCTS simulations | 32 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| V3 B6C128 | 0.099 | 0.492 | 0.987 | 2.190 | 4.909 |
| V3 B8C192 | 0.140 | 0.612 | 1.228 | 2.741 | 5.979 |
| V3 B10C256 | 0.208 | 0.864 | 1.854 | 3.759 | 8.378 |

Measured on 22 deterministic non-terminal fixed states at temperature 0.4, with independent searches. The v0.1.0 replay is no longer available locally, so these measurements are not position-for-position comparable to its report.
