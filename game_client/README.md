# Connect4 3D Game Client

Run from the repository root:

```powershell
conda activate pytorch
python -m pip install -r game_client\requirements.txt
python -m game_client --model-root D:\path\to\models
```

`--model-root` may be repeated. When it is omitted, the client checks this
workspace and a sibling `Connect4_3D_AI_v2.2` training workspace for `save_model`
and `distillation/checkpoints` directories, plus every V3 accepted-champion
directory under `training/runs/local_archive_validation/*/materialized/accepted`
(B4/B6/B8 scales from all synced runs). Model files are read only.

Supported checkpoints:

- v2.2 6x5x5 checkpoints (modern and gravity networks), human first/second
  choice, six-layer 2D board.
- V3 `connect4-v3-model` artifacts (`format == "connect4-v3-model"`) of any
  scale (B4/B6/B8). V3 networks are adapted through
  `arena.agent.V3ModelPredictor`: the absolute FIRST/SECOND role is inferred
  from canonical-board stone parity and the 25-column policy is converted to
  the legacy 150-action space, so the existing MCTS/UI stack is reused.
- Legacy v2.1 eight-layer checkpoints are rejected.
