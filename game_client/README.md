# Connect4 3D v2.2 Game Client

Run from the repository root:

```powershell
conda activate pytorch
python -m pip install -r game_client\requirements.txt
python -m game_client --model-root D:\path\to\models
```

`--model-root` may be repeated. When it is omitted, the client checks this
workspace and a sibling `Connect4_3D_AI_v2.2` training workspace for `save_model`
and `distillation/checkpoints` directories. Model files are read only.

The first release supports v2.2 6x5x5 checkpoints, human first/second choice,
and a six-layer 2D board. Legacy v2.1 eight-layer checkpoints are rejected.
