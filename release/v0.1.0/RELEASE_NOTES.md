# Connect4 3D CubeSprite v0.1.0

The first public CubeSprite release packages the verified offline game client,
four local AI models, the Windows desktop application, and an experimental
Android Beta.

## Highlights

- Play locally as Player vs Player or Player vs AI on a `6 × 5 × 5` board.
- Choose independently between CubeSprite V3, CubeSprite V3 mini,
  v2.2 Balance, and v2.1 High for combat, hints, and win-rate analysis.
- Configure MCTS simulations and temperature for each AI role.
- Enable or disable forced one-move win and mandatory-block tactics.
- Use optional tactical assistance immediately or after a five-second delay.
- Inspect the board through the original six-layer desktop 2D layout or the
  interactive 3D observation view.
- Save, import, export, and inspect replays; request a read-only replay Hint or
  continue a new game from a selected position.
- Run model inference and MCTS fully offline.

## Downloads

### Windows

`Connect4 3D CubeSprite_0.1.0_x64-setup.exe`

- Windows x64 NSIS installer.
- Includes the local Python sidecar and all four ONNX models.
- The installer is not code-signed, so Windows SmartScreen may display an
  unknown-publisher warning.

### Android Beta

`Connect4_3D_CubeSprite_0.1.0-beta-android-arm64.apk`

- Experimental ARM64-only Android build.
- Signed with an Android Debug certificate and intended only for Beta testing.
- Replay storage and replay analysis are not included in this mobile Beta.

### Public model weights

`Connect4_3D_CubeSprite_models_v0.1.0.zip`

Contains exactly four weights-only PyTorch state dictionaries:

- `cubesprite_v3_iter240.pth`
- `cubesprite_v3_mini_iter260.pth`
- `v2.2_balance.pth`
- `v2.1_high.pth`

Optimizer state, trainer state, replay buffers, evaluation state, and other
checkpoint contents are not included.

## SHA-256

| Artifact | SHA-256 |
| --- | --- |
| Windows installer | `16de51d2813c7efdc94fc15dca644a6e8375cb06dd71c79d3494bb2245c77076` |
| Android ARM64 Beta | `708b89c85de090ea8701db62f371974f83be57194087f5a218ed650f508bb038` |
| Public model weights | `873430fc0d9df6a90bb3ccc7a77145aa2b9fc6df476d85b1add19aa419c4ddea` |
| Showcase video | `acdd2494925d11b74c851a866e05b0c9a73219294cf1975d8774ea5288c42e48` |

## Notes and boundaries

- Model strength is not a calibrated human rating. It varies with the
  position, MCTS budget, temperature, hardware, and runtime configuration.
- The training narrative is based partly on the author's retained records and
  recollection; uncertain details are marked accordingly.
- The historical training pipeline remains on the `legacy-training` branch and
  is not presented as fully end-to-end validated.
- Source code is released under MIT. Third-party dependencies and runtimes
  retain their respective upstream licenses.

---

## 中文说明

这是 CubeSprite 的首个公开版本，包含离线游戏客户端、四个本地 AI 模型、
Windows 桌面应用和实验性的 Android Beta。

Windows 安装包为未签名的 x64 NSIS 安装包，可能触发 SmartScreen
“未知发布者”提示。Android APK 仅支持 ARM64，使用 Android Debug 证书签名，
应明确作为 Beta 测试版本分发。公开模型压缩包只包含四个 `.pth` 权重文件，
不包含优化器、训练器、数据池或其它 checkpoint 状态。

模型能力并非经过标定的人类等级；实际表现会受到局面、MCTS 搜索量、温度、
硬件和运行配置影响。模型训练历史中的部分细节来自作者保留的记录与回忆，
不确定之处已在文档中说明。
