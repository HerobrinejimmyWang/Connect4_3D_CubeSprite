# Connect4 3D CubeSprite

CubeSprite `0.2.0-alpha.1` 是 `6 × 5 × 5、连四` 的离线 Windows 桌面游戏，支持 Classic 和 Stage 3 的四种先手限制规则。发布包包含 Tauri 2 应用、React 界面、冻结的 Python sidecar、ONNX Runtime 和可用模型；最终用户无需安装 Python、Conda、Node.js 或 Rust。版本保留在发布元数据和后端，不在 App 页面显示。

## 架构边界

- `src/`：React + TypeScript，只负责界面、动画、语言切换和请求编排。
- `backend/cubesprite_backend/`：唯一的棋局状态源，负责规则、胜负线、MCTS、提示与胜率。
- `src-tauri/`：Windows 容器和 sidecar 生命周期；窗口关闭时终止子进程。
- 前后端只通过 stdin/stdout 的 JSON Lines 通信，不启动 HTTP、WebSocket 或本地监听端口。
- `resources/model_registry.json` 是唯一模型注册表，不扫描用户目录或动态加载 Python 模块。

## 模型

| 模型 | 状态 | 推理适配 |
|---|---|---|
| CubeSprite V4 Flash (Preview1) | 首选、默认 | V3 原生 Stage 3 终端快照；显式棋盘、绝对行棋身份和 32 维规则输入；支持全部五种规则 |
| CubeSprite V3 | 旗舰版 | `iter_0240` 重力感知残差网络，原生 `2 × 6 × 5 × 5` 输入、150 动作 |
| CubeSprite V3 mini | mini 版 | `iter_0260` 轻量重力感知残差网络，原生 `2 × 6 × 5 × 5` 输入、150 动作 |
| v2.2 Balance | 可用 | 原生 `2 × 6 × 5 × 5` 输入、150 动作 |
| V3 B6C128 | 可用 | Stage 1 G150 锚点，6 个残差块、128 通道；适配到产品单输入 ONNX 契约 |
| V3 B8C192 | 可用 | Stage 1 G268 已接受锚点，8 个残差块、192 通道；适配到产品单输入 ONNX 契约 |
| V3 B10C256 | 可用 | Stage 1 G258 最终已接受锚点，10 个残差块、256 通道；适配到产品单输入 ONNX 契约 |

ONNX 发布资源使用 Git LFS。首次检出后运行 `git lfs pull`。注册表记录每个
模型资源的 SHA-256，sidecar 在首次加载前再次核验。如需从本机参考
checkpoint 重新生成，脚本只读 `tmp_built_app/`，并在临时文件通过 ONNX
checker、ONNX Runtime 和 PyTorch 数值比对后原子替换目标文件。

V4 Preview1 单独通过 `python desktop_app/scripts/export_v4_flash.py` 从仓库根目录
`terminal_snapshot.pt` 导出。脚本核验源 SHA-256，严格加载其 V3 架构，并验证
五种规则 × 双方行棋身份的 ONNX/PyTorch 一致性；不转换或续训 Legacy 权重。
默认仍为 256 次搜索、temperature 0.4。主菜单智能度滑条的七档明确应用
16/32/64/128/256/512/1024 次搜索和 temperature 0.5，Advance 打开详细设置。

菜单中先选规则再选旧模型会回到 Classic；先选旧模型再选新规则会切到 V4。
已开始的新规则棋局中，切换到不支持规则的模型会被拒绝，保留棋局并显示红色提示。
游戏内 Instructions 和 AI Settings 的 Back 返回当前棋局。

## 对局回放

对局界面可以把当前步数之前的完整棋局保存到本机。主菜单“回放模式”支持
打开、删除、导入和导出回放；播放器支持逐步浏览、自动播放、2D/3D
观察、从任意未终局位置继续对战，以及按当前 AI 设置计算完整胜率曲线。

回放与胜率分析是两个独立文件：可分享的回放只包含规则版本、落子序列和
终局状态；分析旁车文件记录模型文件哈希、MCTS 配置、执行时间和逐步胜率。
详细格式见 [REPLAY_PROTOCOL.md](REPLAY_PROTOCOL.md)。

保存和导入使用回放协议 v2，包含规则身份、参与者来源及 placement/forced-pass
回合序列。v1 不再支持。训练侧 v2 样例可以逐步播放，并从未终局位置继续。

## Conda 构建环境

开发和发布默认使用 Conda；推荐激活现有 `pytorch` 环境：

```powershell
cd desktop_app
conda activate pytorch
python -m pip install -r backend\requirements-build.txt onnx
python scripts\export_models.py --models all
powershell -ExecutionPolicy Bypass -File scripts\build_sidecar.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

模型导出阶段需要 PyTorch 和 `onnx`；冻结后的 sidecar 不导入或携带 PyTorch/Pygame，只包含 Python、NumPy 与 ONNX Runtime。

## 前端与 Windows 打包

需要 Node.js、pnpm、Rust stable-msvc、Visual C++ Build Tools 和 Windows SDK。应在已初始化 MSVC 的 Developer PowerShell 中执行：

```powershell
cd desktop_app
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm build
pnpm tauri build
```

Tauri 会构建 current-user NSIS 安装包，并内含离线 WebView2 安装资源。生成位置为：

```text
src-tauri\target\release\bundle\nsis\
```

## 后端与协议测试

在仓库根目录运行：

```powershell
conda activate pytorch
python -m unittest discover -s desktop_app\backend\tests -v
python -m unittest discover -s desktop_app\tests_backend -v
python -m compileall connect4_core training arena distillation train_features test game_client desktop_app\backend
```

请求示例：

```json
{"v":1,"type":"request","id":"r1","command":"game.move","params":{"session_id":"...","expected_revision":4,"layer":0,"row":2,"col":2}}
```

响应包含同一请求 ID；所有棋局修改和分析都携带 `session_id + revision`，从而丢弃 Undo、Restart、Exit 或设置变化后的过期 AI 结果。stdout 只输出 JSONL 协议，诊断信息写入 stderr。

## 生成物策略

Windows 实机端到端测试脚本为 `scripts/e2e_windows.py`，连接测试进程专用的
WebView2 调试端口并使用真实安装后的 Tauri/JSONL/ONNX 链路，无模拟后端。
例：`python desktop_app/scripts/e2e_windows.py --exe <installed-cubesprite.exe> --sample <training-v2.json> --output <evidence-directory>`。
仅测试环境需要 Playwright。脚本为测试进程设置独立的 WebView2 数据目录，并将
`CUBESPRITE_DATA_DIR` 设置为证据目录下的绝对路径 `app-data/`，隔离真实后端的回放库。
该环境变量只接受绝对路径；未设置时，正常启动仍使用 Windows Known Folders。
成功测试会清理本次新建的测试回放并保留证据副本。
测试覆盖五种桌面尺寸下的菜单/选择窗口、设置页和说明页边界，以及最小窗口的
游戏/回放/续局弹窗、滑条键盘七档、中英文切换和真实对局回放流程。

- Git 跟踪：源代码、锁文件、图标、manifest、Git LFS ONNX 模型。
- Git 忽略：`node_modules/`、前端 `dist/`、Rust `target/`、PyInstaller 临时目录、冻结 sidecar exe 和安装包。
- `tmp_built_app/` 是本地只读参考材料，不进入提交。
