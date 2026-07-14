# Connect4 3D CubeSprite

CubeSprite `0.1.0` 是固定 `6 × 5 × 5、连四` 规则的离线 Windows 桌面游戏。发布包包含 Tauri 2 应用、React 界面、冻结的 Python sidecar、ONNX Runtime 和可用模型；最终用户无需安装 Python、Conda、Node.js 或 Rust。

## 架构边界

- `src/`：React + TypeScript，只负责界面、动画、语言切换和请求编排。
- `backend/cubesprite_backend/`：唯一的棋局状态源，负责规则、胜负线、MCTS、提示与胜率。
- `src-tauri/`：Windows 容器和 sidecar 生命周期；窗口关闭时终止子进程。
- 前后端只通过 stdin/stdout 的 JSON Lines 通信，不启动 HTTP、WebSocket 或本地监听端口。
- `resources/model_registry.json` 是唯一模型注册表，不扫描用户目录或动态加载 Python 模块。

## 模型

| 模型 | 状态 | 推理适配 |
|---|---|---|
| CubeSprite V3 | 占位 | 当前无权重，界面显示但禁用 |
| CubeSprite V3 mini | 占位 | 当前无权重，界面显示但禁用 |
| v2.2 Balance | 可用 | 原生 `2 × 6 × 5 × 5` 输入、150 动作 |
| v2.1 High | 可用 | 六层棋盘补零为旧版八层单通道输入，200 动作裁为前 150 动作后重新归一化 |

ONNX 发布资源使用 Git LFS。首次检出后运行 `git lfs pull`。如需从本机参考 checkpoint 重新生成，脚本只读 `tmp_built_app/`，并在临时文件通过 ONNX checker、ONNX Runtime 和 PyTorch 数值比对后原子替换目标文件。

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

- Git 跟踪：源代码、锁文件、图标、manifest、Git LFS ONNX 模型。
- Git 忽略：`node_modules/`、前端 `dist/`、Rust `target/`、PyInstaller 临时目录、冻结 sidecar exe 和安装包。
- `tmp_built_app/` 是本地只读参考材料，不进入提交。
