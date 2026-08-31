# Connect4 3D Layer0 Solver

这是仓库内独立子目录中的 5×5 Layer0 求解器原型。它只观察 25 个底层列位，
不加载训练模型，也不改动现有 3D 训练、Arena 或 UI 实现。

## 规则边界

- 棋盘按从上到下、从左到右编号 `1..25`。
- 红方先手，双方轮流在空的 Layer0 格落子。
- 平面内横、竖、两类斜线任意连续四子获胜，共 28 条获胜线。
- 原 3D 游戏在已有底层棋子的列继续落子时，棋子进入更高层。对本 solver 来说这
  是一次 Layer0 棋面未变化的 `pass`，接口和验收 UI 都支持记录这种不可见回合；
  solver 本身不会主动选择更高层落子。
- 棋盘填满时平局；若双方所有潜在连四线都已被对方棋子阻断，也可以提前判平。
- 状态缓存使用正方形的 4 次旋转和 4 次镜像（D4）归一化。等价最优决策会全部
  返回，因此某些局面的最优落子不唯一。

## 运行

仅求解器和测试使用 Python 标准库。界面需要 Python 3.10+；Python 3.10–3.13
安装 pygame，Python 3.14 因官方 wheel 可用性自动安装 API 兼容的 pygame-ce：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_ui.py
```

命令行分析一个局面：

```powershell
python -m layer0_solver.cli analyze --moves "13,9,8,18,14,12,20,2,7,pass,19,1"
python -m layer0_solver.cli replay-known-path
python -m layer0_solver.cli --games 8 --seed 20260827 self-play
python -m layer0_solver.cli solve-root --root-timeout 120
```

`analyze` 的序列包含双方行为；`pass` 表示一次发生在第二层或更高层、Layer0
不可见的落子。输出中的 `win/draw/loss` 始终从当前行棋方视角解释。

## 验收界面

- 默认模式是“人类（红）对 Solver（蓝）”；按钮可交换先后手。
- 点击空格落子；`P` 记录一次不可见的高层落子；`R` 重开；`H` 显示/隐藏 solver
  给出的全部最优格。
- 右侧显示当前局面理论值、最优格、搜索缓存、最近一步和旋转等价提示。
- 所有标成“最优格/等价值”的格都逐一由原生强解确认；不会退化为模板或浅层估值。
- 原生进程常驻并复用换位表。首次空棋盘求解在开发机约 49 秒，界面在后台计算，
  之后同一盘通常明显更快。
- 无窗口渲染检查可运行 `python -B -m tools.render_ui_smoke`；验证截图写入
  `evidence/ui_smoke.png`。

## 验证口径

`tests/test_solver.py` 覆盖：28 条赢线、D4 对称、提前判平、立即获胜/强制防守、
不可见回合、题目给定的 V3 复现路径，以及旋转后的等价最优决策。

pygame、`analyze` 和 `self-play` 都使用 C++ 原生 alpha-beta 强解。运行
`build_native.ps1` 可重新编译。原生进程以 server 模式常驻，避免每一手丢弃根局面
搜索结果。换位表默认在一次查询开始前超过 3200 万条目时清空，以限制连续多局增长；
可通过 `LAYER0_MAX_CACHE_ENTRIES` 调高（最低接受 2600 万，根证明本身约需 2532 万）。

旧版曾把安全填色模板的候选格误称为等价值。例如红方首着 `13` 后，蓝方真正的
平局应手只有 `7, 9, 17, 19`；旧版可能随机走 `18`，其精确值实际为负。当前版本
删除了该验收模板，回归测试明确断言 `18` 不在等价值集合中。

完整根局面的原生证明入口是 `python -m layer0_solver.cli solve-root`。2026-08-24
开发机实测返回平局、25 个首着同值，约 4,174 万节点、48.8 秒、峰值私有内存约
0.95 GiB；机器相关数据与命令保存在 `evidence/2026-08-24_verification.json`。
慢回归会执行根证明并沿随机精确等值分支走完整盘：

```powershell
$env:LAYER0_RUN_SLOW = "1"
python -B -m unittest tests.test_native_root_slow -v
```

这一结果也与 Uiterwijk 2019 年论文 *Solving Strong and Weak 4-in-a-Row* 对强型
5×5 游戏的平局结论一致：
https://ieee-cog.org/2019/papers/paper_115.pdf

2026-08-27 修复后，`--games 8 --seed 20260827 self-play` 覆盖 187 个逐局面强解
步骤，8 局均填满 25 格平局。证据记录在
`evidence/2026-08-27_equivalence_fix.json`。这轮不需要 AlphaZero 辅助判断；原生
穷尽搜索给出的胜/和/负标签比随机模型对局更适合作为等价值回归真值。
