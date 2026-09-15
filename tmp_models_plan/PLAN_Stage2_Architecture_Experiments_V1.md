# Stage 2 模型架构实验实施计划 V1

> 后继计划：面向 Flash、Balance、Pro 的扩展架构与 scaling 实验见
> `PLAN_Stage2_R3_Deployment_Architecture_Scaling_V1.md`。本文件继续作为已经执行的
> Stage 2A/2B 基础契约；后继计划只做增量扩展，不改写这里冻结的数据与运行语义。

## 1. 目标与边界

Stage 2 固定 V3 的规则、25 维动作空间、role/rule 输入、辅助目标和
Stage 1 P6 loss weights，只研究状态表示、主干结构及其 closed-loop
self-play 稳定性。所有新代码必须位于 `training/v3/`；Legacy 训练代码和
checkpoint 只能作为外部历史对照，不得成为 Stage 2 训练输入。

本阶段不根据 Flash Lite、Balance 或 Flagship 的延迟目标淘汰架构。
参数量、FLOPs、CPU/GPU latency 和吞吐仍然记录，三档模型的尺寸搜索在
架构族冻结后另行进行。

## 2. 架构矩阵

所有架构均把 canonical `[N,6,5,5]` 棋盘映射到 `[N,C,5,5]`，复用同一组
policy、WDL、opponent reply、future occupancy 和 moves-left heads。

| Architecture | Representation / trunk | 作用 |
|---|---|---|
| `gravity_resnet` | 14-plane + 2D ResNet | 不变的 Stage 1 control |
| `column_resnet` | shared vertical-column MLP + 2D ResNet | H：重力柱先验 |
| `multiview_resnet` | XY、XZ、YZ 编码并融合 + 2D ResNet | 三视角 representation |
| `raw3d_resnet` | two-player voxels + 3D ResNet + learned height collapse | I：单路 3D control |
| `plane3d_fusion_resnet` | 14-plane branch + raw 3D branch | 压缩表示和体素几何互补性 |
| `column3d_fusion_resnet` | column branch + raw 3D branch | H+I |
| `column_transformer` | 25 column tokens + 2D position embedding + Transformer | G+H |
| `multiview_transformer` | 三视角融合为 25 tokens + Transformer | G 与 multi-view |
| `multiview_winning_resnet` | XY、XZ、YZ 加六个可获胜斜截面 + 2D ResNet | 显式补齐斜向垂直几何 |
| `multiview_winning_transformer` | 增强 multi-view 融合为 25 tokens + Transformer | G 与完整获胜截面组合 |

六个新增截面由 XY 平面中两组对角方向各自的长度 5 主对角线及两条长度 4
邻线构成，即 `x-y∈{-1,0,1}` 与 `x+y-(BOARD_SIZE-1)∈{-1,0,1}`。每个
XY 路径与完整 Z 轴组成一个竖直截面。长度小于 4 的其余角部斜线不能容纳
Connect-4，明确排除。六路使用共享 section encoder，随后按原 XY 坐标回填并在
重叠位置取平均，再与 XY/XZ/YZ 分支融合；该路径集合在 D4 变换下封闭。

`ModelConfig` 严格记录 architecture、width、depth 以及非默认 encoder、branch、
attention 和 MLP 参数。默认派生值属于版本化的架构契约并写入 architecture
matrix。`gravity_resnet` 的序列化、state dict keys 和 forward 数值行为必须与
Stage 1 完全兼容。

## 3. 数据审计与冻结

Stage 2 使用两个严格分开的 B10 V3 data recipe。`standard` 是原 cold-start
训练线，提供三个公平筛选池；`mixed` 是从标准 B10 checkpoint 启动、fresh replay
并采用混合开局温度与 position-balanced sampling 的工程化训练线。两条 lineage
都只接受 checksum 完整、rule registry 一致的 Replay V2 shards，manifest 必须
记录 `data_recipe_id`、run/config hash 和 producer lineage，禁止把两者静默拼成
一个时间序列。

轨迹指标必须逐代包含：anchored strength、平均局长、短局率、policy entropy 和
accepted cadence。缺少指标时停止，不以纯 position 三等分替代。

审计器对五项指标做 robust normalization，并求带 15% 最小长度约束的三段
连续最小 SSE 分割。标准轨迹冻结为 `standard_early`、`standard_mid`、
`standard_late`；mixed 训练线单独冻结为 `mixed_late`。每个 regime 物化：

- 1,000,000 train positions；
- 50,000 validation positions；
- stable game-ID 95/5 split，同一局不得跨 split；
- 固定 seed `271828` 的 sample-ID hash 排序；
- 源 shard checksum、generation 范围、sample-ID digest 和数据集 checksum。

前三池是 architecture screen 的训练基准。`mixed_late` 首轮只作为第四个
validation domain；只有通过前三池筛选的模型才允许获得 mixed 训练数据，并从
同一 `standard_late` checkpoint 分叉，分别执行“继续 standard”与“切换 mixed”
的等量追加训练。两个分支都只继承同一个父 checkpoint 的模型权重，optimizer、
scheduler、sample cursor 和 augmentation stream 全部重新开始，以隔离 data
recipe 的增量效果。

Standard Late 或 Mixed Late 尚未稳定不阻止 Stage 2A；未来可以新增不可变的
`very_late` 扩展池，但不得覆盖已冻结四池。

## 4. Stage 2A：offline successive halving

先以 B6C128 `gravity_resnet`（1,934,020 parameters）的参数量为锚点，搜索各架构 width。目标为锚点
±5%；无可行值时选择不超过锚点且最接近的配置。参数匹配是首轮唯一 efficiency
约束，延迟不参与晋级。

### Round 1

- 10 architectures × 3 train regimes = 30 models；
- seed `271828`，random initialization；
- 训练 250k consumed positions；
- sample order、D4 augmentation token、optimizer、LR、batch 和所有 loss weights
  完全相同；
- 每个模型评估四池，形成 8×3 train ×4 eval cross-regime matrix；
- `mixed_late` 不参与 Round 1 训练。

晋级规则固定为：保留 baseline、十二格 weighted validation loss 宏平均前四，
以及任一 in-regime 单元前二；最多六个架构。超过六个时按 macro total loss、
policy loss、WDL loss、cross-regime variance 的顺序裁剪。

### Round 2

晋级架构的三个模型训练到 1M positions，并用 seeds `271828` 和 `314159`
复验。第一种子严格从 250k checkpoint 续训，第二种子从头训练。记录：

- policy CE、JSD、top-1 agreement；
- WDL CE、Brier、ECE 和 accuracy；
- 三项 auxiliary losses/accuracies；
- cross-regime generalization、seed variance、学习速率和吞吐；
- 固定 checkpoint hash、固定 openings、交换颜色的 256-sim paired matches。

先运行 50 opening pairs。保留 baseline 与两个无数值失败、离线指标有竞争力且
对局无明显退化的候选，再扩展到 200 pairs。单个 loss 或单次胜率不能独立决定
finalist。

### 容量和数据需求复验

首轮 B6 matching 回答“相同参数预算下谁更好”，不能据此否定参数效率曲线较晚
显现的架构。Round 2 后对 baseline 和最多三个候选执行固定的两阶段矩阵：

| 阶段 | 参数锚点 | 数据预算 | 训练池 | 用途 |
|---|---:|---:|---|---|
| A2-small | B6C128，1,934,020 params | 1M | `standard_late` | 延续公平架构筛选 |
| A2-scale | B8C192，5,630,340 params | 1M、3M | `standard_late` | 区分参数收益和数据收益 |
| A3-recipe | 与 A2 最佳 scale 相同 | +1M/+1M | standard continuation / mixed switch | 配对测 mixed 增益 |
| A4-confirm | B10C256，12,325,956 params | 3M | standard 与晋级 mixed | 只验证 baseline 与最多两个 finalist |

A2 的 `scale × positions` 使用完整 2×2，而不是只训练“更大且更多数据”的对角线，
否则无法区分参数饥饿与数据饥饿。只有 B8 相对 B6 显示明确正向 scaling trend，
且收益超过种子方差和置信区间，才进入昂贵的 B10 confirmation。某架构需要更大
规模不是免责条件，而是一条必须用上述交互矩阵验证的假设。

## 5. 统一实验汇总表

每行的主键为 `architecture × parameter_anchor × train_recipe × train_regime × seed ×
consumed_positions × checkpoint_hash`。表中至少包含：

- 数据与复现：四池 manifest/SHA-256、generation range、sample-ID digest、代码与
  config hash、warm-start hash；
- 监督学习：total/policy/WDL/reply/future-occupancy/moves-left loss，policy JSD 与
  top-1 agreement，WDL Brier/ECE/accuracy，三项 auxiliary 指标；
- 跨域：3×4 cross-regime cells、macro mean、cross-regime variance、
  `mixed_late - standard_late` generalization gap；
- 棋力：anchored Elo、95% CI、Elo slope/每百万 positions、paired win/draw/loss、
  先后手分项与 opening-pair consistency；
- closed-loop：accepted cadence、policy entropy、平均/分位数局长、短局率、
  collapse/stability pause、seed variance；
- 效率：parameter count、MACs/FLOPs、训练 positions/s、wall-clock/GPU-hours、峰值
  显存/内存、CPU/GPU batch-1 latency，并记录 sims、batching 和硬件配额。

Loss 用于解释学习与校准，Elo 用于棋力结论，效率用于后续三档尺寸选择；三者不可
互相替代。所有跨架构总表同时报告绝对值、相对 `gravity_resnet` 的差值和置信区间，
不把单一综合分数作为最终结论。

最终交付固定为三份同源产物：

- `training/runs/stage2/reports/final_experiment_matrix.json`：完整机器可读记录与 hash；
- `training/runs/stage2/reports/final_experiment_matrix.csv`：每个 checkpoint/seed/预算一行；
- `tmp_models_plan/STAGE2_FINAL_RESULTS.md`：面向决策的透视表，分别列出 matched-budget、
  scaling、mixed-promotion、Cold/Warm 1M/5M/10M 和效率 Pareto，不隐藏缺失值或失败线。

三者由同一次冻结汇总生成；Markdown 中每个聚合值必须能回指 JSON 的源行和原始
report/checkpoint hash。实验未完成的单元格标记 `pending`，失败标记 `failed:<reason>`，
不得以空白或只汇总成功 runs 的方式造成幸存者偏差。

## 6. Stage 2B：Cold/Warm closed-loop self-play

baseline 加两个 finalist，各用两个种子、两种初始化，共 12 条 closed-loop 线：

- `cold`：random initialization；
- `warm`：同架构、同尺寸、在 `standard_late` 上离线训练至 1M positions 的
  `connect4-v3-model` artifact，只继承模型权重。

Warm Start 不允许跨架构加载 B10 权重，也不继承 donor optimizer、scheduler、
sample cursor、game ID 或 replay；其首个 accepted champion 是经过 model config、
lineage、train regime 与 SHA-256 严格校验的离线模型，之后 replay 全由该架构自己
产生。因此 Cold/Warm 的差异解释为“晚期数据预训练对 closed-loop 启动的影响”，
而不是 donor self-play 数据混入。

所有线先运行 bounded 1M train-position canary，形成第一张等预算矩阵；通过现有
stability guards 后扩展到 5M，作为主结论。只有 5M 时仍有明确上升趋势，或候选间
置信区间仍重叠、结论无法区分时，才将相应的成对 Cold/Warm 线延伸到 10M。不得
只延长其中更有利的一条线。最终同时报告 1M、5M 以及按规则触发的 10M 截面。

Cold/Warm 对局长信号采用不同解释。Cold Start 前 500k positions 的平均局长下降、
短局率上升首先记为学习轨迹信号，不单独判定 collapse；guard 被触发的线记为
`guard_triggered:elo_calibration_required`，而不是直接记为架构失败。对该线冻结触发
前后的 accepted checkpoints，在相同 primary-256 anchor、固定 openings 和交换先后手
条件下测 Elo；只有 Elo 同步下降或停滞，并且 role imbalance、policy entropy 或
accepted cadence 中至少一项也恶化时，才把短局信号升级为 collapse 证据。Warm Start
已经从 standard-late 模型开始，同样的持续恶化更反常，仍作为高优先级暂停信号，
但最终结论同样必须由 anchored Elo 校准。各架构比较优先使用相同 consumed-position
截面；提前触发 guard 的模型同时保留其实际停止位置，避免把删失样本伪装成完整 1M。

继续使用 V3 正式调度器的 accepted-champion-only replay、candidate gate、
inconclusive extension、AMP overflow retry、archive receipt、10-GiB reserve 和
explicit maximum-position bound。主要评价 anchored Elo slope、accepted cadence、
局长、短局率、policy entropy、先后手差异、collapse 和 seed variance。

Stage 2B 后冻结架构族。Flash Lite 的 512-sim CPU 三秒目标、Balance 的 CPU
Pareto 选择和 Flagship 的 GPU scaling 属于随后的尺寸实验。

截至 2026-09-15，Balance closed-loop 的实际执行批次已由
[`PLAN_Stage2_BAL5_Selfplay_V1.md`](PLAN_Stage2_BAL5_Selfplay_V1.md) 具体化。BAL-5
以 `column/winning × none/serial-attention` 四个冻结模型、单 primary seed 运行 cold 1M
与 warm 5M；它替代本节早期“基线加两个 finalist、两个 seed”的通用 Stage 2B 草案，
但不改变随后用第二 seed 复验最终架构结论的要求。

## 7. 可执行入口与产物

```powershell
python -m training.v3.stage2 audit-data --source-dir <standard-replay> --metrics <standard-metrics.jsonl> --mixed-source-dir <mixed-replay> --mixed-metrics <mixed-metrics.jsonl> --output <audit.json>
python -m training.v3.stage2 freeze-data --audit <audit.json> --output-dir <stage2-pools>
python -m training.v3.stage2 calibrate-models --output <architecture-matrix.json>
python -m training.v3.stage2 design-matrix --output <experiment-design.json>
python -m training.v3.stage2 train --config <offline-run.json>
python -m training.v3.stage2 evaluate --config <offline-run.json> --checkpoint <checkpoint.pt> --output <report.json>
python -m training.v3.stage2 summarize --reports <reports...> --output <round1-summary.json>
python -m training.v3.stage2 generate-stage2b --base-config <formal.json> --architecture-matrix <matrix.json> --finalists <finalists.json> --warm-starts <standard-late-warm-starts.json> --output-dir <stage2b-configs>
```

正式配置和摘要可提交；raw pools、checkpoints、logs 和 profiler 输出存放于忽略的
`training/runs/stage2/`。每个报告必须记录 model config、seed、train regime、
consumed positions、replay/checkpoint hashes 和 cross-regime results。

`stage2_mixed_promotion.example.json` 展示 promotion-only mixed 分支；它必须提供
完全相同架构和尺寸的 `warm_start_checkpoint`，缺失或不匹配时拒绝运行。

V3 Elo 使用六 anchor registry `anchored_elo_historical_v3.json`，其 canonical hash
固定为 `806753498c10ce585a9b7586276eaa9037637be2b050072d0b684b9461773b79`。
Stage 2 启动前必须通过 `stage2_elo_protocol_v1.json` 校验六个模型、opening manifest
及冻结的 `primary_256`/`final_512` P050 scale；V1/V2 match 或 scale 不得混入。
`pressure_256v512` 保持独立方向性压力报告，不并入对称 Elo。Stage 2 本身不改写
anchor registry 或 Stage 1 标尺产物。

## 8. 验收标准

- 十种架构均通过 training/search forward、固定 heads 和 strict checkpoint tests；
- 旧 `gravity_resnet` config hash、model artifact 和 state dict 保持兼容；
- column/XZ/YZ/raw-3D 轴向测试，以及六斜截面的路径、玩家视角、padding mask、
  XY 回填和 D4 封闭性测试通过；
- Stage 2B 生成器严格生成 3 architectures ×2 seeds ×2 initializations，Warm
  artifact 若架构、standard_late lineage 或 SHA-256 不匹配必须拒绝；
- pool 分割可复现、等量、game-level 无泄漏，并能检测 manifest/NPZ 篡改；
- offline resume 保持 sample cursor、sample IDs、optimizer、scheduler 和权重一致；
- `compileall`、全部 `test_training_v3_*.py` 和最小 CPU architecture smoke 通过。
