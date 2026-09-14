# Stage 2A-R3 三档部署架构与 Scaling 实验计划 V1

## 1. 定位与当前边界

本轮正式定义为 **Stage 2A-R3：面向 Flash、Balance、Pro 三档部署目标的表示、
架构与 scaling 扩展实验**。它是
`PLAN_Stage2_Architecture_Experiments_V1.md` 的后继计划，不覆盖原计划，也不改写
已冻结的 Stage 2A、Stage 2B 数据、配置或结论。

R3 不再只回答“同参数量下哪个架构更好”，而是依次回答：

1. 更完整的 3D 信息能否提高样本效率、泛化和棋力；
2. 架构收益是否随参数量和数据量扩大而持续；
3. 收益是否超过 seed 方差，并能在 closed-loop 中保持；
4. 哪些架构分别构成 Flash、Balance 和 Pro 的有效 Pareto 前沿。

### 1.1 当前只准备，不启动

本计划批准立即开展的工作仅限于：

- 文档、严格配置、模型组件和统一接口；
- 参数校准、architecture matrix 和实验配置生成；
- geometry-stratified evaluation 与 CPU/GPU benchmark 工具；
- checkpoint、config hash、轴顺序、D4、恢复和最小 smoke 测试；
- 1M/3M 配置模板、报告 schema、dry-run 和复现命令。

在启动门槛满足之前，明确禁止：

- 启动 R3 GPU 训练、离线大规模训练或 self-play；
- 修改当前 Stage 2B 的 manifest、queue、gate cadence、replay、stability guard 或
  retention 语义；
- 用 R3 准备工作中断、抢占或重排当前正式训练线；
- 覆盖已有数据池、checkpoint、结果矩阵或归档 receipt；
- 根据尚未完成的单 seed 轨迹提前宣布 R3 finalist。

所有新训练实现继续只扩展 `training/v3/`，不得借用或修改 Legacy 训练代码。

### 1.2 `R2_PRIMARY_SEED_COMPLETE` 启动门槛

只有同时满足以下条件，R3 才允许从“准备”切换到“执行”：

1. 当前计划范围内 seed `271828` 的 cold/warm 正式线均达到 5M consumed
   positions，或以明确、可复现的 stability guard reason 在安全边界停止；
2. 对这些终态 checkpoint 完成固定 opening、交换先后手的 anchored Elo；
3. checkpoint hash、config hash、queue state、运行报告和归档 receipt 已冻结并可追溯；
4. 已生成 seed271828 的阶段结果矩阵，缺失项明确标记 `pending` 或
   `failed:<reason>`；
5. 已复核 R3 所需的数据、GPU 时间、磁盘 reserve 和 archive staging headroom。

seed `314159` 仍是后续确认工作，不因优先完成 seed271828 而被删除。R3 可以在
`R2_PRIMARY_SEED_COMPLETE` 后启动，不必等待全部 seed314159 线结束；但 seed314159
结果必须与 R3 分开记录，且任何最终架构结论仍须至少两个 seed 支撑。

## 2. 三档模型定义

| 档位 | 主要环境 | 主要延迟口径 | 设计目标 |
|---|---|---|---|
| Flash | 普通 CPU | 512 sims 接近或低于 3 秒 | 即时响应、参数与算子效率、基本棋力 |
| Balance | CPU 可用 | 512 sims 约 10 秒以内 | 更完整地使用 3D 信息，平衡棋力、样本效率和成本 |
| Pro | GPU 推荐 | 512/1024 sims 与 batch scaling | 3D 信息保真、全局建模和棋力上限 |

三档是部署目标，不是强制的 trunk 类型。ResNet 仍可能进入 Flash，紧凑 Transformer
也可能进入 Balance；只有实验支持的完整 3D-token 或深层 Transformer 才进入 Pro。

所有档位同时报告：

- 固定 simulations 的对局与端到端搜索延迟；
- 固定每步时间预算的对局；
- 模型单次推理、完整 MCTS、吞吐和内存/显存。

固定 sims 回答网络和搜索质量；固定时间回答真实部署条件下的综合强度，两者不得
互相代替。参数量只是容量锚点，也不得作为实际延迟的代理。

## 3. 共同实验契约

所有候选继续接收相同 canonical board、role 和 rule features，输出相同的 25-way
policy、3-way WDL、opponent reply、future occupancy 和 moves-left heads。除明确的
Transformer token 路径外，encoder/trunk 最终统一产生 `[N,C,5,5]` 特征图，以复用
现有 learner、heads、MCTS 和 Replay V2 contract。

以下语义保持不变：

- P6 loss weights、occupancy weights、rule/role schemas 和 action contract；
- D4 augmentation、sample order、optimizer、LR schedule 和 batch 语义；
- AMP overflow retry、accepted-champion-only replay、gate extension；
- config/checkpoint hash、archive receipt 和 bounded execution。

不同架构或不同非默认参数必须产生不同 config hash；禁止跨架构 warm-start 或误加载。
现有 `gravity_resnet` 的 state dict key、严格加载和固定输入数值行为保持兼容。

## 4. Flash 实验通道

Flash 优先研究 CPU 友好的二维/柱级结构：

| ID | Architecture | 作用 |
|---|---|---|
| F0 | `gravity_resnet` | Stage 1 固定控制 |
| F1 | `column_resnet` | 轻量垂直柱先验 |
| F2 | `column_bottleneck_resnet` | 检查更高效的宽表示 |
| F3 | compact `column_conv_attention` | 少量 25-token attention 的局部/全局混合 |

### 4.1 Flash 矩阵

| 阶段 | 参数锚点 | 数据量 | 目的 |
|---|---:|---:|---|
| F-1 | B8，约 5.63M | 1M | 完成标准离线评估并回传本地做正式 CPU benchmark |
| F-2 | B10，约 12.33M | 1M | 作为 Flash→Balance 容量边界诊断；只有 512 sims 实测达标才可留在 Flash |
| F-3 | 最佳两项 | 续训到 3M | 验证数据扩展收益和 learning slope |
| F-4 | finalist | 第二 seed | 对差距落入既有 seed 方差的组合确认 |

1M checkpoint 的本地延迟评估至少使用 50–100 个冻结非终局状态，每个状态重复三次，
固定 CPU 线程数、电源模式和 runtime，分别报告 mean、median、p90、p95。战术快捷路径
命中与不命中必须分开统计。只有正式 benchmark 才能用于 3 秒门槛判断。

## 5. Balance 实验通道：3D 信息优先

Balance 的核心不再是单纯强化二维 multi-view，而是系统比较 raw voxel、分解 3D
卷积、learned height collapse，以及 plane/column/multiview 与 raw-3D 的 late fusion。
纯二维网络只作为控制和效率参照。

### 5.1 Balance 架构集合

| ID | Architecture | 表示与实验假设 |
|---|---|---|
| B0 | `gravity_resnet` | 无显式 3D 结构的固定控制 |
| B1 | `raw3d_resnet` | dense voxel 3D baseline，测直接 3D 建模上限与成本 |
| B2 | `raw3d_to2d_resnet` | 浅层 dense 3D encoder、显式高度压缩与 2D trunk，分离 3D 表示和主干成本 |
| B3 | `factorized3d_resnet` | `(1×3×3)+(3×1×1)` 等空间/Z 分解卷积，降低 3D CPU 成本 |
| B4 | `plane3d_fusion_v2` | 14-plane 与浅层 raw-3D 分支在较晚阶段融合 |
| B5 | `column3d_fusion_v2` | vertical-column 先验与 raw-3D 几何互补 |
| B6 | `multiview3d_fusion_resnet` | 强化 XY/XZ/YZ encoder，并保留 raw-3D 分支到较晚阶段 |
| B7 | `winning3d_fusion_resnet` | multi-view、六个 winning sections 与 raw-3D 的完整获胜方向组合 |

B2–B7 应通过可复用组件实现，而不是复制整套模型：

- `Dense3DBranch` 与 `Factorized3DBranch`；
- `MeanHeightCollapse` 与 `LearnedHeightCollapse`；
- `ConcatFusion` 与 `GatedFusion`；
- plane、column、multiview、winning-section encoder；
- 统一 trunk 与 heads adapter。

配置必须严格拒绝与 architecture 无关的 3D branch、collapse、fusion 和 branch-width
字段。每个 architecture matrix 行需记录 encoder、3D branch、fusion、trunk 和 heads
的参数数目及比例，避免把“更强 3D 表示”与“单纯从 trunk 挪走参数”混为一谈。

### 5.2 Balance 主矩阵

250k 只作为功能和数值健康检查，只淘汰实现错误、数值失败或被稳定支配的配置，
不依据小幅 loss 排名激进裁剪。

| 阶段 | 架构 | 参数锚点 | 数据预算 | 目的 |
|---|---|---:|---:|---|
| BAL-1 | B0–B7 | B6，约 1.93M | 1M | 等参数表示筛选、几何评估和正式延迟 |
| BAL-2 | B0 与所有未被稳定支配的 3D 候选 | B8，约 5.63M | 1M→3M | 完整 parameter × data scaling |
| BAL-3 | 最佳两个 3D 架构族+B0 | B10，约 12.33M | 3M，必要时5M | Balance 高容量确认 |
| BAL-4 | 最佳 fusion 架构族 | B8 或 B10 固定总参数 | 1M/3M | 约25%/50% 3D 分支参数占比消融 |

### BAL-4C：同架构 3D depth/width/fusion 数据量诊断

在把优势组合扩展到其它架构并统一训练至 3M 前，先固定
`multiview3d_fusion_resnet`、2D encoder 输出 64、trunk C248/B10、learned height
collapse、FP32、seed 271828 和 `standard_late` 数据顺序，测试完整的：

`volume_channels ∈ {96,128} × volume_blocks ∈ {5,7} × fusion ∈ {concat,attention}`。

这形成 8 个配置。除原计划的 6 个点外，补入 `D96/B7 attention` 与
`D128/B5 attention`，从而分别估计 width、3D depth 和 fusion 主效应，避免只在
`D96/B5` 与 `D128/B7` 比 attention 时发生混杂。

先补齐所有配置的 1M endpoint，并在 1M 与 3M 使用完全相同的 12 条 factorial
对局边：4 条同 D/B 的 concat-attention 边、4 条同 width/fusion 的 B5-B7 边、
4 条同 depth/fusion 的 D96-D128 边。每条边固定 50 opening pairs、交换先后手、
256 simulations。这样可以直接观察各效应是否随训练量改变，而不需要承担 28 组完整
round-robin 的成本。`D96/B3 gated` 作为先前能效对照，`D64/B3 concat` 继续作为
跨轮全局对手；只在主矩阵出现冲突时追加必要对局。

1M 与 3M 之间必须精确续训，保留 optimizer、scheduler 和 sample cursor。完成后按
离线质量、factorial direct Elo、参数/MACs、吞吐和 1M→3M slope 冻结 top2–3；随后
只有在当前 Round 3 的 1M transfer 证据支持跨架构有效性时，才把这些组合扩展到
`column3d_fusion_v2` 与 `winning3d_fusion_resnet`。Elo 巡检按约每 4 组对局 3 小时
估算，而不是机械使用 30 分钟频率。

### BAL-4D：冻结 3D 主线后的 CNN 尾部与 2D encoder 消融

BAL-4C 的 3M anchored Elo 冻结以下三项，不再重开 3D width/depth/fusion 的完整搜索：

- **能效主线**：`column3d_fusion_v2`，2D 输出 E64、3D D96/B5、concat、trunk C248/B10；
- **计算量备用**：同一 column 表示的 D96/B7 concat，只在剩余算力充足或 B5/B7
  定向复核支持深度收益时继续；
- **跨表示参照**：`winning3d_fusion_resnet` 的 E64、D96/B5 concat。

后续不做 representation × tail × encoder-width 的全笛卡尔积。先在能效主线上分别估计
两个小因子，再只组合各自胜出项。

#### BAL-4D-T：CNN trunk 后处理

固定 column E64、D96/B5 concat、C248/B10、FP32、seed 271828 和相同
`standard_late` sample/augmentation 顺序，比较：

| ID | `post_trunk_mode` | 数据流 | 作用 |
|---|---|---|---|
| T0 | 空/none | activated CNN trunk → heads | 复用当前 3M 基线 |
| T1 | `serial_attention` | activated trunk → 2 个 25-token attention blocks → heads | 对齐 CubeSprite V3 的串行 global-context tail |
| T2 | `parallel_attention` | local token-MLP 与 2-block global-attention 并行 → learned alignment → heads | 验证草图 C 的局部/全局双路组织 |

attention 统一为 8 heads、MLP ratio 2.0，不使用 CLS token，也不加入 absolute position
embedding。卷积主干已经生成位置对齐的 5×5 feature map；本轮不把 positional encoding
或 D4 对称性变化混入 tail 因子。T2 的 alignment 是 `concat(local, global) → Linear(C)`，
初始化为两路等权对齐。

#### BAL-4D-E：2D representation encoder 输出宽度

固定无 post-trunk tail、3D D96/B5 concat、C248/B10，并保持 column MLP hidden width
为 248，只改变其送入 fusion 的输出宽度：

| ID | 2D 输出宽度 | 说明 |
|---|---:|---|
| E64 | 64 | 复用当前 3M 基线 |
| E96 | 96 | 与 3D D96 对称，检验 E64 是否为 representation bottleneck |
| E128 | 128 | 检验继续增加 2D 表示容量是否仍有收益 |

这里不同时改变 hidden width。column encoder 的 hidden 参数成本很小，而 fusion 前输出宽度
才直接控制保留到 trunk 的 2D 表示容量；同时改变两者会使解释失焦。

#### 执行与晋级

新训练点只有 T1、T2、E96、E128 四个；T0/E64 是同一个已完成基线。四项先训练到
1M，完成四池交叉验证和各子组 50 opening-pair、256-sim 直赛；随后四项均精确续训到
3M，再重复同一组对局，避免 attention 或较宽 encoder 因 1M 学习较慢被提前淘汰。

- Tail 子组：T0–T1、T0–T2、T1–T2；
- Encoder 子组：E64–E96、E64–E128、E96–E128；
- 不在这一阶段跑完整 anchor matrix；以组内棋力、macro loss、参数、MACs、训练吞吐组成
  Pareto 判断，不能用单一 loss 或单一 Elo 排序替代能效判断；
- 若 tail 与 encoder width 各自都有正收益，只新增一个“最佳 tail + 最佳 width”组合点；
  若任一因子不优于基线，则不强行组合；
- 组合确认后迁移到 D96/B5 concat winning 参照，检验收益是否跨 representation；
- 最终 B10 候选回传本地执行正式 512-sim CPU latency，并只对 finalist 追加 anchored Elo。

完成 BAL-4D 后，如 GPU 预算仍有余量，再固定胜出的 encoder/tail/representation，比较
C248/B10 与 C248/B12。B12 先跑 1M；只有离线曲线和组内直赛未被 B10 稳定支配时才精确
续训到 3M。不得在 tail 或 2D encoder 尚未冻结时同时改变 trunk depth。

B8 的 3M 模型必须从对应 B8 1M checkpoint 精确续训。不得只运行 B6-1M 与 B8-3M
两个对角点，否则无法区分参数收益、数据收益和二者交互。

当前约 12.33M 参数的 BAL-3 模型在 512 sims 下预计均超过 Flash 的 3 秒门槛，故默认
归入 Balance 候选；其已测 256-sim 延迟不能替代 512-sim 正式判据。Balance 的 10 秒
目标在 BAL-1/BAL-2 中仅记录，不作为早期硬淘汰条件。只有进入部署
finalist 后才把它作为产品门槛。对于 3D 架构，参数量相近而 CPU 延迟更高是预期现象，
必须用实际 Elo、固定时间棋力和 geometry-stratified 收益一起判断。

## 6. Pro / Transformer 实验通道

25-token Transformer 已具有足够的全局通信距离。Pro 首先检查 3D 信息是否在
tokenization 或 early fusion 中丢失，其次才研究更深网络和 Attention Residuals。
AttnRes 只能重新访问早期 representation，不能恢复 encoder 已丢弃的 voxel 信息。

### 6.1 PRO-1：位置与 tokenization

| ID | 设计 | 目的 |
|---|---|---|
| A0 | 当前 25 column tokens + absolute position embedding | 固定基线 |
| A1 | A0 + D4-tied 2D relative positional bias | 检验位置关系编码 |
| T1 | 25 layer-aware column tokens | 压柱前显式处理六层顺序 |
| T2 | 25 column + 6 layer-summary tokens | 低成本恢复全层摘要 |
| T3 | 150 cell tokens | 保留完整 `(z,y,x)` voxel 信息 |

PRO-1 统一采用约 B10 参数预算、`standard_late`、1M、单 seed。T2 的 layer tokens 只是
全层摘要，不能宣称能直接访问每个具体格点；只有 T3 完整保留 150 个 cell。

T3 第一版使用固定 Z-pooling 将每柱六个 cell token 还原成 25 个 action features，再
复用现有 heads。暂不同时加入 action-query cross-attention，以免 tokenization 与输出
读出同时改变。

### 6.2 PRO-2：上下文与读出

只在 PRO-1 最佳 tokenization 上比较：

- 现有 FiLM；
- FiLM + 一个联合 rule/role context token；
- 仅当 T3 显示明确收益后，再比较固定 Z-pooling 与 25 action-query cross-attention。

### 6.3 PRO-3：深层信息路由

在最佳 tokenization 上先比较 Vanilla B6 与 Vanilla B12，确认深度确有收益后，再于
相同 B12 宽度比较：

- Vanilla residual stream；
- gated Input Skip；
- Attention Residuals。

Input Skip 是必要控制项。只有 AttnRes 在稳定性、Elo/样本效率和显存/吞吐综合上
超过 Vanilla 与 Input Skip，才进入 B16。MHAR 不进入本轮首批实现。

### 6.4 PRO-4：容量扩展

按约 12M → 25–35M → 约 60M 参数逐点扩展。只有连续两个容量点保持明确正向
scaling slope，且收益超过 seed 方差和置信区间，才启动下一点。Pro 的 CPU 延迟只记录，
不参与淘汰；主要报告 GPU batch 1/8/32/64 延迟、吞吐、峰值显存和 256/512/1024 sims
端到端搜索成本。

## 7. 数据规则

R3 第一轮统一使用冻结的 `standard_late` 数据，所有候选共享 train/validation split、
sample order、augmentation token 和监督目标。原 `standard_early`、`standard_mid`、
`standard_late`、`mixed_late` 四池继续用于 cross-regime evaluation。

`mixed_late` 只提供给已经通过 standard 三池筛选的模型，作为后续 promotion，不得
在第一轮静默混入 `standard_late`。

如现有数据不足，可新增 `expansion_late_v1`，但必须：

1. 预先冻结唯一 donor checkpoint 和 SHA-256；
2. 所有待比较架构使用同一新增池；
3. 固定 game-ID split、sample order、Replay V2 targets 和 checksum；
4. 独立记录 lineage，不覆盖或重命名原四池；
5. baseline 与候选获得相同数据量。

不得为单个“更吃数据”的架构私建训练池。候选自身 self-play 只用于后续 closed-loop
验证，不进入离线横向比较。

## 8. 统一评估规则

每行主键为：

`track × architecture × parameter_anchor × train_recipe × train_regime × seed × consumed_positions × checkpoint_hash`。

### 8.1 监督学习与校准

- total、policy、WDL 和三个 auxiliary losses；
- policy CE、JSD、top-1/top-k agreement；
- WDL CE、Brier、ECE、accuracy；
- cross-regime macro mean、variance 和 mixed-standard gap。

### 8.2 棋力与 closed-loop

- anchored Elo、95% CI、Elo slope/每百万 positions；
- 固定 opening 的 paired win/draw/loss 与颜色交换一致性；
- 先后手分项、accepted cadence、policy entropy；
- 平均及分位数局长、短局率、collapse/stability pause 和 seed variance。

### 8.3 3D geometry-stratified tactical suite

冻结一套不参与训练的战术集合，按真实获胜方向分层：

- XY 水平/纵向/平面对角；
- Z 轴 vertical；
- XZ/YZ；
- 六个长度至少为 4 的 winning sections；
- 3D space diagonals；
- immediate win、forced block、double threat。

每类报告样本数、policy top-1/top-k、目标动作 rank、CE/JSD、WDL Brier/ECE、D4
一致性和 bootstrap CI。该集合用于判断显式 3D 表示是否改善对应几何，不能取代 Elo。

### 8.4 效率与 scaling

- 总参数及 encoder/3D/fusion/trunk/heads 参数占比；
- MACs/FLOPs、训练 positions/s、wall-clock、GPU-hours；
- CPU mean/median/p90/p95 搜索延迟；
- GPU batch latency/throughput、峰值显存；
- `ΔElo/log(parameters)`、每百万 positions 的 Elo/loss slope；
- parameter × data interaction。

Loss 用于解释学习和校准，Elo 用于棋力结论，效率用于三档部署选择，任何一个指标
都不能替代其余两类。

## 9. 晋级、复验与停止规则

- 250k：只排除数值失败、接口失败或被稳定支配的实现；
- 1M：完成 cross-regime、geometry suite、50 opening pairs 和正式 latency；
- 3M：判断 parameter/data scaling slope，不用单点总 loss 排名；
- finalist：至少两个 seed、200 opening pairs；
- 单 seed 只允许形成候选排序，不允许冻结最终架构族；
- 大模型连续两个容量点的收益低于预注册最小收益且成本明显恶化时停止 scaling；
- AttnRes 若不能稳定超过 Vanilla 与 Input Skip，则停止继续加深该分支；
- 不使用单一综合分数，分别生成 Flash、Balance、Pro 三张 Pareto 表。

## 10. 实施分期

### 10.0 2026-09-10 准备批次边界

本批次先完成 Balance 主路径和公共控制面：`raw3d_to2d_resnet`、
`factorized3d_resnet`、四种 3D late-fusion、显式 volume depth、mean/learned
height collapse、concat/gated fusion、分模块参数审计、几何分层评估、正式 CPU latency
证据以及 R2 readiness receipt。`gravity_resnet` 与既有 Stage 2 state dict 保持不变。

Flash 的 `column_bottleneck_resnet` / `column_conv_attention` 与 Pro tokenizer、relative
bias、Input Skip / AttnRes 仍列为计划候选，在 primary seed 证据冻结后接续实现和消融。
这不是把它们视为已通过候选；R3 manifest 会显式区分 `ready_for_smoke` 与
`planned_after_primary_evidence`，任何未实现项不能因 readiness receipt 而直接训练。

### Phase P0：文档与接口冻结（当前可执行）

- 冻结本计划、命名、严格配置字段和报告 schema；
- 定义复用的 3D branch、collapse、fusion 与 Transformer token 接口；
- 定义 `R2_PRIMARY_SEED_COMPLETE` 的机器可读检查结果。

### Phase P1：主要模型实现（当前可执行）

- Flash：bottleneck 与 compact conv-attention；
- Balance：factorized 3D、plane/column 3D fusion v2、multiview/winning 3D fusion；
- Pro：relative positional bias、layer-aware columns、layer-summary 和 150-cell tokenizer；
- 统一 build/forward/checkpoint/config-hash contract。

### Phase P2：评估与生成工具（当前可执行）

- 参数匹配与分模块参数审计；
- geometry-stratified tactical suite；
- CPU/GPU benchmark 与 architecture matrix；
- 1M/3M 配置、resume chain、结果汇总和 Pareto 输出。

### Phase P3：本地验收（当前可执行）

- 全架构 forward/search/checkpoint/config tests；
- 3D 轴顺序、玩家视角、collapse、winning sections 和 D4 tests；
- deterministic dry-run、最小 CPU smoke、`compileall` 和完整 V3 unittest；
- 不启动正式训练。

### Phase E0：启动门槛复核（等待）

- 检查 `R2_PRIMARY_SEED_COMPLETE`；
- 冻结当前 seed271828 结果与归档状态；
- 复核数据、GPU、磁盘和时间预算；
- 生成 R3 execution manifest，并由用户确认进入执行期。

### Phase E1–E3：正式实验（门槛后执行）

1. E1：Flash F-1、Balance BAL-1、Pro PRO-1；
2. E2：晋级候选的 B8/B10、1M/3M scaling 与 PRO-2/PRO-3；
3. E3：第二 seed、200 opening pairs、mixed promotion 和必要的 closed-loop；
4. 按三档分别冻结 Pareto frontier，不从单一总榜选一个通用模型。

## 11. 实现与验收清单

测试至少覆盖：

- 全部新增架构的 training/search forward shape 和五类 heads；
- voxel、column、XY/XZ/YZ、winning sections 的轴顺序和玩家视角；
- dense/factorized 3D branch、mean/learned collapse、concat/gated fusion；
- 150/31/25-token 数量、mask、relative bias 和 attention-head divisibility；
- D4 下的棋盘、policy、section、position/bias 映射；
- 无关字段、未知 architecture、非法分支比例的严格拒绝；
- 每架构 checkpoint round-trip、config hash 差异和跨架构加载失败；
- `gravity_resnet` 旧 checkpoint 与固定输入的回归一致性；
- 参数校准误差、architecture matrix 和配置生成的确定性；
- 中断/恢复后的 sample IDs、optimizer、scheduler 和权重一致；
- 最小 CPU batch smoke、可用时的 GPU dry-run、`compileall` 和全部
  `test_training_v3_*.py` 回归。

正式配置、schema、测试和摘要可提交；数据池、checkpoint、原始日志、benchmark
输出继续放在忽略的 `training/runs/stage2/`。任何生成报告都必须记录代码、配置、数据、
checkpoint、硬件与 opening manifest hash。

## 12. 最终决策输出

R3 最终不生成单一综合冠军，而输出三套部署结论：

- Flash：anchored Elo / 固定时间棋力 vs CPU 512-sim latency；
- Balance：anchored Elo、3D geometry 收益 vs CPU 512-sim latency、GPU-hours；
- Pro：anchored Elo vs GPU search latency、吞吐、显存与 scaling slope。

Balance 的正式决策原则固定为：**优先验证显式 3D 信息的有效性，再比较其实现成本；
multi-view 只有在保留较晚融合，或与 raw-3D 分支形成可验证互补时，才进入 Balance
主候选。**
