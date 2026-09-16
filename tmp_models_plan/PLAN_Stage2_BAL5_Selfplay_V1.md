# Stage 2 BAL-5：冻结 Balance 架构的闭环 Self-play 计划 V1

日期：2026-09-15

## 1. 本轮目标与边界

BAL-5 把离线 `standard_late` 架构证据推进到 classic 规则下的 closed-loop
self-play。R1 只回答四个冻结模型在独立 cold/warm lineage 中的稳定性、迭代能力与
组内棋力；R2 multi-rule 属于 `Stage3-BAL-pre`，本轮只冻结设计，不自动启动。

四个模型统一冻结：

- 2D encoder 输出 `E64`；
- 3D encoder `D96/B7`；
- fusion=`concat`；
- trunk=`C248/B12`；
- representation：`column` / `winning`；
- tail：none / `serial_attention` ×2（8 heads，MLP ratio 2.0）。

离线 donor 使用 FP32、seed 271828、3M `standard_late`。2026-09-16 启动的第一条
closed-loop AMP canary 出现连续 skipped optimizer step、GradScaler 降至
`3.0517578125e-05`，并伴随多个 head 的 loss/grad norm 发散；该 lineage 完整保留为
`failed_amp_canary_20260916`，不计作架构淘汰证据。R1 随后以新 run ID 从随机初始化重开，
四个模型统一使用 FP32 learner。

重开后的 cold 学习率在 0–1M 固定为 `1e-4`。Warm 使用新鲜 optimizer/replay，但不把
成熟 3M donor 重置到原 B10 cold schedule 的 `5e-4`：0–2M 使用 `1e-4`，2M–5M 使用
`5e-5`。这使精度和学习率调整在四个架构间保持一致，并由 config hash 区分旧 canary。

## 2. R1 执行矩阵

| ID | Representation | Tail | Cold | Warm |
|---|---|---|---:|---:|
| C0 | column | none | 1M | 5M |
| C1 | column | serial attention ×2 | 1M | 5M |
| W0 | winning | none | 1M | 5M |
| W1 | winning | serial attention ×2 | 1M | 5M |

Cold 从随机初始化开始。Warm 只加载同架构 3M `standard_late` donor 的模型权重，使用
`model_only_fresh_optimizer_replay_v1`，optimizer、scheduler、replay、sample cursor 与
game IDs 全部重新开始；不同架构或 tail 禁止交叉 warm-start。

### 2.1 Self-play 搜索与探索

- Cold 每代 400 games，Warm 每代 800 games；ply 0–11 强制 256 simulations；
  ply 12 以后按 game 路由为 32/256 simulations，各 50%；
- 基础温度为 ply 0–27 `T=1`、ply 28–49 `T=0.5`、ply 50+ greedy，并保留 Stage 1
  已冻结 Dirichlet 参数；
- 每个 lineage 的前 1M consumed positions 不启用 opening-temperature mixture；
- Warm 在第一个不早于 1M 的原子 generation 边界启用 50/50 mixture：一半 games 的
  ply 0–7 温度从 1.0 降为 0.5，另一半保持 1.0，同时 learner 按两路各 50% consumed
  positions 采样；
- Warm 将每代 self-play games 从 400 提高到 800，并把每代 learner 上限从 Stage 1
  B10 的 256 steps 提高到 512 steps。`train_tokens_per_raw_position` 固定为 4，不增加
  单个新局面的期望复用次数；新增局面供给与训练执行上限同时扩大 2 倍，最多 131,072
  consumed positions/generation。mixture 在累计 consumed positions 首次达到 1M 后的
  下一个 generation 边界启用；切换不发生在 generation 中间，实际边界和 game ID
  写入状态；
- Cold 保持 Stage 1 B10 的 256 steps/generation，且其总上限就是 1M，因此不启用
  mixture。

`opening_temperature_mixture.start_train_positions` 是新的显式学习语义。原始 always-on V1
mixture 的 config hash 在阈值为 0 时保持不变；延迟启用会产生不同 hash。

### 2.2 Gate cadence 与后手保护

- Cold：bootstrap candidate interval 100k，regular interval 200k；
- Warm：两者分别为 150k 与 300k。相比原配置，候选评估频率降为 2/3；
- classic R1 保留 `relative_noninferiority` 后手退化检测，margin=5%，使用与 candidate
  相同的 openings/seeds 生成 accepted-control 证据；
- gate 继续使用 256 simulations、50 initial pairs、25-pair increments、最多 200 pairs，
  inconclusive 不自动接受。

## 3. 队列与停止语义

1. 先并行补齐 W0/W1 两个 3M winning donor；严格验证 config、model artifact、hash、
   train regime、positions 和数值有限性；
2. 物化四模型 cold/warm 配置和完整 SHA-256 manifest；
3. 四个 cold 1M 串行执行。单个 cold 若触发 stability pause、数值失败、artifact/hash
   不一致或未到精确 1M，则标记淘汰；队列继续检查其余 cold，但该模型不进入 warm；
4. 只有通过 cold 检查的模型进入 warm 5M；
5. `archive_required`、硬盘 reserve、gate max-pair inconclusive 或 runner 异常属于操作性
   停止，不得当成架构淘汰，也不得静默跳过；主队列停止等待处理；
6. 首次巡检对准 donor→R1 切换边界。确认 R1 controller 已实际进入 cold 阶段且首个
   generation commit 正常后，巡检改为每 12 小时。

### 3.1 BAL-5 存储与自动归档修订（2026-09-16）

- 150-GiB 云盘的 soft watermark 从 70% 调整到 90%，即约保留 15 GiB；正式执行所需
  `hard_free_gib=10` 与约 4-GiB staging headroom 仍保留，因此两条触发线基本一致；
- generation checkpoint 仍是原子 commit 和精确 resume 的必要状态，不能隔代省略创建；
- verified archive receipt 返回后，云端仅保留最新 generation checkpoint；accepted、
  rejected/candidate gate 模型继续按独立目录及既有策略保留。普通中间 checkpoint 由
  receipt-gated prune 清除，不把“减少长期保留”误实现为破坏 generation commit；
- 巡检遇到 `archive_required` 时，执行增量 bundle → 本地 checksum/materialization →
  receipt 回传 → 云端 revalidated prune，并在确认空间恢复后立即以 `--resume` 重启当前
  job。任何传输、校验或 receipt 失败都停止，不自动继续训练。

## 4. R1 淘汰与评估

Cold 1M 只执行稳定性淘汰，不凭早期短局率下降淘汰；短局率下降可以是学习早期策略的
正常表现。数值失败、持续 stability pause、producer contract 破坏、replay/checkpoint
不一致或明显 collapse 才构成淘汰依据。

Warm 5M 完成后跑两条独立但同协议的组内循环赛：

- terminal line：每个 lineage 最终 checkpoint；
- accepted line：每个 lineage 最后 accepted checkpoint。

两条线都使用固定 openings、交换先后手、相同 search profile。综合得分最低者淘汰；若
第三名与第二名的 Elo 95% CI 不重叠且方向一致，也淘汰第三名，否则保留。必须同时报告
两条线的 W/D/L、Elo、95% CI、先后手分项和 checkpoint hash，不能只用一个合成分数。

## 5. R2 multi-rule 预冻结设计（本轮不执行）

R2 的 primary gate 改为规则分层 gate：

1. 多规则等权 macro paired score 的 95% CI 下界必须大于 50%；
2. 允许预注册的单规则退化容忍带。正式执行前必须在 5% 与 10% 中冻结唯一 hard
   threshold，禁止看结果后选择；
3. 每条规则维护 accepted lineage 的 local-maximum checkpoint、分数、CI 与累计回撤。
   单规则连续 2–3 代退化触发 warning；超过 hard threshold 直接 reject；
4. 取消 classic R1 的后手退化拒绝，但继续报告角色分项；
5. 不同规则使用相互独立的 opening manifests、opening IDs 与 seeds，规则内仍交换先后手。

Multi-rule replay 比较三种 producer 路由：

- Classic：所有规则都由最新 accepted model 产生数据；
- EXP1-fallback：退化规则由该规则 local maximum 单独产生数据，直至最新 accepted 超过它；
- EXP2-mix：退化规则由 local maximum 与最新 accepted 以冻结比例（首选 50/50）混合产生
  数据，直至超过 local maximum。

R2 执行前还需补齐：macro CI 的分层 bootstrap 实现、local-maximum 状态机、规则隔离的
opening lineage、producer-per-rule lineage 与三种 replay recipe 的等预算矩阵。R1 的
classic checkpoint 不自动获得 R2 准入资格，必须先通过上述淘汰规则。

## 6. R1 必留证据

- code commit、resolved config、semantic config hash、model config/hash；
- donor artifact/hash 与 warm-start receipt；
- 每代 producer model、搜索/温度 route 数量、raw/consumed positions、sample-ID digest；
- policy/WDL/aux loss、Brier/ECE、AMP overflow、LR、optimizer/global cursor；
- accepted cadence、每次 gate 的 paired evidence、后手 non-inferiority delta；
- entropy、局长分布、短局率、先后手胜率、stability action；
- wall-clock、positions/s、GPU 利用率、磁盘/归档状态；
- terminal 与 last-accepted 双线循环赛结果。

数据池、checkpoint 与原始日志保留在忽略的 `training/runs/stage2/bal5/`。本地归档以日志、
manifest、汇总报告和每条 lineage 的终点/最后 accepted checkpoint 为主；不追求保留全部
中间 checkpoint 的完全恢复能力。
