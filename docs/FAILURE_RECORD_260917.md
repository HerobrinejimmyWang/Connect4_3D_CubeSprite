# 训练失败案例记录（Failure Record）

**REMARK: Reference only! Summaried by DS_v41. May not suitable for V3 validation.**

记录日期：2026-09-17
记录范围：三条独立训练线（本仓库 V3、`Connect4_3D_AI_v2.2` 传统栈、云端 BAL-5 canary）中可判定的失败 run。

本文件只做**证据归档与归因**，不修改任何既有训练产物、checkpoint、配置或历史 JSON。
所有数字均来自被引用文件的原文；无法确定的地方在 §7 明确列为证据缺口。

---

## 0. 结论速览

| # | Run | 训练栈 | 失败类别 | 根因置信度 |
|---|---|---|---|---|
| F-1 | `bal5_r1_column_no_tail_cold_seed271828`（远程 `failed_amp_canary_20260916`） | V3 `training/v3` formal runner，`learner_amp=true` | **AMP 梯度缩放塌缩 → 连续 24 代零优化步**；恢复后 `moves_left` 头数值发散；最终被 `stability_pause` 停机 | 高 |
| F-2 | `v2.2_large` 主训练（300→400 迭代续训） | Legacy `training/main_train.py` + `trainer.py`，C256 8-block ResNet（29.39 M 参数） | **自对弈最小长度先手胜吸引子**（长期存在）+ **iter 300 后一次被确认负号写反的 virtual loss 线程实验**（触发/放大）：局长塌到 7 步（理论最小）、方差 0.85、97.5% 短局；**导出的权重本身健康**（对外 13-73 步、55% 胜率） | 高（机制与触发），低（各自的量化贡献） |
| F-3 | `midterm_gravity_balanced` / `cubesprite_v3_mini` 共 26 条 run | `distillation/main_distill.py`，`gravity_balanced` 128×6 | 静默 kill（7）、value-head 塌缩（2）、相对 gate 噪声放行（9）、teacher 监督过早归零 | 高 |
| F-4 | `stage1_b10c256_relative_role_guard_coldstart_400g_...` g000003 | V3 `training/v3` | relative-role 冷启动 bootstrap 实现错误，G3 未提交状态被丢弃并从 G2 重放 | 高（仓库自述） |
| F-5 | `stage1_scale_screen_b6c128_2x3080ti` g00222 | V3 `training/v3` | 操作型失败：`too many open files` | 高 |
| F-6 | `bal5_r1_column_no_tail_cold_fp32lr1e4_seed271828`（对照组，未停机但已退化） | V3，FP32 learner | **同一自对弈退化**：11 代后平均局长 16.4→11.0，短局率 0.28→0.76；可赢局面占比 5.2%→17.5% | 高 |
| 附 | §2.2e 引用的 `check_info_guides.md` / `TRAINING_RUNBOOK.md` 阈值（非失败 run） | 作者（人类）在 **V3 蒸馏线实施之前**、从**旧训练栈**经验提炼 | **旧栈经验规则，不能直接作为 V3 的验收依据**；文中"检查者/inspector"均指 **AI agent** | — |

**跨三条训练栈的共同信号**：三条线（Legacy v2.2_large、V3 AMP canary、V3 FP32 对照）全部出现
**自对弈局长塌缩 + 短局率飙升**，且三栈都**没有"存在立即赢/必须封堵则短路"的安全网**。
这是本记录中最重要的跨栈发现。

关键定性（由 §2.3e 的 arena 反证确定）：**这是自对弈动力学问题，不是规则问题，也不是权重问题。**
`v2.2_large` 导出的同一份权重对外部对手打了 40 局、局长 13-73 步、**0 局 ≤ 9 步**、55% 胜率——
7 步塌缩只在"同一网络执双方"时出现。三栈的监控（`policy_entropy`、`value_loss`、
健康门下界）对这类塌缩全部无效，因为塌缩期的熵**上升**而非下降。

---

## 1. F-1：云端 BAL-5 AMP canary（`failed_amp_canary_20260916`）

### 1.1 Run 信息

| 项 | 值 |
|---|---|
| 主机 | `connect4_gpu_2608`（AutoDL 容器 `autosdl-container-s22hgk1338-3d52b6d3`） |
| GPU | 2 × NVIDIA GeForce RTX 3080 Ti（12 GiB each） |
| 远程路径 | `/root/autodl-tmp/Connect4_3D_game_refactor/training/runs/stage2/bal5/failed_amp_canary_20260916` |
| 失效 run | `bal5_r1_column_no_tail_cold_seed271828` |
| 后续 run（同目录内被控制器启动） | `bal5_r1_column_serial_attn2_cold_seed271828` |
| run_id / 状态 | `stopped_at_safe_boundary`, `stop_reason = "stability_pause"` |
| git commit | `b141b5884a086e01e51a8d6e8d045b9b32ff9578`（本仓库存在该 commit） |
| code version | `3.1.0-foundation` |
| config_hash | `a31194d4058d2cc75167dd968300946443a1b5dde81d8c1078252cc87c2b45fb` |
| 时间 | `2026-09-15T10:39:06Z` → `2026-09-15T14:14:02Z`（约 3h35m） |
| 规模 | 44 generations，17,600 games，282,516 raw replay positions，18 GB |
| 磁盘 | 150 GiB 卷，used_fraction 0.4928，`hard_reserve_breached=false`（**非磁盘原因**） |
| 训练预算 | `max_train_positions = 1000000`，实际 `train_positions_consumed = 999936` |

远程 `manifest.json` 自述：

```json
"code_precision": "formal V3 learner AMP enabled for all R1 lineages"
```

本地 `tmp_models_plan/PLAN_Stage2_BAL5_Selfplay_V1.md` 第 20–24 行确认其定位：

> 2026-09-16 启动的第一条 closed-loop AMP canary 出现连续 skipped optimizer step、GradScaler 降至
> `3.0517578125e-05`，并伴随多个 head 的 loss/grad norm 发散；该 lineage 完整保留为
> `failed_amp_canary_20260916`，不计作架构淘汰证据。R1 随后以新 run ID 从随机初始化重开，
> 四个模型统一使用 FP32 learner。

### 1.2 训练栈与配置

- 入口：`python -B -m training.v3 run --config <cfg>.json --execute --max-train-positions 1000000`
- 模型：`architecture=column3d_fusion_v2`，`channels=248`，`blocks=12`，`volume_channels=96`，
  `volume_blocks=7`，`branch_channels=64`，`collapse_mode=learned`，`fusion_mode=concat`，
  `moves_left_classes=301`，`output_schema=policy_wdl_aux_v1`
- Learner：`batch_size=256`，`max_optimizer_steps_per_cycle=256`，`grad_clip_norm=1.0`，
  `weight_decay=1e-4`，`learner_amp=true`，`num_workers=2`
- 损失权重：`policy=1.0`，`wdl=1.0`，`opponent_reply=0.15`，`future_occupancy=0.15`，`moves_left=0.05`
- 学习率表：`0.0005 @0` → `0.00025 @100k` → `0.00015 @500k` → `0.0001 @2M` → `5e-5 @5M`
- 自对弈：`actor_processes=40`，`mcts_lanes_per_actor=4`，`inference_batch_size=32`，
  `full_search_sims=256` / `fast_search_sims=32`（`full_probability=0.5`），400 games/generation，
  `cpuct=1.5`，`virtual_loss=1.0`
- 探索相：`T=1.0, α=0.24, ε=0.06`（ply 0–27）→ `T=0.5`（ply 28–49）→ greedy（ply 50+）
- Replay：`train_tokens_per_raw_position=4.0`，`train_fraction=0.95`，`shard_games=40`，
  `window_alpha=0.75`，`window_beta=0.4`，`window_c=250000`
- Stability：`game_length_pause_start_train_positions=200000`

### 1.3 具体体现

**(a) 连续 24 代零优化步。** `metrics.jsonl` 的 `stage=learner` 记录（`steps`/`positions`）：

| generation | 0 | 1 | 2 | … | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|
| `steps` | 1 | 0 | 0 | … | 0 | 256 | 256 | 144 | 51 |
| `positions` | 256 | 0 | 0 | … | 0 | 65536 | 65536 | 36864 | 13056 |
| `train_positions_consumed` | 256 | 256 | 256 | … | 256 | 65792 | 131328 | 168192 | 181248 |

`generation_commit` 记录中 `optimizer_steps` 完全一致（g0=1，g1…g23=0，g24 起恢复正常）。
早期 `grad_norm`：g0 = **56.318**，随后各代为 **0.0**（代码只在优化步生效时记录 grad_norm，
见 `training/v3/learner.py:794`）。`learning_rate` 全程停在 `0.0005`（调度器不推进）。

**(b) 策略从未脱离随机初始化。** validation 的 policy CE 从 g0 的 3.2384 到 g23 的 3.2367，
44 代只下降 0.0002；WDL accuracy 在 0.483–0.486 之间抖动（`ln 3 = 1.0986`，WDL CE ≈ 0.805 > 1.0986）。
self-play `mean_policy_entropy.full` 在 g0–g36 全程钉在 **3.020–3.049**（`ln 26 ≈ 3.258`，
即几乎均匀分布于 25-26 个合法着法），仅在 g37 一次跳到 2.875 后保持到 g43——
**整段 44 代的熵变化只有 5%**。producer 在 g0–g25 为 `random`，此后为 `candidate`，
**accepted 从未更新过**。

**(c) 第一个候选 checkpoint 即 accepted 到停机。**
`accepted_model_id = "candidate-g000036-s00002552-d00238755"`，来自 generation 36。
g25–g36 的 gate 全部 `not_run` 或未 accept。

**(d) AMP 恢复后 `moves_left` 头发散。** `moves_left_loss`（301 类，权重仅 0.05）轨迹：

| gen | 24 | 28 | 30 | 32 | 33 | 34 | 35 | 36 | 40 | 42 |
|---|---|---|---|---|---|---|---|---|---|---|
| `moves_left_loss` | 3.43 | 22.03 | 50.99 | 99.59 | 146.97 | 190.41 | 244.66 | 281.56 | 279.30 | **285.85** |
| `total_loss` | 4.93 | 5.87 | 7.39 | 9.95 | 12.34 | 14.53 | 17.29 | 19.37 | 19.28 | 19.60 |
| `policy_loss` | 3.3228 | 3.2194 | 3.2248 | 3.2218 | 3.2252 | 3.2269 | 3.2268 | 3.2969 | 3.2975 | 3.2919 |

`moves_left_loss × 0.05` 在 g42 贡献约 **14.3**，占 `total_loss` 19.60 的 73%，而 policy 头
（权重 1.0）只有 3.29。`opponent_reply_loss` 同期 3.24 → 5.92。

**(e) 停机原因。** `run_manifest.json`：`status="stopped_at_safe_boundary"`, `stop_reason="stability_pause"`。
per-generation `stability` stage 显示：

- g0–g27：`action="watch"`，`behavioral_signals=[mean_game_length_low, game_length_variance_low, short_game_rate_high]`，
  全部带 `contextual_signals=["game_length_pause_warmup"]`（`train_positions < 200000` 的豁免）。
- g28–g42：`consecutive_behavioral_alerts` 恒为 1（被 `stability.py:239` 的
  “无联合恶化趋势则重置为 1” 逻辑压住）。
- **g43：`action="pause"`，`consecutive_behavioral_alerts=2`，`first_pause_generation=43`**，
  `contextual=[game_length_variance_critical, value_loss_low]`。

**(f) 进程级异常。** 两个 run 的 log 开头都有 DataLoader worker SIGABRT 遗留：

```
RuntimeError: DataLoader worker (pid 997130) is killed by signal: Aborted.
RuntimeError: DataLoader worker (pid 91990) is killed by signal: Aborted.
...
Exception in thread Thread-41 (_pin_memory_loop):
  ... File ".../torch/utils/data/_utils/pin_memory.py", line 59, in _pin_memory_loop
  ... File ".../multiprocessing/connection.py", line 399, in _recv  -> raise EOFError
V3 error: DataLoader worker (pid 285212) is killed by signal: Terminated.
```

这些是**前一次调度残留的 stderr**（`bal5_r1_column_no_tail_cold` 的 log 第 1–45 行，
出现在 log 主体 JSON 之前），最终那条 `Terminated` 出现在
`bal5_r1_column_serial_attn2_cold_seed271828.log` 末尾——即控制器在
`2026-09-15T14:14:03Z` 启用第二个 job 后，该 job 也立即失败，
与 `controller_state.json` 的 `status="running"`、`active_job=bal5_r1_column_serial_attn2_cold_seed271828`
（`active_started_at=2026-09-15T14:14:03Z`）以及之后**再无日志产出**一致。控制器本身没有写清理状态。

### 1.4 失败原因分析

**主因（高置信度）：AMP 梯度缩放塌缩，且“跳过步”检测逻辑存在缺口。**

代码路径 `training/v3/learner.py:771-793`：

```python
if self.amp_enabled:
    scale_before = float(self.scaler.get_scale())
    self.scaler.scale(total_loss).backward()
    self.scaler.unscale_(self.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    optimizer_step_applied = float(self.scaler.get_scale()) >= scale_before
...
if not optimizer_step_applied:
    break
```

三点观察：

1. **判据不是 PyTorch 的权威信号。** `GradScaler` 已经记录了本迭代是否真的执行了
   `optimizer.step()`，直接读该状态即可；用 `get_scale()` 的前后比较是间接代理。
   当 scale 触及下限（`GradScaler` 下限为 1.0）后，溢出迭代满足 `get_scale() >= scale_before`
   （`1.0 >= 1.0`），**会把被跳过的步误判为已生效**。
2. **反过来说，`break` 语义本身是“每代只跑一个 batch”。** 24 代 `train_positions_consumed`
   恰好每代 +256 = 1 × batch_size，说明循环确实在第一处跳过检测就退出了。
3. **`train_data_ratio` 的读数印证了饥饿状态。** 该值 = `total_positions_consumed / total_positions_added`
   （`replay.py:706-710`）。g0 为 **0.0389**，g1 为 **0.0195**，g23 降到 **0.0017**，
   g24 起恢复到 0.4095 并在 g28 之后稳定 ≥1.0。
   即在零步阶段，learner 每代只消费 256 个位置，而每代新增约 6,400 个。

**为什么 g0 成功而 g1 起全部失败：** g0 跑完全部 256 步（`positions=256` 是首个 batch 的
`positions += batch_count` 计数），g1 起第一个 batch 即被跳过——输入分布相同
（`average_sample_age` g0 = 3488.6），因此可复现地触发了非有限梯度。若每代恰好
halve 一次，65536 / 2^16 = 1.0，**16 代后 scale 触底**，与观测到的 g0 成功、g1–g23 失败、
g24 起恢复的时间线一致。

**次因（中高置信度）：多任务头权重失衡放大了数值不稳定。**
`moves_left` 是 301 类分类头，权重 0.05，但在 AMP 恢复后从 3.43 单调爆炸到 285.85，
同时 `opponent_reply` 从 3.24 涨到 5.92——两个辅助头都在发散，而主 policy 头几乎不动
（3.32 → 3.29）。这表明跨精度训练中辅助头的梯度尺度没有被约束。

**次要观察（需进一步验证）：`grad_norm` 截断发生在 `scaler.unscale_()` 之后、
`scaler.step()` 之前，因此对 `inf/nan` 梯度调用 `clip_grad_norm_` 会把 `grad_norm` 变成
`nan`/`inf`。g0 记录的 `grad_norm=56.318` 是有效值，而跳过代的 grad_norm 未被记录，
所以无法从产物侧确认。**

**缺失证据（重要）**：`metrics.jsonl` 的 learner 记录**没有记录 AMP scale 值**。
PLAN 文档声称 scale 降到 `3.0517578125e-05`（= 2^-15），但产物中无法复核；
也无法区分“scale 触底后误判”与“每个 batch 溢出后 `break`”这两种机制。
建议在 `LearnerMetrics` 中增加 `amp_scale` 与 `amp_skipped_steps` 字段。

### 1.5 与 FP32 对照组的差异（关键反证）

FP32 重启 run `bal5_r1_column_no_tail_cold_fp32lr1e4_seed271828`（本地归档
`training/runs/stage2/archive/bal5/r1/cold/...`）在完全相同的数据配方下：

| 指标 | AMP canary | FP32 对照 |
|---|---|---|
| g0 `steps` | 1 | 103 |
| g0–g27 平均 steps | ~30（g0 + g24–g27） | 68–103 |
| `train_data_ratio` | 0.0389 → 0.0017 | 恒为 4.0000 |
| validation `policy_loss` g0 → 末代 | 3.2384 → 3.2800（上升） | 3.2104 → 2.3490（下降 27%） |
| validation `wdl_accuracy` g0 → 末代 | 0.4859 → 0.5182 | 0.5352 → 0.7310 |
| `moves_left_loss` | 3.92 → 285.85 | 3.92 → 2.19 |
| self-play policy entropy | 3.0325 → 2.8739（−5%，且 g0–g36 完全不动） | 3.0325 → 1.3158（−57%，持续下降） |

**结论：AMP 是 F-1 的直接原因，不是架构问题。** 这也正是 PLAN 把该 run 明确标注为
“不计作架构淘汰证据”的技术依据。

---

## 2. F-2：`v2.2_large` 主训练（Legacy 栈）

### 2.1 Run 信息

| 项 | 值 |
|---|---|
| 日志 | `D:\四字棋3D\Connect4_3D_AI_v2.2\training\checkpoints\train_info.log`（234,253 B，1490 行） |
| checkpoint | `training\checkpoints\checkpoint_300\checkpoint.pth.tar`（1,080,678,995 B）、`model.pth`（117,602,878 B）、`self_play_samples.json`（13,232 B） |
| 权重产物 | `save_model\v2.2_large\model.pth`（117,601,351 B） |
| 时间跨度 | `2026-03-17 13:36:11` → `2026-03-27 00:48:02`（约 9.49 天） |
| 迭代 | 计划 300，**达成 300/300**（`Iter 300/300`，L1444，约 03-26 22:03 结束）；随后目标改为 400 并续训，终结于 **302/400** |
| 日志规模 | **348 个 iteration block / 299 个不同 iteration / 69,600 局 / 9,625,543 新样本** |
| 会话 | **23 个 banner = 2 次 `Start` + 21 次 `Resume`**；其中 **4 个会话跑了 0 个迭代**（L3、L1006、L1015、L1466） |
| 重复执行的迭代 | 21-30 ×3、31-44 ×2、93-96 ×2、97-101 ×3、261-262 ×2、301-302 ×2（`checkpoint_interval=4` + 从最新 checkpoint 恢复） |
| 最终状态 | 日志在 302 后中断，无完成行；未见 `completed`/final report 收尾 |
| **checkpoint 保留策略** | `checkpoint_interval=4`，`max_checkpoints=4`，另有 `best_new`/`best_old`/`best` 轮换（各约 1.08 GB）。**本地只剩 `checkpoint_300/`；迭代 301/302 的 checkpoint 不存在，塌缩后的权重无法检查** |
| 训练目标产物确认 | `checkpoint_300\model.pth` = 143 zip entries、`conv1.weight` 首位、storage 标 `cuda:0`；`save_model\v2.2_large\model.pth` = 145 entries（= 143 + `.format_version` + `.storage_alignment`）、storage 在 `cpu`、139 个 tensor 的 **key 顺序与形状完全一致但 payload 不同（117/139 不同）** ⇒ 同架构、**不同权重**，详见 §2.5（**初稿曾误判为"同权重另存"**）。对照 `v2.2_balence`（47.6 MB / 87 entries）与 `v2.2_fast`（10.2 MB / 73 entries）宽度不同，且与 `distillation/checkpoints/distill_v1.2_balanced|_fast` 写于同日（2026-04-03）⇒ 这两个是**蒸馏学生**，不是本次 run 的产物 |
| 硬件/quota | **部分恢复**：`max_self_play_workers = 128 = cpu_count`、`data_loader_workers = 64 = cpu_count//2` ⇒ 容器可见 **128 vCPU**。**GPU 型号、OS、torch 版本仍无记录** |

模型规模（本地读 `model.pth` 实测）：**139 tensors，29,391,338 参数**；
对照 `v2.2_balence` 为 83 tensors / 11,902,338 参数（2.47×），
`v2.2_fast` 为 10,158,453 B。三个模型 weight key 前缀一致
（`conv1` / `res_blocks.N` / `val_*`），属同族不同宽度。

### 2.2 训练栈与配置

- 入口：Legacy `training/main_train.py` + `training/trainer.py`（**按 AGENTS.md 属 Legacy，不得作为 V3 回退实现**）
- 关键 banner（每次 resume 打印，第 1450 行原文）：

```
train_device=cuda, infer_device=cpu, shared_inference_device=cuda, num_channels=256,
batch_size=512, epochs=4, shared_inference_servers=8, compatible_inference_servers=1
self_play_exploration_strength=1.000
self_play_phase_schedule=opening_stable(<= 9, temp=0.500, alpha=0.500, eps=0.006)
  | early_midgame_probe(<= 30, temp=1.000, alpha=0.240, eps=0.060)
  | mid_lategame_greedy(<= 55, temp=0.500, alpha=0.500, eps=0.005)
  | lategame_greedy(rest, temp=0.000, alpha=0.000, eps=0.000)
exploration_iteration_schedule=1-60(1.100/1.100) | 61-120(1.000/1.000)
  | 121-180(0.900/0.900) | 181-240(0.850/0.850) | >=241(0.800/0.800)
teacher_bootstrap_games=0, teacher_warmup_iters=0, teacher_replay_ratio=0.000
auxiliary_model=v2.1_high_teacher @ /root/autodl-tmp/AI_v2.2 (2)/save_model (old v2.1)/High/best.pth.tar
```

- 自对弈：200 games/iteration，`mcts_sims=1024`（常量），**`self_play_workers=64`**，
  **`num_mcts_threads=8`**（= 64×8 = 512 线程），**`virtual_loss=1.2`**，
  `inference_batch_size=128`，`inference_timeout_s=0.001`，`shared_inference_server_count` 6→10→8（末期 8）
  —— **以上四项取自 `checkpoint_300` 内嵌 `args`，与今日磁盘文件不同（见 §2.2d）**
- 优化器：`Adam(lr=5e-4, wd=1e-4)`，`StepLR(step=40, gamma=0.8)`，floor `1e-4`，`batch_size=512`，
  4 epochs/iter，AMP + `grad_clip=5.0`
- 学习率实测单调正确：`5e-4`(1-40) → `4e-4`(41) → `3.2e-4`(81) → `2.56e-4`(121) → `2.05e-4`(161)
  → `1.64e-4`(201) → `1.31e-4`(241) → `1.05e-4`(281+)。iter 300 与 301 **同为 `1.05e-4`**
- 训练数据：`history_len=20`，`latest_data_weight=1.0`，`source=self_play_only`，
  `teacher_replay_ratio=0.0`（全程），teacher（`v2.1_high_teacher`）**仅用于 eval**
- **短局处理（关键）**：`min_game_steps=1`（`main_train.py:105-106`，注释原文
  “过滤器保持关闭（阈值=1），短局处理交由样本加权完成”）⇒ 348 个 block 中
  **`filtered=0` 无一例外**；`short_game_weight=0.9` / `short_game_step_threshold=10`
  （`parallel_games.py:1011-1012, 1048-1049`）⇒ 一局 7 步的对局仍保留约 90% 样本
- **战术短路（关键）**：`tactical_override_max_step=15`，但
  `tactical_override_prefer_win=False`、`tactical_override_prefer_block=False`
  （`main_train.py:59-61`），而 `_get_tactical_override_policy`
  （`parallel_games.py:205-220`）的两个分支分别以这两个 flag 为条件 ⇒ **该函数恒返回 `None`**，
  即“立即赢/必须防”的短路**从未生效**。`self_play_tactical_max_step` 未设置（=0），
  开局失误纠正样本也从不生成
- 探索末期已衰减到极低：开局温度 `0.5 × 0.8 = 0.4`，Dirichlet `ε = 0.006 × 0.8 = 0.0048`，
  ply ≥ 56 完全 greedy；损失中**没有 entropy bonus**（`trainer.py` 为 MSE value + CE policy）
- 注意：`self_play_phase_schedule` 在会话历史中改过 8 次。banner 变化点：
  03-18 20:14（开局温度 0.9→0.8）、03-18 21:36（→0.7）、03-19 10:52（相切为 ≤9/≤25/≤45）、
  03-19 16:10（servers 6→10）、03-19 20:12（→8）、03-20 13:07（新增 181-240 迭代带宽）、
  03-21 01:25、03-21 20:26、**03-24 00:38（`≤30`/`≤55`，banner 上最后一次可见改动）**。
  **注意：banner 逐字节相同并不等于配置相同——见 §2.2c。**

### 2.2b 跨 lineage 续训（重要，此前未被发现）

主日志**没有** iterations 214-216（`Iter 213/300` → `Iter 217/300` 直接跳号）。
这 3 个迭代由 `training/main_teacher_failure_rl.py` 在
`training/checkpoints_teacher_failure_rl/` 中执行（见 §3.4），它从主 run 的
`checkpoints/checkpoint_213/model.pth` 引导启动（`main_teacher_failure_rl.py:837-860,
934-935, 1090-1096`），受训于 **40% teacher-failure 混合配方**。

主 run 随后**从该兄弟目录的 `checkpoint_216` 恢复**，证据：

- `Teacher-History: replay_pool_samples=34744` 首次出现在 L1035（iter 217）并一直持续到 L1490；
  这与 teacher 日志中 iter 216 的 `history_raw=34744` 对上，而该兄弟 run 的 iter 217 是 46064；
- `no_improve` 计数在 L1040 从 8（L1001）重新变为 2。

**净效果：iterations 217-302 的权重、replay 与 teacher-failure 池都继承自一个
40% teacher 混合的配方，而该池此后对训练完全无贡献**——
从 L1034 起每一行都是 `teacher_replay_samples=0 | teacher_replay_ratio=0.000`。
另有早停事件：iter 213 在连续 8 次失败 eval 后触发早停（L1001-1004），
而当时脚本里打印的 patience 是 **8**，今日 `main_train.py:118` 是 **12**
⇒ **训练期间 `main_train.py` 确实被改动过，且 banner 不打印这些参数。**

### 2.2c 恢复出的 `history_len`：塌缩边界处确实有一处 banner 看不见的配置变更

**这是本记录对 F-2 的方法论核心发现，它推翻了"banner 相同 ⇒ 配置相同"的假设。**

`compose_training_data` 把训练集构造为 `train_examples_history` 的最后 `history_len` 项
（`trainer.py:222-227` 的 `history[-limit:]`；teacher replay = 0）。
因此迭代 T 生效的历史条目数就是那个精确窗口 k，满足
`Σ new_samples[T-k+1..T] == total_samples[T]`。

本记录独立重算了 348 个 iteration block 中所有满足**精确相等**的 k：

| 迭代区间 | 精确匹配的 k | ⇒ 生效 `history_len` |
|---|---|---|
| 30–98、102–208（176 个） | 10 | **10** |
| 209 / 210 / 211 | 11 / 12 / 13 | 上限在 03-21 20:26 会话被抬高 |
| 212–213 | 14 | ≥14 |
| **234–262、278–300（52 个）** | **18** | **18** |
| **301** | **19** | 恢复出来的历史有 18 项 |
| **302** | **20** | **20** |
| 1–30、21–44、93–101、97–101、217–233、261–277、301–302 | 无精确匹配 | 重执行/跨 lineage 恢复窗口，算术不可用 |

**含义：**

1. `history_len` 在整条 run 中至少取过 **10 → ≥14 → 18 → 20** 四个值。
2. **磁盘上的 `main_train.py:102` 写的是 `history_len = 20`，它匹配的是 301-302 的续训段，
   而不是 281-300 段（18）。** 即当前工作树的配置文件反映的是**塌缩后**的状态。
3. 因此 **03-26 23:26 那次 resume 至少改变了一个 banner 不打印的参数**。
   banner 只有 7 行，**从不打印** `history_len`、`teacher_opponent_history_len`、
   `latest_data_weight`、`min_game_steps(_start_iteration)`、`tactical_override_*`、
   `self_play_tactical_max_step`、损失权重、各头 LR 尺度、`weight_decay`、`cpuct`、
   `virtual_loss`、`inference_batch_size`、`inference_timeout_s`、`num_mcts_threads`、
   `self_play_workers`、`eval_games` —— 其中**任何一个都可能同时被改动**。
4. 训练期存在代码改动的其他独立证据：早停信息写"连续 **8** 次评估"（L1004），
   而 `main_train.py:118` 是 `no_improve_eval_patience = 12`；目标 300→400；
   eval 局数在 iter ~167 从 40 改成 30；日志格式在 iter 21 增加
   `long_games`/`short_games`、iter ~167 增加 `Eval-Best-Generations`。
5. **2026-03-26 之前的 `main_train.py` 版本没有任何副本留存。**
   2026-04-17 的 git blob `d7ee2a11` 与今天的工作树文件**逐字节相同**
   （`history_len=20`、patience=12）。**塌缩前的配置只能部分重建。**

**注意 k=18 → 20 本身不足以解释 22 步变 7 步**——
把历史窗口从 18 加到 20 不会直接让局长塌掉。最稳妥的读法是：
**权重/吸引子 tipping 确实发生了，同时边界处存在一个量化不了的配置增量**，
两者叠加，而产物无法区分各自贡献。这就是为什么 §7 把它列为未决项。

### 2.2d `checkpoint_300` 内嵌配置已提取：**磁盘上的代码不是跑 1-300 的那份代码**

子代理用流式 unpickle（不需要 torch；`data.pkl` = 727,977,332 B，峰值 RSS 82 MiB）
成功解出 `checkpoint_300\checkpoint.pth.tar` 的 `args` dict。
**笔者用独立的 `pickletools.genops` 流式扫描复算，逐项吻合：**

| 参数 | 迭代 300 的实际值 | 今日磁盘 `main_train.py` |
|---|---|---|
| `iteration` | **300** | — |
| `history_len` | **18** | **20**（`main_train.py:102`） |
| `num_iterations` | **300** | **400**（`main_train.py:28`） |
| `num_mcts_threads` | **8** | **24**（`main_train.py:55`） |
| `virtual_loss` | **1.2** | **1.5**（`main_train.py:58`） |
| `self_play_workers` | **64** | **48**（`main_train.py:50`） |
| `max_self_play_workers` | **128**（= `cpu_count`） | `cpu_count` |
| `data_loader_workers` | **64**（= `cpu_count//2`） | `max(2, cpu_count//2)` |
| `no_improve_eval_patience` | 12 | 12——但日志在 iter 213 打印的是 **8**（L1004） |
| 其余不变项 | `num_self_play_games=200`、`num_channels=256`、`num_mcts_sims=1024`、`cpuct=1.0`、`batch=512`、`epochs=4`、`lr=5e-4`、`wd=1e-4`、`min_game_steps=1`、`teacher_opponent_history_len=18`、`teacher_replay_ratio=0.0`、**`tactical_override_prefer_win=False`、`tactical_override_prefer_block=False`** | 相同（笔者已验证这四个关键值） |

**五条结论：**

1. **§2.2c 的窗口反解被一个完全独立的方法确认**：checkpoint 里 `history_len = 18`，
   与算术反解在 234-300 段得到的 18 一致；`train_examples_history` 的前三项长度为
   **42,181 / 43,256 / 41,296**，正是日志中 **iter 283/284/285** 的 `new_samples`
   ⇒ 迭代 300 时生效的缓冲区精确等于 **iterations 283-300（18 项）**。
2. **"banner 逐字节相同 ⇒ 配置相同"彻底作废**：迭代 300 的实际配置与今日磁盘文件在
   `history_len`、`num_mcts_threads`、`virtual_loss`、`self_play_workers`、`num_iterations`
   五项上都不同，eval patience 也不同（iter 213 时是 8，存档状态是 12）。
   **磁盘上的 `main_train.py` 是"续训版"修订，而不是跑 iterations 1-300 的那一版；
   2026-03-26 之前的任何修订都没有留存。**
3. **硬件部分恢复**：`max_self_play_workers = 128 = cpu_count`、
   `data_loader_workers = 64 = cpu_count//2` ⇒ 该容器在迭代 300 时可见 **128 vCPU**；
   拓扑为 **64 self-play workers × 8 MCTS threads = 512 线程** + 8 个 CUDA 推理服务。
   **GPU 型号、OS、torch 版本仍然无记录。**
4. **gate 在结构上不可能抓到这次塌缩**：存档的 `next_eval_iteration = 303`，
   而续训从 301 开始、302 之后即死。
   ⇒ **第一次塌缩后的评估原定在迭代 303，从未执行。**
5. **存档的 best 状态**：`best_win_rate = 0.6333`、`best_model_iteration = 295`、
   `older_best = 283 @ 0.6`、`consecutive_no_improve_evals = 1`、`stop_reason = None`。
   scheduler 状态（step_size 40 / gamma 0.8 / `last_epoch` 299 / `_last_lr` 1.048576e-4）
   与日志一致。

### 2.2e 旧栈自己的验收标准：这条 run 从约第 6 个迭代起就在违反

> **作者与适用范围说明（据仓库作者本人说明）**：
> `check_info_guides.md` 与 `TRAINING_RUNBOOK.md` 都是**仓库作者（人类）**撰写的；
> 文中所有"检查者/inspector"指的是 **AI agent**，不是人类。
> 这两份文档写在 **V3 蒸馏线实施之前**，当时 `v2.2_large` 与 `v2.2_balence` 已经训练完成
> —— 也就是说，**这些规则是从旧训练栈的实践中提炼出来的事后经验**。
> **它们不自动适用于新的 V3 训练栈**（见本节末尾的"开局温度"案例）。
> 本记录引用它们，只是为了说明"这条 run 违反了它自己所处时代的标准"，
> **不能当作 V3 的验收依据**。

以下规则**精确描述了本次失败**（原文引用）：

- `distillation\save_model\gravity_balanced\iter_0260\check_info_guides.md:10`：
  > 只要对弈双方在怀疑失误的情况下，游戏对局应该在十手以上……**稳定后自动对每 100 局中不超过 2 局**
  > ……**连续多次迭代中每 100 局有 10 局 short game**，就说明**模型陷入了局部最优，丧失了对抗的基本能力**。

  本 run 实测：**每 200 局有 100-150 局短局，持续约 200 个迭代；末段 195-198/200**
  ⇒ 折算 **975-990 / 1000**，是阈值的约 **100 倍**。
- `check_info_guides.md:13`：iter 40 之后平均局长应在 **18 以上**；
  健康局部最优的 `var_steps` 应在 **70 以上**。本 run 末段 `avg_steps = 7.2`、`var_steps = 0.85`。
- `check_info_guides.md:17`：25 局有效窗口内 **policy entropy 均值应在 0.5 以上**——
  **本 run 的塌缩段熵是 1.08-1.34，完全"合格"** ⇒ 再次说明熵不是可用信号。
- `check_info_guides.md:20`：value loss 低于 0.2 时需要注意（与本 run 的 `value_loss_low` 一致）。
- `distillation\TRAINING_RUNBOOK.md:16-24` 明确列出塌缩指标，并写下关键一句：
  > **Best-vs-best promotion alone is not sufficient evidence of health.
  > A degenerate strategy can be locally strong or intransitive against two saved opponents.**

  **这正是本 run 的 gate 全程保持绿色、而自对弈已经退化的原因。**
- `TRAINING_RUNBOOK.md:9-10`：作者**授权**检查者（AI agent）在明显塌缩时停机、
  回滚到最近健康 checkpoint 并通知用户；且"不要等到下一次计划检查"。
  **这条 run 没有被回滚——它一路跑到 302，日志在 302 后中断。**
- 同一 runbook 与 `TRAINING_HISTORY.md:35` 记录了**同一吸引子的更早两次实例**
  （iter 64→79 平均 16.95→13.94、value loss 0.011；以及"末轮平均 9.91 手、≤12 手 208/240"）
  ⇒ **最小长度退化是这条旧自对弈循环的已知复发失效模式，不是一次性事故。**

**项目自身对"开局多样性"的后续测量也直接印证了机制**
（`experiments\v22_balance_opening_diversity_report.md`，2026-07-20）：
在无扰动下 `T=0.5` 与 `T=0.75` 时，256 次 rollout **收敛到恰好 1 个**有效的 9 手开局
（`T=1.0` 也只有 2 个；只有加 Dirichlet 噪声才恢复到 53-86 个）。
**本 run 的开局相已衰减到温度 0.4 + ε≈0.0048，正落在"无扰动"区间**
——这解释了为什么采样的 5 局全部以 action 12 开局且走同一条线。
同一报告对 310 个不同 9 手开局的战术复盘发现 **206 个含可避免的战术错误**
（194 个 `allows_live_two_upgrade`、8 个 `allows_immediate_win`、4 个 `missed own immediate win`），
其中 **`allows_live_two_upgrade`（送出活三）正是本 run 7 步败局的机制**。

**⚠️ 这批规则的适用范围是有限的：开局温度就是被 V3 明确否定的那一类。**

作者在设计旧栈时，为了让模型的开局更稳定（观察到自己在人类对抗中开局相对固定，
希望自对弈产生更多后期数据），在阶段计划中**明确规定开局温度降为探索期温度的
0.8–0.6 倍**——本 run 正是 `opening_stable(≤9, temp=0.500)` 配合迭代尺度 `0.80`，
即实际 **0.40**。

**这个判断在 V3 中被否定了。** V3 改为**前 28 步全程保持 `temperature = 1.0` 的高探索**
（见 AGENTS.md 与 `training/v3/configs/stage1_*.json` 的 `dynamic_exploration`），
结果是：

- 在 V3 下**训练量不到 1M 时，新模型就已经远超旧三代模型（v1 Elo > 100）**；
- V3 的 **B8 与 B10 模型给出的开局解与作者（人类）的判断完全不同**。

⇒ 这说明**作者在旧栈开局策略上的认知本身可能存在缺陷**，
而 `check_info_guides.md` / `TRAINING_RUNBOOK.md` 里记录的因素
**是旧栈条件下的观察，只能作为参考，不能直接搬到 V3**。
本记录据此在 §6 与 §9 中把所有引用这批阈值的地方都标注为"旧栈经验规则"。

**最后：旧栈里没有任何文件记录本次塌缩。** 没有针对 `v2.2_large` 的复盘；
根目录 `迭代性能记录.xlsx` 止于迭代 159（2026-03-20），**比塌缩早 140 个迭代**。
本次失败**唯一的书面线索就是 §2.2f 那条由作者事后确认的 virtual loss 缺陷**。

### 2.2f 迭代 300 之后的独立实验：**Legacy 栈 virtual loss 的负号写反**

> **来源：仓库作者（人类）本人在本次整理中的直接说明。**
> 本节内容是本记录中**唯一来自人工确认而非产物推断**的关键事实，
> 也是 §7 原先列为"不可判定"的那个洞的答案。

**实验设计**：在 `Iter 300/300` 完成之后，作者做了一次**独立实验**，
目的是验证旧训练栈中 **virtual loss 的并行线程数（`num_mcts_threads`）对训练的影响**。
旧栈此前使用的是 **4 个线程**，实验中逐步上调到更高的值。

**观察结果**：

- 在**原有实现**下，线程数上调后训练**快速崩坏**；
- 作者随即暂停了这个实验；
- 事后检查发现：**原实现中 virtual loss 的符号写反了**。

**这解释了 §2.2c / §2.2d 中那处"banner 看不见的配置变更"**：
迭代 300 存档里的 `num_mcts_threads = 8`、`virtual_loss = 1.2`，
而**磁盘上今日的 `main_train.py` 是 `num_mcts_threads = 24`、`virtual_loss = 1.5`**
—— 磁盘文件记录的正是**这个实验（以及/或后续实验）上调后的参数**，
而不是跑 iterations 1-300 的那一版。**`Iter 301/302` 的塌缩就发生在这个实验之内。**

**机制解释（与既有证据自洽）**：virtual loss 的作用是让并发搜索的多个线程
**不要同时选中同一个叶子**。其效力与并发度成正比——被错误符号污染的
`value_sum` 会随"同一叶子被并发选中的次数"线性累积。
因此 **4 线程时误差尚可承受，线程数上调后误差随并发度放大并迅速破坏搜索**，
与作者观察到的"上调后快速崩坏"以及"回退到 4 线程可用"完全一致。

**本记录的独立佐证**（代码层面核对，两栈约定相反）：

| 栈 | 位置 | virtual-loss 实现 | 符号方向 |
|---|---|---|---|
| Legacy（本次失败 run） | `training/mcts.py:45-58` | `add_virtual_loss`: `value_sum -= virtual_loss`；`revert`: `value_sum += virtual_loss` | **反向（被确认为写反）** |
| V3（重写实现） | `training/v3/search.py:147-159` | `apply_virtual_loss`: `value_sum += amount`；`revert`: `value_sum -= amount` | 正向 |

两个实现在结构上也是分开的（Legacy 单独维护 `virtual_loss_count`，
V3 同时增减 `visit_count`），**V3 不受此缺陷影响，也无需从 Legacy 移植任何修复**。
按 AGENTS.md 的分线规则，这是"不得用 Legacy 代码修 V3 问题"的一个正面例子：
V3 的搜索是独立重写的，恰好绕开了这个缺陷。

**对结论的影响（修改但不颠覆）**：

- §2.4 的"触发点不可判定"**现在可以收敛**：塌缩发生在一次
  **后加的、以线程数为自变量的 virtual loss 实验**中，
  而该实验的实现在事后被确认有符号缺陷。
- 但这**不推翻**最小长度吸引子那条主线：Legacy 栈的 virtual loss 缺陷
  **不解释**为什么游戏会精确停在 7 步（理论最小值），也不解释防守方为何不封堵；
  它解释的是**为什么线程上调后崩溃会加速**。
  两者是**叠加**关系：一个长期存在的吸引子（机制 1-4）加上一个
  在 iter 301 被实验放大/触发的搜索缺陷。
- 因此 §2.4 的机制排序**保持不变**，只是第 1 条前面加了"被一次有缺陷的
  并发搜索实验触发"这一限定。

### 2.3 具体体现：迭代 300 → 301 的断崖

| iter | 时间 | duration | new_samples | **avg_steps** | **var_steps** | entropy | min/max | **short_games** |
|---|---|---|---|---|---|---|---|---|
| 298 | 03-26 18:32 | 01:04:49 | 44,650 | 26.05 | 100.60 | 0.6855 | 7/… | 1 |
| 299 | 03-26 19:37 | 01:28:21 | 42,471 | 24.62 | 106.88 | 0.6774 | 7/… | 6 |
| 300 | 03-26 21:05 | 56:57 | 37,581 | 21.80 | 120.49 | 0.6315 | 7/59 | 32 |
| — | **会话边界** `Resume training session` @ 23:26:51，**配置逐字未变** | | | | | | | |
| **301** | 03-26 23:26 | 20:27 | **10,664** | **7.26** | **0.85** | 1.1546 | 7/15 | **195** |
| 302 | 03-26 23:47 | 21:02 | 10,753 | 7.29 | 1.72 | 1.2360 | 7/22 | **193** |
| — | `Resume` @ 03-27 00:26:52、@ 00:31:03（配置仍逐字未变） | | | | | | | |
| 301' | 03-27 00:31 | 16:58 | 10,604 | **7.22** | **0.79** | 1.3400 | 7/15 | **195** |
| 302' | 03-27 00:48 | 17:03 | 10,411 | **7.17** | **0.39** | 1.0793 | 7/11 | **198** |

同代 loss 同时**下降**：`total` 1.0895 → 1.0931（基本平），
但 `policy` 从 0.8169 升到 0.8276/0.8378，`value` 0.2726 → 0.2614。
即模型并没有“学得更好”，只是分布变了。

**归一化后的对比更清楚**：new_samples 10,664 / avg 7.26 ≈ **1,469 局等效**，
而 37,581 / 21.80 ≈ 1,724 局等效——每局步数减少 67%，总步数只减少 15%。

### 2.3b 塌缩不是从 301 开始的：退化是两阶段

按 20 迭代分块聚合 `avg_steps` 与 `short_games`（`short_games` 日志从 L226 / iter 21 才开始记录）：

| 迭代区间 | 平均 `avg_steps` | 平均 `short_games`(/200) |
|---|---|---|
| 21–40 | **9.25** | **141** |
| 41–60 | 9.90 | 133 |
| 61–80 | 11.87 | 110 |
| 101–120 | 13.04 | 89 |
| 121–140 | 16.15 | 60 |
| 161–180 | 19.34 | 38 |
| 181–200 | 21.17 | 30 |
| 221–240 | 23.37 | 5 |
| 241–260 | 25.08 | 2 |
| 261–280 | 24.74 | 4.5 |
| 281–300 | 24.61 | 10 |
| **301–302** | **7.23** | **195** |

**第一阶段（iter 21 → ~240）：从 9.25 步的长程爬升。** 前 40 个迭代相对理想值有
**141/200 的短局**，模型花了约 200 个迭代才爬出这个吸引子，在 iter 241-280 达到最好的
`avg_steps ≈ 25`、`short ≈ 2-4.5`。

**第二阶段（iter 281-302）：在恢复区里反复抖回吸引子，然后彻底坠落。**
`short_games` 在恢复区内多次闪烁：iter 286 = 91（L1380）、iter 291 = 43（L1402）、
**iter 300 = 32（而 iter 299 只有 6，已是 5 倍恶化）**，
然后 iter 301 = 195、302 = 193/198。

**因此：iter 300 → 301 是"从吸引子边界掉回去"，不是"新引入的缺陷"。**
但**必须与 §2.2c / §2.2f 一起读**：该次 resume 的 banner 逐字节相同，**却确实改变了
banner 不打印的参数**——`history_len` 18 → 20（§2.2c），
以及**`num_mcts_threads` 与 `virtual_loss` 的上调实验**（§2.2f）。
所以"配置未变"这一推论是**错的**。综合读法是：
iter 300 已站在吸引子边界上（自身短局率已是 iter 299 的 5 倍），
而 iter 301 起叠加了**一次被确认有符号缺陷的并发搜索实验**
（virtual loss 负号写反 × 线程数上调），把误差按并发度放大，从而彻底坠落。

### 2.3c 相关性与“熵悖论”

对 294 个迭代做相关（`corr.py`）：

| 变量对 | 相关系数 |
|---|---|
| `corr(short_games, policy_entropy)` | **+0.953** |
| `corr(avg_steps, policy_entropy)` | **−0.856** |
| `corr(avg_steps, value_loss)` | +0.618 |
| `corr(avg_steps, policy_loss)` | −0.689 |

**这不是熵塌缩。** 塌缩期拥有整条 run **最高**的 policy entropy（1.08-1.34，
而健康段是 0.63-0.70）。机制解释：搜索在它到达的那些局面里**无法区分着法**
（`policy_entropy` 高），同时这些局面全是被将死的局面（短局）——
正是“网络在所有着法上都不确定，因为它已经不知道哪一步会输”。
`value_loss` 在此处是**误导性**指标：9 步时代的均值 0.099 远低于 iter 300 的 0.27。

**定量复核（本记录独立完成）**：在 V3 FP32 对照组的 replay shard 上，
“行棋方存在立即制胜着法”的局面占比从 5.2%（gen 0）升到 17.5%（gen 20），
且 gen 20 的这类局面中 **491/811 存在多个制胜点**——与“搜索无法区分着法”一致（见 §5.1）。

### 2.3d 强度结果

- 89 次对自身历史 best 的 eval：41 pass / 48 fail，平均胜率 **0.539**（iter ≥260 时为 0.607）。
- 24 次对 `v2.1_high_teacher` 的 eval：平均胜率 **0.390**，最高 0.675
  ⇒ **训练期间从未超过 v2.1**。
- eval 噪声极大：同一个冻结对手的得分出现过 0.133 与 0.567（L958 vs L944）。
- eval 协议本身无法看见本次失败：30-40 局、greedy、只记录 W/L/D，
  **没有局长、先后手分项或“被杀”指标**；且 iter 301/302 **没有触发任何 eval**。

### 2.3e 决定性反证：导出的权重是健康的，7 步塌缩是**自对弈**现象

塌缩后约 13 小时（2026-03-27 14:38:55）的 arena 对战记录
`arena\history\20260327_143855_v2.1_High_vs_v2.2_1.json`（本地 12,410,811 B，已逐字段核对）：

```json
"config": {"games": 40, "parallel_games": 40, "temperature": 0.1,
           "board_size": 5, "board_layers": 6, "connect_n": 4,
           "immediate_win_check": false, "seed": 42},
"agents.agent2": {"name": "v2.2_1", "model_path": "training/checkpoints/best.pth.tar",
                  "model_config": {"num_channels": 256, "num_res_blocks": 8,
                                   "input_channels": 2, "architecture": "modern",
                                   "board_layers": 6, ...}},
"summary": {"games": 40, "agent1_wins": 18, "agent2_wins": 22,
            "illegal_moves": {"v2.1_High": 0, "v2.2_1": 0},
            "tactical_moves": {"v2.1_High": 0, "v2.2_1": 0}}
```

`agent2` 的 `num_channels=256, num_res_blocks=8, input_channels=2, architecture="modern"`
与本 run 的 `Connect4Net` 完全一致 ⇒ **它就是这个塌缩 run 的 accepted best**。

本记录重新逐局统计该文件的 40 局：

| 指标 | 值 |
|---|---|
| 步数 min / max / mean | **13 / 73 / 32.15** |
| 步数 ≤ 9 的对局 | **0** |
| 步数直方图 | 13,15×4,17,19,21,23×2,24,25×2,27×3,29×4,30,31,33,34×2,35×3,37,39,41×2,43,44,45,46,48,52,63,73 |
| 先手胜 / 后手胜 | **32 / 8（80% 先手胜率）** |
| 非法着法 | 0 |

**结论（高置信度）：**

1. **导出的权重没有损坏。** 同一份权重对外部对手 v2.1_High 取得 22-18（55%），
   对局长度 13-73 步，**没有任何一局 ≤ 9 步**。
   ⇒ 7 步短局**不是权重缺陷，也不是导出缺陷**。
2. **7 步塌缩只发生在自对弈**（同一网络执双方）。这也解释了为什么
   "防守方不封堵"：一份对自身弱点完全盲的副本，正好提供了
   "无法封堵的防守方"。
3. **本游戏有实测约 80% 的先手优势。** 这是最小长度先手胜吸引子的结构性输入：
   一旦 P1 找到一条快速成线且 P2 不封堵，P1 胜是自洽的。
4. **蒸馏学生没有塌缩。** 三个蒸馏 run 的自对弈 `avg_steps` 分别维持在
   10.5-24（v1.1）、14.3-29.3（v1.2_balanced）、12.4-21.2（v1.2_fast），
   只有个别对局的最短值为 7 步
   ⇒ **塌缩不是该架构族或蒸馏配方的固有属性**。
5. **训练期间的 `Eval-Teacher = 0.390` 不等于最终强度。** 它在 `temperature=0`、40 局、
   对**中间 checkpoint** 条件下测得；arena 用 55% 的结果说明该指标严重低估。

唯一的另一份含 7 步局的 arena 记录是
`arena\history\20260404_001735_v2.2_fast_vs_checkpoints.json`（20 局，步数 7/14/7/14 交替，
min 7 / max 18 / avg 10.7，`v2.2_fast` 20-0 胜一个很小的 feature policy）——
对手越弱，"不封堵的防守方"越容易出现，与该机制一致。

### 2.4 失败原因分析（已按交叉验证 + 作者说明修订）

**触发点（高置信度，已三次修订，现可收敛）**：塌缩发生在 `Iter 300/300` → `Iter 301/400` 之间，
且**同一吸引子在 iter 21-40 就已经出现过**（`avg_steps` 9.25、短局 141/200），
所以这不是"续训引入了新缺陷"，而是**一个始终存在的短局吸引子在恢复区边缘被重新捕获**。

**边界处确实有配置变更，且现在知道是什么（高置信度）**：
`history_len` 18 → 20（§2.2c），以及**作者在迭代 300 之后启动的
"virtual loss / MCTS 线程数"独立实验**（§2.2f）——该实验把线程数从 4 上调，
而**事后检查确认原实现的 virtual loss 负号写反**，导致上调后训练快速崩坏，
作者随即暂停了实验。磁盘上今日的 `main_train.py`（`num_mcts_threads=24`、
`virtual_loss=1.5`）记录的正是这个实验上调后的参数，
而迭代 300 存档中是 `8 / 1.2`。
⇒ **"banner 逐字节相同"不能作为"配置未变"的证据，这一点已由作者说明直接确认。**

**机制（高置信度，按"长期吸引子 + 实验触发"两层组织）**：

**A 层——长期存在的结构性缺陷（解释"为什么会退化"，iter 21 起就有效）：**

1. **自对弈没有任何防守安全网。** `tactical_override_prefer_win=False` 与
   `tactical_override_prefer_block=False`（`main_train.py:59-61`）使
   `_get_tactical_override_policy` **恒返回 `None`**（`parallel_games.py:205-220`），
   即"存在立即赢 / 必须封堵"的短路从未生效；`self_play_tactical_max_step` 未设置，
   开局失误纠正样本也从不生成。**V3 的 FP32 对照组在同一位置漏防，是同一个结构性洞**
   （见 §5.1：P2 从不封堵 row 2，41 局 7 步败局）。
2. **短局样本不被过滤、几乎不降权。** `min_game_steps=1` ⇒ 348 个 block 全部
   `filtered=0`；`short_game_weight=0.9` ⇒ 一局 7 步的对局仍贡献约 90% 样本。
   于是**策略最差的时候，新鲜数据从 42k 掉到 10.6k**（iter 301）——
   数据量与数据质量同时崩塌，形成正反馈。
3. **探索已经衰减到几乎为零。** 开局温度 `0.5×0.8 = 0.4`，Dirichlet `ε = 0.0048`，
   ply ≥56 完全 greedy，且损失中**没有 entropy bonus**。
   网络一旦进入确定性短局线，没有任何机制把它推出来。
   （注：这条开局温度策略**后来被 V3 否定**，见 §2.2e。）

**B 层——实验触发（解释"为什么恰好在 iter 301 崩"）：**

0. **Legacy 栈 virtual loss 的负号写反 × 线程数上调**（§2.2f，作者事后确认）。
   该缺陷的误差随并发度线性累积，因此 4 线程可用而上调后快速崩坏——
   这与"塌缩精确发生在 `Iter 300` 之后的线程实验里"在时间上完全吻合。
   **但它不解释为什么游戏精确停在 7 步、也不解释防守方为何不封堵**，
   所以它是**触发/放大器**，不是吸引子的成因。两层是叠加关系。
4. **逐局记录的佐证**：`self_play_samples.json`（iter 300，5 局，rank 1-197，**全部 winner=1**）
   显示 rank1 为 7 步，着法序列 `1:12, 2:6, 3:13, 4:7, 5:11, 6:14, 7:10`。
   解码（`action = layer*25 + row*5 + col`）：P1 走
   `(0,2,2) → (0,2,3) → (0,2,1) → (0,2,0)`（底排 row 2 的 col 0-3，ply 7 成四）；
   P2 走 `(0,1,1) → (0,1,2) → (0,2,4)`——**从未争夺 row 2**，
   在 ply 5 送给对手一个**两端皆空的活三**，ply 6 的“封堵”只是选择了 P1 从哪一端赢。
   rank 2-5 为 15/21/29/59 步，全部 winner=1，且都以 action 12 开局 ⇒ **确定性开局**。

**已排除的假设（经代码核对，均不支持）**：

- **规则/parity bug**：项目自带的精确求解探针 `.tmp\early_win_probe.json`
  （同一开局 P1(0,2,2)、P2(0,1,1)，问“总步数 12 前是否必有一方强制获胜”）
  返回 `force_exists=false, exact=true, visited_nodes=458230`（P1）与 `false`（P2）
  ⇒ **正确的防守方可以活过 ply 12**。本记录另用独立 alpha-beta
  （`tmp/solve_3d_connect4.py`，411 条连线，D4 对称归约）确认空盘深度 7 = 和棋。
  两条独立证据一致：**7 步败局是网络自己的防守失误，不是规则决定的**。
- **权重/导出损坏**（**新增排除项，见 §2.3e**）：塌缩后 13 小时的 arena 对战显示，
  同一份 `best.pth.tar` 对外部对手打了 40 局、13-73 步、**0 局 ≤ 9 步**、22-18 取胜。
  ⇒ 7 步病理只存在于自对弈，权重本身健康。
- **value/reward 符号错误**：目标公式代数正确，MCTS terminal + negamax 自洽。
- **学习率 bug**：实测单调，iter 300 与 301 同为 `1.05e-4`。
- **`.eval()`/dropout 泄漏进自对弈**：推理服务调用 `model.eval()` + `inference_mode`。
- **batch size 效应**。
- **replay 陈旧作为触发器**：塌缩 2 个迭代后退化数据只占约 2.5%
  ⇒ replay 是**维持者**，不是**起因**。

**仍不确定（内容已随作者说明收窄）**：

- **触发机制已由作者确认（§2.2f），但"量化贡献"仍不可分。**
  已知边界处的改动至少包括 `history_len` 18 → 20 与
  `num_mcts_threads` / `virtual_loss` 的线程实验，且该实验的 virtual loss 负号写反。
  **无法从产物区分**：吸引子自身漂移、iter 300 的样本复用、
  与 virtual loss 缺陷各自的贡献比例。
- **Legacy 栈的 virtual loss 缺陷是否还影响 300 之前的迭代，无法回溯。**
  迭代 300 存档是 `num_mcts_threads=8`；旧基线是 4。
  中间的每次 raise 都没有留下记录，因此**无法确定缺陷从哪个迭代起开始显著**，
  也无法排除它参与了 iter 21-40 那次早期退化。
- **2026-03-26 之前的 `main_train.py` 副本不存在。**
  2026-04-17 的 git blob `d7ee2a11` 与今日工作树**逐字节相同**，
  ⇒ **塌缩前的配置只能靠 `checkpoint_300` 的内嵌 `args` 部分重建**（§2.2d）。
  该目录最早的 `training/` commit 是
  `97cfae6 "Initial code-only import"`（2026-04-17），比 run 晚三周。
  目录 mtime 显示是一次性批量拷贝
  （10 个 arena 文件集中在 00:40:49-00:41:35，15 个 training/*.py 在 00:42:02-00:42:21），
  **不能用来判断“训练期间是否被编辑”**（但 §2.2c 第 4 条已用日志算术独立证明编辑发生过）。
- **过渡瞬间的因果**：无法判定是"权重自身漂移到边界"、"iter 300 的 3.07M 样本复用"，
  还是"§2.2f 那次 virtual loss 线程实验"——**因为迭代 301/302 的 checkpoint 不存在**。
  **已部分解决**：`checkpoint_300/checkpoint.pth.tar` 内嵌的 `args` dict 已成功提取
  （§2.2d），`train_examples_history` **确实恰好 18 项**，
  前三项长度 42,181 / 43,256 / 41,296 精确对应 iter 283/284/285
  ⇒ **§2.2c 的窗口反解得到确认**。
  仍未解决的是：**该实验的缺陷在 301/302 两代里具体造成了什么，
  以及它是否也污染了更早的迭代**（无中间记录可查）。

### 2.5 产物归属（已用 SHA-256 定案，含一处更正）

> **更正（重要）**：本记录初稿曾写"`v2.2_large/model.pth` 与 `checkpoint_300/model.pth`
> 是同权重另存"。**这是错的**，笔者已独立重测推翻。
> 初稿只比对了 **storage 的长度多重集**与 zip 元数据差值，**没有比对 payload 内容**。
> 直接对 139 个 tensor entry 逐个算 SHA-256：**117 个 payload 不同，仅 22 个相同**
> （22 个相同的是 BatchNorm 的常量 0/1 张量）。例如 entry `0`（`conv1.weight`，两文件均 55,296 B）
> 在 `checkpoint_300/model.pth` 中是 `sha256 76a6582e…`，在 `v2.2_large/model.pth` 中是 `652c772a…`。
> **结论：同架构，不同权重。**

| 产物 | 大小 | 身份 | 证据 |
|---|---|---|---|
| `save_model\v2.2_large\model.pth` | 117,601,351 B | **accepted-best 的导出（≈迭代 295）**，**不是**塌缩的迭代 300 状态 | 与 `checkpoint_300\model.pth` 同为 139 keys/shapes，但 **117/139 payload 不同**；`checkpoint_300` 内嵌 `best_model_iteration = 295`（`best_win_rate = 0.6333`）；导出命令为 `tools\export_model_pth.py save_model\v2.2_large\best.pth.tar`（`DEFAULT_SOURCE = best.pth.tar`） |
| `save_model\v2.2_balence\model.pth` | 47,629,231 B | **蒸馏学生**（224 ch × 4 blocks，`architecture="modern"`） | 与 `distillation\checkpoints\distill_v1.2_balanced\checkpoint_188\model.pth` **逐字节相同**（sha256 `29D87E1D…`） |
| `save_model\v2.2_fast\model.pth` | 10,158,453 B | **蒸馏学生**（96 ch × 3 blocks） | 与 `distillation\checkpoints\distill_v1.2_fast\checkpoint_48\model.pth` **逐字节相同**（sha256 `E006BB66…`） |

**Lineage 已独立确认**（两个 balanced 蒸馏 checkpoint 内嵌的 `teacher_cache_metadata`
指向其 teacher 路径）：`v2.2_large (256×8, iter 295)` → `v2.2_balence (224×4, iter 188)`
→ `v2.2_fast (96×3, iter 48)`。`check_info_guides.md:3` 也自述该文档是"根据本地上传训练的
v2.2 的 3 个模型"撰写。

三个产物旁**都没有 README 或元数据文件**。另有**两处配置/产物不一致**值得记录：

1. `distillation\distill_config.json` 声明 `fast = 128 ch / 25 iterations / 150 games / window 20`，
   但实际 run（`distill_config.py:38-41` + 权重哈希）是 **96 × 3 / 200 iterations / 128 games / window 12**
   ⇒ **该 JSON preset 不是实际生效值**，用它反推 lineage 会得到错误结论。
2. 教师模型 `save_model (old v2.1)\High\best.pth.tar` 是 **128 ch / 4 blocks / 1 输入通道 /
   200 动作**（5.25 M 参数，`res1..res4`）的 legacy 模型，即 **8 层棋盘**；
   `game_rules.py` 里"action index 0..199"的注释与它一致，
   **与 6 层 / 150 动作的 v2.2 线不同**——跨代评估时需注意棋盘几何差异。

---

## 3. F-3：`gravity_balanced` / `cubesprite_v3_mini` 蒸馏线（26 条 run）

> 本节的完整逐 run 清单、逐类超参与引用行由子代理从
> `D:\四字棋3D\Connect4_3D_AI_v2.2\distillation\save_model\gravity_balanced\iter_0260\`
> 与 `D:\四字棋3D\Connect4_3D_AI_v2.2\experiments\model_architecture\` 逐文件核对得出。

### 3.1 训练栈

- 入口：`python distillation/main_distill.py --config <cfg>.py`（`mp.set_start_method("spawn", force=True)`）
- 配置链（逐级 `importlib` 继承，设置累积）：
  `midterm_gravity_balanced_rerun_alpha.py` → `_guarded` → `_recovery152` →
  `cubesprite_v3_mini_league_rl.py` → `_teacher_recovery` → `_extend220` → `_opening_rl` →
  `_opening_rl_extend238` / `_tactical_opening_rl` → `_extend250` → `_teacher1_extend252` →
  `_alphazero_settle272` → `_tactical_selfplay260`
- 架构 preset `gravity_balanced` = `num_channels=128, num_res_blocks=6, backbone_type="layer2d",
  global_context_blocks=1` → `GravityPolicyValueNet`（`training/experimental_models.py`）
- 精度：配置中 `inference_precision="fp32"`；早期 run preflight 另打印 `train_precision=bf16`；
  2026-06-27 之后 preflight 不再打印 `train_precision`（**有效训练 dtype 无法确认，见 §7**）
- 优化器/批：`AdamW`，`batch_size=1024`，`epochs=4`（cubesprite 段为 2）
- 学习率（继承默认）：`5e-4`(1–4) → `2e-4`(5–20) → `1.5e-4`(21–60) → `1.2e-4`(61–100)
  → `1.0e-4`(101–160) → `8e-5`(161+)；opening/tactical 分支固定 `1.0e-4`
- Teacher：`save_model/v2.2_balence/model.pth`（47,629,231 B）；
  cache 由 `teacher_examples.pth.tar` 换为 `teacher_examples_2000_diverse.pth.tar`（**本地缺失**）
- 搜索：`inference_batch_size=64`，`inference_timeout_s=0.003`，`cpuct=1.0`，`virtual_loss=1.0`，
  `reuse_mcts_tree=True`，`shared_inference_server_count=1`

### 3.2 失败分类

| 类 | 描述 | run 数 | 关键证据 |
|---|---|---|---|
| **F1** | 迭代中途静默挂起后被 kill（watchdog/手动/截断） | **7** | 最后一行是 `Iter N: x/M` 进度条，该迭代的 `Iter N/M \| duration=…` 汇总行从未出现；文件 mtime 比最后进度晚 9 分–47 分；**全库检索 `Traceback|MemoryError|CUDA out of memory|RuntimeError|ValueError|KeyError|SIGKILL|SIGTERM|Killed|nan|BrokenPipe|Exception` 在 6/7 个文件中零命中** |
| **F2** | 锚点 checkpoint 与 `model_compat` 不兼容 | 1 | `File ".../training/model_compat.py", line 139, in infer_model_config` / `action_dim = int(state_dict["prob_fc.bias"].shape[0])` / `KeyError: 'prob_fc.bias'`，发生在 `DistillationTrainer.__init__`，训练开始前 |
| **F3** | 共享推理服务死亡 → 24 个 worker 同时 exit 1 | 1 | `RuntimeError: Distill Anchor Adversarial Iter 181 worker failed with exit codes: [1, 1, ... ×24]`；`mcts.py:328 _recv_loop -> connection.py:399 _recv -> raise EOFError` |
| **F4** | value-head 塌缩 + policy 固化，进程健康 | 2（1 完全 + 1 部分） | `midterm_rerun_alpha`：`self_v` 0.6706(iter5) → **0.0114**(iter79)，**−98%**；同时 `self_pi` **上升** 0.7306 → 0.8008；`var_steps` 90.54 → 11.15；`avg_steps` 30.91 → 13.69；`policy_entropy` 全程 0.66–0.76（**熵监控完全看不到**） |
| **F5** | opening-RL 自对弈饥饿（设计使然，但令健康门失效） | 7 | `Self-Play: games=0 \| new_samples=0 \| avg_steps=0.00 \| var_steps=0.00 \| policy_entropy=0.0000`；`WARNING - Best promotion blocked by health gate at iteration 222: failed=['avg_steps','var_steps','self_value_loss']` |
| **F6** | 拓扑/资源实验 | 6 | 见 §3.3 |
| **F7** | stdout 采集为空 | 1 | `run_20260720.log` **0 字节**，但同 run 的 `train_info.log` 有完整 249→260 记录与 `260/260` 汇总 |
| **F8** | 整轮零 best-model 刷新（平台期） | 9 | `Eval-Best-Generations ... decision=` 统计：`0/4`、`0/3`、`0/3`、`0/3`、`0/1`、`0/2`、`0/1`、`0/5`、`0/3`、`1/N` |
| **F9** | 硬性证据缺口 | 1 | `midterm_guarded_alpha` 差 2 个迭代到 200 被截断，无任何原因记录 |

### 3.3 最尖锐的定量发现

**(a) 蒸馏能力在长程自对弈中净流失（高置信度）。**
`Eval-Teacher` 的 W/L 经 `distill_trainer.py:2966-3003` 核对，是**学生**对固定 teacher 的得分。

- `midterm_guarded_alpha`：iter 4 对 teacher 胜率 **0.825**（teacher-cache 热启动）→
  iter 196 落到 **0.175–0.425 区间**（49 次评测），
  同期 **全部 49 次都打印 `Eval-Health-Gate: healthy=True | failed=[]`**。
- 直接驱动因素：`teacher_mix_schedule` 把 `teacher_ratio` 从 iteration 21 起永久置 **0.0**，
  `teacher_loss_floor` 到 iter 20 衰减到 0.25。
- 反向对照：同仓库 `flagship_wide` 用同样的 gate、同样的 teacher，
  iter 172–188 达到 teacher 胜率 **0.825–0.875**，59 次评测中 45 次 `pass`。
  **所以失败在于 mini 分支的配方，而不是流水线本身。**

**(b) 健康门对塌缩方向不敏感（高置信度）。**
`best_health_gate` 阈值全部是**下界**
（`min_avg_steps=18.0`、`min_var_steps=50.0`、`min_self_value_loss=0.20`），
因此“局长/方差/self_v 一起往下掉”这种塌缩恰好从门的下面穿过去。
`midterm_guarded_alpha` 在 iter 196（`avg_steps=29.88, var_steps=113.40, self_value_loss=0.4967,
teacher_win_rate=0.375`）仍然 `healthy=True`。

**(c) gate 判别力不足（中高置信度）。**
`best_eval_games_per_generation=24`，阈值 `0.550`。24 局在 p=0.5 时 σ≈0.102，
即阈值仅 **0.49σ**。实测同一 `checkpoint_48` 在不同 iter 得分
0.458 / 0.708 / 0.500 / 0.625 / 0.417 / 0.625（摆动 0.29）。
`midterm_rerun_alpha` 在 iter 72 以 `checkpoint_56=0.625, checkpoint_48=0.625` **通过**，
而当时它的绝对局长是 14.78、`self_v` 只有 0.0306。

**(d) GPU 长期贡献率极低（高置信度）。**
teacher cache 生成的服务统计：`requests=51,487,298`，`batches=8,433,279`，
`average_batch_size=6.105`（`batch_capacity=64`，`batch_fill_ratio=0.0954`），
**`timeout_flushes=8,420,669` = 全部 batch 的 99.85%**。
同类统计在归档中每个 self-play server 都是 0.05–0.22 的填充率。

**(e) 首手不对称（高置信度，来自 `v21_high_eval_iter200.json`）。**

```
overall:            win 152 / loss 48  score 0.76   mean_moves 28.93
candidate_first:    games 100  win 99  loss 1   score 0.99  mean_moves 27.07
candidate_second:   games 100  win 53  loss 47  score 0.53  mean_moves 30.79
protocol: 200 games, mcts_sims=512, temperature=0.0, dirichlet_noise=false,
          alternating_first_player=true, seed=20260706
```

先手 0.99、后手 0.53，**46 个百分点的色差**。任何引用 0.76 的强度结论都继承了这个偏差。

### 3.4 `teacher_failure_rl` 支线（独立验证）

- 脚本：`training/main_teacher_failure_rl.py`（1142 行），`TeacherFailureRLTrainer(Trainer)`
- 机制：学生 vs teacher 只保留**失败局**（`teacher_failure_num_games=160`），
  以**固定比例**（`target_teacher_failure_mix_ratio=0.4`）混入 batch，
  历史失败降权 `history_weight=0.5`，新鲜失败 `fresh_weight=1.0`；
  4 轮为一个修复周期后 `subprocess.run` 调 `main_train.py`
- **结果：完全无效。** `training/checkpoints_teacher_failure_rl/FINAL_REPORT.txt` 原文：

```
Number of Evaluations: 1
Average Win Rate vs Previous Best: 0.467
Final Best Win Rate Recorded: 0.000
```

- 但**策略本身是健康的**：`Average of Iteration Mean Steps: 22.292`，
  `Average of Iteration Step Variance: 106.876`，`Average of Iteration Policy Entropy: 0.6032`，
  对 teacher 平均胜率 0.516。**即失败回放机制是“惰性”而非“有害”**——
  它维持了健康分布，却没有产生任何可测的强度增益。

---

## 4. F-4 / F-5：本仓库 V3 自身的失败 run

### 4.1 F-4 `stage1_b10c256_relative_role_guard_coldstart_400g_256_32_2x3080ti` g000003

路径：`training/runs/stage1/archive/materialized/recovery/g000003_failed_relative_bootstrap_20260828T0605Z/`

`recovery_manifest.json`：

```json
"failed_generation": 3,
"latest_committed_generation": 2,
"latest_commit_sha256": "6a405a19c2c83725e7be9e4257738e03de86bde353ec7e9bf674c4df066febae",
"reason": "relative role cold-start bootstrap implementation error",
"recovery_policy": "discard uncommitted G3 state; resume only from committed G2 checkpoint",
"removed_metric_stages": ["selfplay","learner","validation","stability","dynamic_exploration"],
"config_hash": "f530e2830d987bb8618aca675f284867d0624568f4022f5c748ab54242396f12"
```

处置正确：G3 未提交状态全部搬移到 `artifacts/`（含 10 个 replay shard、
`candidate-g000003-s00000408-d00025961.pt` 49,328,227 B、`generation_drafts/g000003.json`），
带 SHA-256，并从 G2 恢复。**这是 V3 的错误处理被正确执行的一次记录，不是流水线缺陷。**

同一 run 的 g0–g2 指标证明流水线本身健康（用于对照 F-1）：

| gen | `steps` | `train_positions_consumed` | val `policy_loss` | val `wdl_accuracy` |
|---|---|---|---|---|
| 0 | 103 | 26,292 | 3.2160 | 0.4859 |
| 1 | 103 | 52,432 | 3.2080 | 0.5095 |
| 2 | 101 | 78,064 | 3.1807 | 0.5761 |
| 3（未提交） | 101 | — | 3.1131 | 0.6276 |

### 4.2 F-5 `stage1_scale_screen_b6c128_2x3080ti` g00222

路径：`training/runs/stage1/archive/materialized/manifests/operator_recovery/g000222.started-empty.too-many-open-files.json`

```json
{
  "artifacts": [],
  "config_hash": "8abf95f8810528460223f49be3084bbb3257e19d982e2b59bd2f116f60d2ecc0",
  "created_at": "2026-08-23T21:18:46.703432+00:00",
  "generation": 222,
  "phase": "started",
  "run_id": "stage1_scale_screen_b6c128_2x3080ti"
}
```

文件名即故障：**`too many open files`**（RLIMIT_NOFILE 耗尽）。
代码侧已有针对性修复的记录，见 `training/v3/learner.py:582-587`：

```python
# A fresh active-window dataset and DataLoader are built for every
# formal generation, so workers cannot persist across calls.  With
# CUDA pin-memory enabled, asking these one-shot loaders for
# persistent workers retains multiprocessing pipe descriptors after
# iteration and eventually exhausts RLIMIT_NOFILE in long runs.
persistent_workers=False,
```

`artifacts: []` 且 `phase: "started"` 说明故障发生在 generation 开始前，未产生任何中间产物。

---

## 5. 跨栈共同信号：自对弈分布退化（F-6）

这是本记录**最重要的发现**：三条互不共享代码的训练栈都出现了同一个现象。

### 5.1 V3 FP32 对照组实测（`bal5_r1_column_no_tail_cold_fp32lr1e4_seed271828`）

本地归档 280 个 replay shard，逐局重建终局盘面后统计。

**局长分布（终局棋子数）**：

| gen | 平均 | 7 | 9 | 11 | 12 | 13 | 15 | 17 | 20 | ≤12 占比 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 18.55 | 2 | 1 | 1 | 41 | 2 | 78 | 39 | 117 | 11% |
| 3 | 14.99 | 1 | 80 | 40 | 39 | 39 | 79 | 2 | 0 | 40% |
| **4** | **9.90** | **41** | **81** | **117** | 40 | 40 | 0 | 1 | 0 | **90%** |
| 10 | 10.67 | 40 | 119 | 118 | 39 | 0 | 0 | 0 | 39 | 90% |
| 20 | 9.85 | 119 | 120 | 79 | 1 | 39 | 40 | 0 | 0 | 80% |

对照 `metrics.jsonl` 的 `health.game_length.mean`：gen 3 = 16.11 → gen 4 = **10.99**，
`short_le_12_rate` 0.3100 → **0.7300**。**转折点正是 producer 从 `random` 切换到第一个
`candidate` 的那一代。**

**WDL 标签**（`replay.py:40-42`：`WDL_WIN=0, WDL_DRAW=1, WDL_LOSS=2`，标签为**该步行棋方**视角）：

| gen | P1 胜 | P1 负 | 平 |
|---|---|---|---|
| 0 | 275 | 125 | 0 |
| 3 | 158 | 242 | 0 |
| **4** | 120 | **280** | 0 |
| 10 | 123 | 277 | 0 |
| 20 | **2** | **398** | 0 |

（g0 的 275 与 `health.results.p1_wins=221` 不同属正常，因为 WDL 记的是被采样位置的
行棋方视角，而 `results` 是整局先手胜负。）

**7 步局是真实胜局，且防守方主动不封堵。** 逐盘复核（`tmp/_verify.py`）：
gen 4 的 400 局中 `(win_by_P1, win_by_P2)` 计数 = `(280, 120)`，
**重力约束违反 0 例**。一个 7 步终局样例（game 1641）：

```
   L1 ..... | ..... | O.... | ..... | .....
   L0 ..OO. | ..... | XXXX. | ..... | .....
```

P1（X）在 (row2, col0..3) 连四成线；P2（O）三子分别在 (r2,c0)、(r2,c1) 列的第 1 层
和 (r2,c2) 列的第 1 层——**完全没有碰 (r2,c3) 列进行封堵**。

**独立求解器结论（`tmp/solve_3d_connect4.py`）**：

```
lines=411 cells=150 playable_columns=25
minimum possible decisive game length = 7 plies (P1 places 4 stones)
depth  5: value from P1 = -0  -> draw
depth  7: value from P1 = -0  -> draw
```

空盘深度 7 搜索（411 条连线、D4 对称归约、置换表）返回 **和棋**，
即 **7 步必胜不存在**。这 41 局 7 步败局是防守方送出的，不是规则决定的。

**“送杀”比例随训练上升**（`tmp/_threat.py`，统计每个被记录局面里
“行棋方存在立即制胜着法”的比例）：

| gen | 记录局面数 | 行棋方可立即制胜 | 占比 | 其中多解 |
|---|---|---|---|---|
| 0 | 6,573 | 342 | **5.2%** | 48 |
| 4 | 4,397 | 604 | **13.7%** | 320 |
| 20 | 4,629 | 811 | **17.5%** | **491** |

即：**超过一半的“可立即制胜”局面存在多个制胜点**——防守方漏的不只是一个点，
而是整片威胁。

### 5.2 机制判读

这不是 AMP 缺陷（F-1 有 FP32 对照），也不是单一实现的 bug（三条栈都有），
而是**自对弈任务本身的退化模式**：

1. 第一个非随机 producer 出现后，某一侧找到一条快速成线路径；
2. 对手（同一网络）没有学会封堵，于是该路径被反复强化；
3. replay 被大量短局填满，`train_positions_consumed` 的有效信息量下降；
4. 短局率高企 → stability 监控在 warmup 期被豁免（`game_length_pause_start_train_positions`），
   或健康门是下界无法察觉 → 继续训练；
5. 最终或触发 `stability_pause`（F-1），或整条 lineage 停在平台期（F-3 F8），
   或坠回吸引子（F-2）。

**结构性共同点（本记录最重要的横向结论）：三条栈都没有“强制防守”安全网。**

| 栈 | 是否强制封堵/强制赢 | 证据 |
|---|---|---|
| Legacy v2.2_large | ❌ 明确关闭 | `tactical_override_prefer_win=False`、`tactical_override_prefer_block=False`（`main_train.py:59-61`）⇒ `_get_tactical_override_policy` 恒 `None`（`parallel_games.py:205-220`） |
| V3 BAL-5 | ❌ 无此机制 | `training/v3/` 中不存在 immediate-win/block 短路；自对弈完全由 MCTS + 策略头决定 |
| 蒸馏线 | ❌ 无此机制 | `distill_trainer.py` 无对应逻辑 |

因此三栈都会出现“防守方漏掉眼前的三连”这一现象。**这是一个可检验的假设**：
只需在任一栈加入“若存在立即制胜/必须封堵着法则短路”的开关，观察局长分布是否停止塌缩。

**当前监控体系对这类塌缩是盲的**，因为：

- `training/v3/stability.py` 的三条行为信号（`mean_game_length_low`、
  `game_length_variance_low`、`short_game_rate_high`）在 warmup 期被
  `game_length_pause_warmup` 整体豁免（`stability.py:192-215`）；
- 该豁免边界由 `game_length_pause_start_train_positions=200000` 决定，
  而 F-1 在 `train_positions_consumed` 达到 181,248（g27）时已进入塌缩区；
- F-2 的 Legacy 栈根本没有该信号（`filtered=0`，且无短局告警）；
- F-3 的健康门是下界，`policy_entropy` 在三条线上都**没有**下降，
  它在 v2.2_large 塌缩时甚至**上升**（0.6315 → 1.1546，与 `short_games` 相关 +0.953）。

**建议的最小可执行检测器（尚未实现，属建议而非现状）**：
对每个 generation 记录
`%games where the side-to-move has an immediate winning move`
（本记录在 V3 replay shard 上实测为 5.2% → 17.5%，见 §5.1）
以及
`%games blocked at ply <= N`，
二者都可在现有 replay shard 上离线计算，且**不依赖 entropy 或 value loss**。

---

## 6. 失败原因横向归纳

| 失败机制 | 表现 | 出现栈 | 现有防护是否生效 |
|---|---|---|---|
| 自对弈最小长度吸引子（无强制防守安全网） | 局长塌缩、送杀率上升、熵**上升** | **三条栈全部** | ❌ 三条栈的监控都看不见 |
| AMP 缩放塌缩 + 跳过步误判 | 连续零优化步、loss 不动、辅助头发散 | V3 | ❌ `get_scale()` 代理判据有缺口；scale 未入 telemetry |
| 续训状态不等价 | （F-2 的可疑点，未证实） | Legacy v2.2_large | ⚠️ 每次 resume 都是风险点；21 次 resume、4 次零迭代会话 |
| teacher 监督过早归零 | 蒸馏能力净流失（0.825→0.375） | 蒸馏 mini 分支 | ❌ health gate 是下界，全部 49 次通过 |
| gate 判别力不足 | 回归被接受、改进被拒绝 | 蒸馏线 | ❌ 24 局 @0.550 ≈ 0.49σ |
| value-head 塌缩 | self_v −98% 而 self_pi 上升 | 蒸馏线 | ❌ 无上界检查 |
| 跨 lineage 恢复污染 | iter 217-302 继承 40% teacher 混合来源的权重/replay | Legacy v2.2_large | ❌ 恢复来源与配方未被记录 |
| 资源/句柄耗尽 | `too many open files` | V3 | ✅ 已有 `persistent_workers=False` 修复与回归 |
| 进程被外部 kill | 无异常、无 traceback、静默停止 | 蒸馏线（7 次） | ❌ 无 dmesg/cgroup/exit code 采集 |
| 共享推理服务死亡 | 24 worker 同时 exit 1 | 蒸馏线 | ⚠️ 有异常抛出，但服务端 stderr 未采集 |

---

## 7. 证据缺口（明确列出，不做推测）

1. **AMP scale 数值未入 telemetry。** `metrics.jsonl` 的 learner 记录没有 `amp_scale` /
   `amp_skipped_steps`。PLAN 文档声称的 `3.0517578125e-05` 无法从产物复核，
   也无法区分“scale 触底误判”与“每 batch 溢出即 break”两种机制。
2. **F-1 无 stderr 分离采集。** `logs/*.log` 把 stdout 与 stderr 混在一个文件里，
   DataLoader SIGABRT 的 traceback 与主日志交错，无法确定哪一条属于哪一次进程。
3. **F-1 的 `bal5_r1_column_serial_attn2_cold_seed271828` 只跑了不到 1 秒。**
   `controller_state.json` 仍为 `status="running"`，控制器未写清理状态，
   因此**无法确认控制器是否仍在运行或已死**。
4. **F-2 无逐局记录。** 主日志只有聚合统计（`avg_steps`/`var_steps`/`short_games`），
   `self_play_samples.json` 只有 iter 300 的 5 局（rank 1-197）、全部 `winner=1`，
   **P1/P2 的整体胜负分布未知**。塌缩瞬间（301/302）没有留下任何逐局记录。
5. **F-2 的塌缩前配置已恢复，但塌缩瞬间的恢复状态不可复原。**
   `checkpoint_300` 内嵌 `args` 给出了迭代 300 的完整配置（见 §2.2d），
   **但 `latest.pth.tar` 不存在**，因此 2026-03-26 23:26 那次 resume
   **实际加载了什么状态**只能从元数据推断，无法验证。
   同时**迭代 301/302 的 checkpoint 不存在**
   （`checkpoint_interval=4` + `max_checkpoints=4`），塌缩后的权重无法检查。
   跨 lineage 恢复的来源 `checkpoints_teacher_failure_rl/checkpoint_216` **也已不存在**
   （该目录只剩 log + FINAL_REPORT）。
   **仍不可判定的是：具体是哪个未记录的旋钮在 3/26 23:26 被翻转**——
   但 `history_len` 18→20 已证明**至少有一个被翻转**。
6. **F-2 的 `best.pth.tar` 本地已不存在。** arena 记录引用
   `training/checkpoints/best.pth.tar`，该文件当前不在本地
   ⇒ 无法对"塌缩期权重"与"arena 用的 best 权重"做同一性验证。
   （但 §2.5 已用哈希确认 `v2.2_large/model.pth` 是 ≈迭代 295 的 accepted-best 导出，
   **不是**塌缩状态，这与 arena 结果一致。）
6. **蒸馏线 16 条 run 的 checkpoint 全部缺失。** 本地归档每个
   `<run>_gravity_balanced\` 目录只有 `train_info.log`，没有任何
   `checkpoint.pth.tar` / `model.pth` / `best.pth.tar`。
   因此 F2（`KeyError: 'prob_fc.bias'`）无法复现，权重级对比全部不可做。
7. **蒸馏线 11 条 run 的 `FINAL_REPORT.txt` 缺失**（只写到了远端 run 目录，从未同步）。
8. **归档完整性清单不覆盖日志来源。**
   `distillation/save_model/gravity_balanced/iter_0260/SHA256SUMS.txt` 只覆盖 5 个顶层成员
   加 `TRAINING_HISTORY.md`、`check_info_guides.md` 与 pre-28 归档，
   **不覆盖 `distillation/checkpoints/` 与 `experiments/` 下的任何文件**。
9. **`teacher_examples_2000_diverse.pth.tar` 本地缺失。** 2026-06-27 之后每条 run 都依赖它，
   远端路径为 `/root/autodl-tmp/Connect4_3D_AI_v2.2/distillation/cache/teacher_examples_2000_diverse.pth.tar`。
10. **无从判定 F1（7 次静默 kill）的具体触发器。** 归档中没有任何
    dmesg / journalctl / nvidia-smi 显存采样 / cgroup 内存日志 / 退出码 / 启动包装脚本。
11. **F-2 的"哪个旋钮在边界处被翻转"已由作者说明回答（§2.2f）**，但
    **该实验的具体实现版本、以及缺陷修复后的代码，都不在归档中**：
    本记录只能引用 `training/mcts.py:45-58` 的**当前**实现来说明符号方向，
    **无法确认 iter 301/302 当时运行的是不是这一版**。
12. ~~F-2 的 7 步是否规则必胜~~ **已解决**：独立 alpha-beta + 项目自带精确探针
    双重确认 7 步非必胜。
13. **未验证 V3 的 `column3d_fusion_v2` 策略头实际动作维度。**
    规则给出 150 个动作（6×5×5），`metrics.jsonl` 中 `visit_counts` 宽度为 25
    （仅列维度），未确认全 150 动作的策略张量形状。
    （旁证：`checkpoint_300` 的 `prob_fc` 是 `(150, 4800)`，与 150 动作一致，
    但这是 **Legacy** 模型，不能代替对 V3 的验证。）

---

## 8. 附：本记录使用的证据与工具

### 8.1 引用的关键文件

**远程（`connect4_gpu_2608`）**
- `.../failed_amp_canary_20260916/manifest.json`
- `.../failed_amp_canary_20260916/controller_state.json`
- `.../failed_amp_canary_20260916/configs/bal5_r1_column_no_tail_cold_seed271828.json`
- `.../failed_amp_canary_20260916/logs/bal5_r1_column_no_tail_cold_seed271828.log`
- `.../failed_amp_canary_20260916/logs/bal5_r1_column_serial_attn2_cold_seed271828.log`
- `.../runs/bal5_r1_column_no_tail_cold_seed271828/{run_manifest.json,metrics/metrics.jsonl,metrics/gate_g*.json}`
- `.../bal5/r1/controller_state.json`（FP32 重启后的控制器状态）

**本仓库**
- `training/v3/learner.py`（AMP 路径 771-793；DataLoader 注释 582-587）
- `training/v3/stability.py`（阈值与 warmup 豁免）
- `training/v3/replay.py`（`TrainTokenBucket` 674-727；WDL 常量 40-42）
- `training/v3/formal_runner.py`（learner 调用 847-853；无 `steps < 1` 断言）
- `training/v3/pipeline.py`（1540-1556，**有** `steps < 1` 断言——与 formal_runner 不一致）
- `training/v3/STATIC_AUDIT.md`（123 行：canary 与 stability_pause 的既有记录）
- `training/v3/README.md`（133-140 行：train-token bucket 与 AMP 跳过语义契约）
- `training/runs/stage1/archive/materialized/recovery/g000003_failed_relative_bootstrap_20260828T0605Z/recovery_manifest.json`
- `training/runs/stage1/archive/materialized/manifests/operator_recovery/g000222.started-empty.too-many-open-files.json`
- `training/runs/stage2/archive/bal5/r1/cold/bal5_r1_column_no_tail_cold_fp32lr1e4_seed271828/`（280 个 replay shard + `materialized/metrics/metrics.jsonl`）
- `connect4_core/game_rules.py`（`BOARD_SIZE=5, MAX_LAYERS=6, CONNECT_N=4`；重力见 `_validate_move`）
- `tmp_models_plan/PLAN_Stage2_BAL5_Selfplay_V1.md`（第 20–28 行 AMP canary 结论）

**`Connect4_3D_AI_v2.2`**
- `training/checkpoints/train_info.log`、`training/checkpoints/checkpoint_300/{checkpoint.pth.tar,model.pth,self_play_samples.json}`
- `save_model/{v2.2_large,v2.2_balence,v2.2_fast}/model.pth`
- **`arena/history/20260327_143855_v2.1_High_vs_v2.2_1.json`**（§2.3e 的决定性反证）
- `arena/history/20260404_001735_v2.2_fast_vs_checkpoints.json`
- `.tmp/early_win_probe.json`（项目自带的精确求解探针）
- `training/checkpoints_teacher_failure_rl/{train_info.log,FINAL_REPORT.txt}`
- `distillation/checkpoints/distill_v1.2_{balanced,fast}/checkpoint_*/model.pth`（产物归属哈希）
- `distillation/distill_config.json`、`distillation/distill_config.py`
- `distillation/save_model/gravity_balanced/iter_0260/**`（16 条 `train_info.log`、`experiments/model_architecture/**`、`TRAINING_HISTORY.md`、`SHA256SUMS.txt`）
- `distillation/save_model/flagship_wide/train_info.log`（健康对照）
- `experiments/model_architecture/midterm_gravity_balanced/v21_high_eval_iter200.json`
- `training/main_train.py`、`training/main_teacher_failure_rl.py`、`training/parallel_games.py`、`training/trainer.py`、`training/model_compat.py`
- `distillation/distill_trainer.py`

### 8.1b 交叉验证说明

本记录的 F-2 部分经过**两条独立路径**取证并相互校验：

1. 笔者自行解析 `train_info.log` 与会话边界（`tmp/_ti.py`、`tmp/_sess.py`）；
2. 一个独立子代理从零重新解析同一日志、并额外做了
   checkpoint zip 结构比对、SHA-256 产物归属、`main_train.py` 训练期改动取证、
   以及 arena 对战记录的逐局统计。

两条路径在塌缩数值（iter 301 = 7.26/0.85/195、iter 302' = 7.17/0.39/198）、
会话边界位置（`Iter 300/300` → `Resume` → `Iter 301/400`）、
7 步起步法序列（P1 `(0,2,2)→(0,2,3)→(0,2,1)→(0,2,0)`，P2 三子均未碰 row 2）、
以及"7 步非规则必胜"（项目自带探针 + 笔者独立求解器）上**完全一致**。

**曾经出现分歧、现已收敛的三处：**

1. **触发点解释。** 笔者初稿认为"续训状态不等价"，子代理证据（同吸引子在 iter 21-40
   即出现 + arena 反证同一权重对外表现正常）使该假设降级。
2. **"banner 相同 ⇒ 配置相同"。** 笔者据 banner 逐字节相同**排除**了"续训改变超参"，
   子代理随后用日志算术反解 `history_len` 并给出 10/≥14/18/20 的轨迹。
   **笔者已独立重算 348 个 iteration block 并精确复现该结果**
   （176 个 block 匹配 k=10、212-213 匹配 k=14、234-262 与 278-300 匹配 k=18、
   301 匹配 k=19、302 匹配 k=20），
   并且**用独立的 `pickletools` 流式扫描复算了 `checkpoint_300` 的内嵌 `args`**
   （`history_len=18`、`num_mcts_threads=8`、`virtual_loss=1.2`、`self_play_workers=64`、
   `max_self_play_workers=128`、`data_loader_workers=64`、`best_model_iteration=295`、
   `next_eval_iteration=303`、`no_improve_eval_patience=12`、`num_iterations=300`、
   `min_game_steps=1`、两个 `tactical_override_*` 均为 `False`）
   ⇒ **初稿的排除是错误的，文档已按修订版写成。**
3. **产物归属。** 初稿写"`v2.2_large/model.pth` 与 `checkpoint_300/model.pth` 是同权重另存"，
   **这是错的**。笔者初测只比对了 storage 长度多重集与 zip 元数据差值，**没有比对 payload**。
   子代理直接逐 entry 比对 CRC32+SHA256，发现 **117/139 payload 不同、仅 22 个常量张量相同**；
   **笔者已独立用 SHA-256 重测并确认 117/139 不同**（例：entry 0 `conv1.weight`
   两文件均 55,296 B，sha256 `76a6582e…` vs `652c772a…`）。
   ⇒ `v2.2_large/model.pth` 是**≈迭代 295 的 accepted-best 导出**，不是塌缩的迭代 300 状态。
   这与 arena 中同一权重对外取得 55%、0 局 ≤9 步的结果一致，**反而强化了 §2.3e 的结论**。

**方法论教训（本记录最值得记住的两条）**：

1. **banner 只覆盖 7 行，不能当作配置指纹；"长度/元数据相同"也不等于"内容相同"。**
   F-2 上这两点各让笔者走了一次弯路。
2. **产物推断有上限，作者说明能一步补齐。**
   §7 曾把"iter 300→301 边界到底改了什么"列为**不可判定**，
   并推测只能靠 `checkpoint_300` 的 `args` 部分回答。
   **实际答案是作者本人给出的：那是迭代 300 之后的一次 virtual loss 线程数实验，
   而该实现的负号写反了**（§2.2f）。
   这类"实现缺陷 + 实验性参数上调"的组合**不会留下任何配置痕迹**
   ——banner 不打印、checkpoint 里也不会有"这个实验正在进行"的标记。
   ⇒ 结论：**对于无书面记录的实验性训练 run，应尽早向作者确认，而不是只在产物里穷举。**

**同时需要保持的边界感**：作者说明也明确了
`check_info_guides.md` / `TRAINING_RUNBOOK.md` 是**旧栈经验规则**，
其中开局温度那条**已被 V3 否定**。因此本记录对这些阈值的引用
一律标注为"参考"，**不构成 V3 的验收依据**。

### 8.2 本次分析新建的临时工具（`tmp/`，非仓库契约）

| 文件 | 用途 |
|---|---|
| `tmp/solve_3d_connect4.py` | 独立 alpha-beta 求解器，判定空盘 7 步是否必胜（答案：否） |
| `tmp/_verify.py` | 从 replay shard 重建终局盘面，校验重力与胜负判定 |
| `tmp/_threat.py` | 统计“行棋方存在立即制胜着法”的局面比例 |
| `tmp/_fp32.py` / `tmp/_learn.py` / `tmp/_sp_table.py` | 解析 `metrics.jsonl` 生成 learner/validation/self-play 表 |
| `tmp/_ti.py` / `tmp/_sess.py` | 解析 v2.2_large `train_info.log` 与定位会话边界 |
| `tmp/_metrics_table.py` / `tmp/_stab.py` / `tmp/_bucket.py` / `tmp/_man.py` / `tmp/_ckpt.py` | 逐代指标、stability 轨迹、token bucket 核账、manifest、checkpoint 结构 |
| `tmp/failure_forensics/` | 从云端拉取的原始证据副本（logs / configs / metrics / manifests） |
| `tmp_models_plan/PLAN_Stage2_BAL5_Selfplay_V1.md` | （既有文件）BAL-5 计划与 AMP canary 结论 |

### 8.3 明确的未修改声明

本次操作**没有**修改、移动或删除任何训练产物、checkpoint、配置、历史 JSON 或既有文档。
新增内容仅为 `docs/FAILURE_RECORD.md`（本文件）与 `tmp/` 下的分析脚本/证据副本。
远程主机上只执行了只读命令（`find`/`ls`/`cat`/`grep`/`wc`）与 `scp` 下载。

---

## 9. 建议的后续动作（按优先级）

1. **给自对弈加"必须封堵/立即赢"安全网，并把它当作可开关的实验而非默认。**
   这是本记录唯一在三条栈上都能解释塌缩的机制（§5.2），
   而 Legacy 栈已证明该机制**被显式关闭**
   （`tactical_override_prefer_win=False` / `prefer_block=False`）。
   建议在 `training/v3/selfplay.py` 增加一个默认关闭的
   `immediate_win_or_block_override` 开关，用 A/B 观察局长分布是否停止塌缩。
2. **把"送杀率"加入 generation 健康指标。** §5.1 的
   `%games where the side-to-move has an immediate winning move`
   （实测 5.2% → 17.5%）与"ply ≤ N 是否被封堵"是目前唯一
   在三条栈上都能提前发现塌缩、且**不依赖 entropy / value loss** 的信号
   （F-2 的熵在塌缩时反而上升，与 `short_games` 相关 +0.953）。
3. **给 AMP 路径加权威跳过判据与 telemetry。** 在 `training/v3/learner.py` 中
   改为读取 `GradScaler` 自身的跳过状态，并把 `amp_scale`、`amp_skipped_steps`
   写入 `LearnerMetrics`；为"scale 触底后仍溢出"补一个回归测试
   （现有 `_OneShotSkippingScaler` 只跳过第一次，无法覆盖该分支）。
4. **统一 `steps < 1` 断言。** `pipeline.py:1553` 有断言而 `formal_runner.py:847-853` 没有，
   导致 formal 路径下 24 代零更新静默通过。建议在 formal runner 中加同一断言，
   或在 `LearnerMetrics` 中标记 `budget_starved`。
5. **重新审视 warmup 豁免边界。** `game_length_pause_start_train_positions=200000`
   让 F-1 的塌缩区（g4–g27，`train_positions_consumed` 65k–181k）完全落在豁免窗口内。
6. **让短局真正进入训练决策，而不是只降权 10%。** Legacy 栈 `min_game_steps=1`
   使 348 个 block 全部 `filtered=0`，且短局率正是强度恶化的先行指标。
   建议把"短局率"作为数据准入或采样权重的显式输入。
7. **为蒸馏线补 gate 置信区间与上界型健康门。** 24 局 @0.550 的判别力不足；
   同时需要 `max`/`min` 双向阈值来捕获 value-head 与局长的**向下**塌缩。
8. **记录恢复来源与配方，禁止静默跨 lineage 恢复。** F-2 的 iter 217-302 继承了
   兄弟目录 40% teacher 混合配方的权重与 replay，而日志里只有一行
   `Teacher-History: replay_pool_samples=34744` 作为线索。
   建议在 run manifest 中记录 `resume_source_run_id` 与 `resume_source_recipe`。
9. **把整个 `args` dict 打进会话开始日志（成本最低、收益最高的一项）。**
   §2.2c 证明 `history_len` 在 run 中取过 10/≥14/18/20 四个值，
   而 banner 只有 7 行、不打印其中任何一项——**塌缩边界处的配置改动因此无法复原**。
   把 `vars(args)` 每次会话开始全量写入 `train_info.log`（约 20 行代码）
   可以彻底消除这一整类歧义。
10. **不要用"banner 相同"作为"配置相同"的证据，也不要用"长度/元数据相同"作为"内容相同"的证据。**
    这是本记录在 F-2 上走了两次弯路的直接教训：
    初稿据 banner 相同排除了"续训改变超参"（被 §2.2c 的窗口反解推翻），
    又据 storage 长度多重集相同断言两个模型"同权重"（被 §2.5 的 payload 哈希推翻）。
11. **把项目的健康标准接入自动停止——但只作为起点，不作为 V3 的真理。**
    `check_info_guides.md` 与 `TRAINING_RUNBOOK.md` 已写下可执行的塌缩判据
    （`avg_steps < 18`、`var_steps < 50`、短局 > 10/100、teacher 胜率 < 0.40），
    且 runbook 明确授权检查者（AI agent）停机回滚。
    **本 run 在 975-990/1000 的短局率下继续跑了约 200 个迭代而无人介入。**
    但注意 §2.2e：这些阈值是**旧栈经验**，其中"开局温度降到 0.8-0.6 倍"这一条
    **已被 V3 否定**（V3 前 28 步全程 `T=1.0`，且不到 1M 训练量即超过旧三代模型 v1 Elo > 100，
    开局解与人类/旧模型都不同）。
    ⇒ 建议做法是**先在 V3 上重新标定阈值**，而不是直接照搬。
12. **对"并发搜索正确性"补一条与线程数无关的回归测试。**
    §2.2f 的缺陷是"4 线程可用、上调后崩坏"，即**症状随并发度放大**。
    V3 的 `search.py` 符号方向正确，但**目前没有测试覆盖
    virtual loss 在 apply/revert 后的值不变性**（`visit_count` 与 `value_sum` 应精确回到原值）。
    建议为 `apply_virtual_loss` / `revert_virtual_loss` 加一条
    "N 次 apply 后 N 次 revert 必须逐位还原"的属性测试——
    这类缺陷在单线程下几乎不可见，只有并发度拉高才暴露。
13. **采集进程级外部证据。** 至少落盘退出码、cgroup 内存峰值、GPU 显存采样与推理服务 stderr；
    当前 7 次静默 kill 完全无法归因。
14. **把 `teacher_mix_schedule` 归零视为一次 A/B 实验而非默认。**
    `midterm_guarded_alpha` 的 0.825→0.375 与 `flagship_wide` 的 0.825–0.875
    说明该设置在 mini 分支上有实质代价。
    （同样地，这条结论建立在旧栈上，V3 需自行验证。）
15. **修正配置与实际生效值不一致的记录。** `distillation\distill_config.json` 声明
    `fast = 128 ch / 25 iterations / 150 games / window 20`，实际是
    **96 × 3 / 200 iterations / 128 games / window 12**（`distill_config.py:38-41` + 权重哈希）。
    F-2 的 banner 同样不打印短局过滤/战术短路/损失权重等关键项，
    且**磁盘上的 `main_train.py` 是续训版修订，不是跑 iterations 1-300 的那一版**。
