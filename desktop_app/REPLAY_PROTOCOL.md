# CubeSprite 回放协议

CubeSprite 回放文件使用 UTF-8 JSON，建议扩展名为
`.c4replay.json`。当前协议标识为：

- `format`: `cubesprite.replay`
- `protocol_version`: `1`
- `rules.format`: `connect4-3d-gravity`
- `rules.version`: `1`

## 可分享回放文件

回放文件只保存复现棋局所需的信息：

- 唯一 ID、显示名称和保存时间；
- 固定的 `6 × 5 × 5`、连四、逐层重力与红方先手规则描述；
- 每一步的序号、动作索引、三维坐标和行棋方；
- 落子总数、对局状态和胜者；
- 对规则与完整落子序列计算的 SHA-256 `fingerprint`。

文件不会保存原对局模式，也不会保存当时使用的 AI。导入端会从空棋盘
逐步重放全部动作，并校验轮流行棋、动作与坐标映射、重力、终局位置和
fingerprint。字段缺失、未知字段、重复 JSON 键、非法数值、越界或终局后
继续落子的文件都会被拒绝。

## 胜率分析旁车文件

胜率曲线单独保存在应用数据目录的 `replay_analysis` 中，不会写回可分享
回放。分析文件通过回放 ID 与 fingerprint 同时关联，包含：

- 使用的模型 ID、显示名称、架构和不可变模型文件哈希；
- MCTS 模拟次数与温度；
- 用于阻止多个应用实例互相覆盖新结果的分析请求代际；
- 开始、完成时间和执行耗时；
- 从第 0 步到回放末步的红蓝双方胜率。

重新计算会使用当前“AI 设置”中的胜率模型和配置，并原子覆盖原分析。
如果回放内容已改变，旧分析不会被加载或写入。

## 本地存储

桌面应用在 Tauri `app_data_dir` 下使用两个互相独立的目录：

```text
replays/
  <id>.c4replay.json
replay_analysis/
  <id>.winrate.json
  .<id>.generation.json  # 内部并发代际标记
```

所有写入先写临时文件，再以原子替换完成；跨线程与多个应用实例的写操作
由同一存储锁串行化。
