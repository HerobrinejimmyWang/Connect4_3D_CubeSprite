# CubeSprite 回放协议

CubeSprite 回放文件使用 UTF-8 JSON，建议扩展名为 `.c4replay.json`。导入器按
`protocol_version` 做严格的 exact-key 分派：V1 保持原样可读；新增的规则与
参与者感知生产端应写 V2。

## V2 顶层结构

V2 顶层字段固定为：

```json
{
  "format": "cubesprite.replay",
  "protocol_version": 2,
  "id": "32 位小写 UUID hex",
  "name": "可修改的显示标题",
  "saved_at": "含时区的 ISO-8601 时间",
  "rules": {},
  "rule_id": "classic",
  "rule_version": 1,
  "participants": [],
  "turns": [],
  "turn_count": 0,
  "placement_count": 0,
  "status": "playing",
  "winner": null,
  "fingerprint": "小写 SHA-256",
  "participant_provenance_hash": "小写 SHA-256"
}
```

`rules` 沿用 V1 的固定棋盘几何描述。`rule_id` 是稳定的规则标识，当前注册表
包含 `classic`、`p1_vertical_ignored`、`p1_vertical_forbidden` 和
`p1_layer0_ignored`；`rule_version` 必须与可执行规则注册表一致。

## 对局双方

`participants` 必须恰好包含 `FIRST/+1` 和 `SECOND/-1` 两项，每项字段固定为：

```json
{
  "seat": "FIRST",
  "player": 1,
  "controller_type": "model",
  "controller_id": "可选的稳定控制器 ID",
  "display_name": "可修改的显示名",
  "model_id": "可选的稳定模型 ID",
  "lineage_hash": null,
  "artifact_sha256": null
}
```

`controller_type` 只能是 `model`、`human`、`random` 或 `external`。未知的可选
标识与哈希写 JSON `null`；存在的 lineage/artifact 哈希必须是小写 SHA-256。
这些身份仅用于记录，不进入神经网络输入。

## 带标签的回合

一次落子固定为：

```json
{
  "ply": 1,
  "kind": "place",
  "player": 1,
  "column": 7,
  "action": 7,
  "layer": 0,
  "row": 1,
  "col": 2
}
```

模型与搜索只输出 `[0,24]` 的 `column`；`action` 和三维坐标由程序按重力
确定地映射到 150 个棋盘坐标。

强制 pass 固定为：

```json
{"ply": 87, "kind": "forced_pass", "player": 1}
```

它不能携带 column/action/坐标，仅在规则引擎判定无合法落子且要求 pass 时
合法。pass 不改变棋盘、切换行棋方、计入 `turn_count`，但不计入
`placement_count`；实际落子重置连续 pass，连续两次强制 pass 判和。

## 完整性边界

`fingerprint` 对 format、协议版本、棋盘几何、rule ID/version 和规范化 tagged
turns 的紧凑 JSON 计算 SHA-256。它覆盖稳定的棋局语义，不覆盖回放 ID、标题、
时间和参与者。

`participant_provenance_hash` 单独覆盖双方的 seat、player、controller type/ID、
model ID、lineage hash 与 artifact hash，并故意排除 `display_name`。因此显示名可
本地化或修改，而稳定身份的变化必须重新计算 provenance hash。

## V1 与分析旁车兼容

V1 顶层和 move 的原 exact-key 校验路径保持不变，不会在读取时静默升级为 V2。
胜率分析仍单独保存在 `replay_analysis`，通过 replay ID 与 gameplay fingerprint
关联；重新计算和原子写入规则不变。
