# CubeSprite v0.2.0-alpha.1 · Windows

本次仅发布 Windows x64 安装包。版本记录在包元数据和后端；App 页面不显示版本号。
最终包已于 2026-09-30 在本机静默安装，并使用独立测试数据目录通过完整原生端到端复测。
最终包 SHA-256：`98c8afb1d0cd57b5878849a9867165e63cf22c537cdf485805e348f424482724`。

## 本次内容

- 注册表首位及默认模型改为 **CubeSprite V4 Flash (Preview1)**。默认搜索仍为
  256 simulations / temperature 0.4。
- 主菜单新增五种互斥规则，沿用 Stage 3 的规则编码、绝对行棋身份和 32 维规则输入。
  禁手落点从合法落子列表中排除，规则要求的 forced pass 自动执行。
- 菜单中旧模型与非 Classic 规则冲突时，按选择顺序切回 Classic 或路由到 V4，
  并显示红色横幅。在已开始的非 Classic 棋局中，拒绝不兼容模型切换，保持棋局。
- AI Intelligence 七档暂定名称，绑定 V4 的 16/32/64/128/256/512/1024 搜索次数及
  temperature 0.5；三类 AI 使用该预设，Advance 可分别细调。
- 游戏页增加 Instructions、AI Settings；Back 回到原棋局。说明中删除 Quick start，
  说明 Classic 与四种调整。棋盘 F1 对应训练协议的 L0。
- 保存/导入使用 replay v2，不兼容 v1；支持训练侧样例、逐回合回放、胜率分析和续局。
  forced pass 在回放中占一个回合，但不计入游戏页的棋子数 `/150`。
- 记录实际参与模型及权重哈希；混合使用多个模型的局不冒充单模型来源。

## 模型来源与边界

- 只读源：仓库根目录 `terminal_snapshot.pt`，V3 原生 Stage 3 generation 32
  evaluation-only 终端快照；未转换 Legacy 权重，未修改训练行为。
- 源 SHA-256：`2524860a99c85ae2e2aa2eb018368e906c7bbcfb1191051532a2d75f42cf6f2f`。
- 发布 ONNX SHA-256：`bbc60a63a72a4ad908006b094c5d7c7a238e204a2a6d857b5cdbfaf09d5e4344`。
- 已通过五种规则 × 双方行棋身份的 PyTorch/ONNX 数值对比
  (`rtol=2e-4, atol=2e-5`) 和实际 16-simulation 搜索合法性检查。
  这些是功能与导出一致性检查，不是棋力评估。

## 安装包与证据

- 最终安装包：本目录的 `CubeSprite_0.2.0-alpha.1_x64-setup.exe`。
- 首轮安装包：`builds/01/CubeSprite_0.2.0-alpha.1_x64-setup.exe`，按要求保留。
  它已通过安装后的界面流程验证，最终包另外修正了 forced-pass 落子数显示。
- `evidence/` 保存最终安装验证结果、界面截图和 v2 保存样例；`checksums.json`
  记录安装包与程序资源哈希。
- 已核验安装后的七个模型及 sidecar 哈希。安装后主程序仅比构建目录程序多出
  Tauri 的 NSIS 包类型标记（`UNK` → `NSS` 三字节），归一化后逐字节相同。
- 源码、模型注册表、ONNX（Git LFS）、小型训练回放测试夹具与测试报告可跟踪。
  安装包、冻结 sidecar、编译缓存和临时 WebView 数据属于本地生成物，不提交 Git。

## 验证范围

- 前端完整 78 项测试通过。TypeScript 和生产构建通过。
- 两个 Python 后端测试入口：42 项、41 项通过（存在共享回放测试）。
- 共享规则回归：12 项通过；语法编译、差异空白检查通过。
- 实际 Windows 安装后，通过 WebView2 操作真实 Tauri/JSONL/ONNX 链路，覆盖默认模型、
  双向兼容路由、七档参数、五规则落子/禁手、2D/3D、说明和设置返回、真实 AI 开局、
  保存 v2、导入训练样例、播放至终局及指定步骤续局。没有模拟后端。
- 验证机已有 WebView2；安装包包含离线 WebView2 安装资源，但未在全新 Windows
  虚拟机上另行验证首次安装 WebView2。

## 构建环境

Windows 11 build 26200；Node 22.22.2；Rust 1.98.1 / MSVC 14.44；
Python 3.12.10；NumPy 2.2.4；ONNX Runtime 1.26.0；PyInstaller 6.20.0。
导出另用 PyTorch 2.8.0、ONNX 1.19.0。

本机补齐的 Rust/MSVC 工具在 `D:/cc/tmp/cubesprite-toolchain/`；导出依赖在
仓库 `.tmp/export-py312/`。本机构建脚本为 `.tmp/build-alpha.ps1`，仅对当前进程
设置 PATH、Rust 目录和 ASCII 临时目录，不改用户 PATH。封装配置覆盖
`bundle.useLocalToolsDir=true`，避免构建进程不能解析系统缓存目录的问题。

常规环境可在 Developer PowerShell 中按 `desktop_app/README.md` 构建；原生端到端脚本：

```powershell
python desktop_app/scripts/e2e_windows.py --exe <安装目录>/cubesprite.exe --sample desktop_app/backend/tests/fixtures/training-v2-sample.c4replay.json --output <证据目录>
```

后续 effort 正式命名、Opening Library、Tutorial、Hint 重构仍在后续范围内。


## alpha.1 界面检漏（同版本重新打包）

- 回放和阵营选择窗口改为独立并排布局，消除覆盖主菜单的问题；缩小主菜单按钮高度。
- 七档滑条使用蓝灰胶囊轨道、内嵌档位点、白色圆形滑块；保留键盘和鼠标操作。
- 中文显示“经典”“AI 智能度”“高级设置”“导入回放”，同步本地化规则提示和说明。
- 按窗口高度调整 AI 设置、一般设置、说明、2D/3D 棋盘和回放工具栏；模型列表及长说明保留内部滚动。
- 原生安装版验证 1536×792、1100×720、1280×720、1440×900、1920×1080 的菜单/侧栏和设置页面，
  验证最小窗口的棋盘、回放及续局弹窗。文档外层不溢出，选择窗口不重叠。
- 源码浏览器另检查中英文六种尺寸（上述五种及 880×576 横屏）。
- 最终原生测试仍运行安装包中的 ONNX、sidecar、Tauri IPC，覆盖七档键盘/鼠标交互与保存、导入、续局。
- 本会话默认 AppData 路径中，独立 Python 同目录原子替换探针也报 WinError 17。
  最终测试通过进程环境变量 `CUBESPRITE_DATA_DIR` 使用工作区内的独立数据目录；未修改系统 Known Folders，
  未弱化原子写入。此环境下默认 AppData 的写入不标记为通过。未设置该变量的正常启动仍用原位置。
- `builds/02` 保留上次交付包及证据，`builds/03`、`builds/04` 保留本次中间包及未通过的预检结果；
  它们不是此次最终交付包。最终包位于本目录，最终成功证据位于 `evidence/`。


## 棋盘可读性补漏（同版本重新打包）

- 压缩游戏页上下留白和状态栏/工具栏占用，保留桌面六层的 4+2 排列。
- 1100×720 中二维棋盘宽度从 740px 增至 960px，棋格边长约从 27px 增至 38px；
  1536×792 恢复到原有 1110px 最大棋盘宽度。三维棋盘也增加可用高度。
- 回放页根据底部工具栏是否换行单独分配棋盘空间，防止底部按钮被挤出视口。
- 源码浏览器通过五种尺寸 × 中英文 × 对局/回放 × 2D/3D 共 40 个布局检查，
  验证棋格点击落子，页面无整体滚动。证据为 `evidence/board-size-geometry.json`。
- 最终安装版端到端复测通过；新增 1100×720、1280×720、1536×792 棋格边长至少 37px
  的检查及对应 2D/3D 边界检查。继续使用独立数据目录，不扩展上文默认 AppData 的验证结论。
- 上一份已交付安装包、校验信息、截图和报告保留于 `builds/05/`。
