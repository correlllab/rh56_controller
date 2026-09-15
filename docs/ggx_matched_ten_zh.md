# GGX × Reflex：共享输入、共享十起点的新实验入口

入口：`tools/run_ggx_matched_ten.py`。旧 `run_ggx_experiment.py` 的单模板协议保留不动。
新入口不读取旧批准来授权动作；默认完全离线，`--execute --capture-live` 才允许进入明确确认的复位/执行流程。

## 已统一的条件

| 条件 | 新入口的处理 |
|---|---|
| 物体与参考观测 | 从指定 Reflex 报告读取原模板、RGB-D、目标掩码和同一份分割点云，核对哈希；不重新裁剪或人工改尺寸 |
| P1–P10 | 保留该 Reflex 记录的 TCP、朝向、点顺序和抓取参考高度；所有物体统一使用新腕分支的 IK 初值，不由 GGX 候选决定 |
| 放回后的定位 | 与 Reflex 相同的 RGB XY 定位、120 px 搜索窗口（继承记录值）、参考深度米制换算；目标与十点同量平移，Z/yaw 冻结 |
| 拍摄/复位位 | 所有物体统一使用 `configs/reflex_scan_wrist_flip.json` 新腕预设，继续使用已有复位流程 |
| 机械臂设置 | 同一份已记录 Reflex 配置：0.20 rad/s、0.02 m/s；加速度 0.20 rad/s²、0.02 m/s² |
| 手设置 | raw speed=1000、force=1000；仍需 `--allow-high-force` 和执行确认；不是实测物理力/速度 |
| 桌面/机械臂模型 | 保留同一标定桌面、CB3 模型、2 mm 桌面规则和控制器检查；不为放行调阈值 |
| 抬升/评分 | 抬升 80 mm，继承 Reflex 的 3 秒 hold，使用同一个评分函数；完整离桌且未滑落才记成功 |
| 失败后的处理 | 每点最多一次物理尝试；闭合/运动开始后不换候选重试。命令完成不自动算成功 |

注意：IK 初值是新腕分支偏好，不保证后续所有关节解逐数相同；实际解与路线仍单独记录和检查。
每次放回允许 XY 有差异，因此匹配的是物体相对任务，而不是强行保持世界坐标完全相同。
用户随后明确选择：**所有新 GGX 实验统一用新腕分支，包括糖盒**。
历史糖盒 Reflex 用旧分支，结果不回写；这项 GGX 执行条件调整预先记录，
不声称它已经证明更容易成功，也不将新结果称为完全相同关节状态的对比。
原十点 Cartesian 坐标、TCP 朝向、点间相对位置和高度均不改。
报告的 `shared_conditions.wrist_branch_policy` 保存原各点关节初值、历史拍摄位和新公共预设。
新预设本身不是已校验的运动授权；首次从当前状态切换仍走原有预检、确认和线缆观察。

## 必须披露、未强行统一的方法差异

1. Reflex 每物体冻结一个 analytical grasp；GGX 冻结原始 top-100，每点按原排名选首个可执行候选。
2. Reflex 三指/四指模式、拇指先准备及原生 .2/.25 秒等待，与 GGX 原始手型和插值闭合不同。
   GGX 保留原有 3 秒闭合＋1 秒稳定，不为“统一”改写已有控制策略；两者都没有可信手反馈。
3. Reflex 允许接近时的预期手指接触；GGX 保留既有目标代理检查。候选与路径拒绝必须单列，不能全算成物理抓取失败。
4. 苹果/罐的 Reflex 使用人工冻结朝向后的近似代理，GGX 保留原实测包围代理；这不是完全相同的目标几何检查。
   未修改 GGX 代理以增加通过率。如果论文要求 checker 完全相同，需另定义协议并复查两边，不能回写旧结果。
5. 未建模相机支架、线缆等障碍仍需现场确认；UR5 自带保护和模型通过均不证明这些路径没有碰撞。

所以应称为 **shared-perception / shared-start comparison with method-specific execution and contact policies**，
不宣称“除 planner 外所有检查和控制完全一致”。

## 冻结候选池的规则

从共享点云生成一次：1000 个原始候选，保存截断前全部候选，再保持 GGX 原分数/顺序取 top-100。
官方场景筛选与本地局部检查完成整个 top-100，缓存其结果；每点仅对已局部通过项按排名检查实际路线。
控制器的纯预检几何拒绝也可以检查下一候选；通信/状态异常、清理异常、场景变化或取消都停止。
视频的 `n` 只作为记录理由的安全否决，不应依靠人工选择“看上去更容易成功”的候选。
已经发出手/臂命令或收到最终运动授权后，任何异常都禁止自动换候选。

## 五个对应的 Reflex 输入

| 物体 | `--reflex-run` |
|---|---|
| Sugar box | `artifacts/reflex_ten_runs/20260913T230536_911538Z/summary.json` |
| Cube | `artifacts/reflex_ten_runs/20260913T235549_436157Z/summary.json` |
| Charger | `artifacts/reflex_ten_runs/20260914T003241_264161Z/summary.json` |
| Apple | `artifacts/reflex_ten_runs/20260914T021908_736921Z/summary.json` |
| Can | `artifacts/reflex_ten_runs/20260914T023308_279132Z/summary.json` |

这是最近 Reflex 的五物体清单：**苹果，而不是旧 GGX 清单中的芥末瓶**。
这些输入用于冻结条件，不替代旧物理结果的核实：cube P4 修正/补试和 can 的手工补录仍保留各自证据边界。

## 命令

只核对条件（不推理、不 IK、不接触设备）：

```bash
.venv312/bin/python tools/run_ggx_matched_ten.py \
  --reflex-run artifacts/reflex_ten_runs/20260913T230536_911538Z/summary.json \
  --validate-only
```

首次离线生成、完整筛选并检查十点：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_matched_ten.py \
  --reflex-run artifacts/reflex_ten_runs/20260913T230536_911538Z/summary.json
```

确认对应物体、摆放朝向和现场路径后，首次生成并进入实机十轮：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_matched_ten.py \
  --reflex-run artifacts/reflex_ten_runs/20260913T230536_911538Z/summary.json \
  --capture-live --execute --allow-high-force
```

已冻结候选池时，在上面的命令增加 `--pool-report <本轮输出目录>/pool/summary.json`，避免再次生成。
若仅需要用新代码重新检查同批候选，使用 `--candidate-run <本轮输出目录>/generation`；
不得因为某批结果差就不记录地重新生成直到通过。

续跑（保留已完成/失败的物理结果，不重试它们）：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_matched_ten.py \
  --resume-from <上轮目录>/summary.json \
  --capture-live --execute --allow-high-force
```

部分运动、未完成复位或清理异常不自动续跑，需先核实状态。
`--point P5` 是单点新运行，不是覆盖旧 P5；补试应另算 attempt。

## 输出与验证

输出在 `artifacts/ggx_matched_ten_runs/`：`summary.json`、`summary.csv`、共享布局、
候选池、逐点全部排名状态、被尝试候选的拒绝原因、执行视频和人工结果。
历史代码漂移单列；保存的模型、配置、标定、观测和坐标被修改会拒绝使用。
此前逐物体历史分支验证报告：`artifacts/ggx_matched_protocol_checks/*_02/summary.json`。
公共新腕分支验证报告：`artifacts/ggx_matched_protocol_checks/*_new_wrist_01/summary.json`。
全部旧 GGX/Reflex 原始实验记录不修改，新增结果不得与旧单模板协议无标记混合。

### 本次实际离线串联

`artifacts/ggx_matched_ten_runs/sugar_offline_pipeline_01/` 使用糖盒 Reflex 保存的同一份点云，
生成一次 1000 个原始候选并保留 top-100：官方通过 83、本地局部通过 8；
P1–P10 均由 rank 4 首先通过名义路线检查（10/10）。
`sugar_offline_pipeline_02/` 复用相同候选池、不重新推理，十点选择完全一致。
两轮均无 UR5 接收/控制连接及手命令；这是离线路径结果，不是物理成功。
该共享 Reflex 场景不同于此前历史糖盒 rank 20 的场景，不能与历史 8/10 合并。
上述两轮使用修改前的糖盒旧腕分支；公共新腕复查单独保存在 `sugar_new_wrist_01/`。
公共新腕复查结果：同一批候选、无新推理，P1–P5/P7–P9 由 rank 4 通过（8/10）；
P6/P10 对全部 8 个局部可用候选均在 `position_to_trial_start` 被桌面检查拒绝。
这是准备路线检查结果，不是物理碰撞证据或 GGX 抓取失败；更换分支并不保证改善覆盖率。
没有为了维持旧分支的 10/10 放宽检查或换一批候选。

实机复用这批已冻结候选的糖盒命令：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_matched_ten.py \
  --reflex-run artifacts/reflex_ten_runs/20260913T230536_911538Z/summary.json \
  --pool-report artifacts/ggx_matched_ten_runs/sugar_offline_pipeline_01/pool/summary.json \
  --capture-live --execute --allow-high-force
```

不需要重复推理或逐个观看 100 个视频。每点一次明确覆盖该点候选搜索的控制器预检授权，
随后只显示可执行路线供最终确认；安全否决可看下一项，取消则结束。
操作者仍须确保是相同物体、同一放置面和朝向、台面/安装不变，路径及线缆畅通。

新增测试覆盖按排名切换、一次物理尝试、取消/异常停止、XY 同量平移、hold、续跑和授权范围。
新腕分支另有两项测试，验证只改初值且不改变任何 TCP，并拒绝朝向或拍摄初值不一致。
新入口专项测试 16/16；全套 1099 通过、下述两个既有失败。
`sim-hand` 导入检查通过。全套测试仍有两个既有失败：拇指 tip site 两模型相差 3 mm，
以及旧模板创建单测缺少 `approval.json` fixture；本次未改模型或旧测试来消除它们。

更换此协议后不要用 `--resume-from` 混接之前旧腕版本的运行；旧结果仍保留，
新运行可用原 `--pool-report` 复用候选并重新计算路径，无需重新推理或更改排名。

## 已完成抬升后的复位修复及本轮续跑

糖盒实机 `20260914T031245_708440Z`：P1 的 rank 17 已记录用户成功，抬升/保持命令完成；
P2 尚未抓取。旧复位强制再向上 80 mm，在控制器 IK 预检中被拒绝，
日志确认未发出张手或机械臂运动命令，控制器清理正常。

新规则仅在有完成 80 mm 抬升/保持的记录、当前姿态仍处于该终点且足够高时，
省去重复上退；先预检完整 MoveJ 返回路径，再经本次复位确认执行张手/返回。
无此证据的初始复位仍沿用原上退规则。姿态不匹配时停止，不用高度猜测替代。
记录不证明张手不会掉物；操作者必须先确保物体已有支撑及返回路径、线缆畅通。

从保存 P2 入口姿态到新拍摄预设的张手名义 MoveJ 路线离线检查：149 个采样，
最小桌面间隙约 7.60 mm。未连接硬件；实际控制器解仍需现场预检，未认证未建模障碍。

尚未手动改变机械臂姿态时的续跑命令（保留 P1，不重试成功点；手动移动后使用下节入口）：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_matched_ten.py \
  --resume-from artifacts/ggx_matched_ten_runs/20260914T031245_708440Z/summary.json \
  --resume-after-reset-fix \
  --capture-live --execute --allow-high-force
```

迁移开关只允许指定旧版本的 `ggx_reset.py` 和 matched 入口发生变化；
其余历史输入、模型和配置必须仍匹配。仅控制器预检失败且无手/臂命令、清理正常的复位
允许续跑；部分动作或状态不明确时仍停止。
修复改变了源代码签名，迁移会用**原候选文件**重新完成局部筛选，不重跑 GGX 推理或更改排名，
不重新签名旧批准，不继承旧路径通过。
日后复用修复前的池也需重新筛选同一批候选，不能直接用旧 `--pool-report` 冒充当前检查。

## 手动移动后的 P2 恢复

`20260914T032339_934780Z` 返回预检通过，但张手阶段发生 0.104211 mm 偏移，
严格旧起点检查中止；日志有张手命令，无机械臂运动命令，清理正常。
随后操作者手动移动，`20260914T032537_430533Z` 对比旧 P1 抬升终点相差
280.855 mm，进入控制器前停止。这不是当前位置离拍摄位的距离。

操作者已确认手完全张开、物体放在桌上、返回路径和线缆畅通，可使用：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_matched_ten.py \
  --resume-from artifacts/ggx_matched_ten_runs/20260914T032537_430533Z/summary.json \
  --resume-after-reset-fix --recover-from-current \
  --capture-live --execute --allow-high-force
```

此入口保留 P1/rank 17 的用户成功记录，从 P2 开始。仅第一次复位不发送张手命令、
不额外上退，也不对比旧抬升终点；读取当前静止姿态，要求位于配置的高位区域，
重新预检返回路径并要求现场 Enter。不会因为手动恢复而豁免返回路径检查。
后续每次抓取完成后恢复正常复位逻辑。未建模的物体、夹具、相机架和线缆仍需现场确认。

正常复位的张手阶段使用已有到位容差（1 mm / 0.5 deg / 0.003 rad）约束同一个固定起点，
不逐帧重置基准；张手稳定后重新计算和预检全部返回路径，再用原严格起点检查确认后运动。
超过该张手阶段范围仍中止；未更改全局阈值、抓取路径或碰撞模型。

显式人工恢复只接纳无机械臂命令且清理正常的失败复位；部分抓取执行、已发出臂运动命令
或清理状态不明仍不可自动续跑。源代码迁移仅接纳列明的旧版本，原始实验报告不覆盖。

此次验证：复位/续跑专项 56/56 通过，`sim-hand` 导入通过；全套 1116 通过，
仍为前述两个既有失败（拇指 tip site 差异、缺失 approval fixture）。
只读核对最新续跑参数确认保留 P1 成功、下一点 P2、旧抬升目标为空，
共享点云和缓存官方筛选签名匹配；未连接相机、手或 UR5，也未验证当前实机返回路线。
