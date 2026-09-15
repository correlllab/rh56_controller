# GGX 一次生成 → 固定十起点实验

协议说明：本文现有实机入口仍是“选一个模板跑十点”。糖盒同批候选池复查已发现其限制：
名义路径覆盖可由 1/10 提高到 8/10。见[离线结果与新协议](ggx_frozen_pool_starts_zh.md)。
不要将本页旧入口误认为已经支持逐起点从 top-100 选择；
新共享 Reflex 起点入口见 [GGX matched 十轮](ggx_matched_ten_zh.md)。

主入口：`tools/run_ggx_experiment.py`。没有 `--execute` 不连接 UR5/手；
只有显式 `--capture-live` 才使用相机。输出：`artifacts/ggx_experiments/<时间>/`。
新物体重新生成一次 GGX；同一物体十点共用自动筛选出的抓姿、闭合序列和物体尺寸。

首次筛选现使用 **GGX 官方场景距离筛选（默认 20 mm）＋CB3 固定桌面检查**。
初筛只选抓姿，不要求扫描位直达路径通过。固定多初值 IK、80/60/40/20 mm 退让规则、
URDF 派生工作包络与每个起点的实际路径分开处理。目标包围盒接触单列为不确定，不进入正常初审；细节和离线重查见
[官方筛选协议](ggx_official_filter_zh.md)。十点定义、XY-only 重定位和实机确认不变。

自动筛选保持 GGX 原始分数、顺序和 top-100：逐个检查，遇到第一个严格局部通过项就停止，
不生成 100 个视频，不把目标接触警告当作通过。记录 `auto_candidate_selected`，
模板保留 `human_approved=false`；这不是人工批准，也不是实机路径通过或抓取成功。
每点执行前仍只看该点检查后的实际执行视频并按 Enter。无需再人工浏览候选。

需要在多个通过项中人工选择时，加 `--manual-select`：只展示严格通过候选，
Enter 选择、`n [原因]` 看下一个、`p` 重放、`r` 标记场景已变、`q` 退出。
仍按原始 top-100 顺序，不显示碰撞未通过的诊断候选。选定后十点使用同一抓姿，
不会在每个点重新挑选。人工选择/否决记录保留，实验应标注为人工辅助选择。
该选项用于新物体生成或 `--candidate-run`，不能修改已冻结的 `--template` / `--approval`。
例如重新拍摄直立糖盒并人工选择：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_experiment.py \
  --capture-live --dimensions-mm 40 90 176 --object-label sugar_box \
  --manual-select --execute --phase grasp-lift --allow-high-force
```

换物体须修改尺寸/名称并重新拍摄。相机支架尚未加入局部碰撞模型；人工选候选
不能替代现场对支架净空的核查。完整路径或复位路径接近支架时不要批准执行。

已有 rank 20 视频仅验证局部接近、闭合和抬升；从十起点过去的路径另查。
真实 OPEN 手型与模型的对应关系仍需现场确认，不能由碰撞通过或视频看起来好推断；
参见 [手型核对](ggx_physical_hand_check_zh.md)。下面命令不会替代现场确认。

## 复用已保存的这批候选（不重新推理）

下面先离线自动筛选并计算十点，不连接相机、机械臂或手：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_experiment.py \
  --candidate-run artifacts/ggx_quick_preview_20260912/fresh_scene \
  --official-filter artifacts/ggx_quick_preview_20260912/fresh_scene/official_scene_filter/summary.json \
  --object-label sugar_box --phase grasp-lift --no-open
```

这是保存图像/起点下的路径评估，不是当前物体位置测量。输出 `template.json`、
`fixed_layout.json`、`summary.csv` 和每个点的 `precheck.json` / `summary.json`。
同一糖盒、同一接触面和朝向，安装/标定未变时，可用输出模板启动下方的实机循环；
会重新确认复位、测 XY、检查当前起点和控制器路径，已有视频批准不继承运动授权。
更换物体必须重新生成，不要复用该糖盒模板。

2026-09-13 已完成此批实际离线串联：`artifacts/ggx_experiments/20260913_rank20_auto_ten/`。
无新推理，自动选 rank 20；P2 全路径通过，其余九点按原规则跳过。
P1/P5 在前往固定起点阶段不可用，其他跳过项在实际路线搜索中未找到通过解；
包括目标代理重叠或分支 IK 耗尽，不能一概记为真实碰撞或 GGX 抓取失败。
这不是十点均可执行的声明，也没有移动任何实机。
P2 的完整离线预览位于
`artifacts/ggx_experiments/20260913_rank20_P2_video/P2/cb3_review.mp4`。

## 开始一轮

以下以 **40 × 40 × 40 mm cube** 为例；换物体改尺寸和名称，不改起点配置。
尺寸顺序为短边、长边、总高度，单位 mm，仅首次用于碰撞代理模型，不用于捏造点云。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_experiment.py \
  --capture-live --dimensions-mm 40 40 40 --object-label cube \
  --execute --phase grasp-lift --allow-high-force
```

流程：

1. 检查整组固定起点；确认物体已有支撑，Enter 张手并返回保存的拍摄位。
2. Enter 拍照，选择物体编号。生成一批 GGX，按原始排名自动选第一个严格局部通过项。
   自动保存未批准的模板，不显示逐候选视频、不多问一次 Enter；没有通过项则停止。
3. 确认物体位置/朝向，Enter 拍照，只重新估计 XY。
4. 依次处理 P1…P10：从该点重选关节分支；本地路径搜索耗尽/几何约束不通过则记录并跳过；可执行的路径显示检查后的视频，Enter 执行。
   仍保留 RTDE 查询确认、实际起点/控制器检查；首次批准不代替后续各点的运动确认。
5. 抓取、抬升后保持。你先让物体获得支撑；下一次 Enter 才张手/退回拍摄位。
   放回同一物体，尽量恢复原朝向和接触面，再 Enter 重测 XY，继续下一个点。

最后一个点完成后保持，不自动松手，也不自动放下物体。`--phase approach-only` 只移动和接近。
当前配置仍为手 speed=1000、force=1000；按已有目标定时闭合，无手部反馈。
机械臂沿用连续 moveJ/moveL、速度和抬升逻辑，没有另写闭合控制器。

只试一个起点加 `--point P2`。同一物体复用模板，不再推理：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_experiment.py \
  --template artifacts/ggx_experiments/<上一轮时间>/template.json \
  --object-label cube --capture-live --execute --allow-high-force
```

这不是自动断点续跑，默认重新遍历十点。换物体、换接触面、相机/安装变化时不能套用旧模板。
也支持 `--approval <已有批准的 summary.json>`；依然重测 XY、检查每条新路径。

## 固定十点，不针对候选调参

配置：`configs/ggx_ten_starts_lab.yaml`。点间向量直接复用
`tools/run_strategy_pregrasp_rate.py:paper_approach_points_mm`，
历史约定为 `--approach-axis y- --start-reference grasp-center`，UR-base 轴，单位 mm：

| 编号 | 历史标签 | X | Y | Z |
|---|---|---:|---:|---:|
| P1 | P1 | 0 | -250 | 0 |
| P2 | P2 | 0 | 0 | 250 |
| P3 | L1_d-150 | -150 | -250 | 100 |
| P4 | L1_d-50 | -50 | -250 | 100 |
| P5 | L1_d+50 | 50 | -250 | 100 |
| P6 | L1_d+150 | 150 | -250 | 100 |
| P7 | L2_d-150 | -150 | -250 | 250 |
| P8 | L2_d-50 | -50 | -250 | 250 |
| P9 | L2_d+50 | 50 | -250 | 250 |
| P10 | L2_d+150 | 150 | -250 | 250 |

整组只平移，不旋转、缩放或单独挪点。张手形状和朝向固定，采用保存的拍摄姿态，
不跟随 GGX 或物体改变。抓取中心沿用已有 `antipodal` 指尖中点定义；
由于手形/朝向相同，TCP 与抓取中心拥有相同的点间向量。

P1 TCP XY 固定为 `[-230,-480]` mm，Z 根据固定桌面保守上界计算，
令模型拇指尖离桌约 25 mm，得到 P1 TCP Z 约 **189.16 mm**。
P2 TCP 约 `[-230,-230,439.16]` mm；P3…P6 高度 289.16 mm，P7…P10 高度 439.16 mm。
P1 整只手/机械臂最小模型桌面净空约 **6.80 mm**，不是整只手都有 25 mm。
全部起点检查通过后才进入流程；起点不合适会报错，不为某个物体自动挪点。

这是固定桌面和已安装手模型的离线结果，不是物理安全认证。未知夹具、张手扫掠和完整自碰撞
未完整建模，现场仍需确认。模型、标定或配置变化后重新验证整组。
旧版针对 rank53 平移搜索得到的高位起点已被此固定布局取代，不用于跨物体比较。

## XY 定位与跳过规则

放回后用目标区域 RGB 模板匹配估计小范围平移，不重分割、重拟合高度/尺寸或 yaw。
米制换算使用首次参考深度与标定射线，参考点高度固定，当前深度不参与定位。
相机仍保存 RGB-D，但 XY 拍摄允许当前深度无效；首次 GGX 点云仍需要有效深度。
默认搜索 ±60 px、XY 平移不超过 40 mm；找不到、歧义或质量不足会停止。
残余 yaw、遮挡、反光或姿态变化可能造成偏差，不能保证所有物体通用或毫米级精度。

抓姿只平移：XY 更新，物体 Z/yaw/尺寸、相对抓姿和闭合目标锁定。
连续跳过期间没有运动，沿用上次 XY；实际动作后必须复位、放回并重新拍摄。

只有**发出运动/手命令前、未收到运动确认的本地几何路径碰撞或具类型的规划搜索耗尽**可跳过，并记录阶段及全部原因：

- `position_to_trial_start`：拍摄位去固定起点途中碰撞；不是该点的抓取失败。
- `start_to_pregrasp` / `approach`：固定起点到抓姿的路径碰撞。
- 其他阶段按实际名称记录，不混为抓取成功/失败。

`skipped_planning_unavailable` 表示本地分支搜索未找到可执行路径，不等于证明无解；
`planning_attempts` 保存各分支的 IK、桌面、工作包络和目标代理不确定原因，不只显示优先标签。
控制器 IK/通信、定位、清理或运动中的错误仍停止，不会自动转到下一点。
全部点碰撞为 `all_points_skipped`（退出码 2），不是成功。
该流程为 GGX moveJ→moveL 策略，并非原论文直线 capsule 可达率实验；
点集相同不代表策略、朝向和碰撞模型相同。

## 输出与离线验证

保存 `fixed_layout.json`、`ten_starts.png`、`template.json`（或复用路径）、
总 `summary.json/summary.csv`、每点 XY 指标/overlay、轨迹、净空、碰撞阶段、
代码/输入哈希和运动授权记录。
`commands_completed_grasp_unverified` 只代表命令完成，真实成功需要人工记录。

纯离线检查，不开相机或机器人：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_experiment.py \
  --template artifacts/ggx_ten_start_runs/rank53_default_20260911/template.json
```

加 `--video` 保存仿真视频；`--trial-captures` 可指定各点保存图像。
未指定时明确使用参考图像，不能称为测得当前位置。
2026-09-11 rank53 保存场景离线结果：P2/P6/P10 通过，其余 7 点路径碰撞跳过。
十点固定桌面终点检查均通过；未为增加路径通过率调点，未执行实机。
两次较晚保存图像的 XY 匹配通过，平移约 0.2 mm；这只是图像一致性，不是物理精度验证。
rank53 此前只有接近动作获认可，未据此宣称抓取成功。

旧 `run_ggx_ten_starts.py` 保留为低层模板/坐标调试入口，默认也使用固定布局；
但不自动生成 GGX/测 XY/按碰撞跳过。其 `--origin-mm` 等覆盖参数不用于上述实验。
