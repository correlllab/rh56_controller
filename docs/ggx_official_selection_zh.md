# GGX 官方候选初审与本地执行分层

2026-09-12。此改动只整理离线候选审核与执行侧交接，没有修改生成器、评分、模型、碰撞距离阈值或真实手命令，没有访问硬件。

## 快速单入口：自动接上本地预览

追加 `--local-preview` 后，先自动做本地评估和视频生成，再显示一次 Enter；不用选择后再复制路径运行第二条命令。
本次按用户要求不跑测试套件，直接用保存场景和新拍的 RGB-D 跑通预览。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_review.py \
  --candidate-run artifacts/ggx_quick_preview_20260912/fresh_scene \
  --feasibility-mode official-only --local-preview
```

只重看本次第一个有本地预览的候选，不重新推理或遍历前面的候选：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_review.py \
  --review-run artifacts/ggx_quick_preview_20260912/fresh_scene/official_rank_015 \
  --local-preview
```

两种视频明确区分：

- `strict_nominal_path_ready`：保存起点到抓取、抬升的原有严格离线路径预览。
- `target_contact_warning_preview`：目标代理接触不挡住**观看**，保留负间隙和失败标记；只显示局部接近、闭合、抬升，不包含从当前/保存起点过来的路径。IK、桌面、工作空间或实际行程失败不会变成这类预览。

原排名与 top-100 不变；无法生成本地预览时记录本地原因后检查下一原始排名，不重新推理。`n` 看下一候选、`p` 重放、`q` 退出。Enter 只记录选择，不执行实机；警告不能被重命名为碰撞通过。

本次新拍照片在 `artifacts/ggx_quick_preview_20260912/current_capture/`，只访问 RealSense，没有连接 UR5 或手。
糖盒主体为 #39，深度顶面高约 175.7 mm，使用此前提供的 `40 × 90 × 176 mm` 竖放尺寸。文字区域另分为 #40，本次没有手工合并分割。
一次 GGX 生成保存了截断前全部 1000 个候选；top-100 中 94 个通过当前分割下的官方筛选。按原顺序首个本地可视化项为 rank 15，类型为目标接触警告，桌面模型最小间隙约 5.0 mm；不是实物安全间隙。
最终视频：`artifacts/ggx_quick_preview_20260912/rank15_current_code/local_rank_015/local_target_contact_warning.mp4`。
这是实际拍摄/推理/预览演示，不是实机部署验证；相机中没有手，也没有同步手型观测，不能用该照片解决 OPEN 映射问题。

## 哪一层决定什么

| 阶段 | 使用的内容 | 结果的含义 |
|---|---|---|
| 生成与排序 | 固定输入、官方生成器、原始 score 和 top-100；生成脚本保留截断前候选 | 不为通过率重排、补入第 101 名或自动换种子 |
| `official-only` 初审 | 现有官方 `filter_colliding_grasps`、官方 Inspire 聚合碰撞网格采样、排除目标后的实测场景点云；默认距离阈值仍为 20 mm | 官方**终态场景距离筛选**，不是目标接触或机器人路径证书 |
| 官方预览 | 保存的同一组官方手表面采样点、实测目标点云、非目标场景点云 | 不使用本地张开手型或包围盒替代官方手；未知遮挡区域没有被证明为空 |
| 单独本地评估 | 原有 CB3、手/TCP 映射、物体代理、固定桌面、同一 `select_geometry → exact_reference_rows → execution` | 本地不可执行或不确定，不能改写成“GGX 生成失败” |
| 实物验证与运动 | 尚待确认的实际 OPEN 手型、随后独立的本轮路径确认 | 官方初审 Enter **不授权运动** |

`legacy` 与 `official-cb3` 旧模式保留，旧审核记录不升级为新协议。
新模式的保存状态为 `official_candidate_selected`，scope 为 `official_ggx_candidate`，始终 `executable=false`。
本地拒绝不会自动另选一个“更容易通过”的官方候选；保留原 rank 与各层结果。

## 今晚：不拍摄、不推理，重看已保存候选

在仓库根目录运行：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_review.py \
  --candidate-run artifacts/ggx_review_runs/20260911T194809_266833Z \
  --feasibility-mode official-only
```

此命令复用该目录的 `simulation.yaml` 和 `inference/summary.json`，重新进行官方场景筛选；不继承旧人工批准。
不需要重新输入尺寸，也不需要 CUDA 推理。官方筛选仍使用独立 GGX 环境，GPU 不可用时其现有 `auto` 模式可在 CPU 做距离计算。
默认自动打开静态双视角 PNG；加 `--no-open` 则手动打开输出路径。

Enter 选择当前官方候选；`n 原因` 看下一项，`p` 重开图，`r` 表示场景已动，`q` 退出。
量化实验应事先固定选择规则，人工只作安全否决并记录原因，不能根据预期抓取成败挑选结果。
保存场景可用于离线回放，但不能据此声称明天实际场景仍然相同。

## 未使用快速单入口时：选择后单独评估本地执行层

把 `本次审核目录` 换为上一步打印的实际路径：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_real.py \
  --approval 本次审核目录/summary.json \
  --no-video
```

这里**不要加 `--execute` 或 `--preflight-only`**：后者需要真实控制器，不是纯离线模式。
默认起点来自保存的 `artifacts/vision_pick_start/start_pose.json`；可用 `--start-snapshot` 指定另一份保存的关节状态，不读取当前机器人。

输出目录中的 `local_assessment.json` 保留官方选择，并单独报告：

- `local_geometry_unavailable`：原有本地接近检查未通过；`geometry` 中保留 IK、桌面、代理接触等原因。
- `local_geometry_ready`：局部接近就绪；还需看 `actual_travel_status` 是否为 `nominal_path_ready`。
- 以上均不表示物理手型已验证、控制器已验证或抓取成功。

实机入口和十起点模板入口不会把这个候选范围自动升级为运动授权。原始模型/参数完全保留，因此之前被本地代理挡住的候选仍可能在**这一层**被记录为不可执行；当前改动没有声称修复实物误差。

## 明天需要完成的唯一新增现场事项

按 [GGX 张开手型实物确认](ggx_physical_hand_check_zh.md) 采集与当前 GGX OPEN 命令绑定的稳定手型观测。
已发现官方与本地张开手型不同，但现有半张开标定照片不足以确定哪套更接近实手。
先用同一套相机/安装变换比较，再决定应修正哪一层；不能因为某套模型放行更多便选它。
因此今天不盲改 raw 命令或关节限制，也不提供“忽略手型差异”的执行开关。

## 结果统计

分别保存官方筛选可用率、本地执行就绪率、已执行抓取成功率和总 trial 下的成功率。
不确定代理接触、IK 搜索耗尽、实际执行失败应分开记录；未执行 trial 不应直接被归为 GGX 抓取失败。

## 本次离线验证

产物位于 `artifacts/ggx_official_selection_smoke_20260912/`，复用了上述保存场景，没有相机采集、重新推理或人为批准。

- `reference_review/`：CLI 跑到首个官方候选预览后因无交互输入取消；完整 top-100 中官方筛选通过 89、拒绝 11。本地检查未参与此统计。
- `reference_rank53/official_rank_053/review_official.png`：原 rank 53 的官方手型双视角点云预览。
- `rank0_local_v2/`：官方通过仍被保留；本地独立报告 `local_execution_unavailable`，没有自动换候选。
- `rank53_local_v2/`：从保存起点完成纯离线接近、闭合、抬升路径准备，`offline_prepared`；仍不是控制器验证或实机抓取成功。
- 仿真环境导入检查通过；相关 225 项测试通过。全套 992 通过、1 项既有失败：`test_planner_and_h12_use_the_same_fingertip_sites`，两个既有 XML 的拇指 tip Y 相差 3 mm。本次未更改这些模型。

新范围的全流程实机执行尚未开放，也没有“确认一下就跳过手型差异”的参数。明天取得观测后仍需根据证据决定映射修正及验证方式；不能将今天的离线准备称为完整实机就绪。
