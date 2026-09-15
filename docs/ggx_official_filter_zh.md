# GGX 官方场景筛选与 CB3 检查

此模式保留原 GGX 候选及置信度排序，不重新推理、不更改抓姿，不新增碰撞求解器。
`run_ggx_experiment.py` 首次生成默认使用此模式；旧 `run_ggx_review.py` 的 legacy 模式保留用于复现。

最新冻结定位及规则统一见 [冻结回放报告](ggx_frozen_diagnosis_zh.md)。下文 49/10 等数字是旧准入规则的历史结果，
不是当前“目标局部检查也通过”的候选数，不能把两个版本的指标混用。

## 固定协议与归因

1. 官方 `graspgenx.utils.collision_filter.filter_colliding_grasps`：使用官方 Inspire
   `coll_mesh.obj`，手表面 2000 点、最多 8192 个场景点，默认距离阈值 **20 mm**。
   固定随机种子并保存实际采样点、代码版本和哈希。阈值不会随通过率自动调整。
2. 输入是标定 ROI 内的实测场景点云，只排除人工选定目标的连通区域，不排除其他障碍。
   缺失深度不补齐；记录有效深度比例。场景点不足报错，不能利用官方空场景的全通过返回值。
3. 官方检查仅针对候选终态与非目标场景的距离。它不检查完整机械臂路径、目标接触或抓取成功。
4. 官方通过项按原排名进入本地 CB3 URDF＋未修改 RH56 模型的抓姿/局部接近检查。
   初筛/执行复用同一有限桌面构造器。桌面 Z 使用固定标定平面在桌面范围内的保守上界；
   与目标碰撞盒底部 Z 分开，不随本轮物体包围盒高度改变。
5. 初筛不检查扫描位直达抓姿的参考行程。先求最终抓姿，采用记录在 `policy` 中的固定 16 个
   数值初值（不作为运动路点）。几何链长上界之外与数值搜索耗尽分开记录。
6. 每个找到的抓姿分支向后求局部接近，按固定 **80/60/40/20 mm** 顺序选取可用距离。
   这是所有物体共用的规则，不按批次通过率调参；20 mm 仅是协议下限，不代表已测安全净空。
   最终 GGX TCP 位姿、接近轴和排名不变。模板保存精确目标；执行用已有 IK 做两段求解，
   先粗定位再从近处细化到旧十点解算采用的 0.05 mm / 0.05 deg，关节等价角在限位内靠近本轮起点。
7. `geometry_eligible` 现在还要求同一路径通过执行侧相同的局部目标/桌面/工作空间谓词，
   目标代理不确定项不再进入正常审核；它仍不是整个实际行程自动通过。

**抓姿选择与路径许可分离。** 初次 Enter 只选已通过局部规则的模板，不授予运动许可。
每个 P_i 都重新检查当前位→P_i→预抓取→接近→闭合/抬升的实际路径，尝试固定规则下的 IK 分支。
任何仍未解决的目标代理接触都不会被放行；它是未执行/不确定，而非真实抓取失败。
几何参考记录显式 `geometry_reference_only`，不能直接转成运动指令。

GGX 配置不再继承 cube 的 X/Y 小方框或 TCP Z>=180 mm。
`workspace_policy: urdf-envelope` 从原 CB3 URDF 链长生成有限包络，保留 TCP Z>=0 的调试底线；
它不是“包络内都安全”的声明。实际全臂/手桌面检查、IK 限位、控制器预检、每轮确认和速度设置不变。
原 cube 配置没有改变。场景中的未知固定物/完整自碰撞仍是限制。

其他限制：未知障碍、完整自碰撞、张手扫掠、动态抓取均未被认证；官方采样距离也不是精确连续几何距离。
手与目标在闭合阶段仍允许预期接触。已有 GGX moveJ/moveL、闭合/抬升、十起点及相对位置不变。
官方筛选作用于首次参考场景的候选选择；十点循环重新测 XY 后的轨迹仍需本地检查和本轮确认，
不能把参考场景的一次官方通过称为所有后续移动路径的官方认证。

## 已保存数据的离线重查

不连接相机、UR5 或手，也不调用 GGX 生成网络：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_ggx_feasibility.py \
  --review-run artifacts/ggx_experiments/20260911T220624_787242Z/initial_ggx
```

换成另一轮的 `initial_ggx` 或原 `ggx_review_runs/<时间>` 即可。
主环境准备标定点云，独立 GGX 环境执行官方函数；无需给 GGX 环境安装相机/机械臂依赖。
默认 CPU/GPU 自动选择并记录，CPU 也可执行筛选；不需要加载生成网络的 checkpoint。

输出 `summary.json`、`summary.csv`、`feasibility.csv`、`official_scene_filter/summary.csv`、
排除的目标 mask、实际送入官方函数的采样点，以及每个本地候选的轨迹/净空/状态。
`--official-filter <已有 summary.json>` 可复用同一候选、同一阈值且哈希有效的官方结果。
`--video` 只渲染可选抓姿的局部参考视频，不授予实机权限。
`--candidate-scope legacy-path` 可复现旧扫描位路径初筛（仍采用当前明确给定配置）。
`--candidate-rank 53` 只复查原排名 53，用于回归诊断；不会重排候选，也不应当作正式实验的挑选规则。

单独准备/运行官方筛选：

```bash
.venv312/bin/python tools/run_ggx_official_filter.py \
  --inference <inference/summary.json> --prepare-only --out artifacts/scene_input_NEW

artifacts/ggx_lab/GraspGenX/.venv/bin/python tools/run_ggx_official_filter.py \
  --scene-input artifacts/scene_input_NEW/scene_input.json --out artifacts/official_filter_NEW
```

直接使用旧交互入口生成新候选时，加 `--feasibility-mode official-cb3`。
阈值可用该入口的 `--scene-collision-mm` 或单独筛选器的 `--threshold-mm` 显式声明；
改变阈值属于协议变更，须单独报告，不能挑选使当前物体通过的阈值。

## 修复前的 2026-09-11 保存批次检查

没有重新生成候选，也没有实机动作：

| 批次 | 官方拒绝 | 官方通过后本地拒绝 | 目标代理不确定 | 全部当前检查通过 |
|---|---:|---:|---:|---:|
| Mustard（22:06 UTC） | 8 | 81 | 11 | 0 |
| Sugar box（22:08 UTC） | 18 | 79 | 3 | 0 |

本地拒绝进一步拆为：Mustard 的 IK 57、工作空间 23、桌面 1；Sugar box 的 IK 55、工作空间 24。
这些不是实际抓取失败数。两批真实执行数均为 0，不能据此计算有意义的“已执行抓取成功率”。
现有批次结果表明，不能把之前的 100 个拒绝都归因给 GGX 官方点云筛选。

旧参考场景 `20260911T194809_266833Z` 的 rank 53 同时通过官方筛选和当前 CB3 检查，
并完成离线候选记录→CB3 路径生成验证。它只对应保存的旧场景，不是物体移动后的实机授权。

## 几何/路径分离后的离线复查

使用同一批候选和同一官方筛选结果，没有重跑网络或更改排名：

| 批次 | 官方拒绝 | 可审核抓姿/局部接近 | 其余几何/接近搜索未通过 |
|---|---:|---:|---:|
| Mustard | 8 | 49 | 43 |
| Sugar box | 18 | 10 | 72 |

两列“可审核”与旧表“全部路径通过”**不是同一指标**。其中 Mustard 48 个、Sugar 10 个仍有目标代理不确定性，
不能用这张表声称实机通过率提高到 49%/10%。Mustard rank 67/97 选择 40 mm，rank 72/77 选择 60 mm，
其余可审核项选择 80 mm，均由同一固定规则产生。

分别对两批排名最高的可审核项（Mustard rank 1、Sugar rank 0）跑了固定十点：均没有找到通过全部
实际路径检查的路线，已记录跳过及各分支原因，没有执行。作为有解回归，旧参考 rank 53 的 P2
通过了新的名义路径检查和连续运动阶段结构检查；不等于控制器或实机验证。

变更了模型使用代码、配置和协议；旧批准/模板的哈希可能过期，应重新生成并审核，不能手改哈希继续执行。

## 官方资产

本地 Inspire 碰撞网格此前只是 Git LFS pointer；本次已下载对应真实资产（约 1.2 MB），
没有使用占位/自造网格。首次安装可在依赖仓库中只获取该文件：

```bash
git -C artifacts/ggx_lab/GraspGenX/ext/gripper_descriptions lfs pull \
  --include=gripper_descriptions/assets/x_grippers/inspire_hand/coll_mesh.obj --exclude=
```

资产和实验输出位于 `artifacts/`，不应提交进主仓库。复现时需保留报告中记录的官方代码及资产版本。
