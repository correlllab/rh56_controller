# 同一视觉输入的 Thumb Reflex：离线接入记录

日期：2026-09-13。当前是开发预览，不是实机执行入口，也不是新的成功率结果。

## 当前恢复：最初的四指 Reflex，盒子已转回原方向

用户决定不采用侧抓，回到最初版本，并已在现场转回糖盒。CLI 默认恢复为 `--fingers 4 --grasp-direction legacy-top`：原始 plane4（拇指加三指，小指张开），观测顶部下 10 mm，保留原 planner 的自动倾角与短轴对齐，不增加侧抓 roll，也不使用 80 mm 侧面接近点。仍为拇指预备 → 直接 moveL 到目标 → 其余手指闭合，再预览 lift。

侧抓开发的代码和报告仅留作显式选择的历史诊断，不是当前实验策略。恢复规则而不是重用旧空间坐标：新拍 9 帧 RGB-D 在 `artifacts/reflex_original_restored_20260913/capture/`，新糖盒为 #28；perception、crop 及 preview 同目录保存。未发手或机械臂指令。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/prepare_reflex_comparison.py \
  --crop-report artifacts/reflex_original_restored_20260913/crop/summary.json \
  --fingers 4 --grasp-direction legacy-top --point P2 --video
```

这是预览命令，不支持 `--execute`。新点云尚未运行 GGX，不能声称新场景已完成两方法 matched evaluation。新增回归测试确认默认四指、原姿态旋转不变、顶部下 10 mm，以及不存在额外侧面接近点。

本轮 40 项定向测试通过，sim-hand import 通过。新观测短边 39.8651 mm，原 solver 解为 40.3841 mm。P2 名义序列完成，采样桌面间隙最小约 32.27 mm；接近段与目标代理最深重叠约 3.54 mm，闭合段约 9.89 mm。仍是接触未经验证的离线诊断，不是实机放行。

## 历史：直接对齐盒子侧面，不强求两个指尖严格等高（已停止采用）

用户再次指出需要的是“hand line up with the box”。此前把拇指/食指严格等高作为硬约束，产生了斜对盒子的姿态。默认现为 `--side-roll box-aligned`：模型手掌法向 +X 平行盒子短轴、指排 +Y 指向世界上方、手腕到手指的 +Z 平行盒子的水平长轴。从靠近底座的一端接近。不根据 IK/碰撞结果调整方向，不再通过额外 roll 求指尖高度相等。

保持五指闭合值、抓取中心、尺寸和模型不变。因原模型的手指几何，指尖不必严格等高：本次拇指约 141.60 mm、食指 150.24 mm；不为消除这 8.64 mm 差异再次倾斜整只手。

新增 `--pose-only`，仅用名义最终姿态 IK 渲染近景侧视与俯视图，不生成十起点路径，不做碰撞放行。输出在 `artifacts/reflex_comparison_previews/20260913_box_aligned_pose_views/`；失败的首次离屏画布尺寸尝试也保留在 `20260913_box_aligned_pose/`。这些静态图不是成功抓取或安全证明，手与物体的真实接触尚未验证。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/prepare_reflex_comparison.py \
  --crop-report artifacts/reflex_side_capture_20260913/crop/summary.json \
  --side-roll box-aligned --pose-only
```

39 项定向测试通过，包含盒子三轴对齐、原闭合值和抓取中心不变。没有相机重拍或硬件动作；未修改旧实验归档。

## 历史：参考照片，保持原手腕轴，只做 wrist roll（用户仍不认可）

用户提供的照片明确了手腕从侧面伸向盒子的姿态。此前只约束指尖高度仍不够：旋转 GUI 抓取平面时，叠加的原 closure tilt 会改变手腕朝向。当前默认 `--side-roll photo-side` 以原五指、点云短轴对齐的手部旋转为基准，**严格保持其 local +Z（手腕到手指的纵轴）不变，只右乘 local Rz(roll)**，再转换回 GUI plane 参数。没有把 RGB 参考照片当作标定，也不再根据碰撞分数选择朝向。

测试除了拇指/食指等高和其余手指向下，还检查 `R_reference.T @ R_new` 只有 Z 轴旋转，以及手/TCP 的纵轴完全不变。中心、原始闭合指令、五指模式、起点和已有接近段数量不变；手腕位置会随围绕抓取中心的旋转正常变化。历史 photo-side 的第一个试算曾强制水平 heading（`20260913_photo_side_P2`），那不满足“保持原朝向”，已被此版取代，未用于实机。

正确约束的预览在 `artifacts/reflex_comparison_previews/20260913_photo_fixed_heading_P2/`。原轴相对水平向下约 20.3°，此倾角保持不变；roll 等价于 +49.26°（该版旧报告写为 -310.74°，仅角度表示相差一圈，不是运动指令；现代码输出已规范为 ±180°）。这是当前模型中达到指尖等高所需的值，不声称从单张照片量出了精确 90°。

拇指/食指预测高度均约 140.78 mm。P2 名义 IK 完成，采样桌面间隙最小约 24.18 mm；但准备/接近存在目标代理重叠（接近最深约 30.34 mm），**仍不是实机放行或成功抓取**。没有改阈值或代理尺寸。35 项定向测试通过；sim-hand import 检查通过。未连接硬件。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/prepare_reflex_comparison.py \
  --crop-report artifacts/reflex_side_capture_20260913/crop/summary.json \
  --fingers 5 --grasp-direction side --side-roll photo-side \
  --point P2 --video
```

## 历史纠正：仅约束拇指与食指等高（不足以保持手腕朝向）

用户明确所需排列是 **thumb Z = index Z > middle Z > ring Z > pinky Z**，不是“手腕在上”或“拇指独自在上”。默认 `--side-roll` 现为 `thumb-index-top`；旧 wrist-above/below 参数仅保留用于重现先前错误理解。

本次只在原侧抓平面内旋转姿态，让拇指/食指指尖等高、其余三指在下。仍调用原 GUI 姿态变换，不改 closure solver、原始手指指令、抓取中心、P1–P10 或已经生成的侧面接近规则。此前提出撤掉接近点的想法未实施，本次按用户“只改方向”的澄清保留已有 80 mm 接近段；它仍不是原单段 Reflex 路径协议。

原错误侧抓已经包含多次旋转，不能机械地在其角度上再加 90°就声称指尖等高。这里用指尖排列作为验证条件；相对上一错误侧抓的平面内修正角记录为 142.176°，不将它冒称为单独的 90°旋转。

新预览：`artifacts/reflex_comparison_previews/20260913_thumb_index_top_P2/`。
模型指尖 base Z：拇指/食指均 145.685 mm，中指 125.247 mm、无名指 103.308 mm、小指 80.811 mm。此为模型预测，不是实测。

**未通过实机条件**：P2 准备段/接近段仍有桌面代理重叠，最小间隙约 -22.1 mm；闭合目标代理重叠约 23.2 mm。保持诊断记录，没有修改碰撞阈值、模型或放行条件。31 项定向测试通过，覆盖指尖排列、抓取中心和原始闭合指令不变。

当前预览命令：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/prepare_reflex_comparison.py \
  --crop-report artifacts/reflex_side_capture_20260913/crop/summary.json \
  --fingers 5 --grasp-direction side --side-roll thumb-index-top \
  --point P2 --video
```

## 历史版本：五指侧抓与新场景（方向理解随后被纠正）

当前默认改为 `--fingers 5 --grasp-direction side`。旧四指预览保持原样；重现旧策略需显式指定 `--fingers 4 --grasp-direction legacy-top`。

侧抓仍用原 `PLANE_5F` 求解及 GUI 的 Plane Rx/Ry/Rz 变换：夹取方向沿点云短轴，抓取平面竖直。固定策略为观测高度的 0.5 处、手腕位于抓取中心上方；短轴正反方向以手腕靠近机器人底座选择，不根据 IK/碰撞结果筛选。`--side-roll wrist-below` 仅提供另一翻转方向的诊断，不代表允许实机执行。

在 P1–P10 不变的前提下，侧抓增加一个距最终位置 80 mm 的侧面接近点，最后一段保持姿态、沿抓取平面的水平法向接近。**这是新增的 approach policy，不是原 Reflex 单段 moveL 的原样回放**；手指仍遵循原来的拇指先摆位、其余四指后闭合。若将此策略用于论文对比，需与 GGX 的路径条件明确匹配或报告差异，不能悄悄沿用旧单段路径协议。

用户重新摆放糖盒后，已用相机单独采集 9 帧 RGB-D：`artifacts/reflex_side_capture_20260913/capture_host/`。#31 是糖盒，保持直立；foreground crop 保存 17488 点，观测短边 44.4636 mm，五指解宽度 46.3249 mm（残差约 +1.86 mm）。宽度没有人为修回 40 mm。深度/模型误差尚未消除，不能保证闭合实物。

新拍摄可以直接用于 analytical 开发预览，无需为了获得点云额外跑 GGX：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/prepare_reflex_comparison.py \
  --crop-report artifacts/reflex_side_capture_20260913/crop/summary.json \
  --fingers 5 --grasp-direction side --side-roll wrist-above \
  --point P2 --video
```

`--crop-report` 检查来源摘要和深度/配置/标定/前景哈希，记录新的点云哈希；**没有新 GGX inference**，因此报告 `ggx_input_match_verified=false`，不是已完成 matched comparison。后续 GGX 应复用这份点云。

方向开发过程均保留：`20260913_fresh_side_P2*` 是最初“仅靠底座”翻转、腕部穿桌的失败诊断；`20260913_fresh_side_wrist_above_P2` 是未消除轴正反歧义、P2 到侧面接近点 IK 未完成的诊断；当前规则见 `20260913_fresh_side_base_facing_P2`。这些不是抓取试次，也不能计入成功率。

当前 P2 虽完成名义 IK，但**仍未通过运动条件**：到侧面接近点的途中桌面间隙接近 0 mm（低于原 2 mm 要求）；随后水平接近在约 36.6% 处出现拇指与代理重叠，闭合阶段最大代理重叠约 25.3 mm。未豁免这些结果，也没有继续调整规则追求通过。视频用于确认所需方向，不用于实机动作授权。此版定向测试 28 项通过，sim-hand import 检查通过。

以下章节描述最初四指版本及仍适用的执行边界。

## 协议边界

保留原 supplied-geometry benchmark；其历史试次数和成功率应从原记录核对，不从本预览推导。新增比较拟使用 5 个物体 × 10 个固定起点 × 2 种方法，即 **100 个计划实验条件**。路径拒绝、未选出候选不计为已执行的 physical trial。

两方法共享保存的 RGB-D、分割点云、固定 P1–P10、物体摆放/XY 重定位规则以及 lift/hold 和成功标准。每物体每方法只冻结一个模板。GGX 保留其原 top-100 排序与选择记录；analytical 由同一份点云估计几何，不人工单独修正宽度或质心。本脚本尚未实现 live 重定位、hold 或结果录入，不能称为已经完成 matched 实机流程。

## 原 Reflex 实际做什么

复用 `GraspExecutor.execute_thumb_reflex` 的位置控制序列：

1. 拇指弯曲和对掌先到最终指令，其余手指张开。
2. 等待 0.2 s。
3. 机械臂 blocking moveL 到最终抓姿。
4. 发送其余手指的最终闭合指令，等待 0.25 s。

此实现**不是触觉反馈触发**的 reflex。write-only 模式可运行这个位置序列；`force_N=0` 禁止可选的自适应力阶段，不表示设备 raw force 为零。当前实验配置的 raw speed/force 都是 1000；本离线脚本只记录，不发送。若论文中旧 reflex 指的是另一种反馈策略，必须先核对，不能默认等同。

使用原视觉 planner 的默认 `PLANE_4F`（拇指加三指，小指张开），不是强行改成 GGX 的五指手型。两方法使用相同初始张手状态，再执行各自方法的预备动作；这些差异要报告。

## 实现及输入

入口：`tools/prepare_reflex_comparison.py`。没有 `--execute`，不连接相机、串口或 RTDE。

- 验证已保存 GGX inference、点云及 crop 的来源和哈希，读取同一个点云文件。
- 复用已有 footprint 最小面积矩形的轴和 observed bounds；不是新加 PCA。
- supplied dimensions 仅沿用为两方法共享的碰撞代理，不用于 analytical 目标宽度或中心。
- 使用原 closure solver、antipodal center、手到 TCP 变换和 raw command 转换。
- 抓取深度默认是观测顶部下 **10 mm**：这是待预先固定的策略参数，不是测量真值，也没有按通过率调优。
- 不继承以前方块人工调的 width/closure/XY 偏移或局部安装修正。
- 保持现有 CB3、RH56、桌面和物体代理模型不变，不缩小代理或放宽阈值。
- 输出 `summary.json`、`summary.csv`、目标、固定起点、逐点轨迹和可选视频；所有结果不可执行。

离线命令（需本机已有保存场景）：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/prepare_reflex_comparison.py \
  --ggx-run artifacts/ggx_quick_preview_20260912/fresh_scene \
  --point P2 --video
```

去掉 `--point P2` 可检查全部十点；去掉 `--video` 可避免渲染。脚本支持 `--help`。

## 糖盒初次结果

输入是原糖盒 GGX 的同一份 11054 点云。测得短边 38.2247 mm，原求解器得到 38.6327 mm（残差 +0.4080 mm）。观测长边 95.0438 mm、高度 179.5977 mm。它们是可见点云估计，不是完整物体真值。

- 十点均求得完整名义 IK 序列，**不等于十点碰撞通过**。
- P2 准备路径和拇指预备阶段未出现目标代理重叠；接近阶段首次重叠在路径参数 98.52%，为拇指末节，约 1.33 mm；最深约 4.51 mm。
- 闭合阶段代理最深重叠约 9.90 mm，涉及食指中节。不能仅凭几何重叠认定真实抓取失败，也不能当作安全接触放行。
- P1/P5 等还存在准备路径或起点的目标重叠，不能将所有问题解释为正常终点接触。
- 当前预览是运动学回放，物体固定；lift 时物体不会随手移动，因此没有模拟抓取成功、滑落或真实抬升接触。
- 采样时间不是实机执行时间。没有相机支架/未知障碍或完整自碰撞认证。

结果目录：

- `artifacts/reflex_comparison_previews/20260913_sugar_shared_cloud/`：第一版十点诊断，代码哈希保留在报告中。
- `artifacts/reflex_comparison_previews/20260913_sugar_P2_video/`：增加首次重叠位置和 Reflex 标签后的 P2 视频与诊断。

这些是旧 GGX 论文证据归档之后新生成的开发材料，**不在此前冻结的 tar.gz 中**。

## 实机前还缺什么

此前 GGX 实机预检按 GGX 手型插值，不能直接认证 Reflex 的“拇指先到位、其余手指后闭合”。2026-09-13 的十起点入口现已使用逐阶段实际 Reflex 手型，并按用户指定允许接近/闭合/抬升时手指与目标接触；准备路径、非手指几何和桌面检查保留。详见 [原始四指 Reflex 十起点脚本](reflex_ten_starts_zh.md)。旧预览及拒绝记录保留，不回写；离线通过不代表已经实机验证。

此外，要核对实际手安装/指令到开口的一致性、原实验 Reflex 的定义，以及共用的 reset、XY 重定位、hold、结果记录流程。未完成这些前不发实机动作，不将本视频当作动作授权。

## 验证

`tests/unit/test_reflex_shared_perception.py` 覆盖原 write-only Reflex 的实际命令顺序和等待时间、禁止读取手反馈、拒绝反馈力模式，以及离线入口不能接收 `--execute`。几何及 IK 使用保存场景做离线诊断，不需要硬件。

本轮验证：sim-hand import 检查通过；Reflex 与已有 vision target 的定向测试 19 项通过。全量 pytest 为 **1010 passed, 2 failed**：

- `test_planner_and_h12_use_the_same_fingertip_sites`：现有 planner 与 UR5 手模型的拇指 tip site 相差 3 mm。这个坐标契约差异需要在接触解释前核对；本轮没有改模型，不能因此宣称 sim/real 或 planner/预检几何已经一致。
- `test_template_creation_does_not_check_scan_travel`：现有 GGX template 测试没有创建现在会被读取的 `approval.json`，触发 FileNotFoundError；本轮未修改该测试或模板代码。
