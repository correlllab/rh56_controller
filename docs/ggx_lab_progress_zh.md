# GGX 与通用桌面视觉：2026-09-09 进度

现有 iterative 成功配置和控制环境没有修改。本轮没有连接相机、UR5 或 RH56。
GGX 推理在 GPU 上运行，抓取验证只运行于 MuJoCo。

## GGX 已完成

使用 [官方源码](https://github.com/NVlabs/GraspGenX)、
[官方权重](https://huggingface.co/adithyamurali/GraspGenXModel) 和
[官方夹爪描述](https://huggingface.co/datasets/adithyamurali/gripper_descriptions)。
均保存在 Git 忽略的 `artifacts/ggx_lab/GraspGenX/`，独立环境为其中的 `.venv`。

- 源码：`b9429097728cb1c430dd78b92edf17ba318aad03`，与仓库旧实验记录一致。
- 权重：`7c834043c11a11417e31d6d5ea9355801e40a2c1`，release generator/discriminator。
- 夹爪描述：`19a03c00d19aeaf052d0f6801f0041982d676e8a`，`inspire_hand`。
- 本机 RTX 2070 8 GB，PyTorch 2.6.0+cu126 / CUDA 12.6 可运行。
- 包版本快照：`artifacts/ggx_lab/environment_freeze.txt`。安装使用独立 installer 的
  `uv pip install --python .venv/bin/python -e .`，没有修改两个 robot venv。
  `uv sync` 会解析不需要的 TensorRT 可选依赖，本轮改用基础包安装。

新增 `tools/run_ggx_cube_inference.py` 调用官方配置、采样和 Isaac-grasp 导出 API，
生成完整 40 mm 方块模型和固定种子的 3500 点输入，并保存点云、模型/资源版本、
候选哈希、`summary.json` / `summary.csv`。无可视化服务器、无硬件控制。
每次仅一次网络采样，不按仿真成功与否重试选候选。

```bash
artifacts/ggx_lab/GraspGenX/.venv/bin/python tools/run_ggx_cube_inference.py \
  --num-grasps 1000 --out artifacts/ggx_lab/cube_next

RH56_CACHE_DIR=/tmp/rh56_review_cache .venv312/bin/python \
  tools/run_graspgenx_success_comparison.py \
  --graspgenx-yaml artifacts/ggx_lab/cube_next/graspgenx_candidates.yml \
  --out artifacts/ggx_lab/cube_next/sim_nominal --error-mm 0 --directions 1
```

第一条必须用 **GGX 的 Python**，第二条用原 **MuJoCo 环境**。输出目录选新名称。
`--help` 不要求 GPU。模型候选处于完整 mesh 坐标系，不是相机/UR 坐标，单位为米。

保留的结果：

| 运行 | 采样 / 输出 | 结果 |
| --- | --- | --- |
| `cube_smoke_01` | 200 / 0 | 输入是 trimesh 数组子类，官方类型检查未转换为 Tensor；已在适配入口转成普通 ndarray |
| `cube_smoke_02` | 200 / 100 | 推理约 0.73 s；原规则选零基 rank 53，仿真未抓起（−0.6 mm） |
| `cube_1000_01` | 1000 / 100 | 推理约 1.14 s；原规则选零基 rank 65，GGX 仿真抬升 48.9 mm，analytical 为 69.5 mm |

时间不含加载模型；最后一行的两个方法各只测试一次零位置误差条件，不能据此宣称
成功率或稳定性。候选按原有置信度、从上方接近及桌面过滤规则选取，没有手选成功姿态。
1000 次采样是恢复旧实验预设，不是根据仿真反馈搜索超参数。未改 MuJoCo 模型。

后续仍需：CB3 实机运动学/IK（现有 GGX 桌面预检用 UR5e）、当前安装变换的适用性、
无手反馈时的 GGX 闭合规则。GGX 仿真用接触反馈停闭合，不能直接搬到当前坏手。
不能把 iterative 的腕部补偿路径改名当 GGX，也不能将其 −Y 5 mm 经验偏移泛化到所有姿态。

## 通用单物体视觉：只有离线预览，不可执行抓取

旧快捷脚本依然是 **40 mm 蓝色直立方块**：宽、高、yaw 并非自动识别。
新 `tools/preview_tabletop_target.py` 仅复用其标定、ROI 和 XY 工作区，使用桌面上方
的深度前景，不按颜色筛选，也不预设物体宽高。使用已有反投影、坐标转换和桌面 API。

```bash
.venv312/bin/python tools/preview_tabletop_target.py \
  --capture artifacts/iterative_cube_runs/20260909T202015_539047Z/capture_00 \
  --out artifacts/tabletop_preview/next
```

输出 `overlay.png`、`foreground.png`、`summary.json` / `summary.csv`。
可重复提供 `--exclude-roi X0 Y0 X1 Y1`，标出人工核对的固定夹具区域。
进入遮罩或 ROI 边界的物体被标记为不完整；多目标不会自动挑最大者。
结果使用独立 schema，始终 `executable_target: false`，不能交给旧抓取脚本。

已保存预览 `artifacts/tabletop_preview/cube_01/` 暴露了实际局限：桌面噪声成为第二个
候选，方块的可见包围框约 39.3 × 63.7 mm，高度分位数约 45.3 mm，尚不能作为
40 mm 方块的可靠尺寸估计。此失败证据保留，不修改旧标定去迎合结果。

现已在操作者移走夹具、手和物体并确认 ready 后，采集 **9 帧空桌背景**，保存在
`artifacts/tabletop_background/empty_20260909_01/`。原始机器记录与
`visual_review.json` 分开保存，背景图已目视检查。新的相机专用入口为：

```bash
.venv312/bin/python tools/capture_tabletop_background.py \
  --camera-serial 836612071918 --frames 9 --out artifacts/tabletop_background/next
```

需先人工清空目标区、保持相机/桌面/桌垫不动，再运行；不会连接机械臂或手，
不会改变激光设置。保存原始帧、时间中值深度、`depth_statistics.npz` 中的
有效帧计数和 MAD，以及标准 capture 元数据和汇总 CSV/JSON。中值仅接受至少
80% 帧有效的像素（九帧中至少八帧），不填补未知深度。

本次整体 ROI 支持率仅约 65.0%，原方块图像区域 `[680,250,780,320]` 为 96.6%。
支持像素的时间 MAD 中位数约 1 mm、95 分位约 2 mm，这是短期重复性，**不是绝对精度**。
不能宣称整个桌面都可可靠检测。下一步放回已知 40 mm 方块验证背景差分，
还需边缘深度异常过滤及多帧尺寸一致性检查。
现已增加显式 `--background` 的深度差分；空桌背景仍无法可靠排除之后移动进来的手。
确认预览选择和测量后再接执行，优先不透明、刚性、轮廓清楚的简单物体。
透明、反光、很薄、柔软或严重遮挡的物体不承诺支持。矩形短轴也不能未经转换就
当作原 planner 的 yaw；可见尺寸不等于完整物体尺寸。

### 放回 40 mm 方块的首次背景差分验证

相机专用采集入口增加 `--scene object`，将本次图像标记为物体测试而不是空桌，
生成 `rh56_tabletop_object_burst/v1` 记录。本次 9 帧位于
`artifacts/tabletop_objects/cube_20260909_01/`，仅采集相机，未访问 UR5 或手。

```bash
.venv312/bin/python tools/preview_tabletop_target.py \
  --capture artifacts/tabletop_objects/cube_20260909_01 \
  --background artifacts/tabletop_background/empty_20260909_01 \
  --out artifacts/tabletop_preview/new_cube_check
```

新检查只接受与背景一致的相机身份、图像参数、内参及深度设置；像素必须有有效
背景，且比背景近 `max(8 mm, 3 × 背景时间MAD)`。不填补未知背景，不按颜色选择。
这一阈值是对约 1–2 mm 时间波动留裕量的实验设定，不是普适传感器误差模型。
相机物理移动仍不能仅靠元数据检查检测到。

结果保留原始包围框（约 39.6 × 51.0 mm），不掩盖深度边缘导致的过大尺寸。
另增 `top_patch`：将前景向内腐蚀 5×5 像素，以内部点的高度中位数/MAD 估计表面，
仅保留高度差在 `max(3 mm, 3 × 内部空间MAD)` 范围内的点，再测其可见矩形。
此方法假设近似水平顶面；圆弧、斜面等不能直接按该矩形当作实际夹取宽度。
算法不读取 40 mm 标称尺寸来生成测量值。

| 检查项 | 结果 |
| --- | --- |
| 9 帧时间中值图的顶面尺寸 | 38.65 × 41.28 mm，高 39.30 mm |
| 各原始帧唯一几何候选 | 9 / 9 |
| 单帧短边范围 | 37.86–40.12 mm |
| 单帧长边范围 | 41.30–44.19 mm |
| 单帧高度范围 | 38.45–40.33 mm |
| 单帧三项尺寸相对 40 mm 的最大偏差 | 4.19 mm |
| 原始帧表面中心最大两两距离 | 1.84 mm |

可视化：`artifacts/tabletop_preview/cube_background_01/overlay.png`。
逐帧及汇总：`artifacts/tabletop_preview/cube_background_raw_frames_01/`，包含
`consistency.json`、`validation.json` 和 `summary.csv`。
这组数据既用于开发也用于检查，只有一个位置/物体，不是独立泛化验证。
40 mm 来自操作者之前的标称尺寸；中心稳定度不代表基座坐标绝对定位精度。
空桌原始帧对其所属背景中值的检查没有可接受候选，但也不是独立留出测试。
近方形顶面的短/长轴可能互换，新增 `yaw_ambiguous` 标记，不直接传给 planner。

所有结果仍为 `executable_target: false`；还没有修改 iterative 的颜色/尺寸输入，
没有发送抓取动作。下一步可在保持背景不变时验证另一种已知尺寸的不透明平顶物体。

### 平放 YCB sugar box：新物体检查失败及原因隔离

操作者提供长 × 短 × 高约 **176 × 90 × 40 mm**。相机采集 9 帧保存在
`artifacts/tabletop_objects/sugar_box_lying_20260909_01/`；RGB 已目视确认目标区只有
一个平放糖盒，没有手或夹具。未连接机械臂或手，未修改相机标定或抓取程序。

保持方块验证后的算法和参数不变，背景差分把同一盒子拆成多个片段：中值图有
两个可接受几何候选，另外两个片段越过旧目标 XY 工作区而被拒绝。
9 个原始帧也全部为两个候选（唯一候选 0/9），不是偶发单帧问题。
主片段顶面估计仅 165.02 × 80.84 × 43.06 mm；它不是完整物体测量，长、短边
分别低估约 10.98、9.16 mm。没有通过自动选最大片段来隐藏歧义。

原因隔离对照仅省略可选的逐像素空桌背景过滤，其余配置不变：同一中值图的顶面
估计变为 **177.16 × 88.85 × 43.39 mm**。目视圈定盒子区域内的深度前景像素
约 31.45% 缺少有效背景；原过滤只保留约 67.41%。这支持背景缺失造成片段断裂
及尺寸低估的解释，不能把未知背景直接当作“没有物体”。但也不能直接全面取消
背景过滤：此前它用于抑制桌面噪声。这项对照不是已经修复的通用检测器。

对照中的完整片段仍因 `outside_target_xy_workspace` 被拒绝，未放宽旧工作区。
尺寸接近标称值不代表绝对定位、夹取宽度、手部可达范围或抓取安全性已验证。
该物体是新的场景检查；标称尺寸来自操作者，不是独立精密测量。

证据目录：

- 原方法：`artifacts/tabletop_preview/sugar_box_lying_01/`。
- 逐帧与 `validation.json` / `summary.csv`：
  `artifacts/tabletop_preview/sugar_box_lying_raw_frames_01/`。
- 无背景过滤的离线对照：
  `artifacts/tabletop_preview/sugar_box_lying_no_background_diagnostic_01/`。

下一步应验证利用可靠背景变化确认物体、利用当前有效深度恢复其连通表面的处理，
而非用标称尺寸补齐边界；需同时回归空桌、40 mm 方块和糖盒，保留歧义拒绝机制。
旧 iterative 基线摘要校验通过，所有本次预览仍为 `executable_target: false`。

### 背景缺失修复：仅离线 supported-surface 模式

`preview_tabletop_target.py` 在提供 `--background` 时现在默认使用
`--background-mode supported-surface`，输出 schema 更新为
`rh56_tabletop_foreground_preview/v2`。可用 `--background-mode pixels` 重现前述
严格逐像素过滤。无背景输入时保留原来的桌面高度前景行为。

改动不是填补深度，也不是取消背景检测：

1. 像素仍必须具有当前实际测得的深度，并高于标定桌面阈值。
2. 背景已知且没有足够靠近变化的像素仍被排除。
3. 背景未知、但当前深度有效的像素可加入连通表面。
4. 每个表面至少要有 150 个可靠变化像素，且占表面至少 25%；否则记录
   `insufficient_background_change_support`，不得成为几何候选。

150 像素沿用前景最小面积尺度，25% 要求变化证据不能只是稀疏种子；两者均为
当前抗散点/弱证据的实验假设，未经过系统阈值扫描，不是概率置信度。新增
`background_support` 保存变化像素数、恢复的未知背景像素数、比例及阈值。
不使用物体的标称长宽进行拟合，不自动选择最大片段。相邻接触物仍可能连在一起；
此模式不支持语义区分手、夹具和目标，也不能保证恢复全部被遮挡的轮廓。

复现中值糖盒检查（输出目录必须不存在）：

```bash
.venv312/bin/python tools/preview_tabletop_target.py \
  --capture artifacts/tabletop_objects/sugar_box_lying_20260909_01 \
  --background artifacts/tabletop_background/empty_20260909_01 \
  --background-mode supported-surface \
  --out artifacts/tabletop_preview/new_supported_sugar_check
```

保存的三场景回归为 `artifacts/tabletop_preview/supported_surface_01/`，包含每场景
中值图、各 9 个原始帧及根目录 `validation.json` / `summary.csv`。

| 场景 | 中值图顶面长 × 短 × 高（mm） | 9 帧检查 |
| --- | --- | --- |
| 空桌 | 无合格目标 | 9/9 帧合格候选数为 0 |
| 40 mm 方块 | 41.39 × 37.90 × 39.19 | 9/9 帧唯一合格候选；尺寸最大标称偏差 3.54 mm |
| 平放糖盒 | 177.16 × 88.85 × 43.39 | 9/9 帧目视对应盒子的表面不再碎裂；尺寸最大标称偏差 3.35 mm |

糖盒原始帧长边范围 177.18–178.41 mm，短边 88.60–89.96 mm，高度
43.00–43.35 mm。**糖盒各帧仍因旧目标 XY 工作区检查被拒绝，合格候选数为 0**；
部分原始帧还存在工作区外的小噪声片段，不能声称整幅图始终只有一个前景。
这里的糖盒尺寸统计使用目视核对区域归属，仅用于诊断，不是自动选择目标。
原始包围框与顶面估计仍分开记录，不把深度边缘尾巴作为实际宽度。

这些是修改算法后复用同一批图像的开发回归，不是独立新场景验证。空桌原始帧
参与了背景中值构建，也不是独立负例；标称尺寸误差不代表基座 XY 绝对误差。
未放宽工作区、未改相机标定/手模型/原始抓取流程；旧 iterative 基线校验通过。

新增 14 项合成测试覆盖背景缺口恢复、当前深度缺失不填补、弱证据拒绝、未变化
夹具排除、多物体分离及边界检查等。相关测试共 36 项通过，sim-hand 导入检查通过；
全套 350 项通过、1 项已知失败：planner 与 H12 的 thumb tip site 位置相差 3 mm，
属于此前已有的模型差异，本次没有修改它。`--help` 和 `git diff --check` 通过。

下一步接入规划前还需检查自动尺寸/yaw 到原 planner 的约定转换和可达夹取宽度，
并区分视觉可见区域与机器人已验证工作区。不能直接把糖盒的约 89 mm 短边替换
进旧 40 mm 方块执行配置，也不能据此宣称 GGX 实机已就绪。

### 在相机图上查看原 Plan 抓取中心

新增离线入口 `tools/preview_tabletop_grasp_center.py`，要求明确指定目视核对过的
组件，不选择最大片段，不把视觉诊断伪装成已通过检查的执行目标：

```bash
.venv312/bin/python tools/preview_tabletop_grasp_center.py \
  --preview artifacts/tabletop_preview/supported_surface_01/sugar/summary.json \
  --component-id 7 --grasp-depth-mm 10 \
  --out artifacts/tabletop_preview/new_sugar_grasp_center
```

调用原 `GraspVizCore` / closure solver 和 `_compute_plan_closures`，逐航点检查
antipodal 中心不变，复用 `camera_calibration.project_target_points` 投影。
原始求解后的拇指—其余指尖均值方向经 yaw 旋转对齐测得短轴，而不是直接把
矩形角度当作 GUI yaw。短轴仍有 180° 歧义，拇指侧尚未经过碰撞/机械臂可达检查。
没有套用方块经验宽度/平移修正，没有修改原 solver、安装修正或执行程序。

结果 `artifacts/tabletop_preview/sugar_grasp_center_01/grasp_center.png`：
红十字为顶面下 10 mm 的原 Plan 中心（UR base mm：−216.60、−629.08、72.70），
黄圈为顶面中心，青线端点 T 为模型拇指端、F 为其余参与指尖的平均位置。
这不是腕部/TCP 位置，也不是 GGX 预测或实时图像。位置源自已保存的 RGB-D。

本次请求短边宽度 88.85 mm，solver 返回宽度 **95.58 mm**，残差 +6.73 mm；
旋转后拇指与其余指尖均值的水平间距为 96.41 mm。保留该差异，不通过改模型
或经验抵消隐藏它。可视化中心成立并不证明闭合宽度足够或所有指尖会同时接触。
糖盒的原工作区拒绝仍保存在汇总中，独立 schema
`rh56_tabletop_grasp_center_diagnostic/v1` 始终 `executable_target: false`。
输出 `summary.json` / `summary.csv`，无 `--execute` 参数；新增 9 项无硬件测试通过。

### 分离桌面视觉区域与机械臂工作区

在操作者同意后，新增独立默认配置 `configs/tabletop_vision_lab.yaml`，schema 为
`rh56_tabletop_vision_config/v1`。只包含相机、标定、桌面高度和图像 ROI，不包含
UR IP、手串口、力、速度、TCP 或抓取目标运动边界；硬件 cycle 的配置加载器
会拒绝此 schema，不能拿它直接执行抓取。

视觉 ROI 选为 `[320,60,1100,620]`（像素，右/下边界不包含），根据已保存的空桌
RGB 目视核对在桌垫内部，避开桌垫边缘和周边杂物。它没有扩大到整张相机图像，
也不宣称整个 ROI 已有小于 5 mm 的测量精度。改变的是不再把旧方块的基座
`200 × 250 mm` 目标范围用于截断大物体；机械臂运动范围和预检没有更改。

预览仍要求当前有效深度、足够背景变化支持、桌面高度条件、完整 ROI/排除区域
边界，以及唯一候选。输出增加 `vision_region`、配置 SHA-256 和
`robot_feasibility_checked: false`；仍是 v2 的非执行诊断，不转成旧抓取目标 schema。
图上青框清楚标注 `VISION ROI ONLY - NOT ROBOT WORKSPACE`。

```bash
.venv312/bin/python tools/preview_tabletop_target.py \
  --config configs/tabletop_vision_lab.yaml \
  --capture artifacts/tabletop_objects/sugar_box_pretrial_20260909_01 \
  --background artifacts/tabletop_background/empty_20260909_01 \
  --out artifacts/tabletop_preview/new_tabletop_region_check
```

前面所有使用旧范围的历史结果需要显式传
`--config configs/ur5_iterative_cube_lab.yaml` 来重现；逐像素背景版本另加
`--background-mode pixels`。旧配置路径不会隐式变成新视觉范围，已验证糖盒预试
数据在旧配置下的组件测量和越界拒绝均完全重现。

回归目录 `artifacts/tabletop_preview/vision_region_01/` 保存四组中值图、36 个原始帧、
根目录 `validation.json` / `summary.csv`：空桌 9/9 无合格候选；方块、第一组糖盒和
预试糖盒各 9/9 唯一候选。预试糖盒顶面仍为 178.93 × 89.97 × 43.75 mm，中心和
尺寸与范围调整前完全相同，没有人为移动或重新拟合它来通过检查。
这仍是已保存场景回归，不是整个桌面的独立精度验证。

新增 12 项测试覆盖纯视觉配置校验、与运动配置隔离、旧范围兼容，以及对象完整
可见时允许越过旧 XY 范围但仍拒绝 ROI 截断。相关测试共 57 项通过；全套 371 项
通过、1 项仍为此前的 planner/H12 拇指参考点差异。sim-hand 导入、`--help`、
`git diff --check` 和冻结的 iterative 基线校验均通过。未改原始执行程序或运动检查。

糖盒现在通过的是**视觉候选筛选**，不是机器人试抓预检。宽度映射差异、当前手
状态/开手路径、安装补偿外推及控制器 IK/路径仍须分别处理；本次没有连接或移动硬件。

### 糖盒宽度差异的定义核对（未改原 planner）

需要更正前面把 `solver_width_residual_mm` 理解为求解误差的说法：在原 plane4
实现中，输入是拇指—参考中指的 XZ 距离；共面修正后返回的 `ClosureResult.width`
是拇指—其余参与指尖均值的 XZ 距离。两者不是同一个量，后者也不是实测接触垫
间隙。原可视化字段保存的是二者的数值差，不应直接据此减掉 7–8 mm。

| 内部输入（mm） | 原参考指求解后距离（mm） | 返回的均值距离（mm） |
| --- | --- | --- |
| 48.0000（方块基线内部输入） | 48.0000 | 49.0258 |
| 88.8486（首组糖盒） | 88.8488 | 95.5751 |
| 89.9721（预试糖盒） | 89.9725 | 97.6855 |

因此，这不是相机把 90 mm 看成 98 mm，也不是参考指求解失败。另一个实际需要
注意的现象是：输入从 89.99 到 90.00 mm 时，无名指 control 从约 0.06924 跳到
0，返回均值宽度从 97.73 到 99.72 mm；90.1 mm 又返回约 0.06312。对应代码是
`coplanar_ctrls` 对非单调伸展端 Z(control) 的反求/最近邻回退，不能仅凭名义宽度
在可达范围内就假定整个多指姿态连续。90 mm 处模型所有参与指尖的世界 Z 跨度
约 19.81 mm；这是几何诊断，不等于实际碰撞或抓取失败的测量。

使用已有 sweep 工具复现，不添加另一套规划器：

```bash
PYTHONPATH=. RH56_CACHE_DIR=/tmp/rh56_review_cache MPLCONFIGDIR=/tmp/rh56_mpl \
  .venv312/bin/python tools/run_planner_sweep.py --modes plane4 \
  --width-min-mm 88 --width-max-mm 92 --width-step-mm 0.1 \
  --object-width-offset-mm 0 --no-plots \
  --out artifacts/tabletop_preview/new_sugar_width_sweep
```

本次输出 `artifacts/tabletop_preview/sugar_width_audit_01/existing_sweep/`：41 个
输入均在名义参考指范围内，输入与返回宽度的 1 mm 差值检查均未通过；这不是
41 次真实抓取失败。分阶段数值保存在同目录上一级的 `width_definition_audit.json`。
注意 sweep 的 offset=0 只是避免预设接触垫修正，并非证明 site 距离等于物体宽度。

原方块的内部 55/60 mm 对应实测约 45/50 mm 空隙，只验证了局部范围，无法独立
确定糖盒该用何种内部输入。不直接套用固定偏移，也不改变已成功的方块基线。
下一步应先在空手且路径确认畅通时做目标宽度附近的空中间距核对，再决定试抓
输入；不需要因此重新做相机或棋盘标定。本次仅离线诊断，未发送手/臂指令，
原 planner 和冻结基线未改动，基线摘要检查通过。

### 单姿态空中间距检查：等待人工测量

操作者确认空手且开手/上抬路径畅通后，使用 `tools/hardware/plan_airgap.py` 做一次
内部宽度 100 mm 的原 plane4 手型检查。机械臂保持原位，没有启动 RTDE 控制脚本。
100 mm 是明确选择的模型输入，位于前述 90 mm 跳变附近之外，并非声称真实开口
等于 100 mm 或已标定成糖盒宽度。返回均值 site 宽度约 106.97 mm。

离线预览：

```bash
.venv312/bin/python tools/hardware/plan_airgap.py --planner-width-mm 100 \
  --out artifacts/tabletop_preview/new_airgap_preview
```

执行额外要求 `--execute --confirm-empty-clear --start-snapshot <新只读记录>`，并在
每次写入前检查正常/静止 UR 状态、原 TCP 范围和起点一致性。单次先发送 force=100，
再 speed=100，最后一次角度 `[1000,1000,900,1000,900,0]`；不清错、不重试、不自动
恢复设置、不查询手的角度或力。无 ACK 时只能报告写出，不能确认设置生效或运动完成。

本次执行 `artifacts/tabletop_preview/sugar_airgap_execute_01/` 记录三次原始 TX；每次
RX 均只有单字节 `00`，不是有效 ACK。UR 只读监视等待 5 秒后正常结束，记录的
TCP 相对起点最大位移约 0.023 mm。当前状态为
`single_shape_sent_awaiting_manual_measurement`，不是手型达到或间隙测量成功。
需操作者观察完全停稳后，沿实际夹取方向测拇指接触垫内侧到中指/食指接触垫内侧
的净间隙，不顶压手指；若角度不便测量则先停止测量而非伸手调整正在运动的手。
手的目标和低设置会在脚本退出后保留；以后试抓必须显式设置所需参数。

新增 11 项无硬件测试通过；全套 382 项通过、1 项此前已有的拇指 site 差异失败。
sim-hand 导入和 `--help` 通过，原冻结方块基线校验通过。

### 糖盒整轮适配已准备；当前回扫描路径预检阻塞

操作者报告内部 100 mm 手型对应拇指—中指约 90 mm，原话与设置另存于
`artifacts/tabletop_preview/sugar_airgap_execute_01/operator_measurement.json`。
这是低设置空手的一次近似测量，不代表力恢复后或受载时的间隙。

新增 `tools/run_sugar_box_trial.py`，仅针对已知平放 176 × 90 × 40 mm 糖盒。
保持独立的视觉 ROI，要求唯一完整候选，尺寸各项偏差 ≤5 mm、顶面空间 MAD ≤3 mm，
已审核的近基座 Y 短轴方向 ±10°。实时执行采集三帧，中心最大两两距离 ≤3 mm、
尺寸极差 ≤3 mm、轴角极差 ≤2°；失败即停，不自动选最大片段。
目标中心仍接受原目标范围检查，TCP/旋转/IK/实际终点与抬升检查全部保留。

试验输入为名义 90 + 近似偏移 10 − 先前总闭合余量 2 = 内部 98 mm，不让视觉
尺寸的小幅噪声直接改变闭合输入。仅这一局部近似得到人工测量支持，2 mm 余量的
实际响应和高力受载状态仍未验证。顶面下 10 mm，位置经验偏移为 0；不把原方块
−Y 5 mm 偏移未经验证直接套用。使用冻结局部安装映射，记录新手型外推。

执行顺序、GuardedBridge、GraspExecutor、开手/停稳/回扫描、抓取和基于实际 TCP
的垂直抬升全部继承 `LiveCycle`，只覆盖视觉输入和 Plan 构建。未修改冻结文件。
对相同输入新增适配器与旧 Plan 的角度指令、TCP 航点、抬升结果一致性测试通过。
原实验速度 0.1 m/s、0.2 m/s²，手 speed1000、force1000，抬升 50 mm 后保持；
执行必须额外 `--execute` 并在完整风险/现场条件提示后按 Enter，实时重拍图像。

离线准备命令：

```bash
.venv312/bin/python tools/run_sugar_box_trial.py \
  --capture artifacts/tabletop_objects/sugar_box_pretrial_20260909_01 \
  --out artifacts/tabletop_preview/new_sugar_offline
```

`--preflight-only --confirm-controller-script` 仅授权控制器 IK/限位检查及脚本启停，
不执行开手、任何机械臂移动、抓取或抬升。读取控制器的本次预检
`artifacts/tabletop_preview/sugar_trial_preflight_01/` **失败**：原地上抬约 44.7 mm 的
段已检查，随后到扫描位的约 106 mm 直线/姿态插值段触发 `Controller IK changed branch`。
此时尚未发送开手或机械臂运动，也尚未完成糖盒抓取段的控制器预检。

不改变任何返回值/原阈值的额外诊断保存在
`artifacts/tabletop_preview/sugar_trial_scan_ik_diagnostic_01/ik_jump_diagnostic.json`。
问题采样间 TCP 只变化约 4.6 mm，腕部关节 6 变化约 0.216 rad（12.4°），超过
原相邻 IK 增量 0.15 rad（8.6°）阈值；关节 5 此时约 0.118 rad（6.7°）。
把差值按 ±π 归一化后结果不变，因此不是简单的 2π 表示回绕；腕部接近对齐是
一种合理解释，但错误标签本身不能证明离散 IK 分支跳变。没有关闭阈值或盲试运动。

当前整轮未执行，手仍保持上一次空中检查的目标和低设置。下一步建议单独评估
回扫描位的低速关节空间路线，以保留已保存的扫描关节姿态；这会改变复位的扫过
路径，需要操作者同意并重新预检/检查净空，不能仅因终点相同就直接执行。
糖盒抓取与抬升部分不因此改成另一套控制流程。

新适配器测试 19 项；与原 cycle 相关测试合计 148 项通过。全套 401 项通过、
1 项仍为先前 planner/H12 拇指 site 差异；sim-hand 导入、`--help` 和冻结基线校验通过。

### 糖盒回扫描位改为保存关节角 MoveJ（仅无运动预检通过）

操作者同意仅调整回位段后，`SugarCycle` 覆盖 entry prepare/preflight 和 move_scan。
未改冻结方块文件、模型、Plan 闭合、开手、抓取腕随动、停稳和实际位姿垂直抬升。
旧 Cartesian 相邻 IK 0.15 rad 检查仍用于原地垂直退离、抓取和抬升；没有关闭检查。

回位使用保存的原始关节角，不做角度回绕，MoveJ 速度 0.10 rad/s、加速度
0.20 rad/s²。这是明确的首次监督试验设置，不是测得的安全极限。
关节直线插值每步最大 0.01 rad，检查控制器关节安全、标定 FK、原 TCP 工作空间、
控制器位姿安全、原回位端点距离/旋转限制，以及保存关节与 TCP 的一致性。
记录全部样本；不声称验证整臂、手或环境碰撞。控制器启动前和预检后检查静止原点。
执行前从实际退离终点重新预检，异步 MoveJ 单次发送，监测状态/工作空间/终点/超时；
失败请求 stopJ，不重试。需 `--execute` 和提示后的 Enter，默认仍不访问硬件。

无运动预检结果：`artifacts/tabletop_preview/sugar_joint_scan_preflight_01/summary.json`，
状态 `entry_preflight_passed_no_motion`，`hardware_commands_attempted=false`，cleanup 无异常。
当前约 Z=298.6 mm，原地垂直退离目标 Z=343.36 mm；之后关节回位路线的采样法兰
范围 X=[−238.79,−226.67]、Y=[−263.09,−157.58]、Z=[343.36,354.57] mm（基座坐标，
不是手指到桌面的净空）。最大单关节行程 61.58°，预计回位约 11 秒。
整轮仍未执行，手保持先前空中手型和低设置。

`--preflight-only` 现在准确表示仅检查回扫描位，不把当前位姿假装成已到扫描位。
抓取控制器预检待实际回位、实时重新检测后执行，失败仍阻止抓取。
本次 161 项相关测试通过；全套 414 通过、1 项此前已有的 planner/H12 拇指 site
3 mm 差异失败。sim-hand 导入通过。机械臂模型及该既有差异未修改。

### 回位已实机完成；修复糖盒实时采集预热差异

经操作者批准执行 `sugar_joint_scan_execute_01`：开手指令、垂直退离、低速 MoveJ
回扫描位均完成，随后检测失败而停止，没有进入抓取/闭合/抬升；cleanup 无异常。
原始记录：`artifacts/tabletop_preview/sugar_joint_scan_execute_01/summary.json`。

失败帧编号 36，前景有 6 个候选，糖盒最大顶面片段仅长 147.54 mm。
保存图像中糖盒内部检查窗口 x=[560,838)、y=[205,335) 的有效深度比例约 56.4%。
发现实时脚本沿用默认预热 30 帧，而先前参考采集工具使用 60 帧。
相机激光150、发射器1、标定、视觉ROI与背景文件均未改变。

不移动机械臂、手或物体的相机诊断九帧存于
`artifacts/tabletop_objects/sugar_scan_vision_diagnostic_01`；60帧预热后九帧全部逐帧
通过现有糖盒目标条件，聚合图同一检查窗口有效深度约84.9%，顶面178.04×89.56×43.38 mm。
这支持启动深度尚未稳定的解释；并非受控重复30/60帧因果实验，也不证明任何场景
只要预热60帧就必定可靠。没有填洞、合并前景片段、降低尺寸或唯一目标阈值。

代码改动（仅视觉）：

- `capture_tabletop_background.py` 提取共享60帧预热常量，既有采集行为不变。
- `SugarCycle.detect` 使用该常量，先完整保存固定三帧再逐帧检测，每帧都必须通过；
  不选最好帧、不补采替代帧；执行模式仍检查UR静止状态与原检测时效。
- 新增互斥 `--camera-check-only`，使用完全相同的检测方法但不连接UR/手、不生成
  运动；每次写summary JSON/CSV、原始图、检测图和预热/帧编号/时间策略元数据。

两次独立启动的新实时检查均通过（每次三帧，不使用旧图）：
`artifacts/tabletop_preview/sugar_warmup60_live_check_01` 和 `_02`。
各组三维中心最大两两差分别0.573、0.776 mm；这是短期一致性，不是绝对定位误差。
第二组顶面尺寸长178.08–178.85、宽89.59–91.02、高43.41–43.50 mm。
默认和运动流程都没有自动重试整轮；本次修复期间没有UR/手连接或指令。

新增6项无硬件测试覆盖相机模式互斥、60帧配置、完整保存固定帧数、失败停相机、
不补采、三帧变化拒绝、实际执行仍检查UR静止。相关215项通过；全套420通过、
1项既有拇指site差异失败，sim-hand导入通过。方块冻结基线保持不变。

### 2026-09-10：切换 GGX，真实直立糖盒点云推理首次跑通

操作者明确暂停原 Plan 试验，改为检查 GGX。此次未连接或移动 UR5/RH56，未重拍
场景，使用刚保存的 `artifacts/tabletop_objects/sugar_upright_20260910_01` RGB-D。
原 Plan 和冻结方块基线没有修改；不是将 iterative 结果重命名为 GGX。

新增离线 `prepare_ggx_tabletop_cloud.py`：显式选择已目视核对的顶面 component19，
用物理尺寸短40、长90、高176 mm 限定朝向包围区域，XY 余量5mm；只保留真实
测得点，排除推定底面以上8mm以下和顶面以上5mm以外的数据。不补深度、不生成
隐藏盒子面，也不宣称完成通用自动分割。
`artifacts/ggx_lab/upright_sugar_real_cloud_01` 保存10,694点、RGB颜色、像素索引、
叠加图和来源哈希。可见Z范围122.43–219.49mm，只覆盖顶面与部分侧面，不能将
这97mm观测跨度误认为实际176mm物高。

扩展已有 `run_ggx_cube_inference.py` 的可选 `--point-cloud` 和必需配套
`--input-frame` 参数，默认完整40mm方块模式不变。真实点云输出使用独立schema，
保持输入UR-base坐标和米制单位；均值平移只恢复一次，不添加物体中心或iterative偏移。
不要将这些base-frame候选直接传给默认还会加物体中心的旧mesh-frame仿真比较入口。

使用原GGX独立环境、官方release生成器/判别器及Inspire-Hand描述，RTX2070执行
单次1000采样、top100、seed20260909。结果
`artifacts/ggx_lab/upright_sugar_real_inference_01`：100候选，推理1.114s（不含加载），
top100评分0.796–0.922。最高分候选接近轴约[0.126,0.124,−0.984]，主要向下。
PointNet++可选扩展加载警告仍存在，但此次配置为ptv3vanilla并成功完成推理。

新增 `preview_ggx_camera_candidates.py`，前3名原始置信度排序、不按碰撞/成功挑选，
用官方open-hand点云直接显示候选姿态，输出
`artifacts/ggx_lab/upright_sugar_real_preview_01/rank_00_overlay.png` 和
`top_candidates_3d.png`。黄十字是描述文件的sweep-volume中心，不是实测接触点；
叠加图不计算遮挡，蓝色点不是实际手指位置或闭合轨迹。

本次证实的是“人工限定真实点云→GGX网络→候选可视化”，不是实机抓取成功。
还需要完整对象/桌面碰撞检查、实际CB3运动学、GGX与安装坐标映射验证，以及针对
write-only手的闭合执行定义；不能直接沿用依靠接触反馈停止的GGX仿真闭合。

本次相关22项无硬件测试通过；全套441通过、1项仍为既有planner/H12拇指site
差异失败。sim-hand导入、各CLI `--help`、冻结基线校验和diff检查通过。

复现（各输出目录须为新目录）：

```bash
.venv312/bin/python tools/prepare_ggx_tabletop_cloud.py \
  --preview artifacts/tabletop_preview/sugar_upright_check_01/summary.json \
  --component-id 19 --dimensions-mm 40 90 176 --out artifacts/ggx_lab/new_real_cloud
artifacts/ggx_lab/GraspGenX/.venv/bin/python tools/run_ggx_cube_inference.py \
  --point-cloud artifacts/ggx_lab/new_real_cloud/object_points_ur_base_m.npy \
  --input-frame ur_base --num-grasps 1000 --topk 100 --out artifacts/ggx_lab/new_real_inference
.venv312/bin/python tools/preview_ggx_camera_candidates.py \
  --inference artifacts/ggx_lab/new_real_inference/summary.json \
  --crop artifacts/ggx_lab/new_real_cloud/summary.json --out artifacts/ggx_lab/new_real_preview
```
