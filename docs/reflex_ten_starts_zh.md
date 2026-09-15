# 两指/三指/四指 Reflex 十起点脚本

## 2026-09-14：两指覆盆子预览

### 最新：按已有标定斜面建模的独立重测

用户确认实际观察来自 5 mm 抓深、抬升前的拇指间隙，随后要求继续十点。按此前明确要求，本次抓深恢复 **10 mm**、额外净空 **0 mm**，但零距离（接触或距离未知）、穿透及非有限距离继续拒绝。独立配置 `configs/reflex_raspberry_calibrated_plane_lab.yaml` 不覆盖原 1.5 mm 配置。

旧碰撞桌面将整个有限桌面内的最高标定点铺成水平面；本物体附近标定桌高 39.804 mm，旧水平上界为 43.539 mm。新 `reflex_desk_surface: calibrated-plane` 保留原拟合法向量、物理测量的基座原点高度 34 mm、完整保守 XY 包络和 2 m 垂直厚度，以剪切棱柱表示倾斜顶面；没有根据观察间隙随意减高度，也没有移动物体或手。复位、离线预览和控制器执行复查使用同一个平面模型。点云、目标位置、两指指令及十点布局不变。此项是显式建模策略修订，不得回写旧实验或称旧拒绝均被证明是误判。

离线记录 `artifacts/reflex_raspberry_20260914/depth10_calibrated_plane_ten_01/`：P2/P3/P4/P5/P7/P8/P9/P10 通过名义检查，最低约 **0.97 mm**；P1 准备段穿桌约 1.145 mm，P6 上臂距离为零且无法证明分离，仍拒绝。没有实机动作。0.97 mm 是模型值，不是物理净空保证；原标定残差及当前安装误差没有因此消失。294 项相关回归、sim-hand 导入检查通过。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_reflex_ten_starts.py \
  --template artifacts/reflex_raspberry_20260914/depth10_calibrated_plane_ten_01/template.json \
  --config configs/reflex_raspberry_calibrated_plane_lab.yaml \
  --object-label raspberry --fingers 2 --grasp-depth-mm 10 \
  --round-object-yaw-deg 90 --round-object-approximation \
  --point all --capture-live --execute --allow-high-force
```

### 历史：水平包络、1.5 mm 净空的重测

最新覆盆子重测：用户明确要求深度 **5 mm**、桌面净空阈值 **1.5 mm**。独立配置 `configs/reflex_raspberry_lab.yaml` 仅新增 `reflex_desk_clearance_mm: 1.5`，其余内容与通用 Reflex 配置相同；默认/旧五物体配置继续为 2 mm。离线轨迹、控制器逐样本预检（含本配置的复位）和最终 Reflex 执行复查读取相同参数，并在报告及最终确认中显示。参数必须为有限正数，不能以零/负数关闭检查；本配置只用于该 Reflex 重测，不声明统一改变 GGX 的其他筛选层。

`artifacts/reflex_raspberry_20260914/depth5_margin1p5_ten_01/` 离线十点中 P2/P3/P4/P5/P7/P8/P9 通过，P1/P6/P10 仍因桌面问题拒绝。几何、抓姿、手指指令、点云及其他检查保持不变，未执行硬件。该 margin 是事后调参，不是实测安全保证；旧 2 mm、4/5/6/10 mm 深度记录以及 P2 实机失败全部保留，不能用本批覆盖。相关 Reflex/控制器/配置/复位回归共 286 项通过，sim-hand 导入检查通过。

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_reflex_ten_starts.py \
  --template artifacts/reflex_raspberry_20260914/depth5_margin1p5_ten_01/template.json \
  --config configs/reflex_raspberry_lab.yaml --object-label raspberry \
  --fingers 2 --grasp-depth-mm 5 \
  --round-object-yaw-deg 90 --round-object-approximation \
  --point all --capture-live --execute --allow-high-force
```

命令是新十点重测（包含 P2），需现场逐步确认；并非已有运动授权或跳过失败检查。

实机更新：深度改为 4 mm（抓点抬高 6 mm）后 P2 名义检查通过，用户确认 speed/force=1000/1000 并执行；`artifacts/reflex_ten_runs/20260914T210212_510691Z/` 记录抓取失败，用户明确确认“未抓住”，不重跑或改成成功。其余九点使用同一冻结模板加 `--points P1 P3 P4 P5 P6 P7 P8 P9 P10`。新参数仅选择原有起点，不重编号，不改变路径、检查或动作确认；本批 planned=9，P2 原记录单独保留，论文汇总时合并为十点。显式子集不能用旧的连续起点 `--resume-from`；如中断，应先核对已执行/部分执行记录，再用模板指定剩余点，不能盲目重试。

后续按用户要求令覆盆子近似盒子朝向机械臂：使用 `--round-object-yaw-deg 90 --round-object-approximation`，盒子 yaw 为 0°。新显式 approximation 参数允许操作者将非圆的测量投影视为无语义朝向的软物体近似；不伪改 `yaw_ambiguous`，不旋转原始点云，不改变默认方向策略。模板/续跑必须保留该参数。`facing_arm_P2/` 已解出全部名义 IK，但接近末段/闭合的 `thumb_proximal/geom_19` 最小桌面间隙为 -3.165 mm，仍未放行；中心、尺寸、深度和手指闭合量与原预览一致。

现支持 `--fingers 2 --grasp-depth-mm 10`，复用原 `GraspMode.LINE_2F`，仅拇指与食指参与，中指、无名指、小指保持张开；三/四指默认值不变。两指只接入原 `legacy-top` 流程，不支持侧抓变体。模板校验同时检查模式、活动指尖、非活动指的仿真值及 raw 张开指令。

覆盆子新观测保存在 `artifacts/reflex_raspberry_20260914/`，用户尺寸先验 25³ mm；观测外包尺寸约 25.98 × 35.25 × 27.40 mm，未缩放点云。`offline_P2_host/` 的 P2 名义检查因 IK 拒绝，视频仅供诊断；没有实机动作。手部力度/速度尚待确认，不把旧刚性物体的 raw=1000 视为覆盆子执行授权，也不把盒子几何预览当作压伤预测。两指相关回归及原 Reflex 测试共 61 项通过，sim-hand 导入检查通过。

入口：`tools/run_reflex_ten_starts.py`。默认离线，不连接 UR5 或手。此脚本有完整的实机分支，但本轮只进行了离线回放和假设备测试，**没有实机验证**。

## 当前糖盒的真实状态

**当前默认已修正为物体相关起点**：`configs/reflex_ten_starts_object_relative_lab.yaml`。原点是物体抓取点（顶面下 35 mm），P1 在 base +Y、即当前台面靠机械臂一侧 250 mm，P2 在原点正上方 250 mm。历史纸面 y- 到 base y+ 采用固定 180° 刚体旋转，横向也随之旋转；距离、顺序和点间几何保持不变。XY 重定位同时平移目标和十个起点，不旋转物体或候选。

最新记录：`artifacts/reflex_ten_runs/20260913_object_relative_depth35_v1/`。P1–P5、P7–P10 名义路径通过；P6 因桌面净空不足跳过。以下旧固定桌面原点记录仅作历史诊断，不能混作本协议结果。未执行实机。

GGX 十轮入口尚沿用其原固定布局，**当前不能声称两种方法已使用相同实际起点**。新模式仍保留共同张手/拍摄朝向，不等同于完全恢复旧实验各策略独立的起始手型/朝向，详见 `reflex_start_anchor_diagnosis_zh.md`。

最新 **35 mm** 离线记录：`artifacts/reflex_ten_runs/20260913_reflex_depth35_v1/`。P2–P10 名义路径通过；P1 仍因准备路径与目标重叠跳过。未访问实机；实际控制器路径仍需当轮预检。相关 49 项测试与 sim-hand 导入检查通过。

2026-09-13 顶面下 10 mm 的离线十点记录：`artifacts/reflex_ten_runs/20260913_reflex_finger_contact_v1/`。

- P1：拍摄位到固定起点途中，与目标代理重叠。
- P2–P10：名义路径通过，记录为 `offline_path_passed_not_executed`。
- 共 9 条名义路径通过、1 条准备路径拒绝；没有执行、没有抓取成功率测量。实际控制器路径仍需当轮检查。

按用户明确指定的 Reflex 接触协议，接近、闭合和抬升阶段允许手指与目标接触，所有重叠仍记录。使用固定手指 body 列表，逐对检查其余几何，避免拇指的较深重叠掩盖手掌/机械臂撞物。拍摄位到 Pi、Pi 处拇指摆位仍检查全部目标碰撞；所有阶段保留桌面 2 mm 检查。未改模型、物体代理、距离阈值、十个起点或抓姿。GGX 默认策略不变。

旧严格策略的全部拒绝记录仍保存在 `20260913_original_offline_v2/`，不得覆盖或算作真实抓取失败。新策略不是对每个手指接触质量的认证，也不模拟物体推移；提前推移、滑落、未抓住须由真实结果记录。

## 固定流程

圆物体可显式使用 `--round-object-yaw-deg 90`：将夹取短轴固定为 robot-base +Y（本实验 P1 朝机械臂的一侧），同时旋转近似盒子的方向，不旋转实测点云，不移动物体中心，不改变尺寸、闭合量、深度和十点布局。该选项要求感知标记 yaw 不确定，且只用于新 crop 的 legacy-top 模式；默认仍沿用实测方向。不是逐点搜索最容易姿态，也不声称全局最优。

该规则保存在冻结 target、scene 及十轮 summary 中；`--template` 和 `--resume-from` 必须继续传入相同的 `--round-object-yaw-deg`，不能静默改方向。旋转后的盒子只是用户指定的圆物体近似，不再声明完全包围原点云。原 GGX 方向、配置和历史记录不变，不能将此人工指定方向条件冒充完全相同的自动朝向条件。

苹果记录：原始 crop 位于 `artifacts/reflex_apple_20260914/crop/`，感知短轴 -45.73° 且 yaw 不确定。按用户要求固定为 +90° 后，近似盒子长轴 yaw 为 0°；新结果在 `facing_arm_offline/`。四指、35 mm 深度及全部十点几何不变，十点均完成名义 IK，P2/P4/P5/P6/P7/P8/P9 通过名义全路径，P1/P10 桌面拒绝，P3 准备段目标碰撞拒绝。此为离线检查，不是控制器/实机成功结果；旧十点 IK 失败记录 `offline_ten/` 保留。

1. 从已审核的 foreground crop 一次性冻结原始四指模板：按用户要求，十轮入口当前默认顶部下 **35 mm**（`--grasp-depth-mm`）、原倾角/短轴对齐、小指保持张开，不做侧抓旋转或增加侧面接近点。与旧 10 mm 相比仅抓取中心及手/TCP 终态下降 25 mm，方向、闭合指令与 P1–P10 不变。深度必须小于实测物体高度，不会自动裁剪。
2. 使用 `reflex_ten_starts_object_relative_lab.yaml`：以物体抓取点为原点、固定实验到 base 的映射；不因候选可达率挪点或改变相对几何。每轮重新检查实际起点与路径。
3. 实机模式下，复用现有有确认的张手、上退、返回拍摄位流程。控制器或复位故障立即停止，不绕过关节限制、不自动重试。
4. 用户摆回同一物体，Enter 拍摄。复用现有 RGB XY tracking，仅更新平移，冻结高度、yaw、尺寸、手型和模板；定位不确定就停止。
5. 检查当前拍摄位张手 → 在拍摄位摆好拇指 → 保持准备手型到 Pi → 原 Reflex 直接 moveL → 其余手指闭合 → lift。名义仿真、控制器预检和实际命令使用同一顺序；准备段使用 `thumb_ctrl`，在原拍摄关节位置检查拇指过渡，不用 GGX closure alpha 代替。
6. 名义路径通过后，Enter 仅允许控制器脚本/预检；检查实际控制器 IK 路径和两次手指过渡，再显示本轮视频；最终 Enter 才授权本轮动作。
7. 执行：高位张手、拇指命令、等待 0.2 s、机械臂到 Pi 再接近抓取终态、其余手指命令、等待 0.25 s，再抬升。无角度/力传感器反馈；控制器到位监控仍保留。0.2 s 是沿用的命令等待，不是实测的手部到位确认。
8. 默认抬升沿用原场景的 80 mm，保持 3 s，然后人工记录 `s` 成功、`f` 失败、Enter 未确认、`q` 记录未确认并结束。成功定义：物体完整离桌并在保持期间未滑落。
9. 下一轮先等待有支撑/路径畅通的复位确认，再重拍和定位。最后保持手不自动松开。

运动前的本地 IK 耗尽/碰撞可记录并跳过；通信、控制器安全、定位、清理或运动后的故障停止整轮。连续未运动的跳过点可复用同一次 XY 测量；不能把旧 XY 用于真实执行后下一轮。

## 命令

### 换成 40 mm 方块

先单独退回拍摄位，避免手臂遮挡（不拍照、不生成模板、不开始抓取）：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_reflex_reset.py \
  --execute --allow-high-force
```

这是现有张手/80 mm 上退/返回拍摄位的薄入口，仍须现场 Enter 确认物体有支撑、张手及全程路径畅通。默认不带 `--execute` 只写未执行记录、不连接设备。相机支架、未知夹具、张开扫掠仍需现场确认。

三指模式现已接通：`--fingers 3 --grasp-depth-mm 15`，直接使用原 `PLANE_3F` 求解器，活动手指为拇指、食指、中指，无名指和小指在最终控制中保持张开。默认仍是四指/35 mm，不改变糖盒协议；不是简单停用第四指，而是按三指几何重新求解中心、闭合量和手腕目标。

需要重新拍摄并审核方块 foreground crop，再用这个**新 crop** 运行十轮，不能使用糖盒的 crop/template/resume。三指模板和续跑必须显式传 `--fingers 3`，深度须传 `--grasp-depth-mm 15`；不一致会报错。模式/深度保存在 summary 和冻结 target。

首次方块感知已完成：`artifacts/reflex_cube_three_20260913/`。相机九帧、前景 #24，`crop/summary.json` 为新现场点云；`offline_ten/template.json` 为三指/15 mm 模板。测量短边 37.081 mm、高度 45.161 mm，与用户标称 40 mm 有偏差，未人工改值；原求解器得到宽度 37.322 mm。抓取点 base XYZ 为 [-217.213, -704.319, 69.960] mm，15 mm 深度相对点云所用上界而非人工测量顶面。

该次十点**全部桌面预检拒绝，未执行实机**。P1 首次拒绝在准备路径，P2–P10 在 Reflex 接近；P2 终态 `wrist_1_link/geom_5` 与当前保守桌面模型重叠约 27.038 mm，不是已允许的手指-目标接触。所有轨迹及视频保存在 `offline_ten/P*/nominal/`，P2 终态截图为 `P2_final_pose.png`。真实桌沿与保守模型的关系尚未重新核实，因此记录为当前模型下的路径拒绝，不是实物抓取失败，也不是已经证明物理碰撞。没有改桌面、阈值或手型强行放行。

后续同抓姿 IK 审计：`tools/audit_reflex_ik_branches.py` 使用原 Mink IK、多初值和限位内腕关节等价角度，保持方块点云、手腕 TCP 终态、三指命令、15 mm 深度、P2 起点及插值路径不变。先枚举终态，再沿原 Cartesian 路径反向求 P2 分支初值，最后完整正向复查准备/接近/闭合/抬升，不能用终态通过替代路径通过。

`artifacts/reflex_cube_three_20260913/ik_wrist_wrap_P2_v3/` 的分支 6、7 通过完整名义路径。分支 6 准备段最小桌面净空约 40.28 mm，接近/闭合约 14.04 mm，抬升约 16.04 mm；旧分支终态为 -27.04 mm。没有改变手掌姿态，只改变机械臂 IK 分支及腕关节的等价角度表示。复核视频：`ik_wrist_flip_P2_video_v4/branch_06/reflex_preview.mp4`。这是离线诊断，不是控制器或物理认证。

按用户随后要求，新腕分支已成为 Reflex 的默认分支：`run_reflex_reset.py` 和 `run_reflex_ten_starts.py` 默认读取 `configs/reflex_scan_wrist_flip.json`，十点布局的 `seed_q` 也使用相同分支。拍摄 TCP 位置/朝向、十点相对位置及抓姿不变；不再每轮主动返回旧腕分支，也不把关节角独立归一化到 ±π。旧实测 `artifacts/vision_pick_start/start_pose.json` 保留不变。

该新配置是名义 IK 初值，不冒充实测编码器数据。实机仍由原控制器 IK/分支/路径检查解析目标，最终到位检查对照控制器实际批准的目标，并保存真实 `scan_snapshot.json`。首次从旧拍摄姿态切换，腕关节 1/3 的净变化约 -157°/-181°；须在开始前确认线缆余量，运动中不要接触线缆。后续新拍摄位到 P2 的对应净变化约 -9°/+24°，不代表整条路径没有转腕或线缆已获认证。

新默认离线重跑保存在 `artifacts/reflex_cube_three_20260913/new_wrist_default_offline/`：P2/P3/P4/P5/P7/P8/P9 通过名义全程检查；P1/P10 准备段、P6 接近段仍被桌面检查拒绝。未放宽阈值或改变模型，未发送实机指令。旧模板含旧配置哈希，不能静默重用；从同一 cube crop 生成新模板。GGX 默认配置没有同步修改，不能据此宣称两种方法的实际关节配置已匹配。拍摄位属于准备流程，分别记录准备路径与抓取路径。

另以原 `reset_goals` 的 80 mm 上退 + MoveJ 返回策略检查名义返回路径：上述七点的 lift 终态返回新拍摄位均通过桌面 2 mm 检查，最小净空约 39.82 mm；旧拍摄姿态首次切换到新姿态的名义路径最小净空约 27.30 mm。使用已有 IK、2 mm/1° Cartesian 采样和默认关节插值，张手状态、固定桌面模型、不含物体，与现有 reset 的几何范围一致。这不包含开手扫掠、线缆、相机支架或控制器实测偏差，不能当作实机运动授权。

### P1：提前准备拇指（后续协议变更）

实测扫描起点对应的旧顺序记录 `artifacts/reflex_ten_runs/20260913T234953_730601Z/P1/` 显示：P1 张开拇指与桌面模型重叠约 15.33 mm。按用户要求，仅将拇指准备提前到高位；不改变十点位置、IK 初值、抓姿、手指命令、等待时间或检查阈值。此顺序用于后续所有 Reflex 点，不针对 P1 单独放行。原记录保留，前后协议不能混称未改变。

新离线记录 `artifacts/reflex_cube_three_20260913/prepare_hand_first_offline/` 使用刚保存的真实扫描起点及原 cube crop。P1 拇指准备阶段桌面净空约 39.67 mm、到 Pi 准备路径约 4.09 mm，原拇指问题已消失；接近初段仍有一帧 `upper_arm_link/geom_3` 桌面距离为 0，低于原 2 mm 要求，因此 P1 仍未放行，不表示实机已发生碰撞。新版视频在该目录 `P1/prepare_first_video/reflex_preview.mp4`。未执行硬件动作。

合成 40×40×40 mm 输入的离线三指解宽度为约 40.397 mm，raw 最终指令 `[1000,1000,634,674,634,0]`，顺序为小指、无名指、中指、食指、拇指俯仰、拇指偏航。这不是新现场感知结果，也不是实机抓取验证；实际命令由新点云模板决定，不手工照抄。

十轮入口当前默认使用 `configs/reflex_real_lab.yaml`：关节速度由 0.10 提至 **0.20 rad/s**，直线速度由 0.01 提至 **0.02 m/s（20 mm/s）**；准备、接近、抬升和复位共用该设置。加速度仍为 0.20 rad/s² 和 0.02 m/s²，手指 speed/force、等待时间、抓姿和检查规则不变。速度上限加倍不等于整轮耗时减半。此为用户请求的调试设置，未实机验证，不能把旧 GGX 数据自动视作相同执行速度。GGX 配置文件未改动；可用 `--config configs/ggx_real_lab.yaml` 恢复原速度。

当前糖盒离线十点（无需硬件）：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_reflex_ten_starts.py \
  --crop-report artifacts/reflex_original_restored_20260913/crop/summary.json \
  --object-label sugar_box --grasp-depth-mm 35
```

加 `--video` 保存离线视频；`--point P2` 只运行指定点。全部路径拒绝时退出码为 2；控制器/通信故障不能当作普通路径跳过。

实机入口（会先显示复位确认；当前没有实机验证，且当轮实际控制器预检和视频确认仍必需）：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_reflex_ten_starts.py \
  --crop-report artifacts/reflex_original_restored_20260913/crop/summary.json \
  --object-label sugar_box --grasp-depth-mm 35 \
  --capture-live --execute --allow-high-force
```

raw speed/force 沿用配置的 1000，必须有 `--allow-high-force` 和 Enter；这不是力反馈阈值，也不是实测物理单位。未显式指定 `--execute` 时，`--capture-live` 只访问相机。

改变深度须用 `--crop-report` 生成新模板；`--template` 不会覆盖冻结深度，参数不一致会报错。旧 10 mm 试验仍保留。独立预览工具 `prepare_reflex_comparison.py` 的历史默认值不变，预览本次设置也须明确传 `--grasp-depth-mm 35`。本次是调参试验，不与旧深度结果混作同一冻结协议。

同一物体可用 `--template <本轮输出/template.json>` 替换 `--crop-report`，冻结数据和依赖哈希必须匹配。换物体必须重新生成审核后的 crop 和模板，不能改旧模板中的标签、宽度或中心。新物体拍摄/分割仍使用现有 perception 工具，本脚本不是一个新检测器。

复位等待确认期间发生的小漂移不再与旧起点做 0.1 mm 预检一致性比较：确认后先用现有到位包络（<1 mm、<0.5°、<0.003 rad）约束前后差异，再读静止姿态并重新生成、预检复位路径。这个包络不是新的传感精度声明。较大移动仍停止；预检过程中和执行前仍保持原 0.1 mm/0.1°/0.01 rad 起点检查，**没有删除检查或自动重试**。

## 记录与可复现性

### XY 搜索与续跑（2026-09-13）

`20260913T225320_494813Z` 保存了 P1 人工失败、P2 人工成功，P3 在拍摄后的 XY 定位失败，尚未执行抓取。失败照片中糖盒相对参考图平移约 -82.19 px，超出旧 ±60 px 搜索窗；扩大到 ±120 px 后，同图匹配 correlation=0.938、peak separation=0.254，估计 XY 平移约 [-36.695, 1.816] mm，跨参考高度的一致性 P90 为 0.257 mm。不是深度检查或碰撞检查造成的退出。

Reflex 入口现默认 `--xy-search-px 120`。只扩大查找区域；匹配质量门槛、40 mm 米制平移界限、冻结高度/yaw 都不变，不在定位失败时盲用旧 XY。GGX 定位器的默认窗口未改。

从这次未执行的 P3 续跑：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_reflex_ten_starts.py \
  --resume-from artifacts/reflex_ten_runs/20260913T225320_494813Z/summary.json \
  --object-label sugar_box --grasp-depth-mm 35 \
  --capture-live --execute --allow-high-force
```

续跑复用原冻结模板，不重新求抓姿；新目录中记录 `resume` 来源及 `prior_trials`，汇总 CSV 带入 P1/P2 原结果，再运行 P3–P10。原目录和原失败记录不覆盖。代码版本可能不同，旧报告和冻结输入均保留哈希关联；运动/布局/感知配置、深度及 hold 不得悄悄改变。仍先确认复位、重新拍摄和检查本轮路径。已完成的动作（即使结果未确认）不自动重做；部分执行/运动已授权但中断的点拒绝自动续跑，须先人工核查。

每轮输出 `summary.json`、`summary.csv`，每点含定位、平移后的目标、名义轨迹及拒绝阶段；进入实机分支后另保存控制器预检、执行视频、连续关节/TCP 轨迹和 raw serial TX/RX。区分：计划条件、开始运动、发出闭合命令、完成 lift/hold、人工成功。完成命令不会自动算成功。

脚本使用 `target-only` 冻结几何，不把冻结模板伪装为可执行批准。坐标采用最新 crop，不沿用早期侧抓场景。相机/标定、代码、模型、模板和定位输入均有哈希；本次给共同 controller preflight 增加了可选逐阶段 hand_ctrl，未指定时 GGX 行为不变。依赖旧代码哈希的模板可能失效，应重新生成，不可静默改哈希。

## 尚未解决、不得省略的限制

- 原始模型与安装/指令到真实指尖的一致性尚未完成验证；此前记录的 3 mm tip site 差异未通过改模型消除。
- 相机支架、夹具和完整自碰撞不在现有模型保证范围内，不能用 UR5 protective stop 替代这些检查。
- 复位沿用旧 open-loop opening/80 mm 上退策略，不检查未知起始手型的张开扫掠或新物体位置；必须在物体有支撑、路径畅通时确认。已知 wrist wrap 超限会停止，未绕过限制。
- 3 s hold 是本轮显式协议参数。历史 GGX 记录不能自动视作采用了同一 hold/success 标准；正式 matched comparison 必须固定并共同使用该标准。
- 该分支是原位置控制 Thumb Reflex，不是触碰反馈触发策略。论文旧 reflex 定义如不同，需明确区分。
