# 十起点原点不一致：2026-09-13

状态：用户已确认 P1 在物体靠机械臂一侧水平 250 mm；Reflex 默认布局已修正，未执行实机。新配置为 `configs/reflex_ten_starts_object_relative_lab.yaml`，固定 180° 纸面到 base 旋转使 P1 为 base +Y、横向为 -X；原点取冻结物体抓取点，XY 重定位同时平移起点与目标。新回放 `artifacts/reflex_ten_runs/20260913_object_relative_depth35_v1/` 中九点名义路径通过、P6 桌面净空拒绝。

GGX 入口尚未切换此布局；旧 GGX/Reflex 固定原点试验不回写，也不能声称已经与新 Reflex 条件 matched。原实验各策略起始手型/朝向的完全一致性仍是独立问题，新模式暂保留共同拍摄朝向。

## 源码依据

- `tools/run_strategy_pregrasp_rate.py` 的 `paper-approach-points` 分支使用 `start_center = grasp_target + approach_vector * distance + lateral_vector * d`，随后设置 `start_center[2] = grasp_target[2] + h`。`start_reference=grasp-center` 再按各策略手型/朝向把抓取中心转换为手基座。
- P1：相对抓取点水平 250 mm、高度差 0；P2：抓取点正上方 250 mm；其余八点：接近距离 250 mm、横向 -150/-50/50/150 mm、高度差 100/250 mm。
- 当前 `tools/ggx_fixed_starts.py` 按固定 `p1_tcp_xy_mm` 和桌面上方拇指 25 mm 定义整组；其接口没有物体位置输入。Reflex 和 GGX 十轮流程均复用了这一固定布局。这保留了点间偏移，却没有保留点组相对物体的位置。
- `tools/run_reflex_ten_starts.py` 的 XY 重定位只平移目标，没有平移已经生成的固定起点。

## 糖盒现场保存数据

来源：`artifacts/reflex_ten_runs/20260913_reflex_depth35_speed2x_P1_rejected_video_v2/` 的 `template.json`、`fixed_layout.json`、`P1/nominal/summary.json`。

UR-base，单位 mm：

| 项目 | X | Y | Z |
|---|---:|---:|---:|
| 物体抓取点（顶面下 35 mm） | -192.408 | -707.212 | 185.265 |
| 当前固定点组原点 | -215.270 | -448.098 | 102.188 |
| 当前 P1 张手中心 | -215.270 | -698.098 | 102.188 |
| 当前 P2 张手中心 | -215.270 | -448.098 | 352.188 |

固定点组原点相对所需抓取点误差为 `[-22.862, +259.114, -83.077]` mm。P1 实际水平接近物体中心约 24.6 mm，而不是 250 mm。P1 准备路径发生目标代理重叠；旧记录保留，不改写为物理抓取失败。

## 修正前须固定的坐标约定

1. 点组原点取物体上预定义的抓取参考点，不是 TCP、物体包围盒中心或固定桌面点。
2. 确认原实验接近轴到当前 UR-base 的映射。历史 `y-` 是所选实验轴，不能仅因符号相同就认定其等于当前底座坐标的负 Y。尤其不能为通过 IK 按候选旋转或镜像点组。
3. 冻结同一套映射、距离、顺序及方法间共享的参考点；每轮仅按已确认的物体定位更新点组和目标。深度为 35 mm 的当前协议应明确记录，不能混同旧深度。
4. 起始手型/朝向与中心到 TCP 的转换也需要与所声明协议一致；仅纠正原点不能证明已完全复现旧实验。
5. 重新区分起点本身、拍摄位到起点、抓取路径的拒绝。修正不代表所有旧拒绝都能通过，更不代表抓取成功。

本诊断未调整代理几何、碰撞阈值、关节限制或历史数据，也未连接硬件。
