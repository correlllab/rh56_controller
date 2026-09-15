# GGX 固定候选池与固定单模板：糖盒离线反事实复查

本记录只涉及离线诊断，不是实机授权，也不覆盖新的五物体实验。
旧实验和现场观察不改写；新结果另列为 **frozen-pool nominal route coverage**。

## 结果

在同一糖盒场景、同一批原始 top-100、同一固定十起点、同一拍摄位、
同一 CB3/手/桌面/物体模型及原路径规则下：

| 起点 | 固定 rank 20 路径复查 | 候选池按原排名的首个路径通过项 |
|---|---|---|
| P1 | 拒绝 | 无；准备阶段拒绝 |
| P2 | 通过 | rank 20 |
| P3 | 拒绝 | rank 31 |
| P4 | 拒绝 | rank 31 |
| P5 | 拒绝 | 无；准备阶段拒绝 |
| P6 | 拒绝 | rank 47 |
| P7 | 拒绝 | rank 26 |
| P8 | 拒绝 | rank 26 |
| P9 | 拒绝 | rank 26 |
| P10 | 拒绝 | rank 26 |

**单模板名义路径覆盖 1/10 → 同批候选池 8/10；恢复原九个拒绝点中的七个。**
rank 20 的十点通过/拒绝状态与历史日志完全一致。
这表明单模板限制是这批低覆盖率的重要因素，不能将原来的九次路径拒绝解释为
“GGX 的同批所有候选在这九点都不可执行”。

8/10 不是实机成功率，也不是 UR5 控制器验证通过率。
P1/P5 的共同张手起点本身与目标代理重叠，之前的起点分段审计已有记录；
换抓姿没有改变该共同准备阶段。代理重叠不是实测碰撞证明。

## 输入、筛选及复现边界

- 原十点：`artifacts/ggx_experiments/20260913_rank20_auto_ten/`。
- 完整局部筛选：`artifacts/ggx_quick_preview_20260913/top100_completion/summary.json`。
- 原生成批：`artifacts/ggx_quick_preview_20260912/fresh_scene/`。
- 原始 top-100 全部保留：6 个官方场景拒绝、87 个局部拒绝、7 个局部通过。
- 局部通过 rank：20、26、31、47、55、57、75；按原排名逐起点检查实际路线。
- 每点首个通过后停止路径搜索；后续候选标记未检查，不能据此称为失败。
- 未重新感知、推理、排序或从 top-100 之外补位；未调阈值、缩小代理、改模型或移动起点。
- 检查覆盖保存拍摄位→固定起点→pregrasp→approach→close→lift 的名义路线。
- 保留历史候选相对场景的微小平移，不重新估计 XY/yaw。

本诊断复用已有局部筛选结果，**没有证明局部筛选本身没有误拒**，也不证明 IK 全局无解。
它只隔离“固定单个模板”与“允许同批其他已通过局部筛选候选”之间的差别。

历史路线的规划、IK、模型及数据依赖按哈希核验；变化的 `ggx_motion.py`
在单独离线进程中加载归档的精确历史版本。
不调用已变化的 reset、起点生成器和 UI/编排入口；使用保存的数值起点。
输入文件与归档字节一致性也逐一核验。哈希、依赖差异和全部逐点 top-100 状态保存在报告内。
未修改原批准文件、原始日志或当前运动模块，未将旧批准重新签名为运动授权。

结果目录：

- `artifacts/ggx_pool_start_audits/sugar_rank20_top100_01/`：第一次复查。
- `artifacts/ggx_pool_start_audits/sugar_rank20_top100_02/`：增加归档输入自动核验后的复跑。

每次输出 `summary.json`、`summary.csv`，以及逐起点/候选的检查细节。
CSV 可直接用于汇总；大文件仍为本地 artifacts，不等于异地备份。

两次复跑均完成，逐点首选 rank 与 rank 20 的通过/拒绝状态完全一致；第二次全部输入哈希复核通过。
新增选择/异常处理测试 5/5 通过；`sim-hand` 导入检查通过。
全套测试 1083 通过、2 个既有失败仍在：两份手模型的拇指 tip site 相差 3 mm，
以及模板创建单测缺少 `approval.json` fixture。此次未改这些模型或原测试来消除失败。

## 离线复跑命令

使用一个尚不存在的新输出目录：

```bash
MUJOCO_GL=egl .venv312/bin/python tools/audit_ggx_frozen_pool_starts.py \
  --run artifacts/ggx_experiments/20260913_rank20_auto_ten \
  --pool artifacts/ggx_quick_preview_20260913/top100_completion/summary.json \
  --archive-files artifacts/ggx_paper_evidence/20260913_session/files \
  --out artifacts/ggx_pool_start_audits/sugar_rank20_top100_repeat
```

脚本没有 `--execute`，不连接相机、手或机械臂；意外异常不能被当作普通候选拒绝吞掉。

## 后续五物体协议

已新增独立入口 `tools/run_ggx_matched_ten.py`；见 [共享 Reflex 条件的十轮流程](ggx_matched_ten_zh.md)。
旧单模板入口和本次历史复查保留不变。

每物体生成一次并冻结整个原始 top-100 集合；每个起点按原排名选择首个通过预定检查的候选，
最多进行一次物理尝试，执行失败后不再换候选重试。人工确认仍用于安全否决，不按预期成功率挑选。
候选为空、局部拒绝、准备拒绝、后续路径拒绝、执行及实际结果分别记录。

这与 [GGX 论文附录 D](https://arxiv.org/html/2606.00998v1) 将 top-100 交给运动规划器筛选的
工业部署思路一致；“按原排名首个通过、每点一次执行”是本实验明确制定的规则，
不是声称逐字复现官方全部 motion-planning protocol。

旧结果应保留为 **fixed-template evaluation**，不能与新候选池实验无标记混合。
历史 GGX 使用固定世界坐标起点；后来的 Reflex 使用物体相对起点且改过腕部分支。
新的 matched comparison 必须核对实际保存的 P1–P10 TCP/关节起点、手部准备状态和计分起始阶段；
只有 P 标签相同不能证明条件相同。本次历史复查不改那些起点以冒充新的匹配实验。

物体清单也需固定：原 GGX 为糖盒、方块、芥末瓶、充电器、罐；最近 Reflex 则包含苹果。
若采用最近 Reflex 的五物体，则 GGX 也应改为同一清单，并使用对应共享感知输入。
