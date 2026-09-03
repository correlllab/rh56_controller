# RH56 analytical control vs. pretrained GraspGen-X

This experiment compares two complete **simulation-only RH56 grasp pipelines**:

- **Analytical synchronized closure:** the paper's four-finger plane solver maps
  target width to the six RH56 actuator commands, while the virtual wrist pose
  is updated throughout closure to keep the antipodal grasp center fixed.
- **Pretrained GraspGen-X:** the official pretrained network generates a 6-D
  Inspire-Hand pose. The hand then follows the open-to-close endpoints shipped
  with the official Inspire-Hand descriptor while the target wrist pose stays
  fixed.

Both pipelines use the same accurate RH56 MuJoCo model, 40 mm cube, object
mass/friction, approach and lift motion, and contact-limited low-level executor.
After opposing contacts are detected, the common executor applies 10% additional
normalized closure, lifts 80 mm, and holds. A trial succeeds when the final
object center is at least 40 mm above its initial height.

This is a planner/control characterization, not a real-hardware result. It does
not include H12/UR5 arm IK, balance, perception, navigation, or wrist-camera
occlusion.

## Generate the pretrained candidates

The official GraspGen-X checkout and its independent environment live at
`/home/tanxuan/workspace/GraspGenX`. The validated run used repository revision
`b9429097728cb1c430dd78b92edf17ba318aad03`. Generate 100 candidates for the
40 mm cube:

```bash
cd /home/tanxuan/workspace/GraspGenX

.venv/bin/python scripts/demo_object_mesh.py \
  --mesh_file assets/sample_data/object_mesh/box.obj \
  --mesh_scale 0.4 \
  --gripper_name inspire_hand \
  --grasp_threshold -1.0 \
  --return_topk \
  --topk_num_grasps 100 \
  --num_grasps 1000 \
  --no-visualization \
  --output_file /home/tanxuan/workspace/rh56_controller/artifacts/graspgenx_success_comparison/graspgenx_candidates.yml
```

The comparison sorts those candidates by learned confidence and selects the
first pose that approaches from above without penetrating the table. Candidate
selection is deliberately not based on MuJoCo lift success.

## Run the comparison

```bash
cd /home/tanxuan/workspace/rh56_controller

.venv312/bin/python -u tools/run_graspgenx_success_comparison.py \
  --graspgenx-yaml artifacts/graspgenx_success_comparison/graspgenx_candidates.yml \
  --out artifacts/graspgenx_success_comparison \
  --error-mm 0 5 10 15 20 25 30 \
  --directions 8

.venv312/bin/python tools/plot_graspgenx_success_comparison.py \
  --input-dir artifacts/graspgenx_success_comparison
```

For a side-by-side nominal-grasp video:

```bash
MUJOCO_GL=egl .venv312/bin/python tools/run_graspgenx_success_comparison.py \
  --graspgenx-yaml artifacts/graspgenx_success_comparison/graspgenx_candidates.yml \
  --out artifacts/graspgenx_success_comparison/video_trial \
  --error-mm 0 \
  --directions 1 \
  --video
```

The primary outputs are `summary.csv`, `trials.csv`, `assumptions.json`,
`success_rate.png`, `directional_lift.png`, `object_displacement.png`, and the
nominal MP4. The CSV files are authoritative; plots are derived views.

## How to interpret this first experiment

Object-position error is applied to the actual cube while both planners keep
using the nominal cube center. It therefore measures execution robustness to a
controlled localization error, not general object recognition. Each non-zero
error magnitude is tested in eight horizontal directions.

One candidate set and one automatically selected GraspGen-X pose are not enough
for a final statistical claim. A paper result should repeat network generation
with fixed, recorded random seeds or saved candidate files and should add more
objects. The current run is a validated baseline and helps choose the final
experimental design.

## Current validated run

The saved 100-candidate set selected confidence rank 59 after top-down and
table-collision filtering. The lift results were:

| Horizontal error | Analytical synchronized | Pretrained GraspGen-X |
|---:|---:|---:|
| 0 mm | 1/1 (100%) | 1/1 (100%) |
| 5 mm | 8/8 (100%) | 8/8 (100%) |
| 10 mm | 8/8 (100%) | 8/8 (100%) |
| 15 mm | 7/8 (87.5%) | 8/8 (100%) |
| 20 mm | 5/8 (62.5%) | 6/8 (75.0%) |
| 25 mm | 5/8 (62.5%) | 3/8 (37.5%) |
| 30 mm | 4/8 (50.0%) | 2/8 (25.0%) |

Across the 48 non-zero-error trials, the analytical method succeeded in 37
(77.1%) and GraspGen-X in 35 (72.9%). This small aggregate difference should
not be treated as a general ranking: GraspGen-X was stronger at 15--20 mm,
while the analytical grasp retained more successful directions at 25--30 mm.
Across all 49 paired conditions, six were won only by the analytical method and
four only by GraspGen-X; a two-sided exact McNemar test gives `p = 0.754`.
Therefore this run does not support a statistically significant overall
success-rate difference.

The nominal trial exposes a clearer control difference. The analytical method
limited maximum cube XY displacement to 1.9 mm, compared with 18.3 mm for the
fixed-pose GraspGen-X closure. GraspGen-X generated a successful grasp pose, but
it did not compensate the RH56 coupled-joint closure motion. This is exactly the
role of the analytical synchronized trajectory, and should be reported as a
separate object-disturbance metric rather than folded into lift success.
