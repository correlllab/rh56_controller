# GraspGen-X UR5e + RH56 table preflight

`tools/preflight_ggx_ur5_table.py` is a simulation-only, deliberately simple
screen before a human reviews a GraspGen-X motion. It does not connect to the
UR controller or RH56 hardware.

The automatic gate checks desk clearance throughout the sampled UR5e + RH56
path and rejects robot-object contact in start, start-to-pregrasp and approach.
The desk height is
specified in the UR base frame. The fixed shoulder fixture contact is ignored;
every other active robot collision geom is checked.

The sampled motion is:

1. the supplied starting arm joints to a GraspGen-X pre-grasp;
2. a linear Cartesian approach resolved through UR5e IK;
3. contact-limited RH56 closure against a fixed primitive object proxy;
4. a vertical Cartesian lift.

Object contact during closure/lift and robot self-collision are not automatic
pass/fail gates in this intentionally narrow tool. A PRECHECK PASS always
requires human review and is not a real-robot safety certificate.

The approach-object gate uses signed distances to all active robot collision
geometries, including fingers, palm, attachment and arm. Touching or penetration
rejects a sample (1e-9 m numerical epsilon, no added empirical clearance margin).
Visual-only geometries and the desk are excluded from this object gate.
Checks are sampled, not continuous, and the primitive object is approximate.
Results record `object_collision_before_closure`, the first phase and minimum
object distance. Closure quality is deliberately not checked for this development
stage. Old desk-only PASS reports must be regenerated before interactive review.

The `robot_object_before_closure/v2` gate handles the reproduced MuJoCo exact-zero
distance failure conservatively: for a zero query with no actual pair contact,
box/mesh vertices are projected onto world/local geom axes. Only a positive
separating interval gap (minus 1e-7 m arithmetic tolerance) may replace the zero.
The replacement is a **lower bound**, not an exact distance. Unknown geometry,
unproven separation, actual contact and negative distances remain rejected.
This is not a switch to `distance < 0`, nor reliance on the absence of contacts.
The observed failure resembles [upstream issue 3383](https://github.com/google-deepmind/mujoco/issues/3383).
Per-sample/candidate `object_distance_corrections` counts are saved for audit.
Older v1 gate reports must also be rerun; no existing experiment is overwritten.

The proxy is fixed, so an arm-lift animation does **not** demonstrate an object
pickup or grasp stability. The existing lift phase commands full GGX closure;
it does not preserve the contact-limited closure amount from the preceding phase.

## Candidate coordinates and oriented objects

The default `--candidate-frame object` preserves the legacy convention:
positions are relative to the object center, with axes aligned to the UR base.
For measured-cloud inference exported in absolute UR base coordinates, specify
`--candidate-frame ur-base`. The adapter rebases the translation internally so
the existing conversion does not add the object center twice; it does not alter
the physical candidate pose or apply empirical grasp offsets.

`--object-yaw-deg` rotates only the proxy around base Z. Its dimensions remain
local X/Y/Z dimensions; this option never rotates the input candidates. Both
options and the resolved object center are recorded in `summary.json`.

The September 10 upright-sugar-box run uses the saved candidate YAML and joint
snapshot, a 90 × 40 × 176 mm proxy, and a horizontal approximation of the measured
desk plane evaluated at the object center:

```bash
MUJOCO_GL=egl .venv312/bin/python tools/preflight_ggx_ur5_table.py \
  --config artifacts/ggx_lab/upright_sugar_sim_01.yaml
```

These local artifacts are not committed. This is an offline UR5e-model preview;
the lab robot is a CB3 UR5. No current hardware state is read or commanded.

## Run the checked-in cube candidate

Replace the table height, object position, object dimensions, and start joints
with the current setup before using the output for review:

```bash
cd /home/tanxuan/workspace/rh56_controller

MUJOCO_GL=egl .venv312/bin/python \
  tools/preflight_ggx_ur5_table.py \
  --graspgenx-yaml \
    artifacts/graspgenx_success_comparison/graspgenx_candidates.yml \
  --table-height-m 0.070 \
  --object-x-m 0.0 \
  --object-y-m -0.50 \
  --object-size-mm 40 40 40 \
  --start-q-deg 0 -90 90 -90 0 0 \
  --desk-clearance-mm 2 \
  --video
```

For repeatable lab runs, first copy
`configs/ggx_ur5_table_lab.example.yaml`, replace its example measurements, and
run:

```bash
MUJOCO_GL=egl .venv312/bin/python \
  tools/preflight_ggx_ur5_table.py \
  --config configs/ggx_ur5_table_lab.yaml
```

Explicit command-line options override YAML fields, including Boolean options;
for example, append `--table-height-m 0.083 --no-video`. Relative paths inside
the YAML are resolved from the repository root. The config path and SHA-256 hash
are saved in `summary.json` together with the final resolved arguments.

Use `--viewer` instead of or in addition to `--video` for interactive review.
Use `--candidate-rank N` to inspect one generated pose. Without it, candidates
are tested in learned-confidence order until five pass. Change that count with
`--review-passes N`.

With `--video`, the standalone tool retains the rejected-candidate diagnostic
video by default. Add `--no-rejected-video` to skip only that video, preserving
all checks, CSV/JSON reports and available failure trajectories. The interactive
review launcher uses this option automatically because it never shows rejected
videos. Passing convenience videos are copied from `review_rank_NNN.mp4`, not
rendered twice. `summary.json` records the video policy and setup/evaluation/render
timings.

The interactive launcher calls this same preflight in-process with a fixed
`PreflightScene`, avoiding repeated imports/model loads. Each candidate is still
evaluated independently from the same start pose; changed scene/check arguments
or pinned input files invalidate reuse. It checks in confidence order and does
not evaluate another candidate while waiting for human approval.

Outputs are written below `--out`:

- `summary.csv` and `summary.json`: automatic gate result and assumptions;
- `candidates.csv`: every checked candidate and rejection reason;
- `human_review.csv`: automatically feasible ranks with blank decision and veto
  fields for the reviewer;
- `trajectory.csv`: sampled joints and table clearance for the passing or
  closest rejected candidate;
- `trajectory_rank_NNN.csv` and `review_rank_NNN.mp4`: individual files for
  every automatically feasible candidate;
- `review.mp4`: convenience copy for the highest-confidence feasible candidate.

Each MP4 has an overlay with the GGX confidence rank, motion phase, closure
amount, current and running-minimum desk clearance, nearest desk geometry, object
clearance (or NOT CHECKED), and PRECHECK PASS/REJECT state. This keeps the scope visible when a
video is copied away from its CSV files.

If no candidate passes, the program exits with status 2 but still writes the
diagnostic files. A generated pose is never modified to make it pass.

For real deployment, review the feasible candidates in confidence order. A
higher-ranked candidate may be vetoed only for a concrete unmodelled execution
or safety reason, which should be written to `human_review.csv`. Execute the
highest-confidence candidate that remains after those vetoes. Do not skip a
feasible candidate merely because a lower rank looks more likely to grasp; that
would turn the baseline into subjective human grasp selection.

## Measurements needed in the lab

- confirm that the robot is a UR5e rather than a CB-series UR5;
- measure desk-top Z relative to the UR base;
- read the current six arm joint angles before each review;
- verify the real TCP-to-RH56 attachment against the MuJoCo model.

Changing the camera calibration, object pose, table height, starting joints, or
mount transform invalidates an earlier preflight result.

Use `docs/ggx_first_test_day_checklist.md` for the staged first-day procedure
and `docs/ggx_first_test_record_template.csv` to retain every attempt and its
failure stage.
