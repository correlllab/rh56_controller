# GraspGen-X UR5e + RH56 table preflight

`tools/preflight_ggx_ur5_table.py` is a simulation-only, deliberately simple
screen before a human reviews a GraspGen-X motion. It does not connect to the
UR controller or RH56 hardware.

The automatic gate covers one question: does the sampled UR5e + RH56 path keep
the requested distance from a horizontal desk plane? The desk height is
specified in the UR base frame. The fixed shoulder fixture contact is ignored;
every other active robot collision geom is checked.

The sampled motion is:

1. the supplied starting arm joints to a GraspGen-X pre-grasp;
2. a linear Cartesian approach resolved through UR5e IK;
3. contact-limited RH56 closure against a fixed primitive object proxy;
4. a vertical Cartesian lift.

Object collision and robot self-collision remain visible in MuJoCo but are not
automatic pass/fail gates in this intentionally narrow tool. A PASS always
requires human review and is not a real-robot safety certificate.

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
amount, current and running-minimum desk clearance, nearest robot geometry, and
automatic PASS/REJECT state. This keeps the main safety evidence visible when a
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
