# Strategy reachability / "no-go" analysis — 2026-09-08

Geometric admissibility of the commanded approach **and closure** motion for
naive, iterative and thumb-reflex closure, over the 15 benchmark objects and the
10 hardware approach conditions.

Answers Reviewer 15's objection that the three closure strategies were selected
empirically without an assessment that they will operate correctly, and delivers
the simulation analysis promised in the author response
(`../IROS26_rh56_rebuttal-2.pdf`).

**Read the doc page first:** [`20260908-strategy_reachability.html`](20260908-strategy_reachability.html)
(published copy: https://claude.ai/code/artifact/a2424428-921c-496b-8e47-3a660cfa051c)

## Headline

Primary configuration: support plane checked against the hand's collision
geometry with no clearance tolerance, object contact against the 13-capsule
proxy, planned contact fingers allowed to touch the object, GraspGen-X scored
over its first 100 ranked proposals as in the hardware protocol. Per-trial rows
in `results/paper_conditions_meshfloor_ggx100_trials.csv` (600 rows = 4
strategies x 15 objects x 10 approach conditions); the three-strategy subset is
`results/paper_conditions_meshfloor_trials.csv`, and
`results/paper_conditions_meshfloor_ggx_trials.csv` is the same run scored over
the first 20 proposals.

| Strategy | Admissible | Support collision | Premature contact | Terminal miss |
|---|---|---|---|---|
| Naive | 72/150 (48.0%) | 76 | 10 | 60 |
| Iterative | 107/150 (71.3%) | 41 | 10 | 0 |
| Reflex | **136/150 (90.7%)** | 12 | 5 | 0 |
| GraspGen-X, first of 100 proposals that executes | 71/150 (47.3%) | 72 | 62 | 30 |
| GraspGen-X, first 20 proposals only | 41/150 (27.3%) | 91 | 92 | 32 |
| Hardware success, 150 trials each | 48.0% / 82.0% / 86.7% | | | |

The failure-mode columns count trials showing each mode and a trial can show
more than one, so they do not sum to the unreachable count.

By object subset (`category` column; delicate = 5 objects x 10 conditions,
ycb_like = 10 x 10):

| Strategy | Delicate (n = 50) | YCB-like (n = 100) |
|---|---|---|
| Naive | 18/50 (36.0%) | 54/100 (54.0%) |
| Iterative | 33/50 (66.0%) | 74/100 (74.0%) |
| Reflex | 45/50 (90.0%) | 91/100 (91.0%) |
| GraspGen-X (100 proposals) | 19/50 (38.0%) | 52/100 (52.0%) |
| GraspGen-X (20 proposals) | 0/50 (0.0%) | 41/100 (41.0%) |

Naive closure admits nothing for the six objects whose grasp point is at or
below 15 mm above the table and carries a median 17.4 mm terminal contact
displacement that no other strategy has. Sensitivity to the support-plane
geometry, the capsule-envelope calibration, and the ablation of the size and
condition corrections are in `results/calibration/README.md` and on the doc
page.

## Numbers quoted in the paper, and where each comes from

Two sentences in the ICRA draft (`ICRA_Inspire_RH56_Dexterous_Hand-4.pdf`,
Sec. III.A and IV.C) do not match the primary run and should be corrected.

**Delicate-subset iterative reachability is 66%, not 71%.** The draft reads
"90% versus 71% reachability" for reflex against iterative on the delicate
objects. The delicate subset has 50 trials, so every rate on it is a multiple
of 2 points; 71% is the full-set iterative rate 71.3% (107/150) transposed into
the subset sentence. The subset values are reflex 45/50 = 90.0% and iterative
33/50 = 66.0%:

```
uv run python - <<'PY'
import pandas as pd
d = pd.read_csv("results/paper_conditions_meshfloor_trials.csv")
print(d.groupby(["strategy","category"]).admissible.agg(["sum","count","mean"]))
PY
```

**Naive closure has 76 support collisions in the primary run, not 57, and the
60 whiffs are a subset of them.** The draft reads "60 misses ('whiffs') and 57
table collisions". The 57 is the naive support-collision count from the
capsule-proxy run with a 10 mm floor tolerance
(`results/paper_conditions_calibrated_trials.csv`, and every `cal_*_f10`
variant), which clears 19 of the 76 and keeps the other 57; the 48.0%,
71.3%, 90.7% rates and the 41 and 12 collisions for iterative and reflex in the
same paragraph are from the primary mesh-floor run, where naive has 76. In that
run the 78 unreachable naive trials partition as

| Modes shown | Trials |
|---|---|
| support collision only | 12 |
| support collision + terminal miss | 56 |
| support collision + premature contact | 4 |
| support collision + terminal miss + premature contact | 4 |
| premature contact only | 2 |
| terminal miss only | 0 |

so |support| = 76, |terminal miss| = 60, |premature| = 10, and their union is
78 = 150 - 72. Every whiff also collides with the table: with the wrist held
at the open-hand pose, the coupled closure carries the contact midpoint 17.4 mm
past the target and the index intermediate phalanx into the support plane (54
of the 60 trials; the thumb distal in the other 6) at a median depth of 6.1 mm,
so the same 60 trials trip both tests. The columns are "trials
showing this mode", not a partition. Corrected sentence: "Naive closure
additionally records 60 terminal misses ('whiffs') and 76 support-surface
collisions, with every whiff also a collision, due to the unaccounted-for
coupled-finger motion."

**GraspGen-X reachability is 47.3% over the first 100 proposals, not 27.3%.**
The draft says a condition is reachable when "at least one of the first 100
proposals" executes, and the hardware protocol (Sec. IV.D) keeps the top 100,
but the run behind 27.3% tested only 20 (`ggx_topk: 20` in
`results/paper_conditions_meshfloor_ggx_assumptions.json`). Rerun at
`--ggx-topk 100` on 2026-09-15 (`paper_conditions_meshfloor_ggx100`): 71/150 =
47.3%, level with naive closure; the three hand-designed strategies are
unchanged. The first executable proposal is the 20th by median (interquartile
range 3rd to 24th, worst 74th; `ggx_candidate_rank` in the CSV is 0-based) and
30 of the 71 successes lie beyond the 20th, which is why the top-20 score is so
much lower. GGX still admits nothing for the seven objects whose grasp point is
at or below 15 mm (big screwdriver, charger, nut, pen, raspberry, small
screwdriver, strawberry), the set on which naive closure whiffs. The
quarter-ball volume sweeps behind Fig. 5 were run at the top 10
(`ggx_topk: 10` in `results/nogo_volume_paper_{strawberry,can}_assumptions.json`);
the figure keeps them (the caption says best of 10). At 100 proposals the
strawberry region stays entirely unreachable, since
`code/ggx_start_independent_check.py` shows all 100 proposals collide with the
table at the proposal pose itself, and the can region goes from 67.7% to 3.4%
unreachable (`results/ggx100/nogo_volume_paper_can.npz`, a grid clipped to 995
of the 1206 poses; the hand strategies read 35.2 / 35.2 / 33.0% on it against
34.0 / 34.0 / 32.1% on the full grid). `ggx_worst`, every one of the 100
executing, is 0% of the region on both grids.

Everything else the draft quotes from this analysis (48.0 / 71.3 / 90.7%;
41 and 12 iterative and reflex collisions; 13 grazes at or under 1.7 mm; the
10.7-point iterative residual; the quarter-ball exclusions 76.9 / 53.3 / 100 /
100% at the strawberry and 34.0 / 32.1 / 34.0 / 67.7% at the can) matches the
primary run.

## Layout

```
code/
  strategy_admissibility.py   strategy waypoint model + admissibility criteria
  run_reachability_sweep.py   sweep runner (paper conditions / dense grid)
  calibrate_capsule_envelope.py  capsule-proxy over-approximation against the mesh
  make_figures.py             figures A-C
  mj_snap_rh56.py             MuJoCo renderer + clearance report ("mujoco-eyes" for RH56)
  ggx_strategy.py             GraspGen-X candidates as a fourth strategy
  run_nogo_volume.py          3-D start-pose sweep around one object
  render_nogo_volume.py       no-go volumes as MuJoCo solids; --strip for one panel per strategy
  compose_nogo_figure.py      stacks the bare strip panels into the lettered paper figure
  render_strips.sh            the strips and the paper figure, with the camera that is part of them
  render_approach_video.py    filmstrip / mp4 of one strategy's commanded motion
  docpage.tmpl.html           doc page template
  build_docpage.py            fills the template from the CSVs
results/                      per-trial CSVs + assumptions JSON per variant; *_stats.json per strip
results/calibration/          capsule-envelope samples, tolerance sweeps, ablations (README.md)
figures/                      PDF (LaTeX) and PNG (doc page); panels/ holds the bare strip panels
figures/20260908-nogo_grid_strawberry_can.png   the paper figure (Fig. 5 bottom)
20260908-strategy_reachability.html   the doc page
20260908-strategy_reachability.tex    standalone LaTeX section (does not touch root_wxie.tex)
```

Doc page, published copy of the same file:
https://claude.ai/code/artifact/a2424428-921c-496b-8e47-3a660cfa051c

## Code this depends on, elsewhere in the repository

Nothing below was copied into this folder.

| Path | Used for |
|---|---|
| `rh56_controller/grasp_geometry.py` | `InspireHandFK`, `ClosureGeometry` — analytical closure solver and FK tables |
| `rh56_controller/capsule_hand_proxy.py` | 13-capsule hand proxy, capsule/object and capsule/plane distances |
| `rh56_controller/paper_v2_objects.py` | the 15 benchmark object proxies and their grasp targets |
| `h1_mujoco/inspire/inspire_grasp_scene.xml` | hand model (floating base, coupled linkages) |
| `rh56_controller/graspgenx_baseline.py` | Isaac-grasp loading and the GGX to MuJoCo frame conversion |
| `tools/export_paper_object_proxy_meshes.py` | object proxy meshes for GGX inference |
| `tools/run_graspgenx_paper_object_inference.py` | GGX inference over the 15 objects |
| `tools/run_strategy_pregrasp_rate.py` | **superseded** pre-grasp-only check; see below |
| `tools/demo_strategy_pregrasp_collision.py` | earlier single-example demo of the same idea |

### Why the existing tool was not enough

`tools/run_strategy_pregrasp_rate.py` checks only whether the hand can translate
to a strategy's pre-grasp pose with the finger shape held fixed. Naive and
reflex closure both approach with the opposing fingers open and both target the
analytical final pose, so it admits them at nearly identical rates and cannot
separate the strategies:

```
$ uv run python tools/run_strategy_pregrasp_rate.py --object paper_sugar_box \
      --mode object-default --start-sampler paper-approach-points --out /tmp/x
  naive:             rate=1.000000  valid=10/10
  iterative_closure: rate=0.500000  valid=5/10
  thumb_reflex:      rate=1.000000  valid=10/10
```

The discriminating phase is closure, which that tool never simulates. This
analysis adds the closure sweep, the terminal-alignment test, and an object-axis
aware grasp yaw.

## Reproduce

```bash
cd /home/humanoid/Programs/rh56_controller
R=paper/9-8_wxie/20260908_strategy_reachability

uv run python $R/code/run_reachability_sweep.py --sampler paper-conditions --out $R/results
uv run python $R/code/run_reachability_sweep.py --sampler paper-conditions \
    --naive-pose-policy final-orientation --tag sens_naive_final_orientation --out $R/results
uv run python $R/code/run_reachability_sweep.py --sampler paper-conditions \
    --contact-policy contact-fingers --tag sens_contact_fingers --out $R/results
uv run python $R/code/run_reachability_sweep.py --sampler grid --tag grid --out $R/results   # ~20 min

# GraspGen-X: meshes and inference need the GraspGen-X environment
GGXPY=~/Programs/hand/external/GraspGenX/.venv/bin/python
$GGXPY tools/export_paper_object_proxy_meshes.py \
    --out artifacts/paper_15_object_grasp_success/proxy_meshes
$GGXPY tools/run_graspgenx_paper_object_inference.py \
    --graspgenx-root ~/Programs/hand/external/GraspGenX \
    --mesh-dir artifacts/paper_15_object_grasp_success/proxy_meshes \
    --out artifacts/paper_15_object_grasp_success/graspgenx_candidates

# four-strategy sweep, primary configuration (20 min; --ggx-topk 20 for the
# earlier paper_conditions_meshfloor_ggx run)
uv run python $R/code/run_reachability_sweep.py --sampler paper-conditions \
    --contact-policy contact-fingers --floor-model mesh --floor-tolerance-mm 0 \
    --ggx-candidates artifacts/paper_15_object_grasp_success/graspgenx_candidates \
    --ggx-topk 100 --tag paper_conditions_meshfloor_ggx100 --out $R/results

# the 3-D no-go volumes behind Fig. 5, GraspGen-X at its best of 10 proposals.
# The grid must span the whole quarter ball: xy +-300 mm and z from the grasp
# point to 300 mm above it (the runner's defaults clip it).
for OBJ_Z in "paper_strawberry 15 315" "paper_can 110 410"; do set -- $OBJ_Z
  uv run python $R/code/run_nogo_volume.py --object $1 --region quarter-ball \
      --full-grid --xy-range-mm -300 300 --z-range-mm $2 $3 \
      --floor-model mesh --floor-tolerance-mm 0 --contact-policy contact-fingers \
      --ggx-candidates artifacts/paper_15_object_grasp_success/graspgenx_candidates \
      --ggx-topk 10 --out $R/results
done
# GraspGen-X at 100 proposals. The runner now prunes proposals that fail with a
# zero-length approach before sweeping (exact: a failure at the proposal pose
# or during closure does not depend on the start). Strawberry: 0 of 100
# survive, so the k = 10 sweep already is the k = 100 answer; check with
#   uv run python $R/code/ggx_start_independent_check.py --object paper_strawberry \
#       --ggx-candidates artifacts/paper_15_object_grasp_success/graspgenx_candidates
# Can: 41 of 100 survive. results/ggx100/nogo_volume_paper_can.npz is a k = 100
# sweep on the runner's default grid (xy +-240 mm, z 0-330 mm: 995 of the 1206
# poses), GGX 3.4% unreachable; the full-grid rerun was stopped.
uv run python $R/code/run_nogo_volume.py --object paper_nut \
    --ggx-candidates artifacts/paper_15_object_grasp_success/graspgenx_candidates \
    --verify-monotonic 6 --out $R/results   # ~40 min
MUJOCO_GL=egl uv run python $R/code/render_nogo_volume.py --object paper_nut \
    --crop-mm 150 --plain-floor --show-hand \
    --out $R/figures/20260908-nogo_volumes_nut.png

uv run python $R/code/make_figures.py
bash $R/code/render_strips.sh            # strips + figures/20260908-nogo_grid_strawberry_can.png
uv run python $R/code/build_docpage.py
```

The primary run and its tolerance variants (`--floor-model mesh
--floor-tolerance-mm 0 --contact-policy contact-fingers --tag
paper_conditions_meshfloor`, plus `--ggx-candidates ... --tag
paper_conditions_meshfloor_ggx`) are the commands in the header comment of
`20260908-strategy_reachability.tex`. The GraspGen-X candidate files live under
`artifacts/`, which is not tracked; the resulting per-trial CSVs are.

Look at any configuration, with the numbers that pair with the picture:

```bash
MUJOCO_GL=egl uv run python $R/code/mj_snap_rh56.py \
    --object paper_nut --strategy all --stage terminal --views side,iso --out /tmp/look.png
MUJOCO_GL=egl uv run python $R/code/mj_snap_rh56.py \
    --object paper_nut --strategy naive --filmstrip 5 --views side --cols 5 --out /tmp/sweep.png
```

Simulation only. No hardware, no dynamics, no friction, no object compliance.
This is a kinematic feasibility model and not a grasp-success prediction.
