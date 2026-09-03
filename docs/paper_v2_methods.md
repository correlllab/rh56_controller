# Paper-V2 Methods Notes

This note explains the methods behind the current paper-v2 figures and scripts.
It is written as an internal methods draft: detailed enough to understand the
logic, but still honest about which parts are first-pass proxies rather than
final physics simulation.

## 1. Analytical Planner Width Characterization

Script:

```bash
python tools/run_planner_sweep.py \
  --modes line plane3 plane4 plane5 \
  --width-min-mm 5 \
  --width-max-mm 115 \
  --width-step-mm 1 \
  --object-width-offset-mm 20 \
  --out artifacts/planner_sweep/
```

The planner sweep characterizes the scalar width range covered by the existing
analytical closure solver in `rh56_controller/grasp_geometry.py`.

The important detail is that the raw solver width is an internal site-to-site
measurement. The MuJoCo sites are embedded inside the fingertips, so the object
width reported in the paper is approximately:

```text
object_width_mm = internal_site_width_mm - 20
```

This is why the paper-facing range is closer to 0-95 mm even though the raw
nominal solver range is around 20.5-122.6 mm for the plane modes.

For each requested internal width and grasp mode, the script:

1. Calls the corresponding analytical closure primitive:
   - `closure.line(width)` for 2-finger pinch.
   - `closure.plane(width, n_fingers=3/4/5)` for planar multi-finger grasps.
2. Records whether the requested internal width lies inside the mode-specific
   nominal solver range.
3. Records the achieved internal width error after the multi-finger
   coplanarity corrections.
4. Writes both internal and object-corrected widths to `summary.csv`.

The output columns intentionally separate:

- `internal_width_mm`: the solver/site-based width.
- `object_width_mm`: the paper-facing corrected width.
- `nominal_feasible`: whether the internal request is inside the nominal range.
- `width_match`: whether the achieved internal width is within tolerance.

`nominal_feasible` and `width_match` are not grasp-success metrics. They only
characterize the analytical geometry.

Cylinder mode is still available in the script, but it is excluded from the
paper-facing default. The reason is that cylinder/power grasp rollout needs
object-aware palm-centering path planning. A bad cylinder rollout is not just a
closure-width issue; the hand has to move around the object and center the palm
relative to it.

The solve-time plot should also be interpreted carefully. The measured time is
offline analytical solve/table-generation time. In the deployed controller, this
can be precomputed into a lookup table:

```text
(grasp mode, object width) -> qpos / control vector
```

So online execution should be discussed as lookup-table execution, not as the
Python solve-time shown in the diagnostic plot.

## 2. Analytical No-Go Volume

Script:

```bash
python tools/run_analytical_grasp_volume.py \
  --objects ycb_cracker_box ycb_sugar_box ycb_potted_meat_can \
  --collision-model capsule \
  --path-hand-shape open \
  --x-range-mm -240 240 \
  --y-range-mm -240 240 \
  --z-range-mm 40 280 \
  --grid-step-mm 60 \
  --yaw-samples 8 \
  --path-samples 8 \
  --out artifacts/analytical_grasp_volume/
```

This is the first-pass implementation of the mentor's proposed no-go volume
analysis. The goal is to characterize where a simple analytical grasp can be
reached by a straight Cartesian move, and where object-aware path planning is
needed.

The question answered by each voxel is:

```text
From this sampled hand-base position, what fraction of sampled yaw poses can
reach the analytical grasp pose with a straight-line Cartesian move without
the swept capsule hand proxy colliding with the object AABB?
```

### Object Model

The current implementation uses coarse YCB-like object proxies:

| Object | Size proxy | Grasp mode | Object width |
|---|---:|---|---:|
| `ycb_cracker_box` | 158 x 71 x 213 mm | `plane5` | 71 mm |
| `ycb_sugar_box` | 89 x 39 x 175 mm | `plane4` | 39 mm |
| `ycb_potted_meat_can` | 101 x 58 x 83 mm | `plane4` | 58 mm |

These are tabletop axis-aligned bounding-box proxies, not final mesh
measurements. The object center is placed at `[0, 0, height / 2]`, so the bottom
face sits on the ground plane instead of half the object spawning under it. The
proxies are good enough for the first characterization figure, but should be
validated against measured mesh extents before final submission.

### Capsule Hand Proxy Verification

Before using a hand-volume proxy in the dense no-go sweep, we now keep a
separate debug step:

```bash
python tools/verify_capsule_hand_proxy.py \
  --object debug_40mm_cube \
  --out artifacts/capsule_proxy_demo/
```

This script does not generate a paper result. Its job is to let us verify the
geometry before trusting the volume heatmap. It:

1. Solves the same analytical grasp pose used by the no-go script.
2. Writes the selected path-hand pose into the MuJoCo hand model.
3. Extracts capsule endpoints from MuJoCo body origins and fingertip sites.
4. Adds a conservative palm capsule envelope in the hand-base frame.
5. Checks every capsule over each straight-line path interval against the
   object AABB using a swept capsule-centerline distance calculation.

The default verifier object is `debug_40mm_cube`, a 40 x 40 x 40 mm cube whose
AABB center is placed at `[0, 0, 20]` mm so its bottom face sits on the ground
plane at `z = 0`. This object size and the default open hand pose are for
debugging the capsule collision logic itself. The YCB-like
objects remain available through `--object ycb_sugar_box`,
`--object ycb_cracker_box`, and `--object ycb_potted_meat_can`.

The useful outputs are:

- `capsules.csv`: capsule endpoints and radii in both hand-base and final-world
  frames.
- `path_samples.csv`: collision/clearance at each path sample.
  This includes raw collision count, ignored near-target collision count, and
  active collision count after the final-contact rule. For rows after the
  start pose, the reported clearance is the minimum over the swept interval,
  not only the endpoint sample.
- `summary.csv`: whether each demo path collides before the ignored final
  contact region.
- `capsule_proxy_demo.png`: visual check of object AABB, capsules, and path
  samples.

The script also supports:

```bash
python tools/verify_capsule_hand_proxy.py --viewer
```

which opens an interactive MuJoCo viewer with the actual hand mesh and the
capsule overlay. This viewer is a visual sanity check only; the CSV outputs are
the reproducible record.

For interactive path debugging, use:

```bash
python tools/capsule_path_gui.py
```

This starts the hand away from the object and lets the hand-base start pose be
moved interactively. By default, the expensive swept-capsule path check is
manual: the mesh moves immediately, the status becomes stale, and the path is
only recomputed after pressing `Enter` or `V`. Use `--auto-check` to restore
the older realtime recomputation behavior. The default is intentionally a
debugging setting:

```text
object = debug_40mm_cube
path_hand_shape = open
final_contact_ignore_mm = 0
final_ignore_groups = fingers
```

This default is intentionally strict: the whole straight-line path must be
collision-free. `path_hand_shape = open` keeps the finger flexion qpos values
at zero but sets `thumb_yaw` to its maximum MuJoCo qpos, corresponding to the
real-hand thumb rotation command moving from raw 1000 to raw 0. For
contact-region experiments, `--final-contact-ignore-mm` can be increased. That
ignore rule is applied per capsule and is not allowed to hide palm or wrist
collisions unless `--final-ignore-groups all` is explicitly requested. The
terminal prints raw/ignored/active collision counts and the final-pose
clearance, so overlap at the analytical target remains visible instead of
hidden.

For comparison with the older, overly permissive behavior, the GUI supports
`--final-ignore-groups all`, but that mode should be treated as visual debugging
only because it can hide palm/object collisions.

Keyboard controls:

```text
W/S: move x + / -
A/D: move y + / -
R/F: move z + / -
J/L: yaw target - / +
=/-: increase/decrease step
Enter or V: validate the current path
Space: animate the latest checked valid path
C: reset the moving hand to the start pose
Q or Esc: quit
```

### Strategy Pre-Grasp Demo

Before turning this into a dense volume sweep, we now keep a one-example demo
for the three paper-level execution strategies:

```bash
python tools/demo_strategy_pregrasp_collision.py \
  --object debug_40mm_cube \
  --mode plane4 \
  --out artifacts/strategy_pregrasp_demo/
```

This is not a paper result. Its job is to verify that the strategy-specific
pre-grasp hand shapes are the ones we intend to analyze:

- `naive`: fully open fingers with thumb yaw already rotated to the final
  opposing direction; thumb bend and non-thumb fingers close together after the
  arm reaches the final grasp pose.
- `iterative_closure`: the analytical planner posture at a wider approach
  width. The default policy is `--iterative-pregrasp-policy planner-max-width`,
  which uses the same mode-specific maximum width available to the analytical
  planner. `--iterative-pregrasp-policy final-plus-preopen` is kept only as a
  comparison/debug option.
- `thumb_reflex`: final thumb bend/yaw with all non-thumb fingers open.

For one fixed object and one fixed analytical grasp target, the script
uses each strategy's pre-grasp hand shape and checks whether the capsule hand
proxy can move from a sampled hand-base start pose to the pre-grasp target without
hitting either:

- the object's box/cylinder/sphere collision primitive, or
- the floor plane at `z = 0`.

It writes `summary.csv`, `assumptions.json`, and
`strategy_pregrasp_demo.png`. This is the gate for deciding whether the
strategy-specific no-go volume is worth running. For example, a low-object
case such as:

```bash
python tools/demo_strategy_pregrasp_collision.py \
  --object debug_20mm_cube \
  --mode plane4 \
  --out artifacts/strategy_pregrasp_demo_20mm/
```

shows that floor collision can dominate the feasibility decision. That is the
behavior we need before claiming that a larger no-go volume comes from an
extended thumb or other strategy-specific hand geometry.

### Strategy Pre-Grasp Feasible Rate

After the one-example demo looks sensible, the first coarse rate script is:

```bash
python tools/run_strategy_pregrasp_rate.py \
  --object debug_40mm_cube \
  --mode line \
  --iterative-pregrasp-policy planner-max-width \
  --start-sampler paper-approach-points \
  --start-reference grasp-center \
  --approach-axis y- \
  --out artifacts/strategy_pregrasp_rate_40mm/
```

This keeps the same object model and strategy definitions as the demo, but
samples the paper-style initial hand positions instead of a rectangular debug
grid. The sampled point is the pre-grasp center, not the raw hand-base pose.
The three strategies are fixed across objects. Object identity changes the
selected analytical grasp mode, target width, and grasp target point; it does
not introduce a hand-tuned pre-grasp shape. For the Plan/iterative strategy,
the default pre-grasp width is the planner's maximum feasible width for the
selected mode, unless `--iterative-width-mm` is explicitly provided.

The object proxy now separates two concepts:

- `aabb_center`: the geometric center of the tabletop collision AABB.
- `grasp_target`: the point where the analytical grasp center should land.

Small/debug objects default to `grasp_target = aabb_center`. For tall upright
paper objects such as bottles, cups, cans, mustard, and sugar boxes, the
default target is 10 mm below the top of the object. A fixed top offset matches
the experimental setup more directly than a height fraction: a 72% target on
a 190 mm bottle would still be about 53 mm below its top. The 10 mm offset is
an explicit first-pass metadata assumption and can be overridden with
`--grasp-target-top-offset-mm`; it should be replaced by measured or annotated
object-specific grasp targets before reporting final paper numbers. Round
objects such as the orange retain a fractional target because "10 mm below
the top" does not describe the intended contact band as well.
Open cups are an explicit exception: Metal Cup and Paper Cup use a rim-level
target (`0 mm` below the top), matching the top-rim approach visible in the
experiments. Their primitive remains a conservative solid cylinder, so this
target choice avoids a false collision with the primitive's nonexistent closed
top face. The 12 mm-thick Pen also uses a top-surface pre-grasp target: placing
the target at its 6 mm geometric center forces the capsule hand into the table,
whereas the top target preserves table clearance for the subsequent pinch.

Capsule/object contact uses a `0.01 mm` numerical epsilon. Analytical contact
can otherwise produce clearances around `-0.002 mm` from optimizer precision,
which incorrectly marks intended fingertip tangency as penetration. The
epsilon is several orders of magnitude below the hand-capsule radii and does
not mask millimetre-scale collision.
The current default point set is:

```text
P1: approach distance = 250 mm, lateral d = 0 mm, h = 0 mm
P2: approach distance = 0 mm, lateral d = 0 mm, h = 250 mm
level 1: approach distance = 250 mm, h = 100 mm,
         lateral d = -150, -50, 50, 150 mm
level 2: approach distance = 250 mm, h = 250 mm,
         lateral d = -150, -50, 50, 150 mm
```

The important geometry is that P1, P2, and the grasp point define the vertical
approach plane. The eight level points are not in that same plane. They are in
the vertical plane through P1 that is perpendicular to the approach plane, so
they keep the same approach distance as P1 and vary laterally by `d`. `h` is
measured upward from the final grasp center. With `--approach-axis y-`, P1 is
250 mm on the `y-` side of the object, the P1-to-grasp movement direction is
world `y+`, and the level-point `d` axis is world `x`.

These points are interpreted as strategy grasp-center waypoints relative to
the grasp target, not raw hand-base positions or raw AABB-center offsets. For
each strategy, the script converts the sampled grasp-center waypoint to the
corresponding hand-base start pose. The hand yaw is set from the approach axis
so the hand faces the object along the
P1-to-grasp movement direction, then applies the paper-view hand yaw offset.
The current default offset is `--paper-hand-yaw-offset-deg -90`, which rotates
the hand 90 degrees right in the world xy plane. A start is counted as feasible
only if the capsule path to that strategy's pre-grasp target avoids both:

- the object's shape-aware collision primitive, and
- the floor plane at `z = 0`.

The script writes `final_grasp_center_error_mm_max` and
`target_grasp_center_error_mm_max_by_strategy` to `assumptions.json`. These
check that the analytical grasp center and each fixed pre-grasp center are
aligned with the requested `grasp_target`, which may differ from the object
AABB center. With the current paper-point default,
`yaw_samples = 1`; yaw sweeps are reserved for later sensitivity analysis, not
the first paper-facing figure.
The default center policy is `antipodal`: the midpoint between the thumb tip
and the centroid of the non-thumb fingertips. The legacy
`contact-centroid` policy is still available for comparison with the viewer's
historical `ClosureResult.midpoint` placement, but it can bias multi-finger
objects toward the finger side.

Two older samplers are still available for debugging:

- `--start-sampler final-offset-grid`: each sampled hand-base start is
  `final_grasp_base(yaw) + [dx, dy, dz]` from the configured offset grid.
- `--start-sampler approach-plane-grid`: samples a rectangular side-approach
  grid. This is useful for coarse exploration but no longer the default paper
  debug geometry.
- `--start-sampler object-top-grid`: samples a horizontal grid above the object.
  This is useful for geometric debugging, but it is less representative of the
  physical approach setup.

The output includes:

- `summary.csv`: aggregate feasible rate per strategy.
- `volume.csv`: one row per strategy, paper point, and yaw angle.
  The paper points are stored as grasp-center references in
  `start_center_*`/`target_center_*`; the actual wrist/base control poses are
  stored separately in `start_*`/`target_*`.
- `strategy_feasible_rate.png`: aggregate bar plot.
- `strategy_paper_approach_points_validity.png`: for paper-point runs, each cell
  shows the valid fraction at that lateral `d/h` point.
- `strategy_approach_plane_validity.png`: for rectangular approach-plane debug
  runs, each cell shows the valid fraction at that lateral/height grid point.
- `strategy_top_grid_yaw_fraction.png`: for top-grid debugging runs, each cell
  shows the fraction of yaw samples that were valid at that grid point.

The current 10-point paper-style run is not a final paper number. It is a
sanity check for whether the strategy-specific collision logic produces
different access rates before scaling to denser grids or additional objects.
The capsule path check is not free: a 3 x 3 x 36 yaw run already takes a few
minutes, so dense sweeps should be staged, cached, or optimized rather than
blindly expanded to full 6D sampling.

For the current paper object set, use:

```bash
python tools/run_strategy_pregrasp_paper_batch.py \
  --out artifacts/strategy_pregrasp_rate_paper_objects_top_10mm/
```

This runs the same 10 paper-style approach points for the 15 grasping objects
shown on the project website: big screwdriver, bottle, can, charger, metal cup,
mustard, orange, pen, small screwdriver, sugar box, egg, nut, paper cup,
raspberry, and strawberry. The first-pass `paper_*` entries use estimated
tabletop primitives in `rh56_controller/paper_v2_objects.py`: Bottle, Can,
Metal Cup, and Paper Cup use upright cylinders; Orange uses a sphere; box-like
and unmeasured irregular objects retain boxes. These are not measured meshes,
so the rates remain exploratory until object dimensions and grasp alignment
are validated.

For grasp-target sensitivity, the rate script also supports:

```bash
python tools/run_strategy_pregrasp_rate.py \
  --object paper_bottle \
  --mode object-default \
  --start-sampler paper-approach-points \
  --start-reference grasp-center \
  --approach-axis y- \
  --grasp-target-top-offset-mm 10 \
  --grasp-target-approach-offset-mm 25 \
  --out artifacts/strategy_pregrasp_rate_bottle_target_sensitivity/
```

The fixed-top-offset, shape-aware batch is written to
`artifacts/strategy_pregrasp_rate_paper_objects_top_10mm/`. Moving the target
to 10 mm below the top changes several large-object results substantially: can,
mustard, and sugar box now contain feasible approach points, confirming that
the previous proportional targets were too low. Shape-aware collision removes
the artificial corners of round-object AABBs without changing the hand
capsules or path definition. This experiment evaluates only the fixed
pre-grasp hand shape along the start-to-target approach path. The target is
the endpoint of path planning, not a simulated closed hand. The experiment
does not simulate or score finger closure after arrival.

For visual verification, render the exact sampled paths with:

```bash
python tools/render_strategy_grid_paths.py \
  --volume artifacts/strategy_pregrasp_rate_40mm/volume.csv \
  --strategy all \
  --out artifacts/strategy_pregrasp_grid_paths_40mm/
```

The GIFs animate the hand mesh with wrist/base position control, but the
visible markers and path lines are the grasp-center points. This distinction is
important: the paper-style points describe where the hand's grasp point should
move, not where the wrist origin should be placed.

To render all 15 objects, three strategies, and 10 start points into one
streamed MP4, use:

```bash
UV_PROJECT_ENVIRONMENT=.venv312 uv run --extra video \
  python tools/render_strategy_paper_trials.py
```

This writes `artifacts/current/all_trials/all_trials.mp4` together with
`trial_index.csv`, which maps every object/strategy/point trial to an exact
timestamp. The video keeps each pre-grasp hand shape fixed during the linear
approach. It does not render or evaluate finger closure after arrival.

Generate the paper-object overview figures with:

```bash
UV_PROJECT_ENVIRONMENT=.venv312 uv run \
  python tools/plot_strategy_pregrasp_overview.py
```

The output under `artifacts/current/visualizations/` separates three questions:

- `object_strategy_feasible_rate.png`: which pre-grasp strategies can approach
  each object from the 10 sampled starts.
- `start_point_feasible_rate.png`: which P1-P10 start locations are generally
  accessible across all, YCB/YCB-like, and delicate objects.
- `failure_modes.png`: whether failed paths are blocked by the object or floor,
  during motion or at the target pose.

### Sweep Scaling / Optimization Plan

For larger sweeps, use a staged plan:

1. Start with the 10 paper-style approach points and `yaw_samples = 1`.
2. Increase position density before adding yaw. For example, use 50 mm spacing
   over the same plane to identify interesting boundary regions.
3. Add yaw only as a sensitivity sweep after the position grid is stable.
4. Use coarse-to-fine refinement: run a sparse grid, then densify near cells
   whose neighboring validity differs.
5. Add an early-exit viability path for dense sweeps. The current function keeps
   detailed clearance diagnostics, which is useful for debugging but slower
   than a boolean "first collision fails" check.
6. Parallelize over strategy/start/yaw samples once the sampler is finalized.
7. Cache all strategy-specific capsule proxies and rotation/grasp-center terms;
   those should not be recomputed for every grid point.

The current verifier has already exposed an important modeling issue: if the
object is placed only by the fingertip analytical midpoint, the final palm
envelope can overlap the object AABB. That is exactly the kind of failure mode
the no-go volume should reveal, but it also means the dense sweep should report
which proxy is being used:

- point hand + inflated object AABB: fastest, weakest physical check.
- capsule hand + object AABB: current paper-facing first pass for hand-volume
  feasibility.
- full MuJoCo mesh collision: best for sparse validation, probably too heavy
  and solver-dependent for the dense primary heatmap.

### Analytical Grasp Pose

For each object:

1. The script selects an analytical grasp mode and object width.
2. It converts object width to internal solver width:

   ```text
   internal_width = object_width + 20 mm
   ```

3. It calls the analytical closure solver to obtain a `ClosureResult`.
4. The resulting `ClosureResult.midpoint` is treated as the grasp midpoint that
   should align with the object center.

For each sampled yaw angle, the final hand-base pose is computed so that the
analytical grasp midpoint lands at the object center:

```text
R_final = Rz(yaw) * R_grasp_tilt
p_base_final = object_center - R_final * grasp_midpoint
```

This means each yaw sample represents a different hand orientation around the
same object, while preserving the analytical closure geometry.

### Volume Sampling

The script samples a regular 3D grid of candidate hand-base start positions:

```text
x: -240 to 240 mm
y: -240 to 240 mm
z: 40 to 280 mm
step: 60 mm
```

At each voxel, it evaluates `yaw_samples` orientations around the object. The
current capsule default is 8 yaw samples.

### Linear-Approach Test

For each voxel and yaw sample:

1. Compute the straight-line path from sampled start position to final analytical
   grasp base position.
2. Reject the pose if the path is longer than `max_linear_move_mm`.
3. Put the hand in the selected path shape. The current default is
   `--path-hand-shape open`, which keeps fingers open and sets thumb yaw to the
   maximum qpos, matching the collision-avoiding pre-grasp shape we want to use.
4. Build a capsule proxy from MuJoCo FK body origins, fingertip sites, and a
   conservative palm envelope.
5. Sweep every capsule along the sampled straight-line path.
6. Reject the yaw sample if any active capsule intersects the object AABB before
   the final allowed contact region.

The older `--collision-model point-inflated` path is still available as a fast
debug baseline, but it is too weak for the paper-facing no-go volume because it
does not model the hand's occupied volume.

The voxel score is:

```text
viability = viable_yaw_samples / total_yaw_samples
```

The no-go volume is the set of voxels where:

```text
viability = 0
```

The script writes:

- `summary.csv`: one row per object.
- `volume.csv`: one row per sampled voxel.
- `*_viability_slices.png`: 2D slices through the volume.
- `*_no_go_volume.png`: 3D scatter visualization.
- `assumptions.json`: all geometry/sampling assumptions.

### What The Current No-Go Volume Shows

The current result is a coarse characterization of where simple analytical
linear approach is plausible under the capsule hand proxy. Each object gets a
mean viability score and a no-go voxel fraction in `summary.csv`; each sampled
voxel keeps the path blocker, clearance, and active collision count in
`volume.csv`.

These numbers should not yet be treated as final physical success rates. They
are a structured way to expose the capability boundary of the analytical method.

### What This Does Not Yet Do

The current no-go volume is not:

- full MuJoCo collision checking,
- full hand mesh sweeping,
- UR5/H1-2 arm IK,
- object pose estimation,
- object dynamics,
- contact stability,
- learning-method comparison.

It is a first-pass analytical reachability proxy. The value is that it gives us
a reproducible figure and data interface for the scientific question:

```text
Where can a kinematically coupled hand succeed with analytical grasping and a
simple linear approach, and where does it need object-aware path planning?
```

### Natural Next Upgrade

The next version should validate and refine the capsule/AABB proxy with more
realistic checks:

1. Use measured YCB mesh extents or actual meshes.
2. Use sparse MuJoCo mesh collision checks as validation points along the path.
3. Sample more than yaw, such as wrist pitch/roll or approach direction.
4. Add arm IK feasibility if the paper wants robot-level reachability.
5. Compare analytical no-go volume against a learned policy or an optimization
   planner, if we want a stronger methods comparison.

## 3. Hybrid Margin / Latency Sweep

Script:

```bash
python tools/run_hybrid_margin_sweep.py \
  --v-fast 1000 \
  --v-contact-list 10 25 50 100 \
  --margin-list 0 5 10 15 20 25 30 40 50 \
  --onset-sigma-units 7.5 \
  --out artifacts/hybrid_margin_sweep/
```

This sweep explains the anticipatory switching logic. It uses the measured
latency summary in `resource/experiment/latency_summary.json` by default. The
current default statistic is p50 latency, about 66.28 ms.

The core probability model is:

```text
p_pre_slow = normal_cdf(margin_units / onset_sigma_units)
```

Intuition:

- margin = 0 means contact may happen before or after switching, so probability
  is around 0.5.
- larger margin means the controller switches to slow/contact mode earlier.
- with margin = 25 and sigma = 7.5, the switch is about 3.33 sigma early, so
  `p_pre_slow` is very high.

Expected latency travel is modeled as:

```text
latency_travel =
  p_pre_slow * v_contact * latency
  + (1 - p_pre_slow) * v_fast * latency
```

So the sweep connects:

- measured latency,
- switch margin,
- contact speed,
- predicted extra travel during sensing latency.

This is still a proxy model, but it makes the margin/contact-speed choice
explicit rather than heuristic.

## 4. Force-Threshold Replay

Script:

```bash
python tools/replay_force_thresholds.py \
  --logs 'experiment_data/peg_in_hole/*.csv' \
  --out artifacts/threshold_replay/
```

This script is infrastructure for future replay. If logs exist, it sweeps force
thresholds and reports precision/recall/detection delay. If logs do not exist,
it writes:

- `expected_log_schema.csv`
- an empty `summary.csv`
- `assumptions.json`

The current repository does not contain matching peg-in-hole logs, so this is
not a paper result yet.

## Recommended Paper Framing

The current methods support this revised paper story:

```text
We characterize the capability boundary of analytical grasping for a
kinematically coupled low-cost dexterous hand. The planner covers a measurable
object-width range, but simple analytical closure plus linear Cartesian approach
has a no-go volume around objects. This explains why object-aware path planning
and latency-aware execution are needed.
```

This is stronger than only saying that the system can execute a few grasps. It
turns the hand's coupling and analytical limitations into the main scientific
object of study.
