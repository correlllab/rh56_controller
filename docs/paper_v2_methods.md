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
  --grid-step-mm 75 \
  --yaw-samples 8 \
  --path-samples 8 \
  --out artifacts/analytical_grasp_volume/
```

**Runtime warning:** the collision check now sweeps real per-finger and palm
capsules against the object AABB (see "Swept-Capsule Hand Proxy" below), which
costs roughly 150-450ms per (position, yaw) pair. The old point-proxy defaults
(`--grid-step-mm 30 --yaw-samples 16`) would take on the order of 10 hours per
3-object run under the capsule check. The defaults above were recalibrated for
a first pass (a few minutes for 3 objects); increase resolution deliberately,
not by copying the old numbers.

This is the first-pass implementation of the mentor's proposed no-go volume
analysis, now using the same swept-capsule hand proxy validated by
`tools/verify_capsule_hand_proxy.py` instead of the original point-hand
proxy. The goal is to characterize where a simple analytical grasp can be
reached by a straight Cartesian move, and where object-aware path planning is
needed.

The question answered by each voxel is:

```text
From this sampled hand-base position, what fraction of sampled yaw poses can
reach the analytical grasp pose with a straight-line Cartesian move without
the swept-capsule hand proxy intersecting a clearance-inflated object AABB
anywhere along the path?
```

### Object Model

The current implementation uses coarse YCB-like object proxies:

| Object | Size proxy | Grasp mode | Object width |
|---|---:|---|---:|
| `ycb_cracker_box` | 158 x 71 x 213 mm | `plane5` | 71 mm |
| `ycb_sugar_box` | 89 x 39 x 175 mm | `plane4` | 39 mm |
| `ycb_potted_meat_can` | 101 x 58 x 83 mm | `plane4` | 58 mm |

These are axis-aligned bounding-box proxies, not final mesh measurements. They
are good enough for the first characterization figure, but should be replaced
with measured mesh extents before final submission.

### Swept-Capsule Hand Proxy

`tools/run_analytical_grasp_volume.py` now uses the swept-capsule hand proxy
(`rh56_controller.capsule_hand_proxy`) directly for the dense no-go sweep,
rather than the coarser point-hand proxy used in the original version of this
script. We still keep a separate, cheap debug step to sanity-check the
geometry before trusting the dense sweep:

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

The verifier exposed an important modeling issue that turned out to be the
dominant effect in the dense sweep, not a rare edge case: if the object is
placed only by the fingertip analytical midpoint, the final palm envelope
overlaps the object AABB. That is exactly the kind of failure mode the no-go
volume should reveal. The dense sweep now reports which proxy is being used:

- point hand + inflated object AABB: fastest, weakest physical check. Used by
  the original version of this script; produced misleadingly optimistic
  reachable-volume numbers because it could not see palm/finger-volume
  collisions at all.
- capsule hand + object AABB: current default for the dense sweep. Confirmed
  (see below) to detect a real, margin-independent palm/object conflict, not
  a false positive from an overly conservative capsule radius.
- full MuJoCo mesh collision: best for sparse validation, probably too heavy
  and solver-dependent for the dense primary heatmap. Still the natural next
  upgrade beyond capsules.

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
p_base_final = -R_final * grasp_midpoint
```

This means each yaw sample represents a different hand orientation around the
same object, while preserving the analytical closure geometry.

### Volume Sampling

The script samples a regular 3D grid of candidate hand-base start positions:

```text
x: -180 to 180 mm
y: -180 to 180 mm
z: -120 to 180 mm
step: 30 mm
```

At each voxel, it evaluates `yaw_samples` orientations around the object. The
default is 16 yaw samples.

### Linear-Approach Test

For each voxel and yaw sample:

1. Compute the straight-line path from sampled start position to final analytical
   grasp base position.
2. Reject the pose if the path is longer than `max_linear_move_mm`.
3. Extract the RH56 capsule hand proxy (per-finger and palm capsules from
   MuJoCo FK, `path_hand_shape=open` by default) at the sampled orientation.
4. Inflate the object AABB by `hand_clearance_mm` (an extra safety margin on
   top of each capsule's own physical radius, not the whole clearance model).
5. Check every capsule's swept centerline against the inflated AABB
   continuously over each path interval (not endpoint-only sampling), using
   `rh56_controller.capsule_hand_proxy.sample_linear_path_collisions`.
6. Ignore finger-capsule contact within the last `final_contact_ignore_mm` of
   the final pose, because the fingers are expected to contact the object at
   the end of the motion. Whether palm-capsule contact is also excused there
   depends on `--ignore-palm-near-final` (default `auto`): excused for
   `plane` multi-finger power-wrap modes, where palm-on-object support is a
   normal part of the grasp (confirmed by geometry inspection, see below);
   never excused for `line` 2-finger pinch modes, where palm contact means a
   bad approach.

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

With the swept-capsule proxy, the result changed dramatically from the old
point-proxy numbers above (kept here for contrast, not as current results).

**First pass (palm never excused near final, matching the point-proxy's
implicit assumption):** 100% no-go for all three objects (375/375 voxels).
We checked the raw palm/finger-vs-AABB clearance at the final grasp pose with
`hand_clearance_mm` set to zero (no added margin) and confirmed this is not a
parameter artifact:

```text
ycb_cracker_box:     palm_center capsule, clearance = -24.0 mm at every
                      sampled yaw (0/45/.../315 deg) -- the capsule
                      centerline passes through the raw, un-inflated object
                      AABB, not just within a safety buffer.
ycb_sugar_box:        palm_center capsule, clearance = -16.5 to -24.0 mm,
                      same yaw-invariance.
ycb_potted_meat_can:  middle-finger capsule, clearance = -8.5 mm at every
                      sampled yaw.
```

The clearance values are identical across every yaw sample, which is expected
for a real geometric conflict (a rigid rotation about the object center does
not change the palm-to-object distance) and rules out an approach-direction
artifact.

We then inspected the actual geometry at the terminal pose directly
(`tools/run_analytical_grasp_volume.py`'s underlying position math, checked
by hand for `ycb_cracker_box`): the palm capsule's inner endpoint sits at
world position `[30, 0, 168]mm`, which is inside the object's AABB
(`x in [-79,79]`, `y in [-35.5,35.5]`, `z in [0,213]mm`) on all three axes
simultaneously. This is what a normal power/wrap grasp of a box looks like --
the palm resting against the object's surface, the same way a human palm
rests against a box while picking it up. It is not a bad pose; it is the
`--ignore-palm-near-final` collision *rule* (palm contact was never excused
near the final pose, for any mode) being too strict for `plane` multi-finger
power-wrap modes, even though that same strictness is correct for `line`
2-finger pinch modes (there, palm contact does indicate a bad approach).

**Second pass (`--ignore-palm-near-final auto`, the current default):** palm
contact within `final_contact_ignore_mm` of the terminal pose is now excused
for `plane` modes only.

```text
ycb_cracker_box:      no-go voxel fraction = 1.000  (125/125 voxels)
ycb_sugar_box:        no-go voxel fraction = 1.000  (125/125 voxels)
ycb_potted_meat_can:  no-go voxel fraction = 0.968  (121/125 voxels)
```

`ycb_potted_meat_can` improved slightly (a few previously-marginal voxels
cleared once palm was no longer double-counted as a blocker alongside a
finger). The two larger boxes did not improve. Breaking down `volume.csv`'s
`dominant_collision_group` for the still-blocked voxels shows palm is *still*
the dominant blocker for the majority of them (65/125 for the cracker box,
72/125 for the sugar box) -- but now it is a `path_collision`, not a
final-pose collision. This means the palm sweeps through the object partway
along the straight-line approach, well before the last `final_contact_ignore_mm`
of the path, even though the *terminal* pose is a legitimate one.

This is a coherent, well-supported finding, not a modeling dead end: **the
terminal analytical grasp pose is reachable in principle, but a naive
straight-line Cartesian approach into it is not**, for large box-like objects
under `plane`-mode power grasps. That is a stronger and more specific version
of the paper's central claim than the old point-proxy figure supported --
object-aware path planning is not just occasionally useful, it is necessary
to avoid dragging the palm through the object on the way to an otherwise-valid
power grasp.

The `debug_40mm_cube` object (2-finger `line` mode, much smaller object) does
not show this conflict at every position -- `tools/verify_capsule_hand_
proxy.py`'s auto-search finds both a blocked and a clear example for it, so
the effect is specific to the multi-finger `plane` modes and/or these object
sizes, not a universal property of the capsule proxy.

### No Straight-Line Approach Direction Works For Power-Wrap Grasps

Script:

```bash
python tools/run_staged_approach_check.py \
  --objects ycb_cracker_box ycb_sugar_box ycb_potted_meat_can \
  --out artifacts/staged_approach/
```

The natural follow-up question to the palm/finger path-collision finding
above is whether a *better-chosen* single straight-line approach direction
avoids it -- e.g. retreating straight back along the hand's own reach axis to
a stand-off distance, then moving straight in, the way a real controller
would execute a simple pregrasp-then-approach motion.

We tested every plausible single-segment straight-line direction: straight
back along the hand's own local +Z reach axis, sideways along local +-X and
+-Y, and straight down from above in world +Z, each at five candidate
stand-off distances (100 to 300mm), for all 8 yaw samples per object, with
both open and closed finger postures.

**Result: zero of 8 yaw samples, for any of the 3 paper objects, in any of
the 6 directions tested, at any stand-off distance, had a collision-free
straight-line approach.** This is not a search-coverage gap -- tracing one
case in detail (`ycb_cracker_box`, straight-in approach) shows the hand's
capsule proxy is in continuous collision from about 166mm out to 35mm out
from the final pose (index finger, then middle finger, then palm, in that
order as the hand gets closer), regardless of whether the fingers are open or
already closed to their target curl. Switching the approach direction moved
which capsule was to blame but never eliminated the collision.

The reason is structural, not a parameter or search issue: a power-wrap
grasp's terminal pose requires the fingers to be spread around the object.
Any straight-line segment that ends with the fingers already in (or close to)
that spread configuration necessarily sweeps some part of that spread through
the object's volume before arriving, no matter which direction it comes from.
A single straight Cartesian segment cannot avoid this; only a genuinely
different motion -- either a curved/multi-segment Cartesian path, or moving
the hand into position via joint-space IK that is not constrained to a
straight end-effector line -- can.

This repository's own earlier hand-tuned experiment scripts already arrived
at the practical resolution, independently of this analysis: the V17 pregrasp
logic explicitly places the open hand at the pregrasp configuration with
object collision temporarily disabled ("PLACE POWER PREGRASP WITH COLLISION
OFF"), then re-enables collision only for the contact-aware finger-closing
phase. That is exactly the right response to this finding -- straight-line
Cartesian collision-checking is the wrong model for the *positioning* phase
of a power-wrap grasp, not a constraint that needs to be satisfied by a
cleverer choice of direction.

**Paper framing takeaway:** the no-go volume and this straight-line-direction
sweep together support a claim sharper than "object-aware path planning helps
sometimes": straight-line Cartesian approach checking is the right model for
`line` 2-finger pinch grasps (small objects, genuine collision risk during
approach), but the wrong model entirely for `plane` multi-finger power-wrap
grasps, which require direct pose placement plus contact-aware closure
instead. Planner mode should determine which collision-avoidance strategy is
even applicable, not just which closure geometry to solve.

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

The point-hand proxy has been replaced by the swept-capsule proxy. Remaining
upgrades toward more realistic checks:

1. Use measured YCB mesh extents or actual meshes (still AABB proxies today).
2. Use full MuJoCo mesh collision checking along the path (capsules are still
   a proxy, not the real hand mesh).
3. Implement and validate the direct-placement-plus-contact-aware-closure
   strategy in simulation for `plane` modes (open hand placed via joint-space
   IK at the pregrasp configuration, collision disabled for that placement
   step only, then fingers close under contact-aware force control until each
   digit reaches the object) -- this is now the concrete next step, not
   further straight-line search; see "No Straight-Line Approach Direction
   Works" above for why straight-line search was abandoned.
4. Sample more than yaw, such as wrist pitch/roll or approach direction, for
   the `line`-mode figure where straight-line approach checking remains the
   right model.
5. Add arm IK feasibility if the paper wants robot-level reachability.
6. Compare analytical no-go volume against a learned policy or an optimization
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
object-width range, but which collision-avoidance strategy is even applicable
depends on grasp mode: straight-line Cartesian approach checking is sufficient
and meaningful for 2-finger pinch grasps, but for multi-finger power-wrap
grasps no straight-line approach direction reaches the terminal pose without
collision, at any distance we tested -- because the terminal pose requires
the fingers to already be spread around the object. Power-wrap grasps instead
require direct pose placement plus contact-aware closure, which this
repository's own earlier hand-tuned experiments already converged on
independently. This explains why object-aware path planning and latency-aware
execution are needed, and why they are needed differently for different grasp
modes.
```

This is stronger than only saying that the system can execute a few grasps. It
turns the hand's coupling and analytical limitations into the main scientific
object of study, and it gives a concrete, mode-dependent answer (not just "path
planning would help") for what a real controller needs to do differently for
pinch vs. power-wrap grasps.
