# Real Robot Setup — UR5e + Inspire RH56

This guide covers everything needed to run `grasp_viz` with the real UR5e arm and Inspire RH56 dexterous hand.

---

## Hardware Prerequisites

| Component | Requirement |
|:--|:--|
| UR5e robot arm | UR software ≥ 5.11, URCap ExternalControl installed |
| Inspire RH56DFX hand | Mounted on UR5e wrist-3 flange; USB–serial adapter to PC |
| Network | PC and UR5e on the same LAN; static IP recommended |
| `ur_rtde` | Installed on the PC (see below) |
| `magpie_control` submodule | Already bundled as `./magpie_control/` |
| `spatialmath` | Installed as a dependency of magpie_control |

---

## Software Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/correlllab/rh56_controller.git
cd rh56_controller
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### 2. Install the uv environment with the real-ur5 profile

```bash
tools/setup_uv_env.sh --profile real-ur5 --python 3.12 --env .venv312 --telemetry
```

Equivalent raw uv command:
```bash
UV_PROJECT_ENVIRONMENT=.venv312 uv sync --extra real-ur5 --extra telemetry
```

This installs `magpie_control` (from `./magpie_control/`), `spatialmath`, and `ur_rtde` in addition to the base packages.

> **Note:** `ur_rtde` is a native extension.  If `uv sync` fails to build it, install it via pip into the venv:
> ```bash
> .venv/bin/pip install ur-rtde
> ```

### 3. Verify serial access for the hand

```bash
ls /dev/ttyUSB*          # find the hand's serial port
sudo chmod 666 /dev/ttyUSB0   # if permission denied (transient)
```

---

## UR5e Network Setup

1. On the UR teach pendant: **Settings → System → Network** — assign a static IP (default used here: `192.168.0.4`).
2. Ensure the **ExternalControl URCap** is installed and its host IP matches the PC.
3. Start the ExternalControl program on the pendant before connecting.

---

## Running the Full System

```bash
uv run python -m rh56_controller.grasp_viz \
    --robot \
    --real-robot \
    --ur5-ip 192.168.0.4 \
    --port /dev/ttyUSB0
```

ROS + rerun synchronized mode:

```bash
uv run python -m rh56_controller.grasp_viz \
   --robot \
   --real-robot \
   --ur5-ip 192.168.0.4 \
   --port /dev/ttyUSB0 \
   --ros-sync \
   --rerun
```

Or via ROS launch:

```bash
ros2 launch rh56_controller grasp_viz_ros.launch.py ur5_ip:=192.168.0.4 serial_port:=/dev/ttyUSB0
```

### Key CLI flags

| Flag | Default | Description |
|:--|:--|:--|
| `--robot` | off | Enable UR5+hand sim viewer with mink differential IK |
| `--real-robot` | off | Enable real UR5 arm control panel |
| `--ur5-ip IP` | `192.168.0.4` | UR5 robot IP address |
| `--ur5-speed M/S` | `0.10` | Arm linear speed (m/s) |
| `--port DEV` | None | Serial port for the RH56 hand |
| `--send-real` | off | Start with "Send to Real" hand mirroring enabled |
| `--no-mink` | off | Skip mink planner (faster startup) |
| `--ros-sync` | off | Publish planner state on ROS2 and accept ROS command topics |
| `--ros-publish-hz HZ` | `20` | ROS2 publish rate for bridge topics |
| `--ros-send-hand-cmd` | off | Publish planned RH56 commands to `hands/cmd` |
| `--rerun` | off | Enable rerun telemetry stream from planner state |

---

## Camera target input to the existing Plan

`grasp_viz --vision-target` loads the `target_validation.yaml` produced by
`tools/analyze_ur5_camera_colored_target.py` into the existing planner. It does
not create another motion executor. The default 4-finger Plan still uses
`_compute_plan_closures` → `_build_world_T_hand` → `execute_plan_waypoints`.

The measurement is a **top-center point in UR base coordinates**, not a TCP
target. For an upright object, the adapter subtracts half the reported object
height along base Z to select a mid-height grasp; `--grasp-depth-mm` overrides
that depth below the top. It uses the existing `antipodal` center policy so the
midpoint between thumb and opposing fingers stays at the selected object center.
Without vision input, the legacy `contact-centroid` default is unchanged.

Width and yaw must be supplied because the current colored-target detector
does not estimate them. `--planner-width-offset-mm` defaults to zero. The
September 4 cube trial used 40 + 20 = 60 mm; the 20 mm is an explicit experimental
setting, not a validated offset for every object. Actual solved width is exported
separately from requested width.

Offline reproduction using the historical September 4 capture (no hardware):

```bash
RH56_CACHE_DIR=/tmp/rh56_plan_cache .venv312/bin/python tools/preview_vision_plan.py \
    --vision-target artifacts/ur5_real_cube_first_grasp/final_pregrasp_visual_analysis_20260904/target_validation.yaml \
    --object-width-mm 40 --planner-width-offset-mm 20 --vision-yaw-deg 90 \
    --out artifacts/vision_plan_preview
```

This writes `plan.json` (schema, assumptions, hand/TCP poses and raw finger
commands) and `summary.csv`. It calls the same Plan generator as the UI, without
opening a camera, robot, hand, or viewer. It is a geometry preview, not an arm IK
or collision safety check. The bundled UR viewer is UR5e; the lab arm is CB3.

Open the existing GUI with the same input for review:

```bash
RH56_CACHE_DIR=/tmp/rh56_plan_cache .venv312/bin/python -m rh56_controller.grasp_viz \
    --robot --no-mink --hand-write-only \
    --vision-target artifacts/ur5_real_cube_first_grasp/final_pregrasp_visual_analysis_20260904/target_validation.yaml \
    --object-width-mm 40 --planner-width-offset-mm 20 --vision-yaw-deg 90
```

This command has no real hardware connection flags. For a later real run, use a
fresh target report and the existing real-arm controls with strategy **Plan**.
Loading a report never refreshes camera calibration or checks scene drift.
Automatic `--send-real` / `--ros-send-hand-cmd` is rejected when loading a vision
target to avoid sending a final finger posture during initialization.

`--hand-write-only` skips executor hand telemetry and rejects nonzero Force (N)
before execution. It also skips the hand-read step when syncing the viewer.
It retains the legacy Plan order and timing: initial hand command, 0.3 s wait,
arm approach, then blocking arm move followed by finger command for each step.
There are no new per-step sleeps or simulation-timed motions. Missing hand
feedback is not reported as successful contact or successful grasp. Existing
GUI speed/force initialization is unchanged; this option is not a low-speed
preset and applies to the grasp executor, not separate force diagnostic panels.

### Explicit-speed trial runner

`tools/run_vision_plan_trial.py` remains an offline preview unless `--execute`,
`--confirm-plan` and a matching `--start-snapshot` are supplied. The legacy
`--confirm-slow-plan` confirmation spelling is also accepted. `--speed-profile`
defaults to `slow`; `planner` selects the original bridge defaults (0.10 m/s,
0.20 m/s²) and hand speed 1000. The trial hand force cap remains 100; selecting
original speed does not restore the GUI's maximum force setting.

`--phase approach-only` runs just the existing Plan's first waypoint and open
approach hand command, without closure steps, lift or release. It is not by
itself a collision guarantee.
With this phase, `--approach-hover-mm 50` stops 50 mm above the original approach
in UR base Z while preserving orientation. It does not subsequently descend.

Experimental `--local-alignment` requires a matching `--alignment-validation`
report with the frozen candidate hash and matching final hand command. It only
adapts the hand-to-TCP frame mapping, not closure geometry or waypoint generation.
This local fit was checked at one hand shape; applying it to open or intermediate
configurations remains an unvalidated extrapolation and execution requires
`--acknowledge-shape-extrapolation`. The runner reports both desired physical-hand
poses and nominal bridge-frame commands. Hardware pose feedback checks the latter;
it does not independently measure actual finger contact geometry.

### Single-cycle camera-to-grasp orchestration

For the operator-confirmed trial_04 iterative cube baseline and a short standalone
launcher, see [the Chinese lab run guide](iterative_cube_lab_zh.md).

`tools/run_vision_pick_cycle.py` connects the existing components in one supervised
cycle, without changing the analytical planner, its width-space waypoints, the
hand-angle mapping, or the original wrist-following sequence:

`open hand -> saved scan pose -> fresh RGB-D detection -> original Plan (approach + closure) -> settle -> vertical lift -> hold`

There is no extra hover/holding waypoint before the Plan. The original executor
moves directly from the scan pose to its own approach waypoint, then executes
the original wrist-following closure sequence. Final hold means no automatic
release after the requested lift, not another motion waypoint.

This is currently an experimental **known-size upright colored-cube** workflow,
not general object recognition or a verified unattended pick-and-place system.
The CLI defaults to an offline saved-image preview. It never connects the robot,
hand, or camera in that mode. Optional hardware libraries are imported only by
hardware paths. Each run creates a new artifact directory, `summary.json`,
`summary.csv`, and a plan; real runs additionally save RGB-D frames, detections,
raw serial TX/RX, stage checkpoints and arm telemetry. Existing outputs are not
overwritten.

The lab example is `configs/ur5_vision_pick_cycle.example.yaml`. Paths resolve
from the repository root. It keeps physical object width separate from the
empirical width offset and the total-gap closure margin:

`internal width = object width + width offset - closure margin`

The September 7 air measurements were approximately `60 -> 50 mm` and
`55 -> 45 mm` (requested internal width -> measured thumb/index inner gap).
The first margin-1 mm/force-100 attempt used internal width 49 mm; the operator
reported a loose partial grasp. At the operator's explicit request the example
now uses offset 10 mm and margin 2 mm for a 40 mm cube, giving internal width
48 mm, with force register 1000. The predicted 38 mm free opening is **outside
the measured range and not verified**, especially after changing force.
The margin is total gap reduction, not per finger; changing 1 to 2 mm requests
only 1 mm additional closure. The raw force register is not measured contact
force. Higher force may let a previously stalled angle target resume. Execution
still requires `--confirm-high-force`; speed does not imply force permission.

Trial 03 completed the full cycle and its operator confirmed a successful pickup,
while reporting index/middle contact before the thumb. The next explicitly
requested comparison uses `grasp_xy_offset_base_mm: [0, -5]`: translate every
original Plan pose 5 mm along UR-base -Y, horizontally toward the finger side
so the thumb starts closer to the cube. This is an experimental placement bias,
not an identified camera/mount calibration error. Hand commands, rotations,
heights and relative closure segments are unchanged. The measured target stays
unchanged in the report; the biased commanded grasp center is recorded separately.
Scan/reset poses are unchanged, and all shifted targets still undergo workspace
and controller preflight. The vector magnitude is capped at 10 mm; missing keys
default to `[0, 0]`, which also restores the successful no-bias baseline. This is
independent of the 2 mm width margin and raw force 1000. A previously checked
unshifted entry route does not certify the shifted route.

`tools/tighten_vision_pick_grip.py` handles only this particular interrupted
margin-1/force-100 to margin-2/force-1000 adjustment without reopening. It reuses
both original Plan endpoints and the frozen target, requires the arm to still
match the completed grasp, checks the small wrist segment with controller IK,
then sends force, wrist compensation and the narrower hand command. It defaults
to offline preview, has separate execution/high-force acknowledgements, logs
TX/RX and summaries, and never lifts, releases or retries. This is not a general
recovery command or confirmation that the object is held.

Offline preview, using a saved unobstructed capture (not the hand-occluded image):

```bash
RH56_CACHE_DIR=/tmp/rh56_plan_cache .venv312/bin/python tools/run_vision_pick_cycle.py \
  --config configs/ur5_vision_pick_cycle.example.yaml \
  --capture artifacts/vision_plan_trial_local_alignment/scene_20260907_173613/camera_check \
  --out artifacts/vision_pick_cycle/preview_new
```

Before the first live cycle, manually position the arm at a reviewed elevated
scan pose, outside the target ROI (the blue thumb brace must not be mistaken for
the blue cube). Read and save that pose without opening the hand or moving the arm:

```bash
.venv312/bin/python tools/run_vision_pick_cycle.py \
  --config configs/ur5_vision_pick_cycle.example.yaml \
  --record-start --out artifacts/vision_pick_start
```

Recording does not certify clearance. Review the example's entry height, TCP and
target bounds and all swept paths before executing. The example refuses to open
below TCP Z=300 mm; that is an explicit entry assumption, **not fingertip clearance**.
By default it does not retreat from a low or arbitrary starting position.
For a new cycle starting with a cube **already supported on the desk**, an
explicit `--reset-from-low --confirm-reset 'OBJECT SUPPORTED AND RESET PATH CLEAR'`
enables this entry sequence: preflight both entry segments, command the hand
open, raise vertically at fixed orientation/XY to the saved scan height (maximum
150 mm), then move to the saved scan pose. It requires operator confirmation of
object support and clear opening/retreat paths; pose alone does not prove either.
This optional entry reset is distinct from the final 50 mm grasp lift and does
not add a hover to the original Plan. Normal elevated entry has no reset waypoint.
The same reset flags must be supplied to controller-only preflight to check the
complete intended route. Do not use this reset to release a suspended object.
The user must verify the opening sweep, current-to-scan route, direct
scan-to-Plan-approach connection and lift route. There is no complete scene/finger
collision detector. Earlier real
desk contact contradicted the absolute model clearance, so simulation alone is
not a clearance guarantee.

The example uses `max_scan_to_approach_mm: 300` for the direct connection.
The saved iterative lab preset now explicitly uses `null`, at the operator's
request, to disable only this arbitrary scan-to-approach length cap. The sampled
controller safety/IK/continuity, workspace and rotation checks still cover the
whole segment; its distance remains logged with `limit_mm: null`. Missing keys
retain the legacy fallback. No intermediate waypoint or planner change is added.
Following the user's explicit approval, `max_reposition_mm` is also 300 mm so
the measured approximately 281 mm return from the interrupted trial to the saved
scan pose can run automatically. This changes only the lab's configurable
entry-to-scan distance cap; `max_plan_segment_mm` stays 50 mm, and entry height,
workspace, rotation, controller safety and IK continuity checks are unchanged.
It does not certify collision clearance or enable recovery from arbitrary poses.
Removing the extra hover changes the entry
path: the earlier scan-to-hover controller check does **not** validate this path.
Review it and rerun controller preflight before the next live cycle. Configs with
`hover_mm` or `max_scan_to_hover_mm` are rejected with migration instructions,
not silently reinterpreted. Without the new approach key, the reposition limit
is used. An arm stopped elsewhere after a failed run must still meet the configured
entry-to-scan limits; the script does not automatically relax them for recovery.

Controller-only preflight is available before moving anything. Supply a saved
unobstructed capture and replace the output directory for each check:

```bash
.venv312/bin/python tools/run_vision_pick_cycle.py \
  --config configs/ur5_vision_pick_cycle.example.yaml \
  --capture artifacts/vision_pick_cycle/start_camera_01 \
  --out artifacts/vision_pick_cycle/controller_check_new \
  --preflight-only --confirm-controller-script --acknowledge-shape-extrapolation
```

This explicitly starts/stops the RTDE control script to query real-controller IK,
but never connects the hand or calls opening, arm movement, grasp or lift methods.
The complete IK chain includes the current pose, saved scan pose, all original Plan
waypoints and lift. It is not fully read-only with respect to robot program state,
and cannot certify obstacle/finger clearance. A subsequent live cycle still
captures a new target and reruns its preflight.

After reviewing the setup, an explicitly authorized run is:

```bash
.venv312/bin/python tools/run_vision_pick_cycle.py \
  --config configs/ur5_vision_pick_cycle.example.yaml \
  --out artifacts/vision_pick_cycle/trial_new \
  --execute --confirm-cycle 'OPEN MOVE DETECT GRASP LIFT' \
  --acknowledge-open-loop --acknowledge-shape-extrapolation --confirm-high-force
```

One initial confirmation covers the whole cycle; there are no per-step prompts.
Execution checks current arm state, zero TCP offset, camera identity/resolution,
three fresh detections, target height, depth noise, target bounds and frame-to-frame
stability. It then checks real-controller IK continuity and pose limits for the
approach, complete Plan and lift. It stops subsequent stages on missing/ambiguous
detection, stale capture, invalid IK, abnormal robot state, timeout or serial
exception/short write. The legacy executor's swallowed finger errors cannot
authorize a lift. It does not retry or automatically release after failure.
An emergency stop remains the operator's responsibility; pending hand targets
cannot be cancelled or their physical completion confirmed through broken readback.

The existing strict final-shape alignment guard in `run_vision_plan_trial.py` is
unchanged. This new workflow validates the frozen candidate against its original
holdout and separately records/gates narrower-shape extrapolation. Saved camera
identity and target-height checks cannot detect every accidental XY camera/platform
shift. The setup must remain fixed or be revalidated.

Opening and pre-lift settling times are explicit configuration assumptions; they
do not replace hand feedback, and the internal Plan timing is unchanged.
Outer settling checks require a continuous stable interval, resetting that
interval on a speed excursion without extending the deadline. The timeout is
the requested interval plus `stationary_wait_grace_s` (default 3 s, an operational
timeout assumption, not a measured hand latency). The existing speed threshold
is unchanged. Abnormal robot state, invalid telemetry, cancellation or out-of-bounds
TCP fails immediately; continuous motion times out and aborts the cycle. Wait
diagnostics, including sampled TCP speeds/poses, go into `summary.json`.
Strict stationary checks at entry, image capture and before lift remain in place.

The final lift now uses the same fresh stationary snapshot to construct its
base-Z target and to measure its planned length. Previously a second noisy
origin could turn the nominal 50 mm into 50.035 mm and trip the exact length
cap. The cap and commanded lift distance are unchanged. Before and after IK,
live telemetry must remain within 0.1 mm / 0.1 degree of that snapshot (joint
change at most 0.01 rad), and the snapshot must be at most one second old when
preflight starts. These are operational drift tolerances, not physical accuracy
claims; recorded stationary TCP span was approximately 0.05 mm. Target/start
snapshots and segment lengths/limits are recorded, including failing segments.

The first supervised trial stopped at the now-removed hover when the old wait
rejected a single above-threshold speed sample; it never reached closure or lift.
Subsequent trials 03 and 04 completed the revised no-hover full cycle, including
the vertical lift, with operator-confirmed pickup. Trial 04 used base -Y 5 mm,
margin 2 mm and raw force 1000; its operator reported "Yes this is perfect!".
These are setup-specific observations, not repeated-trial success statistics
or collision certification. A live result is labeled
`cycle_commands_completed_grasp_unverified`, never automatic
grasp success. The final lift is fixed-orientation base +Z, with no finger command
or automatic release. Operator observations are saved separately from the
unchanged machine summaries; the write-only hand still cannot verify a grasp.

## Real Arm Control Panel

When `--real-robot` is passed, a new panel appears below the grasp sliders:

### Buttons

| Button | Description |
|:--|:--|
| **Teach Mode** | Toggles teach mode (turns red when active). Blocks all arm motion commands. Use to manually position the arm. |
| **Set Pose from Robot** | Reads the current TCP pose from the arm and updates the X/Y/Z and plane orientation sliders. Works in teach mode. |
| **Send to Robot (Arm)** | Executes a single `moveL` to the currently planned grasp pose (arm only; no finger motion). |
| **Simulate Trajectory** | Opens the MuJoCo robot viewer showing the planned target pose via mink IK. Real joint tracking is paused during preview. |
| **GRASP!** | Full coordinated grasp: moves arm to pose, then closes fingers using the selected strategy. |

### Grasp Strategy Radio

Select the finger closing strategy before pressing GRASP!:

| Strategy | Description |
|:--|:--|
| **Naive** | Arm moves to pose, then all fingers close to a fixed pinch posture (750/750/750/750/740/0). No geometry solver. |
| **Plan** | Chunked approach: arm and fingers advance proportionally in `Step (mm)` increments. Most geometry-aware. |
| **Thumb Reflex** | Arm moves to planned pose with only the thumb positioned first before waiting 0.2 s, after which remaining fingers close. |

> Naive strategy greyed out IK Method radio (not relevant for fixed posture).

### Force and Step Controls

| Control | Description |
|:--|:--|
| **Force (N)** | Target contact force in Newtons. Set to 0 to disable force phase. When > 0, activates `adaptive_force_control_iter` after the main trajectory. |
| **Step (mm)** | Chunk size for Plan strategy. Smaller = more intermediate arm/finger waypoints = smoother approach. |

### Cylinder Guard

When **Cylinder** mode is selected and the diameter is below **71 mm**, the GRASP! and Send-to-Robot buttons are automatically disabled. This prevents power-grasp attempts on objects too small to safely wrap.

### Status Area

The bottom strip of the panel shows the last 3 status messages from the executor thread (arm motion feedback, force readings, abort notifications, workspace warnings).

---

## Coordinate Frames

```
World frame: UR5 base, Z up, robot base at origin.
TCP frame:   zero-offset RTDE TCP at the UR tool flange.
Hand frame:  hand_base body at the hand-side mount reference.

_TCP_TO_HAND_POS  = [0, 0, 0.0564]                 (m, along TCP Z-axis)
_TCP_TO_HAND_QUAT = [0.7071068, 0, 0, 0.7071068]  (wxyz, Rz(90°))
```

The `56.4 mm` translation is the physical flange-to-hand fixture distance. It
must not be replaced with the `156.0 mm` wrist-3-to-hand distance from the
integrated UR5e MuJoCo model: that value includes the UR5e `d6 = 99.6 mm`
wrist-to-flange offset, which the zero-offset RTDE TCP has already applied.
The `UR5Bridge` class handles all TCP ↔ hand-frame conversions transparently.

---

## Workspace Limits

| Limit | Value |
|:--|:--|
| Max XY reach | 850 mm |
| Min XY radius (singularity) | 150 mm |
| Min Z | 0 mm (table surface) |

Poses outside these bounds generate warnings in the status area but are **not blocked** — the move still executes. Check workspace before committing to a grasp.

---

## Force Closure Visualization Panel

Click **Force Viz** in the grasp planner UI at any time (sim-only or real-robot mode) to open the contact analysis panel in a separate window.

### Bar chart
Shows calibrated per-finger contact forces in Newtons.
When a real hand is connected, real sensor readings are plotted alongside (or instead of) MuJoCo simulated forces.
The rightmost bar shows the **thumb yaw tangential force** — the component of contact force perpendicular to the thumb's abduction axis, estimated from the yaw motor's back-drive current.

### Grasp wrench space (GWS)
The 3D scatter plot shows the convex hull of all linearized friction-cone primitives.
The **Ferrari–Canny Q** metric measures the radius of the largest ball centered at the origin that fits inside the GWS — higher Q means the grasp can resist larger external wrenches.

### Heuristic force closure
Sensor-only force closure estimate (no simulation geometry required):
- **Lost** if thumb force < 0.3 N
- **Lost** if index force < 0.5 N **and** middle force < 0.3 N
- **Assumed** otherwise

### External wrench check
Enable **Live Wrench** to specify an applied force [Fx, Fy, Fz] in the world frame.
Optionally include gravity (object mass × 9.81 N downward).
The panel reports whether the current GWS can resist the wrench and by what margin (positive = safe).

### Sim geometry toggle
Check **Sim Geometry** to spawn a MuJoCo subprocess that auto-snaps a contact box to the current fingertip positions and computes contact normals.
Uncheck to use real sensor forces only with analytically derived contact normals — no subprocess needed.

---

## Workflow: Planning and Executing a Grasp

1. **Plan the grasp in simulation** (robot viewer + mink IK):
   - Adjust object width, mode (line / plane / cylinder), height (Z), and plane orientation sliders.
   - Click **Simulate Trajectory** to preview the arm pose in the MuJoCo viewer.

2. **Position the real arm**:
   - Enable **Teach Mode** → physically guide the arm to roughly the right position → disable Teach Mode.
   - Click **Set Pose from Robot** to sync the X/Y/Z sliders to the current arm pose.
   - Fine-tune sliders as needed.

3. **Send arm to planned pose**:
   - Click **Send to Robot (Arm)** (no fingers) to verify arm motion is safe.
   - The robot viewer will update to show real joint positions after the move.

4. **Execute full grasp**:
   - Set Force (N) and Step (mm) as desired.
   - Select strategy.
   - Click **GRASP!** — arm moves, fingers close.

5. **Abort**: close the matplotlib window or press Ctrl+C in the terminal. The arm decelerates and stops; the hand freezes.

---

## Real Joint Tracking in Viewer

After each arm move, the robot viewer automatically reflects the real UR5 joint angles (read via RTDE) instead of the mink IK solution. The transition:

- **Real tracking ON** (blue) — viewer shows real joint positions after arm moves.
- **Real tracking OFF** (planned) — viewer shows mink IK for the planned target. Activated by "Simulate Trajectory".

---

## Safety Notes

- The arm moves at `--ur5-speed` (default 0.10 m/s = 100 mm/s). Increase only when confident in the workspace.
- Always test with **Send to Robot (Arm)** before using GRASP! on a new pose.
- Use **Teach Mode** near obstacles; it fully disables all software motion commands.
- The cylinder guard (< 71 mm) prevents unintended power grasps; do not bypass it.
- The executor runs in a background thread; the UI remains responsive. You can monitor status messages without waiting for motion to complete.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|:--|:--|:--|
| `UR5Bridge: Connect failed` | IP wrong, ExternalControl not running | Verify IP; start ExternalControl URCap on pendant |
| `NOT CONNECTED — arm move skipped` | `connect()` failed silently | Check network; re-run with correct `--ur5-ip` |
| `TEACH MODE active — arm move blocked` | Teach mode left on | Click Teach Mode button to disable |
| Arm overshoots / unexpected motion | Speed too high | Lower `--ur5-speed` (0.05–0.08 m/s for precision) |
| `ur_rtde` import error | Not installed | `tools/setup_uv_env.sh --profile real-ur5 --python 3.12 --env .venv312` or `.venv312/bin/pip install ur-rtde` |
| Hand not responding | Wrong port or permissions | `ls /dev/ttyUSB*`, `sudo chmod 666 /dev/ttyUSB0` |
| Cylinder guard active unexpectedly | Mode is Cylinder, width < 71 mm | Increase width slider or switch to plane/line mode |
| Force phase never completes | Force threshold too high | Lower Force (N); check `force_act()` calibration |

---

## Hardware Test Checklist

See the plan file at `.claude/plans/humble-floating-anchor.md` for the full hardware verification protocol, or follow this condensed version on first bring-up:

- [ ] Transform round-trip: `T_hand → T_tcp → T_hand` matches to 1e-9 (no hardware needed).
- [ ] Workspace check: verify WARN messages at r=900mm, r=100mm, z=-10mm poses.
- [ ] UI smoke test (no arm): `uv run ... --robot` — new panel renders, teach mode button changes color, cylinder <71mm greys GRASP!.
- [ ] Connect arm: status shows "Connected to UR5".
- [ ] Teach mode: enable → move arm manually → disable → click "Set Pose from Robot" → sliders update.
- [ ] Send arm (no fingers): click "Send to Robot (Arm)" → arm moves to planned pose; robot viewer shows real joints.
- [ ] Naive grasp (force=0): GRASP! → arm moves, fingers close to pinch posture.
- [ ] Plan grasp (force=2N, step=10mm): chunked approach, force phase messages appear.
- [ ] Thumb Reflex grasp: thumb positions first, then remaining fingers close.
- [ ] Abort: click GRASP! → immediately close figure → arm decelerates.
- [ ] Real joint tracking: after arm move, robot viewer reflects real joint angles, not mink IK.
