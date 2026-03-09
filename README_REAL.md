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

### 2. Install the uv environment with real-robot extras

```bash
uv sync --extra real-robot
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

---

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
Hand frame:  hand_base body (mount point on wrist-3 flange).

_WRIST3_TO_HAND_POS  = [0, 0, 0.156]          (m, along UR5 TCP Z-axis)
_WRIST3_TO_HAND_QUAT = [0.7071068, 0, 0, 0.7071068]  (wxyz, Rz(90°))
```

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
| `ur_rtde` import error | Not installed | `uv sync --extra real-robot` or `.venv/bin/pip install ur-rtde` |
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
