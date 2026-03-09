# Simulation & Visualization — Inspire RH56

## Quick Reference

```bash
## the below four uv-run commands also work gracefully without uv installation, mink just won't be enabled
# Grasp planner (no hardware); mink IK comparison loaded by default
uv run python -m rh56_controller.grasp_viz

# Grasp planner + real hand ("Send to Real" checkbox)
uv run python -m rh56_controller.grasp_viz --port /dev/ttyUSB0

# grasp planner + real hand + real robot + ip
uv run python -m rh56_controller.grasp_viz --port /dev/ttyUSB0 --real-robot --ur5-ip 192.168.0.4

# Grasp planner + UR5 robot arm (enables Robot: Ours / Robot: Mink buttons)
uv run python -m rh56_controller.grasp_viz --robot

# Skip mink planner for faster startup (no mink comparison)
uv run python -m rh56_controller.grasp_viz --no-mink

# Force visualizer, sim-only
python -m rh56_controller.real2sim_viz --sim-only

# Force visualizer with real hand
python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0

# Force visualizer + angle mirroring + thumb tangential force
python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0 --mirror-angles --thumb-tangential

# Hand mirror, sim-only sine-wave demo
python -m rh56_controller.hand_mirror --sim-only

# Hand mirror, real hand → sim
python -m rh56_controller.hand_mirror --port /dev/ttyUSB0

# Hand mirror, sliders → real hand (with in-window toggle)
python -m rh56_controller.hand_mirror --port /dev/ttyUSB0 --sim-to-real

# Rebuild FK cache after XML changes
python -m rh56_controller.grasp_geometry --rebuild

# Recompute thumb yaw lever arm polynomial (updates _THUMB_YAW_REFF_POLY)
python tools/thumb_lever_arm.py --plot
```

---

## Scripts

| Script | One-liner |
|---|---|
| `rh56_controller/grasp_viz.py` | Interactive planner: pick grasp mode/width/height, visualize in FK model and MuJoCo viewer, optionally mirror to real hand or UR5 |
| `rh56_controller/grasp_viz_ui.py` | Tkinter UI layer for grasp_viz: sliders, buttons, real-robot panel, Force Viz launcher |
| `rh56_controller/grasp_viz_force_panel.py` | Force closure analysis panel: per-finger bar chart, GWS, heuristic FC, external wrench check |
| `rh56_controller/grasp_executor.py` | Executes grasp strategies (iterative, reflex, naive) on real UR5+hand; hybrid force control |
| `rh56_controller/real2sim_viz.py` | Real-time force analysis: reads real hand forces + runs sim, computes wrench cones and Ferrari-Canny grasp quality for both |
| `rh56_controller/hand_mirror.py` | Real-to-sim angle mirror with live bar chart; also sim-to-real slider mode for calibration |
| `rh56_controller/mujoco_bridge.py` | Sim backend (library): wraps `inspire_scene.xml`, computes contacts, wrenches, force closure |
| `rh56_controller/grasp_geometry.py` | FK backend (library): MuJoCo FK sweep + closure solver for line/plane/cylinder grasps |
| `rh56_controller/ur5_bridge.py` | UR5 bridge (library): RTDE connection, TCP ↔ hand-frame transforms, moveL wrapper |
| `tools/thumb_lever_arm.py` | Utility: MuJoCo FK sweep to compute `r_eff(θ)` polynomial for yaw tangential force estimation |

---

## grasp_viz.py — Interactive Grasp Planner

Solve antipodal grasp geometry for a target width/height, show results in a matplotlib 3D view, and push joint angles to a MuJoCo passive viewer and optionally to the real hand.

**Controls:**
- Radio buttons: `line` (2-finger), `plane` (3–5 finger), `cylinder` (power)
- Width slider: object width / cylinder diameter in mm
- Z slider: grasp height in world frame (mm)
- X / Y sliders: grasp position in UR5 world frame (robot mode only)
- **Hand: Ours** — open floating-hand viewer with custom planner angles
- **Hand: Mink** — open floating-hand viewer with mink IK planner angles
- **Robot: Ours** — open UR5+hand viewer, arm driven by mink IK, fingers from custom planner (robot mode only)
- **Robot: Mink** — open UR5+hand viewer, arm driven by mink IK, fingers from mink planner (robot mode only)
- Send to Real checkbox: mirror current joint solution to hardware at ~20 Hz

**Multiple viewers:** Each viewer runs in its own subprocess (via `multiprocessing.Process` with the `fork` start method).  This avoids GLFW multi-thread crashes so all four viewer windows can be open simultaneously.

**Mink comparison (matplotlib 3D view):** When mink is loaded, cyan diamond markers show the mink-planner tip positions alongside the orange/blue custom-planner tips.  Gold dashed lines connect mismatched pairs when the position error exceeds 2 mm.

**Viewer conventions:**
- Floating hand (`inspire_grasp_scene.xml`): direct `qpos` write + `mj_kinematics`; equality constraints manually replicated (polycoef coupling)
- Robot mode (`ur5_inspire.xml`): mink differential IK drives arm joints each frame; finger actuator values come from the selected planner

**CLI:**
```
--port /dev/ttyUSB0   connect real hand (enables Send-to-Real and Force Viz real mode)
--robot               enable UR5 robot viewer buttons and real-robot panel
--real-robot          connect to real UR5 arm (requires --robot)
--ur5-ip IP           UR5 robot IP address (default: 192.168.0.4)
--ur5-speed M/S       arm linear speed in m/s (default: 0.10)
--send-real           start with Send-to-Real enabled
--no-mink             skip loading mink planner (faster startup)
--xml PATH            override default inspire_right.xml
--rebuild             force FK table rebuild
```

**Width ranges:**

| Mode | Min (mm) | Max (mm) | Notes |
|---|---|---|---|
| 2-finger line | 8.7 | 122 | Side-approach (tilt=90°) below ~14.1 mm |
| 3/4/5-finger plane | 15.6 | 130 | ~5 mm Z-spread from differing finger lengths |
| cylinder | 28.6 | 103 | Palm mode (tilt=0°, yaw=0) below 71.6 mm |

See [grasp_viz.tex](grasp_viz.tex) for the full mathematical derivation of tilt, width, and closure geometry. See [GRASP_VIZ.md](GRASP_VIZ.md) for implementation notes and documented failed alternatives.

---

## real2sim_viz.py — Real2Sim Force Visualizer

Runs `inspire_scene.xml` (hand + hanging box) while simultaneously reading the real hand's force sensors, then computes and displays grasp quality metrics for both.

**Layout (4-panel Matplotlib):**
```
┌─────────────────────────┬─────────────────────────┐
│  Sim wrench cone (3D)   │  Real wrench cone (3D)  │
│  full friction cone     │  normal-force-only       │
├─────────────────────────┼─────────────────────────┤
│  Per-finger forces (bar)│  Ferrari-Canny over time│
│  real (solid) vs sim    │  Q_sim / Q_real          │
└─────────────────────────┴─────────────────────────┘
```

**Real2sim gap:** The sim normalizes contact primitives (free to scale) → full wrench cone. The real hand fixes Fn from `force_act()` → bounded wrench disk. Q_sim ≥ Q_real always; the gap reflects the loss from not freely scaling contact forces.

**Thumb tangential force** (experimental): enable with `--thumb-tangential` or the in-window checkbox. Uses yaw motor torque proxy + lever arm polynomial to estimate the tangential component perpendicular to the normal. See [THUMB_TANGENTIAL.md](THUMB_TANGENTIAL.md) for calibration details.

**CLI:**
```
--port /dev/ttyUSB0       real hand serial port
--sim-only                disable hardware (sim force sensors only)
--mirror-angles           send sim ctrl angles to real hand each step
--thumb-tangential        enable tangential force estimation at startup
--cone-edges N            friction cone linearization (default 8)
--no-viz                  headless mode (metrics only)
--record FILE.npz         save metrics time-series
```

**Keyboard (MuJoCo viewer):**
- `P` — pause/resume
- `R` — reset to home keyframe
- `S` — print status to console

---

## hand_mirror.py — Real-to-Sim Mirror

Maps real hand joint angles into MuJoCo in real time with a live grouped bar chart (commanded vs. actual read-back). Useful for calibrating joint limits and visualizing mapping errors.

**Modes:**
- **default** (`--port`): real hand drives sim; chart shows real target (blue) vs sim qpos (orange)
- **`--sim-only`**: sine-wave sweep of all joints, no hardware needed
- **`--sim-to-real`**: matplotlib sliders drive sim and real hand simultaneously; in-window toggle button switches between slider control and MuJoCo viewer control panel

**CLI:**
```
--port /dev/ttyUSB0   serial port for real hand
--sim-only            sine-wave demo, no hardware
--sim-to-real         sliders → real hand, with toggle
--xml PATH            override default inspire_scene.xml
```

---

## mujoco_bridge.py — Sim Backend

`SimAnalyzer` wraps `inspire_scene.xml` and provides the contact and wrench analysis used by `real2sim_viz.py`. Not intended to run standalone.

**Key methods:**

| Method | Returns |
|---|---|
| `get_contacts()` | `List[ContactInfo]` — finger name, world position, frame, forces |
| `get_sensor_forces_N()` | `Dict[str, float]` — per-finger site sensor magnitude (N) |
| `compute_sim_wrench_cone(contacts, obj_pos)` | `(N,6)` — full linearized friction cone |
| `compute_real_wrench_cone(contacts, real_forces_N, obj_pos)` | `(M,6)` — normal-force-scaled primitives |
| `compute_real_wrench_cone_with_tangential(...)` | `(M,6)` — as above + thumb yaw tangential |
| `evaluate_force_closure(wrenches)` | `(bool, float)` — FC flag + Ferrari-Canny Q |

---

## grasp_geometry.py — FK Backend

`InspireHandFK` runs a MuJoCo FK sweep at startup and fits scipy interpolators (cached to `.fk_cache.npz`). `ClosureGeometry` uses these to solve antipodal grasps.

**Coupling:** manually replicates the XML equality constraints in the FK sweep (see **Joint coupling** table in the XML Model Files section).

**Cache invalidation:** the cache stores an mtime fingerprint. Delete `.fk_cache.npz` or pass `--rebuild` after changing the XML or coupling equations.

---

## XML Model Files

All XML files live under `h1_mujoco/inspire/`. The hand joint geometry, coupling polycoefs, and actuator limits are derived from the community URDF by [Omkar Kshirsagar](https://github.com/ookkshirsagar/rh56dfx_description).

| File | Used by | Description |
|---|---|---|
| `inspire_right.xml` | real2sim_viz, hand_mirror, grasp_viz_force_panel | Fixed-base right hand, 6 DOF. Source-of-truth for real2sim joint limits. |
| `inspire_right_ur5.xml` | ur5_inspire.xml (included) | Hand body definitions with `gravcomp="1"` on all 13 bodies; included by the robot scene. |
| `inspire_scene.xml` | real2sim_viz, hand_mirror | `inspire_right.xml` + a suspended box for contact experiments. |
| `inspire_grasp_scene.xml` | grasp_viz, grasp_geometry | Floating hand (6-DOF free joint base), calibrated fingertip sites. FK source-of-truth for the analytical planner. |
| `inspire_force_scene.xml` | grasp_viz_force_panel | Fixed-base hand + fixed `object` body (no joint, `gravcomp="1"`) whose pose is set at runtime to auto-snap to the current fingertip centroid. |
| `ur5_inspire.xml` | grasp_viz --robot | UR5e arm + Inspire hand; mink drives arm joints each frame. |

### Joint coupling (equality/mimic constraints, adapted for own damaged hands...)

The RH56's underactuated linkages are modelled as `<equality>` polycoef constraints. The same coefficients must be replicated manually wherever `mj_kinematics` is called directly (i.e. whenever the full constraint solver is bypassed — see note below).

| Joint | Driving joint | Equation |
|---|---|---|
| pinky / ring / middle intermediate | proximal | `q_inter = −0.15 + 1.1169 · q_prox` |
| index intermediate | proximal | `q_inter = −0.05 + 1.1169 · q_prox` |
| thumb intermediate | pitch | `q_inter = 0.15 + 1.33 · q_pitch` |
| thumb distal | pitch | `q_dist = 0.15 + 0.66 · q_pitch` |

> **Why manual replication is needed:** `mj_kinematics` computes forward kinematics from `qpos` but does **not** enforce `<equality>` constraints — those are velocity-level constraints solved only during `mj_step`. Whenever `qpos` is written directly (FK sweep in `grasp_geometry.py`, passive viewer in `grasp_viz.py`), all coupled joints must be set manually using the above equations.

If you update these coefficients, you must update **all four** locations:
1. `inspire_grasp_scene.xml` (source of truth)
2. `inspire_right.xml`
3. `grasp_geometry.py` — `_set_finger_qpos` / `_set_thumb_qpos`
4. `grasp_viz.py` — `_apply_qpos`

Then delete `.fk_cache.npz` so the FK cache rebuilds.

### Actuator ctrl ranges (adapted for our own damaged hand...)

| Actuator | ctrl_min (rad) | ctrl_max (rad) | Notes |
|---|---|---|---|
| pinky | 0 | 1.57 | |
| ring | 0 | 1.57 | |
| middle | 0 | 1.50 | |
| index | 0 | 1.50 | |
| thumb_proximal (bend) | 0.1 | 0.57 | `ctrl_min=0.1` = fully open (real=1000) |
| thumb_yaw | 0 | 1.308 | `ctrl=0` = spread out; `ctrl=1.308` = adducted |

---

## DOF / Index Reference

**Real hand `angle_read()` / `force_act()` / `angle_set()` order:**

| Index | DOF |
|---|---|
| 0 | pinky |
| 1 | ring |
| 2 | middle |
| 3 | index |
| 4 | thumb bend (pitch) |
| 5 | thumb yaw |

**Sign convention (INVERTED):**
```
real = 1000 − round((ctrl − ctrl_min) / (ctrl_max − ctrl_min) × 1000)
  real = 0    → closed / adducted  (ctrl = ctrl_max)
  real = 1000 → open  / abducted   (ctrl = ctrl_min)
```

See the **Actuator ctrl ranges** table in the XML Model Files section above.

---

## Further Reading

- [GRASP_VIZ.md](GRASP_VIZ.md) — Grasp geometry implementation notes: closure solver, tilt derivation, and documented failed alternatives
- [THUMB_TANGENTIAL.md](THUMB_TANGENTIAL.md) — Thumb yaw tangential force estimation: calibration procedure, lever arm polynomial, sign conventions
