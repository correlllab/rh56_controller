# Force Control UI — Usage & Concepts

`force_control_ui.py` provides a real-time debug and command interface for the
`magpie_force_control` admittance controller.  It launches the C++ binary as a
subprocess, streams telemetry at ~50 Hz, and lets you change wrench targets,
compliance axes, stiffness/damping, and pose references without restarting.

---

## Quick start

```bash
# Standalone
uv run python -m rh56_controller.force_control_ui

# Or pop it up from inside grasp_viz (--robot mode):
#   "Force Control" button in the right panel
```

The binary path defaults to
`magpie_force_control/build/force_control_demo`; override via the
**Binary** field or `--binary` CLI flag.

---

## UI layout

```
┌─────────────────────────┬────────────────────────────┐
│  Joint Temperatures      │  Right panel (scrollable)  │
│  (or EEF 3D plot)        │                            │
├─────────────────────────│  • Connection              │
│  Time series plots       │  • Command Frames          │
│  (2-plot or 6-plot)      │  • Wrench target           │
│                          │  • Compliance axes         │
│                          │  • Pose reference          │
│                          │  • Run parameters          │
│                          │  • Advanced (stiff/damp)   │
│                          │  • Logging                 │
│                          │  • Wrench monitor          │
└─────────────────────────┴────────────────────────────┘
```

Toggle **Joint Temps** (checkbox) to replace the top-left area with a live
6-joint temperature bar chart.  Toggle **6-axis plots** to expand the
time-series section from 2 plots (force + torque magnitude) to all 6 axes.

---

## Command Frames

The controller binary works in **mixed frames** (the Hou & Mason default):

| Input type | Natural frame |
|---|---|
| Wrench reference `wd` | **EEF (tool) frame** — sent directly to the controller |
| Position reference | **WORLD frame** — sent as world-frame coordinates |

This is intentional: the tool Z-axis typically points into the contact surface,
so `Fz = -5 N` means "push 5 N into the surface" regardless of arm pose.
World-frame position is convenient for specifying task-space approach directions.

The UI lets you switch each independently:

- **Wrench: EEF** (green, recommended) — enter forces/torques in tool frame.
- **Wrench: WORLD** (orange) — Python rotates `R_eef.T @ w` before sending.
- **Position: WORLD** (blue, recommended) — enter XYZ in robot world frame.
- **Position: EEF** (purple) — Python rotates `R_eef @ delta` before sending.

> **Warning banner** — WORLD wrench + EEF position is a non-standard combination
> that is rarely what you want.  A red warning is shown if you select it.

---

## Wrench target

Enter force/torque components (`Fx Fy Fz Tx Ty Tz`).  Click **Send Wrench**.

Safety limits (editable in Advanced):

| Component | Default limit |
|---|---|
| Force (each axis) | ±20 N |
| Torque (each axis) | ±3 N·m |

Values outside the limits are automatically clamped and the UI entries are
updated to reflect the actual value sent.

---

## Compliance axes

Six checkboxes select which axes the controller regulates with force feedback.
Unchecked axes are position-controlled (spring + damper drive the EEF to the
pose reference, ignoring force error on that axis).

Click **Send Axis** to push the new mask.  The C++ binary rebuilds the
`Tr` permutation matrix so force-controlled axes occupy the first `n_af`
columns — see `buildTr()` in `main.cc`.

---

## Pose reference

Two modes:

- **DELTA** — move the reference by `[dx dy dz]` metres plus a rotation
  `[rx ry rz]` expressed as **Euler XYZ degrees**.  The C++ binary
  integrates this onto the current reference pose.
- **ABSOLUTE** — set the reference pose directly in world frame.

Rotation inputs are clipped to ±360°.  Rotations > 45° trigger a status
warning — proceed carefully; large rotations can cause joint-limit violations or
singularities.

---

## Run parameters

| Field | Meaning |
|---|---|
| Robot IP | UR5 IP address (passed to binary) |
| Duration (s) | How long to run; `0` = run until Stop |
| Rate (Hz) | Control loop rate; 500 Hz is typical |

Changing the rate automatically corrects the `dt` written to the temporary
config file.

---

## Advanced: stiffness and damping

Reveal with the **Advanced** checkbox.  Changes are sent live via `STIFF` /
`DAMP` stdin commands.

Default values (from `config.yaml`):

| | Trans (x/y/z) | Rot (rx/ry/rz) |
|---|---|---|
| Stiffness | 100 | 1 |
| Damping | 2 | 0.2 |
| Inertia | 5 | 0.005 |

---

## Logging

Enable CSV logging with the **Log to CSV** checkbox.  Columns:

```
epoch_s, Fx_s, Fy_s, Fz_s, Tx_s, Ty_s, Tz_s,   # sensed wrench
         Fx_d, Fy_d, Fz_d, Tx_d, Ty_d, Tz_d,   # desired wrench
         tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz,
         ref_x, ref_y, ref_z, ref_rx, ref_ry, ref_rz,
         q0..q5, T0..T5
```

Default log directory: `rh56_controller/logs/`.

---

## Stdin command protocol (reference)

The Python UI speaks to the binary over stdin:

```
WRENCH  Fx Fy Fz Tx Ty Tz          — desired wrench, EEF frame (N / N·m)
AXIS    a0 a1 a2 a3 a4 a5          — compliance mask (1=force-ctrl, 0=pos-ctrl)
RELD    dx dy dz drx dry drz       — relative pose delta (m, axis-angle rad)
REFA    x  y  z  rx  ry  rz        — absolute pose reference (world frame)
STIFF   k0 k1 k2 k3 k4 k5         — diagonal stiffness override
DAMP    d0 d1 d2 d3 d4 d5         — diagonal damping override
STOP                               — graceful shutdown
```

All values are space-separated floats on a single line.

---

## Telemetry protocol (reference)

The binary emits one `TELEM:` line every 10 control steps (~50 Hz at 500 Hz):

```
TELEM: t=<s> tcp=x,y,z,rx,ry,rz ref=... ws=... wd=... q=0,1,2,3,4,5 T=0,1,2,3,4,5
```

Parsed into a float32[37] array: `[t, tcp×6, ref×6, ws×6, wd×6, q×6, T×6]`.

---

## Building the C++ binary

```bash
cd magpie_force_control
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# binary: magpie_force_control/build/force_control_demo
```

---

## Admittance control fundamentals

### Motivation

A purely position-controlled robot is stiff — it will exert arbitrarily large
forces to reach a commanded pose.  For contact-rich tasks (insertion, assembly,
polishing) you need the robot to **yield** to contact forces while still
pursuing a task-space goal.

**Admittance control** achieves this by treating the robot's tool frame as a
virtual mass-spring-damper:

```
M * ẍ  +  D * ẋ  +  K * x  =  -f_error
```

where:

| Symbol | Meaning |
|---|---|
| `M` | virtual inertia (6×6 diagonal) |
| `D` | virtual damping (6×6 diagonal) |
| `K` | virtual stiffness (6×6 diagonal) |
| `x` | displacement from current reference pose |
| `f_error = f_sensed - f_desired` | contact force error |

A large `K` makes the robot stiff (position-dominant).  Small `K` + moderate `D`
makes it compliant — the EEF moves to reduce force error, acting like a damped
spring.

### Hybrid force–velocity control (Hou & Mason 2019)

Pure admittance control applies compliance uniformly to all 6 DOF.  In contact
tasks you often want **rigid position control along some axes** and **force
control along others**.

The Hou & Mason approach partitions the task space:

```
Tr = [e_f1 | e_f2 | … | e_fn_af | e_p1 | e_p2 | … ]
       ← force-controlled axes →  ← position-controlled axes →
```

`Tr` is an orthonormal 6×6 matrix.  `n_af` columns correspond to the
directions in which the controller tracks a force target; the remaining
`6 - n_af` columns are position-controlled.

> **Example — peg insertion:** set the insertion axis (EEF Z) as force-controlled
> (`Fz = -5 N`), all other axes as position-controlled.  The robot pushes with
> constant force along Z while holding a precise lateral position.

The compliance equation only sees the force-controlled component of the
wrench error:

```
f_error_fc = Tr[:, :n_af].T @ (f_sensed - f_desired)
```

The position-controlled component is driven by the stiffness spring alone.

### Integration scheme

Each control step (`dt = 0.002 s` at 500 Hz):

1. Read sensed wrench `f_s` (EEF frame) and current TCP pose.
2. Compute wrench error: `f_err = f_s - f_desired` (both in EEF frame).
3. Project onto force-controlled axes via `Tr`.
4. Integrate the compliance ODE to get a pose correction `Δx`.
5. Add `Δx` to the current reference pose → send as velocity command to robot.

### Frame convention

The binary follows the Hou & Mason convention:

- **`wrench_WTr`** (desired wrench) — expressed in the **tool (EEF) frame**.
  `W` in the variable name refers to the world-frame reference for position;
  the subscript `T` means tool frame for the wrench.
- **`pose_ref`** — expressed in the **world frame**.

This mixing is intentional.  The tool frame is natural for describing contact
forces ("push along the tool Z-axis") while world-frame positions are natural
for specifying approach targets.

Conversion (Python, if you supply world-frame wrenches):

```python
R_eef = aa_to_R(tcp[3:6])        # 3×3 rotation matrix, world←EEF
f_eef = R_eef.T @ f_world         # rotate world wrench into EEF frame
```

### Safety guidelines

- **Start with `Fz = 0`** and verify the robot holds position before applying
  any non-zero force target.
- **Increase force targets gradually** (1–2 N steps); watch the sensed wrench
  stream for oscillations.
- **High stiffness + high inertia** can cause instability.  If the robot
  oscillates, reduce stiffness or increase damping.
- **Max spring force** (`max_spring_force_magnitude` in config) caps the
  stiffness term — a safety net against large position offsets at high `K`.
- The UI's safety clamps (default ±20 N / ±3 N·m) are a first line of defence;
  the C++ spring-force cap is a second.
- **Temperature monitoring**: joint temps are shown in the top-left area.
  Stop immediately if any joint exceeds ~60 °C.

---

## Typical peg-in-hole workflow

```
1. Run grasp_viz --robot, plan grasp, approach hole entry point.
2. Open Force Control UI (button in grasp_viz, or standalone).
3. Set compliance: only Z-axis force-controlled (checkbox row: 0 0 1 0 0 0).
4. Set wrench: Fz = -3 N (gentle contact).
5. Start controller.  Observe sensed wrench stream and time-series plots.
6. Increase |Fz| in 1–2 N increments while monitoring lateral force error.
7. Use Delta pose (XY in world frame) to correct lateral alignment.
8. Once fully inserted, set Fz = 0 and all axes position-controlled.
9. Stop controller.
```
