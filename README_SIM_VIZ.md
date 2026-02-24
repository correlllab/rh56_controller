# Real2Sim Grasp Force Visualizer

Compares the MuJoCo `inspire_scene.xml` simulation against the real RH56
Inspire hand in real time, visualizing the **real2sim gap** that arises from
having only normal force readings on the hardware versus full 3D contact
wrenches in simulation.

## Files

| File | Role |
|---|---|
| `rh56_controller/mujoco_bridge.py` | `SimAnalyzer` — MuJoCo wrapper, wrench/closure math |
| `rh56_controller/real2sim_viz.py`  | Main script — real hand + sim + visualization |

## Dependencies

```
mujoco          # simulation
numpy
scipy           # ConvexHull for Ferrari-Canny (optional, degrades gracefully)
matplotlib      # visualization (optional, --no-viz to skip)
pyserial        # already in rh56_controller
```

## Quick Start

### Simulation only (no hardware required)

```bash
python -m rh56_controller.real2sim_viz --sim-only
```

### With real hand (read only)

```bash
python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0
```

### Mirror sim angles to real hand

```bash
python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0 --mirror-angles
```

### Record session to file

```bash
python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0 --record session.npz
```

### Replay saved session

```bash
python -m rh56_controller.real2sim_viz --replay session.npz
```

## All Options

```
--port          Serial device (e.g. /dev/ttyUSB0)
--hand-id       Hand ID on bus (default: 1)
--model         Path to inspire_scene.xml (auto-detected from repo layout)
--cone-edges    Friction cone edges for sim linearization (default: 8)
--mirror-angles Send sim joint angles to real hand each step
--sim-only      Run without hardware; use sim sensor forces as "real" proxy
--no-viz        Disable Matplotlib window (MuJoCo viewer still shown)
--record FILE   Save metrics to .npz
--replay FILE   Replay a saved .npz recording
```

## Keyboard Controls

| Key | Action |
|---|---|
| `P` | Pause / resume simulation |
| `R` | Reset to home keyframe |
| `S` | Print per-finger force table to console |
| `Q` / `Esc` | Quit |

## Visualization Panels

```
┌─────────────────────────────┬─────────────────────────────┐
│ Sim Wrench Cone (3D)        │ Real Wrench Cone (3D)       │
│ Full friction cone hull     │ Normal-only vectors         │
│ Green=FC  Red=no FC         │ Blue=FC   Red=no FC         │
├─────────────────────────────┼─────────────────────────────┤
│ Per-Finger Normal Force     │ Ferrari-Canny Q over time   │
│ Solid = real (N)            │ Green = Sim (full cone)     │
│ Hatched = sim sensor (N)    │ Blue  = Real (normal only)  │
└─────────────────────────────┴─────────────────────────────┘
```

**MuJoCo viewer:** solid arrows = contact forces on fingers; semi-transparent
arrows = site sensor forces; sphere at object = force-closure status (green/red).

## The Real2Sim Gap

| Property | Simulation | Real Hand |
|---|---|---|
| Contact positions | Exact (MuJoCo) | Borrowed from sim |
| Contact normals | Exact | Borrowed from sim |
| Normal force | Exact | `force_act()` → N |
| Tangential force | Exact (x, y) | **Not available** |

Both sim and real produce **k primitive wrenches per contact**, but they differ
in what the normal force `Fn` represents:

- **Sim**: `Fn` is a free variable (normalized out) → primitives are unit
  vectors on the cone surface → wrench cone is a full scalable cone → `Q`
  captures resistance to *any* magnitude of external wrench.
- **Real**: `Fn` is the *measured* value (fixed) → primitives are actual force
  vectors on the friction-disk boundary `{Fn·n + Ft : |Ft| ≤ μ·Fn}` →
  wrench set is bounded → `Q` captures static equilibrium at the *specific*
  measured force level.

`Q_sim ≥ Q_real`.  The gap reflects the inability to freely scale contact
forces (Fn is constrained by what the sensor reads), not an absence of the
friction cone.

## Finger Ordering

`force_act()` returns `[pinky, ring, middle, index, thumb_bend, thumb_rot]`.
The thumb rotation channel (index 5) has no fingertip contact and is excluded
from wrench analysis.

## Saved `.npz` Schema

```
times           (T,)     simulation time (s)
fc_sim          (T,)     bool, sim force closure
fc_real         (T,)     bool, real force closure
q_sim           (T,)     float, sim Ferrari-Canny metric
q_real          (T,)     float, real Ferrari-Canny metric
num_contacts    (T,)     int, active contacts
real_forces_N   (T, 5)   per-finger real normal forces (N) [thumb,index,middle,ring,pinky]
sim_sensor_N    (T, 5)   per-finger sim sensor magnitudes (N)
object_pos      (T, 3)   object (box) world position
```
