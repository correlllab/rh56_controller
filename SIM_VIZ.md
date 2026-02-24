# SIM_VIZ — Real2Sim Grasp Force Visualizer: Planning Document

## Goal

Build a real-time visualizer that:
1. Runs the MuJoCo `inspire_scene.xml` simulation (upright hand grasping a hanging box).
2. Simultaneously reads the real RH56 Inspire hand via serial (force sensors + optional angle
   mirroring).
3. Computes grasp quality metrics (wrench cone, force closure, Ferrari-Canny) **for both** the
   simulation and the real hand.
4. Displays a side-by-side Matplotlib comparison to visualize the real2sim gap arising from the
   real hand's limited sensing (normal force only vs. full 3D wrench in sim).

---

## The Core Real2Sim Gap

| Property | Simulation | Real Hand |
|---|---|---|
| Contact positions | Known exactly (MuJoCo) | **Unknown** → borrowed from sim |
| Contact normals | Known (contact frame) | **Unknown** → borrowed from sim |
| Normal force | Exact (mj_contactForce) | `force_act()` in grams → N |
| Tangential (friction) force | Exact (x, y components) | **Not available** |
| Site force/torque sensor | Full 3D + torque | **Not available** |

**Consequence:** Both sim and real produce k primitive wrenches per contact.  But they differ in
what Fn represents:

- **Sim**: Fn is a free variable (normalized away) → primitives are unit vectors on the cone
  surface → the wrench cone is a full scalable cone → Q captures resistance to *any* wrench
- **Real**: Fn is *measured* (fixed) → primitives are actual force vectors on the friction
  disk boundary → the wrench set is a bounded disk → Q captures static equilibrium at the
  *specific* measured force level

Ferrari-Canny Q_sim ≥ Q_real.  The gap reflects the loss from not being able to freely
scale contact forces, not from missing the friction cone geometry entirely.

---

## Assumptions

- The real scene replicates `inspire_scene.xml`: same box size, same string, same hand orientation.
- Contact positions and normals in real ≈ those in sim (shared contact geometry assumption).
- `force_act()` returns the normal contact force at each fingertip (in grams), converted to N via
  `N = grams / 1000 * 9.81`.
- Thumb rotation (real index 5) has no corresponding fingertip contact and is ignored for contact
  analysis.

---

## Repository Layout

The `h1_mujoco/` submodule is read-only (not modified). All new code lives in the
`rh56_controller/` ROS package directory (alongside `rh56_hand.py`, `kinematics.py`):

```
rh56_controller/
├── mujoco_bridge.py      # SimAnalyzer: wraps MuJoCo scene, contact/wrench analysis
└── real2sim_viz.py       # Main script: real hand + sim + comparison visualization

SIM_VIZ.md               # This planning document
README_SIM_VIZ.md        # Concise usage document (created at end)
```

---

## Finger Indexing

### Real hand `force_act()` and `angle_set()` order
```
Index  Finger
  0    Pinky
  1    Ring
  2    Middle
  3    Index
  4    Thumb bend
  5    Thumb rotation   ← no fingertip contact; excluded from wrench analysis
```

### Sim actuator order (from `inspire_right.xml`)
```
ctrl[0]  pinky          range [0, 1.57 rad]
ctrl[1]  ring           range [0, 1.57 rad]
ctrl[2]  middle         range [0, 1.57 rad]
ctrl[3]  index          range [0, 1.57 rad]
ctrl[4]  thumb_proximal range [0, 0.60 rad]
ctrl[5]  thumb_yaw      range [0, 1.308 rad]
```

### Finger name → real force index
```python
FINGER_TO_REAL_IDX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3, "thumb": 4}
```

### Sim ctrl → real angle conversion (for `--mirror-angles`)
```python
SIM_CTRL_RANGES = [1.57, 1.57, 1.57, 1.57, 0.6, 1.308]
real_angle[i] = int(clip(ctrl[i] / SIM_CTRL_RANGES[i], 0, 1) * 1000)
```

---

## Data Flow

```
Real hardware (serial)             MuJoCo simulation
──────────────────────             ─────────────────
RealHandReader thread              main-thread sim loop
  force_act() → grams                mj_step()
  grams → N (÷1000 × 9.81)          detect_contacts()
  real_forces_N[5]                   get_sensor_forces_N()
  [locked shared state]              [direct computation]
          │                                  │
          └─────────── Comparator ───────────┘
                            │
                 per-finger: real_N[finger], sim_N[finger]
                 contacts:   position, normal (from sim)
                            │
              ┌─────────────┴─────────────┐
              │                           │
  compute_sim_wrench_cone()    compute_real_wrench_cone()
  full friction cone:          normal-only:
  k edges × n_contacts         1 vector × n_contacts
  (rich, many points)          (sparse, degenerate)
              │                           │
  evaluate_force_closure()    evaluate_force_closure()
  Q_sim, FC_sim               Q_real, FC_real
              │                           │
              └──── Visualization ────────┘
                    (matplotlib thread + mujoco viewer)
```

---

## Module: `mujoco_bridge.py`

### `ContactInfo` dataclass
```
finger_name   str
position      (3,) world frame contact point
frame         (3,3) rows: [normal, tangent1, tangent2]
normal_force  float  N  (from mj_contactForce)
friction_force (2,) N  tangential (sim only)
box_is_geom1  bool
```

### `SimAnalyzer` class

| Method | Description |
|---|---|
| `__init__(xml_path, friction_cone_edges)` | Load model, cache body/site/sensor IDs |
| `reset()` | Reset to "home" keyframe |
| `step()` | `mj_step()` |
| `get_contacts()` | Scan active contacts, return `List[ContactInfo]` |
| `get_sensor_forces_N()` | Site force sensor magnitudes, `Dict[str, float]` |
| `get_object_pos()` | Object body position |
| `get_ctrl_as_real_angles()` | Convert sim ctrl to real hand 0-1000 scale |
| `compute_sim_wrench_cone(contacts, obj_pos)` | Full k-edge linearized friction cone per contact → `(N,6)` |
| `compute_real_wrench_cone(contacts, real_forces_N, obj_pos)` | Normal-only: 1 primitive per contact scaled by real force → `(M,6)` |
| `evaluate_force_closure(wrenches)` | ConvexHull Ferrari-Canny, with 3D fallback → `(bool, float)` |

### Wrench cone computation details

**Sim** (full cone, same as `finger_force_viz.py`):
```
for each contact c:
  for j in 0..k-1:
    theta = 2π j / k
    f = normal + μ * (cos(θ)*t1 + sin(θ)*t2)
    f /= ||f||
    wrench = [f, r × f]   # r = pos - obj_pos
```

**Real** (friction disk — Fn known, tangential direction unknown):
```
for each finger in contact (dominant contact per finger):
  fn = real_forces_N[finger]   # from force_act(), N  (fixed, not a free variable)
  for j in 0..k-1:
    theta = 2π j / k
    f = fn * n + fn * μ * (cos θ * t1 + sin θ * t2)   # disk boundary, actual magnitudes
    wrench = [f, r × f]                                 # k primitives per contact
```

The feasible contact forces form a disk: `{Fn*n + Ft : |Ft| ≤ μ*Fn}`.  We know Fn exactly;
the tangential force can be anything within the friction limit.  We linearize the disk boundary
into k primitives — same count as sim, but using ACTUAL force magnitudes rather than normalized
unit vectors.

Real2sim gap: the sim normalizes primitives (Fn is a free variable → full scalable cone,
any wrench magnitude is achievable by scaling).  The real case fixes Fn → the achievable
wrench set is bounded, and Q_real ≤ Q_sim.

---

## Module: `real2sim_viz.py`

### Thread model

| Thread | Role |
|---|---|
| **Main** | MuJoCo sim stepping + `mujoco.viewer` (must be main thread) |
| **RealHandReader** | Poll `force_act()` at ~50 Hz; update shared state |
| **VizThread** | Matplotlib 4-panel update at ~10 Hz |

Shared state protected by `threading.Lock`.

### CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--port` | — | Serial port for real hand (e.g. `/dev/ttyUSB0`) |
| `--hand-id` | `1` | Hand ID on the serial bus |
| `--model` | auto | Path to `inspire_scene.xml` |
| `--mirror-angles` | off | Send sim ctrl angles to real hand each step |
| `--sim-only` | off | Disable real hardware (sim-vs-sim comparison) |
| `--no-viz` | off | Disable Matplotlib window |
| `--cone-edges` | `8` | Friction cone linearization edges |
| `--record` | — | Save metrics to `.npz` |

### Matplotlib: 4-panel layout

```
┌───────────────────────────┬───────────────────────────┐
│  Panel 1 (3D)             │  Panel 2 (3D)             │
│  Sim wrench cone          │  Real wrench cone         │
│  (full hull, k edges/     │  (normal-only, 1 vec/     │
│   contact, green/red)     │   contact, blue/red)      │
├───────────────────────────┼───────────────────────────┤
│  Panel 3 (bar)            │  Panel 4 (line)           │
│  Per-finger normal force  │  Ferrari-Canny over time  │
│  Real N (solid) vs        │  Q_sim (green)            │
│  Sim sensor N (hatched)   │  Q_real (blue)            │
│  Per-finger colors        │  FC threshold at 0        │
└───────────────────────────┴───────────────────────────┘
```

### MuJoCo viewer overlays

- Solid arrows at contact points: force ON finger from sim (colored by finger)
- Semi-transparent arrows at fingertip sites: sim site sensor force
- Sphere at object center: green (FC=True) or red (FC=False), radius ∝ Q_sim

### Keyboard controls (MuJoCo viewer)

| Key | Action |
|---|---|
| `P` | Pause/resume simulation |
| `R` | Reset to home keyframe |
| `S` | Print detailed status to console |
| `Q`/`Esc` | Quit |

---

## Force unit conversions

```python
# force_act() → Newtons
real_N = grams / 1000.0 * 9.81

# sim mj_contactForce result[0] is in Newtons (MuJoCo SI units)
sim_contact_N = result[0]

# sim site force sensor → Newtons (already SI)
sim_sensor_N = np.linalg.norm(site_xmat @ sensordata[adr:adr+3])
```

---

## Record format (`.npz`)

```
times           (T,)      simulation time
sim_fc          (T,)      bool, sim force closure
real_fc         (T,)      bool, real force closure
sim_q           (T,)      float, sim Ferrari-Canny
real_q          (T,)      float, real Ferrari-Canny
num_contacts    (T,)      int, number of active contacts
real_forces_N   (T, 5)    per-finger real normal forces [thumb,index,middle,ring,pinky]
sim_sensor_N    (T, 5)    per-finger sim sensor magnitudes
object_pos      (T, 3)    object position
```

---

## Dependencies

- `mujoco` (already used by `h1_mujoco/`)
- `numpy`
- `scipy` (ConvexHull for Ferrari-Canny; optional, degrades gracefully)
- `matplotlib` + `mpl_toolkits` (optional, `--no-viz` to skip)
- `pyserial` (already used by `rh56_controller/`)
- `threading` (stdlib)

---

## Implementation Order

1. `mujoco_bridge.py` — SimAnalyzer (pure MuJoCo, no hardware dependency)
2. `real2sim_viz.py` — main script (imports SimAnalyzer + RH56Hand)
3. `README_SIM_VIZ.md` — concise usage doc

---

## Open Questions / Future Work

- **Contact position accuracy**: the sim contact positions shift as the sim evolves. For a frozen
  comparison we could take a snapshot at peak grasp stability.
- **Angle mirroring latency**: serial writes at ~25–50 Hz. Add a `--mirror-rate` flag if needed.
- **Multi-contact per finger**: the real side currently takes the dominant (highest sim normal force)
  contact per finger. Could average or sum instead.
- **Thumb rotation**: ignored for wrench analysis (no fingertip contact site). Could be re-enabled
  if a contact site is added to the XML.
