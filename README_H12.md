# H1-2 Grasp Visualizer — Architecture & Operations Guide

This document explains every component involved in the H1-2 grasp pipeline, how they
connect, what runs where, and what you do and don't need for each mode.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        grasp_viz  (this repo)                            │
│                                                                          │
│  grasp_viz.py ──► GraspVizUI (Tkinter)                                  │
│                       │                                                  │
│                       ├── GraspVizCore (state, geometry)                 │
│                       │       ├── InspireHandFK  (brentq closure)        │
│                       │       ├── ClosureGeometry (brentq closure)       │
│                       │       ├── H12Bridge (ROS2 client, optional)      │
│                       │       └── MinkGraspPlanner (comparison, optional)│
│                       │                                                  │
│                       └── MuJoCo viewer subprocesses (fork)             │
│                               ├── _hand_viewer_worker                   │
│                               ├── _robot_viewer_worker  (UR5 + mink)    │
│                               ├── _h12_robot_viewer_worker (PINK IK)    │
│                               └── _h12_bimanual_viewer_worker (PINK IK) │
└─────────────────────────────────────────────────────────────────────────┘
                              │ ROS2 actions (optional)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    h12_ros2_controller (separate repo)                   │
│                                                                          │
│  frame_task_server.py   — FrameTask action server  (/frame_task)        │
│  dual_arm_server.py     — DualArm action server    (/dual_arm)          │
│  hand_controller_node.py — finger command node                          │
│  joint_state_publisher.py — publishes /joint_states                     │
│                                                                          │
│  core/controller/                                                        │
│    arm_controller.py    — Pinocchio + PINK IK + Unitree SDK              │
│    frame_controller.py  — multi-frame IK tasks                          │
│    hand_controller.py   — finger position commands                      │
│    gravity_comp_controller.py — gravity compensation hold               │
│                                                                          │
│  core/channel_interface.py — Unitree SDK2 low-level motor publisher     │
└─────────────────────────────────────────────────────────────────────────┘
                              │ Unitree SDK2 (DDS/cyclone-dds)
                              ▼
                      Real H1-2 robot
```

---

## Mode Summary

| CLI flags | What runs | ROS2 needed? | Real robot? |
|-----------|-----------|-------------|-------------|
| _(none)_  | Floating hand viewer | No | No |
| `--h12`   | H1-2 MuJoCo sim, PINK IK in-process | No | No |
| `--h12 --bimanual` | Bimanual H1-2 MuJoCo sim, PINK IK for both arms | No | No |
| `--h12 --h12-ros` | H1-2 sim + ROS2 comms bridge active | Yes (ROS2 env) | No |
| `--h12 --real-h12` | Full real robot: ROS2 + H12Bridge + frame_task_server | Yes + h12_ros2_ctrl | **YES** |
| `--robot` | UR5 MuJoCo sim, mink IK | No | No |
| `--real-robot --ur5-ip IP` | UR5 real arm | No (RTDE) | YES |

**Mutual exclusion**: `--real-robot` and `--real-h12` cannot be used together.

---

## Mode 1: `--h12` (Sim-only, no ROS)

```bash
uv run python -m rh56_controller.grasp_viz --h12
uv run python -m rh56_controller.grasp_viz --h12 --bimanual
```

### What happens

1. `GraspVizCore` loads the FK model from `inspire_grasp_scene.xml` and builds the
   brentq closure geometry.  **All finger configuration is solved offline via this
   model — not mink.**
2. You click **"H1-2: Ours"** → launches `_h12_robot_viewer_worker` in a subprocess.
3. The worker loads `h1_2_pos_inspire.xml` (MuJoCo) and the H1-2 URDF from
   `h12_ros2_controller/assets/h1_2/h1_2.urdf` (Pinocchio).
4. Each frame: reads grasp target from shared memory → converts hand-base pose to
   `right_wrist_yaw_link` frame → runs 40 PINK IK iterations → writes qpos to MuJoCo.
5. **No ROS processes, no Unitree SDK, no real hardware.**

### What is NOT happening

- No `frame_task_server`, no ROS nodes, no DDS traffic.
- No joint state publishing to `/joint_states`.
- The `H12Bridge` is not instantiated (it would fail gracefully if tried).

### What IK method is used?

| Part | Method |
|------|--------|
| **Finger configuration** | Brentq geometric closure (`ClosureGeometry`) |
| **Arm (wrist_yaw → grasp target)** | PINK differential IK (Pinocchio), 40 iters/frame |
| **Arm (mink comparison)** | Disabled (no mink comparison for H1-2 currently) |

The width slider range you see is from the **geometric FK sweep** of `inspire_grasp_scene.xml`,
not from mink.  The brentq solver finds the finger config online each frame (< 1 ms).

### File locations

| File | Purpose |
|------|---------|
| `h1_mujoco/inspire/h1_2_pos_inspire.xml` | H1-2 + right hand MuJoCo scene |
| `h1_mujoco/inspire/h1_2_bimanual_inspire.xml` | H1-2 + both hands (bimanual) |
| `h1_mujoco/inspire/inspire_left_h12_qnames.xml` | Left hand body stub (included by bimanual XML) |
| `h12_ros2_controller/assets/h1_2/h1_2.urdf` | Pinocchio URDF for PINK IK |

---

## Mode 2: `--h12 --h12-ros` (Sim with ROS2 bridge)

```bash
# Terminal 1: source ROS2 workspace
source ~/ros2_ws/install/setup.bash
# Terminal 2: run grasp_viz with ROS bridge
uv run python -m rh56_controller.grasp_viz --h12 --h12-ros
```

### What the ROS bridge adds

The `--h12-ros` flag activates `H12Bridge.connect()`, which starts a minimal ROS2
node (`grasp_viz_h12_bridge`) in a background thread.

With this flag:
- **Send H1-2** button becomes active (sends `FrameTask` action to `/frame_task`).
- The robot body reacts to commands just as it would in real life — but through the
  sim controller, not real hardware.

You still need a `frame_task_server` process listening on `/frame_task`:

```bash
# In a sourced ROS2 terminal, from h12_ros2_controller:
python h12_ros2_controller/ros2/frame_task_server.py --debug
```

`--debug` runs the controller in simulation-integration mode (no Unitree SDK, integrates
directly into the Pinocchio model).  This is the correct mode for sim-only ROS testing.

### ROS2 topics & actions produced by frame_task_server

| Name | Type | Description |
|------|------|-------------|
| `/frame_task` | `custom_ros_messages/action/FrameTask` | Move arm to target frame pose |
| `/named_config` | `custom_ros_messages/action/NamedConfig` | Go to named config (e.g. "home") |
| `/frame_names` | `custom_ros_messages/msg/StringArray` | Currently tracked frame names |
| `/frame_targets` | `geometry_msgs/PoseArray` | Target poses for those frames |
| `/frame_poses` | `geometry_msgs/PoseArray` | Actual current poses for those frames |

### What frame_task_server does with the action

1. Receives `FrameTask.Goal` with `frame_names=["right_wrist_yaw_link"]` and `frame_targets=[Pose]`.
2. Calls `FrameController.add_frame_task(task_name, frame_name, T_target)`.
3. Runs PINK IK loop until the frame reaches the target (within threshold) or timeout.
4. In `--debug` mode: calls `control_step_reduced()` → integrates velocity into Pinocchio
   model state (no hardware output).
5. In `--sport` mode: calls `control_step_reduced()` → sends low-level motor commands
   via Unitree SDK2 to the real robot.

---

## Mode 3: `--h12 --real-h12` (Real robot)

```bash
# Terminal 1 — h12_ros2_controller (real robot mode, sourced workspace):
python h12_ros2_controller/ros2/frame_task_server.py --sport

# Terminal 2 — grasp_viz:
uv run python -m rh56_controller.grasp_viz --h12 --real-h12
```

### Full launch sequence

1. **Power on H1-2**, put it in damping/stand mode via the joystick as per Unitree docs.
2. On the robot's onboard computer (or a machine on the same network), start the
   `ChannelFactory` (DDS discovery).  `frame_task_server.py` calls
   `ChannelFactoryInitialize()` which handles this automatically.
3. Start `frame_task_server.py --sport` — this connects via Unitree SDK2 and begins
   reading motor states.
4. Start `grasp_viz --h12 --real-h12`.  `H12Bridge.connect()` is called during init
   and waits up to 5 s for the `/frame_task` action server to become available.
5. Use the sliders to position the grasp target.  Click **Sim H1-2** to preview the
   trajectory in the MuJoCo viewer (PINK IK, no hardware).
6. Click **Send H1-2** to execute: grasp_viz sends a `FrameTask` goal → `frame_task_server`
   runs PINK IK and sends motor commands via Unitree SDK2.
7. When arm is at the target pose, click **GRASP!** — arm moves then fingers close to
   the target width.

### For bimanual real robot

```bash
# Use dual_arm_server instead:
python h12_ros2_controller/ros2/dual_arm_server.py --sport

uv run python -m rh56_controller.grasp_viz --h12 --bimanual --real-h12
```

The `H12Bridge.send_dual_arm()` sends to the `/dual_arm` action.

### What runs on what machine

| Component | Machine | Process |
|-----------|---------|---------|
| `frame_task_server` | Robot PC or workstation on same LAN | ROS2 node, Python |
| `grasp_viz` | Your workstation | Python (uv run) |
| MuJoCo viewer | Your workstation | Forked subprocess |
| Unitree SDK2 DDS | Robot PC → motor boards | C++ runtime (inside SDK2 Python binding) |
| ROS2 DDS | LAN | cyclone-dds (shared domain) |

---

## Bimanual Mode

### Left/Right arm selection

The active arm is determined purely by the grasp Y position:
- Y > 0 (robot's left): **left arm**
- Y ≤ 0 (robot's right): **right arm**

The boundary is `_H12_MIDPLANE_Y = 0.0` in `grasp_viz_workers.py`.

### Graceful switching

When the slider crosses Y = 0:
1. `_update_active_arm()` updates `_active_arm_val` (shared Value).
2. The bimanual worker reads this each frame:
   - **Active arm**: tracks the grasp target (PINK IK to target).
   - **Inactive arm**: PINK IK target is reset to the home-pose SE3 (arm returns to side).
3. Both run within the same PINK IK solve (single QP), so they don't fight each other.

### "Send H1-2" in bimanual mode

For real robot, the bridge sends only the **active** arm via `send_arm()` (single FrameTask).
If the arm switch happens at the same time (Y crosses midplane), the deactivated arm would
need to return to rest — this is handled by sending a `NamedConfig` action to the
deactivated arm's rest config.  This sequencing is planned but not yet implemented;
for the first real robot test, avoid crossing the midplane during a `Send H1-2` call.

---

## Wrist→Hand Attachment Transforms

Both transforms convert the hand-base target (from grasp_viz geometry) to the wrist_yaw
frame that PINK IK targets.

| Side | pos | quat (wxyz) | Rotation matrix |
|------|-----|-------------|----------------|
| **Right** | `[0.054, 0, 0]` | `(0.5, 0.5, 0.5, 0.5)` | `[[0,0,1],[1,0,0],[0,1,0]]` |
| **Left**  | `[0.054, 0, 0]` | `(0.5, 0.5, −0.5, 0.5)` | `[[0,0,1],[1,0,0],[0,−1,0]]` |

The right-hand convention: `wrist_x = base_z + 0.054, wrist_y = base_x, wrist_z = base_y`.
The left-hand convention:  `wrist_x = base_z + 0.054, wrist_y = −base_x, wrist_z = −base_y`.

---

## Button Reference (--h12 mode)

| Button | What it does |
|--------|-------------|
| **H1-2: Ours** | Opens MuJoCo viewer (`h1_2_pos_inspire.xml`) with PINK IK |
| **H1-2: Mink** | N/A (mink comparison planner not yet wired for H1-2 arm) |
| **H1-2: Bimanual** | Opens bimanual viewer (`h1_2_bimanual_inspire.xml`) with PINK IK for both arms |
| **Bimanual mode** | Toggle: enables left/right arm switching based on Y position |
| **Send H1-2** | (real-h12 only) Sends FrameTask action to move real arm |
| **Sim H1-2** | Opens/refreshes sim viewer at current slider-set pose |
| **GRASP!** | (real-h12 only) Send arm then close hand to target width |

---

## ROS2 Setup Requirements

### Packages needed

```
ros-humble-rclpy  (or ros-jazzy-rclpy)
custom_ros_messages  (from correlllab/custom_ros_messages)
h12_ros2_model       (from correlllab/h12_ros2_model)
```

### Build the workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/correlllab/custom_ros_messages
git clone https://github.com/correlllab/h12_ros2_model
ln -s ~/Programs/h12_ros2_controller .
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Run with ROS2

`grasp_viz` itself does **not** call `rclpy.init()` at startup.  `H12Bridge.connect()`
calls it lazily when `--real-h12` or `--h12-ros` is given.  Because `uv run` creates
an isolated Python environment, the ROS2 Python packages must be on the system path.
One reliable way:

```bash
source ~/ros2_ws/install/setup.bash
uv run --active python -m rh56_controller.grasp_viz --h12 --real-h12
```

Or add the ROS2 site-packages to `uv`'s environment via `.env`:

```
# .env (project root)
PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
```

---

## INFO messages on startup (--h12 mode)

```
INFO grasp_viz_core: Initialising FK model...
INFO grasp_viz_core: Loading mink comparison planner...
INFO grasp_viz_core: Mink planner ready.
INFO grasp_viz_core: H12Bridge: ROS2 node started.   ← only with --real-h12 / --h12-ros
```

In sim-only `--h12` mode you will NOT see the H12Bridge line — that is expected.
The PINK IK runs entirely in-process inside the MuJoCo viewer subprocess; no ROS
traffic is involved.

---

## File Quick-Reference

| File | Role |
|------|------|
| `rh56_controller/grasp_viz.py` | CLI entry point |
| `rh56_controller/grasp_viz_core.py` | State, geometry, arm launch, H12Bridge wiring |
| `rh56_controller/grasp_viz_workers.py` | MuJoCo viewer subprocesses (all PINK IK logic) |
| `rh56_controller/grasp_viz_ui.py` | Tkinter UI, button callbacks |
| `rh56_controller/h12_bridge.py` | ROS2 FrameTask / DualArm action client |
| `h1_mujoco/inspire/h1_2_pos_inspire.xml` | H1-2 + right hand MuJoCo scene (with floor/sky) |
| `h1_mujoco/inspire/h1_2_bimanual_inspire.xml` | H1-2 + both hands (bimanual) |
| `h1_mujoco/inspire/inspire_left_h12_qnames.xml` | Left hand body stub (left_ prefixed names) |
| `h12_ros2_controller/ros2/frame_task_server.py` | Single-arm ROS2 action server |
| `h12_ros2_controller/ros2/dual_arm_server.py` | Bimanual ROS2 action server |
| `h12_ros2_controller/core/controller/frame_controller.py` | PINK IK + Unitree SDK controller |

---

## Gravity Compensation

The H1-2 controller (`h12_ros2_controller`) has a `GravityCompController` that holds
the upper body in a fixed pose using integral control.  This is useful before a grasp
to allow manual repositioning.

To use it (via the ROS2 named_config or a separate script), see the gravity_comp_controller
documentation in h12_ros2_controller.  grasp_viz does not currently expose this in the UI
but the `H12Bridge.send_named_config("gravity_comp")` call can be invoked programmatically.
