# ROS2 Port Phase 1 — Completion Summary

**Branch:** `ros`  
**Date:** March 9, 2026  
**Status:** ✅ Phase 1 Complete — Ready for Hardware Validation

---

## What Was Implemented

### 1. rh56_controller ROS2 Bridge (`grasp_viz_ros.py`)

A lightweight, in-process ROS2 bridge that runs alongside the tkinter UI:

**Published Topics:**
- `/grasp_viz/status` — Status messages (std_msgs/String)
- `/grasp_viz/summary_json` — Complete planner state as JSON (std_msgs/String)
- `/grasp_viz/hand_joint_states` — Right hand joint positions (sensor_msgs/JointState)
- `/grasp_viz/ur5_target_pose` — Planned grasp pose (geometry_msgs/PoseStamped)
- `/hands/cmd` — Optional hand command mirroring (custom_ros_messages/MotorCmds)

**Subscribed Topics:**
- `/grasp_viz/set_mode` — Change closure mode
- `/grasp_viz/set_width_mm` — Update solve width
- `/grasp_viz/set_target_width_mm` — Update execution target width
- `/grasp_viz/set_target_pose` — Command new grasp pose

**Rerun Integration:**
- Optional rerun stream (`--rerun` flag)
- Publishes target point and fingertip positions
- Root frame: `grasp_viz_ros` application with `RIGHT_HAND_Z_UP` coordinate system

**Files:**
- [rh56_controller/grasp_viz_ros.py](rh56_controller/grasp_viz_ros.py)
- [rh56_controller/grasp_viz_core.py](rh56_controller/grasp_viz_core.py) (ROS command handlers)
- [rh56_controller/grasp_viz.py](rh56_controller/grasp_viz.py) (CLI flags)
- [launch/grasp_viz_ros.launch.py](launch/grasp_viz_ros.launch.py)

### 2. magpie_force_control ROS2 Wrapper

A Python-based ROS2 package that manages the C++ `force_control_demo` binary:

**Architecture:**
- Process manager node launches C++ binary as subprocess
- Parses stdout for wrench measurements
- Translates ROS topics → CLI arguments
- Provides start/stop services for controller lifecycle

**Topics:**
- `/force_control/wrench_measured` — Real-time F/T sensor readings
- `/force_control/wrench_command` — Controller output
- `/force_control/wrench_setpoint` — Target wrench input
- `/force_control/status` — Controller state

**Services:**
- `/force_control/start` — Launch controller subprocess
- `/force_control/stop` — Terminate controller subprocess

**Files:**
- [magpie_force_control/ros2/magpie_force_control_ros/](magpie_force_control/ros2/magpie_force_control_ros/)
- [magpie_force_control/ros2/magpie_force_control_ros/magpie_force_control_ros/force_control_bridge.py](magpie_force_control/ros2/magpie_force_control_ros/magpie_force_control_ros/force_control_bridge.py)
- [magpie_force_control/ros2/magpie_force_control_ros/launch/force_control.launch.py](magpie_force_control/ros2/magpie_force_control_ros/launch/force_control.launch.py)

### 3. Vendored Yifan Dependencies

Added as git submodules in `magpie_force_control/third_party/`:
- [cpplibrary](https://github.com/badinkajink/cpplibrary) → `third_party/cpplibrary`
- [force_control](https://github.com/badinkajink/force_control) → `third_party/force_control`

**CMake Priority Logic:**
1. Check system-installed libraries first
2. Check vendored `third_party/` submodules second
3. Fall back to `FetchContent` from GitHub as last resort

This ensures deterministic builds and allows custom patches in the forks.

### 4. Documentation

- [ROS2_PORT_PLAN.md](ROS2_PORT_PLAN.md) — Full multi-repo migration tracker
- [README_REAL.md](README_REAL.md) — Updated with ROS2 + rerun usage
- [magpie_force_control/ros2/magpie_force_control_ros/README.md](magpie_force_control/ros2/magpie_force_control_ros/README.md) — Wrapper package docs

---

## Quick Start (Simulation Mode)

Test ROS2 bridge without hardware:

```bash
# Terminal 1: Launch grasp_viz with ROS bridge and rerun
uv run python -m rh56_controller.grasp_viz --robot --ros-sync --rerun

# Terminal 2: Monitor ROS topics
ros2 topic list
ros2 topic echo /grasp_viz/summary_json

# Terminal 3: Command via ROS
ros2 topic pub /grasp_viz/set_mode std_msgs/msg/String "{data: '3-finger plane'}"
ros2 topic pub /grasp_viz/set_width_mm std_msgs/msg/Float64 "{data: 50.0}"
```

---

## Hardware Validation Checklist

### Prerequisites
- [ ] UR5 connected at 192.168.0.4
- [ ] RH56 hand connected via USB serial
- [ ] OptoForce F/T sensor at 192.168.0.3 (for force control)
- [ ] ROS2 workspace sourced: `source /opt/ros/humble/setup.bash`
- [ ] Custom ROS messages built: `colcon build --packages-select custom_ros_messages`

### Test Sequence

#### 1. GraspViz ROS Bridge (5 min)
```bash
# Launch with hardware
uv run python -m rh56_controller.grasp_viz \
    --robot \
    --real-robot \
    --ur5-ip 192.168.0.4 \
    --port /dev/ttyUSB0 \
    --ros-sync \
    --rerun

# In separate terminals:
# ✅ Verify topics appear
ros2 topic list | grep grasp_viz

# ✅ Verify JSON state publishes at ~20 Hz
ros2 topic hz /grasp_viz/summary_json

# ✅ Command mode change from ROS
ros2 topic pub --once /grasp_viz/set_mode std_msgs/msg/String "{data: 'cylinder'}"
# → Check UI updates to cylinder mode

# ✅ Verify rerun opens with live fingertip points
# → Check rerun viewer window shows grasp_viz/* entities
```

#### 2. Force Control Wrapper (10 min)
```bash
# First, build C++ binaries
cd magpie_force_control
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build ROS2 wrapper
cd ../ros2
colcon build --packages-select magpie_force_control_ros
source install/setup.bash

# Launch bridge (manual start mode)
ros2 launch magpie_force_control_ros force_control.launch.py \
    robot_ip:=192.168.0.4 \
    ft_ip:=192.168.0.3

# In another terminal:
# ✅ Start controller via service
ros2 service call /force_control/start std_srvs/srv/Trigger

# ✅ Monitor wrench measurements
ros2 topic echo /force_control/wrench_measured

# ✅ Command a wrench setpoint
ros2 topic pub /force_control/wrench_setpoint geometry_msgs/msg/WrenchStamped "{
  wrench: {force: {z: 2.0}}
}"

# ✅ Stop controller
ros2 service call /force_control/stop std_srvs/srv/Trigger
```

#### 3. Integrated Operation (15 min)
```bash
# Terminal 1: Hand + GraspViz
ros2 launch rh56_controller grasp_viz_ros.launch.py \
    ur5_ip:=192.168.0.4 \
    serial_port:=/dev/ttyUSB0

# Terminal 2: Force Controller
ros2 launch magpie_force_control_ros force_control.launch.py \
    robot_ip:=192.168.0.4 \
    autostart:=false

# Terminal 3: Rerun CLI (optional unified view)
rerun --connect

# Test coordinated grasp:
# 1. Plan grasp in UI (mode, width, pose)
# 2. Execute "GRASP!" from UI
# 3. Start force controller after grasp completes
# 4. Apply force via ROS topic
# 5. Monitor all state in rerun viewer

# ✅ Verify no RTDE connection conflicts between nodes
# ✅ Verify hand command topic receives updates when UI slider moves
# ✅ Verify rerun shows both grasp fingertips and force wrench arrows
```

---

## Known Limitations / Future Work

1. **Concurrent UR5 Control:**  
   GraspViz and force_control both need RTDE access. Coordinate via manual sequencing (grasp first, then activate force control) or implement a motion arbitrator node.

2. **Rerun Unification:**  
   Currently grasp_viz publishes to its own rerun app. Ideally all nodes publish to one shared rerun recording for synchronized timeline view.

3. **magpie_control ROS Branch:**  
   Not yet tested. Need to checkout ros branch and validate UR5_Interface ROS wrapper on this hardware.

4. **Hand Command Mirror:**  
   `--ros-send-hand-cmd` publishes right-hand joint commands to `/hands/cmd`. Left hand repeats last known state. For bimanual, extend to receive explicit left-hand commands.

5. **Action Interface:**  
   Force control wrapper uses start/stop services. Future: add ROS2 action for trajectory-based force profiles.

---

## Git State

**Branch:** `ros` (diverged from `main`)

**Commits:**
- `ecac728` — Complete Phase 1: ROS2 integration for grasp_viz and magpie_force_control
- Submodules updated:
  - `magpie_force_control` → latest with ROS2 wrapper
  - `rerun_rlds_ur5` → added for rerun examples

**Submodule Status:**
- `magpie_control` — unchanged (on main)
- `magpie_force_control` — +1 commit (ROS2 wrapper)
- `mink` — unchanged
- `rerun_rlds_ur5` — new submodule

---

## Next Steps (Phase 2)

1. **Smoke tests on hardware** (this document)
2. **magpie_control ROS branch validation:**
   - Checkout ros branch
   - Test UR5_Interface ROS wrapper
   - Run RTDE connection stress tests
3. **Unified launch file:**
   - Single launch file for hand + grasp_viz + force_control
   - Motion arbitration layer
4. **Rerun unification:**
   - All nodes publish to shared `rec://rh56_system` recording
   - Add wrench arrows, robot mesh, contact forces to viz
5. **Performance profiling:**
   - ROS publish rate impact on UI responsiveness
   - RTDE read latency under concurrent access

---

## Contact / Issues

File issues on [correlllab/rh56_controller](https://github.com/correlllab/rh56_controller) with `[ROS2]` prefix.

For force control C++ questions, reference upstream:
- [yifan-hou/cpplibrary](https://github.com/yifan-hou/cpplibrary)
- [yifan-hou/force_control](https://github.com/yifan-hou/force_control)
