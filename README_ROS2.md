# ROS2 One-Stop Setup and Run Guide

This guide standardizes the project on a single Python 3.10 `uv` environment for ROS2 Humble workflows.

## 1. Prerequisites

System:
- Ubuntu 22.04
- ROS2 Humble installed in `/opt/ros/humble`
- Python 3.10 available as `/usr/bin/python3.10`

OS packages:
```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  build-essential cmake libeigen3-dev libyaml-cpp-dev \
  librtde librtde-dev
```

Optional (if using SDU PPA for newer ur_rtde):
```bash
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt update
sudo apt install -y librtde librtde-dev
```

## 2. Create the single uv Python 3.10 environment

From repo root:
```bash
cd /home/humanoid/Programs/rh56_controller
uv venv --python 3.10 .venv310
UV_PROJECT_ENVIRONMENT=.venv310 uv sync
```

Or run the helper:
```bash
./tools/setup_ros2_uv310.sh
```

Verify core imports in one place:
```bash
source /opt/ros/humble/setup.bash
source .venv310/bin/activate
python -c "import rclpy, mujoco, mink, magpie_control, spatialmath; print('OK')"
```

## 3. Build ROS packages in this workspace

Important: `magpie_force_control_ros` is nested under `magpie_force_control/ros2`, so build with explicit base paths:

```bash
cd /home/humanoid/Programs/rh56_controller
source /opt/ros/humble/setup.bash
source .venv310/bin/activate

colcon build \
  --base-paths . magpie_force_control/ros2 \
  --packages-select rh56_controller magpie_force_control_ros

source install/setup.bash
```

Or run the helper:
```bash
./tools/build_ros2_workspace.sh
source install/setup.bash
```

Verify package discovery:
```bash
ros2 pkg list | grep -E '^(rh56_controller|magpie_force_control_ros)$'
```

## 4. Build C++ force controller binary

The ROS bridge launches `force_control_demo`, so build it first:

```bash
cd /home/humanoid/Programs/rh56_controller/magpie_force_control
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 5. Launch options

### A. Grasp viz + ROS bridge
```bash
cd /home/humanoid/Programs/rh56_controller
source /opt/ros/humble/setup.bash
source .venv310/bin/activate
source install/setup.bash

ros2 launch rh56_controller grasp_viz_ros.launch.py \
  ur5_ip:=192.168.0.4 \
  serial_port:=/dev/ttyUSB0
```

### B. Force control ROS bridge
```bash
cd /home/humanoid/Programs/rh56_controller
source /opt/ros/humble/setup.bash
source .venv310/bin/activate
source install/setup.bash

ros2 launch magpie_force_control_ros force_control.launch.py \
  robot_ip:=192.168.0.4 \
  ft_ip:=192.168.0.3 \
  autostart:=false
```

### C. Unified system launch (Phase 2)
```bash
cd /home/humanoid/Programs/rh56_controller
source /opt/ros/humble/setup.bash
source .venv310/bin/activate
source install/setup.bash

ros2 launch rh56_controller rh56_system.launch.py \
  ur5_ip:=192.168.0.4 \
  serial_port:=/dev/ttyUSB0 \
  enable_hand_driver:=true \
  enable_force_control:=true
```

## 6. Interfaces

### rh56_controller / grasp_viz bridge
Published topics:
- `/grasp_viz/status` (`std_msgs/String`)
- `/grasp_viz/summary_json` (`std_msgs/String`)
- `/grasp_viz/hand_joint_states` (`sensor_msgs/JointState`)
- `/grasp_viz/ur5_target_pose` (`geometry_msgs/PoseStamped`)
- `/hands/cmd` (`custom_ros_messages/MotorCmds`) when `--ros-send-hand-cmd`

Subscribed topics:
- `/grasp_viz/set_mode` (`std_msgs/String`)
- `/grasp_viz/set_width_mm` (`std_msgs/Float64`)
- `/grasp_viz/set_target_width_mm` (`std_msgs/Float64`)
- `/grasp_viz/set_target_pose` (`geometry_msgs/PoseStamped`)

### magpie_force_control_ros
Published topics:
- `/force_control/wrench_measured` (`geometry_msgs/WrenchStamped`)
- `/force_control/wrench_command` (`geometry_msgs/WrenchStamped`)
- `/force_control/status` (`std_msgs/String`)

Subscribed topics:
- `/force_control/wrench_setpoint` (`geometry_msgs/WrenchStamped`)

Services:
- `/force_control/start` (`std_srvs/Trigger`)
- `/force_control/stop` (`std_srvs/Trigger`)

### motion arbitration (Phase 2)
Services:
- `/motion_arbiter/acquire` (`std_srvs/Trigger`)
- `/motion_arbiter/release` (`std_srvs/Trigger`)

## 7. Rerun

- Grasp viz rerun is enabled in `grasp_viz_ros.launch.py` (`--rerun`).
- Force bridge rerun is controlled by launch arg:
  - `rerun_enabled:=true|false`

Example:
```bash
ros2 launch magpie_force_control_ros force_control.launch.py rerun_enabled:=true
```

## 8. Common pitfalls

1. `Package 'magpie_force_control_ros' not found`:
- Build with `--base-paths . magpie_force_control/ros2` and source `install/setup.bash`.

Exact command:
```bash
colcon build --base-paths . magpie_force_control/ros2 --packages-select rh56_controller magpie_force_control_ros
source install/setup.bash
```

2. `rclpy` ABI errors:
- Use Python 3.10 env (`.venv310`) with ROS Humble, not Python 3.12.

3. `force_control_demo` missing:
- Build C++ in `magpie_force_control/build` first.

4. Concurrent UR5 control conflicts:
- Use motion arbiter and avoid simultaneously commanding UR5 from multiple nodes.

5. `No module named 'anyskin'` in grasp viz:
- Re-run `./tools/setup_ros2_uv310.sh` to install runtime deps into `.venv310`.

6. `enable_hand_driver:=true` but `custom_ros_messages` not available:
- `rh56_system.launch.py` now logs a warning and skips `rh56_driver` instead of crashing.
- In this mode, hand serial control still runs through `grasp_viz` using `serial_port:=...`.
- If you need `/hands/*` ROS message interfaces, build/source the workspace that provides `custom_ros_messages`.

## 9. Phase 2 status snapshot

Implemented in this repo:
- Unified launch file (`rh56_system.launch.py`)
- Motion arbitration node (`motion_arbiter_node`)
- Rerun logging hooks in `force_control_bridge`

Next:
- Validate `magpie_control` ROS branch behavior on hardware
- Add quantitative performance profiling runs
- Refine arbitration ownership semantics (owner-aware lock)
