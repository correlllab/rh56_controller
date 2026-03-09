# ROS2 Port Plan and Execution Tracker

This document tracks the multi-repo ROS2 migration for:
- rh56_controller (this workspace)
- magpie_control (UR5 interface)
- magpie_force_control (admittance/force stack)

## Current Execution Status

### Completed in this changeset (Phase 1)
- ✅ Added in-process ROS2 bridge for GraspViz state and command sync.
- ✅ Added optional rerun telemetry stream from GraspViz.
- ✅ Added launch file to run GraspViz with ROS bridge enabled.
- ✅ Added CLI flags to control ROS sync, ROS rate, hand command mirroring, and rerun.
- ✅ Vendored Yifan dependencies (cpplibrary, force_control) as git submodules in magpie_force_control/third_party/.
- ✅ Updated magpie_force_control CMakeLists.txt to prefer vendored submodules over FetchContent.
- ✅ Created magpie_force_control_ros ROS2 Python wrapper package with:
  - Bridge node for subprocess management
  - ROS topics for wrench command/measurement
  - Services for start/stop control
  - Launch file for ROS2 integration
- ✅ Updated documentation for ROS2 usage patterns.

### Next milestones (Phase 2)
- 🔄 Build and smoke-test the magpie_force_control ROS2 wrapper on this hardware.
- 🔄 Validate UR5 connectivity through the magpie_control ros branch (create ros branch if needed).
- 🔄 Add unified ROS bringup launch for hand + viz + force nodes.
- 🔄 Test rerun visualization with live hardware data streams.

## 1) rh56_controller migration (UI + ROS + rerun)

### Implemented interfaces
Published topics:
- /grasp_viz/status (std_msgs/String)
- /grasp_viz/summary_json (std_msgs/String, compact JSON)
- /grasp_viz/hand_joint_states (sensor_msgs/JointState)
- /grasp_viz/ur5_target_pose (geometry_msgs/PoseStamped)

Subscribed topics:
- /grasp_viz/set_mode (std_msgs/String)
- /grasp_viz/set_width_mm (std_msgs/Float64)
- /grasp_viz/set_target_width_mm (std_msgs/Float64)
- /grasp_viz/set_target_pose (geometry_msgs/PoseStamped)

Optional publish:
- /hands/cmd (custom_ros_messages/MotorCmds) when --ros-send-hand-cmd is enabled.

Rerun stream:
- world frame and target/tip points from live planner state.

### Run commands
CLI:
- uv run python -m rh56_controller.grasp_viz --robot --ros-sync --rerun

ROS launch:
- ros2 launch rh56_controller grasp_viz_ros.launch.py

## 2) magpie_control ROS branch validation plan

Use the existing ros branch in magpie_control as the baseline, then verify UR5 behavior on this hardware.

Validation matrix:
1. RTDE connect/disconnect loops (20 cycles) with no stale sockets.
2. teach mode toggle reliability and idempotence.
3. moveL to three canonical waypoints and back-home repeatability.
4. emergency stop behavior from client side.
5. pose read latency and drift under no-motion hold.

Acceptance criteria:
- No RTDE deadlocks.
- No orphan threads after shutdown.
- Mean pose read latency below 20 ms.
- No unexpected reconnect required between trials.

## 3) magpie_force_control ROS2 compatibility plan

Goal: keep high-rate C++ controller while exposing ROS2 Python package APIs.

Architecture:
- Keep force_control_demo and ft sensor code in C++.
- Add ament_python ROS package that:
  - launches C++ binaries as managed subprocesses,
  - republishes measured wrench/state on ROS topics,
  - accepts ROS command goals and writes CLI/config args.

Recommended package shape in magpie_force_control:
- ros2/magpie_force_control_ros/package.xml
- ros2/magpie_force_control_ros/setup.py
- ros2/magpie_force_control_ros/magpie_force_control_ros/bridge_node.py
- ros2/magpie_force_control_ros/launch/force_control.launch.py

## 4) Yifan dependency strategy (fork + submodule)

Requested by project direction: fork and vendor Yifan repos for customization.

Target repos:
- yifan-hou/cpplibrary
- yifan-hou/force_control

Recommended submodule layout:
- magpie_force_control/third_party/cpplibrary
- magpie_force_control/third_party/force_control

Migration notes:
- Point CMake to vendored paths first, fallback to FetchContent second.
- Pin to tested commit SHAs in .gitmodules.
- Keep a patch queue in magpie_force_control/patches for ROS-specific modifications.

## 5) Unified rerun strategy

Use one rerun recording tree namespace per subsystem:
- grasp_viz/* for planner and fingertip state.
- ur5/* for robot pose/joints from ROS topics.
- force_control/* for wrench, setpoints, and controller diagnostics.

This lets you view synchronized timelines in one rerun session.

## 6) Immediate test checklist (this repo)

1. ros2 topic list includes /grasp_viz/* topics after launching GraspViz bridge.
2. Publishing /grasp_viz/set_mode changes planner mode.
3. Publishing /grasp_viz/set_width_mm updates closure solution.
4. /grasp_viz/summary_json updates at configured --ros-publish-hz.
5. rerun opens and renders target + fingertip points.
6. If enabled, /hands/cmd receives 12 motor commands per frame.

## 7) Risk list

- Tkinter and ROS callback concurrency can race if UI widgets are manipulated from ROS threads.
- custom_ros_messages schema drift may break /hands/cmd publishing.
- UR5 and simulation control loops can diverge if timebases are not synchronized.
- High publish rates can overload rerun logging without downsampling.

## 8) Suggested next implementation step

Build the magpie_force_control ROS wrapper package first (minimal bridge with process manager + topics), then add action/service interfaces after hardware smoke tests pass.
