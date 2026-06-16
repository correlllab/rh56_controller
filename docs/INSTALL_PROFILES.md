# Install Profiles

This repository supports several user shapes from one codebase. Pick the
smallest profile that matches the robot stack you need; move up only when you
need another subsystem.

Profiles control Python dependencies. Runtime model/controller assets still
come from git submodules, so clone with submodules or initialize the ones listed
for your profile:

```bash
git submodule update --init --recursive <submodule> ...
```

## Quick Chooser

| Goal | Profile | Python | Extra groups | Required submodules | First command |
|:--|:--|:--:|:--|:--|:--|
| Minimal sim imports / CI smoke test | `sim-core` | 3.12 | none | `h1_mujoco` for viewers | `uv run python -m rh56_controller.grasp_viz --no-mink` |
| Floating RH56 hand planner with Mink comparison | `sim-hand` | 3.12 | `sim-hand` | `h1_mujoco`, `mink` | `uv run python -m rh56_controller.grasp_viz` |
| UR5 + RH56 MuJoCo viewer | `sim-ur5` | 3.12 | `sim-ur5` | `h1_mujoco`, `mink` | `uv run python -m rh56_controller.grasp_viz --robot` |
| H1-2 + RH56 MuJoCo viewer | `sim-h12` | 3.12 | `sim-h12` | `h1_mujoco` | `uv run python -m rh56_controller.grasp_viz --h12` |
| H1-2 and UR5 sim workflows | `sim-h12-ur5` | 3.12 | `sim-h12-ur5` | `h1_mujoco`, `mink`, `magpie_control` | `uv run python -m rh56_controller.grasp_viz --h12` |
| Real RH56 hand only | `real-hand` | 3.12 | `real-hand` | `h1_mujoco` for visualizers | `uv run python -m rh56_controller.hand_mirror --port /dev/ttyUSB0` |
| Real RH56 hand + real UR5, no ROS | `real-ur5` | 3.12 | `real-ur5` | `h1_mujoco`, `mink`, `magpie_control` | `uv run python -m rh56_controller.grasp_viz --robot --real-robot --ur5-ip 192.168.0.4 --port /dev/ttyUSB0` |
| Real RH56 hand + UR5 with ROS2 bridge | `real-ur5-ros` | 3.10 | `real-ur5-ros`, `ros` | `h1_mujoco`, `mink`, `magpie_control` | `ros2 launch rh56_controller grasp_viz_ros.launch.py ur5_ip:=192.168.0.4 serial_port:=/dev/ttyUSB0` |
| Real H1-2 arm, no RH56 serial hand | `real-h12-ros` | 3.10 | `real-h12-ros`, `ros` | `h1_mujoco`; external `h12_ros2_controller` sourced | `uv run python -m rh56_controller.grasp_viz --h12 --real-h12` |
| Real H1-2 + real RH56 hand | `real-h12-hand-ros` | 3.10 | `real-h12-hand-ros`, `ros` | `h1_mujoco`; external `h12_ros2_controller` sourced | `uv run python -m rh56_controller.grasp_viz --h12 --real-h12 --port /dev/ttyUSB0` |
| Maintainer/dev environment | `dev-full` | 3.10 or 3.12 | `dev-full`, `ros` | all submodules | use the workflow-specific command above |

`full` and `real-robot` are kept as backward-compatible aliases for
`dev-full` and `real-ur5`.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules https://github.com/correlllab/rh56_controller.git
cd rh56_controller

# Example: sim hand planner
tools/setup_uv_env.sh --profile sim-hand --python 3.12 --env .venv312
source .venv312/bin/activate
python tools/check_profile_imports.py --profile sim-hand
```

ROS2 Humble workflows should use Python 3.10 and source ROS before running ROS
commands:

```bash
source /opt/ros/humble/setup.bash
tools/setup_uv_env.sh --profile real-h12-hand-ros --python 3.10 --env .venv310 --telemetry
source .venv310/bin/activate
```

## Validation

List all profiles:

```bash
python tools/check_profile_imports.py --list
```

Validate one profile's Python imports and required local paths:

```bash
python tools/check_profile_imports.py --profile sim-ur5
```

If you only want to check Python imports, skip path checks:

```bash
python tools/check_profile_imports.py --profile sim-ur5 --no-paths
```

If you only want to check that submodules/assets are present, skip imports:

```bash
python tools/check_profile_imports.py --profile all --paths-only
```

Run the main sim install checks end-to-end:

```bash
tools/validate_profiles.sh
```

This sets up and validates `sim-hand`, `sim-ur5`, and `sim-h12` in separate
envs matching the README examples: `.venv312`, `.venv312-ur5`, and
`.venv312-h12`. To reuse existing envs:

```bash
tools/validate_profiles.sh --check-only
```

## Boundaries

Base dependencies cover the common numerical and MuJoCo viewer stack:
`numpy`, `scipy`, `matplotlib`, and `mujoco`.

Optional groups add only their subsystem:

| Group | Adds |
|:--|:--|
| `hand` / `real-hand` | `pyserial` for RH56 hardware |
| `mink` | Mink differential IK comparison |
| `ur5` | `magpie_control` and `spatialmath-python` for real UR5 control |
| `h12` | Pinocchio, PINK, DAQP/QPSolvers, Meshcat for H1-2 simulation/control |
| `ros` | Semantic marker only; ROS Python packages come from ROS/apt/colcon |
| `telemetry` | `rerun-sdk` |

When adding a dependency, add it to the narrowest group that needs it, then
update this guide, `tools/setup_uv_env.sh`, and
`tools/check_profile_imports.py` in the same change.
