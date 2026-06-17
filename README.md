# RH56 Controller

Control, simulation, and experiment tooling for the Inspire RH56DFX dexterous hand.

The repository is kept lightweight for users who need to install and run the code. Qualitative videos, project media, and paper-facing results live on the [RH56DFX project page](https://correlllab.github.io/rh56dfx.html); large local datasets and generated outputs are intentionally kept out of Git.

## What Is Here

- Python control APIs for the RH56 hand.
- MuJoCo simulation and grasp-planning tools.
- Optional UR5, H1-2, ROS2, and force-control workflows.
- Profile-based installation so users can install only the stack they need.
- Developer and experiment utilities for calibration, visualization, and validation.

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth 1 https://github.com/correlllab/rh56_controller.git
cd rh56_controller
git submodule update --init --recursive --depth 1 h1_mujoco mink
tools/setup_uv_env.sh --profile sim-hand --python 3.12 --env .venv312
source .venv312/bin/activate
python tools/check_profile_imports.py --profile sim-hand
uv run python -m rh56_controller.grasp_viz
```

For other robot stacks, start with the profile chooser in [docs/INSTALL_PROFILES.md](docs/INSTALL_PROFILES.md).

## Common Profiles

| Goal | Profile | Submodules | First command |
|:--|:--|:--|:--|
| Floating RH56 hand planner | `sim-hand` | `h1_mujoco`, `mink` | `uv run python -m rh56_controller.grasp_viz` |
| UR5 + RH56 simulation | `sim-ur5` | `h1_mujoco`, `mink` | `uv run python -m rh56_controller.grasp_viz --robot` |
| H1-2 + RH56 simulation | `sim-h12` | `h1_mujoco` | `uv run python -m rh56_controller.grasp_viz --h12` |
| Real RH56 hand only | `real-hand` | optional `h1_mujoco` | `uv run python -m rh56_controller.hand_mirror --port /dev/ttyUSB0` |
| Real RH56 hand + UR5 | `real-ur5` | `h1_mujoco`, `mink`, `magpie_control` | `uv run python -m rh56_controller.grasp_viz --robot --real-robot --ur5-ip 192.168.0.4 --port /dev/ttyUSB0` |
| ROS2 Humble workflows | `real-h12-hand-ros` | `h1_mujoco` | `ros2 launch rh56_controller rh56_system.launch.py` |

## Documentation

| Topic | Guide |
|:--|:--|
| Install profile chooser | [docs/INSTALL_PROFILES.md](docs/INSTALL_PROFILES.md) |
| Simulation and visualization | [docs/sim.md](docs/sim.md) |
| Grasp geometry planner | [docs/grasp_viz.md](docs/grasp_viz.md) |
| Real UR5 + RH56 setup | [docs/real.md](docs/real.md) |
| ROS2 workflows | [docs/ros2.md](docs/ros2.md) |
| H1-2 workflows | [docs/h12.md](docs/h12.md) |
| Force control UI | [docs/force_control.md](docs/force_control.md) |
| Force visualization | [docs/force_viz.md](docs/force_viz.md) |
| Local artifacts policy | [docs/ARTIFACTS.md](docs/ARTIFACTS.md) |

## Repository Boundaries

This repo should contain source code, install tooling, small model assets, and written documentation. It should not track raw experiment logs, generated plots, local workspaces, PDFs, or videos. See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the local artifact layout.

## Citation

```bibtex
@misc{tan2026characterizationanalyticalplanninghybrid,
  title={Characterization, Analytical Planning, and Hybrid Force Control for the Inspire RH56DFX Hand},
  author={Xuan Tan and William Xie and Nikolaus Correll},
  year={2026},
  eprint={2603.08988},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2603.08988},
}
```
