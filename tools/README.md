# Tools

The scripts here are grouped by how they should be used.

## Root Helpers

- `setup_uv_env.sh`, `setup_ros2_uv310.sh`, `build_ros2_workspace.sh`: setup and build helpers.
- `check_profile_imports.py`, `validate_profiles.sh`: install-profile validation.
- `thumb_lever_arm.py`: simulation utility for thumb tangential-force calibration.

## Demos

- `demos/demo_h12_rh56_mujoco.py`: direct H1-2 + RH56 MuJoCo demo.

## Hardware

Scripts in `hardware/` talk to real RH56 hardware, serial ports, or timing loops.
Do not run them as part of CI.

## Experiments

Scripts in `experiments/` are interactive or data-collection workflows. Some
also require real RH56 or UR5 hardware. Treat them as maintained manual tools,
not automated tests.
