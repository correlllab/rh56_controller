# Tools

The scripts here are grouped by how they should be used.

## Root Helpers

- `setup_uv_env.sh`, `setup_ros2_uv310.sh`, `build_ros2_workspace.sh`: setup and build helpers.
- `check_profile_imports.py`, `validate_profiles.sh`: install-profile validation.
- `thumb_lever_arm.py`: simulation utility for thumb tangential-force calibration.

## Demos

- `demos/demo_h12_rh56_mujoco.py`: direct H1-2 + RH56 MuJoCo demo.

## Camera Calibration

- `calibrate_ur5_external_camera_sim.py`: fixed external-camera intrinsic and
  eye-to-hand calibration in the UR5+RH56 MuJoCo scene. It writes the reusable
  calibration schema, numeric validation, and image overlays under `artifacts/`.
- `calibrate_ur5_external_camera_real.py`: read-only D435 RGB-D capture and
  fixed-camera eye-to-hand calibration. `--check-only` does not contact the UR;
  `--print-board-only` generates a physical-size SVG without opening hardware;
  normal capture opens only the RTDE receive interface and never commands motion.
  The explicit `--laser-power` flag and saved camera metadata support repeatable
  runtime depth settings without changing robot state.
- `calibrate_ur_pointer.py`: read-only multi-orientation calibration of a rigid
  pointer relative to the currently reported UR TCP, followed by base-frame
  point surveying. It does not require entering a correct TCP on the pendant.
- `analyze_ur5_camera_desk_plane.py`: offline aligned-depth desk-plane check in
  the calibrated UR base frame. It reports plane residual, normal tilt, and a
  clearly qualified base-frame height estimate under `artifacts/`. Pass an
  independent ruler measurement with `--measured-desk-z-mm` to report the
  absolute height discrepancy against a configurable threshold.

## Grasp safety preflight

- `preflight_ggx_ur5_table.py`: simulation-only GraspGen-X candidate screening
  against an adjustable desk, with YAML setup reuse, annotated review videos,
  multiple confidence-ranked PASS options, and explicit human veto records
  with UR5e IK, the RH56 collision model, an adjustable desk height, sampled
  approach/closure/lift motion, CSV diagnostics, and optional review video or
  live viewer. It never imports a hardware bridge; see
  `docs/graspgenx_ur5_table_preflight.md`.

## Hardware

Scripts in `hardware/` talk to real RH56 hardware, serial ports, or timing loops.
Do not run them as part of CI.

## Experiments

Scripts in `experiments/` are interactive or data-collection workflows. Some
also require real RH56 or UR5 hardware. Treat them as maintained manual tools,
not automated tests.
