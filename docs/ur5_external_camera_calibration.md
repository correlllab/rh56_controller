# UR5 + RH56 External Camera Calibration

This workflow validates fixed-camera calibration before the same procedure is
run on hardware. It does not inject artificial errors and it never sends a
hardware command.

## Simulation

Set up the dedicated profile:

```bash
tools/setup_uv_env.sh \
  --profile sim-ur5-vision \
  --python 3.12 \
  --env .venv312
```

Run the calibration:

```bash
MUJOCO_GL=egl .venv312/bin/python \
  tools/calibrate_ur5_external_camera_sim.py \
  --out artifacts/ur5_external_camera_calibration
```

The simulated target is a 9-by-6 inner-corner chessboard with 25 mm squares,
rigidly mounted beside the RH56 hand. The script renders at least 25 UR5 poses,
detects the target from images, calibrates the pinhole intrinsics, and solves
the fixed eye-to-hand transform from OpenCV observations plus robot forward
kinematics. MuJoCo's exact camera pose is hidden from the estimator and used
only for the final numeric check.

Before rendering a capture, the script checks the complete joint-space
segment from the preceding pose. By default, no arm joint moves more than one
degree between collision samples. The board, its conservative 8 mm-radius bracket,
and an 80 x 40 x 30 mm camera housing have collision geometry with a 10 mm
clearance margin. A candidate is rejected if it violates a UR5 joint limit or
creates any non-whitelisted MuJoCo contact. The only whitelist is the fixed
floor-to-shoulder overlap already present in the source UR5 scene.

Important outputs are:

- `camera_calibration.yaml`: camera matrix, distortion, `T_base_camera`, frame
  conventions, assumptions, and simulation truth used for validation.
- `summary.csv`: intrinsic, reprojection, hand-eye residual, and hidden-truth
  errors.
- `capture_summary.csv`: every attempted joint pose and detection result. It
  also records path-check samples, rejection reasons, contact pairs, and an
  explicit `hardware_replay_approved=0` field.
- `calibration_montage.png`: representative detected views.
- `verification.png`: detected corners in green and corners reconstructed from
  robot FK plus the estimated calibration in magenta.
- `printable_chessboard.svg`: the same board with physical millimetre units;
  print at 100% scale and verify the square spacing with a ruler or caliper.

The saved transform uses OpenCV camera axes (+X right, +Y down, +Z forward) and
maps camera-frame points into the UR5 base frame:

```text
p_base = T_base_camera @ p_camera
```

## Hardware Transfer

Do not copy the simulated numeric matrix onto the robot. Reuse the target
dimensions, frame convention, image detector, calibration solver, output
schema, and validation plots, then collect new images and robot poses after the
real camera is mounted.

For the real camera:

1. Print a flat 10-by-7-square board whose 9-by-6 inner corners are spaced
   exactly 25 mm apart, or change `--square-size-mm` to its measured spacing.
2. Rigidly attach it beside the RH56 hand. Its exact mounting transform does
   not have to match simulation and is solved as part of eye-to-hand
   calibration.
3. Keep the external camera fixed for the entire capture and experiment.
4. Collect at least 20–25 views spanning translation and rotation; do not use
   near-identical wrist orientations.
5. Use the camera's native resolution and enable distortion estimation for
   real images.
6. Validate with an image that was not used for calibration before running a
   grasp trial.

Simulation currently provides the verified acquisition/calibration backend.
The simulation collision check is useful for finding obvious modelled
collisions, but it is not a real-robot safety certificate. The current scene
does not know the exact lab table, camera stand, clamps, cables, soft covers,
mount tolerances, controller tracking error, or stopping distance. Do not send
the generated joint poses to the UR5.

For the first hardware dataset, keep acquisition separate from motion:

1. Put the UR controller in reduced/manual mode and set conservative speed and
   force limits using the lab's normal safety procedure.
2. Use freedrive or the teach pendant to move to one visually clear pose at a
   time; do not let this calibration code command the arm.
3. Stop the robot, check clearance around the arm, RH56, board, camera stand,
   table, and cables, then capture one image and the matching
   `T_base_gripper` timestamp/pose.
4. Repeat for 20--25 varied poses. Retreat manually if any clearance is
   uncertain.
5. Estimate distortion and extrinsics from those recorded pairs, then validate
   on held-out views before any grasp experiment.

This repository is currently on the main simulation/paper branch, so a live
UR interface is intentionally not added here. A read-only camera/pose logger
can be connected on the hardware-debug branch once the exact webcam API, UR
pose source, TCP definition, and lab safety setup are known.
