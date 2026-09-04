# UR5 + RH56 External Camera Calibration

This workflow covers both simulation validation and read-only real D435
acquisition. Neither calibration tool sends a robot or RH56 motion command.

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
5. Freeze one RGB-D stream mode. The current D435 workflow uses aligned
   1280-by-720 color and depth at 30 Hz and preserves its factory intrinsics;
   `--intrinsics-source estimate` is available as an explicit comparison.
6. Validate with an image that was not used for calibration before running a
   grasp trial.

A camera aimed vertically down at the desk is supported. The camera angle is
not an input to the solver. A vertical view can simplify tabletop segmentation,
but may increase arm/hand occlusion and makes it especially important to tilt
the wrist-mounted board about multiple axes. Do not collect a sequence that is
only translated under the camera with an unchanged orientation.

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

## Real D435 preflight

Install the read-only calibration profile and test support:

```bash
tools/setup_uv_env.sh \
  --profile real-ur5-vision \
  --python 3.12 \
  --env .venv312 \
  --tests
```

Capture one aligned RGB-D frame without contacting the UR:

```bash
.venv312/bin/python tools/calibrate_ur5_external_camera_real.py \
  --check-only \
  --camera-serial 836612071918 \
  --out artifacts/ur5_external_camera_calibration_real/camera_check
```

This writes raw RGB, annotated detection, raw uint16 depth, a depth preview,
`summary.csv`, camera metadata, and a physical-size `printable_chessboard.svg`.
The default stream is aligned RGB and depth at 1280 by 720, 30 Hz. Keep that
stream mode fixed after calibration.

## Flat-board route when no calibrated TCP exists

For a board that stays flat on the desk, first survey its pose in the UR base
frame. The pendant's active TCP does not have to be accurate for this route.
Do not enter a guessed pointer length. Instead, rigidly mount a short, blunt
point probe and keep the same active TCP configuration selected for the whole
session. Do not use a compliant RH56 fingertip as the probe.

Make a small fixed reference dimple away from the chessboard. Under the lab's
normal manual/reduced-speed procedure, touch the probe tip to exactly that same
dimple at six substantially different wrist orientations. Stop before each
read, then run one command:

```bash
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-reference pose_01
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-reference pose_02
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-reference pose_03
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-reference pose_04
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-reference pose_05
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-reference pose_06
.venv312/bin/python tools/calibrate_ur_pointer.py --solve
```

The solve uses the constraint that the physical tip remains on one fixed base
point while its coordinates relative to the reported TCP remain constant. It
writes `pointer_calibration.yaml` and `summary.csv` under
`artifacts/ur_pointer_calibration/calibration_pointer/`. The default quality
gate requires at least 30 degrees of orientation span, leave-one-out RMS no
larger than 1 mm, and maximum leave-one-out error no larger than 2 mm. A pass
is an internal pointer-calibration check, not yet proof of sub-5-mm camera
accuracy.

After the pointer passes, touch the three marked inner corners in the annotated
camera-check image and record them:

```bash
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-point board_origin
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-point board_x
.venv312/bin/python tools/calibrate_ur_pointer.py --capture-point board_y
```

For the measured 24.2 mm print, `board_x` is eight intervals (193.6 mm) from
the origin and `board_y` is five intervals (121.0 mm) from the origin. These
three points define the complete board pose, including desk-plane height and
orientation. The program only opens `RTDEReceiveInterface`; all arm motion is
manual and no motion command exists in the tool.

## Real eye-to-hand capture

For the compact 140-by-130 mm wrist backing, generate the selected asymmetric
layout without opening either camera or robot:

```bash
.venv312/bin/python tools/calibrate_ur5_external_camera_real.py \
  --print-board-only \
  --pattern-cols 9 \
  --pattern-rows 8 \
  --square-size-mm 12 \
  --page-width-mm 140 \
  --page-height-mm 130 \
  --out artifacts/ur5_external_camera_calibration_real/hand_eye_board_140x130
```

This produces 9-by-8 inner corners on a 10-by-9-square checker field measuring
120 by 108 mm. Print at actual size with fit-to-page disabled. Horizontally,
eight inner-corner intervals should measure 96 mm; vertically, seven intervals
should measure 84 mm. Measure both directions and use the measured average
square spacing rather than assuming the nominal 12 mm.

After rigidly attaching the flat print beside the hand/TCP, run the capture
with the matching pattern and measured square spacing:

```bash
.venv312/bin/python tools/calibrate_ur5_external_camera_real.py \
  --camera-serial 836612071918 \
  --ur-ip 192.168.0.4 \
  --pattern-cols 9 \
  --pattern-rows 8 \
  --square-size-mm 12 \
  --captures 25 \
  --out artifacts/ur5_external_camera_calibration_real/first_lab_dataset
```

The operator moves the arm manually and presses Enter only after the robot is
stopped. The tool imports only `RTDEReceiveInterface`, reads
`getActualTCPPose`, and rejects a sample when the TCP changes while the frame
is captured. It never enables freedrive, writes a register, or constructs an
RTDE control interface. Use `--pose-source manual` when poses will instead be
copied from the teach pendant.

Every fifth accepted view is held out by default. The final
`camera_calibration.yaml` contains `T_base_camera`, `T_tcp_target`, factory or
estimated intrinsics, pose-diversity diagnostics, training and held-out
residuals, and an explicit quality result. `summary.csv`,
`calibration_views.csv`, and `verification_holdout.png` provide the
paper-facing numeric and visual checks. A saved dataset can be recomputed
without hardware using `--calibrate-only --out <dataset>`.
