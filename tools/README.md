# Tools

The scripts here are grouped by how they should be used.

- `run_ggx_feasibility.py`: hardware-free recheck of a saved batch, using the
  official scene-distance filter and the same CB3/fixed-desk builder as execution.
  Saves separate official, IK/workspace/desk, and uncertain target-proxy outcomes.
  No re-inference/reranking, no uncertainty-to-execution bypass. See
  [protocol and limitations](../docs/ggx_official_filter_zh.md).
- `run_ggx_official_filter.py`: calibrated scene preparation in the main environment
  (`--prepare-only`), official function/mesh screening in the GGX environment.
  Default 20 mm, original ranks, selected-target exclusion, saved sampling/versions;
  endpoint-only, not a trajectory certificate or grasp-success measurement.
- `run_ggx_experiment.py`: generate/review GGX once per object, then reuse the
  grasp over setup-wide fixed P1..P10 (fixed position/orientation/open shape,
  independent of candidate). Remeasure RGB XY after replaced-object reset;
  freeze reference Z/yaw/dimensions, no current-depth localization gate. Skip
  typed pre-motion path collisions, stop other faults. Offline by default;
  `--capture-live` enables camera, `--execute` requires per-trial video Enter
  and separately confirmed resets. Saves layout/template, XY overlays and
  summary CSV; no pickup verification or automatic release. See
  [ten-start instructions](../docs/ggx_ten_starts_zh.md).
- `run_ggx_ten_starts.py`: lower-level frozen-template debugging/replay entry;
  same fixed layout by default, optional legacy explicit-coordinate overrides.
  No new GGX inference or automatic XY tracking; use the experiment entry above
  for the cross-object protocol.
- `hardware/plan_airgap.py`: offline single original-Plan hand-shape preview;
  an explicit execute flag, empty-hand/clear-path confirmation and fresh receive-only
  UR snapshot permit one low-setting hand pose for manual gap measurement. No arm
  commands or automatic sweep/retry; raw hand TX/RX and summaries are recorded.
- `run_sugar_box_trial.py`: separate known 176x90x40 mm flat-box input adapter;
  inherits the original `LiveCycle` grasp/lift implementation. Only scan return
  uses saved-joint MoveJ at 0.10 rad/s, with sampled calibrated controller FK and
  safety/workspace checks (not full-arm/hand collision certification). Offline by
  default; `--preflight-only --confirm-controller-script` checks entry only without
  motion. Execution requires Enter and fresh camera data; pick preflight occurs
  at the actual scan pose. `--camera-check-only --out NEW_DIRECTORY` validates
  the same live detection without connecting to robot/hand: shared 60-frame
  warmup, three saved raw frames, all must pass unchanged geometry/consistency
  gates; no frame replacement, fragment merging or motion retry.
  `--execute --reset-only --out NEW_DIRECTORY` requires Enter and runs only
  prepare/open/checked scan return (hand speed100/force100, slow vertical retreat
  if needed, joint return 0.10 rad/s), then stops; no camera, grasp or lift cycle.
  See [lab progress](../docs/ggx_lab_progress_zh.md).

## Saved Lab Cube Baseline

- `run_iterative_cube.py`: saved trial_04 configuration; offline preview by default.
  `--execute` prints safety advice and waits for Enter confirmation, then calls the existing
  single-cycle runner (open/reset/scan/original Plan/lift/hold). No automatic
  release or retry. See [Chinese operating instructions](../docs/iterative_cube_lab_zh.md).

- `run_ggx_cube_inference.py`: offline official GGX Inspire-Hand inference for a
  centered 40 mm cube, using an independent GGX Python environment. Saves seeded
  input clouds, candidate YAML, revisions and summaries; no hardware control.
  Alternatively, `--point-cloud POINTS.npy --input-frame ur_base` consumes a saved
  measured cloud in metres and exports poses in that same frame, restoring the
  input mean once. Never feed these base-frame poses to mesh-frame comparison
  scripts that add an object-center translation.
- `prepare_ggx_tabletop_cloud.py`: offline partial RGB-D crop from an explicitly
  reviewed upright-box top component and supplied SHORT/LONG/HEIGHT dimensions.
  Explicit `--crop-mode foreground` instead selects the saved foreground component
  for non-flat opaque objects with a distinct long axis. Removes isolated points
  only (four neighbours within 5 mm, >=80% and >=100 points retained); never fits
  or rescales points to supplied dimensions. Crop v2 separates operator dimensions,
  observed footprint and enclosing box proxy used by BOTH simulation and CB3.
  The proxy may reject feasible shoulder/neck grasps; it is not an exact shape model.
  Opt-in `--freeze-reference-yaw` accepts near-square initial footprints for
  XY-only repetitions, explicitly recording ambiguous yaw; no proxy shrinking.
  Foreground real handoff requires fixed-reference RGB-D checks, not flat-top matching.
  Reuses camera calibration/deprojection; saves only measured points in UR base,
  colors, overlay and provenance. No filling, hidden box synthesis or universal
  segmentation claim; no robot target or hardware access.
- `preview_ggx_camera_candidates.py`: camera overlays and 3D plots of the highest
  confidence GGX poses using official open-hand descriptor points. Checks source
  hashes and base-frame convention. No collision filter, hand commands or IK;
  score is not a measured success probability.
- `preview_tabletop_target.py`: color-independent saved-depth foreground preview
  with visible extents and explicit fixture exclusions. Never emits an executable
  target; ambiguous foreground requires review. Optional `--background` suppresses
  unchanged depth with measured-noise thresholds; separate `top_patch` estimates
  assume a near-horizontal top and retain raw extents for comparison.
  Default `--background-mode supported-surface` recovers only measured current
  foreground connected across unknown background, with minimum change support;
  `--background-mode pixels` reproduces the original strict pixel gate. Missing
  current depth is never filled; region-boundary and ambiguity checks remain.
  Default `configs/tabletop_vision_lab.yaml` defines a perception-only image ROI,
  independent of robot reachability. Explicit `--config configs/ur5_iterative_cube_lab.yaml`
  retains the original small target XY gate for reproducing older previews.
  See [lab progress and limitations](../docs/ggx_lab_progress_zh.md).
- `capture_tabletop_background.py`: camera-only empty-table RGB-D burst capture;
  saves every raw frame, a supported-depth median, per-pixel MAD/counts and
  summaries. `--scene object` labels a target-validation burst instead of an empty
  background. Does not connect to the robot/hand or change laser settings.
  Object-only `--rgb-only-quality` permits missing depth for RGB XY tracking;
  initial GGX clouds and empty backgrounds still require measured depth.
- `preview_tabletop_grasp_center.py`: offline original Plan center/closing-axis
  overlay on saved RGB for an explicitly reviewed tabletop component. Uses the
  existing antipodal center and calibrated projection; keeps workspace rejection,
  reports solver width residual, and never emits an executable target.

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
- `analyze_ur5_camera_colored_target.py`: offline independent-height check for
  a colored target in a saved aligned RGB-D frame. It rejects farther depth-hole
  background, reports the target top center in the UR base frame, writes a
  `summary.csv`, and can optionally quality-gate an independently measured XY.
- `replay_ur5_camera_calibration.py`: dry-run by default. With the explicit
  `--execute-motion` flag and exact interactive confirmation, it replays a
  saved TCP calibration path using low-speed `moveL`, captures a new dataset,
  recalibrates, and reports camera translation/rotation drift. It checks the
  active TCP offset, start proximity, scalar workspace bounds, protective-stop
  state, target error, and board detection; the operator must still clear and
  watch every recorded straight segment.

## Grasp safety preflight

- `run_ggx_review.py`: camera capture (explicit `--capture-live`) or offline RGB-D
  → known-box crop (or explicit `--crop-mode foreground`) → GGX inference → confidence-ordered per-candidate simulation
  → interactive Enter/next/replay/invalidate/quit review. Offline visual approval
  is saved with hashes. With `--execute`, candidates are NOT visually approved
  first: the CB3/controller path is checked before its execution video is shown.
  Enter then authorizes that exact path; `n` moves on, `p` replays, `r/q` stop.
  A distinct unapproved proposal schema prevents fabricated prior approval.
  UR5e screening remains numerical only; no redundant nominal video is rendered.
  The RTDE-only grant is once per invocation, not permission to move a candidate.
  Final scene/start/TCP/file checks and live safety/deviation monitors remain;
  no IK/geometry/video is recomputed after the execution-video Enter.
  This checked execution video cannot be disabled with `--no-real-video`.
  Optional `--reset-first --capture-live`
  separately confirms open → 80 mm vertical retreat → saved scan-pose return,
  disconnects, then waits for placement Enter before the NEW camera capture.
  Reset uses the same continuous primitive checker/backend, not old object poses;
  object support/clear reset paths require operator confirmation. No implicit reset.
  Reset retains the intentionally slow 0.01 m/s / 0.10 rad/s motion profile;
  per-stage timing separates pre-motion computation from movement duration.
  The shared reset/grasp preflight logs per-method controller query counts/times
  in `controller_preflight_timing`; exact-input calibrated FK is reused only
  within that stationary preflight. IK, safety checks and geometry are not cached.
  Lab `controller_preflight_mode: keypoints` reduces remote safety/FK queries to
  first/middle/last scheduled samples per phase; every Cartesian IK query and local
  geometry sample remains. Interior TCP metadata is explicitly a nominal estimate.
  This reduces remote verification coverage, not a claim of equivalent full
  controller certification. `dense` restores per-sample remote checks (also the
  default for older configs). Commands cannot execute under a different check mode.
  Candidate simulation now reuses one fixed model in-process, lazily in confidence
  order. Rejected candidates retain reports/logs but render no unused videos;
  in offline review each passing video is rendered once, then copied for the
  convenience filename. Executing review instead renders the checked CB3 path.
  Initialization/evaluation/render timings are recorded; approval/cancel stops
  before computing the next rank. Collision checks and motion settings are unchanged.
  If the considered batch has zero automatic passes, default `--fallback-review`
  lazily shows object-only rejects with complete IK and desk-clear sampled paths,
  in confidence order. It reuses computed trajectories; it does not rerun IK.
  Videos retain collision warnings; Enter records a visual-only opinion, never
  real-motion approval (even with `--execute`). Original rejection reports stay
  unchanged. Bad IK/desk paths remain excluded; operator vetoes of passing
  candidates do not trigger fallback. `--no-fallback-review` disables this mode.
  A separately confirmed `--reset-first` still operates before capture as above.
  See `docs/ggx_interactive_review_zh.md`.
- `run_ggx_real.py`: offline approved-candidate CB3 preparation by default;
  `--checked-review` moves all path calculations before the final CB3 video
  approval (Enter executes, n vetoes). `--candidate` accepts only a separately
  recorded unapproved automatic pass and requires that checked-video gate.
  Neither an unreviewed proposal nor a manual collision opinion is motion consent.
  `--preflight-only` explicitly enables camera/UR reads and separately confirmed
  RTDE script startup/IK checks. `--execute` additionally requires final Enter
  before hand writes or motion. Default approach-only; no automatic reset/release.
  Executes one MoveJ pregrasp + one MoveL approach (+ one MoveL lift for full
  grasps), not one stop per collision sample. Numerical CB3/controller/geometry
  checks are mandatory, including closure at the controller-resolved wrist pose.
  Lab `preview_videos: false` skips the two additional CB3 videos, not those checks
  or the initial GGX candidate video. `--video` (review launcher: `--real-video`)
  restores them. `execution.json` and `execution_precheck.json` stay hash-bound.
  Lab raw hand speed/force are both 1000 at the operator's request, including reset
  opening; `--allow-high-force` and the separate Enter confirmations are required.
  Arm speeds and open-loop timing are unchanged. These are protocol settings,
  not measured physical speed/force; no hand feedback is available.
  Lab scene verification uses fixed-reference RGB appearance/edges plus spatial
  valid-depth consistency; fresh capture is retained without re-segmenting a
  depth-fragmented approved target. Missing/changed evidence still rejects;
  this is not an obstacle certificate or a scene-check bypass.
  Lab desk uses an operator-confirmed conservative finite envelope; hardware
  execution is not yet validated. Reference bounds do not follow later objects.
- `preflight_ggx_ur5_table.py`: simulation-only GraspGen-X candidate screening
  against an adjustable desk and pre-closure robot-object contact, with YAML setup reuse, annotated review videos,
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
