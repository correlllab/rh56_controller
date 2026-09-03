# GGX + UR5e + RH56 first test-day checklist

This checklist separates simulation approval, human approval, and the later
real-robot trial. The preflight script is simulation-only and cannot authorize
or command hardware.

## Freeze and record the setup

1. Confirm the robot variant, RH56 mount, TCP definition, desk, object, and
   camera have not moved since calibration.
2. Measure desk-top Z and object pose in the UR base frame.
3. Read and record all six starting arm joint angles rather than relying on the
   example values.
4. Save the camera serial number, calibration file, held-out translation error,
   and held-out rotation error.
5. Copy `configs/ggx_ur5_table_lab.example.yaml`, enter the measured setup, and
   keep that copy with the trial outputs.

## Simulation and human gate

1. Run the preflight with the saved setup YAML.
2. Confirm `summary.json` identifies the intended model, candidate file, config,
   and their hashes.
3. Review every `review_rank_NNN.mp4` in GGX confidence order. The overlay shows
   phase, hand closure, current/running-minimum desk clearance, nearest robot
   geometry, and the final automatic result.
4. In `human_review.csv`, mark PASS or VETO. A veto must name a concrete risk
   omitted by the simplified model, such as cable routing, camera mount,
   singular posture, self-collision, or an unreachable real starting state.
5. Select the highest-confidence automatic PASS that has not been vetoed. Do
   not select a lower rank merely because it appears more likely to grasp.

## Staged real-robot verification

Follow the lab's approved robot-safety procedure, with emergency stop and
reduced-speed controls available. Before any object contact, first verify the
selected path well above the desk, then stop at pre-grasp, then approach slowly.
The operator remains responsible for stopping motion if the real setup differs
from simulation. The current preflight outputs must not be streamed directly to
the robot.

Record each attempt in `docs/ggx_first_test_record_template.csv`, including
automatic result, human decision, elevated dry-run result, grasp/lift outcome,
and the first observed failure stage. Preserve failed trials as data rather than
silently retuning or replacing the GGX candidate.
