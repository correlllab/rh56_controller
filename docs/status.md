# Current Project Status

This page documents the current RH56DFX code-release status for paper-v2
reproducibility work. It intentionally separates simulation-only workflows from
features that need a readable and writable physical RH56 hand.

## Works without hardware

- Profile-based sim-hand installation and import validation.
- Floating RH56 MuJoCo grasp planner.
- Analytical width-to-grasp closure solver.
- Mink comparison planner when the `sim-hand` profile is installed.
- Paper-facing planner and hybrid-margin sweeps that write results under
  `artifacts/`.
- H1-2 sim-only viewer when the optional H1-2 profile dependencies are
  installed.

## Requires working RH56 read/write

- Real hand mirroring.
- Real intrinsic force streaming.
- Real peg-in-hole execution.
- Real grasp execution and object trials.
- Any hardware validation of force thresholds, switch margins, or contact-speed
  sweeps.

## Currently blocked

- Reliable RX/TX read/write on the available RH56 hands.
- Paper-v2 real-hardware refresh experiments.
- Full ROS2 high-level grasp control.
- Real H1-2 plus RH56 integrated manipulation.

## Paper-v2 priority

The main branch of work should remain simulation-only and replay-friendly until
the hardware issue is resolved. New scripts should document assumptions, write
`summary.csv` or a schema file, and avoid requiring ROS2, Unitree SDKs, or real
robot access.
