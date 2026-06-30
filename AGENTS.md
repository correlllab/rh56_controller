# AGENTS.md

## Project context

This repository contains control, simulation, grasp planning, and experiment tooling for the Inspire RH56DFX dexterous hand. The current work is being revised after an IROS rejection. The priority is to make the RH56DFX paper v2 more reproducible, better justified, and more clearly framed as a scientific contribution.

The RH56 physical hands are currently not reliably readable or writable. Unless explicitly requested, do not assume hardware access.

## Current priority

Focus on:

1. Simulation-only reproducibility.
2. Paper-facing experiment scripts.
3. Clear documentation of current capabilities and blocked features.
4. Tests that do not require hardware.
5. Safe hardware diagnostic tools only when working on the `hardware-debug` branch.

Do not prioritize:

- Real H1-2 manipulation.
- Vision integration.
- Standing policy, FAME, or MPC.
- Spray bottle, mouse, bottle-cap, or other new dexterous demos.
- Full ROS2 rewrite of high-level grasp control.
- Custom board design unless direct USB-RS485 works and the intermediate board path fails.

## Repository rules

- Do not require RH56 hardware for sim-only commands or tests.
- Do not require ROS2, Unitree SDK, or real robot dependencies for default tests.
- Optional dependencies should fail gracefully or skip tests with clear messages.
- Keep large artifacts out of Git.
- Write generated outputs under `artifacts/` or another documented local artifact directory.
- Do not commit raw experiment logs, videos, PDFs, local workspaces, or large generated plots unless the user explicitly asks.
- Prefer small, reviewable changes.
- Show diffs and explain assumptions after modifying files.

## Coding expectations

- Use existing repository APIs before inventing new kinematic or controller formulas.
- Add `argparse` CLIs for experiment scripts.
- Every new script should support `--help`.
- Every experiment script should write a `summary.csv` or a clearly named schema file.
- Every experiment script should print the assumptions it used or write them to metadata.
- Hardware scripts must be read-only by default.
- Any write or motion command must require an explicit flag and a clear confirmation mechanism.

## Test expectations

For sim-only changes, try to run:

```bash
python tools/check_profile_imports.py --profile sim-hand
pytest -q
```

If the environment is missing optional dependencies, report that clearly instead of hiding the failure.

## Paper-v2 framing

The next paper should emphasize these claims:

1. RH56DFX raw proprioceptive and force-like signals can be characterized into a usable control basis within measured operating ranges.
2. RH56DFX contact overshoot is governed by contact-phase speed and command-to-sensing latency, enabling latency-aware anticipatory switching.
3. RH56DFX coupled grasp geometry admits a reduced one-dimensional width-to-grasp solution with smooth reachable configurations.
4. The resulting planner and controller improve grasp success over naive closure and support interpretable failure analysis.

Avoid wording that makes empirical parameters sound arbitrary. When discussing values such as contact speed, switch margin, or force thresholds, connect them to measurements, sweeps, sensitivity analysis, or explicit assumptions.

## Safety and hardware caution

When touching hardware-related code:

- Prefer read-only diagnostics.
- Log raw TX and RX bytes.
- Do not move the hand by default.
- Do not send high-force or high-speed commands in diagnostic scripts.
- Keep hardware-debug changes isolated from paper-v2 simulation work unless explicitly requested.
