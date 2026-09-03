# RH56DFX Paper V2 and Codebase Handoff for Codex

## Current handoff snapshot: 2026-06-30

Use this section first. Older sections below are historical planning context.

### Current branch and intent

- Repo: `correlllab/rh56_controller`
- Active branch when this handoff was written: `main`
- Current focus: RH56DFX paper v2, simulation-only reproducibility, and paper-facing evidence for the analytical grasp planner.
- Hardware assumption: no reliable RH56 hardware read/write; do not start hardware experiments unless explicitly requested.

### What changed recently

- Added paper-v2 status and scope docs:
  - `docs/status.md`
  - `docs/not_now.md`
  - `docs/reproduce_paper_figures.md`
  - `docs/paper_v2_methods.md`
  - `docs/figure_review_audit.md`
- Added paper-facing scripts:
  - `tools/run_planner_sweep.py`
  - `tools/run_hybrid_margin_sweep.py`
  - `tools/replay_force_thresholds.py`
  - `tools/run_analytical_grasp_volume.py`
  - `tools/generate_paper_figures.py`
- Added capsule hand proxy work:
  - `rh56_controller/capsule_hand_proxy.py`
  - `rh56_controller/paper_v2_objects.py`
  - `tools/verify_capsule_hand_proxy.py`
  - `tools/capsule_path_gui.py`
  - `tests/unit/test_capsule_hand_proxy.py`
- Added `Makefile` targets for sim checks and paper figure scripts.
- Added `PyYAML` as a base dependency because `grasp_viz --help` imports YAML.

### Current interactive MuJoCo demo

Use:

```bash
.venv312/bin/python tools/capsule_path_gui.py
```

Important controls:

- `W/S`, `A/D`, `R/F`: move the hand-base start pose.
- `J/L`: rotate target yaw.
- `=/-`: change keyboard step size.
- `Enter` or `V`: run the swept-capsule validity check for the current pose.
- `Space`: animate only the latest checked valid path.
- `--auto-check`: restore continuous recomputation, but it is slower.

Current modeling assumptions:

- Default object is `debug_40mm_cube`, a 40 mm cube.
- The cube is placed on the ground plane, not centered on it:
  `aabb_center = [0, 0, 20] mm`, bottom face at `z = 0`.
- Default `path_hand_shape=open` means finger flexion qpos values are `0`,
  while `thumb_yaw` is set to `CTRL_MAX["thumb_yaw"]` so the thumb is fully
  rotated toward the real-hand raw `0` side.
- Collision checking uses a swept capsule centerline vs AABB distance over each
  straight-line path interval, not endpoint-only sampling.
- The GUI defaults to manual checking because the swept check is too expensive
  for smooth pose adjustment.

### Generated artifacts to keep local

Do not commit generated artifacts unless the user explicitly asks. Current
useful local outputs include:

- `artifacts/capsule_proxy_demo/capsule_proxy_demo.png`
- `artifacts/capsule_proxy_demo/summary.csv`
- `artifacts/capsule_proxy_demo/path_samples.csv`
- `artifacts/planner_sweep/summary.csv`
- `artifacts/hybrid_margin_sweep/summary.csv`
- `artifacts/threshold_replay/summary.csv`

The repository now ignores `artifacts/`.

### Validation commands that passed

```bash
.venv312/bin/python -m pytest -q
.venv312/bin/python tools/check_profile_imports.py --profile sim-hand
.venv312/bin/python tools/capsule_path_gui.py --print-initial-and-exit
.venv312/bin/python tools/verify_capsule_hand_proxy.py --out artifacts/capsule_proxy_demo
git diff --check
```

Notes:

- `uv run ...` may fail in Codex sandbox because `~/.cache/uv` can be read-only.
  Use `.venv312/bin/python ...` as the local fallback.
- The GUI itself should be visually checked from the user's normal desktop
  terminal if DISPLAY/GLFW is unavailable in Codex.

### Files intentionally not pushed by default

These are local context or user-report files and should not be committed unless
the user explicitly asks:

- `Review15924.txt`
- `Review176.txt`
- `Review17678.txt`
- `docs/weekly_report_2026-06-26.*`
- `docs/weekly_update_brief_2026-06-26.*`
- LibreOffice lock files such as `docs/.~lock.*#`

### Next likely task

Refine the paper-facing no-go volume figure after the first capsule pass:

1. Run a coarse full sweep with `tools/run_analytical_grasp_volume.py`.
2. Inspect whether the no-go slices are visually interpretable and whether the
   `most_common_blocker` labels match the MuJoCo GUI intuition.
3. Tune grid/yaw/path sampling only after the coarse result is understandable.
4. Decide whether to add sparse MuJoCo mesh-collision validation points before
   using the figure in paper v2.

Keep this line simulation-only and avoid hardware/H1-2 scope.

---

Owner: Xuan Tan
Project: `correlllab/rh56_controller`
Purpose: transfer the current post-IROS-rejection plan into VS Code Codex so it can help execute repo cleanup, simulation experiments, hardware diagnosis support scripts, and paper-facing artifacts.

## 0. How to use this file with Codex

Recommended setup:

1. Open `rh56_controller` in VS Code.
2. Add this file to the repository root as `CODEX_HANDOFF_RH56.md`, or put it in `docs/` and keep it open when prompting Codex.
3. Add the companion `AGENTS_RH56_TEMPLATE.md` file to the repo root as `AGENTS.md`.
4. In the Codex panel, start with one focused task at a time. Do not ask Codex to do the entire project in one prompt.
5. Use the prompts in Section 10 as copy-paste starting points.

Suggested first Codex prompt:

```text
Read AGENTS.md and CODEX_HANDOFF_RH56.md. Do not change hardware-facing code yet. First inspect the repository and propose a minimal PR plan for Phase 1: repo status docs, Makefile commands, and sim-hand/sim-h12 validation. Then implement only the first safe step.
```

## 1. Current situation

The IROS paper was rejected. The core technical work still has value, but the next version needs to better explain the scientific contribution, justify empirical design choices, and provide more rigorous parameter validation.

The RH56 hands are currently blocked by a read/write issue. Because the hand is not reliably usable, the next plan should avoid expanding into real H1-2 tasks, vision, standing policy, or dexterous demos. Focus on work that remains useful even without the physical hand.

Current project page status to keep in mind:

- Development on the RH56 hands is paused because RX/TX read/write has not been resolved.
- ROS2 support is only basic hand control.
- Higher-level antipodal grasping and related control still live in Python/Tkinter.
- The public repo contains control, simulation, grasp planning, install profiles, and calibration or visualization utilities.

Primary near-term goal:

> Convert the existing work from a functional engineering system into a reproducible, explainable, and review-resistant RH56DFX planning and control study.

## 2. What not to prioritize now

Do not make these the main line of work for paper v2:

- Real H1-2 integration.
- Vision integration.
- Standing policy, FAME, or MPC.
- Bottle-cap screwing.
- Spray-bottle trigger pulling.
- Mouse manipulation.
- Large new dexterous demonstrations.
- Full ROS2 rewrite of all high-level hand logic.
- New custom board design unless direct USB-RS485 works but the Unitree or intermediate board path fails.

These can go into `docs/not_now.md` or GitHub issues labeled `not-now`.

## 3. Revised project priorities

Priority order:

1. Repo reproducibility for simulation-only users.
2. Paper-facing simulation experiments and data replays.
3. Hardware diagnosis scripts and logs, time-boxed.
4. Paper narrative rewrite around scientific claims.
5. Optional real experiments only if the hand becomes readable and writable again.

The key idea is to shrink scope, not expand it.

## 4. Branch and file organization

Recommended branches:

```text
main              stable branch, only code that installs and runs
paper-v2          next paper experiments, figures, and docs
hardware-debug    RH56 read/write diagnostic scripts and logs
dev/h12           H1-2 work, not paper-critical right now
```

Recommended new files:

```text
AGENTS.md
CODEX_HANDOFF_RH56.md
docs/status.md
docs/not_now.md
docs/reproduce_paper_figures.md
docs/hardware_debug_plan.md
hardware_debug/README.md
tools/run_planner_sweep.py
tools/run_hybrid_margin_sweep.py
tools/replay_force_thresholds.py
tools/run_grasp_style_coverage.py
tools/run_h12_reachability_sweep.py
tools/rh56_bus_scan.py
```

Recommended local artifact folders, not tracked by Git unless they contain small summary CSVs or plots intentionally committed:

```text
artifacts/planner_sweep/
artifacts/hybrid_margin_sweep/
artifacts/threshold_replay/
artifacts/grasp_style_coverage/
artifacts/h12_reachability/
hardware_debug/logs/
```

## 5. Phase 1: repo cleanup and reproducibility

Goal:

> A new user can clone the repo and run the simulation-only RH56 planner without the physical hand.

Codex should start here before modifying experiment code.

### 5.1 Add `docs/status.md`

Content should be direct and honest:

```md
# Current Project Status

## Works without hardware
- Floating RH56 MuJoCo grasp planner.
- Analytical width-to-grasp solver.
- Planner sweep experiments.
- Force or grasp quality visualization in simulation, where supported.
- H1-2 sim-only viewer with PINK IK, if dependencies are installed.

## Requires working RH56 read/write
- Real hand mirroring.
- Real force sensor streaming.
- Peg-in-hole X-mode.
- Real grasp execution.

## Currently blocked
- Reliable RX/TX read/write on RH56 hands.
- Full ROS2 high-level grasp control.
- Real H1-2 plus RH56 integrated manipulation.
```

### 5.2 Add `docs/not_now.md`

Purpose: prevent scope creep.

Include:

```md
# Not Now

These tasks are deferred until the RH56 read/write issue is resolved or until paper v2 experiments are complete.

- Vision integration.
- Real H1-2 manipulation.
- Standing policy.
- FAME or MPC.
- Spray bottle.
- Mouse manipulation.
- Bottle-cap screwing.
- Full high-level ROS2 port.
```

### 5.3 Add a `Makefile`

Suggested targets:

```makefile
.PHONY: setup-sim check-sim setup-h12 check-h12 test run-planner figures

setup-sim:
	tools/setup_uv_env.sh --profile sim-hand --python 3.12 --env .venv312

check-sim:
	. .venv312/bin/activate && python tools/check_profile_imports.py --profile sim-hand

setup-h12:
	tools/setup_uv_env.sh --profile sim-h12 --python 3.12 --env .venv312-h12

check-h12:
	. .venv312-h12/bin/activate && python tools/check_profile_imports.py --profile sim-h12

test:
	. .venv312/bin/activate && pytest -q

run-planner:
	. .venv312/bin/activate && uv run python -m rh56_controller.grasp_viz

figures:
	. .venv312/bin/activate && python tools/generate_paper_figures.py
```

Do not assume `tools/generate_paper_figures.py` exists. If it does not exist, Codex should create a stub that prints which figure scripts are missing, rather than silently failing.

### 5.4 Add minimal tests

Start with tests that do not require hardware.

Suggested tests:

```text
tests/test_grasp_geometry.py
  test_width_solver_reaches_requested_width
  test_width_solver_rejects_unreachable_widths
  test_width_sweep_is_stable

tests/test_force_mapping.py
  test_raw_to_newtons_valid_range
  test_raw_to_newtons_no_silent_extrapolation

tests/test_cli_smoke.py
  test_import_grasp_viz
  test_import_grasp_geometry
  test_import_mujoco_bridge_if_available
```

Acceptance criteria:

```bash
make check-sim
make test
```

Both should pass on a machine with the sim profile installed. If optional dependencies are missing, tests should skip cleanly with a clear reason.

## 6. Phase 2: paper-facing simulation and replay experiments

The purpose is not to add fancy demos. The purpose is to justify design choices that reviewers may currently see as heuristics.

### 6.1 Experiment A: planner sweep

Question:

> Does the analytical width-to-grasp planner produce smooth, reachable, and reliable grasp configurations across object widths and grasp modes?

Script:

```text
tools/run_planner_sweep.py
```

Example command:

```bash
python tools/run_planner_sweep.py \
  --modes line plane cylinder \
  --width-min 5 \
  --width-max 115 \
  --width-step 1 \
  --heights -40 -20 0 20 40 \
  --out artifacts/planner_sweep/
```

Output CSV schema:

```csv
mode,width_mm,height_mm,success,solve_time_ms,tip_error_mm,coplanarity_error_mm,tilt_deg,joint_margin,collision,notes
```

Plots:

- Reachable width map.
- Tip error vs width.
- Solve time histogram.
- Failure region visualization.

Paper claim supported:

> RH56DFX coupled grasp geometry can be reduced to a one-dimensional width-to-grasp problem with smooth reachable configurations.

Acceptance criteria:

- Script runs without hardware.
- Produces one summary CSV.
- Produces at least one figure.
- Handles unreachable widths without crashing.

### 6.2 Experiment B: hybrid speed and switch-margin sweep

Question:

> Why use low contact speed and a 25-unit anticipatory switch margin?

Script:

```text
tools/run_hybrid_margin_sweep.py
```

Example command:

```bash
python tools/run_hybrid_margin_sweep.py \
  --v-fast 1000 \
  --v-contact-list 10 25 50 100 \
  --margin-list 0 5 10 15 20 25 30 40 50 \
  --latency-model empirical \
  --out artifacts/hybrid_margin_sweep/
```

Minimum model:

```text
Inputs:
- fast speed
- contact speed
- switch margin
- latency distribution or fixed latency estimate
- contact onset uncertainty estimate

Outputs:
- probability of entering low-speed mode before contact
- predicted post-contact motion
- overshoot proxy
- completion time proxy
```

Plots:

- Margin vs probability of pre-contact slow mode.
- Contact speed vs overshoot proxy.
- Time vs overshoot Pareto curve.
- Heatmap of margin and contact speed.

Paper claim supported:

> The hybrid controller is latency-aware. Its switch margin is a measured uncertainty buffer, not a magic number.

Acceptance criteria:

- Script can run using only stored constants or existing logs.
- The 25-unit margin is shown as part of a robust region, not necessarily a single optimal value.
- The output explicitly separates assumptions from measured values.

### 6.3 Experiment C: force threshold replay

Question:

> Are force thresholds robust, or are they brittle hand-tuned constants?

Script:

```text
tools/replay_force_thresholds.py
```

Example command:

```bash
python tools/replay_force_thresholds.py \
  --logs experiment_data/peg_in_hole/*.csv \
  --contact-spike-list 25 50 75 100 125 150 \
  --lateral-spike-list 25 50 75 100 125 150 \
  --window-list 0.1 0.25 0.5 0.75 1.0 \
  --out artifacts/threshold_replay/
```

If logs are not present in Git, script should print a clear message and write a template file:

```text
artifacts/threshold_replay/expected_log_schema.csv
```

Suggested expected log schema:

```csv
trial_id,time_s,index_force_raw,middle_force_raw,ring_force_raw,pinky_force_raw,thumb_force_raw,phase,label_success,label_release_time_s
```

Plots:

- Contact spike threshold vs false positive and false negative rates.
- Lateral spike threshold vs release detection accuracy.
- Moving average window vs detection delay.
- Robust plateau plot.

Paper claim supported:

> Finger-force release logic is robust over a parameter range and performs better than wrist-force-only triggering within the measured task distribution.

Acceptance criteria:

- Script does not require hardware.
- It either runs on logs or writes a clear schema for logs that need to be supplied.
- It produces a summary table even if no plots are generated yet.

### 6.4 Experiment D: grasp style coverage and blocked zones

Question:

> Which grasp styles are geometrically valid for which object dimensions, and where do blocked zones occur?

Script:

```text
tools/run_grasp_style_coverage.py
```

Example command:

```bash
python tools/run_grasp_style_coverage.py \
  --object-widths 10 20 30 40 50 60 70 80 90 100 \
  --object-lengths 20 40 60 80 100 120 160 200 \
  --object-heights 20 40 60 80 \
  --modes line plane cylinder \
  --friction 0.4 0.6 0.8 1.0 \
  --pose-noise-mm 0 5 10 \
  --out artifacts/grasp_style_coverage/
```

Output CSV schema:

```csv
object_width_mm,object_length_mm,object_height_mm,mode,friction,pose_noise_mm,success,contacts,gws_epsilon,external_wrench_margin,collision,blocked_zone,tip_error_mm,notes
```

Plots:

- Coverage map by grasp mode.
- Blocked zones.
- Grasp quality distribution.
- Robustness under pose noise.

Paper claim supported:

> Grasp style selection can be tied to geometric coverage and contact quality, not only intuition.

Acceptance criteria:

- Runs in sim-only mode.
- Does not require H1-2 or real RH56.
- Produces a clear failure reason for invalid cases.

### 6.5 Optional Experiment E: H1-2 reachability smoke test

This is optional and should not become the main paper contribution.

Question:

> If valid RH56 grasp poses are placed on H1-2, which are reachable by the H1-2 arm in simulation?

Script:

```text
tools/run_h12_reachability_sweep.py
```

Example command:

```bash
python tools/run_h12_reachability_sweep.py \
  --targets artifacts/grasp_style_coverage/valid_targets.csv \
  --right-arm \
  --out artifacts/h12_reachability/
```

Output CSV schema:

```csv
target_x,target_y,target_z,width_mm,mode,ik_success,joint_limit_margin,self_collision,table_collision,solve_time_ms,notes
```

Paper claim supported:

> Optional system integration is plausible, but it is not the central claim of the paper.

Acceptance criteria:

- Sim-only.
- No ROS2, no Unitree SDK, no real robot.
- Clearly marked optional.

## 7. Phase 3: hardware debug, time-boxed

Goal:

> Determine whether the RH56 read/write failure is in the hand, the wiring, the power path, the RS485 path, or the intermediate board path.

Do not let this become an open-ended board design project.

### 7.1 Diagnostic setup

Use direct setup first:

```text
PC -> USB-RS485 adapter -> RH56 hand -> bench 24 V supply
```

Do not start through H1-2 or Unitree boards.

Prepare:

```text
bench supply with current limit
multimeter
USB-RS485 adapter, known-good
logic analyzer or oscilloscope if available
spare cable or connector breakout
camera
hardware_debug log file
```

### 7.2 Add `hardware_debug/README.md`

Suggested structure:

```md
# RH56 Hardware Debug Log

## Setup
- Hand serial number:
- Power supply:
- Current limit:
- RS485 adapter:
- Cable path:
- Date:

## Step 1: Power only
- Voltage:
- Idle current:
- Any heat:
- Result:

## Step 2: RS485 physical signal
- TX visible at adapter:
- TX visible at hand connector:
- RX response visible:
- A/B polarity tested:
- Result:

## Step 3: Read-only bus scan
- Baudrate:
- IDs scanned:
- Registers read:
- Raw TX/RX:
- Result:

## Step 4: Small write test, only if read works
- Command:
- Expected motion:
- Observed motion:
- Result:
```

### 7.3 Add `tools/rh56_bus_scan.py`

Purpose:

- Minimal read-only RS485 script.
- Print raw TX and RX bytes.
- Sweep possible hand IDs.
- Avoid moving the hand.
- Avoid using high-level controller abstractions at first.

Example command:

```bash
python tools/rh56_bus_scan.py --port /dev/ttyUSB0 --baud 115200 --ids 1 2 3 4
```

Expected behavior:

```text
ID 1: no response
ID 2: response bytes ... parsed status ...
ID 3: CRC error ...
ID 4: no response
```

Acceptance criteria:

- Does not send motion commands by default.
- Has `--write-test` or equivalent gated behind explicit confirmation.
- Logs raw command and response bytes.
- Exits safely when serial port is unavailable.

### 7.4 Hardware decision tree

Use this logic:

```text
Direct USB-RS485 works, Unitree/mainboard path fails:
  likely issue is intermediate board, wiring, powerbuck, direction control, or shared bus.
  Custom board or microcontroller bridge may be worth discussing.

Direct USB-RS485 fails:
  likely issue is hand, connector, power, ID, protocol, or internal controller.
  Do not start custom board design yet.

TX reaches adapter but not hand connector:
  cable or board path issue.

TX reaches hand connector but no response:
  check power, A/B polarity, ID, baudrate, protocol, hand health.

No TX from adapter:
  software, permissions, adapter, or port issue.
```

## 8. If hardware starts working again

Do not jump to H1-2 or dexterous demos. Do three small, paper-critical real experiments.

### R1: switch margin sweep

```text
margins: 0, 10, 20, 25, 40
v_fast: 1000
v_contact: 25
object or setup: foam block or force gauge contact fixture
n: 10 per condition
metrics: peak force, overshoot, contact time, total time
```

Purpose:

> Show that 25 is in a robust region for anticipatory switching.

### R2: contact speed sweep

```text
v_contact: 10, 25, 50, 100
margin: 25
n: 10 per condition
metrics: peak force overshoot, completion time
```

Purpose:

> Show the speed and overshoot tradeoff.

### R3: force threshold validation

```text
CONTACT_SPIKE: 50, 75, 100
LATERAL_SPIKE: 50, 75, 100
n: 5 or 10 peg-in-hole attempts per setting
```

Purpose:

> Show threshold robustness and reduce the appearance of hand-tuned constants.

## 9. Paper narrative rewrite

The paper should not be framed as only “we improved the RH56 controller.” It should be framed as a study of how to turn a commercial underactuated hand into a reproducible research platform through characterization, analytical planning, and latency-aware force control.

### 9.1 Proposed scientific claims

Claim 1:

> RH56DFX raw proprioceptive and force-like signals can be characterized into a usable control basis within measured operating ranges.

Claim 2:

> RH56DFX contact overshoot is governed by contact-phase speed and command-to-sensing latency, enabling latency-aware anticipatory switching.

Claim 3:

> RH56DFX coupled grasp geometry admits a reduced one-dimensional width-to-grasp solution with smooth reachable configurations.

Claim 4:

> The resulting planner and controller improve grasp success over naive closure and support interpretable failure analysis.

### 9.2 Replace weak contribution bullets

Avoid:

```text
We improve grasping performance of RH56DFX.
```

Use:

```text
We identify and quantify the RH56DFX hardware limitations that dominate contact behavior, including sensor latency, contact overshoot, and uncalibrated force feedback.
```

```text
We formulate RH56DFX antipodal planning as a width-parameterized scalar solve, allowing real-time grasp synthesis for the coupled underactuated mechanism.
```

```text
We derive and validate a latency-aware hybrid speed-force controller whose switching margin is selected from measured latency and contact-onset uncertainty.
```

### 9.3 Claim-to-evidence table for the paper

| Claim | Evidence | Baseline or ablation | Output figure/table |
|---|---|---|---|
| Contact speed controls overshoot | speed sweep | constant high speed vs constant low speed | overshoot vs speed |
| Switch margin is not arbitrary | margin sweep | margin 0, 10, 25, 40 | probability and Pareto plot |
| Analytical planner is reliable | planner sweep | naive width closure or QP where available | reachable width map |
| Finger force helps release | peg-in-hole replay or real tests | wrist-force trigger | threshold sensitivity |
| Grasp style selection is justified | sim coverage | line vs plane vs cylinder | coverage map |

## 10. Copy-paste Codex prompts

Use one at a time.

### Prompt 1: repo inspection

```text
Read AGENTS.md and CODEX_HANDOFF_RH56.md. Inspect the repository. Do not modify files yet. Summarize the current package structure, likely simulation entry points, existing tests, missing tests, and the safest first PR. Focus on sim-only reproducibility. Do not touch hardware-facing code.
```

### Prompt 2: status docs

```text
Create docs/status.md and docs/not_now.md based on CODEX_HANDOFF_RH56.md. Keep the language factual and concise. Do not overpromise code release status. After writing, show the diff and explain any assumptions.
```

### Prompt 3: Makefile

```text
Add a Makefile with setup-sim, check-sim, setup-h12, check-h12, test, run-planner, and figures targets. Use existing scripts where possible. If a target depends on a missing script, create a safe stub that prints a clear message. Do not change package dependencies unless necessary.
```

### Prompt 4: smoke tests

```text
Add minimal pytest smoke tests for sim-only functionality. Tests must not require hardware, ROS2, Unitree SDK, or a real RH56 hand. Optional dependencies should be skipped with clear pytest skip messages. Run pytest if possible and report failures without hiding them.
```

### Prompt 5: planner sweep

```text
Implement tools/run_planner_sweep.py. It should run without hardware, sweep width and grasp mode, write summary.csv, and generate at least one simple plot if matplotlib is available. Use existing planner classes where possible. Do not invent kinematic formulas if existing functions are available. If required APIs are unclear, inspect the repo and adapt.
```

### Prompt 6: margin sweep

```text
Implement tools/run_hybrid_margin_sweep.py as a post-hoc simulation model for switch margin and contact speed. It should separate measured constants from assumptions, write summary.csv, and generate a heatmap or Pareto plot. Keep the model simple and transparent. Do not claim this replaces real experiments.
```

### Prompt 7: force threshold replay

```text
Implement tools/replay_force_thresholds.py. It should accept existing peg-in-hole force logs if present. If logs are absent, it should write an expected_log_schema.csv and exit cleanly. It should sweep contact spike threshold, lateral spike threshold, and moving-average window. Do not require hardware.
```

### Prompt 8: grasp style coverage

```text
Implement tools/run_grasp_style_coverage.py for sim-only coverage analysis across object dimensions, grasp modes, friction values, and pose noise. Start with geometry feasibility and collision or invalid-state reporting. If GWS functions already exist, use them. If not, leave TODO hooks and write available metrics first.
```

### Prompt 9: hardware bus scanner

```text
On branch hardware-debug, implement tools/rh56_bus_scan.py as a read-only RS485 diagnostic tool. It should scan IDs, print raw TX/RX bytes, avoid motion commands by default, and fail safely if the serial port is missing. Do not add write commands unless behind an explicit flag and confirmation.
```

### Prompt 10: paper figure manifest

```text
Create docs/reproduce_paper_figures.md. List every planned paper figure, the script that generates it, input data requirements, output path, and whether it requires hardware. Separate sim-only figures from real-hardware figures.
```

## 11. Review checklist for every Codex PR

Before accepting a Codex change, check:

- Does it run without hardware if it claims to be sim-only?
- Did it avoid touching hardware code unless explicitly asked?
- Did it add clear error messages for missing optional dependencies?
- Did it write outputs under `artifacts/` rather than polluting the repo root?
- Did it avoid committing raw logs, videos, large generated plots, or local workspaces?
- Did it avoid overclaiming scientific guarantees?
- Does the new script have a docstring and `--help` output?
- Is the output CSV schema documented?
- Are assumptions printed or written to metadata?
- Did tests pass, or are failures honestly reported?

## 12. Definition of done for paper-v2 tooling

Minimum acceptable state:

```text
make check-sim passes
a minimal pytest suite passes or skips optional dependencies cleanly
docs/status.md exists
docs/not_now.md exists
docs/reproduce_paper_figures.md exists
planner sweep script produces summary.csv
margin sweep script produces summary.csv
threshold replay script either runs on logs or writes expected schema
grasp style coverage script produces at least geometry feasibility summary
hardware bus scanner exists on hardware-debug branch and is read-only by default
```

Stronger state:

```text
all four paper-facing experiment scripts produce figures
figures can be regenerated from one command
paper claim-to-evidence table is updated with paths to generated outputs
hardware diagnosis log identifies whether the fault is hand-side, cable-side, board-side, or adapter-side
```

## 13. One-sentence operating principle

Do not expand the system until the existing RH56 contribution is reproducible, parameter-justified, and paper-ready.
