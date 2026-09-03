# Paper-V2 Figure Review Audit

This note is the gate for whether a graph belongs in the v2 paper. Each graph
must answer a reviewer-facing question, say what evidence it provides, and say
what it does not prove.

| Graph | Reviewer-facing question | What it shows | What it does not show | Decision |
|---|---|---|---|---|
| `artifacts/planner_sweep/planner_feasible_ranges.png` | Is the analytical planner a characterized method rather than a demo-only heuristic? | The object-width-corrected range for line/plane analytical modes under the repo FK/geometry model. | It is not object-grasp success, friction robustness, collision checking, or hardware validation. Cylinder is excluded from the paper-facing default. | Keep, but describe as nominal feasibility. |
| `artifacts/analytical_grasp_volume/*_viability_slices.png` | Where do simple analytical grasps fail before needing object-aware path planning? | For coarse tabletop object AABBs, each voxel shows the fraction of sampled hand yaw poses whose open-hand capsule proxy can linearly reach the analytical grasp pose without colliding with the object AABB. | It is not full MuJoCo mesh collision, arm IK, learning comparison, or a hardware result. | New core characterization figure; refine before paper submission. |
| `artifacts/planner_sweep/width_error_by_mode.png` | What happens when more fingers are added to the width-parameterized solver? | The scalar width-tracking error after multi-finger coplanarity/geometry corrections. | It does not mean more fingers are worse at grasping; it only exposes a diagnostic limitation of the scalar width metric. | Keep as diagnostic or move to appendix. |
| `artifacts/planner_sweep/solve_time_by_mode.png` | Is precomputation practical? | Offline analytical solve/table-generation timing grouped by planner mode. | It is not online runtime; live control should use lookup-table timing instead. | Appendix or supporting text, not the main planner evidence. |
| `artifacts/hybrid_margin_sweep/pre_slow_probability.png` | How does the anticipatory switch margin enter the controller instead of being a magic number? | The probability of reaching slow mode before contact under the stated onset uncertainty model. | It does not independently measure contact-onset uncertainty. | Keep in main text. |
| `artifacts/hybrid_margin_sweep/margin_speed_heatmap.png` | How do latency, contact speed, and margin jointly affect overshoot risk? | Expected latency travel for each margin/contact-speed pair, using the measured latency summary when available. | It is a proxy model, not real-hardware overshoot validation. | Keep in main text. |
| `artifacts/hybrid_margin_sweep/pareto.png` | What tradeoff is created by slower contact speed and larger switch margin? | A time proxy versus overshoot proxy curve, with robust points marked. | It does not prove an optimal policy without task-specific cost weights. | Keep if space allows; otherwise appendix. |
| `artifacts/threshold_replay/summary.csv` | Are force thresholds robust across logs? | With logs, replay precision/recall and detection delay over threshold sweeps. | With no logs, it provides no validation result. | Keep only after real logs are available. |
| `artifacts/threshold_replay/expected_log_schema.csv` | What data is needed to make force-threshold claims reproducible? | The required CSV schema for future replay. | It is not an experiment result. | Keep in docs, not as a paper figure. |

Current logic change:

- The planner sweep no longer reports `success` as if it were grasp success.
- `nominal_feasible` means the requested scalar width is inside the
  mode-specific nominal solver range and the solve completed.
- `width_match` means the achieved scalar width is within tolerance. This is a
  planner diagnostic only.
- Real grasp success must come from simulation with object/contact assumptions
  or from hardware logs, not from this geometric sweep alone.
