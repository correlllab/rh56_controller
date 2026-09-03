# Paper Figure Reproduction Plan

This manifest maps planned paper-v2 evidence to the scripts that generate it.
Generated outputs should live under `artifacts/` and should not be committed
unless they are intentionally curated small summaries.

| Figure or table | Script | Inputs | Output path | Hardware | Claim |
|---|---|---|---|---|---|
| Planner object-width feasible ranges | `tools/run_planner_sweep.py` | RH56 MuJoCo XML and FK cache generated from repo assets; 20 mm fingertip-site correction | `artifacts/planner_sweep/summary.csv`, `planner_feasible_ranges.png`, `reachable_widths.png` | No | Coupled RH56 geometry produces mode-specific object-width ranges for line/plane analytical grasps; this is not a real grasp-success rate. |
| Analytical no-go volume | `tools/run_analytical_grasp_volume.py` | Coarse YCB-like tabletop object AABBs, analytical grasp poses, open-hand capsule proxy with thumb yaw at max qpos, swept capsule-vs-AABB linear approach check | `artifacts/analytical_grasp_volume/summary.csv`, `volume.csv`, `*_viability_slices.png`, `*_no_go_volume.png` | No | Simple analytical grasping has a characterized no-go volume where a linear Cartesian move cannot reach the grasp pose without object-aware path planning. |
| Capsule proxy debug check | `tools/verify_capsule_hand_proxy.py` | Default 40 mm tabletop cube debug proxy, optional YCB-like proxies, MuJoCo hand FK, capsule hand envelope, swept capsule path check | `artifacts/capsule_proxy_demo/summary.csv`, `capsules.csv`, `path_samples.csv`, `capsule_proxy_demo.png` | No | Debug-only validation that the hand-volume proxy and collision check are geometrically interpretable before using them in a dense no-go sweep. |
| Interactive capsule path sandbox | `tools/capsule_path_gui.py` | Default 40 mm tabletop cube debug proxy, open finger path shape with thumb yaw at max qpos, MuJoCo hand FK, swept capsule path check | MuJoCo GUI plus terminal status; no paper artifact by default | No | Debug-only tool for moving the hand start pose, previewing whether the straight-line capsule path is valid, and animating valid paths. |
| Strategy pre-grasp demo | `tools/demo_strategy_pregrasp_collision.py` | One fixed object/final analytical grasp pose, strategy-specific pre-grasp hand shapes for naive/iterative-closure/thumb-reflex, iterative pre-grasp opened by a small width margin, capsule object/floor collision checks | `artifacts/strategy_pregrasp_demo/summary.csv`, `assumptions.json`, `strategy_pregrasp_demo.png` | No | Debug-only check that the strategy-specific no-go-volume logic is visually and numerically sensible before running dense sweeps. |
| Strategy pre-grasp feasible rate | `tools/run_strategy_pregrasp_rate.py` | Fixed 40 mm cube, final analytical grasp centered on the object AABB, common sampled start hand-base offsets around the final pose, strategy-specific pre-grasp targets, object/floor capsule collision checks | `artifacts/strategy_pregrasp_rate_40mm/summary.csv`, `volume.csv`, `assumptions.json`, `strategy_feasible_rate.png` | No | First coarse rate estimate for how execution strategy changes the collision-free pre-grasp access volume. |
| Planner width-tracking diagnostic | `tools/run_planner_sweep.py` | Same as above | `artifacts/planner_sweep/width_error_by_mode.png`, `summary.csv` | No | Multi-finger corrections introduce scalar internal-width error that should be reported as a planner diagnostic, not as grasp success. |
| Planner offline table-generation time | `tools/run_planner_sweep.py` | Same as above | `artifacts/planner_sweep/solve_time_by_mode.png`, `solve_time_hist.png`, `summary.csv` | No | Offline analytical solves are fast enough to precompute lookup tables; online execution should be reported as lookup, not this solve time. |
| Hybrid margin robustness map | `tools/run_hybrid_margin_sweep.py` | Measured latency summary plus contact-onset uncertainty assumption | `artifacts/hybrid_margin_sweep/summary.csv`, `margin_speed_heatmap.png` | No | The switch margin is a latency-aware uncertainty buffer rather than a magic number. |
| Anticipatory switch probability | `tools/run_hybrid_margin_sweep.py` | Contact-onset uncertainty assumption | `artifacts/hybrid_margin_sweep/pre_slow_probability.png`, `assumptions.json` | No | A 25-unit margin can be interpreted as high probability of switching before contact under the stated uncertainty model. |
| Contact-speed Pareto proxy | `tools/run_hybrid_margin_sweep.py` | Contact-speed list, measured latency, latency model assumptions | `artifacts/hybrid_margin_sweep/pareto.png`, `assumptions.json` | No | Contact-phase speed governs the latency-travel/time tradeoff. |
| Force-threshold replay sensitivity | `tools/replay_force_thresholds.py` | Optional peg-in-hole logs matching `expected_log_schema.csv` | `artifacts/threshold_replay/summary.csv` | Replay only | Finger-force thresholds should be shown as robust over a parameter range. |
| Force-threshold expected schema | `tools/replay_force_thresholds.py` | None, when logs are unavailable | `artifacts/threshold_replay/expected_log_schema.csv` | No | Missing hardware data is documented rather than silently assumed. |

## Notes

- Method details and assumptions are summarized in `docs/paper_v2_methods.md`.
- Real-hardware refresh figures should be added only after RH56 read/write is
  working again.
- The current v2 priority is evidence for latency-aware switching and the
  analytical width solver/no-go volume, not new dexterous demonstrations.
- Every script listed here should support `--help`, write assumptions or
  metadata, and fail gracefully when optional plotting or replay inputs are
  unavailable.
