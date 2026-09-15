#!/usr/bin/env python3
"""Export the existing GUI Plan for a saved camera target, without hardware."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.vision_grasp_target import (
    add_vision_target_arguments, vision_target_from_args,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_vision_target_arguments(parser)
    parser.add_argument("--step-mm", type=float, default=10.0,
                        help="Existing Plan width step (not Cartesian distance)")
    parser.add_argument("--approach-mm", type=float,
                        help="Existing Plan approach width; defaults to solver range maximum")
    parser.add_argument("--out", type=Path, default=Path("artifacts/vision_plan_preview"))
    args = parser.parse_args(argv)
    if args.vision_target is None:
        parser.error("--vision-target is required")
    if not np.isfinite(args.step_mm) or args.step_mm <= 0:
        parser.error("--step-mm must be positive and finite")
    if args.approach_mm is not None and (not np.isfinite(args.approach_mm) or args.approach_mm <= 0):
        parser.error("--approach-mm must be positive and finite")
    target = vision_target_from_args(args)

    # Exactly the GUI state builder and Plan waypoint generator; no duplicate
    # closure, wrist compensation, angle mapping, or hardware motion loop.
    from rh56_controller.grasp_viz_core import GraspVizCore
    from rh56_controller.grasp_executor import GraspExecutor
    from rh56_controller.ur5_bridge import UR5Bridge

    core = GraspVizCore(robot_mode=True, mink_viz=False,
                        vision_target=target, hand_write_only=True)
    try:
        result = core.closure.solve(core._mode, core._width_target_m)
        closures = core._compute_plan_closures(
            args.step_mm, result,
            None if args.approach_mm is None else args.approach_mm / 1000.0,
        )
        executor = GraspExecutor(None, None, core.fk, hand_write_only=True)
        final_yaw_cmd = executor._ctrl_to_real(result.ctrl_values)[5]
        waypoints, rows = [], []
        for index, closure in enumerate(closures):
            hand_T = core._build_world_T_hand(closure)
            tcp_T = UR5Bridge.hand_T_to_tcp_T(hand_T)
            command = executor._ctrl_to_real(closure.ctrl_values)
            command[5] = final_yaw_cmd  # Existing Plan holds final thumb yaw.
            waypoints.append({
                "index": index, "achieved_width_m": closure.width,
                "hand_base_in_ur_base": hand_T.tolist(),
                "zero_tcp_in_ur_base": tcp_T.tolist(),
                "hand_angle_command": command,
            })
            rows.append({
                "index": index, "achieved_width_mm": closure.width * 1000,
                "tilt_deg": closure.tilt_deg,
                **dict(zip(("tcp_x_mm", "tcp_y_mm", "tcp_z_mm"), tcp_T[:3, 3] * 1000)),
                "hand_angle_command": " ".join(map(str, command)),
            })
        report = {
            "schema": "rh56_vision_existing_plan_preview/v1",
            "hardware_contacted": False,
            "strategy": "Plan", "input": target.metadata(),
            "step_mm": args.step_mm, "approach_mm": args.approach_mm,
            "force_N": 0, "hand_write_only": True,
            "limitations": [
                "Geometric preview only; no arm IK, path collision, or current-state validation.",
                "Saved camera target is historical until a fresh measurement is supplied.",
                "UR viewer model is UR5e; laboratory arm is CB3. Viewer is not a safety gate.",
            ],
            "waypoints": waypoints,
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"input": report["input"], "limitations": report["limitations"],
                          "waypoint_count": len(rows), "out": str(args.out.resolve())}, indent=2))
    finally:
        core.cleanup()


if __name__ == "__main__":
    main()
