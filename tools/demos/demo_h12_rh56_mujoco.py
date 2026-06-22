#!/usr/bin/env python3
"""Direct MuJoCo demo for RH56 grasp planning on the H1-2 arm.

This bypasses the Tk grasp_viz UI and opens the H1-2 + RH56/Inspire MuJoCo
viewer directly. It animates the same width-space idea used by grasp_viz:
move the H1-2 wrist from home to the planned wrist target, then close the RH56
hand through planned width waypoints.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"))


def preparse_cache_dir() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if args.cache_dir is not None:
        os.environ["RH56_CACHE_DIR"] = str(args.cache_dir)


preparse_cache_dir()

from rh56_controller.grasp_geometry import GraspMode  # noqa: E402
from rh56_controller.grasp_viz_core import GraspVizCore  # noqa: E402


MODE_MAP = {
    "line": (GraspMode.LINE_2F, 2),
    "plane3": (GraspMode.PLANE_3F, 3),
    "plane4": (GraspMode.PLANE_4F, 4),
    "plane5": (GraspMode.PLANE_5F, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a direct MuJoCo H1-2 + RH56 grasp demo."
    )
    parser.add_argument("--mode", choices=MODE_MAP, default="plane4")
    parser.add_argument("--target-width-mm", type=float, default=40.0)
    parser.add_argument("--start-width-mm", type=float, default=None)
    parser.add_argument("--step-mm", type=float, default=10.0)
    parser.add_argument("--grasp-x", type=float, default=0.5, help="H1-2 viewer X target, positive is in front, metres")
    parser.add_argument("--grasp-y", type=float, default=0.20, help="H1-2 viewer Y target, positive is right; keep roughly within +/-0.2 m")
    parser.add_argument("--grasp-z", type=float, default=0.15, help="H1-2 viewer grasp height, metres")
    parser.add_argument(
        "--wrist-policy",
        choices=("fixed", "sync"),
        default="fixed",
        help=(
            "fixed holds the first wrist target while the RH56 closes; "
            "sync recomputes the wrist target at each width waypoint"
        ),
    )
    parser.add_argument("--arm-duration", type=float, default=2.5)
    parser.add_argument("--segment-duration", type=float, default=0.5)
    parser.add_argument("--hold", type=float, default=4.0)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild RH56 FK cache")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override RH56 FK cache dir; useful on machines with read-only HOME.",
    )
    return parser.parse_args()


def set_result(core: GraspVizCore, result) -> None:
    with core._state_lock:
        core._result = result
    core._custom_ctrl_arr[:] = core._build_ctrl_array(result)
    core._viewer_state_arr[:] = core._build_state_array(result)


def set_ctrl_for_result(
    core: GraspVizCore,
    result,
    final_thumb_yaw: float,
    fixed_wrist_ctrl: np.ndarray | None = None,
) -> None:
    finger_ctrl = dict(result.ctrl_values)
    finger_ctrl["thumb_yaw"] = final_thumb_yaw
    ctrl = core._build_ctrl_array(result, finger_ctrl)
    if fixed_wrist_ctrl is not None:
        ctrl[:6] = fixed_wrist_ctrl
    core._custom_ctrl_arr[:] = ctrl
    core._viewer_state_arr[:] = core._build_state_array(result)


def animate_scalar(shared_value, start: float, end: float, duration: float) -> None:
    steps = max(1, int(duration / 0.033))
    for i in range(steps + 1):
        t = i / steps
        shared_value.value = float(start + t * (end - start))
        time.sleep(duration / steps)


def animate_width_segment(
    core: GraspVizCore,
    start_result,
    end_result,
    final_thumb_yaw: float,
    duration: float,
    fixed_wrist_ctrl: np.ndarray | None = None,
) -> None:
    steps = max(1, int(duration / 0.033))
    for i in range(steps + 1):
        alpha = i / steps
        width = start_result.width + alpha * (end_result.width - start_result.width)
        try:
            result = core.closure.solve(core._mode, width)
        except Exception:
            result = end_result if alpha > 0.5 else start_result
        set_ctrl_for_result(core, result, final_thumb_yaw, fixed_wrist_ctrl)
        time.sleep(duration / steps)


def run_once(core: GraspVizCore, args: argparse.Namespace) -> None:
    target_result = core.closure.solve(core._mode, core._width_target_m)
    start_width = (
        args.start_width_mm / 1000.0
        if args.start_width_mm is not None
        else core._width_range[1]
    )
    start_width = min(start_width, core._width_range[1])
    start_width = max(start_width, target_result.width + 1e-4)
    approach_result = core.closure.solve(core._mode, start_width)

    step_m = max(args.step_mm / 1000.0, 1e-4)
    n_segments = max(1, int(math.ceil((approach_result.width - target_result.width) / step_m)))
    widths = np.linspace(approach_result.width, target_result.width, n_segments + 1)
    results = [core.closure.solve(core._mode, float(width)) for width in widths]
    final_thumb_yaw = target_result.ctrl_values.get("thumb_yaw", 0.0)

    mode_label = getattr(core._mode, "value", str(core._mode))
    print(
        f"[demo_h12_rh56] {mode_label}: "
        f"{approach_result.width * 1000:.1f} mm -> "
        f"{target_result.width * 1000:.1f} mm, "
        f"{len(results) - 1} segments"
    )

    core._sim_grasp_t.value = 1.0
    core._sim_arm_t.value = 0.0
    set_ctrl_for_result(core, approach_result, final_thumb_yaw)
    fixed_wrist_ctrl = (
        np.array(core._custom_ctrl_arr[:6], dtype=float)
        if args.wrist_policy == "fixed"
        else None
    )

    print("[demo_h12_rh56] moving H1-2 wrist from home to grasp target")
    animate_scalar(core._sim_arm_t, 0.0, 1.0, args.arm_duration)

    if fixed_wrist_ctrl is None:
        print("[demo_h12_rh56] closing RH56 fingers with synchronized wrist updates")
    else:
        print("[demo_h12_rh56] closing RH56 fingers while holding wrist target fixed")
    for prev_result, next_result in zip(results, results[1:]):
        animate_width_segment(
            core,
            prev_result,
            next_result,
            final_thumb_yaw,
            args.segment_duration,
            fixed_wrist_ctrl,
        )
    set_ctrl_for_result(core, target_result, final_thumb_yaw, fixed_wrist_ctrl)

    if args.hold > 0:
        print(f"[demo_h12_rh56] holding final grasp for {args.hold:.1f}s")
        time.sleep(args.hold)


def main() -> int:
    args = parse_args()
    mode, n_fingers = MODE_MAP[args.mode]
    core = GraspVizCore(
        rebuild=args.rebuild,
        h12_mode=True,
        bimanual_mode=False,
        mink_viz=False,
    )
    core._mode = mode
    core.set_active_arm_override("right")
    core._grasp_x = args.grasp_x
    core._grasp_y = args.grasp_y
    core._grasp_z = args.grasp_z
    core._width_range = core.closure.width_range(str(mode), n_fingers=n_fingers)
    core._width_target_m = float(np.clip(args.target_width_mm / 1000.0, *core._width_range))
    core._width_m = core._width_target_m

    target_result = core.closure.solve(core._mode, core._width_target_m)
    set_result(core, target_result)
    core._launch_h12_viewer()

    try:
        while True:
            run_once(core, args)
            if not args.repeat:
                break
            print("[demo_h12_rh56] repeat requested; rewinding in 1s")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[demo_h12_rh56] interrupted")
    finally:
        core._h12_stop.set()
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
