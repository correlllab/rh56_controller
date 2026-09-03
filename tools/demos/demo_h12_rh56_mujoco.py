#!/usr/bin/env python3
"""Direct MuJoCo demo for RH56 grasp planning on the H1-2 arm.

This bypasses the Tk grasp_viz UI and opens the H1-2 + RH56/Inspire MuJoCo
viewer directly. It animates the same width-space idea used by grasp_viz:
move the H1-2 wrist from home to the planned wrist target, then close the RH56
hand through planned width waypoints. By default the wrist target is recomputed
at every width so the coupled RH56 closure keeps the antipodal object center
and contact-plane orientation fixed in the world frame. A headless PINK check
rejects unreachable wrist waypoint paths before the viewer starts.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a direct MuJoCo H1-2 + RH56 grasp demo."
    )
    parser.add_argument("--mode", choices=MODE_MAP, default="plane4")
    parser.add_argument("--target-width-mm", type=float, default=40.0)
    parser.add_argument("--start-width-mm", type=float, default=None)
    parser.add_argument("--step-mm", type=float, default=10.0)
    parser.add_argument("--grasp-x", type=float, default=0.25, help="H1-2 pelvis-frame X target, positive is forward, metres")
    parser.add_argument("--grasp-y", type=float, default=-0.20, help="H1-2 pelvis-frame Y target; negative is the robot's right side, metres")
    parser.add_argument("--grasp-z", type=float, default=0.15, help="H1-2 pelvis-frame grasp height, metres")
    parser.add_argument(
        "--plane-rz-deg",
        type=float,
        default=-120.0,
        help="Rotate the grasp/contact plane about world Z; -120 deg is the reachable right-arm smoke-test pose",
    )
    parser.add_argument(
        "--grasp-center-policy",
        choices=("antipodal", "contact-centroid"),
        default="antipodal",
        help=(
            "antipodal (default) locks the yellow object-center marker between "
            "the thumb and opposing fingers; contact-centroid preserves legacy behavior"
        ),
    )
    parser.add_argument(
        "--wrist-policy",
        choices=("fixed", "sync"),
        default="sync",
        help=(
            "sync (default) recomputes the wrist target at each width waypoint "
            "to compensate RH56 coupled-joint closure; fixed holds the first "
            "wrist target as an ablation"
        ),
    )
    parser.add_argument("--arm-duration", type=float, default=2.5)
    parser.add_argument("--segment-duration", type=float, default=0.5)
    parser.add_argument("--hold", type=float, default=4.0)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild RH56 FK cache")
    parser.add_argument(
        "--skip-ik-check",
        action="store_true",
        help="Launch even when the headless H1-2 wrist-waypoint reachability check is skipped",
    )
    parser.add_argument("--ik-check-iters", type=int, default=300)
    parser.add_argument("--ik-position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--ik-orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override RH56 FK cache dir; useful on machines with read-only HOME.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class IKWaypointCheck:
    width_mm: float
    target_position: np.ndarray
    position_error_mm: float
    orientation_error_deg: float


def build_width_results(core: GraspVizCore, args: argparse.Namespace):
    """Build the exact width waypoints shared by preflight and animation."""
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
    results[0] = approach_result
    results[-1] = target_result
    return approach_result, target_result, results


def check_h12_ik_waypoints(
    core: GraspVizCore,
    results,
    *,
    max_iters: int,
) -> list[IKWaypointCheck]:
    """Solve the demo's wrist path headlessly and return per-waypoint errors."""
    import pinocchio as pin
    import pink
    import qpsolvers

    from rh56_controller.grasp_viz_workers import (
        _H12_ARM_JOINTS,
        _H12_EE_FRAME,
        _H12_HOME_Q,
        _build_h12_pin_model,
    )

    if max_iters < 1:
        raise ValueError("--ik-check-iters must be at least 1")

    model = _build_h12_pin_model(pin)
    data = model.createData()
    q0 = pin.neutral(model)
    arm_qidx = []
    for index, joint_name in enumerate(_H12_ARM_JOINTS):
        joint_id = model.getJointId(joint_name)
        qidx = model.joints[joint_id].idx_q
        q0[qidx] = _H12_HOME_Q[index]
        arm_qidx.append(qidx)

    configuration = pink.Configuration(model, data, q0)
    ee_task = pink.tasks.FrameTask(
        _H12_EE_FRAME,
        position_cost=50.0,
        orientation_cost=30.0,
        lm_damping=3.0,
    )
    posture_task = pink.tasks.PostureTask(cost=1e-2)
    posture_task.set_target(q0)
    limits = [
        pink.limits.ConfigurationLimit(model),
        pink.limits.VelocityLimit(model),
    ]
    solver = "daqp" if "daqp" in qpsolvers.available_solvers else None
    if solver is None and qpsolvers.available_solvers:
        solver = qpsolvers.available_solvers[0]
    if solver is None:
        raise RuntimeError("No qpsolvers backend is available for the H1-2 IK check")

    non_arm_qidx = [i for i in range(model.nq) if i not in set(arm_qidx)]
    frame_id = model.getFrameId(_H12_EE_FRAME)
    checks: list[IKWaypointCheck] = []

    for result in results:
        target_matrix, _ = core._h12_wrist_target_world(result)
        ee_task.set_target(pin.SE3(target_matrix))
        for _ in range(max_iters):
            velocity = pink.solve_ik(
                configuration,
                [ee_task, posture_task],
                dt=0.05,
                solver=solver,
                limits=limits,
                safety_break=False,
            )
            configuration.integrate_inplace(velocity, 0.05)
            q_locked = np.array(configuration.q, copy=True)
            q_locked[non_arm_qidx] = q0[non_arm_qidx]
            configuration = pink.Configuration(model, data, q_locked)

        actual = configuration.data.oMf[frame_id].homogeneous
        position_error_mm = 1000.0 * float(
            np.linalg.norm(actual[:3, 3] - target_matrix[:3, 3])
        )
        relative_rotation = target_matrix[:3, :3].T @ actual[:3, :3]
        orientation_error_deg = float(np.degrees(np.arccos(np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        ))))
        checks.append(IKWaypointCheck(
            width_mm=1000.0 * float(result.width),
            target_position=target_matrix[:3, 3].copy(),
            position_error_mm=position_error_mm,
            orientation_error_deg=orientation_error_deg,
        ))

    return checks


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
    approach_result, target_result, results = build_width_results(core, args)
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
    core._plane_rz = math.radians(args.plane_rz_deg)
    core._grasp_center_policy = args.grasp_center_policy
    core._width_range = core.closure.width_range(str(mode), n_fingers=n_fingers)
    core._width_target_m = float(np.clip(args.target_width_mm / 1000.0, *core._width_range))
    core._width_m = core._width_target_m

    approach_result, target_result, results = build_width_results(core, args)
    print(
        "[demo_h12_rh56] assumptions: pelvis-frame target, fixed base/torso, "
        f"right arm, center={args.grasp_center_policy}, plane_rz={args.plane_rz_deg:.1f} deg"
    )
    if not args.skip_ik_check:
        try:
            checks = check_h12_ik_waypoints(
                core,
                results if args.wrist_policy == "sync" else [approach_result],
                max_iters=args.ik_check_iters,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            print(f"[demo_h12_rh56] IK preflight unavailable: {exc}", file=sys.stderr)
            print("[demo_h12_rh56] rerun setup or use --skip-ik-check to override", file=sys.stderr)
            return 2

        worst_position = max(checks, key=lambda row: row.position_error_mm)
        worst_orientation = max(checks, key=lambda row: row.orientation_error_deg)
        print(
            "[demo_h12_rh56] IK preflight: "
            f"max position error={worst_position.position_error_mm:.2f} mm "
            f"at {worst_position.width_mm:.1f} mm; "
            f"max orientation error={worst_orientation.orientation_error_deg:.2f} deg "
            f"at {worst_orientation.width_mm:.1f} mm"
        )
        if (
            worst_position.position_error_mm > args.ik_position_tolerance_mm
            or worst_orientation.orientation_error_deg > args.ik_orientation_tolerance_deg
        ):
            print(
                "[demo_h12_rh56] refusing unreachable wrist path; adjust grasp pose/plane yaw "
                "or use --skip-ik-check for intentional debugging",
                file=sys.stderr,
            )
            return 2

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
