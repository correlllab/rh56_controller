#!/usr/bin/env python3
"""Compare practical Magpie and RH56 closure compensation on H12.

This simulation-only experiment holds the same two-finger grasp center and
pinch frame stationary for both hands.  RH56 closure stops at a provisional
cutoff before the closure-plane knee; no mathematically complete closure is
commanded.  The resulting H12 arm motion is evaluated kinematically together
with a quasi-static whole-body center-of-mass support margin.

The script writes summary.csv, trajectory.csv, assumptions.json, a comparison
plot, and an optional side-by-side video.  ``--live`` opens the exact same
trajectories in interactive MuJoCo viewers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.grasp_geometry import (  # noqa: E402
    ClosureGeometry,
    GraspMode,
    InspireHandFK,
)
from tools.run_h12_gripper_motion_comparison import (  # noqa: E402
    CONDITION_LABELS,
    DEFAULT_MAGPIE_GRIPPER_XML,
    DEFAULT_MAGPIE_XML,
    DEFAULT_RH56_XML,
    MAGPIE_TIP_SITES,
    RH56_TIP_SITES,
    align_axis_near_reference,
    build_condition,
    build_rh56_wrist_targets,
    interpolate_magpie_qpos,
    named_id,
    print_summary,
    render_video,
    run_live,
    settle_magpie_lookup,
    solve_h12_ik,
    wrist_local_magpie_trajectory,
    write_summary,
    write_trajectory,
)


PRACTICAL_CONDITIONS = (
    "magpie_compensated",
    "rh56_practical_compensated",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare compensated Magpie closure with RH56 closure stopped at "
            "a provisional pre-knee cutoff, including static CoM margin."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/h12_practical_gripper_comparison"),
    )
    parser.add_argument("--rh56-xml", type=Path, default=DEFAULT_RH56_XML)
    parser.add_argument("--magpie-xml", type=Path, default=DEFAULT_MAGPIE_XML)
    parser.add_argument(
        "--magpie-gripper-xml",
        type=Path,
        default=DEFAULT_MAGPIE_GRIPPER_XML,
    )
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument(
        "--target-width-mm",
        type=float,
        default=20.0,
        help="Requested object/grasp width before applying the RH56 cutoff.",
    )
    parser.add_argument(
        "--rh56-cutoff-mm",
        type=float,
        default=40.0,
        help=(
            "Provisional minimum RH56 command width. Closure stops here even "
            "when --target-width-mm requests a smaller width."
        ),
    )
    parser.add_argument("--start-width-mm", type=float, default=None)
    parser.add_argument("--grasp-x", type=float, default=0.25)
    parser.add_argument("--grasp-y", type=float, default=-0.20)
    parser.add_argument("--grasp-z", type=float, default=0.15)
    parser.add_argument("--plane-rz-deg", type=float, default=-120.0)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ik-initial-iters", type=int, default=800)
    parser.add_argument("--ik-step-iters", type=int, default=35)
    parser.add_argument("--foot-sole-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--panel-height", type=int, default=420)
    parser.add_argument("--camera-distance", type=float, default=0.85)
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=180.0,
        help="MuJoCo camera azimuth; 180 degrees faces the front of H12.",
    )
    parser.add_argument("--camera-elevation", type=float, default=-10.0)
    parser.add_argument(
        "--video-format",
        choices=("auto", "mp4", "gif"),
        default="auto",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--live-condition",
        choices=("all", *PRACTICAL_CONDITIONS),
        default="all",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.hold_frames < 0:
        raise ValueError("--hold-frames cannot be negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.rh56_cutoff_mm <= 0.0:
        raise ValueError("--rh56-cutoff-mm must be positive")
    if args.foot_sole_tolerance_mm <= 0.0:
        raise ValueError("--foot-sole-tolerance-mm must be positive")
    if args.ik_initial_iters < 1 or args.ik_step_iters < 1:
        raise ValueError("IK iteration counts must be positive")
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise ValueError("panel dimensions must be positive")


def effective_target_width_mm(requested_mm: float, cutoff_mm: float) -> float:
    """Apply a minimum-width cutoff to a descending closure command."""
    return max(float(requested_mm), float(cutoff_mm))


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2-D convex hull using the monotone chain."""
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        raise ValueError("At least three unique support points are required")

    def cross(origin, point_a, point_b) -> float:
        return (
            (point_a[0] - origin[0]) * (point_b[1] - origin[1])
            - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def signed_support_margin_m(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    """Signed minimum edge distance for a counter-clockwise convex polygon."""
    point = np.asarray(point_xy, dtype=float)
    polygon = np.asarray(polygon_xy, dtype=float)
    following = np.roll(polygon, -1, axis=0)
    edges = following - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Support polygon contains a zero-length edge")
    cross = edges[:, 0] * (point[1] - polygon[:, 1]) - edges[:, 1] * (
        point[0] - polygon[:, 0]
    )
    return float(np.min(cross / lengths))


def foot_support_polygon(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    sole_tolerance_m: float,
) -> np.ndarray:
    """Project bottom collision-mesh vertices of both feet to the ground."""
    foot_vertices = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        if "ankle_roll_link" not in body_name:
            continue
        if int(model.geom_contype[geom_id]) == 0:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[start : start + count])
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world = data.geom_xpos[geom_id] + local @ rotation.T
        foot_vertices.append(world)
    if not foot_vertices:
        raise ValueError("No collidable ankle-roll foot meshes were found")

    vertices = np.vstack(foot_vertices)
    sole = vertices[vertices[:, 2] <= vertices[:, 2].min() + sole_tolerance_m]
    return convex_hull_xy(sole[:, :2])


def whole_body_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = np.asarray(model.body_mass[1:], dtype=float)
    return np.sum(data.xipos[1:] * masses[:, None], axis=0) / masses.sum()


def evaluate_static_balance(condition, *, sole_tolerance_m: float) -> float:
    model = mujoco.MjModel.from_xml_path(str(condition.model_path))
    data = mujoco.MjData(model)
    com_rows = []
    polygon = None
    margins = []
    for frame_index, qpos in enumerate(condition.qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if frame_index == 0:
            polygon = foot_support_polygon(
                model,
                data,
                sole_tolerance_m=sole_tolerance_m,
            )
        com = whole_body_com(model, data)
        com_rows.append(com)
        margins.append(1000.0 * signed_support_margin_m(com[:2], polygon))

    condition.com_world = np.asarray(com_rows)
    condition.support_margin_mm = np.asarray(margins)
    condition.support_polygon_xy = polygon
    return float(np.asarray(model.body_mass[1:]).sum())


def magpie_wrist_targets(
    local_anchors: np.ndarray,
    local_frames: np.ndarray,
    *,
    desired_world_frame: np.ndarray,
    target_pelvis: np.ndarray,
) -> np.ndarray:
    targets = []
    for anchor, local_frame in zip(local_anchors, local_frames):
        wrist_rotation = desired_world_frame @ local_frame.T
        target = np.eye(4)
        target[:3, :3] = wrist_rotation
        target[:3, 3] = target_pelvis - wrist_rotation @ anchor
        targets.append(target)
    return np.asarray(targets)


def prepare_conditions(args: argparse.Namespace):
    for path in (args.rh56_xml, args.magpie_xml, args.magpie_gripper_xml):
        if not path.exists():
            raise FileNotFoundError(path)

    fk = InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    width_range = closure.width_range(str(GraspMode.LINE_2F), n_fingers=2)
    lower_mm, upper_mm = (1000.0 * width_range[0], 1000.0 * width_range[1])
    requested_target_mm = float(np.clip(args.target_width_mm, lower_mm, upper_mm))
    cutoff_mm = float(np.clip(args.rh56_cutoff_mm, lower_mm, upper_mm))
    realized_target_mm = effective_target_width_mm(requested_target_mm, cutoff_mm)
    start_mm = (
        upper_mm
        if args.start_width_mm is None
        else float(np.clip(args.start_width_mm, realized_target_mm, upper_mm))
    )
    requested_width_m = np.linspace(start_mm, realized_target_mm, args.frames) / 1000.0
    rh56_results = [
        closure.solve(GraspMode.LINE_2F, float(width))
        for width in requested_width_m
    ]
    rh56_command_width_mm = 1000.0 * np.asarray(
        [result.width for result in rh56_results]
    )
    target_pelvis = np.array([args.grasp_x, args.grasp_y, args.grasp_z])
    rh56_targets, desired_world_frame = build_rh56_wrist_targets(
        rh56_results,
        target_pelvis=target_pelvis,
        plane_rz_rad=math.radians(args.plane_rz_deg),
    )

    print("[practical] solving RH56 pre-knee compensated trajectory...")
    rh56_solution = solve_h12_ik(
        rh56_targets,
        initial_iters=args.ik_initial_iters,
        step_iters=args.ik_step_iters,
    )

    print("[practical] building quasi-static Magpie closure lookup...")
    commands, lookup_qpos, lookup_gaps = settle_magpie_lookup(
        args.magpie_gripper_xml
    )
    _, magpie_qpos, magpie_width_m = interpolate_magpie_qpos(
        requested_width_m,
        commands,
        lookup_qpos,
        lookup_gaps,
    )
    local_anchors, local_frames = wrist_local_magpie_trajectory(
        args.magpie_xml,
        magpie_qpos,
    )
    initial_magpie_wrist_rotation = align_axis_near_reference(
        rh56_targets[0, :3, :3],
        local_frames[0, :, 0],
        desired_world_frame[:, 0],
    )
    magpie_world_frame = initial_magpie_wrist_rotation @ local_frames[0]
    magpie_targets = magpie_wrist_targets(
        local_anchors,
        local_frames,
        desired_world_frame=magpie_world_frame,
        target_pelvis=target_pelvis,
    )
    print("[practical] solving Magpie compensated trajectory...")
    magpie_solution = solve_h12_ik(
        magpie_targets,
        initial_iters=args.ik_initial_iters,
        step_iters=args.ik_step_iters,
    )

    rh56_model = mujoco.MjModel.from_xml_path(str(args.rh56_xml))
    rh56_data = mujoco.MjData(rh56_model)
    mujoco.mj_forward(rh56_model, rh56_data)
    pelvis_id = named_id(rh56_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    target_world = rh56_data.xpos[pelvis_id].copy() + target_pelvis

    conditions = [
        build_condition(
            name="magpie_compensated",
            model_path=args.magpie_xml,
            arm_q=magpie_solution.arm_q,
            command_width_mm=1000.0 * magpie_width_m,
            target_world=target_world,
            tip_sites=MAGPIE_TIP_SITES,
            ik_position_error_mm=magpie_solution.position_error_mm,
            ik_orientation_error_deg=magpie_solution.orientation_error_deg,
            magpie_qpos=magpie_qpos,
        ),
        build_condition(
            name="rh56_practical_compensated",
            model_path=args.rh56_xml,
            arm_q=rh56_solution.arm_q,
            command_width_mm=rh56_command_width_mm,
            target_world=target_world,
            tip_sites=RH56_TIP_SITES,
            ik_position_error_mm=rh56_solution.position_error_mm,
            ik_orientation_error_deg=rh56_solution.orientation_error_deg,
            rh56_results=rh56_results,
            rh56_fk=fk,
        ),
    ]

    total_masses = {}
    for condition in conditions:
        total_masses[condition.name] = evaluate_static_balance(
            condition,
            sole_tolerance_m=args.foot_sole_tolerance_mm / 1000.0,
        )

    metadata = {
        "width_range_mm": [lower_mm, upper_mm],
        "requested_target_width_mm": requested_target_mm,
        "provisional_cutoff_mm": cutoff_mm,
        "realized_target_width_mm": realized_target_mm,
        "start_width_mm": start_mm,
        "target_pelvis_m": target_pelvis.tolist(),
        "target_world_m": target_world.tolist(),
        "plane_rz_deg": args.plane_rz_deg,
        "magpie_lookup_command_range_rad": [
            float(commands[0]),
            float(commands[-1]),
        ],
        "magpie_lookup_gap_range_mm": [
            1000.0 * float(lookup_gaps[-1]),
            1000.0 * float(lookup_gaps[0]),
        ],
        "model_total_mass_kg": total_masses,
    }
    return conditions, metadata


def plot_results(path: Path, conditions) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    metric_specs = (
        ("wrist_path_mm", "Cumulative wrist path (mm)"),
        ("wrist_rotation_path_deg", "Cumulative wrist rotation (deg)"),
        ("arm_joint_travel_rad", "Cumulative arm joint travel (rad)"),
        ("support_margin_mm", "Change in static CoM margin (mm)"),
    )
    colors = {
        "magpie_compensated": "#0072b2",
        "rh56_practical_compensated": "#d55e00",
    }
    for axis, (attribute, ylabel) in zip(axes.ravel(), metric_specs):
        for condition in conditions:
            values = getattr(condition, attribute)
            if attribute == "support_margin_mm":
                # Compare motion-induced change, not the different hands' raw
                # starting mass distributions and initial IK postures.
                values = values - values[0]
            axis.plot(
                condition.actual_width_mm,
                values,
                color=colors[condition.name],
                linewidth=2.4,
                label=CONDITION_LABELS[condition.name],
            )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_xlim(
        max(float(condition.actual_width_mm.max()) for condition in conditions),
        min(float(condition.actual_width_mm.min()) for condition in conditions),
    )
    for axis in axes[1]:
        axis.set_xlabel("Actual fingertip reference width (mm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    figure.suptitle(
        "H12 compensated closure: Magpie vs RH56 provisional cutoff",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    metadata: dict,
    video_path: Path | None,
) -> None:
    payload = {
        "script": "tools/run_h12_practical_gripper_comparison.py",
        "simulation_only": True,
        "uses_hardware": False,
        "purpose": (
            "Compare the H12 arm motion required to hold the same grasp center "
            "and pinch frame during Magpie and practical RH56 closure."
        ),
        "conditions": list(PRACTICAL_CONDITIONS),
        "fixed_states": ["floating base", "legs", "torso", "left arm"],
        "shared_constraints": [
            "same H12 base family",
            "same target grasp center",
            "same world pinch/contact axis",
            "same start and realized final fingertip reference widths",
            "each hand's initial wrist roll is chosen near the same H12 reference pose",
            "each hand's grasp center and complete pinch frame remain stationary thereafter",
        ],
        "rh56_practical_policy": {
            "status": "provisional; not the mentor's exact controller value",
            "requested_target_width_mm": metadata["requested_target_width_mm"],
            "cutoff_width_mm": metadata["provisional_cutoff_mm"],
            "realized_target_width_mm": metadata["realized_target_width_mm"],
            "action_at_cutoff": (
                "Stop the RH56 finger command and end the closure trajectory; "
                "do not command mathematical full closure."
            ),
            "basis": (
                "The prior sensitivity sweep detected a wrist-rotation knee at "
                "approximately 38.1 mm; 40 mm is a conservative reproducible "
                "pre-knee value and matches the checked-in H12 demo default."
            ),
        },
        "static_balance": {
            "metric": (
                "Signed horizontal whole-body CoM distance to the nearest edge "
                "of the convex double-support polygon; positive means inside."
            ),
            "whole_body_com": "MuJoCo body inertial-position mass-weighted mean",
            "support_polygon": (
                "Convex hull of bottom vertices from both collidable foot meshes"
            ),
            "foot_sole_tolerance_mm": args.foot_sole_tolerance_mm,
            "model_total_mass_kg": metadata["model_total_mass_kg"],
            "interpretation": (
                "Quasi-static geometric proxy only; it is not a dynamic balance "
                "or standing-controller stability result."
            ),
        },
        "models": {
            "magpie": str(args.magpie_xml.resolve()),
            "rh56": str(args.rh56_xml.resolve()),
            "isolated_magpie_gripper": str(args.magpie_gripper_xml.resolve()),
        },
        "ik": {
            "solver": "PINK differential IK with non-right-arm joints locked",
            "initial_iterations": args.ik_initial_iters,
            "step_iterations": args.ik_step_iters,
        },
        "grasp_and_rendering": {
            **metadata,
            "visual_screw_is_a_non-contact_proxy": True,
            "video_path": str(video_path.resolve()) if video_path else None,
            "fps": args.fps,
            "moving_frames": args.frames,
            "hold_frames": args.hold_frames,
            "camera_azimuth_deg": args.camera_azimuth,
        },
        "not_evaluated": [
            "dynamic balance or standing-controller recovery",
            "contact force, friction, or grasp success",
            "screw extraction force",
            "mentor's exact hardcoded cutoff and final pose",
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)

    conditions, metadata = prepare_conditions(args)
    write_summary(args.out / "summary.csv", conditions)
    write_trajectory(args.out / "trajectory.csv", conditions, fps=args.fps)
    plot_results(args.out / "comparison_metrics.png", conditions)
    print_summary(conditions)

    video_path = None
    if not args.no_video:
        print("[practical] rendering front-view comparison video...")
        video_path = render_video(conditions, args=args)
        print(f"[practical] video: {video_path}")

    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        metadata=metadata,
        video_path=video_path,
    )
    print(f"[practical] results: {args.out}")

    if args.live:
        run_live(conditions, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
