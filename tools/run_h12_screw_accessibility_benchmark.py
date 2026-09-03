#!/usr/bin/env python3
"""Map H12 screw accessibility with Magpie and RH56 in a surrogate workcell.

The repository does not currently contain Hyundai Ioniq 5 battery-pack CAD.
This simulation-only benchmark therefore uses an explicitly parameterized
surrogate: a horizontal pack surface, a loosened screw, and a low crowned cover
whose near slope starts shortly beyond the screw.  The cover is represented by
two thin inclined plates meeting at a ridge.  It separates two system-level
error sources:

1. base-placement error with an oracle screw pose; and
2. screw-localization error at the nominal base pose.

For each hand and trial, the script solves approach, compensated closure, and
vertical extraction kinematics, then evaluates reachability, right-arm/hand
collision with the surrogate battery, arm joint margin, fingertip closure,
and quasi-static CoM support margin.  It writes summary.csv, trials.csv,
assumptions.json, plots, and an optional representative front-view video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
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
from rh56_controller.grasp_viz_workers import _H12_ARM_JOINTS  # noqa: E402
from tools.run_h12_gripper_motion_comparison import (  # noqa: E402
    CONDITION_COLORS,
    DEFAULT_MAGPIE_GRIPPER_XML,
    DEFAULT_MAGPIE_XML,
    DEFAULT_RH56_XML,
    MAGPIE_TIP_SITES,
    RH56_TIP_SITES,
    add_text,
    align_axis_near_reference,
    build_condition,
    build_rh56_wrist_targets,
    interpolate_magpie_qpos,
    joint_qpos_address,
    named_id,
    settle_magpie_lookup,
    solve_h12_ik,
    wrist_local_magpie_trajectory,
)
from tools.run_h12_practical_gripper_comparison import (  # noqa: E402
    evaluate_static_balance,
    magpie_wrist_targets,
)


HAND_LABELS = {
    "magpie": "Magpie",
    "rh56": "RH56 (40 mm cutoff)",
}

OUTCOME_ORDER = (
    "feasible",
    "unreachable",
    "battery_collision",
    "joint_limit",
    "perception_miss",
    "insufficient_closure",
    "balance_margin",
)

OUTCOME_COLORS = {
    "feasible": "#2ca25f",
    "unreachable": "#756bb1",
    "battery_collision": "#de2d26",
    "joint_limit": "#fd8d3c",
    "perception_miss": "#3182bd",
    "insufficient_closure": "#e6ab02",
    "balance_margin": "#636363",
}


@dataclass(frozen=True)
class TrialSpec:
    scan: str
    base_dx_m: float
    base_dy_m: float
    perception_dx_m: float
    perception_dy_m: float
    perception_dz_m: float = 0.0
    cover_edge_gap_mm: float | None = None

    @property
    def perception_error_mm(self) -> float:
        return 1000.0 * float(np.linalg.norm([
            self.perception_dx_m,
            self.perception_dy_m,
            self.perception_dz_m,
        ]))


@dataclass
class HandTemplate:
    hand: str
    model_path: Path
    tip_sites: tuple[str, str]
    closure_width_mm: np.ndarray
    rh56_results: list | None
    rh56_fk: InspireHandFK | None
    magpie_qpos: np.ndarray | None
    magpie_local_anchors: np.ndarray | None
    magpie_local_frames: np.ndarray | None
    magpie_world_frame: np.ndarray | None


@dataclass(frozen=True)
class WorkcellPose:
    screw_world_m: np.ndarray
    pack_center_world_m: np.ndarray
    pack_top_z_m: float
    cover_slope_centers_world_m: tuple[np.ndarray, np.ndarray]
    cover_slope_quaternions_wxyz: tuple[np.ndarray, np.ndarray]


@dataclass
class TrialResult:
    hand: str
    spec: TrialSpec
    outcome: str
    condition: object
    phases: tuple[str, ...]
    actual_screw_world_m: np.ndarray
    estimated_target_world_m: np.ndarray
    workcell: WorkcellPose
    tracking_error_mm: float
    screw_center_error_mm: float
    min_clearance_mm: float
    min_clearance_body: str
    min_clearance_phase: str
    min_joint_margin_fraction: float
    min_support_margin_mm: float
    max_horizontal_com_shift_mm: float
    closure_sufficient: bool
    reachable: bool
    collision_free: bool
    joint_margin_ok: bool
    perception_ok: bool
    balance_ok: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map Magpie/RH56 screw accessibility under H12 base-placement "
            "and perception error using a parameterized battery surrogate."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/h12_screw_accessibility"),
    )
    parser.add_argument("--rh56-xml", type=Path, default=DEFAULT_RH56_XML)
    parser.add_argument("--magpie-xml", type=Path, default=DEFAULT_MAGPIE_XML)
    parser.add_argument(
        "--magpie-gripper-xml",
        type=Path,
        default=DEFAULT_MAGPIE_GRIPPER_XML,
    )
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument("--rh56-cutoff-mm", type=float, default=40.0)
    parser.add_argument("--start-width-mm", type=float, default=None)
    parser.add_argument("--closure-frames", type=int, default=13)
    parser.add_argument("--approach-frames", type=int, default=5)
    parser.add_argument("--extraction-frames", type=int, default=5)
    parser.add_argument("--approach-distance-mm", type=float, default=80.0)
    parser.add_argument("--extraction-distance-mm", type=float, default=50.0)
    parser.add_argument("--grasp-x", type=float, default=0.45)
    parser.add_argument("--grasp-y", type=float, default=-0.20)
    parser.add_argument("--grasp-z", type=float, default=0.15)
    parser.add_argument(
        "--plane-rz-deg",
        type=float,
        default=-150.0,
        help=(
            "Shared horizontal pinch-axis orientation. -150 deg is the best "
            "reachable RH56 candidate from a 30-deg nominal sweep at the "
            "configured screw pose; it remains a provisional planning value."
        ),
    )
    parser.add_argument("--base-error-mm", type=float, default=150.0)
    parser.add_argument("--base-grid-size", type=int, default=5)
    parser.add_argument(
        "--perception-errors-mm",
        type=float,
        nargs="+",
        default=(0.0, 5.0, 25.0, 50.0),
    )
    parser.add_argument("--perception-directions", type=int, default=4)
    parser.add_argument("--capture-radius-mm", type=float, default=10.0)
    parser.add_argument("--screw-head-diameter-mm", type=float, default=20.0)
    parser.add_argument("--fingertip-pad-radius-mm", type=float, default=10.5)
    parser.add_argument("--screw-head-height-mm", type=float, default=6.0)
    parser.add_argument(
        "--screw-center-above-pack-mm",
        type=float,
        default=25.0,
    )
    parser.add_argument("--pack-depth-m", type=float, default=1.20)
    parser.add_argument("--pack-width-m", type=float, default=2.00)
    parser.add_argument("--pack-thickness-mm", type=float, default=80.0)
    parser.add_argument("--screw-inset-from-edge-mm", type=float, default=100.0)
    parser.add_argument(
        "--cover-edge-gap-mm",
        "--obstacle-gap-mm",
        dest="cover_edge_gap_mm",
        type=float,
        default=15.0,
        help=(
            "Clear distance from the screw-head edge to the near edge of the "
            "sloped cover."
        ),
    )
    parser.add_argument(
        "--cover-edge-gap-sweep-mm",
        "--obstacle-gap-sweep-mm",
        dest="cover_edge_gap_sweep_mm",
        type=float,
        nargs="+",
        default=(5.0, 10.0, 15.0, 20.0, 30.0, 40.0),
        help=(
            "Clear distances from the screw-head edge to the near edge of the "
            "low crowned cover."
        ),
    )
    parser.add_argument("--cover-width-mm", type=float, default=120.0)
    parser.add_argument(
        "--cover-slope-run-mm",
        type=float,
        default=15.0,
        help="Horizontal run of each half of the crowned cover.",
    )
    parser.add_argument(
        "--cover-rise-mm",
        type=float,
        default=10.0,
        help="Ridge height above the low cover edges.",
    )
    parser.add_argument("--cover-thickness-mm", type=float, default=2.0)
    parser.add_argument("--collision-clearance-mm", type=float, default=0.0)
    parser.add_argument(
        "--ik-position-tolerance-mm",
        type=float,
        default=10.0,
        help=(
            "Maximum path IK position error. This is stricter than the checked-in "
            "battery grasp servo's 25 mm acceptance tolerance."
        ),
    )
    parser.add_argument("--ik-orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--joint-margin-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum normalized arm-joint reserve required for feasibility. "
            "The default rejects only hard-limit violations while still "
            "reporting zero reserve; use 0.03 for a conservative 3%% reserve."
        ),
    )
    parser.add_argument("--ik-initial-iters", type=int, default=300)
    parser.add_argument("--ik-step-iters", type=int, default=18)
    parser.add_argument("--foot-sole-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=24)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--panel-height", type=int, default=420)
    parser.add_argument("--camera-distance", type=float, default=0.95)
    parser.add_argument("--camera-azimuth", type=float, default=150.0)
    parser.add_argument("--camera-elevation", type=float, default=-28.0)
    parser.add_argument(
        "--video-format",
        choices=("auto", "mp4", "gif"),
        default="auto",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.base_grid_size < 2:
        raise ValueError("--base-grid-size must be at least 2")
    if args.base_error_mm < 0.0:
        raise ValueError("--base-error-mm cannot be negative")
    if args.perception_directions < 1:
        raise ValueError("--perception-directions must be positive")
    if any(value < 0.0 for value in args.perception_errors_mm):
        raise ValueError("--perception-errors-mm cannot contain negatives")
    if any(value <= 0.0 for value in args.cover_edge_gap_sweep_mm):
        raise ValueError("--cover-edge-gap-sweep-mm values must be positive")
    if args.closure_frames < 2:
        raise ValueError("--closure-frames must be at least 2")
    if args.approach_frames < 1 or args.extraction_frames < 1:
        raise ValueError("approach/extraction frame counts must be positive")
    if args.rh56_cutoff_mm <= 0.0:
        raise ValueError("--rh56-cutoff-mm must be positive")
    if args.pack_depth_m <= 0.0 or args.pack_width_m <= 0.0:
        raise ValueError("pack dimensions must be positive")
    if min(
        args.cover_width_mm,
        args.cover_slope_run_mm,
        args.cover_rise_mm,
        args.cover_thickness_mm,
    ) <= 0.0:
        raise ValueError("sloped-cover dimensions must be positive")
    if args.ik_initial_iters < 1 or args.ik_step_iters < 1:
        raise ValueError("IK iteration counts must be positive")


def build_trial_specs(args: argparse.Namespace) -> list[TrialSpec]:
    extent_m = args.base_error_mm / 1000.0
    grid = np.linspace(-extent_m, extent_m, args.base_grid_size)
    specs = [
        TrialSpec("base_placement", float(dx), float(dy), 0.0, 0.0)
        for dy in grid
        for dx in grid
    ]
    for error_mm in sorted(set(float(value) for value in args.perception_errors_mm)):
        radius_m = error_mm / 1000.0
        if radius_m == 0.0:
            specs.append(TrialSpec("perception", 0.0, 0.0, 0.0, 0.0))
            continue
        for direction in range(args.perception_directions):
            angle = 2.0 * math.pi * direction / args.perception_directions
            specs.append(TrialSpec(
                "perception",
                0.0,
                0.0,
                radius_m * math.cos(angle),
                radius_m * math.sin(angle),
            ))
    for gap_mm in sorted(set(float(value) for value in args.cover_edge_gap_sweep_mm)):
        specs.append(TrialSpec(
            "cover_edge_gap",
            0.0,
            0.0,
            0.0,
            0.0,
            cover_edge_gap_mm=gap_mm,
        ))
    return specs


def combine_motion_targets(
    closure_targets: np.ndarray,
    *,
    approach_frames: int,
    extraction_frames: int,
    approach_distance_m: float,
    extraction_distance_m: float,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    approach_offsets = np.linspace(
        approach_distance_m,
        0.0,
        approach_frames + 1,
    )[:-1]
    approach = np.repeat(closure_targets[:1], approach_frames, axis=0)
    approach[:, 2, 3] += approach_offsets
    extraction_offsets = np.linspace(
        0.0,
        extraction_distance_m,
        extraction_frames + 1,
    )[1:]
    extraction = np.repeat(closure_targets[-1:], extraction_frames, axis=0)
    extraction[:, 2, 3] += extraction_offsets
    targets = np.concatenate((approach, closure_targets, extraction), axis=0)
    phases = (
        ("approach",) * approach_frames
        + ("closure",) * len(closure_targets)
        + ("extraction",) * extraction_frames
    )
    closure_indices = np.arange(
        approach_frames,
        approach_frames + len(closure_targets),
    )
    return targets, phases, closure_indices


def expand_hand_states(values: list | np.ndarray, args: argparse.Namespace):
    first = values[0]
    last = values[-1]
    if isinstance(values, np.ndarray):
        return np.concatenate((
            np.repeat(values[:1], args.approach_frames, axis=0),
            values,
            np.repeat(values[-1:], args.extraction_frames, axis=0),
        ))
    return [first] * args.approach_frames + list(values) + [last] * args.extraction_frames


def build_hand_templates(args: argparse.Namespace) -> tuple[dict[str, HandTemplate], dict]:
    for path in (args.rh56_xml, args.magpie_xml, args.magpie_gripper_xml):
        if not path.exists():
            raise FileNotFoundError(path)

    fk = InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    width_range = closure.width_range(str(GraspMode.LINE_2F), n_fingers=2)
    lower_mm, upper_mm = 1000.0 * width_range[0], 1000.0 * width_range[1]
    cutoff_mm = float(np.clip(args.rh56_cutoff_mm, lower_mm, upper_mm))
    start_mm = (
        upper_mm
        if args.start_width_mm is None
        else float(np.clip(args.start_width_mm, cutoff_mm, upper_mm))
    )
    requested_width_m = np.linspace(start_mm, cutoff_mm, args.closure_frames) / 1000.0
    rh56_results = [
        closure.solve(GraspMode.LINE_2F, float(width))
        for width in requested_width_m
    ]
    rh56_width_mm = 1000.0 * np.asarray([result.width for result in rh56_results])

    print("[accessibility] building quasi-static Magpie closure lookup...")
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

    nominal_target = np.array([args.grasp_x, args.grasp_y, args.grasp_z])
    rh56_nominal_targets, desired_world_frame = build_rh56_wrist_targets(
        rh56_results,
        target_pelvis=nominal_target,
        plane_rz_rad=math.radians(args.plane_rz_deg),
    )
    initial_magpie_wrist_rotation = align_axis_near_reference(
        rh56_nominal_targets[0, :3, :3],
        local_frames[0, :, 0],
        desired_world_frame[:, 0],
    )
    magpie_world_frame = initial_magpie_wrist_rotation @ local_frames[0]

    templates = {
        "magpie": HandTemplate(
            hand="magpie",
            model_path=args.magpie_xml,
            tip_sites=MAGPIE_TIP_SITES,
            closure_width_mm=1000.0 * magpie_width_m,
            rh56_results=None,
            rh56_fk=None,
            magpie_qpos=magpie_qpos,
            magpie_local_anchors=local_anchors,
            magpie_local_frames=local_frames,
            magpie_world_frame=magpie_world_frame,
        ),
        "rh56": HandTemplate(
            hand="rh56",
            model_path=args.rh56_xml,
            tip_sites=RH56_TIP_SITES,
            closure_width_mm=rh56_width_mm,
            rh56_results=rh56_results,
            rh56_fk=fk,
            magpie_qpos=None,
            magpie_local_anchors=None,
            magpie_local_frames=None,
            magpie_world_frame=None,
        ),
    }
    metadata = {
        "rh56_width_range_mm": [lower_mm, upper_mm],
        "closure_start_width_mm": start_mm,
        "closure_cutoff_mm": cutoff_mm,
        "magpie_lookup_command_range_rad": [float(commands[0]), float(commands[-1])],
        "magpie_lookup_gap_range_mm": [
            1000.0 * float(lookup_gaps[-1]),
            1000.0 * float(lookup_gaps[0]),
        ],
    }
    return templates, metadata


def workcell_pose(
    args: argparse.Namespace,
    *,
    pelvis_world_m: np.ndarray,
    spec: TrialSpec,
) -> WorkcellPose:
    nominal_screw = pelvis_world_m + np.array([
        args.grasp_x,
        args.grasp_y,
        args.grasp_z,
    ])
    workcell_shift = np.array([-spec.base_dx_m, -spec.base_dy_m, 0.0])
    screw = nominal_screw + workcell_shift
    pack_top_z = screw[2] - args.screw_center_above_pack_mm / 1000.0
    near_edge_x = nominal_screw[0] - args.screw_inset_from_edge_mm / 1000.0
    pack_center = np.array([
        near_edge_x + args.pack_depth_m / 2.0,
        0.0,
        pack_top_z - args.pack_thickness_mm / 2000.0,
    ]) + workcell_shift
    cover_edge_gap_mm = (
        args.cover_edge_gap_mm
        if spec.cover_edge_gap_mm is None
        else spec.cover_edge_gap_mm
    )
    screw_radius_m = args.screw_head_diameter_mm / 2000.0
    near_edge_x = screw[0] + screw_radius_m + cover_edge_gap_mm / 1000.0
    run_m = args.cover_slope_run_mm / 1000.0
    rise_m = args.cover_rise_mm / 1000.0
    thickness_m = args.cover_thickness_mm / 1000.0
    slope_angle = math.atan2(rise_m, run_m)
    slope_center_z = pack_top_z + 0.5 * (rise_m + thickness_m)
    centers = (
        np.array([near_edge_x + 0.5 * run_m, screw[1], slope_center_z]),
        np.array([near_edge_x + 1.5 * run_m, screw[1], slope_center_z]),
    )
    quaternions = (
        np.array([math.cos(-0.5 * slope_angle), 0.0, math.sin(-0.5 * slope_angle), 0.0]),
        np.array([math.cos(0.5 * slope_angle), 0.0, math.sin(0.5 * slope_angle), 0.0]),
    )
    return WorkcellPose(
        screw_world_m=screw,
        pack_center_world_m=pack_center,
        pack_top_z_m=float(pack_top_z),
        cover_slope_centers_world_m=centers,
        cover_slope_quaternions_wxyz=quaternions,
    )


def add_surrogate_geoms(spec_model: mujoco.MjSpec, args: argparse.Namespace, pose: WorkcellPose) -> None:
    pack = spec_model.worldbody.add_geom()
    pack.name = "surrogate_pack"
    pack.type = mujoco.mjtGeom.mjGEOM_BOX
    pack.size = np.array([
        args.pack_depth_m / 2.0,
        args.pack_width_m / 2.0,
        args.pack_thickness_mm / 2000.0,
    ])
    pack.pos = pose.pack_center_world_m
    pack.rgba = np.array([0.13, 0.20, 0.25, 1.0])
    pack.contype = 1
    pack.conaffinity = 1

    slope_length_m = math.hypot(
        args.cover_slope_run_mm,
        args.cover_rise_mm,
    ) / 1000.0
    for name, center, quaternion in zip(
        ("surrogate_cover_near_slope", "surrogate_cover_far_slope"),
        pose.cover_slope_centers_world_m,
        pose.cover_slope_quaternions_wxyz,
    ):
        slope = spec_model.worldbody.add_geom()
        slope.name = name
        slope.type = mujoco.mjtGeom.mjGEOM_BOX
        slope.size = np.array([
            slope_length_m / 2.0,
            args.cover_width_mm / 2000.0,
            args.cover_thickness_mm / 2000.0,
        ])
        slope.pos = center
        slope.quat = quaternion
        slope.rgba = np.array([0.22, 0.48, 0.62, 1.0])
        slope.contype = 1
        slope.conaffinity = 1

    shaft = spec_model.worldbody.add_geom()
    shaft.name = "surrogate_screw_shaft"
    shaft.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    shaft.size = np.array([
        args.screw_head_diameter_mm / 6000.0,
        max(0.001, (pose.screw_world_m[2] - pose.pack_top_z_m) / 2.0),
        0.0,
    ])
    shaft.pos = np.array([
        pose.screw_world_m[0],
        pose.screw_world_m[1],
        0.5 * (pose.screw_world_m[2] + pose.pack_top_z_m),
    ])
    shaft.rgba = np.array([0.48, 0.48, 0.52, 1.0])
    shaft.contype = 0
    shaft.conaffinity = 0

    head = spec_model.worldbody.add_geom()
    head.name = "surrogate_screw_head"
    head.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    head.size = np.array([
        args.screw_head_diameter_mm / 2000.0,
        args.screw_head_height_mm / 2000.0,
        0.0,
    ])
    head.pos = pose.screw_world_m
    head.rgba = np.array([0.88, 0.72, 0.16, 1.0])
    head.contype = 0
    head.conaffinity = 0


def build_environment_model(
    model_path: Path,
    args: argparse.Namespace,
    pose: WorkcellPose,
) -> mujoco.MjModel:
    spec_model = mujoco.MjSpec.from_file(str(model_path))
    add_surrogate_geoms(spec_model, args, pose)
    return spec_model.compile()


def body_is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == root_id:
            return True
        current = int(model.body_parentid[current])
    return False


def collision_metrics(
    model: mujoco.MjModel,
    qpos_rows: np.ndarray,
    phases: tuple[str, ...],
) -> tuple[float, str, str]:
    obstacle_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "surrogate_pack",
            "surrogate_cover_near_slope",
            "surrogate_cover_far_slope",
        )
    ]
    root_id = named_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "right_shoulder_pitch_link",
    )
    robot_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_contype[geom_id]) != 0
        and body_is_descendant(
            model,
            int(model.geom_bodyid[geom_id]),
            root_id,
        )
    ]
    data = mujoco.MjData(model)
    minimum = float("inf")
    minimum_body = ""
    minimum_phase = ""
    for frame_index, qpos in enumerate(qpos_rows):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for robot_geom in robot_geoms:
            for obstacle_geom in obstacle_ids:
                distance = float(mujoco.mj_geomDistance(
                    model,
                    data,
                    robot_geom,
                    obstacle_geom,
                    0.25,
                    None,
                ))
                if distance < minimum:
                    minimum = distance
                    body_id = int(model.geom_bodyid[robot_geom])
                    minimum_body = (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        or f"body_{body_id}"
                    )
                    minimum_phase = phases[frame_index]
    return 1000.0 * minimum, minimum_body, minimum_phase


def arm_joint_margin_fraction(model_path: Path, arm_q: np.ndarray) -> float:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    margins = []
    for joint_index, joint_name in enumerate(_H12_ARM_JOINTS):
        joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        lower, upper = model.jnt_range[joint_id]
        span = float(upper - lower)
        if span <= 0.0:
            continue
        values = arm_q[:, joint_index]
        margins.extend(np.minimum(values - lower, upper - values) / span)
    return float(np.min(margins))


def classify_outcome(
    *,
    reachable: bool,
    collision_free: bool,
    joint_margin_ok: bool,
    perception_ok: bool,
    closure_sufficient: bool,
    balance_ok: bool,
) -> str:
    if not reachable:
        return "unreachable"
    if not collision_free:
        return "battery_collision"
    if not joint_margin_ok:
        return "joint_limit"
    if not perception_ok:
        return "perception_miss"
    if not closure_sufficient:
        return "insufficient_closure"
    if not balance_ok:
        return "balance_margin"
    return "feasible"


def template_targets(
    template: HandTemplate,
    estimated_target_pelvis_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, object]:
    if template.hand == "rh56":
        closure_targets, _ = build_rh56_wrist_targets(
            template.rh56_results,
            target_pelvis=estimated_target_pelvis_m,
            plane_rz_rad=math.radians(args.plane_rz_deg),
        )
        hand_states = expand_hand_states(template.rh56_results, args)
    else:
        closure_targets = magpie_wrist_targets(
            template.magpie_local_anchors,
            template.magpie_local_frames,
            desired_world_frame=template.magpie_world_frame,
            target_pelvis=estimated_target_pelvis_m,
        )
        hand_states = expand_hand_states(template.magpie_qpos, args)
    targets, phases, _ = combine_motion_targets(
        closure_targets,
        approach_frames=args.approach_frames,
        extraction_frames=args.extraction_frames,
        approach_distance_m=args.approach_distance_mm / 1000.0,
        extraction_distance_m=args.extraction_distance_mm / 1000.0,
    )
    widths = np.concatenate((
        np.repeat(template.closure_width_mm[:1], args.approach_frames),
        template.closure_width_mm,
        np.repeat(template.closure_width_mm[-1:], args.extraction_frames),
    ))
    return targets, phases, widths, hand_states


def evaluate_trial(
    template: HandTemplate,
    trial_spec: TrialSpec,
    args: argparse.Namespace,
    *,
    pelvis_world_m: np.ndarray,
    environment_cache: dict,
) -> TrialResult:
    actual_target_pelvis = np.array([
        args.grasp_x - trial_spec.base_dx_m,
        args.grasp_y - trial_spec.base_dy_m,
        args.grasp_z,
    ])
    perception_offset = np.array([
        trial_spec.perception_dx_m,
        trial_spec.perception_dy_m,
        trial_spec.perception_dz_m,
    ])
    estimated_target_pelvis = actual_target_pelvis + perception_offset
    targets, phases, command_widths, hand_states = template_targets(
        template,
        estimated_target_pelvis,
        args,
    )
    solution = solve_h12_ik(
        targets,
        initial_iters=args.ik_initial_iters,
        step_iters=args.ik_step_iters,
    )
    estimated_target_world = pelvis_world_m + estimated_target_pelvis
    actual_screw_world = pelvis_world_m + actual_target_pelvis
    if template.hand == "rh56":
        condition = build_condition(
            name="rh56_practical_compensated",
            model_path=template.model_path,
            arm_q=solution.arm_q,
            command_width_mm=command_widths,
            target_world=estimated_target_world,
            tip_sites=template.tip_sites,
            ik_position_error_mm=solution.position_error_mm,
            ik_orientation_error_deg=solution.orientation_error_deg,
            rh56_results=hand_states,
            rh56_fk=template.rh56_fk,
        )
    else:
        condition = build_condition(
            name="magpie_compensated",
            model_path=template.model_path,
            arm_q=solution.arm_q,
            command_width_mm=command_widths,
            target_world=estimated_target_world,
            tip_sites=template.tip_sites,
            ik_position_error_mm=solution.position_error_mm,
            ik_orientation_error_deg=solution.orientation_error_deg,
            magpie_qpos=hand_states,
        )
    evaluate_static_balance(
        condition,
        sole_tolerance_m=args.foot_sole_tolerance_mm / 1000.0,
    )

    pose = workcell_pose(args, pelvis_world_m=pelvis_world_m, spec=trial_spec)
    cache_key = (
        template.hand,
        round(trial_spec.base_dx_m, 9),
        round(trial_spec.base_dy_m, 9),
        round(
            args.cover_edge_gap_mm
            if trial_spec.cover_edge_gap_mm is None
            else trial_spec.cover_edge_gap_mm,
            9,
        ),
    )
    if cache_key not in environment_cache:
        environment_cache[cache_key] = build_environment_model(
            template.model_path,
            args,
            pose,
        )
    clearance_mm, clearance_body, clearance_phase = collision_metrics(
        environment_cache[cache_key],
        condition.qpos,
        phases,
    )

    tracking_error = float(solution.position_error_mm.max())
    # Evaluate screw capture at the end of closure, before the commanded
    # extraction deliberately moves the grasp center away from the fixed visual
    # screw proxy.
    closure_end_index = args.approach_frames + args.closure_frames - 1
    final_screw_error = 1000.0 * float(np.linalg.norm(
        condition.anchor_world[closure_end_index] - actual_screw_world
    ))
    joint_margin = arm_joint_margin_fraction(template.model_path, solution.arm_q)
    com_shift = 1000.0 * np.linalg.norm(
        condition.com_world[:, :2] - condition.com_world[0, :2],
        axis=1,
    )
    closure_limit_mm = (
        args.screw_head_diameter_mm + 2.0 * args.fingertip_pad_radius_mm
    )
    closure_sufficient = bool(condition.actual_width_mm[-1] <= closure_limit_mm)
    reachable = bool(
        tracking_error <= args.ik_position_tolerance_mm
        and float(solution.orientation_error_deg.max())
        <= args.ik_orientation_tolerance_deg
    )
    collision_free = bool(clearance_mm >= args.collision_clearance_mm)
    joint_margin_ok = bool(
        joint_margin >= args.joint_margin_fraction - 1e-6
    )
    perception_ok = bool(final_screw_error <= args.capture_radius_mm)
    balance_ok = bool(float(condition.support_margin_mm.min()) > 0.0)
    outcome = classify_outcome(
        reachable=reachable,
        collision_free=collision_free,
        joint_margin_ok=joint_margin_ok,
        perception_ok=perception_ok,
        closure_sufficient=closure_sufficient,
        balance_ok=balance_ok,
    )
    return TrialResult(
        hand=template.hand,
        spec=trial_spec,
        outcome=outcome,
        condition=condition,
        phases=phases,
        actual_screw_world_m=actual_screw_world,
        estimated_target_world_m=estimated_target_world,
        workcell=pose,
        tracking_error_mm=tracking_error,
        screw_center_error_mm=final_screw_error,
        min_clearance_mm=clearance_mm,
        min_clearance_body=clearance_body,
        min_clearance_phase=clearance_phase,
        min_joint_margin_fraction=joint_margin,
        min_support_margin_mm=float(condition.support_margin_mm.min()),
        max_horizontal_com_shift_mm=float(com_shift.max()),
        closure_sufficient=closure_sufficient,
        reachable=reachable,
        collision_free=collision_free,
        joint_margin_ok=joint_margin_ok,
        perception_ok=perception_ok,
        balance_ok=balance_ok,
    )


def trial_row(result: TrialResult) -> dict[str, object]:
    condition = result.condition
    return {
        "hand": result.hand,
        "hand_label": HAND_LABELS[result.hand],
        "scan": result.spec.scan,
        "base_dx_mm": 1000.0 * result.spec.base_dx_m,
        "base_dy_mm": 1000.0 * result.spec.base_dy_m,
        "perception_dx_mm": 1000.0 * result.spec.perception_dx_m,
        "perception_dy_mm": 1000.0 * result.spec.perception_dy_m,
        "perception_dz_mm": 1000.0 * result.spec.perception_dz_m,
        "perception_error_mm": result.spec.perception_error_mm,
        "cover_edge_gap_mm": (
            result.spec.cover_edge_gap_mm
            if result.spec.cover_edge_gap_mm is not None
            else ""
        ),
        "outcome": result.outcome,
        "feasible": result.outcome == "feasible",
        "reachable": result.reachable,
        "collision_free": result.collision_free,
        "joint_margin_ok": result.joint_margin_ok,
        "perception_ok": result.perception_ok,
        "closure_sufficient": result.closure_sufficient,
        "balance_ok": result.balance_ok,
        "max_ik_position_error_mm": result.tracking_error_mm,
        "max_ik_orientation_error_deg": float(
            condition.ik_orientation_error_deg.max()
        ),
        "final_screw_center_error_mm": result.screw_center_error_mm,
        "min_battery_clearance_mm": result.min_clearance_mm,
        "min_clearance_body": result.min_clearance_body,
        "min_clearance_phase": result.min_clearance_phase,
        "min_joint_margin_fraction": result.min_joint_margin_fraction,
        "final_width_mm": float(condition.actual_width_mm[-1]),
        "wrist_path_mm": float(condition.wrist_path_mm[-1]),
        "wrist_rotation_deg": float(condition.wrist_rotation_path_deg[-1]),
        "arm_joint_travel_rad": float(condition.arm_joint_travel_rad[-1]),
        "min_static_support_margin_mm": result.min_support_margin_mm,
        "max_horizontal_com_shift_mm": result.max_horizontal_com_shift_mm,
    }


def write_trials(path: Path, results: list[TrialResult]) -> None:
    rows = [trial_row(result) for result in results]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(results: list[TrialResult]) -> list[dict[str, object]]:
    rows = []
    for hand in HAND_LABELS:
        for scan in ("base_placement", "perception", "cover_edge_gap"):
            subset = [
                result for result in results
                if result.hand == hand and result.spec.scan == scan
            ]
            counts = {
                outcome: sum(result.outcome == outcome for result in subset)
                for outcome in OUTCOME_ORDER
            }
            rows.append({
                "hand": hand,
                "hand_label": HAND_LABELS[hand],
                "scan": scan,
                "trials": len(subset),
                "feasible_trials": counts["feasible"],
                "feasible_rate": counts["feasible"] / len(subset),
                **{f"{outcome}_count": counts[outcome] for outcome in OUTCOME_ORDER[1:]},
                "worst_ik_position_error_mm": max(
                    result.tracking_error_mm for result in subset
                ),
                "minimum_battery_clearance_mm": min(
                    result.min_clearance_mm for result in subset
                ),
                "minimum_joint_margin_fraction": min(
                    result.min_joint_margin_fraction for result in subset
                ),
                "minimum_static_support_margin_mm": min(
                    result.min_support_margin_mm for result in subset
                ),
            })
    return rows


def write_summary(path: Path, results: list[TrialResult]) -> None:
    rows = summary_rows(results)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_base_map(path: Path, results: list[TrialResult]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    base_results = [result for result in results if result.spec.scan == "base_placement"]
    x_values = sorted({1000.0 * result.spec.base_dx_m for result in base_results})
    y_values = sorted({1000.0 * result.spec.base_dy_m for result in base_results})
    outcome_to_code = {name: index for index, name in enumerate(OUTCOME_ORDER)}
    cmap = ListedColormap([OUTCOME_COLORS[name] for name in OUTCOME_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(OUTCOME_ORDER) + 0.5), cmap.N)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), sharex=True, sharey=True)
    for axis, hand in zip(axes, HAND_LABELS):
        matrix = np.zeros((len(y_values), len(x_values)), dtype=int)
        lookup = {
            (1000.0 * result.spec.base_dx_m, 1000.0 * result.spec.base_dy_m): result
            for result in base_results if result.hand == hand
        }
        for row, y_value in enumerate(y_values):
            for column, x_value in enumerate(x_values):
                matrix[row, column] = outcome_to_code[
                    lookup[(x_value, y_value)].outcome
                ]
        axis.imshow(
            matrix,
            origin="lower",
            cmap=cmap,
            norm=norm,
            extent=(
                min(x_values) - 0.5 * (x_values[1] - x_values[0]),
                max(x_values) + 0.5 * (x_values[1] - x_values[0]),
                min(y_values) - 0.5 * (y_values[1] - y_values[0]),
                max(y_values) + 0.5 * (y_values[1] - y_values[0]),
            ),
            interpolation="nearest",
            aspect="equal",
        )
        axis.axhline(0.0, color="white", linewidth=0.8, alpha=0.7)
        axis.axvline(0.0, color="white", linewidth=0.8, alpha=0.7)
        axis.set_title(HAND_LABELS[hand])
        axis.set_xlabel("Robot base X placement error (mm)")
        axis.grid(False)
    axes[0].set_ylabel("Robot base Y placement error (mm)")
    handles = [
        Patch(facecolor=OUTCOME_COLORS[name], label=name.replace("_", " "))
        for name in OUTCOME_ORDER
        if any(result.outcome == name for result in base_results)
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle("H12 screw accessibility under base-placement error")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def perception_success_rates(results: list[TrialResult]) -> dict[str, dict[float, float]]:
    output = {}
    for hand in HAND_LABELS:
        hand_results = [
            result for result in results
            if result.hand == hand and result.spec.scan == "perception"
        ]
        rates = {}
        for error in sorted({result.spec.perception_error_mm for result in hand_results}):
            subset = [
                result for result in hand_results
                if abs(result.spec.perception_error_mm - error) < 1e-8
            ]
            rates[error] = sum(
                result.outcome == "feasible" for result in subset
            ) / len(subset)
        output[hand] = rates
    return output


def plot_perception(path: Path, results: list[TrialResult], capture_radius_mm: float) -> None:
    import matplotlib.pyplot as plt

    rates = perception_success_rates(results)
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    colors = {"magpie": "#0072b2", "rh56": "#d55e00"}
    styles = {
        "magpie": {"marker": "o", "linestyle": "-"},
        "rh56": {"marker": "s", "linestyle": "--"},
    }
    for hand in HAND_LABELS:
        errors = np.asarray(sorted(rates[hand]))
        values = np.asarray([rates[hand][error] for error in errors])
        axis.plot(
            errors,
            100.0 * values,
            **styles[hand],
            linewidth=2.3,
            color=colors[hand],
            label=HAND_LABELS[hand],
        )
    axis.axvline(
        capture_radius_mm,
        color="#333333",
        linestyle="--",
        linewidth=1.3,
        label="configured capture radius",
    )
    axis.set_xlabel("Injected horizontal screw-localization error (mm)")
    axis.set_ylabel("Feasible trial rate across error directions (%)")
    axis.set_ylim(-3.0, 103.0)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Nominal-base robustness to target localization error")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_cover_edge_gap(path: Path, results: list[TrialResult]) -> None:
    import matplotlib.pyplot as plt

    subset = [result for result in results if result.spec.scan == "cover_edge_gap"]
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    colors = {"magpie": "#0072b2", "rh56": "#d55e00"}
    styles = {
        "magpie": {"marker": "o", "linestyle": "-"},
        "rh56": {"marker": "s", "linestyle": "--"},
    }
    for hand in HAND_LABELS:
        hand_results = sorted(
            (result for result in subset if result.hand == hand),
            key=lambda result: result.spec.cover_edge_gap_mm,
        )
        axis.plot(
            [result.spec.cover_edge_gap_mm for result in hand_results],
            [100.0 if result.collision_free else 0.0 for result in hand_results],
            linewidth=2.3,
            color=colors[hand],
            label=HAND_LABELS[hand],
            **styles[hand],
        )
    axis.set_xlabel("Screw-head edge to sloped-cover edge gap (mm)")
    axis.set_ylabel("Collision-free in the tested trajectory (0 or 100%)")
    axis.set_ylim(-3.0, 103.0)
    axis.set_title("Low crowned-cover collision screening")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def select_representative_pair(results: list[TrialResult]) -> tuple[TrialResult, TrialResult]:
    base_results = [result for result in results if result.spec.scan == "base_placement"]
    by_key = {
        (result.hand, result.spec.base_dx_m, result.spec.base_dy_m): result
        for result in base_results
    }
    keys = sorted({
        (result.spec.base_dx_m, result.spec.base_dy_m)
        for result in base_results
    })
    for dx, dy in keys:
        magpie = by_key[("magpie", dx, dy)]
        rh56 = by_key[("rh56", dx, dy)]
        if magpie.outcome == "feasible" and rh56.outcome != "feasible":
            return magpie, rh56
    nominal = min(keys, key=lambda item: abs(item[0]) + abs(item[1]))
    return by_key[("magpie", *nominal)], by_key[("rh56", *nominal)]


def video_extension(video_format: str) -> str:
    if video_format == "gif":
        return ".gif"
    if video_format == "mp4":
        return ".mp4"
    try:
        import imageio_ffmpeg  # noqa: F401
        return ".mp4"
    except ImportError:
        return ".gif"


def render_video(
    pair: tuple[TrialResult, TrialResult],
    *,
    args: argparse.Namespace,
) -> Path:
    extension = video_extension(args.video_format)
    path = args.out / f"representative_trial{extension}"
    models = [
        build_environment_model(result.condition.model_path, args, result.workcell)
        for result in pair
    ]
    data_rows = [mujoco.MjData(model) for model in models]
    renderers = [
        mujoco.Renderer(model, height=args.panel_height, width=args.panel_width)
        for model in models
    ]
    cameras = []
    for result in pair:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = result.actual_screw_world_m
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        cameras.append(camera)
    total_frames = len(pair[0].condition.qpos) + args.hold_frames

    writer = None
    gif_frames = []
    if extension == ".mp4":
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            str(path),
            (args.panel_width * 2, args.panel_height),
            fps=args.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
        )
        writer.send(None)
    try:
        for output_index in range(total_frames):
            panels = []
            for result, model, data, renderer, camera in zip(
                pair, models, data_rows, renderers, cameras
            ):
                frame_index = min(output_index, len(result.condition.qpos) - 1)
                data.qpos[:] = result.condition.qpos[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                panel = add_text(renderer.render(), [
                    f"{HAND_LABELS[result.hand]} | {result.outcome}",
                    (
                        f"base error "
                        f"({1000.0 * result.spec.base_dx_m:+.0f}, "
                        f"{1000.0 * result.spec.base_dy_m:+.0f}) mm"
                    ),
                    f"phase {result.phases[frame_index]}",
                    f"battery clearance {result.min_clearance_mm:.1f} mm",
                    f"max IK error {result.tracking_error_mm:.1f} mm",
                    f"static CoM margin {result.min_support_margin_mm:.1f} mm",
                ])
                panels.append(panel)
            combined = np.ascontiguousarray(np.concatenate(panels, axis=1))
            if writer is not None:
                writer.send(combined)
            else:
                gif_frames.append(combined)
    finally:
        if writer is not None:
            writer.close()
        for renderer in renderers:
            renderer.close()
    if extension == ".gif":
        from PIL import Image
        images = [Image.fromarray(frame).convert("RGB") for frame in gif_frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(round(1000.0 / args.fps))),
            loop=0,
        )
    return path


def live_worker(result: TrialResult, args: argparse.Namespace) -> None:
    os.environ.pop("MUJOCO_GL", None)
    import mujoco.viewer

    model = build_environment_model(result.condition.model_path, args, result.workcell)
    data = mujoco.MjData(model)
    frame_period = 1.0 / float(args.fps)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = result.actual_screw_world_m
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = args.camera_elevation
        while viewer.is_running():
            for qpos in result.condition.qpos:
                if not viewer.is_running():
                    return
                start = time.monotonic()
                data.qpos[:] = qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                remaining = frame_period - (time.monotonic() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
            if args.once:
                while viewer.is_running():
                    time.sleep(0.05)
                return


def run_live(pair: tuple[TrialResult, TrialResult], args: argparse.Namespace) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=live_worker, args=(result, args))
        for result in pair
    ]
    for process in processes:
        process.start()
    print("[accessibility] live viewers started; close both windows to stop")
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    template_metadata: dict,
    results: list[TrialResult],
    representative_pair: tuple[TrialResult, TrialResult],
    video_path: Path | None,
) -> None:
    payload = {
        "script": "tools/run_h12_screw_accessibility_benchmark.py",
        "simulation_only": True,
        "uses_hardware": False,
        "study_status": "surrogate-workcell screening experiment",
        "asset_audit": {
            "h12_battery_pack_geometry_found": False,
            "note": (
                "No Hyundai Ioniq 5 battery pack/workcell MuJoCo, URDF, USD, "
                "or CAD asset was found in the checked workspace. The unrelated "
                "Apptronik Apollo battery_mount_fix.stl is not used."
            ),
            "rh56_h12_mujoco_camera_count": 0,
            "magpie_h12_mujoco_cameras": [
                "head_cam",
                "leftg_hand_cam",
                "hand_cam",
            ],
        },
        "purpose": (
            "Separate screw-access failures caused by robot base placement, "
            "target localization, hand-specific compensation, collision, arm "
            "reach/joint limits, and quasi-static balance."
        ),
        "scans": {
            "base_placement": {
                "grid_size": args.base_grid_size,
                "range_mm": [-args.base_error_mm, args.base_error_mm],
                "perception_error_mm": 0.0,
            },
            "perception": {
                "base_error_mm": [0.0, 0.0],
                "horizontal_error_radii_mm": sorted(set(args.perception_errors_mm)),
                "directions_per_nonzero_radius": args.perception_directions,
                "injection_method": (
                    "Direct 3-D target-position offset after detection; image "
                    "formation, depth segmentation, occlusion, and calibration "
                    "are not rendered in this first-stage benchmark."
                ),
            },
            "sloped_cover_edge_gap": {
                "base_error_mm": [0.0, 0.0],
                "perception_error_mm": 0.0,
                "clear_gaps_mm": sorted(set(args.cover_edge_gap_sweep_mm)),
                "note": (
                    "Gap is measured from the screw-head edge to the near low "
                    "edge of a two-slope crowned-cover proxy along robot +x. "
                    "This is not Hyundai battery-pack CAD."
                ),
            },
        },
        "surrogate_battery": {
            "not_real_pack_cad": True,
            "pack_depth_m": args.pack_depth_m,
            "pack_width_m": args.pack_width_m,
            "pack_thickness_mm": args.pack_thickness_mm,
            "screw_inset_from_near_edge_mm": args.screw_inset_from_edge_mm,
            "screw_head_diameter_mm": args.screw_head_diameter_mm,
            "screw_center_above_pack_mm": args.screw_center_above_pack_mm,
            "cover_edge_gap_mm": args.cover_edge_gap_mm,
            "cover_width_mm": args.cover_width_mm,
            "cover_slope_run_mm_each_side": args.cover_slope_run_mm,
            "cover_rise_mm": args.cover_rise_mm,
            "cover_thickness_mm": args.cover_thickness_mm,
            "layout": (
                "robot -> screw -> short gap -> low crowned cover along +x"
            ),
        },
        "motion": {
            "nominal_target_pelvis_m": [args.grasp_x, args.grasp_y, args.grasp_z],
            "approach_distance_mm": args.approach_distance_mm,
            "extraction_distance_mm": args.extraction_distance_mm,
            "approach_frames": args.approach_frames,
            "closure_frames": args.closure_frames,
            "extraction_frames": args.extraction_frames,
            "plane_rz_deg": args.plane_rz_deg,
            "both_hands_hold_grasp_center_and_pinch_frame_during_closure": True,
            "screw_is_not_dynamically_lifted": True,
            **template_metadata,
        },
        "classification_thresholds": {
            "capture_radius_mm": args.capture_radius_mm,
            "collision_clearance_mm": args.collision_clearance_mm,
            "ik_position_tolerance_mm": args.ik_position_tolerance_mm,
            "ik_orientation_tolerance_deg": args.ik_orientation_tolerance_deg,
            "joint_margin_fraction": args.joint_margin_fraction,
            "fingertip_pad_radius_mm": args.fingertip_pad_radius_mm,
            "closure_sufficient_when_width_at_most_mm": (
                args.screw_head_diameter_mm + 2.0 * args.fingertip_pad_radius_mm
            ),
            "outcome_priority": list(OUTCOME_ORDER[1:]),
        },
        "static_balance": {
            "metric": (
                "Signed horizontal whole-body CoM distance to the convex "
                "double-support polygon."
            ),
            "dynamic_stability_evaluated": False,
        },
        "representative_trial": {
            "base_dx_mm": 1000.0 * representative_pair[0].spec.base_dx_m,
            "base_dy_mm": 1000.0 * representative_pair[0].spec.base_dy_m,
            "magpie_outcome": representative_pair[0].outcome,
            "rh56_outcome": representative_pair[1].outcome,
            "video_path": str(video_path.resolve()) if video_path else None,
        },
        "trial_count": len(results),
        "not_evaluated": [
            "real battery-pack CAD collision fidelity",
            "rendered head-camera detection or calibration",
            "closed-loop hand-camera visual servo",
            "contact force, friction, or screw retention",
            "off-axis jamming in the screw bore",
            "dynamic standing-controller recovery",
        ],
        "paper_reference": str(
            Path(
                "/home/tanxuan/Downloads/"
                "HUMANOIDS_2026___Generalized_Open_Library_of_Embodied_Modules.pdf"
            )
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def print_summary(results: list[TrialResult]) -> None:
    for row in summary_rows(results):
        print(
            f"[accessibility] {row['hand']} {row['scan']}: "
            f"feasible={row['feasible_trials']}/{row['trials']} "
            f"({100.0 * row['feasible_rate']:.1f}%), "
            f"worst IK={row['worst_ik_position_error_mm']:.1f} mm, "
            f"min clearance={row['minimum_battery_clearance_mm']:.1f} mm"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    templates, template_metadata = build_hand_templates(args)
    specs = build_trial_specs(args)

    reference_model = mujoco.MjModel.from_xml_path(str(args.rh56_xml))
    reference_data = mujoco.MjData(reference_model)
    mujoco.mj_forward(reference_model, reference_data)
    pelvis_id = named_id(reference_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_world = reference_data.xpos[pelvis_id].copy()

    environment_cache = {}
    results = []
    total = len(specs) * len(templates)
    completed = 0
    for spec in specs:
        for template in templates.values():
            completed += 1
            print(
                f"[accessibility] trial {completed}/{total}: {template.hand} "
                f"{spec.scan} base=({1000.0 * spec.base_dx_m:+.0f},"
                f"{1000.0 * spec.base_dy_m:+.0f}) mm "
                f"perception={spec.perception_error_mm:.0f} mm"
                + (
                    f" gap={spec.cover_edge_gap_mm:.0f} mm"
                    if spec.cover_edge_gap_mm is not None
                    else ""
                )
            )
            results.append(evaluate_trial(
                template,
                spec,
                args,
                pelvis_world_m=pelvis_world,
                environment_cache=environment_cache,
            ))

    write_trials(args.out / "trials.csv", results)
    write_summary(args.out / "summary.csv", results)
    plot_base_map(args.out / "base_accessibility_map.png", results)
    plot_perception(
        args.out / "perception_robustness.png",
        results,
        args.capture_radius_mm,
    )
    plot_cover_edge_gap(args.out / "sloped_cover_clearance.png", results)
    print_summary(results)

    representative_pair = select_representative_pair(results)
    video_path = None
    if not args.no_video:
        print("[accessibility] rendering representative front-view video...")
        video_path = render_video(representative_pair, args=args)
        print(f"[accessibility] video: {video_path}")
    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        template_metadata=template_metadata,
        results=results,
        representative_pair=representative_pair,
        video_path=video_path,
    )
    print(f"[accessibility] results: {args.out}")
    if args.live:
        run_live(representative_pair, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
