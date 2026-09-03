#!/usr/bin/env python3
"""Run the 15-object x 10-point RH56 grasp-success comparison in MuJoCo.

The trial grid is imported from the earlier RH56 capsule-collision experiment:
P1, P2, and the eight L1/L2 lateral points.  Each trial moves the open hand
from that exact grasp-center offset to its method-specific final grasp, closes
with a shared contact-limited executor, lifts 200 mm at 0.1 m/s, and holds.

The object geometries and masses are exploratory proxies.  This script cannot
score damage to delicate objects and therefore does not reproduce the physical
paper experiment's calibrated 6 N / LLM-estimated force conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK  # noqa: E402
from rh56_controller.graspgenx_baseline import (  # noqa: E402
    GraspGenXCandidate,
    graspgenx_to_mujoco_base_pose,
    load_isaac_grasp_yaml,
)
from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    ObjectSpec,
    tabletop_aabb_center,
    tabletop_grasp_target,
)
from tools.run_graspgenx_success_comparison import (  # noqa: E402
    BASE_ACTUATORS,
    FINGER_ACTUATORS,
    GGX_CLOSE_CTRL,
    GGX_OPEN_CTRL,
    PoseCommand,
    _actuator_ids,
    _analytical_closing_axis,
    _analytical_pose,
    _floor_hand_collision,
    _has_opposing_contacts,
    _initialize_state,
    _lerp_dict,
    _object_hand_contacts,
    _object_position,
    _write_command,
)
from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS  # noqa: E402
from tools.run_strategy_pregrasp_rate import (  # noqa: E402
    approach_basis,
    facing_yaw_for_approach_axis,
    paper_approach_points_mm,
)


DEFAULT_XML = REPO_ROOT / "h1_mujoco/inspire/inspire_grasp_scene.xml"
METHODS = ("iterative", "graspgenx")
DELICATE_OBJECTS = set(PAPER_OBJECTS[-5:])

# Estimated simulation masses only. Replace with measured values before using
# this benchmark for a paper claim.
PAPER_OBJECT_MASS_KG = {
    "paper_big_screwdriver": 0.100,
    "paper_bottle": 0.050,
    "paper_can": 0.100,
    "paper_charger": 0.100,
    "paper_metal_cup": 0.080,
    "paper_mustard": 0.200,
    "paper_orange": 0.150,
    "paper_pen": 0.010,
    "paper_small_screwdriver": 0.040,
    "paper_sugar_box": 0.200,
    "paper_egg": 0.060,
    "paper_nut": 0.010,
    "paper_paper_cup": 0.010,
    "paper_raspberry": 0.004,
    "paper_strawberry": 0.015,
}


@dataclass(frozen=True)
class GridPoint:
    point_id: str
    order: int
    approach_distance_mm: float
    lateral_mm: float
    height_mm: float
    offset_m: np.ndarray


@dataclass(frozen=True)
class RunConfig:
    xml: str
    analytical_yaw_deg: float
    approach_speed_m_s: float
    settle_s: float
    close_s: float
    post_contact_s: float
    lift_m: float
    lift_speed_m_s: float
    hold_s: float
    success_lift_m: float
    contact_compression: float
    opposing_span_fraction: float
    iterative_preopen_m: float
    object_width_offset_m: float
    force_target_n: float
    force_gain_alpha_per_n_s: float


@dataclass
class TrialResult:
    object: str
    label: str
    group: str
    method: str
    point_id: str
    point_order: int
    approach_distance_mm: float
    lateral_mm: float
    height_mm: float
    success: bool
    failure_mode: str
    final_lift_mm: float
    max_lift_mm: float
    max_object_xy_displacement_mm: float
    preclose_object_displacement_mm: float
    opposing_contact_detected: bool
    first_opposing_contact_alpha: float | None
    stopped_closure_alpha: float
    approach_object_contact: bool
    approach_floor_contact: bool
    final_hand_contact_count: int
    final_opposing_contact: bool
    final_object_floor_contact: bool
    final_thumb_normal_force_n: float
    final_opposing_normal_force_n: float
    force_target_n: float
    candidate_rank: int | None
    candidate_name: str
    candidate_confidence: float | None


_WORKER_CLOSURE: ClosureGeometry | None = None
FrameCallback = Callable[[mujoco.MjModel, mujoco.MjData, str], None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-style iterative analytical closure and pretrained GraspGen-X "
            "over the 15 RH56 paper objects and the existing P1-P10 grid."
        )
    )
    parser.add_argument("--objects", nargs="+", choices=PAPER_OBJECTS, default=PAPER_OBJECTS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/graspgenx_candidates"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/results"),
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--points", nargs="+", default=None, help="Optional point IDs, e.g. P1 P2 L1_d-50.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--approach-axis", choices=["x-", "x+", "y-", "y+"], default="y-")
    parser.add_argument("--paper-hand-yaw-offset-deg", type=float, default=-90.0)
    parser.add_argument("--paper-p1-distance-mm", type=float, default=250.0)
    parser.add_argument("--paper-p2-height-mm", type=float, default=250.0)
    parser.add_argument("--paper-level1-height-mm", type=float, default=100.0)
    parser.add_argument("--paper-level2-height-mm", type=float, default=250.0)
    parser.add_argument("--paper-d-list-mm", type=float, nargs="+", default=[-150.0, -50.0, 50.0, 150.0])
    parser.add_argument("--approach-speed-m-s", type=float, default=0.25)
    parser.add_argument("--settle-s", type=float, default=0.15)
    parser.add_argument("--close-s", type=float, default=1.8)
    parser.add_argument("--post-contact-s", type=float, default=0.50)
    parser.add_argument("--lift-mm", type=float, default=200.0)
    parser.add_argument("--lift-speed-m-s", type=float, default=0.10)
    parser.add_argument("--hold-s", type=float, default=0.8)
    parser.add_argument(
        "--success-lift-mm",
        type=float,
        default=180.0,
        help=(
            "Minimum final object-center lift while the object remains in an "
            "opposing grasp. The default requires 90%% of the commanded 200 mm "
            "lift, allowing a small simulation/contact tolerance."
        ),
    )
    parser.add_argument(
        "--contact-compression",
        type=float,
        default=0.10,
        help=(
            "Normalized closure preload when force control first sees opposing "
            "contact. The 0.10 default is the shared simulation calibration used "
            "by both methods; set zero for a purely feedback-driven transition."
        ),
    )
    parser.add_argument("--opposing-span-fraction", type=float, default=0.30)
    parser.add_argument("--force-target-n", type=float, default=6.0)
    parser.add_argument(
        "--force-gain-alpha-per-n-s",
        type=float,
        default=0.08,
        help="One-sided integral gain for maintaining the MuJoCo normal-force target.",
    )
    parser.add_argument(
        "--iterative-preopen-mm",
        type=float,
        default=10.0,
        help=(
            "Width offset above the target for the paper Iterative pre-grasp. "
            "The 10 mm default matches the 38-to-28 mm example in Fig. 4."
        ),
    )
    parser.add_argument(
        "--object-width-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Optional calibration offset from physical object width to the "
            "planner's internal fingertip-site width. The dynamic benchmark "
            "defaults to zero; the capsule-only tools' 20 mm proxy correction "
            "must not be assumed to calibrate MuJoCo contact geometry."
        ),
    )
    parser.add_argument(
        "--topdown-cos",
        type=float,
        default=None,
        help=(
            "Optional downward-approach cosine filter for GraspGen-X. By default "
            "the highest-confidence floor-clear candidate is used because the "
            "paper object set includes side grasps such as the upright bottle."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for name in (
        "approach_speed_m_s",
        "close_s",
        "lift_mm",
        "lift_speed_m_s",
        "success_lift_mm",
        "force_target_n",
        "force_gain_alpha_per_n_s",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.success_lift_mm > args.lift_mm:
        raise ValueError("--success-lift-mm cannot exceed --lift-mm")
    if not 0.0 <= args.contact_compression <= 0.25:
        raise ValueError("--contact-compression must be in [0, 0.25]")
    if not 0.0 < args.opposing_span_fraction <= 1.0:
        raise ValueError("--opposing-span-fraction must be in (0, 1]")
    if args.iterative_preopen_mm <= 0.0:
        raise ValueError("--iterative-preopen-mm must be positive")
    if args.object_width_offset_mm < 0.0:
        raise ValueError("--object-width-offset-mm must be non-negative")
    if args.topdown_cos is not None and not 0.0 <= args.topdown_cos <= 1.0:
        raise ValueError("--topdown-cos must be in [0, 1]")


def build_grid_points(args: argparse.Namespace) -> list[GridPoint]:
    grid_args = SimpleNamespace(
        paper_p1_distance_mm=args.paper_p1_distance_mm,
        paper_p2_height_mm=args.paper_p2_height_mm,
        paper_level1_height_mm=args.paper_level1_height_mm,
        paper_level2_height_mm=args.paper_level2_height_mm,
        paper_d_list_mm=args.paper_d_list_mm,
    )
    raw_points = paper_approach_points_mm(grid_args)
    approach_vector, lateral_vector = approach_basis(args.approach_axis)
    points: list[GridPoint] = []
    for raw in raw_points:
        distance_mm = float(raw["approach_distance_mm"])
        lateral_mm = float(raw["d_mm"])
        height_mm = float(raw["h_mm"])
        offset = (
            approach_vector * distance_mm / 1000.0
            + lateral_vector * lateral_mm / 1000.0
            + np.array([0.0, 0.0, height_mm / 1000.0])
        )
        points.append(
            GridPoint(
                point_id=str(raw["point_id"]),
                order=int(float(raw["order"])),
                approach_distance_mm=distance_mm,
                lateral_mm=lateral_mm,
                height_mm=height_mm,
                offset_m=offset,
            )
        )
    if args.points is not None:
        requested = set(args.points)
        known = {point.point_id for point in points}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown point IDs: {sorted(unknown)}; available: {sorted(known)}")
        points = [point for point in points if point.point_id in requested]
    return points


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build the shared dynamic-execution configuration from CLI arguments."""

    analytical_yaw_deg = (
        facing_yaw_for_approach_axis(args.approach_axis)
        + args.paper_hand_yaw_offset_deg
    )
    return RunConfig(
        xml=str(args.xml.resolve()),
        analytical_yaw_deg=analytical_yaw_deg,
        approach_speed_m_s=args.approach_speed_m_s,
        settle_s=args.settle_s,
        close_s=args.close_s,
        post_contact_s=args.post_contact_s,
        lift_m=args.lift_mm / 1000.0,
        lift_speed_m_s=args.lift_speed_m_s,
        hold_s=args.hold_s,
        success_lift_m=args.success_lift_mm / 1000.0,
        contact_compression=args.contact_compression,
        opposing_span_fraction=args.opposing_span_fraction,
        iterative_preopen_m=args.iterative_preopen_mm / 1000.0,
        object_width_offset_m=args.object_width_offset_mm / 1000.0,
        force_target_n=args.force_target_n,
        force_gain_alpha_per_n_s=args.force_gain_alpha_per_n_s,
    )


def _add_object_model(
    xml_path: str | Path,
    obj: ObjectSpec,
    mass_kg: float,
) -> tuple[mujoco.MjModel, int, int]:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    body = spec.worldbody.add_body(name="benchmark_object", pos=[0.0, 0.0, 0.0])
    body.add_freejoint(name="benchmark_object_free")
    if obj.collision_shape == "box":
        geom_type = mujoco.mjtGeom.mjGEOM_BOX
        geom_size = [value / 2.0 for value in obj.size_m]
    elif obj.collision_shape == "cylinder":
        geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom_size = [min(obj.size_m[0], obj.size_m[1]) / 2.0, obj.size_m[2] / 2.0, 0.0]
    elif obj.collision_shape == "sphere":
        geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom_size = [min(obj.size_m) / 2.0, 0.0, 0.0]
    else:
        raise ValueError(f"Unsupported object shape: {obj.collision_shape}")
    color = [0.96, 0.55, 0.08, 1.0] if obj.name in DELICATE_OBJECTS else [0.12, 0.55, 0.92, 1.0]
    body.add_geom(
        name="benchmark_object_geom",
        type=geom_type,
        size=geom_size,
        mass=mass_kg,
        rgba=color,
        friction=[1.5, 0.01, 0.001],
        condim=6,
    )
    model = spec.compile()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "benchmark_object_geom")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "benchmark_object_free")
    return model, int(geom_id), int(model.jnt_qposadr[joint_id])


def _solve_mode(closure: ClosureGeometry, mode: str, width_m: float):
    if mode == "line":
        return closure.line(width_m)
    if mode.startswith("plane"):
        return closure.plane(width_m, n_fingers=int(mode[-1]))
    raise ValueError(f"Unsupported analytical mode: {mode}")


def _width_range(closure: ClosureGeometry, mode: str) -> tuple[float, float]:
    if mode == "line":
        return closure.width_range("2-finger line", n_fingers=2)
    if mode.startswith("plane"):
        n_fingers = int(mode[-1])
        return closure.width_range(f"{n_fingers}-finger plane", n_fingers=n_fingers)
    raise ValueError(f"Unsupported analytical mode: {mode}")


def _select_candidate(
    *,
    config: RunConfig,
    obj: ObjectSpec,
    candidates: list[GraspGenXCandidate],
    topdown_cos: float | None,
) -> tuple[int, GraspGenXCandidate] | None:
    model, object_geom_id, object_qadr = _add_object_model(
        config.xml, obj, PAPER_OBJECT_MASS_KG[obj.name]
    )
    data = mujoco.MjData(model)
    base_ids = _actuator_ids(model, BASE_ACTUATORS)
    finger_ids = _actuator_ids(model, FINGER_ACTUATORS)
    object_center = tabletop_aabb_center(obj)
    for rank, candidate in enumerate(candidates):
        if topdown_cos is not None and candidate.approach_direction[2] > -topdown_cos:
            continue
        position, rotation = graspgenx_to_mujoco_base_pose(
            candidate, object_center=object_center
        )
        command = PoseCommand(position, rotation, dict(GGX_OPEN_CTRL))
        _initialize_state(
            model,
            data,
            object_qadr,
            object_center,
            command,
            base_ids,
            finger_ids,
        )
        if (
            not _floor_hand_collision(model, data, object_geom_id)
            and not _object_hand_contacts(model, data, object_geom_id)
        ):
            return rank, candidate
    return None


def _get_worker_closure() -> ClosureGeometry:
    global _WORKER_CLOSURE
    if _WORKER_CLOSURE is None:
        _WORKER_CLOSURE = ClosureGeometry(InspireHandFK())
    return _WORKER_CLOSURE


def _object_floor_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_id: int,
) -> bool:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if {int(contact.geom1), int(contact.geom2)} == {floor_id, object_geom_id}:
            if contact.dist <= 1e-5:
                return True
    return False


def _contact_side_normal_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_id: int,
) -> tuple[float, float]:
    """Return summed thumb-side and opposing-finger normal contact forces."""

    thumb_force = 0.0
    opposing_force = 0.0
    wrench = np.zeros(6, dtype=float)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if object_geom_id not in (int(contact.geom1), int(contact.geom2)):
            continue
        other_geom = int(contact.geom2) if int(contact.geom1) == object_geom_id else int(contact.geom1)
        other_body = int(model.geom_bodyid[other_geom])
        if other_body == 0:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
        wrench[:] = 0.0
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        normal_force = abs(float(wrench[0]))
        if body_name.startswith("thumb"):
            thumb_force += normal_force
        elif body_name.startswith(("index", "middle", "ring", "pinky")):
            opposing_force += normal_force
    return thumb_force, opposing_force


def _execute_trial(
    *,
    config: RunConfig,
    obj: ObjectSpec,
    method: str,
    point: GridPoint,
    candidate_selection: tuple[int, GraspGenXCandidate] | None,
    model: mujoco.MjModel,
    object_geom_id: int,
    object_qadr: int,
    frame_callback: FrameCallback | None = None,
) -> TrialResult:
    data = mujoco.MjData(model)
    base_ids = _actuator_ids(model, BASE_ACTUATORS)
    finger_ids = _actuator_ids(model, FINGER_ACTUATORS)
    object_center = tabletop_aabb_center(obj)

    if method == "iterative":
        grasp_target = tabletop_grasp_target(obj)
        closure = _get_worker_closure()
        minimum, maximum = _width_range(closure, obj.mode)
        internal_target_width = obj.grasp_width_m + config.object_width_offset_m
        target_width = float(np.clip(internal_target_width, minimum, maximum))
        approach_width = min(maximum, target_width + config.iterative_preopen_m)
        open_result = _solve_mode(closure, obj.mode, approach_width)
        final_result = _solve_mode(closure, obj.mode, target_width)
        yaw_rad = math.radians(config.analytical_yaw_deg)
        target_open = _analytical_pose(open_result, grasp_target, yaw_rad)
        initial_position = target_open.base_position + point.offset_m

        def approach_command(alpha: float) -> PoseCommand:
            return PoseCommand(
                (1.0 - alpha) * initial_position + alpha * target_open.base_position,
                target_open.base_rotation,
                target_open.finger_ctrl,
            )

        initial = approach_command(0.0)

        def close_command(alpha: float) -> PoseCommand:
            # Once the arm has reached the local pre-grasp, execute the paper's
            # width-space closure about a fixed grasp point. Continue past the
            # nominal target only until opposing contact plus compression.
            width = (1.0 - alpha) * approach_width + alpha * minimum
            return _analytical_pose(
                _solve_mode(closure, obj.mode, float(width)),
                grasp_target,
                yaw_rad,
            )

        closing_axis = _analytical_closing_axis(
            final_result, close_command(1.0).base_rotation
        )
        candidate_rank = None
        candidate_name = f"iterative_{obj.mode}"
        candidate_confidence = None
    else:
        grasp_target = object_center.copy()
        if candidate_selection is None:
            return TrialResult(
                object=obj.name,
                label=obj.label,
                group="delicate" if obj.name in DELICATE_OBJECTS else "ycb",
                method=method,
                point_id=point.point_id,
                point_order=point.order,
                approach_distance_mm=point.approach_distance_mm,
                lateral_mm=point.lateral_mm,
                height_mm=point.height_mm,
                success=False,
                failure_mode="no_executable_candidate",
                final_lift_mm=0.0,
                max_lift_mm=0.0,
                max_object_xy_displacement_mm=0.0,
                preclose_object_displacement_mm=0.0,
                opposing_contact_detected=False,
                first_opposing_contact_alpha=None,
                stopped_closure_alpha=1.0,
                approach_object_contact=False,
                approach_floor_contact=False,
                final_hand_contact_count=0,
                final_opposing_contact=False,
                final_object_floor_contact=True,
                final_thumb_normal_force_n=0.0,
                final_opposing_normal_force_n=0.0,
                force_target_n=config.force_target_n,
                candidate_rank=None,
                candidate_name="",
                candidate_confidence=None,
            )
        candidate_rank, candidate = candidate_selection
        target_position, target_rotation = graspgenx_to_mujoco_base_pose(
            candidate, object_center=object_center
        )
        target_open = PoseCommand(target_position, target_rotation, dict(GGX_OPEN_CTRL))
        initial_position = target_position + point.offset_m

        def approach_command(alpha: float) -> PoseCommand:
            return PoseCommand(
                (1.0 - alpha) * initial_position + alpha * target_position,
                target_rotation,
                dict(GGX_OPEN_CTRL),
            )

        initial = approach_command(0.0)

        def close_command(alpha: float) -> PoseCommand:
            return PoseCommand(
                target_position,
                target_rotation,
                _lerp_dict(GGX_OPEN_CTRL, GGX_CLOSE_CTRL, alpha),
            )

        closing_axis = candidate.rotation[:, 0].copy()
        closing_axis /= np.linalg.norm(closing_axis)
        candidate_name = candidate.name
        candidate_confidence = candidate.confidence

    _initialize_state(
        model,
        data,
        object_qadr,
        object_center,
        initial,
        base_ids,
        finger_ids,
    )
    if frame_callback is not None:
        frame_callback(model, data, "initial")
    initial_object_position = object_center.copy()
    initial_z = float(object_center[2])
    max_z = initial_z
    max_xy = 0.0
    approach_object_contact = False
    approach_floor_contact = False
    first_contact_alpha: float | None = None
    stop_alpha = 1.0
    force_alpha: float | None = None
    timestep = float(model.opt.timestep)

    def update_metrics() -> None:
        nonlocal max_z, max_xy
        position = _object_position(data, object_qadr)
        max_z = max(max_z, float(position[2]))
        max_xy = max(max_xy, float(np.linalg.norm(position[:2] - initial_object_position[:2])))

    def step_for(
        duration: float,
        command_fn: Callable[[float], PoseCommand],
        phase: str,
    ) -> None:
        start_time = float(data.time)
        while data.time - start_time < duration - 1e-12:
            alpha = min(1.0, (data.time - start_time) / max(duration, 1e-9))
            _write_command(data, base_ids, finger_ids, command_fn(alpha))
            mujoco.mj_step(model, data)
            update_metrics()
            if frame_callback is not None:
                frame_callback(model, data, phase)

    step_for(config.settle_s, lambda _alpha: initial, "settle")
    approach_duration = max(
        0.05,
        float(np.linalg.norm(point.offset_m)) / config.approach_speed_m_s,
    )
    approach_start = float(data.time)
    while data.time - approach_start < approach_duration - 1e-12:
        alpha = min(1.0, (data.time - approach_start) / approach_duration)
        command = approach_command(alpha)
        _write_command(data, base_ids, finger_ids, command)
        mujoco.mj_step(model, data)
        update_metrics()
        if frame_callback is not None:
            frame_callback(model, data, "approach")
        if alpha < 0.98 and _object_hand_contacts(model, data, object_geom_id):
            approach_object_contact = True
        if _floor_hand_collision(model, data, object_geom_id):
            approach_floor_contact = True

    preclose_displacement = float(
        np.linalg.norm(_object_position(data, object_qadr) - initial_object_position)
    )
    close_start = float(data.time)
    while data.time - close_start < config.close_s - 1e-12:
        alpha = min(1.0, (data.time - close_start) / config.close_s)
        points = _object_hand_contacts(model, data, object_geom_id)
        opposing = _has_opposing_contacts(
            points,
            _object_position(data, object_qadr),
            closing_axis,
            config.opposing_span_fraction * obj.grasp_width_m,
        )
        if opposing and first_contact_alpha is None:
            first_contact_alpha = alpha
            force_alpha = min(1.0, alpha + config.contact_compression)
        if force_alpha is None:
            command = close_command(alpha)
        else:
            thumb_force, opposing_force = _contact_side_normal_forces(
                model, data, object_geom_id
            )
            force_error = max(
                0.0,
                config.force_target_n - min(thumb_force, opposing_force),
            )
            if abs(force_error) < 0.10:
                force_error = 0.0
            force_alpha = float(
                np.clip(
                    force_alpha
                    + config.force_gain_alpha_per_n_s * force_error * timestep,
                    first_contact_alpha,
                    1.0,
                )
            )
            stop_alpha = force_alpha
            command = close_command(force_alpha)
        _write_command(data, base_ids, finger_ids, command)
        mujoco.mj_step(model, data)
        update_metrics()
        if frame_callback is not None:
            frame_callback(model, data, "close")

    if force_alpha is None:
        force_alpha = 1.0
        stop_alpha = 1.0

    def adaptive_force_command(lift_offset: np.ndarray) -> PoseCommand:
        nonlocal force_alpha, stop_alpha
        if first_contact_alpha is not None:
            thumb_force, opposing_force = _contact_side_normal_forces(
                model, data, object_geom_id
            )
            force_error = max(
                0.0,
                config.force_target_n - min(thumb_force, opposing_force),
            )
            if abs(force_error) < 0.10:
                force_error = 0.0
            force_alpha = float(
                np.clip(
                    force_alpha
                    + config.force_gain_alpha_per_n_s * force_error * timestep,
                    first_contact_alpha,
                    1.0,
                )
            )
            stop_alpha = force_alpha
        command = close_command(force_alpha)
        return PoseCommand(
            command.base_position + lift_offset,
            command.base_rotation,
            command.finger_ctrl,
        )

    zero_lift = np.zeros(3, dtype=float)
    step_for(
        config.post_contact_s,
        lambda _alpha: adaptive_force_command(zero_lift),
        "force settle",
    )
    lift_delta = np.array([0.0, 0.0, config.lift_m])
    lift_duration = config.lift_m / config.lift_speed_m_s
    step_for(
        lift_duration,
        lambda alpha: adaptive_force_command(alpha * lift_delta),
        "lift",
    )
    step_for(
        config.hold_s,
        lambda _alpha: adaptive_force_command(lift_delta),
        "hold",
    )

    final_position = _object_position(data, object_qadr)
    final_lift = float(final_position[2] - initial_z)
    max_lift = float(max_z - initial_z)
    final_contact_points = _object_hand_contacts(model, data, object_geom_id)
    final_opposing = _has_opposing_contacts(
        final_contact_points,
        final_position,
        closing_axis,
        config.opposing_span_fraction * obj.grasp_width_m,
    )
    final_floor_contact = _object_floor_contact(model, data, object_geom_id)
    final_thumb_force, final_opposing_force = _contact_side_normal_forces(
        model, data, object_geom_id
    )
    success = (
        final_lift >= config.success_lift_m
        and final_opposing
        and not final_floor_contact
    )
    if success:
        failure_mode = "ok"
    elif first_contact_alpha is None:
        failure_mode = "no_opposing_contact"
    elif max_lift >= config.success_lift_m:
        failure_mode = "dropped_during_hold"
    elif approach_floor_contact:
        failure_mode = "approach_floor_collision"
    elif approach_object_contact or preclose_displacement > 0.005:
        failure_mode = "approach_object_collision"
    else:
        failure_mode = "slip_during_lift"

    return TrialResult(
        object=obj.name,
        label=obj.label,
        group="delicate" if obj.name in DELICATE_OBJECTS else "ycb",
        method=method,
        point_id=point.point_id,
        point_order=point.order,
        approach_distance_mm=point.approach_distance_mm,
        lateral_mm=point.lateral_mm,
        height_mm=point.height_mm,
        success=success,
        failure_mode=failure_mode,
        final_lift_mm=final_lift * 1000.0,
        max_lift_mm=max_lift * 1000.0,
        max_object_xy_displacement_mm=max_xy * 1000.0,
        preclose_object_displacement_mm=preclose_displacement * 1000.0,
        opposing_contact_detected=first_contact_alpha is not None,
        first_opposing_contact_alpha=first_contact_alpha,
        stopped_closure_alpha=stop_alpha,
        approach_object_contact=approach_object_contact,
        approach_floor_contact=approach_floor_contact,
        final_hand_contact_count=len(final_contact_points),
        final_opposing_contact=final_opposing,
        final_object_floor_contact=final_floor_contact,
        final_thumb_normal_force_n=final_thumb_force,
        final_opposing_normal_force_n=final_opposing_force,
        force_target_n=config.force_target_n,
        candidate_rank=candidate_rank,
        candidate_name=candidate_name,
        candidate_confidence=candidate_confidence,
    )


def _run_object_method_job(
    config: RunConfig,
    object_name: str,
    method: str,
    points: list[GridPoint],
    candidate_selection: tuple[int, GraspGenXCandidate] | None,
) -> list[TrialResult]:
    obj = BUILTIN_OBJECTS[object_name]
    model, object_geom_id, object_qadr = _add_object_model(
        config.xml, obj, PAPER_OBJECT_MASS_KG[object_name]
    )
    return [
        _execute_trial(
            config=config,
            obj=obj,
            method=method,
            point=point,
            candidate_selection=candidate_selection,
            model=model,
            object_geom_id=object_geom_id,
            object_qadr=object_qadr,
        )
        for point in points
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(results: list[TrialResult]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_object: list[dict[str, object]] = []
    for method in METHODS:
        for object_name in PAPER_OBJECTS:
            selected = [row for row in results if row.method == method and row.object == object_name]
            if not selected:
                continue
            successes = sum(row.success for row in selected)
            by_object.append(
                {
                    "method": method,
                    "object": object_name,
                    "label": selected[0].label,
                    "group": selected[0].group,
                    "successes": successes,
                    "trials": len(selected),
                    "success_rate": successes / len(selected),
                    "mean_final_lift_mm": float(np.mean([row.final_lift_mm for row in selected])),
                    "mean_max_xy_displacement_mm": float(
                        np.mean([row.max_object_xy_displacement_mm for row in selected])
                    ),
                    "dominant_failure": max(
                        {mode: sum(row.failure_mode == mode for row in selected) for mode in {r.failure_mode for r in selected}}.items(),
                        key=lambda item: item[1],
                    )[0],
                }
            )

    by_group: list[dict[str, object]] = []
    for method in METHODS:
        for group in ("full", "ycb", "delicate"):
            selected = [
                row for row in results
                if row.method == method and (group == "full" or row.group == group)
            ]
            if not selected:
                continue
            successes = sum(row.success for row in selected)
            by_group.append(
                {
                    "method": method,
                    "group": group,
                    "successes": successes,
                    "trials": len(selected),
                    "success_rate": successes / len(selected),
                    "mean_final_lift_mm": float(np.mean([row.final_lift_mm for row in selected])),
                    "mean_max_xy_displacement_mm": float(
                        np.mean([row.max_object_xy_displacement_mm for row in selected])
                    ),
                }
            )

    by_point: list[dict[str, object]] = []
    for method in METHODS:
        point_ids = sorted({row.point_id for row in results}, key=lambda point_id: next(row.point_order for row in results if row.point_id == point_id))
        for point_id in point_ids:
            selected = [row for row in results if row.method == method and row.point_id == point_id]
            if not selected:
                continue
            successes = sum(row.success for row in selected)
            by_point.append(
                {
                    "method": method,
                    "point_id": point_id,
                    "point_order": selected[0].point_order,
                    "successes": successes,
                    "trials": len(selected),
                    "success_rate": successes / len(selected),
                }
            )
    return by_object, by_group, by_point


def _plot_results(
    out: Path,
    results: list[TrialResult],
    by_object: list[dict[str, object]],
    by_group: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {"iterative": "#1976d2", "graspgenx": "#ef6c00"}
    labels = {"iterative": "Paper Iterative (sim)", "graspgenx": "Pretrained GraspGen-X"}
    selected_methods = [method for method in METHODS if any(row.method == method for row in results)]
    selected_objects = [name for name in PAPER_OBJECTS if any(row.object == name for row in results)]
    point_ids = sorted(
        {row.point_id for row in results},
        key=lambda point_id: next(
            row.point_order for row in results if row.point_id == point_id
        ),
    )
    possible_groups = ["full", "ycb", "delicate"]
    groups = [group for group in possible_groups if any(row["group"] == group for row in by_group)]
    group_label_map = {
        "full": "Full",
        "ycb": "YCB / YCB-like",
        "delicate": "Delicate proxies",
    }
    x = np.arange(len(groups), dtype=float)
    width = 0.72 / len(selected_methods)
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for method_index, method in enumerate(selected_methods):
        rows = {row["group"]: row for row in by_group if row["method"] == method}
        values = [float(rows[group]["success_rate"]) for group in groups]
        positions = x + (method_index - (len(selected_methods) - 1) / 2.0) * width
        bars = axis.bar(positions, values, width, color=colors[method], label=labels[method])
        for bar, group in zip(bars, groups):
            row = rows[group]
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{row['successes']}/{row['trials']}",
                ha="center",
                fontsize=9,
            )
    axis.set_xticks(x, [group_label_map[group] for group in groups])
    axis.set_ylim(0.0, 1.12)
    axis.set_ylabel("Lift success rate")
    axis.set_title(
        f"RH56 proxy grasp success: {len(selected_objects)} objects, "
        f"{len(point_ids)} approach points"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    figure.savefig(out / "group_success_rate.png", dpi=180)
    plt.close(figure)

    object_labels = [BUILTIN_OBJECTS[name].label for name in selected_objects]
    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    x = np.arange(len(selected_objects), dtype=float)
    for method_index, method in enumerate(selected_methods):
        rows = {row["object"]: row for row in by_object if row["method"] == method}
        values = [float(rows[name]["success_rate"]) for name in selected_objects]
        positions = x + (method_index - (len(selected_methods) - 1) / 2.0) * width
        axis.bar(positions, values, width, color=colors[method], label=labels[method])
    axis.set_xticks(x, object_labels, rotation=55, ha="right")
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel(f"Success rate over {len(point_ids)} grid points")
    axis.set_title("Per-object RH56 lift success")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    figure.savefig(out / "object_success_rate.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        len(selected_methods),
        figsize=(6.0 * len(selected_methods), 6.0),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
    )
    for axis, method in zip(axes[0], selected_methods):
        matrix = np.full((len(selected_objects), len(point_ids)), np.nan)
        for object_index, object_name in enumerate(selected_objects):
            for point_index, point_id in enumerate(point_ids):
                match = [
                    row for row in results
                    if row.method == method and row.object == object_name and row.point_id == point_id
                ]
                if match:
                    matrix[object_index, point_index] = float(match[0].success)
        axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(len(point_ids)), point_ids, rotation=55, ha="right")
        axis.set_title(labels[method])
        axis.set_xlabel("Existing collision-test point")
    axes[0, 0].set_yticks(np.arange(len(selected_objects)), object_labels)
    figure.suptitle(
        f"Trial-level success: {len(selected_objects)} objects x {len(point_ids)} approach points"
    )
    figure.savefig(out / "object_point_success.png", dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    points = build_grid_points(args)
    args.out.mkdir(parents=True, exist_ok=True)
    config = build_run_config(args)

    selections: dict[str, tuple[int, GraspGenXCandidate] | None] = {}
    candidate_metadata: dict[str, dict[str, object]] = {}
    if "graspgenx" in args.methods:
        for object_name in args.objects:
            yaml_path = args.candidate_dir / f"{object_name}.yml"
            candidates = load_isaac_grasp_yaml(yaml_path)
            selection = _select_candidate(
                config=config,
                obj=BUILTIN_OBJECTS[object_name],
                candidates=candidates,
                topdown_cos=args.topdown_cos,
            )
            selections[object_name] = selection
            candidate_metadata[object_name] = {
                "yaml": str(yaml_path.resolve()),
                "sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
                "selected_rank": selection[0] if selection else None,
                "selected_name": selection[1].name if selection else None,
                "selected_confidence": selection[1].confidence if selection else None,
                "selected_approach_direction": selection[1].approach_direction.tolist() if selection else None,
            }
            if selection is None:
                print(f"[candidate] {object_name:25s}: NONE")
            else:
                rank, candidate = selection
                print(
                    f"[candidate] {object_name:25s}: rank {rank:2d}, "
                    f"score={candidate.confidence:.3f}, approach_z={candidate.approach_direction[2]:+.3f}"
                )

    jobs = [
        (object_name, method)
        for object_name in args.objects
        for method in args.methods
    ]
    results: list[TrialResult] = []
    if args.workers == 1:
        for object_name, method in jobs:
            rows = _run_object_method_job(
                config,
                object_name,
                method,
                points,
                selections.get(object_name),
            )
            results.extend(rows)
            print(f"[{method:10s}] {object_name:25s}: {sum(row.success for row in rows)}/{len(rows)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_object_method_job,
                    config,
                    object_name,
                    method,
                    points,
                    selections.get(object_name),
                ): (object_name, method)
                for object_name, method in jobs
            }
            for future in as_completed(futures):
                object_name, method = futures[future]
                rows = future.result()
                results.extend(rows)
                print(f"[{method:10s}] {object_name:25s}: {sum(row.success for row in rows)}/{len(rows)}")

    object_order = {name: index for index, name in enumerate(PAPER_OBJECTS)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    results.sort(key=lambda row: (method_order[row.method], object_order[row.object], row.point_order))
    _write_csv(args.out / "trials.csv", [asdict(row) for row in results])
    by_object, by_group, by_point = _summaries(results)
    _write_csv(args.out / "summary_by_object.csv", by_object)
    _write_csv(args.out / "summary_by_group.csv", by_group)
    _write_csv(args.out / "summary_by_point.csv", by_point)

    assumptions = {
        "scope": "simulation-only 15-object primitive-proxy grasp and lift benchmark",
        "trial_count": len(results),
        "methods": args.methods,
        "objects": args.objects,
        "groups": {
            "ycb": [name for name in PAPER_OBJECTS if name not in DELICATE_OBJECTS],
            "delicate": [name for name in PAPER_OBJECTS if name in DELICATE_OBJECTS],
        },
        "grid_source": "tools/run_strategy_pregrasp_rate.py:paper_approach_points_mm",
        "grid_points": [
            {
                "point_id": point.point_id,
                "order": point.order,
                "approach_distance_mm": point.approach_distance_mm,
                "lateral_mm": point.lateral_mm,
                "height_mm": point.height_mm,
                "world_offset_m": point.offset_m.tolist(),
            }
            for point in points
        ],
        "paper_protocol_difference": (
            "The nominal P1-P10 centers match the earlier collision grid and the layout in "
            "Fig. 5. The physical paper additionally randomized lateral position within "
            "+/-30 mm and orientation around the approach axis, but does not publish the "
            "sampled poses or randomization distribution. This run therefore records the "
            "nominal grid centers and holds orientation fixed rather than inventing samples."
        ),
        "success_definition": (
            f"Command a {args.lift_mm:.1f} mm vertical lift at {args.lift_speed_m_s:.3f} m/s, "
            f"hold {args.hold_s:.2f} s, and require final object lift >= {args.success_lift_mm:.1f} mm, "
            "final opposing hand contacts, and no final object-floor contact."
        ),
        "object_proxy_warning": (
            "All objects use estimated box/cylinder/sphere geometry and estimated masses; "
            "they are not measured meshes or masses from the physical experiment."
        ),
        "delicate_object_warning": (
            "The rigid MuJoCo proxies cannot model crushing, puncture, or other damage, so "
            "delicate-object results are lift success only."
        ),
        "force_control_warning": (
            f"Both methods share an approximate MuJoCo contact-force controller targeting "
            f"{args.force_target_n:.1f} N per opposing side after a normalized contact preload. "
            "It is not calibrated against the real RH56 force signals. The rigid-object paper "
            "target was 6 N distributed across active fingers; object-specific delicate-force "
            "targets and damage criteria are not available in this repository."
        ),
        "mass_kg_estimated": {name: PAPER_OBJECT_MASS_KG[name] for name in args.objects},
        "candidate_selection": (
            "First confidence-sorted GraspGen-X candidate whose open hand has neither table penetration nor object contact; "
            "an optional topdown cosine filter is disabled by default because the paper set includes side grasps; "
            "selection never uses lift outcome or P1-P10 path success."
        ),
        "method_execution": {
            "iterative": (
                "Hold the approach-width hand shape while moving from each P1-P10 initial pose to the local pre-grasp, "
                "then reduce width about the fixed grasp point with synchronized wrist compensation and contact-limited compression."
            ),
            "graspgenx": (
                "Move the open hand from the same world-frame waypoint offset to the selected generated grasp pose, "
                "then close with the official Inspire open/close joint convention and the same contact limiter."
            ),
        },
        "graspgenx_candidates": candidate_metadata,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    if not args.no_plots:
        _plot_results(args.out, results, by_object, by_group)

    print(f"Wrote {args.out / 'trials.csv'}")
    print(f"Wrote {args.out / 'summary_by_group.csv'}")
    for row in by_group:
        print(
            f"  {row['method']:10s} {row['group']:8s}: "
            f"{row['successes']}/{row['trials']} = {100.0 * float(row['success_rate']):.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
