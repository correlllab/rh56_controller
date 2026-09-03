#!/usr/bin/env python3
"""Kinematic GGX preflight for UR5e + RH56 desk clearance.

This tool is deliberately simulation-only.  It converts confidence-sorted
GraspGen-X Inspire-hand poses into the checked-in UR5e + RH56 MuJoCo model,
solves arm IK, samples the complete start/pre-grasp/grasp/close/lift path, and
rejects any candidate that brings a robot collision geom too close to the
adjustable desk plane.

The result is a fast screen for human review, not a real-robot safety
certificate.  Nothing in this module imports or calls the hardware bridge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MINK_SRC = REPO_ROOT / "mink" / "src"
if str(MINK_SRC) not in sys.path:
    # Prefer the checked-in robotics package over an unrelated PyPI package
    # that also uses the top-level name ``mink``.
    sys.path.insert(0, str(MINK_SRC))

from rh56_controller.graspgenx_baseline import (  # noqa: E402
    GraspGenXCandidate,
    graspgenx_to_mujoco_base_pose,
    load_isaac_grasp_yaml,
    pregrasp_base_position,
)


DEFAULT_XML = REPO_ROOT / "h1_mujoco/inspire/ur5_inspire.xml"
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINTS = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
)
EEFF_LOCAL_M = np.array([0.070, 0.016, 0.155], dtype=float)
DEFAULT_START_Q_DEG = (0.0, -90.0, 90.0, -90.0, 0.0, 0.0)
GGX_OPEN_CTRL = {
    "pinky": 0.0,
    "ring": 0.0,
    "middle": 0.0,
    "index": 0.0,
    "thumb_proximal": 0.1,
    "thumb_yaw": 1.308,
}
GGX_CLOSE_CTRL = {
    "pinky": 1.47,
    "ring": 1.47,
    "middle": 1.47,
    "index": 1.47,
    "thumb_proximal": 0.57,
    "thumb_yaw": 1.308,
}


@dataclass
class TrajectorySample:
    phase: str
    phase_alpha: float
    qpos: np.ndarray
    closure_alpha: float
    min_desk_clearance_m: float = math.inf
    nearest_geom: str = ""
    desk_collision: bool = False


@dataclass
class CandidateResult:
    rank: int
    name: str
    confidence: float
    passed: bool
    reason: str
    ik_position_error_mm: float | None
    ik_orientation_error_deg: float | None
    min_desk_clearance_mm: float | None
    nearest_geom: str
    first_collision_phase: str
    samples_checked: int


CONFIG_PATH_KEYS = {"graspgenx_yaml", "xml", "out"}


def _config_defaults(
    config_path: Path,
    parser: argparse.ArgumentParser,
) -> dict[str, object]:
    """Load flat YAML defaults, resolving paths from the repository root."""

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = yaml.safe_load(config_path.read_text())
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("--config must contain a YAML mapping")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    valid_keys = set(actions)
    unknown = sorted(set(payload) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown preflight config key(s): {', '.join(unknown)}")

    defaults: dict[str, object] = {}
    for key, value in payload.items():
        action = actions[key]
        if value is None:
            defaults[key] = None
            continue
        try:
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    raise TypeError("expected true or false")
                converted: object = value
            elif action.nargs is not None:
                if not isinstance(value, (list, tuple)):
                    raise TypeError("expected a YAML list")
                converted = [
                    action.type(item) if action.type is not None else item
                    for item in value
                ]
            else:
                converted = action.type(value) if action.type is not None else value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for config key '{key}': {exc}") from exc
        if action.choices is not None and converted not in action.choices:
            choices = ", ".join(str(choice) for choice in action.choices)
            raise ValueError(
                f"Invalid value for config key '{key}'; choose from: {choices}"
            )
        defaults[key] = converted

    for key in CONFIG_PATH_KEYS & defaults.keys():
        value = defaults[key]
        if value is None:
            continue
        path = Path(str(value))
        defaults[key] = path if path.is_absolute() else REPO_ROOT / path
    return defaults


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    tokens = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(tokens)

    parser = argparse.ArgumentParser(
        description=(
            "Screen GraspGen-X candidates with UR5e+RH56 IK and an adjustable "
            "MuJoCo desk plane before human review. Simulation only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Flat YAML defaults file. Relative paths inside it are resolved "
            "from the repository root; explicit CLI options take precedence."
        ),
    )
    parser.add_argument("--graspgenx-yaml", type=Path)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/graspgenx_ur5_table_preflight"),
    )
    parser.add_argument(
        "--table-height-m",
        type=float,
        default=0.070,
        help="Desk top Z in the UR base frame (default: 0.070 m).",
    )
    parser.add_argument(
        "--desk-clearance-mm",
        type=float,
        default=2.0,
        help="Required robot-to-desk clearance, excluding the shoulder fixture.",
    )
    parser.add_argument("--object-x-m", type=float, default=0.0)
    parser.add_argument("--object-y-m", type=float, default=-0.50)
    parser.add_argument(
        "--object-size-mm",
        type=float,
        nargs=3,
        default=[40.0, 40.0, 40.0],
        metavar=("X", "Y", "Z"),
        help="Visual-only object proxy dimensions; Z also sets its center height.",
    )
    parser.add_argument(
        "--object-shape",
        choices=("box", "cylinder", "sphere"),
        default="box",
    )
    parser.add_argument("--standoff-mm", type=float, default=80.0)
    parser.add_argument("--lift-mm", type=float, default=80.0)
    parser.add_argument(
        "--start-q-deg",
        type=float,
        nargs=6,
        default=list(DEFAULT_START_Q_DEG),
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Starting UR joint angles. Replace with the real snapshot before review.",
    )
    parser.add_argument(
        "--candidate-rank",
        type=int,
        default=None,
        help="Check only this confidence rank; otherwise check from rank zero.",
    )
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--review-passes",
        type=int,
        default=5,
        help=(
            "Collect this many automatic PASS candidates for human review "
            "instead of stopping at the first (default: 5)."
        ),
    )
    parser.add_argument("--path-step-deg", type=float, default=1.0)
    parser.add_argument("--cartesian-step-mm", type=float, default=10.0)
    parser.add_argument("--close-samples", type=int, default=30)
    parser.add_argument(
        "--contact-compression",
        type=float,
        default=0.10,
        help="Extra normalized closure after opposing object contacts are found.",
    )
    parser.add_argument(
        "--opposing-span-fraction",
        type=float,
        default=0.30,
        help="Required contact span as a fraction of the smallest object dimension.",
    )
    parser.add_argument("--ik-max-iters", type=int, default=250)
    parser.add_argument("--ik-dt", type=float, default=0.05)
    parser.add_argument("--ik-position-tol-mm", type=float, default=3.0)
    parser.add_argument("--ik-orientation-tol-deg", type=float, default=3.0)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one review MP4 per automatic PASS candidate.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Play the highest-confidence PASS in an interactive MuJoCo viewer.",
    )
    parser.add_argument("--video-width", type=int, default=720)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--camera-azimuth", type=float, default=180.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", type=float, default=1.55)
    if config_args.config is not None:
        parser.set_defaults(**_config_defaults(config_args.config, parser))
    return parser.parse_args(tokens)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.xml.is_file():
        raise FileNotFoundError(args.xml)
    if args.graspgenx_yaml is None:
        raise ValueError("--graspgenx-yaml is required (on the CLI or in --config)")
    if not args.graspgenx_yaml.is_file():
        raise FileNotFoundError(args.graspgenx_yaml)
    if args.desk_clearance_mm < 0.0:
        raise ValueError("--desk-clearance-mm cannot be negative")
    if any(value <= 0.0 for value in args.object_size_mm):
        raise ValueError("--object-size-mm values must be positive")
    if args.standoff_mm < 0.0 or args.lift_mm < 0.0:
        raise ValueError("standoff and lift must be non-negative")
    if args.max_candidates < 1 or args.review_passes < 1 or args.close_samples < 2:
        raise ValueError("candidate and closure sample counts must be positive")
    if not 0.0 <= args.contact_compression <= 0.25:
        raise ValueError("--contact-compression must be in [0, 0.25]")
    if not 0.0 < args.opposing_span_fraction <= 1.0:
        raise ValueError("--opposing-span-fraction must be in (0, 1]")
    if args.path_step_deg <= 0.0 or args.cartesian_step_mm <= 0.0:
        raise ValueError("path sampling steps must be positive")
    if args.ik_max_iters < 1 or args.ik_dt <= 0.0:
        raise ValueError("IK iteration count and timestep must be positive")
    if args.ik_position_tol_mm <= 0.0 or args.ik_orientation_tol_deg <= 0.0:
        raise ValueError("IK tolerances must be positive")
    if args.video_width <= 0 or args.video_height <= 0 or args.fps <= 0:
        raise ValueError("video dimensions and FPS must be positive")
    if args.candidate_rank is not None and args.candidate_rank < 0:
        raise ValueError("--candidate-rank cannot be negative")


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in model: {name}")
    return int(model.jnt_qposadr[joint_id])


def _joint_dof_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in model: {name}")
    return int(model.jnt_dofadr[joint_id])


def _clamp_joint(model: mujoco.MjModel, name: str, value: float) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in model: {name}")
    if model.jnt_limited[joint_id]:
        lower, upper = model.jnt_range[joint_id]
        return float(np.clip(value, lower, upper))
    return float(value)


def set_hand_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    ctrl: dict[str, float],
) -> None:
    """Apply RH56 proximal commands and the checked-in coupling equations."""

    pinky = float(ctrl["pinky"])
    ring = float(ctrl["ring"])
    middle = float(ctrl["middle"])
    index = float(ctrl["index"])
    thumb_pitch = float(ctrl["thumb_proximal"])
    values = {
        "pinky_proximal_joint": pinky,
        "pinky_intermediate_joint": -0.15 + 1.1169 * pinky,
        "ring_proximal_joint": ring,
        "ring_intermediate_joint": -0.15 + 1.1169 * ring,
        "middle_proximal_joint": middle,
        "middle_intermediate_joint": -0.15 + 1.1169 * middle,
        "index_proximal_joint": index,
        "index_intermediate_joint": -0.05 + 1.1169 * index,
        "thumb_proximal_yaw_joint": float(ctrl["thumb_yaw"]),
        "thumb_proximal_pitch_joint": thumb_pitch,
        "thumb_intermediate_joint": 0.15 + 1.33 * thumb_pitch,
        "thumb_distal_joint": 0.15 + 0.66 * thumb_pitch,
    }
    for name, value in values.items():
        qpos[_joint_qpos_address(model, name)] = _clamp_joint(model, name, value)


def _lerp_ctrl(alpha: float) -> dict[str, float]:
    return {
        name: (1.0 - alpha) * GGX_OPEN_CTRL[name] + alpha * GGX_CLOSE_CTRL[name]
        for name in GGX_OPEN_CTRL
    }


def build_model(
    xml_path: Path,
    *,
    table_height_m: float,
    object_center: np.ndarray,
    object_size_m: np.ndarray,
    object_shape: str,
) -> mujoco.MjModel:
    """Build the checked-in robot plus an adjustable desk and visual object."""

    spec = mujoco.MjSpec.from_file(str(xml_path))
    floor = spec.geom("floor")
    if floor is None:
        raise ValueError(f"Model has no geom named 'floor': {xml_path}")
    floor.pos[2] = float(table_height_m)

    body = spec.worldbody.add_body(name="preflight_object", pos=object_center.tolist())
    geom_kwargs: dict[str, object] = {
        "name": "preflight_object_visual",
        "rgba": [0.92, 0.18, 0.08, 0.75],
        # The fixed proxy participates in hand contact so closure can stop
        # before the fingers unrealistically pass through the object.  It is
        # excluded from the desk-clearance metric below.
        "contype": 1,
        "conaffinity": 1,
        "density": 0.0,
        "group": 2,
    }
    if object_shape == "box":
        geom_kwargs.update(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(object_size_m / 2.0).tolist(),
        )
    elif object_shape == "cylinder":
        geom_kwargs.update(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[float(min(object_size_m[:2]) / 2.0), float(object_size_m[2] / 2.0), 0.0],
        )
    else:
        geom_kwargs.update(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[float(min(object_size_m) / 2.0), 0.0, 0.0],
        )
    body.add_geom(**geom_kwargs)
    return spec.compile()


def initial_qpos(model: mujoco.MjModel, arm_q_rad: np.ndarray) -> np.ndarray:
    qpos = model.qpos0.copy()
    for name, value in zip(ARM_JOINTS, arm_q_rad, strict=True):
        qpos[_joint_qpos_address(model, name)] = float(value)
    set_hand_qpos(model, qpos, GGX_OPEN_CTRL)
    return qpos


def target_eeff_transform(base_position: np.ndarray, base_rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = base_rotation
    transform[:3, 3] = base_position + base_rotation @ EEFF_LOCAL_M
    return transform


def _rotation_error_deg(error: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(error[3:])))


def solve_pose_ik(
    model: mujoco.MjModel,
    seed_qpos: np.ndarray,
    target: np.ndarray,
    *,
    max_iters: int,
    dt: float,
    position_tol_m: float,
    orientation_tol_rad: float,
    solver: str,
) -> tuple[np.ndarray, float, float, bool]:
    """Solve one end-effector pose while freezing all RH56 joints."""

    import mink

    configuration = mink.Configuration(model, q=seed_qpos.copy())
    frame_task = mink.FrameTask(
        "eeff",
        "site",
        position_cost=1.0,
        orientation_cost=0.7,
        lm_damping=1e-6,
    )
    frame_task.set_target(mink.SE3.from_matrix(target))
    posture_task = mink.PostureTask(model, cost=1e-4)
    posture_task.set_target_from_configuration(configuration)
    freeze_hand = mink.DofFreezingTask(
        model=model,
        dof_indices=[_joint_dof_address(model, name) for name in HAND_JOINTS],
    )
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, {name: math.pi for name in ARM_JOINTS}),
    ]

    pos_error = math.inf
    ori_error = math.inf
    for _ in range(max_iters):
        error = frame_task.compute_error(configuration)
        pos_error = float(np.linalg.norm(error[:3]))
        ori_error = float(np.linalg.norm(error[3:]))
        if pos_error <= position_tol_m and ori_error <= orientation_tol_rad:
            return configuration.q.copy(), pos_error, ori_error, True
        try:
            velocity = mink.solve_ik(
                configuration,
                [frame_task, posture_task],
                dt,
                solver,
                limits=limits,
                constraints=[freeze_hand],
            )
        except mink.NoSolutionFound:
            break
        configuration.integrate_inplace(velocity, dt)

    error = frame_task.compute_error(configuration)
    pos_error = float(np.linalg.norm(error[:3]))
    ori_error = float(np.linalg.norm(error[3:]))
    passed = pos_error <= position_tol_m and ori_error <= orientation_tol_rad
    return configuration.q.copy(), pos_error, ori_error, passed


def _append_joint_segment(
    model: mujoco.MjModel,
    samples: list[TrajectorySample],
    start_q: np.ndarray,
    end_q: np.ndarray,
    *,
    phase: str,
    max_step_rad: float,
    closure_alpha: float,
) -> None:
    arm_addresses = np.array(
        [_joint_qpos_address(model, name) for name in ARM_JOINTS], dtype=int
    )
    max_change = float(np.max(np.abs(end_q[arm_addresses] - start_q[arm_addresses])))
    count = max(1, int(math.ceil(max_change / max_step_rad)))
    for index in range(1, count + 1):
        alpha = index / count
        qpos = (1.0 - alpha) * start_q + alpha * end_q
        samples.append(TrajectorySample(phase, alpha, qpos, closure_alpha))

def plan_candidate(
    model: mujoco.MjModel,
    start_qpos: np.ndarray,
    candidate: GraspGenXCandidate,
    *,
    object_center: np.ndarray,
    standoff_m: float,
    lift_m: float,
    max_joint_step_rad: float,
    cartesian_step_m: float,
    close_samples: int,
    object_size_m: np.ndarray,
    contact_compression: float,
    opposing_span_fraction: float,
    ik_max_iters: int,
    ik_dt: float,
    ik_position_tol_m: float,
    ik_orientation_tol_rad: float,
    solver: str,
) -> tuple[list[TrajectorySample], float, float, str]:
    """Create a simple joint-sampled pre-grasp, close, and lift trajectory."""

    target_position, target_rotation = graspgenx_to_mujoco_base_pose(
        candidate, object_center=object_center
    )
    pregrasp_position = pregrasp_base_position(
        target_position, candidate.approach_direction, standoff_m
    )

    samples = [TrajectorySample("start", 0.0, start_qpos.copy(), 0.0)]
    worst_pos_error = 0.0
    worst_ori_error = 0.0
    current_q = start_qpos.copy()

    def solve(position: np.ndarray, phase: str) -> tuple[np.ndarray | None, str]:
        nonlocal current_q, worst_pos_error, worst_ori_error
        target = target_eeff_transform(position, target_rotation)
        solved_q, pos_error, ori_error, passed = solve_pose_ik(
            model,
            current_q,
            target,
            max_iters=ik_max_iters,
            dt=ik_dt,
            position_tol_m=ik_position_tol_m,
            orientation_tol_rad=ik_orientation_tol_rad,
            solver=solver,
        )
        worst_pos_error = max(worst_pos_error, pos_error)
        worst_ori_error = max(worst_ori_error, ori_error)
        if not passed:
            return None, f"ik_failed_{phase}"
        return solved_q, ""

    pregrasp_q, reason = solve(pregrasp_position, "pregrasp")
    if pregrasp_q is None:
        return samples, worst_pos_error, worst_ori_error, reason
    _append_joint_segment(
        model,
        samples,
        current_q,
        pregrasp_q,
        phase="start_to_pregrasp",
        max_step_rad=max_joint_step_rad,
        closure_alpha=0.0,
    )
    current_q = pregrasp_q

    approach_distance = float(np.linalg.norm(target_position - pregrasp_position))
    approach_steps = max(1, int(math.ceil(approach_distance / cartesian_step_m)))
    for index in range(1, approach_steps + 1):
        alpha = index / approach_steps
        position = (1.0 - alpha) * pregrasp_position + alpha * target_position
        solved_q, reason = solve(position, "approach")
        if solved_q is None:
            return samples, worst_pos_error, worst_ori_error, reason
        _append_joint_segment(
            model,
            samples,
            current_q,
            solved_q,
            phase="approach",
            max_step_rad=max_joint_step_rad,
            closure_alpha=0.0,
        )
        current_q = solved_q

    object_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "preflight_object_visual"
    )
    close_data = mujoco.MjData(model)
    closing_axis = candidate.rotation[:, 0].copy()
    closing_axis /= np.linalg.norm(closing_axis)
    first_contact_alpha: float | None = None
    stop_alpha = 1.0
    for index in range(1, close_samples + 1):
        alpha = index / close_samples
        qpos = current_q.copy()
        set_hand_qpos(model, qpos, _lerp_ctrl(alpha))
        samples.append(TrajectorySample("close", alpha, qpos, alpha))
        close_data.qpos[:] = qpos
        mujoco.mj_forward(model, close_data)
        points = _object_hand_contact_points(model, close_data, object_geom_id)
        if first_contact_alpha is None and _has_opposing_contacts(
            points,
            object_center,
            closing_axis,
            opposing_span_fraction * float(np.min(object_size_m)),
        ):
            first_contact_alpha = alpha
            stop_alpha = min(1.0, alpha + contact_compression)
        if first_contact_alpha is not None and alpha >= stop_alpha - 1e-12:
            break
    current_q = samples[-1].qpos.copy()

    lift_position = target_position + np.array([0.0, 0.0, lift_m])
    lift_steps = max(1, int(math.ceil(max(lift_m, 1e-12) / cartesian_step_m)))
    for index in range(1, lift_steps + 1):
        alpha = index / lift_steps
        position = (1.0 - alpha) * target_position + alpha * lift_position
        solved_q, reason = solve(position, "lift")
        if solved_q is None:
            return samples, worst_pos_error, worst_ori_error, reason
        set_hand_qpos(model, solved_q, GGX_CLOSE_CTRL)
        _append_joint_segment(
            model,
            samples,
            current_q,
            solved_q,
            phase="lift",
            max_step_rad=max_joint_step_rad,
            closure_alpha=1.0,
        )
        current_q = solved_q

    return samples, worst_pos_error, worst_ori_error, ""


def _object_hand_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_id: int,
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for contact in data.contact:
        if contact.dist > 1e-5:
            continue
        if int(contact.geom1) == object_geom_id:
            other_geom = int(contact.geom2)
        elif int(contact.geom2) == object_geom_id:
            other_geom = int(contact.geom1)
        else:
            continue
        if int(model.geom_bodyid[other_geom]) == 0:
            continue
        points.append(np.asarray(contact.pos, dtype=float).copy())
    return points


def _has_opposing_contacts(
    points: list[np.ndarray],
    object_center: np.ndarray,
    closing_axis: np.ndarray,
    required_span_m: float,
) -> bool:
    if len(points) < 2:
        return False
    projections = np.asarray(
        [(point - object_center) @ closing_axis for point in points], dtype=float
    )
    return float(np.ptp(projections)) >= required_span_m


def _geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    geom_name = model.geom(geom_id).name or f"geom_{geom_id}"
    body_id = int(model.geom_bodyid[geom_id])
    body_name = model.body(body_id).name or f"body_{body_id}"
    return f"{body_name}/{geom_name}"


def desk_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    distance_cap_m: float = 0.50,
) -> tuple[float, str]:
    """Return minimum robot-to-desk distance, skipping the fixed shoulder mount."""

    desk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if desk_id < 0:
        raise ValueError("Model has no desk/floor geom")
    minimum = math.inf
    nearest = ""
    fromto = np.empty(6, dtype=float)
    for geom_id in range(model.ngeom):
        if geom_id == desk_id:
            continue
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        body_name = model.body(int(model.geom_bodyid[geom_id])).name
        if body_name == "preflight_object":
            continue
        if body_name == "shoulder_link":
            # The checked-in model represents the robot fixture as a shoulder
            # capsule passing through the desk plane.  It is the only allowed
            # permanent desk contact.
            continue
        distance = float(
            mujoco.mj_geomDistance(
                model, data, desk_id, geom_id, distance_cap_m, fromto
            )
        )
        if distance < minimum:
            minimum = distance
            nearest = _geom_label(model, geom_id)
    return minimum, nearest


def check_trajectory(
    model: mujoco.MjModel,
    samples: list[TrajectorySample],
    *,
    required_clearance_m: float,
) -> tuple[float, str, str]:
    data = mujoco.MjData(model)
    overall_minimum = math.inf
    overall_nearest = ""
    first_collision_phase = ""
    for sample in samples:
        data.qpos[:] = sample.qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        clearance, nearest = desk_clearance(model, data)
        sample.min_desk_clearance_m = clearance
        sample.nearest_geom = nearest
        sample.desk_collision = clearance < required_clearance_m - 1e-9
        if clearance < overall_minimum:
            overall_minimum = clearance
            overall_nearest = nearest
        if sample.desk_collision and not first_collision_phase:
            first_collision_phase = sample.phase
    return overall_minimum, overall_nearest, first_collision_phase


def evaluate_candidate(
    model: mujoco.MjModel,
    start_qpos: np.ndarray,
    candidate: GraspGenXCandidate,
    rank: int,
    args: argparse.Namespace,
    object_center: np.ndarray,
) -> tuple[CandidateResult, list[TrajectorySample]]:
    samples, pos_error, ori_error, plan_reason = plan_candidate(
        model,
        start_qpos,
        candidate,
        object_center=object_center,
        standoff_m=args.standoff_mm / 1000.0,
        lift_m=args.lift_mm / 1000.0,
        max_joint_step_rad=math.radians(args.path_step_deg),
        cartesian_step_m=args.cartesian_step_mm / 1000.0,
        close_samples=args.close_samples,
        object_size_m=np.asarray(args.object_size_mm, dtype=float) / 1000.0,
        contact_compression=args.contact_compression,
        opposing_span_fraction=args.opposing_span_fraction,
        ik_max_iters=args.ik_max_iters,
        ik_dt=args.ik_dt,
        ik_position_tol_m=args.ik_position_tol_mm / 1000.0,
        ik_orientation_tol_rad=math.radians(args.ik_orientation_tol_deg),
        solver=args.solver,
    )
    minimum, nearest, collision_phase = check_trajectory(
        model,
        samples,
        required_clearance_m=args.desk_clearance_mm / 1000.0,
    )
    if plan_reason:
        reason = plan_reason
    elif collision_phase:
        reason = "desk_clearance"
    else:
        reason = "pass"
    result = CandidateResult(
        rank=rank,
        name=candidate.name,
        confidence=float(candidate.confidence),
        passed=reason == "pass",
        reason=reason,
        ik_position_error_mm=pos_error * 1000.0,
        ik_orientation_error_deg=math.degrees(ori_error),
        min_desk_clearance_mm=minimum * 1000.0,
        nearest_geom=nearest,
        first_collision_phase=collision_phase,
        samples_checked=len(samples),
    )
    return result, samples


def _trajectory_rows(model: mujoco.MjModel, samples: list[TrajectorySample]) -> list[dict[str, object]]:
    arm_addresses = [_joint_qpos_address(model, name) for name in ARM_JOINTS]
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        row: dict[str, object] = {
            "sample": index,
            "phase": sample.phase,
            "phase_alpha": sample.phase_alpha,
            "closure_alpha": sample.closure_alpha,
            "min_desk_clearance_mm": sample.min_desk_clearance_m * 1000.0,
            "nearest_geom": sample.nearest_geom,
            "desk_collision": int(sample.desk_collision),
        }
        for name, address in zip(ARM_JOINTS, arm_addresses, strict=True):
            row[f"{name}_rad"] = float(sample.qpos[address])
            row[f"{name}_deg"] = math.degrees(float(sample.qpos[address]))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _camera(args: argparse.Namespace, object_center: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [object_center[0], 0.5 * object_center[1], args.table_height_m + 0.28]
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    return camera


def annotate_review_frame(
    frame: np.ndarray,
    sample: TrajectorySample,
    result: CandidateResult,
    running_min_clearance_m: float,
) -> np.ndarray:
    """Add the decision-relevant preflight state to an RGB video frame."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    current_mm = sample.min_desk_clearance_m * 1000.0
    running_mm = running_min_clearance_m * 1000.0
    nearest = sample.nearest_geom or "n/a"
    now_state = "COLLISION" if sample.desk_collision else "CLEAR"
    final_state = "PASS" if result.passed else "REJECT"
    lines = [
        f"GGX rank {result.rank} | confidence {result.confidence:.3f}",
        f"phase {sample.phase} | closure {sample.closure_alpha:.2f}",
        f"desk clearance {current_mm:.1f} mm | min so far {running_mm:.1f} mm",
        f"nearest robot geom: {nearest}",
        f"current gate: {now_state} | final automatic result: {final_state}",
    ]

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(11, int(image.width / 55))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow versions before scalable load_default.
        font = ImageFont.load_default()
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_height = max(box[3] - box[1] for box in boxes) + 3
    panel_height = 7 + line_height * len(lines)
    if sample.desk_collision:
        panel_color = (125, 18, 18)
    elif result.passed:
        panel_color = (18, 90, 48)
    else:
        panel_color = (112, 72, 12)
    draw.rectangle((0, 0, image.width, panel_height), fill=panel_color)
    for index, line in enumerate(lines):
        draw.text((7, 4 + index * line_height), line, fill="white", font=font)
    return np.asarray(image)


def write_video(
    path: Path,
    model: mujoco.MjModel,
    samples: list[TrajectorySample],
    result: CandidateResult,
    args: argparse.Namespace,
    object_center: np.ndarray,
) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("MP4 output requires the video dependency profile") from exc

    # Several checked-in MJCFs retain MuJoCo's 640x480 default offscreen
    # framebuffer. Renderer rejects larger requested images unless the model's
    # framebuffer is enlarged first. Keep the XML untouched and size only this
    # in-memory preflight model.
    model.vis.global_.offwidth = max(
        int(model.vis.global_.offwidth), int(args.video_width)
    )
    model.vis.global_.offheight = max(
        int(model.vis.global_.offheight), int(args.video_height)
    )
    renderer = mujoco.Renderer(
        model, height=args.video_height, width=args.video_width
    )
    camera = _camera(args, object_center)
    data = mujoco.MjData(model)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (args.video_width, args.video_height),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    running_min = math.inf
    try:
        for sample in samples:
            running_min = min(running_min, sample.min_desk_clearance_m)
            data.qpos[:] = sample.qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = annotate_review_frame(
                renderer.render(), sample, result, running_min
            )
            writer.send(frame)
    finally:
        writer.close()
        renderer.close()


def play_viewer(
    model: mujoco.MjModel,
    samples: list[TrajectorySample],
    args: argparse.Namespace,
    object_center: np.ndarray,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = _camera(args, object_center).lookat
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = args.camera_elevation
        for sample in samples:
            if not viewer.is_running():
                break
            data.qpos[:] = sample.qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / args.fps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)

    object_size_m = np.asarray(args.object_size_mm, dtype=float) / 1000.0
    object_center = np.array(
        [
            args.object_x_m,
            args.object_y_m,
            args.table_height_m + object_size_m[2] / 2.0,
        ],
        dtype=float,
    )
    model = build_model(
        args.xml,
        table_height_m=args.table_height_m,
        object_center=object_center,
        object_size_m=object_size_m,
        object_shape=args.object_shape,
    )
    start_qpos = initial_qpos(model, np.radians(np.asarray(args.start_q_deg, dtype=float)))
    candidates = load_isaac_grasp_yaml(args.graspgenx_yaml)
    if args.candidate_rank is not None:
        if args.candidate_rank >= len(candidates):
            raise ValueError("--candidate-rank is outside the candidate list")
        ranks = [args.candidate_rank]
    else:
        ranks = list(range(min(len(candidates), args.max_candidates)))

    candidate_results: list[CandidateResult] = []
    passing: list[tuple[CandidateResult, list[TrajectorySample]]] = []
    best_failed: tuple[CandidateResult, list[TrajectorySample]] | None = None
    for rank in ranks:
        result, samples = evaluate_candidate(
            model, start_qpos, candidates[rank], rank, args, object_center
        )
        candidate_results.append(result)
        state = "PASS" if result.passed else "reject"
        print(
            f"[{state}] rank={rank:02d} {result.name} score={result.confidence:.3f} "
            f"reason={result.reason} min_desk={result.min_desk_clearance_mm:.1f} mm"
        )
        if result.passed:
            passing.append((result, samples))
            required_passes = 1 if args.candidate_rank is not None else args.review_passes
            if len(passing) >= required_passes:
                break
        if (
            result.reason == "desk_clearance"
            and (
                best_failed is None
                or result.min_desk_clearance_mm > best_failed[0].min_desk_clearance_mm
            )
        ):
            best_failed = (result, samples)

    _write_csv(
        args.out / "candidates.csv",
        [asdict(result) for result in candidate_results],
    )
    highest_pass = passing[0] if passing else None
    selected_result = highest_pass[0] if highest_pass else None
    review_result = highest_pass if highest_pass is not None else best_failed
    _write_csv(
        args.out / "human_review.csv",
        [
            {
                "rank": result.rank,
                "name": result.name,
                "confidence": result.confidence,
                "automatic_pass": 1,
                "human_decision": "",
                "veto_reason": "",
            }
            for result, _samples in passing
        ],
    )
    summary = {
        "schema": "rh56_ggx_ur5_table_preflight/v1",
        "simulation_only": True,
        "passed": bool(passing),
        "human_review_required": True,
        "automatic_pass_ranks": [result.rank for result, _samples in passing],
        "automatic_passes_collected": len(passing),
        "human_selected_rank": None,
        "selected_rank": selected_result.rank if selected_result else None,
        "selected_name": selected_result.name if selected_result else None,
        "selected_confidence": selected_result.confidence if selected_result else None,
        "minimum_desk_clearance_mm": (
            selected_result.min_desk_clearance_mm if selected_result else None
        ),
        "candidates_checked": len(candidate_results),
        "review_rank": review_result[0].rank if review_result else None,
        "review_is_passing": bool(passing),
        "model": str(args.xml.resolve()),
        "model_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        "candidate_yaml": str(args.graspgenx_yaml.resolve()),
        "candidate_yaml_sha256": hashlib.sha256(args.graspgenx_yaml.read_bytes()).hexdigest(),
        "config": str(args.config.resolve()) if args.config is not None else None,
        "config_sha256": (
            hashlib.sha256(args.config.read_bytes()).hexdigest()
            if args.config is not None
            else None
        ),
        "table_height_m": args.table_height_m,
        "desk_clearance_mm": args.desk_clearance_mm,
        "object_center_m": object_center.tolist(),
        "object_size_mm": [float(value) for value in args.object_size_mm],
        "start_q_deg": [float(value) for value in args.start_q_deg],
        "assumptions": [
            "Desk is an infinite horizontal plane at table_height_m.",
            "The shoulder_link fixture contact with the desk is ignored.",
            "Only robot-to-desk clearance is an automatic collision gate.",
            "The fixed primitive object proxy is used only to stop closure after opposing contact.",
            "Object and self-collision are not automatic pass/fail gates in this simplified tool.",
            "IK and sampled MuJoCo geometry are a human-review aid, not a hardware safety certificate.",
            "No hardware module is imported and no command is sent to UR5 or RH56.",
        ],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(
        args.out / "summary.csv",
        [
            {
                "passed": int(bool(passing)),
                "selected_rank": selected_result.rank if selected_result else "",
                "selected_name": selected_result.name if selected_result else "",
                "selected_confidence": selected_result.confidence if selected_result else "",
                "minimum_desk_clearance_mm": (
                    selected_result.min_desk_clearance_mm if selected_result else ""
                ),
                "candidates_checked": len(candidate_results),
                "human_review_required": 1,
            }
        ],
    )

    if passing:
        for result, samples in passing:
            suffix = f"rank_{result.rank:03d}"
            _write_csv(
                args.out / f"trajectory_{suffix}.csv",
                _trajectory_rows(model, samples),
            )
            if args.video:
                os.environ.setdefault("MUJOCO_GL", "egl")
                write_video(
                    args.out / f"review_{suffix}.mp4",
                    model,
                    samples,
                    result,
                    args,
                    object_center,
                )
        # Stable convenience names always point to the highest-confidence
        # automatically feasible candidate. Human review may veto it.
        _write_csv(
            args.out / "trajectory.csv",
            _trajectory_rows(model, passing[0][1]),
        )
        if args.video:
            write_video(
                args.out / "review.mp4",
                model,
                passing[0][1],
                passing[0][0],
                args,
                object_center,
            )
        if args.viewer:
            play_viewer(model, passing[0][1], args, object_center)
    elif review_result is not None:
        result, samples = review_result
        _write_csv(args.out / "trajectory.csv", _trajectory_rows(model, samples))
        if args.video:
            os.environ.setdefault("MUJOCO_GL", "egl")
            write_video(
                args.out / "review.mp4",
                model,
                samples,
                result,
                args,
                object_center,
            )
        if args.viewer:
            play_viewer(model, samples, args, object_center)

    if not passing:
        detail = (
            f" Closest rejected candidate was rank {review_result[0].rank}."
            if review_result
            else ""
        )
        print(f"No candidate passed.{detail} Review {args.out / 'candidates.csv'}")
        return 2

    result, _samples = passing[0]
    print(
        f"Collected {len(passing)} automatic PASS candidate(s); highest confidence is "
        f"rank {result.rank}. Human review is still required. "
        f"Outputs: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
