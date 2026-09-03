#!/usr/bin/env python3
"""One-example demo for strategy-specific RH56 pre-grasp collision checks.

This is a debugging/demo tool for the paper-v2 no-go-volume direction.  It
compares three execution-strategy pre-grasp hand shapes for one fixed object
and one fixed analytical final grasp pose:

  - naive: fully open fingers with thumb yaw rotated to the final yaw
  - iterative_closure: final analytical posture opened by a small width margin
  - thumb_reflex: final thumb posture with all non-thumb fingers open

For each strategy, the script checks whether the pre-grasp capsule hand proxy
can move from one sampled hand-base start pose to that strategy's pre-grasp
target without hitting the object AABB or the ground plane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

from rh56_controller.capsule_hand_proxy import (
    Capsule,
    build_capsule_proxy,
    capsule_collisions_against_object,
    closure_base_rotation,
    sample_linear_path_collisions,
    set_closure_qpos,
    transform_capsules,
)
from rh56_controller.grasp_geometry import (
    ACTUATOR_NAMES,
    ClosureGeometry,
    ClosureResult,
    InspireHandFK,
)
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS, ObjectSpec
from rh56_controller.paper_v2_objects import tabletop_aabb_center, tabletop_grasp_target


NAIVE_REAL_CMD = [504, 496, 467, 500, 479, 0]
STRATEGIES = ("naive", "iterative_closure", "thumb_reflex")


@dataclass(frozen=True)
class StrategyPose:
    name: str
    label: str
    ctrl_values: dict[str, float]
    target_result: ClosureResult
    notes: str


@dataclass(frozen=True)
class Evaluation:
    strategy: StrategyPose
    start: np.ndarray
    target: np.ndarray
    object_path_collision: bool
    object_target_collision: bool
    floor_path_collision: bool
    floor_target_collision: bool
    min_object_clearance_m: float
    min_floor_clearance_m: float
    nearest_object: str
    max_active_object_collisions: int
    path_rows: list[dict[str, object]]
    capsules_start: list[Capsule]
    capsules_target: list[Capsule]

    @property
    def valid(self) -> bool:
        return not (
            self.object_path_collision
            or self.object_target_collision
            or self.floor_path_collision
            or self.floor_target_collision
        )

    @property
    def dominant_blocker(self) -> str:
        if self.floor_target_collision:
            return "target_floor_collision"
        if self.object_target_collision:
            return "target_object_collision"
        if self.floor_path_collision:
            return "path_floor_collision"
        if self.object_path_collision:
            return "path_object_collision"
        return "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo three strategy-specific pre-grasp collision checks."
    )
    parser.add_argument("--object", choices=sorted(BUILTIN_OBJECTS), default="debug_40mm_cube")
    parser.add_argument(
        "--mode",
        choices=["object-default", "line", "plane3", "plane4", "plane5", "cylinder"],
        default="plane4",
        help="Analytical final grasp mode. Use object-default to use the object metadata.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/strategy_pregrasp_demo"))
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument(
        "--grasp-center-policy",
        choices=["antipodal", "contact-centroid"],
        default="antipodal",
        help=(
            "How to align the object AABB to the analytical grasp. antipodal "
            "uses the midpoint between thumb and non-thumb fingertip centroid; "
            "contact-centroid uses ClosureResult.midpoint for legacy viewer parity."
        ),
    )
    parser.add_argument(
        "--preopen-mm",
        type=float,
        default=5.0,
        help=(
            "Width margin added to the final grasp width for iterative_closure "
            "pre-grasp when --iterative-pregrasp-policy=final-plus-preopen."
        ),
    )
    parser.add_argument(
        "--iterative-pregrasp-policy",
        choices=["planner-max-width", "final-plus-preopen"],
        default="planner-max-width",
        help=(
            "How to choose the Plan/iterative_closure pre-grasp width when "
            "--iterative-width-mm is not set. planner-max-width matches the "
            "interactive planner default."
        ),
    )
    parser.add_argument(
        "--iterative-width-mm",
        type=float,
        default=None,
        help="Explicit iterative_closure pre-grasp width. Overrides --preopen-mm.",
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--start-mm",
        type=float,
        nargs=3,
        default=None,
        help="Manual hand-base start position. If omitted, a small candidate set is scored.",
    )
    parser.add_argument("--path-samples", type=int, default=16)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--floor-z-mm", type=float, default=0.0)
    parser.add_argument(
        "--floor-tolerance-mm",
        type=float,
        default=3.0,
        help=(
            "Small negative floor clearance tolerated for the capsule proxy. "
            "This absorbs proxy-vs-mesh conservatism; reported clearances remain raw."
        ),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def tabletop_object_center(obj: ObjectSpec) -> np.ndarray:
    return tabletop_aabb_center(obj)


def grasp_center_base(result: ClosureResult, policy: str) -> np.ndarray:
    if policy == "contact-centroid":
        return result.midpoint
    if policy != "antipodal":
        raise ValueError(f"Unsupported grasp center policy: {policy}")

    thumb = result.tip_positions.get("thumb")
    nonthumb = [
        point
        for name, point in result.tip_positions.items()
        if name != "thumb"
    ]
    if thumb is None or not nonthumb:
        return result.midpoint
    nonthumb_centroid = np.vstack(nonthumb).mean(axis=0)
    return 0.5 * (thumb + nonthumb_centroid)


def closure_base_position_for_center(
    result: ClosureResult,
    yaw_rad: float,
    *,
    object_center: np.ndarray,
    policy: str,
) -> np.ndarray:
    """Place the selected grasp center at the object center."""

    return object_center - (closure_base_rotation(result, yaw_rad) @ grasp_center_base(result, policy))


def solve_mode(closure: ClosureGeometry, mode: str, width_m: float) -> ClosureResult:
    if mode == "line":
        return closure.line(width_m)
    if mode.startswith("plane"):
        return closure.plane(width_m, n_fingers=int(mode[-1]))
    if mode == "cylinder":
        return closure.cylinder(width_m)
    raise ValueError(f"Unsupported mode: {mode}")


def mode_width_range(closure: ClosureGeometry, mode: str) -> tuple[float, float]:
    if mode == "line":
        return closure.width_range("2-finger line", n_fingers=2)
    if mode.startswith("plane"):
        n = int(mode[-1])
        return closure.width_range(f"{n}-finger plane", n_fingers=n)
    if mode == "cylinder":
        return closure.width_range("cylinder", n_fingers=5)
    raise ValueError(f"Unsupported mode: {mode}")


def pregrasp_width_policy(args: argparse.Namespace) -> str:
    if args.iterative_width_mm is not None:
        return "explicit_iterative_width"
    return str(args.iterative_pregrasp_policy)


def real_cmd_to_ctrl(fk: InspireHandFK, real_cmd: list[int]) -> dict[str, float]:
    ctrl_min = np.array([fk.ctrl_min[name] for name in ACTUATOR_NAMES], dtype=float)
    ctrl_max = np.array([fk.ctrl_max[name] for name in ACTUATOR_NAMES], dtype=float)
    raw = np.clip(np.array(real_cmd, dtype=float) / 1000.0, 0.0, 1.0)
    ctrl = ctrl_min + (1.0 - raw) * (ctrl_max - ctrl_min)
    return {name: float(value) for name, value in zip(ACTUATOR_NAMES, ctrl)}


def open_ctrl(fk: InspireHandFK) -> dict[str, float]:
    return {name: float(fk.ctrl_min[name]) for name in ACTUATOR_NAMES}


def open_ctrl_with_final_thumb_yaw(
    fk: InspireHandFK,
    final_result: ClosureResult,
) -> dict[str, float]:
    ctrl = open_ctrl(fk)
    ctrl["thumb_yaw"] = float(final_result.ctrl_values.get("thumb_yaw", fk.ctrl_min["thumb_yaw"]))
    return ctrl


def strategy_poses(
    fk: InspireHandFK,
    closure: ClosureGeometry,
    *,
    mode: str,
    final_result: ClosureResult,
    target_width_m: float,
    preopen_m: float,
    iterative_width_m: float | None = None,
    iterative_pregrasp_policy: str = "planner-max-width",
) -> list[StrategyPose]:
    lo, hi = mode_width_range(closure, mode)
    if iterative_width_m is not None:
        desired_iterative_width = iterative_width_m
    elif iterative_pregrasp_policy == "planner-max-width":
        desired_iterative_width = hi
    elif iterative_pregrasp_policy == "final-plus-preopen":
        desired_iterative_width = target_width_m + preopen_m
    else:
        raise ValueError(f"Unsupported iterative pre-grasp policy: {iterative_pregrasp_policy}")

    iterative_width = min(hi, max(lo, desired_iterative_width))
    if iterative_width <= target_width_m:
        iterative_width = min(hi, target_width_m + 1e-4)

    try:
        iterative_result = solve_mode(closure, mode, iterative_width)
    except Exception:
        iterative_result = final_result

    iterative_ctrl = dict(iterative_result.ctrl_values)
    iterative_ctrl["thumb_yaw"] = float(final_result.ctrl_values.get("thumb_yaw", 0.0))

    reflex_ctrl = open_ctrl(fk)
    reflex_ctrl["thumb_proximal"] = float(final_result.ctrl_values.get("thumb_proximal", 0.0))
    reflex_ctrl["thumb_yaw"] = float(final_result.ctrl_values.get("thumb_yaw", 0.0))
    naive_ctrl = open_ctrl_with_final_thumb_yaw(fk, final_result)

    return [
        StrategyPose(
            name="naive",
            label="Naive open with thumb yaw",
            ctrl_values=naive_ctrl,
            target_result=final_result,
            notes=(
                "Naive pre-grasp is modeled as fully open fingers with thumb yaw "
                "already rotated to the final opposing direction; thumb bend and "
                "non-thumb fingers close together after the arm reaches the final "
                f"grasp pose. The historical fixed pinch command is {NAIVE_REAL_CMD}."
            ),
        ),
        StrategyPose(
            name="iterative_closure",
            label="Iterative closure",
            ctrl_values=iterative_ctrl,
            target_result=iterative_result,
            notes=(
                "Iterative closure pre-grasp matches the planner approach width "
                "selection: by default the hand starts at the mode's maximum "
                "analytical width, with thumb yaw held at the final analytical "
                "value, then thumb and fingers close through width-space waypoints."
            ),
        ),
        StrategyPose(
            name="thumb_reflex",
            label="Thumb reflex",
            ctrl_values=reflex_ctrl,
            target_result=final_result,
            notes="Final thumb bend/yaw with all non-thumb fingers open.",
        ),
    ]


def build_capsules_for_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_values: dict[str, float],
    *,
    radius_scale: float,
) -> list[Capsule]:
    set_closure_qpos(model, data, ctrl_values)
    return build_capsule_proxy(model, data, radius_scale=radius_scale)


def floor_clearance(capsules: list[Capsule], floor_z_m: float) -> tuple[float, str]:
    best_clearance = math.inf
    best_name = ""
    for capsule in capsules:
        min_z = min(float(capsule.p0[2]), float(capsule.p1[2]))
        clearance = min_z - float(capsule.radius) - floor_z_m
        if clearance < best_clearance:
            best_clearance = clearance
            best_name = f"{capsule.group}/{capsule.name}"
    return best_clearance, best_name


def object_clearance_from_path(rows: list[dict[str, object]]) -> tuple[float, str, int, bool]:
    usable = [row for row in rows if not bool(row["ignored_for_final_contact"])] or rows
    nearest_row = min(usable, key=lambda row: float(row["clearance_m"]))
    nearest = f"{nearest_row['nearest_group']}/{nearest_row['nearest_capsule']}"
    max_active = max(int(row["active_collision_count"]) for row in rows)
    collided = any(bool(row["collision"]) for row in rows)
    return float(nearest_row["clearance_m"]), nearest, max_active, collided


def object_clearance_static(
    capsules: list[Capsule],
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    object_shape: str,
) -> tuple[float, str, bool]:
    records = capsule_collisions_against_object(
        capsules,
        half_extents,
        object_center=aabb_center,
        object_shape=object_shape,
    )
    nearest = min(records, key=lambda record: record.clearance)
    return (
        float(nearest.clearance),
        f"{nearest.capsule_group}/{nearest.capsule_name}",
        any(record.intersects for record in records),
    )


def evaluate_strategy(
    strategy: StrategyPose,
    *,
    capsules_base: list[Capsule],
    start: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    object_shape: str,
    floor_z_m: float,
    floor_tolerance_m: float,
    path_samples: int,
) -> Evaluation:
    capsules_start = transform_capsules(capsules_base, rotation, start)
    capsules_target = transform_capsules(capsules_base, rotation, target)
    path_rows = sample_linear_path_collisions(
        capsules_base,
        start=start,
        final=target,
        rotation=rotation,
        half_extents=half_extents,
        aabb_center=aabb_center,
        object_shape=object_shape,
        path_samples=path_samples,
        final_ignore_m=0.0,
        final_ignore_groups=(),
    )
    path_obj_clearance, nearest_obj, max_active, path_obj_collided = object_clearance_from_path(path_rows)
    target_obj_clearance, target_nearest_obj, target_obj_collided = object_clearance_static(
        capsules_target,
        half_extents,
        aabb_center,
        object_shape,
    )

    start_floor_clearance, start_floor_nearest = floor_clearance(capsules_start, floor_z_m)
    target_floor_clearance, target_floor_nearest = floor_clearance(capsules_target, floor_z_m)
    path_floor_clearance = min(start_floor_clearance, target_floor_clearance)
    floor_nearest = (
        start_floor_nearest
        if start_floor_clearance <= target_floor_clearance
        else target_floor_nearest
    )

    # Prefer target-pose object information when target collision is the blocker.
    nearest = target_nearest_obj if target_obj_collided else nearest_obj
    min_object_clearance = min(path_obj_clearance, target_obj_clearance)
    if path_floor_clearance < min_object_clearance:
        nearest = floor_nearest

    return Evaluation(
        strategy=strategy,
        start=start,
        target=target,
        object_path_collision=path_obj_collided,
        object_target_collision=target_obj_collided,
        floor_path_collision=path_floor_clearance < -floor_tolerance_m,
        floor_target_collision=target_floor_clearance < -floor_tolerance_m,
        min_object_clearance_m=min_object_clearance,
        min_floor_clearance_m=path_floor_clearance,
        nearest_object=nearest,
        max_active_object_collisions=max_active,
        path_rows=path_rows,
        capsules_start=capsules_start,
        capsules_target=capsules_target,
    )


def candidate_starts(final: np.ndarray) -> list[np.ndarray]:
    offsets_mm = [
        (-140, 0, 80),
        (-100, -80, 60),
        (-100, 80, 60),
        (0, -140, 70),
        (0, 140, 70),
        (100, -80, 60),
        (100, 80, 60),
        (140, 0, 80),
        (-80, 0, 30),
        (80, 0, 30),
        (0, -80, 30),
        (0, 80, 30),
    ]
    return [final + np.asarray(offset, dtype=float) / 1000.0 for offset in offsets_mm]


def choose_start(
    strategies: list[StrategyPose],
    capsules_by_strategy: dict[str, list[Capsule]],
    *,
    final_grasp: np.ndarray,
    target_by_strategy: dict[str, np.ndarray],
    rotation_by_strategy: dict[str, np.ndarray],
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    object_shape: str,
    floor_z_m: float,
    floor_tolerance_m: float,
    path_samples: int,
) -> tuple[np.ndarray, list[Evaluation]]:
    best_score = -math.inf
    best_start: np.ndarray | None = None
    best_evals: list[Evaluation] = []

    for start in candidate_starts(final_grasp):
        evals = [
            evaluate_strategy(
                strategy,
                capsules_base=capsules_by_strategy[strategy.name],
                start=start,
                target=target_by_strategy[strategy.name],
                rotation=rotation_by_strategy[strategy.name],
                half_extents=half_extents,
                aabb_center=aabb_center,
                object_shape=object_shape,
                floor_z_m=floor_z_m,
                floor_tolerance_m=floor_tolerance_m,
                path_samples=path_samples,
            )
            for strategy in strategies
        ]
        valid_count = sum(int(e.valid) for e in evals)
        mixed_bonus = 3.0 if 0 < valid_count < len(evals) else 0.0
        floor_bonus = 1.5 * sum(int(e.floor_path_collision or e.floor_target_collision) for e in evals)
        object_bonus = sum(int(e.object_path_collision or e.object_target_collision) for e in evals)
        clearance_bonus = -0.001 * abs(float(np.linalg.norm(start - final_grasp)) * 1000.0 - 150.0)
        score = mixed_bonus + floor_bonus + object_bonus + clearance_bonus
        if score > best_score:
            best_score = score
            best_start = start
            best_evals = evals

    if best_start is None:
        raise RuntimeError("Could not choose a strategy pre-grasp demo start")
    return best_start, best_evals


def write_summary(path: Path, evals: list[Evaluation]) -> None:
    fields = [
        "strategy",
        "label",
        "valid",
        "dominant_blocker",
        "start_x_mm",
        "start_y_mm",
        "start_z_mm",
        "strategy_target_x_mm",
        "strategy_target_y_mm",
        "strategy_target_z_mm",
        "strategy_target_width_mm",
        "object_path_collision",
        "object_target_collision",
        "floor_path_collision",
        "floor_target_collision",
        "min_object_clearance_mm",
        "min_floor_clearance_mm",
        "nearest_blocker",
        "max_active_object_collisions",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in evals:
            writer.writerow(
                {
                    "strategy": e.strategy.name,
                    "label": e.strategy.label,
                    "valid": int(e.valid),
                    "dominant_blocker": e.dominant_blocker,
                    "start_x_mm": f"{e.start[0] * 1000.0:.3f}",
                    "start_y_mm": f"{e.start[1] * 1000.0:.3f}",
                    "start_z_mm": f"{e.start[2] * 1000.0:.3f}",
                    "strategy_target_x_mm": f"{e.target[0] * 1000.0:.3f}",
                    "strategy_target_y_mm": f"{e.target[1] * 1000.0:.3f}",
                    "strategy_target_z_mm": f"{e.target[2] * 1000.0:.3f}",
                    "strategy_target_width_mm": f"{e.strategy.target_result.width * 1000.0:.3f}",
                    "object_path_collision": int(e.object_path_collision),
                    "object_target_collision": int(e.object_target_collision),
                    "floor_path_collision": int(e.floor_path_collision),
                    "floor_target_collision": int(e.floor_target_collision),
                    "min_object_clearance_mm": f"{e.min_object_clearance_m * 1000.0:.3f}",
                    "min_floor_clearance_mm": f"{e.min_floor_clearance_m * 1000.0:.3f}",
                    "nearest_blocker": e.nearest_object,
                    "max_active_object_collisions": e.max_active_object_collisions,
                    "notes": e.strategy.notes,
                }
            )


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    obj: ObjectSpec,
    mode: str,
    target_width_m: float,
    final_grasp: np.ndarray,
    start: np.ndarray,
    strategies: list[StrategyPose],
    target_by_strategy: dict[str, np.ndarray],
) -> None:
    payload = {
        "script": "tools/demo_strategy_pregrasp_collision.py",
        "simulation_only": True,
        "uses_hardware": False,
        "purpose": (
            "One-example check before turning strategy-specific pre-grasp "
            "collision logic into a no-go volume sweep."
        ),
        "object": {
            "name": obj.name,
            "label": obj.label,
            "size_m": obj.size_m,
            "grasp_width_m": obj.grasp_width_m,
            "collision_shape": obj.collision_shape,
            "aabb_center_m": tabletop_object_center(obj).tolist(),
            "grasp_target_fraction": obj.grasp_target_fraction,
            "grasp_target_top_offset_m": obj.grasp_target_top_offset_m,
            "grasp_target_m": tabletop_grasp_target(obj).tolist(),
        },
        "mode": mode,
        "target_width_m": target_width_m,
        "object_width_offset_mm": args.object_width_offset_mm,
        "grasp_center_policy": args.grasp_center_policy,
        "preopen_mm": args.preopen_mm,
        "iterative_width_mm": args.iterative_width_mm,
        "iterative_pregrasp_policy": args.iterative_pregrasp_policy,
        "pregrasp_width_policy": pregrasp_width_policy(args),
        "yaw_deg": args.yaw_deg,
        "start_m": start.tolist(),
        "start_definition": (
            "The sampled start hand-base pose is expressed in world coordinates. "
            "Candidate starts are final_grasp_base plus fixed xyz offsets, so all "
            "strategies are tested from the same world-frame starts."
        ),
        "final_grasp_m": final_grasp.tolist(),
        "path_samples": args.path_samples,
        "floor_z_m": args.floor_z_mm / 1000.0,
        "floor_tolerance_mm": args.floor_tolerance_mm,
        "radius_scale": args.radius_scale,
        "strategies": {
            strategy.name: {
                "label": strategy.label,
                "ctrl_values": {
                    key: float(value)
                    for key, value in strategy.ctrl_values.items()
                },
                "target_m": target_by_strategy[strategy.name].tolist(),
                "target_width_m": float(strategy.target_result.width),
                "notes": strategy.notes,
            }
            for strategy in strategies
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _box_edges(center: np.ndarray, half_extents: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    corners = center + signs * half_extents
    pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return [(corners[i], corners[j]) for i, j in pairs]


def _object_edges(
    center: np.ndarray,
    half_extents: np.ndarray,
    object_shape: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if object_shape == "box":
        return _box_edges(center, half_extents)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    angles = np.linspace(0.0, 2.0 * math.pi, 33)
    if object_shape == "cylinder":
        radius = float(half_extents[0])
        for z in (center[2] - half_extents[2], center[2] + half_extents[2]):
            points = [
                center + np.array([radius * math.cos(a), radius * math.sin(a), z - center[2]])
                for a in angles
            ]
            segments.extend(zip(points[:-1], points[1:]))
        for angle in np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False):
            xy = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
            segments.append(
                (
                    center + xy - np.array([0.0, 0.0, half_extents[2]]),
                    center + xy + np.array([0.0, 0.0, half_extents[2]]),
                )
            )
        return segments
    if object_shape == "sphere":
        radius = float(half_extents[0])
        for axis in range(3):
            points = []
            for angle in angles:
                point = np.zeros(3)
                point[(axis + 1) % 3] = radius * math.cos(angle)
                point[(axis + 2) % 3] = radius * math.sin(angle)
                points.append(center + point)
            segments.extend(zip(points[:-1], points[1:]))
        return segments
    raise ValueError(f"Unsupported object collision shape: {object_shape}")


def maybe_write_plot(
    path: Path,
    *,
    evals: list[Evaluation],
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    object_shape: str,
    floor_z_m: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (path.parent / "plot_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return

    fig = plt.figure(figsize=(14.5, 4.8))
    colors = {
        "naive": "#d55e00",
        "iterative_closure": "#0072b2",
        "thumb_reflex": "#009e73",
    }
    all_points: list[np.ndarray] = []
    for e in evals:
        all_points.extend([e.start, e.target])
        for capsule in e.capsules_start + e.capsules_target:
            all_points.extend([capsule.p0, capsule.p1])
    all_points.append(aabb_center + half_extents)
    all_points.append(aabb_center - half_extents)
    pts = np.vstack(all_points)
    center = pts.mean(axis=0)
    span = max(float((pts.max(axis=0) - pts.min(axis=0)).max()), 0.18)
    radius = span * 0.58

    for idx, e in enumerate(evals, start=1):
        ax = fig.add_subplot(1, len(evals), idx, projection="3d")
        color = colors[e.strategy.name]
        for p0, p1 in _object_edges(aabb_center, half_extents, object_shape):
            ax.plot(
                [p0[0] * 1000, p1[0] * 1000],
                [p0[1] * 1000, p1[1] * 1000],
                [p0[2] * 1000, p1[2] * 1000],
                color="black",
                linewidth=1.2,
            )
        gx, gy = np.meshgrid(
            np.linspace((center[0] - radius) * 1000, (center[0] + radius) * 1000, 2),
            np.linspace((center[1] - radius) * 1000, (center[1] + radius) * 1000, 2),
        )
        gz = np.full_like(gx, floor_z_m * 1000.0)
        ax.plot_surface(gx, gy, gz, color="#dddddd", alpha=0.22, linewidth=0)
        ax.plot(
            [e.start[0] * 1000, e.target[0] * 1000],
            [e.start[1] * 1000, e.target[1] * 1000],
            [e.start[2] * 1000, e.target[2] * 1000],
            color=color,
            linewidth=2.0,
        )
        ax.scatter(
            [e.start[0] * 1000],
            [e.start[1] * 1000],
            [e.start[2] * 1000],
            color=color,
            s=35,
            label="start",
        )
        ax.scatter(
            [e.target[0] * 1000],
            [e.target[1] * 1000],
            [e.target[2] * 1000],
            color="black",
            s=25,
            label="target",
        )
        for capsule in e.capsules_start:
            ax.plot(
                [capsule.p0[0] * 1000, capsule.p1[0] * 1000],
                [capsule.p0[1] * 1000, capsule.p1[1] * 1000],
                [capsule.p0[2] * 1000, capsule.p1[2] * 1000],
                color=color,
                alpha=0.22,
                linewidth=max(1.0, capsule.radius * 1000.0 * 0.35),
            )
        for capsule in e.capsules_target:
            ax.plot(
                [capsule.p0[0] * 1000, capsule.p1[0] * 1000],
                [capsule.p0[1] * 1000, capsule.p1[1] * 1000],
                [capsule.p0[2] * 1000, capsule.p1[2] * 1000],
                color=color,
                alpha=0.78,
                linewidth=max(1.0, capsule.radius * 1000.0 * 0.45),
            )
        status = "valid" if e.valid else f"blocked: {e.dominant_blocker}"
        ax.set_title(
            f"{e.strategy.name}\n{status}\n"
            f"obj {e.min_object_clearance_m * 1000:.1f} mm, "
            f"floor {e.min_floor_clearance_m * 1000:.1f} mm",
            fontsize=9,
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_xlim((center[0] - radius) * 1000, (center[0] + radius) * 1000)
        ax.set_ylim((center[1] - radius) * 1000, (center[1] + radius) * 1000)
        ax.set_zlim(max(-20.0, (center[2] - radius) * 1000), (center[2] + radius) * 1000)
        ax.view_init(elev=22, azim=-54)

    fig.suptitle("Strategy pre-grasp capsule collision demo", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    obj = BUILTIN_OBJECTS[args.object]
    mode = obj.mode if args.mode == "object-default" else args.mode
    width_offset_m = args.object_width_offset_mm / 1000.0
    target_width_m = obj.grasp_width_m + width_offset_m
    preopen_m = args.preopen_mm / 1000.0
    iterative_width_m = (
        args.iterative_width_mm / 1000.0
        if args.iterative_width_mm is not None
        else None
    )
    floor_z_m = args.floor_z_mm / 1000.0
    floor_tolerance_m = args.floor_tolerance_mm / 1000.0

    fk = (
        InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk)
        if args.xml
        else InspireHandFK(rebuild=args.rebuild_fk)
    )
    closure = ClosureGeometry(fk)
    final_result = solve_mode(closure, mode, target_width_m)
    yaw_rad = math.radians(args.yaw_deg)
    aabb_center = tabletop_object_center(obj)
    grasp_target = tabletop_grasp_target(obj)
    half_extents = np.asarray(obj.size_m, dtype=float) / 2.0
    final_grasp = closure_base_position_for_center(
        final_result,
        yaw_rad,
        object_center=grasp_target,
        policy=args.grasp_center_policy,
    )

    strategies = strategy_poses(
        fk,
        closure,
        mode=mode,
        final_result=final_result,
        target_width_m=target_width_m,
        preopen_m=preopen_m,
        iterative_width_m=iterative_width_m,
        iterative_pregrasp_policy=args.iterative_pregrasp_policy,
    )

    model = mujoco.MjModel.from_xml_path(str(args.xml or fk.xml_path))
    data = mujoco.MjData(model)
    capsules_by_strategy = {
        strategy.name: build_capsules_for_ctrl(
            model,
            data,
            strategy.ctrl_values,
            radius_scale=args.radius_scale,
        )
        for strategy in strategies
    }
    target_by_strategy = {
        strategy.name: closure_base_position_for_center(
            strategy.target_result,
            yaw_rad,
            object_center=grasp_target,
            policy=args.grasp_center_policy,
        )
        for strategy in strategies
    }
    rotation_by_strategy = {
        strategy.name: closure_base_rotation(strategy.target_result, yaw_rad)
        for strategy in strategies
    }

    if args.start_mm is None:
        start, evals = choose_start(
            strategies,
            capsules_by_strategy,
            final_grasp=final_grasp,
            target_by_strategy=target_by_strategy,
            rotation_by_strategy=rotation_by_strategy,
            half_extents=half_extents,
            aabb_center=aabb_center,
            object_shape=obj.collision_shape,
            floor_z_m=floor_z_m,
            floor_tolerance_m=floor_tolerance_m,
            path_samples=args.path_samples,
        )
    else:
        start = np.asarray(args.start_mm, dtype=float) / 1000.0
        evals = [
            evaluate_strategy(
                strategy,
                capsules_base=capsules_by_strategy[strategy.name],
                start=start,
                target=target_by_strategy[strategy.name],
                rotation=rotation_by_strategy[strategy.name],
                half_extents=half_extents,
                aabb_center=aabb_center,
                object_shape=obj.collision_shape,
                floor_z_m=floor_z_m,
                floor_tolerance_m=floor_tolerance_m,
                path_samples=args.path_samples,
            )
            for strategy in strategies
        ]

    write_summary(args.out / "summary.csv", evals)
    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        obj=obj,
        mode=mode,
        target_width_m=target_width_m,
        final_grasp=final_grasp,
        start=start,
        strategies=strategies,
        target_by_strategy=target_by_strategy,
    )
    if not args.no_plot:
        maybe_write_plot(
            args.out / "strategy_pregrasp_demo.png",
            evals=evals,
            half_extents=half_extents,
            aabb_center=aabb_center,
            object_shape=obj.collision_shape,
            floor_z_m=floor_z_m,
        )

    print("RH56 strategy pre-grasp demo:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  object: {obj.name}")
    print(f"  mode: {mode}")
    print(f"  pregrasp_width_policy: {pregrasp_width_policy(args)}")
    print(f"  start_mm: {[round(v, 3) for v in (start * 1000.0).tolist()]}")
    print(f"  final_grasp_mm: {[round(v, 3) for v in (final_grasp * 1000.0).tolist()]}")
    for e in evals:
        status = "valid" if e.valid else f"blocked:{e.dominant_blocker}"
        print(
            f"  {e.strategy.name}: {status} "
            f"target_mm={[round(v, 3) for v in (e.target * 1000.0).tolist()]} "
            f"object_clearance_mm={e.min_object_clearance_m * 1000.0:.3f} "
            f"floor_clearance_mm={e.min_floor_clearance_m * 1000.0:.3f}"
        )
    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'assumptions.json'}")
    if not args.no_plot:
        print(f"Wrote {args.out / 'strategy_pregrasp_demo.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
