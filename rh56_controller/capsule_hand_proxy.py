"""Capsule proxy geometry for RH56 analytical-grasp reachability checks.

The helpers in this module intentionally sit between the very coarse
"hand-base point" proxy and full MuJoCo mesh collision.  They use MuJoCo FK to
extract link endpoints, then represent the palm and phalanges with simple
capsules that can be swept through object-space quickly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation

from rh56_controller.grasp_geometry import ClosureResult


@dataclass(frozen=True)
class Capsule:
    """A capsule segment in metres."""

    name: str
    group: str
    p0: np.ndarray
    p1: np.ndarray
    radius: float
    source: str


@dataclass(frozen=True)
class CapsuleCollision:
    """Distance result for one capsule against one object proxy."""

    capsule_name: str
    capsule_group: str
    clearance: float
    distance: float
    radius: float
    t_segment: float
    t_path: float = 0.0
    path_alpha: float = 0.0
    near_final: bool = False

    @property
    def intersects(self) -> bool:
        return self.clearance < -COLLISION_NUMERICAL_EPSILON_M


# Optimized point/segment distances can land a few micrometres below zero at
# exact analytical fingertip contact. This epsilon rejects numerical overlap
# without masking physically meaningful penetration.
COLLISION_NUMERICAL_EPSILON_M = 1e-5


FINGER_RADII_M = {
    "thumb": 0.010,
    "index": 0.008,
    "middle": 0.0085,
    "ring": 0.008,
    "pinky": 0.0075,
}

FINGER_SEGMENTS = {
    "thumb": (
        ("thumb_proximal_base", "thumb_proximal"),
        ("thumb_proximal", "thumb_intermediate"),
        ("thumb_intermediate", "thumb_distal"),
        ("thumb_distal", "site:right_thumb_tip"),
    ),
    "index": (
        ("index_proximal", "index_intermediate"),
        ("index_intermediate", "site:right_index_tip"),
    ),
    "middle": (
        ("middle_proximal", "middle_intermediate"),
        ("middle_intermediate", "site:right_middle_tip"),
    ),
    "ring": (
        ("ring_proximal", "ring_intermediate"),
        ("ring_intermediate", "site:right_ring_tip"),
    ),
    "pinky": (
        ("pinky_proximal", "pinky_intermediate"),
        ("pinky_intermediate", "site:right_pinky_tip"),
    ),
}

# Palm capsules are defined in the hand-base frame.  They intentionally form a
# conservative envelope around the MuJoCo palm boxes in inspire_grasp_scene.xml.
PALM_CAPSULES_BASE = (
    ("palm_center", np.array([0.003, 0.000, 0.030]), np.array([0.001, 0.000, 0.130]), 0.024),
    ("palm_index_side", np.array([0.002, 0.027, 0.055]), np.array([0.001, 0.027, 0.132]), 0.013),
    ("palm_pinky_side", np.array([0.002, -0.027, 0.055]), np.array([0.001, -0.027, 0.132]), 0.013),
)


def closure_base_rotation(result: ClosureResult, yaw_rad: float) -> np.ndarray:
    """Return world-from-hand-base rotation for an analytical closure yaw."""

    cz, sz = math.cos(yaw_rad), math.sin(yaw_rad)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ClosureResult._rot_matrix(result.base_tilt_y)


def closure_base_position(
    result: ClosureResult,
    yaw_rad: float,
    *,
    object_center: np.ndarray | None = None,
) -> np.ndarray:
    """Place the closure midpoint at the object center."""

    center = np.zeros(3, dtype=float) if object_center is None else np.asarray(object_center, dtype=float)
    return center - (closure_base_rotation(result, yaw_rad) @ result.midpoint)


def rotation_to_grasp_scene_euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation matrix to qpos values for the grasp-scene XYZ hinges.

    MuJoCo's three same-body hinge joints match scipy's intrinsic ``zyx``
    convention when values are mapped back to x/y/z joint addresses.
    """

    z_rad, y_rad, x_rad = Rotation.from_matrix(rotation).as_euler("zyx")
    return float(x_rad), float(y_rad), float(z_rad)


def set_closure_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_values: dict[str, float],
    *,
    base_position: np.ndarray | None = None,
    base_rotation: np.ndarray | None = None,
) -> None:
    """Write analytical closure qpos into a MuJoCo grasp-scene model."""

    def qadr(joint_name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(model.jnt_qposadr[jid]) if jid >= 0 else -1

    def set_joint(joint_name: str, value: float) -> None:
        adr = qadr(joint_name)
        if adr >= 0:
            data.qpos[adr] = float(value)

    data.qpos[:] = 0.0

    if base_position is not None:
        set_joint("right_pos_x", float(base_position[0]))
        set_joint("right_pos_y", float(base_position[1]))
        set_joint("right_pos_z", float(base_position[2]))
    if base_rotation is not None:
        rx, ry, rz = rotation_to_grasp_scene_euler_xyz(base_rotation)
        set_joint("right_rot_x", rx)
        set_joint("right_rot_y", ry)
        set_joint("right_rot_z", rz)

    pinky = float(ctrl_values.get("pinky", 0.0))
    ring = float(ctrl_values.get("ring", 0.0))
    middle = float(ctrl_values.get("middle", 0.0))
    index = float(ctrl_values.get("index", 0.0))
    thumb_pitch = float(ctrl_values.get("thumb_proximal", 0.0))
    thumb_yaw = float(ctrl_values.get("thumb_yaw", 0.0))

    set_joint("pinky_proximal_joint", pinky)
    set_joint("pinky_intermediate_joint", -0.15 + 1.1169 * pinky)
    set_joint("ring_proximal_joint", ring)
    set_joint("ring_intermediate_joint", -0.15 + 1.1169 * ring)
    set_joint("middle_proximal_joint", middle)
    set_joint("middle_intermediate_joint", -0.15 + 1.1169 * middle)
    set_joint("index_proximal_joint", index)
    set_joint("index_intermediate_joint", -0.05 + 1.1169 * index)
    set_joint("thumb_proximal_yaw_joint", thumb_yaw)
    set_joint("thumb_proximal_pitch_joint", thumb_pitch)
    set_joint("thumb_intermediate_joint", 0.15 + 1.33 * thumb_pitch)
    set_joint("thumb_distal_joint", 0.15 + 0.66 * thumb_pitch)

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body not found in MuJoCo model: {name}")
    return data.xpos[bid].copy()


def _site_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"Site not found in MuJoCo model: {name}")
    return data.site_xpos[sid].copy()


def _endpoint(model: mujoco.MjModel, data: mujoco.MjData, token: str) -> np.ndarray:
    if token.startswith("site:"):
        return _site_position(model, data, token.split(":", 1)[1])
    return _body_position(model, data, token)


def build_capsule_proxy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    radius_scale: float = 1.0,
    include_palm: bool = True,
    fingers: Iterable[str] = ("thumb", "index", "middle", "ring", "pinky"),
) -> list[Capsule]:
    """Extract hand capsules from the model's current FK state."""

    capsules: list[Capsule] = []
    finger_set = tuple(fingers)
    for finger in finger_set:
        radius = FINGER_RADII_M[finger] * radius_scale
        for idx, (start_token, end_token) in enumerate(FINGER_SEGMENTS[finger]):
            capsules.append(
                Capsule(
                    name=f"{finger}_{idx}",
                    group=finger,
                    p0=_endpoint(model, data, start_token),
                    p1=_endpoint(model, data, end_token),
                    radius=radius,
                    source=f"{start_token}->{end_token}",
                )
            )

    if include_palm:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id < 0:
            raise ValueError("Body not found in MuJoCo model: base")
        base_pos = data.xpos[base_id].copy()
        base_rot = data.xmat[base_id].reshape(3, 3).copy()
        for name, local_p0, local_p1, radius in PALM_CAPSULES_BASE:
            capsules.append(
                Capsule(
                    name=name,
                    group="palm",
                    p0=base_pos + base_rot @ local_p0,
                    p1=base_pos + base_rot @ local_p1,
                    radius=radius * radius_scale,
                    source="hand-base-frame palm envelope",
                )
            )

    return capsules


def transform_capsules(
    capsules: Iterable[Capsule],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[Capsule]:
    """Rigidly transform capsules from hand-base frame to world frame."""

    return [
        Capsule(
            name=capsule.name,
            group=capsule.group,
            p0=translation + rotation @ capsule.p0,
            p1=translation + rotation @ capsule.p1,
            radius=capsule.radius,
            source=capsule.source,
        )
        for capsule in capsules
    ]


def point_aabb_distance(
    point: np.ndarray,
    half_extents: np.ndarray,
    *,
    aabb_center: np.ndarray | None = None,
) -> float:
    """Euclidean distance from a point to an AABB."""

    center = np.zeros(3, dtype=float) if aabb_center is None else np.asarray(aabb_center, dtype=float)
    outside = np.maximum(np.abs(point - center) - half_extents, 0.0)
    return float(np.linalg.norm(outside))


def point_cylinder_distance(
    point: np.ndarray,
    half_extents: np.ndarray,
    *,
    object_center: np.ndarray | None = None,
) -> float:
    """Euclidean distance from a point to a solid upright cylinder."""

    center = (
        np.zeros(3, dtype=float)
        if object_center is None
        else np.asarray(object_center, dtype=float)
    )
    half_extents = np.asarray(half_extents, dtype=float)
    if not np.isclose(half_extents[0], half_extents[1], atol=1e-9):
        raise ValueError("cylinder collision proxies require equal x/y radii")
    radial_outside = float(np.linalg.norm(np.asarray(point)[:2] - center[:2])) - float(
        half_extents[0]
    )
    axial_outside = abs(float(point[2] - center[2])) - float(half_extents[2])
    outside = np.maximum(np.array([radial_outside, axial_outside]), 0.0)
    return float(np.linalg.norm(outside))


def point_sphere_distance(
    point: np.ndarray,
    half_extents: np.ndarray,
    *,
    object_center: np.ndarray | None = None,
) -> float:
    """Euclidean distance from a point to a solid sphere."""

    center = (
        np.zeros(3, dtype=float)
        if object_center is None
        else np.asarray(object_center, dtype=float)
    )
    half_extents = np.asarray(half_extents, dtype=float)
    if not np.allclose(half_extents, half_extents[0], atol=1e-9):
        raise ValueError("sphere collision proxies require equal x/y/z radii")
    return max(float(np.linalg.norm(np.asarray(point) - center) - half_extents[0]), 0.0)


def point_object_distance(
    point: np.ndarray,
    half_extents: np.ndarray,
    *,
    object_center: np.ndarray | None = None,
    object_shape: str = "box",
) -> float:
    """Euclidean distance from a point to a supported solid object proxy."""

    if object_shape == "box":
        return point_aabb_distance(point, half_extents, aabb_center=object_center)
    if object_shape == "cylinder":
        return point_cylinder_distance(
            point,
            half_extents,
            object_center=object_center,
        )
    if object_shape == "sphere":
        return point_sphere_distance(
            point,
            half_extents,
            object_center=object_center,
        )
    raise ValueError(f"Unsupported object collision shape: {object_shape}")


def segment_aabb_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    half_extents: np.ndarray,
    *,
    aabb_center: np.ndarray | None = None,
) -> tuple[float, float]:
    """Numerically compute shortest distance from a segment to an AABB.

    The objective is a convex piecewise-quadratic function of segment parameter,
    so bounded scalar minimization gives a stable capsule-vs-AABB collision
    check without adding a long custom geometry routine.
    """

    delta = p1 - p0
    if np.linalg.norm(delta) < 1e-12:
        return point_aabb_distance(p0, half_extents, aabb_center=aabb_center), 0.0

    def squared_distance(t: float) -> float:
        distance = point_aabb_distance(
            p0 + float(t) * delta,
            half_extents,
            aabb_center=aabb_center,
        )
        return distance * distance

    result = minimize_scalar(
        squared_distance,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return math.sqrt(max(float(result.fun), 0.0)), float(result.x)


def segment_object_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    half_extents: np.ndarray,
    *,
    object_center: np.ndarray | None = None,
    object_shape: str = "box",
) -> tuple[float, float]:
    """Compute shortest centerline distance to a supported object proxy."""

    if object_shape == "box":
        return segment_aabb_distance(
            p0,
            p1,
            half_extents,
            aabb_center=object_center,
        )

    delta = p1 - p0
    if np.linalg.norm(delta) < 1e-12:
        return (
            point_object_distance(
                p0,
                half_extents,
                object_center=object_center,
                object_shape=object_shape,
            ),
            0.0,
        )

    def squared_distance(t: float) -> float:
        distance = point_object_distance(
            p0 + float(t) * delta,
            half_extents,
            object_center=object_center,
            object_shape=object_shape,
        )
        return distance * distance

    result = minimize_scalar(
        squared_distance,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return math.sqrt(max(float(result.fun), 0.0)), float(result.x)


def swept_segment_aabb_distance(
    p0_start: np.ndarray,
    p1_start: np.ndarray,
    p0_end: np.ndarray,
    p1_end: np.ndarray,
    half_extents: np.ndarray,
    *,
    aabb_center: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute shortest distance from a linearly swept segment to an AABB.

    During the capsule-path check the hand orientation is fixed and the hand
    base translates linearly.  Each capsule centerline therefore sweeps a
    ruled surface spanned by the capsule axis and the path translation.  We
    minimize point-to-AABB distance over those two parameters instead of only
    checking the visible samples.
    """

    axis = p1_start - p0_start
    path_delta0 = p0_end - p0_start
    path_delta1 = p1_end - p1_start
    path_delta = 0.5 * (path_delta0 + path_delta1)

    if np.linalg.norm(axis) < 1e-12:
        distance, t_path = segment_aabb_distance(
            p0_start,
            p0_end,
            half_extents,
            aabb_center=aabb_center,
        )
        return distance, 0.0, t_path
    if np.linalg.norm(path_delta) < 1e-12:
        distance, t_segment = segment_aabb_distance(
            p0_start,
            p1_start,
            half_extents,
            aabb_center=aabb_center,
        )
        return distance, t_segment, 0.0

    def squared_distance(params: np.ndarray) -> float:
        t_segment = float(params[0])
        t_path = float(params[1])
        point = p0_start + t_segment * axis + t_path * path_delta
        distance = point_aabb_distance(point, half_extents, aabb_center=aabb_center)
        return distance * distance

    starts = (
        np.array([0.5, 0.5]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    best_fun = math.inf
    best_x = starts[0]
    for guess in starts:
        result = minimize(
            squared_distance,
            guess,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            method="L-BFGS-B",
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 100},
        )
        value = float(result.fun)
        if value < best_fun:
            best_fun = value
            best_x = np.asarray(result.x, dtype=float)

    return math.sqrt(max(best_fun, 0.0)), float(best_x[0]), float(best_x[1])


def swept_segment_object_distance(
    p0_start: np.ndarray,
    p1_start: np.ndarray,
    p0_end: np.ndarray,
    p1_end: np.ndarray,
    half_extents: np.ndarray,
    *,
    object_center: np.ndarray | None = None,
    object_shape: str = "box",
) -> tuple[float, float, float]:
    """Compute shortest distance from a swept segment to an object proxy."""

    if object_shape == "box":
        return swept_segment_aabb_distance(
            p0_start,
            p1_start,
            p0_end,
            p1_end,
            half_extents,
            aabb_center=object_center,
        )

    axis = p1_start - p0_start
    path_delta = 0.5 * ((p0_end - p0_start) + (p1_end - p1_start))
    if np.linalg.norm(axis) < 1e-12:
        distance, t_path = segment_object_distance(
            p0_start,
            p0_end,
            half_extents,
            object_center=object_center,
            object_shape=object_shape,
        )
        return distance, 0.0, t_path
    if np.linalg.norm(path_delta) < 1e-12:
        distance, t_segment = segment_object_distance(
            p0_start,
            p1_start,
            half_extents,
            object_center=object_center,
            object_shape=object_shape,
        )
        return distance, t_segment, 0.0

    def squared_distance(params: np.ndarray) -> float:
        t_segment = float(params[0])
        t_path = float(params[1])
        point = p0_start + t_segment * axis + t_path * path_delta
        distance = point_object_distance(
            point,
            half_extents,
            object_center=object_center,
            object_shape=object_shape,
        )
        return distance * distance

    starts = (
        np.array([0.5, 0.5]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    best_fun = math.inf
    best_x = starts[0]
    for guess in starts:
        result = minimize(
            squared_distance,
            guess,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            method="L-BFGS-B",
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 100},
        )
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x, dtype=float)

    return math.sqrt(max(best_fun, 0.0)), float(best_x[0]), float(best_x[1])


def closest_capsule_to_aabb(
    capsules: Iterable[Capsule],
    half_extents: np.ndarray,
    *,
    aabb_center: np.ndarray | None = None,
) -> CapsuleCollision:
    """Return the capsule with the smallest clearance to an AABB."""

    collisions = capsule_collisions_against_aabb(
        capsules,
        half_extents,
        aabb_center=aabb_center,
    )
    if not collisions:
        raise ValueError("No capsules were provided")
    return min(collisions, key=lambda collision: collision.clearance)


def capsule_collisions_against_aabb(
    capsules: Iterable[Capsule],
    half_extents: np.ndarray,
    *,
    aabb_center: np.ndarray | None = None,
    path_alpha: float = 0.0,
    near_final: bool = False,
) -> list[CapsuleCollision]:
    """Return clearance records for every capsule against an AABB."""

    return capsule_collisions_against_object(
        capsules,
        half_extents,
        object_center=aabb_center,
        object_shape="box",
        path_alpha=path_alpha,
        near_final=near_final,
    )


def capsule_collisions_against_object(
    capsules: Iterable[Capsule],
    half_extents: np.ndarray,
    *,
    object_center: np.ndarray | None = None,
    object_shape: str = "box",
    path_alpha: float = 0.0,
    near_final: bool = False,
) -> list[CapsuleCollision]:
    """Return clearance records for every capsule against an object proxy."""

    collisions: list[CapsuleCollision] = []
    for capsule in capsules:
        distance, t_segment = segment_object_distance(
            capsule.p0,
            capsule.p1,
            half_extents,
            object_center=object_center,
            object_shape=object_shape,
        )
        clearance = distance - capsule.radius
        collisions.append(
            CapsuleCollision(
                capsule_name=capsule.name,
                capsule_group=capsule.group,
                clearance=float(clearance),
                distance=float(distance),
                radius=float(capsule.radius),
                t_segment=float(t_segment),
                path_alpha=float(path_alpha),
                near_final=bool(near_final),
            )
        )
    return collisions


def swept_capsule_collisions_against_aabb(
    capsules_start: Iterable[Capsule],
    capsules_end: Iterable[Capsule],
    half_extents: np.ndarray,
    *,
    alpha_start: float,
    alpha_end: float,
    path_length_m: float,
    final_ignore_m: float,
    aabb_center: np.ndarray | None = None,
) -> list[CapsuleCollision]:
    """Return clearance records for every capsule swept across one path interval."""

    return swept_capsule_collisions_against_object(
        capsules_start,
        capsules_end,
        half_extents,
        object_center=aabb_center,
        object_shape="box",
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        path_length_m=path_length_m,
        final_ignore_m=final_ignore_m,
    )


def swept_capsule_collisions_against_object(
    capsules_start: Iterable[Capsule],
    capsules_end: Iterable[Capsule],
    half_extents: np.ndarray,
    *,
    alpha_start: float,
    alpha_end: float,
    path_length_m: float,
    final_ignore_m: float,
    object_center: np.ndarray | None = None,
    object_shape: str = "box",
) -> list[CapsuleCollision]:
    """Return clearance records for capsules swept past an object proxy."""

    start_list = list(capsules_start)
    end_list = list(capsules_end)
    if len(start_list) != len(end_list):
        raise ValueError("Start and end capsule lists must have the same length")

    collisions: list[CapsuleCollision] = []
    for capsule_start, capsule_end in zip(start_list, end_list):
        if capsule_start.name != capsule_end.name or capsule_start.group != capsule_end.group:
            raise ValueError("Start and end capsule lists must have matching order")
        distance, t_segment, t_path = swept_segment_object_distance(
            capsule_start.p0,
            capsule_start.p1,
            capsule_end.p0,
            capsule_end.p1,
            half_extents,
            object_center=object_center,
            object_shape=object_shape,
        )
        path_alpha = alpha_start + t_path * (alpha_end - alpha_start)
        near_final = path_length_m * (1.0 - path_alpha) <= final_ignore_m
        clearance = distance - capsule_start.radius
        collisions.append(
            CapsuleCollision(
                capsule_name=capsule_start.name,
                capsule_group=capsule_start.group,
                clearance=float(clearance),
                distance=float(distance),
                radius=float(capsule_start.radius),
                t_segment=float(t_segment),
                t_path=float(t_path),
                path_alpha=float(path_alpha),
                near_final=bool(near_final),
            )
        )
    return collisions


def _group_is_ignorable(
    group: str,
    final_ignore_group_set: set[str] | None,
) -> bool:
    return final_ignore_group_set is None or group in final_ignore_group_set


def _sample_collision_decision(
    collisions: list[CapsuleCollision],
    *,
    final_ignore_group_set: set[str] | None,
) -> tuple[CapsuleCollision, CapsuleCollision, int, int, int]:
    """Choose report collision and counts from all capsule-AABB checks.

    Returns:
      report: clearance record that controls validity reporting for this sample.
      raw_nearest: nearest capsule before any final-contact ignore rule.
      raw_collision_count: all intersecting capsules.
      ignored_collision_count: intersecting capsules masked by the final-contact
        rule.
      active_collision_count: intersecting capsules that still block the path.
    """

    if not collisions:
        raise ValueError("No capsule collision records were provided")

    raw_nearest = min(collisions, key=lambda collision: collision.clearance)
    raw_intersections = [collision for collision in collisions if collision.intersects]
    ignored_intersections = [
        collision
        for collision in raw_intersections
        if collision.near_final
        and _group_is_ignorable(collision.capsule_group, final_ignore_group_set)
    ]
    active_intersections = [
        collision
        for collision in raw_intersections
        if not (
            collision.near_final
            and _group_is_ignorable(collision.capsule_group, final_ignore_group_set)
        )
    ]

    if active_intersections:
        report = min(active_intersections, key=lambda collision: collision.clearance)
    else:
        unignored = [
            collision
            for collision in collisions
            if not (
                collision.intersects
                and collision.near_final
                and _group_is_ignorable(collision.capsule_group, final_ignore_group_set)
            )
        ]
        report = min(unignored or collisions, key=lambda collision: collision.clearance)

    return (
        report,
        raw_nearest,
        len(raw_intersections),
        len(ignored_intersections),
        len(active_intersections),
    )


def sample_linear_path_collisions(
    capsules_base: Iterable[Capsule],
    *,
    start: np.ndarray,
    final: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    path_samples: int,
    final_ignore_m: float,
    aabb_center: np.ndarray | None = None,
    object_shape: str = "box",
    final_ignore_groups: Iterable[str] | None = ("thumb", "index", "middle", "ring", "pinky"),
) -> list[dict[str, object]]:
    """Check capsule-vs-object clearance along a straight hand-base path.

    The returned rows are organized by visible path interval.  Row zero checks
    the static start pose; later rows report the continuous swept-capsule
    minimum over the interval ending at that sample.
    """

    if path_samples < 2:
        raise ValueError("path_samples must be >= 2")

    capsules_base = list(capsules_base)
    aabb_center = np.zeros(3, dtype=float) if aabb_center is None else np.asarray(aabb_center, dtype=float)
    final_ignore_group_set = None if final_ignore_groups is None else set(final_ignore_groups)
    path_delta = final - start
    path_length_m = float(np.linalg.norm(path_delta))
    alphas = np.linspace(0.0, 1.0, path_samples)
    if path_length_m > 1e-12 and final_ignore_m > 0.0:
        ignore_start_alpha = 1.0 - final_ignore_m / path_length_m
        if 0.0 < ignore_start_alpha < 1.0:
            alphas = np.unique(np.concatenate([alphas, np.array([ignore_start_alpha])]))
    rows: list[dict[str, object]] = []
    previous_alpha = float(alphas[0])
    previous_position = start + previous_alpha * path_delta
    previous_capsules = transform_capsules(capsules_base, rotation, previous_position)

    for sample_idx, alpha_raw in enumerate(alphas):
        alpha = float(alpha_raw)
        interval_start_alpha = previous_alpha if sample_idx > 0 else alpha
        interval_end_alpha = alpha
        base_position = start + alpha * path_delta
        near_final = bool(path_length_m * (1.0 - alpha) <= final_ignore_m)
        capsules_world = transform_capsules(capsules_base, rotation, base_position)

        if sample_idx == 0:
            collisions = capsule_collisions_against_object(
                capsules_world,
                half_extents,
                object_center=aabb_center,
                object_shape=object_shape,
                path_alpha=alpha,
                near_final=near_final,
            )
            check_type = "static_start"
        else:
            collisions = swept_capsule_collisions_against_object(
                previous_capsules,
                capsules_world,
                half_extents,
                object_center=aabb_center,
                object_shape=object_shape,
                alpha_start=interval_start_alpha,
                alpha_end=interval_end_alpha,
                path_length_m=path_length_m,
                final_ignore_m=final_ignore_m,
            )
            check_type = "swept_interval"

        (
            report,
            raw_nearest,
            raw_collision_count,
            ignored_collision_count,
            active_collision_count,
        ) = _sample_collision_decision(
            collisions,
            final_ignore_group_set=final_ignore_group_set,
        )
        report_alpha = float(report.path_alpha)
        report_position = start + report_alpha * path_delta
        ignored = ignored_collision_count > 0 and active_collision_count == 0
        rows.append(
            {
                "sample_idx": sample_idx,
                "path_check": check_type,
                "interval_alpha_start": float(interval_start_alpha),
                "interval_alpha_end": float(interval_end_alpha),
                "alpha": report_alpha,
                "base_position": report_position.copy(),
                "near_final_contact_region": report.near_final,
                "ignored_for_final_contact": ignored,
                "nearest_capsule": report.capsule_name,
                "nearest_group": report.capsule_group,
                "clearance_m": report.clearance,
                "distance_m": report.distance,
                "radius_m": report.radius,
                "nearest_t_segment": report.t_segment,
                "nearest_t_path": report.t_path,
                "raw_nearest_capsule": raw_nearest.capsule_name,
                "raw_nearest_group": raw_nearest.capsule_group,
                "raw_clearance_m": raw_nearest.clearance,
                "raw_nearest_alpha": raw_nearest.path_alpha,
                "raw_nearest_t_segment": raw_nearest.t_segment,
                "raw_nearest_t_path": raw_nearest.t_path,
                "raw_collision_count": raw_collision_count,
                "ignored_collision_count": ignored_collision_count,
                "active_collision_count": active_collision_count,
                "collision": active_collision_count > 0,
            }
        )
        previous_alpha = alpha
        previous_capsules = capsules_world
    return rows
