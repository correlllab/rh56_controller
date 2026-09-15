#!/usr/bin/env python3
"""Full-motion geometric admissibility for RH56 closure strategies.

The existing repository tool (``tools/run_strategy_pregrasp_rate.py``) checks
only whether the hand can translate to a strategy-specific pre-grasp pose.  It
holds the finger configuration fixed, so it cannot separate naive closure from
reflex closure: both approach with the non-thumb fingers open and both target
the analytical final pose, so both are admitted almost everywhere.

The discriminating phase is closure itself.  On this hand the finger linkages
are coupled, so commanding a width also changes fingertip position and the hand
pose required to keep the same grasp plane.  This module therefore models each
strategy as an ordered sequence of waypoints in the joint space of

    (hand-base position, hand-base rotation, six actuator commands)

and checks the whole executed motion -- approach *and* closure -- against the
object proxy and the support surface.

Strategy definitions follow Sec. IV of the manuscript:

naive
    "moves the hand from an arbitrary pose and fully open to the grasp point,
    like a parallel jaw gripper. Only then does it command the width-
    parameterized pose in one step."  A parallel-jaw gripper's fingertips
    converge on the aperture midpoint, so the hand base is placed such that the
    *fully open* antipodal midpoint sits on the grasp target.  The wrist then
    holds still while the actuators are driven to the final width command.  The
    coupled fingertip travel during that closure is therefore uncompensated.

iterative_closure
    "initiates at the approach-width parameterized pose and steps down until
    closure about the grasp point at the desired width."  Both the wrist pose
    and the actuator commands follow the analytical width sequence, so the
    grasp plane is maintained at every step.

thumb_reflex
    "positions the hand at the final width-parameterized pose while the other
    active fingers stay fully open," then closes the remaining fingers.  The
    wrist is at the final analytical pose for the whole motion.

Admissibility is decided by three failure modes, all geometric:

table_collision
    Any hand capsule penetrates the support plane beyond a tolerance.  This is
    the safety-violating ground collision described in the manuscript.

obstacle_collision
    Any part of the hand touches a neighbouring object.  Nothing about a
    bystander is an intended contact, so no capsule is exempt.  Only present when
    the scene declares obstacles; on an open tabletop this mode never fires.

premature_object_contact
    Any part of the hand other than the planned contact fingertips penetrates
    the object at any point in the motion.  The distal capsules of the thumb and
    of the opposing fingers named by the analytical solution are exempt, since
    those are the grasp contacts; the palm, the inactive fingers, and the
    proximal phalanges are not.  On hardware such contact displaces the object
    instead of grasping it.

terminal_misalignment
    The antipodal contact midpoint actually reached at the end of the motion
    falls outside the object proxy, so the commanded closure no longer straddles
    the object.

Simulation only.  No hardware and no dynamics: this is a kinematic/geometric
feasibility model, and it is not a prediction of grasp success.

Upstream repository code used here (not duplicated):
  rh56_controller/grasp_geometry.py     analytical closure solver + FK tables
  rh56_controller/capsule_hand_proxy.py capsule proxy + object/floor distances
  rh56_controller/paper_v2_objects.py   the 15 paper object proxies
  tools/demo_strategy_pregrasp_collision.py  pre-grasp helpers reused below
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rh56_controller.capsule_hand_proxy import (  # noqa: E402
    Capsule,
    build_capsule_proxy,
    capsule_collisions_against_object,
    closure_base_rotation,
    point_object_distance,
    set_closure_qpos,
    transform_capsules,
)
from rh56_controller.grasp_geometry import (  # noqa: E402
    ACTUATOR_NAMES,
    ClosureGeometry,
    ClosureResult,
    InspireHandFK,
)
from rh56_controller.paper_v2_objects import ObjectSpec, tabletop_aabb_center  # noqa: E402

STRATEGIES = ("naive", "iterative_closure", "thumb_reflex")

FAILURE_MODES = (
    "table_collision",
    "premature_object_contact",
    "terminal_misalignment",
    "obstacle_collision",
)

# Distal capsule of each finger.  These are the planned contact surfaces, so
# they may reach the object; every other capsule touching the object counts as
# premature contact.
_TIP_CAPSULE_SUFFIX = {
    "thumb": "thumb_3",
    "index": "index_1",
    "middle": "middle_1",
    "ring": "ring_1",
    "pinky": "pinky_1",
}


# --------------------------------------------------------------------------
# analytical helpers
# --------------------------------------------------------------------------


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


def open_ctrl(fk: InspireHandFK) -> dict[str, float]:
    """Mechanically fully open hand (every actuator at its ctrl minimum)."""

    return {name: float(fk.ctrl_min[name]) for name in ACTUATOR_NAMES}


def antipodal_center(tips: dict[str, np.ndarray], active: tuple[str, ...]) -> np.ndarray:
    """Midpoint between the thumb tip and the opposing fingertip centroid.

    This is the point a parallel-jaw abstraction treats as "the grasp point",
    and the point the analytical planner servos onto the object.
    """

    thumb = tips.get("thumb")
    opposing = [tips[name] for name in active if name != "thumb" and name in tips]
    if thumb is None or not opposing:
        raise ValueError("antipodal center needs a thumb tip and >=1 opposing tip")
    return 0.5 * (np.asarray(thumb, dtype=float) + np.vstack(opposing).mean(axis=0))


def grasp_yaw_for_object(obj: ObjectSpec) -> float:
    """Yaw that aligns the hand closure axis with the object's graspable axis.

    At zero yaw the hand closes along world x (thumb at +x, opposing fingers at
    -x).  Each object proxy declares ``grasp_width_m``, which is the horizontal
    extent the fingers must straddle, so the closure axis has to lie along
    whichever horizontal axis carries that extent.  Bodies of revolution have
    equal horizontal extents and take zero yaw.
    """

    size = np.asarray(obj.size_m, dtype=float)
    if abs(size[0] - size[1]) < 1e-9:
        return 0.0
    # Align with whichever horizontal extent the planned grasp width matches.
    return 0.0 if abs(size[0] - obj.grasp_width_m) <= abs(size[1] - obj.grasp_width_m) else math.pi / 2.0


def approach_basis(yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (approach direction, lateral direction) for a closure yaw.

    The hand approaches perpendicular to its closure axis, so the fingers
    straddle the object rather than driving into a contact face.
    """

    closure_axis = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    approach = np.cross(np.array([0.0, 0.0, 1.0]), closure_axis)
    approach /= np.linalg.norm(approach)
    return approach, closure_axis


def active_fingers(result: ClosureResult) -> tuple[str, ...]:
    return tuple(result.tip_positions.keys())


def base_for_center(
    center_base: np.ndarray,
    rotation: np.ndarray,
    world_center: np.ndarray,
) -> np.ndarray:
    """Hand-base position putting ``center_base`` (hand frame) on ``world_center``."""

    return np.asarray(world_center, dtype=float) - rotation @ np.asarray(center_base, dtype=float)


# --------------------------------------------------------------------------
# strategy plans
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Waypoint:
    """One commanded hand state."""

    base_position: np.ndarray
    rotation: np.ndarray
    ctrl: dict[str, float]
    phase: str  # "approach" or "closure"


@dataclass
class StrategyPlan:
    name: str
    label: str
    waypoints: list[Waypoint]
    pregrasp_center_base: np.ndarray
    terminal_ctrl: dict[str, float]
    terminal_rotation: np.ndarray
    terminal_base: np.ndarray
    active: tuple[str, ...]
    grasp_target: np.ndarray
    notes: str = ""
    # Optional planner-declared capture volume, as (half_extents, offset) in the
    # hand-base frame.  When present the terminal test asks whether the object
    # lies inside this volume instead of comparing the achieved antipodal
    # midpoint with an analytical grasp target.  GraspGen-X declares such a
    # volume in its gripper descriptor; the analytical strategies do not.
    terminal_volume: tuple[np.ndarray, np.ndarray] | None = None


def _ctrl_lerp(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    return {name: float(a[name] + t * (b[name] - a[name])) for name in ACTUATOR_NAMES}


def build_strategy_plans(
    fk: InspireHandFK,
    closure: ClosureGeometry,
    *,
    mode: str,
    target_width_m: float,
    grasp_target: np.ndarray,
    yaw_rad: float,
    iterative_step_mm: float = 10.0,
    naive_pose_policy: str = "open-pose",
) -> dict[str, StrategyPlan]:
    """Build the commanded waypoint sequence for each closure strategy.

    ``naive_pose_policy`` selects how faithful-vs-charitable the naive baseline
    is.  ``open-pose`` is the literal parallel-jaw reading: the wrist is left at
    the pose used to straddle the object with the hand open, so the full coupled
    tilt change is uncompensated.  ``final-orientation`` grants naive the
    analytically correct grasp *orientation* and leaves only the fingertip
    translation uncompensated; it is reported as a sensitivity check so the
    baseline cannot be dismissed as a straw man.
    """

    lo, hi = mode_width_range(closure, mode)
    width = float(np.clip(target_width_m, lo, hi))
    final = solve_mode(closure, mode, width)
    active = active_fingers(final)
    final_rot = closure_base_rotation(final, yaw_rad)
    final_center_base = antipodal_center(final.tip_positions, active)
    final_base = base_for_center(final_center_base, final_rot, grasp_target)
    final_ctrl = {name: float(final.ctrl_values.get(name, fk.ctrl_min[name])) for name in ACTUATOR_NAMES}

    plans: dict[str, StrategyPlan] = {}

    # ---------------- naive: parallel-jaw placement, wrist held through closure
    naive_ctrl = open_ctrl(fk)
    naive_ctrl["thumb_yaw"] = final_ctrl["thumb_yaw"]
    naive_tips = fk.all_tips(naive_ctrl)
    naive_center_base = antipodal_center(naive_tips, active)
    if naive_pose_policy == "open-pose":
        # Widest analytical solution stands in for the open-hand wrist pose.
        open_result = solve_mode(closure, mode, hi)
        naive_rot = closure_base_rotation(open_result, yaw_rad)
    elif naive_pose_policy == "final-orientation":
        naive_rot = final_rot
    else:
        raise ValueError(f"Unsupported naive pose policy: {naive_pose_policy}")
    naive_base = base_for_center(naive_center_base, naive_rot, grasp_target)
    plans["naive"] = StrategyPlan(
        name="naive",
        label="Naive closure",
        waypoints=[
            Waypoint(naive_base, naive_rot, naive_ctrl, "approach"),
            Waypoint(naive_base, naive_rot, final_ctrl, "closure"),
        ],
        pregrasp_center_base=naive_center_base,
        terminal_ctrl=final_ctrl,
        terminal_rotation=naive_rot,
        terminal_base=naive_base,
        active=active,
        grasp_target=np.asarray(grasp_target, dtype=float),
        notes=(
            "Fully open hand placed so its antipodal midpoint lies on the grasp "
            "target, then the width command is applied with the wrist fixed "
            f"(pose policy: {naive_pose_policy})."
        ),
    )

    # ---------------- iterative closure: wrist tracks the width sequence
    step_m = max(iterative_step_mm, 1.0) / 1000.0
    n_steps = max(int(math.ceil((hi - width) / step_m)), 1)
    widths = np.linspace(hi, width, n_steps + 1)
    iter_waypoints: list[Waypoint] = []
    first_center_base: np.ndarray | None = None
    for idx, w in enumerate(widths):
        try:
            res = solve_mode(closure, mode, float(w))
        except Exception:
            continue
        rot = closure_base_rotation(res, yaw_rad)
        centre = antipodal_center(res.tip_positions, active_fingers(res))
        base = base_for_center(centre, rot, grasp_target)
        ctrl = {name: float(res.ctrl_values.get(name, fk.ctrl_min[name])) for name in ACTUATOR_NAMES}
        # Thumb opposition is set once at the start, as in the planner.
        ctrl["thumb_yaw"] = final_ctrl["thumb_yaw"]
        if first_center_base is None:
            first_center_base = centre
        iter_waypoints.append(Waypoint(base, rot, ctrl, "approach" if idx == 0 else "closure"))
    if first_center_base is None:  # pragma: no cover - solver always yields the endpoint
        first_center_base = final_center_base
        iter_waypoints = [
            Waypoint(final_base, final_rot, final_ctrl, "approach"),
            Waypoint(final_base, final_rot, final_ctrl, "closure"),
        ]
    plans["iterative_closure"] = StrategyPlan(
        name="iterative_closure",
        label="Iterative closure",
        waypoints=iter_waypoints,
        pregrasp_center_base=first_center_base,
        terminal_ctrl=iter_waypoints[-1].ctrl,
        terminal_rotation=iter_waypoints[-1].rotation,
        terminal_base=iter_waypoints[-1].base_position,
        active=active,
        grasp_target=np.asarray(grasp_target, dtype=float),
        notes=(
            f"Wrist and actuators follow the analytical width sequence in "
            f"{len(iter_waypoints) - 1} steps of about {iterative_step_mm:.0f} mm."
        ),
    )

    # ---------------- thumb reflex: wrist at the final pose for the whole motion
    reflex_ctrl = open_ctrl(fk)
    reflex_ctrl["thumb_yaw"] = final_ctrl["thumb_yaw"]
    reflex_ctrl["thumb_proximal"] = final_ctrl["thumb_proximal"]
    reflex_tips = fk.all_tips(reflex_ctrl)
    reflex_center_base = antipodal_center(reflex_tips, active)
    plans["thumb_reflex"] = StrategyPlan(
        name="thumb_reflex",
        label="Thumb reflex",
        waypoints=[
            Waypoint(final_base, final_rot, reflex_ctrl, "approach"),
            Waypoint(final_base, final_rot, final_ctrl, "closure"),
        ],
        pregrasp_center_base=reflex_center_base,
        terminal_ctrl=final_ctrl,
        terminal_rotation=final_rot,
        terminal_base=final_base,
        active=active,
        grasp_target=np.asarray(grasp_target, dtype=float),
        notes=(
            "Thumb is placed at its final bend and opposition while the other "
            "active fingers stay open; the wrist is already at the final "
            "analytical pose, so only the opposing fingers move during closure."
        ),
    )

    return plans


# --------------------------------------------------------------------------
# execution model
# --------------------------------------------------------------------------


@dataclass
class AdmissibilityResult:
    strategy: str
    admissible: bool
    failures: tuple[str, ...]
    min_floor_clearance_m: float
    min_object_clearance_m: float
    premature_clearance_m: float
    obstacle_clearance_m: float
    obstacle_blocker: str
    terminal_offset_m: float
    terminal_outside_object_m: float
    floor_blocker: str
    premature_blocker: str
    first_failure_phase: str
    samples: int
    trace: list[dict[str, object]] = field(default_factory=list)


# --------------------------------------------------------------------------
# exact collision geometry
#
# The capsule proxy is a deliberate over-approximation of the hand, so the
# support-plane test needs a tolerance to stop it reporting collisions the real
# hand does not have.  Choosing that tolerance by matching hardware success
# rates fits a free parameter to the quantity under test.  The alternative used
# here measures it instead: the MuJoCo model already carries the manufacturer
# collision geometry (boxes and a cylinder for the palm, STL meshes for the
# phalanges), and the lowest point of that geometry is exact.
#
# Only the minimum of a linear function is needed, and the minimum of a linear
# function over a point set equals its minimum over the convex hull, so caching
# the hull vertices per actuator command makes the exact test as cheap as the
# capsule test.
# --------------------------------------------------------------------------

_COLLISION_GEOM_GROUP = 3


_VISUAL_GEOM_GROUP = 2


def collision_geom_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    circle_samples: int = 64,
    group: int = _COLLISION_GEOM_GROUP,
) -> tuple[np.ndarray, list[str]]:
    """Surface points of every hand geom in one visual group, in the world frame.

    Returns the points and the owning body name for each point.  The floor plane
    is excluded.  ``group`` selects the collision geometry (3, the default) or
    the visual meshes (2), which are the undecimated part shapes: MuJoCo treats a
    collision mesh as its convex hull, so the collision geometry is itself an
    over-approximation of the part and the two together bracket the real hand.
    """

    points: list[np.ndarray] = []
    owners: list[str] = []
    for gid in range(model.ngeom):
        if int(model.geom_group[gid]) != group:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
        if body in (None, "world"):
            continue
        gtype = int(model.geom_type[gid])
        pos = data.geom_xpos[gid].copy()
        rot = data.geom_xmat[gid].reshape(3, 3).copy()
        size = model.geom_size[gid]
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[gid])
            adr = int(model.mesh_vertadr[mid])
            num = int(model.mesh_vertnum[mid])
            local = np.asarray(model.mesh_vert, dtype=float).reshape(-1, 3)[adr : adr + num]
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
            local = np.array(
                [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float
            )
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius, half = float(size[0]), float(size[1])
            theta = np.linspace(0.0, 2.0 * math.pi, circle_samples, endpoint=False)
            ring = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1)
            local = np.vstack([ring + np.array([0.0, 0.0, half]), ring - np.array([0.0, 0.0, half])])
        elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = float(size[0])
            theta = np.linspace(0.0, 2.0 * math.pi, circle_samples, endpoint=False)
            phi = np.linspace(-math.pi / 2.0, math.pi / 2.0, circle_samples // 4)
            tt, pp = np.meshgrid(theta, phi, indexing="ij")
            local = radius * np.stack(
                [np.cos(pp) * np.cos(tt), np.cos(pp) * np.sin(tt), np.sin(pp)], axis=-1
            ).reshape(-1, 3)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius, half = float(size[0]), float(size[1])
            theta = np.linspace(0.0, 2.0 * math.pi, circle_samples, endpoint=False)
            phi = np.linspace(-math.pi / 2.0, math.pi / 2.0, circle_samples // 4)
            tt, pp = np.meshgrid(theta, phi, indexing="ij")
            cap = radius * np.stack(
                [np.cos(pp) * np.cos(tt), np.cos(pp) * np.sin(tt), np.sin(pp)], axis=-1
            ).reshape(-1, 3)
            local = np.vstack([cap + np.array([0.0, 0.0, half]), cap - np.array([0.0, 0.0, half])])
        else:
            raise ValueError(f"Unsupported collision geom type for body {body}: {gtype}")
        world = pos + local @ rot.T
        points.append(world)
        owners.extend([body] * len(world))
    if not points:
        raise RuntimeError("no hand collision geometry found in the model")
    return np.vstack(points), owners


def collision_hull(points: np.ndarray, owners: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reduce a point set to its convex-hull vertices, keeping owner labels."""

    try:
        from scipy.spatial import ConvexHull  # local import: only needed here

        idx = np.unique(ConvexHull(points).vertices)
    except Exception:  # degenerate cloud, or scipy unavailable
        return points, owners
    return points[idx], [owners[i] for i in idx]


class HandSampler:
    """Caches capsule sets so repeated ctrl values do not re-run FK."""

    def __init__(self, xml_path: str, radius_scale: float = 1.0):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.radius_scale = float(radius_scale)
        self._cache: dict[tuple[int, ...], list[Capsule]] = {}
        self._hull_cache: dict[tuple[int, ...], tuple[np.ndarray, list[str]]] = {}

    @staticmethod
    def _key(ctrl: dict[str, float]) -> tuple[int, ...]:
        return tuple(int(round(ctrl[name] * 1e6)) for name in ACTUATOR_NAMES)

    def capsules(self, ctrl: dict[str, float]) -> list[Capsule]:
        key = self._key(ctrl)
        cached = self._cache.get(key)
        if cached is None:
            set_closure_qpos(self.model, self.data, ctrl)
            cached = build_capsule_proxy(self.model, self.data, radius_scale=self.radius_scale)
            self._cache[key] = cached
        return cached

    def hull(self, ctrl: dict[str, float]) -> tuple[np.ndarray, list[str]]:
        """Convex-hull vertices of the exact collision geometry, hand-base frame."""

        key = self._key(ctrl)
        cached = self._hull_cache.get(key)
        if cached is None:
            set_closure_qpos(self.model, self.data, ctrl)
            cached = collision_hull(*collision_geom_points(self.model, self.data))
            self._hull_cache[key] = cached
        return cached


def _tip_capsule_names(active: tuple[str, ...]) -> set[str]:
    return {_TIP_CAPSULE_SUFFIX[name] for name in active if name in _TIP_CAPSULE_SUFFIX}


def simulate_execution(
    plan: StrategyPlan,
    sampler: HandSampler,
    *,
    start_base: np.ndarray,
    obj: ObjectSpec,
    floor_z_m: float = 0.0,
    floor_tolerance_m: float = 0.003,
    floor_model: str = "capsule",
    approach_samples: int = 16,
    closure_samples: int = 16,
    contact_tolerance_m: float = 0.002,
    contact_policy: str = "tips-only",
    approach_path: str = "direct",
    obstacles: Sequence[ObjectSpec] | None = None,
    obstacle_centers: Sequence[np.ndarray] | None = None,
    keep_trace: bool = False,
) -> AdmissibilityResult:
    """Check one strategy's full commanded motion for geometric admissibility.

    ``contact_policy`` selects how strictly object contact is judged.  Under
    ``tips-only`` only the distal capsule of each planned contact finger may
    touch the object; under ``contact-fingers`` the whole of those fingers may,
    which is the right reading for enveloping grasps on large round objects.

    ``floor_model`` selects the geometry used for the support-plane test.
    ``capsule`` measures the lowest point of the capsule proxy, which
    over-approximates the hand and therefore needs ``floor_tolerance_m`` to
    absorb the over-approximation.  ``mesh`` measures the lowest point of the
    model's own collision geometry, which is exact and needs no tolerance;
    ``calibrate_capsule_envelope.py`` reports the gap between the two.

    ``start_base`` is the world hand-base (wrist) position the motion starts
    from.  The arm commands the wrist, so every strategy is released from the
    same arm state; each then holds its own pre-grasp orientation and finger
    shape, which is precisely the difference under test.
    """

    half_extents = np.asarray(obj.size_m, dtype=float) / 2.0
    aabb_center = tabletop_aabb_center(obj)
    shape = obj.collision_shape
    obstacle_specs = list(obstacles or ())
    obstacle_positions = [np.asarray(c, dtype=float) for c in (obstacle_centers or ())]
    if len(obstacle_positions) != len(obstacle_specs):
        raise ValueError("obstacles and obstacle_centers must have equal length")
    if floor_model not in ("capsule", "mesh"):
        raise ValueError(f"Unsupported floor model: {floor_model}")
    if contact_policy == "tips-only":
        exempt_names = _tip_capsule_names(plan.active)
        exempt_groups: set[str] = set()
    elif contact_policy == "contact-fingers":
        # Permissive variant: the whole of each planned contact finger may touch
        # the object, as it does in an enveloping grasp.  Only the palm and the
        # fingers that take no part in the grasp must stay clear.
        exempt_names = set()
        exempt_groups = set(plan.active)
    else:
        raise ValueError(f"Unsupported contact policy: {contact_policy}")

    first = plan.waypoints[0]
    start_base = np.asarray(start_base, dtype=float)

    # Stage list: free-space approach, then the commanded closure waypoints.
    #
    # "direct" is a single straight line from the start pose to the pre-grasp,
    # the low-complexity motion-planning assumption under test.  "via-overhead"
    # inserts the waypoint an operator would actually use: travel at the start
    # height until above the pre-grasp, then descend.  The difference between the
    # two is how much of the model's pessimism is the straight line itself.
    if approach_path == "direct":
        approach_legs = [(start_base, first.base_position)]
    elif approach_path == "via-overhead":
        overhead = np.array([first.base_position[0], first.base_position[1],
                             max(float(start_base[2]), float(first.base_position[2]))])
        approach_legs = [(start_base, overhead), (overhead, first.base_position)]
    else:
        raise ValueError(f"Unsupported approach path: {approach_path}")

    stages: list[tuple[str, Waypoint, Waypoint, int]] = [
        (
            "approach",
            Waypoint(leg_start, first.rotation, first.ctrl, "approach"),
            Waypoint(leg_end, first.rotation, first.ctrl, "approach"),
            approach_samples,
        )
        for leg_start, leg_end in approach_legs
    ]
    for prev, nxt in zip(plan.waypoints[:-1], plan.waypoints[1:]):
        stages.append(("closure", prev, nxt, closure_samples))

    min_floor = math.inf
    min_object = math.inf
    min_premature = math.inf
    min_obstacle = math.inf
    floor_blocker = ""
    premature_blocker = ""
    obstacle_blocker = ""
    failures: set[str] = set()
    first_failure_phase = ""
    total_samples = 0
    trace: list[dict[str, object]] = []

    for phase, wp_a, wp_b, n_samples in stages:
        for step in range(n_samples + 1):
            t = step / float(n_samples)
            base = wp_a.base_position + t * (wp_b.base_position - wp_a.base_position)
            ctrl = _ctrl_lerp(wp_a.ctrl, wp_b.ctrl, t)
            # Rotation differs between waypoints only for iterative closure, and
            # then only in the tilt angle, so componentwise interpolation with a
            # re-orthonormalisation is exact enough at this sampling density.
            if np.allclose(wp_a.rotation, wp_b.rotation):
                rot = wp_a.rotation
            else:
                blend = wp_a.rotation + t * (wp_b.rotation - wp_a.rotation)
                u, _, vt = np.linalg.svd(blend)
                rot = u @ vt
            capsules = transform_capsules(sampler.capsules(ctrl), rot, base)
            total_samples += 1

            # --- support surface
            if floor_model == "capsule":
                for capsule in capsules:
                    clearance = min(float(capsule.p0[2]), float(capsule.p1[2])) - capsule.radius - floor_z_m
                    if clearance < min_floor:
                        min_floor = clearance
                        floor_blocker = f"{capsule.group}/{capsule.name}"
            else:
                hull_pts, hull_owners = sampler.hull(ctrl)
                heights = hull_pts @ rot[2] + float(base[2])
                low = int(np.argmin(heights))
                clearance = float(heights[low]) - floor_z_m
                if clearance < min_floor:
                    min_floor = clearance
                    floor_blocker = f"mesh/{hull_owners[low]}"
            if min_floor < -floor_tolerance_m and "table_collision" not in failures:
                failures.add("table_collision")
                first_failure_phase = first_failure_phase or phase

            # --- object
            records = capsule_collisions_against_object(
                capsules,
                half_extents,
                object_center=aabb_center,
                object_shape=shape,
            )
            for record in records:
                if record.clearance < min_object:
                    min_object = record.clearance
                # The distal capsules of the planned contact fingers are the
                # grasp contacts, so they are allowed to reach the surface.
                # Every other part of the hand -- palm, inactive fingers, and
                # the proximal phalanges -- must clear the object throughout.
                if record.capsule_name in exempt_names or record.capsule_group in exempt_groups:
                    continue
                if record.clearance < min_premature:
                    min_premature = record.clearance
                    premature_blocker = f"{record.capsule_group}/{record.capsule_name}"
            if min_premature < -contact_tolerance_m and "premature_object_contact" not in failures:
                failures.add("premature_object_contact")
                first_failure_phase = first_failure_phase or phase

            # --- neighbouring objects: no capsule is ever allowed to touch one
            for spec, center in zip(obstacle_specs, obstacle_positions):
                for record in capsule_collisions_against_object(
                    capsules,
                    np.asarray(spec.size_m, dtype=float) / 2.0,
                    object_center=center,
                    object_shape=spec.collision_shape,
                ):
                    if record.clearance < min_obstacle:
                        min_obstacle = record.clearance
                        obstacle_blocker = f"{record.capsule_group}/{record.capsule_name}"
            if min_obstacle < -contact_tolerance_m and "obstacle_collision" not in failures:
                failures.add("obstacle_collision")
                first_failure_phase = first_failure_phase or phase

            if keep_trace:
                trace.append(
                    {
                        "phase": phase,
                        "t": float(t),
                        "base_position": base.copy(),
                        "min_floor_clearance_m": float(min_floor),
                        "min_object_clearance_m": float(min_object),
                    }
                )

    # --- terminal contact alignment
    terminal_tips = sampler_tips(sampler, plan.terminal_ctrl)
    terminal_center_base = antipodal_center(terminal_tips, plan.active)
    terminal_center_world = plan.terminal_base + plan.terminal_rotation @ terminal_center_base
    grasp_target_world = plan.grasp_target
    terminal_offset = float(np.linalg.norm(terminal_center_world - grasp_target_world))

    if plan.terminal_volume is not None:
        # Planner-declared capture volume: the grasp engages the object when the
        # two overlap.  Sampling the volume is enough at this scale and avoids a
        # box-box test against three different object primitives.
        volume_half, volume_offset = plan.terminal_volume
        axes = [np.linspace(-h, h, 5) for h in volume_half]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        points = plan.terminal_base + (grid + volume_offset) @ plan.terminal_rotation.T
        outside = min(
            float(point_object_distance(
                point, half_extents, object_center=aabb_center, object_shape=shape))
            for point in points
        )
    else:
        outside = float(
            point_object_distance(
                terminal_center_world,
                half_extents,
                object_center=aabb_center,
                object_shape=shape,
            )
        )
    if outside > 0.0:
        failures.add("terminal_misalignment")
        first_failure_phase = first_failure_phase or "closure"

    ordered = tuple(mode for mode in FAILURE_MODES if mode in failures)
    return AdmissibilityResult(
        strategy=plan.name,
        admissible=not ordered,
        failures=ordered,
        min_floor_clearance_m=float(min_floor),
        min_object_clearance_m=float(min_object),
        premature_clearance_m=float(min_premature),
        obstacle_clearance_m=float(min_obstacle),
        obstacle_blocker=obstacle_blocker,
        terminal_offset_m=terminal_offset,
        terminal_outside_object_m=outside,
        floor_blocker=floor_blocker,
        premature_blocker=premature_blocker,
        first_failure_phase=first_failure_phase,
        samples=total_samples,
        trace=trace,
    )


def sampler_tips(sampler: HandSampler, ctrl: dict[str, float]) -> dict[str, np.ndarray]:
    """Fingertip positions in the hand-base frame for a commanded ctrl."""

    set_closure_qpos(sampler.model, sampler.data, ctrl)
    tips: dict[str, np.ndarray] = {}
    for finger, site in (
        ("thumb", "right_thumb_tip"),
        ("index", "right_index_tip"),
        ("middle", "right_middle_tip"),
        ("ring", "right_ring_tip"),
        ("pinky", "right_pinky_tip"),
    ):
        sid = mujoco.mj_name2id(sampler.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid >= 0:
            tips[finger] = sampler.data.site_xpos[sid].copy()
    return tips
