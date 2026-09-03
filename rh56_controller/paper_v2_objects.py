"""Shared paper-v2 object proxies for simulation-only analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CollisionShape = Literal["box", "cylinder", "sphere"]


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    label: str
    size_m: tuple[float, float, float]
    grasp_width_m: float
    mode: str
    notes: str
    collision_shape: CollisionShape = "box"
    grasp_target_fraction: tuple[float, float, float] = (0.5, 0.5, 0.5)
    grasp_target_top_offset_m: float | None = None


def tabletop_aabb_center(obj: ObjectSpec) -> np.ndarray:
    """Return the world-space AABB center for a tabletop object proxy."""

    return np.array([0.0, 0.0, obj.size_m[2] / 2.0], dtype=float)


def tabletop_grasp_target(
    obj: ObjectSpec,
    *,
    z_fraction: float | None = None,
    top_offset_m: float | None = None,
) -> np.ndarray:
    """Return the paper-facing grasp target point inside the tabletop AABB.

    Fractions are measured from the AABB minimum corner to maximum corner.
    Tall upright objects can instead specify a distance below their top face,
    which better represents the upper-body grasp used in the experiments.
    Explicit arguments override object metadata; z_fraction and top_offset_m
    are mutually exclusive.
    """

    if z_fraction is not None and top_offset_m is not None:
        raise ValueError("z_fraction and top_offset_m are mutually exclusive")

    fraction = np.asarray(obj.grasp_target_fraction, dtype=float)
    size = np.asarray(obj.size_m, dtype=float)
    if top_offset_m is not None:
        effective_top_offset_m = float(top_offset_m)
    elif z_fraction is None:
        effective_top_offset_m = obj.grasp_target_top_offset_m
    else:
        effective_top_offset_m = None

    if effective_top_offset_m is not None:
        if not 0.0 <= effective_top_offset_m <= obj.size_m[2]:
            raise ValueError(
                "grasp target top offset must lie within the object's height"
            )
        fraction[2] = 1.0 - effective_top_offset_m / obj.size_m[2]
    elif z_fraction is not None:
        fraction[2] = float(z_fraction)

    if np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("grasp target fractions must lie within [0, 1]")
    return tabletop_aabb_center(obj) + (fraction - 0.5) * size


# Coarse YCB-like bounding boxes. These are intentionally metadata, not mesh
# ground truth; replace with measured mesh extents before making a final paper
# table.
BUILTIN_OBJECTS: dict[str, ObjectSpec] = {
    "debug_40mm_cube": ObjectSpec(
        name="debug_40mm_cube",
        label="40 mm cube debug proxy",
        size_m=(0.040, 0.040, 0.040),
        grasp_width_m=0.040,
        mode="line",
        notes=(
            "debug cube used for capsule/path collision checks; not a "
            "paper-facing YCB object"
        ),
    ),
    "debug_20mm_cube": ObjectSpec(
        name="debug_20mm_cube",
        label="20 mm cube debug proxy",
        size_m=(0.020, 0.020, 0.020),
        grasp_width_m=0.020,
        mode="line",
        notes=(
            "small cube used for capsule/path collision debugging; not a "
            "paper-facing YCB object"
        ),
    ),
    "ycb_cracker_box": ObjectSpec(
        name="ycb_cracker_box",
        label="YCB cracker box proxy",
        size_m=(0.158, 0.071, 0.213),
        grasp_width_m=0.071,
        mode="plane5",
        notes="box-like YCB proxy; grasp width uses the short side",
    ),
    "ycb_sugar_box": ObjectSpec(
        name="ycb_sugar_box",
        label="YCB sugar box proxy",
        size_m=(0.089, 0.039, 0.175),
        grasp_width_m=0.039,
        mode="plane4",
        notes="small box-like YCB proxy; grasp width uses the short side",
    ),
    "ycb_potted_meat_can": ObjectSpec(
        name="ycb_potted_meat_can",
        label="YCB potted meat can proxy",
        size_m=(0.101, 0.058, 0.083),
        grasp_width_m=0.058,
        mode="plane4",
        notes="low rectangular can proxy; modeled as a box for linear approach analysis",
    ),
    # Paper object proxies from the grasping-experiments object set. These are
    # first-pass box/cylinder/sphere primitives for simulation-only rate sweeps,
    # not measured meshes. Replace with caliper/mesh measurements before
    # treating the rates as final paper numbers.
    "paper_big_screwdriver": ObjectSpec(
        name="paper_big_screwdriver",
        label="Big Screwdriver",
        size_m=(0.240, 0.035, 0.035),
        grasp_width_m=0.035,
        mode="line",
        notes="estimated proxy; elongated tool grasped by two-finger line mode",
    ),
    "paper_bottle": ObjectSpec(
        name="paper_bottle",
        label="Bottle",
        size_m=(0.075, 0.075, 0.190),
        grasp_width_m=0.075,
        mode="plane5",
        notes="estimated upright cylindrical bottle proxy",
        collision_shape="cylinder",
        grasp_target_top_offset_m=0.010,
    ),
    "paper_can": ObjectSpec(
        name="paper_can",
        label="Can",
        size_m=(0.066, 0.066, 0.120),
        grasp_width_m=0.066,
        mode="plane5",
        notes="estimated upright cylindrical can proxy",
        collision_shape="cylinder",
        grasp_target_top_offset_m=0.010,
    ),
    "paper_charger": ObjectSpec(
        name="paper_charger",
        label="Charger",
        size_m=(0.095, 0.060, 0.030),
        grasp_width_m=0.060,
        mode="plane4",
        notes="estimated low rectangular charger proxy",
    ),
    "paper_metal_cup": ObjectSpec(
        name="paper_metal_cup",
        label="Metal Cup",
        size_m=(0.085, 0.085, 0.095),
        grasp_width_m=0.085,
        mode="plane5",
        notes="estimated upright cylindrical cup proxy grasped at the top rim",
        collision_shape="cylinder",
        grasp_target_top_offset_m=0.0,
    ),
    "paper_mustard": ObjectSpec(
        name="paper_mustard",
        label="Mustard",
        size_m=(0.095, 0.060, 0.191),
        grasp_width_m=0.060,
        mode="plane5",
        notes="estimated YCB-like mustard bottle proxy represented as an AABB",
        grasp_target_top_offset_m=0.010,
    ),
    "paper_orange": ObjectSpec(
        name="paper_orange",
        label="Orange",
        size_m=(0.075, 0.075, 0.075),
        grasp_width_m=0.075,
        mode="plane5",
        notes="estimated spherical fruit proxy",
        collision_shape="sphere",
        grasp_target_fraction=(0.5, 0.5, 0.62),
    ),
    "paper_pen": ObjectSpec(
        name="paper_pen",
        label="Pen",
        size_m=(0.145, 0.012, 0.012),
        grasp_width_m=0.012,
        mode="line",
        notes=(
            "estimated thin pen proxy grasped by two-finger line mode; "
            "pre-grasp target is placed on the top surface for table clearance"
        ),
        grasp_target_top_offset_m=0.0,
    ),
    "paper_small_screwdriver": ObjectSpec(
        name="paper_small_screwdriver",
        label="Small Screwdriver",
        size_m=(0.145, 0.022, 0.022),
        grasp_width_m=0.022,
        mode="line",
        notes="estimated small elongated tool proxy grasped by two-finger line mode",
    ),
    "paper_sugar_box": ObjectSpec(
        name="paper_sugar_box",
        label="Sugar Box",
        size_m=(0.089, 0.039, 0.175),
        grasp_width_m=0.039,
        mode="plane4",
        notes="YCB-like sugar box proxy; same dimensions as ycb_sugar_box",
        grasp_target_top_offset_m=0.010,
    ),
    "paper_egg": ObjectSpec(
        name="paper_egg",
        label="Egg",
        size_m=(0.044, 0.044, 0.058),
        grasp_width_m=0.044,
        mode="line",
        notes="estimated delicate egg proxy represented as an AABB",
    ),
    "paper_nut": ObjectSpec(
        name="paper_nut",
        label="Nut",
        size_m=(0.020, 0.020, 0.020),
        grasp_width_m=0.020,
        mode="line",
        notes="estimated small nut proxy represented as an AABB",
    ),
    "paper_paper_cup": ObjectSpec(
        name="paper_paper_cup",
        label="Paper Cup",
        size_m=(0.078, 0.078, 0.090),
        grasp_width_m=0.078,
        mode="plane5",
        notes="estimated upright cylindrical paper cup proxy grasped at the top rim",
        collision_shape="cylinder",
        grasp_target_top_offset_m=0.0,
    ),
    "paper_raspberry": ObjectSpec(
        name="paper_raspberry",
        label="Raspberry",
        size_m=(0.020, 0.020, 0.020),
        grasp_width_m=0.020,
        mode="line",
        notes="estimated delicate berry proxy represented as an AABB",
    ),
    "paper_strawberry": ObjectSpec(
        name="paper_strawberry",
        label="Strawberry",
        size_m=(0.030, 0.030, 0.035),
        grasp_width_m=0.030,
        mode="line",
        notes="estimated delicate strawberry proxy represented as an AABB",
    ),
}
