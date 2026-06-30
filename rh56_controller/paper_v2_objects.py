"""Shared paper-v2 object proxies for simulation-only analyses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    label: str
    size_m: tuple[float, float, float]
    grasp_width_m: float
    mode: str
    notes: str


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
}
