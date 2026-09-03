import numpy as np
import pytest

from rh56_controller.paper_v2_objects import (
    BUILTIN_OBJECTS,
    ObjectSpec,
    tabletop_grasp_target,
)


def test_tall_paper_object_target_is_ten_mm_below_top():
    bottle = BUILTIN_OBJECTS["paper_bottle"]

    target = tabletop_grasp_target(bottle)

    assert np.isclose(target[2], bottle.size_m[2] - 0.010)
    assert bottle.collision_shape == "cylinder"


def test_explicit_fraction_overrides_object_top_offset():
    bottle = BUILTIN_OBJECTS["paper_bottle"]

    target = tabletop_grasp_target(bottle, z_fraction=0.5)

    assert np.isclose(target[2], bottle.size_m[2] * 0.5)


@pytest.mark.parametrize("object_name", ["paper_metal_cup", "paper_paper_cup"])
def test_open_cup_target_is_at_top_rim(object_name):
    cup = BUILTIN_OBJECTS[object_name]

    target = tabletop_grasp_target(cup)

    assert np.isclose(target[2], cup.size_m[2])


def test_thin_pen_pregrasp_target_is_on_top_surface():
    pen = BUILTIN_OBJECTS["paper_pen"]

    target = tabletop_grasp_target(pen)

    assert np.isclose(target[2], pen.size_m[2])


def test_explicit_top_offset_works_for_center_target_object():
    cube = BUILTIN_OBJECTS["debug_40mm_cube"]

    target = tabletop_grasp_target(cube, top_offset_m=0.010)

    assert np.isclose(target[2], 0.030)


def test_top_offset_and_fraction_cannot_both_be_set():
    obj = ObjectSpec(
        name="test",
        label="test",
        size_m=(0.1, 0.1, 0.1),
        grasp_width_m=0.05,
        mode="line",
        notes="test",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        tabletop_grasp_target(obj, z_fraction=0.5, top_offset_m=0.01)
