from argparse import Namespace

import numpy as np

from tools.run_h12_screw_accessibility_benchmark import (
    build_trial_specs,
    classify_outcome,
    combine_motion_targets,
    workcell_pose,
)


def test_build_trial_specs_separates_base_and_perception_scans():
    args = Namespace(
        base_error_mm=100.0,
        base_grid_size=3,
        perception_errors_mm=(0.0, 10.0),
        perception_directions=4,
        cover_edge_gap_sweep_mm=(10.0, 20.0),
    )

    specs = build_trial_specs(args)
    base = [spec for spec in specs if spec.scan == "base_placement"]
    perception = [spec for spec in specs if spec.scan == "perception"]
    cover_gap = [spec for spec in specs if spec.scan == "cover_edge_gap"]

    assert len(base) == 9
    assert len(perception) == 5
    assert len(cover_gap) == 2
    assert all(spec.perception_error_mm == 0.0 for spec in base)
    assert all(spec.base_dx_m == 0.0 and spec.base_dy_m == 0.0 for spec in perception)
    assert [spec.cover_edge_gap_mm for spec in cover_gap] == [10.0, 20.0]


def test_combine_motion_targets_adds_vertical_approach_and_extraction():
    closure = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    closure[:, 0, 3] = [0.1, 0.2, 0.3]

    targets, phases, closure_indices = combine_motion_targets(
        closure,
        approach_frames=2,
        extraction_frames=2,
        approach_distance_m=0.08,
        extraction_distance_m=0.05,
    )

    assert targets.shape == (7, 4, 4)
    assert phases == (
        "approach",
        "approach",
        "closure",
        "closure",
        "closure",
        "extraction",
        "extraction",
    )
    np.testing.assert_array_equal(closure_indices, np.array([2, 3, 4]))
    assert targets[0, 2, 3] == 0.08
    assert targets[-1, 2, 3] == 0.05
    assert targets[-1, 0, 3] == 0.3


def test_failure_classification_uses_safety_first_priority():
    common = dict(
        reachable=True,
        collision_free=True,
        joint_margin_ok=True,
        perception_ok=True,
        closure_sufficient=True,
        balance_ok=True,
    )
    assert classify_outcome(**common) == "feasible"

    perception = {**common, "perception_ok": False}
    assert classify_outcome(**perception) == "perception_miss"

    collision_and_perception = {
        **common,
        "collision_free": False,
        "perception_ok": False,
    }
    assert classify_outcome(**collision_and_perception) == "battery_collision"

    unreachable_and_collision = {
        **common,
        "reachable": False,
        "collision_free": False,
    }
    assert classify_outcome(**unreachable_and_collision) == "unreachable"


def test_workcell_places_low_crowned_cover_beyond_screw_head():
    args = Namespace(
        grasp_x=0.45,
        grasp_y=-0.20,
        grasp_z=0.15,
        screw_center_above_pack_mm=25.0,
        screw_inset_from_edge_mm=100.0,
        pack_depth_m=1.20,
        pack_thickness_mm=80.0,
        screw_head_diameter_mm=20.0,
        cover_edge_gap_mm=15.0,
        cover_slope_run_mm=15.0,
        cover_rise_mm=10.0,
        cover_thickness_mm=2.0,
    )
    spec = build_trial_specs(Namespace(
        base_error_mm=1.0,
        base_grid_size=2,
        perception_errors_mm=(0.0,),
        perception_directions=4,
        cover_edge_gap_sweep_mm=(15.0,),
    ))[-1]
    pose = workcell_pose(args, pelvis_world_m=np.array([0.0, 0.0, 1.03]), spec=spec)

    near_slope_center = pose.cover_slope_centers_world_m[0]
    near_edge_x = near_slope_center[0] - 0.5 * 0.015
    screw_head_edge_x = pose.screw_world_m[0] + 0.010
    assert np.isclose(near_edge_x - screw_head_edge_x, 0.015)
    assert near_slope_center[1] == pose.screw_world_m[1]
    ridge_center_z = pose.cover_slope_centers_world_m[0][2]
    assert np.isclose(ridge_center_z - pose.pack_top_z_m, 0.006)
