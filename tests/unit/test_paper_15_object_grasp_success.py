from __future__ import annotations

import numpy as np

from tools.run_paper_15_object_grasp_success import build_grid_points, parse_args


def test_paper_facing_defaults_use_shared_force_and_full_lift_threshold():
    args = parse_args([])

    assert args.force_target_n == 6.0
    assert args.contact_compression == 0.10
    assert args.lift_mm == 200.0
    assert args.lift_speed_m_s == 0.10
    assert args.success_lift_mm == 180.0


def test_default_grid_matches_existing_collision_test_geometry():
    points = build_grid_points(parse_args([]))

    assert len(points) == 10
    assert [point.point_id for point in points[:2]] == ["P1", "P2"]
    np.testing.assert_allclose(points[0].offset_m, [0.0, -0.25, 0.0])
    np.testing.assert_allclose(points[1].offset_m, [0.0, 0.0, 0.25])
    np.testing.assert_allclose(points[2].offset_m, [-0.15, -0.25, 0.10])
    np.testing.assert_allclose(points[-1].offset_m, [0.15, -0.25, 0.25])


def test_point_filter_preserves_paper_order():
    args = parse_args(["--points", "P2", "L1_d+50"])
    points = build_grid_points(args)

    assert [point.point_id for point in points] == ["P2", "L1_d+50"]
