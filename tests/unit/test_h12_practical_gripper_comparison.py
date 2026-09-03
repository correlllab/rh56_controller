import numpy as np
import pytest

from tools.run_h12_practical_gripper_comparison import (
    convex_hull_xy,
    effective_target_width_mm,
    magpie_wrist_targets,
    signed_support_margin_m,
)


def test_effective_target_width_stops_at_cutoff():
    assert effective_target_width_mm(20.0, 40.0) == pytest.approx(40.0)
    assert effective_target_width_mm(60.0, 40.0) == pytest.approx(60.0)


def test_convex_hull_and_signed_support_margin():
    points = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [0.0, 0.0],
    ])
    hull = convex_hull_xy(points)

    assert len(hull) == 4
    assert signed_support_margin_m(np.array([0.0, 0.0]), hull) == pytest.approx(1.0)
    assert signed_support_margin_m(np.array([1.2, 0.0]), hull) == pytest.approx(-0.2)


def test_magpie_targets_hold_local_anchor_and_frame_stationary():
    angle = np.deg2rad(25.0)
    local_frames = np.array([
        np.eye(3),
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
    ])
    anchors = np.array([[0.1, 0.0, 0.0], [0.08, 0.02, 0.0]])
    desired = np.eye(3)
    target_point = np.array([0.25, -0.20, 0.15])

    targets = magpie_wrist_targets(
        anchors,
        local_frames,
        desired_world_frame=desired,
        target_pelvis=target_point,
    )

    for target, anchor, local_frame in zip(targets, anchors, local_frames):
        np.testing.assert_allclose(
            target[:3, 3] + target[:3, :3] @ anchor,
            target_point,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            target[:3, :3] @ local_frame,
            desired,
            atol=1e-12,
        )
