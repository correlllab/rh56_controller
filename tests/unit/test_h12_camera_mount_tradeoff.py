import numpy as np

from tools.run_h12_camera_mount_tradeoff import (
    camera_configs,
    look_at_rotation,
    obb_signed_separation,
    parse_args,
    relative_pose,
)


def test_look_at_rotation_points_camera_minus_z_at_target():
    camera = np.array([0.4, -0.3, 1.4])
    target = np.array([0.5, -0.2, 1.1])

    rotation = look_at_rotation(camera, target)
    expected_view = (target - camera) / np.linalg.norm(target - camera)

    np.testing.assert_allclose(-rotation[:, 2], expected_view, atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) > 0.0


def test_relative_pose_reconstructs_child_world_pose():
    parent_position = np.array([0.2, -0.4, 1.0])
    parent_rotation = look_at_rotation(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.2, 0.1, 0.0]),
    )
    child_position = np.array([0.5, -0.1, 1.2])
    child_rotation = np.eye(3)

    local_position, local_rotation = relative_pose(
        parent_position,
        parent_rotation,
        child_position,
        child_rotation,
    )

    np.testing.assert_allclose(
        parent_position + parent_rotation @ local_position,
        child_position,
    )
    np.testing.assert_allclose(
        parent_rotation @ local_rotation,
        child_rotation,
        atol=1e-12,
    )


def test_default_camera_scenarios_separate_head_and_mount_assumptions():
    args = parse_args([])
    configs = {config.key: config for config in camera_configs(args)}

    assert configs["rh56_head"].localization_error_mm == 50.0
    assert configs["rh56_palm_under"].reference_offset_world_m == (0.0, 0.0, 0.055)
    assert configs["rh56_dorsal"].reference_offset_world_m == (0.0, 0.0, 0.020)
    assert configs["rh56_dorsal"].reference_origin == "wrist"
    assert configs["rh56_close_wrist"].reference_offset_world_m == (
        0.0,
        -0.040,
        -0.020,
    )
    assert configs["rh56_close_wrist"].reference_origin == "wrist"
    assert (args.camera_width_mm, args.camera_height_mm, args.camera_depth_mm) == (
        80.0,
        20.0,
        20.0,
    )


def test_obb_signed_separation_distinguishes_gap_and_overlap():
    rotation = np.eye(3)
    half_size = np.array([0.1, 0.1, 0.1])

    gap = obb_signed_separation(
        np.zeros(3),
        rotation,
        half_size,
        np.array([0.25, 0.0, 0.0]),
        rotation,
        half_size,
    )
    overlap = obb_signed_separation(
        np.zeros(3),
        rotation,
        half_size,
        np.array([0.15, 0.0, 0.0]),
        rotation,
        half_size,
    )

    assert np.isclose(gap, 0.05)
    assert np.isclose(overlap, -0.05)
