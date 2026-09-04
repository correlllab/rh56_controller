from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rh56_controller.camera_calibration import (
    average_transforms,
    chessboard_object_points,
    evaluate_hand_eye,
    estimate_eye_to_hand,
    interpolate_joint_positions,
    invert_transform,
    look_at_rotation,
    make_transform,
    mujoco_camera_pose_opencv,
    pinhole_intrinsics_from_fovy,
    pose_vector_xyz_rotvec_to_transform,
    project_target_points,
    rotation_error_deg,
    solve_chessboard_pose,
)


def test_transform_inverse_and_average() -> None:
    first = make_transform(
        Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix(),
        [0.4, -0.5, 0.6],
    )
    assert np.allclose(first @ invert_transform(first), np.eye(4), atol=1e-12)
    assert np.allclose(average_transforms([first, first]), first, atol=1e-12)


def test_ur_pose_vector_conversion() -> None:
    pose = [0.4, -0.5, 0.6, 0.1, -0.2, 0.3]
    transform = pose_vector_xyz_rotvec_to_transform(pose)

    np.testing.assert_allclose(transform[:3, 3], pose[:3])
    np.testing.assert_allclose(
        transform[:3, :3],
        Rotation.from_rotvec(pose[3:]).as_matrix(),
    )

    with pytest.raises(ValueError, match="x, y, z"):
        pose_vector_xyz_rotvec_to_transform([0.0] * 5)


def test_interpolate_joint_positions_caps_step_and_includes_target() -> None:
    samples = interpolate_joint_positions([0.0, -0.1], [0.25, 0.1], 0.1)

    assert samples.shape == (3, 2)
    np.testing.assert_allclose(samples[-1], [0.25, 0.1])
    deltas = np.diff(np.vstack(([0.0, -0.1], samples)), axis=0)
    assert np.max(np.abs(deltas)) <= 0.1 + 1e-12

    stationary = interpolate_joint_positions([1.0], [1.0], 0.1)
    np.testing.assert_allclose(stationary, [[1.0]])


def test_chessboard_points_and_pinhole_intrinsics() -> None:
    points = chessboard_object_points((9, 6), 0.025)
    assert points.shape == (54, 3)
    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    assert np.allclose(points[-1], [0.2, 0.125, 0.0])

    camera_matrix = pinhole_intrinsics_from_fovy(60.0, 640, 480)
    assert camera_matrix[0, 0] == pytest.approx(camera_matrix[1, 1])
    assert camera_matrix[0, 2] == pytest.approx(320.0)
    assert camera_matrix[1, 2] == pytest.approx(240.0)


def test_look_at_and_mujoco_opencv_axis_conversion() -> None:
    rotation = look_at_rotation([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    # MuJoCo local -Z points from the camera toward the target.
    assert np.allclose(rotation @ np.array([0.0, 0.0, -1.0]), [0.0, 0.0, -1.0])
    base_from_camera = mujoco_camera_pose_opencv([0.0, 0.0, 1.0], rotation)
    # OpenCV local +Z points forward, also toward the target.
    assert np.allclose(
        base_from_camera[:3, :3] @ np.array([0.0, 0.0, 1.0]),
        [0.0, 0.0, -1.0],
    )


def test_eye_to_hand_recovers_synthetic_camera() -> None:
    pytest.importorskip("cv2")
    random_generator = np.random.default_rng(7)
    truth_base_from_camera = make_transform(
        Rotation.from_euler("xyz", [0.1, 0.3, -0.2]).as_matrix(),
        [0.4, -0.2, 1.1],
    )
    truth_gripper_from_target = make_transform(
        Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix(),
        [0.05, 0.02, 0.1],
    )

    base_from_gripper: list[np.ndarray] = []
    camera_from_target: list[np.ndarray] = []
    for _ in range(20):
        robot_pose = make_transform(
            Rotation.random(random_state=random_generator).as_matrix(),
            random_generator.uniform([-0.5, -0.6, 0.2], [-0.2, -0.2, 0.7]),
        )
        base_from_gripper.append(robot_pose)
        camera_from_target.append(
            invert_transform(truth_base_from_camera)
            @ robot_pose
            @ truth_gripper_from_target
        )

    result = estimate_eye_to_hand(base_from_gripper, camera_from_target)
    assert np.linalg.norm(
        result.base_from_camera[:3, 3] - truth_base_from_camera[:3, 3]
    ) < 1e-9
    assert rotation_error_deg(result.base_from_camera, truth_base_from_camera) < 1e-8
    assert result.residual_translation_rms_m < 1e-9
    assert result.residual_rotation_rms_deg < 1e-8

    residuals = evaluate_hand_eye(
        base_from_gripper,
        camera_from_target,
        result.base_from_camera,
        result.gripper_from_target,
    )
    assert residuals.translation_m.shape == (20,)
    assert residuals.translation_rms_m < 1e-9
    assert residuals.rotation_rms_deg < 1e-8


def test_chessboard_pnp_recovers_projected_pose() -> None:
    pytest.importorskip("cv2")
    object_points = chessboard_object_points((9, 6), 0.025)
    camera_matrix = np.array(
        [[925.0, 0.0, 647.0], [0.0, 925.0, 364.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    truth = make_transform(
        Rotation.from_euler("xyz", [0.15, -0.25, 0.08]).as_matrix(),
        [-0.08, -0.05, 0.72],
    )
    image_points = project_target_points(
        object_points,
        truth,
        camera_matrix,
        distortion,
    )

    estimated = solve_chessboard_pose(
        object_points,
        image_points,
        camera_matrix,
        distortion,
    )

    np.testing.assert_allclose(estimated[:3, 3], truth[:3, 3], atol=1e-6)
    assert rotation_error_deg(estimated, truth) < 1e-4
