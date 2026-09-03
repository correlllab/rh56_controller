"""Reusable camera-calibration math for the UR5 + RH56 experiments.

The simulation tool uses the same OpenCV conventions expected on hardware:

* ``T_base_camera`` maps points from the OpenCV camera frame into the UR5
  base frame.
* OpenCV camera axes are +X right, +Y down, +Z forward.
* Robot poses are ``T_base_gripper`` transforms.
* Calibration-target observations are ``T_camera_target`` transforms.

OpenCV is an optional dependency.  It is imported only by functions that need
image processing or hand-eye calibration so the normal simulation stack stays
usable without the vision extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


_MUJOCO_TO_OPENCV_CAMERA_AXES = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class IntrinsicCalibration:
    """Pinhole camera calibration returned by OpenCV."""

    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_reprojection_error_px: float
    camera_from_target: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class HandEyeCalibration:
    """Fixed-camera eye-to-hand result and its constant board mount."""

    base_from_camera: np.ndarray
    gripper_from_target: np.ndarray
    residual_translation_rms_m: float
    residual_rotation_rms_deg: float


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "OpenCV is required for camera calibration. Install the vision "
            "extra (for example: uv sync --extra sim-ur5-vision)."
        ) from exc
    if not hasattr(cv2, "calibrateHandEye"):
        raise RuntimeError(
            "This OpenCV build does not provide calibrateHandEye. Use "
            "opencv-python-headless>=4.10,<5."
        )
    return cv2


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and XYZ."""

    rotation_array = np.asarray(rotation, dtype=float)
    translation_array = np.asarray(translation, dtype=float)
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if translation_array.shape != (3,):
        raise ValueError("translation must have shape (3,)")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one rigid 4x4 transform."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average rigid transforms using a chordal rotation mean."""

    matrices = np.asarray(transforms, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) == 0:
        raise ValueError("transforms must be a non-empty sequence of 4x4 matrices")
    mean_rotation = Rotation.from_matrix(matrices[:, :3, :3]).mean().as_matrix()
    mean_translation = matrices[:, :3, 3].mean(axis=0)
    return make_transform(mean_rotation, mean_translation)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return geodesic rotation error between two transforms or rotations."""

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    first_rotation = first_array[:3, :3] if first_array.shape == (4, 4) else first_array
    second_rotation = second_array[:3, :3] if second_array.shape == (4, 4) else second_array
    if first_rotation.shape != (3, 3) or second_rotation.shape != (3, 3):
        raise ValueError("inputs must be 3x3 rotations or 4x4 transforms")
    delta = first_rotation.T @ second_rotation
    return float(np.degrees(Rotation.from_matrix(delta).magnitude()))


def interpolate_joint_positions(
    start: Sequence[float],
    target: Sequence[float],
    max_step_rad: float,
) -> np.ndarray:
    """Interpolate a joint-space segment without exceeding ``max_step_rad``.

    The returned samples exclude ``start`` and include ``target``. A zero-
    length segment still returns one sample so callers can collision-check an
    initial pose with the same code path as every later segment.
    """

    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if start_array.ndim != 1 or target_array.shape != start_array.shape:
        raise ValueError("start and target must be matching one-dimensional arrays")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("joint positions must be finite")
    if not np.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError("max_step_rad must be positive and finite")

    maximum_delta = float(np.max(np.abs(target_array - start_array), initial=0.0))
    sample_count = max(1, int(np.ceil(maximum_delta / max_step_rad)))
    fractions = np.arange(1, sample_count + 1, dtype=float) / sample_count
    return start_array[None, :] + fractions[:, None] * (target_array - start_array)[None, :]


def look_at_rotation(
    camera_position: Sequence[float],
    target_position: Sequence[float],
    *,
    up_hint: Sequence[float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Return a MuJoCo camera-to-world rotation looking at ``target_position``.

    MuJoCo cameras look along local -Z with local +Y pointing up in the image.
    """

    camera = np.asarray(camera_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    up = np.asarray(up_hint, dtype=float)
    forward = target - camera
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-12:
        raise ValueError("camera_position and target_position must differ")
    forward /= forward_norm
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-12:
        raise ValueError("up_hint must not be parallel to the viewing direction")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation to MuJoCo's WXYZ quaternion convention."""

    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def pinhole_intrinsics_from_fovy(
    fovy_deg: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Compute MuJoCo's ideal square-pixel camera matrix from vertical FOV."""

    if not 0.0 < fovy_deg < 180.0:
        raise ValueError("fovy_deg must be in (0, 180)")
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    focal_px = 0.5 * image_height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array(
        [
            [focal_px, 0.0, image_width / 2.0],
            [0.0, focal_px, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def mujoco_camera_pose_opencv(
    camera_position_world: Sequence[float],
    camera_rotation_world: np.ndarray,
) -> np.ndarray:
    """Convert a MuJoCo camera pose to ``T_world_camera_opencv``."""

    rotation_world_mujoco = np.asarray(camera_rotation_world, dtype=float)
    if rotation_world_mujoco.shape != (3, 3):
        raise ValueError("camera_rotation_world must have shape (3, 3)")
    rotation_world_opencv = rotation_world_mujoco @ _MUJOCO_TO_OPENCV_CAMERA_AXES
    return make_transform(rotation_world_opencv, camera_position_world)


def chessboard_object_points(
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> np.ndarray:
    """Return OpenCV chessboard inner-corner coordinates in metres."""

    columns, rows = pattern_size
    if columns < 2 or rows < 2:
        raise ValueError("pattern_size must contain at least 2x2 inner corners")
    if square_size_m <= 0.0:
        raise ValueError("square_size_m must be positive")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
        * float(square_size_m)
    )
    return points


def detect_chessboard(
    image_rgb: np.ndarray,
    pattern_size: tuple[int, int],
) -> np.ndarray | None:
    """Detect ordered chessboard inner corners in an RGB image."""

    cv2 = _require_cv2()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if not found or corners is None:
        return None
    return np.asarray(corners, dtype=np.float32)


def calibrate_intrinsics(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    estimate_distortion: bool = False,
) -> IntrinsicCalibration:
    """Calibrate a pinhole camera and recover every target pose.

    MuJoCo's renderer has no lens distortion, so the simulation tool fixes all
    distortion coefficients to zero by default.  Hardware acquisition should
    set ``estimate_distortion=True``.
    """

    cv2 = _require_cv2()
    if len(object_points) != len(image_points) or len(object_points) < 3:
        raise ValueError("at least three matched calibration views are required")
    width, height = image_size
    flags = 0
    initial_camera_matrix = None
    if not estimate_distortion:
        # MuJoCo renders an ideal centered, square-pixel pinhole camera.  Keep
        # those renderer properties fixed while estimating its focal length
        # from images.  Real cameras should use estimate_distortion=True so
        # principal point, aspect ratio and lens coefficients remain free.
        initial_focal_px = float(max(width, height))
        initial_camera_matrix = np.array(
            [
                [initial_focal_px, 0.0, width / 2.0],
                [0.0, initial_focal_px, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
        )
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [np.asarray(points, dtype=np.float32) for points in object_points],
        [np.asarray(points, dtype=np.float32) for points in image_points],
        (int(width), int(height)),
        initial_camera_matrix,
        None,
        flags=flags,
    )
    camera_from_target: list[np.ndarray] = []
    for rotation_vector, translation_vector in zip(rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(rotation_vector)
        camera_from_target.append(
            make_transform(rotation, np.asarray(translation_vector).reshape(3))
        )
    return IntrinsicCalibration(
        image_width=int(width),
        image_height=int(height),
        camera_matrix=np.asarray(camera_matrix, dtype=float),
        distortion=np.asarray(distortion, dtype=float).reshape(-1),
        rms_reprojection_error_px=float(rms),
        camera_from_target=tuple(camera_from_target),
    )


def estimate_eye_to_hand(
    base_from_gripper: Sequence[np.ndarray],
    camera_from_target: Sequence[np.ndarray],
) -> HandEyeCalibration:
    """Estimate a fixed external camera from a wrist-mounted target.

    For eye-to-hand calibration OpenCV's hand-eye solver is called with the
    inverse robot poses.  Its returned ``camera-to-gripper`` transform then has
    the desired ``camera-to-base`` meaning.  The unknown rigid board mount is
    recovered afterward and used to report equation residuals.
    """

    cv2 = _require_cv2()
    robot_poses = tuple(np.asarray(transform, dtype=float) for transform in base_from_gripper)
    target_poses = tuple(np.asarray(transform, dtype=float) for transform in camera_from_target)
    if len(robot_poses) != len(target_poses) or len(robot_poses) < 3:
        raise ValueError("at least three paired robot/camera poses are required")
    if any(transform.shape != (4, 4) for transform in robot_poses + target_poses):
        raise ValueError("all poses must have shape (4, 4)")

    gripper_from_base = tuple(invert_transform(transform) for transform in robot_poses)
    rotation_camera_to_base, translation_camera_to_base = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in gripper_from_base],
        [transform[:3, 3] for transform in gripper_from_base],
        [transform[:3, :3] for transform in target_poses],
        [transform[:3, 3] for transform in target_poses],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    base_from_camera = make_transform(
        np.asarray(rotation_camera_to_base, dtype=float),
        np.asarray(translation_camera_to_base, dtype=float).reshape(3),
    )

    mount_estimates = [
        invert_transform(base_from_gripper_pose)
        @ base_from_camera
        @ camera_from_target_pose
        for base_from_gripper_pose, camera_from_target_pose in zip(robot_poses, target_poses)
    ]
    gripper_from_target = average_transforms(mount_estimates)

    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for base_from_gripper_pose, camera_from_target_pose in zip(robot_poses, target_poses):
        robot_target = base_from_gripper_pose @ gripper_from_target
        camera_target = base_from_camera @ camera_from_target_pose
        translation_errors.append(
            float(np.linalg.norm(robot_target[:3, 3] - camera_target[:3, 3]))
        )
        rotation_errors.append(rotation_error_deg(robot_target, camera_target))

    return HandEyeCalibration(
        base_from_camera=base_from_camera,
        gripper_from_target=gripper_from_target,
        residual_translation_rms_m=float(np.sqrt(np.mean(np.square(translation_errors)))),
        residual_rotation_rms_deg=float(np.sqrt(np.mean(np.square(rotation_errors)))),
    )


def transform_to_dict(transform: np.ndarray) -> dict[str, object]:
    """Serialize a transform with matrix, XYZ and XYZW quaternion forms."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "matrix": matrix.tolist(),
        "translation_m": matrix[:3, 3].tolist(),
        "quaternion_xyzw": quaternion_xyzw.tolist(),
    }
