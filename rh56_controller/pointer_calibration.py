"""Point-probe calibration from repeated poses at one fixed reference point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rh56_controller.camera_calibration import rotation_error_deg


@dataclass(frozen=True)
class PointerCalibration:
    """A point fixed in a reported TCP frame and fit diagnostics."""

    pointer_in_reported_tcp_m: np.ndarray
    reference_point_base_m: np.ndarray
    residuals_mm: np.ndarray
    leave_one_out_errors_mm: np.ndarray
    linear_system_rank: int
    linear_system_condition: float
    maximum_rotation_span_deg: float

    @property
    def residual_rms_mm(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.residuals_mm))))

    @property
    def residual_max_mm(self) -> float:
        return float(np.max(self.residuals_mm))

    @property
    def leave_one_out_rms_mm(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.leave_one_out_errors_mm))))

    @property
    def leave_one_out_max_mm(self) -> float:
        return float(np.max(self.leave_one_out_errors_mm))


def pointer_point_in_base(
    base_from_reported_tcp: np.ndarray,
    pointer_in_reported_tcp_m: Sequence[float],
) -> np.ndarray:
    """Transform a calibrated pointer-tip point into the UR base frame."""

    transform = np.asarray(base_from_reported_tcp, dtype=float)
    point = np.asarray(pointer_in_reported_tcp_m, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("base_from_reported_tcp must have shape (4, 4)")
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("pointer_in_reported_tcp_m must contain three finite values")
    return transform[:3, :3] @ point + transform[:3, 3]


def _fit_pointer(
    base_from_reported_tcp: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    poses = tuple(np.asarray(pose, dtype=float) for pose in base_from_reported_tcp)
    if len(poses) < 3:
        raise ValueError("at least three poses are required")
    if any(pose.shape != (4, 4) or not np.all(np.isfinite(pose)) for pose in poses):
        raise ValueError("all poses must be finite 4x4 transforms")

    identity = np.eye(3)
    matrix = np.vstack(
        [np.column_stack((pose[:3, :3], -identity)) for pose in poses]
    )
    target = np.concatenate([-pose[:3, 3] for pose in poses])
    solution, _residuals, rank, singular_values = np.linalg.lstsq(
        matrix, target, rcond=None
    )
    if rank < 6:
        raise ValueError(
            "pointer poses are geometrically degenerate; use more varied tool orientations"
        )
    condition = float(singular_values[0] / singular_values[-1])
    return solution[:3], solution[3:], int(rank), condition


def solve_pointer_calibration(
    base_from_reported_tcp: Sequence[np.ndarray],
) -> PointerCalibration:
    """Solve ``R_i p_tip + t_i = p_fixed`` from repeated point contacts.

    The reported TCP does not need to be the pointer tip. Its configuration
    only needs to remain fixed across the capture sequence.
    """

    poses = tuple(np.asarray(pose, dtype=float) for pose in base_from_reported_tcp)
    if len(poses) < 4:
        raise ValueError("at least four poses are required for calibration and validation")
    pointer, reference, rank, condition = _fit_pointer(poses)
    residuals_mm = 1000.0 * np.asarray(
        [np.linalg.norm(pointer_point_in_base(pose, pointer) - reference) for pose in poses]
    )

    leave_one_out_errors: list[float] = []
    for omitted_index, omitted_pose in enumerate(poses):
        training = [pose for index, pose in enumerate(poses) if index != omitted_index]
        try:
            loo_pointer, loo_reference, _loo_rank, _loo_condition = _fit_pointer(training)
        except ValueError:
            continue
        predicted = pointer_point_in_base(omitted_pose, loo_pointer)
        leave_one_out_errors.append(1000.0 * float(np.linalg.norm(predicted - loo_reference)))
    if not leave_one_out_errors:
        raise ValueError(
            "leave-one-out fits were degenerate; collect at least five diverse orientations"
        )

    maximum_rotation_span_deg = max(
        rotation_error_deg(first, second)
        for index, first in enumerate(poses)
        for second in poses[index + 1 :]
    )
    return PointerCalibration(
        pointer_in_reported_tcp_m=pointer,
        reference_point_base_m=reference,
        residuals_mm=residuals_mm,
        leave_one_out_errors_mm=np.asarray(leave_one_out_errors, dtype=float),
        linear_system_rank=rank,
        linear_system_condition=condition,
        maximum_rotation_span_deg=float(maximum_rotation_span_deg),
    )
