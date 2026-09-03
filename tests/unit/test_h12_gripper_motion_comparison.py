import numpy as np
import pytest

from tools.run_h12_gripper_motion_comparison import (
    align_axis_near_reference,
    contact_frame_from_axis,
    rotation_distance_deg,
)


def test_contact_frame_is_orthonormal_and_preserves_axis():
    axis = np.array([0.4, -0.2, 0.9])
    original = axis.copy()
    frame = contact_frame_from_axis(axis)

    np.testing.assert_array_equal(axis, original)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame[:, 0], axis / np.linalg.norm(axis))
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-12)


def test_align_axis_near_reference_maps_local_axis_to_target():
    reference = np.eye(3)
    local_axis = np.array([1.0, 0.0, 0.0])
    desired = np.array([0.0, 0.0, 1.0])

    result = align_axis_near_reference(reference, local_axis, desired)

    np.testing.assert_allclose(result @ local_axis, desired, atol=1e-12)
    assert rotation_distance_deg(reference, result) == pytest.approx(90.0, abs=1e-12)
