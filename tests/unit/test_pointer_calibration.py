from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rh56_controller.camera_calibration import make_transform
from rh56_controller.pointer_calibration import (
    pointer_point_in_base,
    solve_pointer_calibration,
)
from tools.calibrate_ur_pointer import parse_args


def _synthetic_reference_poses() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    pointer_in_tcp = np.array([0.025, -0.018, 0.142])
    reference_in_base = np.array([-0.43, 0.17, 0.09])
    rotations = [
        Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        for angles in (
            (0.0, 0.0, 0.0),
            (25.0, 0.0, 0.0),
            (-20.0, 18.0, 0.0),
            (10.0, -25.0, 20.0),
            (-18.0, -12.0, -22.0),
            (22.0, 16.0, 28.0),
        )
    ]
    poses = [
        make_transform(rotation, reference_in_base - rotation @ pointer_in_tcp)
        for rotation in rotations
    ]
    return poses, pointer_in_tcp, reference_in_base


def test_pointer_calibration_recovers_tip_and_reference() -> None:
    poses, expected_pointer, expected_reference = _synthetic_reference_poses()

    result = solve_pointer_calibration(poses)

    np.testing.assert_allclose(result.pointer_in_reported_tcp_m, expected_pointer, atol=1e-12)
    np.testing.assert_allclose(result.reference_point_base_m, expected_reference, atol=1e-12)
    assert result.residual_max_mm < 1e-9
    assert result.leave_one_out_max_mm < 1e-9
    assert result.maximum_rotation_span_deg > 30.0


def test_pointer_point_is_transformed_into_base() -> None:
    transform = make_transform(
        Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        [0.5, -0.2, 0.1],
    )

    point = pointer_point_in_base(transform, [0.1, 0.0, 0.2])

    np.testing.assert_allclose(point, [0.5, -0.1, 0.3], atol=1e-12)


def test_pointer_calibration_rejects_unvaried_orientation() -> None:
    poses = [make_transform(np.eye(3), [0.1 * index, 0.0, 0.0]) for index in range(5)]

    with pytest.raises(ValueError, match="geometrically degenerate"):
        solve_pointer_calibration(poses)


def test_pointer_cli_requires_one_action_and_has_read_only_defaults() -> None:
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args(["--capture-reference", "pose_01"])
    assert args.ur_ip == "192.168.0.4"
    assert args.capture_reference == "pose_01"
    assert args.out == Path("artifacts/ur_pointer_calibration/calibration_pointer").resolve()


def test_pointer_tool_has_no_rtde_control_import() -> None:
    source = (Path(__file__).parents[2] / "tools/calibrate_ur_pointer.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "rtde_control" not in imported_modules
