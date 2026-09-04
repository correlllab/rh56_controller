from __future__ import annotations

import numpy as np
import pytest

from tools.analyze_ur5_camera_desk_plane import (
    deproject_aligned_depth,
    fit_plane_robust,
    parse_args,
    transform_points,
)


def test_deproject_constant_aligned_depth() -> None:
    pytest.importorskip("cv2")
    depth = np.full((4, 5), 600, dtype=np.uint16)
    camera_matrix = np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]
    )

    points, pixels = deproject_aligned_depth(
        depth,
        camera_matrix,
        np.zeros(5),
        0.001,
        (1, 1, 4, 3),
        1,
        (0.5, 0.7),
    )

    assert len(points) == 6
    np.testing.assert_allclose(points[:, 2], 0.6)
    np.testing.assert_allclose(pixels[0], [1.0, 1.0])


def test_robust_plane_rejects_outlier_and_reports_base_height() -> None:
    xx, yy = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.1, 0.1, 15))
    zz = 0.04 + 0.002 * xx - 0.004 * yy
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    points = np.vstack((points, [0.0, 0.0, 0.2]))

    fit = fit_plane_robust(points, [20.0, 5.0, 2.0])

    assert np.count_nonzero(fit.inlier_mask) == 300
    assert fit.residual_rms_mm < 1e-8
    assert fit.z_at_base_origin_m == pytest.approx(0.04, abs=1e-12)
    assert fit.tilt_from_base_z_deg < 0.3


def test_transform_points_and_cli_validation() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(
        transform_points(transform, np.array([[0.1, 0.2, 0.3]])),
        [[1.1, 2.2, 3.3]],
    )

    with pytest.raises(ValueError, match="depth range"):
        parse_args(["--min-depth-m", "1", "--max-depth-m", "0.5"])

    args = parse_args(["--measured-desk-z-mm", "34", "--max-height-error-mm", "5"])
    assert args.measured_desk_z_mm == 34.0
    assert args.max_height_error_mm == 5.0

    with pytest.raises(ValueError, match="quality thresholds"):
        parse_args(["--max-height-error-mm", "0"])
