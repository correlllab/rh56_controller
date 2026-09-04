from __future__ import annotations

import cv2
import numpy as np
import pytest

from tools.analyze_ur5_camera_colored_target import (
    largest_hsv_component,
    nearest_depth_cluster,
    parse_args,
    plane_z_at_xy_mm,
    pixel_depth_to_base,
)


def test_orange_component_and_nearest_depth_cluster() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = (0, 165, 255)
    mask, centroid, bbox, area = largest_hsv_component(
        image, (5, 140, 140), (35, 255, 255), erosion_px=3
    )
    depth = np.full((80, 100), 590, dtype=np.uint16)
    depth[mask] = 552
    depth[25:35, 35:45] = 590

    surface_m, valid_count, foreground_count, mad_mm = nearest_depth_cluster(
        depth, mask, 0.001, percentile=10.0, band_mm=8.0
    )

    assert bbox == (30, 20, 40, 40)
    assert area == 1600
    assert centroid == pytest.approx([49.5, 39.5])
    assert surface_m == pytest.approx(0.552)
    assert foreground_count < valid_count
    assert mad_mm == pytest.approx(0.0)


def test_roi_excludes_larger_distractor() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[5:75, 2:25] = (255, 100, 0)
    image[25:55, 60:90] = (255, 100, 0)

    _mask, centroid, bbox, area = largest_hsv_component(
        image,
        (85, 80, 50),
        (125, 255, 255),
        erosion_px=3,
        roi=(40, 10, 110, 70),
    )

    assert bbox == (60, 25, 30, 30)
    assert area == 900
    assert centroid == pytest.approx([74.5, 39.5])


def test_pixel_depth_to_base_with_identity_camera() -> None:
    point = pixel_depth_to_base(
        [60.0, 70.0],
        0.5,
        np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
    )

    assert point == pytest.approx([0.05, 0.10, 0.5])


def test_plane_height_uses_independently_anchored_origin() -> None:
    z_mm = plane_z_at_xy_mm([0.01, -0.02, 1.0], 34.0, -200.0, -600.0)

    assert z_mm == pytest.approx(24.0)


def test_cli_requires_complete_expected_xy(tmp_path) -> None:
    required = [
        "--calibration",
        str(tmp_path / "calibration.yaml"),
        "--capture",
        str(tmp_path / "capture"),
        "--desk-z-mm",
        "34",
        "--target-height-mm",
        "40",
    ]
    args = parse_args(required)
    assert args.max_height_error_mm == 5.0

    with pytest.raises(ValueError, match="supplied together"):
        parse_args(required + ["--expected-base-x-mm", "-200"])
