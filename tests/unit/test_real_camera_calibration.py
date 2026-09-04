from __future__ import annotations

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from rh56_controller.camera_calibration import (
    chessboard_object_points,
    invert_transform,
    make_transform,
    project_target_points,
)
from tools.calibrate_ur5_external_camera_real import (
    calibrate_dataset,
    depth_validity_metrics,
    parse_args,
    pose_diversity,
    run_print_board,
    split_calibration_indices,
)


def test_default_mode_is_receive_only_rtde_capture() -> None:
    args = parse_args([])

    assert args.pose_source == "rtde"
    assert not args.check_only
    assert not args.calibrate_only
    assert args.captures == 25
    assert args.intrinsics_source == "factory"
    assert args.mount_description.startswith("fixed external D435")
    assert args.laser_power is None


def test_camera_check_never_needs_a_robot_pose_source() -> None:
    args = parse_args(
        [
            "--check-only",
            "--pose-source",
            "manual",
            "--laser-power",
            "360",
        ]
    )

    assert args.check_only
    assert args.laser_power == 360.0


def test_calibrate_only_requires_existing_output_argument() -> None:
    with pytest.raises(ValueError, match="requires --out"):
        parse_args(["--calibrate-only"])


def test_print_board_only_writes_requested_physical_layout(tmp_path) -> None:
    args = parse_args(
        [
            "--print-board-only",
            "--pattern-cols",
            "9",
            "--pattern-rows",
            "6",
            "--square-size-mm",
            "12",
            "--page-width-mm",
            "130",
            "--page-height-mm",
            "140",
        ]
    )

    assert run_print_board(args, tmp_path) == 0

    svg = (tmp_path / "printable_chessboard.svg").read_text(encoding="utf-8")
    assert 'width="130.000000mm" height="140.000000mm"' in svg
    assert 'x="5.000000" y="28.000000" width="12.000000"' in svg
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "120.0,84.0,130.0,140.0,96.0,60.0" in summary


def test_print_board_rejects_page_smaller_than_checker() -> None:
    with pytest.raises(ValueError, match="smaller than the checker field"):
        parse_args(
            [
                "--print-board-only",
                "--square-size-mm",
                "12",
                "--page-width-mm",
                "119",
                "--page-height-mm",
                "140",
            ]
        )


def test_split_reserves_every_fifth_capture() -> None:
    training, holdout = split_calibration_indices(25, 5)

    assert holdout == [4, 9, 14, 19, 24]
    assert len(training) == 20
    assert set(training).isdisjoint(holdout)


def test_pose_diversity_identifies_multiaxis_rotation() -> None:
    poses = [
        make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        make_transform(Rotation.from_euler("x", 15, degrees=True).as_matrix(), [0.1, 0.0, 0.0]),
        make_transform(Rotation.from_euler("y", 18, degrees=True).as_matrix(), [0.0, 0.1, 0.0]),
        make_transform(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [0.0, 0.0, 0.1]),
    ]

    diversity = pose_diversity(poses)

    assert diversity["translation_span_mm"] > 140.0
    assert diversity["max_rotation_span_deg"] >= 20.0
    assert diversity["rotation_axis_rank_over_2deg"] == 3


def test_depth_validity_reports_full_and_central_regions() -> None:
    depth = np.zeros((8, 8), dtype=np.uint16)
    depth[3:5, 3:5] = 600

    metrics = depth_validity_metrics(depth, 0.001)

    assert metrics["full_valid_depth_fraction"] == pytest.approx(4 / 64)
    assert metrics["center_valid_depth_fraction"] == pytest.approx(1.0)
    assert metrics["center_median_depth_m"] == pytest.approx(0.6)


def test_offline_calibration_recovers_synthetic_fixed_camera(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    camera_matrix = np.array(
        [[925.0, 0.0, 640.0], [0.0, 925.0, 360.0], [0.0, 0.0, 1.0]]
    )
    object_points = chessboard_object_points((9, 6), 0.025)
    truth_base_camera = make_transform(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 1.0])
    truth_tcp_target = make_transform(np.eye(3), [-0.1, -0.0625, 0.08])
    blank_path = tmp_path / "capture.png"
    assert cv2.imwrite(str(blank_path), np.zeros((720, 1280, 3), dtype=np.uint8))

    entries = []
    for index in range(15):
        angles = np.radians(
            [
                -18.0 + 3.0 * index,
                -14.0 + 4.0 * (index % 7),
                -20.0 + 5.0 * (index % 9),
            ]
        )
        base_tcp = make_transform(
            Rotation.from_euler("xyz", angles).as_matrix(),
            [-0.10 + 0.015 * (index % 6), -0.08 + 0.02 * (index % 5), 0.42 + 0.01 * (index % 4)],
        )
        camera_target = invert_transform(truth_base_camera) @ base_tcp @ truth_tcp_target
        corners = project_target_points(
            object_points, camera_target, camera_matrix, np.zeros(5)
        )
        entries.append(
            {
                "index": index,
                "rgb_path": blank_path.name,
                "base_from_tcp": base_tcp.tolist(),
                "corners_px": corners.tolist(),
            }
        )

    manifest = {
        "schema": "rh56_ur5_external_camera_capture/v1",
        "camera": {
            "stream": {"width": 1280, "height": 720, "fps": 30},
            "factory_color_intrinsics": {
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": np.zeros(5).tolist(),
                "distortion_model": "distortion.none",
            },
        },
        "chessboard": {"inner_corners": [9, 6], "square_size_m": 0.025},
        "captures": entries,
        "assumptions": ["synthetic unit-test dataset"],
    }
    (tmp_path / "capture_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    args = parse_args(["--calibrate-only", "--out", str(tmp_path)])

    summary = calibrate_dataset(args, tmp_path)

    assert summary["quality_pass"] == 1
    calibration = yaml.safe_load((tmp_path / "camera_calibration.yaml").read_text())
    estimated = np.asarray(calibration["transforms"]["base_from_camera"]["matrix"])
    np.testing.assert_allclose(estimated[:3, 3], truth_base_camera[:3, 3], atol=1e-5)
