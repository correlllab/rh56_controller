from __future__ import annotations

import argparse

import numpy as np
import pytest
import yaml

from rh56_controller.camera_calibration import make_transform
from tools.replay_ur5_camera_calibration import (
    ReplayPose,
    calibration_drift,
    load_replay_poses,
    parse_args,
    trajectory_metrics,
    validate_trajectory,
)


def _write_manifest(path, count: int = 5) -> None:
    captures = []
    for index in range(count):
        captures.append(
            {
                "index": index,
                "tcp_pose_vector": [
                    -0.2 + 0.01 * index,
                    -0.5,
                    0.2,
                    0.0,
                    2.2,
                    -2.2 + 0.02 * index,
                ],
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "rh56_ur5_external_camera_capture/v1",
                "captures": captures,
            }
        ),
        encoding="utf-8",
    )


def test_default_cli_is_dry_run_and_enforces_low_speed(tmp_path) -> None:
    reference = tmp_path / "capture_manifest.yaml"
    _write_manifest(reference)
    args = parse_args(["--reference", str(reference)])

    assert not args.execute_motion
    assert args.speed_m_s == 0.02
    assert args.accel_m_s2 == 0.05
    assert args.expected_tcp_offset == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="speed"):
        parse_args(["--reference", str(reference), "--speed-m-s", "0.06"])


def test_loads_selected_tcp_poses_and_rejects_too_few(tmp_path) -> None:
    reference = tmp_path / "capture_manifest.yaml"
    _write_manifest(reference, count=7)

    _payload, poses = load_replay_poses(reference, [0, 2, 3, 5, 6])

    assert [pose.reference_index for pose in poses] == [0, 2, 3, 5, 6]
    assert poses[1].tcp_pose_vector[0] == pytest.approx(-0.18)
    with pytest.raises(ValueError, match="at least five"):
        load_replay_poses(reference, [0, 1, 2, 3])


def test_trajectory_metrics_and_limits() -> None:
    poses = [
        ReplayPose(index, np.array([-0.2 + 0.02 * index, -0.5, 0.2, 0, 2.2, -2.2]))
        for index in range(5)
    ]
    metrics = trajectory_metrics(poses)
    args = argparse.Namespace(
        min_tcp_z_m=0.1,
        max_tcp_radius_m=0.8,
        max_segment_mm=30.0,
        max_segment_deg=75.0,
    )

    assert metrics["pose_count"] == 5
    assert metrics["maximum_segment_translation_mm"] == pytest.approx(20.0)
    assert validate_trajectory(metrics, args) == []

    args.max_segment_mm = 10.0
    assert validate_trajectory(metrics, args) == ["segment_translation_above_limit"]


def test_calibration_drift_reports_translation_and_rotation() -> None:
    baseline = np.eye(4)
    observed = make_transform(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        [0.003, 0.004, 0.0],
    )

    translation_mm, rotation_deg = calibration_drift(baseline, observed)

    assert translation_mm == pytest.approx(5.0)
    assert rotation_deg == pytest.approx(90.0)
