from __future__ import annotations

import numpy as np
import yaml

from rh56_controller.graspgenx_baseline import (
    GraspGenXCandidate,
    graspgenx_to_mujoco_base_pose,
    load_isaac_grasp_yaml,
    pregrasp_base_position,
)


def test_load_isaac_grasp_yaml_sorts_confidence(tmp_path):
    path = tmp_path / "grasps.yml"
    payload = {
        "format": "isaac_grasp",
        "format_version": 1.0,
        "grasps": {
            "low": {
                "confidence": 0.2,
                "position": [1.0, 2.0, 3.0],
                "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
            },
            "high": {
                "confidence": 0.9,
                "position": [0.0, 0.0, 0.0],
                "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
            },
        },
    }
    path.write_text(yaml.safe_dump(payload))

    candidates = load_isaac_grasp_yaml(path)

    assert [candidate.name for candidate in candidates] == ["high", "low"]
    np.testing.assert_allclose(candidates[0].rotation, np.eye(3), atol=1e-12)


def test_graspgenx_to_mujoco_base_pose_applies_official_fixed_joint():
    candidate = GraspGenXCandidate(
        name="identity",
        confidence=1.0,
        position=np.array([0.1, 0.2, 0.3]),
        rotation=np.eye(3),
    )

    position, rotation = graspgenx_to_mujoco_base_pose(
        candidate,
        object_center=np.array([0.0, 0.0, 0.02]),
    )

    np.testing.assert_allclose(position, [0.165, 0.19, 0.32], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_pregrasp_retracts_opposite_approach_axis():
    result = pregrasp_base_position(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.0, 0.0, -2.0]),
        0.06,
    )

    np.testing.assert_allclose(result, [0.1, 0.2, 0.36])
