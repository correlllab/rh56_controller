"""Adapters for evaluating GraspGen-X Inspire-Hand grasps with RH56 MuJoCo.

GraspGen-X stores the pose of the root ``world`` link in its Inspire-Hand
descriptor.  The RH56 MuJoCo model uses a differently oriented ``base`` body.
This module keeps that fixed frame conversion in one tested place so the
paper-facing comparison does not hide coordinate-frame conventions inside a
runner script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# Fixed joint in GraspGen-X's official Inspire-Hand URDF:
#   world -> hand_base_link, xyz="0.065 -0.01 0", rpy="pi/2 3pi/4 0"
_GGX_WORLD_TO_URDF_BASE_TRANSLATION = np.array([0.065, -0.01, 0.0], dtype=float)
_GGX_WORLD_TO_URDF_BASE_ROTATION = Rotation.from_euler(
    "xyz", [np.pi / 2.0, 3.0 * np.pi / 4.0, 0.0]
).as_matrix()

# A point expressed in the official URDF hand-base frame is expressed in the
# MuJoCo hand-base frame by p_mj = M @ p_urdf.  This mapping was derived from
# the matching joint origins and link meshes in the two RH56 descriptions.
_URDF_TO_MUJOCO_AXES = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class GraspGenXCandidate:
    """One grasp pose and confidence loaded from Isaac-grasp YAML."""

    name: str
    confidence: float
    position: np.ndarray
    rotation: np.ndarray

    @property
    def approach_direction(self) -> np.ndarray:
        """World/object-frame closing approach axis used by GraspGen-X."""

        return self.rotation[:, 2].copy()


def load_isaac_grasp_yaml(path: str | Path) -> list[GraspGenXCandidate]:
    """Load and confidence-sort GraspGen-X's Isaac-grasp YAML output."""

    yaml_path = Path(path)
    with yaml_path.open() as stream:
        payload = yaml.safe_load(stream)

    if payload.get("format") != "isaac_grasp":
        raise ValueError(f"Unsupported grasp file format in {yaml_path}")
    entries = payload.get("grasps")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"No grasps found in {yaml_path}")

    candidates: list[GraspGenXCandidate] = []
    for name, entry in entries.items():
        orientation = entry["orientation"]
        xyz = np.asarray(orientation["xyz"], dtype=float)
        if xyz.shape != (3,):
            raise ValueError(f"{name}: orientation.xyz must contain three values")
        quaternion_xyzw = np.r_[xyz, float(orientation["w"])]
        candidates.append(
            GraspGenXCandidate(
                name=str(name),
                confidence=float(entry["confidence"]),
                position=np.asarray(entry["position"], dtype=float),
                rotation=Rotation.from_quat(quaternion_xyzw).as_matrix(),
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


def graspgenx_to_mujoco_base_pose(
    candidate: GraspGenXCandidate,
    *,
    object_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a GraspGen-X pose to the RH56 MuJoCo ``base`` body pose.

    GraspGen-X demo meshes are centered before inference and its saved poses
    are transformed back to the input mesh frame.  ``object_center`` therefore
    translates that mesh frame into the MuJoCo world; it does not alter the
    candidate orientation.
    """

    center = np.zeros(3, dtype=float) if object_center is None else np.asarray(object_center, dtype=float)
    if center.shape != (3,):
        raise ValueError("object_center must have shape (3,)")

    fixed_rotation = _GGX_WORLD_TO_URDF_BASE_ROTATION @ _URDF_TO_MUJOCO_AXES.T
    base_rotation = candidate.rotation @ fixed_rotation
    base_position = (
        center
        + candidate.position
        + candidate.rotation @ _GGX_WORLD_TO_URDF_BASE_TRANSLATION
    )
    return base_position, base_rotation


def pregrasp_base_position(
    target_base_position: np.ndarray,
    approach_direction: np.ndarray,
    standoff_m: float,
) -> np.ndarray:
    """Retract a target base pose opposite the GraspGen-X approach axis."""

    direction = np.asarray(approach_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("approach_direction must be non-zero")
    if standoff_m < 0.0:
        raise ValueError("standoff_m must be non-negative")
    return np.asarray(target_base_position, dtype=float) - (standoff_m / norm) * direction
