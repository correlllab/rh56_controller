#!/usr/bin/env python3
"""GraspGen-X as a fourth closure strategy in the admissibility model.

GGX proposes a 6-DoF hand pose rather than an aperture, and its execution is the
same shape as the hand-designed wrist-fixed strategies: approach the proposed
pose with the hand open, then interpolate the descriptor's closure with the wrist
held still.  That makes it a drop-in ``StrategyPlan``, checked by exactly the same
criteria as naive, iterative and reflex closure.

Two things differ from the analytical strategies and are handled here.

The pose is GGX's own, not the analytical grasp target, so ``terminal_misalignment``
tests whether the contact midpoint GGX actually reaches lies inside the object.
The reported ``terminal_offset`` against the analytical target stays as a
diagnostic; it is not a failure criterion for this planner.

GGX returns a ranked candidate list rather than one grasp.  Following the
manuscript protocol, candidates are tested in descending confidence order and the
start pose counts as admissible if any candidate in the tested set executes
cleanly; a start pose with no collision-free candidate is a planning failure.

Upstream code used here:
  rh56_controller/graspgenx_baseline.py  Isaac-grasp loading + GGX->MuJoCo frame
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rh56_controller.graspgenx_baseline import (  # noqa: E402
    GraspGenXCandidate,
    graspgenx_to_mujoco_base_pose,
    load_isaac_grasp_yaml,
)
from rh56_controller.grasp_geometry import ACTUATOR_NAMES, InspireHandFK  # noqa: E402
from strategy_admissibility import StrategyPlan, Waypoint  # noqa: E402

# Open and closed joint values from the official GraspGen-X Inspire-hand
# descriptor (ext/gripper_descriptions/.../inspire_hand/config.json).  These are
# the configurations GGX itself assumes when it proposes a pose, so the closure
# it commands is the descriptor's, not ours.
GGX_OPEN = {
    "thumb_yaw": 1.308,
    "thumb_proximal": 0.0,
    "index": 0.0,
    "middle": 0.0,
    "ring": 0.0,
    "pinky": 0.0,
}
GGX_CLOSE = {
    "thumb_yaw": 1.308,
    "thumb_proximal": 0.6,
    "index": 1.47,
    "middle": 1.47,
    "ring": 1.47,
    "pinky": 1.47,
}

# The descriptor's contacts: GGX closes all five fingers against the thumb.
GGX_CONTACTS = ("thumb", "index", "middle", "ring", "pinky")

# The descriptor's own capture volume: the box GraspGen-X treats as the region a
# grasped object occupies, "sweep_volume" in the same config.json.  Asking
# whether the proposal engages the object on GGX's terms is fairer than applying
# the antipodal-midpoint test, which assumes a precision pinch and is meaningless
# for the descriptor's five-finger enveloping closure.
#
# Only the extents are taken from the descriptor.  Its offset is expressed in the
# GGX URDF frame, and rather than rely on an axis map for a quantity this
# sensitive, the box is centred on the fingertip centroid of the descriptor's own
# closed configuration, measured directly in the MuJoCo hand frame.
_GGX_SWEEP_HALF_EXTENTS = np.array([0.08, 0.045, 0.04], dtype=float) / 2.0


def _ggx_capture_offset(fk: InspireHandFK, close_ctrl: dict[str, float]) -> np.ndarray:
    tips = fk.all_tips(close_ctrl)
    return np.vstack([tips[name] for name in GGX_CONTACTS if name in tips]).mean(axis=0)


def _clipped(fk: InspireHandFK, values: dict[str, float]) -> dict[str, float]:
    return {
        name: float(np.clip(values.get(name, fk.ctrl_min[name]),
                            fk.ctrl_min[name], fk.ctrl_max[name]))
        for name in ACTUATOR_NAMES
    }


def load_candidates(path: str | Path, top_k: int) -> list[GraspGenXCandidate]:
    """Top-k GGX candidates for one object, in descending confidence order."""

    return load_isaac_grasp_yaml(path)[:top_k]


def build_ggx_plan(
    fk: InspireHandFK,
    candidate: GraspGenXCandidate,
    *,
    object_center: np.ndarray,
    grasp_target: np.ndarray,
) -> StrategyPlan:
    """One GGX candidate as a wrist-fixed approach-then-close plan."""

    base, rotation = graspgenx_to_mujoco_base_pose(candidate, object_center=object_center)
    open_ctrl = _clipped(fk, GGX_OPEN)
    close_ctrl = _clipped(fk, GGX_CLOSE)
    return StrategyPlan(
        name="graspgen_x",
        label=f"GraspGen-X ({candidate.name}, p={candidate.confidence:.3f})",
        waypoints=[
            Waypoint(base, rotation, open_ctrl, "approach"),
            Waypoint(base, rotation, close_ctrl, "closure"),
        ],
        pregrasp_center_base=np.zeros(3),  # unused: the start pose is a wrist pose
        terminal_ctrl=close_ctrl,
        terminal_rotation=rotation,
        terminal_base=base,
        active=GGX_CONTACTS,
        grasp_target=np.asarray(grasp_target, dtype=float),
        terminal_volume=(_GGX_SWEEP_HALF_EXTENTS, _ggx_capture_offset(fk, close_ctrl)),
        notes=(
            "GraspGen-X proposes the wrist pose; the descriptor's open-to-close "
            "configuration is interpolated with the wrist held fixed."
        ),
    )


def build_ggx_plans(
    fk: InspireHandFK,
    candidates_path: str | Path,
    *,
    object_center: np.ndarray,
    grasp_target: np.ndarray,
    top_k: int = 10,
) -> list[StrategyPlan]:
    return [
        build_ggx_plan(fk, candidate, object_center=object_center, grasp_target=grasp_target)
        for candidate in load_candidates(candidates_path, top_k)
    ]
