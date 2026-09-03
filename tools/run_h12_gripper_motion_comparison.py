#!/usr/bin/env python3
"""Compare Magpie and RH56 closure-induced H12 arm motion in MuJoCo.

This is a simulation-only kinematic pilot.  It intentionally keeps the H12
base, torso, and legs fixed so that the closure geometry can be measured before
the same trajectories are connected to a dynamic standing controller.

The three conditions are:

1. Magpie with a fixed wrist.
2. RH56 with a fixed wrist (ablation; the grasp center is allowed to drift).
3. RH56 with wrist compensation (the arm follows the coupled-joint motion).

The script writes summary.csv, trajectory.csv, assumptions.json, and an
optional side-by-side video.  ``--live`` opens repeating interactive MuJoCo
viewers using the exact same trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.grasp_geometry import (  # noqa: E402
    ClosureGeometry,
    GraspMode,
    InspireHandFK,
)
from rh56_controller.grasp_viz_workers import (  # noqa: E402
    _H12_ARM_JOINTS,
    _H12_EE_FRAME,
    _H12_HOME_Q,
    _H12_R_HAND_TO_WRIST,
    _H12_T_HAND_TO_WRIST,
    _build_h12_pin_model,
)


DEFAULT_MAGPIE_XML = Path(
    "/home/tanxuan/workspace/Humanoid_Simulation/CL_Assets/"
    "mujoco_assets/h1_2_magpie.xml"
)
DEFAULT_RH56_XML = REPO_ROOT / "h1_mujoco/inspire/h1_2_inspire.xml"
DEFAULT_MAGPIE_GRIPPER_XML = REPO_ROOT / "h1_mujoco/magpie/magpie.xml"

RH56_TIP_SITES = ("right_thumb_tip", "right_index_tip")
MAGPIE_TIP_SITES = ("tip_left", "tip_right")
WRIST_BODY = "right_wrist_yaw_link"

CONDITION_LABELS = {
    "magpie_fixed": "Magpie | fixed wrist",
    "magpie_compensated": "Magpie | compensated wrist",
    "rh56_fixed": "RH56 | fixed wrist",
    "rh56_compensated": "RH56 | compensated wrist",
    "rh56_practical_compensated": "RH56 | provisional cutoff",
}

CONDITION_COLORS = {
    "magpie_fixed": (0.10, 0.62, 0.95, 0.95),
    "magpie_compensated": (0.10, 0.62, 0.95, 0.95),
    "rh56_fixed": (0.95, 0.25, 0.12, 0.95),
    "rh56_compensated": (0.10, 0.78, 0.35, 0.95),
    "rh56_practical_compensated": (0.95, 0.42, 0.12, 0.95),
}

PILOT_CONDITIONS = (
    "magpie_fixed",
    "rh56_fixed",
    "rh56_compensated",
)


@dataclass
class IKSolution:
    arm_q: np.ndarray
    position_error_mm: np.ndarray
    orientation_error_deg: np.ndarray


@dataclass
class Condition:
    name: str
    model_path: Path
    qpos: np.ndarray
    command_width_mm: np.ndarray
    actual_width_mm: np.ndarray
    anchor_world: np.ndarray
    wrist_world: np.ndarray
    wrist_rotation: np.ndarray
    contact_axis_world: np.ndarray
    arm_q: np.ndarray
    ik_position_error_mm: np.ndarray
    ik_orientation_error_deg: np.ndarray
    target_world: np.ndarray
    com_world: np.ndarray | None = None
    support_margin_mm: np.ndarray | None = None
    support_polygon_xy: np.ndarray | None = None

    @property
    def anchor_error_mm(self) -> np.ndarray:
        return 1000.0 * np.linalg.norm(
            self.anchor_world - self.target_world[None, :], axis=1
        )

    @property
    def wrist_path_mm(self) -> np.ndarray:
        steps = np.linalg.norm(np.diff(self.wrist_world, axis=0), axis=1)
        return 1000.0 * np.concatenate(([0.0], np.cumsum(steps)))

    @property
    def wrist_rotation_path_deg(self) -> np.ndarray:
        steps = [
            rotation_distance_deg(a, b)
            for a, b in zip(self.wrist_rotation[:-1], self.wrist_rotation[1:])
        ]
        return np.concatenate(([0.0], np.cumsum(steps)))

    @property
    def arm_joint_travel_rad(self) -> np.ndarray:
        steps = np.abs(np.diff(self.arm_q, axis=0)).sum(axis=1)
        return np.concatenate(([0.0], np.cumsum(steps)))

    @property
    def contact_axis_change_deg(self) -> np.ndarray:
        initial = self.contact_axis_world[0]
        cosine = np.clip(self.contact_axis_world @ initial, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the H12 Magpie-versus-RH56 closure-motion pilot and render "
            "a side-by-side MuJoCo comparison."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/h12_gripper_motion_comparison"),
    )
    parser.add_argument("--rh56-xml", type=Path, default=DEFAULT_RH56_XML)
    parser.add_argument("--magpie-xml", type=Path, default=DEFAULT_MAGPIE_XML)
    parser.add_argument(
        "--magpie-gripper-xml",
        type=Path,
        default=DEFAULT_MAGPIE_GRIPPER_XML,
    )
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument("--target-width-mm", type=float, default=20.0)
    parser.add_argument("--start-width-mm", type=float, default=None)
    parser.add_argument("--grasp-x", type=float, default=0.25)
    parser.add_argument("--grasp-y", type=float, default=-0.20)
    parser.add_argument("--grasp-z", type=float, default=0.15)
    parser.add_argument("--plane-rz-deg", type=float, default=-120.0)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ik-initial-iters", type=int, default=800)
    parser.add_argument("--ik-step-iters", type=int, default=35)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--panel-height", type=int, default=420)
    parser.add_argument("--camera-distance", type=float, default=0.85)
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=180.0,
        help="MuJoCo camera azimuth in degrees; 180 faces the front of H12.",
    )
    parser.add_argument("--camera-elevation", type=float, default=-10.0)
    parser.add_argument(
        "--video-format",
        choices=("auto", "mp4", "gif"),
        default="auto",
        help="auto prefers MP4 and falls back to GIF when imageio-ffmpeg is absent.",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--live-condition",
        choices=("all", *PILOT_CONDITIONS),
        default="all",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play a live trajectory once instead of repeating until the viewer closes.",
    )
    return parser.parse_args(argv)


def joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in {model.names!r}: {name}")
    return int(model.jnt_qposadr[joint_id])


def named_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    result = mujoco.mj_name2id(model, obj_type, name)
    if result < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return result


def rotation_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def align_axis_near_reference(
    reference_rotation: np.ndarray,
    local_axis: np.ndarray,
    desired_world_axis: np.ndarray,
) -> np.ndarray:
    """Align one local axis using the smallest change from a reference rotation."""
    current = reference_rotation @ local_axis
    current /= np.linalg.norm(current)
    desired = np.asarray(desired_world_axis, dtype=float)
    desired /= np.linalg.norm(desired)
    cross = np.cross(current, desired)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(current @ desired, -1.0, 1.0))
    if sine < 1e-9:
        if cosine > 0.0:
            return reference_rotation.copy()
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(current @ trial)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(current, trial)
        axis /= np.linalg.norm(axis)
        delta = 2.0 * np.outer(axis, axis) - np.eye(3)
        return delta @ reference_rotation
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    delta = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))
    return delta @ reference_rotation


def contact_frame_from_axis(
    axis: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Build a repeatable right-handed frame whose X axis is the contact axis."""
    # Copy so normalization does not mutate a caller-owned contact vector.
    x_axis = np.array(axis, dtype=float, copy=True)
    x_axis /= np.linalg.norm(x_axis)
    reference_axis = (
        np.array([0.0, 1.0, 0.0])
        if reference is None
        else np.asarray(reference, dtype=float)
    )
    y_axis = reference_axis - x_axis * float(reference_axis @ x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        reference_axis = np.array([1.0, 0.0, 0.0])
        y_axis = reference_axis - x_axis * float(reference_axis @ x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def rh56_contact_frame_hand(result) -> np.ndarray:
    axis = result.tip_positions["index"] - result.tip_positions["thumb"]
    return contact_frame_from_axis(axis)


def build_rh56_wrist_targets(
    results,
    *,
    target_pelvis: np.ndarray,
    plane_rz_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    plane_rotation = results[0]._plane_rot(0.0, 0.0, plane_rz_rad)
    initial_hand_rotation = (
        plane_rotation @ results[0]._rot_matrix(results[0].base_tilt_y)
    )
    desired_contact_frame = (
        initial_hand_rotation @ rh56_contact_frame_hand(results[0])
    )
    for result in results:
        # Keep both the pinch center and the complete pinch frame stationary.
        # This is stricter than merely tracking the analytical center and is
        # the relevant constraint for keeping a screw extraction axis fixed.
        hand_rotation = (
            desired_contact_frame @ rh56_contact_frame_hand(result).T
        )
        center_hand = result.grasp_center("antipodal")
        hand_position = target_pelvis - hand_rotation @ center_hand

        wrist = np.eye(4)
        wrist[:3, :3] = hand_rotation @ _H12_R_HAND_TO_WRIST
        wrist[:3, 3] = (
            hand_position + hand_rotation @ _H12_T_HAND_TO_WRIST
        )
        targets.append(wrist)
    return np.asarray(targets), desired_contact_frame


def solve_h12_ik(
    targets: np.ndarray,
    *,
    initial_iters: int,
    step_iters: int,
) -> IKSolution:
    import pinocchio as pin
    import pink
    import qpsolvers

    if initial_iters < 1 or step_iters < 1:
        raise ValueError("IK iteration counts must be positive")

    model = _build_h12_pin_model(pin)
    data = model.createData()
    q0 = pin.neutral(model)
    arm_qidx = []
    for index, joint_name in enumerate(_H12_ARM_JOINTS):
        joint_id = model.getJointId(joint_name)
        q_index = model.joints[joint_id].idx_q
        q0[q_index] = _H12_HOME_Q[index]
        arm_qidx.append(q_index)

    configuration = pink.Configuration(model, data, q0)
    ee_task = pink.tasks.FrameTask(
        _H12_EE_FRAME,
        position_cost=50.0,
        orientation_cost=30.0,
        lm_damping=3.0,
    )
    posture_task = pink.tasks.PostureTask(cost=1e-2)
    posture_task.set_target(q0)
    limits = [
        pink.limits.ConfigurationLimit(model),
        pink.limits.VelocityLimit(model),
    ]
    solver = "daqp" if "daqp" in qpsolvers.available_solvers else None
    if solver is None and qpsolvers.available_solvers:
        solver = qpsolvers.available_solvers[0]
    if solver is None:
        raise RuntimeError("No qpsolvers backend is available")

    arm_qidx_set = set(arm_qidx)
    non_arm_qidx = [index for index in range(model.nq) if index not in arm_qidx_set]
    frame_id = model.getFrameId(_H12_EE_FRAME)
    arm_rows = []
    position_errors = []
    orientation_errors = []

    for target_index, target in enumerate(targets):
        ee_task.set_target(pin.SE3(target))
        iterations = initial_iters if target_index == 0 else step_iters
        for _ in range(iterations):
            velocity = pink.solve_ik(
                configuration,
                [ee_task, posture_task],
                dt=0.05,
                solver=solver,
                limits=limits,
                safety_break=False,
            )
            configuration.integrate_inplace(velocity, 0.05)
            q_locked = np.array(configuration.q, copy=True)
            q_locked[non_arm_qidx] = q0[non_arm_qidx]
            configuration = pink.Configuration(model, data, q_locked)

        actual = configuration.data.oMf[frame_id].homogeneous
        position_errors.append(
            1000.0 * float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        )
        orientation_errors.append(
            rotation_distance_deg(target[:3, :3], actual[:3, :3])
        )
        arm_rows.append(np.asarray(configuration.q)[arm_qidx].copy())

    return IKSolution(
        arm_q=np.asarray(arm_rows),
        position_error_mm=np.asarray(position_errors),
        orientation_error_deg=np.asarray(orientation_errors),
    )


def settle_magpie_lookup(
    gripper_xml: Path,
    *,
    samples: int = 121,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actuator, finger qpos, and tip-gap lookup for quasi-static closure."""
    model = mujoco.MjModel.from_xml_path(str(gripper_xml))
    left_site = named_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_left")
    right_site = named_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_right")
    commands = np.linspace(0.0, 2.4, samples)
    qpos_rows = []
    gaps = []

    for command in commands:
        data = mujoco.MjData(model)
        data.ctrl[:] = (command, -command)
        for _ in range(2500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        qpos_rows.append(data.qpos.copy())
        gaps.append(float(np.linalg.norm(
            data.site_xpos[left_site] - data.site_xpos[right_site]
        )))

    return commands, np.asarray(qpos_rows), np.asarray(gaps)


def interpolate_magpie_qpos(
    requested_width_m: np.ndarray,
    commands: np.ndarray,
    lookup_qpos: np.ndarray,
    lookup_gaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    command_for_width = np.interp(
        requested_width_m,
        lookup_gaps[::-1],
        commands[::-1],
    )
    qpos = np.column_stack([
        np.interp(command_for_width, commands, lookup_qpos[:, column])
        for column in range(lookup_qpos.shape[1])
    ])
    actual_gap = np.interp(command_for_width, commands, lookup_gaps)
    return command_for_width, qpos, actual_gap


def set_arm_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    arm_q: np.ndarray,
) -> None:
    for joint_name, value in zip(_H12_ARM_JOINTS, arm_q):
        qpos[joint_qpos_address(model, joint_name)] = value


def set_rh56_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    ctrl_values: dict[str, float],
    fk: InspireHandFK,
) -> None:
    values = {
        name: float(ctrl_values.get(name, fk.ctrl_min[name]))
        for name in (
            "pinky",
            "ring",
            "middle",
            "index",
            "thumb_proximal",
            "thumb_yaw",
        )
    }
    pairs = {
        "pinky_proximal_joint": values["pinky"],
        "pinky_intermediate_joint": -0.15 + 1.1169 * values["pinky"],
        "ring_proximal_joint": values["ring"],
        "ring_intermediate_joint": -0.15 + 1.1169 * values["ring"],
        "middle_proximal_joint": values["middle"],
        "middle_intermediate_joint": -0.15 + 1.1169 * values["middle"],
        "index_proximal_joint": values["index"],
        "index_intermediate_joint": -0.05 + 1.1169 * values["index"],
        "thumb_proximal_yaw_joint": values["thumb_yaw"],
        "thumb_proximal_pitch_joint": values["thumb_proximal"],
        "thumb_intermediate_joint": 0.15 + 1.33 * values["thumb_proximal"],
        "thumb_distal_joint": 0.15 + 0.66 * values["thumb_proximal"],
    }
    for joint_name, value in pairs.items():
        qpos[joint_qpos_address(model, joint_name)] = value


def set_magpie_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    finger_qpos: np.ndarray,
) -> None:
    names = (
        "left_hinge_1",
        "left_hinge_2",
        "left_hinge_3",
        "right_hinge_1",
        "right_hinge_2",
        "right_hinge_3",
    )
    for joint_name, value in zip(names, finger_qpos):
        qpos[joint_qpos_address(model, joint_name)] = value


def wrist_local_magpie_geometry(
    model_path: Path,
    finger_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchors, frames = wrist_local_magpie_trajectory(
        model_path,
        np.asarray(finger_qpos)[None, :],
    )
    return anchors[0], frames[0]


def wrist_local_magpie_trajectory(
    model_path: Path,
    finger_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Magpie pinch centers and frames in the wrist body frame."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    left_site = named_id(model, mujoco.mjtObj.mjOBJ_SITE, MAGPIE_TIP_SITES[0])
    right_site = named_id(model, mujoco.mjtObj.mjOBJ_SITE, MAGPIE_TIP_SITES[1])
    wrist_body = named_id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
    anchors = []
    local_axes = []
    for row in finger_qpos:
        data.qpos[:] = model.qpos0
        set_arm_qpos(model, data.qpos, _H12_HOME_Q)
        set_magpie_qpos(model, data.qpos, row)
        mujoco.mj_forward(model, data)

        midpoint = 0.5 * (
            data.site_xpos[left_site] + data.site_xpos[right_site]
        )
        wrist_position = data.xpos[wrist_body]
        wrist_rotation = data.xmat[wrist_body].reshape(3, 3)
        anchors.append(wrist_rotation.T @ (midpoint - wrist_position))
        local_axis = wrist_rotation.T @ (
            data.site_xpos[right_site] - data.site_xpos[left_site]
        )
        local_axes.append(local_axis / np.linalg.norm(local_axis))

    # Magpie's pinch axis lies close to the default frame-construction
    # reference. Rebuilding every frame independently can therefore introduce
    # arbitrary roll flips. Parallel-transport the frame with the smallest
    # rotation from the previous sample instead.
    frames = [contact_frame_from_axis(local_axes[0])]
    for local_axis in local_axes[1:]:
        frames.append(align_axis_near_reference(
            frames[-1],
            np.array([1.0, 0.0, 0.0]),
            local_axis,
        ))
    return np.asarray(anchors), np.asarray(frames)


def build_condition(
    *,
    name: str,
    model_path: Path,
    arm_q: np.ndarray,
    command_width_mm: np.ndarray,
    target_world: np.ndarray,
    tip_sites: tuple[str, str],
    ik_position_error_mm: np.ndarray,
    ik_orientation_error_deg: np.ndarray,
    rh56_results=None,
    rh56_fk: InspireHandFK | None = None,
    magpie_qpos: np.ndarray | None = None,
) -> Condition:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    site_ids = tuple(
        named_id(model, mujoco.mjtObj.mjOBJ_SITE, site) for site in tip_sites
    )
    wrist_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)

    qpos_rows = []
    anchor_rows = []
    width_rows = []
    wrist_rows = []
    wrist_rotation_rows = []
    contact_axis_rows = []

    for frame_index, current_arm_q in enumerate(arm_q):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        set_arm_qpos(model, data.qpos, current_arm_q)
        if rh56_results is not None:
            if rh56_fk is None:
                raise ValueError("rh56_fk is required with rh56_results")
            set_rh56_qpos(
                model,
                data.qpos,
                rh56_results[frame_index].ctrl_values,
                rh56_fk,
            )
        elif magpie_qpos is not None:
            set_magpie_qpos(model, data.qpos, magpie_qpos[frame_index])
        else:
            raise ValueError("A hand trajectory is required")

        mujoco.mj_forward(model, data)
        point_a = data.site_xpos[site_ids[0]].copy()
        point_b = data.site_xpos[site_ids[1]].copy()
        qpos_rows.append(data.qpos.copy())
        anchor_rows.append(0.5 * (point_a + point_b))
        width_rows.append(1000.0 * float(np.linalg.norm(point_a - point_b)))
        wrist_rows.append(data.xpos[wrist_id].copy())
        wrist_rotation_rows.append(data.xmat[wrist_id].reshape(3, 3).copy())
        contact_axis = point_b - point_a
        contact_axis_rows.append(contact_axis / np.linalg.norm(contact_axis))

    return Condition(
        name=name,
        model_path=model_path,
        qpos=np.asarray(qpos_rows),
        command_width_mm=np.asarray(command_width_mm),
        actual_width_mm=np.asarray(width_rows),
        anchor_world=np.asarray(anchor_rows),
        wrist_world=np.asarray(wrist_rows),
        wrist_rotation=np.asarray(wrist_rotation_rows),
        contact_axis_world=np.asarray(contact_axis_rows),
        arm_q=np.asarray(arm_q),
        ik_position_error_mm=np.asarray(ik_position_error_mm),
        ik_orientation_error_deg=np.asarray(ik_orientation_error_deg),
        target_world=np.asarray(target_world),
    )


def repeat_solution(solution: IKSolution, frames: int) -> IKSolution:
    return IKSolution(
        arm_q=np.repeat(solution.arm_q[:1], frames, axis=0),
        position_error_mm=np.repeat(solution.position_error_mm[:1], frames),
        orientation_error_deg=np.repeat(solution.orientation_error_deg[:1], frames),
    )


def prepare_conditions(args: argparse.Namespace) -> tuple[list[Condition], dict]:
    for path in (args.rh56_xml, args.magpie_xml, args.magpie_gripper_xml):
        if not path.exists():
            raise FileNotFoundError(path)

    fk = InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    width_range = closure.width_range(str(GraspMode.LINE_2F), n_fingers=2)
    target_width_m = float(np.clip(args.target_width_mm / 1000.0, *width_range))
    start_width_m = (
        width_range[1]
        if args.start_width_mm is None
        else float(np.clip(args.start_width_mm / 1000.0, target_width_m, width_range[1]))
    )
    requested_width_m = np.linspace(start_width_m, target_width_m, args.frames)
    rh56_results = [
        closure.solve(GraspMode.LINE_2F, float(width))
        for width in requested_width_m
    ]
    command_width_mm = 1000.0 * np.asarray(
        [result.width for result in rh56_results]
    )
    target_pelvis = np.array([args.grasp_x, args.grasp_y, args.grasp_z])
    rh56_targets, desired_contact_frame = build_rh56_wrist_targets(
        rh56_results,
        target_pelvis=target_pelvis,
        plane_rz_rad=math.radians(args.plane_rz_deg),
    )

    print("[comparison] solving RH56 compensated wrist path...")
    rh56_comp_solution = solve_h12_ik(
        rh56_targets,
        initial_iters=args.ik_initial_iters,
        step_iters=args.ik_step_iters,
    )
    print("[comparison] solving RH56 fixed-wrist pose...")
    rh56_fixed_solution = repeat_solution(
        solve_h12_ik(
            rh56_targets[:1],
            initial_iters=args.ik_initial_iters,
            step_iters=args.ik_step_iters,
        ),
        args.frames,
    )

    print("[comparison] building quasi-static Magpie closure lookup...")
    commands, lookup_qpos, lookup_gaps = settle_magpie_lookup(
        args.magpie_gripper_xml
    )
    _, magpie_qpos, magpie_width_m = interpolate_magpie_qpos(
        requested_width_m,
        commands,
        lookup_qpos,
        lookup_gaps,
    )
    local_open_anchor, local_magpie_contact_frame = wrist_local_magpie_geometry(
        args.magpie_xml, magpie_qpos[0]
    )
    magpie_target = np.eye(4)
    magpie_target[:3, :3] = align_axis_near_reference(
        rh56_targets[0, :3, :3],
        local_magpie_contact_frame[:, 0],
        desired_contact_frame[:, 0],
    )
    magpie_target[:3, 3] = (
        target_pelvis - magpie_target[:3, :3] @ local_open_anchor
    )
    print("[comparison] solving Magpie fixed-wrist pose...")
    magpie_solution = repeat_solution(
        solve_h12_ik(
            magpie_target[None, :, :],
            initial_iters=args.ik_initial_iters,
            step_iters=args.ik_step_iters,
        ),
        args.frames,
    )

    rh56_model = mujoco.MjModel.from_xml_path(str(args.rh56_xml))
    rh56_data = mujoco.MjData(rh56_model)
    mujoco.mj_forward(rh56_model, rh56_data)
    pelvis_id = named_id(rh56_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    target_world = rh56_data.xpos[pelvis_id].copy() + target_pelvis

    conditions = [
        build_condition(
            name="magpie_fixed",
            model_path=args.magpie_xml,
            arm_q=magpie_solution.arm_q,
            command_width_mm=1000.0 * magpie_width_m,
            target_world=target_world,
            tip_sites=MAGPIE_TIP_SITES,
            ik_position_error_mm=magpie_solution.position_error_mm,
            ik_orientation_error_deg=magpie_solution.orientation_error_deg,
            magpie_qpos=magpie_qpos,
        ),
        build_condition(
            name="rh56_fixed",
            model_path=args.rh56_xml,
            arm_q=rh56_fixed_solution.arm_q,
            command_width_mm=command_width_mm,
            target_world=target_world,
            tip_sites=RH56_TIP_SITES,
            ik_position_error_mm=rh56_fixed_solution.position_error_mm,
            ik_orientation_error_deg=rh56_fixed_solution.orientation_error_deg,
            rh56_results=rh56_results,
            rh56_fk=fk,
        ),
        build_condition(
            name="rh56_compensated",
            model_path=args.rh56_xml,
            arm_q=rh56_comp_solution.arm_q,
            command_width_mm=command_width_mm,
            target_world=target_world,
            tip_sites=RH56_TIP_SITES,
            ik_position_error_mm=rh56_comp_solution.position_error_mm,
            ik_orientation_error_deg=rh56_comp_solution.orientation_error_deg,
            rh56_results=rh56_results,
            rh56_fk=fk,
        ),
    ]

    metadata = {
        "width_range_mm": [1000.0 * width_range[0], 1000.0 * width_range[1]],
        "requested_start_width_mm": 1000.0 * start_width_m,
        "requested_target_width_mm": 1000.0 * target_width_m,
        "target_pelvis_m": target_pelvis.tolist(),
        "target_world_m": target_world.tolist(),
        "plane_rz_deg": args.plane_rz_deg,
        "magpie_lookup_command_range_rad": [
            float(commands[0]),
            float(commands[-1]),
        ],
        "magpie_lookup_gap_range_mm": [
            1000.0 * float(lookup_gaps[-1]),
            1000.0 * float(lookup_gaps[0]),
        ],
    }
    return conditions, metadata


def condition_summary(condition: Condition) -> dict[str, object]:
    anchor_error = condition.anchor_error_mm
    com_shift_mm = (
        1000.0
        * np.linalg.norm(
            condition.com_world[:, :2] - condition.com_world[0, :2],
            axis=1,
        )
        if condition.com_world is not None
        else None
    )
    return {
        "condition": condition.name,
        "label": CONDITION_LABELS[condition.name],
        "start_width_mm": float(condition.actual_width_mm[0]),
        "final_width_mm": float(condition.actual_width_mm[-1]),
        "max_anchor_error_mm": float(anchor_error.max()),
        "final_anchor_error_mm": float(anchor_error[-1]),
        "max_contact_axis_change_deg": float(
            condition.contact_axis_change_deg.max()
        ),
        "final_contact_axis_change_deg": float(
            condition.contact_axis_change_deg[-1]
        ),
        "anchor_path_mm": float(
            1000.0
            * np.linalg.norm(np.diff(condition.anchor_world, axis=0), axis=1).sum()
        ),
        "wrist_path_mm": float(condition.wrist_path_mm[-1]),
        "wrist_rotation_deg": float(condition.wrist_rotation_path_deg[-1]),
        "arm_joint_travel_rad": float(condition.arm_joint_travel_rad[-1]),
        "max_ik_position_error_mm": float(
            condition.ik_position_error_mm.max()
        ),
        "max_ik_orientation_error_deg": float(
            condition.ik_orientation_error_deg.max()
        ),
        "static_balance_evaluated": condition.support_margin_mm is not None,
        "max_horizontal_com_shift_mm": (
            float(com_shift_mm.max()) if com_shift_mm is not None else ""
        ),
        "min_static_support_margin_mm": (
            float(condition.support_margin_mm.min())
            if condition.support_margin_mm is not None
            else ""
        ),
        "final_static_support_margin_change_mm": (
            float(
                condition.support_margin_mm[-1]
                - condition.support_margin_mm[0]
            )
            if condition.support_margin_mm is not None
            else ""
        ),
        "dynamic_balance_evaluated": False,
        "contact_success_evaluated": False,
    }


def write_summary(path: Path, conditions: Iterable[Condition]) -> None:
    rows = [condition_summary(condition) for condition in conditions]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory(
    path: Path,
    conditions: Iterable[Condition],
    *,
    fps: int,
) -> None:
    fields = [
        "condition",
        "frame",
        "time_s",
        "command_width_mm",
        "actual_width_mm",
        "anchor_x_m",
        "anchor_y_m",
        "anchor_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "anchor_error_mm",
        "contact_axis_x",
        "contact_axis_y",
        "contact_axis_z",
        "contact_axis_change_deg",
        "wrist_x_m",
        "wrist_y_m",
        "wrist_z_m",
        "cumulative_wrist_path_mm",
        "cumulative_wrist_rotation_deg",
        "cumulative_arm_joint_travel_rad",
        "ik_position_error_mm",
        "ik_orientation_error_deg",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "static_support_margin_mm",
        *_H12_ARM_JOINTS,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            for frame_index in range(len(condition.qpos)):
                row = {
                    "condition": condition.name,
                    "frame": frame_index,
                    "time_s": frame_index / float(fps),
                    "command_width_mm": condition.command_width_mm[frame_index],
                    "actual_width_mm": condition.actual_width_mm[frame_index],
                    "anchor_x_m": condition.anchor_world[frame_index, 0],
                    "anchor_y_m": condition.anchor_world[frame_index, 1],
                    "anchor_z_m": condition.anchor_world[frame_index, 2],
                    "target_x_m": condition.target_world[0],
                    "target_y_m": condition.target_world[1],
                    "target_z_m": condition.target_world[2],
                    "anchor_error_mm": condition.anchor_error_mm[frame_index],
                    "contact_axis_x": condition.contact_axis_world[frame_index, 0],
                    "contact_axis_y": condition.contact_axis_world[frame_index, 1],
                    "contact_axis_z": condition.contact_axis_world[frame_index, 2],
                    "contact_axis_change_deg": (
                        condition.contact_axis_change_deg[frame_index]
                    ),
                    "wrist_x_m": condition.wrist_world[frame_index, 0],
                    "wrist_y_m": condition.wrist_world[frame_index, 1],
                    "wrist_z_m": condition.wrist_world[frame_index, 2],
                    "cumulative_wrist_path_mm": condition.wrist_path_mm[frame_index],
                    "cumulative_wrist_rotation_deg": (
                        condition.wrist_rotation_path_deg[frame_index]
                    ),
                    "cumulative_arm_joint_travel_rad": (
                        condition.arm_joint_travel_rad[frame_index]
                    ),
                    "ik_position_error_mm": (
                        condition.ik_position_error_mm[frame_index]
                    ),
                    "ik_orientation_error_deg": (
                        condition.ik_orientation_error_deg[frame_index]
                    ),
                    "com_x_m": (
                        condition.com_world[frame_index, 0]
                        if condition.com_world is not None
                        else ""
                    ),
                    "com_y_m": (
                        condition.com_world[frame_index, 1]
                        if condition.com_world is not None
                        else ""
                    ),
                    "com_z_m": (
                        condition.com_world[frame_index, 2]
                        if condition.com_world is not None
                        else ""
                    ),
                    "static_support_margin_mm": (
                        condition.support_margin_mm[frame_index]
                        if condition.support_margin_mm is not None
                        else ""
                    ),
                }
                row.update({
                    joint_name: condition.arm_q[frame_index, joint_index]
                    for joint_index, joint_name in enumerate(_H12_ARM_JOINTS)
                })
                writer.writerow(row)


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    metadata: dict,
    video_path: Path | None,
) -> None:
    payload = {
        "script": "tools/run_h12_gripper_motion_comparison.py",
        "simulation_only": True,
        "uses_hardware": False,
        "study_phase": "kinematic pilot before dynamic standing-controller integration",
        "purpose": (
            "Measure closure-induced grasp-center drift and the compensating H12 "
            "right-arm motion for Magpie and RH56."
        ),
        "conditions": list(PILOT_CONDITIONS),
        "fixed_states": ["floating base", "legs", "torso", "left arm"],
        "not_evaluated": [
            "dynamic balance or CoM support-polygon margin",
            "contact force, friction, or grasp success",
            "screw extraction force and extraction-axis error",
            "controller tracking error",
        ],
        "grasp": {
            "mode": "two-finger line",
            "center_policy": "antipodal",
            "target_is_visual_screw_proxy_only": True,
            **metadata,
        },
        "magpie": {
            "wrist_policy": "fixed",
            "finger_motion": "quasi-static MuJoCo 4-bar linkage lookup",
            "model": str(args.magpie_xml.resolve()),
            "isolated_gripper_model": str(args.magpie_gripper_xml.resolve()),
        },
        "rh56": {
            "fixed_condition": "holds the open-grasp wrist pose",
            "compensated_condition": (
                "recomputes the wrist pose at every width to keep both the "
                "analytical antipodal grasp center and pinch frame fixed"
            ),
            "model": str(args.rh56_xml.resolve()),
        },
        "ik": {
            "solver": "PINK differential IK with non-arm joints locked",
            "initial_iterations": args.ik_initial_iters,
            "step_iterations": args.ik_step_iters,
        },
        "rendering": {
            "video_path": str(video_path.resolve()) if video_path else None,
            "fps": args.fps,
            "moving_frames": args.frames,
            "hold_frames": args.hold_frames,
            "panel_size": [args.panel_width, args.panel_height],
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        },
        "paper_reference": str(
            Path(
                "/home/tanxuan/Downloads/"
                "HUMANOIDS_2026___Generalized_Open_Library_of_Embodied_Modules.pdf"
            )
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def add_sphere(scene, point: np.ndarray, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(point, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_capsule(scene, point_a: np.ndarray, point_b: np.ndarray, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(point_a, dtype=np.float64),
        np.asarray(point_b, dtype=np.float64),
    )
    scene.ngeom += 1


def add_cylinder(
    scene,
    center: np.ndarray,
    radius: float,
    half_height: float,
    rgba,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, half_height, 0.0], dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_comparison_geoms(
    scene,
    *,
    target: np.ndarray,
    anchor: np.ndarray,
    contact_axis: np.ndarray,
    color,
) -> None:
    # A simple vertical screw proxy: head centered at the commanded grasp point.
    add_cylinder(scene, target, 0.010, 0.003, (0.78, 0.73, 0.18, 0.82))
    add_cylinder(
        scene,
        target - np.array([0.0, 0.0, 0.028]),
        0.003,
        0.025,
        (0.48, 0.48, 0.50, 0.88),
    )
    add_sphere(scene, target, 0.004, (1.0, 0.9, 0.1, 0.95))
    add_sphere(scene, anchor, 0.005, color)
    axis_half_length = 0.045
    add_capsule(
        scene,
        anchor - axis_half_length * contact_axis,
        anchor + axis_half_length * contact_axis,
        0.0012,
        color,
    )
    if np.linalg.norm(anchor - target) > 1e-4:
        add_capsule(scene, target, anchor, 0.0015, color)


def make_camera(args: argparse.Namespace, target_world: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target_world
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    return camera


def add_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    margin = 10
    line_height = 16
    height = 2 * margin + line_height * len(lines)
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0, 155))
    for line_index, line in enumerate(lines):
        draw.text(
            (margin, margin + line_index * line_height),
            line,
            fill=(255, 255, 255, 255),
        )
    return np.asarray(image)


def video_extension(video_format: str) -> str:
    if video_format == "gif":
        return ".gif"
    if video_format == "mp4":
        return ".mp4"
    try:
        import imageio_ffmpeg  # noqa: F401

        return ".mp4"
    except ImportError:
        return ".gif"


def render_video(
    conditions: list[Condition],
    *,
    args: argparse.Namespace,
) -> Path:
    extension = video_extension(args.video_format)
    if args.video_format == "mp4" and extension != ".mp4":
        raise RuntimeError("MP4 output requires imageio-ffmpeg")
    path = args.out / f"comparison{extension}"

    models = [mujoco.MjModel.from_xml_path(str(item.model_path)) for item in conditions]
    for model in models:
        model.vis.headlight.ambient[:] = 0.45
        model.vis.headlight.diffuse[:] = 0.80
        model.vis.headlight.specular[:] = 0.10
    data_rows = [mujoco.MjData(model) for model in models]
    renderers = [
        mujoco.Renderer(model, height=args.panel_height, width=args.panel_width)
        for model in models
    ]
    options = []
    for _ in models:
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        options.append(option)
    camera = make_camera(args, conditions[0].target_world)
    total_frames = args.frames + args.hold_frames

    writer = None
    gif_frames: list[np.ndarray] = []
    if extension == ".mp4":
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(path),
            (args.panel_width * len(conditions), args.panel_height),
            fps=args.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
        )
        writer.send(None)

    try:
        for output_index in range(total_frames):
            frame_index = min(output_index, args.frames - 1)
            panels = []
            for condition, model, data, renderer, option in zip(
                conditions, models, data_rows, renderers, options
            ):
                data.qpos[:] = condition.qpos[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                add_comparison_geoms(
                    renderer.scene,
                    target=condition.target_world,
                    anchor=condition.anchor_world[frame_index],
                    contact_axis=condition.contact_axis_world[frame_index],
                    color=CONDITION_COLORS[condition.name],
                )
                panel = renderer.render()
                panel = add_text(
                    panel,
                    [
                        CONDITION_LABELS[condition.name],
                        f"width {condition.actual_width_mm[frame_index]:.1f} mm",
                        f"grasp-center error {condition.anchor_error_mm[frame_index]:.1f} mm",
                        (
                            f"contact-axis change "
                            f"{condition.contact_axis_change_deg[frame_index]:.1f} deg"
                        ),
                        (
                            f"wrist {condition.wrist_path_mm[frame_index]:.1f} mm | "
                            f"{condition.wrist_rotation_path_deg[frame_index]:.1f} deg"
                        ),
                        *(
                            [
                                "static CoM margin "
                                f"{condition.support_margin_mm[frame_index]:.1f} mm"
                            ]
                            if condition.support_margin_mm is not None
                            else []
                        ),
                    ],
                )
                panels.append(panel)
            combined = np.ascontiguousarray(np.concatenate(panels, axis=1))
            if writer is not None:
                writer.send(combined)
            else:
                gif_frames.append(combined)
    finally:
        if writer is not None:
            writer.close()
        for renderer in renderers:
            renderer.close()

    if extension == ".gif":
        from PIL import Image

        images = [Image.fromarray(frame).convert("RGB") for frame in gif_frames]
        duration_ms = max(1, int(round(1000.0 / args.fps)))
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
    return path


def live_viewer_worker(
    condition: Condition,
    fps: int,
    once: bool,
    camera_settings: tuple[float, float, float],
) -> None:
    # launch_passive needs the GLFW display backend, not the headless EGL renderer.
    os.environ.pop("MUJOCO_GL", None)
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(condition.model_path))
    data = mujoco.MjData(model)
    frame_period = 1.0 / float(fps)
    distance, azimuth, elevation = camera_settings
    print(f"[live] {CONDITION_LABELS[condition.name]}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = condition.target_world
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation
        while viewer.is_running():
            for frame_index in range(len(condition.qpos)):
                if not viewer.is_running():
                    return
                start = time.monotonic()
                data.qpos[:] = condition.qpos[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                add_comparison_geoms(
                    viewer.user_scn,
                    target=condition.target_world,
                    anchor=condition.anchor_world[frame_index],
                    contact_axis=condition.contact_axis_world[frame_index],
                    color=CONDITION_COLORS[condition.name],
                )
                viewer.sync()
                remaining = frame_period - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if once:
                while viewer.is_running():
                    time.sleep(0.05)
                return


def run_live(conditions: list[Condition], args: argparse.Namespace) -> None:
    selected = (
        conditions
        if args.live_condition == "all"
        else [item for item in conditions if item.name == args.live_condition]
    )
    context = multiprocessing.get_context("spawn")
    processes = []
    settings = (
        args.camera_distance,
        args.camera_azimuth,
        args.camera_elevation,
    )
    for condition in selected:
        process = context.Process(
            target=live_viewer_worker,
            args=(condition, args.fps, args.once, settings),
        )
        process.start()
        processes.append(process)
    print("[comparison] live viewers started; close the window(s) to stop")
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()


def validate_args(args: argparse.Namespace) -> None:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.hold_frames < 0:
        raise ValueError("--hold-frames cannot be negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise ValueError("panel dimensions must be positive")


def print_summary(conditions: Iterable[Condition]) -> None:
    for condition in conditions:
        row = condition_summary(condition)
        print(
            f"[comparison] {condition.name}: "
            f"max grasp-center error={row['max_anchor_error_mm']:.1f} mm, "
            f"contact-axis change={row['max_contact_axis_change_deg']:.1f} deg, "
            f"wrist path={row['wrist_path_mm']:.1f} mm, "
            f"wrist rotation={row['wrist_rotation_deg']:.1f} deg, "
            f"max IK error={row['max_ik_position_error_mm']:.2f} mm"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)

    conditions, metadata = prepare_conditions(args)
    write_summary(args.out / "summary.csv", conditions)
    write_trajectory(args.out / "trajectory.csv", conditions, fps=args.fps)
    print_summary(conditions)

    video_path = None
    if not args.no_video:
        print("[comparison] rendering side-by-side video...")
        video_path = render_video(conditions, args=args)
        print(f"[comparison] video: {video_path}")

    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        metadata=metadata,
        video_path=video_path,
    )
    print(f"[comparison] results: {args.out}")

    if args.live:
        run_live(conditions, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
