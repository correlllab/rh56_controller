#!/usr/bin/env python3
"""Characterize RH56 closure-frame sensitivity on H12.

The RH56 thumb does not provide a unique planar contact surface.  The closure
frame is therefore sensitive to the point selected on the thumb distal body.
This simulation-only experiment keeps the physical finger command trajectory
fixed and perturbs only that local thumb reference point.  It reports how the
implied closure pitch, desired H12 wrist trajectory, and arm IK motion change.

The script also evaluates a practical-cutoff proxy: after a configurable grasp
width, hold wrist orientation fixed while continuing the finger closure and
translate only enough to keep the antipodal center at the target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
    ClosureResult,
    GraspMode,
    InspireHandFK,
)
from rh56_controller.grasp_viz_workers import (  # noqa: E402
    _H12_ARM_JOINTS,
    _H12_R_HAND_TO_WRIST,
    _H12_T_HAND_TO_WRIST,
)
from tools.run_h12_gripper_motion_comparison import (  # noqa: E402
    IKSolution,
    contact_frame_from_axis,
    rotation_distance_deg,
    solve_h12_ik,
)


MODE_MAP = {
    "line": (GraspMode.LINE_2F, 2),
    "plane4": (GraspMode.PLANE_4F, 4),
}


@dataclass(frozen=True)
class ReferenceSpec:
    name: str
    offset_local_m: np.ndarray


@dataclass
class SensitivityTrajectory:
    name: str
    policy: str
    offset_local_m: np.ndarray
    command_width_mm: np.ndarray
    reference_width_mm: np.ndarray
    raw_tilt_deg: np.ndarray
    clipped_tilt_deg: np.ndarray
    center_hand_m: np.ndarray
    wrist_target: np.ndarray
    ik: IKSolution | None = None

    @property
    def wrist_displacement_mm(self) -> np.ndarray:
        return 1000.0 * np.linalg.norm(
            self.wrist_target[:, :3, 3] - self.wrist_target[0, :3, 3],
            axis=1,
        )

    @property
    def wrist_path_mm(self) -> np.ndarray:
        steps = np.linalg.norm(
            np.diff(self.wrist_target[:, :3, 3], axis=0), axis=1
        )
        return 1000.0 * np.concatenate(([0.0], np.cumsum(steps)))

    @property
    def wrist_rotation_from_start_deg(self) -> np.ndarray:
        initial = self.wrist_target[0, :3, :3]
        return np.asarray([
            rotation_distance_deg(initial, current)
            for current in self.wrist_target[:, :3, :3]
        ])

    @property
    def wrist_rotation_path_deg(self) -> np.ndarray:
        steps = [
            rotation_distance_deg(previous, current)
            for previous, current in zip(
                self.wrist_target[:-1, :3, :3],
                self.wrist_target[1:, :3, :3],
            )
        ]
        return np.concatenate(([0.0], np.cumsum(steps)))

    @property
    def arm_joint_travel_rad(self) -> np.ndarray:
        if self.ik is None:
            return np.full(len(self.command_width_mm), np.nan)
        steps = np.abs(np.diff(self.ik.arm_q, axis=0)).sum(axis=1)
        return np.concatenate(([0.0], np.cumsum(steps)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep RH56 grasp width and thumb reference-point placement to "
            "measure closure pitch and H12 arm-motion sensitivity."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/h12_closure_plane_sensitivity"),
    )
    parser.add_argument("--mode", choices=MODE_MAP, default="line")
    parser.add_argument("--samples", type=int, default=81)
    parser.add_argument("--start-width-mm", type=float, default=None)
    parser.add_argument("--end-width-mm", type=float, default=None)
    parser.add_argument(
        "--thumb-offset-mm",
        type=float,
        default=5.0,
        help=(
            "Reference-point perturbation magnitude along each thumb-distal "
            "local axis; ±X, ±Y, and ±Z are evaluated."
        ),
    )
    parser.add_argument(
        "--practical-cutoff-mm",
        type=float,
        default=40.0,
        help=(
            "Provisional closure cutoff. The cutoff-hold proxy stops wrist "
            "rotation here but permits center-preserving translation."
        ),
    )
    parser.add_argument(
        "--knee-threshold-deg-per-mm",
        type=float,
        default=1.0,
    )
    parser.add_argument("--grasp-x", type=float, default=0.25)
    parser.add_argument("--grasp-y", type=float, default=-0.20)
    parser.add_argument("--grasp-z", type=float, default=0.15)
    parser.add_argument("--plane-rz-deg", type=float, default=-120.0)
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument("--skip-ik", action="store_true")
    parser.add_argument("--ik-initial-iters", type=int, default=500)
    parser.add_argument("--ik-step-iters", type=int, default=20)
    parser.add_argument(
        "--ik-variants",
        choices=("baseline", "all"),
        default="all",
        help="Solve arm IK for only baseline policies or for every reference perturbation.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if args.thumb_offset_mm < 0.0:
        raise ValueError("--thumb-offset-mm cannot be negative")
    if args.knee_threshold_deg_per_mm <= 0.0:
        raise ValueError("--knee-threshold-deg-per-mm must be positive")
    if args.ik_initial_iters < 1 or args.ik_step_iters < 1:
        raise ValueError("IK iteration counts must be positive")


def reference_specs(offset_mm: float) -> list[ReferenceSpec]:
    magnitude = offset_mm / 1000.0
    specs = [ReferenceSpec("baseline", np.zeros(3))]
    for axis_index, axis_name in enumerate("xyz"):
        for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
            offset = np.zeros(3)
            offset[axis_index] = sign * magnitude
            specs.append(ReferenceSpec(
                f"thumb_{axis_name}_{sign_name}_{offset_mm:g}mm",
                offset,
            ))
    return specs


def thumb_reference_point(
    fk: InspireHandFK,
    *,
    ctrl_pitch: float,
    ctrl_yaw: float,
    offset_local_m: np.ndarray,
) -> np.ndarray:
    """Evaluate a perturbed site on the moving thumb-distal body."""
    fk._reset_qpos()
    fk._set_thumb_qpos(float(ctrl_pitch), float(ctrl_yaw))
    mujoco.mj_kinematics(fk._model, fk._data)

    site_id = fk._site_ids["thumb"]
    body_id = int(fk._model.site_bodyid[site_id])
    local_position = fk._model.site_pos[site_id] + offset_local_m
    body_rotation = fk._data.xmat[body_id].reshape(3, 3)
    return fk._data.xpos[body_id] + body_rotation @ local_position


def nonthumb_reference(result: ClosureResult) -> np.ndarray:
    points = [
        position
        for name, position in result.tip_positions.items()
        if name != "thumb"
    ]
    if not points:
        raise ValueError("Closure result does not contain a non-thumb contact")
    return np.vstack(points).mean(axis=0)


def wrist_target_from_hand_pose(
    hand_rotation: np.ndarray,
    center_hand: np.ndarray,
    target_pelvis: np.ndarray,
) -> np.ndarray:
    hand_position = target_pelvis - hand_rotation @ center_hand
    wrist = np.eye(4)
    wrist[:3, :3] = hand_rotation @ _H12_R_HAND_TO_WRIST
    wrist[:3, 3] = hand_position + hand_rotation @ _H12_T_HAND_TO_WRIST
    return wrist


def build_reference_trajectory(
    *,
    spec: ReferenceSpec,
    fk: InspireHandFK,
    results: list[ClosureResult],
    command_width_mm: np.ndarray,
    target_pelvis: np.ndarray,
    plane_rz_rad: float,
) -> SensitivityTrajectory:
    centers = []
    contact_frames = []
    reference_widths = []
    raw_tilts = []

    for result in results:
        thumb = thumb_reference_point(
            fk,
            ctrl_pitch=result.ctrl_values["thumb_proximal"],
            ctrl_yaw=result.ctrl_values["thumb_yaw"],
            offset_local_m=spec.offset_local_m,
        )
        opposition = nonthumb_reference(result)
        axis = opposition - thumb
        centers.append(0.5 * (thumb + opposition))
        contact_frames.append(contact_frame_from_axis(axis))
        reference_widths.append(1000.0 * float(np.hypot(axis[0], axis[2])))

        difference = thumb - opposition
        raw_tilts.append(float(np.arctan2(-difference[2], difference[0])))

    centers_array = np.asarray(centers)
    frames_array = np.asarray(contact_frames)
    raw_tilt_rad = np.unwrap(np.asarray(raw_tilts))
    clipped_tilt_rad = np.clip(raw_tilt_rad, -np.pi / 2.0, np.pi / 2.0)

    plane_rotation = ClosureResult._plane_rot(0.0, 0.0, plane_rz_rad)
    # All reference definitions start from the same physical hand orientation.
    initial_hand_rotation = (
        plane_rotation @ ClosureResult._rot_matrix(results[0].base_tilt_y)
    )
    desired_world_contact_frame = initial_hand_rotation @ frames_array[0]
    targets = []
    for center, contact_frame in zip(centers_array, frames_array):
        hand_rotation = desired_world_contact_frame @ contact_frame.T
        targets.append(wrist_target_from_hand_pose(
            hand_rotation,
            center,
            target_pelvis,
        ))

    return SensitivityTrajectory(
        name=spec.name,
        policy="strict_contact_frame",
        offset_local_m=spec.offset_local_m.copy(),
        command_width_mm=command_width_mm.copy(),
        reference_width_mm=np.asarray(reference_widths),
        raw_tilt_deg=np.degrees(raw_tilt_rad),
        clipped_tilt_deg=np.degrees(clipped_tilt_rad),
        center_hand_m=centers_array,
        wrist_target=np.asarray(targets),
    )


def cutoff_hold_trajectory(
    baseline: SensitivityTrajectory,
    *,
    target_pelvis: np.ndarray,
    cutoff_index: int,
) -> SensitivityTrajectory:
    """Hold hand/wrist orientation after the cutoff, while keeping center fixed."""
    targets = baseline.wrist_target.copy()
    cutoff_wrist_rotation = targets[cutoff_index, :3, :3]
    # Convert wrist rotation back to hand rotation.
    hand_rotation = cutoff_wrist_rotation @ _H12_R_HAND_TO_WRIST.T
    for index in range(cutoff_index + 1, len(targets)):
        targets[index] = wrist_target_from_hand_pose(
            hand_rotation,
            baseline.center_hand_m[index],
            target_pelvis,
        )

    clipped = baseline.clipped_tilt_deg.copy()
    clipped[cutoff_index + 1:] = clipped[cutoff_index]
    return SensitivityTrajectory(
        name="baseline_cutoff_hold",
        policy="orientation_hold_after_practical_cutoff",
        offset_local_m=np.zeros(3),
        command_width_mm=baseline.command_width_mm.copy(),
        reference_width_mm=baseline.reference_width_mm.copy(),
        raw_tilt_deg=baseline.raw_tilt_deg.copy(),
        clipped_tilt_deg=clipped,
        center_hand_m=baseline.center_hand_m.copy(),
        wrist_target=targets,
    )


def cumulative_arm_travel(ik: IKSolution | None) -> np.ndarray:
    if ik is None:
        return np.array([])
    steps = np.abs(np.diff(ik.arm_q, axis=0)).sum(axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def nearest_width_index(widths_mm: np.ndarray, target_mm: float) -> int:
    return int(np.argmin(np.abs(widths_mm - target_mm)))


def detect_knee_width(
    widths_mm: np.ndarray,
    rotation_deg: np.ndarray,
    *,
    threshold_deg_per_mm: float,
) -> tuple[float | None, np.ndarray]:
    rate = np.abs(np.gradient(rotation_deg, widths_mm))
    hits = np.flatnonzero(rate >= threshold_deg_per_mm)
    if len(hits) == 0:
        return None, rate
    return float(widths_mm[int(hits[0])]), rate


def solve_trajectory_ik(
    trajectory: SensitivityTrajectory,
    *,
    args: argparse.Namespace,
) -> None:
    trajectory.ik = solve_h12_ik(
        trajectory.wrist_target,
        initial_iters=args.ik_initial_iters,
        step_iters=args.ik_step_iters,
    )


def endpoint_metrics(
    trajectory: SensitivityTrajectory,
    index: int,
) -> dict[str, float]:
    arm_travel = trajectory.arm_joint_travel_rad
    return {
        "command_width_mm": float(trajectory.command_width_mm[index]),
        "reference_width_mm": float(trajectory.reference_width_mm[index]),
        "raw_tilt_deg": float(trajectory.raw_tilt_deg[index]),
        "commanded_tilt_deg": float(trajectory.clipped_tilt_deg[index]),
        "wrist_displacement_mm": float(trajectory.wrist_displacement_mm[index]),
        "wrist_path_mm": float(trajectory.wrist_path_mm[index]),
        "wrist_rotation_from_start_deg": float(
            trajectory.wrist_rotation_from_start_deg[index]
        ),
        "wrist_rotation_path_deg": float(
            trajectory.wrist_rotation_path_deg[index]
        ),
        "arm_joint_travel_rad": float(arm_travel[index]),
        "ik_position_error_mm": (
            float(trajectory.ik.position_error_mm[index])
            if trajectory.ik is not None
            else float("nan")
        ),
        "ik_orientation_error_deg": (
            float(trajectory.ik.orientation_error_deg[index])
            if trajectory.ik is not None
            else float("nan")
        ),
    }


def write_summary(
    path: Path,
    trajectories: list[SensitivityTrajectory],
    *,
    practical_index: int,
    knee_width_mm: float | None,
) -> None:
    rows = []
    for trajectory in trajectories:
        for endpoint, index in (
            ("practical_cutoff", practical_index),
            ("mathematical_endpoint", len(trajectory.command_width_mm) - 1),
        ):
            row = {
                "variant": trajectory.name,
                "policy": trajectory.policy,
                "endpoint": endpoint,
                "thumb_offset_x_mm": 1000.0 * trajectory.offset_local_m[0],
                "thumb_offset_y_mm": 1000.0 * trajectory.offset_local_m[1],
                "thumb_offset_z_mm": 1000.0 * trajectory.offset_local_m[2],
                "detected_knee_width_mm": (
                    knee_width_mm if knee_width_mm is not None else ""
                ),
                **endpoint_metrics(trajectory, index),
            }
            rows.append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sweep(
    path: Path,
    trajectories: list[SensitivityTrajectory],
    *,
    practical_index: int,
    baseline_rotation_rate: np.ndarray,
) -> None:
    fields = [
        "variant",
        "policy",
        "sample",
        "is_at_or_beyond_practical_cutoff",
        "command_width_mm",
        "reference_width_mm",
        "raw_tilt_deg",
        "commanded_tilt_deg",
        "baseline_rotation_rate_deg_per_mm",
        "wrist_x_m",
        "wrist_y_m",
        "wrist_z_m",
        "wrist_displacement_mm",
        "cumulative_wrist_path_mm",
        "wrist_rotation_from_start_deg",
        "cumulative_wrist_rotation_deg",
        "cumulative_arm_joint_travel_rad",
        "ik_position_error_mm",
        "ik_orientation_error_deg",
        *_H12_ARM_JOINTS,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trajectory in trajectories:
            arm_travel = trajectory.arm_joint_travel_rad
            for index in range(len(trajectory.command_width_mm)):
                row = {
                    "variant": trajectory.name,
                    "policy": trajectory.policy,
                    "sample": index,
                    "is_at_or_beyond_practical_cutoff": index >= practical_index,
                    "command_width_mm": trajectory.command_width_mm[index],
                    "reference_width_mm": trajectory.reference_width_mm[index],
                    "raw_tilt_deg": trajectory.raw_tilt_deg[index],
                    "commanded_tilt_deg": trajectory.clipped_tilt_deg[index],
                    "baseline_rotation_rate_deg_per_mm": baseline_rotation_rate[index],
                    "wrist_x_m": trajectory.wrist_target[index, 0, 3],
                    "wrist_y_m": trajectory.wrist_target[index, 1, 3],
                    "wrist_z_m": trajectory.wrist_target[index, 2, 3],
                    "wrist_displacement_mm": trajectory.wrist_displacement_mm[index],
                    "cumulative_wrist_path_mm": trajectory.wrist_path_mm[index],
                    "wrist_rotation_from_start_deg": (
                        trajectory.wrist_rotation_from_start_deg[index]
                    ),
                    "cumulative_wrist_rotation_deg": (
                        trajectory.wrist_rotation_path_deg[index]
                    ),
                    "cumulative_arm_joint_travel_rad": arm_travel[index],
                    "ik_position_error_mm": (
                        trajectory.ik.position_error_mm[index]
                        if trajectory.ik is not None
                        else float("nan")
                    ),
                    "ik_orientation_error_deg": (
                        trajectory.ik.orientation_error_deg[index]
                        if trajectory.ik is not None
                        else float("nan")
                    ),
                }
                for joint_index, joint_name in enumerate(_H12_ARM_JOINTS):
                    row[joint_name] = (
                        trajectory.ik.arm_q[index, joint_index]
                        if trajectory.ik is not None
                        else float("nan")
                    )
                writer.writerow(row)


def plot_results(
    path: Path,
    trajectories: list[SensitivityTrajectory],
    *,
    practical_width_mm: float,
    knee_width_mm: float | None,
    thumb_offset_mm: float,
) -> None:
    import matplotlib.pyplot as plt

    reference_curves = [
        trajectory
        for trajectory in trajectories
        if trajectory.policy == "strict_contact_frame"
    ]
    baseline = next(item for item in trajectories if item.name == "baseline")
    cutoff = next(
        item for item in trajectories if item.name == "baseline_cutoff_hold"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    ax_pitch, ax_translation, ax_rotation, ax_arm = axes.ravel()

    for curve in reference_curves:
        if curve.name == "baseline":
            continue
        color = "#8fa7bf"
        ax_pitch.plot(
            curve.command_width_mm,
            curve.clipped_tilt_deg,
            color=color,
            alpha=0.65,
            linewidth=1.0,
        )
        ax_translation.plot(
            curve.command_width_mm,
            curve.wrist_displacement_mm,
            color=color,
            alpha=0.65,
            linewidth=1.0,
        )
        ax_rotation.plot(
            curve.command_width_mm,
            curve.wrist_rotation_from_start_deg,
            color=color,
            alpha=0.65,
            linewidth=1.0,
        )
        if curve.ik is not None:
            ax_arm.plot(
                curve.command_width_mm,
                curve.arm_joint_travel_rad,
                color=color,
                alpha=0.65,
                linewidth=1.0,
            )

    baseline_style = dict(color="#005f9e", linewidth=2.5, label="baseline thumb site")
    cutoff_style = dict(
        color="#e66101",
        linewidth=2.2,
        linestyle="--",
        label="hold orientation after cutoff",
    )
    ax_pitch.plot(
        baseline.command_width_mm,
        baseline.clipped_tilt_deg,
        **baseline_style,
    )
    ax_pitch.plot(
        cutoff.command_width_mm,
        cutoff.clipped_tilt_deg,
        **cutoff_style,
    )
    ax_translation.plot(
        baseline.command_width_mm,
        baseline.wrist_displacement_mm,
        **baseline_style,
    )
    ax_translation.plot(
        cutoff.command_width_mm,
        cutoff.wrist_displacement_mm,
        **cutoff_style,
    )
    ax_rotation.plot(
        baseline.command_width_mm,
        baseline.wrist_rotation_from_start_deg,
        **baseline_style,
    )
    ax_rotation.plot(
        cutoff.command_width_mm,
        cutoff.wrist_rotation_from_start_deg,
        **cutoff_style,
    )
    if baseline.ik is not None:
        ax_arm.plot(
            baseline.command_width_mm,
            baseline.arm_joint_travel_rad,
            **baseline_style,
        )
    if cutoff.ik is not None:
        ax_arm.plot(
            cutoff.command_width_mm,
            cutoff.arm_joint_travel_rad,
            **cutoff_style,
        )

    ax_pitch.set_ylabel("closure-plane pitch (deg)")
    ax_translation.set_ylabel("wrist displacement (mm)")
    ax_rotation.set_ylabel("wrist rotation from start (deg)")
    ax_arm.set_ylabel("cumulative arm joint travel (rad)")
    for axis in axes[1]:
        axis.set_xlabel("baseline commanded grasp width (mm)")

    for axis in axes.ravel():
        axis.axvline(
            practical_width_mm,
            color="#333333",
            linestyle=":",
            linewidth=1.5,
            label="provisional practical cutoff",
        )
        if knee_width_mm is not None:
            axis.axvline(
                knee_width_mm,
                color="#b2182b",
                linestyle="-.",
                linewidth=1.2,
                label="detected knee",
            )
        axis.grid(True, alpha=0.25)
        axis.set_xlim(
            float(baseline.command_width_mm.max()),
            float(baseline.command_width_mm.min()),
        )

    handles, labels = ax_pitch.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    figure.suptitle(
        "H12–RH56 closure-frame sensitivity to thumb reference point "
        f"(±{thumb_offset_mm:g} mm)",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    width_range_mm: tuple[float, float],
    thumb_site_local_m: np.ndarray,
    practical_index: int,
    knee_width_mm: float | None,
    trajectories: list[SensitivityTrajectory],
) -> None:
    payload = {
        "script": "tools/run_h12_closure_plane_sensitivity.py",
        "simulation_only": True,
        "uses_hardware": False,
        "purpose": (
            "Separate coupled-hand closure motion from closure-frame sensitivity "
            "caused by choosing an arbitrary reference point on the non-planar thumb."
        ),
        "mode": args.mode,
        "physical_command_schedule": (
            "All reference-point variants reuse the baseline ClosureGeometry "
            "finger commands at each baseline width. Only the point on the moving "
            "thumb-distal body changes."
        ),
        "thumb_reference": {
            "site": "right_thumb_tip",
            "parent_body": "thumb_distal",
            "baseline_site_local_m": thumb_site_local_m.tolist(),
            "perturbation_mm": args.thumb_offset_mm,
            "variants": [
                {
                    "name": trajectory.name,
                    "offset_local_mm": (
                        1000.0 * trajectory.offset_local_m
                    ).tolist(),
                }
                for trajectory in trajectories
                if trajectory.policy == "strict_contact_frame"
            ],
        },
        "width_range_mm": list(width_range_mm),
        "samples": args.samples,
        "target_pelvis_m": [args.grasp_x, args.grasp_y, args.grasp_z],
        "plane_rz_deg": args.plane_rz_deg,
        "practical_cutoff": {
            "requested_mm": args.practical_cutoff_mm,
            "sampled_mm": float(
                trajectories[0].command_width_mm[practical_index]
            ),
            "status": "provisional",
            "reason": (
                "40 mm matches the checked-in H12 demo/default grasp width. "
                "The mentor's exact actuation cutoff and hardcoded final pose "
                "were not found in the checked-in controller source."
            ),
            "proxy_policy": (
                "Hold the strict baseline wrist orientation after the cutoff, "
                "while continuing center-preserving translation. This is not "
                "claimed to reproduce the mentor's exact hardcoded pose."
            ),
        },
        "knee_detection": {
            "metric": "baseline wrist rotation-from-start derivative",
            "threshold_deg_per_mm": args.knee_threshold_deg_per_mm,
            "detected_width_mm": knee_width_mm,
        },
        "ik": {
            "enabled": not args.skip_ik,
            "variants": args.ik_variants,
            "initial_iterations": args.ik_initial_iters,
            "step_iterations": args.ik_step_iters,
            "non_arm_joints_locked": True,
        },
        "not_evaluated": [
            "dynamic balance",
            "contact force or grasp success",
            "physical thumb surface normal",
            "mentor's exact hardcoded surface-parallel endpoint",
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)

    mode, finger_count = MODE_MAP[args.mode]
    fk = InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    width_min_m, width_max_m = closure.width_range(
        str(mode), n_fingers=finger_count
    )
    start_width_m = (
        width_max_m
        if args.start_width_mm is None
        else float(np.clip(
            args.start_width_mm / 1000.0,
            width_min_m,
            width_max_m,
        ))
    )
    end_width_m = (
        width_min_m
        if args.end_width_mm is None
        else float(np.clip(
            args.end_width_mm / 1000.0,
            width_min_m,
            start_width_m,
        ))
    )
    requested_widths = np.linspace(start_width_m, end_width_m, args.samples)
    results = [closure.solve(mode, float(width)) for width in requested_widths]
    command_width_mm = np.asarray([1000.0 * result.width for result in results])
    target_pelvis = np.array([args.grasp_x, args.grasp_y, args.grasp_z])

    trajectories = []
    for spec in reference_specs(args.thumb_offset_mm):
        print(f"[sensitivity] building {spec.name}")
        trajectories.append(build_reference_trajectory(
            spec=spec,
            fk=fk,
            results=results,
            command_width_mm=command_width_mm,
            target_pelvis=target_pelvis,
            plane_rz_rad=math.radians(args.plane_rz_deg),
        ))

    baseline = trajectories[0]
    practical_index = nearest_width_index(
        baseline.command_width_mm, args.practical_cutoff_mm
    )
    cutoff = cutoff_hold_trajectory(
        baseline,
        target_pelvis=target_pelvis,
        cutoff_index=practical_index,
    )
    trajectories.append(cutoff)

    knee_width_mm, rotation_rate = detect_knee_width(
        baseline.command_width_mm,
        baseline.wrist_rotation_from_start_deg,
        threshold_deg_per_mm=args.knee_threshold_deg_per_mm,
    )

    if not args.skip_ik:
        for trajectory in trajectories:
            should_solve = (
                args.ik_variants == "all"
                or trajectory.name in ("baseline", "baseline_cutoff_hold")
            )
            if should_solve:
                print(f"[sensitivity] solving H12 IK: {trajectory.name}")
                solve_trajectory_ik(trajectory, args=args)

    write_summary(
        args.out / "summary.csv",
        trajectories,
        practical_index=practical_index,
        knee_width_mm=knee_width_mm,
    )
    write_sweep(
        args.out / "sensitivity.csv",
        trajectories,
        practical_index=practical_index,
        baseline_rotation_rate=rotation_rate,
    )
    plot_results(
        args.out / "closure_plane_sensitivity.png",
        trajectories,
        practical_width_mm=float(baseline.command_width_mm[practical_index]),
        knee_width_mm=knee_width_mm,
        thumb_offset_mm=args.thumb_offset_mm,
    )

    thumb_site_id = fk._site_ids["thumb"]
    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        width_range_mm=(1000.0 * width_min_m, 1000.0 * width_max_m),
        thumb_site_local_m=fk._model.site_pos[thumb_site_id].copy(),
        practical_index=practical_index,
        knee_width_mm=knee_width_mm,
        trajectories=trajectories,
    )

    practical_metrics = endpoint_metrics(baseline, practical_index)
    full_metrics = endpoint_metrics(baseline, len(command_width_mm) - 1)
    cutoff_full_metrics = endpoint_metrics(cutoff, len(command_width_mm) - 1)
    print(
        "[sensitivity] baseline strict at practical cutoff: "
        f"width={practical_metrics['command_width_mm']:.1f} mm, "
        f"wrist={practical_metrics['wrist_displacement_mm']:.1f} mm / "
        f"{practical_metrics['wrist_rotation_from_start_deg']:.1f} deg"
    )
    print(
        "[sensitivity] baseline strict at mathematical endpoint: "
        f"width={full_metrics['command_width_mm']:.1f} mm, "
        f"wrist={full_metrics['wrist_displacement_mm']:.1f} mm / "
        f"{full_metrics['wrist_rotation_from_start_deg']:.1f} deg"
    )
    print(
        "[sensitivity] cutoff-hold at mathematical endpoint: "
        f"wrist={cutoff_full_metrics['wrist_displacement_mm']:.1f} mm / "
        f"{cutoff_full_metrics['wrist_rotation_from_start_deg']:.1f} deg"
    )
    if knee_width_mm is not None:
        print(f"[sensitivity] detected knee: {knee_width_mm:.1f} mm")
    else:
        print("[sensitivity] no knee crossed the requested threshold")
    print(f"[sensitivity] artifacts: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
