#!/usr/bin/env python3
"""Visual and numeric sanity check for the RH56 capsule hand proxy.

This is a debug tool, not a paper-result generator.  It verifies that the
capsule proxy is extracted from the MuJoCo hand FK as intended, then samples a
straight-line hand-base path against the same object AABB proxy used by the
analytical no-go volume script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

from rh56_controller.capsule_hand_proxy import (
    Capsule,
    build_capsule_proxy,
    closure_base_position,
    closure_base_rotation,
    sample_linear_path_collisions,
    set_closure_qpos,
    transform_capsules,
)
from rh56_controller.grasp_geometry import CTRL_MAX, ClosureGeometry, ClosureResult, InspireHandFK
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS, ObjectSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate debug artifacts for the RH56 capsule hand proxy."
    )
    parser.add_argument("--object", choices=sorted(BUILTIN_OBJECTS), default="debug_40mm_cube")
    parser.add_argument("--out", type=Path, default=Path("artifacts/capsule_proxy_demo"))
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument("--mode-override", choices=["line", "plane3", "plane4", "plane5"], default=None)
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--start-mm",
        type=float,
        nargs=3,
        default=None,
        help="Manual hand-base start position. If omitted, auto-selects one blocked and one clear example.",
    )
    parser.add_argument("--path-samples", type=int, default=24)
    parser.add_argument("--final-contact-ignore-mm", type=float, default=0.0)
    parser.add_argument("--auto-grid-step-mm", type=float, default=90.0)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--path-hand-shape",
        choices=["closed", "open"],
        default="open",
        help=(
            "Hand shape used by the capsule path check and plotted hand proxy. "
            "'open' keeps fingers at qpos 0 and sets thumb_yaw to max qpos "
            "(real raw 0, fully rotated)."
        ),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open an interactive MuJoCo viewer with actual hand mesh plus capsule overlay.",
    )
    return parser.parse_args()


def solve_object_grasp(
    closure: ClosureGeometry,
    obj: ObjectSpec,
    mode: str,
    width_offset_m: float,
) -> ClosureResult:
    internal_width_m = obj.grasp_width_m + width_offset_m
    if mode == "line":
        return closure.line(internal_width_m)
    if mode.startswith("plane"):
        return closure.plane(internal_width_m, n_fingers=int(mode[-1]))
    raise ValueError(f"Unsupported mode: {mode}")


def path_ctrl_values(result: ClosureResult, shape: str) -> dict[str, float]:
    if shape == "closed":
        return dict(result.ctrl_values)
    if shape == "open":
        values = {key: 0.0 for key in result.ctrl_values}
        values["thumb_yaw"] = CTRL_MAX["thumb_yaw"]
        return values
    raise ValueError(f"Unsupported path hand shape: {shape}")


def tabletop_object_center(obj: ObjectSpec) -> np.ndarray:
    """Place the object AABB on the ground plane instead of centered on it."""

    return np.array([0.0, 0.0, obj.size_m[2] / 2.0], dtype=float)


def _frange(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(math.floor((hi - lo) / step)) + 1
    return lo + step * np.arange(n, dtype=float)


def _collided(rows: list[dict[str, object]]) -> bool:
    return any(bool(row["collision"]) for row in rows)


def _min_clearance(rows: list[dict[str, object]]) -> float:
    usable = [float(row["clearance_m"]) for row in rows if not bool(row["ignored_for_final_contact"])]
    if not usable:
        usable = [float(row["clearance_m"]) for row in rows]
    return min(usable)


def choose_demo_starts(
    capsules_base: list[Capsule],
    *,
    final: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    path_samples: int,
    final_ignore_m: float,
    auto_grid_step_m: float,
) -> list[tuple[str, np.ndarray, list[dict[str, object]]]]:
    starts: list[tuple[str, np.ndarray, list[dict[str, object]]]] = []
    blocked_best: tuple[float, np.ndarray, list[dict[str, object]]] | None = None
    clear_best: tuple[float, np.ndarray, list[dict[str, object]]] | None = None

    xs = _frange(-0.18, 0.18, auto_grid_step_m)
    ys = _frange(-0.18, 0.18, auto_grid_step_m)
    zs = _frange(-0.06, 0.18, auto_grid_step_m)
    for x in xs:
        for y in ys:
            for z in zs:
                start = np.array([x, y, z], dtype=float)
                if np.linalg.norm(start - final) <= final_ignore_m:
                    continue
                rows = sample_linear_path_collisions(
                    capsules_base,
                    start=start,
                    final=final,
                    rotation=rotation,
                    half_extents=half_extents,
                    aabb_center=aabb_center,
                    path_samples=path_samples,
                    final_ignore_m=final_ignore_m,
                )
                min_clearance = _min_clearance(rows)
                path_len = float(np.linalg.norm(start - final))
                if _collided(rows):
                    score = min_clearance
                    if blocked_best is None or score < blocked_best[0]:
                        blocked_best = (score, start, rows)
                else:
                    # Prefer a readable, non-trivial path that is not just far away.
                    score = abs(path_len - 0.22) - min_clearance
                    if clear_best is None or score < clear_best[0]:
                        clear_best = (score, start, rows)

    if blocked_best is not None:
        starts.append(("blocked_auto", blocked_best[1], blocked_best[2]))
    if clear_best is not None:
        starts.append(("clear_auto", clear_best[1], clear_best[2]))
    if not starts:
        raise RuntimeError("Could not auto-select any capsule-proxy demo starts")
    return starts


def write_capsules_csv(path: Path, capsules_base: list[Capsule], capsules_final: list[Capsule]) -> None:
    fields = [
        "frame",
        "name",
        "group",
        "source",
        "radius_mm",
        "p0_x_mm",
        "p0_y_mm",
        "p0_z_mm",
        "p1_x_mm",
        "p1_y_mm",
        "p1_z_mm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame, capsules in (("hand_base", capsules_base), ("final_world", capsules_final)):
            for capsule in capsules:
                writer.writerow(
                    {
                        "frame": frame,
                        "name": capsule.name,
                        "group": capsule.group,
                        "source": capsule.source,
                        "radius_mm": f"{capsule.radius * 1000.0:.3f}",
                        "p0_x_mm": f"{capsule.p0[0] * 1000.0:.3f}",
                        "p0_y_mm": f"{capsule.p0[1] * 1000.0:.3f}",
                        "p0_z_mm": f"{capsule.p0[2] * 1000.0:.3f}",
                        "p1_x_mm": f"{capsule.p1[0] * 1000.0:.3f}",
                        "p1_y_mm": f"{capsule.p1[1] * 1000.0:.3f}",
                        "p1_z_mm": f"{capsule.p1[2] * 1000.0:.3f}",
                    }
                )


def write_path_samples_csv(
    path: Path,
    cases: list[tuple[str, np.ndarray, list[dict[str, object]]]],
) -> None:
    fields = [
        "case",
        "sample_idx",
        "path_check",
        "interval_alpha_start",
        "interval_alpha_end",
        "alpha",
        "base_x_mm",
        "base_y_mm",
        "base_z_mm",
        "near_final_contact_region",
        "ignored_for_final_contact",
        "collision",
        "nearest_capsule",
        "nearest_group",
        "clearance_mm",
        "distance_mm",
        "radius_mm",
        "nearest_t_segment",
        "nearest_t_path",
        "raw_nearest_capsule",
        "raw_nearest_group",
        "raw_nearest_alpha",
        "raw_nearest_t_segment",
        "raw_nearest_t_path",
        "raw_clearance_mm",
        "raw_collision_count",
        "ignored_collision_count",
        "active_collision_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case_name, _start, rows in cases:
            for row in rows:
                base = np.asarray(row["base_position"], dtype=float)
                writer.writerow(
                    {
                        "case": case_name,
                        "sample_idx": row["sample_idx"],
                        "path_check": row["path_check"],
                        "interval_alpha_start": f"{float(row['interval_alpha_start']):.6f}",
                        "interval_alpha_end": f"{float(row['interval_alpha_end']):.6f}",
                        "alpha": f"{float(row['alpha']):.6f}",
                        "base_x_mm": f"{base[0] * 1000.0:.3f}",
                        "base_y_mm": f"{base[1] * 1000.0:.3f}",
                        "base_z_mm": f"{base[2] * 1000.0:.3f}",
                        "near_final_contact_region": int(bool(row["near_final_contact_region"])),
                        "ignored_for_final_contact": int(bool(row["ignored_for_final_contact"])),
                        "collision": int(bool(row["collision"])),
                        "nearest_capsule": row["nearest_capsule"],
                        "nearest_group": row["nearest_group"],
                        "clearance_mm": f"{float(row['clearance_m']) * 1000.0:.3f}",
                        "distance_mm": f"{float(row['distance_m']) * 1000.0:.3f}",
                        "radius_mm": f"{float(row['radius_m']) * 1000.0:.3f}",
                        "nearest_t_segment": f"{float(row['nearest_t_segment']):.6f}",
                        "nearest_t_path": f"{float(row['nearest_t_path']):.6f}",
                        "raw_nearest_capsule": row["raw_nearest_capsule"],
                        "raw_nearest_group": row["raw_nearest_group"],
                        "raw_nearest_alpha": f"{float(row['raw_nearest_alpha']):.6f}",
                        "raw_nearest_t_segment": f"{float(row['raw_nearest_t_segment']):.6f}",
                        "raw_nearest_t_path": f"{float(row['raw_nearest_t_path']):.6f}",
                        "raw_clearance_mm": f"{float(row['raw_clearance_m']) * 1000.0:.3f}",
                        "raw_collision_count": row["raw_collision_count"],
                        "ignored_collision_count": row["ignored_collision_count"],
                        "active_collision_count": row["active_collision_count"],
                    }
                )


def write_summary_csv(
    path: Path,
    *,
    obj: ObjectSpec,
    mode: str,
    yaw_deg: float,
    final: np.ndarray,
    tip_error_mm: float | None,
    cases: list[tuple[str, np.ndarray, list[dict[str, object]]]],
) -> None:
    fields = [
        "case",
        "object",
        "mode",
        "yaw_deg",
        "start_x_mm",
        "start_y_mm",
        "start_z_mm",
        "final_x_mm",
        "final_y_mm",
        "final_z_mm",
        "path_collides_before_final_ignore",
        "min_clearance_mm",
        "capsule_tip_error_max_mm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case_name, start, rows in cases:
            writer.writerow(
                {
                    "case": case_name,
                    "object": obj.name,
                    "mode": mode,
                    "yaw_deg": f"{yaw_deg:.3f}",
                    "start_x_mm": f"{start[0] * 1000.0:.3f}",
                    "start_y_mm": f"{start[1] * 1000.0:.3f}",
                    "start_z_mm": f"{start[2] * 1000.0:.3f}",
                    "final_x_mm": f"{final[0] * 1000.0:.3f}",
                    "final_y_mm": f"{final[1] * 1000.0:.3f}",
                    "final_z_mm": f"{final[2] * 1000.0:.3f}",
                    "path_collides_before_final_ignore": int(_collided(rows)),
                    "min_clearance_mm": f"{_min_clearance(rows) * 1000.0:.3f}",
                    "capsule_tip_error_max_mm": (
                        f"{tip_error_mm:.3f}" if tip_error_mm is not None else ""
                    ),
                }
            )


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    obj: ObjectSpec,
    mode: str,
    final: np.ndarray,
    rotation: np.ndarray,
    aabb_center: np.ndarray,
    capsules_base: list[Capsule],
) -> None:
    payload = {
        "script": "tools/verify_capsule_hand_proxy.py",
        "simulation_only": True,
        "uses_hardware": False,
        "purpose": "debug visualization and numeric sanity check for the capsule hand proxy",
        "object": {
            "name": obj.name,
            "label": obj.label,
            "size_m": obj.size_m,
            "aabb_center_m": aabb_center.tolist(),
            "bottom_z_m": float(aabb_center[2] - obj.size_m[2] / 2.0),
            "top_z_m": float(aabb_center[2] + obj.size_m[2] / 2.0),
            "grasp_width_m": obj.grasp_width_m,
            "mode": mode,
            "notes": obj.notes,
        },
        "capsule_proxy": {
            "source": (
                "MuJoCo FK body origins and fingertip sites at the selected "
                "path_hand_shape qpos"
            ),
            "capsule_count": len(capsules_base),
            "radius_scale": args.radius_scale,
            "path_hand_shape": args.path_hand_shape,
            "path_hand_shape_note": (
                "open means fingers are at qpos zero while thumb_yaw is at "
                "max qpos, corresponding to real raw 0"
            ),
            "includes_palm": True,
            "collision_test": (
                "minimum swept capsule centerline distance to tabletop object "
                "AABB over each straight-line path interval"
            ),
        },
        "path": {
            "yaw_deg": args.yaw_deg,
            "path_samples": args.path_samples,
            "path_collision_check": (
                "continuous swept-capsule interval checks; not endpoint-only sampling"
            ),
            "final_contact_ignore_mm": args.final_contact_ignore_mm,
            "final_contact_ignore_groups": ["thumb", "index", "middle", "ring", "pinky"],
            "final_contact_ignore_note": "palm/body collisions are still counted inside the final-contact region",
            "final_base_position_m": final.tolist(),
            "base_rotation_world_from_hand": rotation.tolist(),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _tip_endpoint_error_mm(result: ClosureResult, capsules_base: list[Capsule]) -> float:
    errors = []
    for finger, tip in result.tip_positions.items():
        prefix = f"{finger}_"
        finger_capsules = [capsule for capsule in capsules_base if capsule.name.startswith(prefix)]
        if not finger_capsules:
            continue
        endpoint = finger_capsules[-1].p1
        errors.append(float(np.linalg.norm(endpoint - tip) * 1000.0))
    return max(errors) if errors else 0.0


def maybe_write_plot(
    path: Path,
    *,
    obj: ObjectSpec,
    aabb_center: np.ndarray,
    capsules_final: list[Capsule],
    final: np.ndarray,
    cases: list[tuple[str, np.ndarray, list[dict[str, object]]]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_name("plot_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return

    half = np.array(obj.size_m, dtype=float) / 2.0
    group_color = {
        "thumb": "#e45756",
        "index": "#4c78a8",
        "middle": "#72b7b2",
        "ring": "#54a24b",
        "pinky": "#b279a2",
        "palm": "#f58518",
    }

    def draw_box(ax) -> None:
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(
                        (
                            aabb_center
                            + np.array([sx * half[0], sy * half[1], sz * half[2]])
                        )
                        * 1000.0
                    )
        edges = (
            (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
        )
        for a, b in edges:
            pts = np.vstack([corners[a], corners[b]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#222222", linewidth=1.2, alpha=0.65)

    def draw_capsules(ax) -> None:
        for capsule in capsules_final:
            pts = np.vstack([capsule.p0, capsule.p1]) * 1000.0
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=group_color.get(capsule.group, "#777777"),
                linewidth=max(2.0, capsule.radius * 650.0),
                alpha=0.72,
                solid_capstyle="round",
            )

    n = len(cases)
    fig = plt.figure(figsize=(6.0 * n, 5.6))
    axes = []
    for i, (case_name, start, rows) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        axes.append(ax)
        draw_box(ax)
        draw_capsules(ax)
        positions = np.array([row["base_position"] for row in rows], dtype=float) * 1000.0
        colors = []
        for row in rows:
            if bool(row["ignored_for_final_contact"]):
                colors.append("#9a9a9a")
            elif bool(row["collision"]):
                colors.append("#d62728")
            else:
                colors.append("#2ca02c")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#333333", linewidth=1.4)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=22)
        ax.scatter(
            [start[0] * 1000.0],
            [start[1] * 1000.0],
            [start[2] * 1000.0],
            c="#1f77b4",
            s=45,
            label="start",
        )
        ax.scatter(
            [final[0] * 1000.0],
            [final[1] * 1000.0],
            [final[2] * 1000.0],
            c="#111111",
            s=45,
            label="final",
        )
        ax.set_title(f"{case_name}: {'blocked' if _collided(rows) else 'clear'}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.legend(loc="upper left", fontsize=8)

    all_points = [
        np.array(
            [
                aabb_center - half,
                aabb_center + half,
            ]
        )
        * 1000.0
    ]
    all_points.extend(np.vstack([capsule.p0, capsule.p1]) * 1000.0 for capsule in capsules_final)
    for _case_name, start, rows in cases:
        all_points.append(np.array([start, final]) * 1000.0)
        all_points.append(np.array([row["base_position"] for row in rows], dtype=float) * 1000.0)
    stacked = np.vstack(all_points)
    center = stacked.mean(axis=0)
    radius = max(float(np.ptp(stacked[:, axis])) for axis in range(3)) / 2.0
    radius = max(radius, 80.0)
    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=24, azim=-52)

    fig.suptitle(f"Capsule hand proxy verification: {obj.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_viewer(
    *,
    xml_path: str,
    ctrl_values: dict[str, float],
    obj: ObjectSpec,
    aabb_center: np.ndarray,
    rotation: np.ndarray,
    final: np.ndarray,
    capsules_final: list[Capsule],
    cases: list[tuple[str, np.ndarray, list[dict[str, object]]]],
) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    set_closure_qpos(model, data, ctrl_values, base_position=final, base_rotation=rotation)
    mujoco.mj_forward(model, data)
    half = np.array(obj.size_m, dtype=float) / 2.0

    def add_capsule(scn, p0, p1, radius, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
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
            float(radius),
            np.asarray(p0, dtype=np.float64),
            np.asarray(p1, dtype=np.float64),
        )
        scn.ngeom += 1

    def add_sphere(scn, p, radius, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    def draw_overlay(scn) -> None:
        scn.ngeom = 0
        for capsule in capsules_final:
            rgba = (1.0, 0.45, 0.10, 0.42) if capsule.group == "palm" else (0.10, 0.55, 1.0, 0.50)
            add_capsule(scn, capsule.p0, capsule.p1, capsule.radius, rgba)

        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(
                        aabb_center
                        + np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float)
                    )
        edges = (
            (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
        )
        for a, b in edges:
            add_capsule(scn, corners[a], corners[b], 0.0015, (0.0, 0.0, 0.0, 0.7))

        for _case_name, _start, rows in cases:
            for row in rows:
                p = np.asarray(row["base_position"], dtype=float)
                if bool(row["ignored_for_final_contact"]):
                    rgba = (0.55, 0.55, 0.55, 0.9)
                elif bool(row["collision"]):
                    rgba = (1.0, 0.0, 0.0, 0.95)
                else:
                    rgba = (0.0, 0.8, 0.2, 0.85)
                add_sphere(scn, p, 0.005, rgba)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            draw_overlay(viewer.user_scn)
            viewer.sync()
            time.sleep(0.033)


def main() -> int:
    args = parse_args()
    if args.path_samples < 2:
        raise ValueError("--path-samples must be >= 2")
    if args.radius_scale <= 0.0:
        raise ValueError("--radius-scale must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    obj = BUILTIN_OBJECTS[args.object]
    mode = args.mode_override or obj.mode
    yaw_rad = math.radians(args.yaw_deg)

    fk = InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk) if args.xml else InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    result = solve_object_grasp(
        closure,
        obj,
        mode,
        args.object_width_offset_mm / 1000.0,
    )

    xml_path = args.xml or fk.xml_path
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ctrl_values = path_ctrl_values(result, args.path_hand_shape)
    set_closure_qpos(model, data, ctrl_values)

    capsules_base = build_capsule_proxy(model, data, radius_scale=args.radius_scale)
    rotation = closure_base_rotation(result, yaw_rad)
    half_extents = np.array(obj.size_m, dtype=float) / 2.0
    aabb_center = tabletop_object_center(obj)
    final = closure_base_position(result, yaw_rad, object_center=aabb_center)
    capsules_final = transform_capsules(capsules_base, rotation, final)
    final_ignore_m = args.final_contact_ignore_mm / 1000.0

    if args.start_mm is not None:
        start = np.array(args.start_mm, dtype=float) / 1000.0
        rows = sample_linear_path_collisions(
            capsules_base,
            start=start,
            final=final,
            rotation=rotation,
            half_extents=half_extents,
            aabb_center=aabb_center,
            path_samples=args.path_samples,
            final_ignore_m=final_ignore_m,
        )
        cases = [("manual", start, rows)]
    else:
        cases = choose_demo_starts(
            capsules_base,
            final=final,
            rotation=rotation,
            half_extents=half_extents,
            aabb_center=aabb_center,
            path_samples=args.path_samples,
            final_ignore_m=final_ignore_m,
            auto_grid_step_m=args.auto_grid_step_mm / 1000.0,
        )

    tip_error_mm = (
        _tip_endpoint_error_mm(result, capsules_base)
        if args.path_hand_shape == "closed"
        else None
    )
    write_capsules_csv(args.out / "capsules.csv", capsules_base, capsules_final)
    write_path_samples_csv(args.out / "path_samples.csv", cases)
    write_summary_csv(
        args.out / "summary.csv",
        obj=obj,
        mode=mode,
        yaw_deg=args.yaw_deg,
        final=final,
        tip_error_mm=tip_error_mm,
        cases=cases,
    )
    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        obj=obj,
        mode=mode,
        final=final,
        rotation=rotation,
        aabb_center=aabb_center,
        capsules_base=capsules_base,
    )
    if not args.no_plot:
        maybe_write_plot(
            args.out / "capsule_proxy_demo.png",
            obj=obj,
            aabb_center=aabb_center,
            capsules_final=capsules_final,
            final=final,
            cases=cases,
        )

    print("RH56 capsule proxy verification:")
    print("  simulation_only: true")
    print(f"  object: {obj.name}")
    print(f"  mode: {mode}")
    print(f"  path_hand_shape: {args.path_hand_shape}")
    if args.path_hand_shape == "open":
        print("  open_shape_note: fingers qpos 0, thumb_yaw max qpos (real raw 0)")
    print(f"  object_center_mm: {np.round(aabb_center * 1000.0, 1).tolist()}")
    print(f"  yaw_deg: {args.yaw_deg:.1f}")
    print(f"  capsule_count: {len(capsules_base)}")
    if tip_error_mm is not None:
        print(f"  max_active_tip_endpoint_error_mm: {tip_error_mm:.3f}")
    else:
        print("  max_active_tip_endpoint_error_mm: n/a for open hand")
    for case_name, _start, rows in cases:
        print(
            "  "
            f"{case_name}: collided={_collided(rows)} "
            f"min_clearance_mm={_min_clearance(rows) * 1000.0:.2f}"
        )
    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'capsule_proxy_demo.png'}")

    if args.viewer:
        run_viewer(
            xml_path=str(xml_path),
            ctrl_values=ctrl_values,
            obj=obj,
            aabb_center=aabb_center,
            rotation=rotation,
            final=final,
            capsules_final=capsules_final,
            cases=cases,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
