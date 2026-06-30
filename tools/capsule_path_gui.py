#!/usr/bin/env python3
"""Interactive MuJoCo GUI for RH56 capsule path validity checks.

The hand starts away from the object.  Keyboard controls move the hand-base
start pose.  The mesh updates immediately, while the swept capsule validity
check runs only when requested unless ``--auto-check`` is enabled.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from rh56_controller.capsule_hand_proxy import (
    Capsule,
    build_capsule_proxy,
    closure_base_position,
    closure_base_rotation,
    closest_capsule_to_aabb,
    sample_linear_path_collisions,
    set_closure_qpos,
    transform_capsules,
)
from rh56_controller.grasp_geometry import CTRL_MAX, ClosureGeometry, ClosureResult, InspireHandFK
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS, ObjectSpec


@dataclass
class PathStatus:
    rows: list[dict[str, object]]
    path_valid: bool
    min_clearance_m: float
    nearest_capsule: str
    nearest_group: str
    nearest_alpha: float
    current_clearance_m: float
    final_clearance_m: float
    raw_collision_count_max: int
    ignored_collision_count_max: int
    active_collision_count_max: int


@dataclass
class GuiState:
    start: np.ndarray
    current: np.ndarray
    yaw_rad: float
    step_m: float
    running: bool = True
    animating: bool = False
    anim_alpha: float = 0.0
    status_dirty: bool = True
    status: PathStatus | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the RH56 hand in MuJoCo and preview capsule path validity."
    )
    parser.add_argument("--object", choices=sorted(BUILTIN_OBJECTS), default="debug_40mm_cube")
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument("--mode-override", choices=["line", "plane3", "plane4", "plane5"], default=None)
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--start-mm", type=float, nargs=3, default=None)
    parser.add_argument("--step-mm", type=float, default=10.0, help="Keyboard nudge size.")
    parser.add_argument("--path-samples", type=int, default=28)
    parser.add_argument(
        "--auto-check",
        action="store_true",
        help="Continuously recompute swept-capsule validity after pose changes.",
    )
    parser.add_argument(
        "--final-contact-ignore-mm",
        type=float,
        default=0.0,
        help="Ignore collisions this close to the analytical target pose.",
    )
    parser.add_argument(
        "--final-ignore-groups",
        choices=["fingers", "all", "none"],
        default="fingers",
        help="Capsule groups allowed to be ignored near the analytical target.",
    )
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--path-hand-shape",
        choices=["closed", "open"],
        default="open",
        help=(
            "Hand shape used by the capsule path check and moving mesh. "
            "'open' keeps fingers at qpos 0 and sets thumb_yaw to max qpos "
            "(real raw 0, fully rotated)."
        ),
    )
    parser.add_argument(
        "--animation-sec",
        type=float,
        default=2.0,
        help="Duration of the valid-path playback.",
    )
    parser.add_argument(
        "--print-initial-and-exit",
        action="store_true",
        help="Compute the initial status without opening a GUI.",
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


def final_ignore_groups(mode: str) -> tuple[str, ...] | None:
    if mode == "all":
        return None
    if mode == "none":
        return ()
    if mode == "fingers":
        return ("thumb", "index", "middle", "ring", "pinky")
    raise ValueError(f"Unsupported final-ignore-groups mode: {mode}")


def compute_status(
    capsules_base: list[Capsule],
    *,
    start: np.ndarray,
    final: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    path_samples: int,
    final_ignore_m: float,
    final_ignore_groups_mode: str,
) -> PathStatus:
    rows = sample_linear_path_collisions(
        capsules_base,
        start=start,
        final=final,
        rotation=rotation,
        half_extents=half_extents,
        aabb_center=aabb_center,
        path_samples=path_samples,
        final_ignore_m=final_ignore_m,
        final_ignore_groups=final_ignore_groups(final_ignore_groups_mode),
    )
    usable = [row for row in rows if not bool(row["ignored_for_final_contact"])]
    if not usable:
        usable = rows
    min_row = min(usable, key=lambda row: float(row["clearance_m"]))
    colliding = [row for row in rows if bool(row["collision"])]
    first_blocker = colliding[0] if colliding else min_row

    current_capsules = transform_capsules(capsules_base, rotation, start)
    final_capsules = transform_capsules(capsules_base, rotation, final)
    current_clearance = closest_capsule_to_aabb(
        current_capsules,
        half_extents,
        aabb_center=aabb_center,
    ).clearance
    final_clearance = closest_capsule_to_aabb(
        final_capsules,
        half_extents,
        aabb_center=aabb_center,
    ).clearance

    return PathStatus(
        rows=rows,
        path_valid=not colliding,
        min_clearance_m=float(min_row["clearance_m"]),
        nearest_capsule=str(first_blocker["nearest_capsule"]),
        nearest_group=str(first_blocker["nearest_group"]),
        nearest_alpha=float(first_blocker["alpha"]),
        current_clearance_m=float(current_clearance),
        final_clearance_m=float(final_clearance),
        raw_collision_count_max=max(int(row["raw_collision_count"]) for row in rows),
        ignored_collision_count_max=max(int(row["ignored_collision_count"]) for row in rows),
        active_collision_count_max=max(int(row["active_collision_count"]) for row in rows),
    )


def choose_initial_start(
    capsules_base: list[Capsule],
    *,
    requested_start: np.ndarray | None,
    final: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    path_samples: int,
    final_ignore_m: float,
    final_ignore_groups_mode: str,
) -> np.ndarray:
    if requested_start is not None:
        return requested_start

    offsets = (
        np.array([0.20, 0.00, 0.00]),
        np.array([0.00, 0.20, 0.00]),
        np.array([0.00, 0.00, 0.20]),
        np.array([0.20, 0.00, 0.12]),
        np.array([0.00, 0.20, 0.12]),
        np.array([-0.20, 0.00, 0.12]),
        np.array([0.00, -0.20, 0.12]),
    )
    fallback = final + offsets[0]
    for offset in offsets:
        candidate = final + offset
        status = compute_status(
            capsules_base,
            start=candidate,
            final=final,
            rotation=rotation,
            half_extents=half_extents,
            aabb_center=aabb_center,
            path_samples=path_samples,
            final_ignore_m=final_ignore_m,
            final_ignore_groups_mode=final_ignore_groups_mode,
        )
        if status.current_clearance_m > 0.0 and status.path_valid:
            return candidate
    for offset in offsets:
        candidate = final + offset
        status = compute_status(
            capsules_base,
            start=candidate,
            final=final,
            rotation=rotation,
            half_extents=half_extents,
            aabb_center=aabb_center,
            path_samples=path_samples,
            final_ignore_m=final_ignore_m,
            final_ignore_groups_mode=final_ignore_groups_mode,
        )
        if status.current_clearance_m > 0.0:
            return candidate
    return fallback


def status_line(state: GuiState, final: np.ndarray) -> str:
    pose_text = (
        f"start_mm={np.round(state.start * 1000.0, 1).tolist()} "
        f"target_mm={np.round(final * 1000.0, 1).tolist()} "
        f"yaw={math.degrees(state.yaw_rad):.1f} deg "
        f"step={state.step_m * 1000.0:.1f} mm"
    )
    if state.status is None:
        return f"status=not_checked {pose_text}"
    stale_prefix = "status=stale " if state.status_dirty else "status=checked "
    valid_label = "last_valid" if state.status_dirty else "valid"
    return (
        f"{stale_prefix}"
        f"{valid_label}={state.status.path_valid} "
        f"min_clearance={state.status.min_clearance_m * 1000.0:.1f} mm "
        f"current={state.status.current_clearance_m * 1000.0:.1f} mm "
        f"final={state.status.final_clearance_m * 1000.0:.1f} mm "
        f"nearest={state.status.nearest_group}/{state.status.nearest_capsule}"
        f"@{state.status.nearest_alpha:.2f} "
        f"collisions(raw/ignored/active)="
        f"{state.status.raw_collision_count_max}/"
        f"{state.status.ignored_collision_count_max}/"
        f"{state.status.active_collision_count_max} "
        "check=swept_capsules "
        f"{pose_text}"
    )


def print_controls() -> None:
    print("Controls:")
    print("  W/S: move x + / -")
    print("  A/D: move y + / -")
    print("  R/F: move z + / -")
    print("  J/L: yaw target - / +")
    print("  =/-: increase/decrease step")
    print("  Enter or V: validate current path")
    print("  Space: animate latest checked valid path")
    print("  C: reset hand to start")
    print("  Q or Esc: quit")


def add_capsule_geom(scn, p0, p1, radius, rgba) -> None:
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


def add_sphere_geom(scn, p, radius, rgba) -> None:
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


def draw_box(scn, half_extents: np.ndarray, center: np.ndarray) -> None:
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(
                    center
                    + np.array(
                        [
                            sx * half_extents[0],
                            sy * half_extents[1],
                            sz * half_extents[2],
                        ],
                        dtype=float,
                    )
                )
    edges = (
        (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
    )
    for a, b in edges:
        add_capsule_geom(scn, corners[a], corners[b], 0.0015, (0.0, 0.0, 0.0, 0.7))


def draw_path(scn, rows: list[dict[str, object]]) -> None:
    points = [np.asarray(row["base_position"], dtype=float) for row in rows]
    for idx, row in enumerate(rows):
        if bool(row["ignored_for_final_contact"]):
            rgba = (0.55, 0.55, 0.55, 0.85)
        elif bool(row["collision"]):
            rgba = (1.0, 0.05, 0.02, 0.95)
        else:
            rgba = (0.05, 0.8, 0.22, 0.85)
        add_sphere_geom(scn, points[idx], 0.005, rgba)
        if idx > 0:
            add_capsule_geom(scn, points[idx - 1], points[idx], 0.0015, rgba)


def draw_capsule_proxy(
    scn,
    capsules: list[Capsule],
    *,
    valid: bool,
    alpha: float,
    target: bool = False,
    stale: bool = False,
) -> None:
    if target:
        color = (0.08, 0.08, 0.08, alpha)
    elif stale:
        color = (1.0, 0.72, 0.10, alpha)
    elif valid:
        color = (0.05, 0.45, 1.0, alpha)
    else:
        color = (1.0, 0.12, 0.04, alpha)
    if target:
        palm_color = (0.25, 0.25, 0.25, alpha)
    elif stale:
        palm_color = (1.0, 0.72, 0.10, alpha)
    else:
        palm_color = (1.0, 0.55, 0.12, alpha)
    for capsule in capsules:
        rgba = palm_color if capsule.group == "palm" else color
        add_capsule_geom(scn, capsule.p0, capsule.p1, capsule.radius, rgba)


def draw_overlay(
    scn,
    *,
    state: GuiState,
    capsules_base: list[Capsule],
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    final: np.ndarray,
    rotation: np.ndarray,
) -> None:
    scn.ngeom = 0
    stale = bool(state.status_dirty)
    valid = bool(state.status and state.status.path_valid and not stale)
    draw_box(scn, half_extents, aabb_center)
    if state.status is not None and not stale:
        draw_path(scn, state.status.rows)

    current_capsules = transform_capsules(capsules_base, rotation, state.current)
    final_capsules = transform_capsules(capsules_base, rotation, final)
    draw_capsule_proxy(scn, final_capsules, valid=valid, alpha=0.20, target=True)
    draw_capsule_proxy(scn, current_capsules, valid=valid, alpha=0.38, stale=stale)

    add_sphere_geom(scn, state.start, 0.007, (0.0, 0.25, 1.0, 0.95))
    add_sphere_geom(scn, final, 0.007, (0.0, 0.0, 0.0, 0.95))
    if stale:
        status_color = (1.0, 0.72, 0.10, 0.95)
    elif valid:
        status_color = (0.0, 0.85, 0.22, 0.95)
    else:
        status_color = (1.0, 0.05, 0.02, 0.95)
    add_sphere_geom(
        scn,
        aabb_center + np.array([0.0, 0.0, half_extents[2] + 0.045]),
        0.010,
        status_color,
    )


def run_gui(
    *,
    args: argparse.Namespace,
    xml_path: str,
    result: ClosureResult,
    obj: ObjectSpec,
    capsules_base: list[Capsule],
    state: GuiState,
) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    half_extents = np.array(obj.size_m, dtype=float) / 2.0
    aabb_center = tabletop_object_center(obj)
    final_ignore_m = args.final_contact_ignore_mm / 1000.0
    last_print = 0.0

    def current_geometry() -> tuple[np.ndarray, np.ndarray]:
        rotation = closure_base_rotation(result, state.yaw_rad)
        final = closure_base_position(result, state.yaw_rad, object_center=aabb_center)
        return rotation, final

    def recompute_status() -> tuple[np.ndarray, np.ndarray]:
        rotation, final = current_geometry()
        state.status = compute_status(
            capsules_base,
            start=state.start,
            final=final,
            rotation=rotation,
            half_extents=half_extents,
            aabb_center=aabb_center,
            path_samples=args.path_samples,
            final_ignore_m=final_ignore_m,
            final_ignore_groups_mode=args.final_ignore_groups,
        )
        state.status_dirty = False
        return rotation, final

    rotation, final = recompute_status()

    def nudge(delta: np.ndarray) -> None:
        state.start = state.start + delta
        state.current = state.start.copy()
        state.animating = False
        state.status_dirty = True

    def key_callback(key):
        glfw = mujoco.glfw.glfw
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            state.running = False
        elif key == glfw.KEY_W:
            nudge(np.array([state.step_m, 0.0, 0.0]))
        elif key == glfw.KEY_S:
            nudge(np.array([-state.step_m, 0.0, 0.0]))
        elif key == glfw.KEY_A:
            nudge(np.array([0.0, state.step_m, 0.0]))
        elif key == glfw.KEY_D:
            nudge(np.array([0.0, -state.step_m, 0.0]))
        elif key == glfw.KEY_R:
            nudge(np.array([0.0, 0.0, state.step_m]))
        elif key == glfw.KEY_F:
            nudge(np.array([0.0, 0.0, -state.step_m]))
        elif key == glfw.KEY_J:
            state.yaw_rad -= math.radians(5.0)
            state.animating = False
            state.status_dirty = True
        elif key == glfw.KEY_L:
            state.yaw_rad += math.radians(5.0)
            state.animating = False
            state.status_dirty = True
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state.step_m = min(state.step_m * 1.5, 0.1)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state.step_m = max(state.step_m / 1.5, 0.001)
        elif key == glfw.KEY_C:
            state.current = state.start.copy()
            state.animating = False
        elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER, glfw.KEY_V):
            rotation, final = recompute_status()
            print(status_line(state, final))
        elif key == glfw.KEY_SPACE:
            if state.status_dirty and not args.auto_check:
                print("Path status is stale; press Enter or V to validate this pose first.")
                return
            if state.status_dirty:
                recompute_status()
            if state.status and state.status.path_valid:
                state.animating = True
                state.anim_alpha = 0.0
                state.current = state.start.copy()
                print("Animating valid path.")
            else:
                print("Path is blocked; move the start pose or increase final-contact-ignore.")

    print_controls()
    print(status_line(state, final))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.55
        viewer.cam.lookat[:] = [0.06, 0.0, 0.10]

        while viewer.is_running() and state.running:
            rotation, final = current_geometry()
            if state.status_dirty and args.auto_check:
                rotation, final = recompute_status()
                print(status_line(state, final))
                last_print = time.time()

            if state.animating:
                dt = 1.0 / 60.0
                state.anim_alpha = min(1.0, state.anim_alpha + dt / max(args.animation_sec, 1e-6))
                smooth = state.anim_alpha * state.anim_alpha * (3.0 - 2.0 * state.anim_alpha)
                state.current = state.start + smooth * (final - state.start)
                if state.anim_alpha >= 1.0:
                    state.animating = False
                    state.current = final.copy()
                    print("Animation reached target.")

            set_closure_qpos(
                model,
                data,
                path_ctrl_values(result, args.path_hand_shape),
                base_position=state.current,
                base_rotation=rotation,
            )
            mujoco.mj_forward(model, data)
            draw_overlay(
                viewer.user_scn,
                state=state,
                capsules_base=capsules_base,
                half_extents=half_extents,
                aabb_center=aabb_center,
                final=final,
                rotation=rotation,
            )
            viewer.sync()

            # Periodically remind the terminal of the state while the user is
            # moving with repeated keypresses, without flooding every frame.
            if time.time() - last_print > 5.0 and state.status is not None:
                print(status_line(state, final))
                last_print = time.time()
            time.sleep(1.0 / 60.0)


def main() -> int:
    args = parse_args()
    if args.step_mm <= 0.0:
        raise ValueError("--step-mm must be positive")
    if args.path_samples < 2:
        raise ValueError("--path-samples must be >= 2")
    if args.radius_scale <= 0.0:
        raise ValueError("--radius-scale must be positive")

    obj = BUILTIN_OBJECTS[args.object]
    mode = args.mode_override or obj.mode
    fk = InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk) if args.xml else InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    result = solve_object_grasp(
        closure,
        obj,
        mode,
        args.object_width_offset_mm / 1000.0,
    )
    xml_path = str(args.xml or fk.xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ctrl_values = path_ctrl_values(result, args.path_hand_shape)
    set_closure_qpos(model, data, ctrl_values)
    capsules_base = build_capsule_proxy(model, data, radius_scale=args.radius_scale)

    yaw_rad = math.radians(args.yaw_deg)
    rotation = closure_base_rotation(result, yaw_rad)
    half_extents = np.array(obj.size_m, dtype=float) / 2.0
    aabb_center = tabletop_object_center(obj)
    final = closure_base_position(result, yaw_rad, object_center=aabb_center)
    requested_start = np.array(args.start_mm, dtype=float) / 1000.0 if args.start_mm is not None else None
    start = choose_initial_start(
        capsules_base,
        requested_start=requested_start,
        final=final,
        rotation=rotation,
        half_extents=half_extents,
        aabb_center=aabb_center,
        path_samples=args.path_samples,
        final_ignore_m=args.final_contact_ignore_mm / 1000.0,
        final_ignore_groups_mode=args.final_ignore_groups,
    )
    state = GuiState(
        start=start,
        current=start.copy(),
        yaw_rad=yaw_rad,
        step_m=args.step_mm / 1000.0,
    )
    state.status = compute_status(
        capsules_base,
        start=state.start,
        final=final,
        rotation=rotation,
        half_extents=half_extents,
        aabb_center=aabb_center,
        path_samples=args.path_samples,
        final_ignore_m=args.final_contact_ignore_mm / 1000.0,
        final_ignore_groups_mode=args.final_ignore_groups,
    )
    state.status_dirty = False

    print("RH56 capsule path GUI:")
    print(f"  object: {obj.name}")
    print(f"  mode: {mode}")
    print(f"  path_hand_shape: {args.path_hand_shape}")
    if args.path_hand_shape == "open":
        print("  open_shape_note: fingers qpos 0, thumb_yaw max qpos (real raw 0)")
    print(f"  object_center_mm: {np.round(aabb_center * 1000.0, 1).tolist()}")
    print(f"  check_mode: {'auto' if args.auto_check else 'manual'}")
    print(f"  final_contact_ignore_mm: {args.final_contact_ignore_mm:.1f}")
    print(f"  final_ignore_groups: {args.final_ignore_groups}")
    print(status_line(state, final))

    if args.print_initial_and_exit:
        return 0

    run_gui(
        args=args,
        xml_path=xml_path,
        result=result,
        obj=obj,
        capsules_base=capsules_base,
        state=state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
