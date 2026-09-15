#!/usr/bin/env python3
"""Render an RH56 closure-strategy configuration and report its clearances.

The RH56 analogue of the `mujoco-eyes` loop used in the hand repository: look at
the scene instead of inferring it from numbers, and print the numbers that pair
with the picture so the two can be checked against each other.

The strategy geometry is intentionally hard to reason about numerically -- a
"table collision" is a signed clearance on one of thirteen capsules attached to a
hand whose wrist pose depends on the commanded width -- so every claim in the
write-up should be confirmed against a render before it is believed.

Examples
--------
Terminal configuration of all three strategies on the nut, side view::

    MUJOCO_GL=egl uv run python mj_snap_rh56.py --object paper_nut \\
        --strategy all --stage terminal --views side,front,iso --out /tmp/nut.png

Walk the naive closure sweep that drives the fingers into the table::

    MUJOCO_GL=egl uv run python mj_snap_rh56.py --object paper_nut \\
        --strategy naive --filmstrip 6 --out /tmp/naive_sweep.png

Read the PNG afterwards.  Tiles are labelled with strategy, stage, and the
minimum floor/object clearance at that configuration.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_admissibility import (  # noqa: E402
    STRATEGIES,
    ClosureGeometry,
    HandSampler,
    InspireHandFK,
    _ctrl_lerp,
    _tip_capsule_names,
    approach_basis,
    build_strategy_plans,
    grasp_yaw_for_object,
)
from rh56_controller.capsule_hand_proxy import (  # noqa: E402
    capsule_collisions_against_object,
    rotation_to_grasp_scene_euler_xyz,
    transform_capsules,
)
from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
    tabletop_grasp_target,
)

# azimuth, elevation
VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0, -10.0),
    "side": (90.0, -10.0),
    "top": (90.0, -89.0),
    "iso": (45.0, -25.0),
    "iso2": (135.0, -25.0),
    "low": (75.0, 3.0),
}


def add_object_geom(spec: mujoco.MjSpec, obj) -> None:
    """Add the object proxy used by the admissibility model to the scene."""

    center = tabletop_aabb_center(obj)
    half = np.asarray(obj.size_m, dtype=float) / 2.0
    body = spec.worldbody.add_body(name="object_proxy", pos=center)
    if obj.collision_shape == "box":
        gtype, size = mujoco.mjtGeom.mjGEOM_BOX, half
    elif obj.collision_shape == "cylinder":
        gtype, size = mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([half[0], half[2], 0.0])
    elif obj.collision_shape == "sphere":
        gtype, size = mujoco.mjtGeom.mjGEOM_SPHERE, np.array([half[0], 0.0, 0.0])
    else:
        raise ValueError(f"Unsupported collision shape: {obj.collision_shape}")
    geom = body.add_geom(name="object_proxy_geom", type=gtype, size=size)
    geom.rgba = np.array([0.85, 0.45, 0.15, 0.55])
    geom.contype = 0
    geom.conaffinity = 0


def set_scene_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: dict[str, float],
    base_position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    def set_joint(name: str, value: float) -> None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)

    data.qpos[:] = 0.0
    set_joint("right_pos_x", base_position[0])
    set_joint("right_pos_y", base_position[1])
    set_joint("right_pos_z", base_position[2])
    rx, ry, rz = rotation_to_grasp_scene_euler_xyz(rotation)
    set_joint("right_rot_x", rx)
    set_joint("right_rot_y", ry)
    set_joint("right_rot_z", rz)

    pinky = float(ctrl.get("pinky", 0.0))
    ring = float(ctrl.get("ring", 0.0))
    middle = float(ctrl.get("middle", 0.0))
    index = float(ctrl.get("index", 0.0))
    pitch = float(ctrl.get("thumb_proximal", 0.0))
    yaw = float(ctrl.get("thumb_yaw", 0.0))
    for name, value in (
        ("pinky_proximal_joint", pinky),
        ("pinky_intermediate_joint", -0.15 + 1.1169 * pinky),
        ("ring_proximal_joint", ring),
        ("ring_intermediate_joint", -0.15 + 1.1169 * ring),
        ("middle_proximal_joint", middle),
        ("middle_intermediate_joint", -0.15 + 1.1169 * middle),
        ("index_proximal_joint", index),
        ("index_intermediate_joint", -0.05 + 1.1169 * index),
        ("thumb_proximal_yaw_joint", yaw),
        ("thumb_proximal_pitch_joint", pitch),
        ("thumb_intermediate_joint", 0.15 + 1.33 * pitch),
        ("thumb_distal_joint", 0.15 + 0.66 * pitch),
    ):
        set_joint(name, value)
    mujoco.mj_forward(model, data)


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    view: str,
    lookat: np.ndarray,
    distance: float,
    width: int,
    height: int,
) -> Image.Image:
    azimuth, elevation = VIEWS[view]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera)
        return Image.fromarray(renderer.render())


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    draw.rectangle([0, 0, img.width, 6 + 13 * len(lines)], fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((6, 3 + 13 * idx), line, fill=(255, 255, 255))
    return img


def tile(frames: list[Image.Image], cols: int) -> Image.Image:
    if not frames:
        raise ValueError("no frames to tile")
    cols = max(1, min(cols, len(frames)))
    rows = math.ceil(len(frames) / cols)
    w, h = frames[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (24, 24, 24))
    for idx, frame in enumerate(frames):
        sheet.paste(frame, ((idx % cols) * w, (idx // cols) * h))
    return sheet


def clearance_report(
    sampler: HandSampler,
    ctrl: dict[str, float],
    base: np.ndarray,
    rotation: np.ndarray,
    obj,
    exempt: set[str],
) -> tuple[float, str, float, str]:
    capsules = transform_capsules(sampler.capsules(ctrl), rotation, base)
    floor_clear, floor_name = math.inf, ""
    for capsule in capsules:
        value = min(float(capsule.p0[2]), float(capsule.p1[2])) - capsule.radius
        if value < floor_clear:
            floor_clear, floor_name = value, f"{capsule.group}/{capsule.name}"
    records = capsule_collisions_against_object(
        capsules,
        np.asarray(obj.size_m, dtype=float) / 2.0,
        object_center=tabletop_aabb_center(obj),
        object_shape=obj.collision_shape,
    )
    non_contact = [r for r in records if r.capsule_name not in exempt] or records
    worst = min(non_contact, key=lambda r: r.clearance)
    return floor_clear, floor_name, float(worst.clearance), f"{worst.capsule_group}/{worst.capsule_name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", default="paper_nut", choices=sorted(BUILTIN_OBJECTS))
    ap.add_argument("--strategy", default="all", choices=list(STRATEGIES) + ["all"])
    ap.add_argument("--stage", default="terminal", choices=["start", "pregrasp", "terminal"])
    ap.add_argument("--filmstrip", type=int, default=0, help="Frames across the whole motion.")
    ap.add_argument("--views", default="side,iso")
    ap.add_argument("--object-width-offset-mm", type=float, default=20.0)
    ap.add_argument("--naive-pose-policy", default="open-pose", choices=["open-pose", "final-orientation"])
    ap.add_argument("--standoff-mm", type=float, default=250.0)
    ap.add_argument("--lateral-mm", type=float, default=0.0)
    ap.add_argument("--height-mm", type=float, default=250.0)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--distance", type=float, default=0.60)
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    obj = BUILTIN_OBJECTS[args.object]
    grasp_target = tabletop_grasp_target(obj)
    yaw = grasp_yaw_for_object(obj)
    approach_dir, lateral_dir = approach_basis(yaw)

    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    sampler = HandSampler(fk.xml_path)
    plans = build_strategy_plans(
        fk,
        closure,
        mode=obj.mode,
        target_width_m=obj.grasp_width_m + args.object_width_offset_mm / 1000.0,
        grasp_target=grasp_target,
        yaw_rad=yaw,
        naive_pose_policy=args.naive_pose_policy,
    )

    spec = mujoco.MjSpec.from_file(str(fk.xml_path))
    add_object_geom(spec, obj)
    # The scene's default offscreen buffer is 640x480; renders larger than
    # that come back blank unless the framebuffer is grown first.
    spec.visual.global_.offwidth = max(args.width, 640)
    spec.visual.global_.offheight = max(args.height, 480)
    model = spec.compile()
    data = mujoco.MjData(model)

    start_base = (
        grasp_target
        - approach_dir * (args.standoff_mm / 1000.0)
        + lateral_dir * (args.lateral_mm / 1000.0)
        + np.array([0.0, 0.0, args.height_mm / 1000.0])
    )

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for view in views:
        if view not in VIEWS:
            raise SystemExit(f"unknown view {view!r}; choose from {sorted(VIEWS)}")

    frames: list[Image.Image] = []
    print(f"object={args.object} mode={obj.mode} yaw={math.degrees(yaw):.0f}deg "
          f"grasp_target_z={grasp_target[2] * 1000:.0f}mm")

    for strategy in strategies:
        plan = plans[strategy]
        exempt = _tip_capsule_names(plan.active)
        if args.filmstrip > 0:
            # Whole motion: approach as leg 0, then one leg per commanded waypoint.
            legs: list[tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, dict]] = []
            first = plan.waypoints[0]
            legs.append((start_base, first.rotation, first.ctrl,
                         first.base_position, first.rotation, first.ctrl))
            for a, b in zip(plan.waypoints[:-1], plan.waypoints[1:]):
                legs.append((a.base_position, a.rotation, a.ctrl,
                             b.base_position, b.rotation, b.ctrl))
            stages = []
            for idx in range(args.filmstrip):
                u = idx / max(args.filmstrip - 1, 1) * len(legs)
                leg_idx = min(int(u), len(legs) - 1)
                t = u - leg_idx
                b0, r0, c0, b1, r1, c1 = legs[leg_idx]
                base = b0 + t * (b1 - b0)
                ctrl = _ctrl_lerp(c0, c1, t)
                if np.allclose(r0, r1):
                    rot = r0
                else:
                    u_, _, vt = np.linalg.svd(r0 + t * (r1 - r0))
                    rot = u_ @ vt
                stages.append((f"{'approach' if leg_idx == 0 else 'closure'} {idx + 1}/{args.filmstrip}",
                               base, rot, ctrl))
        else:
            first = plan.waypoints[0]
            if args.stage == "start":
                stages = [("start", start_base, first.rotation, first.ctrl)]
            elif args.stage == "pregrasp":
                stages = [("pregrasp", first.base_position, first.rotation, first.ctrl)]
            else:
                stages = [("terminal", plan.terminal_base, plan.terminal_rotation, plan.terminal_ctrl)]

        for stage_name, base, rot, ctrl in stages:
            set_scene_state(model, data, ctrl, base, rot)
            floor_c, floor_n, obj_c, obj_n = clearance_report(sampler, ctrl, base, rot, obj, exempt)
            print(f"  {strategy:18s} {stage_name:22s} floor={floor_c * 1000:8.2f}mm ({floor_n:18s}) "
                  f"object={obj_c * 1000:8.2f}mm ({obj_n})")
            for view in views:
                img = render(model, data, view, grasp_target, args.distance, args.width, args.height)
                frames.append(
                    label(
                        img,
                        f"{strategy} | {stage_name} | {view}\n"
                        f"floor {floor_c * 1000:+.1f}mm  object {obj_c * 1000:+.1f}mm",
                    )
                )

    cols = args.cols or len(views)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tile(frames, cols).save(args.out)
    print(f"wrote {args.out}  ({len(frames)} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
