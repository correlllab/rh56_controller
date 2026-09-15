#!/usr/bin/env python3
"""Side-by-side video of the strategies executing the same approach condition.

The admissibility numbers say which strategy survives which condition. They do
not show *why*, and the difference is a geometric event that takes a few hundred
milliseconds: the moment a swept finger crosses the support plane, or a wrist
held at the wrong pose drags the contacts off the object.

This renders the full commanded motion -- free-space approach, then closure --
for every strategy from one shared start pose, tiles them into one frame, and
encodes the result. Each panel carries the running support-surface and object
clearances, and turns red at the first sampled configuration that violates a
criterion, so the failure is visible at the instant it happens rather than
inferred from a bar chart.

Usage:
    MUJOCO_GL=egl uv run python render_approach_video.py --object paper_nut \\
        --condition P1_level --out ../figures/20260908-approach_P1_nut.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ggx_strategy import build_ggx_plans  # noqa: E402
from mj_snap_rh56 import VIEWS, add_object_geom, render, set_scene_state, tile  # noqa: E402
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
    simulate_execution,
)
from rh56_controller.capsule_hand_proxy import (  # noqa: E402
    capsule_collisions_against_object,
    transform_capsules,
)
from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
    tabletop_grasp_target,
)
from run_reachability_sweep import paper_conditions  # noqa: E402

LABELS = {
    "naive": "Naive closure",
    "iterative_closure": "Iterative closure",
    "thumb_reflex": "Thumb reflex",
    "graspgen_x": "GraspGen-X",
}
OK = (232, 244, 236)
BAD = (250, 224, 224)


def motion_legs(plan, start_base: np.ndarray):
    """(base, rotation, ctrl) at each end of every leg of the commanded motion."""

    first = plan.waypoints[0]
    legs = [(start_base, first.rotation, first.ctrl,
             first.base_position, first.rotation, first.ctrl, "approach")]
    for a, b in zip(plan.waypoints[:-1], plan.waypoints[1:]):
        legs.append((a.base_position, a.rotation, a.ctrl,
                     b.base_position, b.rotation, b.ctrl, "closure"))
    return legs


def sample_motion(plan, start_base: np.ndarray, n_frames: int):
    """Uniformly sample the whole motion so every strategy stays in step."""

    legs = motion_legs(plan, start_base)
    out = []
    for idx in range(n_frames):
        u = idx / max(n_frames - 1, 1) * len(legs)
        leg = min(int(u), len(legs) - 1)
        t = u - leg
        b0, r0, c0, b1, r1, c1, phase = legs[leg]
        base = b0 + t * (b1 - b0)
        ctrl = _ctrl_lerp(c0, c1, t)
        if np.allclose(r0, r1):
            rot = r0
        else:
            u_, _, vt = np.linalg.svd(r0 + t * (r1 - r0))
            rot = u_ @ vt
        out.append((base, rot, ctrl, phase))
    return out


def clearances(sampler, ctrl, base, rot, obj, exempt):
    capsules = transform_capsules(sampler.capsules(ctrl), rot, base)
    floor = min(min(float(c.p0[2]), float(c.p1[2])) - c.radius for c in capsules)
    records = capsule_collisions_against_object(
        capsules, np.asarray(obj.size_m, dtype=float) / 2.0,
        object_center=tabletop_aabb_center(obj), object_shape=obj.collision_shape)
    non_contact = [r for r in records if r.capsule_name not in exempt] or records
    return floor, min(r.clearance for r in non_contact)


def banner(img: Image.Image, title: str, lines: list[str], bad: bool) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 46), BAD if bad else OK)
    out.paste(img, (0, 46))
    draw = ImageDraw.Draw(out)
    draw.text((10, 6), title, fill=(20, 20, 20))
    draw.text((10, 20), lines[0], fill=(70, 70, 70))
    if len(lines) > 1:
        draw.text((10, 32), lines[1], fill=(160, 40, 40) if bad else (70, 70, 70))
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", default="paper_nut", choices=sorted(BUILTIN_OBJECTS))
    ap.add_argument("--condition", default="P1_level")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ggx-candidates", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--view", default="side")
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--distance", type=float, default=0.62)
    ap.add_argument("--lookat-lift-mm", type=float, default=40.0,
                    help="Raise the camera target above the grasp point.")
    ap.add_argument("--contact-policy", default="contact-fingers",
                    choices=["tips-only", "contact-fingers"])
    ap.add_argument("--floor-tolerance-mm", type=float, default=3.0)
    ap.add_argument("--contact-tolerance-mm", type=float, default=2.0)
    args = ap.parse_args()

    obj = BUILTIN_OBJECTS[args.object]
    grasp_target = tabletop_grasp_target(obj)
    yaw = grasp_yaw_for_object(obj)
    approach_dir, lateral_dir = approach_basis(yaw)

    conditions = {c["condition"]: c for c in paper_conditions()}
    if args.condition not in conditions:
        raise SystemExit(f"unknown condition {args.condition!r}; "
                         f"choose from {sorted(conditions)}")
    cond = conditions[args.condition]
    start_base = (
        grasp_target
        - approach_dir * (float(cond["standoff_mm"]) / 1000.0)
        + lateral_dir * (float(cond["lateral_mm"]) / 1000.0)
        + np.array([0.0, 0.0, float(cond["height_mm"]) / 1000.0])
    )

    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    sampler = HandSampler(fk.xml_path)
    plans = build_strategy_plans(
        fk, closure, mode=obj.mode,
        target_width_m=obj.grasp_width_m + 0.020,
        grasp_target=grasp_target, yaw_rad=yaw,
    )
    order = list(STRATEGIES)
    if args.ggx_candidates is not None:
        path = args.ggx_candidates / f"{args.object}.yml"
        if path.exists():
            # Show the candidate GraspGen-X would actually execute: the highest
            # confidence one, not the best of the list after a collision check.
            plans["graspgen_x"] = build_ggx_plans(
                fk, path, object_center=tabletop_aabb_center(obj),
                grasp_target=grasp_target, top_k=1)[0]
            order.append("graspgen_x")

    spec = mujoco.MjSpec.from_file(str(fk.xml_path))
    add_object_geom(spec, obj)
    spec.visual.global_.offwidth = max(args.width, 640)
    spec.visual.global_.offheight = max(args.height, 480)
    for geom in spec.worldbody.geoms:
        if geom.name == "floor":
            geom.material = ""
            geom.rgba = np.array([0.90, 0.90, 0.88, 1.0])
    model = spec.compile()
    data = mujoco.MjData(model)

    verdicts = {}
    for name in order:
        result = simulate_execution(
            plans[name], sampler, start_base=start_base, obj=obj,
            contact_policy=args.contact_policy,
            floor_tolerance_m=args.floor_tolerance_mm / 1000.0,
            contact_tolerance_m=args.contact_tolerance_mm / 1000.0)
        verdicts[name] = result
        print(f"  {LABELS[name]:20s} admissible={result.admissible!s:5s} {result.failures}")

    tracks = {name: sample_motion(plans[name], start_base, args.frames) for name in order}
    exempt = {name: _tip_capsule_names(plans[name].active) for name in order}
    # Centre on the hand rather than the object: the wrist sits well above the
    # grasp point, so aiming at the object leaves the action in the top third.
    lookat = grasp_target + np.array([0.0, 0.0, args.lookat_lift_mm / 1000.0])

    if args.azimuth is not None or args.elevation is not None:
        base_az, base_el = VIEWS[args.view]
        VIEWS["_custom"] = (base_az if args.azimuth is None else args.azimuth,
                            base_el if args.elevation is None else args.elevation)
        args.view = "_custom"

    tmp = Path(tempfile.mkdtemp(prefix="approach_video_"))
    floor_tol = -args.floor_tolerance_mm / 1000.0
    contact_tol = -args.contact_tolerance_mm / 1000.0
    failed_at = {name: None for name in order}

    for frame in range(args.frames):
        panels = []
        for name in order:
            base, rot, ctrl, phase = tracks[name][frame]
            set_scene_state(model, data, ctrl, base, rot)
            floor, objclr = clearances(sampler, ctrl, base, rot, obj, exempt[name])
            bad = floor < floor_tol or objclr < contact_tol
            if bad and failed_at[name] is None:
                failed_at[name] = frame
            latched = failed_at[name] is not None
            img = render(model, data, args.view, lookat, args.distance,
                         args.width, args.height)
            note = f"{phase}   table {floor * 1000:+.0f} mm   object {objclr * 1000:+.0f} mm"
            second = ""
            if latched:
                modes = ", ".join(m.replace("_", " ") for m in verdicts[name].failures)
                second = f"collision from frame {failed_at[name]}  ({modes})"
            panels.append(banner(img, LABELS[name], [note, second], latched))
        tile(panels, len(order)).save(tmp / f"f{frame:04d}.png")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(tmp / "f%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", "20", str(args.out),
    ]
    subprocess.run(cmd, check=True)
    for png in tmp.glob("*.png"):
        png.unlink()
    tmp.rmdir()
    print(f"wrote {args.out}  ({args.frames} frames at {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
