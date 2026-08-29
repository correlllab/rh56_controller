#!/usr/bin/env python3
"""Check whether ANY single straight-line approach reaches a power-wrap grasp.

tools/run_analytical_grasp_volume.py showed that for 'plane' multi-finger
power-wrap modes on box-like objects, a straight-line approach from an
arbitrary sampled direction usually drags the hand through the object partway
along the path, even though the terminal analytical grasp pose is legitimate
(palm-on-object is normal support for a wrap grasp).

This script tests every plausible single-segment straight-line approach
direction (straight in along the hand's own reach axis, sideways from four
directions, and straight down from above) against a range of standoff
distances, per object and yaw sample.

RESULT: for ycb_cracker_box, ycb_sugar_box, and ycb_potted_meat_can under
their respective 'plane' modes, ZERO of 8 yaw samples had a collision-free
straight-line approach in ANY of the 6 directions tested, at any standoff up
to 300mm (see docs/paper_v2_methods.md). This is not a search gap -- it is
evidence that a single straight-line Cartesian segment is structurally
insufficient for a power-wrap grasp: the fingers must end up spread around
the object, so any straight approach that already has them near that spread
configuration grazes the object before arriving, regardless of direction.

The practical resolution already used elsewhere in this repo's own scripts
(e.g. the V17 pregrasp logic: "PLACE POWER PREGRASP WITH COLLISION OFF") is
to place the open hand directly at the pregrasp configuration via joint-space
IK -- which is not constrained to a straight Cartesian line and can route
around the collision this script's straight-line search cannot avoid -- then
close the fingers with contact-aware force control. This script's value is as
a documented negative result motivating that choice, not as a path generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

from rh56_controller.capsule_hand_proxy import (
    build_capsule_proxy,
    closure_base_position,
    closure_base_rotation,
    sample_linear_path_collisions,
    set_closure_qpos,
)
from rh56_controller.grasp_geometry import CTRL_MAX, ClosureGeometry, ClosureResult, InspireHandFK
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS, ObjectSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether any single straight-line approach direction reaches "
            "the analytical power-wrap grasp pose collision-free. Documented "
            "result: none do, for the paper-facing objects/modes; see the "
            "module docstring and docs/paper_v2_methods.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["ycb_cracker_box", "ycb_sugar_box", "ycb_potted_meat_can"],
        choices=sorted(BUILTIN_OBJECTS),
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/staged_approach"))
    parser.add_argument("--xml", default=None)
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument("--mode-override", choices=["line", "plane3", "plane4", "plane5"], default=None)
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument("--yaw-samples", type=int, default=8)
    parser.add_argument(
        "--standoff-mm",
        type=float,
        nargs="+",
        default=[100.0, 150.0, 200.0, 250.0, 300.0],
        help="Candidate retreat distances along the hand's own approach axis (local +Z), tried in order.",
    )
    parser.add_argument("--path-samples", type=int, default=12)
    parser.add_argument("--hand-clearance-mm", type=float, default=10.0)
    parser.add_argument("--final-contact-ignore-mm", type=float, default=35.0)
    parser.add_argument(
        "--ignore-palm-near-final",
        choices=["auto", "always", "never"],
        default="auto",
        help="Same semantics as run_analytical_grasp_volume.py: 'auto' excuses palm contact near the final pose for 'plane' modes only.",
    )
    parser.add_argument("--path-hand-shape", choices=["open", "closed"], default="open")
    parser.add_argument(
        "--directions",
        nargs="+",
        default=[
            "reach_axis_local_+z",
            "side_local_+x",
            "side_local_-x",
            "side_local_+y",
            "side_local_-y",
            "top_down_world_+z",
        ],
        choices=[
            "reach_axis_local_+z",
            "side_local_+x",
            "side_local_-x",
            "side_local_+y",
            "side_local_-y",
            "top_down_world_+z",
        ],
        help="Candidate retreat directions, tried in the given order for each yaw.",
    )
    return parser.parse_args()


def path_ctrl_values(result: ClosureResult, shape: str) -> dict[str, float]:
    if shape == "closed":
        return dict(result.ctrl_values)
    values = {key: 0.0 for key in result.ctrl_values}
    values["thumb_yaw"] = CTRL_MAX["thumb_yaw"]
    return values


def solve_object_grasp(closure: ClosureGeometry, obj: ObjectSpec, mode: str, width_offset_m: float) -> ClosureResult:
    internal_width_m = obj.grasp_width_m + width_offset_m
    if mode == "line":
        return closure.line(internal_width_m)
    if mode.startswith("plane"):
        return closure.plane(internal_width_m, n_fingers=int(mode[-1]))
    raise ValueError(f"Unsupported mode: {mode}")


def tabletop_object_center(obj: ObjectSpec) -> np.ndarray:
    return np.array([0.0, 0.0, obj.size_m[2] / 2.0], dtype=float)


# Candidate retreat directions, tried in order. "reach_axis" (local +Z, the
# direction fingers/palm point) was the first thing tried and confirmed NOT
# to work for plane power-wrap modes: the whole reach envelope (fingers, then
# palm) sweeps through the object over the final ~130mm of that approach,
# regardless of finger posture, because a wrap grasp's terminal pose requires
# the object to sit inside the hand's own reach envelope by construction.
# The remaining directions test the real alternative: approach from the side
# (local +-X/+-Y, perpendicular to the reach axis) or straight down from
# above in world +Z, so the hand is already laterally positioned before it
# ever needs to move through the object's footprint.
RETREAT_DIRECTIONS: dict[str, str] = {
    "reach_axis_local_+z": "local",
    "side_local_+x": "local",
    "side_local_-x": "local",
    "side_local_+y": "local",
    "side_local_-y": "local",
    "top_down_world_+z": "world",
}

RETREAT_VECTORS: dict[str, np.ndarray] = {
    "reach_axis_local_+z": np.array([0.0, 0.0, 1.0]),
    "side_local_+x": np.array([1.0, 0.0, 0.0]),
    "side_local_-x": np.array([-1.0, 0.0, 0.0]),
    "side_local_+y": np.array([0.0, 1.0, 0.0]),
    "side_local_-y": np.array([0.0, -1.0, 0.0]),
    "top_down_world_+z": np.array([0.0, 0.0, 1.0]),
}


def find_staged_approach(
    *,
    capsules_base,
    result: ClosureResult,
    yaw: float,
    aabb_center: np.ndarray,
    half_extents: np.ndarray,
    standoffs_m: list[float],
    path_samples: int,
    final_ignore_m: float,
    final_ignore_groups: tuple[str, ...],
    directions: list[str],
) -> dict[str, object]:
    rotation = closure_base_rotation(result, yaw)
    final = closure_base_position(result, yaw, object_center=aabb_center)

    for direction_name in directions:
        local_vec = RETREAT_VECTORS[direction_name]
        if RETREAT_DIRECTIONS[direction_name] == "local":
            retreat_axis_world = rotation @ local_vec
        else:
            retreat_axis_world = local_vec
        for standoff_m in standoffs_m:
            pregrasp = final - standoff_m * retreat_axis_world
            rows = sample_linear_path_collisions(
                capsules_base,
                start=pregrasp,
                final=final,
                rotation=rotation,
                half_extents=half_extents,
                aabb_center=aabb_center,
                path_samples=path_samples,
                final_ignore_m=final_ignore_m,
                final_ignore_groups=final_ignore_groups,
            )
            collided = any(bool(row["collision"]) for row in rows)
            if not collided:
                return {
                    "found": True,
                    "direction": direction_name,
                    "standoff_mm": standoff_m * 1000.0,
                    "pregrasp_mm": (pregrasp * 1000.0).tolist(),
                    "final_mm": (final * 1000.0).tolist(),
                }

    return {
        "found": False,
        "direction": "",
        "standoff_mm": None,
        "pregrasp_mm": None,
        "final_mm": (final * 1000.0).tolist(),
    }


def evaluate_object(model, data, closure: ClosureGeometry, obj: ObjectSpec, args: argparse.Namespace) -> list[dict[str, object]]:
    mode = args.mode_override or obj.mode
    width_offset_m = args.object_width_offset_mm / 1000.0
    result = solve_object_grasp(closure, obj, mode, width_offset_m)

    ctrl_values = path_ctrl_values(result, args.path_hand_shape)
    set_closure_qpos(model, data, ctrl_values)
    capsules_base = build_capsule_proxy(model, data)

    aabb_center = tabletop_object_center(obj)
    half_extents = np.array(obj.size_m, dtype=float) / 2.0 + args.hand_clearance_mm / 1000.0
    final_ignore_m = args.final_contact_ignore_mm / 1000.0

    if args.ignore_palm_near_final == "always":
        ignore_palm = True
    elif args.ignore_palm_near_final == "never":
        ignore_palm = False
    else:
        ignore_palm = mode.startswith("plane")
    finger_groups = ("thumb", "index", "middle", "ring", "pinky")
    final_ignore_groups = finger_groups + (("palm",) if ignore_palm else ())

    standoffs_m = [v / 1000.0 for v in args.standoff_mm]
    yaws = np.linspace(0.0, 2.0 * math.pi, args.yaw_samples, endpoint=False)

    rows: list[dict[str, object]] = []
    for yaw in yaws:
        outcome = find_staged_approach(
            capsules_base=capsules_base,
            result=result,
            yaw=yaw,
            aabb_center=aabb_center,
            half_extents=half_extents,
            standoffs_m=standoffs_m,
            path_samples=args.path_samples,
            final_ignore_m=final_ignore_m,
            final_ignore_groups=final_ignore_groups,
            directions=args.directions,
        )
        rows.append(
            {
                "object": obj.name,
                "mode": mode,
                "yaw_deg": f"{math.degrees(yaw):.1f}",
                "found": int(outcome["found"]),
                "direction": outcome["direction"],
                "standoff_mm": f"{outcome['standoff_mm']:.1f}" if outcome["standoff_mm"] is not None else "",
                "pregrasp_x_mm": f"{outcome['pregrasp_mm'][0]:.2f}" if outcome["pregrasp_mm"] else "",
                "pregrasp_y_mm": f"{outcome['pregrasp_mm'][1]:.2f}" if outcome["pregrasp_mm"] else "",
                "pregrasp_z_mm": f"{outcome['pregrasp_mm'][2]:.2f}" if outcome["pregrasp_mm"] else "",
                "final_x_mm": f"{outcome['final_mm'][0]:.2f}",
                "final_y_mm": f"{outcome['final_mm'][1]:.2f}",
                "final_z_mm": f"{outcome['final_mm'][2]:.2f}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "object", "mode", "yaw_deg", "found", "direction", "standoff_mm",
        "pregrasp_x_mm", "pregrasp_y_mm", "pregrasp_z_mm",
        "final_x_mm", "final_y_mm", "final_z_mm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fk = InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk) if args.xml else InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    objects = [BUILTIN_OBJECTS[name] for name in args.objects]

    xml_path = args.xml or fk.xml_path
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("RH56 staged-approach assumptions:")
    print("  simulation_only: true")
    print(f"  standoff_candidates_mm: {args.standoff_mm}")
    print(f"  ignore_palm_near_final: {args.ignore_palm_near_final}")
    print(f"  output: {args.out}")

    all_rows: list[dict[str, object]] = []
    for obj in objects:
        rows = evaluate_object(model, data, closure, obj, args)
        all_rows.extend(rows)
        found = sum(r["found"] for r in rows)
        print(f"  {obj.name}: {found}/{len(rows)} yaw samples have a collision-free retreat-then-converge approach")
        direction_counts: dict[str, int] = {}
        for r in rows:
            if r["found"]:
                direction_counts[r["direction"]] = direction_counts.get(r["direction"], 0) + 1
        for direction, count in sorted(direction_counts.items(), key=lambda kv: -kv[1]):
            print(f"    via {direction}: {count}")

    write_csv(args.out / "summary.csv", all_rows)
    (args.out / "assumptions.json").write_text(
        json.dumps(
            {
                "script": "tools/run_staged_approach_check.py",
                "simulation_only": True,
                "approach_model": (
                    "two-waypoint path: pregrasp (retreated along the hand's own "
                    "local +Z approach axis by a candidate standoff distance) -> "
                    "final analytical grasp pose. Tries standoffs in the given "
                    "order and reports the first collision-free one."
                ),
                "standoff_candidates_mm": args.standoff_mm,
                "ignore_palm_near_final": args.ignore_palm_near_final,
                "hand_clearance_mm": args.hand_clearance_mm,
                "final_contact_ignore_mm": args.final_contact_ignore_mm,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
