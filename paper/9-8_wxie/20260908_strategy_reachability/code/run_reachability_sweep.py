#!/usr/bin/env python3
"""Strategy reachability sweep over the 15 paper objects.

Two samplers are provided:

paper-conditions
    The ten approach conditions used in the hardware benchmark: P1 level with
    the grasp point at 250 mm standoff, P2 250 mm directly above, and two
    height levels (100 mm, 250 mm) crossed with four lateral offsets
    (-150, -50, +50, +150 mm).  This gives a per-object admissible count that is
    directly comparable with the 10 hardware trials per object and strategy.

grid
    A dense sweep of approach height above the grasp point against lateral
    offset, at a fixed standoff, used to draw the admissible-region boundary.

Outputs CSV per trial plus a JSON assumptions record.  Simulation only.

Usage:
    uv run python paper/9-8_wxie/20260908_strategy_reachability/code/\\
        run_reachability_sweep.py --sampler paper-conditions --out ../results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_admissibility import (  # noqa: E402
    FAILURE_MODES,
    STRATEGIES,
    ClosureGeometry,
    HandSampler,
    InspireHandFK,
    approach_basis,
    build_strategy_plans,
    grasp_yaw_for_object,
    simulate_execution,
)

from ggx_strategy import build_ggx_plans  # noqa: E402

from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
    tabletop_grasp_target,
)

PAPER_OBJECTS = [
    "paper_big_screwdriver",
    "paper_bottle",
    "paper_can",
    "paper_charger",
    "paper_metal_cup",
    "paper_mustard",
    "paper_orange",
    "paper_pen",
    "paper_small_screwdriver",
    "paper_sugar_box",
    "paper_egg",
    "paper_nut",
    "paper_paper_cup",
    "paper_raspberry",
    "paper_strawberry",
]

# Object-set split used by the hardware benchmark.
DELICATE = {"paper_egg", "paper_nut", "paper_paper_cup", "paper_raspberry", "paper_strawberry"}


def paper_conditions(
    p1_distance_mm: float = 250.0,
    p2_height_mm: float = 250.0,
    level_heights_mm: tuple[float, float] = (150.0, 250.0),
    lateral_mm: tuple[float, ...] = (-30.0, -10.0, 10.0, 30.0),
) -> list[dict[str, object]]:
    """The ten hardware approach conditions, as offsets from the grasp point.

    The hardware protocol samples ten approach poses per object from the grasp
    plane and includes exactly one adversarial pose level with the grasp point,
    which is the case that exposes premature object contact. The remaining nine
    start above the grasp point. Lateral jitter is +/-30 mm, as stated in the
    protocol; an earlier reconstruction used +/-150 mm, which placed starts far
    outside the sampled region and is the single largest source of the
    sim-hardware gap reported in results/calibration.
    """

    points: list[dict[str, object]] = [
        {"condition": "P1_level", "standoff_mm": p1_distance_mm, "lateral_mm": 0.0, "height_mm": 0.0},
        {"condition": "P2_overhead", "standoff_mm": 0.0, "lateral_mm": 0.0, "height_mm": p2_height_mm},
    ]
    for level_idx, height in enumerate(level_heights_mm, start=1):
        for lateral in lateral_mm:
            points.append(
                {
                    "condition": f"L{level_idx}_d{lateral:+g}",
                    "standoff_mm": p1_distance_mm,
                    "lateral_mm": float(lateral),
                    "height_mm": float(height),
                }
            )
    return points


def grid_conditions(
    height_range_mm: tuple[float, float],
    height_step_mm: float,
    lateral_range_mm: tuple[float, float],
    lateral_step_mm: float,
    standoff_mm: float,
) -> list[dict[str, object]]:
    heights = np.arange(height_range_mm[0], height_range_mm[1] + 1e-9, height_step_mm)
    laterals = np.arange(lateral_range_mm[0], lateral_range_mm[1] + 1e-9, lateral_step_mm)
    points: list[dict[str, object]] = []
    for height in heights:
        for lateral in laterals:
            points.append(
                {
                    "condition": f"h{height:+.0f}_d{lateral:+.0f}",
                    "standoff_mm": float(standoff_mm),
                    "lateral_mm": float(lateral),
                    "height_mm": float(height),
                }
            )
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=PAPER_OBJECTS)
    parser.add_argument("--sampler", choices=["paper-conditions", "grid"], default="paper-conditions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default=None, help="Filename stem override.")
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument("--naive-pose-policy", choices=["open-pose", "final-orientation"], default="open-pose")
    parser.add_argument("--iterative-step-mm", type=float, default=10.0)
    parser.add_argument("--approach-samples", type=int, default=24)
    parser.add_argument("--closure-samples", type=int, default=24)
    parser.add_argument("--floor-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--floor-model", choices=["capsule", "mesh"], default="capsule",
        help="Support-plane geometry: capsule proxy (needs a tolerance) or the model's exact collision meshes (needs none).")
    parser.add_argument("--contact-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--contact-policy",
        choices=["tips-only", "contact-fingers"],
        default="tips-only",
        help=(
            "tips-only: only the distal capsule of each planned contact finger "
            "may touch the object. contact-fingers: the whole of those fingers "
            "may, as in an enveloping grasp."
        ),
    )
    parser.add_argument(
        "--approach-path",
        choices=["direct", "via-overhead"],
        default="direct",
        help=(
            "direct is one straight line from the start pose to the pre-grasp. "
            "via-overhead travels at the start height until above the pre-grasp, "
            "then descends."
        ),
    )
    parser.add_argument("--standoff-mm", type=float, default=250.0)
    parser.add_argument("--condition-lateral-mm", type=float, nargs="+", default=None,
                        help="Override the lateral jitter of the paper conditions.")
    parser.add_argument("--condition-heights-mm", type=float, nargs=2, default=None,
                        help="Override the two elevated approach heights.")
    parser.add_argument("--height-range-mm", type=float, nargs=2, default=[0.0, 300.0])
    parser.add_argument("--height-step-mm", type=float, default=20.0)
    parser.add_argument("--lateral-range-mm", type=float, nargs=2, default=[-150.0, 150.0])
    parser.add_argument("--lateral-step-mm", type=float, default=25.0)
    parser.add_argument(
        "--ggx-candidates",
        type=Path,
        default=None,
        help=(
            "Directory of GraspGen-X Isaac-grasp YAML files, one per object. "
            "Adds GGX as a fourth strategy."
        ),
    )
    parser.add_argument(
        "--ggx-topk",
        type=int,
        default=20,
        help=(
            "Candidates tested per start pose, in descending confidence order. "
            "The start pose is admissible if any of them executes cleanly."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.sampler.replace("-", "_")

    if args.sampler == "paper-conditions":
        kwargs = {"p1_distance_mm": args.standoff_mm}
        if args.condition_lateral_mm is not None:
            kwargs["lateral_mm"] = tuple(args.condition_lateral_mm)
        if args.condition_heights_mm is not None:
            kwargs["level_heights_mm"] = tuple(args.condition_heights_mm)
        conditions = paper_conditions(**kwargs)
    else:
        conditions = grid_conditions(
            tuple(args.height_range_mm),
            args.height_step_mm,
            tuple(args.lateral_range_mm),
            args.lateral_step_mm,
            args.standoff_mm,
        )

    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    sampler = HandSampler(fk.xml_path, radius_scale=args.radius_scale)

    rows: list[dict[str, object]] = []
    for obj_name in args.objects:
        obj = BUILTIN_OBJECTS[obj_name]
        grasp_target = tabletop_grasp_target(obj)
        yaw = grasp_yaw_for_object(obj)
        approach_dir, lateral_dir = approach_basis(yaw)
        target_width = obj.grasp_width_m + args.object_width_offset_mm / 1000.0
        plans = build_strategy_plans(
            fk,
            closure,
            mode=obj.mode,
            target_width_m=target_width,
            grasp_target=grasp_target,
            yaw_rad=yaw,
            iterative_step_mm=args.iterative_step_mm,
            naive_pose_policy=args.naive_pose_policy,
        )
        ggx_plans = []
        if args.ggx_candidates is not None:
            candidate_file = args.ggx_candidates / f"{obj_name}.yml"
            if candidate_file.exists():
                ggx_plans = build_ggx_plans(
                    fk,
                    candidate_file,
                    object_center=tabletop_aabb_center(obj),
                    grasp_target=grasp_target,
                    top_k=args.ggx_topk,
                )

        for cond in conditions:
            start_base = (
                grasp_target
                - approach_dir * (float(cond["standoff_mm"]) / 1000.0)
                + lateral_dir * (float(cond["lateral_mm"]) / 1000.0)
                + np.array([0.0, 0.0, float(cond["height_mm"]) / 1000.0])
            )
            for strategy in STRATEGIES:
                result = simulate_execution(
                    plans[strategy],
                    sampler,
                    start_base=start_base,
                    obj=obj,
                    floor_tolerance_m=args.floor_tolerance_mm / 1000.0,
                    floor_model=args.floor_model,
                    contact_tolerance_m=args.contact_tolerance_mm / 1000.0,
                    contact_policy=args.contact_policy,
                    approach_path=args.approach_path,
                    approach_samples=args.approach_samples,
                    closure_samples=args.closure_samples,
                )
                row: dict[str, object] = {
                    "object": obj_name,
                    "object_label": obj.label,
                    "category": "delicate" if obj_name in DELICATE else "ycb_like",
                    "size_x_mm": obj.size_m[0] * 1000.0,
                    "size_y_mm": obj.size_m[1] * 1000.0,
                    "size_z_mm": obj.size_m[2] * 1000.0,
                    "volume_cm3": float(np.prod(obj.size_m)) * 1e6,
                    "grasp_width_mm": obj.grasp_width_m * 1000.0,
                    "grasp_height_mm": float(grasp_target[2]) * 1000.0,
                    "mode": obj.mode,
                    "collision_shape": obj.collision_shape,
                    "yaw_deg": math.degrees(yaw),
                    "condition": cond["condition"],
                    "standoff_mm": cond["standoff_mm"],
                    "lateral_mm": cond["lateral_mm"],
                    "height_mm": cond["height_mm"],
                    "strategy": strategy,
                    "admissible": int(result.admissible),
                    "min_floor_clearance_mm": result.min_floor_clearance_m * 1000.0,
                    "min_object_clearance_mm": result.min_object_clearance_m * 1000.0,
                    "premature_clearance_mm": result.premature_clearance_m * 1000.0,
                    "terminal_offset_mm": result.terminal_offset_m * 1000.0,
                    "terminal_outside_object_mm": result.terminal_outside_object_m * 1000.0,
                    "floor_blocker": result.floor_blocker,
                    "premature_blocker": result.premature_blocker,
                    "first_failure_phase": result.first_failure_phase,
                    "samples": result.samples,
                }
                for mode in FAILURE_MODES:
                    row[mode] = int(mode in result.failures)
                row["ggx_candidate_rank"] = -1
                row["ggx_candidates_tested"] = 0
                rows.append(row)

            if ggx_plans:
                # Descending confidence order; the first candidate that executes
                # cleanly is the one GGX would use.  No clean candidate is a
                # planning failure, as in the manuscript protocol.
                chosen = None
                chosen_rank = -1
                for rank, plan in enumerate(ggx_plans):
                    candidate_result = simulate_execution(
                        plan,
                        sampler,
                        start_base=start_base,
                        obj=obj,
                        floor_tolerance_m=args.floor_tolerance_mm / 1000.0,
                        floor_model=args.floor_model,
                        contact_tolerance_m=args.contact_tolerance_mm / 1000.0,
                        contact_policy=args.contact_policy,
                        approach_samples=args.approach_samples,
                        closure_samples=args.closure_samples,
                    )
                    if chosen is None:
                        chosen, chosen_rank = candidate_result, rank
                    if candidate_result.admissible:
                        chosen, chosen_rank = candidate_result, rank
                        break
                row = {
                    "object": obj_name,
                    "object_label": obj.label,
                    "category": "delicate" if obj_name in DELICATE else "ycb_like",
                    "size_x_mm": obj.size_m[0] * 1000.0,
                    "size_y_mm": obj.size_m[1] * 1000.0,
                    "size_z_mm": obj.size_m[2] * 1000.0,
                    "volume_cm3": float(np.prod(obj.size_m)) * 1e6,
                    "grasp_width_mm": obj.grasp_width_m * 1000.0,
                    "grasp_height_mm": float(grasp_target[2]) * 1000.0,
                    "mode": obj.mode,
                    "collision_shape": obj.collision_shape,
                    "yaw_deg": math.degrees(yaw),
                    "condition": cond["condition"],
                    "standoff_mm": cond["standoff_mm"],
                    "lateral_mm": cond["lateral_mm"],
                    "height_mm": cond["height_mm"],
                    "strategy": "graspgen_x",
                    "admissible": int(chosen.admissible),
                    "min_floor_clearance_mm": chosen.min_floor_clearance_m * 1000.0,
                    "min_object_clearance_mm": chosen.min_object_clearance_m * 1000.0,
                    "premature_clearance_mm": chosen.premature_clearance_m * 1000.0,
                    "terminal_offset_mm": chosen.terminal_offset_m * 1000.0,
                    "terminal_outside_object_mm": chosen.terminal_outside_object_m * 1000.0,
                    "floor_blocker": chosen.floor_blocker,
                    "premature_blocker": chosen.premature_blocker,
                    "first_failure_phase": chosen.first_failure_phase,
                    "samples": chosen.samples,
                }
                for mode in FAILURE_MODES:
                    row[mode] = int(mode in chosen.failures)
                row["ggx_candidate_rank"] = chosen_rank if chosen.admissible else -1
                row["ggx_candidates_tested"] = len(ggx_plans)
                rows.append(row)

    csv_path = args.out / f"{tag}_trials.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    assumptions = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "hardware_required": False,
        "sampler": args.sampler,
        "n_conditions": len(conditions),
        "n_objects": len(args.objects),
        "n_trials": len(rows),
        "xml_path": str(fk.xml_path),
        "object_width_offset_mm": args.object_width_offset_mm,
        "naive_pose_policy": args.naive_pose_policy,
        "iterative_step_mm": args.iterative_step_mm,
        "approach_samples": args.approach_samples,
        "closure_samples": args.closure_samples,
        "floor_tolerance_mm": args.floor_tolerance_mm,
        "floor_model": args.floor_model,
        "contact_tolerance_mm": args.contact_tolerance_mm,
        "contact_policy": args.contact_policy,
        "approach_path": args.approach_path,
        "ggx_candidates": str(args.ggx_candidates) if args.ggx_candidates else None,
        "ggx_topk": args.ggx_topk if args.ggx_candidates else None,
        "radius_scale": args.radius_scale,
        "conditions": conditions,
        "failure_modes": list(FAILURE_MODES),
        "notes": (
            "Geometric admissibility of the commanded approach and closure "
            "motion against the object proxy and the support plane. Not a "
            "grasp-success prediction: no dynamics, friction, or object "
            "compliance are modelled."
        ),
    }
    (args.out / f"{tag}_assumptions.json").write_text(json.dumps(assumptions, indent=2))

    # Console summary
    print(f"trials: {len(rows)}  objects: {len(args.objects)}  conditions: {len(conditions)}")
    present = list(STRATEGIES) + (["graspgen_x"] if args.ggx_candidates else [])
    for strategy in present:
        subset = [r for r in rows if r["strategy"] == strategy]
        if not subset:
            continue
        admitted = sum(int(r["admissible"]) for r in subset)
        line = f"  {strategy:18s} {admitted:4d}/{len(subset):4d} = {admitted / len(subset):6.1%}"
        for mode in FAILURE_MODES:
            line += f"  {mode.split('_')[0]}={sum(int(r[mode]) for r in subset):3d}"
        print(line)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
