#!/usr/bin/env python3
"""Sweep a 3-D grid of start hand poses around one object, per strategy.

The paper-condition sweep answers "how often does each strategy work". This one
answers "where can it start from", by walking a box of start hand-base positions
around a single object and recording, for every column, the lowest height from
which the commanded approach and closure executes cleanly.

The result is a no-go volume per strategy: the region of start poses the strategy
locks out. Because the failures are driven by clearance above the support
surface, the admissible set is a region above a surface, and a per-column floor
height captures it compactly.

GraspGen-X is swept twice, which brackets how much its ranked candidate list
matters:

  ggx_best   the manuscript protocol -- admissible if any of the top-k
             candidates executes cleanly
  ggx_worst  admissible only if every one of the top-k does, i.e. the risk taken
             by trusting the ranking without a collision check

Simulation only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ggx_strategy import build_ggx_plans  # noqa: E402
from strategy_admissibility import (  # noqa: E402
    STRATEGIES,
    ClosureGeometry,
    HandSampler,
    InspireHandFK,
    approach_basis,
    build_strategy_plans,
    grasp_yaw_for_object,
    simulate_execution,
)

from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
    tabletop_grasp_target,
)

LAYERS = list(STRATEGIES) + ["ggx_best", "ggx_worst"]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", default="paper_strawberry", choices=sorted(BUILTIN_OBJECTS))
    ap.add_argument("--out", type=Path, default=here.parent / "results")
    ap.add_argument("--xy-range-mm", type=float, nargs=2, default=[-240.0, 240.0])
    ap.add_argument("--xy-step-mm", type=float, default=30.0)
    ap.add_argument("--z-range-mm", type=float, nargs=2, default=[0.0, 330.0])
    ap.add_argument("--z-step-mm", type=float, default=30.0)
    ap.add_argument("--object-width-offset-mm", type=float, default=20.0)
    ap.add_argument("--contact-policy", choices=["tips-only", "contact-fingers"],
                    default="contact-fingers")
    ap.add_argument("--approach-samples", type=int, default=16)
    ap.add_argument("--closure-samples", type=int, default=16)
    ap.add_argument("--floor-tolerance-mm", type=float, default=3.0)
    ap.add_argument("--floor-model", choices=["capsule", "mesh"], default="capsule",
        help="Support-plane geometry: capsule proxy (needs a tolerance) or the model's exact collision meshes (needs none).")
    ap.add_argument("--contact-tolerance-mm", type=float, default=2.0)
    ap.add_argument("--iterative-step-mm", type=float, default=15.0)
    ap.add_argument("--ggx-candidates", type=Path, default=None)
    ap.add_argument("--ggx-topk", type=int, default=10)
    ap.add_argument(
        "--yaw-mode",
        choices=["fixed", "radial"],
        default="fixed",
        help=(
            "fixed keeps one grasp yaw everywhere, so the admissible set is "
            "essentially a ceiling. radial turns the hand to face the object "
            "from wherever it starts, which is what an operator would do and "
            "makes the no-go set depend on approach azimuth."
        ),
    )
    ap.add_argument(
        "--clutter",
        default=None,
        choices=sorted(BUILTIN_OBJECTS),
        help="Place a neighbouring object on the table beside the target.",
    )
    ap.add_argument(
        "--clutter-offset-mm",
        type=float,
        nargs=2,
        default=[0.0, 120.0],
        help="Neighbour position relative to the target, in the world xy plane.",
    )
    ap.add_argument(
        "--region",
        choices=["box", "quarter-ball"],
        default="box",
        help=(
            "box sweeps the full xy grid at every height. quarter-ball keeps "
            "only start poses within --region-radius-mm of the grasp point that "
            "are above it and on the approach side, which is the set an arm "
            "would actually be commanded from and removes the poses behind and "
            "below the object that dominate a box grid."
        ),
    )
    ap.add_argument("--region-radius-mm", type=float, default=300.0)
    ap.add_argument(
        "--grasp-height-mm",
        type=float,
        default=None,
        help=(
            "Override the grasp point height above the table. Grasping a tall "
            "object low on its body is the case with structure in it: the table "
            "bounds the start set from below and the object's own body blocks "
            "the descent from above, so the no-go set is not a height field."
        ),
    )
    ap.add_argument(
        "--full-grid",
        action="store_true",
        help=(
            "Evaluate every cell instead of scanning each column to its lowest "
            "admissible height. The column scan assumes admissibility is "
            "monotone in height, which makes the no-go set a height field -- a "
            "sheet. Objects tall enough to be approached above their own base "
            "break that assumption, and only the full grid can represent the "
            "resulting overhangs."
        ),
    )
    ap.add_argument("--verify-monotonic", type=int, default=0,
                    help="Sample this many columns and check the monotone-in-height assumption.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    obj = BUILTIN_OBJECTS[args.object]
    if args.grasp_height_mm is None:
        grasp_target = tabletop_grasp_target(obj)
    else:
        grasp_target = tabletop_grasp_target(
            obj, z_fraction=float(args.grasp_height_mm) / 1000.0 / obj.size_m[2])
    aabb_center = tabletop_aabb_center(obj)
    yaw = grasp_yaw_for_object(obj)

    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    sampler = HandSampler(fk.xml_path)

    target_width = obj.grasp_width_m + args.object_width_offset_mm / 1000.0

    _plan_cache: dict[int, dict] = {}

    def plans_for(yaw_rad: float) -> dict:
        """Strategy plans at a given grasp yaw, cached by angle.

        Only the hand-base rotation depends on yaw, but rebuilding through the
        analytical solver keeps the terminal pose exactly consistent with the
        rest of the analysis rather than rotating a cached result by hand.
        """

        key = int(round(math.degrees(yaw_rad) * 4))
        cached = _plan_cache.get(key)
        if cached is None:
            cached = build_strategy_plans(
                fk, closure, mode=obj.mode, target_width_m=target_width,
                grasp_target=grasp_target, yaw_rad=yaw_rad,
                iterative_step_mm=args.iterative_step_mm,
            )
            _plan_cache[key] = cached
        return cached

    plans = plans_for(yaw)
    ggx_plans = []
    if args.ggx_candidates is not None:
        path = args.ggx_candidates / f"{args.object}.yml"
        if path.exists():
            ggx_plans = build_ggx_plans(
                fk, path, object_center=aabb_center,
                grasp_target=grasp_target, top_k=args.ggx_topk,
            )

    obstacles, obstacle_centers = [], []
    if args.clutter is not None:
        neighbour = BUILTIN_OBJECTS[args.clutter]
        offset = np.array([args.clutter_offset_mm[0], args.clutter_offset_mm[1], 0.0]) / 1000.0
        obstacles = [neighbour]
        obstacle_centers = [tabletop_aabb_center(neighbour)
                            + np.array([aabb_center[0], aabb_center[1], 0.0]) + offset]
        print(f"  neighbour: {args.clutter} at "
              f"{np.round(obstacle_centers[0] * 1000).astype(int)} mm")

    xs = np.arange(args.xy_range_mm[0], args.xy_range_mm[1] + 1e-9, args.xy_step_mm)
    ys = np.arange(args.xy_range_mm[0], args.xy_range_mm[1] + 1e-9, args.xy_step_mm)
    zs = np.arange(args.z_range_mm[0], args.z_range_mm[1] + 1e-9, args.z_step_mm)

    def check(plan, start_base) -> bool:
        return simulate_execution(
            plan, sampler, start_base=start_base, obj=obj,
            floor_tolerance_m=args.floor_tolerance_mm / 1000.0,
            floor_model=args.floor_model,
            contact_tolerance_m=args.contact_tolerance_mm / 1000.0,
            contact_policy=args.contact_policy,
            approach_samples=args.approach_samples,
            closure_samples=args.closure_samples,
            obstacles=obstacles or None,
            obstacle_centers=obstacle_centers or None,
        ).admissible

    def ggx_outcomes(start_base) -> tuple[bool, bool]:
        """(any candidate admissible, every candidate admissible).

        Stops as soon as both are decided, which is usually after two
        candidates.
        """

        saw_true = saw_false = False
        for plan in ggx_plans:
            if check(plan, start_base):
                saw_true = True
            else:
                saw_false = True
            if saw_true and saw_false:
                break
        return saw_true, not saw_false

    # The failures are driven by clearance, so admissibility is monotone in
    # height: once a column works it keeps working higher up.  Scanning each
    # column from the bottom and stopping at the first admissible height is then
    # equivalent to the full grid at a fraction of the cost.  --verify-monotonic
    # samples columns and checks that assumption instead of trusting it.
    above = float(zs[-1] + (zs[1] - zs[0]))
    floors = {name: np.full((len(xs), len(ys)), above, dtype=float) for name in LAYERS}
    evaluations = 0
    yaw_used = np.full((len(xs), len(ys)), yaw, dtype=float)

    approach_dir, _lateral_dir = approach_basis(yaw)

    def column_yaw_at(x: float, y: float) -> float:
        if args.yaw_mode == "radial" and (abs(x) > 1e-6 or abs(y) > 1e-6):
            return math.atan2(-y, -x) - math.pi / 2.0
        return yaw

    def in_region(x: float, y: float, z: float) -> bool:
        """Is this start pose inside the region being swept?

        The quarter ball is the set of poses above the grasp point, on the side
        the arm approaches from, and within reach of it.
        """

        if args.region == "box":
            return True
        dz = z - grasp_target[2] * 1000.0
        if dz < -1e-9:
            return False
        if float(np.dot(np.array([x, y, 0.0]), approach_dir)) > 1e-9:
            return False
        return math.sqrt(x * x + y * y + dz * dz) <= args.region_radius_mm + 1e-9

    sampled = np.array(
        [[[in_region(float(x), float(y), float(z)) for z in zs] for y in ys] for x in xs],
        dtype=bool)

    if args.full_grid:
        dense = {name: np.zeros((len(xs), len(ys), len(zs)), dtype=bool) for name in LAYERS}
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                yaw_used[i, j] = column_yaw_at(x, y)
                column_plans = plans_for(float(yaw_used[i, j]))
                for k, z in enumerate(zs):
                    if not sampled[i, j, k]:
                        continue
                    start = np.array([grasp_target[0] + x / 1000.0,
                                      grasp_target[1] + y / 1000.0,
                                      z / 1000.0])
                    for name in STRATEGIES:
                        evaluations += 1
                        dense[name][i, j, k] = check(column_plans[name], start)
                    if ggx_plans:
                        best, worst = ggx_outcomes(start)
                        evaluations += len(ggx_plans)
                        dense["ggx_best"][i, j, k] = best
                        dense["ggx_worst"][i, j, k] = worst
            print(f"  column row {i + 1}/{len(xs)}  ({evaluations} evaluations)", flush=True)
        for name in LAYERS:
            hits = dense[name].any(axis=2)
            idx = np.argmax(dense[name], axis=2)
            floors[name] = np.where(hits, zs[idx], above)
        grids = dense
        return _write(args, obj, xs, ys, zs, grasp_target, aabb_center, yaw,
                      yaw_used, grids, floors, sampled, obstacles,
                      obstacle_centers, ggx_plans)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Face the object from wherever the hand starts.  The closure axis
            # lies along the hand's local x, and approach_basis puts the approach
            # along +y of that frame, so the yaw that walks the hand inward from
            # (x, y) is the bearing of the inward direction less a quarter turn.
            column_yaw = column_yaw_at(x, y)
            yaw_used[i, j] = column_yaw
            column_plans = plans_for(column_yaw)
            pending = set(LAYERS) if ggx_plans else set(STRATEGIES)
            for k, z in enumerate(zs):
                if not pending:
                    break
                if not sampled[i, j, k]:
                    continue
                start = np.array([grasp_target[0] + x / 1000.0,
                                  grasp_target[1] + y / 1000.0,
                                  z / 1000.0])
                for name in list(pending & set(STRATEGIES)):
                    evaluations += 1
                    if check(column_plans[name], start):
                        floors[name][i, j] = z
                        pending.discard(name)
                if ggx_plans and ({"ggx_best", "ggx_worst"} & pending):
                    best, worst = ggx_outcomes(start)
                    evaluations += len(ggx_plans)
                    if best and "ggx_best" in pending:
                        floors["ggx_best"][i, j] = z
                        pending.discard("ggx_best")
                    if worst and "ggx_worst" in pending:
                        floors["ggx_worst"][i, j] = z
                        pending.discard("ggx_worst")
        print(f"  column row {i + 1}/{len(xs)}  ({evaluations} evaluations)", flush=True)

    # Rebuild the dense grids the renderer expects from the floor heights.
    grids = {}
    for name in LAYERS:
        grid = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
        for k, z in enumerate(zs):
            grid[:, :, k] = floors[name] <= z + 1e-9
        grids[name] = grid

    if args.verify_monotonic:
        rng = np.random.default_rng(0)
        bad = 0
        for _ in range(args.verify_monotonic):
            i = int(rng.integers(len(xs)))
            j = int(rng.integers(len(ys)))
            check_plans = plans_for(float(yaw_used[i, j]))
            for name in STRATEGIES:
                truth = []
                for z in zs:
                    start = np.array([grasp_target[0] + xs[i] / 1000.0,
                                      grasp_target[1] + ys[j] / 1000.0,
                                      z / 1000.0])
                    truth.append(check(check_plans[name], start))
                truth = np.asarray(truth)
                if truth.any() and not truth[np.argmax(truth):].all():
                    bad += 1
        print(f"  monotonicity check: {bad} non-monotone columns "
              f"of {args.verify_monotonic * len(STRATEGIES)} sampled")

    return _write(args, obj, xs, ys, zs, grasp_target, aabb_center, yaw,
                  yaw_used, grids, floors, sampled, obstacles,
                  obstacle_centers, ggx_plans)


def _write(args, obj, xs, ys, zs, grasp_target, aabb_center, yaw, yaw_used,
           grids, floors, sampled, obstacles, obstacle_centers, ggx_plans) -> int:
    """Write the volume archive and its assumptions record.

    Shared by both sweep paths so the column scan and the full grid emit
    byte-identical schemas; only ``sampled`` and the assumptions record
    distinguish them.
    """

    total = int(sampled.sum()) if args.region != "box" else len(xs) * len(ys) * len(zs)
    tag = args.object + ("" if args.clutter is None else f"_{args.clutter}")
    out = args.out / f"nogo_volume_{tag}.npz"
    np.savez_compressed(
        out,
        xs_mm=xs, ys_mm=ys, zs_mm=zs,
        grasp_target_m=grasp_target,
        aabb_center_m=aabb_center,
        object_size_m=np.asarray(obj.size_m),
        **({"clutter_center_m": obstacle_centers[0],
            "clutter_size_m": np.asarray(obstacles[0].size_m)} if obstacles else {}),
        yaw_rad=yaw,
        yaw_used_rad=yaw_used,
        sampled=sampled,
        **{f"admissible_{name}": grids[name] for name in LAYERS},
        **{f"floor_mm_{name}": floors[name] for name in LAYERS},
    )
    (args.out / f"nogo_volume_{tag}_assumptions.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "object": args.object,
        "collision_shape": obj.collision_shape,
        "grasp_mode": obj.mode,
        "grasp_height_mm": float(grasp_target[2] * 1000.0),
        "contact_policy": args.contact_policy,
        "yaw_mode": args.yaw_mode,
        "region": args.region,
        "region_radius_mm": args.region_radius_mm if args.region != "box" else None,
        "full_grid": bool(args.full_grid),
        "clutter": args.clutter,
        "clutter_offset_mm": args.clutter_offset_mm if args.clutter else None,
        "iterative_step_mm": args.iterative_step_mm,
        "ggx_topk": args.ggx_topk if ggx_plans else None,
        "n_ggx_candidates": len(ggx_plans),
        "grid": {
            "xy_range_mm": args.xy_range_mm, "xy_step_mm": args.xy_step_mm,
            "z_range_mm": args.z_range_mm, "z_step_mm": args.z_step_mm,
            "n_poses": total,
            "n_cells": len(xs) * len(ys) * len(zs),
        },
        "layers": LAYERS,
    }, indent=2))

    print()
    for name in LAYERS:
        grid = grids[name] & sampled
        if not grid.any() and name.startswith("ggx") and not ggx_plans:
            continue
        n = int(grid.sum())
        print(f"  {name:18s} admissible {n:6d}/{total}  = {n / max(total, 1):6.1%}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
