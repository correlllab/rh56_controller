#!/usr/bin/env python3
"""Which GraspGen-X proposals fail no matter where the hand starts.

A proposal's approach leg runs from the start pose to the proposed pose; its
closure and terminal checks happen at the proposed pose with the wrist held
still, so a failure there is a property of the proposal and not of the start.
Running every proposal with a zero-length approach therefore separates the
candidates the start-pose sweep can never rescue from the ones it might. When
all of them fail this way the no-go volume is the whole sampled region at any
depth k, and the 3-D sweep need not be run to say so.

Usage:
    uv run python ggx_start_independent_check.py --object paper_strawberry \\
        --ggx-candidates ../../../../artifacts/paper_15_object_grasp_success/graspgenx_candidates
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ggx_strategy import build_ggx_plans  # noqa: E402
from strategy_admissibility import HandSampler, InspireHandFK, simulate_execution  # noqa: E402
from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
    tabletop_grasp_target,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True, choices=sorted(BUILTIN_OBJECTS))
    ap.add_argument("--ggx-candidates", type=Path, required=True)
    ap.add_argument("--ggx-topk", type=int, default=100)
    # The volume sweep's settings, so the verdict transfers to the figure.
    ap.add_argument("--floor-model", default="mesh")
    ap.add_argument("--floor-tolerance-mm", type=float, default=0.0)
    ap.add_argument("--contact-policy", default="contact-fingers")
    ap.add_argument("--contact-tolerance-mm", type=float, default=2.0)
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()

    obj = BUILTIN_OBJECTS[args.object]
    fk = InspireHandFK()
    sampler = HandSampler(fk.xml_path)
    plans = build_ggx_plans(fk, args.ggx_candidates / f"{args.object}.yml",
                            object_center=tabletop_aabb_center(obj),
                            grasp_target=tabletop_grasp_target(obj), top_k=args.ggx_topk)

    phases = collections.Counter()
    modes = collections.Counter()
    survivors = []
    for rank, plan in enumerate(plans):
        result = simulate_execution(
            plan, sampler, start_base=plan.waypoints[0].base_position, obj=obj,
            floor_tolerance_m=args.floor_tolerance_mm / 1000.0,
            floor_model=args.floor_model,
            contact_tolerance_m=args.contact_tolerance_mm / 1000.0,
            contact_policy=args.contact_policy,
            approach_samples=args.samples, closure_samples=args.samples,
        )
        if result.admissible:
            survivors.append(rank)
        else:
            phases[result.first_failure_phase] += 1
            for mode in sorted(result.failures):
                modes[mode] += 1

    n = len(plans)
    print(f"{args.object}: {n} proposals, {len(survivors)} survive a zero-length approach")
    print(f"  start-independent failures by first phase: {dict(phases)}")
    print(f"  failure modes (a proposal can show several): {dict(modes)}")
    if survivors:
        print(f"  proposals a start pose could still rescue (0-based rank): {survivors}")
    else:
        print("  no proposal executes from any start pose: the no-go volume is the whole "
              f"sampled region at k = {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
