#!/usr/bin/env python3
"""Measure the capsule proxy's over-approximation of the hand near the support plane.

The admissibility model decides ``table_collision`` from the lowest point of a
capsule proxy of the hand.  The capsules are a deliberate over-approximation --
the palm capsules are documented as "a conservative envelope around the MuJoCo
palm boxes" -- so a bare capsule test reports support-plane collisions the real
hand does not have.  The sweep absorbs that with a tolerance, and the value used
(10 mm) was chosen because it reproduced the hardware success rates.  That makes
the tolerance a parameter fitted to the quantity under test, and any agreement
with hardware is then partly circular.

This script measures the tolerance instead.  For every hand state the strategies
actually command, it computes two numbers in the same world frame:

    z_capsule  lowest point of the capsule proxy
    z_mesh     lowest point of the model's own collision geometry
               (palm boxes and cylinder, phalanx STL meshes)

and their difference

    delta = z_mesh - z_capsule

which is how far the capsule envelope hangs below the hand it stands for, along
the direction the support-plane test looks.  A capsule test with tolerance
``tau`` and a mesh test with tolerance zero agree exactly when ``tau = delta``,
so the distribution of ``delta`` over the sampled motions *is* the tolerance,
with its spread giving the range over which the answer is uncertain.

The gap depends on the hand's configuration and on its orientation, but not on
its position, and -- because the wrist rotation is ``Rz(yaw) @ Ry(tilt)`` and
``Rz`` leaves the world vertical fixed -- not on yaw either.  The script checks
that invariance numerically rather than assuming it.

Simulation only; no hardware.  Upstream code used here:
  rh56_controller/capsule_hand_proxy.py   capsule proxy construction
  code/strategy_admissibility.py          collision-geometry hull + strategy plans

Usage:
    uv run python paper/9-8_wxie/20260908_strategy_reachability/code/\
        calibrate_capsule_envelope.py --out ../results/calibration
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
    STRATEGIES,
    _VISUAL_GEOM_GROUP,
    collision_geom_points,
    collision_hull,
    ClosureGeometry,
    HandSampler,
    InspireHandFK,
    _ctrl_lerp,
    approach_basis,
    build_strategy_plans,
    grasp_yaw_for_object,
    transform_capsules,
)

from run_reachability_sweep import PAPER_OBJECTS, paper_conditions  # noqa: E402

from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_grasp_target,
)

# Samples with the hand far above the table never reach the support-plane test,
# so including them would dilute the statistic with configurations that cannot
# change any decision.  This band is the height range over which a tolerance of
# a few millimetres could flip a trial.
DECISION_BAND_MM = 25.0


def capsule_floor_height(capsules, rot: np.ndarray, base: np.ndarray) -> tuple[float, str]:
    """Lowest point of the capsule proxy, and the capsule that owns it."""

    placed = transform_capsules(capsules, rot, base)
    best = math.inf
    owner = ""
    for capsule in placed:
        z = min(float(capsule.p0[2]), float(capsule.p1[2])) - capsule.radius
        if z < best:
            best = z
            owner = f"{capsule.group}/{capsule.name}"
    return best, owner


def mesh_floor_height(hull: tuple[np.ndarray, list[str]], rot: np.ndarray, base: np.ndarray) -> tuple[float, str]:
    """Lowest point of the exact collision geometry, and the body that owns it."""

    points, owners = hull
    heights = points @ rot[2] + float(base[2])
    low = int(np.argmin(heights))
    return float(heights[low]), owners[low]


class VisualHullSampler:
    """Convex-hull vertices of the undecimated visual meshes, cached per command.

    MuJoCo collides a mesh as its convex hull, so the collision geometry already
    over-approximates the part it stands for. Measuring the visual meshes as well
    turns the single tolerance into a chain of one-sided bounds: capsule proxy
    encloses collision geometry encloses part.
    """

    def __init__(self, sampler: HandSampler):
        self.sampler = sampler
        self._cache: dict[tuple[int, ...], tuple[np.ndarray, list[str]]] = {}

    def hull(self, ctrl: dict[str, float]) -> tuple[np.ndarray, list[str]]:
        key = HandSampler._key(ctrl)
        cached = self._cache.get(key)
        if cached is None:
            self.sampler.capsules(ctrl)  # writes qpos and runs FK on the shared model
            cached = collision_hull(*collision_geom_points(
                self.sampler.model, self.sampler.data, group=_VISUAL_GEOM_GROUP))
            self._cache[key] = cached
        return cached


def stage_samples(plan, start_base: np.ndarray, approach_samples: int, closure_samples: int):
    """Yield (phase, t, base, rot, ctrl) over the same motion the sweep checks."""

    first = plan.waypoints[0]
    stages = [("approach", (np.asarray(start_base, dtype=float), first.base_position,
                            first.rotation, first.rotation, first.ctrl, first.ctrl), approach_samples)]
    for prev, nxt in zip(plan.waypoints[:-1], plan.waypoints[1:]):
        stages.append(("closure", (prev.base_position, nxt.base_position,
                                   prev.rotation, nxt.rotation, prev.ctrl, nxt.ctrl), closure_samples))

    for phase, (base_a, base_b, rot_a, rot_b, ctrl_a, ctrl_b), n in stages:
        for step in range(n + 1):
            t = step / float(n)
            base = base_a + t * (base_b - base_a)
            ctrl = _ctrl_lerp(ctrl_a, ctrl_b, t)
            if np.allclose(rot_a, rot_b):
                rot = rot_a
            else:
                blend = rot_a + t * (rot_b - rot_a)
                u, _, vt = np.linalg.svd(blend)
                rot = u @ vt
            yield phase, t, base, rot, ctrl


def yaw_invariance_check(sampler: HandSampler, ctrl: dict[str, float], tilt_deg: float) -> float:
    """Largest spread in delta over world-z yaw, which should be numerically zero."""

    caps = sampler.capsules(ctrl)
    hull = sampler.hull(ctrl)
    tilt = math.radians(tilt_deg)
    ry = np.array([[math.cos(tilt), 0.0, math.sin(tilt)], [0.0, 1.0, 0.0],
                   [-math.sin(tilt), 0.0, math.cos(tilt)]])
    deltas = []
    for yaw_deg in (0.0, 45.0, 90.0, 137.0, 210.0, 315.0):
        yaw = math.radians(yaw_deg)
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        rot = rz @ ry
        base = np.zeros(3)
        z_caps, _ = capsule_floor_height(caps, rot, base)
        z_mesh, _ = mesh_floor_height(hull, rot, base)
        deltas.append(z_mesh - z_caps)
    return float(np.ptp(deltas))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", default=PAPER_OBJECTS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="capsule_envelope")
    ap.add_argument("--object-width-offset-mm", type=float, default=20.0)
    ap.add_argument("--iterative-step-mm", type=float, default=10.0)
    ap.add_argument("--naive-pose-policy", choices=["open-pose", "final-orientation"], default="open-pose")
    ap.add_argument("--approach-samples", type=int, default=24)
    ap.add_argument("--closure-samples", type=int, default=24)
    ap.add_argument("--standoff-mm", type=float, default=250.0)
    ap.add_argument("--decision-band-mm", type=float, default=DECISION_BAND_MM)
    ap.add_argument("--radius-scale", type=float, default=1.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conditions = paper_conditions(p1_distance_mm=args.standoff_mm)
    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    sampler = HandSampler(fk.xml_path, radius_scale=args.radius_scale)
    visual = VisualHullSampler(sampler)

    rows: list[dict[str, object]] = []
    for obj_name in args.objects:
        obj = BUILTIN_OBJECTS[obj_name]
        grasp_target = tabletop_grasp_target(obj)
        yaw = grasp_yaw_for_object(obj)
        approach_dir, lateral_dir = approach_basis(yaw)
        plans = build_strategy_plans(
            fk, closure,
            mode=obj.mode,
            target_width_m=obj.grasp_width_m + args.object_width_offset_mm / 1000.0,
            grasp_target=grasp_target,
            yaw_rad=yaw,
            iterative_step_mm=args.iterative_step_mm,
            naive_pose_policy=args.naive_pose_policy,
        )
        for cond in conditions:
            start_base = (
                grasp_target
                - approach_dir * (float(cond["standoff_mm"]) / 1000.0)
                + lateral_dir * (float(cond["lateral_mm"]) / 1000.0)
                + np.array([0.0, 0.0, float(cond["height_mm"]) / 1000.0])
            )
            for strategy in STRATEGIES:
                plan = plans[strategy]
                for phase, t, base, rot, ctrl in stage_samples(
                    plan, start_base, args.approach_samples, args.closure_samples
                ):
                    caps = sampler.capsules(ctrl)
                    hull = sampler.hull(ctrl)
                    z_caps, caps_owner = capsule_floor_height(caps, rot, base)
                    z_mesh, mesh_owner = mesh_floor_height(hull, rot, base)
                    z_vis, vis_owner = mesh_floor_height(visual.hull(ctrl), rot, base)
                    # Wrist pitch about the hand-frame y axis, which is the only
                    # part of the wrist rotation the vertical test can see.
                    tilt_deg = math.degrees(math.asin(float(np.clip(rot[0, 2], -1.0, 1.0))))
                    rows.append({
                        "object": obj_name,
                        "condition": cond["condition"],
                        "strategy": strategy,
                        "phase": phase,
                        "t": round(float(t), 4),
                        "grasp_height_mm": float(grasp_target[2]) * 1000.0,
                        "wrist_pitch_deg": tilt_deg,
                        "z_capsule_mm": z_caps * 1000.0,
                        "z_mesh_mm": z_mesh * 1000.0,
                        "z_visual_mm": z_vis * 1000.0,
                        "delta_mm": (z_mesh - z_caps) * 1000.0,
                        "delta_visual_mm": (z_vis - z_mesh) * 1000.0,
                        "capsule_owner": caps_owner,
                        "mesh_owner": mesh_owner,
                        "visual_owner": vis_owner,
                    })
        print(f"  {obj_name}: {len(rows)} samples so far", flush=True)

    csv_path = args.out / f"{args.tag}_samples.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    delta = np.array([r["delta_mm"] for r in rows], dtype=float)
    delta_vis = np.array([r["delta_visual_mm"] for r in rows], dtype=float)
    z_caps = np.array([r["z_capsule_mm"] for r in rows], dtype=float)
    band = np.abs(z_caps) <= args.decision_band_mm
    delta_band = delta[band]

    def describe(values: np.ndarray) -> dict[str, float]:
        return {
            "n": int(values.size),
            "min_mm": float(np.min(values)),
            "p05_mm": float(np.percentile(values, 5)),
            "median_mm": float(np.median(values)),
            "mean_mm": float(np.mean(values)),
            "p95_mm": float(np.percentile(values, 95)),
            "max_mm": float(np.max(values)),
        }

    per_strategy = {
        s: describe(delta_band[np.array([r["strategy"] for r in rows])[band] == s])
        for s in STRATEGIES
        if np.any((np.array([r["strategy"] for r in rows])[band] == s))
    }
    owners = {}
    for r, in_band in zip(rows, band):
        if in_band:
            owners[r["capsule_owner"]] = owners.get(r["capsule_owner"], 0) + 1

    # Invariance check on a representative closed-hand state.
    probe_ctrl = {name: float(fk.ctrl_min[name]) for name in fk.ctrl_min}
    invariance = max(yaw_invariance_check(sampler, probe_ctrl, tilt) for tilt in (0.0, 20.0, 45.0))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "hardware_required": False,
        "purpose": (
            "Measure the capsule proxy's vertical over-approximation against the model's own "
            "collision geometry, so the support-plane tolerance is a measured quantity rather "
            "than a parameter fitted to hardware success rates."
        ),
        "xml_path": str(fk.xml_path),
        "n_objects": len(args.objects),
        "n_conditions": len(conditions),
        "n_samples": int(delta.size),
        "decision_band_mm": args.decision_band_mm,
        "delta_all": describe(delta),
        "delta_visual_all": describe(delta_vis),
        "delta_visual_decision_band": describe(delta_vis[band]),
        "collision_encloses_visual_everywhere": bool(np.all(delta_vis >= -1e-6)),
        "delta_decision_band": describe(delta_band),
        "delta_by_strategy_in_band": per_strategy,
        "binding_capsule_counts_in_band": dict(sorted(owners.items(), key=lambda kv: -kv[1])),
        "yaw_invariance_max_spread_mm": invariance,
        "capsule_encloses_mesh_everywhere": bool(np.all(delta >= -1e-6)),
        "recommended_tolerance_mm": float(np.median(delta_band)),
        "bounding_tolerances_mm": [float(np.percentile(delta_band, 5)), float(np.percentile(delta_band, 95))],
        "samples_csv": csv_path.name,
    }
    json_path = args.out / f"{args.tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"  samples                        {delta.size}")
    print(f"  capsule encloses mesh          {summary['capsule_encloses_mesh_everywhere']}")
    print(f"  yaw invariance spread          {invariance*1000:.4f} um")
    print(f"  delta, all samples             median {summary['delta_all']['median_mm']:.2f} mm "
          f"[{summary['delta_all']['min_mm']:.2f}, {summary['delta_all']['max_mm']:.2f}]")
    print(f"  delta, within {args.decision_band_mm:.0f} mm of table   median "
          f"{summary['delta_decision_band']['median_mm']:.2f} mm "
          f"[p05 {summary['delta_decision_band']['p05_mm']:.2f}, "
          f"p95 {summary['delta_decision_band']['p95_mm']:.2f}]  n={delta_band.size}")
    for s, d in per_strategy.items():
        print(f"    {s:20s} median {d['median_mm']:.2f} mm  [p05 {d['p05_mm']:.2f}, p95 {d['p95_mm']:.2f}]")
    print(f"  collision encloses visual      {summary['collision_encloses_visual_everywhere']}")
    print(f"  collision-vs-visual, in band   median "
          f"{summary['delta_visual_decision_band']['median_mm']:.2f} mm "
          f"[p05 {summary['delta_visual_decision_band']['p05_mm']:.2f}, "
          f"p95 {summary['delta_visual_decision_band']['p95_mm']:.2f}]")
    print(f"  binding capsule (in band)      {list(summary['binding_capsule_counts_in_band'])[:4]}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
