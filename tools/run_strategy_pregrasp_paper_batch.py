#!/usr/bin/env python3
"""Run strategy pre-grasp feasible-rate sweeps for the paper object set."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-object strategy pre-grasp feasible-rate sweeps."
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=sorted(BUILTIN_OBJECTS),
        default=PAPER_OBJECTS,
        help="Objects to sweep. Defaults to the paper grasping-experiment set.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/strategy_pregrasp_rate_paper_objects_top_10mm"),
    )
    parser.add_argument("--approach-axis", choices=["x-", "x+", "y-", "y+"], default="y-")
    parser.add_argument("--path-samples", type=int, default=10)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--floor-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--paper-hand-yaw-offset-deg", type=float, default=-90.0)
    parser.add_argument(
        "--grasp-target-z-fraction",
        type=float,
        default=None,
        help="Override all object metadata z fractions for sensitivity runs.",
    )
    parser.add_argument(
        "--grasp-target-top-offset-mm",
        type=float,
        default=None,
        help=(
            "Override all object metadata and place targets this distance "
            "below object tops. Mutually exclusive with the z-fraction override."
        ),
    )
    parser.add_argument(
        "--grasp-target-approach-offset-mm",
        type=float,
        default=0.0,
        help="Shift all grasp targets toward the approach side by this distance.",
    )
    parser.add_argument(
        "--iterative-pregrasp-policy",
        choices=["planner-max-width", "final-plus-preopen"],
        default="planner-max-width",
        help="Default matches the interactive Plan strategy.",
    )
    parser.add_argument("--preopen-mm", type=float, default=5.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_summary(path: Path, object_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    assumptions = json.loads(path.with_name("assumptions.json").read_text())
    target = assumptions["object"]["grasp_target_m"]
    target_fraction = assumptions["object"].get(
        "effective_grasp_target_fraction",
        assumptions["object"]["grasp_target_fraction"],
    )
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            obj = BUILTIN_OBJECTS[object_name]
            rows.append(
                {
                    "object": object_name,
                    "label": obj.label,
                    "mode": obj.mode,
                    "collision_shape": obj.collision_shape,
                    "grasp_width_mm": f"{obj.grasp_width_m * 1000.0:.3f}",
                    "grasp_target_fraction_z": f"{float(target_fraction[2]):.3f}",
                    "grasp_target_top_offset_mm": assumptions["object"].get(
                        "effective_grasp_target_top_offset_mm"
                    )
                    or "",
                    "grasp_target_x_mm": f"{float(target[0]) * 1000.0:.3f}",
                    "grasp_target_y_mm": f"{float(target[1]) * 1000.0:.3f}",
                    "grasp_target_z_mm": f"{float(target[2]) * 1000.0:.3f}",
                    "strategy": row["strategy"],
                    "valid_samples": row["valid_samples"],
                    "total_samples": row["total_samples"],
                    "feasible_rate": row["feasible_rate"],
                    "dominant_blocker": row["dominant_blocker"],
                    "source_summary": str(path),
                }
            )
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "object",
        "label",
        "mode",
        "collision_shape",
        "grasp_width_mm",
        "grasp_target_fraction_z",
        "grasp_target_top_offset_mm",
        "grasp_target_x_mm",
        "grasp_target_y_mm",
        "grasp_target_z_mm",
        "strategy",
        "valid_samples",
        "total_samples",
        "feasible_rate",
        "dominant_blocker",
        "source_summary",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if (
        args.grasp_target_z_fraction is not None
        and args.grasp_target_top_offset_mm is not None
    ):
        raise SystemExit(
            "--grasp-target-z-fraction and --grasp-target-top-offset-mm "
            "are mutually exclusive"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    rate_script = Path(__file__).with_name("run_strategy_pregrasp_rate.py")
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for object_name in args.objects:
        object_out = args.out / object_name
        cmd = [
            sys.executable,
            str(rate_script),
            "--object",
            object_name,
            "--mode",
            "object-default",
            "--preopen-mm",
            str(args.preopen_mm),
            "--iterative-pregrasp-policy",
            args.iterative_pregrasp_policy,
            "--start-sampler",
            "paper-approach-points",
            "--start-reference",
            "grasp-center",
            "--approach-axis",
            args.approach_axis,
            "--path-samples",
            str(args.path_samples),
            "--radius-scale",
            str(args.radius_scale),
            "--floor-tolerance-mm",
            str(args.floor_tolerance_mm),
            "--paper-hand-yaw-offset-deg",
            str(args.paper_hand_yaw_offset_deg),
            "--grasp-target-approach-offset-mm",
            str(args.grasp_target_approach_offset_mm),
            "--out",
            str(object_out),
        ]
        if args.grasp_target_z_fraction is not None:
            cmd.extend(
                [
                    "--grasp-target-z-fraction",
                    str(args.grasp_target_z_fraction),
                ]
            )
        if args.grasp_target_top_offset_mm is not None:
            cmd.extend(
                [
                    "--grasp-target-top-offset-mm",
                    str(args.grasp_target_top_offset_mm),
                ]
            )
        if args.no_plots:
            cmd.append("--no-plots")

        print(f"[paper-batch] {object_name}")
        result = subprocess.run(cmd, cwd=_REPO_ROOT, text=True, capture_output=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            failures.append(
                {
                    "object": object_name,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                }
            )
            continue

        rows.extend(read_summary(object_out / "summary.csv", object_name))

    write_summary(args.out / "summary.csv", rows)
    assumptions = {
        "script": "tools/run_strategy_pregrasp_paper_batch.py",
        "simulation_only": True,
        "uses_hardware": False,
        "objects": args.objects,
        "object_proxy_warning": (
            "paper_* objects use estimated tabletop box/cylinder/sphere "
            "primitives, not measured meshes. Treat rates as exploratory until "
            "dimensions are measured or replaced with meshes."
        ),
        "iterative_pregrasp_policy": args.iterative_pregrasp_policy,
        "preopen_mm": args.preopen_mm,
        "approach_axis": args.approach_axis,
        "paper_hand_yaw_offset_deg": args.paper_hand_yaw_offset_deg,
        "grasp_target_z_fraction_override": args.grasp_target_z_fraction,
        "grasp_target_top_offset_mm_override": args.grasp_target_top_offset_mm,
        "grasp_target_approach_offset_mm": args.grasp_target_approach_offset_mm,
        "path_samples": args.path_samples,
        "radius_scale": args.radius_scale,
        "floor_tolerance_mm": args.floor_tolerance_mm,
        "failures": failures,
    }
    (args.out / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")

    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
