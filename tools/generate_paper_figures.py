#!/usr/bin/env python3
"""Paper-v2 figure command manifest.

By default this script prints the reproducible commands rather than running all
experiments. Use --run-smoke to execute a quick, low-cost generation pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SMOKE_COMMANDS = [
    [
        sys.executable,
        "tools/run_planner_sweep.py",
        "--modes",
        "line",
        "plane3",
        "plane4",
        "plane5",
        "--width-min-mm",
        "5",
        "--width-max-mm",
        "35",
        "--width-step-mm",
        "15",
        "--out",
        "artifacts/archive/smoke/planner_sweep_smoke",
    ],
    [
        sys.executable,
        "tools/run_analytical_grasp_volume.py",
        "--objects",
        "debug_40mm_cube",
        "--x-range-mm",
        "-80",
        "80",
        "--y-range-mm",
        "-80",
        "80",
        "--z-range-mm",
        "80",
        "160",
        "--grid-step-mm",
        "80",
        "--yaw-samples",
        "4",
        "--path-samples",
        "4",
        "--max-linear-move-mm",
        "280",
        "--out",
        "artifacts/archive/smoke/analytical_grasp_volume_capsule_smoke",
    ],
    [
        sys.executable,
        "tools/run_hybrid_margin_sweep.py",
        "--margin-list",
        "0",
        "25",
        "--v-contact-list",
        "25",
        "100",
        "--out",
        "artifacts/archive/smoke/hybrid_margin_sweep_smoke",
    ],
    [
        sys.executable,
        "tools/replay_force_thresholds.py",
        "--out",
        "artifacts/archive/smoke/threshold_replay_smoke",
    ],
]

FULL_COMMANDS = [
    "python tools/run_planner_sweep.py --modes line plane3 plane4 plane5 --width-min-mm 5 --width-max-mm 115 --width-step-mm 1 --object-width-offset-mm 20 --out artifacts/planner_sweep/",
    "python tools/run_analytical_grasp_volume.py --objects ycb_cracker_box ycb_sugar_box ycb_potted_meat_can --collision-model capsule --path-hand-shape open --x-range-mm -240 240 --y-range-mm -240 240 --z-range-mm 40 280 --grid-step-mm 60 --yaw-samples 8 --path-samples 8 --out artifacts/analytical_grasp_volume/",
    "python tools/run_strategy_pregrasp_rate.py --object debug_40mm_cube --mode plane4 --out artifacts/strategy_pregrasp_rate_40mm/",
    "python tools/run_hybrid_margin_sweep.py --v-fast 1000 --v-contact-list 10 25 50 100 --margin-list 0 5 10 15 20 25 30 40 50 --onset-sigma-units 7.5 --out artifacts/hybrid_margin_sweep/",
    "python tools/replay_force_thresholds.py --logs 'experiment_data/peg_in_hole/*.csv' --out artifacts/threshold_replay/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or smoke-run paper-v2 figure commands.")
    parser.add_argument("--run-smoke", action="store_true", help="Run low-cost smoke commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [
        path
        for path in (
            Path("tools/run_planner_sweep.py"),
            Path("tools/run_analytical_grasp_volume.py"),
            Path("tools/run_strategy_pregrasp_rate.py"),
            Path("tools/run_hybrid_margin_sweep.py"),
            Path("tools/replay_force_thresholds.py"),
        )
        if not path.exists()
    ]
    if missing:
        print("Missing figure script(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    if not args.run_smoke:
        print("Paper-v2 figure commands:")
        for command in FULL_COMMANDS:
            print(f"  {command}")
        print("\nUse --run-smoke to execute a quick validation pass.")
        return 0

    for command in SMOKE_COMMANDS:
        print("+ " + " ".join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
