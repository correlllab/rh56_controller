#!/usr/bin/env python3
"""Audit cached GraspGen-X target poses for open-hand table clearance.

This is a pre-execution feasibility diagnostic, not a grasp-success test.  It
is especially useful for small tabletop objects, where an object-centric 6-D
grasp pose can put the RH56 palm or inactive fingers through the support plane.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

from rh56_controller.graspgenx_baseline import (  # noqa: E402
    graspgenx_to_mujoco_base_pose,
    load_isaac_grasp_yaml,
)
from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
)
from tools.run_graspgenx_success_comparison import (  # noqa: E402
    BASE_ACTUATORS,
    FINGER_ACTUATORS,
    GGX_OPEN_CTRL,
    PoseCommand,
    _actuator_ids,
    _floor_hand_collision,
    _initialize_state,
    _object_hand_contacts,
)
from tools.run_paper_15_object_grasp_success import (  # noqa: E402
    DEFAULT_XML,
    PAPER_OBJECT_MASS_KG,
    _add_object_model,
)
from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify cached GraspGen-X candidates as collision-free, "
            "floor-colliding, object-contacting, or both at the open target pose."
        )
    )
    parser.add_argument("--objects", nargs="+", choices=PAPER_OBJECTS, default=PAPER_OBJECTS)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/graspgenx_candidates"),
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/graspgenx_table_clearance"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["grasp_width_mm"]))
    labels = [str(row["label"]) for row in ordered]
    total = np.array([float(row["candidate_count"]) for row in ordered])
    categories = [
        ("clear_count", "Collision-free", "#2e7d32"),
        ("floor_only_count", "Floor only", "#ef6c00"),
        ("object_only_count", "Object only", "#1976d2"),
        ("floor_and_object_count", "Floor + object", "#c62828"),
    ]
    figure, axis = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    bottom = np.zeros(len(ordered), dtype=float)
    x = np.arange(len(ordered))
    for key, label, color in categories:
        values = np.array([float(row[key]) for row in ordered]) / total
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of cached candidates")
    axis.set_title("GraspGen-X open-hand target feasibility on a tabletop")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}

    for object_name in args.objects:
        obj = BUILTIN_OBJECTS[object_name]
        yaml_path = args.candidate_dir / f"{object_name}.yml"
        candidates = load_isaac_grasp_yaml(yaml_path)
        model, object_geom_id, object_qadr = _add_object_model(
            args.xml,
            obj,
            PAPER_OBJECT_MASS_KG[object_name],
        )
        data = mujoco.MjData(model)
        base_ids = _actuator_ids(model, BASE_ACTUATORS)
        finger_ids = _actuator_ids(model, FINGER_ACTUATORS)
        center = tabletop_aabb_center(obj)
        counts = {
            "clear": 0,
            "floor_only": 0,
            "object_only": 0,
            "floor_and_object": 0,
        }
        clear_ranks: list[int] = []
        for rank, candidate in enumerate(candidates):
            position, rotation = graspgenx_to_mujoco_base_pose(
                candidate,
                object_center=center,
            )
            _initialize_state(
                model,
                data,
                object_qadr,
                center,
                PoseCommand(position, rotation, dict(GGX_OPEN_CTRL)),
                base_ids,
                finger_ids,
            )
            floor = _floor_hand_collision(model, data, object_geom_id)
            object_contact = bool(_object_hand_contacts(model, data, object_geom_id))
            if floor and object_contact:
                counts["floor_and_object"] += 1
            elif floor:
                counts["floor_only"] += 1
            elif object_contact:
                counts["object_only"] += 1
            else:
                counts["clear"] += 1
                clear_ranks.append(rank)

        candidate_count = len(candidates)
        rows.append(
            {
                "object": object_name,
                "label": obj.label,
                "grasp_width_mm": obj.grasp_width_m * 1000.0,
                "minimum_extent_mm": min(obj.size_m) * 1000.0,
                "candidate_count": candidate_count,
                "clear_count": counts["clear"],
                "clear_fraction": counts["clear"] / candidate_count,
                "first_clear_rank": clear_ranks[0] if clear_ranks else "",
                "floor_only_count": counts["floor_only"],
                "object_only_count": counts["object_only"],
                "floor_and_object_count": counts["floor_and_object"],
            }
        )
        hashes[object_name] = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
        print(
            f"{obj.label:18s}: {counts['clear']:3d}/{candidate_count} open-hand "
            "targets clear the object and table"
        )

    summary_path = args.out / "candidate_clearance.csv"
    _write_csv(summary_path, rows)
    if not args.no_plot:
        _plot(args.out / "candidate_clearance.png", rows)
    metadata = {
        "scope": "pre-execution open-hand collision feasibility only",
        "not_success_rate": True,
        "object_proxy_warning": (
            "Objects are the current estimated primitive proxies, so these results "
            "characterize the present pipeline rather than physical objects."
        ),
        "candidate_selection_implication": (
            "A zero clear count means the object-only candidate set needs scene-aware "
            "generation, grasp repair, or a different pre-grasp hand shape before dynamics."
        ),
        "candidate_sha256": hashes,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out / "assumptions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
