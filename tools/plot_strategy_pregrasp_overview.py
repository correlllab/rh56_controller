#!/usr/bin/env python3
"""Create paper-facing overview plots for the pre-grasp path experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS


STRATEGIES = ("naive", "iterative_closure", "thumb_reflex")
STRATEGY_LABELS = {
    "naive": "Naive",
    "iterative_closure": "Iterative",
    "thumb_reflex": "Reflex",
}
DELICATE_OBJECTS = {
    "paper_egg",
    "paper_nut",
    "paper_paper_cup",
    "paper_raspberry",
    "paper_strawberry",
}
CATEGORY_ORDER = ("All", "YCB / YCB-like", "Delicate")
BLOCKER_ORDER = (
    "path_object_collision",
    "target_object_collision",
    "path_floor_collision",
    "target_floor_collision",
)
BLOCKER_LABELS = {
    "path_object_collision": "Object during path",
    "target_object_collision": "Object at target",
    "path_floor_collision": "Floor during path",
    "target_floor_collision": "Floor at target",
}
BLOCKER_COLORS = {
    "path_object_collision": "#0072B2",
    "target_object_collision": "#D55E00",
    "path_floor_collision": "#009E73",
    "target_floor_collision": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-object pre-grasp path feasibility and failure modes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/current"),
        help="Batch artifact root containing summary.csv and object directories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/current/visualizations"),
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def object_category(object_name: str) -> str:
    return "Delicate" if object_name in DELICATE_OBJECTS else "YCB / YCB-like"


def ordered_objects(summary_rows: list[dict[str, str]]) -> list[str]:
    present = {row["object"] for row in summary_rows}
    ordered = [name for name in PAPER_OBJECTS if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def normalized_rate_rows(
    summary_rows: list[dict[str, str]], objects: list[str]
) -> list[dict[str, object]]:
    row_by_key = {(row["object"], row["strategy"]): row for row in summary_rows}
    output: list[dict[str, object]] = []
    for object_name in objects:
        for strategy in STRATEGIES:
            row = row_by_key[(object_name, strategy)]
            output.append(
                {
                    "object": object_name,
                    "label": row["label"],
                    "category": object_category(object_name),
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "valid_trials": int(row["valid_samples"]),
                    "total_trials": int(row["total_samples"]),
                    "feasible_rate": float(row["feasible_rate"]),
                }
            )
    return output


def load_volume_rows(root: Path, objects: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for object_name in objects:
        for row in read_rows(root / object_name / "volume.csv"):
            row["object"] = object_name
            row["category"] = object_category(object_name)
            rows.append(row)
    return rows


def point_summary_rows(volume_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    point_order = sorted(
        {
            (int(float(row["point_order"])), row["point_id"])
            for row in volume_rows
            if row.get("point_order") and row.get("point_id")
        }
    )
    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in volume_rows:
        for category in ("All", row["category"]):
            key = (category, row["strategy"], row["point_id"])
            counts[key][0] += int(row["valid"])
            counts[key][1] += 1

    output: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        for strategy in STRATEGIES:
            for order, point_id in point_order:
                valid, total = counts[(category, strategy, point_id)]
                output.append(
                    {
                        "category": category,
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "point_order": order,
                        "point_id": point_id,
                        "valid_trials": valid,
                        "total_trials": total,
                        "feasible_rate": valid / total if total else 0.0,
                    }
                )
    return output


def failure_summary_rows(volume_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in volume_rows:
        if int(row["valid"]):
            continue
        blocker = row["dominant_blocker"]
        if blocker not in BLOCKER_LABELS:
            blocker = "other"
        for category in ("All", row["category"]):
            counts[(category, row["strategy"], blocker)] += 1
            totals[(category, row["strategy"])] += 1

    output: list[dict[str, object]] = []
    blockers = (*BLOCKER_ORDER, "other")
    for category in CATEGORY_ORDER:
        for strategy in STRATEGIES:
            total = totals[(category, strategy)]
            for blocker in blockers:
                count = counts[(category, strategy, blocker)]
                output.append(
                    {
                        "category": category,
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "failure_mode": blocker,
                        "failure_mode_label": BLOCKER_LABELS.get(blocker, "Other"),
                        "failure_count": count,
                        "total_failures": total,
                        "failure_fraction": count / total if total else 0.0,
                    }
                )
    return output


def annotate_heatmap(ax, matrix: np.ndarray) -> None:
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                continue
            color = "black" if value >= 0.68 else "white"
            ax.text(
                col_index,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )


def plot_object_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    objects: list[str],
    dpi: int,
) -> None:
    row_by_key = {(row["object"], row["strategy"]): row for row in rows}
    labels = [str(row_by_key[(name, STRATEGIES[0])]["label"]) for name in objects]
    matrix = np.asarray(
        [
            [float(row_by_key[(name, strategy)]["feasible_rate"]) for strategy in STRATEGIES]
            for name in objects
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.4), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(STRATEGIES)), [STRATEGY_LABELS[name] for name in STRATEGIES])
    ax.set_yticks(range(len(objects)), labels)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Collision-free approach rate by object and pre-grasp strategy", pad=12)
    ax.set_xlabel("Fixed pre-grasp hand shape")
    delicate_start = next(
        (index for index, name in enumerate(objects) if name in DELICATE_OBJECTS),
        len(objects),
    )
    if 0 < delicate_start < len(objects):
        ax.axhline(delicate_start - 0.5, color="white", linewidth=2.5)
    annotate_heatmap(ax, matrix)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Feasible trials / 10 starts")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def short_point_label(point_id: str) -> str:
    if "_d" in point_id:
        level, offset = point_id.split("_d", maxsplit=1)
        return f"{level}\n{offset}"
    return point_id


def plot_point_heatmaps(path: Path, rows: list[dict[str, object]], dpi: int) -> None:
    point_order = sorted(
        {(int(row["point_order"]), str(row["point_id"])) for row in rows}
    )
    point_ids = [point_id for _, point_id in point_order]
    row_by_key = {
        (row["category"], row["strategy"], row["point_id"]): row for row in rows
    }

    fig, axes = plt.subplots(
        len(CATEGORY_ORDER),
        1,
        figsize=(11.5, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    image = None
    for ax, category in zip(axes, CATEGORY_ORDER):
        matrix = np.asarray(
            [
                [
                    float(row_by_key[(category, strategy, point_id)]["feasible_rate"])
                    for point_id in point_ids
                ]
                for strategy in STRATEGIES
            ]
        )
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(STRATEGIES)), [STRATEGY_LABELS[name] for name in STRATEGIES])
        ax.tick_params(axis="both", length=0)
        ax.set_title(category, loc="left", fontsize=10, fontweight="bold")
        annotate_heatmap(ax, matrix)

    axes[-1].set_xticks(range(len(point_ids)), [short_point_label(name) for name in point_ids])
    axes[-1].set_xlabel("Sampled grasp-center start point")
    fig.suptitle("Approach feasibility by start point", fontsize=14)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Feasible fraction across objects")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_failure_modes(path: Path, rows: list[dict[str, object]], dpi: int) -> None:
    row_by_key = {
        (row["category"], row["strategy"], row["failure_mode"]): row for row in rows
    }
    fig, axes = plt.subplots(1, len(CATEGORY_ORDER), figsize=(13.0, 4.4), constrained_layout=True)
    for ax, category in zip(axes, CATEGORY_ORDER):
        left = np.zeros(len(STRATEGIES), dtype=float)
        for blocker in BLOCKER_ORDER:
            values = np.asarray(
                [
                    int(row_by_key[(category, strategy, blocker)]["failure_count"])
                    for strategy in STRATEGIES
                ],
                dtype=float,
            )
            bars = ax.barh(
                range(len(STRATEGIES)),
                values,
                left=left,
                color=BLOCKER_COLORS[blocker],
                label=BLOCKER_LABELS[blocker],
            )
            for bar, value in zip(bars, values):
                if value >= 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
            left += values
        ax.set_yticks(range(len(STRATEGIES)), [STRATEGY_LABELS[name] for name in STRATEGIES])
        ax.invert_yaxis()
        ax.set_title(category, fontsize=10, fontweight="bold")
        ax.set_xlabel("Failed trials")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Why collision-free approaches fail", fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_summary = args.root / "summary.csv"
    summary_rows = read_rows(source_summary)
    objects = ordered_objects(summary_rows)
    rate_rows = normalized_rate_rows(summary_rows, objects)
    volume_rows = load_volume_rows(args.root, objects)
    point_rows = point_summary_rows(volume_rows)
    failure_rows = failure_summary_rows(volume_rows)

    write_rows(
        args.out / "summary.csv",
        rate_rows,
        [
            "object",
            "label",
            "category",
            "strategy",
            "strategy_label",
            "valid_trials",
            "total_trials",
            "feasible_rate",
        ],
    )
    write_rows(
        args.out / "start_point_summary.csv",
        point_rows,
        [
            "category",
            "strategy",
            "strategy_label",
            "point_order",
            "point_id",
            "valid_trials",
            "total_trials",
            "feasible_rate",
        ],
    )
    write_rows(
        args.out / "failure_mode_summary.csv",
        failure_rows,
        [
            "category",
            "strategy",
            "strategy_label",
            "failure_mode",
            "failure_mode_label",
            "failure_count",
            "total_failures",
            "failure_fraction",
        ],
    )

    plot_object_heatmap(args.out / "object_strategy_feasible_rate.png", rate_rows, objects, args.dpi)
    plot_point_heatmaps(args.out / "start_point_feasible_rate.png", point_rows, args.dpi)
    plot_failure_modes(args.out / "failure_modes.png", failure_rows, args.dpi)

    assumptions = {
        "script": "tools/plot_strategy_pregrasp_overview.py",
        "simulation_only": True,
        "uses_hardware": False,
        "source_summary": str(source_summary),
        "source_volumes": [str(args.root / name / "volume.csv") for name in objects],
        "objects": objects,
        "strategies": list(STRATEGIES),
        "trial_count": len(volume_rows),
        "interpretation": {
            "object_strategy_feasible_rate": (
                "Each cell is the fraction of the 10 sampled starts whose fixed "
                "pre-grasp hand shape reaches the target without object or floor collision."
            ),
            "start_point_feasible_rate": (
                "Each cell pools individual binary trials across objects in the named category."
            ),
            "failure_modes": (
                "Each failed trial contributes once using its dominant_blocker label."
            ),
            "closure": "Finger closure after arrival is neither rendered nor scored.",
        },
    }
    (args.out / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")

    print("RH56 strategy pre-grasp overview:")
    print(f"  objects: {len(objects)}")
    print(f"  trials: {len(volume_rows)}")
    for name in (
        "object_strategy_feasible_rate.png",
        "start_point_feasible_rate.png",
        "failure_modes.png",
        "summary.csv",
        "start_point_summary.csv",
        "failure_mode_summary.csv",
        "assumptions.json",
    ):
        print(f"Wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
