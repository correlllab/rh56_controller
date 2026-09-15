#!/usr/bin/env python3
"""Figures for the strategy reachability analysis.

Reads the CSVs written by run_reachability_sweep.py and writes PDF (for LaTeX)
and PNG (for the HTML doc page) into ../figures.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rh56_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots 1-4 of the reference palette, in fixed order.
NAIVE, ITER, REFLEX, GGX = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ALL_STRATEGIES = ["naive", "iterative_closure", "thumb_reflex", "graspgen_x"]
COLORS = {"naive": NAIVE, "iterative_closure": ITER, "thumb_reflex": REFLEX,
          "graspgen_x": GGX}
LABELS = {"naive": "Naive", "iterative_closure": "Iterative", "thumb_reflex": "Reflex",
          "graspgen_x": "GraspGen-X"}
# Populated from the loaded CSV so a run without GGX still works.
STRATEGIES = ["naive", "iterative_closure", "thumb_reflex"]

SUFFIX = ""

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e2de"
SURFACE = "#ffffff"

# Hardware success rates from the manuscript (Sec. VI).
HARDWARE = {
    "overall": {"naive": 0.480, "iterative_closure": 0.820, "thumb_reflex": 0.867},
    "ycb_like": {"naive": 0.620, "iterative_closure": 0.760, "thumb_reflex": 0.900},
    "delicate": {"naive": 0.200, "iterative_closure": 0.940, "thumb_reflex": 0.800},
}

FAILURE_MODES = ["table_collision", "premature_object_contact", "terminal_misalignment"]
FAILURE_LABELS = {
    "table_collision": "Support-surface\ncollision",
    "premature_object_contact": "Premature object\ncontact",
    "terminal_misalignment": "Terminal contact\nmisalignment",
}


def object_geometry(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Map object -> (grasp width mm, grasp-point height above the table mm)."""

    geom: dict[str, tuple[float, float]] = {}
    for row in rows:
        geom.setdefault(row["object"],
                        (float(row["grasp_width_mm"]), float(row["grasp_height_mm"])))
    return geom


def object_tick(obj: str, geom: dict[str, tuple[float, float]]) -> str:
    """Tick label carrying both numbers.

    A bare parenthesised millimetre value next to an object name reads as the
    object's size, and the ordering variable here is the grasp-point height, so
    the two are named.
    """

    width, height = geom[obj]
    name = obj.replace("paper_", "").replace("_", " ")
    return f"{name}  {width:.0f} mm wide, {height:.0f} mm up"


def style_axes(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_dir / stem}.pdf/.png")


# ---------------------------------------------------------------- main figure
def figure_main(rows: list[dict], out_dir: Path) -> None:
    """The two-panel body figure: per-object admissibility, and failure modes.

    No hardware overlay: the simulated admissibility and the measured success
    rate are different quantities, and putting them on one axis invites the
    reader to treat the model as a success predictor. The comparison belongs in
    the text.
    """

    geom = object_geometry(rows)
    objects = {o: h for o, (_, h) in geom.items()}
    order = sorted(objects, key=lambda o: objects[o])
    counts = defaultdict(dict)
    for obj in order:
        for strategy in STRATEGIES:
            subset = [r for r in rows if r["object"] == obj and r["strategy"] == strategy]
            counts[obj][strategy] = sum(int(r["admissible"]) for r in subset)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.62, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(order))
    h = 0.78 / len(STRATEGIES)
    centre = (len(STRATEGIES) - 1) / 2.0
    for idx, strategy in enumerate(STRATEGIES):
        vals = [counts[o][strategy] for o in order]
        pos = y + (centre - idx) * h
        ax.barh(pos, vals, height=h * 0.88, color=COLORS[strategy],
                label=LABELS[strategy], linewidth=0)
        # A zero-length bar is indistinguishable from an absent one, and zero is
        # the reading that carries the result for naive closure.
        zeros = [k for k, v in enumerate(vals) if v == 0]
        if zeros:
            ax.scatter([0.055] * len(zeros), pos[zeros], marker="|", s=26,
                       color=COLORS[strategy], linewidths=1.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([object_tick(o, geom) for o in order],
                       fontsize=8.6, color=INK)
    ax.set_xlabel("Admissible approach conditions (of 10)", fontsize=9, color=INK2)
    ax.set_xlim(0, 12.6)
    ax.set_xticks(range(0, 11, 2))
    style_axes(ax, grid_axis="x")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.6, loc="lower right",
              bbox_to_anchor=(1.0, 0.02), labelcolor=INK)
    ax.set_title("A   By object, ordered by grasp-point height above the table",
                 fontsize=10, color=INK, loc="left", pad=8)

    x = np.arange(len(FAILURE_MODES))
    w = 0.78 / len(STRATEGIES)
    for idx, strategy in enumerate(STRATEGIES):
        subset = [r for r in rows if r["strategy"] == strategy]
        vals = [sum(int(r[m]) for r in subset) for m in FAILURE_MODES]
        bars = ax2.bar(x + (idx - centre) * w, vals, width=w * 0.88,
                       color=COLORS[strategy], linewidth=0)
        for rect, val in zip(bars, vals):
            ax2.text(rect.get_x() + rect.get_width() / 2, val + 2, str(val),
                     ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([FAILURE_LABELS[m] for m in FAILURE_MODES], fontsize=8.4, color=INK)
    ax2.set_ylabel(f"Trials (of {len([r for r in rows if r['strategy'] == STRATEGIES[0]])})",
                   fontsize=9, color=INK2)
    ax2.set_ylim(0, max(
        sum(int(r[m]) for r in rows if r["strategy"] == st)
        for st in STRATEGIES for m in FAILURE_MODES) * 1.28)
    style_axes(ax2)
    ax2.set_title("B   Failure modes (a trial can trigger more than one)",
                  fontsize=10, color=INK, loc="left", pad=8)

    fig.tight_layout()
    save(fig, out_dir, f"20260908-fig_main_admissibility{SUFFIX}")


# ---------------------------------------------------------------- figure A
def figure_per_object(rows: list[dict], out_dir: Path) -> None:
    geom = object_geometry(rows)
    objects = {o: h for o, (_, h) in geom.items()}
    order = sorted(objects, key=lambda o: objects[o])

    counts = defaultdict(dict)
    totals = defaultdict(dict)
    for obj in order:
        for strategy in STRATEGIES:
            subset = [r for r in rows if r["object"] == obj and r["strategy"] == strategy]
            counts[obj][strategy] = sum(int(r["admissible"]) for r in subset)
            totals[obj][strategy] = len(subset)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 5.4), gridspec_kw={"width_ratios": [2.05, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(order))
    h = 0.78 / len(STRATEGIES)
    centre = (len(STRATEGIES) - 1) / 2.0
    for idx, strategy in enumerate(STRATEGIES):
        vals = [counts[o][strategy] for o in order]
        ax.barh(
            y + (centre - idx) * h,
            vals,
            height=h * 0.88,
            color=COLORS[strategy],
            label=LABELS[strategy],
            linewidth=0,
        )
    labels = [object_tick(o, geom) for o in order]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("Admissible approach conditions (of 10)", fontsize=9, color=INK2)
    ax.set_xlim(0, 10.4)
    ax.set_xticks(range(0, 11, 2))
    style_axes(ax, grid_axis="x")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    ax.set_title(
        "Per object, ordered by grasp-point height above the table",
        fontsize=9.5, color=INK, loc="left", pad=8,
    )

    # Boundary annotation: naive is inadmissible everywhere below ~50 mm.
    naive_zero = [i for i, o in enumerate(order) if counts[o]["naive"] == 0]
    if naive_zero:
        edge = max(naive_zero) + 0.5
        ax.axhline(edge, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(
            0.15, edge + 0.30,
            "naive admits nothing for any object above this line",
            fontsize=7.6, color=MUTED, ha="left", va="top",
        )

    # Right panel: simulated admissibility vs measured hardware success.
    groups = ["overall", "ycb_like", "delicate"]
    group_labels = ["All 15", "YCB-like", "Delicate"]
    x = np.arange(len(groups))
    w = 0.78 / len(STRATEGIES)
    for idx, strategy in enumerate(STRATEGIES):
        sim = []
        for group in groups:
            if group == "overall":
                subset = [r for r in rows if r["strategy"] == strategy]
            else:
                subset = [r for r in rows if r["strategy"] == strategy and r["category"] == group]
            sim.append(sum(int(r["admissible"]) for r in subset) / len(subset))
        offset = (idx - (len(STRATEGIES) - 1) / 2.0) * w
        ax2.bar(x + offset, sim, width=w * 0.88, color=COLORS[strategy],
                linewidth=0, label=LABELS[strategy])
        # A zero-height bar is invisible; say so rather than leave a gap.
        for xi, value in zip(x, sim):
            if value < 0.005:
                ax2.text(xi + offset, 0.012, "0%", ha="center", va="bottom",
                         fontsize=7.4, color=COLORS[strategy], fontweight="bold")
        if strategy in HARDWARE["overall"]:
            hw = [HARDWARE[g][strategy] for g in groups]
            ax2.scatter(x + offset, hw, s=34, facecolor=SURFACE,
                        edgecolor=COLORS[strategy], linewidth=1.6, zorder=5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(group_labels, fontsize=8.5, color=INK)
    ax2.set_ylabel("Fraction", fontsize=9, color=INK2)
    ax2.set_ylim(0, 1.0)
    style_axes(ax2)
    ax2.set_title(
        "Bars: simulated geometric admissibility\nRings: measured hardware success",
        fontsize=9.5, color=INK, loc="left", pad=8,
    )
    save(fig, out_dir, f"20260908-fig_a_per_object_admissibility{SUFFIX}")


# ---------------------------------------------------------------- figure B
def figure_failure_modes(rows: list[dict], out_dir: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    fig.patch.set_facecolor(SURFACE)

    x = np.arange(len(FAILURE_MODES))
    w = 0.78 / len(STRATEGIES)
    for idx, strategy in enumerate(STRATEGIES):
        subset = [r for r in rows if r["strategy"] == strategy]
        vals = [sum(int(r[m]) for r in subset) for m in FAILURE_MODES]
        bars = ax.bar(x + (idx - (len(STRATEGIES) - 1) / 2.0) * w, vals, width=w * 0.88,
                      color=COLORS[strategy], linewidth=0, label=LABELS[strategy])
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, str(val),
                    ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([FAILURE_LABELS[m] for m in FAILURE_MODES], fontsize=8.2, color=INK)
    ax.set_ylabel("Trials (of 150)", fontsize=9, color=INK2)
    ax.set_ylim(0, 165)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title("Failure modes (a trial can trigger more than one)",
                 fontsize=9.5, color=INK, loc="left", pad=8)

    # By approach condition, ordered by height above the grasp point.
    cond_key = {}
    for r in rows:
        cond_key.setdefault(r["condition"], (float(r["height_mm"]), float(r["lateral_mm"])))
    conds = sorted(cond_key, key=lambda c: cond_key[c])
    for strategy in STRATEGIES:
        vals = []
        for cond in conds:
            subset = [r for r in rows if r["strategy"] == strategy and r["condition"] == cond]
            vals.append(sum(int(r["admissible"]) for r in subset) / len(subset))
        ax2.plot(range(len(conds)), vals, marker="o", markersize=5.5, linewidth=2.0,
                 color=COLORS[strategy], label=LABELS[strategy],
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax2.set_xticks(range(len(conds)))
    ax2.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=7.0, color=INK)
    ax2.set_xlim(-0.4, len(conds) - 0.6)
    ax2.set_ylabel("Admissible fraction", fontsize=9, color=INK2)
    ax2.set_ylim(-0.03, 1.0)
    style_axes(ax2)
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    ax2.set_title("By approach condition (P1 is level with the grasp point)",
                  fontsize=9.5, color=INK, loc="left", pad=8)
    save(fig, out_dir, f"20260908-fig_b_failure_modes{SUFFIX}")


# ---------------------------------------------------------------- figure C
def figure_grid_maps(rows: list[dict], out_dir: Path, objects: list[str]) -> None:
    """Admissible region in (approach height, lateral offset) per strategy."""

    cmap = LinearSegmentedColormap.from_list("adm", ["#f2f5f9", "#2a78d6"])
    fig, axes = plt.subplots(
        len(objects), len(STRATEGIES),
        figsize=(2.85 * len(STRATEGIES) + 1.1, 2.5 * len(objects) + 0.7),
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    heights = sorted({float(r["height_mm"]) for r in rows})
    laterals = sorted({float(r["lateral_mm"]) for r in rows})

    for r_idx, obj in enumerate(objects):
        obj_rows = [r for r in rows if r["object"] == obj]
        gw = float(obj_rows[0]["grasp_width_mm"])
        gz = float(obj_rows[0]["grasp_height_mm"])
        for c_idx, strategy in enumerate(STRATEGIES):
            ax = axes[r_idx][c_idx]
            grid = np.full((len(heights), len(laterals)), np.nan)
            for row in obj_rows:
                if row["strategy"] != strategy:
                    continue
                i = heights.index(float(row["height_mm"]))
                j = laterals.index(float(row["lateral_mm"]))
                grid[i, j] = int(row["admissible"])
            ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                      extent=[laterals[0], laterals[-1], heights[0], heights[-1]])
            rate = np.nanmean(grid)
            ax.set_title(
                f"{LABELS[strategy]} — {rate:.0%} admissible",
                fontsize=8.6, color=COLORS[strategy], loc="left", pad=5,
            )
            ax.tick_params(colors=INK2, labelsize=7.4, length=3, width=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            if c_idx == 0:
                ax.set_ylabel(
                    f"{obj.replace('paper_', '').replace('_', ' ')}\n"
                    f"{gw:.0f} mm wide, {gz:.0f} mm up"
                    "\n\nheight above grasp point (mm)",
                    fontsize=7.8, color=INK,
                )
            if r_idx == len(objects) - 1:
                ax.set_xlabel("lateral offset (mm)", fontsize=8, color=INK2)

    fig.suptitle(
        "Filled = the commanded approach and closure is geometrically admissible from that start pose",
        fontsize=9, color=INK2, y=1.005, x=0.02, ha="left",
    )
    fig.tight_layout()
    save(fig, out_dir, f"20260908-fig_c_admissible_regions{SUFFIX}")


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results", type=Path, default=here.parent / "results")
    ap.add_argument("--figures", type=Path, default=here.parent / "figures")
    ap.add_argument("--trials", default="paper_conditions_trials.csv")
    ap.add_argument("--suffix", default="", help="Appended to each figure stem.")
    ap.add_argument("--grid-objects", nargs="+",
                    default=["paper_nut", "paper_charger", "paper_can"])
    args = ap.parse_args()

    global STRATEGIES, SUFFIX
    SUFFIX = args.suffix
    paper = load(args.results / args.trials)
    present = {r["strategy"] for r in paper}
    STRATEGIES = [s for s in ALL_STRATEGIES if s in present]
    print(f"strategies: {', '.join(STRATEGIES)}")
    figure_main(paper, args.figures)
    figure_per_object(paper, args.figures)
    figure_failure_modes(paper, args.figures)

    grid_path = args.results / "grid_trials.csv"
    if grid_path.exists():
        figure_grid_maps(load(grid_path), args.figures, args.grid_objects)
    else:
        print(f"skipping region maps: {grid_path} not present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
