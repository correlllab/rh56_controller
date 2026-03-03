#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).parent
DEFAULT_OUT_STEM = Path(__file__).with_name("speed_sweep_contact_force")
FORCE_LIMIT_G = 500.0
FIGSIZE = (3.5, 2.45)
DPI = 300


def parse_args() -> argparse.Namespace:
    default_csv = find_latest_csv()
    parser = argparse.ArgumentParser(
        description="Plot contact-stage force traces for the speed sweep experiment."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Input CSV file. Default: {default_csv.name}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_STEM,
        help="Output path stem, without extension.",
    )
    parser.add_argument(
        "--force-limit",
        type=float,
        default=FORCE_LIMIT_G,
        help="Horizontal reference line in raw force units.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure window after saving.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "lines.linewidth": 1.6,
        }
    )


def find_latest_csv() -> Path:
    candidates = sorted(DATA_DIR.glob("middlefinger_speed_sweep_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No sweep CSV found in {DATA_DIR}. Expected middlefinger_speed_sweep_*.csv"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_contact_data(csv_path: Path) -> list[tuple[str, pd.DataFrame]]:
    df = pd.read_csv(csv_path)
    df["Timestamp_Epoch"] = pd.to_numeric(df["Timestamp_Epoch"], errors="coerce")
    df["Commanded_Speed"] = pd.to_numeric(df["Commanded_Speed"], errors="coerce")
    df["Middle_Force_g"] = pd.to_numeric(df["Middle_Force_g"], errors="coerce")

    df = df.dropna(subset=["Timestamp_Epoch", "Middle_Force_g"])
    df = df[df["Stage"] != "open"].copy()

    traces: list[tuple[str, pd.DataFrame]] = []

    speed_df = df[df["Mode"] == "speed"].copy()
    for speed in sorted(speed_df["Commanded_Speed"].dropna().unique(), reverse=True):
        segment = speed_df[speed_df["Commanded_Speed"] == speed].sort_values(
            "Timestamp_Epoch"
        )
        if segment.empty:
            continue
        traces.append((fr"$v = {int(speed)}$", rebase_time(segment)))

    hybrid_df = df[df["Mode"] == "hybrid"].sort_values("Timestamp_Epoch")
    if not hybrid_df.empty:
        traces.append(("Hybrid", rebase_time(hybrid_df)))

    return traces


def rebase_time(segment: pd.DataFrame) -> pd.DataFrame:
    segment = segment.copy()
    segment["Time_s"] = segment["Timestamp_Epoch"] - segment["Timestamp_Epoch"].iloc[0]
    return segment


def plot_contact_traces(
    traces: list[tuple[str, pd.DataFrame]], out_stem: Path, force_limit: float
) -> None:
    if not traces:
        raise ValueError("No contact-stage traces found to plot.")

    configure_style()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for label, segment in traces:
        ax.plot(segment["Time_s"], segment["Middle_Force_g"], label=label)

    ax.axhline(
        force_limit,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Force Limit ({force_limit:.0f})",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (raw units)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        framealpha=0.95,
        borderpad=0.35,
        handlelength=2.0,
        columnspacing=0.9,
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pgf"), bbox_inches="tight")


def main() -> None:
    args = parse_args()
    traces = load_contact_data(args.csv)
    plot_contact_traces(traces, out_stem=args.out, force_limit=args.force_limit)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
