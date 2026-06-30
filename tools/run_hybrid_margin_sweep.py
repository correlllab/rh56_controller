#!/usr/bin/env python3
"""Run a transparent latency-aware hybrid margin sweep.

The model is intentionally simple: it turns measured latency and contact-onset
uncertainty into a reproducible sensitivity table. It is a paper-facing
post-hoc model, not a replacement for real hardware validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

DEFAULT_LATENCY_MS = 66.0
LATENCY_STAT_KEYS = {
    "mean": "latency_mean_s",
    "p50": "p50",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep RH56 hybrid switch margins and contact speeds without hardware."
    )
    parser.add_argument("--v-fast", type=float, default=1000.0)
    parser.add_argument(
        "--v-contact-list",
        type=float,
        nargs="+",
        default=[10.0, 25.0, 50.0, 100.0],
    )
    parser.add_argument(
        "--margin-list",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=None,
        help=(
            "Override command-to-sensing latency in ms. By default the script "
            "loads --latency-summary and uses --latency-stat."
        ),
    )
    parser.add_argument(
        "--latency-summary",
        type=Path,
        default=Path("resource/experiment/latency_summary.json"),
        help="Measured latency summary JSON. Ignored when --latency-ms is passed.",
    )
    parser.add_argument(
        "--latency-stat",
        choices=sorted(LATENCY_STAT_KEYS),
        default="p50",
        help="Latency statistic to use from --latency-summary.",
    )
    parser.add_argument("--onset-sigma-units", type=float, default=7.5)
    parser.add_argument(
        "--units-per-speed-second",
        type=float,
        default=1.0,
        help="Assumed command-position units moved per speed unit per second.",
    )
    parser.add_argument(
        "--nominal-move-units",
        type=float,
        default=500.0,
        help="Nominal free-space closure distance for the completion-time proxy.",
    )
    parser.add_argument("--robust-probability", type=float, default=0.99)
    parser.add_argument(
        "--max-overshoot-proxy",
        type=float,
        default=5.0,
        help="Maximum expected latency travel units for robust_region=true.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/hybrid_margin_sweep"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_latency(args: argparse.Namespace) -> tuple[float, dict[str, object]]:
    if args.latency_ms is not None:
        return args.latency_ms, {
            "source": "cli",
            "latency_ms": args.latency_ms,
            "note": "explicit --latency-ms override",
        }

    key = LATENCY_STAT_KEYS[args.latency_stat]
    if args.latency_summary.exists():
        payload = json.loads(args.latency_summary.read_text())
        if key not in payload:
            raise ValueError(f"{args.latency_summary} does not contain {key}")
        latency_s = float(payload[key])
        return latency_s * 1000.0, {
            "source": str(args.latency_summary),
            "stat": args.latency_stat,
            "json_key": key,
            "latency_s": latency_s,
            "latency_ms": latency_s * 1000.0,
            "trials": payload.get("trials"),
            "valid_count": payload.get("valid_count"),
            "movement_eps": payload.get("movement_eps"),
            "speed": payload.get("speed"),
        }

    return DEFAULT_LATENCY_MS, {
        "source": "fallback_default",
        "latency_ms": DEFAULT_LATENCY_MS,
        "requested_summary": str(args.latency_summary),
        "note": "latency summary not found",
    }


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "v_fast",
        "v_contact",
        "margin_units",
        "latency_ms",
        "onset_sigma_units",
        "p_pre_slow",
        "latency_travel_units",
        "overshoot_proxy",
        "time_proxy",
        "robust_region",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_assumptions(path: Path, args: argparse.Namespace) -> None:
    payload = {
        "script": "tools/run_hybrid_margin_sweep.py",
        "simulation_only": True,
        "uses_hardware": False,
        "model_status": "post-hoc sensitivity model; not a replacement for hardware validation",
        "measured_constants": {
            "latency_ms": args.latency_ms,
            "latency": args.latency_metadata,
        },
        "assumptions": {
            "onset_sigma_units": args.onset_sigma_units,
            "contact_onset_error_distribution": "zero-mean Gaussian in command units",
            "p_pre_slow": "normal_cdf(margin_units / onset_sigma_units)",
            "latency_travel_units": (
                "expected distance traveled during latency, mixing contact speed and "
                "fast speed by p_pre_slow"
            ),
            "overshoot_proxy": "same as expected latency travel units",
            "time_proxy": (
                "max(nominal_move_units - margin_units, 0) / v_fast + "
                "min(margin_units, nominal_move_units) / v_contact"
            ),
            "units_per_speed_second": args.units_per_speed_second,
            "nominal_move_units": args.nominal_move_units,
        },
        "robust_region_definition": {
            "p_pre_slow_at_least": args.robust_probability,
            "overshoot_proxy_at_most": args.max_overshoot_proxy,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def maybe_write_plots(out_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(
            f"matplotlib unavailable; plots skipped: {exc}\n"
        )
        return

    margins = sorted({float(row["margin_units"]) for row in rows})
    speeds = sorted({float(row["v_contact"]) for row in rows})
    grid = np.zeros((len(speeds), len(margins)))
    robust_grid = np.zeros((len(speeds), len(margins)), dtype=bool)
    prob_grid = np.zeros((len(speeds), len(margins)))
    for row in rows:
        i = speeds.index(float(row["v_contact"]))
        j = margins.index(float(row["margin_units"]))
        grid[i, j] = float(row["overshoot_proxy"])
        prob_grid[i, j] = float(row["p_pre_slow"])
        robust_grid[i, j] = str(row["robust_region"]).strip().lower() == "true"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(margins)), [f"{m:g}" for m in margins], rotation=45)
    ax.set_yticks(range(len(speeds)), [f"{s:g}" for s in speeds])
    ax.set_xlabel("Switch margin [command units]")
    ax.set_ylabel("Contact speed [raw speed units]")
    ax.set_title("Predicted post-switch latency travel (lower is better)")
    for i, speed in enumerate(speeds):
        for j, margin in enumerate(margins):
            if robust_grid[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.8))
            if abs(speed - 25.0) < 1e-9 and abs(margin - 25.0) < 1e-9:
                ax.plot(j, i, marker="*", ms=15, color="#ffcc33", mec="black", mew=0.8)
                ax.text(j + 0.25, i + 0.18, "25/25", color="white", fontsize=8, weight="bold")
    ax.text(
        0.01,
        -0.24,
        "White boxes meet the robust-region rule; star marks the paper default margin/contact speed.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.colorbar(im, ax=ax, label="Expected travel during latency [command units]")
    fig.tight_layout()
    fig.savefig(out_dir / "margin_speed_heatmap.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for speed in speeds:
        subset = [row for row in rows if float(row["v_contact"]) == speed]
        subset.sort(key=lambda row: float(row["time_proxy"]))
        xs = [float(row["time_proxy"]) for row in subset]
        ys = [float(row["overshoot_proxy"]) for row in subset]
        robust = [str(row["robust_region"]).strip().lower() == "true" for row in subset]
        line = ax.plot(xs, ys, marker="o", label=f"v_contact={speed:g}")[0]
        color = line.get_color()
        ax.scatter(
            [x for x, ok in zip(xs, robust) if ok],
            [y for y, ok in zip(ys, robust) if ok],
            facecolors="none",
            edgecolors=color,
            linewidths=2.0,
            s=90,
        )
        for row in subset:
            if abs(float(row["v_contact"]) - 25.0) < 1e-9 and abs(float(row["margin_units"]) - 25.0) < 1e-9:
                x = float(row["time_proxy"])
                y = float(row["overshoot_proxy"])
                ax.plot(x, y, marker="*", ms=16, color="#ffcc33", mec="black", mew=0.8)
                ax.annotate("default 25/25", (x, y), xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Completion-time proxy [s]")
    ax.set_ylabel("Overshoot proxy [command units]")
    ax.set_title("Speed-margin tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    probs = prob_grid[0, :]
    ax.plot(margins, probs, marker="o", color="#2f6f9f")
    ax.axvline(25.0, color="#8a4f2a", ls="--", lw=1.2)
    ax.axhline(0.99, color="#777777", ls=":", lw=1.2)
    ax.annotate("25-unit margin", (25.0, np.interp(25.0, margins, probs)), xytext=(8, -22), textcoords="offset points", fontsize=8)
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Switch margin [command units]")
    ax.set_ylabel("P(slow mode before contact)")
    ax.set_title("Anticipatory switch probability from contact-onset uncertainty")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pre_slow_probability.png", dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.latency_ms, args.latency_metadata = resolve_latency(args)

    if args.onset_sigma_units <= 0:
        raise ValueError("--onset-sigma-units must be positive")
    if args.v_fast <= 0 or any(v <= 0 for v in args.v_contact_list):
        raise ValueError("speeds must be positive")

    print("RH56 hybrid margin sweep assumptions:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  latency_ms: {args.latency_ms:g}")
    print(f"  latency_source: {args.latency_metadata['source']}")
    print(f"  onset_sigma_units: {args.onset_sigma_units:g}")
    print(f"  output: {args.out}")

    latency_s = args.latency_ms / 1000.0
    rows: list[dict[str, object]] = []
    for v_contact in args.v_contact_list:
        contact_latency_travel = v_contact * args.units_per_speed_second * latency_s
        fast_latency_travel = args.v_fast * args.units_per_speed_second * latency_s
        for margin in args.margin_list:
            p_pre_slow = normal_cdf(margin / args.onset_sigma_units)
            expected_latency_travel = (
                p_pre_slow * contact_latency_travel
                + (1.0 - p_pre_slow) * fast_latency_travel
            )
            fast_distance = max(args.nominal_move_units - margin, 0.0)
            slow_distance = min(max(margin, 0.0), args.nominal_move_units)
            time_proxy = (
                fast_distance / (args.v_fast * args.units_per_speed_second)
                + slow_distance / (v_contact * args.units_per_speed_second)
            )
            robust = (
                p_pre_slow >= args.robust_probability
                and expected_latency_travel <= args.max_overshoot_proxy
            )
            rows.append(
                {
                    "v_fast": f"{args.v_fast:.6f}",
                    "v_contact": f"{v_contact:.6f}",
                    "margin_units": f"{margin:.6f}",
                    "latency_ms": f"{args.latency_ms:.6f}",
                    "onset_sigma_units": f"{args.onset_sigma_units:.6f}",
                    "p_pre_slow": f"{p_pre_slow:.9f}",
                    "latency_travel_units": f"{expected_latency_travel:.6f}",
                    "overshoot_proxy": f"{expected_latency_travel:.6f}",
                    "time_proxy": f"{time_proxy:.6f}",
                    "robust_region": robust,
                }
            )

    write_csv(args.out / "summary.csv", rows)
    write_assumptions(args.out / "assumptions.json", args)
    if not args.no_plots:
        maybe_write_plots(args.out, rows)

    robust_count = sum(1 for row in rows if row["robust_region"] is True)
    print(f"Wrote {args.out / 'summary.csv'} ({robust_count}/{len(rows)} robust rows)")
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
