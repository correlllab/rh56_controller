#!/usr/bin/env python3
"""Replay peg-in-hole force-threshold choices from stored logs.

When logs are absent, this script writes the expected schema and an empty
summary table. It never fabricates validation results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)


REQUIRED_COLUMNS = [
    "trial_id",
    "time_s",
    "index_force_raw",
    "middle_force_raw",
    "ring_force_raw",
    "pinky_force_raw",
    "thumb_force_raw",
    "phase",
    "label_success",
    "label_release_time_s",
]

SUMMARY_FIELDS = [
    "contact_spike_threshold",
    "lateral_spike_threshold",
    "window_s",
    "n_trials",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "precision",
    "recall",
    "mean_detection_delay_s",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay RH56 force thresholds from peg-in-hole logs without hardware."
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=[],
        help="CSV logs or glob patterns matching expected_log_schema.csv.",
    )
    parser.add_argument(
        "--contact-spike-list",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
    )
    parser.add_argument(
        "--lateral-spike-list",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
    )
    parser.add_argument(
        "--window-list",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/threshold_replay"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def expand_logs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted({path for path in paths if path.exists() and path.is_file()})


def write_schema(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_COLUMNS)
        writer.writerow(
            [
                "trial_001",
                "0.000",
                "0",
                "0",
                "0",
                "0",
                "0",
                "approach|contact|insert|release",
                "true",
                "1.250",
            ]
        )


def write_empty_summary(path: Path, notes: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "contact_spike_threshold": "",
                "lateral_spike_threshold": "",
                "window_s": "",
                "n_trials": 0,
                "true_positive": 0,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0,
                "precision": "",
                "recall": "",
                "mean_detection_delay_s": "",
                "notes": notes,
            }
        )


def write_metadata(path: Path, args: argparse.Namespace, logs: list[Path]) -> None:
    payload = {
        "script": "tools/replay_force_thresholds.py",
        "simulation_only": True,
        "uses_hardware": False,
        "input_logs": [str(path) for path in logs],
        "required_columns": REQUIRED_COLUMNS,
        "assumptions": {
            "baseline": "first sample in each trial",
            "contact_signal": "max index/middle/ring/pinky/thumb force minus baseline",
            "lateral_signal": "max middle/ring/pinky force minus baseline",
            "detection": "first moving-average sample exceeding either threshold",
            "positive_label": "label_release_time_s is present",
        },
        "sweep": {
            "contact_spike_list": args.contact_spike_list,
            "lateral_spike_list": args.lateral_spike_list,
            "window_list": args.window_list,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_trials(paths: list[Path]) -> dict[str, list[dict[str, object]]]:
    trials: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
            for row in reader:
                trial_id = str(row["trial_id"])
                row["_source"] = str(path)
                trials[trial_id].append(row)
    for rows in trials.values():
        rows.sort(key=lambda row: float(row["time_s"]))
    return trials


def bool_label(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "success"}


def maybe_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def moving_average_detection(
    rows: list[dict[str, object]],
    contact_threshold: float,
    lateral_threshold: float,
    window_s: float,
) -> float | None:
    if not rows:
        return None
    force_cols = [
        "index_force_raw",
        "middle_force_raw",
        "ring_force_raw",
        "pinky_force_raw",
        "thumb_force_raw",
    ]
    lateral_cols = ["middle_force_raw", "ring_force_raw", "pinky_force_raw"]
    baseline = {col: float(rows[0][col]) for col in force_cols}
    window: deque[tuple[float, float, float]] = deque()

    for row in rows:
        t = float(row["time_s"])
        contact_signal = max(float(row[col]) - baseline[col] for col in force_cols)
        lateral_signal = max(float(row[col]) - baseline[col] for col in lateral_cols)
        window.append((t, contact_signal, lateral_signal))
        while window and t - window[0][0] > window_s:
            window.popleft()
        avg_contact = sum(item[1] for item in window) / len(window)
        avg_lateral = sum(item[2] for item in window) / len(window)
        if avg_contact >= contact_threshold or avg_lateral >= lateral_threshold:
            return t
    return None


def replay(args: argparse.Namespace, trials: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for contact_threshold in args.contact_spike_list:
        for lateral_threshold in args.lateral_spike_list:
            for window_s in args.window_list:
                tp = fp = tn = fn = 0
                delays: list[float] = []
                for rows in trials.values():
                    first = rows[0]
                    label_time = maybe_float(first["label_release_time_s"])
                    positive = label_time is not None
                    detected_time = moving_average_detection(
                        rows, contact_threshold, lateral_threshold, window_s
                    )
                    detected = detected_time is not None
                    if positive and detected:
                        tp += 1
                        delays.append(float(detected_time) - float(label_time))
                    elif positive and not detected:
                        fn += 1
                    elif not positive and detected:
                        fp += 1
                    else:
                        tn += 1
                precision = tp / (tp + fp) if (tp + fp) else None
                recall = tp / (tp + fn) if (tp + fn) else None
                mean_delay = sum(delays) / len(delays) if delays else None
                summary.append(
                    {
                        "contact_spike_threshold": f"{contact_threshold:.6f}",
                        "lateral_spike_threshold": f"{lateral_threshold:.6f}",
                        "window_s": f"{window_s:.6f}",
                        "n_trials": len(trials),
                        "true_positive": tp,
                        "false_positive": fp,
                        "true_negative": tn,
                        "false_negative": fn,
                        "precision": "" if precision is None else f"{precision:.6f}",
                        "recall": "" if recall is None else f"{recall:.6f}",
                        "mean_detection_delay_s": "" if mean_delay is None else f"{mean_delay:.6f}",
                        "notes": "replayed_from_logs",
                    }
                )
    return summary


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plot(out_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(
            f"matplotlib unavailable; plots skipped: {exc}\n"
        )
        return
    if not rows:
        return

    xs = [float(row["contact_spike_threshold"]) for row in rows]
    ys = [float(row["recall"]) if row["recall"] != "" else 0.0 for row in rows]
    colors = [float(row["window_s"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(xs, ys, c=colors, cmap="viridis", s=18)
    ax.set_xlabel("Contact spike threshold [raw units]")
    ax.set_ylabel("Recall")
    ax.set_title("Force-threshold replay sensitivity")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Window [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_recall.png", dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logs = expand_logs(args.logs)
    write_metadata(args.out / "assumptions.json", args, logs)

    if not logs:
        write_schema(args.out / "expected_log_schema.csv")
        write_empty_summary(
            args.out / "summary.csv",
            "no_logs_found_expected_schema_written",
        )
        print("No replay logs found.")
        print(f"Wrote {args.out / 'expected_log_schema.csv'}")
        print(f"Wrote {args.out / 'summary.csv'}")
        return 0

    trials = load_trials(logs)
    rows = replay(args, trials)
    write_summary(args.out / "summary.csv", rows)
    if not args.no_plots:
        maybe_write_plot(args.out, rows)

    print(f"Loaded {len(logs)} log file(s), {len(trials)} trial(s)")
    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
