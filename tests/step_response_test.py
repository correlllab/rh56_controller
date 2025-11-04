#!/usr/bin/env python3
"""
RH56 single-finger step-response capture and analysis (supports code-defined step plans, multi-run comparison, optional plotting).

- Samples are recorded to CSV files.
- Metrics: 10–90% rise time, settling (window/strict), first-peak/global overshoot, RMSE (full vs target; steady vs reference), steady-state error.
- Reference selection: target or final.
- Multi-run modes (zip or cross) save per-run CSVs, plus a combined plot and summary CSV.
- Plotting: single figure with multiple traces; toggles and paths configured at the top of this file.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

# ====== (1) Adjust defaults here ======
DEFAULT_PORT = "/dev/tty.usbserial-2130"
DEFAULT_FINGER = 5
DEFAULT_TARGET_ANGLE = 1000
DEFAULT_SPEED = 1000
DEFAULT_DURATION = 4.0
DEFAULT_SAMPLE_RATE = 165.0
DEFAULT_OUTPUT_PATH = Path("./logs/thumb_rotate_step.csv")
DEFAULT_SUMMARY_PATH = Path("./logs/summary_thumb_rotate.csv")

# Preparation step (fully open hand)
DEFAULT_PREP_ANGLE = 1000
DEFAULT_PREP_SPEED = 1000
DEFAULT_PREP_WAIT = 1.0
DEFAULT_REPREP_BETWEEN = True   # Re-open before each step when running multiple tests

# Metric calculation options
DEFAULT_REFERENCE = "final"         # {"target","final"}
DEFAULT_SETTLING_RULE = "to_end"    # {"window","to_end"}
DEFAULT_PEAK_MODE = "first"         # {"first","global"}
DEFAULT_FINAL_WINDOW = 0.5
DEFAULT_INTERPOLATE = True

# Plot configuration
DEFAULT_PLOT = True
DEFAULT_PLOT_PATH = Path("./logs/thumb_rotate_step_plot.png")
DEFAULT_SHOW_PLOT = False

# ====== (2) Define the step plan in code ======
# When USE_CODE_STEP_PLAN is True the script uses CODE_STEP_PLAN / CODE_SWEEP and ignores --targets/--speeds/--pairing.
USE_CODE_STEP_PLAN = True

# Option A: explicit list (executed sequentially; each target/speed pair can differ).
# Example: run three steps target=400@800, 600@800, 800@800.
CODE_STEP_PLAN: List[Dict[str, int]] = [
    {"target": 500, "speed": 1000},
    {"target": 500, "speed": 750},
    {"target": 500, "speed": 500},
    {"target": 500, "speed": 250},
    {"target": 500, "speed": 100},
]

# Option B: automatic sweep (mutually exclusive with CODE_STEP_PLAN; CODE_STEP_PLAN wins when non-empty).
# mode="zip"  pairs targets/speeds in order (lengths must match or speed list length=1 for broadcast).
# mode="cross" evaluates the full Cartesian product.
CODE_SWEEP = {
    "enabled": False,
    "mode": "cross",               # "zip" or "cross"
    "targets": [400, 600, 800],    # Leave empty to skip
    "speeds":  [600, 900],         # Leave empty to skip
    "label_prefix": "sweep"
}
# =====================================

try:
    from rh56_controller.rh56_hand import RH56Hand
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rh56_controller.rh56_hand import RH56Hand

Sample = Tuple[float, int]


def prepare_hand(hand: "RH56Hand", prep_angle: int, prep_speed: int, wait_time: float) -> Optional[List[int]]:
    target_angles = [prep_angle] * 6
    target_speeds = [prep_speed] * 6
    hand.speed_set(target_speeds)
    hand.angle_set(target_angles)
    if wait_time > 0:
        time.sleep(wait_time)
    return hand.angle_read()


def _rmse(arr: Sequence[float], ref: float) -> float:
    if not arr:
        return float("nan")
    return math.sqrt(sum((x - ref) ** 2 for x in arr) / len(arr))


def _crossing_time(
    times: Sequence[float],
    values: Sequence[float],
    threshold: float,
    increasing: bool,
    interpolate: bool,
) -> Optional[float]:
    for i in range(1, len(values)):
        v0, v1 = values[i - 1], values[i]
        t0i, t1i = times[i - 1], times[i]
        if increasing:
            if v0 < threshold <= v1:
                if not interpolate or v1 == v0:
                    return t1i
                alpha = (threshold - v0) / (v1 - v0)
                return t0i + alpha * (t1i - t0i)
        else:
            if v0 > threshold >= v1:
                if not interpolate or v1 == 0:
                    return t1i
                alpha = (threshold - v0) / (v1 - v0)
                return t0i + alpha * (t1i - t0i)
    return None


def compute_metrics(
    samples: Sequence[Sample],
    initial: int,
    target: int,
    settle_band: float,
    settle_abs: Optional[float],
    settle_window: float,
    reference: str = DEFAULT_REFERENCE,
    final_window: float = DEFAULT_FINAL_WINDOW,
    settling_rule: str = DEFAULT_SETTLING_RULE,
    peak_mode: str = DEFAULT_PEAK_MODE,
    interpolate: bool = DEFAULT_INTERPOLATE,
) -> Dict[str, Optional[float]]:
    if not samples:
        return {
            "rise_time": None, "settling_time": None, "overshoot_pct": None,
            "rms_error": None, "rms_error_steady": None, "steady_state_error": None,
            "reference_used": reference, "settling_rule": settling_rule,
            "eps": None, "final_est": None, "thr10": None, "thr90": None, "ref_value": None,
        }

    times = [float(t) for t, _ in samples]
    values = [float(v) for _, v in samples]
    total_T = times[-1] - times[0] if len(times) > 1 else 0.0

    # Estimate final value using a trailing-window average
    if total_T > 0 and final_window > 0:
        cutoff = times[-1] - final_window
        start_idx = 0
        for i, t in enumerate(times):
            if t >= max(times[0], cutoff):
                start_idx = i
                break
        final = sum(values[start_idx:]) / max(1, len(values) - start_idx)
    else:
        final = values[-1]

    ref_value = float(target) if reference == "target" else float(final)

    amplitude = ref_value - float(initial)
    abs_amp = abs(amplitude)
    eps = settle_abs if settle_abs is not None else (abs_amp * settle_band if abs_amp > 0 else settle_band)

    # Rise time (10% -> 90%)
    thr10 = float(initial) + 0.1 * amplitude
    thr90 = float(initial) + 0.9 * amplitude
    increasing = amplitude >= 0.0

    t10 = _crossing_time(times, values, thr10, increasing, interpolate)
    t90 = _crossing_time(times, values, thr90, increasing, interpolate)
    rise_time = (t90 - t10) if (t10 is not None and t90 is not None and t90 >= t10) else None

    # Overshoot
    overshoot_pct = 0.0
    if abs_amp > 0:
        t_cross_ref = _crossing_time(times, values, ref_value, increasing, interpolate)
        start_idx = 0
        if t_cross_ref is not None:
            for i, t in enumerate(times):
                if t >= t_cross_ref:
                    start_idx = i
                    break

        if peak_mode == "global":
            peak_val = max(values[start_idx:]) if increasing else min(values[start_idx:])
        else:
            # First peak after crossing the reference
            found = False
            if increasing:
                peak_val = ref_value
                for i in range(max(start_idx, 1), len(values) - 1):
                    if values[i - 1] <= values[i] > values[i + 1]:
                        peak_val = values[i]; found = True; break
                if not found:
                    peak_val = max(values[start_idx:]) if start_idx < len(values) else ref_value
            else:
                peak_val = ref_value
                for i in range(max(start_idx, 1), len(values) - 1):
                    if values[i - 1] >= values[i] < values[i + 1]:
                        peak_val = values[i]; found = True; break
                if not found:
                    peak_val = min(values[start_idx:]) if start_idx < len(values) else ref_value

        overshoot_pct = max(
            0.0,
            ((peak_val - ref_value) if increasing else (ref_value - peak_val)) / abs_amp * 100.0,
        )

    # Settling time
    avg_dt = (times[-1] - times[0]) / max(1, len(times) - 1)
    window_samples = max(1, int(math.ceil(settle_window / max(1e-12, avg_dt))))

    settling_time: Optional[float] = None
    if settling_rule == "window":
        for i in range(len(values)):
            j = i + window_samples
            if j > len(values): break
            if all(abs(v - ref_value) <= eps for v in values[i:j]):
                settling_time = times[i]; break
    else:  # Strictly remain within the band until the end
        last_violate = None
        for i, v in enumerate(values):
            if abs(v - ref_value) > eps:
                last_violate = i
        if last_violate is None:
            settling_time = times[0]
        elif last_violate + 1 < len(times):
            settling_time = times[last_violate + 1]
        else:
            settling_time = None

    # Error metrics
    rms_error = _rmse(values, float(target))  # Full trace relative to target
    steady_start = max(0, len(times) - window_samples)
    rms_error_steady = _rmse(values[steady_start:], ref_value)
    steady_state_error = final - float(target)

    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot_pct": overshoot_pct,
        "rms_error": rms_error,
        "rms_error_steady": rms_error_steady,
        "steady_state_error": steady_state_error,
        "reference_used": reference,
        "settling_rule": settling_rule,
        "eps": eps,
        "final_est": final,
        "thr10": thr10,
        "thr90": thr90,
        "ref_value": ref_value,
    }


def collect_samples(
    hand: RH56Hand,
    finger: int,
    target_angle: int,
    speed: int,
    duration: float,
    sample_rate: float,
    initial_angles: Sequence[int],
) -> List[Sample]:
    speeds = [0] * 6
    speeds[finger] = speed
    hand.speed_set(speeds)

    target_angles = list(initial_angles)
    target_angles[finger] = target_angle

    history: List[Sample] = [(0.0, int(initial_angles[finger]))]

    start_time = time.perf_counter()
    sample_interval = 1.0 / sample_rate
    next_sample = start_time + sample_interval
    end_time = start_time + duration

    hand.angle_set(target_angles)

    while True:
        now = time.perf_counter()
        if now < next_sample:
            time.sleep(max(0.0, next_sample - now))
            continue

        angles = hand.angle_read()
        if angles:
            timestamp = time.perf_counter() - start_time
            history.append((timestamp, int(angles[finger])))

        next_sample += sample_interval
        if time.perf_counter() >= end_time:
            break

    return history


def write_csv(output_path: Path, samples: Iterable[Sample], target: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_s", "target_value", "measured_value"])
        for timestamp, value in samples:
            writer.writerow([f"{timestamp:.6f}", target, value])


def write_summary(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label","finger","target_angle","speed",
        "rise_time","settling_time","overshoot_pct",
        "rms_error","rms_error_steady","steady_state_error",
        "reference_used","settling_rule","eps","final_est"
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_response_multi(series: List[Dict[str, object]], plot_path: Path, show: bool = False) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for item in series:
        samples = item["samples"]  # List[Sample]
        metrics = item["metrics"]  # Dict
        label   = item["label"]    # str
        target  = item["target"]   # int

        t = [s[0] for s in samples]
        y = [s[1] for s in samples]
        ax.plot(t, y, label=label)  # Use default color cycle

        # Target/reference lines
        ax.axhline(float(target), linestyle="--")
        ref_value = metrics.get("ref_value")
        if ref_value is not None:
            ax.axhline(ref_value, linestyle=":")

        # Settling band
        eps = metrics.get("eps")
        if eps is not None and ref_value is not None:
            upper = ref_value + eps
            lower = ref_value - eps
            ax.fill_between(t, lower, upper, alpha=0.08)

        # Key vertical markers
        if metrics.get("rise_time") is not None:
            ax.axvline(metrics["rise_time"], linestyle="--")
        if metrics.get("settling_time") is not None:
            ax.axvline(metrics["settling_time"], linestyle="--")

    ax.set_title("RH56 Single-Finger Step Response (Multi-Run)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (controller units)")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=170)
    if show:
        plt.show()
    plt.close(fig)


def _parse_csv_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _build_run_plan_from_code() -> List[Dict[str, object]]:
    """Use CODE_STEP_PLAN when provided; otherwise build a sweep plan if CODE_SWEEP is enabled."""
    plan: List[Dict[str, object]] = []
    if CODE_STEP_PLAN:
        for i, item in enumerate(CODE_STEP_PLAN, start=1):
            plan.append({
                "target": int(item["target"]),
                "speed": int(item["speed"]),
                "label": f"code-{i} (a={item['target']}, s={item['speed']})"
            })
        return plan

    if CODE_SWEEP.get("enabled", False):
        mode = CODE_SWEEP.get("mode", "zip")
        tg  = CODE_SWEEP.get("targets", []) or []
        sp  = CODE_SWEEP.get("speeds", []) or []
        prefix = CODE_SWEEP.get("label_prefix", "sweep")

        if not tg:
            raise ValueError("CODE_SWEEP enabled but 'targets' empty.")
        if not sp:
            raise ValueError("CODE_SWEEP enabled but 'speeds' empty.")

        if mode == "zip":
            if len(sp) == 1 and len(tg) > 1:
                sp = [sp[0]] * len(tg)
            if len(tg) != len(sp):
                raise ValueError("CODE_SWEEP zip mode requires equal lengths or single speed broadcast.")
            for i, (a, s) in enumerate(zip(tg, sp), start=1):
                plan.append({"target": int(a), "speed": int(s), "label": f"{prefix}-{i} (a={a}, s={s})"})
        else:  # cross
            idx = 1
            for a in tg:
                for s in sp:
                    plan.append({"target": int(a), "speed": int(s), "label": f"{prefix}-{idx} (a={a}, s={s})"})
                    idx += 1
        return plan

    # No code-defined plan: fall back to CLI-generated configuration
    return []


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run single-finger step tests on the RH56 hand.")
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--finger", type=int, default=DEFAULT_FINGER)
    p.add_argument("--target-angle", type=int, default=DEFAULT_TARGET_ANGLE,
                   help="Used only when not using code-defined plan and --targets is empty.")
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED,
                   help="Used only when not using code-defined plan and --speeds is empty.")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--settle-band", type=float, default=0.02)
    p.add_argument("--settle-abs", type=float, default=None)
    p.add_argument("--settle-window", type=float, default=0.5)
    p.add_argument("--hand-id", type=int, default=1)
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                   help="Base CSV path (per run will append suffix like _s{speed}_a{angle}.csv)")
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH,
                   help="Summary CSV path for multi-run.")
    p.add_argument("--prep-angle", type=int, default=DEFAULT_PREP_ANGLE)
    p.add_argument("--prep-speed", type=int, default=DEFAULT_PREP_SPEED)
    p.add_argument("--prep-wait", type=float, default=DEFAULT_PREP_WAIT)
    p.add_argument("--no-reprep", action="store_true",
                   help="Do NOT re-prepare between runs (default is to prepare each time).")

    # Metric options
    p.add_argument("--reference", choices=["target", "final"], default=DEFAULT_REFERENCE)
    p.add_argument("--settling-rule", choices=["window", "to_end"], default=DEFAULT_SETTLING_RULE)
    p.add_argument("--peak-mode", choices=["first", "global"], default=DEFAULT_PEAK_MODE)
    p.add_argument("--final-window", type=float, default=DEFAULT_FINAL_WINDOW)
    p.add_argument("--no-interpolate", action="store_true")

    # Plotting options
    p.add_argument("--plot", action="store_true", default=DEFAULT_PLOT)
    p.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    p.add_argument("--show-plot", action="store_true", default=DEFAULT_SHOW_PLOT)

    # Multi-run configuration via CLI (used when no code-defined plan is active)
    p.add_argument("--targets", type=str, default=None,
                   help="Comma-separated target angles, e.g., '400,600,800'.")
    p.add_argument("--speeds", type=str, default=None,
                   help="Comma-separated speeds, e.g., '600,900'. If length=1, reused for all.")
    p.add_argument("--pairing", choices=["zip","cross"], default="zip")
    p.add_argument("--label-prefix", type=str, default="step")
    return p.parse_args(argv)


def _fmt_print(name: str, val: Optional[float], unit: str = "", precision: int = 4):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        print(f"{name}: N/A")
    else:
        print(f"{name}: {val:.{precision}f}{(' ' + unit) if unit else ''}")


def _run_single_step(
    hand: RH56Hand,
    args: argparse.Namespace,
    target_angle: int,
    speed: int,
    initial_angles: Optional[Sequence[int]],
) -> Tuple[List[Sample], Dict[str, Optional[float]]]:
    if initial_angles is None:
        initial_angles = hand.angle_read()
        if not initial_angles:
            raise RuntimeError("Unable to read initial angles.")

    samples = collect_samples(
        hand=hand,
        finger=args.finger,
        target_angle=target_angle,
        speed=speed,
        duration=args.duration,
        sample_rate=args.sample_rate,
        initial_angles=initial_angles,
    )

    metrics = compute_metrics(
        samples=samples,
        initial=initial_angles[args.finger],
        target=target_angle,
        settle_band=args.settle_band,
        settle_abs=args.settle_abs,
        settle_window=args.settle_window,
        reference=args.reference,
        final_window=args.final_window,
        settling_rule=args.settling_rule,
        peak_mode=args.peak_mode,
        interpolate=not args.no_interpolate,
    )
    return samples, metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # Build the run plan, preferring the code-defined configuration
    run_plan: List[Dict[str, object]] = []
    if USE_CODE_STEP_PLAN:
        try:
            run_plan = _build_run_plan_from_code()
            if run_plan:
                print("Using step plan from code (USE_CODE_STEP_PLAN=True).")
            else:
                print("USE_CODE_STEP_PLAN=True but no CODE_STEP_PLAN/CODE_SWEEP; falling back to CLI.")
        except Exception as e:
            print(f"Code-defined plan error: {e}. Falling back to CLI.", file=sys.stderr)
            run_plan = []

    if not run_plan:
        tg_list = _parse_csv_list(args.targets) if args.targets else None
        sp_list = _parse_csv_list(args.speeds) if args.speeds else None
        if tg_list is None and sp_list is None:
            run_plan = [{"target": args.target_angle, "speed": args.speed, "label": f"{args.label_prefix}-1 (a={args.target_angle}, s={args.speed})"}]
        else:
            run_plan = []
            tg_list = tg_list or [args.target_angle]
            sp_list = sp_list or [args.speed]
            if args.pairing == "zip":
                if len(sp_list) == 1 and len(tg_list) > 1:
                    sp_list = [sp_list[0]] * len(tg_list)
                if len(tg_list) != len(sp_list):
                    print("zip pairing requires equal lengths (or single speed broadcast).", file=sys.stderr)
                    return 2
                for i, (a, s) in enumerate(zip(tg_list, sp_list), start=1):
                    run_plan.append({"target": a, "speed": s, "label": f"{args.label_prefix}-{i} (a={a}, s={s})"})
            else:  # cross
                idx = 1
                for a in tg_list:
                    for s in sp_list:
                        run_plan.append({"target": a, "speed": s, "label": f"{args.label_prefix}-{idx} (a={a}, s={s})"})
                        idx += 1

    output_base = args.output
    summary_rows: List[Dict[str, object]] = []
    plot_series: List[Dict[str, object]] = []

    hand = RH56Hand(args.port, hand_id=args.hand_id, baudrate=args.baudrate)
    try:
        print(f"Preparing hand: set all to angle {args.prep_angle} @ speed {args.prep_speed}, wait {args.prep_wait:.2f}s")
        init_angles = prepare_hand(hand, args.prep_angle, args.prep_speed, args.prep_wait)
        if not init_angles:
            init_angles = hand.angle_read()
        if not init_angles:
            print("Unable to read angles from the hand after preparation.", file=sys.stderr)
            return 1

        for idx, step in enumerate(run_plan, start=1):
            tgt = int(step["target"]); spd = int(step["speed"])
            label = str(step.get("label", f"step-{idx} (a={tgt}, s={spd})"))

            print(f"\n=== Run {idx}/{len(run_plan)}: target={tgt}, speed={spd} ===")

            # Optionally re-open the hand before each subsequent run
            if idx > 1 and DEFAULT_REPREP_BETWEEN and (not args.no_reprep):
                print(f"Re-preparing hand before run {idx}...")
                init_angles = prepare_hand(hand, args.prep_angle, args.prep_speed, args.prep_wait)
                if not init_angles:
                    init_angles = hand.angle_read()
                if not init_angles:
                    print("Unable to read angles from the hand before this run.", file=sys.stderr)
                    return 1

            samples, metrics = _run_single_step(hand, args, tgt, spd, init_angles)

            # Save per-run CSV with suffix
            out_path = output_base.with_name(f"{output_base.stem}_s{spd}_a{tgt}{output_base.suffix}")
            write_csv(out_path, samples, tgt)
            print(f"CSV saved to: {out_path}")

            total_time = samples[-1][0] if samples else 0.0
            eff_rate = (len(samples) - 1) / total_time if total_time > 0 and len(samples) > 1 else 0.0
            print(f"Samples: {len(samples)} (effective ≈ {eff_rate:.1f} Hz)")
            print(f"Initial: {init_angles[args.finger]} | Target: {tgt}")
            eps = metrics["eps"]; eps_str = (f"{eps:.3f}" if isinstance(eps, (int,float)) and not math.isnan(eps) else "N/A")
            print(f"Reference: {metrics['reference_used']} | Settling: {metrics['settling_rule']} | eps: {eps_str}")

            _fmt_print("Rise Time", metrics["rise_time"], "s")
            _fmt_print("Settling Time", metrics["settling_time"], "s")
            _fmt_print("Overshoot", metrics["overshoot_pct"], "%", 2)
            _fmt_print("RMSE(full vs target)", metrics["rms_error"])
            _fmt_print("RMSE(steady vs ref)", metrics["rms_error_steady"])
            _fmt_print("Steady-state Error (final-target)", metrics["steady_state_error"])

            summary_rows.append({
                "label": label,
                "finger": args.finger,
                "target_angle": tgt,
                "speed": spd,
                **metrics
            })
            plot_series.append({"label": label, "samples": samples, "metrics": metrics, "target": tgt})

    finally:
        try:
            hand.ser.close()
        except Exception:
            pass

    # Write summary file when running multiple steps
    if len(summary_rows) > 1:
        write_summary(args.summary, summary_rows)
        print(f"\nSummary saved to: {args.summary}")

    # Generate comparison plot
    if args.plot:
        plot_response_multi(plot_series, args.plot_path, show=args.show_plot)
        print(f"Plot saved to: {args.plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
