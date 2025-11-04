#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RH56 force-limit stop overshoot test: compare overshoot under different speeds at a fixed max-force.

Workflow per run:
  1) Set max-force limit (via RH56Hand API).
  2) Pre-open to baseline, then command a large step toward "close" (e.g., 0) at given speed.
  3) Sample angle (and force if available) at high rate.
  4) Trigger point = when force ~ reaches limit (force-mode) OR velocity falls below vel_eps for vel_window (stall-mode).
  5) Keep sampling until "stopped" = velocity below stop_eps for stop_window.
  6) Metrics: time_to_trigger, overshoot vs peak, stop vs peak, final-stop force, plus classic step metrics.

Outputs:
  - Per-run CSV (t, target, angle, [force], flags)
  - Summary CSV aggregating metrics over speeds
  - One multi-trace plot (default matplotlib colors)
"""

from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ===== (1) Defaults you can edit here =====
DEFAULT_PORT = "/dev/tty.usbserial-2130"
DEFAULT_BAUD = 115200
DEFAULT_FINGER = 3

# Step & sampling
DEFAULT_DURATION = 15.0
DEFAULT_SAMPLE_RATE = 165.0
DEFAULT_PREP_ANGLE = 1000
DEFAULT_PREP_SPEED = 1000
DEFAULT_PREP_WAIT = 1.0
DEFAULT_REPREP_BETWEEN = True

# Targeting: we'll command a large "close" step (e.g., 0)
DEFAULT_CLOSE_TARGET = 0

# Force limit & detection
DEFAULT_FORCE_LIMIT = 500           # <-- 请按你设备单位填写（例如 0~100 或 mA/ADC）
DEFAULT_FORCE_LIMIT_METHODS = [
    "force_set",              # RH56 API for per-finger force thresholds (0-1000)
    "force_limit_set",
    "set_force_limit",
    "max_force_set",
    "current_limit_set",
    "set_current_limit",
]
DEFAULT_FORCE_READ_METHODS = ["force_read", "current_read"]

# Trigger detection
DEFAULT_TRIGGER_MODE = "force"     # {"force","stall"}
DEFAULT_FORCE_TRIGGER_RATIO = 0.95 # force-mode: trigger when force >= ratio * limit
DEFAULT_VEL_EPS = 2                # stall-mode: |Δangle| < vel_eps (units per sample)
DEFAULT_VEL_WINDOW = 3             # number of consecutive samples for trigger
DEFAULT_STOP_EPS = 1               # stop condition velocity eps
DEFAULT_STOP_WINDOW = 5            # consecutive samples

# Plot & output
DEFAULT_PLOT = True
DEFAULT_PLOT_PATH = Path("./logs/force_stop_2_plot.png")
DEFAULT_OUTPUT_PATH = Path("./logs/force_stop_2_run.csv")
DEFAULT_SUMMARY_PATH = Path("./logs/force_stop_2_summary.csv")
DEFAULT_SHOW_PLOT = False

# Comparison plan (code-defined): vary speed at a fixed force limit
USE_CODE_STEP_PLAN = True
CODE_STEP_PLAN: List[Dict[str, int]] = [
    {"speed": 1000, "force_limit": DEFAULT_FORCE_LIMIT},
    {"speed": 500,  "force_limit": DEFAULT_FORCE_LIMIT},
    {"speed": 250,  "force_limit": DEFAULT_FORCE_LIMIT},
    {"speed": 100,  "force_limit": DEFAULT_FORCE_LIMIT},
    {"speed": 50,   "force_limit": DEFAULT_FORCE_LIMIT},
    {"speed": 25,   "force_limit": DEFAULT_FORCE_LIMIT},
    {"speed": 10,   "force_limit": DEFAULT_FORCE_LIMIT},
]
# =========================================

try:
    from rh56_controller.rh56_hand import RH56Hand
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rh56_controller.rh56_hand import RH56Hand

Sample = Tuple[float, int]

# ----- small helpers -----
def _get_method(obj, names: List[str]) -> Optional:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def prepare_hand(hand: "RH56Hand", prep_angle: int, prep_speed: int, wait_time: float) -> Optional[List[int]]:
    hand.speed_set([prep_speed]*6)
    hand.angle_set([prep_angle]*6)
    if wait_time > 0: time.sleep(wait_time)
    return hand.angle_read()

def _rmse(arr: Sequence[float], ref: float) -> float:
    if not arr: return float("nan")
    return math.sqrt(sum((x-ref)**2 for x in arr) / len(arr))

def compute_classic_metrics(samples: Sequence[Tuple[float,int]], initial: int, target: int,
                            settle_band: float, settle_abs: Optional[float],
                            settle_window: float, reference: str, final_window: float,
                            interpolate: bool=True) -> Dict[str, Optional[float]]:
    if not samples:
        return {k: None for k in ["rise_time","settling_time","overshoot_pct","rms_error","rms_error_steady","steady_state_error","ref_value","eps","final_est"]}
    times = [float(t) for t,_ in samples]
    values = [float(v) for _,v in samples]
    # final estimate
    if len(times) >= 2:
        cutoff = times[-1] - final_window
        i0 = 0
        for i,t in enumerate(times):
            if t >= max(times[0], cutoff): i0 = i; break
        final = sum(values[i0:]) / max(1, len(values)-i0)
    else:
        final = values[-1]
    ref = float(target) if reference=="target" else float(final)
    amp = ref - float(initial); aabs = abs(amp)
    eps = settle_abs if settle_abs is not None else (aabs*settle_band if aabs>0 else settle_band)
    # rise time
    thr10 = float(initial) + 0.1*amp
    thr90 = float(initial) + 0.9*amp
    inc = amp >= 0
    def _cth(thr: float) -> Optional[float]:
        for i in range(1,len(values)):
            v0,v1 = values[i-1], values[i]; t0,t1 = times[i-1], times[i]
            if inc:
                if v0 < thr <= v1:
                    if not interpolate or v1==v0: return t1
                    a = (thr-v0)/(v1-v0); return t0 + a*(t1-t0)
            else:
                if v0 > thr >= v1:
                    if not interpolate or v1==v0: return t1
                    a = (thr-v0)/(v1-v0); return t0 + a*(t1-t0)
        return None
    t10, t90 = _cth(thr10), _cth(thr90)
    rise = (t90-t10) if (t10 is not None and t90 is not None and t90>=t10) else None
    # overshoot%
    if inc:
        peak = max(values)
        ovs = max(0.0, (peak - ref)/(aabs if aabs else 1.0) * 100.0)
    else:
        trough = min(values)
        ovs = max(0.0, (ref - trough)/(aabs if aabs else 1.0) * 100.0)
    # settling (strict to end)
    avgdt = (times[-1]-times[0])/max(1,len(times)-1)
    win = max(1, int(math.ceil(settle_window/max(1e-12,avgdt))))
    last_out = None
    for i,v in enumerate(values):
        if abs(v-ref) > eps: last_out = i
    if last_out is None: settle = times[0]
    elif last_out+1 < len(times): settle = times[last_out+1]
    else: settle = None
    # RMSE
    rmse = _rmse(values, float(target))
    steady_start = max(0, len(times)-win)
    rmse_steady = _rmse(values[steady_start:], ref)
    sse = final - float(target)
    return {
        "rise_time": rise, "settling_time": settle, "overshoot_pct": ovs,
        "rms_error": rmse, "rms_error_steady": rmse_steady, "steady_state_error": sse,
        "ref_value": ref, "eps": eps, "final_est": final
    }

def collect_until_stop(
    hand: RH56Hand,
    finger: int,
    target_angle: int,
    speed: int,
    duration: float,
    sample_rate: float,
    initial_angles: Sequence[int],
    trigger_mode: str,
    force_limit: Optional[float],
    force_trigger_ratio: float,
    force_read_method: Optional[str],
    vel_eps: int,
    vel_window: int,
    stop_eps: int,
    stop_window: int,
) -> Tuple[List[Tuple[float,int,Optional[float],str]], Dict[str, Optional[float]]]:
    """
    Return:
      history: [(t, angle, force_or_None, flag)], where flag in {"", "TRIGGER", "STOP"}
      events: {
          "t_cmd","t_trigger","t_stop",
          "angle_trigger","angle_stop","angle_peak",
          "force_trigger","force_stop","force_peak"
      }
    """
    # set speed only for this finger
    speeds = [0]*6; speeds[finger] = speed; hand.speed_set(speeds)
    target_angles = list(initial_angles); target_angles[finger] = target_angle

    force_reader = None
    if force_read_method:
        force_reader = getattr(hand, force_read_method, None)

    history: List[Tuple[float,int,Optional[float],str]] = []
    initial_angle = int(initial_angles[finger])
    direction = 1 if target_angle >= initial_angle else -1
    angle_peak = initial_angle
    force_peak: Optional[float] = None

    start = time.perf_counter()
    sample_dt = 1.0 / sample_rate
    next_t = start + sample_dt
    end = start + duration

    # issue command
    hand.angle_set(target_angles)
    t_cmd = time.perf_counter()

    # For velocity estimation
    last_angles: List[int] = list(initial_angles)
    vel_buf: List[float] = []

    triggered = False
    stopped = False
    t_trigger = None; a_trigger = None; f_trigger = None
    t_stop = None; a_stop = None; f_stop = None

    while True:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
            continue

        angles = hand.angle_read()
        fval = None
        if force_reader:
            try:
                fr = force_reader()
                if fr and 0 <= finger < len(fr):
                    fval = float(fr[finger])
            except Exception:
                fval = None

        if angles:
            t_rel = time.perf_counter() - start
            ang = int(angles[finger])
            history.append((t_rel, ang, fval, ""))

            # update peak tracking
            if direction >= 0:
                if ang > angle_peak:
                    angle_peak = ang
            else:
                if ang < angle_peak:
                    angle_peak = ang
            if fval is not None:
                if (force_peak is None) or (fval > force_peak):
                    force_peak = fval

            # vel estimation (units per sample)
            dv = ang - int(last_angles[finger])
            last_angles = list(angles)
            vel_buf.append(abs(dv))
            if len(vel_buf) > max(vel_window, stop_window):
                vel_buf.pop(0)

            # trigger detection
            if not triggered:
                trig = False
                if trigger_mode == "force" and fval is not None and force_limit is not None:
                    trig = (fval >= force_trigger_ratio * float(force_limit))
                else:  # stall
                    if len(vel_buf) >= vel_window:
                        trig = all(v <= vel_eps for v in vel_buf[-vel_window:])
                if trig:
                    triggered = True
                    t_trigger = t_rel; a_trigger = ang; f_trigger = fval
                    history[-1] = (t_rel, ang, fval, "TRIGGER")

            # stop detection (after trigger)
            if triggered and not stopped:
                if len(vel_buf) >= stop_window and all(v <= stop_eps for v in vel_buf[-stop_window:]):
                    stopped = True
                    t_stop = t_rel; a_stop = ang; f_stop = fval
                    history[-1] = (t_rel, ang, fval, "STOP")
                    # We can break early, but keep a bit more samples for context if time allows
                    # break

        next_t += sample_dt
        if time.perf_counter() >= end or (triggered and stopped):
            break

    events = {
        "t_cmd": t_cmd - start,
        "t_trigger": t_trigger,
        "t_stop": t_stop,
        "angle_trigger": a_trigger,
        "angle_stop": a_stop,
        "force_trigger": f_trigger,
        "force_stop": f_stop,
        "angle_peak": angle_peak,
        "force_peak": force_peak,
    }
    return history, events

def write_csv(output_path: Path, rows: Iterable[Tuple[float,int,Optional[float],str]], target: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s","target_value","angle","force_or_none","flag"])
        for t, ang, fval, flag in rows:
            w.writerow([f"{t:.6f}", target, ang, ("" if fval is None else f"{fval:.3f}"), flag])

def write_summary(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label","finger","speed","force_limit",
        "force_peak","force_delta","force_ratio"
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {}
            for k in fields:
                val = r.get(k)
                row[k] = "" if val is None else val
            w.writerow(row)

def plot_series_multi(series, plot_path: Path, show: bool=False) -> None:
    import matplotlib
    if not show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not series: return
    fig, ax = plt.subplots(figsize=(9,5))
    for item in series:
        rows = item["rows"]
        label = item["label"]
        target = item["target"]
        t = [r[0] for r in rows]; y = [r[2] for r in rows]
        ax.plot(t, y, label=label)
        ax.axhline(float(target), linestyle="--")
        # mark trigger/stop
        for (tt, yy, _, flag) in rows:
            if flag == "TRIGGER":
                ax.axvline(tt, linestyle=":")
            elif flag == "STOP":
                ax.axvline(tt, linestyle="-.")
    ax.set_title("Force-limit stop overshoot (angle traces)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (controller units)")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=170)
    if show: plt.show()
    plt.close(fig)

def _parse_list_int(csv_str: Optional[str]) -> Optional[List[int]]:
    if not csv_str: return None
    return [int(x.strip()) for x in csv_str.split(",") if x.strip()]

def _build_plan_from_code() -> List[Dict[str, int]]:
    return CODE_STEP_PLAN[:] if CODE_STEP_PLAN else []

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Force-limit overshoot tester (direct).")
    ap.add_argument("--port", default=DEFAULT_PORT); ap.add_argument("--baudrate", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--finger", type=int, default=DEFAULT_FINGER)
    ap.add_argument("--close-target", type=int, default=DEFAULT_CLOSE_TARGET)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument("--prep-angle", type=int, default=DEFAULT_PREP_ANGLE)
    ap.add_argument("--prep-speed", type=int, default=DEFAULT_PREP_SPEED)
    ap.add_argument("--prep-wait", type=float, default=DEFAULT_PREP_WAIT)
    ap.add_argument("--no-reprep", action="store_true")

    # classic metrics opts (keep minimal)
    ap.add_argument("--settle-band", type=float, default=0.02)
    ap.add_argument("--settle-abs", type=float, default=None)
    ap.add_argument("--settle-window", type=float, default=0.5)
    ap.add_argument("--reference", choices=["target","final"], default="final")
    ap.add_argument("--final-window", type=float, default=0.5)

    # trigger detection & force methods
    ap.add_argument("--trigger-mode", choices=["force","stall"], default=DEFAULT_TRIGGER_MODE)
    ap.add_argument("--force-limit", type=float, default=DEFAULT_FORCE_LIMIT)
    ap.add_argument("--force-trigger-ratio", type=float, default=DEFAULT_FORCE_TRIGGER_RATIO)
    ap.add_argument("--limit-method", type=str, default=None,
                    help="RH56Hand method to set force/current limit (auto-try if omitted).")
    ap.add_argument("--force-read-method", type=str, default=None,
                    help="RH56Hand method to read force/current array (auto-try if omitted).")
    ap.add_argument("--vel-eps", type=int, default=DEFAULT_VEL_EPS)
    ap.add_argument("--vel-window", type=int, default=DEFAULT_VEL_WINDOW)
    ap.add_argument("--stop-eps", type=int, default=DEFAULT_STOP_EPS)
    ap.add_argument("--stop-window", type=int, default=DEFAULT_STOP_WINDOW)

    # plotting & outputs
    ap.add_argument("--plot", action="store_true", default=DEFAULT_PLOT)
    ap.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    ap.add_argument("--show-plot", action="store_true", default=DEFAULT_SHOW_PLOT)

    # CLI plan (fallback)
    ap.add_argument("--speeds", type=str, default=None, help="e.g. '1000,500,250'")
    ap.add_argument("--label-prefix", type=str, default="speed")
    args = ap.parse_args(argv)

    # Build plan
    plan = _build_plan_from_code() if USE_CODE_STEP_PLAN else []
    if not plan:
        sp_list = _parse_list_int(args.speeds) if args.speeds else [1000, 500]
        plan = [{"speed": s, "force_limit": args.force_limit} for s in sp_list]

    # Init device
    hand = RH56Hand(args.port, hand_id=1, baudrate=args.baudrate)
    # Detect limit setter & force reader
    limit_setter = getattr(hand, args.limit_method, None) if args.limit_method else _get_method(hand, DEFAULT_FORCE_LIMIT_METHODS)
    force_reader_name = args.force_read_method or (_get_method(hand, DEFAULT_FORCE_READ_METHODS).__name__ if _get_method(hand, DEFAULT_FORCE_READ_METHODS) else None)

    if args.trigger_mode == "force" and (not force_reader_name):
        print("[warn] force trigger requested but no force-read method found; falling back to 'stall' mode.")
        args.trigger_mode = "stall"

    summary_rows: List[Dict[str, object]] = []
    plot_data = []

    try:
        print(f"Preparing hand to angle {args.prep_angle} @ speed {args.prep_speed}...")
        init = prepare_hand(hand, args.prep_angle, args.prep_speed, args.prep_wait)
        if not init:
            init = hand.angle_read()
        if not init:
            print("Unable to read angles after preparation.", file=sys.stderr)
            return 1

        for idx, step in enumerate(plan, start=1):
            spd = int(step["speed"]); fl = float(step["force_limit"])
            label = f"{args.label_prefix}-{idx} (speed={spd}, Fmax={fl})"
            print(f"\n=== Run {idx}/{len(plan)}: {label} ===")

            # Set force limit if possible
            if limit_setter:
                try:
                    # Try typical signatures: list of 6, or single value for all fingers
                    limit_value = int(round(fl))
                    try:
                        limit_setter([limit_value]*6)
                    except TypeError:
                        limit_setter(limit_value)
                except Exception as e:
                    print(f"[warn] Failed to set force limit with {limit_setter.__name__}: {e}")
            else:
                print("[warn] No force-limit setter detected. Please pass --limit-method or adjust DEFAULT_FORCE_LIMIT_METHODS.")

            # Optionally re-open
            if idx>1 and DEFAULT_REPREP_BETWEEN and (not args.no_reprep):
                init = prepare_hand(hand, args.prep_angle, args.prep_speed, args.prep_wait) or init

            # Sample until stop
            rows, ev = collect_until_stop(
                hand=hand, finger=args.finger, target_angle=args.close_target,
                speed=spd, duration=args.duration, sample_rate=args.sample_rate,
                initial_angles=init,
                trigger_mode=args.trigger_mode,
                force_limit=fl,
                force_trigger_ratio=args.force_trigger_ratio,
                force_read_method=force_reader_name,
                vel_eps=args.vel_eps, vel_window=args.vel_window,
                stop_eps=args.stop_eps, stop_window=args.stop_window,
            )
            # Save run CSV
            out_path = args.output.with_name(f"{args.output.stem}_s{spd}_F{int(fl)}{args.output.suffix}")
            write_csv(out_path, rows, args.close_target)
            print(f"CSV saved: {out_path}")

            # Force peak metrics
            force_peak = ev.get("force_peak")
            if force_peak is not None:
                force_delta = force_peak - fl
                force_ratio = (force_delta / fl) if fl else None
            else:
                force_delta = None
                force_ratio = None

            summary_rows.append({
                "label": label,
                "finger": args.finger,
                "speed": spd,
                "force_limit": fl,
                "force_peak": force_peak,
                "force_delta": force_delta,
                "force_ratio": force_ratio,
            })
            plot_data.append({"rows": rows, "label": label, "target": args.close_target})

            if force_peak is not None:
                print(f"Force peak: {force_peak:.1f} (limit {fl})")
                if force_delta is not None:
                    print(f"Delta vs limit: {force_delta:+.1f}")
                if force_ratio is not None:
                    print(f"Overshoot ratio: {force_ratio:.3f} ({force_ratio*100:.1f}%)")
            else:
                print("Force peak unavailable (no force data).")

    finally:
        try: hand.ser.close()
        except Exception: pass

    # Write summary & plot
    if summary_rows:
        write_summary(args.summary, summary_rows)
        print(f"\nSummary saved: {args.summary}")
    if args.plot:
        plot_series_multi(plot_data, args.plot_path, show=args.show_plot)
        print(f"Plot saved: {args.plot_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
