#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RH56 延迟测试（运动量阈值法，直连，无 ROS）

逻辑：
- 发命令记 t_cmd
- 轮询 angle_read；一旦 |measured - initial_at_cmd| ≥ movement_eps -> 记 t_move
- latency = t_move - t_cmd
- 重复 N 轮（默认 10），输出均值/分位数 + CSV

用法示例：
python3 latency_movement_threshold.py --port /dev/tty.usbserial-2130 \
  --finger 2 --baseline 1000 --delta -300 --movement-eps 10 --trials 10
"""

from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
from typing import List, Optional

# ===== 默认参数（可在文件头直接改） =====
DEFAULT_PORT = "/dev/tty.usbserial-2130"
DEFAULT_BAUD = 115200
DEFAULT_FINGER = 2
DEFAULT_BASELINE = 1000       # 每轮前复位到此角度（若 --no-reset 则跳过复位）
DEFAULT_DELTA = -300          # 发令角度 = baseline + delta（自动夹到 0..1000）
DEFAULT_SPEED = 1000
DEFAULT_TRIALS = 10
DEFAULT_MOVEMENT_EPS = 10     # 触发阈值 |meas - initial_at_cmd| ≥ eps
DEFAULT_MAX_WAIT = 1.0        # 单轮最大等待秒数
DEFAULT_SETTLE_EPS = 10       # 复位判稳阈值
DEFAULT_SETTLE_TIMEOUT = 1.0  # 复位判稳超时
DEFAULT_INTERTRIAL = 0.05     # 轮间最小间隔
DEFAULT_OUT_DIR = Path("./logs_latency")
# =====================================

try:
    from rh56_controller.rh56_hand import RH56Hand
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rh56_controller.rh56_hand import RH56Hand

def clamp(v: int, lo=0, hi=1000) -> int:
    return max(lo, min(hi, int(v)))

def percentiles(xs, ps=(50, 90, 95, 99)):
    if not xs: return {}
    xs = sorted(xs); n = len(xs)
    out = {}
    for p in ps:
        k = (p/100)*(n-1); f = math.floor(k); c = math.ceil(k)
        out[f"p{p}"] = xs[f] if f==c else xs[f] + (xs[c]-xs[f])*(k-f)
    return out

def read_angles(hand) -> Optional[List[int]]:
    a = hand.angle_read()
    return list(a) if a else None

def wait_settle(hand, finger: int, target: int, eps: int, timeout: float):
    deadline = time.perf_counter() + max(timeout, 0.001)
    while time.perf_counter() < deadline:
        a = read_angles(hand)
        if a and abs(int(a[finger]) - target) <= eps:
            return True
        time.sleep(0.0)
    return False

def run(args) -> int:
    hand = RH56Hand(args.port, hand_id=1, baudrate=args.baud)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    latencies: List[float] = []

    baseline = clamp(args.baseline)
    cmd_angle = clamp(baseline + args.delta)

    try:
        # 可选复位一次
        if not args.no_reset:
            hand.speed_set([args.speed]*6)
            hand.angle_set([baseline]*6)
            wait_settle(hand, args.finger, baseline, args.settle_eps, args.settle_timeout)

        for i in range(1, args.trials + 1):
            loop_start = time.perf_counter()

            # 复位到 baseline，保证起点一致
            if not args.no_reset:
                hand.speed_set([args.speed]*6)
                hand.angle_set([baseline]*6)
                wait_settle(hand, args.finger, baseline, args.settle_eps, args.settle_timeout)

            # 发令前读取“发令瞬间的基准值”
            a0 = read_angles(hand) or [baseline]*6
            init_val = int(a0[args.finger])

            # 仅修改被测手指目标角，其余保持当前
            sp = [0]*6; sp[args.finger] = args.speed
            hand.speed_set(sp)
            target_vec = list(a0); target_vec[args.finger] = cmd_angle

            # 发令并记时
            t_cmd = time.perf_counter()
            hand.angle_set(target_vec)

            # 轮询直到“位移≥阈值”或超时
            latency = float("nan")
            deadline = t_cmd + max(args.max_wait, 0.001)
            last_meas = None
            while time.perf_counter() < deadline:
                meas = read_angles(hand)
                if meas:
                    last_meas = int(meas[args.finger])
                    if abs(last_meas - init_val) >= args.movement_eps:
                        t_move = time.perf_counter()
                        latency = t_move - t_cmd
                        break
                time.sleep(0.0)

            csv_rows.append([i, t_cmd, (t_move if math.isfinite(latency) else float("nan")), latency, init_val, cmd_angle, (last_meas if last_meas is not None else float("nan"))])
            if math.isfinite(latency):
                latencies.append(latency)

            # 轮间间隔（不强制固定频率，只保证最小间隔）
            sleep_t = (loop_start + args.intertrial) - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        try:
            hand.ser.close()
        except Exception:
            pass

    # 输出 CSV
    csv_path = out_dir / "latency_log.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial","t_cmd","t_move","latency_s","init_at_cmd","cmd_angle","last_measured"])
        w.writerows(csv_rows)

    # 汇总
    valids = [x for x in latencies if math.isfinite(x)]
    stats = {
        "trials": args.trials,
        "valid_count": len(valids),
        "invalid_count": args.trials - len(valids),
        "invalid_rate": (args.trials - len(valids)) / args.trials if args.trials else None,
        "latency_mean_s": (sum(valids)/len(valids)) if valids else None,
        **percentiles(valids, ps=(50,90,95,99)),
        "finger": args.finger,
        "baseline": baseline,
        "cmd_angle": cmd_angle,
        "delta": args.delta,
        "speed": args.speed,
        "movement_eps": args.movement_eps,
        "max_wait": args.max_wait,
        "reset_each_trial": (not args.no_reset),
    }
    json_path = out_dir / "latency_summary.json"
    with json_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Movement-threshold Latency Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    return 0

def build_parser():
    p = argparse.ArgumentParser(description="RH56 latency by movement-threshold (direct, no ROS).")
    p.add_argument("--port", type=str, default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--finger", type=int, default=DEFAULT_FINGER)
    p.add_argument("--baseline", type=int, default=DEFAULT_BASELINE)
    p.add_argument("--delta", type=int, default=DEFAULT_DELTA,
                   help="command step relative to baseline, clamped to [0,1000] (e.g., -300).")
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--movement-eps", type=int, default=DEFAULT_MOVEMENT_EPS)
    p.add_argument("--max-wait", type=float, default=DEFAULT_MAX_WAIT)
    p.add_argument("--settle-eps", type=int, default=DEFAULT_SETTLE_EPS)
    p.add_argument("--settle-timeout", type=float, default=DEFAULT_SETTLE_TIMEOUT)
    p.add_argument("--intertrial", type=float, default=DEFAULT_INTERTRIAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-reset", action="store_true",
                   help="Do NOT reset to baseline before each trial.")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)

if __name__ == "__main__":
    sys.exit(main())
