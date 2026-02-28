#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-finger simple controller for RH56 (Index + Thumb bend)

Behavior:
1) Start from open (index=1000, thumb=1000)
2) Fast move to target angles with speed=1000
3) Set speed=25, set force thresholds (user-defined), re-send the same target angles
4) Log: timestamp_epoch, index/thumb angle, index/thumb force (raw g), plus converted N
5) Press Enter: open hand, stop logging, save CSV
6) Ctrl+C or any exception: open hand, stop logging, DO NOT save CSV
"""

import sys
import time
import csv
import argparse
import threading
from pathlib import Path
from typing import Optional, List

# --- Make repo import work like rh56_peginhole.py ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from rh56_controller.rh56_hand import RH56Hand

# ---------------- USER SETTINGS (edit these) ----------------
# Two fingers we control
IDX_INDEX = 3  # Index finger  (force_act/angle_read index) :contentReference[oaicite:3]{index=3}
IDX_THUMB = 4  # Thumb bend    (force_act/angle_read index) :contentReference[oaicite:4]{index=4}

# Start (open) pose (same style as your peginhole script) :contentReference[oaicite:5]{index=5}
OPEN_ANGLES = [1000, 1000, 1000, 1000, 1000, 0]

# Target angles (TBD)
TARGET_INDEX_ANGLE = 650  # TBD: 0-1000
TARGET_THUMB_ANGLE = 650  # TBD: 0-1000

# Speeds
FAST_SPEED = 1000
SLOW_SPEED = 25

# Force thresholds in raw unit (g), 0-1000. force_set unit is gram :contentReference[oaicite:6]{index=6}
INDEX_FORCE_G = 300  # TBD
THUMB_FORCE_G = 300  # TBD

# Logging rate (seconds)
LOG_DT = 0.01

# Optional: angle settling check after fast move
ANGLE_TOL = 8
ANGLE_WAIT_TIMEOUT_S = 5.0
# -----------------------------------------------------------


def apply_speed_all(hand: RH56Hand, speed: int) -> None:
    hand.speed_set(
        [speed] * 6
    )  # same pattern as your script :contentReference[oaicite:7]{index=7}


def apply_angles(hand: RH56Hand, angles: List[int]) -> None:
    hand.angle_set(list(angles))


def wait_until_angles(
    hand: RH56Hand, target: List[int], idxs: List[int], tol: int, timeout_s: float
) -> bool:
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        cur = hand.angle_read()
        if cur is None:
            time.sleep(0.02)
            continue
        ok = True
        for i in idxs:
            if abs(int(cur[i]) - int(target[i])) > tol:
                ok = False
                break
        if ok:
            return True
        time.sleep(0.02)
    return False


# Raw (g) -> Newton conversion (copied from your peginhole script) :contentReference[oaicite:8]{index=8}
def index_g_to_N(raw_g: int) -> float:
    # return (raw_g * 0.007478) - 0.414
    return raw_g


def thumb_g_to_N(raw_g: int) -> float:
    # return (raw_g * 0.012547) + 0.384
    return raw_g


def build_target_angles() -> List[int]:
    target = OPEN_ANGLES.copy()
    target[IDX_INDEX] = int(TARGET_INDEX_ANGLE)
    target[IDX_THUMB] = int(TARGET_THUMB_ANGLE)
    return target


def build_force_thresholds() -> List[int]:
    thr = [1000] * 6
    thr[IDX_INDEX] = int(INDEX_FORCE_G)
    thr[IDX_THUMB] = int(THUMB_FORCE_G)
    return thr


def save_csv(records: List[dict], start_epoch: float) -> str:
    start_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_epoch))
    filename = f"twofinger_log_{start_time_str}.csv"

    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Timestamp_Epoch",
                "Index_Angle",
                "Thumb_Angle",
                "Index_Force_g",
                "Thumb_Force_g",
                "Index_Force_N",
                "Thumb_Force_N",
            ]
        )
        for r in records:
            w.writerow(
                [
                    f"{r['ts']:.6f}",
                    r["idx_angle"],
                    r["th_angle"],
                    r["idx_g"],
                    r["th_g"],
                    f"{r['idx_N']:.4f}",
                    f"{r['th_N']:.4f}",
                ]
            )

    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand serial port")
    parser.add_argument("--hand-id", type=int, default=1, help="Hand ID")
    args = parser.parse_args()

    print(f"Connecting RH56 on port={args.port}, hand_id={args.hand_id}...")
    hand = RH56Hand(port=args.port, hand_id=args.hand_id)

    target_angles = build_target_angles()
    force_thr = build_force_thresholds()

    stop_event = threading.Event()
    save_event = threading.Event()

    def wait_for_enter():
        input("Press Enter to open hand, stop logging, and save CSV...\n")
        save_event.set()
        stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    records: List[dict] = []
    start_epoch = time.time()

    try:
        # 0) Start open
        apply_speed_all(hand, FAST_SPEED)
        apply_angles(hand, OPEN_ANGLES)
        time.sleep(0.3)

        # 1) Fast move to target
        print(f"Fast move to target angles (speed={FAST_SPEED}) ...")
        apply_speed_all(hand, FAST_SPEED)
        apply_angles(hand, target_angles)
        _ = wait_until_angles(
            hand,
            target_angles,
            [IDX_INDEX, IDX_THUMB],
            tol=ANGLE_TOL,
            timeout_s=ANGLE_WAIT_TIMEOUT_S,
        )

        # 2) Slow + force threshold, then re-command same target angles
        print(
            f"Set slow speed={SLOW_SPEED}, set force thresholds, re-send target angles ..."
        )
        apply_speed_all(hand, SLOW_SPEED)
        try:
            hand.force_set(force_thr)
        except Exception as e:
            print(f"Warning: force_set failed: {e}")

        apply_angles(hand, target_angles)

        # 3) Log until Enter
        print("Logging started...")
        while not stop_event.is_set():
            ts = time.time()

            angles = hand.angle_read()
            forces = hand.force_act()
            if angles is None or forces is None:
                time.sleep(LOG_DT)
                continue

            idx_angle = int(angles[IDX_INDEX])
            th_angle = int(angles[IDX_THUMB])
            idx_g = int(forces[IDX_INDEX])
            th_g = int(forces[IDX_THUMB])

            records.append(
                {
                    "ts": ts,
                    "idx_angle": idx_angle,
                    "th_angle": th_angle,
                    "idx_g": idx_g,
                    "th_g": th_g,
                    "idx_N": index_g_to_N(idx_g),
                    "th_N": thumb_g_to_N(th_g),
                }
            )

            time.sleep(LOG_DT)

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C. Will open hand but NOT save CSV.")
    except Exception as e:
        print(f"\nException: {e}. Will open hand but NOT save CSV.")
    finally:
        # Always open on exit
        try:
            apply_speed_all(hand, FAST_SPEED)
            apply_angles(hand, OPEN_ANGLES)
            time.sleep(0.3)
        except Exception:
            pass

        try:
            hand.ser.close()
        except Exception:
            pass

    # Save only when Enter was pressed
    if save_event.is_set() and len(records) > 0:
        filename = save_csv(records, start_epoch)
        print(f"Saved: {filename} ({len(records)} samples)")
    else:
        print(f"No CSV saved. Samples collected: {len(records)}")


if __name__ == "__main__":
    main()
