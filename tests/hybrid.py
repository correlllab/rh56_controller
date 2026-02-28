#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-finger simple controller for RH56 (Index + Thumb bend)

Changes vs your hybrid.py:
- Start logging immediately after connecting (before any motion commands)
- Press Enter: open hand first, then stop logging, then save CSV
- Ctrl+C or exceptions: open hand, stop logging, DO NOT save CSV
- Do NOT change your existing angle commands in main()
"""

import sys
import time
import csv
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Tuple

# --- Make repo import work like rh56_peginhole.py ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from rh56_controller.rh56_hand import RH56Hand

# ---------------- USER SETTINGS (edit these) ----------------
IDX_INDEX = 3
IDX_THUMB = 4

OPEN_ANGLES = [1000, 1000, 1000, 1000, 650, 0]

TARGET_INDEX_ANGLE = 650
TARGET_THUMB_ANGLE = 650

FAST_SPEED = 1000
SLOW_SPEED = 25

INDEX_FORCE_G = 500
THUMB_FORCE_G = 500

LOG_DT = 0.01

ANGLE_TOL = 8
ANGLE_WAIT_TIMEOUT_S = 5.0
# -----------------------------------------------------------


def apply_speed_all(hand: RH56Hand, speed: int) -> None:
    hand.speed_set([speed] * 6)


def apply_angles(hand: RH56Hand, angles: List[int]) -> None:
    hand.angle_set(list(angles))


def wait_until_angles(
    read_angles_fn,
    target: List[int],
    idxs: List[int],
    tol: int,
    timeout_s: float,
) -> bool:
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        cur = read_angles_fn()
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

    # keep these as in your file (even if not used later)
    target_angles = build_target_angles()
    force_thr = build_force_thresholds()

    stop_event = threading.Event()
    save_event = threading.Event()

    # Prevent logger reads and command writes interleaving on serial
    io_lock = threading.Lock()

    records: List[dict] = []
    start_epoch = time.time()

    # Locked wrappers
    def cmd_speed_all(speed: int) -> None:
        with io_lock:
            apply_speed_all(hand, speed)

    def cmd_angles(angles: List[int]) -> None:
        with io_lock:
            apply_angles(hand, angles)

    def cmd_force_set(thr: List[int]) -> None:
        with io_lock:
            hand.force_set(list(thr))

    def read_state() -> Tuple[Optional[List[int]], Optional[List[int]]]:
        with io_lock:
            angles = hand.angle_read()
            forces = hand.force_act()
        return angles, forces

    def read_angles_only() -> Optional[List[int]]:
        with io_lock:
            return hand.angle_read()

    # Logger thread starts immediately
    def logger_loop():
        while not stop_event.is_set():
            ts = time.time()
            angles, forces = read_state()
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

    log_thread = threading.Thread(target=logger_loop, daemon=True)
    log_thread.start()
    print("Logging started immediately.")

    # Enter handler: open first, then stop, then save
    def wait_for_enter():
        try:
            input("Press Enter to open hand, stop logging, and save CSV...\n")
        except Exception:
            # stdin closed or non-interactive
            stop_event.set()
            return

        # open first (so open motion is included in log)
        try:
            cmd_speed_all(FAST_SPEED)
            cmd_angles(OPEN_ANGLES)
            time.sleep(0.3)
        except Exception:
            pass

        save_event.set()
        stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    try:
        # 0) Start open
        cmd_speed_all(FAST_SPEED)
        cmd_angles(OPEN_ANGLES)
        time.sleep(1)
        if stop_event.is_set():
            raise KeyboardInterrupt()

        # 1) Fast move to target (KEEP your angles unchanged here)
        print(f"Fast move to target angles (speed={FAST_SPEED}) ...")
        cmd_speed_all(FAST_SPEED)
        cmd_angles([1000, 1000, 1000, 800, 650, 0])  # unchanged
        time.sleep(0.5)
        if stop_event.is_set():
            raise KeyboardInterrupt()

        # 2) Slow + force threshold, then re-command (KEEP your angles unchanged here)
        print(
            f"Set slow speed={SLOW_SPEED}, set force thresholds, re-send target angles ..."
        )
        cmd_speed_all(SLOW_SPEED)
        try:
            cmd_force_set(force_thr)
        except Exception as e:
            print(f"Warning: force_set failed: {e}")

        cmd_angles([1000, 1000, 1000, 0, 650, 0])  # unchanged

        # Main thread just waits now, logger is already running
        while not stop_event.is_set():
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted. Will open hand but NOT save CSV.")
    except Exception as e:
        print(f"\nException: {e}. Will open hand but NOT save CSV.")
    finally:
        # stop logger
        stop_event.set()
        try:
            log_thread.join(timeout=1.0)
        except Exception:
            pass

        # Always open on exit (your original behavior)
        try:
            cmd_speed_all(FAST_SPEED)
            cmd_angles(OPEN_ANGLES)
            time.sleep(0.3)
        except Exception:
            pass

        try:
            with io_lock:
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
