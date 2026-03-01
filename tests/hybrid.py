#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 middle-finger contact experiment (speed sweep + hybrid)

What this script does:
- Use ONLY the middle finger (index=2) for both command + logging
- Run contact trials at speeds: 1000, 500, 250, 100, 50, 25, and HYBRID
  - HYBRID: speed=1000 to PREP, then speed=25 to CLOSE
- Auto-finish each trial when:
  - force threshold is set to FORCE_TARGET_G (default 500g)
  - after force reading first exceeds FORCE_TRIGGER_G (default 300g),
    within a rolling STABLE_WINDOW_S (default 0.5s), the force stays
    "roughly stable" around FORCE_TARGET_G (default 500 ± 25g)
- After the last (HYBRID) trial finishes:
  - open hand
  - stop logging
  - save ONE CSV containing all trials
- Ctrl+C / exceptions:
  - open hand
  - stop logging
  - DO NOT save CSV (unless --save-partial is used)
"""

import sys
import time
import csv
import argparse
import threading
from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple, Deque, Dict, Any

# --- Make repo import work like your hybrid.py ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from rh56_controller.rh56_hand import RH56Hand

# ---------------- USER SETTINGS (edit these) ----------------
IDX_MIDDLE = 2  # middle finger is "2" per your note

# You define these three poses. Keep length=6.
# Convention in your repo seems to use 0..1000 scale per finger.
DEFAULT_OPEN_ANGLES = [1000, 1000, 1000, 1000, 650, 0]
DEFAULT_PREP_ANGLES = [1000, 1000, 890, 1000, 650, 0]  # example placeholder 878 + 12
DEFAULT_CLOSE_ANGLES = [1000, 1000, 0, 1000, 650, 0]  # example placeholder

# Speed sweep (non-hybrid trials)
SPEEDS = [1000, 500, 250, 100, 50, 25]

# Hybrid behavior: 1000 to PREP, then 25 to CLOSE
HYBRID_APPROACH_SPEED = 1000
HYBRID_CONTACT_SPEED = 25

# Force threshold for middle finger (raw g units in your current script)
FORCE_TARGET_G = 500
FORCE_TRIGGER_G = 300

# "Stable" detection parameters (edit if needed)
STABLE_WINDOW_S = 0.5
STABLE_BAND_G = 25
STABLE_FRAC_IN_BAND = 0.80  # fraction of samples inside [target±band] over the window

# Logging
LOG_DT = 0.01

# Timeouts / pauses (safety)
RESET_SPEED = 1000
POSE_SETTLE_S = 0.35
TRIAL_TIMEOUT_S = 20.0

# Angle wait (only used in HYBRID to ensure prep reached before slow contact)
ANGLE_TOL = 8
ANGLE_WAIT_TIMEOUT_S = 5.0
# -----------------------------------------------------------


def apply_speed_all(hand: RH56Hand, speed: int) -> None:
    hand.speed_set([int(speed)] * 6)


def apply_angles(hand: RH56Hand, angles: List[int]) -> None:
    hand.angle_set(list(map(int, angles)))


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


def middle_g_to_N(raw_g: int) -> float:
    # Put your calibration here if you want N; keep raw by default.
    return float(raw_g)


def build_force_thresholds() -> List[int]:
    thr = [1000] * 6
    thr[IDX_MIDDLE] = int(FORCE_TARGET_G)
    return thr


def save_csv(records: List[Dict[str, Any]], start_epoch: float) -> str:
    start_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_epoch))
    filename = f"middlefinger_speed_sweep_{start_time_str}.csv"

    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Timestamp_Epoch",
                "Trial",
                "Mode",
                "Commanded_Speed",
                "Stage",
                "Middle_Angle",
                "Middle_Force_g",
                "Middle_Force_N",
            ]
        )
        for r in records:
            w.writerow(
                [
                    f"{r['ts']:.6f}",
                    r["trial"],
                    r["mode"],
                    r["cmd_speed"],
                    r["stage"],
                    r["mid_angle"],
                    r["mid_g"],
                    f"{r['mid_N']:.4f}",
                ]
            )

    return filename


def force_is_stable_over_window(
    window_forces: List[int],
    target: int,
    band: int,
    frac_in_band: float,
) -> bool:
    if not window_forces:
        return False
    lo = target - band
    hi = target + band
    inside = sum(1 for f in window_forces if lo <= f <= hi)
    return (inside / max(1, len(window_forces))) >= frac_in_band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand serial port")
    parser.add_argument("--hand-id", type=int, default=1, help="Hand ID")
    parser.add_argument(
        "--save-partial",
        action="store_true",
        help="Save CSV even if interrupted/failed before finishing all trials",
    )
    args = parser.parse_args()

    print(f"Connecting RH56 on port={args.port}, hand_id={args.hand_id}...")
    hand = RH56Hand(port=args.port, hand_id=args.hand_id)

    force_thr = build_force_thresholds()

    stop_event = threading.Event()
    completed_all_trials = False

    # Prevent logger reads and command writes interleaving on serial
    io_lock = threading.Lock()

    # Shared state for logger context
    state_lock = threading.Lock()
    run_state = {
        "trial": "init",
        "mode": "init",
        "cmd_speed": -1,
        "stage": "init",
    }

    records: List[Dict[str, Any]] = []
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

    def set_run_state(**kwargs) -> None:
        with state_lock:
            for k, v in kwargs.items():
                run_state[k] = v

    def get_run_state_copy() -> Dict[str, Any]:
        with state_lock:
            return dict(run_state)

    # Logger thread starts immediately
    def logger_loop():
        while not stop_event.is_set():
            ts = time.time()
            angles, forces = read_state()
            if angles is None or forces is None:
                time.sleep(LOG_DT)
                continue

            mid_angle = int(angles[IDX_MIDDLE])
            mid_g = int(forces[IDX_MIDDLE])

            s = get_run_state_copy()
            records.append(
                {
                    "ts": ts,
                    "trial": s["trial"],
                    "mode": s["mode"],
                    "cmd_speed": s["cmd_speed"],
                    "stage": s["stage"],
                    "mid_angle": mid_angle,
                    "mid_g": mid_g,
                    "mid_N": middle_g_to_N(mid_g),
                }
            )
            time.sleep(LOG_DT)

    log_thread = threading.Thread(target=logger_loop, daemon=True)
    log_thread.start()
    print("Logging started immediately.")

    def go_pose(label: str, angles: List[int], speed: int) -> None:
        set_run_state(stage=label, cmd_speed=int(speed))
        cmd_speed_all(int(speed))
        cmd_angles(angles)
        time.sleep(POSE_SETTLE_S)

    def run_contact_trial(mode: str, speed: int) -> bool:
        """
        Returns True if stable criterion met before timeout; else False.
        """
        if mode == "speed":
            trial_name = f"speed_{speed}"
        else:
            trial_name = "hybrid"

        set_run_state(trial=trial_name, mode=mode)

        # Always reset to OPEN between trials
        go_pose("open", DEFAULT_OPEN_ANGLES, RESET_SPEED)

        # Force threshold for this trial
        set_run_state(stage="force_set")
        try:
            cmd_force_set(force_thr)
        except Exception as e:
            print(f"Warning: force_set failed: {e}")

        # Start contact motion
        if mode == "speed":
            set_run_state(stage="contact", cmd_speed=int(speed))
            cmd_speed_all(int(speed))
            cmd_angles(DEFAULT_CLOSE_ANGLES)
        else:
            # Hybrid: 1000 to PREP, then 25 to CLOSE
            set_run_state(stage="approach", cmd_speed=int(HYBRID_APPROACH_SPEED))
            cmd_speed_all(int(HYBRID_APPROACH_SPEED))
            cmd_angles(DEFAULT_PREP_ANGLES)

            ok = wait_until_angles(
                read_angles_only,
                DEFAULT_PREP_ANGLES,
                idxs=[IDX_MIDDLE],
                tol=ANGLE_TOL,
                timeout_s=ANGLE_WAIT_TIMEOUT_S,
            )
            if not ok:
                print(
                    "Warning: HYBRID prep pose not reached before timeout (continuing)."
                )

            set_run_state(stage="contact", cmd_speed=int(HYBRID_CONTACT_SPEED))
            cmd_speed_all(int(HYBRID_CONTACT_SPEED))
            cmd_angles(DEFAULT_CLOSE_ANGLES)

        # Peak detection: track running max after trigger, end when max stops increasing for STABLE_WINDOW_S.
        t0 = time.time()
        seen_trigger = False
        max_g = -(10**9)
        t_last_new_max: Optional[float] = None

        while (time.time() - t0) < TRIAL_TIMEOUT_S:
            if stop_event.is_set():
                return False

            _, forces = read_state()
            if forces is None:
                time.sleep(0.01)
                continue

            g = int(forces[IDX_MIDDLE])
            now = time.time()

            if g >= FORCE_TRIGGER_G:
                if not seen_trigger:
                    seen_trigger = True
                    max_g = g
                    t_last_new_max = now
                else:
                    if g > (max_g + MAX_REFRESH_EPS_G):
                        max_g = g
                        t_last_new_max = now

            if seen_trigger and (t_last_new_max is not None):
                if (now - t_last_new_max) >= STABLE_WINDOW_S:
                    return True

            time.sleep(0.01)

        return False

    try:
        # Trial plan: speeds then hybrid at the end
        plan: List[Tuple[str, int]] = [("speed", s) for s in SPEEDS] + [("hybrid", -1)]
        print("Trial plan:", ", ".join([f"{m}:{v}" for m, v in plan]))

        for mode, spd in plan:
            if mode == "speed":
                ok = run_contact_trial("speed", spd)
                label = f"speed={spd}"
            else:
                ok = run_contact_trial("hybrid", HYBRID_CONTACT_SPEED)
                label = "hybrid"

            if ok:
                print(f"[DONE] {label}: stable force detected.")
            else:
                print(f"[FAIL] {label}: timeout or interrupted before stable force.")
                raise RuntimeError(f"Trial failed: {label}")

        completed_all_trials = True

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
    except Exception as e:
        print(f"\nException: {e}")
    finally:
        # stop logger
        stop_event.set()
        try:
            log_thread.join(timeout=1.0)
        except Exception:
            pass

        # Always open on exit
        try:
            set_run_state(stage="open_exit", cmd_speed=int(RESET_SPEED))
            cmd_speed_all(RESET_SPEED)
            cmd_angles(DEFAULT_OPEN_ANGLES)
            time.sleep(POSE_SETTLE_S)
        except Exception:
            pass

        try:
            with io_lock:
                hand.ser.close()
        except Exception:
            pass

    # Save policy
    if (completed_all_trials or args.save_partial) and len(records) > 0:
        filename = save_csv(records, start_epoch)
        if completed_all_trials:
            print(f"Saved: {filename} ({len(records)} samples)")
        else:
            print(f"Saved PARTIAL: {filename} ({len(records)} samples)")
    else:
        print(f"No CSV saved. Samples collected: {len(records)}")


if __name__ == "__main__":
    main()
