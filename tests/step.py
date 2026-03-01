#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 middle-finger position control experiment (Full Speed vs Step Control)

Summary:
- Use ONLY the middle finger for command + logging
- Run 2 trials:
  1. full_speed: Command 0 directly at speed 1000.
  2. step_control: Command 999 to 0 step-by-step at speed 1000. No sleep between steps, poll as fast as possible.
- Inter-trial pause is 1s.
- Continuous data logging of timestamp, angles, and raw forces.
"""

import sys
import time
import csv
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# --- Make repo import work like your original scripts ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from rh56_controller.rh56_hand import RH56Hand


# ---------------- USER SETTINGS (edit these) ----------------
IDX_MIDDLE = 3  # middle finger index

# You define these poses. Keep length=6.
DEFAULT_OPEN_ANGLES = [1000, 1000, 1000, 1000, 650, 0]
DEFAULT_CLOSE_ANGLES = [1000, 1000, 1000, 0, 650, 0]

# Motion settings
SPEED = 1000
STEP_SIZE = 1

# Logging
LOG_DT = 0.006

# Timeouts / pauses (safety)
POSE_SETTLE_S = 0.5  # Time given to physically reach an open/reset pose
INTER_TRIAL_PAUSE_S = 1.0  # 1s pause between trials
TRIAL_TIMEOUT_S = 15.0  # Maximum time allowed for a single trial to complete
# -----------------------------------------------------------


def apply_speed_all(hand: RH56Hand, speed: int) -> None:
    hand.speed_set([int(speed)] * 6)


def apply_angles(hand: RH56Hand, angles: List[int]) -> None:
    hand.angle_set(list(map(int, angles)))


def save_csv(records: List[Dict[str, Any]], start_epoch: float) -> str:
    start_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_epoch))
    filename = f"middlefinger_position_control_{start_time_str}.csv"

    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Timestamp_Epoch",
                "Trial",
                "Commanded_Speed",
                "Stage",
                "Middle_Angle",
                "Middle_Force_g",
            ]
        )
        for r in records:
            w.writerow(
                [
                    f"{r['ts']:.6f}",
                    r["trial"],
                    r["cmd_speed"],
                    r["stage"],
                    r["mid_angle"],
                    r["mid_g"],
                ]
            )

    return filename


def main() -> None:
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

    stop_event = threading.Event()
    completed_all_trials = False

    # Prevent logger reads and command writes interleaving on serial
    io_lock = threading.Lock()

    # Shared state for logger context
    state_lock = threading.Lock()
    run_state = {
        "trial": "init",
        "cmd_speed": -1,
        "stage": "init",
        "recording": True,
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
    def logger_loop() -> None:
        while not stop_event.is_set():
            ts = time.time()
            angles, forces = read_state()
            if angles is None or forces is None:
                time.sleep(LOG_DT)
                continue

            mid_angle = int(angles[IDX_MIDDLE])
            mid_g = int(forces[IDX_MIDDLE])

            s = get_run_state_copy()
            if not s.get("recording", True):
                time.sleep(LOG_DT)
                continue

            records.append(
                {
                    "ts": ts,
                    "trial": s["trial"],
                    "cmd_speed": s["cmd_speed"],
                    "stage": s["stage"],
                    "mid_angle": mid_angle,
                    "mid_g": mid_g,
                }
            )
            time.sleep(LOG_DT)

    log_thread = threading.Thread(target=logger_loop, daemon=True)
    log_thread.start()
    print("Logging started immediately.")

    def go_pose(label: str, angles: List[int], speed: int) -> None:
        # Record the movement as it happens
        set_run_state(stage=label, cmd_speed=int(speed), recording=True)
        cmd_speed_all(int(speed))
        cmd_angles(angles)
        time.sleep(POSE_SETTLE_S)

    def run_position_trial(trial_type: str) -> bool:
        """
        Returns True if the finger successfully reached position 0 before timeout.
        """
        set_run_state(trial=trial_type, stage="inter_trial", recording=False)

        # 1. Reset to OPEN between trials (recorded)
        go_pose("open", DEFAULT_OPEN_ANGLES, SPEED)

        # 2. Inter-trial pause (unrecorded)
        set_run_state(stage="inter_trial_pause", recording=False)
        time.sleep(INTER_TRIAL_PAUSE_S)

        # 3. Start motion
        set_run_state(stage="closing", cmd_speed=SPEED, recording=True)
        t0 = time.time()

        if trial_type == "full_speed":
            cmd_speed_all(SPEED)
            cmd_angles(DEFAULT_CLOSE_ANGLES)

            # Fast poll until target reached
            while (time.time() - t0) < TRIAL_TIMEOUT_S:
                if stop_event.is_set():
                    return False
                cur_angles = read_angles_only()
                if cur_angles is not None:
                    if int(cur_angles[IDX_MIDDLE]) <= 0:
                        return True

        elif trial_type == "step_control":
            cmd_speed_all(SPEED)
            target_angles = list(DEFAULT_OPEN_ANGLES)

            # Step down from 999 to 0
            for target_pos in range(999, -1, -STEP_SIZE):
                target_angles[IDX_MIDDLE] = target_pos
                cmd_angles(target_angles)

                # Fast poll loop without sleep to confirm position reached
                while (time.time() - t0) < TRIAL_TIMEOUT_S:
                    if stop_event.is_set():
                        return False
                    cur_angles = read_angles_only()
                    if cur_angles is not None:
                        # Once the current angle is less than or equal to target, break out to send next command
                        if int(cur_angles[IDX_MIDDLE]) <= target_pos:
                            break

            # Ensure final step reached 0 completely
            return target_pos == 0

        return False

    try:
        plan: List[str] = ["full_speed", "step_control"]
        print("Trial plan:", ", ".join(plan))

        for trial_name in plan:
            ok = run_position_trial(trial_name)

            if ok:
                print(f"[DONE] {trial_name}: successfully reached position 0.")
            else:
                print(f"[FAIL] {trial_name}: timeout or interrupted before reaching 0.")
                raise RuntimeError(f"Trial failed: {trial_name}")

        completed_all_trials = True

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
    except Exception as e:
        print(f"\nException: {e}")
    finally:
        stop_event.set()
        try:
            log_thread.join(timeout=1.0)
        except Exception:
            pass

        # Always open on exit
        try:
            set_run_state(stage="open_exit", cmd_speed=SPEED, recording=False)
            cmd_speed_all(SPEED)
            cmd_angles(DEFAULT_OPEN_ANGLES)
            time.sleep(POSE_SETTLE_S)
        except Exception:
            pass

        try:
            with io_lock:
                hand.ser.close()
        except Exception:
            pass

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
