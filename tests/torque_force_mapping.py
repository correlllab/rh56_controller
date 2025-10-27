#!/usr/bin/env python3
"""
Torque-force mapping experiment script.

Features:
1. Select a finger to test (others hold at angle 1000).
2. Configure a force limit (0-1000) for the selected finger.
3. Automatically close the finger at speed 1 while sampling force/current at ~60 Hz.
   Reporting starts after readings stabilize (difference <= 25 for several samples),
   then prints 1 Hz averages of force and current.
4. Press Enter to halt the test, restore the hand to the open position, and repeat.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    import select  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    select = None  # type: ignore


# Ensure project root is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from rh56_controller.rh56_hand import RH56Hand  # noqa: E402


FINGER_LABELS = {
    0: "Pinky",
    1: "Ring",
    2: "Middle",
    3: "Index",
    4: "Thumb bend",
    5: "Thumb rotation",
}

READ_INTERVAL = 1.0 / 60.0
STABILITY_THRESHOLD = 25
STABILITY_COUNT_REQUIRED = 10


def prompt_finger() -> int:
    """Prompt the user to choose a finger index."""
    print("\nSelect finger to test:")
    for idx, name in FINGER_LABELS.items():
        print(f"  {idx} - {name}")

    while True:
        raw = input("Finger index (0-5) or 'q' to quit: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return -1
        if raw.isdigit():
            value = int(raw)
            if value in FINGER_LABELS:
                return value
        print("Invalid choice. Enter a number from 0 to 5, or 'q' to exit.")


def prompt_force_limit() -> Optional[int]:
    """Ask for a force limit in grams (0-1000)."""
    while True:
        raw = input("Force limit for selected finger (0-1000, or 'q' to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        if raw.isdigit():
            value = int(raw)
            if 0 <= value <= 1000:
                return value
        print("Please enter an integer between 0 and 1000 or 'q' to exit.")


def apply_force_limit(hand: RH56Hand, finger: int, limit: int) -> bool:
    """Set force thresholds so only the selected finger uses the provided limit."""
    limits = [1000] * 6
    limits[finger] = limit
    response = hand.force_set(limits)
    return response is not None


def set_all_angles(hand: RH56Hand, angles: List[int]) -> None:
    """Wrapper for angle_set with basic error handling."""
    response = hand.angle_set(angles)
    if response is None:
        print("Warning: Failed to apply angle command.")


def set_speeds(hand: RH56Hand, speeds: List[int]) -> None:
    """Wrapper for speed_set with basic error handling."""
    response = hand.speed_set(speeds)
    if response is None:
        print("Warning: Failed to apply speed command.")


def wait_for_enter(timeout: float) -> Optional[str]:
    """Check whether user input is available within timeout seconds."""
    if select and hasattr(select, "select"):
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip().lower()
        return None
    # Fallback: simple sleep without non-blocking input
    time.sleep(timeout)
    return None


def run_measurement(hand: RH56Hand, finger: int, force_limit: int) -> None:
    """Run the measurement loop for the configured finger and force limit."""
    print(f"\nSetting force limit {force_limit}g on {FINGER_LABELS[finger]}...")
    if not apply_force_limit(hand, finger, force_limit):
        print("Failed to set force limit; aborting this run.")
        return

    open_angles = [1000] * 6
    set_all_angles(hand, open_angles)

    speeds = [1000] * 6
    speeds[finger] = 1
    set_speeds(hand, speeds)

    target_angles = open_angles.copy()
    target_angles[finger] = 0
    print(f"Closing {FINGER_LABELS[finger]} to angle 0...")
    set_all_angles(hand, target_angles)

    prev_force: Optional[int] = None
    stable_counter = 0
    stable = False
    last_report_time = time.time()
    acc_force = 0.0
    acc_current = 0.0
    sample_count = 0
    start_time = time.time()

    print("Sampling force and current at ~60 Hz. Press Enter to stop.")

    try:
        while True:
            loop_start = time.time()

            forces = hand.force_act()
            currents = hand.current_read()

            if forces is None or currents is None:
                print("Warning: Failed to read force/current; retrying...")
                time.sleep(READ_INTERVAL)
                continue

            force_value = forces[finger]
            current_value = currents[finger]

            if prev_force is not None and abs(force_value - prev_force) <= STABILITY_THRESHOLD:
                stable_counter += 1
            else:
                stable_counter = 0

            prev_force = force_value

            if not stable and stable_counter >= STABILITY_COUNT_REQUIRED:
                stable = True
                last_report_time = loop_start
                acc_force = 0.0
                acc_current = 0.0
                sample_count = 0
                print("Force readings stabilized; starting averaged output (1 Hz).")

            if stable:
                acc_force += force_value
                acc_current += current_value
                sample_count += 1

                if loop_start - last_report_time >= 1.0:
                    avg_force = acc_force / sample_count if sample_count else 0.0
                    avg_current = acc_current / sample_count if sample_count else 0.0
                    elapsed = loop_start - start_time
                    print(f"[+{elapsed:5.2f}s] Avg force: {avg_force:7.2f} g | Avg current: {avg_current:7.2f} mA")
                    last_report_time = loop_start
                    acc_force = 0.0
                    acc_current = 0.0
                    sample_count = 0

            user_input = wait_for_enter(max(0.0, READ_INTERVAL - (time.time() - loop_start)))
            if user_input is not None:
                if user_input in {"q", "quit", "exit"}:
                    raise KeyboardInterrupt
                print("Stop requested by user.")
                break

    except KeyboardInterrupt:
        print("\nMeasurement interrupted by user.")

    finally:
        print("Restoring hand to open state and resetting speeds.")
        set_speeds(hand, [1000] * 6)
        set_all_angles(hand, open_angles)
        time.sleep(0.2)  # brief pause to ensure command execution


def main() -> None:
    try:
        hand = RH56Hand(port="/dev/ttyUSB0", hand_id=1)
        print("RH56 hand initialized.")
    except Exception as exc:  # pragma: no cover - hardware init failure handling
        print(f"Failed to initialize RH56 hand: {exc}")
        return

    try:
        while True:
            finger = prompt_finger()
            if finger < 0:
                print("Exiting experiment.")
                return

            while True:
                force_limit = prompt_force_limit()
                if force_limit is None:
                    print("Exiting experiment.")
                    return

                run_measurement(hand, finger, force_limit)

                follow_up = input(
                    "\nPress Enter to set a new force limit, "
                    "'finger' to choose another finger, or 'q' to quit: "
                ).strip().lower()

                if follow_up in {"q", "quit", "exit"}:
                    print("Exiting experiment.")
                    return
                if follow_up in {"finger", "f"}:
                    break  # break inner loop to select new finger
                # otherwise loop continues to request new force limit for same finger

    except KeyboardInterrupt:
        print("\nExperiment terminated by user.")
        set_speeds(hand, [1000] * 6)
        set_all_angles(hand, [1000] * 6)


if __name__ == "__main__":
    main()
