#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive grasp experiment script for direct serial control of the RH56 hand.

Features
--------
1) Three predefined grasp presets (1 / 2 / 3) extracted from the former precision
   logic. Each preset:
       - Applies its prepare (pre-grasp) posture immediately.
       - On first Enter: closes the selected fingers at a configured speed.
       - On second Enter: re-opens the hand to the default open pose.
2) Special force-reactive pinch (input 'x'):
       - Prepares a two-finger posture (index + thumb).
       - After Enter, monitors raw thumb force for a +75 g jump to auto-trigger
         the closing motion, limits force for those fingers, and continuously
         streams index force readings at ~60 Hz (0.5 s rolling averages).
       - Once the averaged index force exceeds 500 g, it detects a +75 g spike
         relative to the previous average, then looks for a -25 g drop compared
         to the prior average before automatically re-opening.
3) Utility commands:
       - 'o' : clear errors and fully open.
       - 'c' : clear errors only.
       - 's' : print status/angle/force snapshots.
       - 'q' : quit the program.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Allow running the script from repo root or tests/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from rh56_controller.rh56_hand import RH56Hand  # noqa: E402

# Finger indices for readability
PINKY = 0
RING = 1
MIDDLE = 2
INDEX = 3
THUMB_BEND = 4
THUMB_ROT = 5

DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]


@dataclass
class GraspPreset:
    key: str
    name: str
    description: str
    prepare_angles: List[int]
    close_angles: List[int]
    close_speed: int = 500
    restore_speed: int = 1000


def build_presets() -> Dict[str, GraspPreset]:
    """Define the fixed grasp patterns used during experiments."""
    presets: Dict[str, GraspPreset] = {
        "1": GraspPreset(
            key="1",
            name="Precision pinch",
            description=(
                "Index + thumb pinch. Thumb bend preset 650, thumb rotation 150. "
                "Index closes to 0; thumb bend curls for contact."
            ),
            prepare_angles=[1000, 1000, 1000, 1000, 700, 150],
            close_angles=[1000, 1000, 1000, 0, 800, 150],
            close_speed=500,
        ),
        "2": GraspPreset(
            key="2",
            name="Tripod grasp",
            description="Index + middle + thumb. Thumb bend preset 850, rotation 0.",
            prepare_angles=[1000, 1000, 1000, 1000, 850, 0],
            close_angles=[1000, 1000, 0, 0, 850, 0],
            close_speed=500,
        ),
        "3": GraspPreset(
            key="3",
            name="Power grasp",
            description="Index + middle + ring + thumb wrap (thumb bend 1000).",
            prepare_angles=[1000, 1000, 1000, 1000, 1000, 0],
            close_angles=[0, 0, 0, 0, 1000, 0],
            close_speed=550,
        ),
    }
    return presets


def apply_angles(hand: RH56Hand, angles: Sequence[int], label: str) -> None:
    """Wrapper to send angles with a short log."""
    response = hand.angle_set(list(angles))
    if response is None:
        print(f"[WARN] Failed to apply {label} command.")
    else:
        print(f"[OK] {label}: {list(angles)}")


def apply_speed(hand: RH56Hand, speed: int, label: str) -> None:
    """Set identical speeds for all DOF."""
    values = [speed] * 6
    response = hand.speed_set(values)
    if response is None:
        print(f"[WARN] Failed to set speeds for {label}.")
    else:
        print(f"[OK] Speeds set to {speed} for {label}.")


def clear_and_open(hand: RH56Hand) -> None:
    """Clear errors and fully open the hand silently."""
    hand.clear_errors()
    hand.angle_set(DEFAULT_OPEN)


def wait_for_action(prompt: str) -> str:
    """
    Wait for user input.
    Returns: 'continue' (Enter), 'back', or 'quit'.
    """
    raw = input(prompt).strip().lower()
    if raw in {"q", "quit", "exit"}:
        return "quit"
    if raw in {"b", "back"}:
        return "back"
    if raw:
        print("Press Enter to continue, 'b' to go back, or 'q' to quit.")
        return wait_for_action(prompt)
    return "continue"


def run_preset_mode(hand: RH56Hand, preset: GraspPreset) -> bool:
    """Execute the two-stage Enter interaction for a grasp preset."""
    print(f"\n=== Mode {preset.key}: {preset.name} ===")
    print(preset.description)
    apply_angles(hand, preset.prepare_angles, "Prepare posture")

    action = wait_for_action("Press Enter to CLOSE, 'b' to change mode, 'q' to quit: ")
    if action == "quit":
        return False
    if action == "back":
        return True
    hand.force_set([1000] * 6)  # reset forces before closing
    apply_speed(hand, preset.close_speed, "closing")
    apply_angles(hand, preset.close_angles, "Closing fingers")

    action = wait_for_action("Press Enter to OPEN, 'b' to change mode, 'q' to quit: ")
    if action == "quit":
        return False
    # Even if user presses 'b', still open before returning
    apply_speed(hand, preset.restore_speed, "re-opening")
    apply_angles(hand, DEFAULT_OPEN, "Opening fingers")
    print("Run complete. Returning to main menu.\n")
    return action != "quit"


# --- Force-reactive pinch mode configuration ---
FORCE_LIMIT = 800
FORCE_SPIKE_THRESHOLD = 75  # g increase between averaged samples triggers spike detection
FORCE_DROP_TOLERANCE = 25    # g drop between averaged samples required to release
FORCE_SPIKE_MIN_FORCE = 500  # Index force must exceed this before spike detection runs
FORCE_SAMPLE_INTERVAL = 1.0 / 60.0  # seconds between raw readings (~60 Hz)
FORCE_AVG_WINDOW = 0.5       # seconds of data per averaged sample
FORCE_MAX_DURATION = 20.0    # fallback timeout in seconds
FORCE_TARGET_FINGERS = [INDEX, THUMB_BEND]
FORCE_MODE_PRESET_KEY = "1"  # reuse preset 1 posture/closing pattern


def read_forces(hand: RH56Hand) -> Optional[List[int]]:
    """Helper to read force data with retry logging."""
    forces = hand.force_act()
    if forces is None:
        print("[WARN] Unable to read force data.")
    return forces


def run_force_reactive_mode(hand: RH56Hand, preset: GraspPreset) -> bool:
    """Special mode 'x': two-finger grasp with force monitoring."""
    print("\n=== Force-reactive pinch mode (x) ===")
    print(
        "Flow: Prepare posture -> Enter to start -> limit forces -> monitor thumb force "
        "for a +75 g jump (auto close) ->\n"
        "monitor 0.5 s averaged index force at ~60 Hz. Once the averaged force exceeds "
        f"{FORCE_SPIKE_MIN_FORCE} g and the next average jumps by {FORCE_SPIKE_THRESHOLD} g\n"
        f"relative to the previous window, we mark a spike; once a "
        f"{FORCE_DROP_TOLERANCE} g decrease between consecutive averages is observed, the hand opens automatically."
    )

    while True:
        apply_angles(hand, preset.prepare_angles, "Force mode prepare posture")
        print("[INFO] Force-reactive cycle armed (auto loop). Press Ctrl+C to exit mode.")

        limits = [1000] * 6
        for idx in FORCE_TARGET_FINGERS:
            limits[idx] = FORCE_LIMIT
        response = hand.force_set(limits)
        if response is None:
            print("[WARN] Failed to set force limits; continuing anyway.")
        else:
            print(f"[OK] Force limits applied ({FORCE_LIMIT} g) for {FORCE_TARGET_FINGERS}.")

        print(
            f"Monitoring raw thumb force for a +{FORCE_SPIKE_THRESHOLD} g jump to trigger closure "
            "(no baseline threshold)."
        )
        thumb_prev: Optional[float] = None
        thumb_start = time.time()
        while True:
            thumb_forces = read_forces(hand)
            if not thumb_forces:
                time.sleep(FORCE_SAMPLE_INTERVAL)
                continue

            thumb_force = float(thumb_forces[THUMB_BEND])
            if thumb_prev is not None:
                thumb_delta = thumb_force - thumb_prev
                if thumb_delta >= FORCE_SPIKE_THRESHOLD:
                    print(
                        f"[INFO] Thumb spike detected (+{thumb_delta:.1f} g vs last sample). "
                        "Initiating closure."
                    )
                    break
            thumb_prev = thumb_force
            if time.time() - thumb_start >= FORCE_MAX_DURATION:
                print("[WARN] Thumb spike not detected within timeout. Closing anyway.")
                break
            time.sleep(FORCE_SAMPLE_INTERVAL)

        apply_speed(hand, preset.close_speed, "force mode closing")
        apply_angles(hand, preset.close_angles, "Closing for force mode")

        print(
            f"Streaming index force at ~60 Hz and averaging every {FORCE_AVG_WINDOW:.1f} s. "
            f"Waiting for averaged readings to exceed {FORCE_SPIKE_MIN_FORCE} g, then looking for "
            f"+{FORCE_SPIKE_THRESHOLD} g spikes followed by -{FORCE_DROP_TOLERANCE} g drops between "
            "consecutive averaged samples."
        )

        start_time = time.time()
        window_start = start_time
        window_samples: List[float] = []
        prev_avg: Optional[float] = None
        monitor_active = False
        spike_detected = False

        while True:
            forces = read_forces(hand)
            if not forces:
                time.sleep(FORCE_SAMPLE_INTERVAL)
                continue

            index_force = float(forces[INDEX])
            now = time.time()
            elapsed = now - start_time

            if elapsed >= FORCE_MAX_DURATION:
                print("[WARN] Timeout reached without detecting drop. Opening hand.")
                break

            window_samples.append(index_force)
            window_duration = now - window_start
            if window_duration < FORCE_AVG_WINDOW:
                time.sleep(FORCE_SAMPLE_INTERVAL)
                continue

            sample_count = len(window_samples)
            if sample_count == 0:
                window_start = now
                time.sleep(FORCE_SAMPLE_INTERVAL)
                continue

            avg_force = sum(window_samples) / sample_count
            print(
                f"[{elapsed:6.2f}s] Index avg ({sample_count} samples / {FORCE_AVG_WINDOW:.1f}s window) "
                f"= {avg_force:7.2f} g"
            )

            window_samples = []
            window_start = now

            if prev_avg is None:
                prev_avg = avg_force
                time.sleep(FORCE_SAMPLE_INTERVAL)
                continue

            delta = avg_force - prev_avg

            if not monitor_active:
                if avg_force >= FORCE_SPIKE_MIN_FORCE:
                    monitor_active = True
                    print(
                        f"[INFO] Averaged force exceeded {FORCE_SPIKE_MIN_FORCE} g. "
                        "Spike detection armed."
                    )
                prev_avg = avg_force
                time.sleep(FORCE_SAMPLE_INTERVAL)
                continue

            if not spike_detected:
                if delta >= FORCE_SPIKE_THRESHOLD:
                    spike_detected = True
                    print(
                        f"[INFO] Spike detected (+{delta:.1f} g vs last average). "
                        f"Waiting for a -{FORCE_DROP_TOLERANCE} g drop."
                    )
            else:
                drop = -delta
                if drop >= FORCE_DROP_TOLERANCE:
                    print(
                        f"[INFO] Drop detected ({drop:.1f} g decrease vs last average). Opening hand."
                    )
                    break

            prev_avg = avg_force
            time.sleep(FORCE_SAMPLE_INTERVAL)

        print("[INFO] Clearing errors before reopening.")
        clear_response = hand.clear_errors()
        if clear_response is None:
            print("[WARN] Failed to clear errors before reopening.")
        else:
            print("[OK] Errors cleared before reopening.")

        apply_speed(hand, 500, "force mode reopening")
        apply_angles(hand, DEFAULT_OPEN, "Opening fingers (pass 1/2)")
        time.sleep(1.0)
        apply_angles(hand, DEFAULT_OPEN, "Opening fingers (pass 2/2)")
        hand.force_set([1000] * 6)
        print("Force-reactive cycle complete. Returning to prepare stage.\n")


def print_status(hand: RH56Hand) -> None:
    """Display quick hand telemetry."""
    status = hand.status_read()
    angles = hand.angle_read()
    forces = hand.force_act()
    print("Status snapshot:")
    print(f"  Raw status: {status}")
    print(f"  Angles: {angles}")
    print(f"  Forces: {forces}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive experiment harness for RH56 grasp trials (direct serial connection)."
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial device for the RH56 hand (default matches macOS USB adapter).",
    )
    parser.add_argument(
        "--hand-id",
        type=int,
        default=1,
        help="Modbus slave ID (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        hand = RH56Hand(port=args.port, hand_id=args.hand_id)
        print(f"[OK] RH56 hand initialized on {args.port} (ID {args.hand_id}).")
    except Exception as exc:
        print(f"[ERR] Failed to initialize RH56 hand: {exc}")
        return

    presets = build_presets()

    try:
        while True:
            print("\nAvailable grasp modes:")
            for key in ("1", "2", "3"):
                preset = presets[key]
                print(f"  {key}. {preset.name} - {preset.description}")
            print("  x. Force-reactive pinch (auto open after force drop)")
            print("  o. Clear errors and fully open")
            print("  c. Clear errors only")
            print("  s. Print status snapshot")
            print("  q. Quit\n")

            choice = input("Select a mode (1/2/3/x/o/c/s/q): ").strip().lower()

            if choice in presets:
                if not run_preset_mode(hand, presets[choice]):
                    break
                continue

            if choice == "x":
                preset = presets.get(FORCE_MODE_PRESET_KEY)
                if not preset:
                    print(f"[ERR] Force mode preset '{FORCE_MODE_PRESET_KEY}' missing.")
                    continue
                if not run_force_reactive_mode(hand, preset):
                    break
                continue

            if choice == "o":
                clear_and_open(hand)
                print("[OK] Hand opened.\n")
                continue

            if choice == "c":
                response = hand.clear_errors()
                if response is None:
                    print("[WARN] Failed to clear errors.")
                else:
                    print("[OK] Errors cleared.")
                continue

            if choice == "s":
                print_status(hand)
                continue

            if choice in {"q", "quit", "exit"}:
                print("Exiting experiment.")
                break

            print("Unknown selection. Please choose 1, 2, 3, x, o, c, s, or q.")

    except KeyboardInterrupt:
        print("\n[INFO] Experiment interrupted by user.")

    finally:
        try:
            apply_angles(hand, DEFAULT_OPEN, "Final open")
        except Exception:
            pass


if __name__ == "__main__":
    main()
