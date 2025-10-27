#!/usr/bin/env python3
"""
Fixed-thumb grasp controller with two grasp presets.

Mode 1 (precision grasp):
    - Prompts for object length and width in centimeters.
    - Determines number of fingers:
        * length < 5 cm  -> use index + thumb (two-finger strategy, thumb rotation = 150)
        * 5 cm ≤ length ≤ 10 cm -> use index + middle + thumb (three fingers, thumb rotation = 0)
        * length > 10 cm -> use index + middle + ring + thumb (four fingers, thumb rotation = 0)
    - Thumb bend preset from width:
        * width < 3 cm  -> 650
        * 3 cm ≤ width < 5 cm -> 800
        * width ≥ 5 cm -> 1000
    - Prepares hand as [1000, 1000, 1000, 1000, thumb_bend, thumb_rotation] and
      closes the selected fingers at speed 500 before restoring speeds to 1000.

Mode 2 (full-hand grasp):
    - Matches the previous Mode 1 hand behavior (all fingers close from 1000 to 0).

Common utilities:
    - 'g' : prepare-to-grab posture (index/thumb bend 1000, others 0)
    - 'o' : fully open hand (all joints 1000)
    - 'c' : clear errors
    - 's' : print status
    - 'q' : quit
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


# Ensure project root is on sys.path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from rh56_controller.rh56_hand import RH56Hand  # noqa: E402


# Finger index mapping
PINKY_IDX = 0
RING_IDX = 1
MIDDLE_IDX = 2
INDEX_IDX = 3
THUMB_BEND_IDX = 4
THUMB_ROTATION_IDX = 5

HAND_PORT = "/dev/ttyUSB0"
HAND_ID = 1


@dataclass
class ModeConfig:
    description: str
    initial_angles: List[int]
    close_angles: List[int]
    speed_override: Optional[List[int]] = None
    speed_restore: Optional[List[int]] = None
    prepare_angles: Optional[List[int]] = None


MODE_PRESETS = {
    "2": ModeConfig(
        description="Mode 2 (full-hand grasp): thumb bend 1000, all fingers start at 1000. '1' closes all to 0.",
        initial_angles=[1000, 1000, 1000, 1000, 1000, 0],
        close_angles=[0, 0, 0, 0, 0, 0],
        prepare_angles=[1000, 1000, 1000, 1000, 1000, 0],
    ),
}


def apply_angles(hand: RH56Hand, angles: Sequence[int], label: str) -> None:
    """Send angle command and print a brief summary."""
    response = hand.angle_set(list(angles))
    if response is not None:
        print(f"{label}: {list(angles)}")
    else:
        print(f"Failed to apply angles for {label}.")


def prepare_to_grab(hand: RH56Hand, mode: ModeConfig) -> None:
    """Apply the mode-specific pre-grasp posture."""
    posture = mode.prepare_angles or mode.initial_angles
    apply_angles(hand, posture, "Prepare-to-grab posture")


def fully_open(hand: RH56Hand) -> None:
    """Clear errors and open all joints to 1000 without logging."""
    hand.clear_errors()
    hand.angle_set([1000] * 6)


def select_mode() -> str:
    """Prompt user until a valid mode choice (1 or 2) is selected."""
    print("\nAvailable modes:")
    print("  1. Mode 1 (precision grasp with fixed thumb)")
    print("  2. Mode 2 (full-hand grasp)")

    while True:
        choice = input("Select mode (1/2): ").strip()
        if choice in {"1", "2"}:
            return choice
        print("Invalid selection. Please enter '1' or '2'.")


def prompt_object_length() -> float:
    """Ask the user for object length in centimeters."""
    while True:
        raw = input("Enter object length (cm): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number (e.g., 4.5).")
            continue
        if value <= 0:
            print("Length must be positive.")
            continue
        return value


def prompt_object_width() -> float:
    """Ask the user for object width in centimeters."""
    while True:
        raw = input("Enter object width (cm): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number (e.g., 2.8).")
            continue
        if value <= 0:
            print("Width must be positive.")
            continue
        return value


def configure_precision_mode(length_cm: float, width_cm: float) -> ModeConfig:
    """Construct precision grasp configuration based on object dimensions."""
    if length_cm < 5.0:
        fingers = 2
        thumb_rotation = 150
    elif length_cm <= 10.0:
        fingers = 3
        thumb_rotation = 0
    else:
        fingers = 4
        thumb_rotation = 0

    if width_cm < 3.0:
        thumb_bend = 650
    elif width_cm < 5.0:
        thumb_bend = 800
    else:
        thumb_bend = 1000

    description = (
        f"Mode 1 (precision grasp): length={length_cm:.2f} cm, width={width_cm:.2f} cm -> "
        f"{fingers}-finger strategy, thumb bend {thumb_bend}, thumb rotation {thumb_rotation}."
    )

    initial = [1000, 1000, 1000, 1000, thumb_bend, thumb_rotation]
    close = initial.copy()
    if fingers >= 2:
        close[INDEX_IDX] = 0
    if fingers >= 3:
        close[MIDDLE_IDX] = 0
    if fingers >= 4:
        close[RING_IDX] = 0

    speed_override = [500] * 6
    speed_restore = [1000] * 6
    return ModeConfig(description, initial, close, speed_override, speed_restore, prepare_angles=initial.copy())


def prompt_speed_limit(default: int = 1000) -> int:
    """Prompt for a global speed limit (0-1000)."""
    prompt = f"Enter speed limit (0-1000, press Enter for {default}): "
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        if raw.isdigit():
            value = int(raw)
            if 0 <= value <= 1000:
                return value
        print("Please enter a number between 0 and 1000 (or press Enter for default).")


def main() -> None:
    try:
        hand = RH56Hand(port=HAND_PORT, hand_id=HAND_ID)
        print("RH56 hand initialized.")
    except Exception as exc:  # pragma: no cover - hardware init error handling
        print(f"Failed to initialize RH56 hand: {exc}")
        return

    mode_choice = select_mode()

    if mode_choice == "1":
        length_cm = prompt_object_length()
        width_cm = prompt_object_width()
        mode = configure_precision_mode(length_cm, width_cm)
        print(f"\nSelected Mode 1 with object length {length_cm:.2f} cm, width {width_cm:.2f} cm")
        print(mode.description)
    else:
        mode = MODE_PRESETS[mode_choice]
        print(f"\n{mode.description}")

    speed_limit = prompt_speed_limit()
    speed_values = [speed_limit] * 6
    if hand.speed_set(speed_values) is not None:
        print(f"Speed limit applied: {speed_limit} for all fingers.")
    else:
        print("Failed to apply speed limit (continuing with previous settings).")

    # Apply initial posture for the selected mode
    apply_angles(hand, mode.initial_angles, "Initial posture")

    print("\nCommands:")
    print("  1 - execute grasp action for current mode")
    print("  g - prepare-to-grab posture (mode-specific)")
    print("  o - clear errors then fully open hand (silent)")
    print("  c - clear errors")
    print("  s - print status")
    print("  q - quit\n")

    while True:
        cmd = input("Enter command: ").strip().lower()

        if cmd in {"q", "quit", "exit"}:
            print("Exiting controller.")
            break

        if cmd == "1":
            if mode.speed_override:
                if hand.speed_set(mode.speed_override) is not None:
                    print(f"Applied temporary speed override: {mode.speed_override[0]}")
                else:
                    print("Failed to apply temporary speed override.")

            apply_angles(hand, mode.close_angles, "Grasp command")

            if mode.speed_restore:
                if hand.speed_set(mode.speed_restore) is not None:
                    print("Speeds restored to 1000.")
                else:
                    print("Failed to restore speeds to 1000.")
            continue

        if cmd == "g":
            prepare_to_grab(hand, mode)
            continue

        if cmd == "o":
            fully_open(hand)
            continue

        if cmd == "c":
            print("Clearing errors...")
            response = hand.clear_errors()
            if response is not None:
                print("Errors cleared.")
            else:
                print("Failed to clear errors.")
            continue

        if cmd == "s":
            print("Hand status:")
            status = hand.status_read()
            angles = hand.angle_read()
            forces = hand.force_act()
            print(f"  Raw status: {status}")
            print(f"  Angles: {angles}")
            print(f"  Forces: {forces}")
            continue

        print("Unknown command. Available commands: 1, g, o, c, s, q.")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
