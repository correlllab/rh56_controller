#!/usr/bin/env python3
"""
Fixed-thumb grasp controller with two grasp presets.

Mode 1 (large object):
    - Thumb bend starts at 1000, thumb rotation at 0.
    - All other fingers start at 1000.
    - Command '1' closes every finger (including thumb rotation) to 0.

Mode 2 (small object pinch):
    - Thumb bend fixed at 550, thumb rotation at 0.
    - Index finger starts at 1000, remaining fingers at 0.
    - Command '1' closes only the index finger to 0 (thumb stays fixed).

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
from typing import List, Sequence


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


@dataclass(frozen=True)
class ModeConfig:
    description: str
    initial_angles: List[int]
    close_angles: List[int]


MODE_PRESETS = {
    "1": ModeConfig(
        description="Mode 1 (large object): thumb bend 1000, all fingers start at 1000. '1' closes all to 0.",
        initial_angles=[1000, 1000, 1000, 1000, 1000, 0],
        close_angles=[0, 0, 0, 0, 0, 0],
    ),
    "2": ModeConfig(
        description="Mode 2 (small object pinch): thumb bend 550, others 0 except index 1000. '1' closes index to 0.",
        initial_angles=[0, 0, 0, 1000, 550, 0],
        close_angles=[0, 0, 0, 0, 550, 0],
    ),
}


def apply_angles(hand: RH56Hand, angles: Sequence[int], label: str) -> None:
    """Send angle command and print a brief summary."""
    response = hand.angle_set(list(angles))
    if response is not None:
        print(f"{label}: {list(angles)}")
    else:
        print(f"Failed to apply angles for {label}.")


def prepare_to_grab(hand: RH56Hand) -> None:
    """Set index/thumb bend to 1000, others to 0."""
    posture = [0, 0, 0, 1000, 1000, 0]
    apply_angles(hand, posture, "Prepare-to-grab posture")


def fully_open(hand: RH56Hand) -> None:
    """Open all joints to 1000."""
    posture = [1000] * 6
    apply_angles(hand, posture, "Fully open posture")


def select_mode() -> ModeConfig:
    """Prompt user until a valid mode is selected."""
    print("\nAvailable modes:")
    for mode_id, cfg in MODE_PRESETS.items():
        print(f"  {mode_id}. {cfg.description}")

    while True:
        choice = input("Select mode (1/2): ").strip()
        if choice in MODE_PRESETS:
            return MODE_PRESETS[choice]
        print("Invalid selection. Please enter '1' or '2'.")


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

    mode = select_mode()
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
    print("  g - prepare-to-grab posture (index/thumb bend 1000, others 0)")
    print("  o - fully open hand (all joints 1000)")
    print("  c - clear errors")
    print("  s - print status")
    print("  q - quit\n")

    while True:
        cmd = input("Enter command: ").strip().lower()

        if cmd in {"q", "quit", "exit"}:
            print("Exiting controller.")
            break

        if cmd == "1":
            apply_angles(hand, mode.close_angles, "Grasp command")
            continue

        if cmd == "g":
            prepare_to_grab(hand)
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
