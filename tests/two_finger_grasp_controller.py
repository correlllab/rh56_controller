#!/usr/bin/env python3
"""
Interactive two-finger grasp controller using workspace lookup data.

Features
--------
1. Mode selection for speed and force presets.
2. Distance-based lookup of index/thumb controller positions.
3. Multiple grasp strategies selectable via the GRASP_STRATEGY variable.
4. Continuous monitoring of angles and forces until the user stops sampling.

Run:
    python tests/two_finger_grasp_controller.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

try:
    import select  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    select = None  # type: ignore


# Resolve project paths so imports work when invoked from repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from rh56_controller.rh56_hand import RH56Hand  # noqa: E402
from workspace_lookup import WorkspaceLookup  # noqa: E402


# ----- Configurable strategy toggles -----
# Update these constants to try different motion strategies.
GRASP_STRATEGY = "simultaneous"
# Options: "simultaneous", "stepped", "thumb_first", "index_first"

STEP_SIZE = 20          # Default controller step per increment
STEP_DELAY_SEC = 0.05   # Delay between intermediate updates
FIXED_THUMB_POSITION = 550  # Default thumb bend position for the fixed-thumb mode

HAND_PORT = "/dev/ttyUSB0"
HAND_ID = 1

# Finger index mapping for the low-level API.
PINKY_IDX = 0
RING_IDX = 1
MIDDLE_IDX = 2
INDEX_IDX = 3
THUMB_BEND_IDX = 4
THUMB_ROTATION_IDX = 5


@dataclass
class ModeConfig:
    speeds: List[int]
    force_limits: List[int]
    description: str
    use_synced_steps: bool = False
    step_size: int = STEP_SIZE
    step_delay: float = STEP_DELAY_SEC
    fixed_thumb_position: Optional[int] = None
    lock_thumb: bool = False
    force_index_close: bool = False


def prompt_int(prompt: str, minimum: int, maximum: int, default: Optional[int] = None) -> int:
    """Prompt the user for an integer within [minimum, maximum]."""
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        if raw.isdigit():
            value = int(raw)
            if minimum <= value <= maximum:
                return value
        print(f"Please enter a number between {minimum} and {maximum}.")


def select_mode() -> ModeConfig:
    """Prompt once for mode selection and return the corresponding configuration."""
    print("\nMode selection (enter 1, 2, 3, or 4):")
    print("  1. High-speed grasp (speed=1000, force limit=1000)")
    print("  2. Slow speed (speed=1) with custom force limit")
    print("  3. Custom speed and force limit")
    print(f"  4. Fixed-thumb pinch (thumb bend fixed at {FIXED_THUMB_POSITION}, index closes to 0)")

    while True:
        choice = input("Select mode: ").strip()
        if choice == "1":
            speeds = [1000] * 6
            forces = [1000] * 6
            step_size = prompt_int(
                f"Enter synchronized step size (1-1000, default {STEP_SIZE}): ",
                1,
                1000,
                default=STEP_SIZE,
            )
            return ModeConfig(
                speeds,
                forces,
                f"Mode 1: high-speed with synchronized stepping (step size {step_size})",
                use_synced_steps=True,
                step_size=step_size,
                step_delay=STEP_DELAY_SEC,
            )
        if choice == "2":
            custom_force = prompt_int("Enter force limit for thumb/index (0-1000): ", 0, 1000)
            speeds = [1] * 6
            forces = [1000] * 6
            forces[INDEX_IDX] = custom_force
            forces[THUMB_BEND_IDX] = custom_force
            return ModeConfig(speeds, forces, f"Mode 2: slow speed, force limit={custom_force}")
        if choice == "3":
            custom_speed = prompt_int("Enter speed value for thumb/index (0-1000): ", 0, 1000)
            custom_force = prompt_int("Enter force limit for thumb/index (0-1000): ", 0, 1000)
            speeds = [1000] * 6
            speeds[INDEX_IDX] = custom_speed
            speeds[THUMB_BEND_IDX] = custom_speed
            forces = [1000] * 6
            forces[INDEX_IDX] = custom_force
            forces[THUMB_BEND_IDX] = custom_force
            return ModeConfig(
                speeds,
                forces,
                f"Mode 3: custom speed={custom_speed}, force limit={custom_force}",
            )
        if choice == "4":
            speeds = [1000] * 6
            forces = [1000] * 6
            return ModeConfig(
                speeds,
                forces,
                f"Mode 4: fixed thumb at {FIXED_THUMB_POSITION}, index closes to 0",
                fixed_thumb_position=FIXED_THUMB_POSITION,
                lock_thumb=True,
                force_index_close=True,
            )
        print("Invalid mode. Please enter 1, 2, 3, or 4.")


def configure_hand(hand: RH56Hand, mode: ModeConfig) -> None:
    """Apply speed and force settings to the physical hand."""
    print(f"\nApplying {mode.description}")
    if response := hand.speed_set(mode.speeds):
        print("Speed settings updated.")
    else:
        print("Failed to update speed settings (continuing).")
    if response := hand.force_set(mode.force_limits):
        print("Force limits updated.")
    else:
        print("Failed to update force limits (continuing).")


def build_angle_command(index_val: int, thumb_val: int, base_value: int = 0) -> List[int]:
    """Construct a full 6-DOF angle list using index and thumb values."""
    angles = [base_value] * 6
    angles[INDEX_IDX] = index_val
    angles[THUMB_BEND_IDX] = thumb_val
    return angles


def interpolate_values(start: int, target: int, step: int) -> List[int]:
    """Generate intermediate values from start to target using linear interpolation."""
    if start == target:
        return []
    step = max(1, step)
    total_delta = target - start
    steps = max(1, math.ceil(abs(total_delta) / step))
    return [int(round(start + total_delta * (i / steps))) for i in range(1, steps + 1)]


def apply_angles(hand: RH56Hand, angles: Sequence[int]) -> None:
    """Send the angle command to the hand."""
    hand.angle_set(list(angles))


def move_with_strategy(
    hand: RH56Hand,
    target_angles: List[int],
    strategy: str,
    step_size: int,
    step_delay: float,
    sync_steps: bool = False,
    sync_step_size: int = STEP_SIZE,
    sync_step_delay: float = STEP_DELAY_SEC,
) -> None:
    """Dispatch movement according to the configured strategy."""
    strategy = strategy.lower()
    current_angles = hand.angle_read() or [1000] * 6

    if sync_steps:
        move_indices = [INDEX_IDX, THUMB_BEND_IDX]
        start_angles = current_angles[:]
        max_delta = max(
            abs(target_angles[idx] - start_angles[idx]) for idx in move_indices
        )
        if max_delta == 0:
            apply_angles(hand, target_angles)
            return

        steps = max(1, math.ceil(max_delta / max(1, sync_step_size)))
        print(f"Synchronized stepping ({steps} steps, step size {sync_step_size}).")

        for step in range(1, steps + 1):
            next_angles = start_angles[:]
            for idx in range(6):
                if idx in move_indices:
                    start_val = start_angles[idx]
                    end_val = target_angles[idx]
                    next_angles[idx] = int(
                        round(start_val + (end_val - start_val) * (step / steps))
                    )
                else:
                    next_angles[idx] = target_angles[idx]
            apply_angles(hand, next_angles)
            time.sleep(sync_step_delay)

        # Ensure final command matches target precisely.
        apply_angles(hand, target_angles)
        return

    if strategy == "simultaneous":
        apply_angles(hand, target_angles)
        return

    if strategy == "stepped":
        move_indices = [INDEX_IDX, THUMB_BEND_IDX]
        sequences = {
            idx: interpolate_values(current_angles[idx], target_angles[idx], step_size)
            for idx in move_indices
        }
        steps = max((len(seq) for seq in sequences.values()), default=0)
        for step in range(steps):
            next_angles = current_angles[:]
            for idx in move_indices:
                seq = sequences[idx]
                if step < len(seq):
                    next_angles[idx] = seq[step]
                else:
                    next_angles[idx] = target_angles[idx]
            apply_angles(hand, next_angles)
            time.sleep(step_delay)
            current_angles = next_angles
        return

    if strategy in {"thumb_first", "index_first"}:
        order = [THUMB_BEND_IDX, INDEX_IDX] if strategy == "thumb_first" else [INDEX_IDX, THUMB_BEND_IDX]
        for idx in order:
            sequences = interpolate_values(current_angles[idx], target_angles[idx], step_size)
            for value in sequences:
                current_angles[idx] = value
                apply_angles(hand, current_angles)
                time.sleep(step_delay)
            current_angles[idx] = target_angles[idx]
            apply_angles(hand, current_angles)
            time.sleep(step_delay)
        return

    print(f"Unknown strategy '{strategy}', defaulting to simultaneous.")
    apply_angles(hand, target_angles)


def monitor_hand(hand: RH56Hand, interval: float = 0.5) -> None:
    """
    Periodically print angle and force readings until the user stops monitoring.
    Accepts 's' (or 'stop') followed by Enter to exit the loop.
    """
    print("Monitoring angles and forces. Type 's' then Enter to stop.")
    iteration = 0
    try:
        while True:
            angles = hand.angle_read()
            forces = hand.force_act()
            iteration += 1
            print(f"[{iteration:02d}] Angles: {angles} | Forces: {forces}")

            if select and hasattr(select, "select"):
                ready, _, _ = select.select([sys.stdin], [], [], interval)
                if ready:
                    command = sys.stdin.readline().strip().lower()
                    if command in {"s", "stop"}:
                        print("Monitoring stopped.")
                        return
                    print("Continuing monitoring. Type 's' to stop.")
            else:
                time.sleep(interval)
                # Fallback prompt (only used where non-blocking input is unavailable).
                if sys.platform.startswith("win"):  # pragma: no cover - Windows specific
                    command = input("Enter 's' to stop monitoring or press Enter to continue: ").strip().lower()
                    if command in {"s", "stop"}:
                        print("Monitoring stopped.")
                        return
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")


def main() -> None:
    """Entry point."""
    try:
        hand = RH56Hand(port=HAND_PORT, hand_id=HAND_ID)
        print("RH56 hand initialized.")
    except Exception as exc:  # pragma: no cover - hardware init failure handling
        print(f"Failed to initialize RH56 hand: {exc}")
        return

    lookup = WorkspaceLookup()
    mode = select_mode()
    configure_hand(hand, mode)

    min_cm, max_cm = lookup.get_distance_range_cm()
    print(f"\nDistance lookup ready. Reachable range: {min_cm:.2f} - {max_cm:.2f} cm")
    print("Enter a distance in cm, 'o' to fully open the hand, or 'q' to quit.")

    while True:
        user_input = input("\nTarget distance (cm) or command: ").strip().lower()

        if user_input in {"q", "quit", "exit"}:
            print("Exiting controller.")
            break

        if user_input in {"o", "open"}:
            open_angles = [1000] * 6
            print("Opening hand to 1000 on all joints.")
            apply_angles(hand, open_angles)
            continue

        if user_input == "g":
            grasp_prep = [0] * 6
            grasp_prep[INDEX_IDX] = 1000
            grasp_prep[THUMB_BEND_IDX] = 550  # 550 is a good prep position for grabbing small objects
            print("Preparing for grasp: setting index/thumb bend to 1000, others to 0.")
            apply_angles(hand, grasp_prep)
            continue

        try:
            target_distance = float(user_input)
        except ValueError:
            print("Invalid input. Enter a distance in centimeters, 'o', or 'q'.")
            continue

        if mode.force_index_close:
            index_pos = 0
            thumb_pos = mode.fixed_thumb_position if mode.fixed_thumb_position is not None else 0
            print("\nFixed-thumb mode: ignoring distance input and applying preset positions.")

            try:
                actual_distance_m = lookup.get_distance_for_positions(index_pos, thumb_pos)
            except Exception:
                actual_distance_m = float("nan")

            def coords_for(finger: str, position: int) -> tuple[float, float]:
                table = lookup.tables[finger]
                row = table[table['position'] == position]
                if len(row) == 0:
                    return (float("nan"), float("nan"))
                return (float(row['x_world'].iloc[0]), float(row['y_world'].iloc[0]))

            index_coords = coords_for('index', index_pos)
            thumb_coords = coords_for('thumb_bend', thumb_pos)
            print(f"  Index position: {index_pos}")
            print(f"  Thumb position: {thumb_pos}")
            if not math.isnan(actual_distance_m):
                print(f"  Achieved distance (cm): {actual_distance_m * 100:.3f}")
            print(f"  Index world coords: ({index_coords[0]:.4f}, {index_coords[1]:.4f}) m")
            print(f"  Thumb world coords: ({thumb_coords[0]:.4f}, {thumb_coords[1]:.4f}) m")
        else:
            result = lookup.get_positions_for_distance_cm(target_distance, return_details=True)
            index_pos = result["index_position"]
            thumb_pos = result["thumb_position"]
            index_coords = result["index_coords"]
            thumb_coords = result["thumb_coords"]

            print("\nLookup result:")
            print(f"  Target distance: {target_distance:.2f} cm")
            print(f"  Achieved distance: {result['actual_distance'] * 100:.3f} cm")
            print(f"  Index position: {index_pos} (theta={result['index_theta']:.2f}°)")
            print(f"  Thumb position: {thumb_pos} (theta={result['thumb_theta']:.2f}°)")
            print(f"  Index world coords: ({index_coords[0]:.4f}, {index_coords[1]:.4f}) m")
            print(f"  Thumb world coords: ({thumb_coords[0]:.4f}, {thumb_coords[1]:.4f}) m")

        target_angles = build_angle_command(index_pos, thumb_pos)
        move_with_strategy(
            hand,
            target_angles,
            GRASP_STRATEGY,
            STEP_SIZE,
            STEP_DELAY_SEC,
            sync_steps=mode.use_synced_steps,
            sync_step_size=mode.step_size,
            sync_step_delay=mode.step_delay,
        )

        monitor_hand(hand, interval=0.5)


if __name__ == "__main__":
    main()
