#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torque-force mapping experiment script (DIRECT, interactive, multi-run).

保留原有逻辑与交互：
1) 选择手指（其余保持角度 1000）。
2) 为所选手指设置力度上限 (0-1000)。
3) 以 speed=1 持续关闭该手指，约 60 Hz 采样 force/current；
   读取稳定（连续若干样本 |Δforce| <= 阈值）后开始 1 Hz 的平均输出。
4) 按 Enter 结束当前轮次，手打开复位，继续下一轮；可继续设置新力度或换手指。

说明：
- 直连 RH56Hand；使用你的仓库 API：hand.force_set([...])、hand.force_act()、hand.current_read()。
- 其它逻辑不改：稳定阈值/计数、输出格式、速度=1、复位与提示等。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    import select  # type: ignore
except ImportError:  # Windows 等无 select 情况
    select = None  # type: ignore

# 确保项目根可导入
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

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
    """交互选择手指索引。"""
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
    """输入力度上限（0-1000）。"""
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
    """仅为所选手指设置力度上限，其余保持 1000。"""
    limits = [1000] * 6
    limits[finger] = limit
    try:
        resp = hand.force_set(limits)
    except Exception as e:
        print(f"Error: force_set failed: {e}")
        return False
    return resp is not None


def set_all_angles(hand: RH56Hand, angles: List[int]) -> None:
    """angle_set 封装，失败给出警告。"""
    try:
        resp = hand.angle_set(angles)
        if resp is None:
            print("Warning: Failed to apply angle command.")
    except Exception as e:
        print(f"Warning: angle_set exception: {e}")


def set_speeds(hand: RH56Hand, speeds: List[int]) -> None:
    """speed_set 封装，失败给出警告。"""
    try:
        resp = hand.speed_set(speeds)
        if resp is None:
            print("Warning: Failed to apply speed command.")
    except Exception as e:
        print(f"Warning: speed_set exception: {e}")


def wait_for_enter(timeout: float) -> Optional[str]:
    """在 timeout 秒内检查是否有用户输入（回车/quit）。"""
    if select and hasattr(select, "select"):
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except Exception:
            time.sleep(timeout)
            return None
        if ready:
            return sys.stdin.readline().strip().lower()
        return None
    # 无 select 的平台简单 sleep
    time.sleep(timeout)
    return None


def run_force_calibration(hand: RH56Hand) -> None:
    """Force sensor calibration workflow, adapted from the standalone demo."""
    print("\n=== Force Sensor Calibration Mode ===")

    angles = hand.angle_read()
    print(f"Pre-calibration angles: {angles}")
    forces = hand.force_act()
    if forces:
        print("Pre-calibration finger forces:")
        for idx in range(6):
            name = FINGER_LABELS.get(idx, f"Finger {idx}")
            print(f"  {name}: {forces[idx]:4d} g")
    else:
        print("Warning: Unable to read pre-calibration forces.")

    input("Press Enter to start calibration...")
    print("Starting calibration, process will take about 15 seconds...")
    try:
        hand.gesture_force_clb(1)
    except Exception as exc:
        print(f"Failed to trigger calibration: {exc}")
        return

    sequence = [
        ("Calibrating: All fingers opening...", 3),
        ("Calibrating: Four fingers bending...", 3),
        ("Calibrating: Four fingers opening...", 3),
        ("Calibrating: Thumb bending...", 3),
        ("Calibrating: Thumb opening...", 3),
    ]
    for message, delay in sequence:
        print(message)
        time.sleep(delay)

    print("Calibration complete, waiting for system stabilization...")
    time.sleep(2)

    finger_names = [FINGER_LABELS.get(i, f"Finger {i}") for i in range(6)]
    print("Reading post-calibration forces...")
    attempts = 0
    post_forces: Optional[List[int]] = None
    while attempts < 3:
        post_forces = hand.force_act()
        if post_forces:
            print(f"Post-calibration finger forces (Attempt {attempts + 1}):")
            for name, value in zip(finger_names, post_forces):
                print(f"  {name}: {value:4d} g")
            break
        print(f"Attempt {attempts + 1} failed to read forces, retrying...")
        attempts += 1
        time.sleep(1)

    if not post_forces:
        print("Warning: Unable to read post-calibration forces, check device connection.")

    save = input("Save parameters? (y/n): ").strip().lower()
    if save == "y":
        try:
            hand.save_parameters()
            print("Parameters saved.")
        except Exception as exc:
            print(f"Failed to save parameters: {exc}")

    test = input("Test post-calibration force sensor response? (y/n): ").strip().lower()
    if test == "y":
        print("Press each finger in sequence to observe force changes (Ctrl+C to exit)...")
        try:
            while True:
                forces = hand.force_act()
                if forces:
                    readings = "  ".join(f"{name}: {value:4d}" for name, value in zip(finger_names, forces))
                    print(f"Current forces (g): {readings}")
                else:
                    print("Unable to read forces.")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nForce sensor test complete.")


def run_measurement(hand: RH56Hand, finger: int, force_limit: int) -> None:
    """按当前配置执行一次测量。"""
    print(f"\nSetting force limit {force_limit}g on {FINGER_LABELS[finger]}...")
    if not apply_force_limit(hand, finger, force_limit):
        print("Failed to set force limit; aborting this run.")
        return

    open_angles = [1000] * 6
    set_all_angles(hand, open_angles)

    speeds = [1000] * 6
    speeds[finger] = 10  # 按你的逻辑固定为 1
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

            # 读取力与电流
            try:
                forces = hand.force_act()
                currents = hand.current_read()
            except Exception as e:
                print(f"Warning: read exception: {e}")
                time.sleep(READ_INTERVAL)
                continue

            if forces is None or currents is None:
                print("Warning: Failed to read force/current; retrying...")
                time.sleep(READ_INTERVAL)
                continue

            force_value = forces[finger]
            current_value = currents[finger]

            # 稳定判据：|Δforce| <= 阈值 连续 N 次
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

            # 稳定后开始 1 Hz 平均输出
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

            # 非阻塞检查用户输入（Enter 结束；q 退出整个程序）
            user_input = wait_for_enter(max(0.0, READ_INTERVAL - (time.time() - loop_start)))
            if user_input is not None:
                if user_input in {"q", "quit", "exit"}:
                    raise KeyboardInterrupt
                # 其它输入（包括空行/回车）结束当前轮
                print("Stop requested by user.")
                break

    except KeyboardInterrupt:
        print("\nMeasurement interrupted by user.")

    finally:
        print("Restoring hand to open state and resetting speeds.")
        try:
            set_speeds(hand, [1000] * 6)
            set_all_angles(hand, open_angles)
        finally:
            time.sleep(0.2)  # brief pause to ensure command execution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torque-force mapping experiment script (direct serial connection, interactive multi-run)."
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbserial-2130",
        help="Serial port for the RH56 hand (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--hand-id",
        type=int,
        default=1,
        help="Modbus slave ID of the RH56 hand (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=("mapping", "calibration"),
        default="mapping",
        help="Select 'mapping' for the torque-force experiment or 'calibration' to run the force sensor calibration routine.",
    )
    args = parser.parse_args()

    try:
        hand = RH56Hand(port=args.port, hand_id=args.hand_id)
        print(f"RH56 hand initialized on {args.port} (ID {args.hand_id}).")
    except Exception as exc:
        print(f"Failed to initialize RH56 hand: {exc}")
        return

    try:
        if args.mode == "calibration":
            run_force_calibration(hand)
            return

        while True:
            # 选择手指（支持多轮）
            finger = prompt_finger()
            if finger < 0:
                print("Exiting experiment.")
                return

            while True:
                # 设置力度上限（支持多轮）
                force_limit = prompt_force_limit()
                if force_limit is None:
                    print("Exiting experiment.")
                    return

                # 执行一次测量
                run_measurement(hand, finger, force_limit)

                # 后续选择：继续该手指新力度、或切换手指、或退出
                follow_up = input(
                    "\nPress Enter to set a new force limit, "
                    "'finger' to choose another finger, or 'q' to quit: "
                ).strip().lower()

                if follow_up in {"q", "quit", "exit"}:
                    print("Exiting experiment.")
                    return
                if follow_up in {"finger", "f"}:
                    break  # 返回上一层，重新选手指
                # 其它输入（或直接回车）继续在同一手指上设新力度

    except KeyboardInterrupt:
        print("\nExperiment terminated by user.")
        try:
            set_speeds(hand, [1000] * 6)
            set_all_angles(hand, [1000] * 6)
        except Exception:
            pass


if __name__ == "__main__":
    main()
