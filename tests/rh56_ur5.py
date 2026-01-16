#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run coordinated UR5 + RH56 trials with poses defined directly in this file.

Procedure (per trial):
  1) clear errors + open hand
  2) set hand "prepare posture" (from your presets)
  3) move arm to *_prep pose
  4) close hand (preset speed + close angles)
  5) move arm to *_end pose (lift)
  6) optional shake (between end and shake pose)
  7) return to *_prep pose (or "return_to_start" if enabled)
  8) open hand

Usage:
  python3 run_arm_hand_trial.py --trial tripod --shake 4
  python3 run_arm_hand_trial.py --trial precision
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# --- UR5 + RH56 ---
from magpie_control import ur5
from rh56_controller.rh56_hand import RH56Hand

# --- Reuse your hand presets/utilities from experiment.py ---
from experiment import build_presets, apply_angles, apply_speed, clear_and_open, DEFAULT_OPEN


# =========================
#  Poses
# =========================

pose_precision_prep = np.array([
    [-0.988, -0.142,  0.061, -0.328],
    [-0.11,   0.37,  -0.923, -0.381],
    [ 0.109, -0.918, -0.381,  0.157],
    [ 0.,     0. ,    0. ,    1.   ]
], dtype=float)

pose_precision_end = np.array([
    [-0.988, -0.142,  0.061, -0.328],
    [-0.11,   0.37,  -0.923, -0.381],
    [ 0.109, -0.918, -0.381,  0.300],
    [ 0.,     0. ,    0. ,    1.   ]
], dtype=float)

pose_tripod_prep = np.array([
    [-0.991, -0.099,  0.091, -0.331],
    [-0.13,   0.538, -0.833, -0.404],
    [ 0.034, -0.837, -0.546,  0.194],
    [ 0.,     0.,     0.,     1.   ]
], dtype=float)

pose_tripod_end = np.array([
    [-0.991, -0.099,  0.091, -0.331],
    [-0.13,   0.538, -0.833, -0.404],
    [ 0.034, -0.837, -0.546,  0.300],
    [ 0.,     0.,     0.,     1.   ]
], dtype=float)

pose_tripod_shake = np.array([
    [-0.991, -0.099,  0.091, -0.331],
    [-0.13,   0.538, -0.833, -0.404],
    [ 0.034, -0.837, -0.546,  0.350],
    [ 0.,     0.,     0.,     1.   ]
], dtype=float)


TRIALS: Dict[str, Dict[str, Optional[np.ndarray]]] = {
    # "precision" == pinch-style in your naming; we’ll map it to preset key "1"
    "precision": {
        "prep": pose_precision_prep,
        "end": pose_precision_end,
        "shake": None,  # no shake pose provided; can add later if you want
    },
    "tripod": {
        "prep": pose_tripod_prep,
        "end": pose_tripod_end,
        "shake": pose_tripod_shake,
    },
}

# Map trial name -> your preset mode key in experiment.py presets dict
# (based on your earlier description: 1=pinch, 2=tripod)
TRIAL_TO_MODE = {
    "precision": "1",
    "tripod": "2",
}


def _log(stage: str, msg: str = "") -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] [{stage}] {msg}".rstrip())


def _assert_pose(T: np.ndarray, name: str) -> None:
    if not isinstance(T, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray")
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError(f"{name} has NaN/Inf")


def shake_between(robot, A: np.ndarray, B: np.ndarray, cycles: int, linSpeed: float, linAccel: float) -> None:
    _log("SHAKE", f"{cycles} cycles")
    for _ in range(cycles):
        robot.moveL(A, linSpeed=linSpeed, linAccel=linAccel, asynch=False)
        robot.moveL(B, linSpeed=linSpeed, linAccel=linAccel, asynch=False)


def run_trial(
    robot,
    hand: RH56Hand,
    trial_name: str,
    linSpeed: float,
    linAccel: float,
    settle_s: float,
    close_wait_s: float,
    shake_cycles: int,
    return_to_start: bool,
) -> None:
    if trial_name not in TRIALS:
        raise ValueError(f"Unknown trial '{trial_name}'. Options: {list(TRIALS.keys())}")

    poses = TRIALS[trial_name]
    prep = poses["prep"]
    end = poses["end"]
    shake_pose = poses["shake"]

    _assert_pose(prep, f"{trial_name}.prep")
    _assert_pose(end, f"{trial_name}.end")
    if shake_pose is not None:
        _assert_pose(shake_pose, f"{trial_name}.shake")

    mode_key = TRIAL_TO_MODE[trial_name]
    presets = build_presets()
    if mode_key not in presets:
        raise ValueError(f"Preset mode '{mode_key}' not in build_presets() keys: {list(presets.keys())}")
    preset = presets[mode_key]

    start_pose = robot.get_tcp_pose() if return_to_start else None

    # 1) safe start
    _log("INIT", "Clear errors + open hand")
    clear_and_open(hand)
    time.sleep(0.5)

    # 2) prep hand
    _log("HAND", f"Prepare posture: {trial_name} (mode {mode_key}: {preset.name})")
    apply_angles(hand, preset.prepare_angles, "Prepare posture")
    time.sleep(0.5)

    # 3) move arm to prep pose
    _log("ARM", f"MoveL -> {trial_name}.prep")
    robot.moveL(prep, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

    # 4) close hand
    time.sleep(5)
    _log("HAND", f"Close (speed={preset.close_speed})")
    apply_speed(hand, preset.close_speed, "closing speed")
    apply_angles(hand, preset.close_angles, "closing angles")
    time.sleep(2)

    # 5) lift
    _log("ARM", f"MoveL -> {trial_name}.end (lift)")
    robot.moveL(end, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

    # 6) optional shake
    if shake_cycles > 0:
        if shake_pose is None:
            _log("SHAKE", "No shake pose provided for this trial; skipping")
        else:
            # shake between end and shake_pose
            shake_between(robot, end, shake_pose, shake_cycles, linSpeed, linAccel)
            # return to end after shake
            robot.moveL(end, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

    # 7) return
    if return_to_start and start_pose is not None:
        _log("ARM", "Return to START pose")
        robot.moveL(start_pose, linSpeed=linSpeed, linAccel=linAccel, asynch=False)
    else:
        _log("ARM", f"Return to {trial_name}.prep")
        robot.moveL(prep, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

    # 8) open hand
    time.sleep(1)
    _log("HAND", f"Open (speed={preset.restore_speed})")
    apply_speed(hand, preset.restore_speed, "opening speed")
    apply_angles(hand, DEFAULT_OPEN, "opening angles")
    time.sleep(0.5)

    # 9) lift arm higher to clear
    robot.moveL(pose_precision_end, linSpeed=linSpeed, linAccel=linAccel, asynch=False)
    clear_and_open(hand)
    time.sleep(0.5)
    clear_and_open(hand)
    _log("DONE", "Trial complete")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", choices=list(TRIALS.keys()), default="precision",
                    help="Which pose set to run")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="RH56 serial port")
    ap.add_argument("--hand-id", type=int, default=1, help="RH56 hand ID")

    ap.add_argument("--linSpeed", type=float, default=0.5)
    ap.add_argument("--linAccel", type=float, default=0.75)
    ap.add_argument("--settle-s", type=float, default=0.2)
    ap.add_argument("--close-wait-s", type=float, default=0.6)

    ap.add_argument("--shake", type=int, default=0, help="Shake cycles (tripod supports shake pose)")
    ap.add_argument("--return-to-start", action="store_true",
                    help="Return arm to starting TCP pose instead of returning to prep pose")
    return ap.parse_args()


def main():
    args = parse_args()

    _log("CONNECT", "Hand")
    hand = RH56Hand(port=args.port, hand_id=args.hand_id)

    _log("CONNECT", "UR5")
    robot = ur5.UR5_Interface()
    robot.start()

    try:
        run_trial(
            robot=robot,
            hand=hand,
            trial_name=args.trial,
            linSpeed=args.linSpeed,
            linAccel=args.linAccel,
            settle_s=args.settle_s,
            close_wait_s=args.close_wait_s,
            shake_cycles=args.shake,
            return_to_start=args.return_to_start,
        )
    finally:
        # Always leave the hand open
        try:
            _log("CLEANUP", "Open hand")
            clear_and_open(hand)
            apply_angles(hand, DEFAULT_OPEN, "final open")
        except Exception as e:
            _log("CLEANUP", f"Hand cleanup failed: {e}")

        # If your UR5 interface exposes stop/close, call it here.
        # try: robot.stop()
        # except: pass


if __name__ == "__main__":
    main()
