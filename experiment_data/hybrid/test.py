#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 middle-finger contact experiment (interactive single setting).

Workflow:
- Prompt the user once at startup for:
  - speed (integer) OR hybrid mode ("h" / "hybrid")
  - force limit
- Run the same setting for 20 trials
- Save one CSV in the same folder as this script
- Open hand and exit
"""

import sys
import time
import csv
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from rh56_controller.rh56_hand import RH56Hand


IDX_MIDDLE = 3

DEFAULT_OPEN_ANGLES = [1000, 1000, 1000, 1000, 650, 0]
DEFAULT_PREP_ANGLES = [1000, 1000, 1000, 700, 650, 0]
DEFAULT_CLOSE_ANGLES = [1000, 1000, 1000, 0, 650, 0]

TRIAL_COUNT = 20

HYBRID_APPROACH_SPEED = 1000
HYBRID_CONTACT_SPEED = 25

FORCE_TRIGGER_RATIO = 0.6

STABLE_WINDOW_S = 0.5
MAX_REFRESH_EPS_G = 2
MIN_PEAK_G_TO_END: Optional[int] = None

LOG_TARGET_HZ = 165.0
LOG_DT = 0.006

RESET_SPEED = 1000
POSE_SETTLE_S = 0.5
PREP_SETTLE_S = 0.2
FORCE_SETTLE_S = 0.2
INTER_TRIAL_GAP_S = 1.0
TRIAL_TIMEOUT_S = 20.0

ANGLE_TOL = 8
ANGLE_WAIT_TIMEOUT_S = 5.0


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
    return float(raw_g)


def build_force_thresholds(force_limit_g: int) -> List[int]:
    thr = [1000] * 6
    thr[IDX_MIDDLE] = int(force_limit_g)
    return thr


def compute_force_trigger(force_limit_g: int) -> int:
    return max(1, min(int(force_limit_g), int(force_limit_g * FORCE_TRIGGER_RATIO)))


def prompt_user_config() -> Tuple[str, int, int]:
    while True:
        raw = input("请输入速度（整数）或 hybrid 模式（输入 h）: ").strip().lower()
        if raw in {"h", "hybrid"}:
            mode = "hybrid"
            speed = HYBRID_CONTACT_SPEED
            break
        try:
            speed = int(raw)
            if speed <= 0:
                raise ValueError
            mode = "speed"
            break
        except ValueError:
            print("无效输入。请输入正整数速度，或输入 h。")

    while True:
        raw = input("请输入 force limit（正整数）: ").strip()
        try:
            force_limit_g = int(raw)
            if force_limit_g <= 0:
                raise ValueError
            break
        except ValueError:
            print("无效输入。force limit 必须是正整数。")

    return mode, speed, force_limit_g


def save_csv(
    records: List[Dict[str, Any]],
    start_epoch: float,
    mode: str,
    speed: int,
    force_limit_g: int,
) -> Path:
    start_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_epoch))
    speed_label = "hybrid" if mode == "hybrid" else f"speed{speed}"
    filename = (
        f"middlefinger_{speed_label}_force{force_limit_g}_"
        f"{TRIAL_COUNT}trials_{start_time_str}.csv"
    )
    filepath = SCRIPT_DIR / filename

    with filepath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Sample_Index",
                "Experiment_Elapsed_s",
                "Trial_Elapsed_s",
                "Timestamp_Epoch",
                "Trial_Index",
                "Trial",
                "Mode",
                "Commanded_Speed",
                "Stage",
                "Middle_Angle",
                "Middle_Force_g",
                "Middle_Force_N",
                "Force_Limit_g",
                "Force_Trigger_g",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r["sample_index"],
                    f"{r['experiment_elapsed_s']:.6f}",
                    f"{r['trial_elapsed_s']:.6f}",
                    f"{r['ts']:.6f}",
                    r["trial_index"],
                    r["trial"],
                    r["mode"],
                    r["cmd_speed"],
                    r["stage"],
                    r["mid_angle"],
                    r["mid_g"],
                    f"{r['mid_N']:.4f}",
                    r["force_limit_g"],
                    r["force_trigger_g"],
                ]
            )

    return filepath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand serial port")
    parser.add_argument("--hand-id", type=int, default=2, help="Hand ID")
    parser.add_argument(
        "--save-partial",
        action="store_true",
        help="Save CSV even if interrupted/failed before finishing all trials",
    )
    args = parser.parse_args()

    mode, speed, force_limit_g = prompt_user_config()
    force_trigger_g = compute_force_trigger(force_limit_g)
    force_thr = build_force_thresholds(force_limit_g)

    if mode == "hybrid":
        print(
            f"配置: mode=hybrid, approach_speed={HYBRID_APPROACH_SPEED}, "
            f"contact_speed={HYBRID_CONTACT_SPEED}, force_limit={force_limit_g}, "
            f"trigger={force_trigger_g}, trials={TRIAL_COUNT}, "
            f"log_gap={LOG_DT}s(~{1.0 / LOG_DT:.1f}Hz, target={LOG_TARGET_HZ:.1f}Hz)"
        )
    else:
        print(
            f"配置: mode=speed, speed={speed}, force_limit={force_limit_g}, "
            f"trigger={force_trigger_g}, trials={TRIAL_COUNT}, "
            f"log_gap={LOG_DT}s(~{1.0 / LOG_DT:.1f}Hz, target={LOG_TARGET_HZ:.1f}Hz)"
        )

    print(f"Connecting RH56 on port={args.port}, hand_id={args.hand_id}...")
    hand = RH56Hand(port=args.port, hand_id=args.hand_id)

    stop_event = threading.Event()
    completed_all_trials = False

    io_lock = threading.Lock()
    state_lock = threading.Lock()
    records_lock = threading.Lock()
    run_state = {
        "trial_index": 0,
        "trial": "init",
        "mode": "init",
        "cmd_speed": -1,
        "stage": "init",
        "recording": False,
        "force_limit_g": force_limit_g,
        "force_trigger_g": force_trigger_g,
        "trial_start_mono": None,
    }

    records: List[Dict[str, Any]] = []
    trial_sample_counts: Dict[int, int] = {}
    start_epoch = time.time()
    start_mono = time.perf_counter()

    def cmd_speed_all(speed_value: int) -> None:
        with io_lock:
            apply_speed_all(hand, speed_value)

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

    def get_trial_sample_count(trial_index: int) -> int:
        with records_lock:
            return trial_sample_counts.get(trial_index, 0)

    def logger_loop() -> None:
        sample_index = 0
        next_tick = time.perf_counter()
        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)

            sample_mono = time.perf_counter()
            ts = time.time()
            angles, forces = read_state()
            if angles is None or forces is None:
                next_tick += LOG_DT
                if sample_mono > (next_tick + LOG_DT):
                    next_tick = sample_mono + LOG_DT
                continue

            mid_angle = int(angles[IDX_MIDDLE])
            mid_g = int(forces[IDX_MIDDLE])

            s = get_run_state_copy()
            if not s.get("recording", True):
                next_tick += LOG_DT
                if sample_mono > (next_tick + LOG_DT):
                    next_tick = sample_mono + LOG_DT
                continue

            trial_start_mono = s.get("trial_start_mono")
            if trial_start_mono is None:
                trial_elapsed_s = 0.0
            else:
                trial_elapsed_s = max(0.0, sample_mono - float(trial_start_mono))

            sample_index += 1
            with records_lock:
                records.append(
                    {
                        "sample_index": sample_index,
                        "experiment_elapsed_s": max(0.0, sample_mono - start_mono),
                        "trial_elapsed_s": trial_elapsed_s,
                        "ts": ts,
                        "trial_index": s["trial_index"],
                        "trial": s["trial"],
                        "mode": s["mode"],
                        "cmd_speed": s["cmd_speed"],
                        "stage": s["stage"],
                        "mid_angle": mid_angle,
                        "mid_g": mid_g,
                        "mid_N": middle_g_to_N(mid_g),
                        "force_limit_g": s["force_limit_g"],
                        "force_trigger_g": s["force_trigger_g"],
                    }
                )
                trial_idx = int(s["trial_index"])
                trial_sample_counts[trial_idx] = trial_sample_counts.get(trial_idx, 0) + 1

            next_tick += LOG_DT
            if sample_mono > (next_tick + LOG_DT):
                next_tick = sample_mono + LOG_DT

    log_thread = threading.Thread(target=logger_loop, daemon=True)
    log_thread.start()
    print("Logging started immediately.")

    def go_pose(label: str, angles: List[int], speed_value: int) -> None:
        set_run_state(stage=label, cmd_speed=int(speed_value), recording=True)
        cmd_speed_all(int(speed_value))
        cmd_angles(angles)
        time.sleep(POSE_SETTLE_S)

        set_run_state(stage=f"{label}_settle", cmd_speed=int(speed_value), recording=False)

    def run_contact_trial(trial_index: int) -> Dict[str, Any]:
        trial_name = f"{mode}_{trial_index:02d}"
        trial_start_epoch = time.time()
        trial_start_mono = time.perf_counter()
        set_run_state(
            trial_index=trial_index,
            trial=trial_name,
            mode=mode,
            stage="inter_trial",
            recording=False,
            force_limit_g=force_limit_g,
            force_trigger_g=force_trigger_g,
            trial_start_mono=trial_start_mono,
        )

        go_pose("open", DEFAULT_OPEN_ANGLES, RESET_SPEED)
        set_run_state(stage="inter_trial_gap", recording=False)
        time.sleep(INTER_TRIAL_GAP_S)

        set_run_state(stage="force_set", recording=False)
        try:
            cmd_force_set(force_thr)
        except Exception as e:
            print(f"Warning: force_set failed: {e}")
        time.sleep(FORCE_SETTLE_S)

        if mode == "speed":
            set_run_state(stage="contact", cmd_speed=int(speed), recording=True)
            cmd_speed_all(int(speed))
            cmd_angles(DEFAULT_CLOSE_ANGLES)
        else:
            set_run_state(
                stage="approach",
                cmd_speed=int(HYBRID_APPROACH_SPEED),
                recording=True,
            )
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

            set_run_state(stage="prep_settle", recording=False)
            time.sleep(PREP_SETTLE_S)

            set_run_state(
                stage="contact",
                cmd_speed=int(HYBRID_CONTACT_SPEED),
                recording=True,
            )
            cmd_speed_all(int(HYBRID_CONTACT_SPEED))
            cmd_angles(DEFAULT_CLOSE_ANGLES)

        t0 = time.time()
        seen_trigger = False
        max_g = -(10**9)
        t_last_new_max: Optional[float] = None
        contact_start_epoch = time.time()

        while (time.time() - t0) < TRIAL_TIMEOUT_S:
            if stop_event.is_set():
                finished_at_epoch = time.time()
                set_run_state(stage="contact_interrupted", recording=False)
                return {
                    "trial_index": trial_index,
                    "trial": trial_name,
                    "mode": mode,
                    "requested_speed": speed,
                    "force_limit_g": force_limit_g,
                    "force_trigger_g": force_trigger_g,
                    "threshold_crossed": seen_trigger,
                    "peak_force_g": "" if max_g == -(10**9) else max_g,
                    "plateau_detected": False,
                    "result": "interrupted",
                    "contact_duration_s": max(0.0, finished_at_epoch - contact_start_epoch),
                    "samples_recorded": get_trial_sample_count(trial_index),
                    "started_at_epoch": trial_start_epoch,
                    "finished_at_epoch": finished_at_epoch,
                }

            _, forces = read_state()
            if forces is None:
                time.sleep(0.01)
                continue

            g = int(forces[IDX_MIDDLE])
            now = time.time()

            if g >= force_trigger_g:
                if not seen_trigger:
                    seen_trigger = True
                    max_g = g
                    t_last_new_max = now
                elif g > (max_g + MAX_REFRESH_EPS_G):
                    max_g = g
                    t_last_new_max = now

            if seen_trigger and (t_last_new_max is not None):
                if (now - t_last_new_max) >= STABLE_WINDOW_S:
                    if (MIN_PEAK_G_TO_END is None) or (max_g >= MIN_PEAK_G_TO_END):
                        finished_at_epoch = time.time()
                        set_run_state(stage="contact_complete", recording=False)
                        return {
                            "trial_index": trial_index,
                            "trial": trial_name,
                            "mode": mode,
                            "requested_speed": speed,
                            "force_limit_g": force_limit_g,
                            "force_trigger_g": force_trigger_g,
                            "threshold_crossed": True,
                            "peak_force_g": max_g,
                            "plateau_detected": True,
                            "result": "success",
                            "contact_duration_s": max(0.0, finished_at_epoch - contact_start_epoch),
                            "samples_recorded": get_trial_sample_count(trial_index),
                            "started_at_epoch": trial_start_epoch,
                            "finished_at_epoch": finished_at_epoch,
                        }

            time.sleep(0.01)

        finished_at_epoch = time.time()
        set_run_state(stage="contact_timeout", recording=False)
        return {
            "trial_index": trial_index,
            "trial": trial_name,
            "mode": mode,
            "requested_speed": speed,
            "force_limit_g": force_limit_g,
            "force_trigger_g": force_trigger_g,
            "threshold_crossed": seen_trigger,
            "peak_force_g": "" if max_g == -(10**9) else max_g,
            "plateau_detected": False,
            "result": "timeout",
            "contact_duration_s": max(0.0, finished_at_epoch - contact_start_epoch),
            "samples_recorded": get_trial_sample_count(trial_index),
            "started_at_epoch": trial_start_epoch,
            "finished_at_epoch": finished_at_epoch,
        }

    try:
        for i in range(1, TRIAL_COUNT + 1):
            print(f"[{i}/{TRIAL_COUNT}] Running trial...")
            summary = run_contact_trial(i)
            if summary["result"] == "success":
                print(f"[DONE] trial {i}/{TRIAL_COUNT}")
            else:
                raise RuntimeError(
                    f"Trial failed: {i}/{TRIAL_COUNT} ({summary['result']})"
                )

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

    if (completed_all_trials or args.save_partial) and len(records) > 0:
        filepath = save_csv(records, start_epoch, mode, speed, force_limit_g)
        if completed_all_trials:
            print(f"Saved: {filepath} ({len(records)} samples)")
        else:
            print(f"Saved PARTIAL: {filepath} ({len(records)} samples)")
    else:
        print(f"No CSV saved. Samples collected: {len(records)}")


if __name__ == "__main__":
    main()

