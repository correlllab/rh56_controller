#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 Peg-in-Hole - Hand Control Only (UR5 Removed)
功能:
  1. 连接 RH56: 执行 "抓取反射" 逻辑。
  2. 移除 UR5: 防止与 Jupyter Notebook 冲突。
  3. 数据保存: 保存 Unix Epoch 时间戳 (time.time()) 以便与其他数据对齐。
  4. 绘图: 仅绘制手部数据。

使用方法:
  python3 rh56_peginhole.py --port /dev/ttyUSB0
"""

import sys
import time
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# --- RH56 Only (UR5 Removed) ---
from rh56_controller.rh56_hand import RH56Hand

DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
CLOSE_ANGLES = [1000, 1000, 1000, 0, 600, 150] 

IDX_FINGER_A = 3    # Index finger
IDX_FINGER_B = 4    # Thumb finger

# 阈值设定
THUMB_SPIKE_THRESH = 50      
INDEX_STABLE_THRESH = 300    
INDEX_STABLE_TIME = 2.0      
MODE_X_SPIKE = 50            
MODE_X_DROP = 25             
MODE_X_WINDOW = 0.5          

def apply_angles(hand, angles, label=""):
    hand.angle_set(list(angles))

def apply_speed(hand, speed, label=""):
    hand.speed_set([speed] * 6)

def run_hand_control_task(hand):
    print(f"[{time.strftime('%H:%M:%S')}] START TASK (Hand Control Only)...")
    
    # Data Lists
    timestamps = [] # Will store time.time()
    forces_finger_a = [] 
    forces_finger_b = [] 
    phases = []
    
    current_phase = 0
    
    # Helper variables for logic
    thumb_baseline = None
    stable_start_t = None
    
    window_samples = []
    window_start_t = time.time()
    prev_avg = None
    spike_detected = False
    
    # Initialize Hand
    apply_speed(hand, 500, "init")
    apply_angles(hand, [1000, 1000, 1000, 1000, 600, 150], "Prepare")
    time.sleep(1.0)
    
    task_running = True
    
    try:
        while task_running:
            # --- 1. Capture Time (Unix Epoch) ---
            loop_rate = 0.006
            current_epoch = time.time() 
            
            # --- 2. Read Hand Data ---
            hand_data = hand.force_act()
            raw_index = hand_data[IDX_FINGER_A]
            raw_thumb = hand_data[IDX_FINGER_B]
            
            f_a_newton = (raw_index * 0.007478) - 0.414
            f_b_newton = (raw_thumb * 0.012547) + 0.384
            
            # --- 3. Store Data ---
            timestamps.append(current_epoch)
            forces_finger_a.append(f_a_newton)
            forces_finger_b.append(f_b_newton)
            phases.append(current_phase)
            
            # --- 4. Logic Control ---
            # Note: Logic timers still use monotonic/relative logic for stability, 
            # but we save Epoch time for data sync.
            
            if current_phase == 0:
                if thumb_baseline is None: thumb_baseline = raw_thumb
                if raw_thumb < thumb_baseline: thumb_baseline = raw_thumb
                else: thumb_baseline += (raw_thumb - thumb_baseline) * 0.05
                
                if (raw_thumb - thumb_baseline) > THUMB_SPIKE_THRESH:
                    print(f">>> [Phase 0->1] Contact! Closing Hand.")
                    apply_speed(hand, 50, "Grip")
                    apply_angles(hand, CLOSE_ANGLES, "Close")
                    current_phase = 1
                    stable_start_t = None

            elif current_phase == 1:
                if raw_index > INDEX_STABLE_THRESH:
                    if stable_start_t is None: stable_start_t = time.time()
                    if (time.time() - stable_start_t) >= INDEX_STABLE_TIME:
                        print(f">>> [Phase 1->2] Grip Stable. Ready for insertion.")
                        current_phase = 2
                        window_samples = []
                        window_start_t = time.time()
                        prev_avg = None
                        spike_detected = False
                else:
                    stable_start_t = None

            elif current_phase == 2:
                window_samples.append(raw_thumb)
                if (time.time() - window_start_t) >= MODE_X_WINDOW:
                    if len(window_samples) > 0:
                        avg_force = sum(window_samples) / len(window_samples)
                        if prev_avg is not None:
                            delta = avg_force - prev_avg
                            if not spike_detected:
                                if delta >= MODE_X_SPIKE:
                                    spike_detected = True
                                    print(f"    [Phase 2] Spike (+{delta:.1f}).")
                            else:
                                drop = -delta
                                if drop >= MODE_X_DROP:
                                    print(f">>> [Phase 2->3] DROP (-{drop:.1f}). Opening!")
                                    apply_speed(hand, 1000, "Fast Open")
                                    apply_angles(hand, DEFAULT_OPEN, "Open")
                                    task_running = False 
                                    current_phase = 3
                        prev_avg = avg_force
                        window_samples = []
                        window_start_t = time.time()

            
            elapsed = time.time() - current_epoch
            remaining = loop_rate - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    print(f"Task Finished. Points: {len(timestamps)}")
    return timestamps, forces_finger_a, forces_finger_b, phases

def save_csv(timestamps, f_idx, f_thumb, phases):
    # Use the first timestamp for the filename to keep it unique
    start_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamps[0]))
    filename = f"hand_data_{start_time_str}.csv"
    print(f"Saving data to {filename}...")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header changed to Timestamp_Epoch for synchronization
        writer.writerow(["Timestamp_Epoch", "Index_Force_N", "Thumb_Force_N", "Phase"])
        # Rows
        for i in range(len(timestamps)):
            writer.writerow([
                f"{timestamps[i]:.6f}",  # High precision for epoch time
                f"{f_idx[i]:.4f}", 
                f"{f_thumb[i]:.4f}", 
                phases[i]
            ])
    print("CSV Save Complete.")

def plot_results(timestamps, f_idx, f_thumb, phases):
    # --- IROS Style Config ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'serif',
        'lines.linewidth': 2
    })
    
    # Calculate Relative Time for Plotting (Easier to read than Epoch)
    if len(timestamps) > 0:
        t_start = timestamps[0]
        rel_times = [t - t_start for t in timestamps]
    else:
        rel_times = []

    plt.figure(figsize=(10, 6))
    
    p_arr = np.array(phases)
    t_arr = np.array(rel_times)
    
    if len(t_arr) > 0:
        max_f = max(max(f_idx), max(f_thumb)) * 1.1 if f_idx else 10
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==0), color='gray', alpha=0.15, label='Wait')
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==1), color='orange', alpha=0.15, label='Stable')
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==2), color='green', alpha=0.15, label='Insert')

    plt.plot(rel_times, f_idx, label='Index Finger', color='blue')
    plt.plot(rel_times, f_thumb, label='Thumb Finger', color='red')
    
    thresh_val = (300 * 0.007478 - 0.414)
    plt.axhline(y=thresh_val, color='blue', linestyle=':', alpha=0.8, label='Thresh')
    
    plt.title("Hand Finger Forces (Synced)")
    plt.xlabel("Time (s) - relative to start")
    plt.ylabel("Force (N)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    
    plt.tight_layout()
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"hand_plot_{timestamp_str}.png"
    # plt.savefig(save_path, dpi=300)
    # print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand Serial port")
    parser.add_argument("--hand-id", type=int, default=1, help="Hand ID")
    args = parser.parse_args()

    print(f"Connecting to RH56 Hand on {args.port}...")
    try:
        hand = RH56Hand(port=args.port, hand_id=args.hand_id)
    except Exception as e:
        print(f"Hand connection failed: {e}")
        return

    # No UR5 connection here anymore!

    try:
        t, f_a, f_b, phases = run_hand_control_task(hand)
        
        if len(t) > 0:
            save_csv(t, f_a, f_b, phases)
            plot_results(t, f_a, f_b, phases)
        
    finally:
        print("Cleaning up...")
        try: 
            for _ in range(3):
                apply_angles(hand, DEFAULT_OPEN, "Final Open")
                time.sleep(0.5)
        except: pass
        print("Done.")

if __name__ == "__main__":
    main()