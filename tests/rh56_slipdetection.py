#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 Slip Detection (Clean Recording + CSV + IROS Plot)
功能:
  1. 键盘控制流程 (Enter开始 -> Enter结束)。
  2. 数据保存: 自动保存 Timeseries CSV 和 Events CSV。
  3. 绘图: IROS 格式 (Font 14)。

使用方法:
  python3 rh56_slipdetection.py --port /dev/ttyUSB0
"""

import sys
import time
import argparse
import select
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

# --- UR5 + RH56 ---
from magpie_control import ur5
from rh56_controller.rh56_hand import RH56Hand

DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
GRIP_ANGLES  = [1000, 1000, 0, 0, 850, 0] 

IDX_MIDDLE = 2
IDX_INDEX = 3
SLIP_POS_DIFF_THRESH = 20.0 

def apply_angles(hand, angles):
    hand.angle_set(list(angles))

def apply_speed(hand, speed):
    hand.speed_set([speed] * 6)

def is_enter_pressed():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        sys.stdin.readline() 
        return True
    return False

def run_clean_session(hand, robot):
    # --- 准备 ---
    print(f"[{time.strftime('%H:%M:%S')}] System Ready.")
    apply_speed(hand, 200)
    apply_angles(hand, [1000, 1000, 0, 0, 500, 150]) 
    time.sleep(1.0)
    
    while is_enter_pressed(): pass
    print("\n=== Press [ENTER] to START Grip & Recording ===\n")

    while True:
        if is_enter_pressed():
            print(">>> START DETECTED! Recording started.")
            break
        time.sleep(0.05)

    # --- 开始记录 (T=0) ---
    apply_speed(hand, 50)
    apply_angles(hand, GRIP_ANGLES)
    
    start_t = time.monotonic()
    times = []
    forces_wrist = []      
    val_index = []         
    val_middle = []        
    events = []
    
    events.append((0.0, "Start & Grip"))
    
    baseline_locked = False
    initial_grip_pos = None
    reclosing_count = 0
    task_running = True
    
    print(">>> Recording... Press [ENTER] again to STOP.")

    try:
        while task_running:
            loop_now = time.monotonic()
            current_t = loop_now - start_t
            
            # 1. 结束检查
            if is_enter_pressed():
                print(f">>> [{current_t:.2f}s] STOP DETECTED.")
                events.append((current_t, "Stop & Open"))
                task_running = False
                break 

            # 2. 数据读取
            hand_data = hand.force_act() 
            v_idx = hand_data[IDX_INDEX]
            v_mid = hand_data[IDX_MIDDLE]
            
            wrist_wrench = robot.get_ft_data()
            f_wrist = np.linalg.norm(wrist_wrench[:3]) if (wrist_wrench and len(wrist_wrench)>=3) else 0.0
            
            times.append(current_t)
            forces_wrist.append(f_wrist)
            val_index.append(v_idx)
            val_middle.append(v_mid)
            
            # 3. 逻辑控制
            if current_t < 2.0:
                pass
            else:
                if not baseline_locked:
                    initial_grip_pos = [v_mid, v_idx]
                    baseline_locked = True
                    print(f">>> [{current_t:.2f}s] Locked Baseline: {initial_grip_pos}")
                    events.append((current_t, "Monitoring Active"))
                
                diff_mid = abs(v_mid - initial_grip_pos[0])
                diff_idx = abs(v_idx - initial_grip_pos[1])
                max_diff = max(diff_mid, diff_idx)
                
                if (max_diff > SLIP_POS_DIFF_THRESH) and (reclosing_count == 0):
                    print(f"!!! [{current_t:.2f}s] SLIP (Diff={max_diff:.1f}) -> RE-GRIP!")
                    events.append((current_t, "Auto: Re-Grip"))
                    apply_angles(hand, GRIP_ANGLES) 
                    reclosing_count = 1 
                
                if max_diff < (SLIP_POS_DIFF_THRESH / 2):
                    reclosing_count = 0

            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(">>> Session Ended.")
    apply_speed(hand, 500)
    apply_angles(hand, DEFAULT_OPEN)
    
    return times, forces_wrist, val_index, val_middle, events

def save_slip_data(times, f_wrist, val_idx, val_mid, events):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # 1. 保存主要数据
    csv_filename = f"slip_data_{timestamp}.csv"
    print(f"Saving timeseries to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_s", "Wrist_Force_N", "Index_Val", "Middle_Val"])
        for i in range(len(times)):
            writer.writerow([f"{times[i]:.4f}", f"{f_wrist[i]:.4f}", val_idx[i], val_mid[i]])
            
    # 2. 保存事件日志 (方便单独查看)
    event_filename = f"slip_events_{timestamp}.csv"
    print(f"Saving events to {event_filename}...")
    with open(event_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_s", "Event_Description"])
        for evt in events:
            writer.writerow([f"{evt[0]:.4f}", evt[1]])
            
    return timestamp

def plot_clean_results(times, f_wrist, val_idx, val_mid, events, timestamp_str):
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
    # -------------------------

    plt.figure(figsize=(10, 10))
    
    # Subplot 1
    plt.subplot(3, 1, 1)
    plt.plot(times, f_wrist, color='black', label='UR5 Wrist Force')
    plt.title("Robot Force (Passive)")
    plt.ylabel("Force (N)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.legend()
    
    # Subplot 2
    plt.subplot(3, 1, 2)
    plt.plot(times, val_idx, label=f'Index ({IDX_INDEX})', color='blue')
    plt.plot(times, val_mid, label=f'Middle ({IDX_MIDDLE})', color='orange')
    
    # Events
    for t_event, name in events:
        if t_event > times[-1]: continue
        color = 'red' if 'Re-Grip' in name else 'green'
        linestyle = '--' if 'Re-Grip' in name else ':'
        plt.axvline(x=t_event, color=color, linestyle=linestyle)
        plt.text(t_event, plt.ylim()[0] + (plt.ylim()[1]-plt.ylim()[0])*0.02, 
                 name, rotation=90, color=color, verticalalignment='bottom', fontsize=12)

    monitor_start = next((t for t, n in events if "Monitoring" in n), None)
    if monitor_start and len(times) > 0:
        plt.axvspan(monitor_start, times[-1], color='green', alpha=0.05, label='Monitor Zone')

    plt.title("Hand Finger Feedback")
    plt.ylabel("Sensor Value")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)

    # Subplot 3 (Diff)
    plt.subplot(3, 1, 3)
    if monitor_start:
        try:
            start_idx = next(i for i, t in enumerate(times) if t >= monitor_start)
            base_mid = val_mid[start_idx]
            base_idx = val_idx[start_idx]
            
            diff_m_arr = [abs(v - base_mid) for v in val_mid[start_idx:]]
            diff_i_arr = [abs(v - base_idx) for v in val_idx[start_idx:]]
            t_slice = times[start_idx:]
            
            plt.plot(t_slice, diff_i_arr, color='blue', alpha=0.6, label='Diff Index')
            plt.plot(t_slice, diff_m_arr, color='orange', alpha=0.6, label='Diff Middle')
            plt.axhline(y=SLIP_POS_DIFF_THRESH, color='red', linestyle='--', label='Threshold')
        except: pass
            
    plt.title("Calculated Slip Deviation")
    plt.xlabel("Time (s)")
    plt.ylabel("Delta")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.legend()
    
    plt.tight_layout()
    save_path = f"slip_plot_{timestamp_str}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--robot-ip", default="192.168.0.5")
    args = parser.parse_args()

    print(f"Connect Hand: {args.port}...")
    try: hand = RH56Hand(port=args.port, hand_id=1)
    except: return

    print("Connect UR5...")
    try:
        robot = ur5.UR5_Interface()
        robot.start()
        robot.start_ft_sensor(ip_address=args.robot_ip, poll_rate=100)
    except: return

    try:
        t, f_w, v_i, v_m, evts = run_clean_session(hand, robot)
        if len(t) > 0:
            ts = save_slip_data(t, f_w, v_i, v_m, evts)
            plot_clean_results(t, f_w, v_i, v_m, evts, ts)
        else:
            print("No data collected.")
            
    finally:
        try: apply_angles(hand, DEFAULT_OPEN)
        except: pass

if __name__ == "__main__":
    main()