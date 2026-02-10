#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 Peg-in-Hole - Hand Control + Passive UR5 Sensing
功能:
  1. 连接 RH56: 执行 "抓取反射" 逻辑。
  2. 连接 UR5: 被动读取 F/T 传感器。
  3. 数据保存: 自动保存为 CSV。
  4. 绘图: 符合 IROS 格式 (Font 14)。

使用方法:
  python3 rh56_peginhole_passive_ur5.py --port /dev/ttyUSB0
"""

import sys
import time
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

try:
    from rh56_controller.rh56_hand import RH56Hand
    from magpie_control import ur5
except ImportError as e:
    print(f"错误: 缺少必要的驱动模块 ({e})。")
    sys.exit(1)

DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
CLOSE_ANGLES = [1000, 1000, 1000, 0, 500, 150] 

IDX_FINGER_A = 3    # Index finger
IDX_FINGER_B = 4    # Thumb finger

# 阈值设定
THUMB_SPIKE_THRESH = 50      
INDEX_STABLE_THRESH = 300    
INDEX_STABLE_TIME = 2.0      
MODE_X_SPIKE = 75            
MODE_X_DROP = 25             
MODE_X_WINDOW = 0.5          

def apply_angles(hand, angles, label=""):
    hand.angle_set(list(angles))

def apply_speed(hand, speed, label=""):
    hand.speed_set([speed] * 6)

def run_passive_sensing_task(hand, robot):
    print(f"[{time.strftime('%H:%M:%S')}] START TASK (Hand Control + UR5 Passive Read)...")
    
    times = []
    forces_finger_a = [] 
    forces_finger_b = [] 
    forces_wrist = []    
    phases = []
    
    current_phase = 0
    start_t = time.monotonic()
    
    thumb_baseline = None
    stable_start_t = None
    
    window_samples = []
    window_start_t = time.monotonic()
    prev_avg = None
    spike_detected = False
    
    apply_speed(hand, 200, "init")
    apply_angles(hand, [1000, 1000, 1000, 1000, 500, 150], "Prepare")
    time.sleep(1.0)
    
    task_running = True
    
    try:
        while task_running:
            loop_start = time.monotonic()
            current_t = loop_start - start_t
            
            # --- 读取数据 ---
            hand_data = hand.force_act()
            raw_index = hand_data[IDX_FINGER_A]
            raw_thumb = hand_data[IDX_FINGER_B]
            
            f_a_newton = (raw_index * 0.007478) - 0.414
            f_b_newton = (raw_thumb * 0.012547) + 0.384
            
            wrist_wrench = robot.get_ft_data()
            if wrist_wrench and len(wrist_wrench) >= 3:
                f_wrist_newton = np.linalg.norm(wrist_wrench[:3])
            else:
                f_wrist_newton = 0.0

            times.append(current_t)
            forces_finger_a.append(f_a_newton)
            forces_finger_b.append(f_b_newton)
            forces_wrist.append(f_wrist_newton)
            phases.append(current_phase)
            
            # --- 逻辑控制 ---
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
                    if stable_start_t is None: stable_start_t = time.monotonic()
                    if (time.monotonic() - stable_start_t) >= INDEX_STABLE_TIME:
                        print(f">>> [Phase 1->2] Grip Stable. Ready for insertion.")
                        current_phase = 2
                        window_samples = []
                        window_start_t = time.monotonic()
                        prev_avg = None
                        spike_detected = False
                else:
                    stable_start_t = None

            elif current_phase == 2:
                window_samples.append(raw_index)
                if (time.monotonic() - window_start_t) >= MODE_X_WINDOW:
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
                        window_start_t = time.monotonic()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    print(f"Task Finished. Points: {len(times)}")
    return times, forces_wrist, forces_finger_a, forces_finger_b, phases

def save_csv(times, f_wrist, f_idx, f_thumb, phases):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"peginhole_data_{timestamp}.csv"
    print(f"Saving data to {filename}...")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["Time_s", "Wrist_Force_N", "Index_Force_N", "Thumb_Force_N", "Phase"])
        # Rows
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.4f}", 
                f"{f_wrist[i]:.4f}", 
                f"{f_idx[i]:.4f}", 
                f"{f_thumb[i]:.4f}", 
                phases[i]
            ])
    print("CSV Save Complete.")

def plot_results(times, f_wrist, f_idx, f_thumb, phases):
    # --- IROS Style Config ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'serif', # Times New Roman style
        'lines.linewidth': 2
    })
    # -------------------------

    plt.figure(figsize=(10, 8)) # Standardize size
    
    # Subplot 1
    plt.subplot(2, 1, 1)
    plt.plot(times, f_wrist, label='UR5 Wrist Force', color='black')
    plt.title("UR5 Wrist Force")
    plt.ylabel("Force (N)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlim(left=0)
    
    # Subplot 2
    plt.subplot(2, 1, 2)
    p_arr = np.array(phases)
    t_arr = np.array(times)
    
    if len(t_arr) > 0:
        max_f = max(max(f_idx), max(f_thumb)) * 1.1 if f_idx else 10
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==0), color='gray', alpha=0.15, label='Wait')
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==1), color='orange', alpha=0.15, label='Stable')
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==2), color='green', alpha=0.15, label='Insert')

    plt.plot(times, f_idx, label='Index Finger', color='blue')
    plt.plot(times, f_thumb, label='Thumb Finger', color='red')
    
    thresh_val = (300 * 0.007478 - 0.414)
    plt.axhline(y=thresh_val, color='blue', linestyle=':', alpha=0.8, label='Thresh')
    
    plt.title("Hand Finger Forces")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"peginhole_plot_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand Serial port")
    parser.add_argument("--hand-id", type=int, default=1, help="Hand ID")
    parser.add_argument("--robot-ip", default="192.168.0.5", help="UR5 IP")
    args = parser.parse_args()

    print(f"Connecting to RH56 Hand on {args.port}...")
    try:
        hand = RH56Hand(port=args.port, hand_id=args.hand_id)
    except Exception as e:
        print(f"Hand connection failed: {e}")
        return

    print("Connecting to UR5 Interface (Passive)...")
    try:
        robot = ur5.UR5_Interface()
        robot.start()
        robot.start_ft_sensor(ip_address=args.robot_ip, poll_rate=100)
    except Exception as e:
        print(f"UR5 connection failed: {e}")
        return

    try:
        t, f_w, f_a, f_b, phases = run_passive_sensing_task(hand, robot)
        
        if len(t) > 0:
            save_csv(t, f_w, f_a, f_b, phases)
            plot_results(t, f_w, f_a, f_b, phases)
        
    finally:
        print("Cleaning up...")
        try: apply_angles(hand, DEFAULT_OPEN, "Final Open")
        except: pass
        print("Done.")

if __name__ == "__main__":
    main()