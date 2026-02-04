#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 Peg-in-Hole - Hand Control + Passive UR5 Sensing
只控制手，不控制机械臂运动，但读取机械臂力传感器数据用于绘图。

功能:
  1. 连接 RH56: 执行 "抓取反射" 逻辑 (接触 -> 抓稳 -> 入孔监测 -> 松开)。
  2. 连接 UR5: 仅开启 F/T 传感器 (Force/Torque) 读取腕部受力，绝对不发送运动指令。
  3. 绘图: 结束时生成双子图 (上图: 机械臂受力, 下图: 手指受力)。

使用方法:
  python3 rh56_peginhole_passive_ur5.py --port /dev/ttyUSB0
"""

import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    # 导入手部驱动
    from rh56_controller.rh56_hand import RH56Hand
    # 导入机械臂驱动 (仅用于读取)
    from magpie_control import ur5
except ImportError as e:
    print(f"错误: 缺少必要的驱动模块 ({e})。")
    print("请确保 magpie_control 和 rh56_controller 在 PYTHONPATH 中。")
    sys.exit(1)

DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
CLOSE_ANGLES = [1000, 1000, 1000, 0, 500, 150] 

# 传感器索引
IDX_FINGER_A = 3    # Index finger
IDX_FINGER_B = 4    # Thumb finger

# 阈值设定
THUMB_SPIKE_THRESH = 50      # 接触触发
INDEX_STABLE_THRESH = 300    # 抓稳阈值
INDEX_STABLE_TIME = 2.0      # 稳定时间
MODE_X_SPIKE = 75            # 入孔阻力
MODE_X_DROP = 25             # 入孔跌落
MODE_X_WINDOW = 0.5          # 窗口时间

def apply_angles(hand, angles, label=""):
    hand.angle_set(list(angles))

def apply_speed(hand, speed, label=""):
    hand.speed_set([speed] * 6)

def run_passive_sensing_task(hand, robot):
    print(f"[{time.strftime('%H:%M:%S')}] START TASK (Hand Control + UR5 Passive Read)...")
    print(">>> Hand is waiting for contact...")
    print(">>> Robot sensor is active (No movement commands will be sent).")
    
    # 数据容器
    times = []
    forces_finger_a = [] # Index
    forces_finger_b = [] # Thumb
    forces_wrist = []    # UR5 Wrist (Magnitude)
    phases = []
    
    # 状态机
    # 0: Wait Contact -> 1: Check Stability -> 2: Monitor Insertion -> 3: Done
    current_phase = 0
    
    start_t = time.monotonic()
    
    # 逻辑变量
    thumb_baseline = None
    stable_start_t = None
    
    # Phase 2 变量
    window_samples = []
    window_start_t = time.monotonic()
    prev_avg = None
    spike_detected = False
    
    # 初始化手部姿态
    apply_speed(hand, 200, "init")
    apply_angles(hand, [1000, 1000, 1000, 1000, 500, 150], "Prepare")
    time.sleep(1.0)
    
    task_running = True
    
    try:
        while task_running:
            loop_start = time.monotonic()
            current_t = loop_start - start_t
            
            # --- A. 读取数据 (Hand + Robot) ---
            
            # 1. RH56 数据
            hand_data = hand.force_act()
            raw_index = hand_data[IDX_FINGER_A]
            raw_thumb = hand_data[IDX_FINGER_B]
            
            # 映射牛顿力
            f_a_newton = (raw_index * 0.007478) - 0.414
            f_b_newton = (raw_thumb * 0.012547) + 0.384
            
            # 2. UR5 数据 (被动读取)
            # get_ft_data 返回 [Fx, Fy, Fz, Tx, Ty, Tz]
            wrist_wrench = robot.get_ft_data()
            if wrist_wrench and len(wrist_wrench) >= 3:
                # 计算合力大小 (Magnitude)
                f_wrist_newton = np.linalg.norm(wrist_wrench[:3])
            else:
                f_wrist_newton = 0.0

            # 记录所有数据
            times.append(current_t)
            forces_finger_a.append(f_a_newton)
            forces_finger_b.append(f_b_newton)
            forces_wrist.append(f_wrist_newton)
            phases.append(current_phase)
            
            
            # [Phase 0] 等待接触
            if current_phase == 0:
                if thumb_baseline is None: thumb_baseline = raw_thumb
                # 基线漂移跟随
                if raw_thumb < thumb_baseline: thumb_baseline = raw_thumb
                else: thumb_baseline += (raw_thumb - thumb_baseline) * 0.05
                
                if (raw_thumb - thumb_baseline) > THUMB_SPIKE_THRESH:
                    print(f">>> [Phase 0->1] Contact! Closing Hand.")
                    apply_speed(hand, 50, "Grip")
                    apply_angles(hand, CLOSE_ANGLES, "Close")
                    current_phase = 1
                    stable_start_t = None

            # [Phase 1] 抓稳检测
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

            # [Phase 2] 入孔检测 (检测 Drop)
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
                                    print(f"    [Phase 2] Resistance Spike (+{delta:.1f}).")
                            else:
                                drop = -delta
                                if drop >= MODE_X_DROP:
                                    print(f">>> [Phase 2->3] DROP detected (-{drop:.1f}). Opening Hand!")
                                    apply_speed(hand, 1000, "Fast Open")
                                    apply_angles(hand, DEFAULT_OPEN, "Open")
                                    task_running = False # 结束
                                    current_phase = 3
                        
                        prev_avg = avg_force
                        window_samples = []
                        window_start_t = time.monotonic()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    print(f"Task Finished. Collected {len(times)} data points.")
    return times, forces_wrist, forces_finger_a, forces_finger_b, phases

def plot_results(times, f_wrist, f_idx, f_thumb, phases):
    plt.figure(figsize=(12, 8))
    
    # --- 上图：UR5 腕部受力 ---
    plt.subplot(2, 1, 1)
    plt.plot(times, f_wrist, label='UR5 Wrist Force (N)', color='black', linewidth=1.2)
    plt.title("UR5 Wrist Force (Passive Monitoring)")
    plt.ylabel("Force Magnitude (N)")
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # --- 下图：手指受力 ---
    plt.subplot(2, 1, 2)
    
    # 阶段背景色
    p_arr = np.array(phases)
    t_arr = np.array(times)
    if len(t_arr) > 0:
        max_f = max(max(f_idx), max(f_thumb)) * 1.1 if f_idx else 10
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==0), color='gray', alpha=0.15, label='Wait Contact')
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==1), color='orange', alpha=0.15, label='Check Stability')
        plt.fill_between(t_arr, 0, max_f, where=(p_arr==2), color='green', alpha=0.15, label='Monitor Insertion')

    plt.plot(times, f_idx, label='Index Finger (N)', color='blue')
    plt.plot(times, f_thumb, label='Thumb Finger (N)', color='red')
    
    # 阈值线
    thresh_val = (300 * 0.007478 - 0.414)
    plt.axhline(y=thresh_val, color='blue', linestyle=':', alpha=0.6, label='Stability Thresh')
    
    plt.title("Hand Finger Forces (Reflex Logic)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = "peginhole_passive_ur5_data.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand Serial port")
    parser.add_argument("--hand-id", type=int, default=1, help="Hand ID")
    parser.add_argument("--robot-ip", default="192.168.0.5", help="UR5 IP Address for F/T sensor")
    args = parser.parse_args()

    # 1. 连接手
    print(f"Connecting to RH56 Hand on {args.port}...")
    try:
        hand = RH56Hand(port=args.port, hand_id=args.hand_id)
    except Exception as e:
        print(f"Hand connection failed: {e}")
        return

    # 2. 连接 Robot (Passive)
    print("Connecting to UR5 Interface (Passive)...")
    try:
        robot = ur5.UR5_Interface()
        robot.start()
        # 仅启动力传感器读取，不发送动作
        print(f"Starting F/T sensor stream on {args.robot_ip}...")
        robot.start_ft_sensor(ip_address=args.robot_ip, poll_rate=100)
    except Exception as e:
        print(f"UR5 connection failed: {e}")
        return

    try:
        # 运行主循环
        t, f_w, f_a, f_b, phases = run_passive_sensing_task(hand, robot)
        
        # 绘图
        plot_results(t, f_w, f_a, f_b, phases)
        
    finally:
        print("Cleaning up...")
        # 手部恢复
        try:
            apply_angles(hand, DEFAULT_OPEN, "Final Open")
        except: pass
        

        try:
            pass 
        except: pass
        print("Done.")

if __name__ == "__main__":
    main()