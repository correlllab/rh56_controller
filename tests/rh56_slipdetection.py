#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RH56 Slip Detection (Clean Recording Version)
功能:
  1. 待机: 等待用户按下 [ENTER] 键启动任务。
  2. 任务区间 (两次Enter之间):
     - T=0s: 立即闭合手，开始记录数据。
     - T=0~2s: 抓取稳定期。
     - T>2s:  锁定基准位，开始监测滑移 (Diff > 20 -> Re-Grip)。
  3. 结束: 再次按下 [ENTER] 键 -> 停止记录，松开手，生成图表。

使用方法:
  python3 rh56_slip_clean.py --port /dev/ttyUSB0
"""

import sys
import time
import argparse
import select
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 导入驱动
# ==========================================
try:
    from rh56_controller.rh56_hand import RH56Hand
    from magpie_control import ur5
except ImportError as e:
    print(f"错误: 缺少驱动模块 ({e})。请检查 PYTHONPATH。")
    sys.exit(1)

# ==========================================
# 2. 参数定义
# ==========================================
DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
GRIP_ANGLES  = [1000, 1000, 0, 0, 850, 0] 

IDX_MIDDLE = 2
IDX_INDEX = 3

SLIP_POS_DIFF_THRESH = 20.0 # 滑移判定阈值

def apply_angles(hand, angles):
    hand.angle_set(list(angles))

def apply_speed(hand, speed):
    hand.speed_set([speed] * 6)

# ==========================================
# 3. 非阻塞键盘检测
# ==========================================
def is_enter_pressed():
    """检测是否有 Enter 键按下，非阻塞"""
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        sys.stdin.readline() # 消耗按键
        return True
    return False

# ==========================================
# 4. 核心逻辑 (优化版)
# ==========================================
def run_clean_session(hand, robot):
    # --- A. 准备阶段 ---
    print(f"[{time.strftime('%H:%M:%S')}] System Ready.")
    print("Initializing Hand position...")
    apply_speed(hand, 200)
    apply_angles(hand, [1000, 1000, 0, 0, 500, 150]) # 预备姿态
    time.sleep(1.0)
    
    # 清空输入缓冲区
    while is_enter_pressed(): pass

    print("\n================================================")
    print("  WAITING FOR START...")
    print("  Press [ENTER] to START Grip & Recording")
    print("================================================\n")

    # 阻塞等待第一次 Enter
    while True:
        if is_enter_pressed():
            print(">>> START DETECTED! Recording started.")
            break
        time.sleep(0.05)

    # --- B. 记录与执行阶段 (Start T=0) ---
    
    # 立即执行动作：闭合
    apply_speed(hand, 50)
    apply_angles(hand, GRIP_ANGLES)
    
    # 初始化变量
    start_t = time.monotonic()
    times = []
    forces_wrist = []      
    val_index = []         
    val_middle = []        
    events = []
    
    # 记录初始事件
    events.append((0.0, "Start & Grip"))
    
    # 状态标志
    baseline_locked = False
    initial_grip_pos = None
    reclosing_count = 0
    
    task_running = True
    
    print(">>> Recording... Press [ENTER] again to STOP.")

    try:
        while task_running:
            loop_now = time.monotonic()
            current_t = loop_now - start_t
            
            # 1. 检查结束信号 (2nd Enter)
            if is_enter_pressed():
                print(f">>> [{current_t:.2f}s] STOP DETECTED. Stopping...")
                events.append((current_t, "Stop & Open"))
                task_running = False
                break # 跳出循环，去处理松手和画图

            # 2. 读取数据 (Passive)
            hand_data = hand.force_act() 
            v_idx = hand_data[IDX_INDEX]
            v_mid = hand_data[IDX_MIDDLE]
            
            wrist_wrench = robot.get_ft_data()
            f_wrist = np.linalg.norm(wrist_wrench[:3]) if (wrist_wrench and len(wrist_wrench)>=3) else 0.0
            
            # 3. 存储数据
            times.append(current_t)
            forces_wrist.append(f_wrist)
            val_index.append(v_idx)
            val_middle.append(v_mid)
            
            # 4. 逻辑控制 (Stabilize -> Monitor)
            
            # [0.0s - 2.0s] 稳定期 (忽略滑移检测)
            if current_t < 2.0:
                # 只是等待手抓紧，不做任何干预
                pass
                
            # [> 2.0s] 锁定基准 & 监测期
            else:
                # 如果是刚进入 2.0s，锁定一次基准值
                if not baseline_locked:
                    initial_grip_pos = [v_mid, v_idx]
                    baseline_locked = True
                    print(f">>> [{current_t:.2f}s] Stable. Baseline Locked: {initial_grip_pos}")
                    events.append((current_t, "Monitoring Active"))
                
                # 执行滑移检测
                diff_mid = abs(v_mid - initial_grip_pos[0])
                diff_idx = abs(v_idx - initial_grip_pos[1])
                max_diff = max(diff_mid, diff_idx)
                
                # 触发 Re-Grip
                if (max_diff > SLIP_POS_DIFF_THRESH) and (reclosing_count == 0):
                    print(f"!!! [{current_t:.2f}s] SLIP DETECTED (Diff={max_diff:.1f}) -> RE-GRIP!")
                    events.append((current_t, "Auto: Re-Grip"))
                    
                    apply_angles(hand, GRIP_ANGLES) # 再次发送闭合指令
                    reclosing_count = 1 
                
                # 简单的迟滞复位
                if max_diff < (SLIP_POS_DIFF_THRESH / 2):
                    reclosing_count = 0

            time.sleep(0.01) # 100Hz 采样率

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # --- C. 结束清理 ---
    print(">>> Session Ended. Opening Hand...")
    apply_speed(hand, 500)
    apply_angles(hand, DEFAULT_OPEN)
    
    print(f"Total Data Points: {len(times)}")
    return times, forces_wrist, val_index, val_middle, events

# ==========================================
# 5. 绘图函数
# ==========================================
def plot_clean_results(times, f_wrist, val_idx, val_mid, events):
    plt.figure(figsize=(12, 10))
    
    # 子图1: UR5 受力
    plt.subplot(3, 1, 1)
    plt.plot(times, f_wrist, color='black', label='UR5 Wrist Force')
    plt.title("Robot Force (Passive)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.xlim(left=0) # 从 0 开始
    plt.legend()
    
    # 子图2: 手指原始数值
    plt.subplot(3, 1, 2)
    plt.plot(times, val_idx, label=f'Index ({IDX_INDEX})', color='blue')
    plt.plot(times, val_mid, label=f'Middle ({IDX_MIDDLE})', color='orange')
    
    # 绘制事件线
    for t_event, name in events:
        if t_event > times[-1]: continue # 防止画出界
        
        color = 'red' if 'Re-Grip' in name else 'green'
        linestyle = '--' if 'Re-Grip' in name else ':'
        plt.axvline(x=t_event, color=color, linestyle=linestyle)
        # 标签稍微错开一点
        plt.text(t_event, plt.ylim()[0] + (plt.ylim()[1]-plt.ylim()[0])*0.05, 
                 name, rotation=90, color=color, verticalalignment='bottom')

    # 标记检测开始区域
    # 找到 "Monitoring Active" 的时间点
    monitor_start = next((t for t, n in events if "Monitoring" in n), None)
    if monitor_start and len(times) > 0:
        plt.axvspan(monitor_start, times[-1], color='green', alpha=0.05, label='Monitor Zone')

    plt.title("Hand Finger Feedback")
    plt.ylabel("Position/Force Value")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(left=0)

    # 子图3: 模拟 Diff 可视化 (帮助分析阈值是否合理)
    plt.subplot(3, 1, 3)
    if monitor_start:
        # 只画 monitor 之后的数据
        # 找到 monitor_start 对应的索引
        try:
            start_idx = next(i for i, t in enumerate(times) if t >= monitor_start)
            # 获取基准值 (大约是 start_idx 时刻的值)
            base_mid = val_mid[start_idx]
            base_idx = val_idx[start_idx]
            
            # 计算 Diff 数组
            diff_m_arr = [abs(v - base_mid) for v in val_mid[start_idx:]]
            diff_i_arr = [abs(v - base_idx) for v in val_idx[start_idx:]]
            t_slice = times[start_idx:]
            
            plt.plot(t_slice, diff_i_arr, color='blue', alpha=0.6, label='Diff Index')
            plt.plot(t_slice, diff_m_arr, color='orange', alpha=0.6, label='Diff Middle')
            plt.axhline(y=SLIP_POS_DIFF_THRESH, color='red', linestyle='--', label='Threshold (20.0)')
        except:
            pass
            
    plt.title("Calculated Slip Deviation (Diff)")
    plt.xlabel("Time (s)")
    plt.ylabel("Delta Value")
    plt.grid(True)
    plt.xlim(left=0)
    plt.legend()
    
    plt.tight_layout()
    save_path = "slip_clean_record.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# ==========================================
# 6. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Hand Serial port")
    parser.add_argument("--robot-ip", default="192.168.0.5", help="UR5 IP")
    args = parser.parse_args()

    print(f"Connecting to RH56 Hand on {args.port}...")
    try:
        hand = RH56Hand(port=args.port, hand_id=1)
    except Exception:
        print("Hand connect failed.")
        return

    print("Connecting to UR5 Interface (Passive)...")
    try:
        robot = ur5.UR5_Interface()
        robot.start()
        robot.start_ft_sensor(ip_address=args.robot_ip, poll_rate=100)
    except Exception:
        print("UR5 connect failed.")
        return

    try:
        # 运行
        t, f_w, v_i, v_m, evts = run_clean_session(hand, robot)
        # 绘图
        if len(t) > 0:
            plot_clean_results(t, f_w, v_i, v_m, evts)
        else:
            print("No data collected.")
            
    finally:
        print("Cleaning up...")
        try: apply_angles(hand, DEFAULT_OPEN)
        except: pass

if __name__ == "__main__":
    main()