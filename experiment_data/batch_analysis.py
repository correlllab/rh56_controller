import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 1. 设置 IROS 论文绘图风格
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'mathtext.fontset': 'cm'
})

# ==========================================
# 2. 数据处理函数
# ==========================================

def analyze_pair(hand_file, ur5_file):
    try:
        # 读取数据
        h_df = pd.read_csv(hand_file).sort_values('Timestamp_Epoch')
        u_df = pd.read_csv(ur5_file).sort_values('timestamp_epoch')
        
        # 提取核心指标
        results = {}
        
        # --- A. 计算灵敏度 (MDF) ---
        # 使用 Phase 0 (空载) 的 3倍标准差
        p0_h = h_df[h_df['Phase'] == 0]
        if p0_h.empty: return None
        
        t_end_p0 = p0_h['Timestamp_Epoch'].max()
        p0_u = u_df[u_df['timestamp_epoch'] <= t_end_p0]
        
        h_noise = (p0_h['Index_Force_N'] + p0_h['Thumb_Force_N']).std()
        u_noise = p0_u['wrist_force_N'].std()
        print(f"Debug: {os.path.basename(hand_file)} - Hand Noise: {h_noise:.3f} N, Wrist Noise: {u_noise:.3f} N")

        results['Hand_MDF'] = 3 * h_noise
        results['Wrist_MDF'] = 3 * u_noise
        
        # --- B. 计算稳定性 (CV) ---
        # 使用 Phase 2 (抓取保持) 的 变异系数 (Std/Mean)
        p2_h = h_df[h_df['Phase'] == 2]
        if not p2_h.empty:
            t_start, t_end = p2_h['Timestamp_Epoch'].min(), p2_h['Timestamp_Epoch'].max()
            p2_u = u_df[(u_df['timestamp_epoch'] >= t_start) & (u_df['timestamp_epoch'] <= t_end)]
            
            h_force = p2_h['Index_Force_N'] + p2_h['Thumb_Force_N']
            u_force = p2_u['wrist_force_N']
            
            if h_force.mean() > 0.1:
                results['Hand_CV'] = h_force.std() / h_force.mean()
            if u_force.mean() > 0.1:
                results['Wrist_CV'] = u_force.std() / u_force.mean()
            
        return results
        
    except Exception as e:
        print(f"Skipping {hand_file}: {e}")
        return None

# ==========================================
# 3. 批量执行
# ==========================================

# hand_files = sorted(glob.glob('hand_data_*.csv'))
# ur5_files = sorted(glob.glob('ur5_task_data_*.csv'))
all_hand_files = glob.glob('hand_data_*.csv')
all_ur5_files = glob.glob('ur5_task_data_*.csv')

# 2. Filter out any file containing "_old"
hand_files = sorted([f for f in all_hand_files if '_old' not in f])
ur5_files = sorted([f for f in all_ur5_files if '_old' not in f])

print(f"Found {len(all_hand_files)} total files.")
print(f"Processing {len(hand_files)} clean files (excluded {len(all_hand_files) - len(hand_files)} '_old' files)...")
print(f"Processing {len(hand_files)} experiments...")
all_results = []

for h_f in hand_files:
    try:
        # 简单配对
        h_start = pd.read_csv(h_f, nrows=1)['Timestamp_Epoch'].iloc[0]
        best_u = min(ur5_files, key=lambda u: abs(pd.read_csv(u, nrows=1)['timestamp_epoch'].iloc[0] - h_start))
        res = analyze_pair(h_f, best_u)
        if res: all_results.append(res)
    except: continue

df = pd.DataFrame(all_results)
print(f"Analysis done. Valid trials: {len(df)}")

# ==========================================
# 4. 生成两张核心图表
# ==========================================

def create_boxplot(data_col1, data_col2, title, ylabel, filename):
    if df.empty or data_col1 not in df.columns: return
    
    plt.figure(figsize=(5, 5))
    
    # 准备数据
    data = [df[data_col1].dropna(), df[data_col2].dropna()]
    labels = ['Hand', 'Wrist']
    
    # 绘制箱线图
    bp = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    # 配色：Hand(蓝), Wrist(红)
    colors = ['#aec7e8', '#ff9896']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加提升倍数标注 (Improvement Ratio)
    # 自动计算中位数比值
    med_hand = df[data_col1].median()
    med_wrist = df[data_col2].median()
    
    if med_hand > 0:
        ratio = med_wrist / med_hand
        # 根据指标不同，显示的文字不同
        if "MDF" in data_col1:
            text = f"~{ratio:.1f}x More Sensitive" # 灵敏度
        else:
            text = f"~{ratio:.1f}x More Stable"    # 稳定性
            
        plt.text(1.5, df[data_col2].max()*0.9, text, 
                 ha='center', fontsize=12, fontweight='bold', color='darkred',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

# --- 1. MDF (灵敏度) ---
create_boxplot('Hand_MDF', 'Wrist_MDF', 
               'Fig. 1: Minimum Detectable Force (Sensitivity)', 
               'MDF ($N$) [Lower is Better]', 
               'Fig1_Sensitivity_MDF.png')

# --- 2. CV (稳定性) ---
create_boxplot('Hand_CV', 'Wrist_CV', 
               'Fig. 2: Insert Stability', 
               'Coefficient of Variation (CV)', 
               'Fig2_Stability_CV.png')

print("\nSuccess! Generated 2 figures.")


# ==========================================
# 5. PRINT FINAL STATISTICS FOR PAPER
# ==========================================

print("\n" + "="*40)
print("FINAL RESULTS (For Paper / Manual Record)")
print("="*40)

# 1. Calculate the Raw Sigma (Noise) from the MDF (MDF = 3 * Sigma)
# We divide by 3 because your code calculated MDF as 3 * std_dev
hand_sigmas = df['Hand_MDF'] / 3
wrist_sigmas = df['Wrist_MDF'] / 3

# 2. Print the "Average" (Typical Performance)
print(f"\n[TYPICAL NOISE] (Use these for 'Sensors have a noise floor of X')")
print(f"Hand Average Sigma:  {hand_sigmas.mean():.4f} N")
print(f"Wrist Average Sigma: {wrist_sigmas.mean():.4f} N")

# 3. Print the "Worst Case" (Conservative Thresholds)
print(f"\n[WORST CASE] (Use these for setting safe thresholds)")
print(f"Hand Max Sigma:      {hand_sigmas.max():.4f} N")
print(f"Wrist Max Sigma:     {wrist_sigmas.max():.4f} N")

# 4. Print the Improvement Ratio
improvement = wrist_sigmas.mean() / hand_sigmas.mean()
print(f"\n[CONCLUSION]")
print(f"The Hand is {improvement:.1f}x more sensitive than the Wrist.")
print("="*40)