import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Data ---
# Replace with your actual file paths
ur5_file = 'ur5_task_data_1770759843.csv'
hand_file = 'hand_data_20260210-144401.csv'

ur5_df = pd.read_csv(ur5_file)
hand_df = pd.read_csv(hand_file)

# --- 2. Time Synchronization ---
# Find the global start time (earliest timestamp from both files)
t_start = min(ur5_df['timestamp_epoch'].min(), hand_df['Timestamp_Epoch'].min())

# Create relative time columns (seconds since start)
ur5_df['time_rel'] = ur5_df['timestamp_epoch'] - t_start
hand_df['time_rel'] = hand_df['Timestamp_Epoch'] - t_start

# --- 3. IROS Style Configuration ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'font.family': 'serif', # Times New Roman style
    'lines.linewidth': 2
})

# --- 4. Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: UR5 Wrist Force
ax1.plot(ur5_df['time_rel'], ur5_df['wrist_force_N'], label='UR5 Wrist Force', color='black')
ax1.set_title("UR5 Wrist Force")
ax1.set_ylabel("Force (N)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# Subplot 2: Hand Forces
t_hand = hand_df['time_rel']
f_idx = hand_df['Index_Force_N']
f_thumb = hand_df['Thumb_Force_N']
phases = hand_df['Phase']

# Calculate max force for shading height
max_f = max(f_idx.max(), f_thumb.max()) * 1.1 if not f_idx.empty else 10

# Define Phase Colors
phase_map = {
    0: ('gray', 'Wait'),
    1: ('orange', 'Stable'),
    2: ('green', 'Insert'),
    3: ('red', 'Drop/Open')
}

# Add background shading for phases
for phase_id, (color, label) in phase_map.items():
    mask = (phases == phase_id)
    if mask.any():
        ax2.fill_between(t_hand, 0, max_f, where=mask, 
                         color=color, alpha=0.15, label=label)

# Plot Finger Forces
ax2.plot(t_hand, f_idx, label='Index Finger', color='blue')
ax2.plot(t_hand, f_thumb, label='Thumb Finger', color='red')

# Add Threshold Line (from original logic)
thresh_val = (300 * 0.007478 - 0.414)
ax2.axhline(y=thresh_val, color='blue', linestyle=':', alpha=0.8, label='Thresh')

ax2.set_title("Hand Finger Forces")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (N)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(left=0)

# Deduplicate legend entries (fixes multiple shading labels)
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.tight_layout()

# Save and Show
save_path = "combined_plot_iros.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()