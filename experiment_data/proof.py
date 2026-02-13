import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FOLDER = '.' 
OUTPUT_FOLDER = 'final_fixed_trim'
HAND_FILE_PATTERN = 'hand_data*.csv'
UR5_FILE_PATTERN = 'ur5_task_data*.csv'
INTERP_POINTS = 500

# --- USER TUNING VARIABLE ---
TRIM_SECONDS_FROM_END = 1.8  # Change this value to cut more or less from the end
# ----------------------------

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_01(series):
    """Normalizes a series to the range [0, 1]."""
    if len(series) == 0: return series
    rng = series.max() - series.min()
    if rng == 0: return series - series.min()
    return (series - series.min()) / rng

def calculate_lag_normalized(t1, y1, t2, y2):
    """Calculates time shift to align y2 to y1."""
    # Normalize for robust correlation
    y1_n = normalize_01(y1)
    y2_n = normalize_01(y2)
    
    dt = 0.01
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    
    if t_end - t_start < 0.5: return 0.0
    
    t_grid = np.arange(t_start, t_end, dt)
    
    f1 = interp1d(t1, y1_n, kind='linear', fill_value="extrapolate")
    f2 = interp1d(t2, y2_n, kind='linear', fill_value="extrapolate")
    
    y1_g = f1(t_grid)
    y2_g = f2(t_grid)
    
    y1_d = signal.detrend(y1_g)
    y2_d = signal.detrend(y2_g)
    
    corr = signal.correlate(y1_d, y2_d, mode='full')
    lags = signal.correlation_lags(len(y1_d), len(y2_d), mode='full')
    return lags[np.argmax(corr)] * dt

# ==========================================
# MAIN PROCESSING
# ==========================================

hand_files = sorted(glob.glob(os.path.join(DATA_FOLDER, HAND_FILE_PATTERN)))
ur5_files = sorted(glob.glob(os.path.join(DATA_FOLDER, UR5_FILE_PATTERN)))
pairs = list(zip(hand_files, ur5_files))

all_thumb = []
all_wrist = []
all_lags = []

print(f"Processing {len(pairs)} pairs. Trimming last {TRIM_SECONDS_FROM_END}s...")

for i, (hand_f, ur5_f) in enumerate(pairs):
    try:
        df_hand = pd.read_csv(hand_f)
        df_ur5 = pd.read_csv(ur5_f)
        
        # 1. Base Timeline (Wrist)
        wrist_start = df_ur5['timestamp_epoch'].min()
        wrist_end = df_ur5['timestamp_epoch'].max()
        
        # 2. Apply Fixed Trim
        cutoff_time = wrist_end - TRIM_SECONDS_FROM_END
        
        if cutoff_time <= wrist_start:
            print(f"  -> Skipping Pair {i+1}: Recording too short for trim.")
            continue
            
        # 3. Slice Data (Trimmed Window)
        hand_slice = df_hand[(df_hand['Timestamp_Epoch'] >= wrist_start) & 
                             (df_hand['Timestamp_Epoch'] <= cutoff_time)].copy()
        wrist_slice = df_ur5[(df_ur5['timestamp_epoch'] >= wrist_start) & 
                             (df_ur5['timestamp_epoch'] <= cutoff_time)].copy()
        
        if hand_slice.empty or wrist_slice.empty: continue
            
        # 4. Calculate Lag (on clean data)
        lag = calculate_lag_normalized(
            hand_slice['Timestamp_Epoch'].values, hand_slice['Thumb_Force_N'].values,
            wrist_slice['timestamp_epoch'].values, wrist_slice['wrist_force_N'].values
        )
        all_lags.append(lag)
        
        # 5. Normalize & Align
        duration = cutoff_time - wrist_start
        
        # Force Normalization (0-1)
        h_force = normalize_01(hand_slice['Thumb_Force_N'].values)
        w_force = normalize_01(wrist_slice['wrist_force_N'].values)
        
        # Time Normalization (0-100%)
        h_time_pct = ((hand_slice['Timestamp_Epoch'].values - wrist_start) / duration) * 100
        w_time_shifted = wrist_slice['timestamp_epoch'].values + lag
        w_time_pct = ((w_time_shifted - wrist_start) / duration) * 100
        
        # Interpolate
        x_grid = np.linspace(0, 100, INTERP_POINTS)
        f_h = interp1d(h_time_pct, h_force, kind='linear', bounds_error=False, fill_value=np.nan)(x_grid)
        f_w = interp1d(w_time_pct, w_force, kind='linear', bounds_error=False, fill_value=np.nan)(x_grid)
        
        all_thumb.append(f_h)
        all_wrist.append(f_w)
        
    except Exception as e:
        print(f"Error pair {i}: {e}")

# ==========================================
# PLOTTING
# ==========================================
if all_thumb:
    h_mean = np.nanmean(all_thumb, axis=0)
    w_mean = np.nanmean(all_wrist, axis=0)
    h_std = np.nanstd(all_thumb, axis=0)
    w_std = np.nanstd(all_wrist, axis=0)
    
    avg_lag = np.mean(all_lags)
    
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 100, INTERP_POINTS)
    
    plt.plot(x, h_mean, 'b-', label='Hand Thumb (Mean)', linewidth=2)
    plt.fill_between(x, h_mean - h_std, h_mean + h_std, color='blue', alpha=0.15)
    
    plt.plot(x, w_mean, 'orange', label=f'Wrist Force (Shifted {avg_lag:.2f}s)', linewidth=2)
    plt.fill_between(x, w_mean - w_std, w_mean + w_std, color='orange', alpha=0.15)
    
    plt.title(f"Aligned Force Profile (Trimmed Last {TRIM_SECONDS_FROM_END}s)")
    plt.xlabel("Task Progress (%)")
    plt.ylabel("Normalized Force")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"{OUTPUT_FOLDER}/average_fixed_trim.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Analysis saved to {save_path}")
else:
    print("No valid data found.")