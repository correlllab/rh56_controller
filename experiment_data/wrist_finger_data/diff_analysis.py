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
DATA_FOLDER = '.'  # Set to your data folder
OUTPUT_FOLDER = 'aligned_phase2_analysis'
HAND_FILE_PATTERN = 'hand_data*.csv'
UR5_FILE_PATTERN = 'ur5_task_data*.csv'

# Analysis Parameters
SMOOTHING_WINDOW = 20
INTERP_POINTS = 100  # Number of points to normalize Phase 2 to (0-100% of Phase 2)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_time_range(filepath, time_col):
    try:
        df = pd.read_csv(filepath, usecols=[time_col])
        return df[time_col].min(), df[time_col].max()
    except:
        return None, None

def min_max_scale(series):
    rng = series.max() - series.min()
    if rng == 0: return series - series.min()
    return (series - series.min()) / rng

def calculate_optimal_lag(t1, y1, t2, y2):
    """Calculates time shift to align y2 to y1."""
    # Common grid
    dt = 0.01
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    if t_end - t_start < 1.0: return 0.0 # Too short overlap
    
    t_grid = np.arange(t_start, t_end, dt)
    
    # Interpolate
    f1 = interp1d(t1, y1, kind='linear', fill_value="extrapolate")
    f2 = interp1d(t2, y2, kind='linear', fill_value="extrapolate")
    
    y1_new = f1(t_grid)
    y2_new = f2(t_grid)
    
    # Detrend
    y1_c = y1_new - np.mean(y1_new)
    y2_c = y2_new - np.mean(y2_new)
    
    # Cross-corr
    corr = signal.correlate(y1_c, y2_c, mode='full')
    lags = signal.correlation_lags(len(y1_c), len(y2_c), mode='full')
    lag_idx = lags[np.argmax(corr)]
    
    return lag_idx * dt

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Match Files
hand_files = glob.glob(os.path.join(DATA_FOLDER, HAND_FILE_PATTERN))
ur5_files = glob.glob(os.path.join(DATA_FOLDER, UR5_FILE_PATTERN))

pairs = []
used_ur5 = set()
for h_file in hand_files:
    h_start, h_end = get_time_range(h_file, 'Timestamp_Epoch')
    if not h_start: continue
    
    best_match = None
    max_overlap = 0
    for u_file in ur5_files:
        if u_file in used_ur5: continue
        u_start, u_end = get_time_range(u_file, 'timestamp_epoch')
        if not u_start: continue
        
        overlap = min(h_end, u_end) - max(h_start, u_start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = u_file
            
    if best_match and max_overlap > 1.0:
        pairs.append((h_file, best_match))
        used_ur5.add(best_match)

print(f"Found {len(pairs)} pairs.")

# 2. Process Each Pair
all_thumb_p2 = []
all_wrist_p2 = []
all_lags = []

for i, (h_path, u_path) in enumerate(pairs):
    try:
        # Load
        hand_df = pd.read_csv(h_path).sort_values('Timestamp_Epoch')
        ur5_df = pd.read_csv(u_path).sort_values('timestamp_epoch')
        
        # Calculate Lag
        lag = calculate_optimal_lag(hand_df['Timestamp_Epoch'], hand_df['Thumb_Force_N'],
                                    ur5_df['timestamp_epoch'], ur5_df['wrist_force_N'])
        all_lags.append(lag)
        
        # Apply Lag to UR5
        ur5_df['timestamp_epoch'] += lag
        
        # Sync (Interpolate UR5 to Hand timestamps)
        f_ur5 = interp1d(ur5_df['timestamp_epoch'], ur5_df['wrist_force_N'], 
                         kind='linear', fill_value="extrapolate", bounds_error=False)
        
        # Filter to overlapping range
        start = max(hand_df['Timestamp_Epoch'].min(), ur5_df['timestamp_epoch'].min())
        end = min(hand_df['Timestamp_Epoch'].max(), ur5_df['timestamp_epoch'].max())
        mask = (hand_df['Timestamp_Epoch'] >= start) & (hand_df['Timestamp_Epoch'] <= end)
        df = hand_df[mask].copy()
        df['wrist_aligned'] = f_ur5(df['Timestamp_Epoch'])
        
        # Extract Phase 2 ONLY
        phase2 = df[df['Phase'] == 2].copy()
        
        if len(phase2) < 50:
            print(f"Skipping pair {i+1}: Not enough Phase 2 data.")
            continue
            
        # Smooth
        phase2['thumb_smooth'] = phase2['Thumb_Force_N'].rolling(window=SMOOTHING_WINDOW, center=True).mean()
        phase2['wrist_smooth'] = phase2['wrist_aligned'].rolling(window=SMOOTHING_WINDOW, center=True).mean()
        phase2 = phase2.dropna()
        
        # Normalize Force (0-1)
        thumb_n = min_max_scale(phase2['thumb_smooth'])
        wrist_n = min_max_scale(phase2['wrist_smooth'])
        
        # Time Normalize Phase 2 (0-100%)
        t_current = np.linspace(0, 1, len(phase2))
        t_target = np.linspace(0, 1, INTERP_POINTS)
        
        f_thumb = interp1d(t_current, thumb_n, kind='linear')
        f_wrist = interp1d(t_current, wrist_n, kind='linear')
        
        all_thumb_p2.append(f_thumb(t_target))
        all_wrist_p2.append(f_wrist(t_target))
        
    except Exception as e:
        print(f"Error in pair {i+1}: {e}")

# 3. Aggregate Phase 2
if all_thumb_p2:
    print(f"\nAverage Lag Detected: {np.mean(all_lags):.3f}s")
    
    thumb_mat = np.array(all_thumb_p2)
    wrist_mat = np.array(all_wrist_p2)
    
    thumb_mean = np.nanmean(thumb_mat, axis=0)
    thumb_std = np.nanstd(thumb_mat, axis=0)
    wrist_mean = np.nanmean(wrist_mat, axis=0)
    wrist_std = np.nanstd(wrist_mat, axis=0)
    
    x_axis = np.linspace(0, 100, INTERP_POINTS)
    
    plt.figure(figsize=(10, 6))
    
    # Wrist
    plt.plot(x_axis, wrist_mean, 'orange', label='Wrist Force (Mean)', linewidth=2)
    plt.fill_between(x_axis, wrist_mean - wrist_std, wrist_mean + wrist_std, color='orange', alpha=0.2)
    
    # Thumb
    plt.plot(x_axis, thumb_mean, 'blue', label='Thumb Force (Mean)', linewidth=2, linestyle='--')
    plt.fill_between(x_axis, thumb_mean - thumb_std, thumb_mean + thumb_std, color='blue', alpha=0.2)
    
    plt.title(f'Phase 2 Interaction Agreement (n={len(all_thumb_p2)})\n')
    plt.xlabel('Interaction Progress (%)')
    plt.ylabel('Normalized Force (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f'{OUTPUT_FOLDER}/phase2_grand_average.png'
    plt.savefig(save_path)
    print(f"Saved aligned Phase 2 plot to: {save_path}")
    
    print(f"Correlation of Mean Trends: {np.corrcoef(thumb_mean, wrist_mean)[0,1]:.3f}")
else:
    print("No valid Phase 2 data found.")