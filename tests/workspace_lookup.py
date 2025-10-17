#!/usr/bin/env python3
"""
Workspace Lookup Tool - Distance-based lookup for two-finger grasping.
Given a target distance between index and thumb, find the optimal controller positions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


class WorkspaceLookup:
    """
    Distance-based lookup tool for two-finger grasping (index and thumb).
    Finds optimal controller positions based on desired finger separation distance.
    """
    
    def __init__(self, workspace_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the workspace lookup tool.
        
        Args:
            workspace_dir: Directory containing the workspace CSV files. If None,
                defaults to the project's workspace_data folder.
        """
        default_workspace_dir = Path(__file__).resolve().parents[1] / "workspace_data"
        workspace_path = Path(workspace_dir).expanduser() if workspace_dir else default_workspace_dir
        self.workspace_dir = workspace_path.resolve()
        self.tables: Dict[str, pd.DataFrame] = {}
        
        # Load workspace tables
        self._load_tables()
        
        # Pre-compute distance table between index and thumb
        self._build_distance_table()
    
    def _load_tables(self):
        """Load workspace CSV files for index and thumb_bend."""
        if not self.workspace_dir.exists():
            raise FileNotFoundError(
                f"Workspace directory {self.workspace_dir} not found. "
                "Please run generate_workspace_table.py first."
            )
        
        # We only need index and thumb_bend for two-finger grasping
        required_fingers = ['index', 'thumb_bend']
        
        for finger_name in required_fingers:
            csv_file = self.workspace_dir / f"{finger_name}_workspace.csv"
            if not csv_file.exists():
                raise FileNotFoundError(
                    f"Workspace file {csv_file} not found. "
                    "Please run generate_workspace_table.py first."
                )
            
            self.tables[finger_name] = pd.read_csv(csv_file)
            print(f"Loaded workspace table for {finger_name} ({len(self.tables[finger_name])} positions)")
    
    def _build_distance_table(self):
        """
        Pre-compute distances between all index-thumb position combinations.
        This creates a lookup table for fast distance-based queries.
        
        Constraints applied:
        1. Index Y >= Thumb Y (fingers not crossed)
        2. Index theta >= Thumb theta (index bends more for better grasp)
        """
        print("Building distance lookup table...")
        
        index_df = self.tables['index']
        thumb_df = self.tables['thumb_bend']
        
        # Create arrays for fast computation
        index_positions = index_df['position'].values
        thumb_positions = thumb_df['position'].values
        
        index_x = index_df['x_world'].values
        index_y = index_df['y_world'].values
        thumb_x = thumb_df['x_world'].values
        thumb_y = thumb_df['y_world'].values
        
        index_theta = index_df['theta_deg'].values
        thumb_theta = thumb_df['theta_deg'].values
        
        # Build distance table
        distance_data = []
        
        for i, idx_pos in enumerate(index_positions):
            for j, thumb_pos in enumerate(thumb_positions):
                # Calculate Euclidean distance between fingertips
                distance = np.sqrt((index_x[i] - thumb_x[j])**2 + 
                                 (index_y[i] - thumb_y[j])**2)
                
                # Apply constraints for valid grasp configurations:
                # 1. Fingers not crossed: Index Y >= Thumb Y
                # 2. Index bends more than thumb: Index theta >= Thumb theta
                #    This ensures index is always more curved for proper grasping
                if index_y[i] >= thumb_y[j] and index_theta[i] >= thumb_theta[j]:
                    distance_data.append({
                        'index_position': idx_pos,
                        'thumb_position': thumb_pos,
                        'distance': distance,
                        'index_x': index_x[i],
                        'index_y': index_y[i],
                        'thumb_x': thumb_x[j],
                        'thumb_y': thumb_y[j],
                        'index_theta': index_theta[i],
                        'thumb_theta': thumb_theta[j],
                    })
        
        self.distance_table = pd.DataFrame(distance_data)
        print(f"Distance table built: {len(self.distance_table)} valid configurations")
        print(f"Distance range: {self.distance_table['distance'].min()*1000:.2f} - "
              f"{self.distance_table['distance'].max()*1000:.2f} mm")
    
    def get_positions_for_distance(
        self, 
        target_distance: float,
        return_details: bool = False,
        prefer_index_bend: bool = True,
        distance_tolerance_cm: float = 0.5,
        thumb_position_range: Tuple[int, int] = (600, 900)
    ) -> Tuple[int, int] | dict:
        """
        Find the optimal index and thumb positions for a target separation distance.
        
        Strategy: 
        1. Prefer configurations where thumb stays in optimal range (600-900) for better grasp
        2. For small distances, prefer index finger bends more to avoid thumb over-bending
        3. Balance between distance accuracy and natural finger posture
        
        Args:
            target_distance: Target distance between fingertips (in meters)
            return_details: If True, return detailed dictionary; if False, return just positions
            prefer_index_bend: If True, prefer larger index theta for small distances
            distance_tolerance_cm: Tolerance in cm for finding candidates (default: 0.5cm)
            thumb_position_range: Preferred thumb position range (default: 600-900)
            
        Returns:
            If return_details=False: Tuple of (index_position, thumb_position)
            If return_details=True: Dictionary with positions, angles, coords, and actual distance
        """
        # Find all rows within tolerance
        distance_errors = np.abs(self.distance_table['distance'] - target_distance)
        tolerance_m = distance_tolerance_cm / 100.0
        
        # Get candidates within tolerance
        candidates = self.distance_table[distance_errors <= tolerance_m].copy()
        
        # If no candidates within tolerance, expand search
        if len(candidates) == 0:
            # Expand tolerance gradually until we find candidates
            for expanded_tolerance in [1.0, 2.0, 5.0, 10.0]:  # cm
                tolerance_m = expanded_tolerance / 100.0
                candidates = self.distance_table[distance_errors <= tolerance_m].copy()
                if len(candidates) > 0:
                    break
            
            # If still no candidates, use all data
            if len(candidates) == 0:
                candidates = self.distance_table.copy()
        
        # Apply thumb position preference filter
        thumb_min, thumb_max = thumb_position_range
        in_thumb_range = (candidates['thumb_position'] >= thumb_min) & (candidates['thumb_position'] <= thumb_max)
        candidates_in_range = candidates[in_thumb_range].copy()
        
        # If we have candidates in the preferred thumb range, use them
        if len(candidates_in_range) > 0:
            candidates_to_use = candidates_in_range
        else:
            # Otherwise use all candidates but penalize those outside range
            candidates_to_use = candidates
        
        # Smart selection based on distance and finger configuration
        if prefer_index_bend and target_distance <= 0.05:  # For distances <= 5cm
            # For small distances: Multi-factor optimization
            # 1. Prefer higher index theta (more index bend)
            # 2. Reward larger theta difference (index >> thumb)
            # 3. Softly prefer thumb in optimal range (600-900)
            # 4. Consider distance accuracy
            
            # Normalize index theta (0-90° -> 0-1)
            index_score = candidates_to_use['index_theta'] / 90.0
            
            # Theta difference score: reward when index bends much more than thumb
            # Already filtered for index_theta >= thumb_theta in distance table
            theta_diff = candidates_to_use['index_theta'] - candidates_to_use['thumb_theta']
            theta_diff_score = np.clip(theta_diff / 30.0, 0, 1)  # Normalize to 30° difference
            
            # Thumb position score: soft preference for 600-900 range
            # More flexible than before - allows deviation
            thumb_center = (thumb_min + thumb_max) / 2
            thumb_range_size = thumb_max - thumb_min
            thumb_deviation = np.abs(candidates_to_use['thumb_position'] - thumb_center) / (thumb_range_size / 2)
            thumb_score = np.clip(1.0 - 0.5 * thumb_deviation, 0, 1)  # Softer penalty
            
            # Distance accuracy score (closer is better)
            distance_score = 1.0 - np.clip(distance_errors[candidates_to_use.index] / tolerance_m, 0, 1)
            
            # Combined score with weights
            candidates_to_use['score'] = (
                0.35 * index_score +        # 35% - Prefer more index bend
                0.30 * theta_diff_score +   # 30% - Reward large theta difference
                0.15 * thumb_score +        # 15% - Soft thumb position preference
                0.20 * distance_score       # 20% - Distance accuracy
            )
            
            best_idx = candidates_to_use['score'].idxmax()
            row = candidates_to_use.loc[best_idx]
        else:
            # For larger distances: Balance between theta difference, thumb position, and distance
            
            # Theta difference score: still important for large distances
            theta_diff = candidates_to_use['index_theta'] - candidates_to_use['thumb_theta']
            theta_diff_score = np.clip(theta_diff / 20.0, 0, 1)  # Lower threshold for large distances
            
            # Thumb position score: soft preference
            thumb_center = (thumb_min + thumb_max) / 2
            thumb_range_size = thumb_max - thumb_min
            thumb_deviation = np.abs(candidates_to_use['thumb_position'] - thumb_center) / (thumb_range_size / 2)
            thumb_score = np.clip(1.0 - 0.5 * thumb_deviation, 0, 1)
            
            # Distance accuracy score
            distance_score = 1.0 - np.clip(distance_errors[candidates_to_use.index] / tolerance_m, 0, 1)
            
            # Combined score
            candidates_to_use['score'] = (
                0.25 * theta_diff_score +   # 25% - Theta difference still matters
                0.20 * thumb_score +        # 20% - Soft thumb position preference
                0.55 * distance_score       # 55% - Distance accuracy is priority
            )
            
            best_idx = candidates_to_use['score'].idxmax()
            row = candidates_to_use.loc[best_idx]
        
        if not return_details:
            return (int(row['index_position']), int(row['thumb_position']))
        
        result = {
            'index_position': int(row['index_position']),
            'thumb_position': int(row['thumb_position']),
            'index_theta': row['index_theta'],
            'thumb_theta': row['thumb_theta'],
            'index_coords': (row['index_x'], row['index_y']),
            'thumb_coords': (row['thumb_x'], row['thumb_y']),
            'actual_distance': row['distance'],
            'target_distance': target_distance,
            'distance_error': row['distance'] - target_distance,
        }
        
        return result
    
    def get_positions_for_distance_cm(
        self,
        target_distance_cm: float,
        return_details: bool = False,
        prefer_index_bend: bool = True,
        distance_tolerance_cm: float = 0.5,
        thumb_position_range: Tuple[int, int] = (600, 900)
    ) -> Tuple[int, int] | dict:
        """
        Convenience method to find positions using centimeters instead of meters.
        
        Args:
            target_distance_cm: Target distance between fingertips (in centimeters)
            return_details: If True, return detailed dictionary
            prefer_index_bend: If True, prefer larger index theta for small distances
            distance_tolerance_cm: Tolerance in cm for finding candidates (default: 0.5cm)
            thumb_position_range: Preferred thumb position range (default: 600-900)
            
        Returns:
            Same as get_positions_for_distance()
        """
        return self.get_positions_for_distance(
            target_distance_cm / 100.0, 
            return_details,
            prefer_index_bend,
            distance_tolerance_cm,
            thumb_position_range
        )
    
    def get_distance_range(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum achievable distances between fingers.
        
        Returns:
            Tuple of (min_distance_m, max_distance_m)
        """
        return (
            self.distance_table['distance'].min(),
            self.distance_table['distance'].max()
        )
    
    def get_distance_range_cm(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum achievable distances in centimeters.
        
        Returns:
            Tuple of (min_distance_cm, max_distance_cm)
        """
        min_m, max_m = self.get_distance_range()
        return (min_m * 100, max_m * 100)
    
    def is_distance_achievable(self, target_distance: float, tolerance: float = 0.005) -> bool:
        """
        Check if a target distance is achievable within tolerance.
        
        Args:
            target_distance: Target distance in meters
            tolerance: Acceptable error in meters (default: 5mm)
            
        Returns:
            True if distance is achievable within tolerance
        """
        min_error = np.abs(self.distance_table['distance'] - target_distance).min()
        return min_error <= tolerance
    
    def get_all_positions_near_distance(
        self,
        target_distance: float,
        tolerance: float = 0.005
    ) -> pd.DataFrame:
        """
        Get all position combinations that achieve a distance within tolerance.
        
        Args:
            target_distance: Target distance in meters
            tolerance: Acceptable error in meters (default: 5mm)
            
        Returns:
            DataFrame with all matching configurations
        """
        distance_errors = np.abs(self.distance_table['distance'] - target_distance)
        mask = distance_errors <= tolerance
        return self.distance_table[mask].copy()
    
    def get_distance_for_positions(self, index_position: int, thumb_position: int) -> float:
        """
        Calculate the distance between fingers for given controller positions.
        
        Args:
            index_position: Index finger controller position (0-1000)
            thumb_position: Thumb controller position (0-1000)
            
        Returns:
            Distance between fingertips in meters
        """
        # Look up coordinates from the original tables
        index_row = self.tables['index'][self.tables['index']['position'] == index_position]
        thumb_row = self.tables['thumb_bend'][self.tables['thumb_bend']['position'] == thumb_position]
        
        if len(index_row) == 0 or len(thumb_row) == 0:
            raise ValueError("Invalid position values")
        
        index_x = index_row['x_world'].iloc[0]
        index_y = index_row['y_world'].iloc[0]
        thumb_x = thumb_row['x_world'].iloc[0]
        thumb_y = thumb_row['y_world'].iloc[0]
        
        distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
        return distance
    
    def print_distance_info(self):
        """Print information about achievable distances."""
        min_dist, max_dist = self.get_distance_range()
        
        print(f"\n{'='*60}")
        print(f"Two-Finger Grasp Distance Information")
        print(f"{'='*60}")
        print(f"Valid configurations: {len(self.distance_table)}")
        print(f"Distance range: {min_dist*100:.2f} - {max_dist*100:.2f} cm")
        
        # Find configuration for minimum distance
        min_idx = self.distance_table['distance'].idxmin()
        min_row = self.distance_table.loc[min_idx]
        print(f"\nMinimum distance configuration:")
        print(f"  Index position: {int(min_row['index_position'])} (θ={min_row['index_theta']:.2f}°)")
        print(f"  Thumb position: {int(min_row['thumb_position'])} (θ={min_row['thumb_theta']:.2f}°)")
        print(f"  Distance: {min_row['distance']*100:.3f} cm")
        
        # Find configuration for maximum distance
        max_idx = self.distance_table['distance'].idxmax()
        max_row = self.distance_table.loc[max_idx]
        print(f"\nMaximum distance configuration:")
        print(f"  Index position: {int(max_row['index_position'])} (θ={max_row['index_theta']:.2f}°)")
        print(f"  Thumb position: {int(max_row['thumb_position'])} (θ={max_row['thumb_theta']:.2f}°)")
        print(f"  Distance: {max_row['distance']*100:.3f} cm")
        print(f"{'='*60}\n")


def interactive_demo():
    """Interactive real-time demo for distance-based lookup."""
    print("\n" + "="*70)
    print(" RH56 Two-Finger Grasp - Interactive Distance Lookup")
    print("="*70)
    
    # Initialize the lookup tool
    print("\nLoading workspace data...")
    lookup = WorkspaceLookup()
    
    # Print distance information
    lookup.print_distance_info()
    
    # Get distance range
    min_dist_cm, max_dist_cm = lookup.get_distance_range_cm()
    
    print(f"{'='*70}")
    print(" Live distance lookup")
    print(f"{'='*70}")
    print("Enter a target distance in centimeters to get controller positions.")
    print(f"Reachable distance range: {min_dist_cm:.3f} - {max_dist_cm:.2f} cm")
    print("Type 'q' or 'quit' to exit.")
    print(f"{'='*70}\n")
    
    while True:
        try:
            # Get user input
            user_input = input("Enter target distance (cm): ").strip()
            
            # Check for quit command
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nSession ended.")
                break
            
            # Parse distance
            try:
                target_distance_cm = float(user_input)
            except ValueError:
                print("ERROR: Please enter a valid number.\n")
                continue
            
            # Check if distance is in valid range
            if target_distance_cm < min_dist_cm or target_distance_cm > max_dist_cm:
                print(f"WARNING: Distance is outside the reachable range ({min_dist_cm:.3f} - {max_dist_cm:.2f} cm).")
                print("         Searching for the closest reachable configuration...\n")
            
            # Find positions
            result = lookup.get_positions_for_distance_cm(target_distance_cm, return_details=True)
            
            # Display results
            print(f"\n{'-'*70}")
            print(" Query result")
            print(f"{'-'*70}")
            print(f"Target distance:     {target_distance_cm:.2f} cm")
            print(f"Achieved distance:   {result['actual_distance']*100:.3f} cm")
            print(f"Error:               {result['distance_error']*100:.3f} cm")
            print("\nController positions:")
            print(f"  - Index: {result['index_position']:4d}  (θ = {result['index_theta']:5.2f}°)")
            print(f"  - Thumb: {result['thumb_position']:4d}  (θ = {result['thumb_theta']:5.2f}°)")
            print("\nFingertip coordinates (world frame):")
            print(f"  - Index: ({result['index_coords'][0]*100:6.3f}, {result['index_coords'][1]*100:6.3f}) cm")
            print(f"  - Thumb: ({result['thumb_coords'][0]*100:6.3f}, {result['thumb_coords'][1]*100:6.3f}) cm")
            print(f"{'-'*70}\n")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
            break
        except Exception as e:
            print(f"ERROR: {e}\n")
    
    print(f"{'='*70}")
    print("Thank you for using the tool!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    interactive_demo()
