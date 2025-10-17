#!/usr/bin/env python3
"""
Generate forward kinematics workspace tables for RH56 hand fingers.
This script creates lookup tables mapping controller positions (0-1000) to 
end-effector positions in both local and world coordinates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

# Add parent directory to path to import kinematics module
# Ensure package root is on sys.path so rh56_controller package resolves reliably
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from rh56_controller.kinematics import HandKinematics

DEFAULT_WORKSPACE_DIR = PACKAGE_ROOT / "workspace_data"

class WorkspaceGenerator:
    def __init__(
        self,
        thumb_splay_deg: float = 58.0,
        thumb_base_offset: Optional[Tuple[float, float]] = None,
    ):
        """Initialize the workspace generator with hand kinematics."""
        self.hand_kinematics = HandKinematics(
            thumb_splay_deg=thumb_splay_deg,
            thumb_base_offset=thumb_base_offset,
        )
        self.thumb_splay_deg = thumb_splay_deg
        self.thumb_base_offset = tuple(self.hand_kinematics.thumb_base_offset)
        
        # Controller position to angle mapping
        self.min_position = 0
        self.max_position = 1000
        
        # Kinematic angle (theta) ranges for different finger types
        # Position 0 (closed) -> theta_max, Position 1000 (open) -> theta_min (0°)
        # Four fingers: theta 0° - 90°
        # Thumb: theta 0° - 45° (limited range, perpendicular to index when open)
        self.theta_ranges = {
            'pinky': (0.0, 90.0),
            'ring': (0.0, 90.0),
            'middle': (0.0, 90.0),
            'index': (0.0, 90.0),
            'thumb_bend': (0.0, 45.0),      # Thumb has limited range
            'thumb_rotation': (0.0, 90.0),  # Rotation uses full range
        }
    
    def position_to_theta(self, position: int, finger_name: str) -> float:
        """
        Convert controller position (0-1000) to kinematic angle theta.
        Position 0 (closed) -> theta_max
        Position 1000 (open) -> theta_min (0°)
        
        Args:
            position: Controller position value (0-1000)
            finger_name: Name of the finger to get correct theta range
            
        Returns:
            Theta angle in degrees (range depends on finger type)
        """
        if not (self.min_position <= position <= self.max_position):
            raise ValueError(f"Position must be between {self.min_position} and {self.max_position}")
        
        if finger_name not in self.theta_ranges:
            raise ValueError(f"Unknown finger: {finger_name}")
        
        theta_min, theta_max = self.theta_ranges[finger_name]
        
        # Inverted linear interpolation: position 0 -> theta_max, position 1000 -> theta_min (0°)
        theta = theta_max - (position / self.max_position) * (theta_max - theta_min)
        return theta
    
    def theta_to_position(self, theta: float, finger_name: str) -> int:
        """
        Convert kinematic angle theta to controller position (0-1000).
        Theta theta_max (closed) -> position 0
        Theta 0° (open) -> position 1000
        
        Args:
            theta: Kinematic angle in degrees
            finger_name: Name of the finger to get correct theta range
            
        Returns:
            Controller position (0-1000)
        """
        if finger_name not in self.theta_ranges:
            raise ValueError(f"Unknown finger: {finger_name}")
        
        theta_min, theta_max = self.theta_ranges[finger_name]
        
        if not (theta_min <= theta <= theta_max):
            raise ValueError(f"Theta must be between {theta_min} and {theta_max} degrees for {finger_name}")
        
        # Inverted linear interpolation: theta_max -> 0, theta 0° -> 1000
        position = ((theta_max - theta) / (theta_max - theta_min)) * self.max_position
        return int(round(position))
    
    def generate_finger_workspace(self, finger_name: str, step: int = 1) -> pd.DataFrame:
        """
        Generate workspace table for a single finger.
        
        Args:
            finger_name: Name of the finger (e.g., "index", "thumb_bend")
            step: Step size for controller position (default: 1)
            
        Returns:
            DataFrame with columns: position, angle, x_local, y_local, x_world, y_world
        """
        if finger_name not in self.hand_kinematics.fingers:
            raise ValueError(f"Unknown finger: {finger_name}. Available: {list(self.hand_kinematics.fingers.keys())}")
        
        finger_model = self.hand_kinematics.fingers[finger_name]
        positions = range(self.min_position, self.max_position + 1, step)
        
        data = []
        for pos in positions:
            # Convert position to theta angle
            theta = self.position_to_theta(pos, finger_name)
            
            # Get forward kinematics using theta
            joint_world, end_world = finger_model.forward_kinematics(theta)
            
            # Calculate local coordinates (relative to finger base)
            end_local = finger_model.world_to_local(end_world)
            
            data.append({
                'position': pos,
                'theta_deg': theta,
                'x_local': end_local[0],
                'y_local': end_local[1],
                'x_world': end_world[0],
                'y_world': end_world[1],
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_all_fingers_workspace(self, finger_names: list = None, step: int = 1) -> dict:
        """
        Generate workspace tables for multiple fingers.
        
        Args:
            finger_names: List of finger names. If None, generates for all fingers.
            step: Step size for controller position (default: 1)
            
        Returns:
            Dictionary mapping finger names to their workspace DataFrames
        """
        if finger_names is None:
            finger_names = list(self.hand_kinematics.fingers.keys())
        
        workspace_tables = {}
        for finger_name in finger_names:
            print(f"Generating workspace for {finger_name}...")
            workspace_tables[finger_name] = self.generate_finger_workspace(finger_name, step)
        
        return workspace_tables
    
    def save_tables_to_csv(
        self,
        workspace_tables: dict,
        output_dir: Union[str, Path, None] = None,
    ):
        """
        Save workspace tables to CSV files.
        
        Args:
            workspace_tables: Dictionary of finger names to DataFrames
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir) if output_dir is not None else DEFAULT_WORKSPACE_DIR
        output_path.mkdir(parents=True, exist_ok=True)
        
        for finger_name, df in workspace_tables.items():
            filepath = output_path / f"{finger_name}_workspace.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved {finger_name} workspace to {filepath}")
    
    def plot_workspace(
        self,
        workspace_tables: dict,
        save_path: Optional[Union[str, Path]] = None,
    ):
        """
        Plot the workspace for all fingers.
        
        Args:
            workspace_tables: Dictionary of finger names to DataFrames
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot in world coordinates (convert to cm)
        for finger_name, df in workspace_tables.items():
            # Convert meters to centimeters
            x_world_cm = df['x_world'] * 100
            y_world_cm = df['y_world'] * 100
            
            # Plot trajectory
            line, = ax1.plot(x_world_cm, y_world_cm, '-', label=f"{finger_name}", linewidth=2, alpha=0.7)
            color = line.get_color()
            
            # Mark start point (position 0)
            ax1.plot(x_world_cm.iloc[0], y_world_cm.iloc[0], 'o', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            
            # Mark end point (position 1000)
            ax1.plot(x_world_cm.iloc[-1], y_world_cm.iloc[-1], 's', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        
        # Add separate legend entries for start/end markers
        ax1.plot([], [], 'o', color='gray', markersize=8, markeredgecolor='black', 
                markeredgewidth=1.5, label='○ Pos=0 (Closed, θ=max)')
        ax1.plot([], [], 's', color='gray', markersize=8, markeredgecolor='black', 
                markeredgewidth=1.5, label='□ Pos=1000 (Open, θ=0°)')
        
        ax1.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
        ax1.set_title('Finger Workspace - World Coordinates', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='best', ncol=2)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axis('equal')
        
        # Plot in local coordinates (convert to cm)
        for finger_name, df in workspace_tables.items():
            # Convert meters to centimeters
            x_local_cm = df['x_local'] * 100
            y_local_cm = df['y_local'] * 100
            
            # Plot trajectory
            line, = ax2.plot(x_local_cm, y_local_cm, '-', label=f"{finger_name}", linewidth=2, alpha=0.7)
            color = line.get_color()
            
            # Mark start point (position 0)
            ax2.plot(x_local_cm.iloc[0], y_local_cm.iloc[0], 'o', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            
            # Mark end point (position 1000)
            ax2.plot(x_local_cm.iloc[-1], y_local_cm.iloc[-1], 's', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        
        # Add separate legend entries for start/end markers
        ax2.plot([], [], 'o', color='gray', markersize=8, markeredgecolor='black', 
                markeredgewidth=1.5, label='○ Pos=0 (Closed, θ=max)')
        ax2.plot([], [], 's', color='gray', markersize=8, markeredgecolor='black', 
                markeredgewidth=1.5, label='□ Pos=1000 (Open, θ=0°)')
        
        ax2.set_xlabel('X Local (cm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Y Local (cm)', fontsize=12, fontweight='bold')
        ax2.set_title('Finger Workspace - Local Coordinates', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, loc='best', ncol=2)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved workspace plot to {save_path}")
        
        plt.show()
    
    def find_nearest_position(self, df: pd.DataFrame, target_x: float, target_y: float, 
                             coordinate_type: str = 'world') -> dict:
        """
        Find the controller position that gets closest to a target (x, y) point.
        
        Args:
            df: Workspace DataFrame for a finger
            target_x: Target x coordinate
            target_y: Target y coordinate
            coordinate_type: 'world' or 'local' coordinates
            
        Returns:
            Dictionary with position, angle, actual coordinates, and distance
        """
        if coordinate_type == 'world':
            x_col, y_col = 'x_world', 'y_world'
        elif coordinate_type == 'local':
            x_col, y_col = 'x_local', 'y_local'
        else:
            raise ValueError("coordinate_type must be 'world' or 'local'")
        
        # Calculate distances to target
        distances = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
        min_idx = distances.idxmin()
        
        result = {
            'position': int(df.loc[min_idx, 'position']),
            'theta_deg': df.loc[min_idx, 'theta_deg'],
            'x': df.loc[min_idx, x_col],
            'y': df.loc[min_idx, y_col],
            'distance': distances[min_idx],
        }
        
        return result


def main():
    """Main function to generate workspace tables."""
    generator = WorkspaceGenerator()
    
    # Generate workspace for thumb and index (as requested)
    print("=" * 60)
    print("Generating Finger Workspace Tables")
    print("=" * 60)
    print(f"Thumb base splay angle: {generator.thumb_splay_deg:.2f}°")
    print(f"Thumb base offset: {generator.thumb_base_offset}\n")
    
    # Primary fingers: thumb_bend and index
    primary_fingers = ["thumb_bend", "index"]
    workspace_tables = generator.generate_all_fingers_workspace(primary_fingers, step=1)
    
    # Save to CSV
    primary_workspace_dir = DEFAULT_WORKSPACE_DIR
    generator.save_tables_to_csv(workspace_tables, output_dir=primary_workspace_dir)
    
    # Generate plot
    generator.plot_workspace(workspace_tables, save_path=primary_workspace_dir / "workspace_plot_primary.png")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for finger_name, df in workspace_tables.items():
        print(f"\n{finger_name.upper()}:")
        print(f"  Number of positions: {len(df)}")
        print(f"  Theta range: {df['theta_deg'].min():.2f}° - {df['theta_deg'].max():.2f}°")
        print(f"  X range (world): {df['x_world'].min():.4f} - {df['x_world'].max():.4f} m")
        print(f"  Y range (world): {df['y_world'].min():.4f} - {df['y_world'].max():.4f} m")
        print(f"  Max reach (world): {np.sqrt(df['x_world']**2 + df['y_world']**2).max():.4f} m")
    
    # Example: Find position for a target point
    print("\n" + "=" * 60)
    print("Example: Inverse Lookup")
    print("=" * 60)
    
    # Example target for index finger
    target_x, target_y = -0.05, 0.05  # meters, world coordinates
    result = generator.find_nearest_position(
        workspace_tables["index"], 
        target_x, 
        target_y, 
        coordinate_type='world'
    )
    print(f"\nTarget point: ({target_x}, {target_y}) in world coordinates")
    print(f"Index finger - Nearest position: {result['position']}")
    print(f"  Theta: {result['theta_deg']:.2f}°")
    print(f"  Actual position: ({result['x']:.4f}, {result['y']:.4f})")
    print(f"  Distance from target: {result['distance']:.4f} m")
    
    # Optional: Generate for all fingers (commented out by default)
    print("\n" + "=" * 60)
    print("Generating workspace for ALL fingers...")
    print("=" * 60)
    all_workspace_tables = generator.generate_all_fingers_workspace(step=10)  # Use step=10 for faster generation
    all_fingers_dir = DEFAULT_WORKSPACE_DIR / "all_fingers"
    generator.save_tables_to_csv(all_workspace_tables, output_dir=all_fingers_dir)
    generator.plot_workspace(all_workspace_tables, save_path=DEFAULT_WORKSPACE_DIR / "workspace_plot_all.png")
    
    print("\n" + "=" * 60)
    print("Done! Workspace tables generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
