import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import scipy.optimize as optimize

# ================== Basic Finger Class ==================
class Finger:
    def __init__(self, l1, l2, base_offset, is_thumb=False):
        """
        Parameters:
            l1, l2 - Link lengths
            base_offset - Base point offset (x, y) in world coordinates
            is_thumb - Whether it is a thumb (reverse motion)
        """
        self.l1 = l1
        self.l2 = l2
        self.base_offset = np.array(base_offset)
        self.is_thumb = is_thumb
        
        self.min_reach = abs(self.l1 - self.l2)  # Minimum reachable distance
        self.max_reach = self.l1 + self.l2       # Maximum reachable distance

    def forward_kinematics(self, theta):
        """Return position in world coordinates"""
        theta_rad = np.radians(theta)
        
        if self.is_thumb:
            x_joint = self.l1 * np.sin(theta_rad)
            y_joint_local = -self.l1 * np.cos(theta_rad)
            x_end = x_joint + self.l2 * np.sin(2*theta_rad)
            y_end_local = y_joint_local - self.l2 * np.cos(2*theta_rad)
        else:
            x_joint = self.l1 * np.sin(theta_rad)
            y_joint_local = self.l1 * np.cos(theta_rad)
            x_end = x_joint + self.l2 * np.sin(2*theta_rad)
            y_end_local = y_joint_local + self.l2 * np.cos(2*theta_rad)

        joint_world = self.base_offset + np.array([x_joint, y_joint_local])
        end_world = self.base_offset + np.array([x_end, y_end_local])
        
        return joint_world, end_world

    def calculate_alpha(self, end_world):
        """Calculate the alpha angle in world coordinates"""
        end_local = end_world - self.base_offset
        x, y = end_local
        
        if self.is_thumb:
            angle_rad = np.arctan2(x, -y)
            alpha_rad = np.pi - angle_rad
        else:
            angle_rad = np.arctan2(x, y)
            alpha_rad = np.pi - angle_rad
            
        return np.degrees(alpha_rad) % 360

    def is_reachable(self, target_world):
        """Check if the target point is within the reachable workspace"""
        target_local = target_world - self.base_offset
        distance = np.linalg.norm(target_local)
        return self.min_reach <= distance <= self.max_reach

    def inverse_kinematics(self, target_world, tolerance=0.01):
        """Solve for the target point in world coordinates"""
        if not self.is_reachable(target_world):
            raise ValueError(f"Target point {target_world} is out of the workspace range "
                           f"(Min: {self.min_reach:.2f}, Max: {self.max_reach:.2f})")
        
        target_local = target_world - self.base_offset
        
        best_theta = 0
        min_error = float('inf')
        
        for theta in np.linspace(0, 90, 1000):
            _, end_world = self.forward_kinematics(theta)
            error = np.linalg.norm(end_world - target_world)
            
            if error < min_error:
                min_error = error
                best_theta = theta
                
        return best_theta

# ================== Cooperative Control System ==================
class HandSystem:
    def __init__(self):
        # Define finger base points (world coordinates)
        self.index = Finger(
            l1=3.2, 
            l2=4.6,
            base_offset=(0, 0),  # Index finger base point
            is_thumb=False
        )
        
        self.thumb = Finger(
            l1=5.5,
            l2=5.0,
            base_offset=(-2.9, -7.7),  # Thumb base point
            is_thumb=True
        )

    def reach_common_target(self, target_world, tolerance=0.01):
        """Control both fingers to reach the same target point"""
        # Check if both fingers can reach the target point
        if not self.index.is_reachable(target_world):
            raise ValueError("Target point is out of the index finger's reachable range")
        if not self.thumb.is_reachable(target_world):
            raise ValueError("Target point is out of the thumb's reachable range")
            
        try:
            # Solve for the index finger
            theta_index = self.index.inverse_kinematics(target_world, tolerance)
            joint_index, end_index = self.index.forward_kinematics(theta_index)
            alpha_index = self.index.calculate_alpha(end_index)
            
            # Solve for the thumb
            theta_thumb = self.thumb.inverse_kinematics(target_world, tolerance)
            joint_thumb, end_thumb = self.thumb.forward_kinematics(theta_thumb)
            alpha_thumb = self.thumb.calculate_alpha(end_thumb)
            
            # Check the error between the actual position and the target position
            index_error = np.linalg.norm(end_index - target_world)
            thumb_error = np.linalg.norm(end_thumb - target_world)
            
            if max(index_error, thumb_error) > tolerance:
                print(f"Warning: Failed to achieve the target precision")
                print(f"Index finger error: {index_error:.3f}")
                print(f"Thumb error: {thumb_error:.3f}")
            
            return {
                'target': target_world,
                'index': {
                    'theta': theta_index,
                    'alpha': alpha_index,
                    'position': end_index,
                    'joint': joint_index,
                    'error': index_error
                },
                'thumb': {
                    'theta': theta_thumb,
                    'alpha': alpha_thumb,
                    'position': end_thumb,
                    'joint': joint_thumb,
                    'error': thumb_error
                }
            }
            
        except ValueError as e:
            print(f"Motion planning failed: {str(e)}")
            return None

    def plot_hand(self, result):
        """Visualize the coordinated motion of both hands"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot base points
        ax.plot(*self.index.base_offset, 'gs', markersize=12, label='Index Base')
        ax.plot(*self.thumb.base_offset, 'rs', markersize=12, label='Thumb Base')
        
        # Plot index finger
        index_data = result['index']
        ax.plot([self.index.base_offset[0], index_data['joint'][0]],
                [self.index.base_offset[1], index_data['joint'][1]], 
                'b-o', lw=3, label='Index Finger')
        ax.plot([index_data['joint'][0], index_data['position'][0]],
                [index_data['joint'][1], index_data['position'][1]], 
                'b-o', lw=3)
        
        # Plot thumb
        thumb_data = result['thumb']
        ax.plot([self.thumb.base_offset[0], thumb_data['joint'][0]],
                [self.thumb.base_offset[1], thumb_data['joint'][1]], 
                'r-o', lw=3, label='Thumb')
        ax.plot([thumb_data['joint'][0], thumb_data['position'][0]],
                [thumb_data['joint'][1], thumb_data['position'][1]], 
                'r-o', lw=3)
        
        # Annotate angles
        def draw_angle(ax, center, start, end, radius, color, label):
            arc = Arc(center, 2*radius, 2*radius, theta1=start, theta2=end, 
                     color=color, lw=1.5)
            ax.add_patch(arc)
            mid_angle = np.radians((start + end)/2)
            ax.text(center[0] + radius*1.3*np.cos(np.radians(mid_angle)),
                    center[1] + radius*1.3*np.sin(np.radians(mid_angle)),
                    label, color=color, ha='center', va='center')
        
        # Index finger angle
        draw_angle(ax, self.index.base_offset, 90, 90-index_data['theta'], 
                  1.5, 'blue', f'θ_i={index_data["theta"]:.1f}°')
        
        # Thumb angle
        draw_angle(ax, self.thumb.base_offset, 270, 270+thumb_data['theta'], 
                  1.2, 'red', f'θ_t={thumb_data["theta"]:.1f}°')
        
        # Plot target point and actual position
        target = result['target']
        ax.plot(*target, 'k*', markersize=20, label='Target')
        
        # Plot error circle
        error_circle = Circle(target, 0.1, fill=False, color='gray', linestyle='--')
        ax.add_patch(error_circle)
        
        # Set coordinate system
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':')
        ax.set_xlabel("X World Coordinate")
        ax.set_ylabel("Y World Coordinate")
        ax.legend()
        
        # Add error information
        plt.title(f"Hand Coordination System\n" + 
                 f"Target: ({target[0]:.2f}, {target[1]:.2f})\n" +
                 f"Index Error: {index_data['error']:.3f}, " +
                 f"Thumb Error: {thumb_data['error']:.3f}")
        plt.show()

# ================== Usage Example ==================
if __name__ == "__main__":
    hand = HandSystem()
    
    target_world = np.array([4, -4])
    #target_world = np.array([1, -4])
    
    result = hand.reach_common_target(target_world, tolerance=7.0)
    
    if result:
        print("\n=== Motion Planning Result ===")
        print(f"\nTarget Position: ({target_world[0]:.2f}, {target_world[1]:.2f})")
        
        print("\nIndex Finger Result:")
        print(f"Joint Angle: {result['index']['theta']:.2f}°")
        print(f"End Angle α: {result['index']['alpha']:.2f}°")
        print(f"Actual Position: ({result['index']['position'][0]:.2f}, {result['index']['position'][1]:.2f})")
        print(f"Position Error: {result['index']['error']:.3f}")
        
        print("\nThumb Result:")
        print(f"Joint Angle: {result['thumb']['theta']:.2f}°")
        print(f"End Angle α: {result['thumb']['alpha']:.2f}°")
        print(f"Actual Position: ({result['thumb']['position'][0]:.2f}, {result['thumb']['position'][1]:.2f})")
        print(f"Position Error: {result['thumb']['error']:.3f}")
        
        hand.plot_hand(result)
