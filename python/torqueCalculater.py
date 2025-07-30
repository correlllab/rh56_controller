import numpy as np
from controller import RH56Hand

def _calculate_kinematics(l1: float, l2: float, theta_deg: float) -> dict:
    """
    Calculates kinematic properties for a single finger angle.
    """
    theta_rad = np.radians(theta_deg)

    x_joint = l1 * np.sin(theta_rad)
    y_joint = l1 * np.cos(theta_rad)
    x_end = x_joint + l2 * np.sin(2 * theta_rad)
    y_end = y_joint + l2 * np.cos(2 * theta_rad)

    d1 = np.sqrt(x_end**2 + y_end**2)

    # Calculate the angle of d1 with respect to the positive x-axis
    angle_d1_rad = np.arctan2(x_end, y_end)

    # Calculate alpha1 as the angle with the positive y-axis (vertical)
    alpha1_calculated_rad = np.pi - angle_d1_rad
    alpha1_calculated_deg = np.degrees(alpha1_calculated_rad)

    return {
        'x_joint': x_joint,
        'y_joint': y_joint,
        'x_end': x_end,
        'y_end': y_end,
        'd1': d1,
        'alpha1_deg': alpha1_calculated_deg,
    }

class TorqueCalculator:
    def __init__(self, hand: RH56Hand, l1: float, l2: float):
        """
        Initialize the torque calculator
        
        Args:
            hand: RH56Hand controller instance
            l1: Length of the first phalanx
            l2: Length of the second phalanx
        """
        self.hand = hand
        self.l1 = l1
        self.l2 = l2
        
    def calculate_torque(self, finger_index: int, angle_deg: float = None) -> float:
        """
        Calculate the torque for a specific finger
        
        Args:
            finger_index: Index of the finger (0-5)
            angle_deg: Optional. The angle in degrees to use for calculation.
                       If None, the current angle is read from the hand.
            
        Returns:
            float: Calculated torque in N⋅m.
        """
        # Get force reading for the specific finger
        forces = self.hand.force_act()
        if forces is None or finger_index >= len(forces):
            raise ValueError("Failed to read force values or invalid finger index")
        
        force_grams = forces[finger_index]
        # Convert force from grams to Newtons (1 kg = 9.81 N)
        force_newtons = (force_grams / 1000.0) * 9.81
        
        if angle_deg is None:
            # Get current angle of the finger if not provided
            angles = self.hand.angle_read()
            if angles is None:
                raise ValueError("Failed to read angle values")
            # Convert the angle reading (0-1000) to degrees (0-180)
            angle_deg = (angles[finger_index] / 1000.0) * 180.0
        
        # Simulate finger to get geometric parameters
        kinematics = _calculate_kinematics(self.l1, self.l2, angle_deg)
        
        d1 = kinematics['d1']
        alpha1_deg = kinematics['alpha1_deg']
        
        alpha1_rad = np.radians(alpha1_deg)
        
        # Calculate torque: τ = F × d × sin(α)
        torque = force_newtons * d1 * np.sin(alpha1_rad)
        
        return torque

    def get_all_torques(self) -> list:
        """
        Calculate torques for all fingers
        
        Returns:
            list: Torques for all fingers in N⋅m
        """
        return [self.calculate_torque(i) for i in range(6)]