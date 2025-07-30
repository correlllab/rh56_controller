import numpy as np

# Forward-declare RH56Hand to resolve circular dependency for type hinting
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional
if TYPE_CHECKING:
    from .rh56_hand import RH56Hand


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
        'd1': d1,
        'alpha1_deg': alpha1_calculated_deg,
    }

class TorqueCalculator:
    def __init__(self, hand: 'RH56Hand', finger_lengths: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        Initialize the torque calculator.
        
        Args:
            hand: RH56Hand controller instance.
            finger_lengths: Optional dictionary mapping finger index to (l1, l2) lengths.
        """
        self.hand = hand
        # Default finger lengths if not provided
        self.finger_lengths = finger_lengths or {
            0: (0.032, 0.041),  # Pinky
            1: (0.032, 0.046),  # Ring
            2: (0.032, 0.051),  # Middle
            3: (0.032, 0.046),  # Index
            4: (0.050, 0.050),  # Thumb Bend
            5: (0.050, 0.050),  # Thumb Rotation
        }

    def get_finger_lengths(self, finger_index: int) -> Tuple[float, float]:
        """Returns the l1, l2 lengths for a specific finger."""
        if finger_index not in self.finger_lengths:
            raise ValueError(f"Lengths for finger {finger_index} not defined.")
        return self.finger_lengths[finger_index]

    def calculate_torque(self, finger_index: int, angle_deg: float = None) -> float:
        """
        Calculate the torque for a specific finger.
        
        Args:
            finger_index: Index of the finger (0-5).
            angle_deg: Optional. The angle in degrees to use for calculation.
                       If None, the current angle is read from the hand.
            
        Returns:
            float: Calculated torque in N·m.
        """
        forces = self.hand.force_act()
        if forces is None or finger_index >= len(forces):
            raise ValueError("Failed to read force values or invalid finger index")
        
        force_grams = forces[finger_index]
        force_newtons = (force_grams / 1000.0) * 9.81
        
        if angle_deg is None:
            angles = self.hand.angle_read()
            if angles is None:
                raise ValueError("Failed to read angle values")
            angle_deg = (angles[finger_index] / 1000.0) * 180.0
        
        l1, l2 = self.get_finger_lengths(finger_index)
        kinematics = _calculate_kinematics(l1, l2, angle_deg)
        
        d1 = kinematics['d1']
        alpha1_deg = kinematics['alpha1_deg']
        alpha1_rad = np.radians(alpha1_deg)
        
        torque = force_newtons * d1 * np.sin(alpha1_rad)
        return torque

    def get_all_torques(self) -> List[float]:
        """Calculate torques for all fingers."""
        return [self.calculate_torque(i) for i in range(6)]


# ================== Kinematic Model and Coordination ==================
class FingerModel:
    def __init__(self, l1: float, l2: float, base_offset: Tuple[float, float], is_thumb: bool = False):
        """
        A 2D kinematic model for a single finger.

        Args:
            l1: Length of the first phalanx.
            l2: Length of the second phalanx.
            base_offset: (x, y) position of the finger's base in the hand's frame.
            is_thumb: Flag for thumb's reverse kinematics.
        """
        self.l1 = l1
        self.l2 = l2
        self.base_offset = np.array(base_offset)
        self.is_thumb = is_thumb
        self.min_reach = abs(self.l1 - self.l2)
        self.max_reach = self.l1 + self.l2

    def forward_kinematics(self, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates joint and end-effector positions from a joint angle."""
        theta_rad = np.radians(theta)
        if self.is_thumb:
            x_joint = self.l1 * np.sin(theta_rad)
            y_joint_local = -self.l1 * np.cos(theta_rad)
            x_end = x_joint + self.l2 * np.sin(2 * theta_rad)
            y_end_local = y_joint_local - self.l2 * np.cos(2 * theta_rad)
        else:
            x_joint = self.l1 * np.sin(theta_rad)
            y_joint_local = self.l1 * np.cos(theta_rad)
            x_end = x_joint + self.l2 * np.sin(2 * theta_rad)
            y_end_local = y_joint_local + self.l2 * np.cos(2 * theta_rad)

        joint_world = self.base_offset + np.array([x_joint, y_joint_local])
        end_world = self.base_offset + np.array([x_end, y_end_local])
        return joint_world, end_world
    
    def is_reachable(self, target_world: np.ndarray) -> bool:
        """Checks if a target point is within the finger's workspace."""
        target_local = target_world - self.base_offset
        distance = np.linalg.norm(target_local)
        return self.min_reach <= distance <= self.max_reach

    def inverse_kinematics(self, target_world: np.ndarray) -> float:
        """Solves for the joint angle required to reach a target point using a search."""
        if not self.is_reachable(target_world):
            # Return the angle that gets closest to the target
            target_local = target_world - self.base_offset
            dist = np.linalg.norm(target_local)
            if dist > self.max_reach: # Target is too far, fully extend
                return 90.0
            else: # Target is too close, fully retract
                return 0.0
    
        best_theta = 0
        min_error = float('inf')
        for theta in np.linspace(0, 90, 1000):
            _, end_world = self.forward_kinematics(theta)
            error = np.linalg.norm(end_world - target_world)
            if error < min_error:
                min_error = error
                best_theta = theta
        return best_theta

class HandKinematics:
    """
    Manages the kinematic models for all fingers of the hand.
    """
    def __init__(self):
        # These base offsets are estimates and should be configured based on
        # the hand's physical properties and a defined coordinate frame.
        # Units are in meters.
        finger_params = {
            "pinky":  {'l1': 0.032, 'l2': 0.041, 'base_offset': (0.035, 0), 'is_thumb': False},
            "ring":   {'l1': 0.032, 'l2': 0.046, 'base_offset': (0.012, 0), 'is_thumb': False},
            "middle": {'l1': 0.032, 'l2': 0.051, 'base_offset': (-0.012, 0), 'is_thumb': False},
            "index":  {'l1': 0.032, 'l2': 0.046, 'base_offset': (-0.035, 0), 'is_thumb': False},
            "thumb_bend": {'l1': 0.050, 'l2': 0.050, 'base_offset': (-0.025, -0.075), 'is_thumb': True},
        }
        self.fingers = {name: FingerModel(**params) for name, params in finger_params.items()}
        self.finger_names = list(self.fingers.keys())

    def solve_ik_for_finger(self, finger_name: str, target_world: np.ndarray) -> float:
        """
        Calculates the required joint angle for a single finger to reach a target.

        Args:
            finger_name: The name of the finger (e.g., "index").
            target_world: A numpy array for the (x, y) target position.

        Returns:
            The required joint angle in degrees (0-90).
        """
        if finger_name not in self.fingers:
            raise ValueError(f"Unknown finger: {finger_name}")
        return self.fingers[finger_name].inverse_kinematics(target_world)