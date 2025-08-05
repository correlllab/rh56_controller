import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_srvs.srv import Trigger

from custom_ros_messages.msg import MotorCmd, MotorCmds, MotorState, MotorStates
from custom_ros_messages.action import HandAdaptiveForce
from custom_ros_messages.srv import SetHandAngles

from .rh56_hand import RH56Hand

import threading
import time
import math
from typing import List, Optional, Tuple
import numpy as np

class RH56Driver(Node):
    """
    ROS 2 bimanual driver for the Inspire RH56DFX hands.

    This node handles the communication with the hand hardware, publishes sensor
    data (joint angles and forces), and provides services for controlling the hand.
    It controls both a right hand (ID 1) and a left hand (ID 2) on the same serial bus.
    """
    def __init__(self):
        super().__init__('rh56_driver')

        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('publish_rate', 50.0)

        # Get parameters
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.get_logger().info(f"Connecting to hands on port '{serial_port}'")

        try:
            self.righthand = RH56Hand(port=serial_port, hand_id=1)
            self.lefthand = RH56Hand(port=serial_port, hand_id=2)
            self.get_logger().info("Successfully connected to both hands.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to connect to the hands: {e}")
            rclpy.shutdown()
            return

        # Define joint names for both hands
        self._right_joint_names = [f'right_{j}' for j in ['pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotation']]
        self._left_joint_names = [f'left_{j}' for j in ['pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotation']]
        self._all_joint_names = self._right_joint_names + self._left_joint_names

        # Publisher for combined state, matching the C++ package
        self.hand_state_pub = self.create_publisher(MotorStates, 'hands/state', 10)
        
        # Subscriber for combined command, matching the C++ package
        self.hand_cmd_sub = self.create_subscription(
            MotorCmds,
            'hands/cmd',
            self.hand_cmd_callback,
            10
        )

        # Services - now namespaced for each hand
        self.create_service(Trigger, 'hands/right/calibrate_force_sensors', lambda r, s: self.calibrate_callback(r, s, self.righthand))
        self.create_service(Trigger, 'hands/left/calibrate_force_sensors',  lambda r, s: self.calibrate_callback(r, s, self.lefthand))

        self.create_service(Trigger, 'hands/right/save_parameters', lambda r, s: self.save_callback(r, s, self.righthand))
        self.create_service(Trigger, 'hands/left/save_parameters',  lambda r, s: self.save_callback(r, s, self.lefthand))

        self._gesture_library = {
            "open":  [1000] * 6,
            "close": [0] * 6,
            "pinch": [1000, 1000, 0, 0, 1000, 0],
            "point": [0, 0, 0, 1000, 1000, 1000],
        }

        for gesture_name, angles in self._gesture_library.items():
            self.create_service(
                Trigger, f'hands/right/{gesture_name}',
                lambda req, res, a=angles, g=gesture_name: self.gesture_callback(req, res, a, [self.righthand], g)
            )
            self.create_service(
                Trigger, f'hands/left/{gesture_name}',
                lambda req, res, a=angles, g=gesture_name: self.gesture_callback(req, res, a, [self.lefthand], g)
            )
            self.create_service(
                Trigger, f'hands/{gesture_name}',
                lambda req, res, a=angles, g=gesture_name: self.gesture_callback(req, res, a, [self.righthand, self.lefthand], g)
            )

        self.create_service(
            SetHandAngles,
            'hands/set_angles',
            self.set_joint_angles_callback
        )

        self.right_action_server = ActionServer(
            self,
            HandAdaptiveForce,
            'hands/right/adaptive_force_control',
            lambda goal_handle: self.adaptive_force_callback(goal_handle, self.righthand)
        )

        self.left_action_server = ActionServer(
            self,
            HandAdaptiveForce,
            'hands/left/adaptive_force_control',
            lambda goal_handle: self.adaptive_force_callback(goal_handle, self.lefthand)
        )

        # Threading lock for safe serial communication
        self.hand_lock = threading.Lock()

        # Start the main publishing loop in a separate thread
        self.publisher_thread = threading.Thread(target=self.publish_loop)
        self.publisher_thread.daemon = True
        self.publisher_thread.start()

        self.get_logger().info("RH56 Bimanual Driver node started successfully.")

    def publish_loop(self):
        """Continuously reads sensor data and publishes it."""
        rate = self.create_rate(self.publish_rate)
        while rclpy.ok():
            with self.hand_lock:
                right_angles = self.righthand.angle_read()
                right_forces = self.righthand.force_act()
                right_temps = self.righthand.temp_read()
                
                left_angles = self.lefthand.angle_read()
                left_forces = self.lefthand.force_act()
                left_temps = self.lefthand.temp_read()

            if not (right_angles and right_forces and left_angles and left_forces):
                self.get_logger().warn("Incomplete data read from one or both hands.", throttle_duration_sec=5)
                rate.sleep()
                continue

            now = self.get_clock().now().to_msg()
            all_angles = right_angles + left_angles
            all_forces = right_forces + left_forces
            all_limits = self.righthand.force_limits + self.lefthand.force_limits
            all_temps  = right_temps + left_temps

            # --- Populate and publish the MotorStates message ---
            motor_states_msg = MotorStates()
            for i in range(12):
                state = MotorState()
                # Populate the state message based on the unitree_msgs definition
                state.mode = 0  # Mode is not used by the hand controller, set to 0
                state.q = (all_angles[i] / 1000.0) * math.pi
                state.dq = 0.0
                state.ddq = 0.0
                state.tau = float(all_forces[i])
                state.tau_lim = float(all_limits[i])
                state.temperature = float(all_temps[i])
                # unused
                state.q_raw = 0.0
                state.dq_raw = 0.0
                state.tau_raw = 0.0
                state.tau_lim_raw = 0.0

                motor_states_msg.motor_states.append(state)
            self.hand_state_pub.publish(motor_states_msg)

            # --- Populate and publish the standard JointState message for RViz ---
            # js_msg = JointState()
            # js_msg.header.stamp = now
            # js_msg.name = self._all_joint_names
            # js_msg.position = [s.q for s in motor_states_msg.motor_states]
            # js_msg.effort = [s.tau_est for s in motor_states_msg.motor_states]
            # self.joint_state_pub.publish(js_msg)

            rate.sleep()

    def hand_cmd_callback(self, msg: MotorCmds):
        """Receives MotorCmds and sends them to the respective hands."""
        cmds = msg.motor_commands
        if len(cmds) != 12:
            self.get_logger().warn(f"Received MotorCmds with {len(cmds)} commands, expected 12.")
            return

        # Right hand commands (first 6)
        right_pos_rad = [cmd.q for cmd in cmds[:6]]
        right_angles_raw = [(p / math.pi) * 1000.0 for p in right_pos_rad]
        right_angles = [max(0, min(1000, int(a))) for a in right_angles_raw]

        # Left hand commands (last 6)
        left_pos_rad = [cmd.q for cmd in cmds[6:]]
        left_angles_raw = [(p / math.pi) * 1000.0 for p in left_pos_rad]
        left_angles = [max(0, min(1000, int(a))) for a in left_angles_raw]

        with self.hand_lock:
            # self.righthand.angle_set(right_angles)
            # self.lefthand.angle_set(left_angles)
            self.send_angles_concurrent([(self.righthand, right_angles), (self.lefthand, left_angles)])

    def calibrate_callback(self, request: Trigger.Request, response: Trigger.Response, hand: RH56Hand):
        hand_name = "right" if hand.hand_id == 1 else "left"
        self.get_logger().info(f"Force sensor calibration service called for {hand_name} hand. This will take ~15 seconds.")
        with self.hand_lock:
            hand.gesture_force_clb(1)
            time.sleep(15) # Wait for the hardware calibration routine to finish
        response.success = True
        response.message = f"Force sensor calibration completed for {hand_name} hand."
        self.get_logger().info(f"Calibration finished for {hand_name} hand.")
        return response

    def save_callback(self, request: Trigger.Request, response: Trigger.Response, hand: RH56Hand):
        hand_name = "right" if hand.hand_id == 1 else "left"
        self.get_logger().info(f"Save parameters service called for {hand_name} hand.")
        with self.hand_lock:
            hand.save_parameters()
        response.success = True
        response.message = f"Parameters saved to {hand_name} hand's non-volatile memory."
        return response

    # def adaptive_force_callback(self, request: AdaptiveForce.Request, response: AdaptiveForce.Response, hand: RH56Hand):
    #     hand_name = "right" if hand.hand_id == 1 else "left"
    #     self.get_logger().info(f"Adaptive force control service called for {hand_name} hand.")

    #     with self.hand_lock:
    #         results = hand.adaptive_force_control(
    #             target_forces=list(request.target_forces),
    #             target_angles=list(request.target_angles),
    #             step_size=request.step_size,
    #             max_iterations=request.max_iterations
    #         )

    #     if results and results.get('final_forces') is not None:
    #         response.success = True
    #         response.final_forces = results['final_forces']
    #         response.final_angles = results['final_angles']
    #         self.get_logger().info(f"Adaptive force control for {hand_name} hand finished successfully.")
    #     else:
    #         response.success = False
    #         self.get_logger().error(f"Adaptive force control for {hand_name} hand failed.")

    #     return response

    def adaptive_force_callback(self, goal_handle, hand):
        hand_name = "right" if hand.hand_id == 1 else "left"
        self.get_logger().info(f"Adaptive force control action called for {hand_name} hand.")

        goal = goal_handle.request
        feedback_msg = HandAdaptiveForce.Feedback()
        result_msg = HandAdaptiveForce.Result()

        with self.hand_lock:
            for step in hand.adaptive_force_control_iter(
                target_forces=list(goal.target_forces),
                target_angles=list(goal.target_angles),
                step_size=goal.step_size,
                max_iterations=goal.max_iterations
            ):
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(f"Goal canceled for {hand_name}")
                    goal_handle.canceled()
                    return HandAdaptiveForce.Result(success=False)

                # Emit feedback if available
                feedback_msg.forces = step["forces"]
                feedback_msg.angles = step["angles"]
                goal_handle.publish_feedback(feedback_msg)

                if step.get("done"):
                    result_msg.success = True
                    result_msg.final_forces = step["final_forces"]
                    result_msg.final_angles = step["final_angles"]
                    goal_handle.succeed()
                    return result_msg

        # If it exits the loop without "done"
        result_msg.success = False
        return result_msg

    def gesture_callback(self, request, response, angles: List[int], hands: List[RH56Hand], gesture_name: Optional[str] = None):
        with self.hand_lock:
            pairs = []
            for hand in hands:
                hand_label = "right" if hand.hand_id == 1 else "left"
                if gesture_name:
                    self.get_logger().info(f"Setting {hand_label} hand to gesture '{gesture_name}'")
                else:
                    self.get_logger().info(f"Setting {hand_label} hand to raw joint values")
                # Ensure angles are within valid range
                angles = np.clip(angles, 0, 1000).astype(int).tolist()
                pairs.append((hand, angles))
            self.send_angles_concurrent(pairs)

        response.success = True
        hand_msg = " and ".join(["right" if h.hand_id == 1 else "left" for h in hands])
        response.message = f"Set gesture '{gesture_name}' on {hand_msg} hand" if gesture_name else f"Set raw joint angles on {hand_msg} hand"
        return response

    def set_joint_angles_callback(self, request, response):
        hand_str = request.hand.lower()
        hands = []
        if hand_str in ["left", "both"]:
            hands.append(self.lefthand)
        if hand_str in ["right", "both"]:
            hands.append(self.righthand)
        if not hands:
            response.success = False
            response.message = f"Invalid hand spec: '{request.hand}'"
            return response

        if len(request.angles) != 6:
            response.success = False
            response.message = "Expected exactly 6 joint angles."
            return response

        # Clamp and convert
        clamped_angles = [max(0, min(1000, int(a))) for a in request.angles]
        self.gesture_callback(request, response, clamped_angles, hands, None)  # gesture_name=None
        response.success = True
        response.message = f"Set joint angles for '{hand_str}'"
        return response
    
    def send_angles_concurrent(self, hand_angle_pairs: List[Tuple[RH56Hand, List[int]]]):
        threads = []
        for hand, angles in hand_angle_pairs:
            t = threading.Thread(target=hand.angle_set, args=(angles,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

def main(args=None):
    rclpy.init(args=args)
    
    # Use a MultiThreadedExecutor to handle callbacks and the publisher thread concurrently
    executor = MultiThreadedExecutor()
    driver_node = RH56Driver()
    
    # Check if the node was initialized correctly before spinning
    if rclpy.ok():
        executor.add_node(driver_node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            driver_node.destroy_node()
            executor.shutdown()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
