import time
from typing import Callable, Iterable

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from custom_ros_messages.action import HandAdaptiveForce
from custom_ros_messages.msg import MotorCmds, MotorState, MotorStates
from custom_ros_messages.srv import SetHandAngles

from .hand_ros_core import (
    DOF_PER_HAND,
    HAND_ORDER,
    JOINT_NAMES,
    RIGHT,
    LEFT,
    SerialHandManager,
    motor_state_records,
    parse_hand_ids,
)


GESTURE_LIBRARY = {
    "open": [1000] * DOF_PER_HAND,
    "close": [0] * DOF_PER_HAND,
    "pinch": [1000, 1000, 0, 0, 1000, 0],
    "point": [0, 0, 0, 1000, 1000, 1000],
}


class RH56Driver(Node):
    """ROS 2 driver for one or two Inspire RH56DFX hands.

    The node is the single serial owner.  It publishes a 12-element state array
    in right[6] + left[6] order and accepts the same 12-element command layout.
    If only one hand is enabled, the disabled side is published with mode=-1 and
    incoming commands for that side are ignored.
    """

    def __init__(self):
        super().__init__("rh56_driver")
        self._ready = False

        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("hand_ids", "1,2")
        self.declare_parameter("publish_rate", 50.0)

        serial_port = self.get_parameter("serial_port").get_parameter_value().string_value
        hand_ids_param = self.get_parameter("hand_ids").get_parameter_value().string_value
        self.publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        try:
            hand_ids = parse_hand_ids(hand_ids_param)
        except ValueError as exc:
            self.get_logger().fatal(f"Invalid hand_ids parameter: {exc}")
            return

        self.get_logger().info(
            f"Connecting to RH56 hand IDs {list(hand_ids)} on port '{serial_port}'"
        )
        try:
            self._manager = SerialHandManager(serial_port=serial_port, hand_ids=hand_ids)
        except Exception as exc:
            self.get_logger().fatal(f"Failed to connect to configured hand(s): {exc}")
            return

        self._all_joint_names = [
            f"{side}_{joint}" for side in HAND_ORDER for joint in JOINT_NAMES
        ]

        self.hand_state_pub = self.create_publisher(MotorStates, "hands/state", 10)
        self.joint_state_pub = self.create_publisher(JointState, "hands/joint_states", 10)
        self.hand_cmd_sub = self.create_subscription(
            MotorCmds,
            "hands/cmd",
            self.hand_cmd_callback,
            10,
        )

        self._register_services()
        self._register_actions()

        period_s = 1.0 / max(1.0, float(self.publish_rate))
        self._publish_timer = self.create_timer(period_s, self.publish_once)
        self._ready = True
        self.get_logger().info(
            "RH56 driver ready for enabled sides: "
            + ", ".join(self._manager.enabled_sides)
        )

    def _register_services(self) -> None:
        for side in self._manager.enabled_sides:
            self.create_service(
                Trigger,
                f"hands/{side}/calibrate_force_sensors",
                lambda req, res, s=side: self.calibrate_callback(req, res, (s,)),
            )
            self.create_service(
                Trigger,
                f"hands/{side}/save_parameters",
                lambda req, res, s=side: self.trigger_for_sides(
                    req, res, (s,), "save parameters", self._manager.save_parameters
                ),
            )
            self.create_service(
                Trigger,
                f"hands/{side}/clear_errors",
                lambda req, res, s=side: self.trigger_for_sides(
                    req, res, (s,), "clear errors", self._manager.clear_errors
                ),
            )

        self.create_service(
            Trigger,
            "hands/calibrate_force_sensors",
            lambda req, res: self.calibrate_callback(req, res, self._manager.enabled_sides),
        )
        self.create_service(
            Trigger,
            "hands/save_parameters",
            lambda req, res: self.trigger_for_sides(
                req, res, self._manager.enabled_sides, "save parameters", self._manager.save_parameters
            ),
        )
        self.create_service(
            Trigger,
            "hands/clear_errors",
            lambda req, res: self.trigger_for_sides(
                req, res, self._manager.enabled_sides, "clear errors", self._manager.clear_errors
            ),
        )

        self.create_service(SetHandAngles, "hands/set_angles", self.set_angles_callback)
        self.create_service(SetHandAngles, "hands/set_speeds", self.set_speeds_callback)
        self.create_service(SetHandAngles, "hands/set_force_limits", self.set_force_limits_callback)
        self.create_service(SetHandAngles, "hands/set_current_limits", self.set_current_limits_callback)

        for gesture_name, angles in GESTURE_LIBRARY.items():
            self.create_service(
                Trigger,
                f"hands/{gesture_name}",
                lambda req, res, a=angles, g=gesture_name: self.gesture_callback(
                    req, res, self._manager.enabled_sides, a, g
                ),
            )
            for side in self._manager.enabled_sides:
                self.create_service(
                    Trigger,
                    f"hands/{side}/{gesture_name}",
                    lambda req, res, s=side, a=angles, g=gesture_name: self.gesture_callback(
                        req, res, (s,), a, g
                    ),
                )

    def _register_actions(self) -> None:
        self._action_servers = []
        if RIGHT in self._manager.enabled_sides:
            self._action_servers.append(
                ActionServer(
                    self,
                    HandAdaptiveForce,
                    "hands/right/adaptive_force_control",
                    lambda goal_handle: self.adaptive_force_callback(goal_handle, RIGHT),
                )
            )
        if LEFT in self._manager.enabled_sides:
            self._action_servers.append(
                ActionServer(
                    self,
                    HandAdaptiveForce,
                    "hands/left/adaptive_force_control",
                    lambda goal_handle: self.adaptive_force_callback(goal_handle, LEFT),
                )
            )

    def publish_once(self) -> None:
        states_by_side, errors = self._manager.read_states()
        if errors:
            self.get_logger().warn("; ".join(errors), throttle_duration_sec=5)

        records = motor_state_records(states_by_side)
        motor_states_msg = MotorStates()
        for record in records:
            state = MotorState()
            state.mode = int(record["mode"])
            state.q = float(record["q"])
            state.dq = float(record["dq"])
            state.ddq = float(record["ddq"])
            state.tau = float(record["tau"])
            state.tau_lim = float(record["tau_lim"])
            if hasattr(state, "current"):
                state.current = float(record["current"])
            state.temperature = float(record["temperature"])
            state.q_raw = float(record["q_raw"])
            state.dq_raw = float(record["dq_raw"])
            state.tau_raw = float(record["tau_raw"])
            state.tau_lim_raw = float(record["tau_lim_raw"])
            motor_states_msg.motor_states.append(state)
        self.hand_state_pub.publish(motor_states_msg)

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self._all_joint_names
        joint_state_msg.position = [float(record["q"]) for record in records]
        joint_state_msg.effort = [float(record["tau"]) for record in records]
        self.joint_state_pub.publish(joint_state_msg)

    def hand_cmd_callback(self, msg: MotorCmds) -> None:
        cmds = msg.motor_commands
        if len(cmds) != DOF_PER_HAND * 2:
            self.get_logger().warn(
                f"Received MotorCmds with {len(cmds)} commands, expected {DOF_PER_HAND * 2}."
            )
            return
        try:
            self._manager.apply_angle_commands_rad([cmd.q for cmd in cmds])
        except Exception as exc:
            self.get_logger().warn(f"Failed to apply hand command: {exc}")

    def _set_values_callback(
        self,
        request,
        response,
        label: str,
        setter: Callable[[str, Iterable[float]], object],
    ):
        values = list(request.angles)
        hand_selector = request.hand.lower().strip()

        def operation(side: str) -> None:
            setter(side, values)

        ok, message = self._manager.apply_to_sides(hand_selector, operation)
        response.success = ok
        response.message = (
            f"{label} applied to {message}" if ok else f"{label} failed: {message}"
        )
        return response

    def set_angles_callback(self, request, response):
        return self._set_values_callback(
            request, response, "angles", self._manager.set_angles_raw
        )

    def set_speeds_callback(self, request, response):
        return self._set_values_callback(
            request, response, "speeds", self._manager.set_speeds_raw
        )

    def set_force_limits_callback(self, request, response):
        return self._set_values_callback(
            request, response, "force limits", self._manager.set_force_limits_raw
        )

    def set_current_limits_callback(self, request, response):
        return self._set_values_callback(
            request, response, "current limits", self._manager.set_current_limits_raw
        )

    def trigger_for_sides(
        self,
        _request,
        response,
        sides: Iterable[str],
        label: str,
        operation: Callable[[str], object],
    ):
        try:
            for side in sides:
                operation(side)
        except Exception as exc:
            response.success = False
            response.message = f"{label} failed: {exc}"
            return response
        response.success = True
        response.message = f"{label} applied to {', '.join(sides)}"
        return response

    def calibrate_callback(self, _request, response, sides: Iterable[str]):
        try:
            for side in sides:
                self.get_logger().info(
                    f"Force sensor calibration requested for {side} hand; waiting ~15 seconds."
                )
                self._manager.calibrate_force_sensors(side)
            time.sleep(15)
        except Exception as exc:
            response.success = False
            response.message = f"force sensor calibration failed: {exc}"
            return response
        response.success = True
        response.message = f"force sensor calibration completed for {', '.join(sides)}"
        return response

    def gesture_callback(self, _request, response, sides: Iterable[str], angles: list[int], gesture_name: str):
        try:
            for side in sides:
                self._manager.set_angles_raw(side, angles)
        except Exception as exc:
            response.success = False
            response.message = f"gesture '{gesture_name}' failed: {exc}"
            return response
        response.success = True
        response.message = f"gesture '{gesture_name}' applied to {', '.join(sides)}"
        return response

    def adaptive_force_callback(self, goal_handle, side: str):
        self.get_logger().info(f"Adaptive force control requested for {side} hand.")

        goal = goal_handle.request
        feedback_msg = HandAdaptiveForce.Feedback()
        result_msg = HandAdaptiveForce.Result()

        try:
            for step in self._manager.adaptive_force_control_iter(
                side,
                target_forces=list(goal.target_forces),
                target_angles=list(goal.target_angles),
                step_size=goal.step_size,
                max_iterations=goal.max_iterations,
            ):
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(f"Goal canceled for {side} hand")
                    goal_handle.canceled()
                    result_msg.success = False
                    return result_msg

                feedback_msg.forces = step.get("forces", [])
                feedback_msg.angles = step.get("angles", [])
                goal_handle.publish_feedback(feedback_msg)

                if step.get("done"):
                    result_msg.success = True
                    result_msg.final_forces = step.get("final_forces", [])
                    result_msg.final_angles = step.get("final_angles", [])
                    goal_handle.succeed()
                    return result_msg
        except Exception as exc:
            self.get_logger().warn(f"Adaptive force control failed for {side} hand: {exc}")

        result_msg.success = False
        return result_msg


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    driver_node = RH56Driver()

    if not driver_node._ready:
        driver_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    executor.add_node(driver_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        driver_node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
