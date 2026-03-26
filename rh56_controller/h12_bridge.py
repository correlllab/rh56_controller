"""
h12_bridge.py — Direct rclpy bridge for H1-2 arm + hand control.

Requires rclpy to be importable (source /opt/ros/humble/setup.bash before running).

Run with:
    source /opt/ros/humble/setup.bash
    UV_PROJECT_ENVIRONMENT=.venv310 uv run python -m rh56_controller.grasp_viz --h12 --bimanual --real-h12

Actions used:
  /frame_task   (custom_ros_messages/FrameTask)  — single arm
  /dual_arm     (custom_ros_messages/DualArm)    — bimanual
  /named_config (custom_ros_messages/NamedConfig)

Topics subscribed:
  /right_ee_pose  (geometry_msgs/PoseStamped)
  /left_ee_pose   (geometry_msgs/PoseStamped)

Topics published:
  /right_hand_cmd  (std_msgs/Float64MultiArray)  — 6 floats, [0=close .. 1=open]
  /left_hand_cmd   (std_msgs/Float64MultiArray)  — same
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional
import numpy as np

_log = logging.getLogger(__name__)


class H12Bridge:
    """
    Direct rclpy client for H1-2 arm + Inspire hand control.

    Call connect() once (requires rclpy importable), then use:
      get_wrist_pose(frame_name)  → 4×4 np.ndarray or None
      send_arm(frame_name, T)     → bool
      send_dual_arm(right_T, left_T) → bool
      send_hand_cmd(values, arm)  → None   (values: 6 floats in [0,1])
      send_named_config(name)     → bool
      disconnect()
    """

    def __init__(self, bimanual: bool = False) -> None:
        self._bimanual   = bimanual
        self._node       = None
        self._executor   = None
        self._spin_thread: Optional[threading.Thread] = None
        self._connected  = False
        self.last_error  = ""
        self._ee_poses: dict[str, object] = {}
        self._right_hand_pub = None
        self._left_hand_pub  = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import MultiThreadedExecutor
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import Float64MultiArray

            if not rclpy.ok():
                rclpy.init()

            self._node = Node("grasp_viz_h12")

            # Subscribe to EE poses published by dual_arm_server at 100 Hz.
            # Only log the first message per side to avoid 100 Hz log flooding.
            def _right_cb(msg):
                if "right_wrist_yaw_link" not in self._ee_poses:
                    _log.info("H12Bridge: /right_ee_pose first msg pos=[%.3f,%.3f,%.3f]",
                              msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
                self._ee_poses["right_wrist_yaw_link"] = msg
            def _left_cb(msg):
                if "left_wrist_yaw_link" not in self._ee_poses:
                    _log.info("H12Bridge: /left_ee_pose first msg received")
                self._ee_poses["left_wrist_yaw_link"] = msg
            self._node.create_subscription(PoseStamped, "/right_ee_pose", _right_cb, 10)
            self._node.create_subscription(PoseStamped, "/left_ee_pose",  _left_cb,  10)

            # Joint states for IK seeding — optional, skip if sensor_msgs unavailable
            try:
                from sensor_msgs.msg import JointState
                self._node.create_subscription(
                    JointState, "/joint_states",
                    lambda msg: setattr(self, "_joint_state_msg", msg), 10)
            except ImportError:
                _log.warning("H12Bridge: sensor_msgs not available, /joint_states disabled")

            # Publishers for hand commands (hand_controller_node subscribes)
            self._right_hand_pub = self._node.create_publisher(
                Float64MultiArray, "/right_hand_cmd", 10)
            self._left_hand_pub = self._node.create_publisher(
                Float64MultiArray, "/left_hand_cmd", 10)

            # Heartbeat timer: without at least one timer, some DDS/rclpy builds never
            # wake the executor's wait-set for subscriptions — callbacks would silently
            # never fire even though topics are publishing.
            self._node.create_timer(0.5, lambda: None)

            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self._node)

            self._connected = True
            self._spin_thread = threading.Thread(
                target=self._spin_loop, daemon=True, name="h12-spin")
            self._spin_thread.start()

            _log.info("H12Bridge: connected (ROS2 direct).")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("H12Bridge.connect failed: %s", exc)
            return False

    def _spin_loop(self):
        try:
            self._executor.spin()
        except Exception as exc:
            _log.error("H12Bridge spin loop crashed: %s", exc, exc_info=True)

    def disconnect(self) -> None:
        self._connected = False
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pose query (reads from cached topic subscription)
    # ------------------------------------------------------------------

    def get_joint_states(self, timeout: float = 2.0) -> Optional[dict]:
        """
        Return latest joint states as {joint_name: angle_rad}, or None if not yet received.
        Waits up to `timeout` seconds for the first message.
        """
        t0 = time.monotonic()
        while self._joint_state_msg is None and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        msg = self._joint_state_msg
        if msg is None:
            return None
        return {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    def get_wrist_pose(self, frame_name: str,
                       base_frame: str = "pelvis",
                       timeout: float = 5.0) -> Optional[np.ndarray]:
        """Return 4×4 world→wrist transform (pelvis frame), or None on failure."""
        from scipy.spatial.transform import Rotation

        t0 = time.monotonic()
        while frame_name not in self._ee_poses and time.monotonic() - t0 < timeout:
            time.sleep(0.05)

        msg = self._ee_poses.get(frame_name)
        if msg is None:
            self.last_error = f"No pose received for {frame_name} within {timeout}s"
            _log.info("H12Bridge.get_wrist_pose FAILED: %s", self.last_error)
            return None

        p = msg.pose.position
        q = msg.pose.orientation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [p.x, p.y, p.z]
        _log.info("H12Bridge.get_wrist_pose OK [%s]: pos=[%.3f, %.3f, %.3f]",
                  frame_name, p.x, p.y, p.z)
        return T

    # ------------------------------------------------------------------
    # Single-arm: FrameTask action
    # ------------------------------------------------------------------

    def send_arm(self, frame_name: str, T: np.ndarray,
                 timeout: float = 15.0) -> bool:
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import FrameTask

            client = ActionClient(self._node, FrameTask, "frame_task")
            if not client.wait_for_server(timeout_sec=5.0):
                self.last_error = "frame_task server not available"
                _log.warning("H12Bridge.send_arm FAILED: %s", self.last_error)
                return False

            pose = _mat_to_pose(T)
            goal = FrameTask.Goal()
            goal.frame_names   = [frame_name]
            goal.frame_targets = [pose]

            _log.info("H12Bridge.send_arm → %s pos=[%.3f, %.3f, %.3f]",
                      frame_name, T[0, 3], T[1, 3], T[2, 3])
            ok = _send_action(client, goal, timeout)
            if ok:
                _log.info("H12Bridge.send_arm OK")
            else:
                self.last_error = "goal rejected or timed out"
                _log.warning("H12Bridge.send_arm FAILED")
            return ok
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("H12Bridge.send_arm FAILED: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Dual-arm: DualArm action
    # ------------------------------------------------------------------

    def send_dual_arm(self,
                      right_T: Optional[np.ndarray],
                      left_T:  Optional[np.ndarray],
                      timeout: float = 15.0) -> bool:
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import DualArm

            client = ActionClient(self._node, DualArm, "dual_arm")
            if not client.wait_for_server(timeout_sec=5.0):
                self.last_error = "dual_arm server not available"
                _log.warning("H12Bridge.send_dual_arm FAILED: %s", self.last_error)
                return False

            goal = DualArm.Goal()
            if right_T is not None:
                goal.right_target = _mat_to_pose(right_T)
                goal.move_right   = True
            if left_T is not None:
                goal.left_target = _mat_to_pose(left_T)
                goal.move_left   = True

            _log.info("H12Bridge.send_dual_arm right=%s left=%s",
                      "yes" if right_T is not None else "no",
                      "yes" if left_T  is not None else "no")
            ok = _send_action(client, goal, timeout)
            if ok:
                _log.info("H12Bridge.send_dual_arm OK")
            else:
                self.last_error = "goal rejected or timed out"
                _log.warning("H12Bridge.send_dual_arm FAILED")
            return ok
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("H12Bridge.send_dual_arm FAILED: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Hand control
    # ------------------------------------------------------------------

    def send_hand_cmd(self, values: list, arm: str = "right") -> None:
        """
        Publish hand command.

        values: 6 floats in [0, 1], order [pinky, ring, middle, index, thumb_bend, thumb_yaw]
                0.0 = fully closed, 1.0 = fully open.
        arm: "right" or "left"
        """
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = [float(v) for v in values]
        pub = self._right_hand_pub if arm == "right" else self._left_hand_pub
        if pub is not None:
            pub.publish(msg)
            _log.info("H12Bridge.send_hand_cmd arm=%s vals=%s",
                      arm, [f"{v:.2f}" for v in values])

    # ------------------------------------------------------------------
    # Named configuration
    # ------------------------------------------------------------------

    def send_named_config(self, name: str, timeout: float = 15.0) -> bool:
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import NamedConfig

            client = ActionClient(self._node, NamedConfig, "named_config")
            if not client.wait_for_server(timeout_sec=5.0):
                self.last_error = "named_config server not available"
                return False

            goal = NamedConfig.Goal()
            goal.config_name = name
            ok = _send_action(client, goal, timeout)
            if not ok:
                self.last_error = "goal rejected or timed out"
            return ok
        except Exception as exc:
            self.last_error = str(exc)
            return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mat_to_pose(T: np.ndarray):
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation
    p = Pose()
    p.position.x = float(T[0, 3])
    p.position.y = float(T[1, 3])
    p.position.z = float(T[2, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p


def _send_action(client, goal, timeout: float) -> bool:
    """Send an action goal and block until done. Returns True on success."""
    done  = threading.Event()
    ok    = [False]

    def _on_result(future):
        ok[0] = True
        done.set()

    def _on_goal(future):
        gh = future.result()
        if not gh or not gh.accepted:
            done.set()
            return
        gh.get_result_async().add_done_callback(_on_result)

    client.send_goal_async(goal).add_done_callback(_on_goal)
    done.wait(timeout=timeout)
    return ok[0]
