"""
h12_bridge.py — ROS2 client bridge for real H1-2 arm control.

Wraps the h12_ros2_controller ROS2 action servers:
  - /frame_task   (FrameTask action)  — single arm: right or left
  - /dual_arm     (DualArm action)    — bimanual (both arms simultaneously)
  - /named_config (NamedConfig action) — go to a named configuration

Usage (from grasp_viz_core):
    bridge = H12Bridge(bimanual=False)
    bridge.connect()                   # init ROS2 node; returns True on success
    bridge.send_arm("right_wrist_yaw_link", T_4x4)
    bridge.send_named_config("home")
    bridge.disconnect()

The bridge starts a minimal ROS2 node in a background thread.  rclpy is only
imported lazily so that grasp_viz still works without a ROS2 environment.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

_log = logging.getLogger(__name__)


def _matrix_to_pose(T: np.ndarray):
    """Convert 4×4 homogeneous matrix to geometry_msgs/Pose."""
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


class H12Bridge:
    """
    Thin ROS2 action client for H1-2 arm control.

    Designed to work alongside grasp_viz without blocking the Tkinter event loop.
    All action calls are synchronous within their own thread — callers should use
    threading.Thread to avoid blocking the UI.
    """

    def __init__(self, bimanual: bool = False) -> None:
        self._bimanual = bimanual
        self._node     = None
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._connected = False
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise rclpy and start a background spin thread.  Returns True on success."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor

            if not rclpy.ok():
                rclpy.init()

            self._node     = Node("grasp_viz_h12_bridge")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            self._spin_thread = threading.Thread(
                target=self._spin_loop, daemon=True, name="h12-bridge-spin")
            self._spin_thread.start()

            self._connected = True
            _log.info("H12Bridge: ROS2 node started.")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("H12Bridge.connect failed: %s", exc)
            return False

    def disconnect(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
        self._connected = False

    def _spin_loop(self):
        try:
            while self._connected and self._executor is not None:
                self._executor.spin_once(timeout_sec=0.05)
        except Exception as exc:
            _log.debug("H12Bridge spin loop exited: %s", exc)

    # ------------------------------------------------------------------
    # Single-arm: FrameTask action
    # ------------------------------------------------------------------

    def send_arm(self, frame_name: str, T: np.ndarray,
                 timeout: float = 15.0) -> bool:
        """
        Move a single arm so that `frame_name` reaches pose `T` (4×4 matrix,
        world frame).  Blocks until the action completes or times out.
        Returns True on success.

        frame_name: "right_wrist_yaw_link" or "left_wrist_yaw_link"
        """
        if not self._connected or self._node is None:
            _log.warning("H12Bridge not connected.")
            return False
        try:
            import rclpy
            from rclpy.action import ActionClient
            from custom_ros_messages.action import FrameTask

            client = ActionClient(self._node, FrameTask, "frame_task")
            if not client.wait_for_server(timeout_sec=5.0):
                _log.warning("H12Bridge: frame_task server not available.")
                return False

            goal = FrameTask.Goal()
            goal.frame_names   = [frame_name]
            goal.frame_targets = [_matrix_to_pose(T)]

            future = client.send_goal_async(goal)
            _wait_future(future, timeout)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                _log.warning("H12Bridge: frame_task goal rejected.")
                return False

            result_future = goal_handle.get_result_async()
            _wait_future(result_future, timeout)
            result = result_future.result()
            _log.info("H12Bridge: frame_task done (status %s).",
                      getattr(result, "status", "?"))
            return True
        except Exception as exc:
            _log.warning("H12Bridge.send_arm error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Dual-arm: DualArm action
    # ------------------------------------------------------------------

    def send_dual_arm(self,
                      right_T: Optional[np.ndarray],
                      left_T:  Optional[np.ndarray],
                      timeout: float = 15.0) -> bool:
        """
        Move both arms simultaneously via the /dual_arm action.
        Pass None to leave an arm at its current pose.
        """
        if not self._connected or self._node is None:
            _log.warning("H12Bridge not connected.")
            return False
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import DualArm

            client = ActionClient(self._node, DualArm, "dual_arm")
            if not client.wait_for_server(timeout_sec=5.0):
                _log.warning("H12Bridge: dual_arm server not available.")
                return False

            goal = DualArm.Goal()
            if right_T is not None:
                goal.right_target = _matrix_to_pose(right_T)
                goal.move_right   = True
            if left_T is not None:
                goal.left_target = _matrix_to_pose(left_T)
                goal.move_left   = True

            future = client.send_goal_async(goal)
            _wait_future(future, timeout)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                _log.warning("H12Bridge: dual_arm goal rejected.")
                return False

            result_future = goal_handle.get_result_async()
            _wait_future(result_future, timeout)
            _log.info("H12Bridge: dual_arm done.")
            return True
        except Exception as exc:
            _log.warning("H12Bridge.send_dual_arm error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Named configuration (e.g. "home", "rest")
    # ------------------------------------------------------------------

    def send_named_config(self, name: str, timeout: float = 15.0) -> bool:
        """Send the robot to a named configuration defined in h12_ros2_controller."""
        if not self._connected or self._node is None:
            _log.warning("H12Bridge not connected.")
            return False
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import NamedConfig

            client = ActionClient(self._node, NamedConfig, "named_config")
            if not client.wait_for_server(timeout_sec=5.0):
                _log.warning("H12Bridge: named_config server not available.")
                return False

            goal = NamedConfig.Goal()
            goal.config_name = name

            future = client.send_goal_async(goal)
            _wait_future(future, timeout)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                _log.warning("H12Bridge: named_config '%s' rejected.", name)
                return False

            result_future = goal_handle.get_result_async()
            _wait_future(result_future, timeout)
            _log.info("H12Bridge: named_config '%s' done.", name)
            return True
        except Exception as exc:
            _log.warning("H12Bridge.send_named_config error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected


def _wait_future(future, timeout: float) -> None:
    """Spin-wait for a rclpy Future with a hard timeout."""
    import rclpy
    t0 = time.monotonic()
    while not future.done():
        rclpy.spin_once(future._node if hasattr(future, "_node") else None,
                        timeout_sec=0.01)
        if time.monotonic() - t0 > timeout:
            break
