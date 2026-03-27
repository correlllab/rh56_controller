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
        self._joint_state_msg = None
        self._ee_poses: dict[str, object] = {}
        self._frame_poses: dict[str, object] = {}
        self._frame_names_latest = []
        self._ee_pose_recv_time: dict[str, float] = {}
        self._last_pose_read_stamp: dict[str, tuple[int, int]] = {}
        self._subscriptions = []
        self._timers = []
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
            from geometry_msgs.msg import PoseStamped, PoseArray
            from std_msgs.msg import Float64MultiArray
            from custom_ros_messages.msg import StringArray

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
                self._ee_pose_recv_time["right_wrist_yaw_link"] = time.monotonic()
            def _left_cb(msg):
                if "left_wrist_yaw_link" not in self._ee_poses:
                    _log.info("H12Bridge: /left_ee_pose first msg received")
                self._ee_poses["left_wrist_yaw_link"] = msg
                self._ee_pose_recv_time["left_wrist_yaw_link"] = time.monotonic()
            self._subscriptions.append(
                self._node.create_subscription(PoseStamped, "/right_ee_pose", _right_cb, 10)
            )
            self._subscriptions.append(
                self._node.create_subscription(PoseStamped, "/left_ee_pose", _left_cb, 10)
            )

            # Fallback pose source from frame_task_server.
            # /frame_names and /frame_poses are aligned arrays.
            def _frame_names_cb(msg):
                self._frame_names_latest = list(msg.data)

            def _frame_poses_cb(msg):
                if not self._frame_names_latest:
                    return
                n = min(len(self._frame_names_latest), len(msg.poses))
                for i in range(n):
                    self._frame_poses[self._frame_names_latest[i]] = msg.poses[i]

            self._subscriptions.append(
                self._node.create_subscription(StringArray, "/frame_names", _frame_names_cb, 10)
            )
            self._subscriptions.append(
                self._node.create_subscription(PoseArray, "/frame_poses", _frame_poses_cb, 10)
            )

            # Joint states for IK seeding — optional, skip if sensor_msgs unavailable
            try:
                from sensor_msgs.msg import JointState
                self._subscriptions.append(
                    self._node.create_subscription(
                        JointState, "/joint_states",
                        lambda msg: setattr(self, "_joint_state_msg", msg), 10)
                )
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
            self._timers.append(self._node.create_timer(0.5, lambda: None))

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
        self._subscriptions.clear()
        self._timers.clear()
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

        last_stamp = self._last_pose_read_stamp.get(frame_name)
        t0 = time.monotonic()
        while (
            frame_name not in self._ee_poses
            and frame_name not in self._frame_poses
            and time.monotonic() - t0 < timeout
        ):
            time.sleep(0.05)

        # Prefer a fresh /right_ee_pose|/left_ee_pose sample when available.
        # If callbacks are active, this prevents repeatedly reusing an older
        # cached message on successive Set Pose calls.
        fresh_wait_s = min(0.25, max(0.0, timeout))
        tw = time.monotonic()
        while time.monotonic() - tw < fresh_wait_s:
            msg = self._ee_poses.get(frame_name)
            if msg is None or not hasattr(msg, "header"):
                break
            stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
            if last_stamp is None or stamp != last_stamp:
                break
            time.sleep(0.01)

        msg = self._ee_poses.get(frame_name)
        pose_msg = msg.pose if msg is not None else self._frame_poses.get(frame_name)
        if pose_msg is None:
            self.last_error = (
                f"No pose received for {frame_name} within {timeout}s "
                f"(checked /right_ee_pose|/left_ee_pose and /frame_poses)"
            )
            _log.info("H12Bridge.get_wrist_pose FAILED: %s", self.last_error)
            return None

        p = pose_msg.position
        q = pose_msg.orientation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [p.x, p.y, p.z]
        if msg is not None and hasattr(msg, "header"):
            stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
            self._last_pose_read_stamp[frame_name] = stamp
            age_s = time.monotonic() - self._ee_pose_recv_time.get(frame_name, time.monotonic())
            _log.info(
                "H12Bridge.get_wrist_pose OK [%s]: pos=[%.3f, %.3f, %.3f] quat=[%.4f, %.4f, %.4f, %.4f] stamp=%d.%09d age=%.3fs",
                frame_name, p.x, p.y, p.z, q.x, q.y, q.z, q.w, stamp[0], stamp[1], age_s,
            )
            return T
        _log.info("H12Bridge.get_wrist_pose OK [%s]: pos=[%.3f, %.3f, %.3f] quat=[%.4f, %.4f, %.4f, %.4f]",
                  frame_name, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        return T

    # ------------------------------------------------------------------
    # Single-arm: FrameTask action
    # ------------------------------------------------------------------

    def send_arm(self, frame_name: str, T: np.ndarray,
                 timeout: float = 15.0) -> bool:
        try:
            from rclpy.action import ActionClient
            try:
                from custom_ros_messages.action import FrameTask
            except Exception as exc:
                _log.warning("H12Bridge.send_arm: FrameTask unavailable (%s); falling back to dual_arm", exc)
                return self._send_arm_via_dual_arm(frame_name, T, timeout)

            client = ActionClient(self._node, FrameTask, "frame_task")
            if not client.wait_for_server(timeout_sec=5.0):
                _log.warning("H12Bridge.send_arm: frame_task unavailable; falling back to dual_arm")
                return self._send_arm_via_dual_arm(frame_name, T, timeout)

            pose = _mat_to_pose(T)
            goal = FrameTask.Goal()
            goal.frame_names   = [frame_name]
            goal.frame_targets = [pose]

            # Server-side convergence thresholds are commonly ~5 mm / 0.02 rad.
            # Warn when target delta is below threshold to explain no visible motion.
            cur_T = self.get_wrist_pose(frame_name, timeout=0.3)
            if cur_T is not None:
                dp = float(np.linalg.norm(T[:3, 3] - cur_T[:3, 3]))
                dth = _rotation_delta_rad(cur_T[:3, :3], T[:3, :3])
                if dp > 0.20 or dth > 1.2:
                    self.last_error = (
                        "target jump too large for safe single-step command "
                        f"(Δpos={dp*1000.0:.1f} mm, Δrot={np.degrees(dth):.1f} deg)"
                    )
                    _log.warning("H12Bridge.send_arm BLOCKED: %s", self.last_error)
                    return False
                if dp < 5e-3 and dth < 2e-2:
                    _log.info(
                        "H12Bridge.send_arm: tiny delta (%.1f mm, %.2f deg) below typical controller thresholds; motion may be skipped",
                        dp * 1000.0, np.degrees(dth),
                    )

            _log.info("H12Bridge.send_arm → %s pos=[%.3f, %.3f, %.3f]",
                      frame_name, T[0, 3], T[1, 3], T[2, 3])
            ok, err = _send_action(client, goal, timeout)
            if ok:
                _log.info("H12Bridge.send_arm OK")
            else:
                self.last_error = err or "goal rejected or timed out"
                _log.warning("H12Bridge.send_arm FAILED: %s", self.last_error)
            return ok
        except Exception as exc:
            self.last_error = str(exc)
            _log.warning("H12Bridge.send_arm FAILED: %s", exc)
            return False

    def _send_arm_via_dual_arm(self, frame_name: str, T: np.ndarray,
                               timeout: float = 15.0) -> bool:
        arm = frame_name.lower()
        if "right" in arm:
            return self.send_dual_arm(right_T=T, left_T=None, timeout=timeout)
        if "left" in arm:
            return self.send_dual_arm(right_T=None, left_T=T, timeout=timeout)
        self.last_error = f"cannot infer arm side from frame_name '{frame_name}'"
        _log.warning("H12Bridge.send_arm fallback FAILED: %s", self.last_error)
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
            request_right = right_T is not None
            request_left = left_T is not None

            # For DualArm schemas without move_right/move_left fields, both targets
            # must still be populated. Hold the non-requested side at its current pose.
            if right_T is None:
                right_T = self.get_wrist_pose("right_wrist_yaw_link", timeout=1.0)
            if left_T is None:
                left_T = self.get_wrist_pose("left_wrist_yaw_link", timeout=1.0)
            if right_T is None or left_T is None:
                self.last_error = "missing current wrist pose for dual_arm fallback"
                _log.warning("H12Bridge.send_dual_arm FAILED: %s", self.last_error)
                return False

            goal.right_target = _mat_to_pose(right_T)
            goal.left_target = _mat_to_pose(left_T)

            if hasattr(goal, "move_right"):
                goal.move_right = bool(request_right)
            if hasattr(goal, "move_left"):
                goal.move_left = bool(request_left)

            _log.info(
                "H12Bridge.send_dual_arm request_right=%s request_left=%s right_pos=[%.3f,%.3f,%.3f] left_pos=[%.3f,%.3f,%.3f]",
                "yes" if request_right else "no",
                "yes" if request_left else "no",
                right_T[0, 3], right_T[1, 3], right_T[2, 3],
                left_T[0, 3], left_T[1, 3], left_T[2, 3],
            )
            ok, err = _send_action(client, goal, timeout)
            if ok:
                _log.info("H12Bridge.send_dual_arm OK")
            else:
                self.last_error = err or "goal rejected or timed out"
                _log.warning("H12Bridge.send_dual_arm FAILED: %s", self.last_error)
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
            ok, err = _send_action(client, goal, timeout)
            if not ok:
                self.last_error = err or "goal rejected or timed out"
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

    R_raw = np.array(T[:3, :3], dtype=float)
    if not np.all(np.isfinite(R_raw)):
        raise ValueError("target rotation contains non-finite values")
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    q = Rotation.from_matrix(R).as_quat()  # xyzw
    qn = float(np.linalg.norm(q))
    if qn <= 0.0 or not np.isfinite(qn):
        raise ValueError("invalid quaternion norm from target rotation")
    q = q / qn
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p


def _send_action(client, goal, timeout: float) -> tuple[bool, str]:
    """Send an action goal and block until done. Returns (ok, error_message)."""
    done  = threading.Event()
    ok    = [False]
    err   = [""]

    def _on_result(future):
        try:
            action_result = future.result()
            result_msg = getattr(action_result, "result", None)
            if result_msg is not None and hasattr(result_msg, "success"):
                ok[0] = bool(result_msg.success)
                if not ok[0]:
                    err[0] = getattr(result_msg, "message", "") or "action returned success=False"
            else:
                ok[0] = True
        except Exception as exc:
            ok[0] = False
            err[0] = f"result error: {exc}"
        finally:
            done.set()

    def _on_goal(future):
        try:
            gh = future.result()
        except Exception as exc:
            err[0] = f"goal send failed: {exc}"
            done.set()
            return
        if not gh or not gh.accepted:
            err[0] = "goal rejected"
            done.set()
            return
        try:
            gh.get_result_async().add_done_callback(_on_result)
        except Exception as exc:
            err[0] = f"failed waiting for action result: {exc}"
            done.set()

    client.send_goal_async(goal).add_done_callback(_on_goal)
    completed = done.wait(timeout=timeout)
    if not completed:
        return False, f"timeout waiting for action result ({timeout:.1f}s)"
    return ok[0], err[0]


def _rotation_delta_rad(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices."""
    R = R_a.T @ R_b
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(tr))
