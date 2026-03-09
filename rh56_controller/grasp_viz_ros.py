"""ROS2 bridge for GraspViz state, commands, and optional rerun telemetry.

This module runs inside the same process as the tkinter UI and mirrors planner
state to ROS topics at a fixed rate. It is intentionally lightweight and keeps
all heavy geometry in GraspVizCore.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

import numpy as np


class GraspVizRosBridge:
    """Publish GraspViz state to ROS2 and optionally mirror hand commands."""

    def __init__(
        self,
        core,
        publish_hz: float = 20.0,
        send_hand_cmd: bool = False,
        rerun_enabled: bool = False,
    ) -> None:
        self._core = core
        self._publish_hz = max(1.0, float(publish_hz))
        self._send_hand_cmd = bool(send_hand_cmd)
        self._rerun_enabled = bool(rerun_enabled)

        self._rclpy = None
        self._node = None
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

        self._pub_status = None
        self._pub_summary = None
        self._pub_joint = None
        self._pub_pose = None
        self._pub_hand_cmd = None
        self._last_left_q = [0.0] * 6

        self._rr = None

    def start(self) -> bool:
        """Start ROS2 node + publisher timer in a background executor thread."""
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node

            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64
            from std_msgs.msg import String

            self._rclpy = rclpy

            if not rclpy.ok():
                rclpy.init(args=None)

            class _BridgeNode(Node):
                pass

            self._node = _BridgeNode("grasp_viz_bridge")
            self._pub_status = self._node.create_publisher(String, "grasp_viz/status", 10)
            self._pub_summary = self._node.create_publisher(String, "grasp_viz/summary_json", 10)
            self._pub_joint = self._node.create_publisher(JointState, "grasp_viz/hand_joint_states", 10)
            self._pub_pose = self._node.create_publisher(PoseStamped, "grasp_viz/ur5_target_pose", 10)

            self._node.create_subscription(String, "grasp_viz/set_mode", self._on_set_mode, 10)
            self._node.create_subscription(Float64, "grasp_viz/set_width_mm", self._on_set_width_mm, 10)
            self._node.create_subscription(
                Float64,
                "grasp_viz/set_target_width_mm",
                self._on_set_target_width_mm,
                10,
            )
            self._node.create_subscription(
                PoseStamped,
                "grasp_viz/set_target_pose",
                self._on_set_target_pose,
                10,
            )

            if self._send_hand_cmd:
                try:
                    from custom_ros_messages.msg import MotorCmd, MotorCmds, MotorStates

                    self._pub_hand_cmd = self._node.create_publisher(MotorCmds, "hands/cmd", 10)
                    self._node.create_subscription(MotorStates, "hands/state", self._on_hand_state, 10)
                    self._motor_cmd_cls = MotorCmd
                    self._motor_cmds_cls = MotorCmds
                except Exception as exc:
                    self._node.get_logger().warning(
                        f"custom_ros_messages unavailable; hand command publish disabled: {exc}"
                    )
                    self._pub_hand_cmd = None

            period_s = 1.0 / self._publish_hz
            self._node.create_timer(period_s, self._on_publish_timer)

            self._executor = MultiThreadedExecutor(num_threads=2)
            self._executor.add_node(self._node)
            self._running = True
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True, name="grasp-viz-ros")
            self._spin_thread.start()

            self._setup_rerun()
            self._publish_status("ROS bridge started")
            return True
        except Exception:
            self.stop()
            return False

    def stop(self) -> None:
        """Stop timer/executor and close node cleanly."""
        self._running = False
        try:
            if self._executor is not None:
                self._executor.shutdown(timeout_sec=0.5)
        except Exception:
            pass
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        self._executor = None
        self._node = None
        self._pub_status = None
        self._pub_summary = None
        self._pub_joint = None
        self._pub_pose = None
        self._pub_hand_cmd = None

    def _publish_status(self, text: str) -> None:
        if self._pub_status is None:
            return
        from std_msgs.msg import String

        msg = String()
        msg.data = text
        self._pub_status.publish(msg)

    def _on_set_mode(self, msg) -> None:
        self._core.ros_set_mode(str(msg.data).strip())

    def _on_set_width_mm(self, msg) -> None:
        self._core.ros_set_width_m(float(msg.data) / 1000.0)

    def _on_set_target_width_mm(self, msg) -> None:
        self._core.ros_set_target_width_m(float(msg.data) / 1000.0)

    def _on_set_target_pose(self, msg) -> None:
        self._core.ros_set_target_pose(
            x_m=float(msg.pose.position.x),
            y_m=float(msg.pose.position.y),
            z_m=float(msg.pose.position.z),
        )

    def _on_hand_state(self, msg) -> None:
        try:
            qs = [float(s.q) for s in msg.motor_states]
            if len(qs) >= 12:
                self._last_left_q = qs[6:12]
        except Exception:
            return


    def _on_publish_timer(self) -> None:
        if not self._running:
            return
        snap = self._core.get_ros_snapshot()
        if snap is None:
            return

        self._publish_summary(snap)
        self._publish_joint_state(snap)
        self._publish_target_pose(snap)
        self._publish_hand_cmd(snap)
        self._publish_rerun(snap)

    def _publish_summary(self, snap: Dict[str, Any]) -> None:
        if self._pub_summary is None:
            return
        from std_msgs.msg import String

        msg = String()
        msg.data = json.dumps(snap, separators=(",", ":"))
        self._pub_summary.publish(msg)

    def _publish_joint_state(self, snap: Dict[str, Any]) -> None:
        if self._pub_joint is None:
            return
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = [
            "right_pinky",
            "right_ring",
            "right_middle",
            "right_index",
            "right_thumb_bend",
            "right_thumb_rotation",
        ]
        msg.position = [
            float(snap["ctrl_rad"][0]),
            float(snap["ctrl_rad"][1]),
            float(snap["ctrl_rad"][2]),
            float(snap["ctrl_rad"][3]),
            float(snap["ctrl_rad"][4]),
            float(snap["ctrl_rad"][5]),
        ]
        self._pub_joint.publish(msg)

    def _publish_target_pose(self, snap: Dict[str, Any]) -> None:
        if self._pub_pose is None:
            return
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "ur5_base"
        msg.pose.position.x = float(snap["target_pose"]["x"])
        msg.pose.position.y = float(snap["target_pose"]["y"])
        msg.pose.position.z = float(snap["target_pose"]["z"])
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        self._pub_pose.publish(msg)

    def _publish_hand_cmd(self, snap: Dict[str, Any]) -> None:
        if self._pub_hand_cmd is None:
            return
        cmds = self._motor_cmds_cls()
        cmds.motor_commands = []

        right = [float(v) for v in snap["ctrl_rad"]]
        if len(right) != 6:
            return
        both = right + self._last_left_q
        for q in both:
            c = self._motor_cmd_cls()
            c.mode = 0
            c.q = float(q)
            c.dq = 0.0
            c.tau = 0.0
            cmds.motor_commands.append(c)
        self._pub_hand_cmd.publish(cmds)

    def _setup_rerun(self) -> None:
        if not self._rerun_enabled:
            return
        try:
            import rerun as rr

            self._rr = rr
            rr.init("grasp_viz_ros", spawn=True)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
        except Exception:
            self._rr = None

    def _publish_rerun(self, snap: Dict[str, Any]) -> None:
        if self._rr is None:
            return
        rr = self._rr
        try:
            rr.set_time_seconds("wall_clock", time.time())

            tp = snap["target_pose"]
            rr.log(
                "grasp_viz/target",
                rr.Points3D([[tp["x"], tp["y"], tp["z"]]], colors=[[255, 180, 0]], radii=[0.008]),
            )

            tip_pts = []
            for key in ("thumb", "index", "middle", "ring", "pinky"):
                p = snap["tips_world_m"].get(key)
                if p is not None:
                    tip_pts.append(p)
            if tip_pts:
                rr.log("grasp_viz/tips", rr.Points3D(tip_pts, colors=[[80, 220, 255]], radii=[0.006]))
        except Exception:
            return