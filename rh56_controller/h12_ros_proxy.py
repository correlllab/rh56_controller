#!/usr/bin/env python3
"""
h12_ros_proxy.py — ROS2 subprocess proxy for H12Bridge.

Run under Python 3.10 with ROS2 Humble sourced.  Reads newline-delimited
JSON commands from stdin, executes ROS2 action calls, writes JSON results
to stdout.

Command format:
  {"id": <int>, "cmd": "<name>", ...args...}

Result format:
  {"id": <int>, "ok": <bool>, "error": "<str>"}

Supported commands:
  connect     {}
  disconnect  {}
  send_arm    {"frame_name": str, "T": [[4x4 floats]], "timeout": float}
  send_dual_arm {"right_T": [[4x4]]|null, "left_T": [[4x4]]|null, "timeout": float}
  send_named_config {"name": str, "timeout": float}
"""

import json
import sys
import threading
import time


def _reply(msg_id, ok, error=""):
    line = json.dumps({"id": msg_id, "ok": ok, "error": error})
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _matrix_to_pose(T):
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation
    p = Pose()
    p.position.x = float(T[0][3])
    p.position.y = float(T[1][3])
    p.position.z = float(T[2][3])
    import numpy as np
    q = Rotation.from_matrix(np.array(T)[:3, :3]).as_quat()  # xyzw
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p


def _wait_future(future, executor, timeout):
    t0 = time.monotonic()
    while not future.done():
        executor.spin_once(timeout_sec=0.01)
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("action timed out")


class Proxy:
    def __init__(self):
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._connected = False
        self._tf_buffer = None
        self._tf_listener = None
        self._subscriptions = []

    def connect(self, msg_id):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor
            from geometry_msgs.msg import PoseStamped, PoseArray
            from custom_ros_messages.msg import StringArray

            if not rclpy.ok():
                rclpy.init()

            self._node = Node("grasp_viz_h12_proxy")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            # Persistent subscriptions — populated by the spin thread
            self._ee_poses: dict[str, object] = {}
            self._subscriptions.append(self._node.create_subscription(
                PoseStamped, "/right_ee_pose",
                lambda msg: self._ee_poses.__setitem__("right_wrist_yaw_link", msg), 1))
            self._subscriptions.append(self._node.create_subscription(
                PoseStamped, "/left_ee_pose",
                lambda msg: self._ee_poses.__setitem__("left_wrist_yaw_link", msg), 1))

            # Fallback source from frame_task_server publisher.
            self._frame_poses: dict[str, object] = {}
            self._frame_names_latest = []
            self._subscriptions.append(self._node.create_subscription(
                StringArray, "/frame_names",
                lambda msg: setattr(self, "_frame_names_latest", list(msg.data)), 1))

            def _frame_poses_cb(msg):
                names = self._frame_names_latest
                if not names:
                    return
                n = min(len(names), len(msg.poses))
                for i in range(n):
                    self._frame_poses[names[i]] = msg.poses[i]

            self._subscriptions.append(
                self._node.create_subscription(PoseArray, "/frame_poses", _frame_poses_cb, 1)
            )

            self._connected = True
            self._spin_thread = threading.Thread(
                target=self._spin_loop, daemon=True, name="h12-proxy-spin")
            self._spin_thread.start()

            _reply(msg_id, True)
        except Exception as exc:
            _reply(msg_id, False, str(exc))

    def _spin_loop(self):
        try:
            while self._connected and self._executor is not None:
                self._executor.spin_once(timeout_sec=0.05)
        except Exception:
            pass

    def disconnect(self, msg_id):
        self._connected = False
        self._subscriptions.clear()
        # Let the spin thread exit before touching the executor / rclpy
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        _reply(msg_id, True)

    def send_arm(self, msg_id, frame_name, T, timeout):
        try:
            from rclpy.action import ActionClient
            try:
                from custom_ros_messages.action import FrameTask
            except Exception as exc:
                self._send_arm_via_dual_arm(msg_id, frame_name, T, timeout,
                                            reason=f"FrameTask unavailable: {exc}")
                return

            client = ActionClient(self._node, FrameTask, "frame_task")
            if not client.wait_for_server(timeout_sec=5.0):
                self._send_arm_via_dual_arm(msg_id, frame_name, T, timeout,
                                            reason="frame_task server not available")
                return

            goal = FrameTask.Goal()
            goal.frame_names   = [frame_name]
            goal.frame_targets = [_matrix_to_pose(T)]

            future = client.send_goal_async(goal)
            _wait_future(future, self._executor, timeout)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                _reply(msg_id, False, "goal rejected")
                return

            result_future = goal_handle.get_result_async()
            _wait_future(result_future, self._executor, timeout)
            _reply(msg_id, True)
        except Exception as exc:
            _reply(msg_id, False, str(exc))

    def _send_arm_via_dual_arm(self, msg_id, frame_name, T, timeout, reason=""):
        side = str(frame_name).lower()
        if "right" in side:
            right_T, left_T = T, None
        elif "left" in side:
            right_T, left_T = None, T
        else:
            suffix = f" ({reason})" if reason else ""
            _reply(msg_id, False, f"cannot infer arm from frame_name '{frame_name}'{suffix}")
            return

        if reason:
            sys.stderr.write(f"[proxy] send_arm fallback to dual_arm: {reason}\n")
            sys.stderr.flush()
        self.send_dual_arm(msg_id, right_T=right_T, left_T=left_T, timeout=timeout)

    def send_dual_arm(self, msg_id, right_T, left_T, timeout):
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import DualArm

            client = ActionClient(self._node, DualArm, "dual_arm")
            if not client.wait_for_server(timeout_sec=5.0):
                _reply(msg_id, False, "dual_arm server not available")
                return

            goal = DualArm.Goal()
            if right_T is not None:
                goal.right_target = _matrix_to_pose(right_T)
                goal.move_right   = True
            if left_T is not None:
                goal.left_target = _matrix_to_pose(left_T)
                goal.move_left   = True

            future = client.send_goal_async(goal)
            _wait_future(future, self._executor, timeout)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                _reply(msg_id, False, "goal rejected")
                return

            result_future = goal_handle.get_result_async()
            _wait_future(result_future, self._executor, timeout)
            _reply(msg_id, True)
        except Exception as exc:
            _reply(msg_id, False, str(exc))

    def get_wrist_pose(self, msg_id, frame_name, base_frame, timeout):
        """Read wrist pose from cached /right_ee_pose or /left_ee_pose topic."""
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation

            t0 = time.monotonic()
            while (
                frame_name not in self._ee_poses
                and frame_name not in self._frame_poses
                and time.monotonic() - t0 < timeout
            ):
                time.sleep(0.05)

            msg = self._ee_poses.get(frame_name)
            pose_msg = msg.pose if msg is not None else self._frame_poses.get(frame_name)
            if pose_msg is None:
                _reply(msg_id, False,
                       f"No pose for {frame_name} within {timeout}s (checked /right_ee_pose|/left_ee_pose and /frame_poses)")
                return

            p = pose_msg.position
            q = pose_msg.orientation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = [p.x, p.y, p.z]
            sys.stderr.write(f"[proxy] pose ok: {frame_name} pos=[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]\n")
            sys.stderr.flush()
            line = json.dumps({"id": msg_id, "ok": True, "error": "", "T": T.tolist()})
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception as exc:
            _reply(msg_id, False, str(exc))

    def send_named_config(self, msg_id, name, timeout):
        try:
            from rclpy.action import ActionClient
            from custom_ros_messages.action import NamedConfig

            client = ActionClient(self._node, NamedConfig, "named_config")
            if not client.wait_for_server(timeout_sec=5.0):
                _reply(msg_id, False, "named_config server not available")
                return

            goal = NamedConfig.Goal()
            goal.config_name = name

            future = client.send_goal_async(goal)
            _wait_future(future, self._executor, timeout)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                _reply(msg_id, False, f"named_config '{name}' rejected")
                return

            result_future = goal_handle.get_result_async()
            _wait_future(result_future, self._executor, timeout)
            _reply(msg_id, True)
        except Exception as exc:
            _reply(msg_id, False, str(exc))


def main():
    proxy = Proxy()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"[proxy] bad JSON: {exc}\n")
            sys.stderr.flush()
            continue

        msg_id = msg.get("id", -1)
        cmd    = msg.get("cmd", "")

        if cmd == "connect":
            proxy.connect(msg_id)
        elif cmd == "disconnect":
            proxy.disconnect(msg_id)
            break
        elif cmd == "send_arm":
            proxy.send_arm(msg_id,
                           msg["frame_name"], msg["T"], msg.get("timeout", 15.0))
        elif cmd == "send_dual_arm":
            proxy.send_dual_arm(msg_id,
                                msg.get("right_T"), msg.get("left_T"),
                                msg.get("timeout", 15.0))
        elif cmd == "send_named_config":
            proxy.send_named_config(msg_id, msg["name"], msg.get("timeout", 15.0))
        elif cmd == "get_wrist_pose":
            proxy.get_wrist_pose(msg_id,
                                 msg["frame_name"],
                                 msg.get("base_frame", "world"),
                                 msg.get("timeout", 5.0))
        else:
            _reply(msg_id, False, f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
