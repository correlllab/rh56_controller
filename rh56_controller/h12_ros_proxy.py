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

    def connect(self, msg_id):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor

            if not rclpy.ok():
                rclpy.init()

            self._node = Node("grasp_viz_h12_proxy")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

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
            from custom_ros_messages.action import FrameTask

            client = ActionClient(self._node, FrameTask, "frame_task")
            if not client.wait_for_server(timeout_sec=5.0):
                _reply(msg_id, False, "frame_task server not available")
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
        try:
            import rclpy
            from tf2_ros import Buffer, TransformListener
            from geometry_msgs.msg import TransformStamped
            import numpy as np
            from scipy.spatial.transform import Rotation

            tf_buffer   = Buffer()
            tf_listener = TransformListener(tf_buffer, self._node)  # noqa: F841

            # Poll until transform is available
            t0 = time.monotonic()
            ts: TransformStamped = None
            while time.monotonic() - t0 < timeout:
                try:
                    ts = tf_buffer.lookup_transform(
                        base_frame, frame_name,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5),
                    )
                    break
                except Exception:
                    self._executor.spin_once(timeout_sec=0.1)

            if ts is None:
                _reply(msg_id, False, f"TF lookup {base_frame}→{frame_name} timed out")
                return

            t = ts.transform.translation
            q = ts.transform.rotation  # xyzw convention in geometry_msgs
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = [t.x, t.y, t.z]
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
