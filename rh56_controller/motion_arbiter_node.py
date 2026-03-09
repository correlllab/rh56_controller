"""Simple ROS2 motion arbitration lock for UR5 command ownership.

This node provides two services:
- /motion_arbiter/acquire (Trigger): success only when lock is free
- /motion_arbiter/release (Trigger): always releases lock

Use this to prevent concurrent UR5 commanding by independent ROS nodes.
"""

import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class MotionArbiterNode(Node):
    def __init__(self):
        super().__init__("motion_arbiter")
        self._lock = threading.Lock()
        self._held = False

        self._acquire_srv = self.create_service(
            Trigger, "motion_arbiter/acquire", self._on_acquire
        )
        self._release_srv = self.create_service(
            Trigger, "motion_arbiter/release", self._on_release
        )

        self.get_logger().info("MotionArbiterNode ready")

    def _on_acquire(self, _req, res):
        with self._lock:
            if self._held:
                res.success = False
                res.message = "busy"
                return res
            self._held = True
        res.success = True
        res.message = "acquired"
        return res

    def _on_release(self, _req, res):
        with self._lock:
            self._held = False
        res.success = True
        res.message = "released"
        return res


def main(args=None):
    rclpy.init(args=args)
    node = MotionArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
