"""
grasp_viz.py — Entry point for the Inspire RH56 antipodal grasp geometry planner.

Usage:
    uv run python -m rh56_controller.grasp_viz
    uv run python -m rh56_controller.grasp_viz --robot
    uv run python -m rh56_controller.grasp_viz --real-robot --ur5-ip 192.168.0.4 \\
                                                --port /dev/ttyUSB0

Options:
    --xml PATH        Path to inspire_right.xml (default: bundled)
    --rebuild         Force rebuild of FK cache
    --port DEV        Serial port for real hand   (e.g. /dev/ttyUSB0)
    --robot           Enable UR5+hand robot viewer / IK planning
    --h12             Enable H1-2+hand robot viewer with PINK IK
    --bimanual        Enable bimanual mode (both arms; requires --h12)
    --send-real       Start with Send-to-Real enabled (requires --port)
    --no-mink         Skip mink IK comparison planner (faster startup)
    --real-robot      Enable real UR5 arm control (requires --ur5-ip)
    --ur5-ip IP       UR5 robot IP address        (e.g. 192.168.0.4)
    --ur5-speed M/S   UR5 linear speed in m/s     (default: 0.10)
    --real-h12        Connect to real H1-2 via ROS2 frame_task_server
    --h12-ros         Enable ROS2 comms for H1-2 sim (no real hardware)

Architecture:
    This file is a thin CLI wrapper.  All state and geometry live in
    GraspVizCore (grasp_viz_core.py); the tkinter UI layer lives in
    GraspVizUI (grasp_viz_ui.py); MuJoCo subprocess workers live in
    grasp_viz_workers.py.
"""

import argparse
import logging
import multiprocessing
import sys

from .grasp_geometry import _DEFAULT_XML
from .grasp_viz_ui import GraspVizUI


def main():
    parser = argparse.ArgumentParser(
        description="Inspire RH56 Grasp Geometry Visualizer"
    )
    parser.add_argument("--xml", default=_DEFAULT_XML,
                        help="Path to inspire_right.xml")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of FK cache")
    parser.add_argument("--port", default=None,
                        help="Serial port for real hand (e.g. /dev/ttyUSB0)")
    parser.add_argument("--robot", action="store_true",
                        help="Enable UR5+hand robot viewer buttons")
    parser.add_argument("--h12", action="store_true",
                        help="Enable H1-2+hand robot viewer with PINK IK")
    parser.add_argument("--bimanual", action="store_true",
                        help="Enable bimanual mode in H1-2 viewer (requires --h12)")
    parser.add_argument("--real-h12", action="store_true",
                        help="Connect to real H1-2 via ROS2 frame_task_server (requires --h12)")
    parser.add_argument("--h12-ros", action="store_true",
                        help="Enable ROS2 comms for H1-2 sim (without real robot)")
    parser.add_argument("--send-real", action="store_true",
                        help="Start with Send-to-Real enabled (requires --port)")
    parser.add_argument("--no-mink", action="store_true",
                        help="Disable mink IK comparison planner (faster startup)")
    parser.add_argument("--real-robot", action="store_true",
                        help="Enable real UR5 arm control panel (requires --ur5-ip)")
    parser.add_argument("--ur5-ip", default=None,
                        help="UR5 robot IP address (e.g. 192.168.0.4)")
    parser.add_argument("--ur5-speed", type=float, default=0.10,
                        help="UR5 arm linear speed in m/s (default 0.10)")
    parser.add_argument("--ros-sync", action="store_true",
                        help="Enable ROS2 state bridge for grasp_viz")
    parser.add_argument("--ros-publish-hz", type=float, default=20.0,
                        help="ROS publish rate in Hz (default 20)")
    parser.add_argument("--ros-send-hand-cmd", action="store_true",
                        help="Publish planned right-hand command to hands/cmd")
    parser.add_argument("--rerun", action="store_true",
                        help="Enable rerun telemetry stream from grasp_viz")
    parser.add_argument("--log-file", default=None,
                        help="Write Python logs to file (default: stderr only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging (default: INFO)")
    # ros2 launch injects extra CLI tokens (e.g. --ros-args ...).
    # Ignore unknown args so this non-rclpy app still works under launch.
    args, _unknown = parser.parse_known_args()

    _handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        _handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=_handlers,
    )

    # Mutual exclusion: cannot connect to both real UR5 and real H1-2 simultaneously
    if args.real_robot and args.real_h12:
        parser.error("--real-robot and --real-h12 are mutually exclusive. "
                     "Connect to only one real robot at a time.")

    viz = GraspVizUI(
        xml_path=args.xml,
        rebuild=args.rebuild,
        port=args.port,
        robot_mode=args.robot,
        h12_mode=args.h12,
        bimanual_mode=args.bimanual,
        send_real=args.send_real,
        mink_viz=not args.no_mink,
        real_robot=args.real_robot,
        ur5_ip=args.ur5_ip,
        ur5_speed=args.ur5_speed,
        real_h12=args.real_h12,
        h12_ros=args.h12_ros,
        ros_sync=args.ros_sync,
        ros_publish_hz=args.ros_publish_hz,
        ros_send_hand_cmd=args.ros_send_hand_cmd,
        rerun_viz=args.rerun,
    )
    viz.run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
