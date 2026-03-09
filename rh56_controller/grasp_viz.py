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
    --send-real       Start with Send-to-Real enabled (requires --port)
    --no-mink         Skip mink IK comparison planner (faster startup)
    --real-robot      Enable real UR5 arm control (requires --ur5-ip)
    --ur5-ip IP       UR5 robot IP address        (e.g. 192.168.0.4)
    --ur5-speed M/S   UR5 linear speed in m/s     (default: 0.10)

Architecture:
    This file is a thin CLI wrapper.  All state and geometry live in
    GraspVizCore (grasp_viz_core.py); the tkinter UI layer lives in
    GraspVizUI (grasp_viz_ui.py); MuJoCo subprocess workers live in
    grasp_viz_workers.py.
"""

import argparse
import multiprocessing

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
    args = parser.parse_args()

    viz = GraspVizUI(
        xml_path=args.xml,
        rebuild=args.rebuild,
        port=args.port,
        robot_mode=args.robot,
        send_real=args.send_real,
        mink_viz=not args.no_mink,
        real_robot=args.real_robot,
        ur5_ip=args.ur5_ip,
        ur5_speed=args.ur5_speed,
        ros_sync=args.ros_sync,
        ros_publish_hz=args.ros_publish_hz,
        ros_send_hand_cmd=args.ros_send_hand_cmd,
        rerun_viz=args.rerun,
    )
    viz.run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
