from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
import glob
import os


def _extra_env_from_venv():
    venv = os.environ.get("VIRTUAL_ENV", "")
    py_paths = []
    if venv:
        py_paths.extend(glob.glob(os.path.join(venv, "lib", "python*", "site-packages")))
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        py_paths.append(existing)
    if not py_paths:
        return {}
    return {"PYTHONPATH": os.pathsep.join(py_paths)}


def _build_grasp_viz_node(context):
    serial_port = LaunchConfiguration("serial_port").perform(context).strip()
    ur5_ip = LaunchConfiguration("ur5_ip").perform(context).strip()

    args = [
        "--robot",
        "--ros-sync",
        "--rerun",
        "--no-mink",
        "--ros-publish-hz",
        LaunchConfiguration("ros_publish_hz"),
    ]

    if serial_port:
        args.extend(["--port", serial_port])
    if ur5_ip:
        args.extend(["--real-robot", "--ur5-ip", ur5_ip])

    python_exe = "/usr/bin/python3"
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv:
        candidate = os.path.join(venv, "bin", "python")
        if os.path.exists(candidate):
            python_exe = candidate

    return [
        ExecuteProcess(
            cmd=[python_exe, "-m", "rh56_controller.grasp_viz", *args],
            name="grasp_viz",
            output="screen",
            additional_env=_extra_env_from_venv(),
        )
    ]


def generate_launch_description():
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="",
        description="Optional serial port for RH56 hand (e.g. /dev/ttyUSB0)",
    )
    ur5_ip_arg = DeclareLaunchArgument(
        "ur5_ip",
        default_value="",
        description="Optional UR5 IP for real-robot mode",
    )
    ros_publish_hz_arg = DeclareLaunchArgument(
        "ros_publish_hz",
        default_value="20.0",
        description="ROS bridge publish rate in Hz",
    )

    return LaunchDescription([
        serial_port_arg,
        ur5_ip_arg,
        ros_publish_hz_arg,
        OpaqueFunction(function=_build_grasp_viz_node),
    ])
