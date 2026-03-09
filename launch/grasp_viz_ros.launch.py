from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

    grasp_viz_node = Node(
        package="rh56_controller",
        executable="grasp_viz",
        name="grasp_viz",
        output="screen",
        arguments=[
            "--robot",
            "--ros-sync",
            "--rerun",
            "--ros-publish-hz",
            LaunchConfiguration("ros_publish_hz"),
            "--port",
            LaunchConfiguration("serial_port"),
            "--ur5-ip",
            LaunchConfiguration("ur5_ip"),
        ],
    )

    return LaunchDescription([
        serial_port_arg,
        ur5_ip_arg,
        ros_publish_hz_arg,
        grasp_viz_node,
    ])
