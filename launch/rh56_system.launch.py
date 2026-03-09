from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

try:
    import custom_ros_messages.msg  # noqa: F401

    HAS_CUSTOM_ROS_MESSAGES = True
except Exception:
    HAS_CUSTOM_ROS_MESSAGES = False


def generate_launch_description():
    ur5_ip_arg = DeclareLaunchArgument(
        "ur5_ip", default_value="192.168.0.4", description="UR5 robot IP"
    )
    serial_port_arg = DeclareLaunchArgument(
        "serial_port", default_value="/dev/ttyUSB0", description="RH56 serial port"
    )
    enable_hand_driver_arg = DeclareLaunchArgument(
        "enable_hand_driver", default_value="false", description="Launch RH56 hand ROS driver"
    )
    enable_force_control_arg = DeclareLaunchArgument(
        "enable_force_control", default_value="false", description="Launch force control ROS bridge"
    )

    motion_arbiter = Node(
        package="rh56_controller",
        executable="motion_arbiter_node",
        name="motion_arbiter",
        output="screen",
    )

    grasp_viz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("rh56_controller"),
                "launch",
                "grasp_viz_ros.launch.py",
            ])
        ),
        launch_arguments={
            "ur5_ip": LaunchConfiguration("ur5_ip"),
            "serial_port": LaunchConfiguration("serial_port"),
        }.items(),
    )

    hand_driver = Node(
        package="rh56_controller",
        executable="rh56_driver",
        name="rh56_driver",
        output="screen",
        parameters=[{"serial_port": LaunchConfiguration("serial_port")}],
        condition=IfCondition(LaunchConfiguration("enable_hand_driver")),
    )

    missing_hand_driver_deps_warning = LogInfo(
        msg=(
            "enable_hand_driver:=true requested, but custom_ros_messages is not "
            "available in the current environment. Skipping rh56_driver; grasp_viz "
            "will continue with direct hand serial control."
        ),
        condition=IfCondition(LaunchConfiguration("enable_hand_driver")),
    )

    force_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("magpie_force_control_ros"),
                "launch",
                "force_control.launch.py",
            ])
        ),
        launch_arguments={
            "robot_ip": LaunchConfiguration("ur5_ip"),
            "autostart": "false",
            "use_motion_arbiter": "true",
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_force_control")),
    )

    actions = [
        ur5_ip_arg,
        serial_port_arg,
        enable_hand_driver_arg,
        enable_force_control_arg,
        motion_arbiter,
        grasp_viz_launch,
        force_control_launch,
    ]

    if HAS_CUSTOM_ROS_MESSAGES:
        actions.append(hand_driver)
    else:
        actions.append(missing_hand_driver_deps_warning)

    return LaunchDescription(actions)
