from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    Generates the launch description for the bimanual RH56 controller node.
    """
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='The serial port to which both RH56 hands are connected.'
    )
    hand_ids_arg = DeclareLaunchArgument(
        'hand_ids',
        default_value='1,2',
        description='Comma-separated RH56 hand IDs to enable, e.g. 1, 2, or 1,2.'
    )

    # Define the node
    rh56_driver_node = Node(
        package='rh56_controller',
        executable='rh56_driver',
        name='rh56_driver', # The node name is defined in the driver itself
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'hand_ids': LaunchConfiguration('hand_ids'),
        }]
    )

    return LaunchDescription([
        serial_port_arg,
        hand_ids_arg,
        rh56_driver_node,
    ])
