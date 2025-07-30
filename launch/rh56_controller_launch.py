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

    # Define the node
    rh56_driver_node = Node(
        package='rh56_controller',
        executable='rh56_driver',
        name='rh56_driver', # The node name is defined in the driver itself
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
        }]
    )

    return LaunchDescription([
        serial_port_arg,
        rh56_driver_node,
    ])