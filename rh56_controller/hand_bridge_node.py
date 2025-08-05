import rclpy
from rclpy.node import Node

from custom_ros_messages.msg import MotorCmd, MotorCmds, MotorState, MotorStates
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_ as DDS_MotorStates, MotorCmds_ as DDS_MotorCmds, MotorCmd_ as DDS_MotorCmd
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorState_ as DDS_MotorState
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_ as MotorCmd_default

NUM_HAND_DOF = 12  # Update this if your setup uses a different DOF count


class HandBridgeNode(Node):
    def __init__(self):
        super().__init__('hand_bridge_node')

        # ROS2 Subscribers and Publishers
        self.ros_state_sub = self.create_subscription(
            MotorStates,
            '/hands/state',  # ROS2-native state topic
            self.ros_state_callback,
            10
        )
        self.ros_cmd_pub = self.create_publisher(
            MotorCmds,
            '/hands/cmd',  # ROS2-native cmd topic
            10
        )

        # CycloneDDS Publishers and Subscribers
        self.dds_state_pub = ChannelPublisher('/rt/inspire/state', DDS_MotorStates)
        self.dds_state_pub.Init()

        self.dds_cmd_sub = ChannelSubscriber('/rt/inspire/cmd', DDS_MotorCmds)
        self.dds_cmd_sub.Init(self.dds_cmd_callback, 10)

        self.get_logger().info("HandBridgeNode initialized.")

    def ros_state_callback(self, msg: MotorStates):
        dds_msg = DDS_MotorStates()
        dds_msg.states = []

        for i in range(min(NUM_HAND_DOF, len(msg.motor_states))):
            ros_state = msg.motor_states[i]
            dds_state = DDS_MotorState()
            dds_state.q = float(ros_state.q)
            # Fill other fields only if needed
            dds_msg.states.append(dds_state)

        self.dds_state_pub.Write(dds_msg)

    def dds_cmd_callback(self, msg: DDS_MotorCmds):
        ros_msg = MotorCmds()
        ros_msg.motor_commands = []

        for i in range(min(NUM_HAND_DOF, len(msg.cmds))):
            dds_cmd = msg.cmds[i]
            ros_cmd = MotorCmd()
            ros_cmd.q = float(dds_cmd.q)
            ros_cmd.mode = int(dds_cmd.mode)
            ros_cmd.dq = float(dds_cmd.dq)
            ros_cmd.tau = float(dds_cmd.tau)
            # Optional: fill other fields with zeros or ignore
            ros_msg.motor_commands.append(ros_cmd)

        self.ros_cmd_pub.publish(ros_msg)


def main(args=None):
    rclpy.init(args=args)
    bridge_node = HandBridgeNode()

    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
