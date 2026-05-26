#!/usr/bin/env python3
"""
Bridges /joint_states (from joint_state_publisher_gui sliders)
to /joint_trajectory_controller/joint_trajectory (for Gazebo)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class JointSliderBridge(Node):
    def __init__(self):
        super().__init__('joint_slider_bridge')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.joint_names = ['R1', 'R2', 'R3']
        self.get_logger().info('Joint Slider Bridge started - move the sliders!')

    def joint_state_callback(self, msg):
        # Filter only our robot joints
        positions = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
            else:
                positions.append(0.0)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1s

        traj.points = [point]
        self.publisher.publish(traj)

def main(args=None):
    rclpy.init(args=args)
    node = JointSliderBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
