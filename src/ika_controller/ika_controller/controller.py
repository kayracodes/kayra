#!/usr/bin/env python3
import rclpy
from rake_core.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from rclpy.time import Time
from rclpy.constants import S_TO_NS
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import math
from tf_transformations import quaternion_from_euler
from ika_msgs.msg import MotorFeedback
import json
from types import SimpleNamespace


class OdometryNodeConfig:
    def __init__(self, wheel_radius=0.125, wheel_separation=0.92):
        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation


class OdometryNode(Node):
    def __init__(self):
        super().__init__("odom_node")

        self.front_left_wheel_prev_pos = 0.0
        self.front_right_wheel_prev_pos = 0.0
        self.rear_left_wheel_prev_pos = 0.0
        self.rear_right_wheel_prev_pos = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().info(
                "No config found, using default parameters for odom node"
            )
            return

        self.speed_conversion = np.array(
            [
                [
                    self.config.wheel_radius / 2,
                    self.config.wheel_radius / 2,
                ],
                [
                    self.config.wheel_radius / self.config.wheel_separation,
                    -self.config.wheel_radius / self.config.wheel_separation,
                ],
            ]
        )
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "base_footprint"
        self.odom_msg.pose.pose.orientation.x = 0.0
        self.odom_msg.pose.pose.orientation.y = 0.0
        self.odom_msg.pose.pose.orientation.z = 0.0
        self.odom_msg.pose.pose.orientation.w = 1.0
        self.motor_feedback_msg = MotorFeedback()
        # Set the covariance matrix for position (x, y, z, roll, pitch, yaw)

        # TF Package
        self.br = TransformBroadcaster(self)
        self.transform_stamped = TransformStamped()
        self.transform_stamped.header.frame_id = "odom"
        self.transform_stamped.child_frame_id = "base_footprint"

        self.prev_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10
        )
        self.wheel_cmd_pub = self.create_publisher(
            Float64MultiArray, "simple_velocity_controller/commands", 10
        )
        self.motor_feedback_pub = self.create_publisher(
            MotorFeedback, "/motor_feedback", 10
        )
        self.vel_sub_ = self.create_subscription(
            Twist, "/ika_nav/cmd_vel", self.velCallback, 10
        )
        # Timers
        self.create_timer(0.05, self.publish_odom)

        self.get_logger().info("Odom node has been started")

    def get_default_config(self):
        return json.loads(json.dumps(OdometryNodeConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def velCallback(self, msg):
        wheel_speed_msg = Float64MultiArray()

        if not self.mobility:
            wheel_speed_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.wheel_cmd_pub.publish(wheel_speed_msg)
            return

        robot_speed = np.array([[msg.linear.x], [msg.angular.z]])
        wheel_speed = np.matmul(np.linalg.inv(self.speed_conversion), robot_speed)

        wheel_speed_msg.data = [
            wheel_speed[0, 0],
            wheel_speed[1, 0],
            wheel_speed[0, 0],
            wheel_speed[1, 0],
        ]

        self.wheel_cmd_pub.publish(wheel_speed_msg)

    def joint_callback(self, msg):

        dp_front_right = msg.position[0] - self.front_right_wheel_prev_pos
        dp_front_left = msg.position[1] - self.front_left_wheel_prev_pos
        dp_rear_right = msg.position[2] - self.rear_right_wheel_prev_pos
        dp_rear_left = msg.position[3] - self.rear_left_wheel_prev_pos

        dp_right = (dp_front_right + dp_rear_right) / 2.0
        dp_left = (dp_front_left + dp_rear_left) / 2.0

        dt = Time.from_msg(msg.header.stamp) - self.prev_time
        dt_sec = dt.nanoseconds / S_TO_NS

        self.front_right_wheel_prev_pos = msg.position[0]
        self.front_left_wheel_prev_pos = msg.position[1]
        self.rear_right_wheel_prev_pos = msg.position[2]
        self.rear_left_wheel_prev_pos = msg.position[3]
        self.prev_time = Time.from_msg(msg.header.stamp)

        fi_left = dp_left / dt_sec
        fi_right = dp_right / dt_sec

        linear = (
            self.config.wheel_radius * fi_right + self.config.wheel_radius * fi_left
        ) / 2
        angular = (
            self.config.wheel_radius * fi_right - self.config.wheel_radius * fi_left
        ) / self.config.wheel_separation

        d_s = (
            self.config.wheel_radius * dp_right + self.config.wheel_radius * dp_left
        ) / 2
        d_theta = (
            self.config.wheel_radius * dp_right - self.config.wheel_radius * dp_left
        ) / self.config.wheel_separation
        self.theta += d_theta
        self.x += d_s * math.cos(self.theta)
        self.y += d_s * math.sin(self.theta)

        q = quaternion_from_euler(0, 0, self.theta)

        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_msg.pose.pose.position.x = self.x
        self.odom_msg.pose.pose.position.y = self.y
        self.odom_msg.pose.pose.orientation.x = q[0]
        self.odom_msg.pose.pose.orientation.y = q[1]
        self.odom_msg.pose.pose.orientation.z = q[2]
        self.odom_msg.pose.pose.orientation.w = q[3]
        self.odom_msg.twist.twist.linear.x = linear
        self.odom_msg.twist.twist.angular.z = angular

        self.motor_feedback_msg.v = linear
        self.motor_feedback_msg.w = angular
        # Enhanced logging with more debug info
        # radius = abs(linear / (angular + 1e-6)) if abs(angular) > 1e-6 else float("inf")
        # self.get_logger().info(
        #     f"v: {linear:.4f}, w: {angular:.4f}, R: {radius:.4f}, dt: {dt_sec:.6f}, "
        #     f"dp_left: {dp_left:.6f}, dp_right: {dp_right:.6f}"
        # )

        self.transform_stamped.transform.translation.x = self.x
        self.transform_stamped.transform.translation.y = self.y
        self.transform_stamped.transform.rotation.x = q[0]
        self.transform_stamped.transform.rotation.y = q[1]
        self.transform_stamped.transform.rotation.z = q[2]
        self.transform_stamped.transform.rotation.w = q[3]
        self.transform_stamped.header.stamp = self.get_clock().now().to_msg()

    def publish_odom(self):
        # self.odom_pub.publish(self.odom_msg)

        self.motor_feedback_msg.dt = 0.05  # Same period as the timer
        self.motor_feedback_pub.publish(self.motor_feedback_msg)
        # self.br.sendTransform(self.transform_stamped)
        pass


def main(args=None):
    rclpy.init(args=args)
    odometry_node = OdometryNode()
    try:
        Node.run_node(odometry_node)
    except KeyboardInterrupt:
        pass
    finally:
        odometry_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
