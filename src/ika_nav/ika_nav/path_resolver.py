#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from ika_nav.pure_pursuit import PurePursuit
import math
from ika_utils.motion_model_utils import normalize_angle, get_yaw_from_quaternion
import numpy as np
from rclpy.executors import MultiThreadedExecutor


class Pose:
    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None


class PathResolverNodeconfs:
    def __init__(self):
        self.forward_speed = 1.5
        self.reverse_speed = -0.4
        self.radius_multiplier = 1.2
        self.radius_max = 4.0
        self.radius_start = 0.7
        self.angular_aggression = 1.8
        self.max_angular_speed = 0.8


class PathResolverNode(Node):
    def __init__(self):
        super().__init__("autonav_nav_resolver")

        self.conf = PathResolverNodeconfs()
        self.pose = None
        self.pure_pursuit = PurePursuit()
        self.backCount = -1
        self.status = -1
        self.path_subscriber = self.create_subscription(
            Path, "/ika_nav/planned_path", self.on_path_received, 1
        )
        self.odom = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_callback, 1
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 1)

        self.create_timer(0.05, self.resolve)

    def on_reset(self):
        self.pose = None
        self.backCount = -1

    def odom_callback(self, msg: Odometry):
        self.pose = Pose()
        self.pose.x = msg.pose.pose.position.x
        self.pose.y = msg.pose.pose.position.y
        self.pose.theta = get_yaw_from_quaternion(msg.pose.pose.orientation)

    def on_path_received(self, msg: Path):
        self.points = [x.pose.position for x in msg.poses]
        self.pure_pursuit.set_points([(point.x, point.y) for point in self.points])

    def resolve(self):
        if self.pose is None:
            return

        cur_pos = (self.pose.x, self.pose.y)
        lookahead = None
        radius = self.conf.radius_start
        while lookahead is None and radius <= self.conf.radius_max:
            lookahead = self.pure_pursuit.get_lookahead_point(
                cur_pos[0], cur_pos[1], radius
            )
            radius *= self.conf.radius_multiplier

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        if self.backCount == -1 and (
            lookahead is not None
            and ((lookahead[1] - cur_pos[1]) ** 2 + (lookahead[0] - cur_pos[0]) ** 2)
            > 0.25
        ):
            angle_diff = math.atan2(
                lookahead[1] - cur_pos[1], lookahead[0] - cur_pos[0]
            )
            error = normalize_angle(angle_diff - self.pose.theta) / math.pi
            self.get_logger().info(f"Angle diff: {angle_diff}, Error: {error}")
            forward_speed = self.conf.forward_speed * (1 - abs(error)) ** 3
            cmd_vel.linear.x = forward_speed
            cmd_vel.angular.z = self.clamp(
                error * self.conf.angular_aggression,
                -self.conf.max_angular_speed,
                self.conf.max_angular_speed,
            )

            if self.status == 0:
                self.status = 1
        else:
            if self.backCount == -1:
                self.backCount = 8
                self.get_logger().info("Switching to reverse mode")
            else:
                self.status = 0
                self.backCount -= 1
                self.get_logger().info(f"Reversing, backCount: {self.backCount}")

            cmd_vel.linear.x = self.conf.reverse_speed

        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(
            f"Cmd Vel: linear.x={cmd_vel.linear.x}, angular.z={cmd_vel.angular.z}"
        )

    # UTILITY FUNCTIONS
    def clamp(self, a, a_min, a_max):
        if a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        return a


def main():
    rclpy.init()
    node = PathResolverNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
