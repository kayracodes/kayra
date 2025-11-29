#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Float32MultiArray
from ika_utils.costmap_utils import normalize_angle

"""
Subs: /odom, /path
pubs: /cmd_vel, /debug_topic
"""


class Pose:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


class PathTrackerConfig:
    def __init__(self):
        self.L = 0.95
        self.kp_angular = 4.0
        self.max_angular_velocity = 2.5
        self.max_linear_velocity = 0.8
        self.min_linear_velocity = 0.3
        self.waypoint_tolerance = (
            0.1  # Tolerance for reaching waypoints (roughly half waypoint spacing)
        )
        self.backward_speed = -0.4
        self.back_iters = 10
        self.map_resolution = 0.08  # meters per pixel
        self.map_width = 75  # pixels
        self.map_height = 75  # pixels
        self.map_origin_x = 0.0  # meters
        self.map_origin_y = -3.0  # meters


class PathTracker(Node):
    def __init__(self):
        super().__init__("path_tracker")

        # Control parameters
        self.conf = PathTrackerConfig()

        # State variables
        self.current_waypoint_idx = 1
        self.current_pose = Pose()
        self.waypoints = None
        self.is_recovery_mode = False
        self.back_count = -1
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)
        self.cmd_vel_msg = Twist()
        # self.debug_pub = self.create_publisher(Float32MultiArray, "/ika_nav/", 10)

        self.path_sub = self.create_subscription(
            Path, "/ika_nav/planned_path", self.path_callback, 10
        )
        # Shutdown handler
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Path Tracker initialized")

    def path_callback(self, msg):
        self.get_logger().info("Received new path")
        """Alternative callback for ROS Path messages"""
        if msg.poses is None or len(msg.poses) <= 1:
            self.waypoints = None
            return
        self.waypoints = []
        for pose in msg.poses:
            point = pose.pose.position
            self.waypoints.append((point.x, point.y))

    def control_loop(self):
        if self.waypoints is None:
            self.get_logger().warn("No path given, cannot control robot")
            return
        elif len(self.waypoints) == 1:
            self.get_logger().warn("Entering recovery")
            self.is_recovery_mode = True

        if self.is_recovery_mode:
            if self.back_count < self.conf.back_iters:
                self.back_count += 1
            else:
                self.is_recovery_mode = False
                self.back_count = -1
            self.get_logger().info("Recovery mode active, stopping robot")
            self.cmd_vel_msg.linear.x = self.conf.backward_speed
            self.cmd_vel_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel_msg)
            return

        pose_x, pose_y = self.current_pose.x, self.current_pose.y
        next_waypoint_idx = self.current_waypoint_idx + 10
        if next_waypoint_idx >= len(self.waypoints):
            next_waypoint_idx = len(self.waypoints) - 1
        next_waypoint = self.waypoints[next_waypoint_idx]
        error = math.atan2(next_waypoint[1] - pose_y, next_waypoint[0] - pose_x)
        self.cmd_vel_msg.angular.z = self.clamp(
            error * self.conf.kp_angular,
            -self.conf.max_angular_velocity,
            self.conf.max_angular_velocity,
        )
        self.cmd_vel_msg.linear.x = (
            self.conf.max_linear_velocity * (1 - abs(error) / math.pi) ** 3
        )
        self.get_logger().info(
            f"Cmd Vel: linear.x={self.cmd_vel_msg.linear.x}, angular.z={self.cmd_vel_msg.angular.z}"
        )
        self.cmd_vel_pub.publish(self.cmd_vel_msg)

    def shutdown_callback(self):
        """Callback to handle shutdown"""
        self.get_logger().info("Shutting down Path Tracker")
        self.cmd_vel_msg.linear.x = 0.0
        self.cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd_vel_msg)
        self.get_logger().info("Path Tracker shutdown complete")

    # UTILITY FUNCTIONS
    def pixel_to_pose(self, pixel_x, pixel_y):
        x = pixel_x * self.map_resolution + self.map_origin_x
        y = pixel_y * self.map_resolution + self.map_origin_y
        return x, y

    def clamp(self, a, a_min, a_max):
        if a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        return a


def main(args=None):
    rclpy.init(args=args)

    node = PathTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_callback()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
