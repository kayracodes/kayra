#!/usr/bin/env python3

import rclpy
from rake_core.node import Node
from rake_core.states import SystemModeEnum, SystemStateEnum, DeviceStateEnum
from rake_msgs.msg import SystemState, DeviceState
import json
import numpy as np
import math
from types import SimpleNamespace
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Float32MultiArray
from ika_utils.costmap_utils import normalize_angle
from rclpy.exceptions import ROSInterruptException

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
        self.def_kp_angular = 7.0
        self.def_kd_angular = 0.5
        self.def_max_angular_velocity = 3.0
        self.def_max_linear_velocity = 0.8
        self.def_min_linear_velocity = 0.2
        self.waypoint_tolerance = (
            0.1  # Tolerance for reaching waypoints (roughly half waypoint spacing)
        )
        self.next_waypoints = 13  # Number of waypoints to look ahead
        self.backward_speed = -0.2
        self.map_resolution = 0.08  # meters per pixel
        self.map_width = 75  # pixels
        self.map_height = 75  # pixels
        self.map_origin_x = 0.0  # meters
        self.map_origin_y = -3.0  # meters

        self.recovery_aggression_coeff = 1.1
        self.recovery_rotation_time = 3.0

        # State-dependant speeds
        self.slow_auto_max_speed = 0.5
        self.fast_auto_max_speed = 1.1
        self.cautious_auto_max_speed = 0.7
        # State-dependant kp, kd values
        self.sharp_turn_kp_angular = 9.0
        self.sharp_turn_kd_angular = 0.3


class PathTracker(Node):
    def __init__(self):
        super().__init__("path_tracker")
        # State variables
        self.current_waypoint_idx = 1
        self.current_pose = Pose()
        self.waypoints = None
        self.is_recovery_mode = False
        self.recovery_aggression = 1.0
        self.recovery_time = 0
        self.prev_error = 0.0

    def init(self):
        # Publishers
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return
        # define the runtime changable parameters first
        self.kp_angular = self.config.def_kp_angular
        self.kd_angular = self.config.def_kd_angular
        self.max_angular_velocity = self.config.def_max_angular_velocity
        self.max_linear_velocity = self.config.def_max_linear_velocity
        self.min_linear_velocity = self.config.def_min_linear_velocity

        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)
        self.cmd_vel_msg = Twist()
        # self.debug_pub = self.create_publisher(Float32MultiArray, "/ika_nav/", 10)

        self.path_sub = self.create_subscription(
            Path, "/ika_nav/planned_path", self.path_callback, 10
        )
        # Shutdown handler
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Path Tracker initialized")

    def system_state_transition(self, old_state: SystemState, new_state: SystemState):
        # Device & System State Changes
        if not hasattr(self, "config"):
            return
        if (
            (
                new_state.state > SystemStateEnum.SEMI_AUTONOMOUS
                and new_state.state < SystemStateEnum.TARGET_HITTING
            )
            and self.device_state == DeviceStateEnum.READY
            and new_state.mobility
        ):
            self.set_device_state(DeviceStateEnum.WORKING)
        elif not (
            new_state.state > SystemStateEnum.SEMI_AUTONOMOUS
            and new_state.state < SystemStateEnum.TARGET_HITTING
            and self.device_state == DeviceStateEnum.WORKING
        ):
            self.set_device_state(DeviceStateEnum.READY)
            self.on_reset()

        elif (
            not new_state.mobility
            and old_state.mobility
            and self.device_state == DeviceStateEnum.WORKING
        ):
            self.set_device_state(DeviceStateEnum.READY)
            self.on_reset()

        # Configuration Changes

        updated_config = self.get_state_config_changes(new_state.state)
        if not updated_config:
            return
        try:
            for k, v in updated_config.items():
                setattr(self, k, v)
        except Exception as e:
            self.get_logger().error(f"Error updating state config: {e}")

        self.get_logger().info(f"Runtime Changable Params: {updated_config}")

    def get_state_config_changes(self, state: SystemStateEnum):
        # Device Configuration Changes
        if state == SystemStateEnum.SLOW_AUTONOMOUS:
            return {
                "kp_angular": self.config.def_kp_angular,
                "kd_angular": self.config.def_kd_angular,
                "max_angular_velocity": self.config.def_max_angular_velocity,
                "max_linear_velocity": self.config.slow_auto_max_speed,
                "min_linear_velocity": self.config.def_min_linear_velocity,
            }
        elif state == SystemStateEnum.FAST_AUTONOMOUS:
            return {
                "kp_angular": self.config.def_kp_angular,
                "kd_angular": self.config.def_kd_angular,
                "max_angular_velocity": self.config.def_max_angular_velocity,
                "max_linear_velocity": self.config.fast_auto_max_speed,
                "min_linear_velocity": self.config.def_min_linear_velocity,
            }
        elif state == SystemStateEnum.CAUTIOUS_AUTONOMOUS:
            return {
                "kp_angular": self.config.def_kp_angular,
                "kd_angular": self.config.def_kd_angular,
                "max_angular_velocity": self.config.def_max_angular_velocity,
                "max_linear_velocity": self.config.cautious_auto_max_speed,
                "min_linear_velocity": self.config.def_min_linear_velocity,
            }
        elif state == SystemStateEnum.SHARP_TURN_AUTONOMOUS:
            return {
                "kp_angular": self.config.sharp_turn_kp_angular,
                "kd_angular": self.config.sharp_turn_kd_angular,
                "max_angular_velocity": self.config.def_max_angular_velocity,
                "max_linear_velocity": self.config.def_max_linear_velocity,
                "min_linear_velocity": self.config.def_min_linear_velocity,
            }
        else:
            return {
                "kp_angular": self.config.def_kp_angular,
                "kd_angular": self.config.def_kd_angular,
                "max_angular_velocity": self.config.def_max_angular_velocity,
                "max_linear_velocity": self.config.def_max_linear_velocity,
                "min_linear_velocity": self.config.def_min_linear_velocity,
            }

    def get_default_config(self):
        return json.loads(json.dumps(PathTrackerConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def path_callback(self, msg):
        """Alternative callback for ROS Path messages"""
        if len(msg.poses) < 10:
            self.waypoints = []
            return
        self.is_recovery_mode = False
        self.current_waypoint_idx = 2
        self.waypoints = []
        for pose in msg.poses:
            point = pose.pose.position
            self.waypoints.append((point.x, point.y))

    def control_loop(self):
        if not (
            self.device_state == DeviceStateEnum.WORKING
            and (
                SystemStateEnum.SEMI_AUTONOMOUS
                < self.system_state
                < SystemStateEnum.TARGET_HITTING
            )
        ):
            self.get_logger().warn("System not in valid state for control")
            return

        if self.waypoints is None:
            self.get_logger().warn("No path given, cannot control robot")
            return
        elif len(self.waypoints) == 0:
            self.get_logger().warn("Entering recovery")
            self.is_recovery_mode = True

        if self.is_recovery_mode:
            self.recovery_aggression *= self.config.recovery_aggression_coeff
            angular_cmd = self.clamp(
                self.recovery_aggression * self.prev_error * self.kp_angular,
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
            if self.recovery_time > self.config.recovery_rotation_time:
                angular_cmd = -angular_cmd
                self.recovery_aggression = 1.0

            self.cmd_vel_msg.angular.z = angular_cmd
            self.cmd_vel_msg.linear.x = self.config.backward_speed
            self.cmd_vel_pub.publish(self.cmd_vel_msg)

            self.recovery_time += 0.1
            self.get_logger().info("Recovery mode active, stopping robot")
            return

        # reset recovery variables
        self.recovery_time = 0.0
        self.recovery_aggression = 1.0

        # determine and publish velocity commands
        pose_x, pose_y = self.current_pose.x, self.current_pose.y
        next_waypoint_idx = self.current_waypoint_idx + self.config.next_waypoints
        if next_waypoint_idx >= len(self.waypoints):
            self.current_waypoint_idx = len(self.waypoints) - 1
            next_waypoint_idx = len(self.waypoints) - 1
        next_waypoint = self.waypoints[next_waypoint_idx]
        error = math.atan2(next_waypoint[1] - pose_y, next_waypoint[0] - pose_x)
        delta_error = (error - self.prev_error) / 0.1
        self.prev_error = error
        angular_cmd = self.clamp(
            error * self.kp_angular + delta_error * self.kd_angular,
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )
        # angular_cmd = 0.0 if abs(angular_cmd) < 0.55 else angular_cmd
        self.cmd_vel_msg.angular.z = angular_cmd
        self.cmd_vel_msg.linear.x = (
            self.max_linear_velocity * (1 - abs(error) / math.pi) ** 5
        )
        self.get_logger().info(
            f"Cmd Vel: linear.x={self.cmd_vel_msg.linear.x}, angular.z={self.cmd_vel_msg.angular.z}"
        )
        self.cmd_vel_pub.publish(self.cmd_vel_msg)

    def on_reset(self):
        self.waypoints = []
        self.cmd_vel_msg = Twist()
        self.cmd_vel_msg.linear.x = 0.0
        self.cmd_vel_msg.angular.z = 0.0
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
    try:
        rclpy.init()
        node = PathTracker()
        node.init()
        Node.run_node(node)
    except ROSInterruptException:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
