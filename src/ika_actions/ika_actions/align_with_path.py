#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rake_core.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from types import SimpleNamespace

import numpy as np
import math
import json
import time
from rake_core.constants import Actions
from ika_actions.action import AlignWithPath
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist


class AlignWithPathConfig:
    """Configuration class for align with path action parameters"""

    def __init__(self):
        # PD Controller parameters (same as path_tracker.py)
        self.kp_angular = 10.0  # Proportional gain
        self.kd_angular = 0.2  # Derivative gain

        # Alignment thresholds
        self.angle_tolerance = 0.025  # radians (approximately 1.43 degrees)
        self.max_execution_time = 10.0  # seconds

        # Control limits
        self.max_angular_velocity = 3.0  # rad/s
        self.min_angular_velocity = 1.0  # rad/s

        # Control frequency
        self.control_frequency = 10.0  # Hz (same as path_tracker control_loop)
        self.control_period = 1.0 / self.control_frequency  # 0.1 seconds


class AlignWithPathServer(Node):
    """Action server for aligning robot heading with planned path"""

    def __init__(self):
        super().__init__("align_with_path_server")
        self.callback_group = ReentrantCallbackGroup()

    def init(self):
        # Action server
        self._action_server = ActionServer(
            self,
            AlignWithPath,
            Actions.ALIGN_WITH_PATH,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            "/ika_nav/planned_path",
            self.path_callback,
            10,
            callback_group=self.callback_group,
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)

        # State variables
        self.waypoints = None
        self.prev_error = 0.0

        self.get_logger().info("Align with path action server initialized")

    def get_default_config(self):
        return json.loads(json.dumps(AlignWithPathConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def path_callback(self, msg):
        """Callback to store latest path data"""

        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        self.get_logger().info(f"Received new path with {len(msg.poses)} poses")
        if len(msg.poses) < 10:
            self.waypoints = []
            return

        self.waypoints = []
        for pose in msg.poses:
            point = pose.pose.position
            self.waypoints.append((point.x, point.y))

    def goal_callback(self, goal_request):
        """Accept goal only if path is available"""
        if self.waypoints is None or len(self.waypoints) == 0:
            self.get_logger().warn("Goal rejected: No path available")
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Goal accepted: Path available with {len(self.waypoints)} waypoints"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation"""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def calculate_angular_error(self):
        """Calculate angular error using the same approach as path_tracker.py"""
        if not self.waypoints or len(self.waypoints) < 2:
            return 0.0

        # Use same logic as path_tracker.py: look from 2nd to 45th element (or last if fewer)
        start_idx = 1  # 2nd element (0-indexed)
        end_idx = min(45, len(self.waypoints) - 1)  # 45th element or last

        if start_idx >= len(self.waypoints):
            start_idx = len(self.waypoints) - 1
        if end_idx <= start_idx:
            end_idx = (
                start_idx + 1 if start_idx + 1 < len(self.waypoints) else start_idx
            )

        # Calculate direction vector from start to end point
        start_point = self.waypoints[start_idx]
        end_point = self.waypoints[end_idx]

        # Calculate angle using atan2 (same as path_tracker line 124)
        error = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

        return error

    def pd_controller(self, error, dt):
        """PD controller for angular velocity calculation (same as path_tracker.py)"""
        if dt <= 0:
            dt = self.config.control_period  # Use default control period

        # Proportional term
        p_term = self.config.kp_angular * error

        # Derivative term
        delta_error = (error - self.prev_error) / dt
        d_term = self.config.kd_angular * delta_error

        # Calculate control output
        control_output = p_term + d_term

        # Apply limits (same clamp logic as path_tracker)
        control_output = self.clamp(
            control_output,
            -self.config.max_angular_velocity,
            self.config.max_angular_velocity,
        )

        # Apply minimum velocity threshold
        control_output = (
            control_output
            if abs(control_output) >= self.config.min_angular_velocity
            else np.sign(control_output) * self.config.min_angular_velocity
        )
        self.last_error = error
        return control_output

    def clamp(self, a, a_min, a_max):
        """Utility function to clamp values (same as path_tracker.py)"""
        if a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        return a

    def execute_callback(self, goal_handle):
        """Execute path alignment action"""
        self.get_logger().info("Executing path alignment...")

        # Initialize feedback and result
        feedback_msg = AlignWithPath.Feedback()
        result = AlignWithPath.Result()

        # Reset controller state
        self.prev_error = 0.0
        start_time = time.time()

        try:
            # Initial status: searching for path alignment
            feedback_msg.status = AlignWithPath.Feedback.STATUS_SEARCHING
            feedback_msg.elapsed_time = 0.0
            feedback_msg.angular_error = 0.0
            feedback_msg.message = "Starting path alignment..."
            goal_handle.publish_feedback(feedback_msg)

            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if self.mobility == 0:
                    self.get_logger().warn(
                        "System mobility is disabled, aborting path alignment"
                    )
                    goal_handle.abort()
                    result.status = AlignWithPath.Result.STATUS_FAILED
                    result.message = "System mobility is disabled"
                    return result

                # Check timeout
                if elapsed_time > self.config.max_execution_time:
                    self.get_logger().warn("Path alignment timeout exceeded")
                    self._publish_zero_velocity()

                    feedback_msg.status = AlignWithPath.Feedback.STATUS_FAILED
                    feedback_msg.elapsed_time = elapsed_time
                    feedback_msg.message = f"Timeout: Could not align within {self.config.max_execution_time}s"
                    goal_handle.publish_feedback(feedback_msg)

                    result.status = AlignWithPath.Result.STATUS_TIMEOUT
                    result.message = f"Timeout: Could not align within {self.config.max_execution_time}s"
                    goal_handle.abort()
                    return result

                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Goal was cancelled")
                    self._publish_zero_velocity()

                    feedback_msg.status = AlignWithPath.Feedback.STATUS_FAILED
                    feedback_msg.elapsed_time = elapsed_time
                    feedback_msg.message = "Goal cancelled by client"
                    goal_handle.publish_feedback(feedback_msg)

                    goal_handle.canceled()
                    result.status = AlignWithPath.Result.STATUS_CANCELLED
                    result.message = "Goal cancelled by client"
                    return result

                # Check if we still have path data
                if self.waypoints is None or len(self.waypoints) == 0:
                    self.get_logger().warn("Path lost during alignment")
                    self._publish_zero_velocity()

                    feedback_msg.status = AlignWithPath.Feedback.STATUS_FAILED
                    feedback_msg.elapsed_time = elapsed_time
                    feedback_msg.message = "Path data lost during alignment"
                    goal_handle.publish_feedback(feedback_msg)

                    result.status = AlignWithPath.Result.STATUS_NO_PATH
                    result.message = "Path data lost during alignment"
                    goal_handle.abort()
                    return result

                # Calculate current angular error
                angular_error = self.calculate_angular_error()
                angular_error_degrees = angular_error * 180.0 / math.pi

                # Check if alignment is achieved
                if abs(angular_error) < self.config.angle_tolerance:
                    self.get_logger().info(
                        f"Path alignment successful! Final error: {angular_error_degrees:.2f}°"
                    )
                    self._publish_zero_velocity()

                    feedback_msg.status = AlignWithPath.Feedback.STATUS_ALIGNED
                    feedback_msg.elapsed_time = elapsed_time
                    feedback_msg.angular_error = angular_error_degrees
                    feedback_msg.message = f"Successfully aligned with path (error: {angular_error_degrees:.2f}°)"
                    goal_handle.publish_feedback(feedback_msg)

                    result.status = AlignWithPath.Result.STATUS_SUCCEEDED
                    result.message = f"Successfully aligned with path (error: {angular_error_degrees:.2f}°)"
                    goal_handle.succeed()
                    return result

                # Calculate control command using PD controller
                angular_velocity = self.pd_controller(
                    angular_error, self.config.control_period
                )

                # Publish control command
                cmd_msg = Twist()
                cmd_msg.angular.z = angular_velocity
                self.cmd_vel_pub.publish(cmd_msg)

                # Publish feedback with aligning status
                feedback_msg.status = AlignWithPath.Feedback.STATUS_ALIGNING
                feedback_msg.elapsed_time = elapsed_time
                feedback_msg.angular_error = angular_error_degrees
                feedback_msg.message = (
                    f"Aligning with path... Error: {angular_error_degrees:.2f}°"
                )
                goal_handle.publish_feedback(feedback_msg)

                self.get_logger().debug(
                    f"Path alignment: Error: {angular_error_degrees:.2f}°, "
                    f"Cmd: {angular_velocity:.3f} rad/s, Time: {elapsed_time:.1f}s"
                )

                # Sleep to maintain control frequency
                time.sleep(self.config.control_period)

        except Exception as e:
            self.get_logger().error(f"Error during path alignment execution: {str(e)}")
            self._publish_zero_velocity()

            feedback_msg.status = AlignWithPath.Feedback.STATUS_FAILED
            feedback_msg.elapsed_time = time.time() - start_time
            feedback_msg.angular_error = 0.0
            feedback_msg.message = f"Execution error: {str(e)}"
            goal_handle.publish_feedback(feedback_msg)

            result.status = AlignWithPath.Result.STATUS_FAILED
            result.message = f"Execution error: {str(e)}"
            goal_handle.abort()
            return result

    def _publish_zero_velocity(self):
        """Publish zero velocity command to stop the robot"""
        cmd_msg = Twist()
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)

    align_with_path_server = AlignWithPathServer()

    # Use MultiThreadedExecutor for handling action callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(align_with_path_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        align_with_path_server.get_logger().info(
            "Shutting down align with path server..."
        )
    finally:
        align_with_path_server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
