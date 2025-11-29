#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rake_core.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import time
import json
from rake_core.constants import Actions
from types import SimpleNamespace
from ika_actions.action import RampAlignment
from ika_msgs.msg import RampFeedback
from geometry_msgs.msg import Twist


class RampAlignmentConfig:
    """Configuration class for ramp alignment parameters"""

    def __init__(self):
        # PD Controller parameters
        self.kp = 12.0  # Proportional gain
        self.kd = 0.1  # Derivative gain

        # Alignment thresholds
        self.angle_tolerance = 0.5 / 180 * np.pi  # radians
        self.max_execution_time = 13.0  # seconds

        # Control limits
        self.max_angular_velocity = 3.0  # rad/s
        self.min_angular_velocity = 1.2  # rad/s


class RampAlignmentServer(Node):
    """Action server for aligning robot with detected ramps"""

    def __init__(self):
        super().__init__("ramp_alignment_server")
        self.callback_group = ReentrantCallbackGroup()

    def init(self):
        # Action server
        self._action_server = ActionServer(
            self,
            RampAlignment,
            Actions.RAMP_ALIGNMENT,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Subscribers
        self.ramp_sub = self.create_subscription(
            RampFeedback,
            "/ika_vision/ramp_detected",
            self.ramp_callback,
            10,
            callback_group=self.callback_group,
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)

        # State variables
        self.current_ramp_data = None
        self.last_error = 0.0
        self.last_time = None

        self.get_logger().info("Ramp alignment action server initialized")

    def get_default_config(self):
        return json.loads(json.dumps(RampAlignmentConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def ramp_callback(self, msg):
        """Callback to store latest ramp detection data"""
        self.current_ramp_data = msg

    def goal_callback(self, goal_request):
        """Accept goal only if ramp is detected"""
        if self.current_ramp_data is None:
            self.get_logger().warn("Goal rejected: No ramp data available")
            return GoalResponse.REJECT

        if (
            not self.current_ramp_data.detected
            or len(self.current_ramp_data.ramps) == 0
        ):
            self.get_logger().warn("Goal rejected: No ramp detected")
            return GoalResponse.REJECT

        self.get_logger().info("Goal accepted: Ramp detected, starting alignment")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation"""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def calculate_ramp_angle(self, ramp_lines):
        """Calculate average angle from detected ramp lines"""
        if not ramp_lines:
            return 0.0

        sum_x, sum_y = 0.0, 0.0
        for ramp in ramp_lines:
            dx = ramp.x2 - ramp.x1
            dy = ramp.y2 - ramp.y1
            sum_x += dx
            sum_y += dy

        # Calculate average angle in radians
        avg_angle = np.arctan2(-sum_y, sum_x)
        return avg_angle

    def pd_controller(self, error, dt):
        """PD controller for angular velocity calculation"""
        if dt <= 0:
            dt = 0.01  # Prevent division by zero

        # Proportional term
        p_term = self.config.kp * error

        # Derivative term
        d_term = self.config.kd * (error - self.last_error) / dt

        # Calculate control output
        control_output = p_term + d_term

        # Apply limits
        control_output = np.clip(
            control_output,
            -self.config.max_angular_velocity,
            self.config.max_angular_velocity,
        )

        # Apply minimum velocity threshold
        # if (
        #     abs(control_output) > 0
        #     and abs(control_output) < self.config.min_angular_velocity
        # ):
        control_output = (
            control_output
            if abs(control_output) >= self.config.min_angular_velocity
            else np.sign(control_output) * self.config.min_angular_velocity
        )
        self.last_error = error
        return control_output

    def execute_callback(self, goal_handle):
        """Execute ramp alignment action"""
        self.get_logger().info("Executing ramp alignment...")

        # Initialize feedback and result
        feedback_msg = RampAlignment.Feedback()
        result = RampAlignment.Result()

        # Reset controller state
        self.last_error = 0.0
        self.last_time = time.time()
        start_time = self.last_time

        try:
            # Initial status: searching for ramp
            feedback_msg.status = RampAlignment.Feedback.STATUS_SEARCHING
            feedback_msg.ramp_angle = 0.0
            feedback_msg.message = "Searching for ramp..."
            goal_handle.publish_feedback(feedback_msg)

            while True:
                current_time = time.time()
                dt = current_time - self.last_time
                elapsed_time = current_time - start_time

                if self.mobility == 0:
                    self.get_logger().warn(
                        "System mobility is disabled, aborting ramp alignment"
                    )
                    goal_handle.abort()
                    result.status = RampAlignment.Result.STATUS_FAILED
                    result.message = "System mobility is disabled"
                    return result

                # Check timeout
                if elapsed_time > self.config.max_execution_time:
                    self.get_logger().warn("Alignment timeout exceeded")
                    self._publish_zero_velocity()

                    feedback_msg.status = RampAlignment.Feedback.STATUS_FAILED
                    feedback_msg.message = f"Timeout: Could not align within {self.config.max_execution_time}s"
                    goal_handle.publish_feedback(feedback_msg)
                    goal_handle.abort()

                    result.status = RampAlignment.Result.STATUS_TIMEOUT
                    result.message = f"Timeout: Could not align within {self.config.max_execution_time}s"
                    return result

                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Goal was cancelled")
                    self._publish_zero_velocity()

                    feedback_msg.status = RampAlignment.Feedback.STATUS_FAILED
                    feedback_msg.message = "Goal cancelled by client"
                    goal_handle.publish_feedback(feedback_msg)

                    goal_handle.canceled()
                    result.status = RampAlignment.Result.STATUS_CANCELLED
                    result.message = "Goal cancelled by client"
                    return result

                # Check if we still have ramp data
                if (
                    self.current_ramp_data is None
                    or not self.current_ramp_data.detected
                    or len(self.current_ramp_data.ramps) == 0
                ):
                    self.get_logger().warn("Ramp lost during alignment")
                    self._publish_zero_velocity()

                    feedback_msg.status = RampAlignment.Feedback.STATUS_FAILED
                    feedback_msg.message = "Ramp detection lost during alignment"
                    goal_handle.publish_feedback(feedback_msg)

                    result.status = RampAlignment.Result.STATUS_RAMP_LOST
                    result.message = "Ramp detection lost during alignment"
                    goal_handle.abort()
                    return result

                # Calculate current ramp angle
                ramp_angle = self.calculate_ramp_angle(self.current_ramp_data.ramps)
                angular_error = ramp_angle  # Error is the angle itself (target is 0)

                # Check if alignment is achieved
                if abs(angular_error) < self.config.angle_tolerance:
                    self.get_logger().info(
                        f"Alignment successful! Final error: {angular_error:.2f}°"
                    )
                    self._publish_zero_velocity()

                    feedback_msg.status = RampAlignment.Feedback.STATUS_ALIGNED
                    feedback_msg.ramp_angle = (
                        ramp_angle * 180 / np.pi
                    )  # Convert to degrees
                    feedback_msg.message = f"Successfully aligned (error: {angular_error * 180 / np.pi:.2f}°)"
                    goal_handle.publish_feedback(feedback_msg)

                    result.status = RampAlignment.Result.STATUS_SUCCEEDED
                    result.message = f"Successfully aligned with ramp (error: {angular_error * 180 / np.pi:.2f}°)"
                    goal_handle.succeed()
                    return result

                # Calculate control command using PD controller
                angular_velocity = self.pd_controller(angular_error, dt)

                # Publish control command
                cmd_msg = Twist()
                cmd_msg.angular.z = angular_velocity
                cmd_msg.linear.x = 0.0
                self.cmd_vel_pub.publish(cmd_msg)

                # Publish feedback with aligning status
                feedback_msg.status = RampAlignment.Feedback.STATUS_ALIGNING
                feedback_msg.ramp_angle = ramp_angle
                feedback_msg.message = f"Aligning... Error: {angular_error:.2f}°"
                goal_handle.publish_feedback(feedback_msg)

                self.get_logger().debug(
                    f"Angle: {ramp_angle:.2f}°, Error: {angular_error:.2f}°, "
                    f"Cmd: {angular_velocity:.3f} rad/s"
                )

                self.last_time = current_time

                # Sleep to maintain reasonable control frequency
                time.sleep(0.05)  # 20Hz control loop

        except Exception as e:
            self.get_logger().error(f"Error during alignment execution: {str(e)}")
            self._publish_zero_velocity()

            feedback_msg.status = RampAlignment.Feedback.STATUS_FAILED
            feedback_msg.message = f"Execution error: {str(e)}"
            goal_handle.publish_feedback(feedback_msg)

            result.status = RampAlignment.Result.STATUS_FAILED
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

    ramp_alignment_server = RampAlignmentServer()
    ramp_alignment_server.init()
    Node.run_node(ramp_alignment_server)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
