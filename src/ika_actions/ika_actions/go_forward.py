#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rake_core.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import json
from types import SimpleNamespace

from ika_actions.action import GoForward
from geometry_msgs.msg import Twist


class GoForwardConfig:
    """Configuration class for go forward action parameters"""

    def __init__(self):
        # Velocity limits
        self.min_velocity = 0.2  # m/s
        self.max_velocity = 1.2  # m/s

        # Control frequency
        self.control_frequency = 20.0  # Hz
        self.control_period = 1.0 / self.control_frequency  # seconds


class GoForwardServer(Node):
    """Action server for moving robot forward with specified speed and time"""

    def __init__(self):
        super().__init__("go_forward_server")
        self.callback_group = ReentrantCallbackGroup()

    def init(self):
        # Action server
        self._action_server = ActionServer(
            self,
            GoForward,
            "go_forward",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)

        self.get_logger().info("Go forward action server initialized")

    def get_default_config(self):
        return json.loads(json.dumps(GoForwardConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def goal_callback(self, goal_request):
        """Validate goal parameters and accept/reject accordingly"""
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        speed = goal_request.speed
        time_duration = goal_request.time

        # Validate speed limits
        if speed < self.config.min_velocity or speed > self.config.max_velocity:
            self.get_logger().warn(
                f"Goal rejected: Speed {speed:.2f} m/s outside valid range "
                f"[{self.config.min_velocity}, {self.config.max_velocity}]"
            )
            return GoalResponse.REJECT

        # Validate time duration
        if time_duration <= 0:
            self.get_logger().warn(
                f"Goal rejected: Invalid time duration {time_duration:.2f}s"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Goal accepted: Moving forward at {speed:.2f} m/s for {time_duration:.2f}s"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation"""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute go forward action"""
        self.get_logger().info("Executing go forward action...")

        # Get goal parameters
        speed = goal_handle.request.speed
        time_duration = goal_handle.request.time

        # Initialize feedback and result
        feedback_msg = GoForward.Feedback()
        result = GoForward.Result()

        # Reset timing
        start_time = time.time()

        try:
            # Initial status: starting movement
            feedback_msg.status = GoForward.Feedback.STATUS_RUNNING
            feedback_msg.elapsed_time = 0.0
            feedback_msg.message = f"Starting forward motion at {speed:.2f} m/s"
            goal_handle.publish_feedback(feedback_msg)

            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Goal was cancelled")
                    self._publish_zero_velocity()

                    feedback_msg.status = GoForward.Feedback.STATUS_FAILED
                    feedback_msg.elapsed_time = elapsed_time
                    feedback_msg.message = "Goal cancelled by client"
                    goal_handle.publish_feedback(feedback_msg)

                    goal_handle.canceled()
                    result.status = GoForward.Result.STATUS_CANCELLED
                    result.message = "Goal cancelled by client"
                    return result

                # Check if time duration is reached
                if elapsed_time >= time_duration:
                    self.get_logger().info(
                        f"Forward motion completed! Duration: {elapsed_time:.2f}s"
                    )
                    self._publish_zero_velocity()

                    feedback_msg.status = GoForward.Feedback.STATUS_COMPLETED
                    feedback_msg.elapsed_time = elapsed_time
                    feedback_msg.message = f"Successfully completed forward motion"
                    goal_handle.publish_feedback(feedback_msg)

                    result.status = GoForward.Result.STATUS_SUCCEEDED
                    result.message = (
                        f"Successfully moved forward for {elapsed_time:.2f}s"
                    )
                    goal_handle.succeed()
                    return result

                # Publish forward velocity command
                cmd_msg = Twist()
                cmd_msg.linear.x = speed
                self.cmd_vel_pub.publish(cmd_msg)

                # Publish feedback with running status
                feedback_msg.status = GoForward.Feedback.STATUS_RUNNING
                feedback_msg.elapsed_time = elapsed_time
                feedback_msg.message = (
                    f"Moving forward... {elapsed_time:.1f}/{time_duration:.1f}s"
                )
                goal_handle.publish_feedback(feedback_msg)

                self.get_logger().debug(
                    f"Forward motion: {elapsed_time:.2f}s / {time_duration:.2f}s "
                    f"at {speed:.2f} m/s"
                )

                # Sleep to maintain control frequency
                time.sleep(self.config.control_period)

        except Exception as e:
            self.get_logger().error(f"Error during forward motion execution: {str(e)}")
            self._publish_zero_velocity()

            feedback_msg.status = GoForward.Feedback.STATUS_FAILED
            feedback_msg.elapsed_time = time.time() - start_time
            feedback_msg.message = f"Execution error: {str(e)}"
            goal_handle.publish_feedback(feedback_msg)

            result.status = GoForward.Result.STATUS_FAILED
            result.message = f"Execution error: {str(e)}"
            goal_handle.abort()
            return result

    def _publish_zero_velocity(self):
        """Publish zero velocity command to stop the robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)

    go_forward_server = GoForwardServer()
    go_forward_server.init()
    Node.run_node(go_forward_server)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
