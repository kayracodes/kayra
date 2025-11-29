#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rake_core.constants import Topics, Services
from rake_msgs.srv import WaitForEvent
import json
import sys
import select
import termios
import tty
from threading import Lock
from typing import Optional


class EventCaller(Node):
    def __init__(self):
        super().__init__("event_caller")
        self.ramp_undetected_client = self.create_client(
            WaitForEvent,
            Services.RAMP_UNDETECTED,
        )
        self.ramp_detected_client = self.create_client(
            WaitForEvent,
            Services.RAMP_DETECTED,
        )
        self.sign_undetected_client = self.create_client(
            WaitForEvent,
            Services.SIGN_UNDETECTED,
        )
        self.ramp_close_client = self.create_client(
            WaitForEvent,
            Services.RAMP_CLOSE,
        )

        self.caller_lock = Lock()
        self.caller_busy = False
        self.current_service_name = None
        self.current_future = None

        # Store original terminal settings for keyboard input
        self.original_settings = None
        if sys.stdin.isatty():
            self.original_settings = termios.tcgetattr(sys.stdin)

        # Setup non-blocking keyboard input
        self._setup_keyboard_input()

        # Change timer to 0.1 seconds for responsive keyboard input
        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Event Caller Node initialized")
        self.get_logger().info("Commands:")
        self.get_logger().info("  r - Call ramp_undetected service")
        self.get_logger().info("  R - Call ramp_detected service")
        self.get_logger().info("  s - Call sign_undetected service (sign_id=0)")
        self.get_logger().info("  q - Quit")

    def _setup_keyboard_input(self):
        """Setup non-blocking keyboard input"""
        if sys.stdin.isatty():
            tty.setraw(sys.stdin.fileno())

    def _restore_terminal(self):
        """Restore original terminal settings"""
        if self.original_settings and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)

    def _get_keyboard_input(self) -> Optional[str]:
        """Get keyboard input in non-blocking manner"""
        if not sys.stdin.isatty():
            return None

        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def timer_callback(self):
        with self.caller_lock:
            # First check if caller is busy
            if self.caller_busy:
                # We're waiting for a service response, don't process new commands
                return

            # Caller is free, check for keyboard commands
            key = self._get_keyboard_input()

            if key is None:
                return

            # Handle keyboard commands
            if key == "r":
                self._call_ramp_undetected_service()
            elif key == "R":
                self._call_ramp_detected_service()
            elif key == "s":
                self._call_sign_undetected_service()
            elif key == "C":
                self._call_ramp_close_service()
            elif key == "q":
                self.get_logger().info("Quitting...")
                self._restore_terminal()
                rclpy.shutdown()
            else:
                self.get_logger().info(f"Unknown command: {key}")

    def _call_ramp_undetected_service(self):
        """Call the ramp undetected service"""
        if not self.ramp_undetected_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Ramp undetected service not available")
            return

        self.get_logger().info("Calling ramp_undetected service...")

        # You can handle the request creation here
        request = WaitForEvent.Request()
        request.event_type = "ramp_undetected"
        request.param_names = ["timeout"]
        request.param_values = ["30.0"]

        # Mark caller as busy
        self.caller_busy = True
        self.current_service_name = "ramp_undetected"

        # Call service asynchronously
        self.current_future = self.ramp_undetected_client.call_async(request)
        self.current_future.add_done_callback(self._service_response_callback)

    def _call_ramp_detected_service(self):
        """Call the ramp detected service"""
        if not self.ramp_detected_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Ramp detected service not available")
            return

        self.get_logger().info("Calling ramp_detected service...")

        # You can handle the request creation here
        request = WaitForEvent.Request()
        request.event_type = "ramp_detected"
        request.param_names = ["timeout"]
        request.param_values = ["30.0"]

        # Mark caller as busy
        self.caller_busy = True
        self.current_service_name = "ramp_detected"

        # Call service asynchronously
        self.current_future = self.ramp_detected_client.call_async(request)
        self.current_future.add_done_callback(self._service_response_callback)

    def _call_ramp_close_service(self):
        """Call the ramp close service"""
        if not self.ramp_close_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Ramp close service not available")
            return

        self.get_logger().info("Calling ramp_close service...")

        # You can handle the request creation here
        request = WaitForEvent.Request()
        request.event_type = "ramp_close"
        request.param_names = [
            "distance_threshold",
            "min_lines",
            "timeout",
            "ramp_type",
        ]
        request.param_values = ["0.75", "0", "30.0", "farthest"]

        # Mark caller as busy
        self.caller_busy = True
        self.current_service_name = "ramp_close"

        # Call service asynchronously
        self.current_future = self.ramp_close_client.call_async(request)
        self.get_logger().info("Calling ramp_close service...")

        # Mark caller as busy
        self.caller_busy = True
        self.current_service_name = "ramp_close"

        # Call service asynchronously
        self.current_future = self.ramp_close_client.call_async(request)
        self.current_future.add_done_callback(self._service_response_callback)

    def _call_sign_undetected_service(self):
        """Call the sign undetected service with sign_id=7"""
        if not self.sign_undetected_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Sign undetected service not available")
            return

        self.get_logger().info("Calling sign_undetected service (sign_id=0)...")

        # You can handle the request creation here
        request = WaitForEvent.Request()
        request.event_type = "sign_undetected"
        request.param_names = ["sign_id", "timeout"]
        request.param_values = ["4", "60.0"]

        # Mark caller as busy
        self.caller_busy = True
        self.current_service_name = "sign_undetected"

        # Call service asynchronously
        self.current_future = self.sign_undetected_client.call_async(request)
        self.current_future.add_done_callback(self._service_response_callback)

    def _service_response_callback(self, future):
        """Callback for when service call completes"""
        with self.caller_lock:
            service_name = self.current_service_name

            try:
                response = future.result()
                self.get_logger().info(
                    f"Service {service_name} completed: "
                    f"success={response.success}, message='{response.message}'"
                )
            except Exception as e:
                self.get_logger().error(f"Service {service_name} failed: {str(e)}")
            finally:
                # Mark caller as free
                self.caller_busy = False
                self.current_service_name = None
                self.current_future = None

                self.get_logger().info("Event caller is now free for new commands")

    def destroy_node(self):
        """Clean shutdown"""
        self._restore_terminal()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EventCaller()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
