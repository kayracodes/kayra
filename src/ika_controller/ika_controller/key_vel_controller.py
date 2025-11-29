#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import readchar
import threading
import time


class KeyVelocityController(Node):
    def __init__(self):
        super().__init__("key_vel_controller")

        # Create publisher for velocity commands
        self.vel_publisher = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)

        # Initialize velocity values
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.linear_increment = 0.2  # Speed increment (m/s)
        self.angular_increment = 0.5  # Angular increment (rad/s)

        # Max speeds
        self.max_linear_speed = 2.0  # m/s
        self.max_angular_speed = 2.0  # rad/s

        # Keys for control
        self.control_keys = {
            "w": "forward",
            "s": "stop",
            "a": "rotate_left",
            "d": "rotate_right",
            "x": "backward",
            "q": "quit",
        }

        # Create timer for publishing at 10Hz
        self.timer = self.create_timer(0.1, self.publish_velocity)

        # Flag to control the input thread
        self.running = True

        # Start keyboard input thread
        self.input_thread = threading.Thread(target=self.read_keyboard_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        # Print instructions
        self.get_logger().info(
            """
        Key Velocity Controller Started
        -------------------------------
        W - Forward
        S - Backward
        A - Rotate Left
        D - Rotate Right
        X - Stop
        Q - Quit
        """
        )

    def read_keyboard_input(self):
        """Thread function to read keyboard input"""
        while self.running:
            try:
                # Read a single character (non-blocking)
                key = readchar.readchar().lower()

                if key in self.control_keys:
                    self.process_key(key)

                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"Error reading keyboard input: {e}")

    def process_key(self, key):
        """Process the pressed key and update velocities"""
        action = self.control_keys[key]

        if action == "forward":
            self.linear_speed = min(
                self.linear_speed + self.linear_increment, self.max_linear_speed
            )
            # self.get_logger().info(f"Forward: {self.linear_speed:.2f} m/s")

        elif action == "backward":
            self.linear_speed = max(
                self.linear_speed - self.linear_increment, -self.max_linear_speed
            )
            # self.get_logger().info(f"Backward: {self.linear_speed:.2f} m/s")

        elif action == "rotate_left":
            self.angular_speed = min(
                self.angular_speed + self.angular_increment, self.max_angular_speed
            )
            # self.get_logger().info(f"Rotate Left: {self.angular_speed:.2f} rad/s")

        elif action == "rotate_right":
            self.angular_speed = max(
                self.angular_speed - self.angular_increment, -self.max_angular_speed
            )
            # self.get_logger().info(f"Rotate Right: {self.angular_speed:.2f} rad/s")

        elif action == "stop":
            self.linear_speed = 0.0
            self.angular_speed = 0.0
            self.get_logger().info("Stopped")

        elif action == "quit":
            self.get_logger().info("Shutting down...")
            self.running = False
            rclpy.shutdown()

        self.get_logger().info(
            f"v: {self.linear_speed:.2f} m/s, {self.angular_speed:.2f} rad/s, R={self.linear_speed / (self.angular_speed + 0.00001)}"
        )

    def publish_velocity(self):
        """Publish current velocity commands at regular intervals"""
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed
        self.vel_publisher.publish(twist)

    def destroy_node(self):
        """Clean up resources before shutdown"""
        self.running = False
        if self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyVelocityController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
