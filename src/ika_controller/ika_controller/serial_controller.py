#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import Jetson.GPIO as GPIO


class SerialController(Node):
    def __init__(self):
        super().__init__("serial_controller")
        self.mode = None
        self.joy_cmd = None
        self.auto_cmd = None
        self.command_str = ""

        # GPIO Setup
        self.gpio_pin = 7  # Using board pin 7 (GPIO 216) - you can change this
        GPIO.setmode(GPIO.BOARD)  # Use board pin numbering
        GPIO.setup(self.gpio_pin, GPIO.OUT)
        GPIO.output(self.gpio_pin, GPIO.LOW)  # Initialize to LOW
        self.get_logger().info(f"GPIO pin {self.gpio_pin} initialized to LOW")

        self.joy_cmd_sub = self.create_subscription(
            String, "/ika_controller/joy_cmd", self.joy_cmd_callback, 10
        )
        self.auto_cmd_sub = self.create_subscription(
            String, "/ika_controller/auto_cmd", self.auto_cmd_callback, 10
        )

        for option in ["ACM0", "USB0", "USB1"]:
            self.serial_port = f"/dev/tty{option}"  # Default serial port
            try:
                self.socket = serial.Serial(self.serial_port, 115200, timeout=10)
                break
            except serial.SerialException as e:
                self.get_logger().error(
                    f"Failed to open serial port {self.serial_port}: {e}"
                )
                self.socket = None

    def joy_cmd_callback(self, msg: String):
        self.joy_cmd = msg.data
        self.get_logger().info(f"Joy command set to: {self.joy_cmd}")
        self.mode = self.joy_cmd.data[0]

    def auto_cmd_callback(self, msg: String):
        self.auto_cmd = msg.data
        self.get_logger().info(f"Auto command set to: {self.auto_cmd}")

    def control_gpio_from_joy_cmd(self):
        """Control GPIO pin based on last character of joy_cmd"""
        if self.joy_cmd is None or len(self.joy_cmd) == 0:
            return

        last_char = self.joy_cmd[-1]  # Get last character

        if last_char == "1":
            GPIO.output(self.gpio_pin, GPIO.HIGH)
            self.get_logger().info(f"GPIO pin {self.gpio_pin} set to HIGH")
        elif last_char == "0":
            GPIO.output(self.gpio_pin, GPIO.LOW)
            self.get_logger().info(f"GPIO pin {self.gpio_pin} set to LOW")
        else:
            self.get_logger().warn(
                f"Invalid GPIO control character: {last_char} (expected '0' or '1')"
            )

    def send_serial_command(self):
        if self.mode is None:
            self.get_logger().warn("Mode not set, cannot send command.")
            return

        if self.mode == "M":
            self.command_str = self.joy_cmd
        elif self.mode == "A":
            self.command_str = self.auto_cmd
        elif self.mode == "E":
            self.command_str = f"E{self.joy_cmd[1:]}"

        if self.socket.is_open:
            try:
                self.socket.write(self.command_str.encode())
                self.get_logger().info(f"Sent command: {self.command_str}")
            except serial.SerialException as e:
                self.get_logger().error(f"Failed to send command: {e}")
        else:
            self.get_logger().error("Serial port is not open, cannot send command.")

        # Control GPIO based on last character of joystick message
        self.control_gpio_from_joy_cmd()

    def __del__(self):
        """Cleanup GPIO when node is destroyed"""
        try:
            GPIO.cleanup()
            self.get_logger().info("GPIO cleanup completed")
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SerialController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
