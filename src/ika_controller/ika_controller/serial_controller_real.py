#!/usr/bin/env python3
import rclpy
from rake_core.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from rclpy.time import Time
from rclpy.constants import S_TO_NS
import numpy as np
from geometry_msgs.msg import Twist
import math
from tf_transformations import quaternion_from_euler
from ika_msgs.msg import MotorFeedback, IsInWater
import json
from types import SimpleNamespace
from rake_msgs.msg import ResetCommand, ShootCommand
from rake_core.states import SystemStateEnum
import struct
import serial


class OdometryNodeConfig:
    def __init__(self, wheel_radius=0.125, wheel_separation=0.92):
        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation
        self.odom_period = 0.05
        self.command_send_period = 0.1


class OdometryNode(Node):
    def __init__(self):
        super().__init__("odom_node")

        self.v = 0.0
        self.w = 0.0

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().info(
                "No config found, using default parameters for odom node"
            )
            return

        # Initialize Serial Communication
        self.init_serial_communication()

        # Define Constants
        self.rpm_to_omega = 2 * math.pi / 6000

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
        # State Variables
        self.reset_cmd = ResetCommand()
        self.shoot_cmd = ShootCommand()
        self.cmd_vel = Twist()

        # Define Motor Feedback
        self.motor_feedback_msg = MotorFeedback()
        # Publishers
        self.motor_feedback_pub = self.create_publisher(
            MotorFeedback, "/ika_controller/motor_feedback", 10
        )
        # Subscriptions
        self.vel_sub_ = self.create_subscription(
            Twist, "/ika_nav/cmd_vel", self.vel_callback, 10
        )
        self.reset_cmd_sub = self.create_subscription(
            ResetCommand, "/ika_nav/reset_command", self.reset_callback, 10
        )
        self.shoot_cmd_sub = self.create_subscription(
            ShootCommand, "/ika_nav/shoot_command", self.shoot_callback, 10
        )
        # Timers
        self.create_timer(self.config.odom_period, self.read_serial)
        self.create_timer(self.config.command_send_period, self.send_command)

        self.get_logger().info("Odom node has been started")

    def get_default_config(self):
        return json.loads(json.dumps(OdometryNodeConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def init_serial_communication(self):
        for option in ["USB1", "USB0", "ACM0"]:
            self.serial_port = f"/dev/tty{option}"  # Default serial port
            try:
                self.socket = serial.Serial(self.serial_port, 115200, timeout=10)
                break
            except serial.SerialException as e:
                self.get_logger().info(
                    f"Failed to open serial port {self.serial_port}: {e}"
                )
        # send 0 command for communication initializiation
        self.socket.write(b"M00.0000.000000CF")

    def vel_callback(self, msg):
        self.cmd_vel = msg

    def reset_callback(self, msg):
        self.reset_cmd = msg

    def shoot_callback(self, msg):
        self.shoot_cmd = msg

    def read_serial(self):
        try:
            data = self.socket.read(size=25)
            flv = float(struct.unpack("i", data[0:4])[0]) * self.rpm_to_omega
            frv = -float(struct.unpack("i", data[4:8])[0]) * self.rpm_to_omega
            mlv = float(struct.unpack("i", data[8:12])[0]) * self.rpm_to_omega
            # rlv = float(struct.unpack("h", data[10:12])[0]) * self.rpm_to_omega
            mrv = -float(struct.unpack("i", data[12:16])[0]) * self.rpm_to_omega
            rlv = float(struct.unpack("i", data[16:20])[0]) * self.rpm_to_omega
            rrv = -float(struct.unpack("i", data[20:24])[0]) * self.rpm_to_omega
            # rrv = float(struct.unpack("h", data[22:24])[0]) * self.rpm_to_omega
            self.is_in_water = int(struct.unpack("h", data[24:25]))

            v_r = (frv + mrv + rrv) / 3
            v_l = (flv + mlv + rlv) / 3
            v = (v_r * self.wheel_radius + v_l * self.wheel_radius) / 2
            omega = (
                v_r * self.wheel_radius - v_l * self.wheel_radius
            ) / self.wheel_separation

            self.motor_feedback_msg.v = v
            self.motor_feedback_msg.omega = omega
            self.motor_feedback_msg.dt = self.config.odom_period

            self.get_logger().info(
                f"{v_r * self.wheel_radius}, {v_l * self.wheel_radius}"
            )

            is_in_water_msg = IsInWater()
            is_in_water_msg.is_in_water = True if self.is_in_water else False
            self.is_in_water_pub.publish(is_in_water_msg)
            self.motor_feedback_pub.publish(self.motor_feedback_msg)

        except TimeoutError as e:
            self.get_logger().info(f"Socket timed out: {e}")
        except ValueError as e:
            self.get_logger().info(f"Cannot read odom value: {e}")
        except Exception as e:
            self.get_logger().info(f"Odom reading error: {e}")

    def send_command(self):
        if self.mobility == 0:
            command_str = "M00.0000.000000CF"
            self.socket.write(command_str.encode())
            return

        v_cmd, w_cmd = self.cmd_vel.linear.x, self.cmd_vel.angular.z
        v_r, right_direct, v_l, left_direct = self.compute_wheel_velocities(
            v_cmd, w_cmd
        )
        reset_on = self.reset_cmd.reset_on
        laser_on, tilt_dir, pan_dir = (
            self.shoot_cmd.laser_on,
            self.shoot_cmd.tilt_dir,
            self.shoot_cmd.pan_dir,
        )
        command_str = self.generate_command_string(
            v_r, right_direct, v_l, left_direct, laser_on, tilt_dir, pan_dir, reset_on
        )

        self.socket.write(command_str.encode())

    def compute_wheel_velocities(self, v_cmd, w_cmd):
        v = np.array([[v_cmd], [w_cmd]])
        transform_matrix = np.array([[1.0, 0.46], [1.0, -0.46]])
        wheel_velocities = transform_matrix @ v

        wheel_vel_right = round(wheel_velocities[0, 0], 2)
        wheel_vel_left = round(wheel_velocities[1, 0], 2)

        right_direct = 1 if wheel_vel_right > 0 else 0
        left_direct = 1 if wheel_vel_left > 0 else 0

        max_actual = max(abs(wheel_vel_right), abs(wheel_vel_left))

        if max_actual > 1.0:
            scale = 1.0 / max_actual
            wheel_vel_right *= scale
            wheel_vel_left *= scale

        return wheel_vel_right, right_direct, wheel_vel_left, left_direct

    def generate_command_string(
        self, v_r, right_direct, v_l, left_direct, laser_on, tilt_dir, pan_dir, reset_on
    ):
        mode_char = "M"
        if (
            self.system_state == SystemStateEnum.MANUAL
            or self.system_state == SystemStateEnum.ACTION_CONTROL_TEST
        ):
            mode_char = "M"
        elif self.system_state in [
            SystemStateEnum.SEMI_AUTONOMOUS,
            SystemStateEnum.IDLE_AUTONOMOUS,
            SystemStateEnum.SLOW_AUTONOMOUS,
            SystemStateEnum.FAST_AUTONOMOUS,
            SystemStateEnum.CAUTIOUS_AUTONOMOUS,
            SystemStateEnum.SHARP_TURN_AUTONOMOUS,
            SystemStateEnum.TARGET_HITTING,
            SystemStateEnum.ACTION_CONTROLLED,
        ]:
            mode_char = "A"
        else:
            mode_char = "E"

        command_str = f"{mode_char}{right_direct}{v_r}{left_direct}{v_l}{tilt_dir}{pan_dir}{laser_on}{reset_on}CF"
        return command_str


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
