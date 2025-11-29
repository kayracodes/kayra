#!/usr/bin/env python3
import pygame
from pygame.locals import *
import time
import pygame.locals
from threading import Thread
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


WHEEL_SEPERATION = 0.93
WHEEL_RADIUS = 0.125
PI = 3.14159
ANG_VEL_THRESHOLD = 0.05
LIN_VEL_THRESHOLD = 0.05


class JoyController(Node):
    def __init__(self):
        super().__init__("joy_controller")
        self.mode = "M"
        self.tilt_on = False
        self.pan_on = False
        self.laser_on = False
        self.tilt_dir = "0"
        self.pan_dir = "0"
        self.command_str = ""
        self.lin_vel = 0
        self.ang_vel = 0
        self.left_direct = 0
        self.right_direct = 0
        self.wheel_vel_right = 0
        self.wheel_vel_left = 0
        self.reset_card = False
        self.last_reset = time.time()

        self.check_connection()

        # Create QoS profile for reliable network communication
        # qos_profile = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=DurabilityPolicy.TRANSIENT_LOCAL,
        #     depth=10,
        # )

        # Mode Pub
        self.mode_pub = self.create_publisher(String, "/ika_controller/mode", 10)
        self.cmd_pub = self.create_publisher(String, "/ika_controller/joy_cmd", 10)
        self.timer = self.create_timer(0.1, self.publish_loop)

        self.get_logger().info("JoyController initialized")

    def check_connection(self):
        pygame.init()
        pygame.joystick.init()
        self.joy_sticks = [
            pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())
        ]

        is_device_found = False
        while len(self.joy_sticks) == 0:
            print("No joy sticks are connected, please plug in a joystick")

            for event in pygame.event.get():
                if event == pygame.JOYDEVICEADDED:
                    joy_index = event
                    self.joy_sticks.append(pygame.joystick.Joystick(joy_index))
                    is_device_found = True
            if is_device_found:
                print("Joy stick device initialized")
            time.sleep(1)
        else:
            print("Joy device found")

    def publish_loop(self):
        for event in pygame.event.get():
            if event.type == JOYBUTTONDOWN:
                if event.button == 2:
                    self.mode = "M"  # manuel
                    # print("Button 2 pressed, manuel mode")
                elif event.button == 0:
                    self.mode = "A"  # auto
                    # print("Button 0 pressed, autonomous mode")
                elif event.button == 3:
                    self.mode = "E"

                elif event.button == 1:
                    self.laser_on = not self.laser_on

                elif event.button == 5:
                    self.reset_card = True

            elif event.type == JOYHATMOTION:
                # Handle D-pad (hat) input for tilt and pan control
                hat_x, hat_y = event.value

                # Handle pan (left/right movement)
                if hat_x == -1:  # Left
                    self.tilt_on = True
                    self.tilt_dir = "1"  # Left
                elif hat_x == 1:  # Right
                    self.tilt_on = True
                    self.pan_dir = "0"  # Right
                else:
                    self.tilt_on = False
                    self.tilt_dir = "0"  # Stop

                # Handle tilt (up/down movement)
                if hat_y == 1:  # Up
                    self.pan_on = True
                    self.pan_dir = "1"  # Up
                elif hat_y == -1:  # Down
                    self.pan_on = True
                    self.pan_dir = "0"  # Down
                else:
                    self.pan_on = False
                    self.pan_dir = "0"  # Stop

            elif event.type == JOYAXISMOTION:
                if event.axis == 1:
                    lin_val = -event.value
                    if abs(lin_val) <= 0.05:
                        self.lin_vel = 0
                    elif lin_val > 0:
                        self.lin_vel = round(lin_val, 2)
                    elif lin_val < 0:
                        self.lin_vel = round(lin_val, 2)
                elif event.axis == 3:
                    ang_val = -event.value
                    if abs(ang_val) <= 0.05:
                        self.ang_vel = 0
                    elif ang_val > 0:
                        self.ang_vel = round(ang_val, 2)
                    elif ang_val < 0:
                        self.ang_vel = round(ang_val, 2)

        self.compute_wheel_velocities()

        self.command_str = self.build_command_string()
        self.get_logger().info(f"Command: {self.command_str}")

        # Publish with debug info
        msg = String(data=self.command_str)
        self.cmd_pub.publish(msg)
        self.get_logger().debug(
            f"Published to /ika_controller/joy_cmd: {self.command_str}"
        )

        mode_msg = String(data=self.mode)
        self.mode_pub.publish(mode_msg)
        self.get_logger().debug(f"Published to /ika_controller/mode: {self.mode}")

    def compute_wheel_velocities(self):
        v = np.array([[self.lin_vel], [self.ang_vel]])
        transform_matrix = np.array([[1.0, 0.46], [1.0, -0.46]])
        wheel_velocities = transform_matrix @ v

        self.wheel_vel_right = round(wheel_velocities[0, 0], 2)
        self.wheel_vel_left = round(wheel_velocities[1, 0], 2)

        self.right_direct = 1 if self.wheel_vel_right > 0 else 0
        self.left_direct = 1 if self.wheel_vel_left > 0 else 0

        max_actual = max(abs(self.wheel_vel_right), abs(self.wheel_vel_left))

        if max_actual > 1.0:
            scale = 1.0 / max_actual
            self.wheel_vel_right *= scale
            self.wheel_vel_left *= scale

    # UTILITY FUNCTIONS
    def build_command_string(self):
        # Build command string with mode, movement, tilt, pan, and laser info
        # mode + laser_on + tilt_on + tilt_dir + pan_on + pan_dir
        mode = self.mode
        laser_on = "1" if self.laser_on else "0"
        tilt_on = "1" if self.tilt_on else "0"
        tilt_dir = self.tilt_dir if self.tilt_on else "0"
        pan_on = "1" if self.pan_on else "0"
        pan_dir = self.pan_dir if self.pan_on else "0"
        command_str = f"{mode}{laser_on}{tilt_on}{tilt_dir}{pan_on}{pan_dir}"

        if self.reset_card == True:
            if time.time() - self.last_reset < 2.0:
                self.reset_card = False
            else:
                self.last_reset = time.time()

        reset_card = "1" if self.reset_card else "0"

        tilt_cmd, pan_cmd = "", ""
        if tilt_on == "0":
            tilt_cmd = "0"
        elif tilt_dir == "1":
            tilt_cmd = "1"
        elif tilt_dir == "0":
            tilt_cmd = "2"

        if pan_on == "0":
            pan_cmd = "0"
        elif pan_dir == "1":
            pan_cmd = "1"
        elif pan_dir == "0":
            pan_cmd = "2"
            pan_cmd = "2"

        command_str = f"{self.mode}{self.right_direct}{abs((self.wheel_vel_right)):.2f}{self.left_direct}{(abs(self.wheel_vel_left)):.2f}{tilt_cmd}{pan_cmd}{laser_on}{reset_card}CF"
        if self.lin_vel < 0:
            command_str = f"{self.mode}{self.right_direct}{abs((self.wheel_vel_left)):.2f}{self.left_direct}{abs((self.wheel_vel_right)):.2f}{tilt_cmd}{pan_cmd}{laser_on}{reset_card}CF"
        return command_str


if __name__ == "__main__":
    rclpy.init()
    joy_controller = JoyController()
    rclpy.spin(joy_controller)
    joy_controller.destroy_node()
    rclpy.shutdown()
