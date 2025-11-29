#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum
from ika_msgs.msg import RoadSignArray, RoadSign
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import String
import serial
import struct
import math
import time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from tf_transformations import euler_matrix
from threading import Thread


class Target:
    def __init__(self, x_px=None, y_px=None, radius_px=None):
        self.x_px = x_px
        self.y_px = y_px
        self.radius_px = radius_px


class VelCmd:
    def __init__(self):
        self.v = 0.0
        self.omega = 0.0


class CameraShooter(Node):
    def __init__(self):
        super().__init__("camera_shooter")
        self.mode = None
        self.bridge = CvBridge()

        self.target = Target()
        # self.traffic_signs = RoadSignArray()
        self.laser_on = False
        self.tilt_complete = False
        self.pan_complete = False
        self.is_target_hit = False
        # Camera Parameters
        self.cam_matrix = np.array(
            [
                [1.44827013e03, 0.00000000e00, 9.38693779e02],
                [0.00000000e00, 1.44831655e03, 5.26119185e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            dtype=np.float32,
        )
        self.cx = int(self.cam_matrix[0, 2])
        self.cy = int(self.cam_matrix[1, 2])

        self.cam_dist_coeffs = np.array(
            [[0.05292199, -0.21002977, -0.00152838, -0.00097731, 0.2083675]],
            dtype=np.float32,
        )
        self.best_angles = [0.0, 0.0, 0.0]

        self.R_l_c = euler_matrix(np.radians(0.6), 0.0, np.radians(1.0))[:3, :3].astype(
            np.float32
        )
        laser_offset = 4.45
        laser_z_offset = 0.0  # Z-direction offset
        self.t_vec_l_c = np.array(
            [[0.0], [laser_offset], [laser_z_offset]], dtype=np.float32
        )
        self.best_laser_offset = laser_offset  # Store the best laser offset
        self.best_laser_z_offset = laser_z_offset  # Store the best laser z offset
        # Joy Stick Commands
        self.debug_img = None

        # Config params
        self.tilt_sensitivity = 0.1
        self.pan_sensitivity = 0.1
        self.road_signs = RoadSignArray()
        self.sign_id_map = {
            0: "1",
            1: "10",
            2: "11",
            3: "12",
            4: "2",
            5: "3",
            6: "4",
            7: "4-END",
            8: "5",
            9: "6",
            10: "7",
            11: "8",
            12: "9",
            13: "STOP",
        }
        self.last_sign = None

        self.left_direct = 0
        self.right_direct = 0
        self.wheel_vel_right = 0
        self.wheel_vel_left = 0

        self.auto_vel = VelCmd()

        # Serial Communication
        self.mode_sub = self.create_subscription(
            String, "/ika_controller/mode", self.mode_callback, 10
        )

        self.shoot_cam_sub = self.create_subscription(
            Image,
            "/shoot_camera/image_raw",
            self.target_camera_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_subscription(RoadSignArray, "/road_signs", self.signs_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, "/ika_nav/cmd_vel", self.cmd_vel_callback, 10
        )
        # Publishers
        self.debug_img_pub = self.create_publisher(
            Image,
            "/shoot_camera/debug_image",
            10,
        )

        self.auto_cmd_pub = self.create_publisher(
            String, "/ika_controller/auto_cmd", 10
        )

        self.create_timer(
            0.05, self.control_loop, callback_group=MutuallyExclusiveCallbackGroup()
        )

    def target_camera_callback(self, msg: Image):
        # if self.mode != "A":
        #     # self.get_logger().warn("Not in autonomous mode, ignoring camera frame.")
        #     return
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.undistort(cv_image, self.cam_matrix, self.cam_dist_coeffs)
        avg_center, avg_radius, debug_img = self.detect_target(cv_image)

        if avg_center is not None and avg_radius is not None:
            self.debug_img = debug_img
            self.target.x_px, self.target.y_px = avg_center
            self.target.radius_px = avg_radius
            self.debug_img_pub.publish(
                self.bridge.cv2_to_imgmsg(self.debug_img, "bgr8")
            )
        else:
            self.target.x_px = None
            self.target.y_px = None
            self.target.radius_px = None

    def road_signs_callback(self, msg: RoadSignArray):
        self.road_signs = msg
        if len(self.road_signs.signs) > 0:
            self.last_sign = self.sign_id_map[self.road_signs.signs[-1].id]
            self.get_logger().info(f"Last detected sign: {self.last_sign}")

    def mode_callback(self, msg):
        if msg.data in ["M", "A", "E"]:
            self.mode = msg.data
            self.get_logger().info(f"Mode changed to: {self.mode}")
        else:
            self.get_logger().warn(f"Invalid mode received: {msg.data}. Ignoring.")

    def cmd_vel_callback(self, msg: Twist):
        self.auto_vel.v = msg.linear.x
        self.auto_vel.omega = msg.angular.z

    def detect_target(self, img):
        cv2.circle(img, (self.cx, self.cy), 4, (0, 0, 255), -1)
        img_blur = cv2.GaussianBlur(img, (5, 5), 1.5)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        # HSV filter for white pixels: low saturation and high value
        lower = (0, 0, 41)  # H: any hue, S: low saturation, V: bright values
        upper = (
            180,
            255,
            255,
        )  # H: max hue in OpenCV (179), S: low saturation, V: max brightness
        mask = cv2.inRange(hsv, lower, upper)

        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=5,
            param1=100,
            param2=20,
            minRadius=16,
            maxRadius=20,
        )

        average_center = (0, 0)
        average_radius = 0.0
        if circles is None:
            self.get_logger().warn("No circles detected.")
            return None, None, img

        circles = np.uint16(np.around(circles))
        centers = []
        radii = []
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            # if abs(center[0] - 960) > 960 * 0.35:
            #     continue
            # if self.validate_target(center):
            centers.append(center)
            radii.append(radius)
            cv2.circle(img, center, 1, (0, 0, 255), 1)
            cv2.circle(img, center, radius, (0, 255, 0), 1)

        if len(centers) > 0:
            avg_x = float(np.mean([c[0] for c in centers]))
            avg_y = float(np.mean([c[1] for c in centers]))
            average_center = (avg_x, avg_y)
            # average_radius = float(np.mean(radii))

            average_radius = max(radii)
            cv2.circle(img, (int(avg_x), int(avg_y)), 1, (255, 0, 0), -1)
            # self.target_radius_px = average_radius
            # self.target_center_px = average_center

            # self.world_coordinates(self.center, self.radii)
            return (average_center, average_radius, img)
        return None, None, img

    def world_coordinates(self, center, radius):
        if self.cam_matrix is None or self.cam_dist_coeffs is None:
            self.get_logger().warn(
                "Camera parameters not set, cannot compute world coordinates."
            )
            return None, None

        cx, cy = center
        radius_px = radius
        real_radius = 9.0

        image_points = np.array(
            [
                [cx, cy],
                [cx + radius_px, cy],
                [cx - radius_px, cy],
                [cx, cy + radius_px],
                [cx, cy - radius_px],
            ],
            dtype=np.float32,
        )

        object_points = np.array(
            [
                [0, 0, 0],
                [real_radius, 0, 0],
                [-real_radius, 0, 0],
                [0, real_radius, 0],
                [0, -real_radius, 0],
            ],
            dtype=np.float32,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.cam_matrix,
            self.cam_dist_coeffs,
        )

        if success:
            # tvec represents the translation from the camera frame to the object's frame.
            # It gives the coordinates of the object's origin in the camera's coordinate system.
            x, y, z = tvec.ravel()
            self.get_logger().info(
                f"Target coordinates in Camera Frame: X={x:.2f}cm Y={y:.2f}cm Z={z:.2f}cm"
            )
            # t_vec_t_c = np.array([tvec[2], -tvec[0], -tvec[1]])
            return rvec, tvec
        else:
            self.get_logger().warn("solvePnP failed.")
            return None, None

    def control_loop(self):
        if self.mode != "A":
            return

        if self.last_sign == "12" and not self.is_target_hit:
            self.auto_vel.v = 0.0
            self.auto_vel.omega = 0.0
            self.get_logger().info("readying for target hitting, stopping robot.")

            if (
                self.target.x_px is None
                or self.target.y_px is None
                or self.target.radius_px is None
            ):
                self.pan_on = False
                self.tilt_on = False

                self.command_str = self.generate_serial_cmd()
                # self.get_logger().info(f"Command: {self.command_str}")

                self.get_logger().warn("waiting for new camera frame.")
                return
            rvec, tvec_t_c = self.world_coordinates(
                (self.target.x_px, self.target.y_px), self.target.radius_px
            )
            if rvec is None or tvec_t_c is None:
                # self.get_logger().warn("Failed to compute world coordinates.")
                return
            pan, tilt = self.return_tilt_and_pan(tvec_t_c, self.t_vec_l_c, self.R_l_c)
            if tilt is None or pan is None:
                self.get_logger().warn("Failed to compute tilt and pan.")
                return

            self.get_logger().info(f"tilt: {tilt:.2f}, pan: {pan:.2f}")

            if not self.tilt_complete:
                if abs(tilt) > self.tilt_sensitivity:
                    self.tilt_on = True
                    self.tilt_dir = "1" if tilt > 0.0 else "0"
                else:
                    self.get_logger().info("TILT COMPLETE!!!")
                    self.tilt_on = False
                    self.tilt_complete = True
            else:
                if abs(pan) > self.pan_sensitivity:
                    self.pan_on = True
                    self.pan_dir = "1" if pan < 0.0 else "0"
                else:
                    self.get_logger().info("PAN COMPLETE!!!")
                    self.pan_on = False
                    self.pan_complete = True

            if self.pan_complete and self.tilt_complete:
                # Last check before firing
                if abs(tilt) > self.tilt_sensitivity:
                    self.tilt_complete = False
                elif abs(pan) > self.pan_sensitivity:
                    self.pan_complete = False
                else:
                    self.get_logger().info("Target hit!")
                    self.laser_on = True
                    self.is_target_hit = True
                    self.pan_on = False
                    self.tilt_on = False
                    self.pan_complete = False
                    self.tilt_complete = False
                    self.is_target_hit = True
                    self.last_sign = None
                    self.get_logger().info("Target hit, resetting state.")

        else:
            self.compute_wheel_velocities()
            self.get_logger().info(
                "Sending auto cmd: {self.auto_vel.v:.2f}, {self.auto_vel.omega:.2f}"
            )

        self.command_str = self.generate_serial_cmd()

        self.auto_cmd_pub.publish(String(data=self.command_str))
        # self.get_logger().info(f"Command: {self.command_str}")

    # UTILITY FUNCTIONS

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

    def return_tilt_and_pan(self, t_vec_t_c, t_vec_l_c, R_l_c):
        if t_vec_t_c is None:
            self.get_logger().warn("No targets detected, cannot send commands.")
            return None, None

        # Convert pnp translation to camera_frame coordinates
        P_laser_to_target_in_cam = (t_vec_t_c - t_vec_l_c).reshape(3, 1)
        assert P_laser_to_target_in_cam.shape == (
            3,
            1,
        ), f"Expected shape (3, 1), got {P_laser_to_target_in_cam.shape}"

        P_t_l = np.dot(R_l_c, P_laser_to_target_in_cam)
        assert P_t_l.shape == (
            3,
            1,
        ), f"Expected shape (3, 1), got {P_t_l.shape}"

        self.get_logger().info(
            f"Target position in Laser Frame: X={P_t_l[0,0]:.2f}cm Y={P_t_l[1,0]:.2f}cm Z={P_t_l[2,0]:.2f}cm"
        )
        pan = math.atan2(P_t_l[0, 0], P_t_l[2, 0])
        tilt = math.atan2(P_t_l[1, 0], math.sqrt(P_t_l[0, 0] ** 2 + P_t_l[2, 0] ** 2))

        # tilt = 0.0 if abs(tilt) < 0.01 else tilt
        # pan = 0.0 if abs(pan) < 0.01 else pan

        tilt_in_deg = np.degrees(tilt)
        pan_in_deg = np.degrees(pan)

        self.get_logger().info(
            f"Tilt needed: {tilt_in_deg:.4f} deg, Pan needed: {pan_in_deg:.4f} deg"
        )

        return tilt_in_deg, pan_in_deg

    def generate_serial_cmd(self):
        mode = self.mode
        laser_on = "1" if self.laser_on else "0"
        tilt_on = "1" if self.tilt_on else "0"
        tilt_dir = self.tilt_dir if self.tilt_on else "0"
        pan_on = "1" if self.pan_on else "0"
        pan_dir = self.pan_dir if self.pan_on else "0"
        # command_str = f"{mode}{laser_on}{tilt_on}{tilt_dir}{pan_on}{pan_dir}"

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

        # wheel cmd part
        if self.auto_vel.v >= 0:
            cmd_str = f"{self.right_direct}{abs((self.wheel_vel_right)):.2f}{self.left_direct}{(abs(self.wheel_vel_left)):.2f}"
        else:
            cmd_str = f"{self.right_direct}{abs((self.wheel_vel_left)):.2f}{self.left_direct}{abs((self.wheel_vel_right)):.2f}"

        return f"{self.mode}{cmd_str}{tilt_cmd}{pan_cmd}{laser_on}CF"

    def optimize_camera(self):
        time.sleep(3)
        self.get_logger().info("Starting camera optimization...")

        if self.target.x_px is None or self.target.y_px is None:
            self.get_logger().warn("Target not detected, cannot optimize camera.")
            return
        rvec, tvec_t_c = self.world_coordinates(
            (self.target.x_px, self.target.y_px), self.target.radius_px
        )
        if rvec is None or tvec_t_c is None:
            self.get_logger().warn("Failed to compute world coordinates.")
            return

        self.get_logger().info(
            f"Using the PnP solution {tvec_t_c[0, 0]:.2f}cm {tvec_t_c[1, 0]:.2f}cm {tvec_t_c[2, 0]:.2f}cm"
        )

        min_error = float("inf")
        best_P_t_l = np.zeros((3, 1), dtype=np.float32)
        best_R_l_c = np.eye(3, dtype=np.float32)
        best_t_vec_l_c = self.t_vec_l_c.copy()

        self.get_logger().info(
            "Starting laser offset, z-offset and angle optimization..."
        )

        # Iterate over laser offset values (y-direction) and z-direction offset
        for laser_offset in np.arange(3.5, 5.0, 0.1):  # Search range for laser offset
            for laser_z_offset in np.arange(0.5, 2.0, 0.1):  # Search range for z offset
                t_vec_l_c_current = np.array(
                    [[0.0], [laser_offset], [laser_z_offset]], dtype=np.float32
                )

                for roll in np.arange(-3.0, 3.0, 0.1):
                    for pitch in np.arange(-3.0, 3.0, 0.1):
                        for yaw in np.arange(-3.0, 3.0, 0.1):
                            R_l_c = euler_matrix(roll, pitch, yaw)[:3, :3].astype(
                                np.float32
                            )
                            P_laser_to_target_in_cam = (
                                tvec_t_c - t_vec_l_c_current
                            ).reshape(3, 1)
                            P_t_l = np.dot(R_l_c, P_laser_to_target_in_cam)

                            assert P_t_l.shape == (
                                3,
                                1,
                            ), f"Expected shape (3, 1), got {P_t_l.shape}"

                            # Calculate error without modifying P_t_l
                            # Assuming you want to minimize x and y components (keep only z)
                            error = np.sqrt(P_t_l[0, 0] ** 2 + P_t_l[1, 0] ** 2)

                            if error < min_error:
                                min_error = error
                                best_R_l_c = R_l_c.copy()  # Important: make a copy
                                best_P_t_l = P_t_l.copy()  # Important: make a copy
                                best_t_vec_l_c = (
                                    t_vec_l_c_current.copy()
                                )  # Store best laser offset
                                self.best_angles = [roll, pitch, yaw]
                                self.best_laser_offset = (
                                    laser_offset  # Update best laser offset
                                )
                                self.best_laser_z_offset = (
                                    laser_z_offset  # Update best laser z offset
                                )
                                self.get_logger().info(
                                    f"New best - Y_Offset={laser_offset:.2f}, Z_Offset={laser_z_offset:.2f}, Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}, Error={error:.4f}"
                                )
                                self.get_logger().info(
                                    f"Best P_t_l: [{P_t_l[0,0]:.2f}, {P_t_l[1,0]:.2f}, {P_t_l[2,0]:.2f}]"
                                )

        # Update class variables with best results
        self.R_l_c = best_R_l_c
        self.t_vec_l_c = best_t_vec_l_c  # Update the laser offset vector

        self.get_logger().info(f"Final Best P_t_l:\n{best_P_t_l}")
        self.get_logger().info(f"Final Best R_l_c:\n{self.R_l_c}")
        self.get_logger().info(
            f"Final Best Laser Y-Offset: {self.best_laser_offset:.2f}cm"
        )
        self.get_logger().info(
            f"Final Best Laser Z-Offset: {self.best_laser_z_offset:.2f}cm"
        )
        self.get_logger().info(f"Final Best t_vec_l_c:\n{self.t_vec_l_c}")
        self.get_logger().info(f"Final minimum error: {min_error:.4f}")

        self.get_logger().info(
            f"Target position in Laser Frame: X={best_P_t_l[0,0]:.2f}cm Y={best_P_t_l[1,0]:.2f}cm Z={best_P_t_l[2,0]:.2f}cm"
        )


# 1 sağ 2 sola x'te
# y'de 1 yukarı 2 aşağı
def main(args=None):
    rclpy.init(args=args)
    node = CameraShooter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
