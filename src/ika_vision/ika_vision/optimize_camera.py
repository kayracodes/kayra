#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf_transformations import euler_matrix


class CameraOptimizer(Node):
    def __init__(self):
        super().__init__("camera_optimizer")

        self.bridge = CvBridge()

        self.best_roll = 0.0  # tilt from cam->laser: positive for up
        self.best_pitch = 0.0  # pan error from cam->laser: positive for right
        self.best_offset = 0.0  # y offset from cam->laser: positive for down
        self.shoot_img = None

        # Configuration Params
        self.max_roll = 2.0
        self.max_pitch = 2.0
        self.min_offset = 4.15
        self.max_offset = 4.75

        # Camera Coefficients
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

        # Subscribers
        self.shoot_cam_sub = self.create_subscription(
            Image, "/shoot_camera/image_raw", self.process_image, 10
        )

        self.R_l_c = euler_matrix(np.radians(-0.4), 0.0, np.radians(1.0))[
            :3, :3
        ].astype(np.float32)
        laser_offset = 4.45
        self.t_vec_l_c = np.array([[0.0], [laser_offset], [0.0]], dtype=np.float32)

    def process_image(self, msg):
        self.shoot_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def return_optimal_configuration(self, img):
        best_roll, best_pitch, best_offset = 0.0, 0.0, 0.0
        best_err = float("inf")
        # Implement your optimization logic here
        for laser_offset in np.arange(self.min_offset, self.max_offset, 0.1):
            for pitch in np.arange(-self.max_pitch, self.max_pitch, 0.1):
                for roll in np.arange(-self.max_roll, self.max_roll, 0.1):
                    # Set the variables
                    R_l_c = euler_matrix(
                        np.radians(roll),
                        np.radians(pitch),
                    )[
                        :3, :3
                    ].astype(np.float32)
                    t_vec_l_c = np.array(
                        [[0.0], [laser_offset], [0.0]], dtype=np.float32
                    )

                    # find the error in current setup
                    err = self._get_errors_from_image(img, R_l_c, t_vec_l_c)
                    if err is not None and err < best_err:
                        best_err = err
                        best_roll = roll
                        best_pitch = pitch
                        best_offset = laser_offset

        return best_roll, best_pitch, best_offset

    # UTILITY FUNCTİONS
    def _detect_target(self, img):
        """Target detection function (exact copy from shoot_controller)"""
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
            minRadius=15,
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

            if self.is_within_sign(center):
                # ignore targets that are detected within road signs
                cv2.circle(img, center, radius, (255, 0, 0), 4)
                continue

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

    def _world_coordinates(self, center, radius):
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

    def _return_tilt_and_pan(self, t_vec_t_c, t_vec_l_c, R_l_c):
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

    def _get_errors_from_image(self, img, R_l_c, t_vec_l_c):
        avg_center, avg_radius, _ = self._detect_target(img)
        if avg_center is None or avg_radius is None:
            self.get_logger().info("No valid target detected.")
            return None
        _, t_vec_t_c = self._world_coordinates(avg_center, avg_radius)

        if t_vec_t_c is None:
            self.get_logger().info("No valid target detected.")
            return None
        tilt_err, pan_err = self._return_tilt_and_pan(t_vec_t_c, t_vec_l_c, R_l_c)

        err = tilt_err**2 + pan_err**2
        return err

    def camera_optimization_cycle(self):
        import sys, select, termios, tty

        self.get_logger().info(
            "Camera optimization cycle started. Press 'o' to optimize, 'q' to quit."
        )
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == "o":
                        if self.shoot_img is not None:
                            roll, pitch, offset = self.return_optimal_configuration(
                                self.shoot_img
                            )
                            self.get_logger().info(
                                f"Optimal config: roll={roll:.2f}, pitch={pitch:.2f}, offset={offset:.2f}"
                            )
                        else:
                            self.get_logger().info("No image received yet.")
                    elif key == "q":
                        self.get_logger().info("Shutting down node.")
                        rclpy.shutdown()
                        break
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = CameraOptimizer()
    node.camera_optimization_cycle()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
