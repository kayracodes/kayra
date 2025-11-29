#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from ika_msgs.msg import RoadSign, RoadSignArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
from tf_transformations import euler_from_matrix, quaternion_matrix
from tf2_ros import TransformListener, Buffer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import math


class ShootCameraControllerConfig:
    def __init__(self):
        self.cam_topic = "/shoot_camera/image_raw/compressed"
        self.cam_frame_id = "shoot_camera"
        self.cam_width = 1920
        self.cam_height = 1080
        self.cam_fps = 8


class ShootCameraController(Node):
    def __init__(self):
        super().__init__("shoot_camera_controller")
        self.conf = ShootCameraControllerConfig()
        self.bridge = CvBridge()
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        self.cam_matrix = None
        self.cam_dist_coeffs = None
        self.target_center_px = None
        self.target_radius_px = None

        self.current_tilt = 0.0
        self.tilt_cmd = None
        self.current_pan = 0.0
        self.pan_cmd = None

        self.t_vec_l_c = np.array(
            [[0.0], [0.0], [0.02]], dtype=np.float64
        )  # 1 unit along z-axis
        # self.r_vec_t_c = None
        self.t_vec_t_c = None

        self.debug_img = None

        self.road_signs = RoadSignArray()
        self.create_subscription(
            RoadSignArray, "/ika_vision/road_signs", self.road_signs_callback, 10
        )
        self.shoot_cam_sub = self.create_subscription(
            CompressedImage, self.conf.cam_topic, self.shoot_cam_callback, 10
        )
        self.shoot_debug_pub = self.create_publisher(
            Image, "/shoot_camera/debug_image", 10
        )
        self.shoot_pos_cmd_pub = self.create_publisher(
            Float64MultiArray, "/shoot_camera_controller/commands", 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.cam_info_pub = self.create_subscription(
            CameraInfo, "/shoot_camera/camera_info", self.camera_info_callback, 10
        )

        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(
            0.2, self.send_control_commands, callback_group=self.control_cb_group
        )

    def joint_state_callback(self, msg):
        self.current_tilt = msg.position[5]
        self.current_pan = msg.position[4]

    def road_signs_callback(self, msg):
        self.get_logger().info(f"Received {len(msg.signs)} road signs")
        self.road_signs = msg

    def shoot_cam_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        self.target_center_px, self.target_radius_px, self.debug_img = (
            self.detect_target(img)
        )
        if self.target_center_px is None or self.target_radius_px is None:
            # self.get_logger().warn("Target not detected, skipping control commands.")
            return

        _, self.t_vec_t_c = self.world_coordinates(
            self.target_center_px, self.target_radius_px
        )

        debug_img_msg = self.bridge.cv2_to_imgmsg(self.debug_img)
        debug_img_msg.header.stamp = self.get_clock().now().to_msg()

        self.shoot_debug_pub.publish(debug_img_msg)

    def send_control_commands(self):
        if self.t_vec_t_c is None:
            self.get_logger().warn("No targets detected, cannot send commands.")
            return

        # Convert pnp translation to camera_frame coordinates
        t_vec_t_c = np.array(
            [self.t_vec_t_c[2], -self.t_vec_t_c[0], -self.t_vec_t_c[1]]
        )
        quad_l_c = self.buffer.lookup_transform(
            "shoot_laser", self.conf.cam_frame_id, rclpy.time.Time()
        ).transform.rotation
        R_l_c = np.array(
            quaternion_matrix(
                [
                    quad_l_c.x,
                    quad_l_c.y,
                    quad_l_c.z,
                    quad_l_c.w,
                ]
            )[:3, :3]
        )
        P_laser_to_target_in_cam = (t_vec_t_c - self.t_vec_l_c).reshape(3, 1)
        assert P_laser_to_target_in_cam.shape == (
            3,
            1,
        ), f"Expected shape (3, 1), got {P_laser_to_target_in_cam.shape}"

        P_t_l = np.dot(R_l_c, P_laser_to_target_in_cam)
        assert P_t_l.shape == (
            3,
            1,
        ), f"Expected shape (3, 1), got {P_t_l.shape}"

        # self.get_logger().info(
        #     f"Target position in Laser Frame: X={P_t_l[0,0]:.2f}cm Y={P_t_l[1,0]:.2f}cm Z={P_t_l[2,0]:.2f}cm"
        # )
        tilt = math.atan2(P_t_l[1, 0], P_t_l[2, 0])
        pan = math.atan2(P_t_l[0, 0], math.sqrt(P_t_l[2, 0] ** 2 + P_t_l[1, 0] ** 2))

        # tilt = 0.0 if abs(tilt) < 0.01 else tilt
        # pan = 0.0 if abs(pan) < 0.01 else pan

        # Normalize angles to [-pi, pi] before sending as commands
        self.tilt_cmd = np.clip(tilt + self.current_tilt, -np.pi / 2, np.pi / 2)
        self.pan_cmd = np.clip(pan + self.current_pan, -np.pi / 2, np.pi / 2)

        tilt_in_deg = np.degrees(tilt)
        pan_in_deg = np.degrees(pan)

        self.get_logger().info(
            f"Tilt needed: {tilt_in_deg:.4f} deg, Pan needed: {pan_in_deg:.4f} deg"
        )

        cmd_msg = Float64MultiArray()
        cmd_msg.data = [self.pan_cmd, self.tilt_cmd]
        self.shoot_pos_cmd_pub.publish(cmd_msg)

    def camera_info_callback(self, msg):
        """Callback to handle camera info updates."""
        self.cam_matrix = np.array(msg.k).reshape(3, 3)
        self.cam_dist_coeffs = np.array(msg.d).reshape(-1, 1)

    # UTILITY FUNCTIONS
    def detect_target(self, img):
        img_blur = cv2.GaussianBlur(img, (5, 5), 1.5)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        lower = (0, 0, 35)
        upper = (255, 100, 170)
        mask = cv2.inRange(hsv, lower, upper)

        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=5,
            param1=100,
            param2=20,
            minRadius=10,
            maxRadius=25,
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

            # if (
            #     abs(center[0] - self.conf.cam_width / 2) > self.conf.cam_width * 0.35
            #     or center[1] > self.conf.cam_height * 0.5
            # ):
            #     continue
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
            average_radius = float(np.mean(radii))

            average_radius = max(radii)
            cv2.circle(img, (int(avg_x), int(avg_y)), 1, (255, 0, 0), -1)
            # self.target_radius_px = average_radius
            # self.target_center_px = average_center

            # self.world_coordinates(self.center, self.radii)
            return (average_center, average_radius, img)
        return None, None, img

    def is_within_sign(self, center):
        for sign in self.road_signs.signs:
            if (
                sign.x < center[0] < sign.x + sign.w
                and sign.y < center[1] < sign.y + sign.h
            ):
                return True
        return False

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
            # self.get_logger().info(
            #     f"Target coordinates in Camera Frame: X={x:.2f}cm Y={y:.2f}cm Z={z:.2f}cm"
            # )
            return rvec, tvec
        else:
            self.get_logger().warn("solvePnP failed.")
            return None, None


def main(args=None):
    rclpy.init(args=args)
    node = ShootCameraController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
