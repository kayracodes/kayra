#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rake_core.node import Node
from rake_core.states import SystemModeEnum, SystemStateEnum
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import cv2
import numpy as np
import math
import json
from types import SimpleNamespace
from rake_core.constants import Actions
from ika_actions.action import LockTarget
from cv_bridge import CvBridge
from ika_msgs.msg import RoadSign, RoadSignArray
from rake_msgs.msg import ShootCommand
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_matrix, quaternion_matrix, euler_matrix
from threading import Lock


class Target:
    def __init__(self, x_px=None, y_px=None, radius_px=None):
        self.x_px = x_px
        self.y_px = y_px
        self.radius_px = radius_px


class LockTargetConfig:
    """Configuration class for lock target action parameters"""

    def __init__(self):
        # Timeout for target locking
        self.max_execution_time = 10.0  # seconds
        # Camera configuration (from ShootCameraControllerConfig)
        self.cam_frame_id = "shoot_camera"
        self.cam_width = 1920
        self.cam_height = 1080
        self.cam_fps = 8
        self.cam_laser_offset = 0.02

        # Control frequency
        self.control_period = 0.05

        self.tilt_sensitivity = 0.3
        self.pan_sensitivity = 0.3
        # Hough Circle params
        self.min_radius = 10
        self.max_radius = 25


class LockTargetServer(Node):
    """Action server for locking onto detected targets using pan/tilt control"""

    def __init__(self):
        super().__init__("lock_target_server")
        self.execution_lock = Lock()
        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # CV Bridge for image processing
        self.bridge = CvBridge()

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        self.target = Target()
        self.debug_img = None
        self.road_signs = RoadSignArray()
        # Control variables for action execution
        self.is_executing = False
        self.laser_on = False
        self.tilt_complete = False
        self.pan_complete = False
        self.is_target_hit = False
        self.pan_dir = 0
        self.tilt_dir = 0

        # Initialize error tracking variables
        self.current_tilt_error = 0.0
        self.current_pan_error = 0.0

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

        # TF Values
        self.R_l_c = euler_matrix(np.radians(-0.4), 0.0, np.radians(1.0))[
            :3, :3
        ].astype(np.float32)
        laser_offset = 4.45
        self.t_vec_l_c = np.array([[0.0], [laser_offset], [0.0]], dtype=np.float32)

        # Action Server
        self._action_server = ActionServer(
            self,
            LockTarget,
            Actions.LOCK_TARGET_REAL,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Subscribers (from shoot_controller)
        self.shoot_cam_sub = self.create_subscription(
            CompressedImage,
            "/shoot_camera/image_raw",
            self.shoot_cam_callback,
            10,
        )
        self.create_subscription(
            RoadSignArray, "/ika_vision/road_signs", self.road_signs_callback, 10
        )

        # Publishers (from shoot_controller)
        self.shoot_debug_pub = self.create_publisher(
            Image, "/shoot_camera/debug_image", 10
        )
        self.shoot_cmd_pub = self.create_publisher(
            ShootCommand, "/shoot_camera/shoot_command", 10
        )

        # Timer
        self.get_logger().info("Lock target action server initialized")

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def get_default_config(self):
        return json.loads(json.dumps(LockTargetConfig().__dict__))

    def road_signs_callback(self, msg):
        """Road signs callback (exact copy from shoot_controller)"""
        self.get_logger().debug(f"Received {len(msg.signs)} road signs")
        self.road_signs = msg

    def shoot_cam_callback(self, msg):
        # """Camera callback (exact copy from shoot_controller)"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.undistort(cv_image, self.cam_matrix, self.cam_dist_coeffs)
        avg_center, avg_radius, debug_img = self.detect_target(cv_image)

        if avg_center is not None and avg_radius is not None:
            self.debug_img = debug_img
            self.target.x_px, self.target.y_px = avg_center
            self.target.radius_px = avg_radius
            self.shoot_debug_pub.publish(
                self.bridge.cv2_to_imgmsg(self.debug_img, "bgr8")
            )
        else:
            self.debug_img = debug_img
            self.target.x_px = None
            self.target.y_px = None
            self.target.radius_px = None
            self.shoot_debug_pub.publish(
                self.bridge.cv2_to_imgmsg(self.debug_img, "bgr8")
            )

    def detect_target(self, img):
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

    def is_within_sign(self, center):
        """Check if target is within road sign (exact copy from shoot_controller)"""
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
            self.get_logger().info(
                f"Target coordinates in Camera Frame: X={x:.2f}cm Y={y:.2f}cm Z={z:.2f}cm"
            )
            # t_vec_t_c = np.array([tvec[2], -tvec[0], -tvec[1]])
            return rvec, tvec
        else:
            self.get_logger().warn("solvePnP failed.")
            return None, None

    def _send_shoot_command(self):
        """Send shoot command to shoot controller (exact copy from shoot_controller)"""
        shoot_msg = ShootCommand()
        shoot_msg.laser_on = int(self.laser_on)
        shoot_msg.tilt_dir = self.tilt_dir
        shoot_msg.pan_dir = self.pan_dir
        self.shoot_cmd_pub.publish(shoot_msg)
        self.get_logger().debug(f"Published shoot command: {shoot_msg}")

    def _reset_state_variables(self):
        # Reset Target & Debug Image
        self.target = Target()
        self.debug_img = None
        # Reset Control variables for action execution
        self.is_executing = False
        self.laser_on = False
        self.tilt_complete = False
        self.pan_complete = False
        self.is_target_hit = False
        self.pan_dir = 0
        self.tilt_dir = 0
        self.current_tilt_error = 0.0
        self.current_pan_error = 0.0

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

    def goal_callback(self, goal_request):
        """Accept goal only if target is detected"""
        # Try to detect target in current frame
        with self.execution_lock:
            if self.is_executing:
                self.get_logger().warn("Goal rejected: Target is already being tracked")
                return GoalResponse.REJECT

            if (
                self.target.x_px is None
                or self.target.y_px is None
                or self.target.radius_px is None
            ):
                self.get_logger().warn("Goal rejected: No target detected")
                return GoalResponse.REJECT

        angular_tolerance = min(
            self.config.pan_sensitivity, self.config.tilt_sensitivity
        )
        if angular_tolerance <= 0:
            self.get_logger().warn(
                f"Goal rejected: Invalid angular tolerance {angular_tolerance}"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Goal accepted: Target detected, tolerance: {angular_tolerance}°"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation"""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute target locking action"""
        self.get_logger().info("Executing target locking...")

        # Initialize feedback and result
        feedback_msg = LockTarget.Feedback()
        result = LockTarget.Result()

        # Reset Timing & State Variables
        with self.execution_lock:
            start_time = time.time()
            self.is_executing = True

        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Check if system mobility is disabled, if so abort
                if self.mobility == 0:
                    self.get_logger().warn(
                        "System mobility is disabled, aborting target locking"
                    )
                    goal_handle.abort()
                    with self.execution_lock:
                        result.success = False
                        result.message = "System mobility is disabled"
                        self._reset_state_variables()
                        return result

                # Check timeout
                if elapsed_time > self.config.max_execution_time:
                    self.get_logger().warn("Target locking timeout exceeded")

                    with self.execution_lock:
                        result.success = False
                        result.message = f"Timeout: Could not lock target within {self.config.max_execution_time}s"
                        self._reset_state_variables()
                        return result

                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Goal was cancelled")

                    goal_handle.canceled()
                    with self.execution_lock:
                        result.success = False
                        result.message = "Goal cancelled by client"
                        self._reset_state_variables()
                        return result

                # Check if we still have target
                if (
                    self.target.x_px is None
                    or self.target.y_px is None
                    or self.target.radius_px is None
                ):
                    self.pan_dir = 0
                    self.tilt_dir = 0
                    self._send_shoot_command()
                    time.sleep(self.config.control_period)
                    continue
                # Calculate control commands
                rvec, tvec_t_c = self.world_coordinates(
                    (self.target.x_px, self.target.y_px), self.target.radius_px
                )
                if rvec is None or tvec_t_c is None:
                    self.get_logger().warn("Failed to compute world coordinates.")
                    self.pan_dir = 0
                    self.tilt_dir = 0
                    self._send_shoot_command()
                    time.sleep(self.config.control_period)
                    continue

                pan, tilt = self.return_tilt_and_pan(
                    tvec_t_c, self.t_vec_l_c, self.R_l_c
                )
                if tilt is None or pan is None:
                    self.get_logger().warn("Failed to compute tilt and pan.")
                    time.sleep(self.config.control_period)
                    self.pan_dir = 0
                    self.tilt_dir = 0
                    self._send_shoot_command()
                    continue

                if not self.tilt_complete:
                    if abs(tilt) > self.config.tilt_sensitivity:
                        self.tilt_dir = 1 if tilt > 0 else 2
                        self.laser_on = False
                    else:
                        self.get_logger().info("TILT COMPLETE")
                        self.tilt_complete = True
                        self.tilt_dir = 0
                else:
                    if abs(pan) > self.config.pan_sensitivity:
                        self.pan_dir = 1 if pan > 0 else 2
                        self.laser_on = False
                    else:
                        self.get_logger().info("PAN COMPLETE")
                        self.pan_complete = True
                        self.pan_dir = 0

                if self.pan_complete and self.tilt_complete:
                    # Last check before firing
                    if abs(tilt) > self.config.tilt_sensitivity:
                        self.tilt_complete = False
                    elif abs(pan) > self.config.pan_sensitivity:
                        self.pan_complete = False
                    else:
                        self.get_logger().info("PAN TILT COMPLETE")
                        self.laser_on = True
                        self.tilt_dir = 0
                        self.pan_dir = 0
                        self.is_target_hit = True
                        self.get_logger().info("Target hit, resetting state.")

                        with self.execution_lock:
                            result.success = True
                            result.message = "Target successfully locked and hit"
                            self._reset_state_variables()
                            goal_handle.succeed()
                            return result

                # Update current errors for feedback
                self.current_tilt_error = tilt
                self.current_pan_error = pan

                # Publish feedback
                feedback_msg.tilt_loss = self.current_tilt_error
                feedback_msg.pan_loss = self.current_pan_error
                goal_handle.publish_feedback(feedback_msg)

                # Publish control commands
                self.get_logger().info(
                    f"Target locking: Tilt error: {tilt:.2f}°, "
                    f"Pan error: {pan:.2f}°, Time: {elapsed_time:.1f}s"
                )
                self._send_shoot_command()
                # Sleep to maintain control frequency
                time.sleep(self.config.control_period)

        except Exception as e:
            self.get_logger().error(f"Error during target locking execution: {str(e)}")

            with self.execution_lock:
                result.success = False
                result.message = f"Execution error: {str(e)}"
                self.is_executing = False
                return result


def main(args=None):
    rclpy.init(args=args)

    lock_target_server = LockTargetServer()
    lock_target_server.init()
    Node.run_node(lock_target_server)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
