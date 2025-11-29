#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import cv2
import numpy as np
import math

from rake_core.constants import Actions
from ika_actions.action import LockTarget
from cv_bridge import CvBridge
from ika_msgs.msg import RoadSign, RoadSignArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_matrix, quaternion_matrix
from tf2_ros import TransformListener, Buffer


class LockTargetConfig:
    """Configuration class for lock target action parameters"""

    def __init__(self):
        # Timeout for target locking
        self.max_execution_time = 10.0  # seconds

        # Camera configuration (from ShootCameraControllerConfig)
        self.cam_topic = "/shoot_camera/image_raw/compressed"
        self.cam_frame_id = "shoot_camera"
        self.cam_width = 1920
        self.cam_height = 1080
        self.cam_fps = 8

        # Control frequency
        self.control_frequency = 5.0  # Hz (same as shoot_controller timer)
        self.control_period = 1.0 / self.control_frequency


class LockTargetServer(Node):
    """Action server for locking onto detected targets using pan/tilt control"""

    def __init__(self):
        super().__init__("lock_target_server")

        self.config = LockTargetConfig()
        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.detect_target_cb_group = MutuallyExclusiveCallbackGroup()
        # CV Bridge for image processing
        self.bridge = CvBridge()

        # TF2 setup
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Action server
        self._action_server = ActionServer(
            self,
            LockTarget,
            Actions.LOCK_TARGET,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Camera parameters (from shoot_controller)
        self.cam_matrix = None
        self.cam_dist_coeffs = None
        self.target_center_px = None
        self.target_radius_px = None

        # Joint states (from shoot_controller)
        self.current_tilt = 0.0
        self.tilt_cmd = None
        self.current_pan = 0.0
        self.pan_cmd = None

        # Transform vectors (from shoot_controller)
        self.t_vec_l_c = np.array(
            [[0.0], [0.0], [0.02]], dtype=np.float64
        )  # 1 unit along z-axis
        self.t_vec_t_c = None

        self.debug_img = None
        self.road_signs = RoadSignArray()

        # Subscribers (from shoot_controller)
        self.create_subscription(
            RoadSignArray, "/ika_vision/road_signs", self.road_signs_callback, 10
        )
        self.shoot_cam_sub = self.create_subscription(
            CompressedImage,
            self.config.cam_topic,
            self.shoot_cam_callback,
            10,
            callback_group=self.detect_target_cb_group,
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.cam_info_pub = self.create_subscription(
            CameraInfo, "/shoot_camera/camera_info", self.camera_info_callback, 10
        )

        # Publishers (from shoot_controller)
        self.shoot_debug_pub = self.create_publisher(
            Image, "/shoot_camera/debug_image", 10
        )
        self.shoot_pos_cmd_pub = self.create_publisher(
            Float64MultiArray, "/shoot_camera_controller/commands", 10
        )

        # Control variables for action execution
        self.current_tilt_error = 0.0
        self.current_pan_error = 0.0
        self.is_executing = False

        self.get_logger().info("Lock target action server initialized")

    def joint_state_callback(self, msg):
        """Joint state callback (exact copy from shoot_controller)"""
        self.current_tilt = msg.position[5]
        self.current_pan = msg.position[4]

    def road_signs_callback(self, msg):
        """Road signs callback (exact copy from shoot_controller)"""
        self.get_logger().debug(f"Received {len(msg.signs)} road signs")
        self.road_signs = msg

    def shoot_cam_callback(self, msg):
        """Camera callback (exact copy from shoot_controller)"""
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        self.target_center_px, self.target_radius_px, self.debug_img = (
            self.detect_target(img)
        )
        if self.debug_img is not None:
            debug_img_msg = self.bridge.cv2_to_imgmsg(self.debug_img)
            debug_img_msg.header.stamp = self.get_clock().now().to_msg()
            self.shoot_debug_pub.publish(debug_img_msg)

        if self.target_center_px is None or self.target_radius_px is None:
            return

        _, self.t_vec_t_c = self.world_coordinates(
            self.target_center_px, self.target_radius_px
        )

    def camera_info_callback(self, msg):
        """Camera info callback (exact copy from shoot_controller)"""
        self.cam_matrix = np.array(msg.k).reshape(3, 3)
        self.cam_dist_coeffs = np.array(msg.d).reshape(-1, 1)

    def detect_target(self, img):
        """Target detection function (exact copy from shoot_controller)"""
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
            return None, None, img

        circles = np.uint16(np.around(circles))
        centers = []
        radii = []
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            if self.is_within_sign(center):
                # ignore targets that are detected within road signs
                # cv2.circle(img, center, radius, (255, 0, 0), 4)
                continue
            centers.append(center)
            radii.append(radius)

            cv2.circle(img, center, 1, (0, 0, 255), 1)
            cv2.circle(img, center, radius, (0, 255, 0), 1)

        # Add tilt and pan error text to debug image
        tilt_text = f"Tilt Error: {self.current_tilt_error:.3f} deg"
        pan_text = f"Pan Error: {self.current_pan_error:.3f} deg"

        # Choose color based on error magnitude (green if < 0.3 deg, red otherwise)
        tilt_color = (0, 255, 0) if abs(self.current_tilt_error) < 0.3 else (0, 0, 255)
        pan_color = (0, 255, 0) if abs(self.current_pan_error) < 0.3 else (0, 0, 255)

        # Add text to top-left corner
        cv2.putText(
            img, tilt_text, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tilt_color, 4
        )
        cv2.putText(
            img, pan_text, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pan_color, 4
        )

        if len(centers) > 0:
            avg_x = float(np.mean([c[0] for c in centers]))
            avg_y = float(np.mean([c[1] for c in centers]))
            average_center = (avg_x, avg_y)
            average_radius = float(np.mean(radii))

            average_radius = max(radii)
            cv2.circle(img, (int(avg_x), int(avg_y)), 1, (255, 0, 0), -1)

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
        """World coordinates calculation (exact copy from shoot_controller)"""
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
            return rvec, tvec
        else:
            self.get_logger().warn("solvePnP failed.")
            return None, None

    def calculate_control_commands(self):
        """Calculate pan/tilt commands (exact copy from shoot_controller send_control_commands)"""
        if self.t_vec_t_c is None:
            return None, None

        try:
            # Convert pnp translation to camera_frame coordinates
            t_vec_t_c = np.array(
                [self.t_vec_t_c[2], -self.t_vec_t_c[0], -self.t_vec_t_c[1]]
            )
            quad_l_c = self.buffer.lookup_transform(
                "shoot_laser", self.config.cam_frame_id, rclpy.time.Time()
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

            tilt = math.atan2(P_t_l[1, 0], P_t_l[2, 0])
            pan = math.atan2(
                P_t_l[0, 0], math.sqrt(P_t_l[2, 0] ** 2 + P_t_l[1, 0] ** 2)
            )

            # Normalize angles to [-pi, pi] before sending as commands
            self.tilt_cmd = np.clip(tilt + self.current_tilt, -np.pi / 2, np.pi / 2)
            self.pan_cmd = np.clip(pan + self.current_pan, -np.pi / 2, np.pi / 2)

            return tilt, pan

        except Exception as e:
            self.get_logger().error(f"Error calculating control commands: {str(e)}")
            return None, None

    def goal_callback(self, goal_request):
        """Accept goal only if target is detected"""
        # Try to detect target in current frame
        if self.target_center_px is None or self.target_radius_px is None:
            self.get_logger().warn("Goal rejected: No target detected")
            return GoalResponse.REJECT

        angular_tolerance = goal_request.angular_tolerance
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

        # Get goal parameters
        angular_tolerance_deg = goal_handle.request.angular_tolerance
        angular_tolerance_rad = math.radians(angular_tolerance_deg)

        # Initialize feedback and result
        feedback_msg = LockTarget.Feedback()
        result = LockTarget.Result()

        # Reset timing
        start_time = time.time()
        self.is_executing = True

        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Check timeout
                if elapsed_time > self.config.max_execution_time:
                    self.get_logger().warn("Target locking timeout exceeded")

                    result.success = False
                    result.message = f"Timeout: Could not lock target within {self.config.max_execution_time}s"
                    self.is_executing = False
                    goal_handle.abort()
                    return result

                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Goal was cancelled")

                    goal_handle.canceled()
                    result.success = False
                    result.message = "Goal cancelled by client"
                    self.is_executing = False
                    return result

                # Check if we still have target
                if self.target_center_px is None or self.target_radius_px is None:
                    self.get_logger().warn("Target lost during locking")

                    result.success = False
                    result.message = "Target lost during locking"
                    self.is_executing = False
                    goal_handle.abort()
                    return result

                # Calculate control commands
                tilt_error, pan_error = self.calculate_control_commands()

                if tilt_error is None or pan_error is None:
                    self.get_logger().warn("Failed to calculate control commands")
                    time.sleep(self.config.control_period)
                    continue

                # Convert errors to degrees for feedback
                tilt_error_deg = abs(math.degrees(tilt_error))
                pan_error_deg = abs(math.degrees(pan_error))

                # Update current errors for feedback
                self.current_tilt_error = tilt_error_deg
                self.current_pan_error = pan_error_deg

                # Publish feedback
                feedback_msg.tilt_loss = tilt_error_deg
                feedback_msg.pan_loss = pan_error_deg
                goal_handle.publish_feedback(feedback_msg)

                # Check if target is locked within tolerance
                if (
                    tilt_error_deg < angular_tolerance_deg
                    and pan_error_deg < angular_tolerance_deg
                ):
                    self.get_logger().info(
                        f"Target locked successfully! Tilt error: {tilt_error_deg:.2f}°, "
                        f"Pan error: {pan_error_deg:.2f}°"
                    )

                    result.success = True
                    result.message = (
                        f"Successfully locked target (tilt: {tilt_error_deg:.2f}°, "
                        f"pan: {pan_error_deg:.2f}°)"
                    )
                    goal_handle.succeed()
                    self.is_executing = False
                    return result

                # Publish control commands
                if self.tilt_cmd is not None and self.pan_cmd is not None:
                    cmd_msg = Float64MultiArray()
                    cmd_msg.data = [self.pan_cmd, self.tilt_cmd]
                    self.shoot_pos_cmd_pub.publish(cmd_msg)

                self.get_logger().debug(
                    f"Target locking: Tilt error: {tilt_error_deg:.2f}°, "
                    f"Pan error: {pan_error_deg:.2f}°, Time: {elapsed_time:.1f}s"
                )

                # Sleep to maintain control frequency
                time.sleep(self.config.control_period)

        except Exception as e:
            self.get_logger().error(f"Error during target locking execution: {str(e)}")

            result.success = False
            result.message = f"Execution error: {str(e)}"
            self.is_executing = False
            goal_handle.abort()
            return result


def main(args=None):
    rclpy.init(args=args)

    lock_target_server = LockTargetServer()

    # Use MultiThreadedExecutor for handling action callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(lock_target_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        lock_target_server.get_logger().info("Shutting down lock target server...")
    finally:
        lock_target_server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
