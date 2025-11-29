#!/usr/bin/env python3
import rclpy
import numpy as np
import cv2
import sys
from types import SimpleNamespace
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from rake_core.node import Node
from rake_msgs.msg import SystemState
from rake_core.states import SystemStateEnum, DeviceStateEnum, SystemModeEnum
from nav_msgs.msg import OccupancyGrid
from ika_msgs.msg import RampFeedback, RampInfo, RampDistance
from rclpy.exceptions import ROSInterruptException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import json

hsv_thresholds = {
    "red": [(118, 104, 81), (179, 255, 255)],
    "orange": [(0, 104, 177), (111, 255, 255)],
    "white": [(0, 0, 174), (179, 50, 255)],
}
hsv_thresholds_real = {
    # "dirt": [(15, 30, 30), (150, 255, 220)],
    "red": [(119, 27, 0), (179, 255, 255)],
    "white": [(0, 0, 200), (179, 200, 255)],
    "orange": [(0, 55, 100), (179, 255, 255)],
}

map_width = 75
map_height = 75
resolution = 0.08  # meters per pixel


class VisionTransformerConfig:
    def __init__(self):
        self.points_src = [(46, 0), (274, 0), (320, 240), (0, 240)]

        # perspective mask will be resized to turn into occupancy grid, but higher resolution version is used for ramp detection
        self.points_dst = [
            (0, 0),
            (2 * map_width, 0),
            (2 * 45, 2 * map_height),
            (2 * 30, 2 * map_height),
        ]

        self.unknown_inflation_right = [
            (map_width, 0),
            (map_width, map_height),
            (45, map_height),
        ]
        self.unknown_inflation_left = [
            (0, 0),
            (0, map_height),
            (30, map_height),
        ]
        self.unknown_right = [
            (2 * map_width, 0),
            (2 * map_width, 2 * map_height),
            (2 * 45, 2 * map_height),
        ]
        self.unknown_left = [
            (0, 0),
            (0, 2 * map_height),
            (2 * 30, 2 * map_height),
        ]

        self.blur_iters = 2
        self.blur_kernel = (3, 3)

        # State-specific inflation radius parameters (static config only)
        self.default_inflation_radius = 1.2
        self.sharp_turn_inflation_radius = 1.8
        self.cautious_inflation_radius = 1.2

        # ramp_detection parameters
        self.min_line_length = 20
        self.max_line_gap = 5
        self.max_detection_angle = 15


class VisionTransformer(Node):
    def __init__(self):
        # NOTE I may wanna initialize the default config here by calling self.config_updated(self.get_default_config()), but not now
        super().__init__("vision_transformer")
        self.cv_bridge = CvBridge()
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.occ_grid_cb_group = MutuallyExclusiveCallbackGroup()
        self.debug_cb_group = MutuallyExclusiveCallbackGroup()

        # Runtime changable parameters

        self.occupancy_grid_msg = self.init_occ_grid_msg()
        self.pers_img = None
        self.occ_grid = None
        self.pers_mask = None
        self.line_mask = None
        self.ramp_lines = []
        self.min_ramp_distance = None
        self.max_ramp_distance = None

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return
        # define the runtime changable parameters first
        self.inflation_radius = self.config.default_inflation_radius
        self.morph_occ_grid = True
        # define others
        self.front_cam_sub = self.create_subscription(
            CompressedImage,
            "/front_camera/image_raw/compressed",
            self.front_cam_callback,
            10,
            callback_group=self.cb_group,
        )
        self.pers_img_pub = self.create_publisher(Image, "/ika_vision/pers_img", 10)
        self.pers_mask_pub = self.create_publisher(Image, "/ika_vision/pers_mask", 10)
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, "/ika_vision/occupancy_grid", 10
        )
        self.ramp_detected_pub = self.create_publisher(
            RampFeedback, "/ika_vision/ramp_detected", 10
        )
        self.ramp_distance_pub = self.create_publisher(
            RampDistance, "/ika_vision/ramp_distance", 10
        )

        self.create_timer(
            0.125,
            self.publish_occupancy_grid,
            callback_group=self.occ_grid_cb_group,
        )
        self.create_timer(0.2, self.publish_ramp_feedback)
        self.create_timer(
            0.25, self.publish_debug_image, callback_group=self.debug_cb_group
        )
        # Vision Transformer will work at all the system_states
        self.set_device_state(DeviceStateEnum.WORKING)

    def system_state_transition(self, old_state: SystemState, new_state: SystemState):
        """Handle system state transitions and update parameters accordingly"""

        # Update inflation radius based on new state using static config values
        self.get_logger().info(
            f"System state transition: {old_state.state} -> {new_state.state}"
        )
        if not hasattr(self, "config"):
            # config not loaded yet return
            return
        updated_config = self.get_state_config_changes(new_state.state)
        if not updated_config:
            return
        try:
            for k, v in updated_config.items():
                setattr(self, k, v)
        except Exception as e:
            self.get_logger().error(f"Error updating state config: {e}")

        self.get_logger().info(f"Runtime Changable Params: {updated_config}")

    def get_state_config_changes(self, state: SystemStateEnum):
        # Device Configuration Changes
        if (
            state == SystemStateEnum.SLOW_AUTONOMOUS
            or state == SystemStateEnum.FAST_AUTONOMOUS
            or state == SystemStateEnum.IDLE_AUTONOMOUS
        ):
            return {
                "inflation_radius": self.config.default_inflation_radius,
                "morph_occ_grid": True,
            }
        elif state == SystemStateEnum.CAUTIOUS_AUTONOMOUS:
            return {
                "inflation_radius": self.config.cautious_inflation_radius,
                "morph_occ_grid": False,
            }
        elif state == SystemStateEnum.SHARP_TURN_AUTONOMOUS:
            return {
                "inflation_radius": self.config.sharp_turn_inflation_radius,
                "morph_occ_grid": True,
            }
        else:
            return {
                "inflation_radius": self.config.default_inflation_radius,
                "morph_occ_grid": True,
            }

    def get_default_config(self):
        return json.loads(json.dumps(VisionTransformerConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def front_cam_callback(self, msg):
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        # create occupancy grid
        self.occ_grid, self.pers_img, self.pers_mask = self.apply_transform(img)
        # detect ramps
        if self.pers_img is not None or self.pers_mask is not None:
            (
                self.ramp_lines,
                self.line_mask,
                self.min_ramp_distance,
                self.max_ramp_distance,
            ) = self.detect_ramp(self.pers_img, self.pers_mask)

    def apply_transform(self, img):
        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        img = self.blur(img)

        mask = self.hsv_filter_real(img)
        mask = self.perspective_transform(mask)
        # if self.ramp_lines is not None and len(self.ramp_lines) > 0:
        #     mask = self.mark_ramps(mask, self.ramp_lines)
        # if self.line_mask is not None:
        #     mask = self.mark_lines(mask, self.line_mask)

        kernel = np.ones((3, 3), np.uint8)
        occ_grid = cv2.resize(mask, (map_width, map_height))
        if self.morph_occ_grid:
            # apply morphological operations to mask
            occ_grid = cv2.morphologyEx(occ_grid, cv2.MORPH_OPEN, kernel)

        pers_img = self.perspective_transform(img)

        # apply morphological operations to pers_mask
        # kernel = np.ones((3, 3), np.uint8)
        pers_mask = self.unknown_area(mask)
        pers_mask = cv2.morphologyEx(pers_mask, cv2.MORPH_CLOSE, kernel)
        pers_mask = cv2.morphologyEx(pers_mask, cv2.MORPH_OPEN, kernel)
        pers_mask = cv2.cvtColor(pers_mask, cv2.COLOR_GRAY2BGR)

        return occ_grid, pers_img, pers_mask

    def hsv_filter(self, img):
        # Reset color filter for current frame

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # assert img_hsv.shape[:2] == (320, 240), "Image shape mismatch"
        mask = np.zeros_like(img_hsv[:, :, 0])
        for color, (lower, upper) in hsv_thresholds.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask_color = cv2.inRange(img_hsv, lower, upper)
            has_color = np.any(mask_color)

            if has_color:
                mask = cv2.bitwise_or(mask, mask_color)

        return mask

    def hsv_filter_real(self, img):
        # Reset color filter for current frame

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # assert img_hsv.shape[:2] == (320, 240), "Image shape mismatch"
        mask = np.zeros_like(img_hsv[:, :, 0])
        for color, (lower, upper) in hsv_thresholds.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask_color = cv2.inRange(img_hsv, lower, upper)
            cv2.morphologyEx(
                mask_color,
                cv2.MORPH_OPEN,
                np.ones((9, 9), np.uint8),
                iterations=1,
                dst=mask_color,
            )
            cv2.morphologyEx(
                mask_color,
                cv2.MORPH_CLOSE,
                np.ones((9, 9), np.uint8),
                iterations=1,
                dst=mask_color,
            )
            mask = cv2.bitwise_or(mask, mask_color)

        # combined_mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=3)
        return mask

    def detect_ramp(self, pers_img, pers_mask):
        pers_img = np.where(pers_mask == 0, pers_img, 255)
        edges = cv2.Canny(pers_img, 50, 150, apertureSize=3)
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

        # Keep only strong horizontal edges
        horizontal_edges = np.abs(sobely) > np.abs(sobelx)
        strong_edges = (np.abs(sobely) > 50) & horizontal_edges
        # line_img = cv2.cvtColor(255 * strong_edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # detect ramp
        lines = cv2.HoughLinesP(
            strong_edges.astype(np.uint8),
            1,
            np.pi / 180,
            18,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        horizontal_lines = []
        min_ramp_distance, max_ramp_distance = None, None
        if lines is not None:
            sum_x, sum_y = 0.0, 0.0
            hor_sum_x, hor_sum_y = 0.0, 0.0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate line angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                sum_x += x2 - x1
                sum_y += y2 - y1
                # self.get_logger().info(f"Detected horizontal line: {line}")
                # Keep nearly horizontal lines (±15 degrees)
                if (
                    abs(angle) < self.config.max_detection_angle
                    or abs(abs(angle) - 180) < self.config.max_detection_angle
                ):
                    hor_sum_x += x2 - x1
                    hor_sum_y += y2 - y1
                    # length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    horizontal_lines.append((x1, y1, x2, y2))
                    # self.get_logger().info(f"Detected horizontal line: {line}")
            avg_angle = np.arctan2(-sum_y, sum_x) * 180 / np.pi
            if len(horizontal_lines) > 0:
                angle_hor = np.arctan2(-hor_sum_y, hor_sum_x) * 180 / np.pi
                self.get_logger().info(
                    f"{len(horizontal_lines)} hor lines, hor_angle: {angle_hor:.2f}"
                )
                closest_ramp = min(
                    horizontal_lines, key=lambda line: 150 - (line[1] + line[3]) / 2
                )
                min_ramp_distance = round(
                    0.04 * (150 - (closest_ramp[1] + closest_ramp[3]) / 2), 2
                )
                farthest_ramp = max(
                    horizontal_lines, key=lambda line: 150 - (line[1] + line[3]) / 2
                )
                max_ramp_distance = round(
                    0.04 * (150 - (farthest_ramp[1] + farthest_ramp[3]) / 2), 2
                )
            self.get_logger().info(f"angle: {avg_angle:.2f}")
        else:
            self.get_logger().warn("No lines detected in the image.")
        return horizontal_lines, strong_edges, min_ramp_distance, max_ramp_distance

    def perspective_transform(self, img):
        src = np.array(self.config.points_src, np.float32)
        dst = np.array(self.config.points_dst, np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(img, M, (2 * map_width, 2 * map_height))

    def unknown_area(self, mask):
        # occupancy mask for pers_img and pers_mask
        unknown_right = np.array(self.config.unknown_right, np.int32)
        unknown_left = np.array(self.config.unknown_left, np.int32)

        cv2.fillPoly(mask, [unknown_right], 255)
        cv2.fillPoly(mask, [unknown_left], 255)

        return mask

    def unknown_area_inflation(self, mask, mask_value):
        # occupancy mask for actual grid data that will be inflated
        unknown_right = np.array(self.config.unknown_inflation_right, np.int32)
        unknown_left = np.array(self.config.unknown_inflation_left, np.int32)

        cv2.fillPoly(mask, [unknown_right], mask_value)
        cv2.fillPoly(mask, [unknown_left], mask_value)

        return mask

    def init_occ_grid_msg(self):
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.frame_id = "base_link"
        occupancy_grid_msg.info.resolution = (
            resolution  # Set the resolution of the grid
        )
        occupancy_grid_msg.info.width = map_width
        occupancy_grid_msg.info.height = map_height
        occupancy_grid_msg.info.origin.position.x = 0.0
        occupancy_grid_msg.info.origin.position.y = -3.0
        occupancy_grid_msg.info.origin.position.z = 0.0

        return occupancy_grid_msg

    def blur(self, img):
        for _ in range(self.config.blur_iters):
            img = cv2.blur(img, self.config.blur_kernel)
        return img

    def apply_inflation(self, occupancy_grid: np.array):
        inflation_radius_px = int(self.inflation_radius / resolution)
        obstacle_mask = ((occupancy_grid == 100) | (occupancy_grid == 50)).astype(
            np.uint8
        )
        # Get both distance and labels
        distance_transform, labels = cv2.distanceTransformWithLabels(
            1 - obstacle_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )

        inflated_grid = occupancy_grid.copy()
        within_radius_mask = distance_transform <= inflation_radius_px

        # Find coordinates of occupied cells
        occupied_coords = np.column_stack(np.where(obstacle_mask == 1))
        # # Flatten occupancy grid for easy lookup
        # flat_occupancy = occupancy_grid.flatten()

        # For each unoccupied cell within radius, get occupancy value of closest occupied cell
        rows, cols = np.where(within_radius_mask)
        for r, c in zip(rows, cols):
            label = labels[r, c]
            if label == 0:
                continue  # No occupied cell found
            # OpenCV labels start from 1
            occ_r, occ_c = occupied_coords[label - 1]
            occ_value = occupancy_grid[occ_r, occ_c]
            dist = distance_transform[r, c]
            inflated_value = occ_value * (1 - (dist / inflation_radius_px))
            inflated_grid[r, c] = np.clip(inflated_value, 0, 100)

        return inflated_grid

    def publish_occupancy_grid(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        if self.occ_grid is None:
            self.get_logger().warn("No occ. grid data to publish.")
            return

        occ_grid = ((self.occ_grid // 255.0) * 100).astype(np.uint8)
        occ_grid = self.unknown_area_inflation(occ_grid, 50)
        # kernel = np.ones((3, 3), np.uint8)
        # occ_grid = cv2.morphologyEx(occ_grid, cv2.MORPH_CLOSE, kernel)
        # occ_grid = cv2.morphologyEx(occ_grid, cv2.MORPH_OPEN, kernel)

        occ_grid = np.transpose(occ_grid, (1, 0))
        occ_grid = cv2.flip(occ_grid, 1)
        occ_grid = cv2.flip(occ_grid, 0)

        occ_grid = self.apply_inflation(occ_grid)
        flattened_grid = occ_grid.flatten()

        self.occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.occupancy_grid_msg.data = flattened_grid.tolist()
        self.occupancy_grid_pub.publish(self.occupancy_grid_msg)

        # msg = self.cv_bridge.cv2_to_imgmsg(self.pers_img)
        # self.pers_img_pub.publish(msg)

    def publish_ramp_feedback(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        if self.pers_mask is None or self.ramp_lines is None:
            ramp_msg = RampFeedback()
            ramp_msg.detected = False
            self.ramp_detected_pub.publish(ramp_msg)
            return
        # Create ramp_detected msg
        ramp_msg = RampFeedback()
        ramp_msg.detected = len(self.ramp_lines) > 0
        for line in self.ramp_lines:
            x1, y1, x2, y2 = line
            ramp_info = RampInfo()
            ramp_info.x1 = int(x1)
            ramp_info.y1 = int(y1)
            ramp_info.x2 = int(x2)
            ramp_info.y2 = int(y2)
            ramp_msg.ramps.append(ramp_info)

        self.ramp_detected_pub.publish(ramp_msg)
        # Closest / Farthest Ramp Distance
        ramp_distance_msg = RampDistance()
        if self.min_ramp_distance is None or self.max_ramp_distance is None:
            return
        ramp_distance_msg.min_distance = self.min_ramp_distance
        ramp_distance_msg.max_distance = self.max_ramp_distance
        self.ramp_distance_pub.publish(ramp_distance_msg)

    def publish_debug_image(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        if self.line_mask is not None:
            line_img = cv2.cvtColor(
                255 * self.line_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            for line in self.ramp_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            pers_img_msg = self.cv_bridge.cv2_to_imgmsg(line_img)
            self.pers_img_pub.publish(pers_img_msg)

        if self.pers_mask is not None:
            pers_mask_msg = self.cv_bridge.cv2_to_imgmsg(self.pers_mask)
            self.pers_mask_pub.publish(pers_mask_msg)


def main():
    try:
        rclpy.init()
        node = VisionTransformer()
        node.init()
        Node.run_node(node)
    except ROSInterruptException:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
