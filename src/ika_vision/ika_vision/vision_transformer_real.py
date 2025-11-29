#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import OccupancyGrid
from ika_msgs.msg import ColorFilter

from rclpy.exceptions import ROSInterruptException
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from ament_index_python import get_package_share_directory
import os


class VisionTransformerConfig:
    def __init__(self):
        self.map_width = 75
        self.map_height = 75
        self.resolution = 0.08  # meters per pixel
        self.points_src = [(46, 0), (274, 0), (320, 240), (0, 240)]
        self.points_dst = [
            (0, 0),
            (self.map_width, 0),
            (45, self.map_height),
            (30, self.map_height),
        ]

        self.unknown_right = [
            (self.map_width, 0),
            (self.map_width, self.map_height),
            (58, self.map_height),
            (57, 63),
        ]
        self.unknown_left = [
            (0, 0),
            (0, self.map_height),
            (17, self.map_height),
            (18, 63),
        ]
        self.hsv_thresholds = {
            "blue": [(0, 104, 51), (112, 255, 255)],
        }
        self.blur_iters = 2
        self.blur_kernel = (3, 3)
        self.inflation_radius = 1.3  # meters


class VisionTransformer(Node):
    def __init__(self):
        super().__init__("vision_transformer")
        self.cv_bridge = CvBridge()
        self.conf = VisionTransformerConfig()
        self.cb_group = MutuallyExclusiveCallbackGroup()

        self.front_cam_sub = self.create_subscription(
            Image,
            "/front_camera/image_raw",
            self.front_cam_callback,
            10,
            callback_group=self.cb_group,
        )
        self.pers_img_pub = self.create_publisher(Image, "/ika_vision/pers_img", 10)
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, "/ika_vision/occupancy_grid", 10
        )
        self.color_filter_pub = self.create_publisher(
            ColorFilter, "/ika_vision/color_filter", 10
        )

        self.occupancy_grid_msg = self.init_occ_grid_msg()
        self.pers_img = None
        self.occ_grid = None
        self.color_filter = ColorFilter()

        self.create_timer(
            0.125, self.publish_occupancy_grid, callback_group=self.cb_group
        )

    def front_cam_callback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg)
        self.occ_grid = self.apply_transform(img)
        assert self.occ_grid.shape == (75, 75), "Occupancy grid shape mismatch"
        self.pers_img = cv2.cvtColor(self.occ_grid, cv2.COLOR_GRAY2BGR)

    def apply_transform(self, img):
        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        img = self.blur(img)
        mask = self.hsv_filter(img)
        mask = self.perspective_transform(mask)
        return mask

    def hsv_filter(self, img):
        # Reset color filter for current frame
        self.color_filter = ColorFilter()

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # assert img_hsv.shape[:2] == (320, 240), "Image shape mismatch"
        mask = np.zeros_like(img_hsv[:, :, 0])
        for color, (lower, upper) in self.conf.hsv_thresholds.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask_color = cv2.inRange(img_hsv, lower, upper)
            has_color = np.any(mask_color)

            if has_color:
                if color == "red":
                    self.color_filter.is_red = True
                elif color == "orange":
                    self.color_filter.is_orange = True
                elif color == "white":
                    self.color_filter.is_white = True
                elif color == "dirt":
                    pixel_count = np.sum(mask_color > 0)
                    # self.get_logger().info(f"Dirt detected! Pixels: {pixel_count}")
                    # Save dirt mask for debugging
                    # cv2.imwrite("/tmp/debug_mask_dirt.jpg", mask_color)
                    self.color_filter.is_dirt = True

            if color != "dirt":
                mask = cv2.bitwise_or(mask, mask_color)

        return mask

    def perspective_transform(self, img):
        src = np.array(self.conf.points_src, np.float32)
        dst = np.array(self.conf.points_dst, np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(img, M, (self.conf.map_width, self.conf.map_height))

    def unknown_area(self, mask):
        unknown_right = np.array(self.conf.unknown_right, np.int32)
        unknown_left = np.array(self.conf.unknown_left, np.int32)

        cv2.fillPoly(mask, [unknown_right], 100)
        cv2.fillPoly(mask, [unknown_left], 100)

        return mask

    def init_occ_grid_msg(self):
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.frame_id = "base_link"
        occupancy_grid_msg.info.resolution = 0.08  # Set the resolution of the grid
        occupancy_grid_msg.info.width = self.conf.map_width
        occupancy_grid_msg.info.height = self.conf.map_height
        occupancy_grid_msg.info.origin.position.x = 0.0
        occupancy_grid_msg.info.origin.position.y = -3.0
        occupancy_grid_msg.info.origin.position.z = 0.0

        return occupancy_grid_msg

    def blur(self, img):
        for _ in range(self.conf.blur_iters):
            img = cv2.blur(img, self.conf.blur_kernel)
        return img

    def apply_inflation(self, occupancy_grid: np.array):
        inflation_radius_px = int(self.conf.inflation_radius / self.conf.resolution)
        obstacle_mask = (occupancy_grid == 100).astype(np.uint8)
        distance_transform = cv2.distanceTransform(1 - obstacle_mask, cv2.DIST_L2, 5)

        inflated_grid = occupancy_grid.copy()
        within_radius_mask = (distance_transform <= inflation_radius_px) & (
            (occupancy_grid != 100) & (occupancy_grid != -1)
        )

        inflated_grid[within_radius_mask] = (
            100 * (1 - (distance_transform[within_radius_mask] / inflation_radius_px))
        ).astype(np.int8)

        return inflated_grid

    def publish_occupancy_grid(self):
        # Resize the image to 150x150
        if self.occ_grid is None:
            self.get_logger().warn("No occ. grid data to publish.")
            return

        occ_grid = ((self.occ_grid // 255.0) * 100).astype(np.uint8)
        occ_grid = self.unknown_area(occ_grid)
        kernel = np.ones((3, 3), np.uint8)
        occ_grid = cv2.morphologyEx(occ_grid, cv2.MORPH_CLOSE, kernel)
        occ_grid = cv2.morphologyEx(occ_grid, cv2.MORPH_OPEN, kernel)

        occ_grid = np.transpose(occ_grid, (1, 0))
        occ_grid = cv2.flip(occ_grid, 1)
        occ_grid = cv2.flip(occ_grid, 0)

        occ_grid = self.apply_inflation(occ_grid)
        flattened_grid = occ_grid.flatten()

        self.occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.occupancy_grid_msg.data = flattened_grid.tolist()
        self.occupancy_grid_pub.publish(self.occupancy_grid_msg)

        msg = self.cv_bridge.cv2_to_imgmsg(self.pers_img)
        self.pers_img_pub.publish(msg)

        self.color_filter_pub.publish(self.color_filter)


def main():
    try:
        rclpy.init()
        node = VisionTransformer()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except ROSInterruptException:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
