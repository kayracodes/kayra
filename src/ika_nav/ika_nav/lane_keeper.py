#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import cv2
from cv_bridge import CvBridge


class LaneKeeperConfig:
    def __init__(self):
        pass


class LaneKeeper(Node):
    def __init__(self):
        super().__init__("lane_keeper")
        self.conf = LaneKeeperConfig()
        self.bridge = CvBridge()

        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10
        )

    def map_callback(self, msg):

def main(args=None):
    rclpy.init(args=args)
    node = LaneKeeper()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
