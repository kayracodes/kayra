#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

cv_bridge = CvBridge()


class Hsvfilter(Node):

    def __init__(self):
        super().__init__("hsv_filter_node")
        self.get_logger().info("Hsv filter node has been started")

        self.camera_sub = self.create_subscription(
            Image, "/front_camera/image_raw", self.camera_callback, 10
        )
        self.hsv_pub = self.create_publisher(Image, "/hsv_filter", 10)
        self.init_trackbar()

    def init_trackbar(self):
        # Create three sets of trackbars for white, red, orange
        for color in ["White", "Red", "Orange"]:
            win = f"Trackbars_{color}"
            cv2.namedWindow(win)
            cv2.createTrackbar(f"H Min", win, 0, 179, lambda x: None)
            cv2.createTrackbar(f"H Max", win, 179, 179, lambda x: None)
            cv2.createTrackbar(f"S Min", win, 0, 255, lambda x: None)
            cv2.createTrackbar(f"S Max", win, 255, 255, lambda x: None)
            cv2.createTrackbar(f"V Min", win, 0, 255, lambda x: None)
            cv2.createTrackbar(f"V Max", win, 255, 255, lambda x: None)

    def camera_callback(self, msg):
        img = cv_bridge.imgmsg_to_cv2(msg)
        masks = []
        for color in ["White", "Red", "Orange"]:
            masks.append(self.hsv_filter(img, color))
        # Combine masks with bitwise OR
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = cv2.bitwise_or(final_mask, mask)
        filtered_raw = cv_bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
        self.hsv_pub.publish(filtered_raw)
        cv2.imshow("Filtered Image", final_mask)
        cv2.waitKey(1)

    def hsv_filter(self, img, color_name):
        win = f"Trackbars_{color_name}"
        img_blur = cv2.GaussianBlur(img, (7, 7), 1.5)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H Min", win)
        h_max = cv2.getTrackbarPos("H Max", win)
        s_min = cv2.getTrackbarPos("S Min", win)
        s_max = cv2.getTrackbarPos("S Max", win)
        v_min = cv2.getTrackbarPos("V Min", win)
        v_max = cv2.getTrackbarPos("V Max", win)

        lower = (h_min, s_min, v_min)
        upper = (h_max, s_max, v_max)
        mask = cv2.inRange(img_hsv, lower, upper)
        return mask


def main(args=None):
    rclpy.init(args=args)
    node = Hsvfilter()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
