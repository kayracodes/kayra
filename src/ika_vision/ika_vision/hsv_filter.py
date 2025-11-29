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
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("H Min", "Trackbars", 0, 179, lambda x: None)
        cv2.createTrackbar("H Max", "Trackbars", 179, 179, lambda x: None)
        cv2.createTrackbar("S Min", "Trackbars", 0, 255, lambda x: None)
        cv2.createTrackbar("S Max", "Trackbars", 255, 255, lambda x: None)
        cv2.createTrackbar("V Min", "Trackbars", 0, 255, lambda x: None)
        cv2.createTrackbar("V Max", "Trackbars", 255, 255, lambda x: None)

    def camera_callback(self, msg):
        img = cv_bridge.imgmsg_to_cv2(msg)
        filtered = self.hsv_filter(img)
        filtered_raw = cv_bridge.cv2_to_imgmsg(filtered, encoding="mono8")
        self.hsv_pub.publish(filtered_raw)
        cv2.imshow("Filtered Image", filtered)
        cv2.waitKey(1)

    def hsv_filter(self, img):
        img_blur = cv2.GaussianBlur(img, (7, 7), 1.5)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H Min", "Trackbars")
        h_max = cv2.getTrackbarPos("H Max", "Trackbars")
        s_min = cv2.getTrackbarPos("S Min", "Trackbars")
        s_max = cv2.getTrackbarPos("S Max", "Trackbars")
        v_min = cv2.getTrackbarPos("V Min", "Trackbars")
        v_max = cv2.getTrackbarPos("V Max", "Trackbars")

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
