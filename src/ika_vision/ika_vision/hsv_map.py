#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

cv_bridge = CvBridge()


class Hsvmap(Node):

    def __init__(self):
        super().__init__("hsv_map_node")
        self.get_logger().info("Hsv map node has been started")

        self.camera_sub = self.create_subscription(
            Image, "/shoot_camera/image_raw", self.camera_callback, 10
        )
        self.hsv_pub = self.create_publisher(Image, "/hsv_map", 10)

    def camera_callback(self, msg):
        img = cv_bridge.imgmsg_to_cv2(msg)
        filtered = self.hsv_map(img)
        filtered_raw = cv_bridge.cv2_to_imgmsg(filtered, encoding="mono8")
        self.hsv_pub.publish(filtered_raw)
        cv2.imshow("Filtered Image", filtered)
        cv2.waitKey(1)

    def hsv_map(self, img):
        img_blur = cv2.GaussianBlur(img, (7, 7), 1.5)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        lower_red = (118, 104, 81)
        upper_red = (179, 255, 255)
        mask_red = cv2.inRange(img_hsv, lower_red, upper_red)

        lower_orange = (0, 104, 177)
        upper_orange = (111, 255, 255)
        mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)

        lower_white = (0, 0, 174)
        upper_white = (179, 50, 255)
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_red, mask_white)
        combined_mask = cv2.bitwise_or(combined_mask, mask_orange)

        inverted_mask = cv2.bitwise_not(combined_mask)

        kernel = np.ones((7, 7), np.uint8)
        clean_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_OPEN, kernel)

        return clean_mask


def main(args=None):
    rclpy.init(args=args)
    node = Hsvmap()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
