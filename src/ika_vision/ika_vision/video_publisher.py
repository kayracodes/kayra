#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
import time
from cv_bridge import CvBridge


class VideoPublisher(Node):
    def __init__(self, video_path):
        super().__init__("video_publisher")
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(
            CompressedImage, "/front_camera/image_raw/compressed", 10
        )
        self.image_publisher = self.create_publisher(
            Image, "/front_camera/image_raw", 10
        )
        self.cap = cv2.VideoCapture(video_path)
        self.timer_period = 1.0 / 8.0  # 8 FPS
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video file reached.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            return
        # Compressed Image
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = cv2.imencode(".jpg", frame)[1].tobytes()
        self.publisher_.publish(msg)

        # Raw Image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(img_msg)
        self.get_logger().info("Published frame")


def main(args=None):
    rclpy.init(args=args)
    import sys

    if len(sys.argv) < 2:
        print("Usage: ros2 run ika_vision video_publisher <video_path>")
        return
    video_path = sys.argv[1]
    node = VideoPublisher(video_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
