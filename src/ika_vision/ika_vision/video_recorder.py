#!/usr/bin/env python3
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
from pathlib import Path


class VideoRecorder(Node):
    def __init__(self):
        super().__init__("video_recorder")

        # Declare and get camera topic parameter
        self.declare_parameter("camera_topic", "/front_camera/image_raw")
        camera_topic = (
            self.get_parameter("camera_topic").get_parameter_value().string_value
        )

        # Video parameters
        self.fps = 8
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Setup data directory
        self.data_dir = Path("/home/rosgeek/Documents/ika_ws/src/ika_vision/video")
        self.data_dir.mkdir(exist_ok=True)

        # cv_bridge for ROS2 Image conversion
        self.bridge = CvBridge()

        # Video writer setup
        self.video_writer = None
        self.video_path = None
        self.frame_count = 0

        # Subscribe to camera topic
        self.image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f"Video Recorder started. Subscribed to: {camera_topic}. Saving to: {self.data_dir}"
        )

    def get_next_video_number(self):
        """Find the next available video number."""
        video_paths = list(self.data_dir.glob("video*.mp4"))
        return len(video_paths) + 1

    def image_callback(self, msg):
        """Process incoming ROS2 Image messages and write to video."""
        try:
            # Convert ROS2 Image message to OpenCV image using cv_bridge
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if frame is None:
                self.get_logger().warn("Failed to convert ROS2 Image to OpenCV image")
                return

            # Initialize video writer with first frame
            if self.video_writer is None:
                height, width = frame.shape[:2]
                video_number = self.get_next_video_number()
                self.video_path = self.data_dir / f"video{video_number}.mp4"

                self.video_writer = cv2.VideoWriter(
                    str(self.video_path), self.fourcc, self.fps, (width, height)
                )

                self.get_logger().info(f"Started recording: {self.video_path.name}")
                self.get_logger().info(f"Resolution: {width}x{height} @ {self.fps} FPS")

            # Write frame to video
            self.video_writer.write(frame)
            self.frame_count += 1

            # Log progress every 5 seconds of video
            if self.frame_count % (self.fps * 5) == 0:
                duration = self.frame_count / self.fps
                self.get_logger().info(
                    f"Recorded {duration:.1f} seconds ({self.frame_count} frames)"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def destroy_node(self):
        """Clean up when node shuts down."""
        if self.video_writer is not None:
            self.video_writer.release()
            duration = self.frame_count / self.fps
            self.get_logger().info(
                f"Video saved: {self.video_path.name} ({duration:.1f}s, {self.frame_count} frames)"
            )

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        recorder = VideoRecorder()
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.destroy_node()
        rclpy.shutdown()
    finally:
        if "recorder" in locals():
            try:
                recorder.destroy_node()
                rclpy.shutdown()
            except Exception as e:
                pass


if __name__ == "__main__":
    main()
