#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import threading
import os
import time
import cv_bridge
import subprocess
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor

camera_width = 1920
camera_height = 1080
# right_udev = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_CA62B6DF-video"
# right_udev = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_944FA6DF-video"
right_udev = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_200901010001-video"


class CameraNodeConfig:
    def __init__(self):
        self.refresh_rate = 8
        self.output_width = 1920
        self.output_height = 1080
        self.scan_rate = 1.0


class CameraNode(Node):
    def __init__(self, udev_path):
        super().__init__("shoot_camera_node")
        self.config = CameraNodeConfig()
        self.camera_publisher = self.create_publisher(
            Image, "/shoot_camera/image_raw", 10
        )
        self.bridge = cv_bridge.CvBridge()

        self.camera_thread = None
        self.camera_kill = False
        self.camera_path = udev_path
        self.capture = None

        self.create_thread()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def cleanup(self):
        """Cleanup method to properly shutdown camera resources"""
        self.get_logger().info("Cleaning up camera resources...")
        self.camera_kill = True

        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.get_logger().info("Camera cleanup completed")

    def create_thread(self):
        if self.camera_thread is not None:
            self.camera_kill = True
            if self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2.0)
            self.camera_thread = None

        self.camera_kill = False
        self.camera_thread = threading.Thread(target=self.camera_worker)
        self.camera_thread.daemon = (
            True  # Make thread daemon so it dies with main process
        )
        self.camera_thread.start()

    def camera_worker(self):
        capture = None
        try:
            while rclpy.ok() and not self.camera_kill:
                try:
                    if not os.path.exists(self.camera_path):
                        self.get_logger().warn(
                            f"Camera path {self.camera_path} does not exist."
                        )
                        time.sleep(self.config.scan_rate)
                        continue

                    capture = cv2.VideoCapture(self.camera_path)
                    if capture is None:
                        capture = cv2.VideoCapture(
                            self.camera_path[:-1] + str(1 - int(self.camera_path[-1]))
                        )
                    if capture is None or not capture.isOpened():
                        self.get_logger().warn(
                            f"Failed to open camera at {self.camera_path}."
                        )
                        self.get_logger().info(self.camera_path)
                        time.sleep(self.config.scan_rate)
                        continue

                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.output_width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.output_height)

                    # Store capture reference for cleanup
                    self.capture = capture

                except Exception as e:
                    self.get_logger().error(f"Error initializing camera: {e}")
                    time.sleep(self.config.scan_rate)
                    continue

                while rclpy.ok() and capture is not None and not self.camera_kill:
                    try:
                        ret, frame = capture.read()

                        if not ret or frame is None:
                            self.get_logger().warn(
                                f"Failed to read frame from camera at {self.camera_path}."
                            )
                            break

                        frame = cv2.resize(
                            frame, (self.config.output_width, self.config.output_height)
                        )

                        # Only publish if node is still active
                        if not self.camera_kill and rclpy.ok():
                            self.camera_publisher.publish(
                                self.bridge.cv2_to_imgmsg(frame, "bgr8")
                            )

                        time.sleep(1.0 / self.config.refresh_rate)

                    except Exception as e:
                        self.get_logger().error(f"Error reading frame: {e}")
                        break

                # Clean up capture for this iteration
                if capture is not None:
                    capture.release()
                    capture = None
                    self.capture = None

        except Exception as e:
            self.get_logger().error(f"Camera worker error: {e}")
        finally:
            # Final cleanup
            if capture is not None:
                capture.release()
                capture = None
            self.capture = None
            self.get_logger().info("Camera worker thread ended")


if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down camera node...")
        if "node_right" in locals():
            node_right.cleanup()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.init()
        right_index0 = subprocess.run(
            ["v4l2-ctl", "--device=2", "--all"], stdout=subprocess.PIPE
        )

        correct_right = None

        if "exposure_time_absolute" in right_index0.stdout.decode("utf-8"):
            correct_right = 0
        else:
            correct_right = 1

        node_right = CameraNode(
            right_udev + "-index" + str(correct_right),
        )
        executor = MultiThreadedExecutor()

        executor.add_node(node_right)

        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, shutting down...")
        finally:
            # Cleanup
            node_right.cleanup()
            node_right.destroy_node()
            executor.shutdown()
            rclpy.shutdown()

    except Exception as e:
        print(f"Error in camera node: {e}")
        if "node_right" in locals():
            node_right.cleanup()
        sys.exit(1)
