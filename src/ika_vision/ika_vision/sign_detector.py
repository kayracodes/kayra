#!/usr/bin/env python3
import rclpy
from rake_core.node import Node
from rake_core.constants import SIGN_ID_MAP, ID_SIGN_MAP
from ika_msgs.msg import RoadSign, RoadSignArray
from sensor_msgs.msg import CompressedImage, Image
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from types import SimpleNamespace
import cv2
import numpy as np
import json
from cv_bridge import CvBridge
import os


class SignDetectorConfig:
    def __init__(self):
        self.model_path = os.path.join(
            get_package_share_directory("ika_vision"), "ai_models", "sign_model_v4.pt"
        )
        self.confidence_threshold = 0.55
        self.image_topic = "/shoot_camera/image_raw"
        self.signs_topic = "/ika_vision/road_signs"
        self.debug_img_topic = "/sign_detector/debug_image"
        self.inference_size = 832
        self.img_width = 1920
        self.img_height = 1080


class SignDetector(Node):
    def __init__(self):
        super().__init__("sign_detector")

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        self.model = self.load_yolo_model()
        self.bridge = CvBridge()
        self.signs_pub = self.create_publisher(
            RoadSignArray, self.config.signs_topic, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.config.image_topic, self.image_callback, 10
        )
        self.debug_img_pub = self.create_publisher(
            Image, self.config.debug_img_topic, 10
        )

    def get_default_config(self):
        return json.loads(json.dumps(SignDetectorConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def image_callback(self, msg):
        img = cv2.resize(
            self.bridge.imgmsg_to_cv2(msg, "bgr8"),
            (self.config.inference_size, self.config.inference_size),
        )
        if self.model is None:
            return

        results = self.model(img, imgsz=self.config.inference_size)
        road_signs = []
        debug_img = img.copy()

        for result in results:
            for box in result.boxes:
                if box.conf[0] > self.config.confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    w, h = x2 - x1, y2 - y1
                    cls_id = int(box.cls[0])

                    road_sign = RoadSign()
                    road_sign.id = cls_id
                    road_sign.x = int(
                        x1 * self.config.img_width / self.config.inference_size
                    )
                    road_sign.y = int(
                        y1 * self.config.img_height / self.config.inference_size
                    )
                    road_sign.w = int(
                        w * self.config.img_width / self.config.inference_size
                    )
                    road_sign.h = int(
                        h * self.config.img_height / self.config.inference_size
                    )
                    road_signs.append(road_sign)

                    # Draw bounding box on debug image
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = (
                        f"{SIGN_ID_MAP.get(cls_id, 'Unknown')} | ({box.conf[0]:.2f})"
                    )
                    cv2.putText(
                        debug_img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

        if road_signs:
            # Sort by area (w*h) in descending order
            road_signs.sort(key=lambda s: s.w * s.h, reverse=True)

            # Highlight the largest bounding box
            largest_sign = road_signs[0]
            x1, y1, w, h = (
                largest_sign.x * self.config.inference_size // self.config.img_width,
                largest_sign.y * self.config.inference_size // self.config.img_height,
                largest_sign.w * self.config.inference_size // self.config.img_width,
                largest_sign.h * self.config.inference_size // self.config.img_height,
            )
            cv2.rectangle(
                debug_img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 4
            )  # Thicker border

            # Publish road signs
            road_sign_array = RoadSignArray()
            road_sign_array.signs = road_signs
            self.signs_pub.publish(road_sign_array)

        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img)
        self.debug_img_pub.publish(debug_msg)

    # UTILITY FUNCTIONS
    def load_yolo_model(self):
        if not os.path.exists(self.config.model_path):
            self.get_logger().error(
                f"Model file not found at {self.config.model_path}. Please check the path."
            )
            return None
        return YOLO(self.config.model_path)


def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()
    node.init()
    Node.run_node(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
