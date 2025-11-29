#!/usr/bin/env python3

import rclpy
from rake_core.node import Node
from sensor_msgs.msg import Image, CompressedImage, Imu, NavSatFix
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rake_msgs.srv import LoadConfig, SaveConfig, UpdateConfig, LoadMission
from rake_msgs.msg import SystemState
from ika_msgs.msg import RampFeedback, RampDistance, RoadSign, RoadSignArray, IsInWater
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rake_core.constants import Services
import cv2
import asyncio
import websockets
import threading
import json
import time
import numpy as np
import base64


async_loop = asyncio.new_event_loop()


class Limiter:
    def __init__(self) -> None:
        self.limits = {}
        self.nextAllowance = {}

    def setLimit(self, key, limit):
        self.limits[key] = limit
        self.nextAllowance[key] = 0

    def use(self, key):
        if key not in self.limits:
            return True

        nextUsageAfter = self.nextAllowance[key]
        if nextUsageAfter == 0:
            self.nextAllowance[key] = time.time() + (1.0 / self.limits[key])
            return True

        if time.time() > nextUsageAfter:
            self.nextAllowance[key] = time.time() + (1.0 / self.limits[key])
            return True

        return False


class BroadcastNode(Node):
    def __init__(self):
        super().__init__("igvc_broadcast_node")
        self.port = 4000
        self.host = "0.0.0.0"
        self.client = []
        self.bridge = CvBridge()
        self.current_file = "rake.json"
        self.command_handler = {
            "load_config": self.handle_load_config,
            "save_config": self.handle_save_config,
            "update_config": self.handle_update_config,
            "get_current_config": self.handle_get_current_config,
            "estop_enter": self.handle_estop_enter,
            "estop_exit": self.handle_estop_exit,
            "send_planner": self.handle_send_planner,
        }

        self.limiter = Limiter()
        self.limiter.setLimit("/front_camera/image_raw", 5)
        self.limiter.setLimit("/rear_camera/image_raw", 5)
        self.limiter.setLimit("/shoot_camera/debug_image", 5)
        self.limiter.setLimit("/sign_detector/debug_image", 5)
        self.limiter.setLimit("/ika_nav/debug_image", 5)
        self.limiter.setLimit("/ika_vision/pers_img", 5)

        # Publishers

        self.rear_camera_sub = self.create_subscription(
            Image, "/rear_camera/image_raw", self.rear_camera_callback, 10
        )

        self.shoot_camera_sub = self.create_subscription(
            Image,
            "/shoot_camera/debug_image",
            self.shoot_camera_callback,
            10,
        )

        self.front_camera_sub = self.create_subscription(
            Image,
            "/front_camera/image_raw",
            self.front_camera_callback,
            10,
        )

        self.odom_global_sub = self.create_subscription(
            Odometry, "/odom_noisy", self.odom_noisy_callback, 10
        )

        # self.debug_sub = self.create_subscription(
        #     Image, "/sign_detector/debug_image", self.debug_callback, 10
        # )

        self.pers_mask_sub = self.create_subscription(
            Image, "/ika_vision/pers_mask", self.pers_mask_callback, 10
        )

        self.pers_img_sub = self.create_subscription(
            Image, "/ika_vision/pers_img", self.pers_img_callback, 10
        )

        self.ika_debug = self.create_subscription(
            Image, "/ika_nav/debug_image", self.ika_debug_callback, 10
        )

        self.gps_sub = self.create_subscription(
            NavSatFix, "/gps/fix", self.gps_callback, 10
        )

        self.cmd_sub = self.create_subscription(
            Twist, "/ika_nav/cmd_vel", self.cmd_callback, 10
        )

        self.system_state_sub = self.create_subscription(
            SystemState, "/rake/system_state", self.system_state_callback, 10
        )
        self.road_sign_sub = self.create_subscription(
            RoadSignArray, "/ika_vision/road_signs", self.sign_callback, 10
        )

        self.ramp_sub = self.create_subscription(
            RampDistance, "/ika_vision/ramp_distance", self.ramp_callback, 10
        )
        self.in_water_sub = self.create_subscription(
            IsInWater, "/ika_controller/is_in_water", self.in_water_callback, 10
        )

        self.mission_status_sub = self.create_subscription(
            String, "/mission_controller/status", self.mission_status_callback, 10
        )

        self.load_config_client = self.create_client(LoadConfig, Services.LOAD_CONFIG)

        self.save_config_client = self.create_client(SaveConfig, Services.SAVE_CONFIG)

        self.load_mission_client = self.create_client(
            LoadMission, Services.LOAD_MISSION
        )

        self.thread = threading.Thread(target=self.start_websocket_server)
        self.thread.daemon = True
        self.thread.start()

    def start_websocket_server(self):
        asyncio.set_event_loop(async_loop)
        async_loop.run_until_complete(self.start_server())

    async def start_server(self):
        async with websockets.serve(self.handler, self.host, self.port):
            self.get_logger().info(
                f"WebSocket server started on {self.host}:{self.port}"
            )
            await asyncio.Future()

    async def handler(self, websocket):

        self.client.append(websocket)
        self.get_logger().info(
            f"New client connected, Total clients: {len(self.client)}"
        )

        consumer_task = asyncio.create_task(self.consumer(websocket))
        server_task = asyncio.create_task(self.producer(websocket))
        pending = await asyncio.wait(
            [consumer_task, server_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            for t in task:
                t.cancel()

        if websocket in self.client:
            self.client.remove(websocket)

    def broadcast_image(self, topic_name, msg):
        if not self.client:
            return

        if not self.limiter.use(topic_name):
            return

        cvimg = self.bridge.imgmsg_to_cv2(msg)
        _, img_enc = cv2.imencode(".png", cvimg)

        img_base64 = base64.b64encode(img_enc.tobytes()).decode("utf-8")
        packet = {"topic": topic_name, "format": "png_base64", "image": img_base64}
        self.push_temp(packet)

    def message_to_dict(self, msg):
        result = {}

        if hasattr(msg, "get_fields_and_field_types"):
            for key in msg.get_fields_and_field_types().keys():
                value = getattr(msg, key)

                if hasattr(value, "get_fields_and_field_types"):
                    result[key] = self.message_to_dict(value)
                elif isinstance(value, (list, tuple)):
                    if (
                        hasattr(value[0], "get_fields_and_field_types")
                        if value
                        else False
                    ):
                        result[key] = [self.message_to_dict(item) for item in value]
                    else:
                        result[key] = list(value)
                elif isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                else:
                    result[key] = value

        return result

    def push_temp(self, message, u_id=None):
        json_msg = json.dumps(message)
        asyncio.run_coroutine_threadsafe(self.broadcast(json_msg), async_loop)

    def push(self, topic, data):
        packet = {"mode": "data", "topic": topic}

        data_dict = self.message_to_dict(data)
        packet.update(data_dict)

        json_msg = json.dumps(packet)
        asyncio.run_coroutine_threadsafe(self.broadcast(json_msg), async_loop)

    async def broadcast(self, message):
        for ws in list(self.client):
            try:
                await ws.send(message)
            except:
                self.client.remove(ws)

    async def producer(self, websocket):
        while True:
            await asyncio.sleep(0.1)

    async def consumer(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                mode = data.get("mode")
                handler_command = self.command_handler.get(mode)
                if handler_command:
                    await handler_command(data)
            except Exception as e:
                self.get_logger().error(f"Err: {e}")

    async def handle_load_config(self, data):
        preset_name = data.get("preset_name", "")
        if not preset_name:
            self.get_logger().warn("No preset name provided")
            return

        if not self.load_config_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("LoadConfig service not available")
            return

        self.current_file = f"{preset_name}.json"
        request = LoadConfig.Request()
        request.preset_name = preset_name

        future = self.load_config_client.call_async(request)
        self.get_logger().info(f"{request.preset_name} config file loaded")

    async def handle_save_config(self, data):
        preset_name = data.get("preset_name", "")
        if not preset_name:
            self.get_logger().warn("No preset name entered.")
            return

        if not self.save_config_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("SaveConfig service not available")
            return

        request = SaveConfig.Request()
        request.preset_name = preset_name

        future = self.save_config_client.call_async(request)
        self.get_logger().info(f"{request.preset_name} config file saved")

    async def handle_update_config(self, data):
        device = data.get("device")
        param = data.get("param")
        value = data.get("value")

        if not device or not param:
            self.get_logger().warn(
                f"Missing device or param - device: '{device}', param: '{param}'"
            )
            return

        try:
            current_config = self.node_configs.get(device, {})
            current_config[param] = value

            self.node_configs[device] = current_config

            request = UpdateConfig.Request()
            request.device = device
            request.json = json.dumps(current_config)

            future = self.config_updated_client.call_async(request)

            self.get_logger().info(
                f"Updated config for device '{device}': {param} = {value}"
            )
        except Exception as e:
            self.get_logger().error(f"Update failed {e}")

    async def handle_send_planner(self, data):
        try:
            file_name = data.get("file_name")
            if not file_name:
                self.get_logger().warn("No file name provided for planner data")
                return

            if not self.load_mission_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("LoadMission service not available")
                return

            request = LoadMission.Request()
            request.file_name = file_name

            future = self.load_mission_client.call_async(request)
            self.get_logger().info(f"Sent planner data: {file_name}")

        except Exception as e:
            self.get_logger().error(f"Failed to send planner data: {str(e)}")

    async def handle_get_current_config(self, data):
        try:
            for device, config in self.node_configs.items():
                cfg_msg = {"mode": "config_data", "device": device, "config": config}
                json_msg = json.dumps(cfg_msg)
                await self.broadcast(json_msg)

            self.get_logger().info("Current configuration sent to clients")

        except Exception as e:
            self.get_logger().error(f"Failed to get current config: {str(e)}")

    async def handle_estop_enter(self, data):
        try:
            self.get_logger().info("Emergency stop activated")
            self.set_system_state(
                state=self.system_state, mode=self.system_mode, mobility=0
            )
            self.get_logger().info("System mobility set to 0")

        except Exception as e:
            self.get_logger().error(f"Failed to handle estop_enter: {str(e)}")

    async def handle_estop_exit(self, data):
        try:
            self.get_logger().info("Emergency stop deactivated")
            self.set_system_state(
                state=self.system_state, mode=self.system_mode, mobility=1
            )
            self.get_logger().info("System mobility set to 1")

        except Exception as e:
            self.get_logger().error(f"Failed to handle estop_exit: {str(e)}")

    def rear_camera_callback(self, msg):
        img_cv = self.bridge.imgmsg_to_cv2(msg)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_rgb = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
        self.broadcast_image("/rear_camera/image_raw", img_rgb)

    def shoot_camera_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg)
        cv_img = cv2.resize(cv_img, (640, 480))
        img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        self.broadcast_image("/shoot_camera/debug_image", img_msg)

    def front_camera_callback(self, msg):
        img_cv = self.bridge.imgmsg_to_cv2(msg)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_rgb = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
        self.broadcast_image("/front_camera/image_raw", img_rgb)

    def pers_img_callback(self, msg):
        self.broadcast_image("/ika_vision/pers_img", msg)

    # def debug_callback(self, msg):
    #     with self.image_lock:
    #         self.sign_detector_img = self.bridge.imgmsg_to_cv2(msg)

    #     self.broadcast_combine_img()

    def mission_status_callback(self, msg):
        self.push("/mission_controller/status", msg)

    def in_water_callback(self, msg):
        self.push("/ika_vision/in_water", msg)

    def ika_debug_callback(self, msg):
        self.broadcast_image("/ika_nav/debug_image", msg)

    def pers_mask_callback(self, msg):
        self.broadcast_image("/ika_vision/pers_mask", msg)

    def sign_callback(self, msg):
        self.push("/ika_vision/road_signs", msg)

    def system_state_callback(self, msg):
        self.push("/rake/system_state", msg)

    def odom_noisy_callback(self, msg):
        self.push("/odom_noisy", msg)

    def ramp_callback(self, msg):
        self.push("/ika_vision/ramp_distance", msg)

    def cmd_callback(self, msg):
        self.push("/ika_nav/cmd_vel", msg)

    def gps_callback(self, msg):
        self.push("/gps/fix", msg)


def main(args=None):
    rclpy.init()
    node = BroadcastNode()
    Node.run_node(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
