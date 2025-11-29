#!/usr/bin/env python3
import json
import rclpy
from rake_core.node import Node
from rake_core.constants import Topics, Services
from rake_msgs.msg import SystemState, ConfigUpdated, DeviceState
from rake_msgs.srv import UpdateConfig, UpdateSystemState, UpdateDeviceState
from types import SimpleNamespace
import time


class ExampleNode2Config:
    def __init__(self):
        self.first_num = 10
        self.second_num = 20
        self.first_str = "Foo"
        self.second_str = "Bar"


class ExampleNode2(Node):
    def __init__(self):
        super().__init__("example_node2")

    def init(self):
        self.config_changer()

    def get_default_config(self):
        return json.loads(json.dumps(ExampleNode2Config().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )
        # self.get_logger().info(f"Updated Config: {self.jdump(self.config)}")

    def info_logger(self):
        self.get_logger().info(
            f"Config: first_num={self.config.first_num}, "
            f"second_num={self.config.second_num}, "
            f"first_str='{self.config.first_str}', "
            f"second_str='{self.config.second_str}'"
        )
        self.get_logger().info(
            f"system_state={self.system_state}, system_mode={self.system_mode}, mobility={self.mobility}"
        )
        for device, state in self.device_states.items():
            self.get_logger().info(f"Device {device} state: {state}")
        for node, config in self.node_configs.items():
            self.get_logger().info(f"Node {node} config: {config}")

    def config_changer(self):
        config = self.node_configs["example_node1"]
        config["first_num"] = 15
        config["second_num"] = 25
        config["first_str"] = "Foo"
        config["second_str"] = "Bar"
        config = json.loads(self.jdump(config))
        self.update_config("example_node1", config)


def main():
    rclpy.init()
    node = ExampleNode2()
    Node.run_node(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
