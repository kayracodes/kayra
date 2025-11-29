#!/usr/bin/env python3
import json
import rclpy
from rake_core.node import Node
from rake_core.constants import Topics, Services
from rake_msgs.msg import SystemState, ConfigUpdated, DeviceState
from rake_msgs.srv import UpdateConfig, UpdateSystemState, UpdateDeviceState
from types import SimpleNamespace
import time


class ExampleNode1Config:
    def __init__(self):
        self.first_num = 10
        self.second_num = 20
        self.first_str = "Foo"
        self.second_str = "Bar"


class ExampleNode1(Node):
    def __init__(self):
        super().__init__("example_node1")

    def init(self):
        pass

    def get_default_config(self):
        return json.loads(json.dumps(ExampleNode1Config().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )
        # self.get_logger().info(f"Updated Config: {self.jdump(self.config)}")

    def config_changer(self):
        self.get_logger().info("Changing config...")
        self.get_logger().info(f"Current config: {self.jdump(self.config)}")
        new_config = self.config.copy()
        new_config.first_num += 1
        new_config.second_num += 1
        new_config.first_str = "UpdatedFoo"
        json_config = json.loads(self.jdump(new_config))
        self.update_config(self.id, json_config)


def main():
    rclpy.init()
    node = ExampleNode1()
    node.init()  # Initialize the node
    Node.run_node(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
