#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rake_core.constants import Topics, Services
from rake_msgs.msg import SystemState, ConfigUpdated, DeviceState
from rake_msgs.srv import (
    UpdateConfig,
    UpdateSystemState,
    UpdateDeviceState,
    LoadConfig,
    SaveConfig,
)
from rake_core.states import SystemStateEnum, DeviceStateEnum, SystemModeEnum
import json
import os
import signal


class Broker(Node):
    def __init__(self):
        super().__init__("broker")
        self.declare_parameter("state", SystemStateEnum.MANUAL)
        self.declare_parameter("mode", SystemModeEnum.SIMULATION)
        self.declare_parameter("mobility", 1)
        self.state = self.get_parameter("state").get_parameter_value().integer_value
        self.mode = self.get_parameter("mode").get_parameter_value().integer_value
        self.mobility = (
            self.get_parameter("mobility").get_parameter_value().integer_value
        )

        self.node_configs = {}
        self.device_states = {}

        self.system_state_pub = self.create_publisher(
            SystemState, Topics.SYSTEM_STATE, 10
        )
        self.config_updated_pub = self.create_publisher(
            ConfigUpdated, Topics.CONFIG_UPDATE, 10
        )
        self.device_state_pub = self.create_publisher(
            DeviceState, "/rake/device_state", 10
        )

        self.update_config_srv = self.create_service(
            UpdateConfig, Services.CONFIG_UPDATE, self.on_update_config_called
        )
        self.update_system_state_srv = self.create_service(
            UpdateSystemState,
            Services.SYSTEM_STATE,
            self.on_update_system_state_called,
        )
        self.update_device_state_srv = self.create_service(
            UpdateDeviceState,
            "/rake/device_state_client",
            self.on_update_device_state_called,
        )
        self.load_config_srv = self.create_service(
            LoadConfig, Services.LOAD_CONFIG, self.on_load_config_called
        )

        self.save_config_srv = self.create_service(
            SaveConfig, Services.SAVE_CONFIG, self.on_save_config_called
        )

        self.config_dir = os.path.join(os.getcwd(), "src", "rake_core", "config")

        self.get_logger().info("Broker node started")

    def on_load_config_called(self, request, response):
        config_file = os.path.join(self.config_dir, f"{request.preset_name}.json")
        if not os.path.exists(config_file):
            response.success = False
            self.get_logger().info(f"Config file not found: {request.preset_name}")
            return response

        try:
            with open(config_file, "r") as f:
                loaded_configs = json.load(f)
        except Exception as e:
            self.get_logger().warn(f"Could not load config: {e}")
            response.success = False
            return response

        for device, config in loaded_configs.items():
            self.node_configs[device] = config
            cfg_msg = ConfigUpdated()
            cfg_msg.device = device
            cfg_msg.json = json.dumps(config)
            self.config_updated_pub.publish(cfg_msg)
            self.get_logger().info(f"Config loaded for {device}")

        response.success = True
        return response

    def on_save_config_called(self, request, response):
        config_file = os.path.join(self.config_dir, f"{request.preset_name}.json")

        self.get_logger().info(f"Saving config to {config_file}")
        try:
            with open(config_file, "w") as f:
                json.dump(self.node_configs, f)
            response.success = True
            self.get_logger().info(f"Config saved {config_file}")

        except Exception as e:
            response.success = False
            self.get_logger().warn(f"Failed to save config: {e}")

        return response

    def on_update_config_called(self, request, response):
        """
        Callback for the UpdateConfig service.
        Updates the configuration of the system and publishes a ConfigUpdated message.
        """
        try:
            device = request.device
            config = json.loads(request.json)
            self.node_configs[device] = config

            config_updated_msg = ConfigUpdated()
            config_updated_msg.device = device
            config_updated_msg.json = json.dumps(config)
            self.config_updated_pub.publish(config_updated_msg)
            response.success = True

            self.get_logger().info(f"Configuration updated for {device}")
        except Exception as e:
            response.success = False
            self.get_logger().error(f"Failed to update config: {str(e)}")

        return response

    def on_update_system_state_called(self, request, response):
        """
        Callback for the UpdateSystemState service.
        Updates the system state and publishes a SystemState message.
        """
        try:
            state = SystemState()
            state.state = request.state
            state.mode = request.mode
            state.mobility = request.mobility

            if state.state == SystemStateEnum.SHUTDOWN:
                self.get_logger().info("Received shutdown signal, shutting down")
                self.create_timer(5, self.shutdown_callback)

            self.system_state_pub.publish(state)

            # update system state attributes
            self.state = state.state
            self.mode = state.mode
            self.mobility = state.mobility

            response.success = True
            self.get_logger().info("System state updated successfully")
            self.get_logger().info(
                f"System state: {state.state}, Mode: {state.mode}, Mobility: {state.mobility}"
            )

        except Exception as e:
            response.success = False
            self.get_logger().error(f"Failed to update system state: {str(e)}")

        return response

    def on_update_device_state_called(self, request, response):
        """
        Callback for the UpdateDeviceState service.
        Updates the state of a device and publishes a DeviceState message.
        """
        try:
            device_state = DeviceState()
            device_state.device = request.device
            device_state.state = request.state

            if self.device_states.get(device_state.device) == None:
                # then publish system state, node configs and device states
                system_state = SystemState()
                system_state.state = self.state
                system_state.mode = self.mode
                system_state.mobility = self.mobility
                self.system_state_pub.publish(system_state)

                for device_id, state in self.device_states.items():
                    device_state_msg = DeviceState()
                    device_state_msg.device = device_id
                    device_state_msg.state = state
                    self.device_state_pub.publish(device_state_msg)

                for device_id, node_config in self.node_configs.items():
                    config_updated_msg = ConfigUpdated()
                    config_updated_msg.device = device_id
                    config_updated_msg.json = json.dumps(node_config)
                    self.config_updated_pub.publish(config_updated_msg)

            self.device_states[device_state.device] = device_state.state
            self.device_state_pub.publish(device_state)
            response.success = True
            # self.get_logger().info(
            #     f"Device {device_state.device} state updated to {DeviceStateEnum(device_state.state).name}"
            # )

        except Exception as e:
            response.success = False
            self.get_logger().error(f"Failed to update device state: {e}")
        return response

    def shutdown_callback(self):
        self.get_logger().info("Shutting down...")
        self.destroy_node()
        os.kill(os.getpid(), signal.SIGINT)
        self.get_logger().info("Broker node has been shut down.")


if __name__ == "__main__":
    rclpy.init()
    broker = Broker()
    rclpy.spin(broker)
    rclpy.shutdown()
