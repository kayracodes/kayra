#!/usr/bin/env python3
import rclpy
from rclpy.node import Node as ROSNode
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rake_core.constants import Topics, Services
from rake_core.states import SystemStateEnum, DeviceStateEnum, SystemModeEnum
from rake_msgs.msg import SystemState, ConfigUpdated, DeviceState
from rake_msgs.srv import UpdateConfig, UpdateSystemState, UpdateDeviceState
import os
import signal
import json
from threading import Thread
import time


class Node(ROSNode):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.id = node_name
        self.system_state = SystemStateEnum.MANUAL
        self.system_mode = SystemModeEnum.SIMULATION
        self.mobility = 1
        self.device_states = {}
        self.node_configs = {}

        self.system_state_callback_group = MutuallyExclusiveCallbackGroup()
        self.config_updated_callback_group = MutuallyExclusiveCallbackGroup()
        self.update_device_state_callback_group = MutuallyExclusiveCallbackGroup()

        self.system_state_subscriber = self.create_subscription(
            SystemState, Topics.SYSTEM_STATE, self.on_system_state, 10
        )
        self.config_updated_subscriber = self.create_subscription(
            ConfigUpdated,
            Topics.CONFIG_UPDATE,
            self.on_config_updated,
            10,
        )
        self.device_state_subscriber = self.create_subscription(
            DeviceState, "/rake/device_state", self.on_device_state, 10
        )

        self.system_state_client = self.create_client(
            UpdateSystemState,
            Services.SYSTEM_STATE,
            callback_group=self.system_state_callback_group,
        )
        self.config_updated_client = self.create_client(
            UpdateConfig,
            Services.CONFIG_UPDATE,
            callback_group=self.config_updated_callback_group,
        )
        self.update_device_state_client = self.create_client(
            UpdateDeviceState,
            "/rake/device_state_client",
            callback_group=self.update_device_state_callback_group,
        )
        booting_thread = Thread(target=self.boot, daemon=True)
        booting_thread.start()

    def boot(self):
        time.sleep(1)  # Simulate some boot time
        self.set_device_state(DeviceStateEnum.BOOTING)

    def init(self):
        """
        Called for initialization of the node.
        """

    def on_system_state(self, msg: SystemState):
        """
        Called when a system state success is received.

        :param msg: The system state success.
        """

        # If the system state is shutdown, kill this node killing the proces
        if msg.state == SystemStateEnum.SHUTDOWN:
            self.get_logger().info("Received shutdown signal, shutting down")
            os.kill(os.getpid(), signal.SIGKILL)
            return

        oldState = SystemState()
        oldState.state = self.system_state
        oldState.mode = self.system_mode
        oldState.mobility = self.mobility
        self.system_state = msg.state
        self.system_mode = msg.mode
        self.mobility = msg.mobility

        self.system_state_transition(oldState, msg)

    def system_state_transition(self, old_state: SystemState, new_state: SystemState):
        """
        Called after any update to the system state (state, mode, or mobility).

        :param old_state: The old system state.
        :param new_state: The new system state.
        """
        pass

    def on_config_updated(self, msg: ConfigUpdated):
        self.node_configs[msg.device] = json.loads(msg.json)
        if msg.device is None or msg.device != self.id:
            return

        try:
            parsed_json = json.loads(msg.json)
            self.config_updated(parsed_json)
        except Exception as e:
            self.get_logger().error("Failed to parse config update: " + str(e))

    def config_updated(self, json):
        """
        Called when the configuration is updated.
        """
        pass

    def on_device_state(self, msg: DeviceState):
        """
        Called when a device state success is received.

        :param msg: The device state success.
        """
        self.device_states[msg.device] = msg.state
        if msg.device is None or msg.device != self.id:
            return

        self.device_state = msg.state
        if msg.state == DeviceStateEnum.BOOTING:
            default_config = self.get_default_config()
            if default_config is None:
                self.get_logger().error("Default config is None, cannot set it")
                return

            request = UpdateConfig.Request()
            request.device = self.id
            request.json = json.dumps(default_config)

            while not self.config_updated_client.wait_for_service(timeout_sec=2.0):
                if not rclpy.ok():
                    self.get_logger().error("Interrupted while waiting for service")
                    return

            try:
                result = self.config_updated_client.call(request)
                if not result.success:
                    self.get_logger().error("Failed to set default config.")
            except Exception as e:
                self.get_logger().error("Failed to set default config: " + str(e))

            # system can now init itself
            self.get_logger().info("Default config set, initializing...")
            self.config_updated(default_config)
            self.set_device_state(DeviceStateEnum.READY)
            self.init()

    def update_config(self, device, config):
        """
        Calls service to update the configuration of the node.
        """
        request = UpdateConfig.Request()
        request.device = device
        request.json = json.dumps(config)

        while not self.config_updated_client.wait_for_service(timeout_sec=2.0):
            if not rclpy.ok():
                self.get_logger().error("Interrupted while waiting for service")
                return

        try:
            result = self.config_updated_client.call(request)
            if not result.success:
                self.get_logger().error("Failed to set default config: ")
        except Exception as e:
            self.get_logger().error("Failed to set default config: " + str(e))

    def set_device_state(self, state: DeviceStateEnum):
        """
        Calls service to update the device state of the node.
        """
        request = UpdateDeviceState.Request()
        request.device = self.id
        request.state = state

        while not self.update_device_state_client.wait_for_service(timeout_sec=2.0):
            if not rclpy.ok():
                self.get_logger().error("Interrupted while waiting for service")
                return

        try:
            result = self.update_device_state_client.call(request)
            if not result.success:
                self.get_logger().error("Failed to set device state: ")
        except Exception as e:
            self.get_logger().error("Failed to set device state: " + str(e))

    def set_system_state(self, state: SystemStateEnum, mode=None, mobility=None):
        """
        Calls service to update the system state of the node.
        """
        request = UpdateSystemState.Request()
        request.state = state
        request.mode = mode if mode is not None else self.system_mode
        request.mobility = mobility if mobility is not None else self.mobility

        while not self.system_state_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error("Interrupted while waiting for service")
                return

        try:
            result = self.system_state_client.call(request)
            if not result.success:
                self.get_logger().error("Failed to set system state: ")
        except Exception as e:
            self.get_logger().error("Failed to set system state: " + str(e))

    def run_node(node):
        """
        Runs the node with the correct ROS parameters and specifications

        :param node: The node to run.
        """

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
        executor.remove_node(node)

    def run_nodes(nodes):
        """
        Runs the nodes with the correct ROS parameters and specifications

        :param nodes: The nodes to run.
        """

        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        executor.spin()
        for node in nodes:
            executor.remove_node(node)

    def jdump(self, obj):
        isDict = isinstance(obj, dict)
        if not isDict:
            return json.dumps(obj.__dict__)
        else:
            return json.dumps(obj)

    def get_default_config(self):
        """
        Returns the default configuration for this node.
        """
        return {}
