#!/usr/bin/env python3
"""
System State Manager
Manages robot operational states and parameter configurations
Integrates with rake_core for centralized state management
"""

import rclpy
import json
import os
from typing import Dict, Any
from ament_index_python.packages import get_package_share_directory

# Import your rake_core components
from rake_core.core import Core
from rake_core.states import SystemState, DeviceState

# Messages
from std_msgs.msg import String


class RobotSystemState:
    """Enhanced system states for parkour robot"""

    # Basic autonomy modes
    NORMAL_AUTONOMY = "normal_autonomy"
    FAST_AUTONOMY = "fast_autonomy"
    SLOW_AUTONOMY = "slow_autonomy"
    SLOW_RAMP_AUTONOMY = "slow_ramp_autonomy"

    # Specialized navigation modes
    SHARP_TURN_AUTONOMY = "sharp_turn_autonomy"
    TRAFFIC_CONE_MODE = "traffic_cone_mode"

    # Operational modes
    MANUAL_MODE = "manual_mode"
    EMERGENCY_STOP = "emergency_stop"

    # Task-specific modes
    RAMP_APPROACH_MODE = "ramp_approach_mode"
    WATER_CROSSING_MODE = "water_crossing_mode"
    SHOOTING_MODE = "shooting_mode"


class SystemStateManager(Core):
    """
    Enhanced system state manager that extends rake_core.Core
    Manages parameter configurations for different robot operational states
    """

    def __init__(self):
        super().__init__("system_state_manager")

        # Current system state
        self.current_system_state = RobotSystemState.NORMAL_AUTONOMY

        # Load parameter configurations from JSON file
        self.parameter_configs = self._load_parameter_configs()

        # Publishers for state information
        self.system_state_pub = self.create_publisher(
            String, "/system_state_manager/current_state", 10
        )

        # Service for external state change requests
        from std_srvs.srv import SetString

        self.state_change_service = self.create_service(
            SetString, "set_system_state", self.handle_state_change_request
        )

        # Timer for publishing current state
        self.create_timer(1.0, self.publish_current_state)

        self.get_logger().info(
            f"System State Manager initialized in {self.current_system_state}"
        )

    def _load_parameter_configs(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load parameter configurations from JSON file
        Format: SYSTEM_STATE::NODE_NAME::ATTRIBUTE_NAME -> ATTRIBUTE_VALUE
        """
        try:
            config_file = os.path.join(
                get_package_share_directory("ika_controller"),
                "config",
                "system_state_configs.json",
            )

            with open(config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.get_logger().warn("System state config file not found, using defaults")
            return self._get_default_configs()
        except Exception as e:
            self.get_logger().error(f"Failed to load config file: {e}")
            return self._get_default_configs()

    def _get_default_configs(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Default parameter configurations for each system state"""
        return {
            RobotSystemState.NORMAL_AUTONOMY: {
                "path_planner": {
                    "max_linear_velocity": 0.5,
                    "max_angular_velocity": 1.0,
                    "path_resolution": 0.05,
                    "planning_frequency": 10.0,
                },
                "path_tracker": {
                    "kp_linear": 1.0,
                    "kp_angular": 2.0,
                    "lookahead_distance": 0.3,
                    "max_linear_velocity": 0.5,
                    "max_angular_velocity": 1.0,
                },
                "vision_transformer": {
                    "inflation_radius": 1.2,
                    "detection_frequency": 8.0,
                    "hsv_threshold_confidence": 0.6,
                },
                "sign_detector": {
                    "confidence_threshold": 0.55,
                    "detection_frequency": 10.0,
                },
            },
            RobotSystemState.FAST_AUTONOMY: {
                "path_planner": {
                    "max_linear_velocity": 0.8,
                    "max_angular_velocity": 1.5,
                    "path_resolution": 0.1,
                    "planning_frequency": 15.0,
                },
                "path_tracker": {
                    "kp_linear": 1.2,
                    "kp_angular": 2.5,
                    "lookahead_distance": 0.5,
                    "max_linear_velocity": 0.8,
                    "max_angular_velocity": 1.5,
                },
                "vision_transformer": {
                    "inflation_radius": 1.0,
                    "detection_frequency": 12.0,
                    "hsv_threshold_confidence": 0.5,
                },
                "sign_detector": {
                    "confidence_threshold": 0.5,
                    "detection_frequency": 15.0,
                },
            },
            RobotSystemState.SLOW_RAMP_AUTONOMY: {
                "path_planner": {
                    "max_linear_velocity": 0.2,
                    "max_angular_velocity": 0.5,
                    "path_resolution": 0.02,
                    "planning_frequency": 5.0,
                },
                "path_tracker": {
                    "kp_linear": 0.8,
                    "kp_angular": 1.5,
                    "lookahead_distance": 0.15,
                    "max_linear_velocity": 0.2,
                    "max_angular_velocity": 0.5,
                },
                "vision_transformer": {
                    "inflation_radius": 1.5,
                    "detection_frequency": 15.0,
                    "hsv_threshold_confidence": 0.7,
                },
                "sign_detector": {
                    "confidence_threshold": 0.6,
                    "detection_frequency": 20.0,
                },
            },
            RobotSystemState.TRAFFIC_CONE_MODE: {
                "path_planner": {
                    "max_linear_velocity": 0.3,
                    "max_angular_velocity": 0.8,
                    "path_resolution": 0.03,
                    "planning_frequency": 12.0,
                },
                "path_tracker": {
                    "kp_linear": 0.9,
                    "kp_angular": 1.8,
                    "lookahead_distance": 0.2,
                    "max_linear_velocity": 0.3,
                    "max_angular_velocity": 0.8,
                },
                "vision_transformer": {
                    "inflation_radius": 0.8,
                    "detection_frequency": 20.0,
                    "hsv_threshold_confidence": 0.8,
                    # Enhanced orange detection for traffic cones
                    "hsv_thresholds": {
                        "orange": [[0, 104, 177], [111, 255, 255]],
                        "red": [[118, 104, 81], [179, 255, 255]],
                        "white": [[0, 0, 174], [179, 50, 255]],
                    },
                },
                "sign_detector": {
                    "confidence_threshold": 0.7,
                    "detection_frequency": 25.0,
                },
            },
            RobotSystemState.SHOOTING_MODE: {
                "path_planner": {
                    "max_linear_velocity": 0.1,
                    "max_angular_velocity": 0.3,
                    "path_resolution": 0.01,
                    "planning_frequency": 2.0,
                },
                "path_tracker": {
                    "kp_linear": 0.5,
                    "kp_angular": 1.0,
                    "lookahead_distance": 0.1,
                    "max_linear_velocity": 0.1,
                    "max_angular_velocity": 0.3,
                },
                "vision_transformer": {
                    "inflation_radius": 2.0,
                    "detection_frequency": 5.0,
                    "hsv_threshold_confidence": 0.9,
                },
                "sign_detector": {
                    "confidence_threshold": 0.8,
                    "detection_frequency": 30.0,
                },
            },
        }

    def handle_state_change_request(self, request, response):
        """Handle external requests to change system state"""
        new_state = request.data

        if self._is_valid_state(new_state):
            success = self.change_system_state(new_state)
            response.success = success
            response.message = (
                f"System state changed to {new_state}"
                if success
                else "State change failed"
            )
        else:
            response.success = False
            response.message = f"Invalid system state: {new_state}"

        return response

    def change_system_state(self, new_state: str) -> bool:
        """
        Change the system state and update all node configurations
        """
        if not self._is_valid_state(new_state):
            self.get_logger().error(f"Invalid system state: {new_state}")
            return False

        if new_state == self.current_system_state:
            self.get_logger().info(f"Already in state {new_state}")
            return True

        self.get_logger().info(
            f"Changing system state: {self.current_system_state} -> {new_state}"
        )

        # Get parameter configuration for new state
        if new_state not in self.parameter_configs:
            self.get_logger().error(f"No configuration found for state {new_state}")
            return False

        state_config = self.parameter_configs[new_state]

        # Update configuration for each registered node
        for node_name, node_params in state_config.items():
            self._update_node_configuration(node_name, node_params)

        # Update current state
        old_state = self.current_system_state
        self.current_system_state = new_state

        # Trigger system state change in rake_core
        self._trigger_system_state_change(old_state, new_state)

        self.get_logger().info(f"System state successfully changed to {new_state}")
        return True

    def _update_node_configuration(self, node_name: str, params: Dict[str, Any]):
        """Update configuration for a specific node using rake_core services"""
        try:
            # Use rake_core's configuration update mechanism
            self.update_node_config(node_name, params)
            self.get_logger().debug(f"Updated {node_name} configuration: {params}")
        except Exception as e:
            self.get_logger().error(f"Failed to update {node_name} configuration: {e}")

    def _trigger_system_state_change(self, old_state: str, new_state: str):
        """Trigger system state change notification through rake_core"""
        try:
            # Use rake_core's system state change mechanism
            self.change_system_state_internal(old_state, new_state)
        except Exception as e:
            self.get_logger().error(f"Failed to trigger system state change: {e}")

    def _is_valid_state(self, state: str) -> bool:
        """Check if the given state is valid"""
        valid_states = [
            RobotSystemState.NORMAL_AUTONOMY,
            RobotSystemState.FAST_AUTONOMY,
            RobotSystemState.SLOW_AUTONOMY,
            RobotSystemState.SLOW_RAMP_AUTONOMY,
            RobotSystemState.SHARP_TURN_AUTONOMY,
            RobotSystemState.TRAFFIC_CONE_MODE,
            RobotSystemState.MANUAL_MODE,
            RobotSystemState.EMERGENCY_STOP,
            RobotSystemState.RAMP_APPROACH_MODE,
            RobotSystemState.WATER_CROSSING_MODE,
            RobotSystemState.SHOOTING_MODE,
        ]
        return state in valid_states

    def publish_current_state(self):
        """Publish current system state"""
        msg = String()
        msg.data = self.current_system_state
        self.system_state_pub.publish(msg)

    def get_current_state_config(self, node_name: str = None) -> Dict[str, Any]:
        """Get configuration for current state (optionally for specific node)"""
        if self.current_system_state not in self.parameter_configs:
            return {}

        state_config = self.parameter_configs[self.current_system_state]

        if node_name:
            return state_config.get(node_name, {})

        return state_config


def main(args=None):
    rclpy.init(args=args)

    state_manager = SystemStateManager()

    try:
        rclpy.spin(state_manager)
    except KeyboardInterrupt:
        pass
    finally:
        state_manager.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
