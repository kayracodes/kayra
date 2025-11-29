#!/usr/bin/env python3
"""
Event Detection Service Node
Provides services for "until_x_happens" functionality by monitoring perception stack
"""
import json
import rclpy

# from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import threading
from typing import Dict, Any, Optional

# Messages
from ika_msgs.msg import RoadSignArray, RampFeedback, IsInWater
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# Services - You'll need to create these service definitions
from rake_msgs.srv import WaitForEvent
from rake_core.constants import Topics, Services, SIGN_ID_MAP

# Import your rake_core base class
from rake_core.node import Node


class EventDetector(Node):
    """
    Service-based event detection node that monitors perception stack
    and provides "until_x_happens" functionality
    """

    def __init__(self):
        super().__init__("event_detector")

        # Callback groups for concurrent service handling
        self.service_callback_group = ReentrantCallbackGroup()
        self.subscription_callback_group = ReentrantCallbackGroup()

        # Current perception state
        self.current_signs = []
        self.current_ramp_feedback = None
        self.current_odom = None
        self.sign_detection_history = {}  # Track when signs were last seen
        self.ramp_detection_history = {}

        # Event monitoring state
        self.active_monitors = {}  # Dictionary of active event monitors
        self.monitor_lock = threading.Lock()

        # Subscribers to perception stack
        self.sign_sub = self.create_subscription(
            RoadSignArray,
            "/ika_vision/road_signs",
            self.sign_callback,
            10,
            callback_group=self.subscription_callback_group,
        )

        self.ramp_sub = self.create_subscription(
            RampFeedback,
            "/ika_vision/ramp_detected",
            self.ramp_callback,
            10,
            callback_group=self.subscription_callback_group,
        )

        self.in_water_sub = self.create_subscription(
            IsInWater,
            "/ika_controller/is_in_water",
            self.in_water_callback,
            10,
            callback_group=self.subscription_callback_group,
        )
        # self.odom_sub = self.create_subscription(
        #     Odometry,
        #     "/odom",
        #     self.odom_callback,
        #     10,
        #     callback_group=self.subscription_callback_group,
        # )

        # Services for different event types
        self.create_service(
            WaitForEvent,
            Services.SIGN_UNDETECTED,
            self.handle_sign_undetected,
            callback_group=self.service_callback_group,
        )

        self.create_service(
            WaitForEvent,
            Services.RAMP_UNDETECTED,
            self.handle_ramp_undetected,
            callback_group=self.service_callback_group,
        )

        self.create_service(
            WaitForEvent,
            Services.RAMP_CLOSE,
            self.handle_ramp_close,
            callback_group=self.service_callback_group,
        )

        self.create_service(
            WaitForEvent,
            Services.RAMP_DETECTED,
            self.handle_ramp_detected,
            callback_group=self.service_callback_group,
        )

        self.create_service(
            WaitForEvent,
            Services.IN_WATER,
            self.handle_in_water,
            callback_group=self.service_callback_group,
        )
        self.create_service(
            WaitForEvent,
            Services.OUT_WATER,
            self.handle_out_water,
            callback_group=self.service_callback_group,
        )
        # Timer for cleaning up old detection history
        self.create_timer(1.0, self.cleanup_detection_history)

        self.get_logger().info("Event Detection Service Node initialized")

    def in_water_callback(self, msg: IsInWater):
        """Update current in-water state and check active monitors"""
        self.current_in_water = msg
        current_time = time.time()
        with self.monitor_lock:
            completed_monitors = []
            for monitor_id, monitor in self.active_monitors.items():
                if monitor["type"] == "in_water" and msg.is_in_water == True:
                    completed_monitors.append(monitor_id)
                    self.get_logger().info("In water detected")
                elif monitor["type"] == "out_water" and msg.is_in_water == False:
                    self.get_logger().info("Out of water detected")
                    completed_monitors.append(monitor_id)
            for monitor_id in completed_monitors:
                self._complete_monitor(monitor_id, success=True)

    def sign_callback(self, msg: RoadSignArray):
        """Update current sign detections and history"""
        self.current_signs = msg.signs
        current_time = time.time()

        # Update detection history for each detected sign
        for sign in msg.signs:
            sign_id = int(sign.id)
            self.sign_detection_history[sign_id] = current_time

        # Check active monitors for sign-related events
        with self.monitor_lock:
            completed_monitors = []
            for monitor_id, monitor in self.active_monitors.items():
                if monitor["type"] == "sign_undetected":
                    if self._check_sign_undetected(monitor):
                        completed_monitors.append(monitor_id)

            # Complete monitors that have detected their events
            for monitor_id in completed_monitors:
                self._complete_monitor(monitor_id, success=True)

    def ramp_callback(self, msg: RampFeedback):
        """Update current ramp detection state"""
        self.current_ramp_feedback = msg
        current_time = time.time()

        if msg.detected and len(msg.ramps) > 0:
            self.ramp_detection_history["ramp"] = current_time

        # Check active monitors for ramp-related events
        with self.monitor_lock:
            completed_monitors = []
            for monitor_id, monitor in self.active_monitors.items():
                if monitor["type"] == "ramp_detected":
                    if msg.detected:
                        completed_monitors.append(monitor_id)
                elif monitor["type"] == "ramp_undetected":
                    if self._check_ramp_undetected(monitor):
                        completed_monitors.append(monitor_id)
                elif monitor["type"] == "ramp_close":
                    if self._check_ramp_close(monitor, msg):
                        completed_monitors.append(monitor_id)

            # Complete monitors that have detected their events
            for monitor_id in completed_monitors:
                self._complete_monitor(monitor_id, success=True)

    # def odom_callback(self, msg: Odometry):
    #     """Update current odometry for distance-based events"""
    #     self.current_odom = msg

    def _parse_request_params(self, request) -> Dict[str, Any]:
        """Convert param_names and param_values arrays to dictionary"""
        params = {}

        # Ensure both arrays have same length
        if len(request.param_names) != len(request.param_values):
            self.get_logger().error(
                "param_names and param_values arrays have different lengths"
            )
            return params

        # Convert arrays to dictionary
        for name, value in zip(request.param_names, request.param_values):
            # Try to convert to appropriate type
            params[name] = self._convert_param_value(value)

        return params

    def _convert_param_value(self, value_str: str) -> Any:
        """Convert string parameter value to appropriate type"""
        # Try to convert to number first
        try:
            # Try int first
            if "." not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            pass

        # Try boolean
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Return as string if no conversion possible
        return value_str

    def handle_sign_undetected(self, request, response):
        """
        Service to wait until a specific sign is no longer detected
        Request parameters: sign_id (string), timeout (float)
        """
        event_params = self._parse_request_params(request)
        sign_id = str(event_params.get("sign_id", ""))
        timeout = float(event_params.get("timeout", 30.0))

        if not sign_id:
            response.success = False
            response.message = "Missing required parameter: sign_id"
            return response

        # Check if sign is already undetected
        if self._is_sign_currently_undetected(sign_id):
            response.success = True
            response.message = f"Sign {sign_id} already undetected"
            return response

        # Create monitor for this event
        monitor_id = self._create_monitor(
            monitor_type="sign_undetected",
            params={"sign_id": sign_id, "timeout": timeout},
            request=request,
        )

        # Wait for the event (this will block the service call)
        success = self._wait_for_monitor_completion(monitor_id, timeout)

        response.success = success
        response.message = (
            f"Sign {sign_id} undetected"
            if success
            else "Timeout waiting for sign undetection"
        )
        return response

    def handle_ramp_undetected(self, request, response):
        """Service to wait until ramp is no longer detected"""
        event_params = self._parse_request_params(request)
        timeout = float(event_params.get("timeout", 30.0))

        # Check if ramp is already undetected
        if not self.current_ramp_feedback or not self.current_ramp_feedback.detected:
            response.success = True
            response.message = "Ramp already undetected"
            return response

        # Create monitor for this event
        monitor_id = self._create_monitor(
            monitor_type="ramp_undetected", params={"timeout": timeout}, request=request
        )

        # Wait for the event
        success = self._wait_for_monitor_completion(monitor_id, timeout)

        response.success = success
        response.message = (
            "Ramp undetected" if success else "Timeout waiting for ramp undetection"
        )
        return response

    def handle_ramp_close(self, request, response):
        """Service to wait until ramp is detected within specified distance threshold"""
        event_params = self._parse_request_params(request)
        distance_threshold = event_params.get("distance_threshold", 1.2)  # meters
        min_lines = event_params.get("min_lines", 2)
        timeout = event_params.get("timeout", 30.0)
        ramp_type = event_params.get(
            "ramp_type", "farthest"
        )  # farthest or closest ramp

        # Check if condition is already met
        if self._is_ramp_close_now(distance_threshold, min_lines, ramp_type):
            response.success = True
            response.message = f"Ramp already close (< {distance_threshold}m)"
            return response

        # Create monitor for this event
        monitor_id = self._create_monitor(
            monitor_type="ramp_close",
            params={
                "distance_threshold": distance_threshold,
                "ramp_type": ramp_type,
                "min_lines": min_lines,
                "timeout": timeout,
            },
            request=request,
        )

        # Wait for the event
        success = self._wait_for_monitor_completion(monitor_id, timeout)

        response.success = success
        response.message = (
            f"Ramp close detected" if success else "Timeout waiting for ramp close"
        )
        return response

    def handle_ramp_detected(self, request, response):
        """Service to wait until any ramp is detected"""
        event_params = self._parse_request_params(request)
        timeout = float(event_params.get("timeout", 30.0))

        # Check if ramp is already detected
        if self.current_ramp_feedback and self.current_ramp_feedback.detected:
            response.success = True
            response.message = "Ramp already detected"
            return response

        # Create monitor for this event
        monitor_id = self._create_monitor(
            monitor_type="ramp_detected", params={"timeout": timeout}, request=request
        )

        # Wait for the event
        success = self._wait_for_monitor_completion(monitor_id, timeout)

        response.success = success
        response.message = (
            "Ramp detected" if success else "Timeout waiting for ramp detection"
        )
        return response

    def handle_in_water(self, request, response):
        """Service to wait until robot is in water (IsInWater True) before timeout"""
        event_params = self._parse_request_params(request)
        timeout = float(event_params.get("timeout", 30.0))

        # Check if already in water
        if hasattr(self, "current_in_water") and self.current_in_water.is_in_water:
            response.success = True
            response.message = "Robot already in water"
            return response

        # Create monitor for this event
        monitor_id = self._create_monitor(
            monitor_type="in_water", params={"timeout": timeout}, request=request
        )

        # Wait for the event
        success = self._wait_for_monitor_completion(monitor_id, timeout)

        response.success = success
        response.message = (
            "Robot entered water" if success else "Timeout waiting for in_water event"
        )
        return response

    def handle_out_water(self, request, response):
        """Service to wait until robot is out of water (IsInWater False) before timeout"""
        event_params = self._parse_request_params(request)
        timeout = float(event_params.get("timeout", 30.0))

        # Check if already out of water
        if hasattr(self, "current_in_water") and not self.current_in_water.is_in_water:
            response.success = True
            response.message = "Robot already out of water"
            return response

        # Create monitor for this event
        monitor_id = self._create_monitor(
            monitor_type="out_water", params={"timeout": timeout}, request=request
        )

        # Wait for the event
        success = self._wait_for_monitor_completion(monitor_id, timeout)

        response.success = success
        response.message = (
            "Robot exited water" if success else "Timeout waiting for out_water event"
        )
        return response

    def _create_monitor(
        self, monitor_type: str, params: Dict[str, Any], request
    ) -> str:
        """Create a new event monitor"""
        import uuid

        monitor_id = str(uuid.uuid4())

        with self.monitor_lock:
            self.active_monitors[monitor_id] = {
                "type": monitor_type,
                "params": params,
                "start_time": time.time(),
                "completed": False,
                "success": False,
                "completion_event": threading.Event(),
                "request": request,
            }

        self.get_logger().info(f"Created monitor {monitor_id} for {monitor_type}")
        return monitor_id

    def _wait_for_monitor_completion(self, monitor_id: str, timeout: float) -> bool:
        """Wait for a monitor to complete within timeout"""
        monitor = self.active_monitors.get(monitor_id)
        if not monitor:
            return False

        # Wait for completion or timeout
        completed = monitor["completion_event"].wait(timeout)

        with self.monitor_lock:
            if monitor_id in self.active_monitors:
                success = self.active_monitors[monitor_id]["success"]
                del self.active_monitors[monitor_id]
                return completed and success

        return False

    def _complete_monitor(self, monitor_id: str, success: bool):
        """Mark a monitor as completed"""
        if monitor_id in self.active_monitors:
            self.active_monitors[monitor_id]["completed"] = True
            self.active_monitors[monitor_id]["success"] = success
            self.active_monitors[monitor_id]["completion_event"].set()

            self.get_logger().info(
                f"Completed monitor {monitor_id} with success={success}"
            )

    def _check_sign_undetected(self, monitor: Dict[str, Any]) -> bool:
        """Check if a sign is no longer detected"""
        sign_id = monitor["params"]["sign_id"]
        return self._is_sign_currently_undetected(sign_id)

    def _check_ramp_undetected(self, monitor: Dict[str, Any]) -> bool:
        """Check if ramp is no longer detected"""
        return not self.current_ramp_feedback or not self.current_ramp_feedback.detected

    def _check_ramp_close(
        self, monitor: Dict[str, Any], ramp_msg: RampFeedback
    ) -> bool:
        """Check if ramp is close based on line detection criteria"""
        distance_threshold = monitor["params"]["distance_threshold"]
        min_lines = monitor["params"]["min_lines"]
        ramp_type = monitor["params"]["ramp_type"]

        return self._is_ramp_close_now(distance_threshold, min_lines, ramp_type)

    def _is_sign_currently_undetected(self, sign_id: str) -> bool:
        """Check if a specific sign is currently not detected"""
        # Convert sign_id to int for comparison with detected signs
        try:
            target_id = int(sign_id)
        except ValueError:
            return True  # Invalid sign_id means it's "undetected"

        # Check if sign is in current detections
        detected_ids = [sign.id for sign in self.current_signs]
        return target_id not in detected_ids

    def _is_ramp_close_now(
        self, distance_threshold: float, min_lines: int, ramp_type: str
    ) -> bool:
        if not self.current_ramp_feedback or not self.current_ramp_feedback.detected:
            return False

        # Check if we have enough lines
        if len(self.current_ramp_feedback.ramps) < min_lines:
            return False

        if ramp_type == "closest":
            closest_ramp = min(
                self.current_ramp_feedback.ramps, key=lambda r: 150 - (r.y1 + r.y2) / 2
            )
            distance = 0.04 * min(150 - (closest_ramp.y1 + closest_ramp.y2) / 2, 150)
        else:  # farthest
            farthest_ramp = max(
                self.current_ramp_feedback.ramps, key=lambda r: 150 - (r.y1 + r.y2) / 2
            )
            distance = 0.04 * min(150 - (farthest_ramp.y1 + farthest_ramp.y2) / 2, 150)

        self.get_logger().info(f"Ramp not close: {distance:.2f}m")
        if distance < distance_threshold:
            self.get_logger().info(
                f"Ramp is close: {distance:.2f}m < {distance_threshold:.2f}m"
            )
            return True
        return False

    def cleanup_detection_history(self):
        """Clean up old detection history entries"""
        current_time = time.time()
        cleanup_threshold = 5.0  # seconds

        # Clean up sign detection history
        expired_signs = [
            sign_id
            for sign_id, last_seen in self.sign_detection_history.items()
            if current_time - last_seen > cleanup_threshold
        ]
        for sign_id in expired_signs:
            del self.sign_detection_history[sign_id]

        # Clean up ramp detection history
        if "ramp" in self.ramp_detection_history:
            if current_time - self.ramp_detection_history["ramp"] > cleanup_threshold:
                del self.ramp_detection_history["ramp"]

        # Check for timed-out monitors
        with self.monitor_lock:
            timed_out_monitors = []
            for monitor_id, monitor in self.active_monitors.items():
                if current_time - monitor["start_time"] > monitor["params"].get(
                    "timeout", 30.0
                ):
                    timed_out_monitors.append(monitor_id)

            for monitor_id in timed_out_monitors:
                self._complete_monitor(monitor_id, success=False)


def main(args=None):
    rclpy.init(args=args)

    event_detector = EventDetector()

    # Use MultiThreadedExecutor for concurrent service handling
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
    executor.add_node(event_detector)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        event_detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
