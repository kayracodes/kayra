#!/usr/bin/env python3
"""
Mission Controller - Central state machine for autonomous parkour execution
Handles traffic sign detection and coordinates action execution sequences
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from enum import Enum
import time
from threading import Lock

# Messages and Actions
from ika_msgs.msg import RoadSignArray, RoadSign
from ika_actions.action import GoForward, AlignWithPath, RampAlignment
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# Action client result states
from rclpy.action.client import ClientGoalHandle, GoalStatus


class RobotState(Enum):
    """Main robot states"""

    IDLE = "idle"
    PROCESSING_SIGN = "processing_sign"
    EXECUTING_ACTIONS = "executing_actions"
    ERROR = "error"
    WAITING_FOR_COMPLETION = "waiting_for_completion"


class SignType(Enum):
    """Traffic sign types mapped to your detection IDs"""

    SIGN_1 = "1"
    SIGN_2 = "2"
    SIGN_3 = "3"
    SIGN_4 = "4"
    SIGN_4_END = "4-END"
    SIGN_5 = "5"
    SIGN_6 = "6"
    SIGN_7 = "7"
    SIGN_8 = "8"
    SIGN_9 = "9"
    SIGN_10 = "10"
    SIGN_11 = "11"
    SIGN_12 = "12"
    STOP = "STOP"


class ActionSequence:
    """Represents a sequence of actions to execute for a specific sign"""

    def __init__(self, sign_type: SignType, actions: list):
        self.sign_type = sign_type
        self.actions = actions  # List of action dictionaries
        self.current_action_index = 0
        self.is_complete = False
        self.failed = False


class MissionController(Node):
    """Central mission controller implementing the state machine"""

    def __init__(self):
        super().__init__("mission_controller")

        # State management
        self.current_state = RobotState.IDLE
        self.state_lock = Lock()
        self.current_sequence = None
        self.last_detected_sign = None
        self.sign_detection_cooldown = 2.0  # seconds
        self.last_sign_time = 0.0

        # Callback groups for concurrent operations
        self.sign_callback_group = MutuallyExclusiveCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        # Subscribers
        self.sign_sub = self.create_subscription(
            RoadSignArray,
            "/ika_vision/road_signs",
            self.sign_detection_callback,
            10,
            callback_group=self.sign_callback_group,
        )

        # Publishers
        self.state_pub = self.create_publisher(String, "/mission_controller/state", 10)
        self.status_pub = self.create_publisher(
            String, "/mission_controller/status", 10
        )

        # Action clients
        self.go_forward_client = ActionClient(
            self, GoForward, "go_forward", callback_group=self.action_callback_group
        )
        self.align_path_client = ActionClient(
            self,
            AlignWithPath,
            "align_with_path",
            callback_group=self.action_callback_group,
        )
        self.ramp_align_client = ActionClient(
            self,
            RampAlignment,
            "ramp_alignment",
            callback_group=self.action_callback_group,
        )

        # Define action sequences for each sign
        self.action_sequences = self._define_action_sequences()

        # Status timer
        self.create_timer(0.5, self.publish_status)

        self.get_logger().info("Mission Controller initialized in IDLE state")

    def _define_action_sequences(self) -> dict:
        """Define action sequences for each traffic sign based on your notes"""
        sequences = {}

        # Sign 1: align_with_path_plan -> set_to_slow_ramp_autonomy
        sequences[SignType.SIGN_1] = [
            {"type": "align_with_path"},
            {"type": "set_mode", "mode": "slow_ramp_autonomy"},
        ]

        # Sign 2: slow_ramp_autonomy
        sequences[SignType.SIGN_2] = [
            {"type": "set_mode", "mode": "slow_ramp_autonomy"}
        ]

        # Sign 3: align_with_path_plan -> set_to_slow_ramp_auto -> auto_till_close_ramp_seen->align_with_the_ramp -> go_forward(10 seconds) -> set_to_normal_auto
        sequences[SignType.SIGN_3] = [
            {"type": "align_with_path"},
            {"type": "set_mode", "mode": "slow_ramp_auto"},
            {"type": "wait_for_ramp"},
            {"type": "align_ramp"},
            {"type": "go_forward", "speed": 0.3, "time": 10.0},
            {"type": "set_mode", "mode": "normal_auto"},
        ]

        # Sign 4: autonomy_till_ramp_close -> align_with_path_plan -> set_to_fast_autonomy
        sequences[SignType.SIGN_4] = [
            {"type": "autonomy_till_ramp_close"},
            {"type": "align_with_path"},
            {"type": "set_mode", "mode": "fast_autonomy"},
        ]

        # Sign 4-END: autonomy_till_sign_undetected -> set_to_normal_autonomy -> ch occ_grid kernel+ increase inflation for steep turn
        sequences[SignType.SIGN_4_END] = [
            {"type": "autonomy_till_sign_undetected"},
            {"type": "set_mode", "mode": "normal_autonomy"},
            {"type": "adjust_navigation_params"},
        ]

        # Sign 5: auto_till_ramp_detected -> align_with_ramp -> go_forward_till_out_of_water -> set occ to traffic cone configuration
        sequences[SignType.SIGN_5] = [
            {"type": "auto_till_ramp_detected"},
            {"type": "align_ramp"},
            {"type": "go_forward_till_water_exit"},
            {"type": "set_navigation_config", "config": "traffic_cone"},
        ]

        # Sign 6: set occ to traffic cone configuration
        sequences[SignType.SIGN_6] = [
            {"type": "set_navigation_config", "config": "traffic_cone"}
        ]

        # Sign 7: auto_till_2_undetected -> align_with_path_plan -> set_to_slow_robot_autonomy
        sequences[SignType.SIGN_7] = [
            {"type": "auto_till_sign_undetected", "target_sign": "2"},
            {"type": "align_with_path"},
            {"type": "set_mode", "mode": "slow_robot_autonomy"},
        ]

        # Sign 9: slow_auto_till(2 lines seen with length > threshold && closer than 1.2 meters) -> align_robot_with_ramp -> go_forward_till_ramp_unseen -> go_forward(2 seconds) -> hold_pos(2 seconds)
        sequences[SignType.SIGN_9] = [
            {"type": "slow_auto_till_ramp_close"},
            {"type": "align_ramp"},
            {"type": "go_forward_till_ramp_unseen"},
            {"type": "go_forward", "speed": 0.3, "time": 2.0},
            {"type": "hold_position", "time": 2.0},
        ]

        # Sign 10 || 11: this is when we're climbing the ramp -> go_forward(2 secs)
        sequences[SignType.SIGN_10] = [
            {"type": "go_forward", "speed": 0.3, "time": 2.0}
        ]
        sequences[SignType.SIGN_11] = [
            {"type": "go_forward", "speed": 0.3, "time": 2.0}
        ]

        # Sign 12: find_target -> lock_target -> shoot_target -> go_forward_till_12_unseen -> hold_pos(2 seconds) -> set_to_normal_autonomy
        sequences[SignType.SIGN_12] = [
            {"type": "find_target"},
            {"type": "lock_target"},
            {"type": "shoot_target"},
            {"type": "go_forward_till_sign_unseen", "target_sign": "12"},
            {"type": "hold_position", "time": 2.0},
            {"type": "set_mode", "mode": "normal_autonomy"},
        ]

        return sequences

    def sign_detection_callback(self, msg: RoadSignArray):
        """Handle incoming sign detections"""
        with self.state_lock:
            # Only process signs if we're in IDLE state
            if self.current_state != RobotState.IDLE:
                return

            # Check cooldown to avoid rapid sign switching
            current_time = time.time()
            if current_time - self.last_sign_time < self.sign_detection_cooldown:
                return

            if len(msg.signs) == 0:
                return

            # Get the largest (closest/most prominent) sign
            largest_sign = max(msg.signs, key=lambda s: s.w * s.h)

            # Map detection ID to sign type
            sign_id_map = {
                0: SignType.SIGN_1,
                1: SignType.SIGN_10,
                2: SignType.SIGN_11,
                3: SignType.SIGN_12,
                4: SignType.SIGN_2,
                5: SignType.SIGN_3,
                6: SignType.SIGN_4,
                7: SignType.SIGN_4_END,
                8: SignType.SIGN_5,
                9: SignType.SIGN_6,
                10: SignType.SIGN_7,
                11: SignType.SIGN_8,
                12: SignType.SIGN_9,
                13: SignType.STOP,
            }

            detected_sign = sign_id_map.get(largest_sign.id)
            if detected_sign is None:
                self.get_logger().warn(f"Unknown sign ID: {largest_sign.id}")
                return

            self.get_logger().info(f"Sign detected: {detected_sign.value}")
            self.last_sign_time = current_time
            self.last_detected_sign = detected_sign

            # Transition to processing state
            self._transition_to_processing(detected_sign)

    def _transition_to_processing(self, sign_type: SignType):
        """Transition to sign processing state"""
        self.current_state = RobotState.PROCESSING_SIGN
        self.get_logger().info(f"Processing sign: {sign_type.value}")

        # Create action sequence for this sign
        if sign_type in self.action_sequences:
            actions = self.action_sequences[sign_type]
            self.current_sequence = ActionSequence(sign_type, actions)

            # Start executing the sequence
            self._transition_to_execution()
        else:
            self.get_logger().warn(
                f"No action sequence defined for sign: {sign_type.value}"
            )
            self._transition_to_idle()

    def _transition_to_execution(self):
        """Transition to action execution state"""
        self.current_state = RobotState.EXECUTING_ACTIONS
        self.get_logger().info("Starting action sequence execution")

        # Execute the first action in the sequence
        self._execute_next_action()

    def _execute_next_action(self):
        """Execute the next action in the current sequence"""
        if self.current_sequence is None or self.current_sequence.is_complete:
            self._transition_to_idle()
            return

        if self.current_sequence.current_action_index >= len(
            self.current_sequence.actions
        ):
            # Sequence complete
            self.current_sequence.is_complete = True
            self.get_logger().info("Action sequence completed successfully")
            self._transition_to_idle()
            return

        # Get current action
        action = self.current_sequence.actions[
            self.current_sequence.current_action_index
        ]
        self.get_logger().info(f"Executing action: {action}")

        # Execute based on action type
        if action["type"] == "go_forward":
            self._execute_go_forward(action)
        elif action["type"] == "align_with_path":
            self._execute_align_with_path(action)
        elif action["type"] == "align_ramp":
            self._execute_align_ramp(action)
        elif action["type"] == "set_mode":
            self._execute_set_mode(action)
        elif action["type"] == "set_system_state":
            self._execute_set_system_state(action)
        elif action["type"] == "hold_position":
            self._execute_hold_position(action)
        elif action["type"] == "wait_until_event":
            self._execute_wait_until_event(action)
        elif action["type"] == "go_forward_until_event":
            self._execute_go_forward_until_event(action)
        else:
            # For actions not yet implemented, just move to next
            self.get_logger().warn(
                f"Action type '{action['type']}' not yet implemented, skipping"
            )
            self._action_completed(success=True)

    # UTILITY FUNCTIONS: abstraction of action server  goal_sending, service_calling & method calling
    def _execute_go_forward(self, action):
        """Execute go forward action"""
        if not self.go_forward_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Go forward action server not available")
            self._action_completed(success=False)
            return

        goal = GoForward.Goal()
        goal.speed = action.get("speed", 0.3)
        goal.time = action.get("time", 1.0)

        future = self.go_forward_client.send_goal_async(goal)
        future.add_done_callback(self._go_forward_response_callback)

    def _execute_align_with_path(self, action):
        """Execute align with path action"""
        if not self.align_path_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Align path action server not available")
            self._action_completed(success=False)
            return

        goal = AlignWithPath.Goal()
        future = self.align_path_client.send_goal_async(goal)
        future.add_done_callback(self._align_path_response_callback)

    def _execute_align_ramp(self, action):
        """Execute ramp alignment action"""
        if not self.ramp_align_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Ramp alignment action server not available")
            self._action_completed(success=False)
            return

        goal = RampAlignment.Goal()
        future = self.ramp_align_client.send_goal_async(goal)
        future.add_done_callback(self._ramp_align_response_callback)

    def _execute_set_mode(self, action):
        """Execute mode change - publish to appropriate topic"""
        mode = action.get("mode", "normal_auto")
        # This would publish to your mode topic
        self.get_logger().info(f"Setting mode to: {mode}")
        # Add actual mode publishing here based on your system
        self._action_completed(success=True)

    def _execute_set_system_state(self, action):
        """Execute system state change using system state manager"""
        from std_srvs.srv import SetString

        state = action.get("state", "normal_autonomy")

        # Call system state manager service
        client = self.create_client(SetString, "set_system_state")

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("System state manager service not available")
            self._action_completed(success=False)
            return

        request = SetString.Request()
        request.data = state

        future = client.call_async(request)
        future.add_done_callback(lambda f: self._system_state_callback(f, state))

    def _system_state_callback(self, future, state: str):
        """Handle system state change response"""
        try:
            response = future.result()
            self.get_logger().info(
                f"System state change to {state}: {response.success}"
            )
            self._action_completed(success=response.success)
        except Exception as e:
            self.get_logger().error(f"System state change to {state} failed: {str(e)}")
            self._action_completed(success=False)

    def _execute_hold_position(self, action):
        """Execute hold position for specified time"""
        hold_time = action.get("time", 2.0)
        self.get_logger().info(f"Holding position for {hold_time} seconds")

        # Create a timer to complete this action after the hold time
        self.hold_timer = self.create_timer(
            hold_time, lambda: self._action_completed(success=True)
        )
        # Make it single-shot
        self.hold_timer.cancel()
        self.hold_timer = self.create_timer(hold_time, self._hold_timer_callback)

    def _hold_timer_callback(self):
        """Callback for hold position timer"""
        self.hold_timer.cancel()
        self._action_completed(success=True)

    def _execute_wait_until_event(self, action):
        """Execute wait until event action using event detection services"""
        event_type = action.get("event")
        params = action.get("params", {})

        if event_type == "ramp_close":
            self._call_event_service("wait_for_ramp_close", params)
        elif event_type == "ramp_undetected":
            self._call_event_service("wait_for_ramp_undetected", params)
        elif event_type == "sign_undetected":
            self._call_event_service("wait_for_sign_undetected", params)
        elif event_type == "ramp_detected":
            self._call_event_service("wait_for_ramp_detected", params)
        else:
            self.get_logger().error(f"Unknown event type: {event_type}")
            self._action_completed(success=False)

    def _execute_go_forward_until_event(self, action):
        """Execute go forward until an event occurs"""
        # Start go forward action
        go_forward_params = action.get("action_params", {})
        event_type = action.get("event")
        event_params = action.get("event_params", {})

        # Store current action for potential cancellation
        self.current_conditional_action = {
            "type": "go_forward_until_event",
            "event": event_type,
            "event_params": event_params,
        }

        # Start the go forward action with long duration
        long_duration_action = {
            "type": "go_forward",
            "speed": go_forward_params.get("speed", 0.3),
            "time": 999.0,  # Very long duration, will be cancelled by event
        }

        # Execute go forward
        self._execute_go_forward(long_duration_action)

        # Simultaneously start monitoring for the event
        self._start_event_monitoring(event_type, event_params)

    def _call_event_service(self, service_name: str, params: dict):
        """Call an event detection service"""
        from rake_msgs.srv import WaitForEvent

        client = self.create_client(WaitForEvent, service_name)

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Event service {service_name} not available")
            self._action_completed(success=False)
            return

        request = WaitForEvent.Request()
        request.event_type = service_name
        request.param_names = list(params.keys())
        request.param_values = [str(v) for v in params.values()]

        future = client.call_async(request)
        future.add_done_callback(
            lambda f: self._event_service_callback(f, service_name)
        )

    def _event_service_callback(self, future, service_name: str):
        """Handle event service response"""
        try:
            response = future.result()
            self.get_logger().info(
                f"Event service {service_name} completed: {response.success}"
            )
            self._action_completed(success=response.success)
        except Exception as e:
            self.get_logger().error(f"Event service {service_name} failed: {str(e)}")
            self._action_completed(success=False)

    def _start_event_monitoring(self, event_type: str, params: dict):
        """Start monitoring for an event in parallel with action execution"""
        # This would run in parallel and cancel the main action when event occurs
        # Implementation depends on your specific needs
        pass

    # Action completion callbacks
    def _go_forward_response_callback(self, future):
        """Handle go forward action response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Go forward goal rejected")
            self._action_completed(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._go_forward_result_callback)

    def _go_forward_result_callback(self, future):
        """Handle go forward action result"""
        result = future.result().result
        success = result.status == GoForward.Result.STATUS_SUCCEEDED
        self._action_completed(success=success)

    def _align_path_response_callback(self, future):
        """Handle align path action response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Align path goal rejected")
            self._action_completed(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._align_path_result_callback)

    def _align_path_result_callback(self, future):
        """Handle align path action result"""
        result = future.result().result
        success = result.status == AlignWithPath.Result.STATUS_SUCCEEDED
        self._action_completed(success=success)

    def _ramp_align_response_callback(self, future):
        """Handle ramp alignment action response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Ramp alignment goal rejected")
            self._action_completed(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._ramp_align_result_callback)

    def _ramp_align_result_callback(self, future):
        """Handle ramp alignment action result"""
        result = future.result().result
        success = result.status == RampAlignment.Result.STATUS_SUCCEEDED
        self._action_completed(success=success)

    def _action_completed(self, success: bool):
        """Handle action completion"""
        if not success:
            self.get_logger().error("Action failed, returning to idle")
            self.current_sequence.failed = True
            self._transition_to_idle()
            return

        # Move to next action
        self.current_sequence.current_action_index += 1
        self._execute_next_action()

    def _transition_to_idle(self):
        """Transition back to idle state"""
        self.current_state = RobotState.IDLE
        self.current_sequence = None
        self.get_logger().info("Returned to IDLE state")

    def publish_status(self):
        """Publish current state and status information"""
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

        status_msg = String()
        if self.current_sequence is not None:
            action_info = f"Action {self.current_sequence.current_action_index + 1}/{len(self.current_sequence.actions)}"
            status_msg.data = f"State: {self.current_state.value}, Sign: {self.current_sequence.sign_type.value}, {action_info}"
        else:
            status_msg.data = f"State: {self.current_state.value}"

        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)

    mission_controller = MissionController()

    # Use MultiThreadedExecutor to handle concurrent callbacks
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
    executor.add_node(mission_controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        mission_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
