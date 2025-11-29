#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rake_core.node import Node
from rake_msgs.msg import SystemState, DeviceState, ShootCommand, ResetCommand
from rake_msgs.srv import WaitForEvent, LoadMission
from rake_core.states import SystemStateEnum, DeviceStateEnum
from rake_core.constants import SIGN_ID_MAP, Actions, Topics, Services
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import json
from types import SimpleNamespace
from std_msgs.msg import String, Bool
from ika_msgs.msg import RampFeedback, RampInfo, RoadSignArray, RoadSign, IsInWater
from ika_actions.action import AlignWithPath, RampAlignment, LockTarget
from threading import Lock, Thread, Event
from enum import Enum
import time
import threading
from geometry_msgs.msg import Twist
import os


class RobotState(Enum):
    """Main robot states"""

    IDLE = "idle"
    PROCESSING_SIGN = "processing_sign"
    EXECUTING_ACTIONS = "executing_actions"
    ERROR = "error"
    WAITING_FOR_COMPLETION = "waiting_for_completion"


class ExecutionAction:
    def __init__(
        self, type, action=None, action_params=None, event=None, event_params=None
    ):
        self.type = type
        self.action = action
        self.action_params = action_params
        self.event = event
        self.event_params = event_params


class ExecutionSequence:
    def __init__(self, actions: list[ExecutionAction]):
        self.actions = actions
        self.current_action_index = 0
        self.is_completed = False
        self.failed = False


class MissionPlannerConfig:
    def __init__(self):
        # to define configurable parameters of the Mission Planner Node
        self.retry_failed_actions = False  # Whether to retry failed actions
        self.max_retries = 5  # Maximum number of retries per action
        self.critical_action_types = []  # Actions that should never be skipped
        self.recoverable_errors = [
            "timeout",
            "no_path",
            "Action server not available",
        ]  # Error types that are recoverable
        self.sign_detection_cooldown = 2.0  # Cooldown period for sign detection
        self.sign_validation_iters = (
            2  # Number of consistent detections to validate sign
        )


class MissionPlanner(Node):
    def __init__(self):
        super().__init__("mission_planner")
        self.current_state = RobotState.IDLE
        self.state_lock = Lock()
        self.last_detected_sign = None
        self.last_sign_time = 0.0
        self.stop_flags = {}
        self.current_threads = []
        self.current_action_future = None
        self.event_detected = False
        self.stop_event = Event()
        self.action_completion_pending = False
        self.current_retry_count = 0  # Track retries for current action

        self.is_after_hitting_target = False
        self.is_fifth_passed = False

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().info("Config not loaded yet")
            return
        # Constants
        self.mission_dir = os.path.join(os.getcwd(), "src", "ika_controller", "mission")
        self.default_configuration_path = os.path.join(
            self.mission_dir, "trial_mission.json"
        )
        self.execution_sequence = self._get_execution_sequence_demo(0)

        # Sign Detection Validation
        self.detection_count = 0

        # callback groups
        self.action_callback_group = ReentrantCallbackGroup()
        self.sign_detection_callback_group = MutuallyExclusiveCallbackGroup()
        self.event_detector_callback_group = MutuallyExclusiveCallbackGroup()
        self.load_mission_callback_group = MutuallyExclusiveCallbackGroup()
        # Subscribers
        self.sign_sub = self.create_subscription(
            RoadSignArray,
            "/ika_vision/road_signs",
            self.sign_detection_callback,
            10,
            callback_group=self.sign_detection_callback_group,
        )

        self.after_hitting_target_flag_sub = self.create_subscription(
            Bool,
            "/ika_nav/after_hit_target_flag",
            self.after_hitting_target_flag_callback,
            10,
        )
        # Publishers
        self.status_pub = self.create_publisher(
            String, "/mission_controller/status", 10
        )
        self.velocity_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)
        self.shoot_cmd_pub = self.create_publisher(
            ShootCommand, "/ika_nav/shoot_command", 10
        )

        self.reset_cmd_pub = self.create_publisher(
            ResetCommand, "/ika_nav/reset_command", 10
        )
        # Action Clients
        self.align_ramp_client = ActionClient(
            self,
            RampAlignment,
            Actions.RAMP_ALIGNMENT,
            callback_group=self.action_callback_group,
        )
        self.align_with_path_client = ActionClient(
            self,
            AlignWithPath,
            Actions.ALIGN_WITH_PATH,
            callback_group=self.action_callback_group,
        )
        self.lock_target_client = ActionClient(
            self,
            LockTarget,
            Actions.LOCK_TARGET_REAL,
            callback_group=self.action_callback_group,
        )

        # Service Clients - Fix: Use existing clients instead of creating new ones
        self.event_clients = {
            "ramp_undetected": self.create_client(
                WaitForEvent,
                Services.RAMP_UNDETECTED,
                callback_group=self.event_detector_callback_group,
            ),
            "ramp_detected": self.create_client(
                WaitForEvent,
                Services.RAMP_DETECTED,
                callback_group=self.event_detector_callback_group,
            ),
            "sign_undetected": self.create_client(
                WaitForEvent,
                Services.SIGN_UNDETECTED,
                callback_group=self.event_detector_callback_group,
            ),
            "ramp_close": self.create_client(
                WaitForEvent,
                Services.RAMP_CLOSE,
                callback_group=self.event_detector_callback_group,
            ),
            "in_water": self.create_client(
                WaitForEvent,
                Services.IN_WATER,
                callback_group=self.event_detector_callback_group,
            ),
            "out_water": self.create_client(
                WaitForEvent,
                Services.OUT_WATER,
                callback_group=self.event_detector_callback_group,
            ),
        }

        # Load Mission Server
        self.load_mission_srv = self.create_service(
            LoadMission,
            Services.LOAD_MISSION,
            callback=self.on_load_mission_request,
            callback_group=self.load_mission_callback_group,
        )

        # Status Timer
        self.completion_timer = self.create_timer(0.1, self._check_completion_callback)

        self.status_pub.publish(String(data="Mission Planner Node initialized"))

    def get_default_config(self):
        return json.loads(json.dumps(MissionPlannerConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def json_to_mission_configuration(self, json_object) -> dict:
        """Takes the json object, turns it into a mission configuration in the format given by _get_execution_sequence function"""
        mission_configuration = {}
        for sign, action_sequence in json_object.items():
            action_list = []
            for action in action_sequence.get("actions", []):
                action_list.append(
                    ExecutionAction(
                        type=action.get("type"),
                        action=action.get("action", None),
                        action_params=action.get("action_params", None),
                        event=action.get("event", None),
                        event_params=action.get("event_params", None),
                    )
                )
            mission_configuration[sign] = action_list
        return mission_configuration

    def system_state_transition(self, old_state, new_state):
        if (
            new_state.state == SystemStateEnum.MANUAL
            or new_state.state == SystemStateEnum.ACTION_CONTROL_TEST
            or new_state.mobility == 0
        ):
            self.get_logger().info(
                "Transitioned to MANUAL or mobility 0, cancelling current sequence"
            )
            # self.status_pub.publish(
            #     String(
            #         data="Transitioned to MANUAL or mobility 0, cancelling current sequence"
            #     )
            # )
            self.cancel_current_sequence(
                reason="System transitioned to MANUAL or mobility 0"
            )

    def on_load_mission_request(self, request, response):
        """Load a mission if the robot is in MANUAL state"""
        file_name = request.file_name
        file_path = os.path.join(self.mission_dir, f"{file_name}.json")
        if not os.path.isfile(file_path):
            self.get_logger().error(f"Mission file not found: {file_path}")
            response.success = False
            return response

        try:
            if self.system_state != SystemStateEnum.MANUAL:
                self.get_logger().warn("Cannot load mission: not in MANUAL state")
                response.success = False
                return response
            with open(file_path, "r") as f:
                mission_json = json.load(f)
            self.cancel_current_sequence(reason="Loading new mission")

            self.mission_configuration = self.json_to_mission_configuration(
                mission_json
            )

            # self.get_logger().info(f"Mission loaded from {file_path}")
            self.status_pub.publish(String(data="Mission loaded successfully"))
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Failed to load mission: {e}")
            response.success = False
        return response

    def after_hitting_target_flag_callback(self, msg: Bool):
        self.is_after_hitting_target = msg.data
        if self.is_after_hitting_target:
            self.get_logger().info("After hitting target flag set to True")
            self.cancel_current_sequence(reason="After hitting target flag set")

    def sign_detection_callback(self, msg: RoadSignArray):
        """Handle incoming sign detections"""
        with self.state_lock:
            # Only process signs if we're in IDLE state
            if self.is_after_hitting_target:
                self._transition_to_processing(-1)
                self.status_pub.publish(
                    String(data=f"Processing after hitting target with special sign -1")
                )
                return
            if (
                self.system_state == SystemStateEnum.MANUAL
                or self.system_state == SystemStateEnum.ACTION_CONTROL_TEST
                or self.mobility == 0
            ):
                return
            if self.current_state != RobotState.IDLE:
                return

            # Check cooldown to avoid rapid sign switching
            current_time = time.time()
            if current_time - self.last_sign_time < self.config.sign_detection_cooldown:
                return

            if len(msg.signs) == 0:
                return

            # Get the largest (closest/most prominent) sign
            largest_sign = max(msg.signs, key=lambda s: s.w * s.h)

            detected_sign = largest_sign.id
            if detected_sign is None:
                self.get_logger().warn(f"Unknown sign ID: {detected_sign}")
                return

            # Validate sign detection by requiring consistent detections
            if detected_sign == self.last_detected_sign:
                self.detection_count += 1
                if self.detection_count >= self.config.sign_validation_iters:
                    self.get_logger().info(
                        f"Sign validated: {SIGN_ID_MAP.get(detected_sign)}"
                    )
                    self.detection_count = 0
                    # Transition to processing state
                    self._transition_to_processing(detected_sign)
                else:
                    self.get_logger().info(
                        f"Sign detection count: {self.detection_count} for sign {SIGN_ID_MAP.get(detected_sign)}"
                    )

            else:
                self.detection_count = 1
            self.last_sign_time = current_time
            self.last_detected_sign = detected_sign
            self.status_pub.publish(
                String(
                    data=f"Sign Detected: {SIGN_ID_MAP.get(detected_sign)}, Detection Count: {self.detection_count}"
                )
            )

            # Transition to processing state
            self._transition_to_processing(detected_sign)

    def _transition_to_idle(self):
        """Transition back to idle state"""
        # NOTE: Stop & Reset the Motors
        self.current_state = RobotState.IDLE
        self.execution_sequence = None
        # self.get_logger().info("Returned to IDLE state")
        self.status_pub.publish(String(data="Returned to IDLE state"))

    def cancel_current_sequence(self, reason="Manual cancellation"):
        """Cancel the current action sequence - can be called externally"""
        with self.state_lock:
            if self.current_state == RobotState.EXECUTING_ACTIONS:
                self.get_logger().info(f"Cancelling current sequence: {reason}")
                # self.status_pub.publish(
                #     String(data=f"Cancelling current sequence: {reason}")
                # )

                # Clean up ongoing operations
                self._cleanup_threads()

                # Cancel any active action
                if self.current_action_future is not None:
                    if not self.current_action_future.done():
                        self.current_action_future.cancel()
                    self.current_action_future = None

                # Reset state
                self.event_detected = False
                self.execution_sequence = None
                self._transition_to_idle()
            else:
                self.get_logger().info(
                    f"No active sequence to cancel (state: {self.current_state.value})"
                )
                # self.status_pub.publish(
                #     String(
                #         data=f"No active sequence to cancel (state: {self.current_state.value})"
                #     )
                # )

    def _transition_to_processing(self, sign_id):
        """Transition to sign processing state"""
        self.current_state = RobotState.PROCESSING_SIGN
        self.status_pub.publish(
            String(data=f"Processing sign: {SIGN_ID_MAP.get(sign_id)}")
        )

        # Create action sequence for this sign
        action_list = self._get_execution_sequence_demo(sign_id)["actions"]
        if len(action_list) == 0:
            self.status_pub.publish(
                String(
                    data=f"No actions defined for sign {SIGN_ID_MAP.get(sign_id)}, returning to IDLE"
                )
            )
            self._transition_to_idle()
            return
        self.execution_sequence = ExecutionSequence(action_list)
        self._transition_to_execution()

    def _transition_to_execution(self):
        """Transition to action execution state"""
        self.current_state = RobotState.EXECUTING_ACTIONS
        # self.get_logger().info("Starting action sequence execution")
        self.status_pub.publish(String(data="Starting action sequence execution"))

        # Execute the first action in the sequence
        self._execute_next_action()

    def _execute_next_action(self):
        if self.execution_sequence is None or self.execution_sequence.is_completed:
            # self.get_logger().info("Action sequence completed")
            self.status_pub.publish(String(data="Action sequence completed"))
            self._transition_to_idle()
            return

        if self.execution_sequence.current_action_index >= len(
            self.execution_sequence.actions
        ):
            # Sequence complete
            self.execution_sequence.is_complete = True
            # self.get_logger().info("Action sequence completed successfully")
            self.status_pub.publish(
                String(data="Action sequence completed successfully")
            )
            self._transition_to_idle()
            return

        current_action = self.execution_sequence.actions[
            self.execution_sequence.current_action_index
        ]
        if current_action.type == "action":
            if current_action.action == "align_ramp":
                self._execute_align_ramp()
            elif current_action.action == "align_with_path":
                self._execute_align_with_path()
            elif current_action.action == "lock_target":
                self._execute_lock_target(
                    angular_tolerance=current_action.action_params.get(
                        "angular_tolerance", 1.0
                    )
                )
            else:
                self.get_logger().error(
                    f"Unknown action: {current_action.action}, skipping"
                )
                self.execution_sequence.current_action_index += 1
                self._execute_next_action()

        elif current_action.type == "set_system_state":
            # Handle system state changes
            state = current_action.action_params.get("state")
            if state:
                # self.get_logger().info(f"Setting system state to: {state}")
                self.status_pub.publish(
                    String(data=f"Setting system state to: {state}")
                )
                self.set_system_state(state)
                # Complete immediately for state changes
                self._action_completed(success=True)
            else:
                self.get_logger().error(
                    "No state specified for set_system_state action"
                )
                self.execution_sequence.current_action_index += 1
                self._execute_next_action()
        elif current_action.type == "method":
            self._execute_method(
                method_name=current_action.action,
                method_params=current_action.action_params,
            )
        elif current_action.type == "wait_event":
            self._call_event_service(
                service_name=current_action.event,
                params=current_action.event_params or {},
            )
        elif current_action.type == "action_until":
            self._execute_action_until_event(
                action=current_action.action,
                action_params=current_action.action_params or {},
                event=current_action.event,
                event_params=current_action.event_params or {},
            )
        elif current_action.type == "method_until":
            self._execute_method_until_event(
                method_name=current_action.action,
                method_params=current_action.action_params or {},
                event=current_action.event,
                event_params=current_action.event_params or {},
            )
        else:
            self.get_logger().error(
                f"Unknown action type: {current_action.type}, skipping"
            )
            self.execution_sequence.current_action_index += 1
            self._execute_next_action()

    def _get_execution_sequence(self, sign_id):
        sign_str = SIGN_ID_MAP.get(sign_id, None)
        execution_sequence = self.mission_configuration.get(sign_str, [])
        return {"actions": execution_sequence}

        if sign_str == "1":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.5, "time": 6.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 4.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.5, "time": 10.0},
                    ),
                ]
            }
        elif sign_str == "5":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.CAUTIOUS_AUTONOMOUS},
                    ),
                ]
            }

        elif sign_str == "12":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 2.0},
                    ),
                    ExecutionAction(
                        type="action",
                        action="lock_target",
                        action_params={"angular_tolerance": 0.2},
                    ),
                    ExecutionAction(
                        type="method",
                        action="hit_target",
                        action_params={"duration": 3.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 2.0},
                    ),
                ]
            }

        elif sign_str == "4":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method_until",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 8.0},
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 1.2,
                            "min_lines": 0,
                            "timeout": 8.0,
                            "ramp_type": "closest",
                        },
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method_until",
                        action="go_forward",
                        action_params={"speed": 0.4, "time": 8.0},
                        event="ramp_undetected",
                        event_params={"timeout": 10.0},
                    ),
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.FAST_AUTONOMOUS},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="sign_undetected",
                        event_params={"sign_id": str(sign_id)},
                    ),
                ]
            }

        elif sign_str == "4-END":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.FAST_AUTONOMOUS},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_detected",
                        event_params={"timeout": 10.0},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_undetected",
                        event_params={"timeout": 10.0},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="sign_undetected",
                        event_params={"sign_id": str(sign_id)},
                    ),
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.6, "time": 3.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 5.0},
                    ),
                ]
            }

        else:
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.CAUTIOUS_AUTONOMOUS},
                    ),
                ]
            }

    def get_default_mission_configuration(self):
        with open(self.default_configuration_path, "r") as f:
            mission_json = json.load(f)
        return self.json_to_mission_configuration(mission_json)

    def _get_execution_sequence_demo(self, sign_id):
        # convert to colors
        sign_str = SIGN_ID_MAP.get(sign_id, None)

        if sign_str == "5" and self.is_fifth_passed:
            sign_str = "7"

        if sign_str == "after_hit_target":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.45, "time": 6.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 4.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.7, "time": 20.0},
                    ),
                ]
            }

        elif sign_str == "1":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 50.0},
                    ),
                ]
            }

        elif sign_str == "2":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 2.0},
                    ),
                    ExecutionAction(
                        type="method_until",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 8.0},
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 0.22,
                            "min_lines": 0,
                            "timeout": 8.0,
                            "ramp_type": "farthest",
                        },
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 25.0},
                    ),
                ]
            }

        elif sign_str == "3":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 2.0},
                    ),
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="method_until",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 8.0},
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 0.22,
                            "min_lines": 0,
                            "timeout": 8.0,
                            "ramp_type": "farthest",
                        },
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 25.0},
                    ),
                ]
            }

        elif sign_str == "4":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.SLOW_AUTONOMOUS},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_detected",
                        event_params={"timeout": 30.0},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 0.3,
                            "min_lines": 0,
                            "timeout": 15.0,
                            "ramp_type": "closest",
                        },
                    ),
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.8, "time": 30.0},
                    ),
                    ExecutionAction(
                        type="method_until",
                        action="go_forward",
                        action_params={"speed": 0.4, "time": 7.0},
                        event="ramp_undetected",
                        event_params={"timeout": 7.0},
                    ),
                ]
            }

        elif sign_str == "5":
            self.is_fifth_passed = True
            return
            {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.SLOW_AUTONOMOUS},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_detected",
                        event_params={"timeout": 30.0},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 0.3,
                            "min_lines": 0,
                            "timeout": 15.0,
                            "ramp_type": "farthest",
                        },
                    ),
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.4, "time": 32.0},
                    ),
                ]
            }

        elif sign_str == "6":
            return {
                "actions": [
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.SLOW_AUTONOMOUS},
                    ),
                ]
            }

        elif sign_str == "7":
            return {
                "actions": [
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_detected",
                        event_params={"timeout": 15.0},
                    ),
                    ExecutionAction(
                        type="wait_event",
                        action=None,
                        action_params=None,
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 0.4,
                            "min_lines": 0,
                            "timeout": 12.0,
                            "ramp_type": "farthest",
                        },
                    ),
                    ExecutionAction(
                        type="set_system_state",
                        action=None,
                        action_params={"state": SystemStateEnum.ACTION_CONTROLLED},
                    ),
                    ExecutionAction(
                        type="action",
                        action="align_ramp",
                        action_params={},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.4, "time": 20.0},
                    ),
                    ExecutionAction(
                        type="method_until",
                        action="go_forward",
                        action_params={"speed": 0.2, "time": 8.0},
                        event="ramp_close",
                        event_params={
                            "distance_threshold": 0.3,
                            "min_lines": 0,
                            "timeout": 8.0,
                            "ramp_type": "farthest",
                        },
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.5, "time": 5.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 4.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.5, "time": 6.0},
                    ),
                    ExecutionAction(
                        type="method",
                        action="go_forward",
                        action_params={"speed": 0.0, "time": 10.0},
                    ),
                ]
            }

    def _check_completion_callback(self):
        """Timer callback to check for pending action completions"""
        if self.action_completion_pending:
            self.action_completion_pending = False
            self._action_completed(success=True)

    def _signal_completion_from_thread(self):
        """Signal completion from a thread - to be called by timer"""
        self.action_completion_pending = True

    def _action_completed(self, success=True, error_msg=None):
        """Handle completion of any action/method/event in the sequence"""
        if not success:
            self.get_logger().error(f"Action failed: {error_msg}")

            # Check if this is a recoverable error and we should retry
            current_action = None
            if (
                self.execution_sequence is not None
                and self.execution_sequence.current_action_index
                < len(self.execution_sequence.actions)
            ):
                current_action = self.execution_sequence.actions[
                    self.execution_sequence.current_action_index
                ]

            should_retry = (
                self.config.retry_failed_actions
                and self.current_retry_count < self.config.max_retries
                and current_action is not None
                and self._is_recoverable_error(error_msg, current_action)
            )

            if should_retry:
                self.current_retry_count += 1
                # self.get_logger().info(
                #     f"Retrying action (attempt {self.current_retry_count}/{self.config.max_retries})"
                # )
                self.status_pub.publish(
                    String(
                        data=f"Retrying action (attempt {self.current_retry_count}/{self.config.max_retries})"
                    )
                )

                # Clean up before retry
                self._cleanup_threads()
                self.event_detected = False
                if self.current_action_future is not None:
                    if not self.current_action_future.done():
                        self.current_action_future.cancel()
                    self.current_action_future = None

                # Retry the current action
                self._execute_next_action()
                return

            # No retry or max retries reached - cancel sequence
            self.get_logger().error(
                f"Action failed permanently after {self.current_retry_count} retries"
            )

            # Clean up any running threads and actions before cancelling sequence
            self._cleanup_threads()

            # Cancel any ongoing action
            if self.current_action_future is not None:
                if not self.current_action_future.done():
                    self.current_action_future.cancel()
                self.current_action_future = None

            # Reset flags
            self.event_detected = False
            self.current_retry_count = 0

            # Cancel sequence and transition to idle
            # self.get_logger().info("Cancelling sequence due to unsuccessful action")
            self.status_pub.publish(
                String(data="Cancelling sequence due to unsuccessful action")
            )
            self.execution_sequence = None
            self._transition_to_idle()
            return

        # Success - reset retry counter and continue
        self.current_retry_count = 0

        # Clean up any running threads
        self._cleanup_threads()

        # Reset flags
        self.event_detected = False
        self.current_action_future = None

        # Move to next action
        if self.execution_sequence is not None:
            self.execution_sequence.current_action_index += 1
            self._execute_next_action()

    def _is_recoverable_error(self, error_msg, action):
        """Check if an error is recoverable based on configuration"""
        if error_msg is None:
            return False

        error_msg_lower = error_msg.lower()

        # Check if error type is in recoverable list
        for recoverable_error in self.config.recoverable_errors:
            if recoverable_error.lower() in error_msg_lower:
                return True

        # Never retry critical actions
        if action.type in self.config.critical_action_types:
            return False

        return False

    def _cleanup_threads(self):
        """Clean up all running threads"""
        # Signal all threads to stop
        self.stop_event.set()

        # Get current thread to avoid joining itself
        current_thread = threading.current_thread()

        # Wait for threads to finish
        for thread in self.current_threads:
            if thread.is_alive() and thread != current_thread:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    self.get_logger().warn(
                        f"Thread {thread.name} did not stop gracefully"
                    )

        # Clear thread list
        self.current_threads.clear()

        # Reset stop event for next use
        self.stop_event.clear()

    # Execute Action Methods
    def _execute_align_ramp(self):
        """Execute ramp alignment action"""
        if not self.align_ramp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Ramp alignment action server not available")
            self._action_completed(
                success=False, error_msg="Action server not available"
            )
            return

        goal = RampAlignment.Goal()

        # self.get_logger().info(f"Sending ramp alignment goal")
        self.status_pub.publish(String(data=f"Sending ramp alignment goal"))
        future = self.align_ramp_client.send_goal_async(goal)
        future.add_done_callback(self._align_ramp_response_callback)
        self.current_action_future = future

    def _align_ramp_response_callback(self, future):
        """Handle ramp alignment goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Ramp alignment goal rejected")
                self._action_completed(success=False, error_msg="Goal rejected")
                return

            # self.get_logger().info("Ramp alignment goal accepted")
            self.status_pub.publish(String(data="Ramp alignment goal accepted"))
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._align_ramp_result_callback)
        except Exception as e:
            self.get_logger().error(f"Ramp alignment goal failed: {str(e)}")
            self._action_completed(success=False, error_msg=str(e))

    def _align_ramp_result_callback(self, future):
        """Handle ramp alignment result"""
        try:
            result = future.result().result
            if result.status == result.STATUS_SUCCEEDED:
                # self.get_logger().info("Ramp alignment completed successfully")
                self.status_pub.publish(
                    String(data="Ramp alignment completed successfully")
                )
                self._action_completed(success=True)
            else:
                # self.get_logger().error(f"Ramp alignment failed: {result.message}")
                self.status_pub.publish(
                    String(data=f"Ramp alignment failed: {result.message}")
                )
                self._action_completed(success=False, error_msg=result.message)
        except Exception as e:
            self.get_logger().error(f"Ramp alignment result error: {str(e)}")
            self._action_completed(success=False, error_msg=str(e))

    def _execute_align_with_path(self):
        """Execute align with path action"""

        if not self.align_with_path_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Align with path action server not available")
            self._action_completed(
                success=False, error_msg="Action server not available"
            )
            return

        goal = AlignWithPath.Goal()
        # Add any specific goal parameters if needed

        self.status_pub.publish(String(data="Sending align with path goal"))
        future = self.align_with_path_client.send_goal_async(goal)
        future.add_done_callback(self._align_with_path_response_callback)
        self.current_action_future = future

    def _align_with_path_response_callback(self, future):
        """Handle align with path goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                # self.get_logger().error("Align with path goal rejected")
                self.status_pub.publish(String(data="Align with path goal rejected"))
                self._action_completed(success=False, error_msg="Goal rejected")
                return

            # self.get_logger().info("Align with path goal accepted")
            self.status_pub.publish(String(data="Align with path goal accepted"))
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._align_with_path_result_callback)
        except Exception as e:
            self.get_logger().error(f"Align with path goal failed: {str(e)}")
            self._action_completed(success=False, error_msg=str(e))

    def _align_with_path_result_callback(self, future):
        """Handle align with path result"""
        try:
            result = future.result().result
            if (
                result.status == result.STATUS_SUCCEEDED
            ):  # Use STATUS_SUCCEEDED for successful alignment
                # self.get_logger().info("Align with path completed successfully")
                self.status_pub.publish(
                    String(data="Align with path completed successfully")
                )
                self._action_completed(success=True)
            else:
                status_msg = {
                    result.STATUS_FAILED: "Failed",
                    result.STATUS_TIMEOUT: "Timeout",
                    result.STATUS_CANCELLED: "Cancelled",
                    result.STATUS_NO_PATH: "No path found",
                }.get(result.status, f"Unknown status {result.status}")

                # self.get_logger().error(
                #     f"Align with path failed: {status_msg} - {result.message}"
                # )
                self.status_pub.publish(
                    String(
                        data=f"Align with path failed: {status_msg} - {result.message}"
                    )
                )
                self._action_completed(
                    success=False, error_msg=f"{status_msg}: {result.message}"
                )
        except Exception as e:
            self.get_logger().error(f"Align with path result error: {str(e)}")
            self._action_completed(success=False, error_msg=str(e))

    def _execute_lock_target(self, angular_tolerance=1.0):
        """Execute lock target action"""

        if not self.lock_target_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Lock target action server not available")
            self._action_completed(
                success=False, error_msg="Action server not available"
            )
            return

        goal = LockTarget.Goal()
        goal.angular_tolerance = float(angular_tolerance)

        self.status_pub.publish(
            String(
                data=f"Sending lock target goal: angular_tolerance={angular_tolerance}"
            )
        )
        future = self.lock_target_client.send_goal_async(goal)
        future.add_done_callback(self._lock_target_response_callback)
        self.current_action_future = future

    def _lock_target_response_callback(self, future):
        """Handle lock target goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Lock target goal rejected")
                self._action_completed(success=False, error_msg="Goal rejected")
                return

            self.status_pub.publish(String(data="Lock target goal accepted"))
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._lock_target_result_callback)
        except Exception as e:
            self.get_logger().error(f"Lock target goal failed: {str(e)}")
            self._action_completed(success=False, error_msg=str(e))

    def _lock_target_result_callback(self, future):
        """Handle lock target result"""
        try:
            result = future.result().result
            if result.success:
                self.status_pub.publish(
                    String(data="Lock target completed successfully")
                )
                self._action_completed(success=True)
            else:
                self.get_logger().error(f"Lock target failed: {result.message}")
                self._action_completed(success=False, error_msg=result.message)
        except Exception as e:
            self.get_logger().error(f"Lock target result error: {str(e)}")
            self._action_completed(success=False, error_msg=str(e))

    # WaitEvent Service Callers
    def _call_event_service(self, service_name: str, params: dict):
        """Call an event detection service"""
        from rake_msgs.srv import WaitForEvent

        if service_name not in self.event_clients:
            self.get_logger().error(f"Unknown event service: {service_name}")
            self._action_completed(
                success=False, error_msg=f"Unknown service: {service_name}"
            )
            return

        client = self.event_clients[service_name]

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Event service {service_name} not available")
            self._action_completed(success=False, error_msg="Service not available")
            return

        request = WaitForEvent.Request()
        request.event_type = service_name
        request.param_names = list(params.keys())
        request.param_values = [str(v) for v in params.values()]

        self.get_logger().info(
            f"Calling event service: {service_name} with params: {params}"
        )
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

    # Execute Method Methods
    def _execute_method(self, method_name, method_params):
        """Execute a method-based action (go_forward, hold_position, etc.)"""
        if method_name == "go_forward":
            self._go_forward(
                speed=method_params.get("speed", 0.5),
                time=method_params.get("time", 5.0),
            )
        elif method_name == "hold_position":
            self._hold_position(
                direction=method_params.get("direction", "forward"),
                duration=method_params.get("duration", 3.0),
            )
        elif method_name == "reset_motors":
            self._reset_motors(duration=method_params.get("duration", 3.0))
        elif method_name == "hit_target":
            self._hit_target(duration=method_params.get("duration", 3.0))
        else:
            self.get_logger().error(f"Unknown method: {method_name}")
            self._action_completed(
                success=False, error_msg=f"Unknown method: {method_name}"
            )

    def _go_forward(self, speed=0.5, time=5.0):
        """Move robot forward at specified speed for specified time"""
        self.status_pub.publish(
            String(data=f"Going forward: speed={speed}, time={time}")
        )

        def forward_thread():
            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = 0.0

            start_time = self.get_clock().now()
            rate = self.create_rate(10)  # 10 Hz

            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < time:
                if self.stop_event.is_set():
                    break
                self.velocity_pub.publish(twist)
                rate.sleep()

            # Stop the robot
            stop_twist = Twist()
            self.velocity_pub.publish(stop_twist)

            if not self.stop_event.is_set():
                self._signal_completion_from_thread()

        thread = Thread(target=forward_thread, name="go_forward_thread")
        self.current_threads.append(thread)
        thread.start()

    def _hold_position(self, direction="forward", duration=3.0):
        """Hold position in specified direction for duration"""
        self.status_pub.publish(
            String(data=f"Holding position: direction={direction}, duration={duration}")
        )

        def hold_thread():
            twist = Twist()
            if direction == "forward":
                twist.linear.x = 0.1  # Small forward force
            elif direction == "backward":
                twist.linear.x = -0.1  # Small backward force
            else:
                twist.linear.x = 0.0  # Just hold

            start_time = self.get_clock().now()
            rate = self.create_rate(10)  # 10 Hz

            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
                if self.stop_event.is_set():
                    break
                self.velocity_pub.publish(twist)
                rate.sleep()

            # Stop the robot
            stop_twist = Twist()
            self.velocity_pub.publish(stop_twist)

            if not self.stop_event.is_set():
                self._signal_completion_from_thread()

        thread = Thread(target=hold_thread, name="hold_position_thread")
        self.current_threads.append(thread)
        thread.start()

    def _reset_motors(self, duration=3.0):
        self.status_pub.publish(String(data=f"Resetting motors for {duration} seconds"))

        def reset_thread():
            reset_cmd = ResetCommand()
            rate = self.create_rate(10)  # 10 Hz

            num_iters = int(duration * 10)
            for i in range(num_iters):
                if i == num_iters // 2:
                    reset_cmd.reset_on = True
                    self.reset_cmd_pub.publish(reset_cmd)
                else:
                    # reset is sent, now publish 0 velocity
                    reset_cmd.reset_on = False
                    self.reset_cmd_pub.publish(reset_cmd)

                    # publish 0 velocity
                    zero_velocity = Twist()
                    zero_velocity.linear.x = 0.0
                    zero_velocity.angular.z = 0.0
                    self.velocity_pub.publish(zero_velocity)
                rate.sleep()

            if not self.stop_event.is_set():
                self._signal_completion_from_thread()

        thread = Thread(target=reset_thread, name="reset_motors_thread")
        self.current_threads.append(thread)
        thread.start()

    def _hit_target(self, duration=3.0):
        """Execute target hitting behavior"""
        self.status_pub.publish(String(data=f"Hitting target for {duration} seconds"))

        def hit_thread():
            shoot_cmd = ShootCommand()
            shoot_cmd.pan_dir = 0
            shoot_cmd.tilt_dir = 0
            shoot_cmd.laser_on = 1

            start_time = self.get_clock().now()
            rate = self.create_rate(10)  # 10 Hz

            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
                if self.stop_event.is_set():
                    break
                self.shoot_cmd_pub.publish(shoot_cmd)
                rate.sleep()

            # Stop the robot
            shoot_cmd.laser_on = 0
            self.shoot_cmd_pub.publish(shoot_cmd)

            if not self.stop_event.is_set():
                self._signal_completion_from_thread()

        thread = Thread(target=hit_thread, name="hit_target_thread")
        self.current_threads.append(thread)
        thread.start()

    def _execute_action_until_event(self, action, action_params, event, event_params):
        """Execute action until event is detected"""
        self.status_pub.publish(
            String(data=f"Executing action {action} until event {event}")
        )

        # Reset event flag
        self.event_detected = False

        # Start event monitoring in background
        event_thread = Thread(
            target=self._monitor_event_until,
            args=(event, event_params),
            name=f"event_monitor_{event}",
        )
        self.current_threads.append(event_thread)
        event_thread.start()

        # Start the action
        if action == "align_ramp":
            self._execute_align_ramp_until_event(**action_params)
        elif action == "align_with_path":
            self._execute_align_with_path_until_event(**action_params)
        elif action == "lock_target":
            self._execute_lock_target_until_event(**action_params)
        else:
            self.get_logger().error(f"Unknown action for until: {action}")
            self._action_completed(success=False, error_msg=f"Unknown action: {action}")

    def _execute_method_until_event(
        self, method_name, method_params, event, event_params
    ):
        """Execute method until event is detected"""
        self.status_pub.publish(
            String(data=f"Executing method {method_name} until event {event}")
        )

        # Reset event flag
        self.event_detected = False

        # Start event monitoring in background
        event_thread = Thread(
            target=self._monitor_event_until,
            args=(event, event_params),
            name=f"event_monitor_{event}",
        )
        self.current_threads.append(event_thread)
        event_thread.start()

        # Start the method
        if method_name == "go_forward":
            self._go_forward_until_event(**method_params)
        elif method_name == "hold_position":
            self._hold_position_until_event(**method_params)
        elif method_name == "hit_target":
            self._hit_target_until_event(**method_params)
        else:
            self.get_logger().error(f"Unknown method for until: {method_name}")
            self._action_completed(
                success=False, error_msg=f"Unknown method: {method_name}"
            )

    def _monitor_event_until(self, event, event_params):
        """Monitor event in background thread"""
        if event not in self.event_clients:
            self.get_logger().error(f"Unknown event service: {event}")
            return

        client = self.event_clients[event]

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Event service {event} not available")
            return

        request = WaitForEvent.Request()
        request.event_type = event
        request.param_names = list(event_params.keys())
        request.param_values = [str(v) for v in event_params.values()]

        try:
            # This will block until event is detected or timeout
            response = client.call(request)
            if response.success:
                self.status_pub.publish(
                    String(data=f"Event {event} detected, stopping current action")
                )
                self.event_detected = True
                self.stop_event.set()  # Signal other threads to stop
        except Exception as e:
            self.get_logger().error(f"Event monitoring failed: {str(e)}")

    def _execute_align_ramp_until_event(self):
        """Execute ramp alignment until event"""
        if not self.align_ramp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Ramp alignment action server not available")
            self._action_completed(
                success=False, error_msg="Action server not available"
            )
            return

        goal = RampAlignment.Goal()

        future = self.align_ramp_client.send_goal_async(goal)

        def response_callback(future):
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self._action_completed(success=False, error_msg="Goal rejected")
                    return

                # Monitor for completion or event
                def monitor_completion():
                    result_future = goal_handle.get_result_async()
                    while not result_future.done() and not self.event_detected:
                        time.sleep(0.1)

                    if self.event_detected:
                        # Cancel the action
                        goal_handle.cancel_goal_async()
                        self.status_pub.publish(
                            String(data="Cancelled ramp alignment due to event")
                        )

                    self._action_completed(success=True)

                monitor_thread = Thread(
                    target=monitor_completion, name="action_monitor"
                )
                self.current_threads.append(monitor_thread)
                monitor_thread.start()

            except Exception as e:
                self._action_completed(success=False, error_msg=str(e))

        future.add_done_callback(response_callback)

    def _execute_align_with_path_until_event(self):
        """Execute align with path until event"""
        if not self.align_with_path_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Align with path action server not available")
            self._action_completed(
                success=False, error_msg="Action server not available"
            )
            return

        goal = AlignWithPath.Goal()
        future = self.align_with_path_client.send_goal_async(goal)

        def response_callback(future):
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self._action_completed(success=False, error_msg="Goal rejected")
                    return

                def monitor_completion():
                    result_future = goal_handle.get_result_async()
                    while not result_future.done() and not self.event_detected:
                        time.sleep(0.1)

                    if self.event_detected:
                        goal_handle.cancel_goal_async()
                        self.status_pub.publish(
                            String(data="Cancelled align with path due to event")
                        )

                    self._action_completed(success=True)

                monitor_thread = Thread(
                    target=monitor_completion, name="action_monitor"
                )
                self.current_threads.append(monitor_thread)
                monitor_thread.start()

            except Exception as e:
                self._action_completed(success=False, error_msg=str(e))

        future.add_done_callback(response_callback)

    def _execute_lock_target_until_event(self, angular_tolerance=1.0):
        """Execute lock target until event"""
        if not self.lock_target_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Lock target action server not available")
            self._action_completed(
                success=False, error_msg="Action server not available"
            )
            return

        goal = LockTarget.Goal()
        goal.angular_tolerance = float(angular_tolerance)
        future = self.lock_target_client.send_goal_async(goal)

        def response_callback(future):
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self._action_completed(success=False, error_msg="Goal rejected")
                    return

                def monitor_completion():
                    result_future = goal_handle.get_result_async()
                    while not result_future.done() and not self.event_detected:
                        time.sleep(0.1)

                    if self.event_detected:
                        goal_handle.cancel_goal_async()
                        self.status_pub.publish(
                            String(data="Cancelled lock target due to event")
                        )

                    self._action_completed(success=True)

                monitor_thread = Thread(
                    target=monitor_completion, name="action_monitor"
                )
                self.current_threads.append(monitor_thread)
                monitor_thread.start()

            except Exception as e:
                self._action_completed(success=False, error_msg=str(e))

        future.add_done_callback(response_callback)

    def _go_forward_until_event(self, speed=0.5, time=5.0):
        """Go forward until event is detected"""

        def forward_until_thread():
            twist = Twist()
            twist.linear.x = float(speed)

            start_time = self.get_clock().now()
            rate = self.create_rate(10)

            while (
                (self.get_clock().now() - start_time).nanoseconds / 1e9 < time
                and not self.event_detected
                and not self.stop_event.is_set()
            ):
                self.velocity_pub.publish(twist)
                rate.sleep()

            # Stop the robot
            stop_twist = Twist()
            self.velocity_pub.publish(stop_twist)

            if self.event_detected:
                self.status_pub.publish(String(data="Stopped go_forward due to event"))

            self._signal_completion_from_thread()

        thread = Thread(target=forward_until_thread, name="go_forward_until_thread")
        self.current_threads.append(thread)
        thread.start()

    def _hold_position_until_event(self, direction="forward", duration=3.0):
        """Hold position until event is detected"""

        def hold_until_thread():
            twist = Twist()
            if direction == "forward":
                twist.linear.x = 0.1
            elif direction == "backward":
                twist.linear.x = -0.1
            else:
                twist.linear.x = 0.0

            start_time = self.get_clock().now()
            rate = self.create_rate(10)

            while (
                (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration
                and not self.event_detected
                and not self.stop_event.is_set()
            ):
                self.velocity_pub.publish(twist)
                rate.sleep()

            # Stop the robot
            stop_twist = Twist()
            self.velocity_pub.publish(stop_twist)

            if self.event_detected:
                self.status_pub.publish(
                    String(data="Stopped hold_position due to event")
                )

            self._signal_completion_from_thread()

        thread = Thread(target=hold_until_thread, name="hold_position_until_thread")
        self.current_threads.append(thread)
        thread.start()

    def _hit_target_until_event(self, duration=3.0):
        """Hit target until event is detected"""

        def hit_until_thread():
            twist = Twist()
            twist.linear.x = 0.0

            start_time = self.get_clock().now()
            rate = self.create_rate(10)

            while (
                (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration
                and not self.event_detected
                and not self.stop_event.is_set()
            ):
                self.velocity_pub.publish(twist)
                rate.sleep()

            # Stop the robot
            stop_twist = Twist()
            self.velocity_pub.publish(stop_twist)

            if self.event_detected:
                self.status_pub.publish(String(data="Stopped hit_target due to event"))

            self._signal_completion_from_thread()

        thread = Thread(target=hit_until_thread, name="hit_target_until_thread")
        self.current_threads.append(thread)
        thread.start()

    # UTILITY FUNCTIONS: abstraction of action server  goal_sending, service_calling & method calling


def main(args=None):
    rclpy.init(args=args)
    mission_planner = MissionPlanner()

    try:
        Node.run_node(mission_planner)
    except KeyboardInterrupt:
        pass
    finally:
        mission_planner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
