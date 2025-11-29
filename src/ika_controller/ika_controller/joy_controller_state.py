#!/usr/bin/env python3
import pygame
from pygame.locals import *
import time
import rclpy
from rake_core.node import Node
import numpy as np
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rake_msgs.msg import ShootCommand, ResetCommand
from rake_core.states import SystemStateEnum, DeviceStateEnum, SystemModeEnum
from rake_core.constants import Topics, Services, Actions
from ika_actions.action import LockTarget, AlignWithPath, RampAlignment
from rclpy.action import ActionClient
import json
from types import SimpleNamespace
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Twist
from threading import Lock
from std_msgs.msg import Bool

WHEEL_SEPERATION = 0.93
WHEEL_RADIUS = 0.125
PI = 3.14159
ANG_VEL_THRESHOLD = 0.05
LIN_VEL_THRESHOLD = 0.05


class JoyControllerConfig:
    def __init__(self):
        self.control_command_topic = "/ika_controller/control_command"
        self.max_linear_velocity = 0.8
        self.max_angular_velocity_simulation = 3.0
        self.max_angular_velocity_practice = 1.6
        self.lock_target_tolerance = 0.6


class JoyController(Node):
    def __init__(self):
        super().__init__("joy_controller")

    def init(self):
        if not hasattr(self, "config"):
            self.get_logger().warn("Config not yet received")
            return

        self.check_connection()
        self.publish_loop_cb = MutuallyExclusiveCallbackGroup()
        self.tilt_on = False
        self.pan_on = False
        self.laser_on = False
        self.tilt_dir = "0"
        self.pan_dir = "0"
        self.command_str = ""
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.left_direct = 0
        self.right_direct = 0
        self.wheel_vel_right = 0.0
        self.wheel_vel_left = 0.0
        self.reset_card = False
        self.last_reset = time.time()
        self.cmd_vel_msg = Twist()
        self.reset_msg = ResetCommand()
        self.shoot_command_msg = ShootCommand()

        # Separate goal handles for each action
        self.goal_handles = {
            "locking_target": None,
            "aligning_ramp": None,
            "aligning_path": None,
        }
        self.action_futures = {
            "locking_target": None,
            "aligning_ramp": None,
            "aligning_path": None,
        }

        # Action Variables
        self.action_status = {
            "locking_target": False,
            "aligning_ramp": False,
            "aligning_path": False,
        }

        self.max_angular_velocity = (
            self.config.max_angular_velocity_simulation
            if self.system_mode == SystemModeEnum.SIMULATION
            else self.config.max_angular_velocity_practice
        )
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)
        self.shoot_command_pub = self.create_publisher(
            ShootCommand, "/ika_nav/shoot_command", 10
        )
        self.reset_command_pub = self.create_publisher(
            ResetCommand, "/ika_nav/reset_command", 10
        )
        self.after_hit_target_flag_pub = self.create_publisher(
            Bool, "/ika_nav/after_hit_target_flag", 10
        )

        # Action Clients
        self.action_clients = {
            "lock_target": (
                ActionClient(
                    self,
                    LockTarget,
                    Actions.LOCK_TARGET,
                )
                if self.system_mode == SystemModeEnum.SIMULATION
                else ActionClient(
                    self,
                    LockTarget,
                    Actions.LOCK_TARGET_REAL,
                )
            ),
            "align_with_path": ActionClient(
                self, AlignWithPath, Actions.ALIGN_WITH_PATH
            ),
            "ramp_alignment": ActionClient(self, RampAlignment, Actions.RAMP_ALIGNMENT),
        }
        # Action Goals
        self.action_goals = {
            "lock_target": LockTarget.Goal(
                angular_tolerance=self.config.lock_target_tolerance
            ),
            "align_with_path": AlignWithPath.Goal(),
            "ramp_alignment": RampAlignment.Goal(),
        }
        # Timers
        self.timer = self.create_timer(
            0.1, self.publish_loop, callback_group=self.publish_loop_cb
        )

        self.get_logger().info("JoyController initialized")

    def check_connection(self):
        pygame.init()
        pygame.joystick.init()
        self.joy_sticks = [
            pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())
        ]

        is_device_found = False
        while len(self.joy_sticks) == 0:
            print("No joy sticks are connected, please plug in a joystick")

            for event in pygame.event.get():
                if event == pygame.JOYDEVICEADDED:
                    joy_index = event
                    self.joy_sticks.append(pygame.joystick.Joystick(joy_index))
                    is_device_found = True
            if is_device_found:
                print("Joy stick device initialized")
            time.sleep(1)
        else:
            print("Joy device found")

    def system_state_transition(self, old_state, new_state):
        self.get_logger().info(
            f"System state transition: {old_state.state} -> {new_state.state}"
        )
        if (
            old_state.state != SystemStateEnum.MANUAL
            and new_state.state == SystemStateEnum.MANUAL
        ):
            self.get_logger().info("Entering MANUAL mode, starting joystick control")
            self.on_reset()
        elif (
            old_state.state == SystemStateEnum.MANUAL
            and new_state.state != SystemStateEnum.MANUAL
        ):
            self.get_logger().info("Exiting MANUAL mode, stopping joystick control")
            self.on_reset()

        if new_state.mode != old_state.mode:
            self.get_logger().info(
                f"System mode changed: {old_state.mode} -> {new_state.mode}"
            )

    def get_default_config(self):
        return json.loads(json.dumps(JoyControllerConfig().__dict__))

    def config_updated(self, json_object):
        self.config = json.loads(
            self.jdump(json_object), object_hook=lambda d: SimpleNamespace(**d)
        )

    def publish_loop(self):
        (
            new_state,
            new_mobility,
            laser_on,
            reset_card,
            pan_cmd,
            tilt_cmd,
            lin_vel,
            ang_vel,
            lock_target,
            align_with_path,
            ramp_alignment,
        ) = self.parse_joy_command()

        if new_mobility is not None or new_state is not None:
            new_system_state = self.system_state if new_state is None else new_state
            self.set_system_state(new_system_state, mobility=new_mobility)

        # handle after_hit_target_flag
        if align_with_path:
            after_hitting_target_msg = Bool()
            after_hitting_target_msg.data = True
            self.after_hit_target_flag_pub.publish(after_hitting_target_msg)
            self.get_logger().info("Align with path triggered after hitting target")

        elif not align_with_path:
            after_hitting_target_msg = Bool()
            after_hitting_target_msg.data = False
            self.after_hit_target_flag_pub.publish(after_hitting_target_msg)
        # Robot control can only be done in the manual state
        if (
            self.system_state == SystemStateEnum.MANUAL
            or self.system_state == SystemStateEnum.ACTION_CONTROL_TEST
        ):
            if reset_card is not None:
                if reset_card == True:
                    if time.time() - self.last_reset < 3.0:
                        self.reset_card = False
                    else:
                        self.last_reset = time.time()
                        self.reset_card = True

                    self.get_logger().info("Reset Issued")
                    self.reset_msg.reset_on = self.reset_card
                    self.reset_command_pub.publish(self.reset_msg)
            else:
                self.reset_card = False
                self.reset_msg.reset_on = self.reset_card
                self.reset_command_pub.publish(self.reset_msg)

            # handle align with path
            # if align_with_path and not self.action_status["aligning_path"]:
            #     if self.system_state != SystemStateEnum.ACTION_CONTROL_TEST:
            #         self.set_system_state(SystemStateEnum.ACTION_CONTROL_TEST)
            #     else:
            #         self._execute_align_with_path()
            #         self.set_system_state(SystemStateEnum.MANUAL)

            # elif align_with_path and self.action_status["aligning_path"]:
            #     self._cancel_align_with_path()
            #     self.get_logger().info("Align with path cancelled by user")
            #     self.set_system_state(SystemStateEnum.MANUAL)

            # handle lock target
            if lock_target and not self.action_status["locking_target"]:
                self.set_system_state(SystemStateEnum.ACTION_CONTROL_TEST)
                self._execute_lock_target()

            elif lock_target and self.action_status["locking_target"]:
                # Cancel the current action properly
                self._cancel_lock_target()
                self.get_logger().info("Lock target cancelled by user")
                self.set_system_state(SystemStateEnum.MANUAL)

            # handle ramp alignment
            elif ramp_alignment and not self.action_status["aligning_ramp"]:
                self.set_system_state(SystemStateEnum.ACTION_CONTROL_TEST)
                self._execute_ramp_alignment()
            elif ramp_alignment and self.action_status["aligning_ramp"]:
                self._cancel_ramp_alignment()
                self.get_logger().info("Ramp alignment cancelled by user")
                self.set_system_state(SystemStateEnum.MANUAL)

            # handle linear and angular velocity if not aligning
            if (
                not self.action_status["aligning_path"]
                and not self.action_status["aligning_ramp"]
            ):
                if lin_vel is not None:
                    self.cmd_vel_msg.linear.x = lin_vel

                if ang_vel is not None:
                    self.cmd_vel_msg.angular.z = ang_vel
                self.cmd_vel_pub.publish(self.cmd_vel_msg)

            # handle laser command if lock_target isn't being executed
            if not self.action_status["locking_target"]:
                if laser_on is not None:
                    self.laser_on = laser_on
                    self.shoot_command_msg.laser_on = laser_on
                    self.get_logger().info(f"Laser toggled to: {self.laser_on}")

                if pan_cmd is not None:
                    self.shoot_command_msg.pan_dir = pan_cmd

                if tilt_cmd is not None:
                    self.shoot_command_msg.tilt_dir = tilt_cmd
                self.shoot_command_pub.publish(self.shoot_command_msg)

    def parse_joy_command(self):
        (
            new_state,
            new_mobility,
            laser_on,
            reset_card,
            pan_cmd,
            tilt_cmd,
            lin_vel,
            ang_vel,
            lock_target,
            align_with_path,
            ramp_alignment,
        ) = (None, None, None, None, None, None, None, None, None, None, None)
        for event in pygame.event.get():
            if event.type == JOYBUTTONDOWN:
                if event.button == 2:
                    new_state = SystemStateEnum.MANUAL
                elif event.button == 0:
                    new_state = SystemStateEnum.IDLE_AUTONOMOUS
                    # print("Button 0 pressed, autonomous mode")
                elif event.button == 3:
                    new_mobility = 0 if self.mobility else 1
                elif event.button == 1:
                    laser_on = not self.laser_on

                elif event.button == 5:
                    reset_card = True
                elif event.button == 8:
                    align_with_path = True
                elif event.button == 9:
                    lock_target = True
                elif event.button == 10:
                    ramp_alignment = True

            elif event.type == JOYHATMOTION:
                # Handle D-pad (hat) input for tilt and pan control
                hat_x, hat_y = event.value

                # Handle pan (left/right movement)
                if hat_x == -1:  # Left
                    pan_cmd = 1
                elif hat_x == 1:  # Right
                    pan_cmd = 2
                else:
                    pan_cmd = 0

                # Handle tilt (up/down movement)
                if hat_y == 1:  # Up
                    tilt_cmd = 1
                elif hat_y == -1:  # Down
                    tilt_cmd = 2
                else:
                    tilt_cmd = 0

            elif event.type == JOYAXISMOTION:
                if event.axis == 1:
                    lin_val = -event.value * self.config.max_linear_velocity
                    if abs(lin_val) <= 0.05:
                        lin_vel = 0.0
                    elif lin_val > 0:
                        lin_vel = round(lin_val, 2)
                    elif lin_val < 0:
                        lin_vel = round(lin_val, 2)
                elif event.axis == 3:
                    ang_val = -event.value * self.max_angular_velocity
                    if abs(ang_val) <= 0.05:
                        ang_vel = 0.0
                    elif ang_val > 0:
                        ang_vel = round(ang_val, 2)
                    elif ang_val < 0:
                        ang_vel = round(ang_val, 2)

        return (
            new_state,
            new_mobility,
            laser_on,
            reset_card,
            pan_cmd,
            tilt_cmd,
            lin_vel,
            ang_vel,
            lock_target,
            align_with_path,
            ramp_alignment,
        )

    def _execute_lock_target(self):
        """Execute lock target action"""

        if not self.action_clients["lock_target"].wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Lock target action server not available")
            self.action_status["locking_target"] = False
            return

        self.action_status["locking_target"] = True
        self.goal_handles["locking_target"] = None  # Reset goal handle

        goal = LockTarget.Goal()
        goal.angular_tolerance = float(self.config.lock_target_tolerance)

        self.get_logger().info(
            f"Sending lock target goal: angular_tolerance={self.config.lock_target_tolerance}"
        )
        future = self.action_clients["lock_target"].send_goal_async(goal)
        self.action_futures["locking_target"] = future
        future.add_done_callback(self._lock_target_response_callback)

    def _cancel_lock_target(self):
        """Cancel the current lock target action"""
        if self.goal_handles["locking_target"] is not None:
            self.get_logger().info("Cancelling lock target action")
            cancel_future = self.goal_handles["locking_target"].cancel_goal_async()
            cancel_future.add_done_callback(
                lambda f: self._cancel_response_callback(f, "locking_target")
            )
        else:
            # No active goal handle, just reset state
            self.get_logger().info("No active lock target goal to cancel")
            self.action_status["locking_target"] = False
            self.action_futures["locking_target"] = None

    def _lock_target_response_callback(self, future):
        """Handle lock target goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Lock target goal rejected")
                self.reset_action_variables("locking_target")
                return

            self.get_logger().info("Lock target goal accepted")
            self.goal_handles["locking_target"] = (
                goal_handle  # Store goal handle for cancellation
            )
            result_future = goal_handle.get_result_async()
            self.action_futures["locking_target"] = result_future
            result_future.add_done_callback(self._lock_target_result_callback)
        except Exception as e:
            self.get_logger().error(f"Lock target goal failed: {str(e)}")
            self.reset_action_variables("locking_target")

    def _lock_target_result_callback(self, future):
        """Handle lock target result"""
        try:
            # Check if the future was cancelled or has no result
            if future.cancelled():
                self.get_logger().info("Lock target action was cancelled")
                self.reset_action_variables("locking_target")
                return

            result_response = future.result()
            if result_response is None:
                self.get_logger().warn("Lock target result is None (likely cancelled)")
                self.reset_action_variables("locking_target")
                return

            result = result_response.result
            if result.success:
                self.action_status["locking_target"] = False
                self.get_logger().info("Lock target completed successfully")
                self.set_system_state(
                    SystemStateEnum.MANUAL
                )  # Return to manual after completion
            else:
                self.get_logger().error(f"Lock target failed: {result.message}")
                self.action_status["locking_target"] = False
                self.set_system_state(SystemStateEnum.MANUAL)

        except Exception as e:
            self.get_logger().error(f"Lock target result error: {str(e)}")
            self.action_status["locking_target"] = False
            self.set_system_state(SystemStateEnum.MANUAL)
        finally:
            # Always clean up
            self.action_futures["locking_target"] = None
            self.goal_handles["locking_target"] = None

    def _cancel_response_callback(self, future, action_name):
        """Handle cancel response"""
        try:
            cancel_response = future.result()
            self.get_logger().info(f"{action_name} cancel response received")
        except Exception as e:
            self.get_logger().error(
                f"Cancel response error for {action_name}: {str(e)}"
            )
        finally:
            # Always reset state after cancel attempt
            self.reset_action_variables(action_name)

    def _execute_ramp_alignment(self):
        """Execute ramp alignment action"""

        if not self.action_clients["ramp_alignment"].wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Ramp alignment action server not available")
            self.action_status["aligning_ramp"] = False
            return

        self.action_status["aligning_ramp"] = True
        self.goal_handles["aligning_ramp"] = None  # Reset goal handle

        goal = self.action_goals["ramp_alignment"]

        self.get_logger().info("Sending ramp alignment goal")
        future = self.action_clients["ramp_alignment"].send_goal_async(goal)
        self.action_futures["aligning_ramp"] = future
        future.add_done_callback(self._ramp_alignment_response_callback)

    def _cancel_ramp_alignment(self):
        """Cancel the current ramp alignment action"""
        if self.goal_handles["aligning_ramp"] is not None:
            self.get_logger().info("Cancelling ramp alignment action")
            cancel_future = self.goal_handles["aligning_ramp"].cancel_goal_async()
            cancel_future.add_done_callback(
                lambda f: self._cancel_response_callback(f, "aligning_ramp")
            )
        else:
            # No active goal handle, just reset state
            self.get_logger().info("No active ramp alignment goal to cancel")
            self.action_status["aligning_ramp"] = False
            self.action_futures["aligning_ramp"] = None

    def _ramp_alignment_response_callback(self, future):
        """Handle ramp alignment goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Ramp alignment goal rejected")
                self.reset_action_variables("aligning_ramp")
                return

            self.get_logger().info("Ramp alignment goal accepted")
            self.goal_handles["aligning_ramp"] = (
                goal_handle  # Store goal handle for cancellation
            )
            result_future = goal_handle.get_result_async()
            self.action_futures["aligning_ramp"] = result_future
            result_future.add_done_callback(self._ramp_alignment_result_callback)
        except Exception as e:
            self.get_logger().error(f"Ramp alignment goal failed: {str(e)}")
            self.reset_action_variables("aligning_ramp")

    def _ramp_alignment_result_callback(self, future):
        """Handle ramp alignment result"""
        try:
            # Check if the future was cancelled or has no result
            if future.cancelled():
                self.get_logger().info("Ramp alignment action was cancelled")
                self.reset_action_variables("aligning_ramp")
                return

            result_response = future.result()
            if result_response is None:
                self.get_logger().warn(
                    "Ramp alignment result is None (likely cancelled)"
                )
                self.reset_action_variables("aligning_ramp")
                return

            result = result_response.result
            if result.status == 0:  # STATUS_SUCCEEDED
                self.action_status["aligning_ramp"] = False
                self.get_logger().info("Ramp alignment completed successfully")
                self.set_system_state(
                    SystemStateEnum.MANUAL
                )  # Return to manual after completion
            else:
                self.get_logger().error(f"Ramp alignment failed: {result.message}")
                self.action_status["aligning_ramp"] = False
                self.set_system_state(SystemStateEnum.MANUAL)

        except Exception as e:
            self.get_logger().error(f"Ramp alignment result error: {str(e)}")
            self.action_status["aligning_ramp"] = False
            self.set_system_state(SystemStateEnum.MANUAL)
        finally:
            # Always clean up
            self.action_futures["aligning_ramp"] = None
            self.goal_handles["aligning_ramp"] = None

    def _execute_align_with_path(self):
        """Execute align with path action"""

        if not self.action_clients["align_with_path"].wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Align with path action server not available")
            self.action_status["aligning_path"] = False
            return

        self.action_status["aligning_path"] = True
        self.goal_handles["aligning_path"] = None  # Reset goal handle

        goal = self.action_goals["align_with_path"]

        self.get_logger().info("Sending align with path goal")
        future = self.action_clients["align_with_path"].send_goal_async(goal)
        self.action_futures["aligning_path"] = future
        future.add_done_callback(self._align_with_path_response_callback)

    def _cancel_align_with_path(self):
        """Cancel the current align with path action"""
        if self.goal_handles["aligning_path"] is not None:
            self.get_logger().info("Cancelling align with path action")
            cancel_future = self.goal_handles["aligning_path"].cancel_goal_async()
            cancel_future.add_done_callback(
                lambda f: self._cancel_response_callback(f, "aligning_path")
            )
        else:
            # No active goal handle, just reset state
            self.get_logger().info("No active align with path goal to cancel")
            self.action_status["aligning_path"] = False
            self.action_futures["aligning_path"] = None

    def _align_with_path_response_callback(self, future):
        """Handle align with path goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Align with path goal rejected")
                self.reset_action_variables("aligning_path")
                return

            self.get_logger().info("Align with path goal accepted")
            self.goal_handles["aligning_path"] = (
                goal_handle  # Store goal handle for cancellation
            )
            result_future = goal_handle.get_result_async()
            self.action_futures["aligning_path"] = result_future
            result_future.add_done_callback(self._align_with_path_result_callback)
        except Exception as e:
            self.get_logger().error(f"Align with path goal failed: {str(e)}")
            self.reset_action_variables("aligning_path")

    def _align_with_path_result_callback(self, future):
        """Handle align with path result"""
        try:
            # Check if the future was cancelled or has no result
            if future.cancelled():
                self.get_logger().info("Align with path action was cancelled")
                self.reset_action_variables("aligning_path")
                return

            result_response = future.result()
            if result_response is None:
                self.get_logger().warn(
                    "Align with path result is None (likely cancelled)"
                )
                self.reset_action_variables("aligning_path")
                return

            result = result_response.result
            if result.status == 0:  # STATUS_SUCCEEDED
                self.action_status["aligning_path"] = False
                self.get_logger().info("Align with path completed successfully")
                self.set_system_state(
                    SystemStateEnum.MANUAL
                )  # Return to manual after completion
            else:
                self.get_logger().error(f"Align with path failed: {result.message}")
                self.action_status["aligning_path"] = False
                self.set_system_state(SystemStateEnum.MANUAL)

        except Exception as e:
            self.get_logger().error(f"Align with path result error: {str(e)}")
            self.action_status["aligning_path"] = False
            self.set_system_state(SystemStateEnum.MANUAL)
        finally:
            # Always clean up
            self.action_futures["aligning_path"] = None
            self.goal_handles["aligning_path"] = None

    def reset_action_variables(self, current_action: str = None):
        if current_action:
            self.action_status[current_action] = False
            self.action_futures[current_action] = None
            self.goal_handles[current_action] = None
        else:
            # Reset all action statuses
            for action in self.action_status:
                self.action_status[action] = False
                self.action_futures[action] = None
                self.goal_handles[action] = None

    def on_reset(self):
        self.cmd_vel_msg = Twist()
        self.reset_msg = ResetCommand()
        self.shoot_command_msg = ShootCommand()
        # Cancel any ongoing actions
        if self.action_status["locking_target"]:
            self._cancel_lock_target()
        if self.action_status["aligning_ramp"]:
            self._cancel_ramp_alignment()
        if self.action_status["aligning_path"]:
            self._cancel_align_with_path()


def main(args=None):
    rclpy.init(args=args)
    node = JoyController()
    node.init()
    Node.run_node(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
