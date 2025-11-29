#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate launch description for IKA simulation setup.

    This launch file starts:
    - Core system in simulation mode
    - Gazebo simulation with IKA robot
    - ROS2 control stack
    - Vision processing nodes
    - Action servers
    - Navigation stack
    - Mission planning
    - User interface
    """

    # Core node - Launch in SIMULATION mode, MANUAL state, without mobility
    core_node = Node(
        name="broker",
        package="rake_core",
        executable="core.py",
        parameters=[
            {"mode": 0},  # SIMULATION mode
            {"state": 0},  # MANUAL state
            {"mobility": 0},  # Without mobility
        ],
    )

    # Gazebo and Rviz launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("ika_description"),
                        "launch",
                        "ika_gazebo.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={"world_name": "ika_parkur"}.items(),
    )

    # ROS2 Control launch
    controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("ika_controller"),
                        "launch",
                        "ika_controllers.launch.py",
                    ]
                )
            ]
        )
    )

    # Vision Stack nodes
    vision_transformer_node = Node(
        package="ika_vision",
        executable="vision_transformer.py",
        name="vision_transformer",
    )

    sign_detector_node = Node(
        package="ika_vision", executable="sign_detector.py", name="sign_detector"
    )

    # Action Server nodes
    align_ramp_node = Node(
        package="ika_actions", executable="align_ramp.py", name="align_ramp"
    )

    align_with_path_node = Node(
        package="ika_actions", executable="align_with_path.py", name="align_with_path"
    )

    lock_target_node = Node(
        package="ika_actions", executable="lock_target.py", name="lock_target"
    )

    # Event Detector node
    event_detector_node = Node(
        package="ika_controller", executable="event_detector.py", name="event_detector"
    )

    # Joy Controller node
    joy_controller_node = Node(
        package="ika_controller",
        executable="joy_controller_state.py",
        name="joy_controller",
    )

    # Navigation Stack nodes
    path_planner_node = Node(
        package="ika_nav", executable="path_planner.py", name="path_planner"
    )

    path_tracker_node = Node(
        package="ika_nav", executable="path_tracker.py", name="path_tracker"
    )

    # Mission Planner node
    mission_planner_node = Node(
        package="ika_controller",
        executable="mission_planner.py",
        name="mission_planner",
    )

    # User Interface node
    ui_node = Node(package="ika_utils", executable="ui.py", name="ui")

    return LaunchDescription(
        [
            # Core system
            core_node,
            # Simulation environment
            gazebo_launch,
            controllers_launch,
            # Vision processing
            vision_transformer_node,
            sign_detector_node,
            # Action servers
            align_ramp_node,
            align_with_path_node,
            lock_target_node,
            # Control and detection
            # event_detector_node,
            # joy_controller_node,
            # Navigation
            path_planner_node,
            path_tracker_node,
            # Mission planning
            # mission_planner_node,
            # User interface
            ui_node,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
