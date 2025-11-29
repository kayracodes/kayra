import os
from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from ament_index_python import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)


def generate_launch_description():
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Whether to use simulation time",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",  # name of the state broadcaster
            "--controller-manager",  # namespace
            "/controller_manager",
        ],
    )
    simple_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "simple_velocity_controller",  # name of the state broadcaster
            "--controller-manager",  # namespace
            "/controller_manager",
        ],
    )

    shoot_camera_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "shoot_camera_controller",  # name of the state broadcaster
            "--controller-manager",  # namespace
            "/controller_manager",
        ],
    )
    # controller_noisy_node = Node(
    #     package="igvc_controller",
    #     executable="noisy_controller.py",
    #     name="noisy_controller_node",
    #     output="screen",
    #     parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    # )
    controller_node = Node(
        package="ika_controller",
        executable="controller.py",
        name="simple_controller_node",
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )
    return LaunchDescription(
        [
            declare_use_sim_time,
            joint_state_broadcaster_spawner,
            simple_controller,
            shoot_camera_controller,
            controller_node,
        ]
    )
