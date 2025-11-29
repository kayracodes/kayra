from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os
from os import pathsep
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    ika_description = get_package_share_directory("ika_description")
    ika_description_prefix = get_package_prefix("ika_description")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(ika_description, "urdf", "kayra.urdf.xacro"),
        description="Absolute path to robot urdf file",
    )

    robot_description = ParameterValue(
        Command(["xacro ", LaunchConfiguration("model")]), value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    # rviz2 = Node(
    # package="rviz2",
    # executable="rviz2",
    # name="rviz2",
    # output="screen",
    # arguments=["-d", os.path.join(ika_description, "rviz", "display.rviz")],
    # parameters=[{"use_sim_time": True}],
    # )

    vision_node = Node(
        package="ika_vision",
        executable="vision_transformer_real.py",
        name="vision_transformer_node",
        output="screen",
    )

    sign_detector_node = Node(
        package="ika_vision",
        executable="sign_detector.py",
        name="sign_detector_node",
    )

    shoot_test_node = Node(
        package="ika_vision",
        executable="shoot_test_real.py",
        name="shoot_test_node",
    )

    path_planner_node = Node(
        package="ika_nav",
        executable="path_planner_real.py",
        name="path_planner_node",
    )

    path_tracker_node = Node(
        package="ika_nav",
        executable="path_tracker.py",
        name="path_tracker_node",
    )
    # serial_node = Node(
    # package="ika_controller",
    # executable="serial_controller.py",
    # name="ika_serial_node",
    # output="screen",
    # )
    ui_node = Node(
        package="ika_utils",
        executable="ui.py",
        name="ui_node",
    )

    return LaunchDescription(
        [
            model_arg,
            robot_state_publisher_node,
            # serial_node,
            vision_node,
            # rviz2,
            sign_detector_node,
            shoot_test_node,
            path_planner_node,
            path_tracker_node,
            ui_node,
        ]
    )
