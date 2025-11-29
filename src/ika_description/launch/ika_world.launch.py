import os
from os import pathsep
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
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
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


def generate_launch_description():
    ika_description = get_package_share_directory("ika_description")
    ika_description_prefix = get_package_prefix("ika_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    # is_new_model_arg = DeclareLaunchArgument(
    #     name="new_model",
    #     default_value="true",
    #     description="Whether to use the new model or not",
    # )

    # model_arg = DeclareLaunchArgument(
    #     name="model",
    #     default_value=os.path.join(ika_description, "urdf", f"ika_bot2.urdf.xacro"),
    #     description="Absolute path to robot urdf file",
    # )

    world_name_arg = DeclareLaunchArgument(
        name="world_name", default_value="ika_parkur"
    )

    world_path = PathJoinSubstitution(
        [
            ika_description,
            "worlds",
            PythonExpression(
                expression=["'", LaunchConfiguration("world_name"), "'", " + '.sdf'"]
            ),
        ]
    )

    model_path = os.path.join(ika_description, "models")
    model_path += pathsep + os.path.join(ika_description_prefix, "share")

    env_var = SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path)

    # robot_description = ParameterValue(
    #     Command(["xacro ", LaunchConfiguration("model")]), value_type=str
    # )

    # robot_state_publisher_node = Node(
    #     package="robot_state_publisher",
    #     executable="robot_state_publisher",
    #     parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    # )

    # rviz2 = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="screen",
    #     arguments=["-d", os.path.join(ika_description, "rviz", "display.rviz")],
    #     parameters=[{"use_sim_time": True}],
    # )

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")
        ),
        launch_arguments={
            "world": world_path,
            "use_sim_time": "true",
        }.items(),
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
        }.items(),
    )

    # spawn_robot = Node(
    #     package="ika_description",
    #     executable="spawn_entity_qos.py",
    #     arguments=[
    #         "-entity",
    #         "ika",
    #         "-topic",
    #         "robot_description",
    #     ],
    #     output="screen",
    # )

    # ika_controllers = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory("ika_controller"),
    #             "launch",
    #             "ika_controller.launch.py",
    #         )
    #     )
    # )

    return LaunchDescription(
        [
            env_var,
            # model_arg,
            world_name_arg,
            # is_new_model_arg,
            # rviz2,
            start_gazebo_server,
            start_gazebo_client,
            # robot_state_publisher_node,
            # spawn_robot,
            # ika_controllers,
        ]
    )
