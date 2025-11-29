#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    # pkg_share = get_package_share_directory("ika_vision")

    # Path to parameter file
    # params_file = os.path.join(pkg_share, "config", "camera_shooter_params.yaml")

    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyUSB0",
        description="Serial port for camera shooter communication",
    )

    laser_offset_arg = DeclareLaunchArgument(
        "laser_offset",
        default_value="0.015",
        description="Laser offset parameter in cm",
    )

    tilt_sensitivity_arg = DeclareLaunchArgument(
        "tilt_sensitivity",
        default_value="1.0",
        description="Tilt sensitivity in degrees",
    )

    pan_sensitivity_arg = DeclareLaunchArgument(
        "pan_sensitivity", default_value="1.0", description="Pan sensitivity in degrees"
    )

    state_arg = DeclareLaunchArgument(
        "state",
        default_value="MANUAL",
        description="Initial camera shooter state (MANUAL/AUTONOMOUS)",
    )

    camera_udev_arg = DeclareLaunchArgument(
        "camera_udev",
        default_value="/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_CA62B6DF-video",
        description="Camera device path",
    )

    # Joy Controller Node
    joy_controller_node = Node(
        package="ika_controller",
        executable="joy_controller.py",
        name="joy_controller",
        output="screen",
        parameters=[],
        remappings=[("/ika_controller/joy_cmd", "/ika_controller/joy_cmd")],
    )

    # Camera Node
    camera_node = Node(
        package="ika_controller",
        executable="camera_node.py",
        name="shoot_camera_node",
        output="screen",
        parameters=[{"camera_udev": LaunchConfiguration("camera_udev")}],
        remappings=[("/shoot_camera/image_raw", "/shoot_camera/image_raw")],
    )

    # Camera Shooter Node (shoot_test.py)
    camera_shooter_node = Node(
        package="ika_vision",
        executable="shoot_test.py",
        name="camera_shooter",
        output="screen",
        parameters=[
            # params_file,
            {"serial_port": LaunchConfiguration("serial_port")},
            {"laser_offset": LaunchConfiguration("laser_offset")},
            {"tilt_sensitivity": LaunchConfiguration("tilt_sensitivity")},
            {"pan_sensitivity": LaunchConfiguration("pan_sensitivity")},
            {"state": LaunchConfiguration("state")},
        ],
        remappings=[
            ("/shoot_camera/image_raw", "/shoot_camera/image_raw"),
            ("/ika_controller/joy_cmd", "/ika_controller/joy_cmd"),
            ("/camera_shooter/debug_image", "/camera_shooter/debug_image"),
            ("/camera_shooter/debug_info", "/camera_shooter/debug_info"),
        ],
    )

    # Optional: RViz for visualization
    rviz_config_path = os.path.join(
        get_package_share_directory("ika_description"), "rviz", "shoot.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
        output="screen",
        condition=lambda context: os.path.exists(rviz_config_path),
    )

    return LaunchDescription(
        [
            # Launch arguments
            serial_port_arg,
            laser_offset_arg,
            tilt_sensitivity_arg,
            pan_sensitivity_arg,
            state_arg,
            camera_udev_arg,
            # Nodes
            joy_controller_node,
            camera_node,
            camera_shooter_node,
            # rviz_node,  # Uncomment if you want to launch RViz
        ]
    )
