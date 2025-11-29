from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="ika_vision",
                executable="video_recorder.py",
                name="video_recorder_front",
                parameters=[{"camera_topic": "/front_camera/image_raw"}],
            ),
            Node(
                package="ika_vision",
                executable="video_recorder.py",
                name="video_recorder_shoot",
                parameters=[{"camera_topic": "/shoot_camera/image_raw"}],
            ),
            Node(
                package="ika_vision",
                executable="video_recorder.py",
                name="video_recorder_rear",
                parameters=[{"camera_topic": "/rear_camera/image_raw"}],
            ),
        ]
    )
