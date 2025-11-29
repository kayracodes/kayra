from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    rear_camera_node = Node(
        package="ika_controller",
        executable="camera_node_real.py",
        name="rear_camera",
        parameters=[
            {
                "width": 320,
                "height": 240,
                "topic_name": "/rear_camera/image_raw",
                "udev_path": "/dev/v4l/by-id/usb-046d_081b_A063E250-video-index0",
            }
        ],
        output="screen",
    )

    shoot_camera_node = Node(
        package="ika_controller",
        executable="camera_node_real.py",
        name="shoot_camera",
        parameters=[
            {
                "width": 320,
                "height": 240,
                "topic_name": "/shoot_camera/image_raw",
                "udev_path": "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_CA62B6DF-video-index0",
            }
        ],
        output="screen",
    )

    front_camera_node = Node(
        package="ika_controller",
        executable="camera_node_real.py",
        name="front_camera",
        parameters=[
            {
                "width": 320,
                "height": 240,
                "topic_name": "/front_camera/image_raw",
                "udev_path": "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_944FA6DF-video-index0",
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            front_camera_node,
            shoot_camera_node,
            rear_camera_node,
        ]
    )
