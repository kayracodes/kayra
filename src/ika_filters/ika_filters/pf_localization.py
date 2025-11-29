#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from ika_msgs.msg import MotorFeedback
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from ika_filters.particle_filter_fixed import ParticleFilter


class PFLocalization(Node):
    def __init__(self):
        super().__init__("pf_localization")
        self.map_to_base_tf = self.init_tf_stamped()
        self.odom_msg = self.init_odom_msg()
        self.pf = ParticleFilter()
        self.motorfeedback_sub = self.create_subscription(
            MotorFeedback, "/motor_feedback", self.on_motor_feedback, 10
        )
        self.gps_sub = self.create_subscription(
            NavSatFix, "/gps/fix", self.on_gps_fix, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/odometry/filtered", 10)

    def on_motor_feedback(self, msg: MotorFeedback):
        avg_particle = self.pf.feedback(msg)

        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_msg.pose.pose.position.x = avg_particle.x
        self.odom_msg.pose.pose.position.y = avg_particle.y
        self.odom_msg.pose.pose.position.z = 0.0

        quat = quaternion_from_euler(0, 0, avg_particle.theta)
        self.odom_msg.pose.pose.orientation.x = quat[0]
        self.odom_msg.pose.pose.orientation.y = quat[1]
        self.odom_msg.pose.pose.orientation.z = quat[2]
        self.odom_msg.pose.pose.orientation.w = quat[3]

        self.odom_msg.twist.twist.linear.x = avg_particle.vx
        self.odom_msg.twist.twist.linear.y = avg_particle.vy
        self.odom_msg.twist.twist.angular.z = avg_particle.omega

        self.map_to_base_tf.header.stamp = self.get_clock().now().to_msg()
        self.map_to_base_tf.transform.translation.x = avg_particle.x
        self.map_to_base_tf.transform.translation.y = avg_particle.y
        self.map_to_base_tf.transform.translation.z = 0.0
        self.map_to_base_tf.transform.rotation = self.odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(self.map_to_base_tf)
        self.odom_pub.publish(self.odom_msg)

    def on_gps_fix(self, msg: NavSatFix):
        gps_x, gps_y = self.pf.gps_feedback(msg)
        if gps_x is None or gps_y is None:
            return
        self.get_logger().info(
            f"Origin lat: {self.pf.origin_lat}, Origin lon: {self.pf.origin_lon}, GPS Position: x={gps_x}, y={gps_y}"
        )

    # UTILITY FUNCTIONS
    def init_tf_stamped(self):
        tf_stamped = TransformStamped()
        tf_stamped.header.frame_id = "map"
        tf_stamped.child_frame_id = "base_footprint"
        return tf_stamped

    def init_odom_msg(self):
        odom_msg = Odometry()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_footprint"
        return odom_msg


def main(args=None):
    rclpy.init(args=args)
    node = PFLocalization()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
