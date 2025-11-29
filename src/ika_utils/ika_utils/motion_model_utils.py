import numpy as np
from numpy import cos, sin, tan, sqrt
import tf_transformations
from tf_transformations import (
    euler_from_quaternion,
)
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


def forward_motion_model_linearized():

    def state_transition_function_g(mu=None, u=None, delta_t=None):
        x, y, theta, v_x, v_y, w, ax, ay, alpha = mu

        v = u[0]
        w = u[1]

        g = np.array(
            [
                x + v * cos(theta) * delta_t + 0.5 * ax * delta_t**2,
                y + v * sin(theta) * delta_t + 0.5 * ay * delta_t**2,
                theta + w * delta_t + 0.5 * alpha * delta_t**2,
                v_x + ax * delta_t,
                v_y + ay * delta_t,
                w + alpha * delta_t,
                ax,
                ay,
                alpha,
            ]
        )
        assert g.shape == (
            9,
        ), f"State transition function g shape is {g.shape}, expected (9,)"
        return g

    def state_transition_jacobian_G(mu=None, u=None, delta_t=None):
        x, y, theta, v_x, v_y, w, ax, ay, alpha = mu
        v = u[0]
        w = u[1]

        G = np.array(
            [
                [
                    1.0,
                    0.0,
                    -v * sin(theta) * delta_t,
                    0.0,
                    0.0,
                    0.0,
                    delta_t**2 / 2.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    v * cos(theta) * delta_t,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    delta_t**2 / 2.0,
                    0.0,
                ],
                [0.0, 0.0, 1.0, 0.0, 0.0, delta_t, 0.0, 0.0, delta_t**2 / 2.0],
                [0.0, 0.0, -v * sin(theta), 0.0, 0.0, 0.0, delta_t, 0.0, 0.0],
                [0.0, 0.0, v * cos(theta), 0.0, 0.0, 0.0, 0.0, delta_t, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert G.shape == (9, 9), f"Jacobian G shape is {G.shape}, expected (9, 9)"
        return G

    return state_transition_function_g, state_transition_jacobian_G


def angular_motion_model_linearized():
    def state_transition_function_g(mu=None, u=None, delta_t=None):
        x, y, theta, v_x, v_y, w, ax, ay, alpha = mu

        v = u[0]
        w = u[1]

        g = np.array(
            [
                x,
                y,
                theta + w * delta_t + 0.5 * alpha * delta_t**2,
                0.0,
                0.0,
                w + alpha * delta_t,
                0.0,
                0.0,
                alpha,
            ]
        )
        assert g.shape == (
            9,
        ), f"State transition function g shape is {g.shape}, expected (9,)"
        return g

    def state_transition_jacobian_G(mu=None, u=None, delta_t=None):
        x, y, theta, v_x, v_y, w, ax, ay, alpha = mu
        v = u[0]
        w = u[1]

        G = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, delta_t, 0.0, 0.0, delta_t**2 / 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert G.shape == (9, 9), f"Jacobian G shape is {G.shape}, expected (9, 9)"
        return G

    return state_transition_function_g, state_transition_jacobian_G


def observation_model_local_ekf_linearized():
    def observation_function_h(mu):
        x, y, theta, v_x, v_y, w, ax, ay, alpha = mu
        return np.array([[theta], [v_x], [v_y], [w], [ax], [ay], [alpha]])

    def observation_jacobian_H():
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

    return observation_function_h, observation_jacobian_H


def observation_model_global_ekf_linearized():
    def observation_function_h(mu):
        x, y, theta, v_x, v_y, w, ax, ay, alpha = mu
        return np.array([[x], [y], [theta], [v_x], [v_y], [w], [ax], [ay], [alpha]])

    def observation_jacobian_H():
        return np.eye(9).astype(np.float32)

    return observation_function_h, observation_jacobian_H


def get_yaw_from_quaternion(quaternion):
    """
    Convert a quaternion to yaw angle.
    """
    _, _, yaw = euler_from_quaternion(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    )
    return yaw


def normalize_angle(yaw):
    """
    Normalize the angle to be within the range [-pi, pi].
    """
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def odom_to_pose2D(odom):
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    yaw = get_yaw_from_quaternion(odom.pose.pose.orientation)
    return (x, y, yaw)


def control_input_from_odom_msg(msg: Odometry):
    """
    Extract control input from odometry message.
    """
    v = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z

    return np.array([v, w])


def get_normalized_pose2D(initial_pose, current_pose):
    # Check if the initial pose is set
    if initial_pose:
        x, y, yaw = current_pose
        init_x, init_y, init_yaw = initial_pose

        # Adjust position
        x -= init_x
        y -= init_y

        # Adjust orientation
        yaw -= init_yaw
        yaw = normalize_angle(yaw)

        return (x, y, yaw)
    else:
        return (0.0, 0.0, 0.0)  # Default pose if initial pose not set


def transform_matrix_from_pose2D(x, y, yaw):
    """
    Convert a 2D pose to a transformation matrix.
    """
    return np.array(
        [[cos(yaw), -sin(yaw), x], [sin(yaw), cos(yaw), y], [0.0, 0.0, 1.0]]
    )


def cov_array_from_stddevs(std_devs):
    """
    Create a covariance matrix from standard deviations.
    """
    return np.diag(std_devs**2)


def compute_slip_correction(w, k):
    return 1 / (1 + k * abs(w))


def transform_to_matrix(transform):
    """
    Convert a TransformStamped message to a 4x4 transformation matrix.

    Args:
        transform: A geometry_msgs.msg.TransformStamped object

    Returns:
        A 4x4 numpy array representing the transformation matrix
    """
    translation = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
    ]

    rotation = [
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w,
    ]

    # Create the transformation matrix
    trans_mat = tf_transformations.translation_matrix(translation)
    rot_mat = tf_transformations.quaternion_matrix(rotation)

    return np.matmul(trans_mat, rot_mat)


def matrix_to_transform(matrix):
    """
    Convert a 4x4 transformation matrix to a TransformStamped message.

    Args:
        matrix: A 4x4 numpy array representing the transformation matrix

    Returns:
        A geometry_msgs.msg.TransformStamped object
    """
    transform = TransformStamped()

    # Extract translation from matrix
    translation = tf_transformations.translation_from_matrix(matrix)
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = translation[2]

    # Extract rotation from matrix
    quaternion = tf_transformations.quaternion_from_matrix(matrix)
    transform.transform.rotation.x = quaternion[0]
    transform.transform.rotation.y = quaternion[1]
    transform.transform.rotation.z = quaternion[2]
    transform.transform.rotation.w = quaternion[3]

    return transform
