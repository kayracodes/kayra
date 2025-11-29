#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image, NavSatFix
import cv2
import numpy as np
from cv_bridge import CvBridge
from ika_utils.costmap_utils import Costmap
from ika_utils.nav_utils import (
    ll2xy,
    get_yaw_from_quaternion,
    normalize_angle,
)
from math import sqrt
from heapq import heappush, heappop
import numpy as np
from ika_utils.motion_model_utils import (
    get_yaw_from_quaternion,
    transform_matrix_from_pose2D,
)


class Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def get_pose_vector(self):
        return np.array([self.x, self.y, 1.0], dtype=np.float32)


class PathPlannerconf:
    def __init__(self):
        self.simulation_waypoints = [
            # (42.67972912390755, -83.2027464538608),
            # (42.679743698654, -83.202864106066),
            # (42.67988111661, -83.202865617513),
            (42.6799276890720, -83.20298411283),
        ]
        self.goal_tolerance = 0.6  # meters
        self.goal_cost_threshold = 35  # cost threshold for goal selection
        # Define motion primitives for A* search (8-directional movement)
        self.motion_primitives = [
            (1, 0),  # right
            (-1, 0),  # left
            (0, 1),  # up
            (0, -1),  # down
            (1, 1),  # diagonal right-up
            (1, -1),  # diagonal right-down
            (-1, 1),  # diagonal left-up
            (-1, -1),  # diagonal left-down
        ]


class PathPlanner(Node):
    def __init__(self):
        super().__init__("path_planner")
        self.conf = PathPlannerconf()
        self.cv_bridge = CvBridge()

        self.costmap = None
        self.best_pos = None
        self.best_score = None
        self.path = Path()
        self.path_frame = "base_link"  # Default frame for path
        self.path.header.frame_id = self.path_frame
        # Choose which frame to publish path in: "base_link" or "map"

        self.waypoints = self.conf.simulation_waypoints
        self.current_lat = None
        self.current_long = None
        self.pose = Pose()

        # Publishers
        self.path_pub = self.create_publisher(Path, "/ika_nav/planned_path", 10)
        self.debug_img_pub = self.create_publisher(Image, "/ika_nav/debug_image", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/ika_nav/goal", 10)
        # subscribers
        self.occupancy_grid_sub = self.create_subscription(
            OccupancyGrid,
            "/ika_vision/occupancy_grid",
            self.grid_callback,
            10,
        )
        self.gps_sub = self.create_subscription(
            NavSatFix,
            "/gps/fix",
            self.gps_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.odom_callback,
            10,
        )

        self.create_timer(0.2, self.create_path)

    def create_path(self):
        # calls find_path and if valid path is found, publishes debug information
        if self.costmap is None or self.best_pos is None:
            return
        goal = self.best_pos
        start = (0, 38)
        path = self.find_path(start, goal)

        if path is None:
            self.get_logger().info("No path found")
            self.path.poses = [PoseStamped()]
            return

        self.get_logger().info(f"Path found to {goal}")
        # Convert to poses based on selected frame
        if self.path_frame == "map":
            path_poses = [self.pixel_to_world_pose(p[0], p[1]) for p in path]
        else:  # base_link
            path_poses = [self.pixel_to_base_pose(p[0], p[1]) for p in path]

        self.path.poses = self.poses_to_path(path_poses)
        self.get_logger().info(
            f"Path length: {len(self.path.poses)}\n"
            f"Start: {start}, Goal: {goal}\n"
            f"Publishing in frame: {self.path_frame}\n"
            f"first_goal: {path_poses[0]}"
        )
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = "base_link"
        self.path_pub.publish(self.path)

    def grid_callback(self, msg):
        if self.costmap is None:
            self.costmap = Costmap(
                msg.info.width,
                msg.info.height,
                msg.info.resolution,
                msg.info.origin.position.x,
                msg.info.origin.position.y,
            )
        self.costmap.update_data(
            np.array(msg.data).reshape(msg.info.height, msg.info.width)
        )
        # next_waypoint = self.get_next_goal()
        # if next_waypoint is None:
        #     self.get_logger().info("No more waypoints to process.")
        #     return
        # self.get_logger().info(f"Processing waypoint: {next_waypoint}")

        self.best_pos = None
        self.best_score = None

        depth = 0
        temp_best_pos = (0, 38)
        temp_best_score = -100000000
        frontier = set()
        explored = set()
        frontier.add((0, 38))

        while depth < 75 and len(frontier) > 0:
            cur_frontier = frontier.copy()
            for pos in cur_frontier:
                x, y = pos[0], pos[1]
                # score = x * 2.1 - y * 0.3 + depth * 0.5 - heading_err_gps * 8
                score = (
                    x * 2.0
                    + abs(75 - y) * 1.1
                    + depth * 1.2
                    - self.costmap.data[y, x] / 2.0
                )
                if score > temp_best_score:
                    temp_best_score = score
                    temp_best_pos = (x, y)

                frontier.remove(pos)
                explored.add(pos)

                # add right, left and front
                if (
                    x < 74
                    and (x + 1, y) not in explored
                    and self.costmap.data[y, x + 1] <= self.conf.goal_cost_threshold
                    and self.costmap.data[y, x + 1] != -1
                ):
                    frontier.add((x + 1, y))
                if (
                    y > 0
                    and (x, y - 1) not in explored
                    and self.costmap.data[y - 1, x] <= self.conf.goal_cost_threshold
                    and self.costmap.data[y - 1, x] != -1
                ):
                    frontier.add((x, y - 1))
                if (
                    y < 74
                    and (x, y + 1) not in explored
                    and self.costmap.data[y + 1, x] <= self.conf.goal_cost_threshold
                    and self.costmap.data[y + 1, x] != -1
                ):
                    frontier.add((x, y + 1))
            depth += 1

        # self.goal_point.point.x, self.goal_point.point.y = self.costmap.pixelToPose(
        #     temp_best_pos[0], temp_best_pos[1]
        # )
        # self.goal_point.header.stamp = self.get_clock().now().to_msg()
        # self.goal_point_pub.publish(self.goal_point)

        self.best_pos = temp_best_pos
        self.best_score = temp_best_score

        goal_point = self.get_goal_point()
        if goal_point is not None:
            # self.get_logger().info(
            #     f"Best position: {self.best_pos}, Score: {self.best_score}"
            # )
            self.goal_pub.publish(goal_point)

    def find_path(self, start, goal):
        # convert start to array indexes
        res = self.costmap.resolution
        looked_at = np.zeros((75, 75))
        open_set = [start]
        path = {}
        search_dirs = []

        for x, y in self.conf.motion_primitives:
            score = sqrt((x * res) ** 2 + (y * res) ** 2)
            search_dirs.append((x, y, score))

        def h(pt):
            return sqrt((pt[0] - goal[0]) ** 2 + (pt[1] - goal[1]) ** 2)

        def d(to_pt, cost):
            return cost + self.costmap.data[to_pt[1], to_pt[0]] * 2.5

        gScore = {}
        gScore[start] = 0

        def getG(pt):
            if pt in gScore:
                return gScore[pt]
            else:
                gScore[pt] = 1000000000
                return 1000000000  # Infinity

        fScore = {}
        fScore[start] = h(start)
        next_current = [(1, start)]

        while len(open_set) > 0:
            current = heappop(next_current)[1]
            looked_at[current[0], current[1]] = 1
            if abs(current[0] - goal[0]) < 3 and abs(current[1] - goal[1]) < 3:
                self.get_logger().info(
                    f"Found path to {goal} with path length {len(path)}"
                )
                return self.recreate_path(path, current)
            open_set.remove(current)

            for x, y, cost in search_dirs:
                neighbor = (current[0] + x, current[1] + y)
                if (
                    neighbor[0] < 0
                    or neighbor[0] >= 75
                    or neighbor[1] < 0
                    or neighbor[1] >= 75
                    or self.costmap.data[neighbor[1] - 1, neighbor[0] - 1] == -1
                    or self.costmap.data[neighbor[1] - 1, neighbor[0] - 1]
                    > self.conf.goal_cost_threshold
                ):
                    continue

                tent_g_score = getG(current) + d(neighbor, cost)
                if tent_g_score < getG(neighbor):
                    path[neighbor] = current
                    gScore[neighbor] = tent_g_score
                    fScore[neighbor] = tent_g_score + 2.8 * h(neighbor)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
                        heappush(next_current, (fScore[neighbor], neighbor))

        # No path found
        self.get_logger().warn(f"No path found from {start} to {goal}")
        return None

    def recreate_path(self, path, current):
        total_path = [current]
        while current in path:
            current = path[current]
            total_path.append(current)

        return total_path[::-1]

    def gps_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_long = msg.longitude

    def odom_callback(self, msg):
        self.pose = Pose(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            theta=get_yaw_from_quaternion(msg.pose.pose.orientation),
        )

    # UTILITY FUNCTIONS
    def poses_to_path(self, poses):
        path = Path()
        for pose in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.path_frame
            pose_stamped.header.stamp = path.header.stamp
            pose_stamped.pose.position.x = float(pose[0])
            pose_stamped.pose.position.y = float(pose[1])
            path.poses.append(pose_stamped)
        return path.poses

    def get_dist_to_goal(self, current_lat, current_long, goal_lat, goal_long):
        # calculate the distance to the goal in meters
        dx, dy = ll2xy(current_lat, current_long, goal_lat, goal_long)
        return np.sqrt(dx**2 + dy**2)

    def get_next_goal(self):
        if len(self.waypoints) == 0:
            return None
        if self.current_lat is None or self.current_long is None:
            return (42.6799276890720, -84.20298411283)

        waypoint_candidate = self.waypoints[0]
        goal_dist = self.get_dist_to_goal(
            self.current_lat,
            self.current_long,
            waypoint_candidate[0],
            waypoint_candidate[1],
        )
        if goal_dist < self.conf.goal_tolerance:
            self.waypoints.pop(0)
            return self.get_next_goal()
        else:
            return waypoint_candidate

    def get_goal_point(self):
        if self.best_pos is None or self.best_score is None:
            self.get_logger().warn("No best position or score available.")
            return None

        goal_point = PoseStamped()
        goal_point.header.frame_id = "base_link"
        goal_point.header.stamp = self.get_clock().now().to_msg()
        goal_point.pose.position.x, goal_point.pose.position.y = (
            self.costmap.pixelToPose(self.best_pos[0], self.best_pos[1])
        )
        goal_point.pose.orientation.w = 1.0

        return goal_point

    def pixel_to_base_pose(self, x_px, y_px):
        """Convert pixel coordinates to base_link frame coordinates"""
        base_x, base_y = self.costmap.pixelToPose(x_px, y_px)
        return base_x, base_y

    def pixel_to_world_pose(self, x_px, y_px):
        """Convert pixel coordinates to map frame coordinates"""
        # Step 1: Convert pixel to base_link coordinates
        base_x, base_y = self.costmap.pixelToPose(x_px, y_px)

        # Step 2: Transform from base_link to world frame
        # Create transformation matrix from base_link to world
        transform_matrix = transform_matrix_from_pose2D(
            self.pose.x, self.pose.y, self.pose.theta
        )

        # Apply transformation to base_link coordinates
        base_point = np.array([base_x, base_y, 1.0])
        world_point = np.dot(transform_matrix, base_point)
        return world_point[0], world_point[1]

    def shutdown_callback(self):
        """Callback to handle shutdown"""
        self.get_logger().info("Shutting down Path Planner")
        self.path.poses = [PoseStamped()]
        self.path_pub.publish(self.path)
        self.get_logger().info("Path Planner shutdown complete")


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
