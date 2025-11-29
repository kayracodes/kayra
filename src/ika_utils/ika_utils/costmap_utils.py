# /usr/bin/env .venv/bin/python3
import cv2
import math
import numpy as np
from ika_utils.motion_model_utils import transform_matrix_from_pose2D, normalize_angle

POSE_CHANGED_THRESHOLD = 0.05
ANGLE_CHANGED_THRESHOLD = 0.05

PRIOR_PROB = 0.5
OCC_PROB = 0.9
FREE_PROB = 0.1


def prob2logodds(p):
    if p == 0.0:
        return -float("inf")
    elif p == 1.0:
        return float("inf")
    if p < 0.0 or p > 1.0:
        p = 0.5
    return math.log(p / (1 - p))


def logodds2prob(l):
    try:
        return 1 - (1 / (1 + math.exp(l)))
    except OverflowError:
        return 1.0 if l > 0 else 0.0


class Costmap:
    def __init__(self, pixel_width, pixel_height, resolution, origin_x=0, origin_y=0):
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.resolution = resolution
        self.width = pixel_width * resolution
        self.height = pixel_height * resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.data = np.ones((pixel_height, pixel_width), dtype=np.int8) * -1
        self.probability_map = np.ones(
            (pixel_height, pixel_width), dtype=np.float32
        ) * prob2logodds(PRIOR_PROB)

    def update_data(self, new_data):
        if new_data.shape != self.data.shape:
            raise ValueError("New data shape does not match existing data shape.")

        if np.max(new_data) > 100 or np.min(new_data) < -1:
            raise ValueError(
                f"New data value should be between -1 and 100. Got {np.min(new_data)} to {np.max(new_data)}"
            )
        self.data = new_data
        self.probability_map = np.vectorize(prob2logodds)(new_data / 100.0)

    def get_data(self):
        return self.data.flatten().tolist()

    def pixelToPose(self, x_px, y_px):
        x = x_px * self.resolution + self.origin_x
        y = y_px * self.resolution + self.origin_y
        return x, y

    def poseToPixel(self, x_m, y_m):
        x_px = int((x_m - self.origin_x) / self.resolution)
        y_px = int((y_m - self.origin_y) / self.resolution)
        return x_px, y_px

    def poseOnMap(self, x_m, y_m):
        x_px, y_px = self.poseToPixel(x_m, y_m)
        if (x_px >= 0 and x_px < self.pixel_width) and (
            y_px >= 0 and y_px < self.pixel_height
        ):
            return True
        return False

    def shift_rotate(self, delta_x, delta_y, angle):
        # Create a transformation matrix
        new_grid = np.ones_like(self.data) * -1
        for h in range(self.pixel_height):
            for w in range(self.pixel_width):
                x, y = self.pixelToPose(w, h)
                new_x = math.cos(angle) * (x - delta_x) + math.sin(angle) * (
                    y - delta_y
                )
                new_y = -math.sin(angle) * (x - delta_x) + math.cos(angle) * (
                    y - delta_y
                )
                # NOTE: Practical test results explain this if clause(not in theory)
                if abs(angle) < 0.03:
                    if self.poseOnMap(new_x, y):
                        map_x, map_y = self.poseToPixel(new_x, y)
                        if self.data[h, w] != -1:
                            new_grid[map_y, map_x] = self.data[h, w]

                elif self.poseOnMap(new_x, new_y):
                    map_x, map_y = self.poseToPixel(new_x, new_y)
                    if self.data[h, w] != -1:
                        new_grid[map_y, map_x] = self.data[h, w]
        self.update_data(new_grid)

    def sensor_model(self, old_odds, new_odds) -> float:
        return old_odds + new_odds - prob2logodds(PRIOR_PROB)

    @classmethod
    def integrate_occupancy(cls, g_map, l_map):
        if l_map is None:
            return
        new_grid = g_map.data.copy()

        for hl in range(l_map.pixel_height):
            for wl in range(l_map.pixel_width):
                wg, hg = wl + 60, hl
                new_grid[hg, wg] = l_map.data[hl, wl]
                # old_odds = g_map.probability_map[hg, wg]
                # if l_map.data[hl, wl] == 100:
                #     new_odds = prob2logodds(OCC_PROB)
                # elif l_map.data[hl, wl] == 0:
                #     new_odds = prob2logodds(FREE_PROB)
                # else:
                #     new_odds = prob2logodds(PRIOR_PROB)
                # prob = g_map.sensor_model(old_odds, new_odds)
                # new_grid[hg, wg] = logodds2prob(prob) * 100

        g_map.update_data(new_grid)


class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def add(self, delta_x, delta_y, angle):
        self.x += delta_x
        self.y += delta_y
        self.theta += angle

    def get_pose(self):
        return (self.x, self.y, self.theta)

    @classmethod
    def pose_diff(cls, pose1, pose2):
        delta_x = pose2.x - pose1.x
        delta_y = pose2.y - pose1.y
        angle = normalize_angle(pose2.theta - pose1.theta)
        return delta_x, delta_y, angle

    @classmethod
    def pose_changed(cls, pose1, pose2):
        delta_x, delta_y, angle = cls.pose_diff(pose1, pose2)
        if (
            abs(delta_x) > POSE_CHANGED_THRESHOLD
            or abs(delta_y) > POSE_CHANGED_THRESHOLD
            or abs(angle) > ANGLE_CHANGED_THRESHOLD
        ):
            return True
        return False


def bresenham(x0, y0, x1, y1):
    line = []

    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx = xsign
        xy = 0
        yx = 0
        yy = ysign
    else:
        tmp = dx
        dx = dy
        dy = tmp
        xx = 0
        xy = ysign
        yx = xsign
        yy = 0

    D = 2 * dy - dx
    y = 0

    for i in range(dx + 1):
        line.append((x0 + i * xx + y * yx, y0 + i * xy + y * yy))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

    return line


def is_line_free(x0, y0, x1, y1, costmap, threshold):
    line = bresenham(x0, y0, x1, y1)
    for x, y in line:
        if costmap[y][x] > threshold:
            return False
    return True


def compute_heading(start, goal):
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    return math.atan2(dy, dx)


# Collusion Checking Functions
def interpolate_line(start, goal, step_size):
    # Generate intermediate points along the line
    distance = math.hypot(goal[0] - start[0], goal[1] - start[1])

    if distance < step_size:
        return [start, goal]
    num_steps = int(distance / step_size)
    points = []
    for i in range(num_steps + 1):
        ratio = i / num_steps
        x = start[0] + ratio * (goal[0] - start[0])
        y = start[1] + ratio * (goal[1] - start[1])
        points.append((x, y))
    return points


def generate_footprint_polygon(center, width, length, heading):
    # Rectangle centered at (0, 0)
    half_w = width / 2
    half_l = length / 2
    rect = np.array(
        [[0.0, -half_w], [0.0, half_w], [half_l, half_w], [half_l, -half_w]]
    )

    # Rotate
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    rotated_rect = np.dot(rect, R)

    # Translate
    rotated_rect += np.array(center)

    return np.int32(rotated_rect)


def check_path(
    costmap_img,
    base_pose,
    goal_pose,
    footprint_width,
    footprint_length,
    cost_threshold,
    step_size=5,
):
    """
    base_pose and goal_pose in pixel coordinates: (x, y)
    footprint_width, footprint_length in pixels
    """
    heading = compute_heading(base_pose, goal_pose)
    points = interpolate_line(base_pose, goal_pose, step_size)
    points.pop(0)
    for center in points:
        # Get rotated rectangle (robot footprint at this step)
        polygon = generate_footprint_polygon(
            center, footprint_width, footprint_length, heading
        )

        # Create a mask
        mask = np.zeros_like(costmap_img, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Masked region
        cost_values = cv2.bitwise_and(costmap_img, costmap_img, mask=mask)

        # Check if any pixel inside mask exceeds threshold
        if np.any(cost_values > cost_threshold):
            return False

    return True


def check_rotation(
    costmap_img,
    center,
    footprint_width,
    footprint_length,
    rotation_angle,
    rotation_step=10,
    cost_threshold=50,
):
    """
    center: robot's center (x, y) in pixels
    """
    rotation_steps = int(rotation_angle / rotation_step)
    angle = rotation_steps

    while angle < abs(rotation_angle):
        angle += rotation_step
        heading = angle * 2 * np.pi / 180
        polygon = generate_footprint_polygon(
            center, footprint_width, footprint_length, heading
        )

        mask = np.zeros_like(costmap_img, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        cost_values = cv2.bitwise_and(costmap_img, costmap_img, mask=mask)

        if np.any(cost_values > cost_threshold):
            return False

    return True
