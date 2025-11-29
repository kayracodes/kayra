from math import pi, sqrt, sin, cos, tan, atan2, asin, atan, degrees, radians
import math
import re
import numpy as np

import numpy as np
import tf_transformations
from tf_transformations import (
    euler_from_quaternion,
)
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class GoalPose:
    def __init__(
        self,
        goal_x,
        goal_y,
        angle_diff,
        x_px,
        y_px,
        waypoint_goal_x,
        waypoint_goal_y,
        cost,
    ):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.angle_diff = angle_diff

        self.x_px = x_px
        self.y_px = y_px
        self.cost = cost
        self.waypoint_goal_x = waypoint_goal_x
        self.waypoint_goal_y = waypoint_goal_y


# Constants from heading_calculator.py
WGS84_A = 6378137.0  # major axis
WGS84_B = 6356752.31424518  # minor axis
WGS84_F = 0.0033528107  # ellipsoid flattening
WGS84_E = 0.0818191908  # first eccentricity
WGS84_EP = 0.0820944379  # second eccentricity

# UTM Parameters
UTM_K0 = 0.9996  # scale factor
UTM_FE = 500000.0  # false easting
UTM_FN_N = 0.0  # false northing, northern hemisphere
UTM_FN_S = 10000000.0  # false northing, southern hemisphere
UTM_E2 = WGS84_E * WGS84_E  # e^2
UTM_E4 = UTM_E2 * UTM_E2  # e^4
UTM_E6 = UTM_E4 * UTM_E2  # e^6
UTM_EP2 = UTM_E2 / (1 - UTM_E2)  # e'^2

RADIANS_PER_DEGREE = pi / 180.0
DEGREES_PER_RADIAN = 180.0 / pi


# Utility Functions from heading_calculator.py
def ll2xy(lat, lon, origin_lat, origin_lon):
    """
    Geonav: Lat/Long to X/Y
    Convert latitude and longitude in dec. degress to x and y in meters
    relative to the given origin location.  Converts lat/lon and orgin to UTM and then takes the difference

    Args:
      lat (float): Latitude of location
      lon (float): Longitude of location
      orglat (float): Latitude of origin location
      orglon (float): Longitude of origin location

    Returns:
      tuple: (x,y) where...
        x is Easting in m (local grid)
        y is Northing in m  (local grid)
    """
    outmy, outmx, outmzone = LLtoUTM(origin_lat, origin_lon)
    utmy, utmx, utmzone = LLtoUTM(lat, lon)
    if not (outmzone == utmzone):
        print(
            "WARNING: geonav_conversion: origin and location are in different UTM zones!"
        )
    y = utmy - outmy
    x = utmx - outmx
    return (x, y)


def xy2ll(x, y, orglat, orglon):
    """Convert x,y to lat/lon"""
    outmy, outmx, outmzone = LLtoUTM(orglat, orglon)
    utmy = outmy + y
    utmx = outmx + x
    return UTMtoLL(utmy, utmx, outmzone)


def UTMLetterDesignator(Lat):
    """
    Determine the correct UTM letter designator for the given latitude
    """
    if (84 >= Lat) and (Lat >= 72):
        return "X"
    elif (72 > Lat) and (Lat >= 64):
        return "W"
    elif (64 > Lat) and (Lat >= 56):
        return "V"
    elif (56 > Lat) and (Lat >= 48):
        return "U"
    elif (48 > Lat) and (Lat >= 40):
        return "T"
    elif (40 > Lat) and (Lat >= 32):
        return "S"
    elif (32 > Lat) and (Lat >= 24):
        return "R"
    elif (24 > Lat) and (Lat >= 16):
        return "Q"
    elif (16 > Lat) and (Lat >= 8):
        return "P"
    elif (8 > Lat) and (Lat >= 0):
        return "N"
    elif (0 > Lat) and (Lat >= -8):
        return "M"
    elif (-8 > Lat) and (Lat >= -16):
        return "L"
    elif (-16 > Lat) and (Lat >= -24):
        return "K"
    elif (-24 > Lat) and (Lat >= -32):
        return "J"
    elif (-32 > Lat) and (Lat >= -40):
        return "H"
    elif (-40 > Lat) and (Lat >= -48):
        return "G"
    elif (-48 > Lat) and (Lat >= -56):
        return "F"
    elif (-56 > Lat) and (Lat >= -64):
        return "E"
    elif (-64 > Lat) and (Lat >= -72):
        return "D"
    elif (-72 > Lat) and (Lat >= -80):
        return "C"
    else:
        return "Z"  # Error flag, latitude is outside the UTM limits


def LLtoUTM(Lat, Long):
    """
    Convert lat/long to UTM coords
    """
    a = WGS84_A
    eccSquared = UTM_E2
    k0 = UTM_K0

    # Make sure the longitude is between -180.00 .. 179.9
    LongTemp = (Long + 180.0) - int((Long + 180.0) / 360.0) * 360.0 - 180.0

    LatRad = Lat * RADIANS_PER_DEGREE
    LongRad = LongTemp * RADIANS_PER_DEGREE
    ZoneNumber = int((LongTemp + 180.0) / 6.0) + 1

    if Lat >= 56.0 and Lat < 64.0 and LongTemp >= 3.0 and LongTemp < 12.0:
        ZoneNumber = 32
    # Special zones for Svalbard
    if Lat >= 72.0 and Lat < 84.0:
        if LongTemp >= 0.0 and LongTemp < 9.0:
            ZoneNumber = 31
        elif LongTemp >= 9.0 and LongTemp < 21.0:
            ZoneNumber = 33
        elif LongTemp >= 21.0 and LongTemp < 33.0:
            ZoneNumber = 35
        elif LongTemp >= 33.0 and LongTemp < 42.0:
            ZoneNumber = 37
    # +3 puts origin in middle of zone
    LongOrigin = (ZoneNumber - 1.0) * 6.0 - 180.0 + 3.0
    LongOriginRad = LongOrigin * RADIANS_PER_DEGREE

    # Compute the UTM Zone from the latitude and longitude
    UTMZone = "%d%s" % (ZoneNumber, UTMLetterDesignator(Lat))

    eccPrimeSquared = (eccSquared) / (1.0 - eccSquared)
    N = a / sqrt(1 - eccSquared * sin(LatRad) * sin(LatRad))
    T = tan(LatRad) * tan(LatRad)
    C = eccPrimeSquared * cos(LatRad) * cos(LatRad)
    A = cos(LatRad) * (LongRad - LongOriginRad)

    M = a * (
        (
            1
            - eccSquared / 4.0
            - 3.0 * eccSquared * eccSquared / 64.0
            - 5.0 * eccSquared * eccSquared * eccSquared / 256.0
        )
        * LatRad
        - (
            3.0 * eccSquared / 8.0
            + 3.0 * eccSquared * eccSquared / 32.0
            + 45.0 * eccSquared * eccSquared * eccSquared / 1024.0
        )
        * sin(2.0 * LatRad)
        + (
            15.0 * eccSquared * eccSquared / 256.0
            + 45.0 * eccSquared * eccSquared * eccSquared / 1024.0
        )
        * sin(4.0 * LatRad)
        - (35.0 * eccSquared * eccSquared * eccSquared / 3072.0) * sin(6.0 * LatRad)
    )

    UTMEasting = (
        k0
        * N
        * (
            A
            + (1.0 - T + C) * A * A * A / 6.0
            + (5.0 - 18.0 * T + T * T + 72 * C - 58.0 * eccPrimeSquared)
            * A
            * A
            * A
            * A
            * A
            / 120.0
        )
        + 500000.0
    )

    UTMNorthing = k0 * (
        M
        + N
        * tan(LatRad)
        * (
            A * A / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C * C) * A * A * A * A / 24.0
            + (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * eccPrimeSquared)
            * A
            * A
            * A
            * A
            * A
            * A
            / 720.0
        )
    )
    if Lat < 0:
        # 10000000 meter offset for southern hemisphere
        UTMNorthing += 10000000.0

    return (UTMNorthing, UTMEasting, UTMZone)


def UTMtoLL(UTMNorthing, UTMEasting, UTMZone):
    """
    Converts UTM coords to lat/long
    """
    k0 = UTM_K0
    a = WGS84_A
    eccSquared = UTM_E2
    e1 = (1 - sqrt(1 - eccSquared)) / (1 + sqrt(1 - eccSquared))

    x = UTMEasting - 500000.0  # remove 500,000 meter offset for longitude
    y = UTMNorthing

    ZoneLetter = re.findall("([a-zA-Z])", UTMZone)[0]
    ZoneNumber = float(UTMZone.split(ZoneLetter)[0])

    if ZoneLetter < "N":
        # remove 10,000,000 meter offset used for southern hemisphere
        y -= 10000000.0

    # +3 puts origin in middle of zone
    LongOrigin = (ZoneNumber - 1) * 6.0 - 180.0 + 3.0
    eccPrimeSquared = (eccSquared) / (1.0 - eccSquared)
    M = y / k0
    mu = M / (
        a
        * (
            1.0
            - eccSquared / 4.0
            - 3.0 * eccSquared * eccSquared / 64.0
            - 5.0 * eccSquared * eccSquared * eccSquared / 256.0
        )
    )
    phi1Rad = mu + (
        (3.0 * e1 / 2.0 - 27.0 * e1 * e1 * e1 / 32.0) * sin(2.0 * mu)
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1 * e1 * e1 * e1 / 32.0) * sin(4.0 * mu)
        + (151.0 * e1 * e1 * e1 / 96.0) * sin(6.0 * mu)
    )

    N1 = a / sqrt(1.0 - eccSquared * sin(phi1Rad) * sin(phi1Rad))
    T1 = tan(phi1Rad) * tan(phi1Rad)
    C1 = eccPrimeSquared * cos(phi1Rad) * cos(phi1Rad)
    R1 = a * (1.0 - eccSquared) / pow(1 - eccSquared * sin(phi1Rad) * sin(phi1Rad), 1.5)
    D = x / (N1 * k0)
    Lat = phi1Rad - (
        (N1 * tan(phi1Rad) / R1)
        * (
            D * D / 2.0
            - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * eccPrimeSquared)
            * D
            * D
            * D
            * D
            / 24.0
            + (
                61.0
                + 90.0 * T1
                + 298.0 * C1
                + 45.0 * T1 * T1
                - 252.0 * eccPrimeSquared
                - 3.0 * C1 * C1
            )
            * D
            * D
            * D
            * D
            * D
            * D
            / 720.0
        )
    )

    Lat = Lat * DEGREES_PER_RADIAN

    Long = (
        D
        - (1.0 + 2.0 * T1 + C1) * D * D * D / 6.0
        + (
            5.0
            - 2.0 * C1
            + 28.0 * T1
            - 3.0 * C1 * C1
            + 8.0 * eccPrimeSquared
            + 24.0 * T1 * T1
        )
        * D
        * D
        * D
        * D
        * D
        / 120.0
    ) / cos(phi1Rad)
    Long = LongOrigin + Long * DEGREES_PER_RADIAN

    return (Lat, Long)


def world_to_map_transform(self, x, y, z, yaw_offset):
    """
    Convert UTM coordinates to map coordinates
    """
    transform_matrix = np.zeros((3, 3))
    transform_matrix[0, 0] = cos(yaw_offset)
    transform_matrix[0, 1] = sin(yaw_offset)
    transform_matrix[1, 0] = -sin(yaw_offset)
    transform_matrix[1, 1] = cos(yaw_offset)
    transform_matrix[2, 2] = 1

    inv_transform_matrix = np.linalg.inv(transform_matrix)
    map_tf_vector = np.dot(inv_transform_matrix, np.array([x, y, z]))
    return map_tf_vector


def ll2distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two lat/lon points using the Haversine formula
    """
    delta_x, delta_y = ll2xy(lat1, lon1, lat2, lon2)
    return sqrt(delta_x**2 + delta_y**2)


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
