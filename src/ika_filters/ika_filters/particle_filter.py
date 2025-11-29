#!/usr/bin/env python3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from ika_utils.nav_utils import ll2xy
from ika_msgs.msg import MotorFeedback
from ika_utils.motion_model_utils import normalize_angle
import numpy as np
import math
import random


class Particle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, vx=0.0, vy=0.0, omega=0.0, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.weight = weight


class ParticleFilter:
    def __init__(self, seed_heading=None, slip_factor=0.75, rotate_factor=0.6):
        self.num_particles = 750
        self.gps_noise = 0.5  # Adjusted for better GPS noise simulation
        self.process_noise = [0.15, 0.15, 0.3, 0.1, 0.1]  # x, y, theta, v, w
        self.origin_lat = None
        self.origin_lon = None
        self.seed_heading = seed_heading
        self.slip_factor = slip_factor
        self.rotate_factor = rotate_factor
        self.init_particles()

    def init_particles(self):
        self.particles = []
        if self.seed_heading is not None:
            self.particles = [
                Particle(
                    theta=self.seed_heading + np.random.normal(0, 0.1),
                )
                for _ in range(self.num_particles)
            ]
        else:
            self.particles = [
                Particle(theta=i / self.num_particles * 2 * np.pi)
                for i in range(self.num_particles)
            ]

    def feedback(self, motor_feedback: MotorFeedback):
        # take v, w, dt and update particles
        avg_particle = Particle()

        sum_weight = 0.0
        sum_avg_theta_x = 0.0
        sum_avg_theta_y = 0.0

        for particle in self.particles:
            particle = self.kinematic_model(
                particle, motor_feedback.v, motor_feedback.w, motor_feedback.dt
            )

            weight = particle.weight**2
            avg_particle.x += particle.x * weight
            avg_particle.y += particle.y * weight
            avg_particle.vx += particle.vx * weight
            avg_particle.vy += particle.vy * weight
            avg_particle.omega += particle.omega * weight
            avg_particle.weight += weight

            sum_avg_theta_x += math.cos(particle.theta) * weight
            sum_avg_theta_y += math.sin(particle.theta) * weight

            sum_weight += weight

        if sum_weight < 0.000001:
            sum_weight = 0.000001

        avg_particle.x /= sum_weight
        avg_particle.y /= sum_weight
        avg_particle.theta = math.atan2(
            sum_avg_theta_y / sum_weight, sum_avg_theta_x / sum_weight
        )
        avg_particle.vx /= sum_weight
        avg_particle.vy /= sum_weight
        avg_particle.omega /= sum_weight

        return Particle(
            x=avg_particle.x,
            y=avg_particle.y,
            theta=avg_particle.theta,
            vx=avg_particle.vx,
            vy=avg_particle.vy,
            omega=avg_particle.omega,
        )

    def gps_feedback(self, gps_msg: NavSatFix):
        if self.origin_lat is None or self.origin_lon is None:
            self.origin_lat = gps_msg.latitude
            self.origin_lon = gps_msg.longitude
            return None, None

        gps_x, gps_y = ll2xy(
            gps_msg.latitude, gps_msg.longitude, self.origin_lat, self.origin_lon
        )

        # if gps_x**2 + gps_y**2 > 5.0**2:
        #     self.on_reset()
        #     return None, None

        for particle in self.particles:
            dist_sqrt = np.sqrt((particle.x - gps_x) ** 2 + (particle.y - gps_y) ** 2)
            particle.weight = max(math.exp(-dist_sqrt / (2 * self.gps_noise**2)), 1e-10)

        self.resample()
        return [gps_x, gps_y]

    def resample(self):
        weights = [particle.weight for particle in self.particles]
        weights_sum = sum(weights)
        if weights_sum <= 0.00001:
            weights_sum = 0.00001
        weights = [weight / weights_sum for weight in weights]

        new_particles = random.choices(self.particles, weights, k=self.num_particles)
        self.particles = []

        for particle in new_particles:
            rand_x = np.random.normal(0, self.process_noise[0])
            rand_y = np.random.normal(0, self.process_noise[1])
            rand_v = np.random.normal(0, self.process_noise[2])
            rand_omega = np.random.normal(0, self.process_noise[3])
            rand_theta = np.random.normal(0, self.process_noise[4])

            # Create new particle to avoid modifying shared objects
            new_particle = Particle(
                x=particle.x + rand_x * math.cos(particle.theta),
                y=particle.y + rand_y * math.sin(particle.theta),
                theta=particle.theta + rand_theta * self.rotate_factor,
                vx=particle.vx + rand_v,
                vy=particle.vy,
                omega=particle.omega + rand_omega,
                weight=particle.weight,
            )

            self.particles.append(new_particle)

    def on_reset(self):
        self.origin_lat = None
        self.origin_lon = None
        self.init_particles()

    # UTILITY FUNCTIONS
    def kinematic_model(self, particle, v, w, dt):
        x = particle.x + v * math.cos(particle.theta) * dt
        y = particle.y + v * math.sin(particle.theta) * dt
        theta = normalize_angle(particle.theta + w * dt)
        omega = w * self.rotate_factor
        vx = v * self.slip_factor * math.cos(particle.theta)
        vy = v * self.slip_factor * math.sin(particle.theta)

        return Particle(x, y, theta, vx, vy, omega, particle.weight)
