from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.lib.tools import get_rotation_matrix


@dataclass
class CarParameters:
    """
    Based on Tesla's model S 100D
    https://www.car.info/en-se/tesla/model-s/model-s-100-kwh-awd-16457112/specs
    """

    length: float = 4.97
    width: float = 1.964
    tire_diameter: float = 0.4826
    tire_width: float = 0.265
    axle_track: float = 1.7
    wheelbase: float = 2.96
    rear_overhang: float = 0.5 * (length - wheelbase)
    l_: float = 0.5
    l_r: float = l_ * wheelbase
    l_f: float = l_ * wheelbase
    max_steer: float = np.radians(33)
    max_velocity: float = 5
    color: str = 'black'


class CarDescription:
    def __init__(self):
        """
        Description of a car for visualising vehicle control in Matplotlib.
        All calculations are done w.r.t the vehicle's rear axle to reduce computation steps.

        At every time step
        :return outlines:               (ndarray) vehicle's outlines [x, y]
        :return front_right_wheel:      (ndarray) vehicle's front-right axle [x, y]
        :return rear_right_wheel:       (ndarray) vehicle's rear-right axle [x, y]
        :return front_left_wheel:       (ndarray) vehicle's front-left axle [x, y]
        :return rear_left_wheel:        (ndarray) vehicle's rear-right axle [x, y]
        """

        rear_axle_to_front_bumper = CarParameters.length - CarParameters.rear_overhang
        centerline_to_wheel_centre = 0.5 * CarParameters.axle_track
        centerline_to_side = 0.5 * CarParameters.width

        vehicle_vertices = np.array([
            (-CarParameters.rear_overhang, centerline_to_side),
            (rear_axle_to_front_bumper, centerline_to_side),
            (rear_axle_to_front_bumper, -centerline_to_side),
            (-CarParameters.rear_overhang, -centerline_to_side)
        ])

        half_tyre_width = 0.5 * CarParameters.tire_width
        centerline_to_inwards_rim = centerline_to_wheel_centre - half_tyre_width
        centerline_to_outwards_rim = centerline_to_wheel_centre + half_tyre_width

        # Rear right wheel vertices
        wheel_vertices = np.array([
            (-CarParameters.tire_diameter, -centerline_to_inwards_rim),
            (CarParameters.tire_diameter, -centerline_to_inwards_rim),
            (CarParameters.tire_diameter, -centerline_to_outwards_rim),
            (-CarParameters.tire_diameter, -centerline_to_outwards_rim)
        ])

        self.outlines = np.concatenate([vehicle_vertices, [vehicle_vertices[0]]])
        self.rear_right_wheel = np.concatenate([wheel_vertices, [wheel_vertices[0]]])

        # Reflect the wheel vertices about the x-axis
        self.rear_left_wheel = self.rear_right_wheel.copy()
        self.rear_left_wheel[:, 1] *= -1

        # Translate the wheel vertices to the front axle
        front_left_wheel = self.rear_left_wheel.copy()
        front_right_wheel = self.rear_right_wheel.copy()
        front_left_wheel[:, 0] += CarParameters.wheelbase
        front_right_wheel[:, 0] += CarParameters.wheelbase

        get_face_center = lambda vertices: np.array([
            0.5 * (vertices[0][0] + vertices[2][0]),
            0.5 * (vertices[0][1] + vertices[2][1])
        ])

        # Translate front wheels to origin
        self.fr_wheel_center = get_face_center(front_right_wheel)
        self.fl_wheel_center = get_face_center(front_left_wheel)
        self.fr_wheel_origin = front_right_wheel - self.fr_wheel_center
        self.fl_wheel_origin = front_left_wheel - self.fl_wheel_center

        # Class variables
        self.x: float = None
        self.y: float = None
        self.yaw_vector = np.empty((2, 2))

    def transform(self, point: np.ndarray) -> np.ndarray:
        # Vector rotation
        point = point.dot(self.yaw_vector).T

        # Vector translation
        point[0, :] += self.x
        point[1, :] += self.y

        return point

    def frenet_to_cartesian(self, x_road: np.ndarray):
        # TODO: Check that it's the x, y position of the center of the rear axle
        # TODO: Hardcoding for a circular path, in general, need to have an analytical expression of the path
        s, e, mu, v, d = x_road

        r = 50
        yaw = mu + (np.pi/2 + (s/r))
        xy = r * np.array([np.cos(s/r), np.sin(s/r)])
        xy += e * np.array([np.cos(np.pi + (s/r)), np.sin(np.pi + (s/r))])
        xy += CarParameters.l_r * np.array([np.cos(yaw), np.sin(yaw)])
        return xy[0], xy[1], v, yaw, d

    def plot_car(self, x_road: np.ndarray) -> Tuple[np.ndarray, ...]:
        x_global = self.frenet_to_cartesian(x_road)

        self.x, self.y, _, yaw, steer = x_global

        # Rotation matrices
        self.yaw_vector = get_rotation_matrix(yaw)
        steer_vector = get_rotation_matrix(steer)

        # Rotate the wheels about its position
        front_right_wheel = self.fr_wheel_origin.copy()
        front_left_wheel = self.fl_wheel_origin.copy()
        front_right_wheel = front_right_wheel @ steer_vector
        front_left_wheel = front_left_wheel @ steer_vector
        front_right_wheel += self.fr_wheel_center
        front_left_wheel += self.fl_wheel_center

        outlines = self.transform(self.outlines)
        rear_right_wheel = self.transform(self.rear_right_wheel)
        rear_left_wheel = self.transform(self.rear_left_wheel)
        front_right_wheel = self.transform(front_right_wheel)
        front_left_wheel = self.transform(front_left_wheel)

        return self.x, self.y, outlines, front_right_wheel, rear_right_wheel, front_left_wheel, rear_left_wheel
