from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from math import sin, cos, atan2
from scipy.interpolate import CubicSpline


def initialise_cubic_spline(x: ArrayLike, y: ArrayLike, ds: float, bc_type: str) -> Tuple[CubicSpline, np.ndarray]:
    distance = np.concatenate((np.zeros(1), np.cumsum(np.hypot(np.ediff1d(x), np.ediff1d(y)))))
    points = np.array([x, y]).T
    s = np.arange(0, distance[-1], ds)

    try:
        cs = CubicSpline(distance, points, bc_type=bc_type, axis=0, extrapolate=False)

    except ValueError as e:
        raise ValueError(
            f"{e} If you are getting a sequence error, do check if your input dataset contains consecutive duplicate(s).")

    return cs, s


def generate_cubic_spline(x: ArrayLike, y: ArrayLike, ds: float = 0.05, bc_type: str = 'natural') -> Tuple[
    np.ndarray, ...]:
    cs, s = initialise_cubic_spline(x, y, ds, bc_type)

    dx, dy = cs.derivative(1)(s).T
    yaw = np.arctan2(dy, dx)

    ddx, ddy = cs.derivative(2)(s).T
    curvature = (ddy * dx - ddx * dy) / ((dx * dx + dy * dy) ** 1.5)

    cx, cy = cs(s).T
    return cx, cy, yaw, curvature


def normalise_angle(angle):
    return atan2(sin(angle), cos(angle))


def get_rotation_matrix(angle: float) -> np.ndarray:
    cos_angle = cos(angle)
    sin_angle = sin(angle)

    return np.array([
        (cos_angle, sin_angle),
        (-sin_angle, cos_angle)
    ])
