import csv
from dataclasses import dataclass
from typing import List

import numpy as np

from tools import generate_cubic_spline


def generate_circular_track(radius: float, num_points: int) -> List:
    # Generate evenly spaced angles around the circle
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate x and y coordinates using polar to Cartesian conversion
    x_coordinates = radius * np.cos(angles)
    y_coordinates = radius * np.sin(angles)

    # Create a list of (x, y) coordinate pairs
    return list(zip(x_coordinates, y_coordinates))


def generate_ellipse_track(semi_minor: float, semi_major: float, num_points: int) -> List:
    # Generate evenly spaced angles
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate x and y coordinates using parametric equations for an ellipse
    x_coordinates = semi_minor * np.cos(angles)
    y_coordinates = semi_major * np.sin(angles)

    # Create a list of (x, y) coordinate pairs
    return list(zip(x_coordinates, y_coordinates))


@dataclass
class CircularPath:
    x_t: np.ndarray         # Path x-coordinate
    y_t: np.ndarray         # Path y-coordinate
    yaw_t: np.ndarray       # Path orientation
    k: np.ndarray           # Path curvature
    s: np.ndarray           # Path arc length
    width: float            # Path width
    radius: float           # Path radius

    def __init__(self, radius: float, width: float, file_path='../visualization/tracks/circle.csv', circle=True):
        if circle:
            with open(file_path, newline='') as f:
                rows = list(csv.reader(f, delimiter=','))

            ds = 0.05
            x, y = [[float(i) for i in row] for row in zip(*rows[1:])]
            self.x_t, self.y_t, self.yaw_t, self.k = generate_cubic_spline(x, y, ds)
            self.radius = radius
            self.length = 2 * np.pi * self.radius
            self.s = np.linspace(0, self.length, self.x_t.size)
            self.width = width


if __name__ == "__main__":
    # Generate points
    points = generate_circular_track(radius=50, num_points=100)
    # points = generate_ellipse_track(semi_minor=60, semi_major=30, num_points=100)

    # Specify the CSV file path
    csv_file = 'tracks/circle.csv'
    # csv_file = 'tracks/ellipse.csv'

    # Write the coordinates to a CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X-axis', 'Y-axis'])  # Write header
        writer.writerows(points)

    print(f'CSV file "{csv_file}" generated successfully.')
