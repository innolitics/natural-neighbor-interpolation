import unittest

import numpy as np

from naturalneighbor import natural_neighbor


def generate_grid(*final_shape):
    xmax, ymax, zmax = [dim - 1 for dim in final_shape]
    points = []
    x_range = np.arange(0, xmax + 1)
    y_range = np.arange(0, ymax + 1)
    z_range = np.arange(0, zmax + 1)
    for x in x_range:
        for y in y_range:
            for z in z_range:
                points.append((x, y, z))

    grid = tuple(zip(*points))
    return grid


class TestInputValidation(unittest.TestCase):
    def test_mismatched_num_values_and_points(self):
        points = np.random.rand(100, 3)
        values = np.linalg.norm(points)
        coord_max = (10, 10, 10)
        grid = generate_grid(*coord_max)
        interpolated_values = natural_neighbor(points, values, grid, max(coord_max))
        self.assertEqual(interpolated_values.shape, (10, 10, 10))
