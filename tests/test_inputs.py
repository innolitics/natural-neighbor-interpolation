import unittest

import numpy as np

from naturalneighbor import natural_neighbor


class TestInputValidation(unittest.TestCase):
    def test_mismatched_num_values_and_points(self):
        points = np.random.rand(100, 3)
        values = np.linalg.norm(points)
        grid = np.mgrid[0:1:10j, 0:1:10j, 0:1,10j]
        interpolated_values = natural_neighbor(points, values, grid)
        self.assertEqual(interpolated_values.shape, (10, 10, 10))
