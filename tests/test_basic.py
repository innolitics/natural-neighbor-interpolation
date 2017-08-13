import unittest

import numpy as np

from naturalneighbor import natural_neighbor


class TestBasic(unittest.TestCase):
    def test_output_shape_correct(self):
        points = np.random.rand(100, 3)
        values = np.linalg.norm(points)
        grid = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j]
        interpolated_values = natural_neighbor(points, values, grid, 10)
        self.assertEqual(interpolated_values.shape, (10, 10, 10))
