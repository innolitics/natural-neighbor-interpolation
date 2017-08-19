import unittest

import numpy as np

from naturalneighbor.naturalneighbor import convert_xyz_to_ijk

class TestBasic(unittest.TestCase):
    def test_output_shape_correct(self):
        points = np.random.rand(100, 3)
        values = np.linalg.norm(points)
        grid = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j]
        interpolated_values = naturalneighbor(points, values, grid)
        #self.assertEqual(interpolated_values.shape, (10, 10, 10))

    def test_convert_xyz_to_ijk(self):
        interpolated_grid_ranges = np.array([
            [1, 10, 1],
            [2, 6, 2],
            [1, 10, 2],
        ])
        np.testing.assert_allclose(
            [[0,0,0]],
            convert_xyz_to_ijk(np.array([[1,2,1]]).T, interpolated_grid_ranges).T
        )
        np.testing.assert_allclose(
            [[9,2,4]],
            convert_xyz_to_ijk(np.array([[10,6,9]]).T, interpolated_grid_ranges).T
        )