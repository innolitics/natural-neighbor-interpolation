import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

import naturalneighbor


def test_identity_3x3():
    starts = np.array([0, 0, 0])
    steps = np.array([1, 1, 1])
    num_dimensions = 3
    num_points = 10
    points_xyz = np.random.rand(num_points, num_dimensions)
    actual_points_ijk = naturalneighbor._xyz_to_ijk(points_xyz, starts, steps)
    expected_points_ijk = points_xyz
    assert_allclose(actual_points_ijk, expected_points_ijk, rtol=0, atol=1e-10)


def test_shift_3x3():
    starts = np.array([0, -1, 2])
    steps = np.array([1, 1, 1])
    num_dimensions = 3
    num_points = 10
    points_xyz = np.array([[0, 0, 0], [1, 1, 1]])
    actual_points_ijk = naturalneighbor._xyz_to_ijk(points_xyz, starts, steps)
    expected_points_ijk = np.array([[0, 1, -2], [1, 2, -1]])
    assert_allclose(actual_points_ijk, expected_points_ijk, rtol=0, atol=1e-10)


def test_shift_and_scale_3x3():
    starts = np.array([0, -1, 2])
    steps = np.array([1, 0.5, 0.125])
    num_dimensions = 3
    num_points = 10
    points_xyz = np.array([[0, 0, 0], [1, 1, 1]])
    actual_points_ijk = naturalneighbor._xyz_to_ijk(points_xyz, starts, steps)
    expected_points_ijk = np.array([[0, 2, -16], [1, 4, -8]])
    assert_allclose(actual_points_ijk, expected_points_ijk, rtol=0, atol=1e-10)
