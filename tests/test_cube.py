import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

import naturalneighbor
print(dir(naturalneighbor))


def known_cube(value_0_1_1=0, value_1_1_1=0, side_length=1):
    known_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],  # index 6
        [1, 1, 1],  # index 7
    ], dtype=np.float)*side_length

    known_values = np.zeros((8,), dtype=np.float)
    known_values[6] = value_0_1_1
    known_values[7] = value_1_1_1

    return known_points, known_values


def test_interpolate_on_known_points():
    '''
    If we interpolate precisely on the same grid as our known points, we
    should receive the exact same values back.
    '''
    known_points, _ = known_cube()
    known_values = np.random.rand(8)

    interpolation_grid_ranges = [
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
    ]

    actual_interpolated_values = naturalneighbor.natural_neighbor(
        known_points,
        known_values,
        interpolation_grid_ranges
    )

    expected_interpolated_values = np.array((2, 2, 2), dtype=np.float)
    for value, (i, j, k) in zip(known_values, known_points):
        expected_interpolated_values[i, j, k] = value

    assert_allclose(actual_interpolated_values, expected_interpolated_values, rtol=0, atol=0)


def test_interpolate_between_cube_edges():
    value_1_1_1 = 3
    value_0_1_1 = 7
    known_points, known_values = known_cube(value_0_1_1, value_1_1_1)

    interpolation_grid_ranges = [
        [0, 1, 0.5],
        [0, 1, 0.5],
        [0, 1, 0.5],
    ]

    interpolated_values = naturalneighbor.natural_neighbor(
        known_points,
        known_values,
        interpolation_grid_ranges
    )

    actual_edge_value = interpolated_values[1, 2, 2]
    expected_edge_value = (3.0 + 7.0)/2
    assert_almost_equal(actual_edge_value, expected_edge_value)
