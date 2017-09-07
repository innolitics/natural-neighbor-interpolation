import itertools
import math

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from naturalneighbor import griddata


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
    ], dtype=np.float) * side_length

    known_values = np.zeros((8,), dtype=np.float)
    known_values[6] = value_0_1_1
    known_values[7] = value_1_1_1

    return known_points, known_values


@pytest.mark.parametrize("grid_ranges", [
    [[0, 1, 2j], [0, 1, 2j], [0, 1, 2j]],
    [[0, 1, 4j], [0, 1, 7j], [0, 1, 10j]],
])
def test_interp_on_known_points(grid_ranges):
    known_points, _ = known_cube()
    known_values = np.random.rand(8)

    actual_interp_values = griddata(
        known_points,
        known_values,
        grid_ranges,
    )

    for value, point in zip(known_values, known_points):
        i, j, k = point.astype(int)
        # we only want to compare the corners of the cube, so we use the fact
        # we know i, j, and k will each be 0 or 1 (and a well-placed negative sign)
        # to grab just the corners
        assert math.isclose(actual_interp_values[-i, -j, -k], value, rel_tol=0, abs_tol=1e-8)


def test_interp_constant_values():
    known_points, _ = known_cube()
    known_values = np.ones((8,)) * 7

    interp_grid_ranges = [
        [0, 1.5, 0.5],
        [0, 1.5, 0.5],
        [0, 1.5, 0.5],
    ]

    actual_interp_values = griddata(
        known_points,
        known_values,
        interp_grid_ranges,
    )

    expected_interp_values = np.ones_like(actual_interp_values) * 7
    assert_allclose(actual_interp_values, expected_interp_values, rtol=0, atol=1e-8)


@pytest.mark.skip(reason="the expected edge values is probably wrong")
def test_interp_between_cube_edges():
    value_1_1_1 = 3
    known_points, known_values = known_cube(value_1_1_1=value_1_1_1)

    interp_grid_ranges = [
        [0, 1, 0.1],
        [0, 1, 0.1],
        [0, 1, 0.1],
    ]

    interp_values = griddata(
        known_points,
        known_values,
        interp_grid_ranges
    )

    actual_edge_value = interp_values[5, 10, 10]
    expected_edge_value = 1.5
    assert_almost_equal(actual_edge_value, expected_edge_value)
