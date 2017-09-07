import itertools
import math

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from naturalneighbor import griddata


def known_cube(side_length=1):
    return np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=np.float) * side_length


@pytest.mark.parametrize("grid_ranges", [
    [[0, 1, 2j], [0, 1, 2j], [0, 1, 2j]],
    [[0, 1, 4j], [0, 1, 7j], [0, 1, 10j]],
])
def test_interp_on_known_points(grid_ranges):
    known_points = known_cube()
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
    known_points = known_cube()
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


def test_interp_between_cube_edges():
    known_points = known_cube()
    known_values = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    num_points = 1001
    interp_grid_ranges = [
        [0, 1, 2j],
        [0, 1, 2j],
        [0, 1, num_points*1j],
    ]

    actual_interp_values = griddata(
        known_points,
        known_values,
        interp_grid_ranges
    )

    expected_interp_values = np.linspace(0, 1, num_points)

    # note the tolerance is loose, because discrete natural neighbor does
    # introduce errors into the interpolated values
    assert_allclose(actual_interp_values[0, 0, :], expected_interp_values, rtol=0, atol=1e-2)
