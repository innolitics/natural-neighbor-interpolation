import scipy.interpolate
import numpy as np

import naturalneighbor


def test_output_size_matches_scipy():
    points = np.random.rand(10, 3)
    values = np.random.rand(10)

    grid_ranges = [
        [0, 4, 0.6],  # step isn't a multiple
        [-3, 3, 1.0],  # step is a multiple
        [0, 1, 3],  # step is larger than stop - start
    ]

    mesh_grids = tuple(np.mgrid[
        grid_ranges[0][0]:grid_ranges[0][1]:grid_ranges[0][2],
        grid_ranges[1][0]:grid_ranges[1][1]:grid_ranges[1][2],
        grid_ranges[2][0]:grid_ranges[2][1]:grid_ranges[2][2],
    ])

    scipy_result = scipy.interpolate.griddata(points, values, mesh_grids)
    nn_result = naturalneighbor.griddata(points, values, grid_ranges)

    assert scipy_result.shape == nn_result.shape
