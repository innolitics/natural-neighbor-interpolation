import numpy as np

from . import _naturalneighbor


def naturalneighbor(points, values, interpolated_grid_ranges):
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Points must be a Nx3 dimensional array where N>0")
    if values.ndim != 1 or values.shape[0] != points.shape[0]:
        raise ValueError("Values must be a 1 dimensional and have same number of values as points.")
    points_arr = np.ascontiguousarray(points, dtype=np.double)
    values_arr = np.ascontiguousarray(values, dtype=np.double)
    interpolated_grid_ranges_arr = np.ascontiguousarray(interpolated_grid_ranges, dtype=np.int)
    if interpolated_grid_ranges_arr.shape[0] == 3 and interpolated_grid_ranges_arr.shape[1] == 3:
        raise ValueError("Interpolated grid ranges must be an array like object of size 3x3 where the first dimension "
                         "is x,y,z and the second is (min value, max value, number of points).")

    interpolated_values = _naturalneighbor.natural_neighbor(points_arr, values_arr, interpolated_grid_ranges_arr)
    return interpolated_values
