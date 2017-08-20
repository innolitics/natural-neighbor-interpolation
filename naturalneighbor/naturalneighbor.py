import numpy as np

import cnaturalneighbor


def griddata(known_points, known_values, interp_ranges):
    if known_points.ndim != 2:
        raise ValueError("known_points must be a NxD array with N>0")

    if known_values.ndim != 1:
        raise ValueError("known_values must be a 1D array")

    num_dimensions = known_points.shape[1]

    if num_dimensions != 3:
        raise NotImplementedError("Currently only support D=3")

    num_known_points = known_points.shape[0]
    num_known_values = known_values.shape[0]

    if num_known_points != num_known_values:
        raise ValueError("Number of known_points != number of known_values")

    interp_ranges = np.array(interp_ranges, dtype=np.double, order='C')

    if interp_ranges.shape != (num_dimensions, 3):
        raise ValueError("Invalid interp_ranges: should be Dx3")

    starts = interp_ranges[:, 0]
    stops = interp_ranges[:, 1]
    steps = interp_ranges[:, 2]

    if np.any(steps <= 0):
        raise ValueError("Invalid interp_ranges: step <= 0")

    if np.any(stops - starts <= 0):
        raise ValueError("Invalid interp_ranges: start < stop")

    interp_values_shape = np.floor(1 + (stops - starts)/steps).astype(np.int)
    interp_values = np.zeros(interp_values_shape, dtype=np.double)

    known_points_ijk = _xyz_to_ijk(known_points, starts, steps)
    known_points_ijk = np.ascontiguousarray(known_points_ijk, dtype=np.double)

    known_values = np.ascontiguousarray(known_values, dtype=np.double)

    cnaturalneighbor.griddata(
        known_points_ijk,
        known_values,
        interp_values,
    )

    return interp_values


def _xyz_to_ijk(points_xyz, starts, steps):
    return (points_xyz - starts)/steps
