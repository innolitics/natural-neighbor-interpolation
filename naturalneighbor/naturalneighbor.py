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

    interp_ranges_arr = np.array(interp_ranges)

    if interp_ranges_arr.shape != (num_dimensions, 3):
        raise ValueError("Invalid interp_ranges: should be Dx3")

    starts = np.real(interp_ranges_arr[:, 0])
    stops = np.real(interp_ranges_arr[:, 1])
    ranges = stops - starts
    steps = interp_ranges_arr[:, 2].astype(np.complex)

    real_steps = np.imag(steps) == 0
    imag_steps = np.imag(steps) != 0

    if np.any(steps[real_steps] <= 0):
        raise ValueError("Invalid interp_ranges: real step <= 0")

    if np.any(steps[imag_steps] == 0):
        raise ValueError("Invalid interp_ranges: imag step == 0")

    if np.any(ranges <= 0):
        raise ValueError("Invalid interp_ranges: start < stop")

    # mimick the somewhat complex behavior of np.mgrid, since usually scipy
    # user's will be using mgrid along with griddata, and we want to make it
    # easy to switch over
    output_shape = np.empty(num_dimensions, dtype=np.double)
    step_sizes = np.empty(num_dimensions, dtype=np.double)
    for i, (start, stop, step) in enumerate(interp_ranges):
        temp_grid = np.mgrid[start:stop:step]
        output_shape[i] = len(temp_grid)

        if output_shape[i] > 1:
            step_sizes[i] = (temp_grid[-1] - temp_grid[0]) / (output_shape[i] - 1)
        else:
            step_sizes[i] = 1

    interp_values_shape = np.floor(output_shape).astype(np.int)
    interp_values = np.zeros(interp_values_shape, dtype=np.double)

    known_points_ijk = _xyz_to_ijk(known_points, starts, step_sizes)
    known_points_ijk = np.ascontiguousarray(known_points_ijk, dtype=np.double)

    known_values = np.ascontiguousarray(known_values, dtype=np.double)

    cnaturalneighbor.griddata(
        known_points_ijk,
        known_values,
        interp_values,
    )

    return interp_values


def _xyz_to_ijk(points_xyz, starts, steps):
    return (points_xyz - starts) / steps
