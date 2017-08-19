import numpy as np

# from . import _naturalneighbor

def apply_affine(affine_matrix, A):
    mm, m = A.shape
    assert mm == 3
    A1 = np.vstack((A, np.ones((1, m), dtype=float)))
    A1_transformed = affine_matrix @ A1
    A_transformed = A1_transformed[:3, :]
    return A_transformed


def convert_xyz_to_ijk(points_xyz, interpolated_grid_ranges):
    mins = interpolated_grid_ranges[:, 0]
    scales = interpolated_grid_ranges[:, 2]
    # scale then translate
    ijk2xyz = np.array([
        [1, 0, 0, mins[0]],
        [0, 1, 0, mins[1]],
        [0, 0, 1, mins[2]],
        [0, 0, 0, 1],
    ]) @ np.array([
        [scales[0], 0, 0, 0],
        [0, scales[1], 0, 0],
        [0, 0, scales[2], 0],
        [0, 0, 0, 1]
    ])

    xyz2ijk = np.linalg.inv(ijk2xyz)
    return apply_affine(xyz2ijk, points_xyz)

def naturalneighbor(points_xyz, values, interpolated_grid_ranges):
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3 or points_xyz.shape[0] == 0:
        raise ValueError("Points must be a Nx3 dimensional array where N>0")
    if values.ndim != 1 or values.shape[0] != points_xyz.shape[0]:
        raise ValueError("Values must be a 1 dimensional and have same number of values as points.")
    interpolated_grid_ranges_arr = np.ascontiguousarray(interpolated_grid_ranges, dtype=np.int)
    if interpolated_grid_ranges_arr.shape[0] == 3 and interpolated_grid_ranges_arr.shape[1] == 3:
        raise ValueError("Interpolated grid ranges must be an array like object of size 3x3 where the first dimension "
                         "is x,y,z and the second is (min value, max value, number of points).")

    values_arr = np.ascontiguousarray(values, dtype=np.double)
    points_ijk = np.ascontiguousarray(convert_xyz_to_ijk(points_xyz, interpolated_grid_ranges))
    interpolated_values = _naturalneighbor.natural_neighbor(points_ijk, values_arr, interpolated_grid_ranges_arr)
    return interpolated_values
