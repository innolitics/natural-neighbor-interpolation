'''
Comparison of natural neighbor and linear barycentric interpolation.
'''
import numpy as np
import scipy.interpolate
import matplotlib as mpl
mpl.use('Agg')  # so it can run on Travis without a display
import matplotlib.pyplot as plt

import naturalneighbor


def error_str(errors):
    numerical_error = errors[~np.isnan(errors)]
    mean_err = np.mean(numerical_error)
    std_err = np.std(numerical_error)
    max_err = np.max(numerical_error)
    return "(Mean={:.2f}, Std={:.2f} Max={:.2f})".format(mean_err, std_err, max_err)


def compare_interp_for_func(func, func_as_string, image_name):
    coord_max = 60
    xmax = coord_max
    ymax = coord_max
    zmax = coord_max
    final_shape = (xmax, ymax, zmax)
    num_known_points = 100

    known_points = np.round(np.random.rand(num_known_points, 3) * np.min([xmax, ymax, zmax]))

    grid_ranges = [
        [0, xmax, 1],
        [0, ymax, 1],
        [0, zmax, 1],
    ]

    grid = np.mgrid[0:xmax:1, 0:ymax:1, 0:zmax:1]

    known_values = np.array([func(*point) for point in known_points], dtype=np.float64)
    true_values = np.reshape([func(x, y, z) for x, y, z in zip(*grid)], final_shape)

    linear_interp = scipy.interpolate.griddata(known_points, known_values, tuple(grid), method='linear')

    nn_interp = naturalneighbor.griddata(known_points, known_values, grid_ranges)
    nn_interp[np.isnan(linear_interp)] = float('nan')

    nn_interp_slice = nn_interp[:, :, 20]
    linear_interp_slice = linear_interp[:, :, 20]
    true_values_slice = true_values[:, :, 20]

    nn_interp_err = np.abs(nn_interp_slice - true_values_slice)
    linear_interp_err = np.abs(linear_interp_slice - true_values_slice)

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(true_values_slice)
    ax1.set_title("True Values\n{}".format(func_as_string))

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(nn_interp_err)
    nn_error_str = error_str(nn_interp_err)
    ax2.set_title("Natural Neighbor Abs Error\n{}".format(nn_error_str))

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(linear_interp_err)
    linear_error_str = error_str(linear_interp_err)
    ax3.set_title("Linear Barycentric Abs Error\n{}".format(linear_error_str))

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(nn_interp_slice)
    ax5.set_title("Natural Neighbor Values")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(linear_interp_slice)
    ax6.set_title("Linear Barycentric Values")

    plt.savefig(image_name, dpi=100)


if __name__ == '__main__':
    np.random.seed(100)

    compare_interp_for_func(
        (lambda x, y, z: np.sin(y / 10) + np.sin(x / 10)),
        'sin(y/10) + sin(x/10)',
        'sin_sin_comparison.png',
    )

    compare_interp_for_func(
        (lambda x, y, z: x + np.sin(x / 10) / 10),
        'x + sin(x/10)/10',
        'linear_comparison.png',
    )
