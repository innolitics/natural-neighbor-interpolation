'''
Comparison of natural neighbor and linear barycentric interpolation.
'''
import numpy as np
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt

import naturalneighbor

mpl.use('TkAgg')  # Required for matplotlib on MacOS systems


def display_method_error(method, interp_values, truth):
    error = (truth - interp_values)
    numerical_error = error[~np.isnan(error)]
    mean_err = np.mean(np.abs(numerical_error))
    std_err = np.std(np.abs(numerical_error))
    max_err = np.max(np.abs(numerical_error))
    print(f'''
    {method} Error Statistics:
        Mean absolute error ({method}): {mean_err}
        Standard deviation of error ({method}): {std_err}
        Max absolute error ({method}): {max_err}
        Standard Deviation of absolute error ({method}): {max_err}
    ''')


if __name__ == '__main__':
    coord_max = 60
    xmax = coord_max
    ymax = coord_max
    zmax = coord_max
    final_shape = (xmax, ymax, zmax)
    num_known_points = 100

    known_points = np.round(np.random.rand(num_known_points, 3) * np.min([xmax, ymax, zmax]))

    grid_ranges = [
        [0, xmax - 1, 1],
        [0, ymax - 1, 1],
        [0, zmax - 1, 1],
    ]

    grid = np.mgrid[0:xmax:1, 0:ymax:1, 0:zmax:1]

    def f(x, y, z):
        return np.sin(y / 10) + np.sin(x / 10)

    known_values = np.array([f(*point) for point in known_points], dtype=np.float64)
    true_values = np.reshape([f(x, y, z) for x, y, z in zip(*grid)], final_shape)

    nn_interp = naturalneighbor.griddata(known_points, known_values, grid_ranges)
    display_method_error('Natural Neighbor', nn_interp, true_values)

    linear_interp = scipy.interpolate.griddata(known_points, known_values, tuple(grid), method='linear')
    display_method_error('Linear Barycentric', linear_interp, true_values)

    plt.figure(1)
    plt.imshow(true_values[:, :, 20])
    plt.title("True Values")
    plt.figure(2)
    plt.imshow(nn_interp[:, :, 20])
    plt.title("Natural Neighbor")
    plt.figure(3)
    plt.title("Linear Barycentric")
    plt.imshow(linear_interp[:, :, 20])
    plt.show()
