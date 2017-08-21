'''
Comparison of natural neighbor and linear barycentric interpolation.
'''
import numpy as np
from scipy.interpolate import griddata
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from naturalneighbor import natural_neighbor


def linear_barycentric_interpolation(known_points, known_vals, grid_ranges):
    grid = np.mgrid[*(slice(r) for r in grid_ranges)]
    return griddata(known_points, known_vals, grid, method='linear')


def display_method_error(method, interpolated_values, truth):
    error = (truth - interpolated_values)
    numerical_error = error[~np.isnan(error)]
    mean_err = np.mean(np.abs(numerical_error))
    std_err = np.std(np.abs(numerical_error))
    max_err = np.max(np.abs(numerical_error))
    print(f'''
    {method} Error Statistics:
        Mean absolute error ({method}): {mean_err}
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

    def f(x, y, z):
        return np.sin(y/10) + np.sin(x/10)

    known_vals = np.array([f(*point) for point in known_points], dtype=np.float64)
    true_vals = np.reshape([f(x,y,z) for x,y,z in zip(*grid)], final_shape)

    grid_ranges = [
        [0, xmax, 1],
        [0, ymax, 1],
        [0, zmax, 1],
    ]

    print("Beginning Natural Neighbor Interpolation")
    nn_interp = natural_neighbor(known_points, known_vals, grid_ranges)
    linear_interp = linear_barycentric_interpolation(known_points, known_vals, grid, final_shape)
    linear_interp = np.reshape(linear_interp, final_shape)
    display_method_error('Linear Barycentric', linear_interp, true_vals)
    display_method_error('Natural Neighbor', nn_interp, true_vals)

    plt.figure(1)
    plt.imshow(true_vals[:,:,20])
    plt.title("True Values")
    plt.figure(2)
    plt.imshow(nn_interp[:,:,20])
    plt.title("Natural Neighbor")
    plt.figure(3)
    plt.title("Linear Barycentric")
    plt.imshow(linear_interp[:,:,20])
    plt.show()
