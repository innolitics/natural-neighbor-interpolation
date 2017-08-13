'''
Testing natural neighbor interpolation.
'''
import numpy as np
from scipy.interpolate import griddata
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from naturalneighbor import natural_neighbor



def generate_input_grid(final_shape, num_known_points):
    xmax, ymax, zmax = [dim-1 for dim in final_shape]
    points = []
    x_range = np.arange(0, xmax+1)
    y_range = np.arange(0, ymax+1)
    z_range = np.arange(0, zmax+1)
    for x in x_range:
        for y in y_range:
            for z in z_range:
                points.append((x,y,z))

    grid = tuple(zip(*points))
    data_coords = np.round(np.random.rand(num_known_points, 3) * np.min([xmax, ymax, zmax]))
    known_points = np.array(data_coords)
    return (grid, known_points)


def linear_barycentric_interpolation(data_coords, known_vals, grid, final_shape):
    gridded = griddata(data_coords, known_vals, grid, method='linear')
    return np.reshape(gridded, final_shape)


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
    coord_max = 30
    xmax = coord_max
    ymax = coord_max
    zmax = coord_max
    final_shape = (xmax, ymax, zmax)
    num_known_points = 50
    grid, known_points = generate_input_grid(final_shape, num_known_points)

    def f(x, y, z):
        return np.sin(y/10) + np.sin(x/10)

    known_vals = np.array([f(*point) for point in known_points], dtype=np.float64)
    true_vals = np.reshape([f(x,y,z) for x,y,z in zip(*grid)], final_shape)

    print("Beginning Interpolation")
    nn_interp = natural_neighbor(known_points, known_vals, grid, coord_max)
    linear_interp = linear_barycentric_interpolation(known_points, known_vals, grid, final_shape)
    nn_interp = np.reshape(nn_interp, final_shape)
    nn_interp[np.isnan(linear_interp)] = float('NaN')
    true_vals[np.isnan(linear_interp)] = float('NaN')

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
