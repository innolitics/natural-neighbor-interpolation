#include <cmath>

#include <vector>
#include <algorithm>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "kdtree.h"
#include "geometry.h"

static char module_docstring[] = "Discrete natural neighbor interpolation in 3D.";

static char nn_docstring[] = "Calculate the natural neighbor interpolation of a dataset.";

static PyObject* naturalneighbor_natural_neighbor(PyObject* module, PyObject* args);

static PyMethodDef module_methods[] = {
    {"natural_neighbor", naturalneighbor_natural_neighbor, METH_VARARGS, nn_docstring}
};

static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "naturalneighbor",
   module_docstring,
   -1,  // TODO: investigate whether we can support sub-interpreters
   module_methods
};

PyMODINIT_FUNC PyInit_naturalneighbor(void) {
    PyObject* m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }

    import_array();  // Enable numpy functionality
    return m;
}

typedef geometry::Point<double, 3> Point;

static PyObject* naturalneighbor_natural_neighbor(PyObject* module, PyObject* args) {
    PyArrayObject *known_coords, *known_values, *interpolated_coord_ranges, *interpolated_values;

    if (!PyArg_ParseTuple(args, "O!O!O!", 
                &PyArray_Type, &known_coords,
                &PyArray_Type, &known_values,
                &PyArray_Type, &interpolated_coord_ranges,
                &PyArray_Type, &interpolated_values)) {
        return NULL;
    }

    npy_intp* known_coordinates_dims = PyArray_DIMS(known_coords);
    double* known_values = (double*)PyArray_GETPTR1(known_values, 0);

    int num_known_points = known_coordinates_dims[0];

    // TODO: think of a way to avoid using these intermediate vectors
    auto known_coords_vec = new std::vector<Point>(num_known_points);
    auto known_values_vec = new std::vector<double>(num_known_points);

    for (npy_intp i = 0; i < num_known_points; i++) {
        known_coords_vec[i] = Point(
                *(double*)PyArray_GETPTR2(known_coords, i, 0),
                *(double*)PyArray_GETPTR2(known_coords, i, 1),
                *(double*)PyArray_GETPTR2(known_coords, i, 2));

        known_values_vec[i] = *(double*)PyArray_GETPTR1(known_values_vec, i);
    }

    kdtree::kdtree<double> *tree = new kdtree::kdtree<double>();
    for (std::size_t i = 0; i < num_known_points; i++) {
        tree->add(&known_coords_vec[i], &known_values_vec[i]);
    }
    tree->build();

    auto interpolation_values = new std::vector<double>(interpolation_points.size(), 0.0);
    auto contribution_counter = new std::vector<int>(interpolation_points.size(), 0);

    int xscale = coord_max*coord_max;
    int yscale = coord_max;

    // Scatter method discrete Sibson
    // For each interpolation point p, search neighboring interpolation points
    // within a sphere of radius r, where r = distance to nearest known point.
    for (std::size_t i = 0; i < interpolation_points.size(); i++) {
        const kdtree::QueryResult *q = tree->nearest_iterative(interpolation_points[i]);
        double comparison_distance = q->distance;
        int r = floor(comparison_distance);
        int px = interpolation_points[i][0];
        int py = interpolation_points[i][1];
        int pz = interpolation_points[i][2];
        // Search neighboring interpolation points within a bounding box
        // of r indices. From this subset of points, calculate their distance
        // and tally the ones that fall within the sphere of radius r surrounding
        // interpolation_points[i].

        auto x_neighborhood_min = std::clamp(px - r, 0, coord_max);
        auto x_neighborhood_max = std::clamp(px + r, 0, coord_max);
        auto y_neighborhood_min = std::clamp(py - r, 0, coord_max);
        auto y_neighborhood_max = std::clamp(py + r, 0, coord_max);
        auto z_neighborhood_min = std::clamp(pz - r, 0, coord_max);
        auto z_neighborhood_max = std::clamp(pz + r, 0, coord_max);

        for (auto x = x_neighborhood_min; x < x_neighborhood_max; x++) {
            for (auto y = y_neighborhood_min; y < y_neighborhood_max; y++) {
                for (auto z = z_neighborhood_min; z < z_neighborhood_max; z++) {

                    int idx = x*xscale + y*yscale + z;
                    double distance_x = interpolation_points[idx][0] - px;
                    double distance_y = interpolation_points[idx][1] - py;
                    double distance_z = interpolation_points[idx][2] - pz;
                    if (distance_x*distance_x + distance_y*distance_y
                            + distance_z*distance_z > comparison_distance){
                        continue;
                    }
                    (*interpolation_values)[idx] += q->value;
                    (*contribution_counter)[idx] += 1;
                }
            }
        }
    }

    for (std::size_t i = 0; i < interpolation_values->size(); i++) {
        if ((*contribution_counter)[i] != 0) {
            (*interpolation_values)[i] /= (*contribution_counter)[i];
        } else {
            (*interpolation_values)[i] = 0; //TODO: this is just 0, better way to mark NAN?
        }
    }

    delete contribution_counter;
    delete tree;
    delete known_coords_vec;
    delete known_values_vec;

    return interpolation_values;
}
}
