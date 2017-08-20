#include <cmath>

#include <vector>
#include <algorithm>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "kdtree.h"
#include "geometry.h"

static char module_docstring[] = "Discrete natural neighbor interpolation in 3D.";

static char griddata_docstring[] = "Calculate the natural neighbor interpolation of a dataset.";

static PyObject* cnaturalneighbor_griddata(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {
    {"griddata", cnaturalneighbor_griddata, METH_VARARGS, griddata_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "cnaturalneighbor",
   module_docstring,
   -1,  // TODO: investigate whether we can support sub-interpreters
   module_methods
};

PyMODINIT_FUNC PyInit_cnaturalneighbor(void) {
    PyObject* m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }

    import_array();  // Enable numpy functionality
    return m;
}

typedef geometry::Point<double, 3> Point;

std::size_t clamp(std::size_t val, std::size_t min, std::size_t max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    } else {
        return val;
    }
}

static PyObject* cnaturalneighbor_griddata(PyObject* self, PyObject* args) {
    // TODO: remove contribute counter from inputs
    PyArrayObject *known_coords, *known_values, *interpolated_values, *contribution_counter;

    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                &PyArray_Type, &known_coords,
                &PyArray_Type, &known_values,
                &PyArray_Type, &interpolated_values,
                &PyArray_Type, &contribution_counter)) {
        return NULL;
    }

    npy_intp* known_coordinates_dims = PyArray_DIMS(known_coords);
    npy_intp* interpolated_values_shape = PyArray_DIMS(interpolated_values);
    double* interpolated_values_ptr = (double*)PyArray_GETPTR1(interpolated_values, 0);
    double* contribution_counter_ptr = (double*)PyArray_GETPTR1(contribution_counter, 0);

    std::size_t num_known_points = known_coordinates_dims[0];

    // TODO: think of a way to avoid using these intermediate vectors
    auto known_coords_vec = new std::vector<Point>(num_known_points);
    auto known_values_vec = new std::vector<double>(num_known_points);

    for (std::size_t i = 0; i < num_known_points; i++) {
        (*known_coords_vec)[i] = Point(
                *(double*)PyArray_GETPTR2(known_coords, i, 0),
                *(double*)PyArray_GETPTR2(known_coords, i, 1),
                *(double*)PyArray_GETPTR2(known_coords, i, 2));

        (*known_values_vec)[i] = *(double*)PyArray_GETPTR1(known_values, i);
    }

    kdtree::kdtree<double> *tree = new kdtree::kdtree<double>();
    for (std::size_t i = 0; i < num_known_points; i++) {
        tree->add(&(*known_coords_vec)[i], &(*known_values_vec)[i]);
    }
    tree->build();

    // Scatter method discrete Sibson
    // For each interpolation point p, search neighboring interpolation points
    // within a sphere of radius r, where r = distance to nearest known point.
    std::size_t ni = interpolated_values_shape[0];
    std::size_t nj = interpolated_values_shape[1];
    std::size_t nk = interpolated_values_shape[2];
    for (std::size_t i = 0; i < ni; i++) {
        for (std::size_t j = 0; j < nj; j++) {
            for (std::size_t k = 0; k < nk; k++) {
                const kdtree::QueryResult *q = tree->nearest_iterative(
                    Point(i, j, k)
                );
                
                double distance_squared = q->distance;
                int r = ceil(sqrt(distance_squared));
                // Search neighboring interpolation points within a bounding box
                // of r indices. From this subset of points, calculate their distance
                // and tally the ones that fall within the sphere of radius r surrounding
                // interpolation_points[i].

                auto i_neighborhood_min = clamp(i - r, 0, ni);
                auto i_neighborhood_max = clamp(i + r, 0, ni);
                auto j_neighborhood_min = clamp(j - r, 0, nj);
                auto j_neighborhood_max = clamp(j + r, 0, nj);
                auto k_neighborhood_min = clamp(k - r, 0, nk);
                auto k_neighborhood_max = clamp(k + r, 0, nk);


                for (auto i_neighborhood = i_neighborhood_min; i_neighborhood < i_neighborhood_max; i_neighborhood++) {
                    for (auto j_neighborhood = j_neighborhood_min; j_neighborhood < j_neighborhood_max; j_neighborhood++) {
                        for (auto k_neighborhood = k_neighborhood_min; k_neighborhood < k_neighborhood_max; k_neighborhood++){
                            double distance_i = i - i_neighborhood;
                            double distance_j = j - j_neighborhood;
                            double distance_k = k - k_neighborhood;
                            if (distance_i*distance_i + distance_j*distance_j
                                    + distance_k*distance_k > distance_squared){
                                continue;
                            }
                            interpolated_values_ptr[i + ni*j + ni*nj*k] += q->value;
                            contribution_counter_ptr[i + ni*j + ni*nj*k] += 1;
                        }
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < ni; i++) {
        for (std::size_t j = 0; j < nj; j++) {
            for (std::size_t k = 0; k < nk; k++) {
                if (contribution_counter_ptr[i + ni*j + ni*nj*k] != 0) {
                    interpolated_values_ptr[i + ni*j + ni*nj*k] /= contribution_counter_ptr[i + ni*j + ni*nj*k];
                }
            }
        }
    }

    delete tree;
    Py_RETURN_NONE;
}
