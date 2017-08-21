#include <cmath>

#include <vector>
#include <algorithm>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "kdtree.h"
#include "geometry.h"


typedef geometry::Point<double, 3> Point;


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
   -1,
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
    PyArrayObject *known_points_ijk, *known_values, *interp_values;

    if (!PyArg_ParseTuple(args, "O!O!O!",
                &PyArray_Type, &known_points_ijk,
                &PyArray_Type, &known_values,
                &PyArray_Type, &interp_values)) {
        return NULL;
    }

    npy_intp* known_points_ijk_dims = PyArray_DIMS(known_points_ijk);
    std::size_t num_known_points = known_points_ijk_dims[0];

    npy_intp* interp_values_shape = PyArray_DIMS(interp_values);
    std::size_t ni = interp_values_shape[0];
    std::size_t nj = interp_values_shape[1];
    std::size_t nk = interp_values_shape[2];

    double* known_values_ptr = (double*)PyArray_GETPTR1(known_values, 0);
    double* interp_values_ptr = (double*)PyArray_GETPTR1(interp_values, 0);

    // TODO: make kd-tree manage memeory for points and values and remove these
    // intermediate vectors; perhaps would be best to pass-by-value into the
    // kdtree.add for our purposes
    auto known_points_ijk_vec = new std::vector<Point>(num_known_points);

    for (std::size_t i = 0; i < num_known_points; i++) {
        (*known_points_ijk_vec)[i] = Point(
                *(double*)PyArray_GETPTR2(known_points_ijk, i, 0),
                *(double*)PyArray_GETPTR2(known_points_ijk, i, 1),
                *(double*)PyArray_GETPTR2(known_points_ijk, i, 2));
    }

    kdtree::kdtree<double> *tree = new kdtree::kdtree<double>();
    for (std::size_t i = 0; i < num_known_points; i++) {
        tree->add(&(*known_points_ijk_vec)[i], known_values_ptr + i);
    }
    tree->build();

    auto contribution_counter = new double[ni*nj*nk]();

    for (std::size_t i = 0; i < ni; i++) {
        for (std::size_t j = 0; j < nj; j++) {
            for (std::size_t k = 0; k < nk; k++) {
                auto query_point = Point(i, j, k);
                auto nearest_known_point = tree->nearest_iterative(query_point);

                double distance_sq_query_to_known = nearest_known_point->distance;
                int roi_radius = ceil(sqrt(distance_sq_query_to_known));

                auto i_roi_min = clamp(i - roi_radius, 0, ni - 1);
                auto i_roi_max = clamp(i + roi_radius, 0, ni - 1);
                auto j_roi_min = clamp(j - roi_radius, 0, nj - 1);
                auto j_roi_max = clamp(j + roi_radius, 0, nj - 1);
                auto k_roi_min = clamp(k - roi_radius, 0, nk - 1);
                auto k_roi_max = clamp(k + roi_radius, 0, nk - 1);

                for (std::size_t i_roi = i_roi_min; i_roi <= i_roi_max; i_roi++) {
                    for (std::size_t j_roi = j_roi_min; j_roi <= j_roi_max; j_roi++) {
                        for (std::size_t k_roi = k_roi_min; k_roi <= k_roi_max; k_roi++) {
                            double deltai_2 = (i - i_roi)*(i - i_roi);
                            double deltaj_2 = (j - j_roi)*(j - j_roi);
                            double deltak_2 = (k - k_roi)*(k - k_roi);
                            double distance_sq_roi_to_known = deltai_2 + deltaj_2 + deltak_2;

                            if (distance_sq_roi_to_known <= distance_sq_query_to_known) {
                                std::size_t indice = nj*nk*i_roi + nk*j_roi + k_roi;
                                interp_values_ptr[indice] += nearest_known_point->value;
                                contribution_counter[indice] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < ni; i++) {
        for (std::size_t j = 0; j < nj; j++) {
            for (std::size_t k = 0; k < nk; k++) {
                auto indice = nj*nk*i + nk*j + k;
                if (contribution_counter[indice] != 0) {
                    interp_values_ptr[indice] /= contribution_counter[indice];
                }
            }
        }
    }

    delete[] contribution_counter;
    delete tree;
    delete known_points_ijk_vec;

    Py_RETURN_NONE;
}
