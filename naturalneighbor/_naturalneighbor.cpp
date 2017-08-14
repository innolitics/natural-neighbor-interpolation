#include <vector>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "geometry.h"
#include "nn.h"

static char module_docstring[] = "Discrete natural neighbor interpolation in 3D.";

static char nn_docstring[] = "Calculate the natural neighbor interpolation of a dataset.";

static PyObject *naturalneighbor_natural_neighbor(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"natural_neighbor", naturalneighbor_natural_neighbor, METH_VARARGS, nn_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "naturalneighbor",
   module_docstring,
   -1,
   module_methods
};

PyMODINIT_FUNC PyInit_naturalneighbor(void) {
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) { return NULL; }

    import_array(); // Enable numpy functionality
    return m;
}

typedef geometry::Point<double, 3> Point;

static PyObject *naturalneighbor_natural_neighbor(PyObject *self, PyObject *args) {
    int coord_max;
    PyObject *known_coord_obj, *known_values_obj, *interpolation_points_obj;

    if (!PyArg_ParseTuple(args, "OOOi", &known_coord_obj,
                          &known_values_obj, &interpolation_points_obj, &coord_max)) {
        return NULL;
    }
    PyObject *known_coord_numpy_arr = PyArray_FROM_OTF(known_coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *known_values_numpy_arr = PyArray_FROM_OTF(known_values_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *interpolation_points_numpy_arr = PyArray_FROM_OTF(interpolation_points_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (known_coord_numpy_arr == NULL || known_values_numpy_arr == NULL || interpolation_points_numpy_arr == NULL) {
        Py_XDECREF(known_coord_numpy_arr);
        Py_XDECREF(known_values_numpy_arr);
        Py_XDECREF(interpolation_points_numpy_arr);
        return NULL;
    }

    npy_intp *known_coordinates_dims = PyArray_DIMS(known_coord_numpy_arr);
    double *known_values = (double*)PyArray_GETPTR1(known_values_numpy_arr, 0);
    npy_intp *interpolation_points_dims = PyArray_DIMS(interpolation_points_numpy_arr);

    //TODO: Add in bounds checking if user passes invalid shape
    int num_known_points = known_coordinates_dims[0];
    std::vector<Point> known_coords(num_known_points);
    std::vector<double> known_vals(num_known_points);
    for (npy_intp i = 0; i < num_known_points; i++) {
        known_coords[i] = (Point(*(double*)PyArray_GETPTR2(known_coord_numpy_arr, i, 0),
                                 *(double*)PyArray_GETPTR2(known_coord_numpy_arr, i, 1),
                                 *(double*)PyArray_GETPTR2(known_coord_numpy_arr, i, 2)));
        known_vals[i] = *((double*)PyArray_GETPTR1(known_values_numpy_arr, i));
    }

    int num_interpolation_points = interpolation_points_dims[1];
    std::vector<Point> interp_points(num_interpolation_points);
    for (npy_intp i = 0; i < num_interpolation_points; i++) {
        interp_points[i] = (Point(*(double*)PyArray_GETPTR2(interpolation_points_numpy_arr, 0, i),
                                  *(double*)PyArray_GETPTR2(interpolation_points_numpy_arr, 1, i),
                                  *(double*)PyArray_GETPTR2(interpolation_points_numpy_arr, 2, i)));
    }

    std::vector<double> *interpolation_values = natural_neighbor(known_coords,
                                                                 known_vals,
                                                                 interp_points,
                                                                 coord_max);
    npy_intp dims[] = {num_interpolation_points};
    PyObject* result = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    for (int i = 0; i < num_interpolation_points; i++) {
        *((double*)PyArray_GETPTR1(result, i)) = (*interpolation_values)[i];
    }

    printf("Calculation completed.\n");
    Py_DECREF(known_coord_numpy_arr);
    Py_DECREF(known_values_numpy_arr);
    Py_DECREF(interpolation_points_numpy_arr);

    delete interpolation_values;

    return result;
}
