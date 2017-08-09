#include "Python.h"
#include "numpy/arrayobject.h"
#include <boost/geometry.hpp>
#include <vector>
#include "nn.h"

static char module_docstring[] = "This module calculates natural neighbor interpolation for a given dataset in 3D.";

static char nn_docstring[] = "Calculate the natural neighbor interpolation of a dataset.";

static PyObject *naturalneighbor_natural_neighbor(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"natural_neighbor", naturalneighbor_natural_neighbor, METH_VARARGS, nn_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "naturalneighbor",   /* name of module */
   module_docstring, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
  module_methods
};

PyMODINIT_FUNC PyInit_naturalneighbor(void) {
    /* PyObject *m = Py_InitModule3("naturalneighbor", module_methods, module_docstring); */
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) { return NULL; }

    import_array(); // Enable numpy functionality
    return m;
}

typedef boost::geometry::model::point <double, 3, boost::geometry::cs::cartesian> Point;
static PyObject *naturalneighbor_natural_neighbor(PyObject *self, PyObject *args) {
    int coord_max;
    PyObject *known_coord_obj, *known_values_obj, *interpolation_points_obj;

    if (!PyArg_ParseTuple(args, "OOOi", &known_coord_obj,
                          &known_values_obj, &interpolation_points_obj, &coord_max)) {
        return NULL;
    }
    //TODO: NPY_IN_ARRAY or NPY_ARRAY_IN_ARRAY?
    //https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#PyArray_FROM_OTF
    PyObject *known_coord_numpy_arr = PyArray_FROM_OTF(known_coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *known_values_numpy_arr = PyArray_FROM_OTF(known_values_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *interpolation_points_numpy_arr = PyArray_FROM_OTF(interpolation_points_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (known_coord_numpy_arr == NULL || known_values_numpy_arr == NULL || interpolation_points_numpy_arr == NULL) {
        Py_XDECREF(known_coord_numpy_arr);
        Py_XDECREF(known_values_numpy_arr);
        Py_XDECREF(interpolation_points_numpy_arr);
        return NULL;
    }

    double *known_coordinates = (double*)PyArray_DATA(known_coord_numpy_arr);
    int *known_coordinates_dims = (int*)PyArray_DIMS(known_coord_numpy_arr);
    double *known_values = (double*)PyArray_DATA(known_coord_numpy_arr);
    double *interpolation_points = (double*)PyArray_DATA(interpolation_points_numpy_arr);
    int *interpolation_points_dims = (int*)PyArray_DIMS(interpolation_points_numpy_arr);

    //TODO: Add in bounds checking if user passes invalid shape
    int num_known_points = known_coordinates_dims[0];
    std::vector<Point> known_coords(num_known_points);
    std::vector<double> known_vals(num_known_points);
    for (int i = 0; i < num_known_points; i++) {
        known_coords.push_back(Point(*(double*)PyArray_GETPTR2(known_coord_numpy_arr, i, 0),
                                     *(double*)PyArray_GETPTR2(known_coord_numpy_arr, i, 1),
                                     *(double*)PyArray_GETPTR2(known_coord_numpy_arr, i, 2)));
        known_vals.push_back(known_values[i]);
    }

    int num_interpolation_points = interpolation_points_dims[0];
    std::vector<Point> interp_points(num_interpolation_points);
    for (int i = 0; i < num_interpolation_points; i++) {
        interp_points.push_back(Point(*(double*)PyArray_GETPTR2(interpolation_points_numpy_arr, i, 0),
                                      *(double*)PyArray_GETPTR2(interpolation_points_numpy_arr, i, 1),
                                      *(double*)PyArray_GETPTR2(interpolation_points_numpy_arr, i, 2)));
    }


    std::vector<double> *interpolation_values = natural_neighbor(known_coords,
                                                                 known_vals,
                                                                 interp_points,
                                                                 coord_max);

    Py_DECREF(known_coord_numpy_arr);
    Py_DECREF(known_values_numpy_arr);
    Py_DECREF(interpolation_points_numpy_arr);

    npy_intp dims[] = {num_interpolation_points};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    for (int i = 0; i < num_interpolation_points; i++) {
        double interp_value = (*interpolation_values)[i];
        ((double*) PyArray_DATA(result))[i] = interp_value;
    }

    //TODO: Convert function result into numpy array
    return result;
}
