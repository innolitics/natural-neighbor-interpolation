#include <python/Python.h>
#include <numpy/arrayobject.h>
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

PyMODINIT_FUNC init_naturalneighbor(void) {
    PyObject *m = Py_InitModule3("_naturalneighbor", module_methods, module_docstring);
    if (m == NULL) { return; }

    import_array(); // Enable numpy functionality
}

typedef boost::geometry::model::point <double, 3, boost::geometry::cs::cartesian> Point;
static PyObject *naturalneighbor_natural_neighbor(PyObject *self, PyObject *args) {
    int coord_max;
    PyObject *known_coord_obj, *known_values_obj, *interpolation_points_obj;

    if (!PyArg_ParseTuple(args, "OOOi", &known_coord_obj,
                          &known_values_obj, &interpolation_points_obj, &coord_max)) {
        return NULL;
    }
    //TODO: points aren't doubles, change the specified type here
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
    double *known_values = (double*)PyArray_DATA(known_coord_numpy_arr);
    double *interpolation_points = (double*)PyArray_DATA(interpolation_points_numpy_arr);

    //TODO: Work these parameters into vector<Point> and vector<double>

    std::vector<double> interpolation_values = natural_neighbor(known_coordinates,
                                                                known_values,
                                                                interpolation_points,
                                                                coord_max);

    Py_DECREF(known_coord_numpy_arr);
    Py_DECREF(known_values_numpy_arr);
    Py_DECREF(interpolation_points_numpy_arr);

    //TODO: Convert function result into numpy array
    PyObject *ret = Py_BuildValue("O", interpolation_values);
    return ret;
}
