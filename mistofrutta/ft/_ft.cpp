#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <iostream>
#include "ft.hpp"

static PyObject *ft_cubic(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _ftMethods[] = {
    {"ft_cubic", ft_cubic, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _ft = {
    PyModuleDef_HEAD_INIT,
    "_ft",
    NULL, // Module documentation
    -1,
    _ftMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__ft(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_ft);
    }
    
    
//////// The actual functions of the modules

static PyObject *ft_cubic(PyObject *self, PyObject *args) {

    int32_t M, N;
    double a, delta;
    PyObject *h_o, *R_o, *I_o;
    
    if(!PyArg_ParseTuple(args, "OddO", 
                &h_o, &a, &delta, &R_o)) 
                return NULL;
    
    // Get the PyArrayObjects. This will also cast the datatypes if needed.
    PyArrayObject *h_a = (PyArrayObject*) PyArray_FROM_OT(h_o, NPY_COMPLEX128);
    PyArrayObject *R_a = (PyArrayObject*) PyArray_FROM_OT(R_o, NPY_FLOAT64);
    
    // Extract the lenghts of h and R, as their shape[0].
    M = *(PyArray_SHAPE(h_a));
    N = *(PyArray_SHAPE(R_a));
    
    // Create the numpy array to be returned
    I_o = PyArray_SimpleNew(1, PyArray_SHAPE(R_a), NPY_COMPLEX128);
    PyArrayObject *I_a = (PyArrayObject*) PyArray_FROM_OT(I_o, NPY_COMPLEX128);
    Py_INCREF(I_o);
    
        
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (h_a == NULL || R_a == NULL || I_a == NULL) {
        Py_XDECREF(h_a);
        Py_XDECREF(R_a);
        Py_XDECREF(I_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    std::complex<double> *h = (std::complex<double>*)PyArray_DATA(h_a);
    double *R = (double*)PyArray_DATA(R_a);
    std::complex<double> *I = (std::complex<double>*)PyArray_DATA(I_a);
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    ft_cubic(h,M,a,delta,R,N,I);
          
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(h_a);
    Py_XDECREF(R_a);
    Py_XDECREF(I_a);
    
    // Return the computed Fourier integral
    return I_o;
}
