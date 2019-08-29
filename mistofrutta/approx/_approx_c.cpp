//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <stdint.h>

static PyObject *trigamma(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _approx_cMethods[] = {
    {"trigamma", trigamma, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _approx_c = {
    PyModuleDef_HEAD_INIT,
    "_approx_c",
    NULL, // Module documentation
    -1,
    _approx_cMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__approx_c(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_approx_c);
    }
    
    
//////// The actual functions of the modules

static PyObject *trigamma(PyObject *self, PyObject *args) {

    int sizeX;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiO", &X_o, &sizeX, &out_o)) return NULL;
    
    PyObject *X_a = PyArray_FROM_OTF(X_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *out_a = PyArray_FROM_OTF(out_o, NPY_FLOAT64, NPY_IN_ARRAY);
        
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (X_a == NULL || out_a == NULL) {
        Py_XDECREF(X_a);
        Py_XDECREF(out_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    double *X = (double*)PyArray_DATA(X_a);
    double *out = (double*)PyArray_DATA(out_a);
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    // From wikipedia. https://en.wikipedia.org/wiki/Trigamma_function
    // Recurrence relation to make argument large, and Laurent series for 
    // asymptotic expansion.
    
    int order = 7;
    // Bernoulli numbers (B+)
    double ber[7] = {1.0,0.5,1./6.,-1./3.,1./42.,-1./3.,5./66.};
    
    double xi;
    double xp3;
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        out[i] = 1./pow(xi,2.0) + 1./pow(xi+1.0,2.0) + 1./pow(xi+2.0,2.);
        for (int k=0;k<order;k++) {
            xp3 = X[i]+3.;
            out[i] += ber[k]/pow(xp3,(double)k+1);
        }
    }   
        
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(X_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

