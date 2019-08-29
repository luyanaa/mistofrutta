//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <stdint.h>

static PyObject *trigamma(PyObject *self, PyObject *args);
static PyObject *fastexp(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _approx_cMethods[] = {
    {"exp", fastexp, METH_VARARGS, ""},
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


static PyObject *fastexp(PyObject *self, PyObject *args) {

    int sizeX, order;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiOi", &X_o, &sizeX, &out_o, &order)) return NULL;
    
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
    
    double xi;
    double y;
    // see http://codingforspeed.com/using-faster-exponential-approximation
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        
        y = 1.0 + xi / 4096.0;
        y *= y; y *= y; y *= y; y *= y;
        y *= y; y *= y; y *= y; y *= y;
        y *= y; y *= y; y *= y; y *= y;
        out[i] = y;
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

static PyObject *trigamma(PyObject *self, PyObject *args) {

    int sizeX, order;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiOi", &X_o, &sizeX, &out_o, &order)) return NULL;
    
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
    
    // Bernoulli numbers (B+)
    // https://en.wikipedia.org/wiki/Bernoulli_number
    double ber[21] = {1.0,0.5,1./6.,0.0,-1./30.,0.0,1./42.,0.0,-1./30.,0.0,5./66.,0.0,-691./2730.,0.,7./6.,0.0,-3617./510.,0.0,43867./798.,0.0,-1746611./330.};
    
    double xi;
    double xp3;
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        out[i] = 1./pow(xi,2.0) + 1./pow(xi+1.0,2.0) + 1./pow(xi+2.0,2.);
        for (int k=0;k<order;k++) {
            if(ber[k]!=0.0){
                xp3 = X[i]+3.;
                out[i] += ber[k]/pow(xp3,(double)k+1);
            }
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

