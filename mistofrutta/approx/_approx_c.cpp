#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <stdint.h>

static PyObject *fastexp(PyObject *self, PyObject *args);
static PyObject *fastlog(PyObject *self, PyObject *args);
static PyObject *gamma_orig(PyObject *self, PyObject *args);
static PyObject *digamma(PyObject *self, PyObject *args);
static PyObject *trigamma(PyObject *self, PyObject *args);

double _fastexp(double x);
double _digamma(double x, int order, int order2);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _approx_cMethods[] = {
    {"exp", fastexp, METH_VARARGS, ""},
    {"log", fastlog, METH_VARARGS, ""},
    {"gamma", gamma_orig, METH_VARARGS, ""},
    {"digamma", digamma, METH_VARARGS, ""},
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

////////////////////////////////////////////////////////////////////////////////////
// Actual stuff that has to stay in this module
////////////////////////////

double _fastexp(double x){
    double y;
    y = 1.0 + x / 4096.0;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    return y;
}

double _fastlog(double x) {
    //return pow(1.+x/pow(2.0,order),order);
    double y;
    y = (x-1.)/(x+1.);
    y = 2.0*(y+0.3333333*y*y*y+0.2*y*y*y*y*y);
    return y;
}

double _digamma(double x, int order, int order2) {
    double coeff[15] = {-0.5,-1./12.,0.0,1./120.,0.0,-1./256.,0.0,1./240.,0.0,-5./660.,0.0,691./32760.,0.0,-1./12.};
    
    double xporder2 = x+order2;
    double y = log(xporder2);
    
    for(int j=0;j<order2;j++) {
        y -= 1./(x+j);
    }
    
    for (int k=0;k<order;k++) {
        if(coeff[k]!=0.0){
            y += coeff[k]/xporder2;
            //out[i] += coeff[k]/pow(xporder2,(double)k+1.); // VERSION 1
        }
        xporder2 *= xporder2;
    }
    
    return y;
}


static PyObject *fastexp(PyObject *self, PyObject *args) {

    int sizeX, order;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiOi", &X_o, &sizeX, &out_o, &order)) return NULL;
    
    PyArrayObject *X_a = (PyArrayObject*) PyArray_FROM_OT(X_o, NPY_FLOAT64);
    PyArrayObject *out_a = (PyArrayObject*) PyArray_FROM_OT(out_o, NPY_FLOAT64);
        
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
    Py_XDECREF(out_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *fastlog(PyObject *self, PyObject *args) {

    int sizeX;//, order;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiO", &X_o, &sizeX, &out_o)) return NULL;
    
    PyArrayObject *X_a = (PyArrayObject*) PyArray_FROM_OT(X_o, NPY_FLOAT64);
    PyArrayObject *out_a = (PyArrayObject*) PyArray_FROM_OT(out_o, NPY_FLOAT64);
        
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
    //double y;
    // see http://codingforspeed.com/using-faster-exponential-approximation
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        out[i] = _fastlog(xi);
    }        
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(X_a);
    Py_XDECREF(out_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *gamma_orig(PyObject *self, PyObject *args) {

    int sizeX;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiO", &X_o, &sizeX, &out_o)) return NULL;
    
    PyArrayObject *X_a = (PyArrayObject*) PyArray_FROM_OT(X_o, NPY_FLOAT64);
    PyArrayObject *out_a = (PyArrayObject*) PyArray_FROM_OT(out_o, NPY_FLOAT64);
        
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
    
    for(int i=0;i<sizeX;i++) {
        out[i] = tgamma(X[i]);
    }   
        
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(X_a);
    Py_XDECREF(out_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *digamma(PyObject *self, PyObject *args) {

    int sizeX, order, order2;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiOii", &X_o, &sizeX, &out_o, &order, &order2)) return NULL;
    
    PyArrayObject *X_a = (PyArrayObject*) PyArray_FROM_OT(X_o, NPY_FLOAT64);
    PyArrayObject *out_a = (PyArrayObject*) PyArray_FROM_OT(out_o, NPY_FLOAT64);
        
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
    
    
    Py_BEGIN_ALLOW_THREADS
    //double coeff[15] = {-0.5,-1./12.,0.0,1./120.,0.0,-1./256.,0.0,1./240.,0.0,-5./660.,0.0,691./32760.,0.0,-1./12.};
    
    double xi;
    //double xporder2;
    
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        out[i] = _digamma(xi, order, order2);
        /**xporder2 = xi+order2;
        out[i] = log(xporder2);
        
        for(int j=0;j<order2;j++) {
            out[i] -= 1./(xi+j);
        }
        
        for (int k=0;k<order;k++) {
            if(coeff[k]!=0.0){
                out[i] += coeff[k]/xporder2;
                //out[i] += coeff[k]/pow(xporder2,(double)k+1.); // VERSION 1
            }
            xporder2 *= xporder2;
        }**/
    }
    Py_END_ALLOW_THREADS   
        
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(X_a);
    Py_XDECREF(out_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *trigamma(PyObject *self, PyObject *args) {

    int sizeX, order;
    PyObject *X_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OiOi", &X_o, &sizeX, &out_o, &order)) return NULL;
    
    PyArrayObject *X_a = (PyArrayObject*) PyArray_FROM_OT(X_o, NPY_FLOAT64);
    PyArrayObject *out_a = (PyArrayObject*) PyArray_FROM_OT(out_o, NPY_FLOAT64);
        
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
    int order2 = 5;
    
    double xi;
    double xporder2;
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        out[i] = 0.0;
        
        for(int j=0;j<order2;j++) {
            out[i] += 1./pow(xi+j,2.0);// + 1./pow(xi+1.0,2.0) + 1./pow(xi+2.0,2.) + 1./pow(xi+3.0,2.);
        }
        xporder2 = xi+order2;
        
        for (int k=0;k<order;k++) {
            if(ber[k]!=0.0){
                //out[i] += ber[k]/(double)xporder2; // VERSION 2 5 times faster, but less precision
                out[i] += ber[k]/pow(xporder2,(double)k+1.); // VERSION 1
            }
            //xporder2 *= xporder2; // VERSION 2
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
    Py_XDECREF(out_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

