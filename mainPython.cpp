#include <iostream>
#include <Python.h>
#include "include/masCone.h"
#include <numpy/arrayobject.h>

using namespace std;

int main() {

    vector<double> test;
    test.push_back(2);
    test.push_back(1);
    test.push_back(3);
    test.push_back(7);
    test.push_back(6);
    test.push_back(4);
    test.push_back(5);
    masCone testCone(test);

    vector<double> testSignal;
    testSignal.push_back(2);
    testSignal.push_back(1);
    testSignal.push_back(4);
    testSignal.push_back(4);
    testSignal.push_back(6);
    testSignal.push_back(4);
    testSignal.push_back(2);

}

static vector<double> getDoubleVector(PyObject *seq) {

    int seqlen;
    int i;

    seqlen = PySequence_Fast_GET_SIZE(seq);
    vector<double> data;
    for (i = 0; i < seqlen; i++) {
        PyObject *fitem;
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if (!item) {
        }
        fitem = PyNumber_Float(item);
        if (!fitem) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_TypeError, "all items must be numbers");
        }
        data.push_back(PyFloat_AS_DOUBLE(fitem));
        Py_DECREF(fitem);
    }
    Py_DECREF(seq);
    return data;
}

static PyObject *approximation(PyObject *self, PyObject *args) {

    PyObject *signalSeq;
    PyObject *coneSeq;

    vector<double> signal;
    vector<double> coneData;
    double *dbar;
    int seqlen;
    int i;

    if (!PyArg_ParseTuple(args, "O|O", &signalSeq, &coneSeq))
        return 0;

    signalSeq = PySequence_Fast(signalSeq, "argument must be iterable");
    coneSeq = PySequence_Fast(coneSeq, "argument must be iterable");

    if (!signalSeq || !coneSeq)
        return 0;

    signal = getDoubleVector(signalSeq);
    coneData = getDoubleVector(coneSeq);

    masCone testCone(coneData);

    vector<double> testSignal;
    testSignal = testCone.approximation(signal);

    PyObject *result = PyList_New(0);

    for (i = 0; i < testSignal.size(); i++) {
        PyList_Append(result, PyFloat_FromDouble(testSignal[i]));
    }

    return result;
}


static PyObject *proximity(PyObject *self, PyObject *args) {

    PyObject *signalSeq;
    PyObject *coneSeq;

    vector<double> signal;
    vector<double> coneData;
    double *dbar;
    int seqlen;
    int i;

    if (!PyArg_ParseTuple(args, "O|O", &signalSeq, &coneSeq))
        return 0;

    signalSeq = PySequence_Fast(signalSeq, "argument must be iterable");
    coneSeq = PySequence_Fast(coneSeq, "argument must be iterable");

    if (!signalSeq || !coneSeq)
        return 0;

    signal = getDoubleVector(signalSeq);
    coneData = getDoubleVector(coneSeq);

    masCone testCone(coneData);

    double result;
    result = testCone.getProximity(signal);

    PyObject *resultPy = PyFloat_FromDouble(result);

    return resultPy;
}

static PyObject *distance(PyObject *self, PyObject *args) {

    PyObject *ASeq;
    PyObject *BSeq;

    vector<double> A;
    vector<double> B;
    double *dbar;
    int seqlen;
    int i;

    if (!PyArg_ParseTuple(args, "O|O", &ASeq, &BSeq))
        return 0;

    ASeq = PySequence_Fast(ASeq, "argument must be iterable");
    BSeq = PySequence_Fast(BSeq, "argument must be iterable");

    if (!ASeq || !BSeq)
        return 0;

    A = getDoubleVector(ASeq);
    B = getDoubleVector(BSeq);

    double result;
    result = masCone::distance(A, B);

    PyObject *resultPy = PyFloat_FromDouble(result);

    return resultPy;
}

static PyObject *getChunkProximity(PyObject *self, PyObject *args) {

    PyObject *signalSeq;
    PyObject *coneSeq;

    vector<double> signal;
    vector<double> coneData;
    double *dbar;
    int seqlen;
    int i;

    if (!PyArg_ParseTuple(args, "O|O", &signalSeq, &coneSeq))
        return 0;

    signalSeq = PySequence_Fast(signalSeq, "argument must be iterable");
    coneSeq = PySequence_Fast(coneSeq, "argument must be iterable");

    if (!signalSeq || !coneSeq)
        return 0;

    signal = getDoubleVector(signalSeq);
    coneData = getDoubleVector(coneSeq);

    masCone testCone(coneData);

    vector<double> testSignal;
    testSignal = testCone.getChunkProximity(signal);

    PyObject *result = PyList_New(0);

    for (i = 0; i < testSignal.size(); i++) {
        PyList_Append(result, PyFloat_FromDouble(testSignal[i]));
    }

    return result;
}

static PyObject *compareTwoSignals(PyObject *self, PyObject *args) {
    PyObject *signalSeq;
    PyObject *coneSeq;

    vector<double> signal;
    vector<double> coneData;
    double *dbar;
    int seqlen;
    int i;

    if (!PyArg_ParseTuple(args, "O|O", &signalSeq, &coneSeq))
        return 0;

    signalSeq = PySequence_Fast(signalSeq, "argument must be iterable");
    coneSeq = PySequence_Fast(coneSeq, "argument must be iterable");

    if (!signalSeq || !coneSeq)
        return 0;

    signal = getDoubleVector(signalSeq);
    coneData = getDoubleVector(coneSeq);
    int samples = 50;

    masCone testCone(coneData);

    vector<double> testSignal;
    testSignal = testCone.getChunkProximity(signal);

    PyObject *result = PyList_New(0);

    for (i = 0; i < testSignal.size(); i++) {
        PyList_Append(result, PyFloat_FromDouble(testSignal[i]));
    }

    return result;
}

static PyObject *test(PyObject *self, PyObject *args) {
    cout << "HELLO " << endl;
}

PyMODINIT_FUNC PyInit_mas(void) {

    Py_Initialize();
    import_array();

    static PyMethodDef methods[] = {
            {"approximation",     approximation,     METH_VARARGS,
                    "approximation"},
            {"proximity",         proximity,         METH_VARARGS,
                    "proximity"},
            {"distance",          distance,          METH_VARARGS,
                    "distance"},
            {"getChunkProximity", getChunkProximity, METH_VARARGS,
                    "getChunkProximity"},
            {"test",              test,              METH_VARARGS,
                    "test"},
            {NULL, NULL, 0, NULL}
    };

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "mas",     /* m_name */
            "This is a module",  /* m_doc */
            -1,                  /* m_size */
            methods,    /* m_methods */
            NULL,                /* m_reload */
            NULL,                /* m_traverse */
            NULL,                /* m_clear */
            NULL,                /* m_free */
    };
#endif


#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);

#else
    PyObject *m = Py_InitModule("mas",
        methods, "This is a module");
#endif
}


