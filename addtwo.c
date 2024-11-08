#include <Python.h>

static PyObject* add(PyObject* self, PyObject* args) {
    double a, b;
    if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }
    return Py_BuildValue("d", a + b);
}

static PyMethodDef AddTwoMethods[] = {
    {"add", add, METH_VARARGS, "Add two numbers"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef addtwo = {
    PyModuleDef_HEAD_INIT,
    "addtwo",
    NULL,
    -1,
    AddTwoMethods
};

PyMODINIT_FUNC PyInit_addtwo(void) {
    return PyModule_Create(&addtwo);
}