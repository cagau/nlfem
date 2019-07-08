#include <Python.h>

static PyObject *chiffre_caesar(PyObject *self, PyObject *args);

static PyMethodDef ChiffreMethods[] =
        {
                {"caesar", chiffre_caesar, METH_VARARGS,
                                "Perform Caesar cipher encryption."},
                {0, 0, 0, 0}
        };

static PyModuleDef ChiffreModule =
        {
                PyModuleDef_HEAD_INIT,
                "chiffre",
                "Performs insane encryption operations",
                -1,
                ChiffreMethods
        };

PyMODINIT_FUNC PyInit_chiffre(void)
{
    return PyModule_Create(&ChiffreModule);
}


static PyObject *chiffre_caesar(PyObject *self, PyObject *args)
{
    char *text, *encrypted, *c, *e;
    PyObject *result = 0;
    int cipher, length;

    if(!PyArg_ParseTuple(args, "si", &text, &cipher))
        return 0;

    length = strlen(text);
    encrypted = (char *)malloc(length+1);
    encrypted[length] = '\0';

    for(c = text, e = encrypted; *c; c++, e++)
        *e = ((*c - 'A' + cipher) % 26) + 'A';
    result = Py_BuildValue("s", encrypted);
    free(encrypted);

    return result;
}