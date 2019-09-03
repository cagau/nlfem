# This file cimports the C-Functions from clinalg.c and makes them avaialbe to python via def. This is usually
# necessary only for few "entrance" functions to the actual C or Cython Code. However, if setup_linalg.py is
# executed (via python setup_linalg.py build_ext --inplace -f) a standalone package is created which also contains
# the import-able def function you find below.


from linalg cimport doubleVec_any as c_doubleVec_any
cimport numpy as np

def py_doubleVec_any(np.ndarray[double, ndim=1, mode="c"] a, int len):
    return c_doubleVec_any(&a[0], len)