# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

# distutils: language = c++
# distutils: sources = RoadDensityPyMod.cpp

cdef extern from "Cassemble.h":
    # Assembly algorithm with BFS -----------------------------------------------------------------------------------------
    void par_assemble(double *, int, int, double *, long *, double *,int, int, int, int, double *, int, double *, double *, int, double*, double, long *, int, int) nogil
    # void par_assembleMass(double *, long *, double *, int, int, int, double *, double *) nogil
    # void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *) nogil

    # Mass matrix evaluation ----------------------------------------------------------------------------------------------
    # void par_evaluateMass(double *, double *, long *, double *, int, int, int, double *, double *) nogil

    # DEBUG Helpers and test functions
    int retriangulate(double *, double *, double, double *, int) nogil
    # double model_f(double *) nogil
    void toPhys(double *, double *, double *) nogil
    void toRef(double *, double *, double *) nogil
    void model_basisFunction(double *, double *) nogil



