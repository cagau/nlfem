# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

# distutils: language = c++
# distutils: sources = RoadDensityPyMod.cpp
from libcpp cimport bool

cdef extern from "Cassemble.h" nogil:
    # Assembly algorithm with BFS -----------------------------------------------------------------------------------------
    void par_assemble(  double * ptrAd,
                        const int K_Omega,
                        const int K,
                        double * fd,
                        const long * ptrTriangles,
                        const double * ptrVerts,
            # Number of Triangles and number of Triangles in Omega
                        const int J, const int J_Omega,
            # Number of vertices (in case of CG = K and K_Omega)
                        const int L, const int L_Omega,
                        const double * Px, const int nPx, const double * dx,
                        const double * Py, const int nPy, const double * dy,
                        const double sqdelta,
                        const long * ptrNeighbours,
                        const int is_DiscontinuousGalerkin,
                        const int is_NeumannBoundary,
                        const int dim
    )
    # void par_assembleMass(double *, long *, double *, int, int, int, double *, double *) nogil
    # void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *) nogil

    # Mass matrix evaluation ----------------------------------------------------------------------------------------------
    # void par_evaluateMass(double *, double *, long *, double *, int, int, int, double *, double *) nogil

    # DEBUG Helpers and test functions
    # int retriangulate(double *, double *, double, double *, int) nogil
    # double model_f(double *) nogil
    # void toPhys(double *, double *, double *) nogil
    # void toRef(double *, double *, double *) nogil
    # void model_basisFunction(double *, double *) nogil



