# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

# distutils: language = c++
# distutils: sources = RoadDensityPyMod.cpp

cdef extern from "Cassemble.h":
    struct QuadratureStruct:
        double * Px
        double * Py
        double * dx
        double * dy
        int nPx
        int nPy

    struct MeshStruct:
        int K_Omega
        int K
        long * ptrTriangles
        double * ptrVerts
        # Number of Triangles and number of Triangles in Omega
        int J
        int J_Omega
        # Number of vertices (in case of CG = K and K_Omega)
        int L
        int L_Omega
        double sqdelta
        long * ptrNeighbours
        int is_DiscontinuousGalerkin
        int is_NeumannBoundary
        int dim
        int dVertex

ctypedef QuadratureStruct QuadratureType
ctypedef MeshStruct MeshType

cdef extern from "Cassemble.h":
    # Assembly algorithm with BFS -----------------------------------------------------------------------------------------
    void par_assemble(  const MeshType& mesh,
                        const QuadratureType& quadRule,
                        double * Ad,
                        double * fd
                        ) nogil
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



