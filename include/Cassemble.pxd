# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

# distutils: language = c++
# distutils: sources = RoadDensityPyMod.cpp
from libcpp cimport bool
from libcpp.string cimport string
# Assembly algorithm with BFS -----------------------------------------------------------------------------------------

cdef extern from "Cassemble.h" nogil:
    void par_assemble( const string path_spAd,
        const int K_Omega,
        const int K,
        double * fd,
        const long * ptrTriangles,
        const long * ptrLabelTriangles,
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
        const string  str_model_kernel,
        const string str_model_f,
        const string str_integration_method,
        const int is_PlacePointOnCap,
        const int dim
    ) nogil

    # Mass matrix evaluation ----------------------------------------------------------------------------------------------
    void par_evaluateMass(double *vd, double *ud, long *Elements, long *ElementLabels, double *Verts, int K_Omega, int J, int nP,
                     double *P, double *dx, const int dim) nogil
    void constructAdjaciencyGraph(const int dim, const int nE, const long * elements, long * neighbours) nogil
    # DEBUG Helpers and test functions
    int method_retriangulate(const double * x_center, const double * TE,
                           const double sqdelta, double * re_Triangle_list,
                           int is_placePointOnCap) nogil
