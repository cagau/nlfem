# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

# distutils: language = c++
# distutils: sources = RoadDensityPyMod.cpp
from libcpp cimport bool

cdef extern from "Cassemble2D.h" nogil:
    # Assembly algorithm with BFS -----------------------------------------------------------------------------------------
    void par_assemble2D(  double * Ad,
                        int K,
                        double * fd,
                        const long * Triangles,
                        const long * ptrLabelTriangles,
                        const double * Verts,
                        # Number of Triangles and number of Triangles in Omega
                        int J, int J_Omega,
                        # Number of vertices (in case of CG = K and K_Omega)
                        int L, int L_Omega,
                        double * Px, int nPx, double * dx,
                        double * Py, int nPy, double * dy,
                        double sqdelta,
                        const long * Neighbours,
                        int is_DiscontinuousGalerkin,
                        int is_NeumannBoundary) nogil




