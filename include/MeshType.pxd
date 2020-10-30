cdef extern from "MeshType.h":
    cdef cppclass MeshType:
            MeshType(const long * ptrElements_,
                 const long * ptrElementLabels_,
                 const double * ptrVerts_,
                 const long nE_,
                 const long nE_Omega_,
                 const int nV_,
                 const int nV_Omega_,
                 const long * ptrNeighborIndices_,
                 const long * ptrNeighborIndexPtr_,
                 const long dim_,
                 const double maxDiameter_,
                 const long outdim,
                 const long nNeighbours
                 ) except +
